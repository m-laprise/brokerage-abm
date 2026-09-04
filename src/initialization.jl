"""
    initialization.jl

Initialize agent types, the matching environment, the network, the broker,
histories, and the selected prediction models.
"""

using LinearAlgebra: norm, normalize, dot
using Random: AbstractRNG
using Graphs: add_vertex!, neighbors
using StableRNGs: StableRNG

# ─────────────────────────────────────────────────────────────────────────────
# Curve geometry and agent types
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_curve_geometry(d, s, rng) -> CurveGeometry

Draw random frequencies f_k ~ U{1,...,5} and phases θ_k ~ U[0,2π) for the
sinusoidal type curve with s active dimensions.
"""
function generate_curve_geometry(d::Int, s::Int, rng::AbstractRNG)::CurveGeometry
    freqs = [rand(rng, 1:5) for _ in 1:s]
    phases = [2π * rand(rng) for _ in 1:s]
    return CurveGeometry(d, s, freqs, phases)
end

"""
    curve_point(t, geo) -> Vector{Float64}

Evaluate the sinusoidal curve at position t ∈ [0,1]. Returns a unit vector.
"""
function curve_point(t::Float64, geo::CurveGeometry)::Vector{Float64}
    v = zeros(geo.d)
    for k in 1:geo.s
        v[k] = sin(2π * geo.freqs[k] * t + geo.phases[k])
    end
    n = norm(v)
    return n > 1e-12 ? v ./ n : v
end

"""
    generate_agent_types(N, geo, sigma_x, rng) -> types

Draw N agent types at random curve positions with noise, projected to the unit sphere.
Types remain in draw order, so their placement in the initial Watts-Strogatz graph
is not assortative by construction.
"""
function generate_agent_types(
    N::Int, geo::CurveGeometry, sigma_x::Float64, rng::AbstractRNG
)::Vector{Vector{Float64}}
    d = geo.d
    sigma_per_dim = sigma_x / sqrt(d)

    # Draw types
    types = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        t_i = rand(rng)
        cp = curve_point(t_i, geo)
        noisy = cp .+ sigma_per_dim .* randn(rng, d)
        n = norm(noisy)
        types[i] = n > 1e-12 ? noisy ./ n : noisy
    end

    return types
end

"""
    generate_matching_dgp(params, rng) -> NamedTuple

Generate the type geometry, realized agent types, and matching environment used at
the start of a simulation. The supplied RNG is advanced exactly through these DGP
draws so initialization and reporting analyses can share one implementation.
"""
function generate_matching_dgp(params::ModelParams, rng::AbstractRNG)
    geo = generate_curve_geometry(params.d, params.s, rng)
    agent_types = generate_agent_types(params.N, geo, params.sigma_x, rng)
    env = generate_matching_env(
        params.d,
        params.rho,
        params.delta,
        params.sigma_eps,
        agent_types,
        rng;
        sigma_x=params.sigma_x,
        curve_geo=geo,
        constant_signal_scale=params.constant_signal_scale,
    )
    return (; curve_geo=geo, agent_types, env)
end

# ─────────────────────────────────────────────────────────────────────────────
# Full initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    initialize_model(params) -> ModelState

Complete model initialization following `paper/appendices/simulation_pseudocode.tex`
(`Initialize`):
1. Agent types on sinusoidal curve
2. Matching function (c, A, B)
3. Calibration (q_cal, r, phi, c_s)
4. Network (Watts-Strogatz + broker node)
5. Broker roster seeding
6. Agent history seeding from all initial non-broker graph edges
7. Broker history seeding from existing roster-roster graph edges
8. State variables (satisfaction, reputation)
9. Initial fitting of the selected prediction model
"""
function initialize_model(params::ModelParams)::ModelState
    rng = StableRNG(params.seed)
    p = params
    d = p.d
    N = p.N

    # ── Agent types and matching environment (A, B, c) ──
    dgp = generate_matching_dgp(p, rng)
    geo = dgp.curve_geo
    agent_types = dgp.agent_types
    env = dgp.env

    # ── Calibration ──
    cal = calibrate(env, agent_types, p, rng)

    # ── Network ──
    G = build_network(N, p.k, p.p_rewire, rng)

    # ── Broker setup ──
    broker_node = N + 1
    n_roster_seed = roster_target_size(p)

    # Initialize broker NN
    d_broker = broker_pair_feature_dim(d)
    h_broker = broker_hidden_width(p)
    h_agent = agent_hidden_width(p)
    broker_nn = init_neural_net(d_broker, h_broker, rng)
    broker_grad = NNGradBuffers(broker_nn)

    broker = Broker(;
        node_id=broker_node,
        roster=Set{Int}(),
        current_clients=Set{Int}(),
        history_party1_types=Matrix{Float64}(undef, d, 64),
        history_party2_types=Matrix{Float64}(undef, d, 64),
        history_q=Vector{Float64}(undef, 64),
        history_count=0,
        history_retained_party=(
            if p.learning_model == :ridge && p.ridge_broker_variant == :single_principal
                Vector{UInt8}(undef, 64)
            else
                UInt8[]
            end
        ),
        retain_one_party=(
            p.learning_model == :ridge && p.ridge_broker_variant == :single_principal
        ),
        nn=broker_nn,
        nn_grad=broker_grad,
        predict_buf=zeros(h_broker),
        ridge=(
            if p.learning_model == :ridge
                RidgeModel(
                    p.ridge_broker_variant in (:additive, :single_principal) ? d : d_broker,
                    Q_OFFSET,
                )
            else
                nothing
            end
        ),
        n_new_obs=0,
        train_X=Matrix{Float64}(undef, d_broker, 128),
        train_q=Vector{Float64}(undef, 128),
        last_reputation=0.0,     # set from seed data below
        has_had_clients=false,
    )

    # Seed roster with random agents
    roster_candidates = collect(1:N)
    shuffle!(rng, roster_candidates)
    for i in 1:min(n_roster_seed, N)
        aid = roster_candidates[i]
        push!(broker.roster, aid)
        add_broker_edge!(G, aid, broker_node)
    end

    # ── Create agents ──
    initial_hist_cap = 16
    initial_train_cap = 16
    agents = Vector{Agent}(undef, N)
    for i in 1:N
        nn = init_neural_net(d, h_agent, rng)
        agents[i] = Agent(;
            id=i,
            type=agent_types[i],
            active_matches=ActiveMatch[],
            history_X=Matrix{Float64}(undef, d, initial_hist_cap),
            history_q=Vector{Float64}(undef, initial_hist_cap),
            history_count=0,
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
            ridge=(p.learning_model == :ridge ? RidgeModel(d, Q_OFFSET) : nothing),
            n_new_obs=0,
            train_X=Matrix{Float64}(undef, d, initial_train_cap),
            train_q=Vector{Float64}(undef, initial_train_cap),
            partner_sum=zeros(N),
            partner_count=zeros(Int, N),
            satisfaction_self=0.0,   # set from seed data below
            satisfaction_broker=0.0, # no broker experience at init
            tried_broker=false,
            periods_alive=0,
        )
    end

    # ── Agent history seeding from all initial non-broker graph edges ──
    edge_i = Int[]
    edge_j = Int[]
    edge_q = Float64[]
    for i in 1:N
        for j in neighbors(G, i)
            (j <= i || j > N) && continue
            q = match_output(agents[i].type, agents[j].type, env, rng)
            record_agent_history!(agents[i], agents[j].type, q)
            update_partner_mean!(agents[i], j, q)
            record_agent_history!(agents[j], agents[i].type, q)
            update_partner_mean!(agents[j], i, q)
            push!(edge_i, i)
            push!(edge_j, j)
            push!(edge_q, q)
        end
    end

    # ── Broker history seeding from existing roster-roster graph edges ──
    roster_edge_indices = [
        idx for idx in eachindex(edge_i) if
        (edge_i[idx] in broker.roster) && (edge_j[idx] in broker.roster)
    ]
    shuffle!(rng, roster_edge_indices)
    n_broker_seed = min(100, length(roster_edge_indices))
    for idx in @view roster_edge_indices[1:n_broker_seed]
        record_broker_history!(
            broker, agents[edge_i[idx]].type, agents[edge_j[idx]].type, edge_q[idx]; rng=rng
        )
    end

    # ── State variables (from seed data, not q_cal) ──
    # Broker reputation: mean of seed broker match outcomes
    if broker.history_count > 0
        broker.last_reputation =
            sum(broker.history_q[k] for k in 1:broker.history_count) / broker.history_count
        broker.has_had_clients = true
    end
    # Agent self-satisfaction: mean of seed match outcomes
    # Agent broker-satisfaction: broker reputation (market prior, not personal experience)
    for i in 1:N
        n = agents[i].history_count
        if n > 0
            agents[i].satisfaction_self = sum(agents[i].history_q[k] for k in 1:n) / n
        end
        agents[i].satisfaction_broker = broker.last_reputation
    end

    # Build model state
    state = ModelState(;
        params=p,
        rng=rng,
        period=0,
        env=env,
        cal=cal,
        curve_geo=geo,
        agents=agents,
        broker=broker,
        G=G,
        accum=PeriodAccumulators(),
        cached_network=CachedNetworkMeasures(),
    )

    # ── Initial predictor fitting ──
    # Initialization is period 0. Its cumulative history boundary lets rolling
    # windows exclude seed observations once W_p completed simulation periods
    # are available.
    for agent in agents
        push!(agent.obs_period_marks, agent.history_count)
        if agent.history_count > 0
            agent.n_new_obs = agent.history_count  # treat all seed data as new
            train_agent_predictor!(agent, p)
        end
    end
    push!(broker.obs_period_marks, broker.history_count)
    if broker.history_count > 0
        broker.n_new_obs = broker.history_count
        train_broker_predictor!(broker, agents, p, rng)
    end

    return state
end
