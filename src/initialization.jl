"""
    initialization.jl

Model initialization: agent types, matching environment, network, broker, history seeding,
and neural network initial training.
"""

using LinearAlgebra: norm, normalize, dot
using MultivariateStats: fit, PCA, predict
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
    generate_agent_types(N, geo, sigma_x, rng; sort_by_pc1=false) -> (types, inv_order)

Draw N agent types at random curve positions with noise, projected to the unit sphere.
When sort_by_pc1=true, types are sorted by first principal component for
type-assortative Watts-Strogatz ordering. When false (default), types are in random
order, producing a non-assortative initial network.
"""
function generate_agent_types(
    N::Int, geo::CurveGeometry, sigma_x::Float64, rng::AbstractRNG; sort_by_pc1::Bool=false
)::Tuple{Vector{Vector{Float64}},Vector{Int}}
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

    if sort_by_pc1
        # Sort by first principal component
        type_matrix = hcat(types...)  # d x N
        pca_model = fit(PCA, type_matrix; maxoutdim=1)
        pc1 = predict(pca_model, type_matrix)[1, :]
        sort_order = sortperm(pc1)
        sorted_types = types[sort_order]
        inv_order = invperm(sort_order)
        return (sorted_types, inv_order)
    else
        # Random ordering (no type assortativity in initial network)
        return (types, collect(1:N))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Full initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    initialize_model(params) -> ModelState

Complete model initialization following `simulation_pseudocode.tex` (`Initialize`):
1. Agent types on sinusoidal curve
2. Matching function (c, A, B)
3. Calibration (q_cal, r, phi, c_s)
4. Network (Watts-Strogatz + broker node)
5. Broker roster seeding
6. Agent history seeding from all initial non-broker graph edges
7. Broker history seeding from existing roster-roster graph edges
8. State variables (satisfaction, reputation)
9. Neural network initial training (E_init steps)
"""
function initialize_model(params::ModelParams; sort_by_pc1::Bool=false)::ModelState
    rng = StableRNG(params.seed)
    p = params
    d = p.d
    N = p.N

    # ── Agent types ──
    geo = generate_curve_geometry(d, p.s, rng)
    sorted_types, _ = generate_agent_types(N, geo, p.sigma_x, rng; sort_by_pc1=sort_by_pc1)

    # ── Ideal type c ──
    # (perturbation of a random curve position)

    # ── Matching environment (A, B, c) ──
    env = generate_matching_env(
        d, p.rho, p.delta, p.sigma_eps, sorted_types, rng; sigma_x=p.sigma_x, curve_geo=geo
    )

    # ── Calibration ──
    cal = calibrate(env, sorted_types, p, rng)

    # ── Network ──
    G = build_network(N, p.k, p.p_rewire, rng)

    # ── Broker setup ──
    broker_node = N + 1
    n_roster_seed = roster_target_size(N)

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
        history_Xi=Matrix{Float64}(undef, d, 64),
        history_Xj=Matrix{Float64}(undef, d, 64),
        history_q=Vector{Float64}(undef, 64),
        history_count=0,
        nn=broker_nn,
        nn_grad=broker_grad,
        predict_buf=zeros(h_broker),
        n_new_obs=0,
        train_X=Matrix{Float64}(undef, d_broker, 128),
        train_q=Vector{Float64}(undef, 128),
        last_reputation=0.0,     # set from seed data below
        has_had_clients=false,
        capture_confidence_mae=0.0,
        capture_confidence_ready=false,
        capture_error_count=0,
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
            type=sorted_types[i],
            active_matches=ActiveMatch[],
            history_X=Matrix{Float64}(undef, d, initial_hist_cap),
            history_q=Vector{Float64}(undef, initial_hist_cap),
            history_count=0,
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
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
            broker, agents[edge_i[idx]].type, agents[edge_j[idx]].type, edge_q[idx]
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

    # ── Initial neural network training ──
    for agent in agents
        if agent.history_count > 0
            agent.n_new_obs = agent.history_count  # treat all seed data as new
            train_agent_nn_impl!(agent, p, true)
        end
    end
    if broker.history_count > 0
        broker.n_new_obs = broker.history_count
        train_broker_nn!(broker, p)
    end

    return state
end
