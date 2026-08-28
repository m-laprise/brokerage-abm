"""
    types.jl

Agent types, model state, and supporting structs for BrokerageABM v0.3.
Unimodal matching market: N agents + 1 broker on a single network G.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Neural network
# ─────────────────────────────────────────────────────────────────────────────

"""One-hidden-layer ReLU network: y = w2' * relu(W1 * z + b1) + b2."""
mutable struct NeuralNet
    W1::Matrix{Float64}   # h x d_in
    b1::Vector{Float64}   # h
    w2::Vector{Float64}   # h
    b2::Float64           # scalar
end

"""Pre-allocated gradient buffers matching a NeuralNet's shape.
Owning these per-NN (rather than per-thread) keeps training thread-safe:
each agent's NN and its buffers can be trained concurrently without locks."""
mutable struct NNGradBuffers
    dW1::Matrix{Float64}  # h x d_in  (gradient of W1)
    db1::Vector{Float64}  # h         (gradient of b1)
    dw2::Vector{Float64}  # h         (gradient of w2)
    db2::Base.RefValue{Float64}       # scalar gradient of b2

    # DifferentiationInterface/Enzyme parameter-gradient scratch.
    theta::Vector{Float64}
    dtheta::Vector{Float64}

    # Adam optimizer state, sized lazily to the packed parameter count and
    # persisted across training periods. Warm-starting the moment estimates
    # alongside the weights mirrors the spec's per-period warm start; the
    # per-parameter second moment is what lets training recover the
    # low-curvature interaction/gain directions that vanilla GD starves.
    m::Vector{Float64}                # first moment (mean of gradients)
    v::Vector{Float64}                # second moment (mean of squared gradients)
    adam_t::Base.RefValue{Int}        # Adam timestep for bias correction
end

"""Create zero-initialized gradient buffers matching `nn`."""
function NNGradBuffers(nn::NeuralNet)
    h, d_in = size(nn.W1)
    return NNGradBuffers(
        zeros(h, d_in),
        zeros(h),
        zeros(h),
        Ref(0.0),
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Ref(0),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Current-period match tracking
# ─────────────────────────────────────────────────────────────────────────────

"""A single current-period relationship in an agent's match list."""
struct ActiveMatch
    partner_id::Int
    channel::Symbol      # :self or :broker
end

# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

"""Market participant with a type, prediction model, match history, and satisfaction scores."""
Base.@kwdef mutable struct Agent
    id::Int
    type::Vector{Float64}                    # x_i on the unit sphere, length d

    # Current-period relationships (distinct counterparties; no K-based capacity limit)
    active_matches::Vector{ActiveMatch} = ActiveMatch[]

    # Experience history: d x capacity matrix (column-major, doubling growth)
    # Column j holds the partner type from the j-th match
    history_X::Matrix{Float64}               # d x capacity
    history_q::Vector{Float64}               # realized outputs, matching columns of history_X
    history_count::Int = 0                   # total observations recorded

    # Cumulative history_count at the end of initialization (period 0) and each
    # later period alive; defines the period-based training window.
    obs_period_marks::Vector{Int} = Int[]

    # Neural network and prediction buffer
    nn::NeuralNet
    nn_grad::NNGradBuffers                   # pre-allocated gradient buffers
    predict_buf::Vector{Float64}             # length agent_hidden_width(params)
    n_new_obs::Int = 0                       # observations since last training (for adaptive schedule)
    train_X::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)  # contiguous training scratch
    train_q::Vector{Float64} = Float64[]                     # matching q scratch

    # Per-partner average tracking (direct-indexed by partner agent ID)
    partner_sum::Vector{Float64}             # length N; sum of realized q for matches with partner j
    partner_count::Vector{Int}               # length N; count of matches with partner j

    # Satisfaction indices (EWMA)
    satisfaction_self::Float64 = 0.0
    satisfaction_broker::Float64 = 0.0
    tried_broker::Bool = false

    # Tenure
    periods_alive::Int = 0

    # Cumulative match counter: every accepted match the agent has participated
    # in, any role, any channel.
    # Reset to zero on entry; not decremented on match expiration.
    n_matches_any::Int = 0
end

"""Number of valid history entries for an agent."""
effective_history_size(agent::Agent) = agent.history_count

"""True if the agent currently has a relationship with `partner_id`."""
@inline function has_current_match(agent::Agent, partner_id::Int)::Bool
    @inbounds for am in agent.active_matches
        am.partner_id == partner_id && return true
    end
    return false
end

"""True if the agent currently participates in at least one broker-channel match."""
function has_active_broker_match(agent::Agent)
    any(am -> am.channel == :broker, agent.active_matches)
end

"""Mean realized output with partner j, or NaN if no prior match."""
function partner_mean(agent::Agent, j::Int)
    c = agent.partner_count[j]
    return c > 0 ? agent.partner_sum[j] / c : NaN
end

"""Record a new observation to agent's history, growing the buffer if needed."""
function record_agent_history!(
    agent::Agent, partner_type::AbstractVector{Float64}, q::Float64
)
    agent.history_count += 1
    agent.n_new_obs += 1
    n = agent.history_count
    cap = size(agent.history_X, 2)

    # Grow buffer if needed (doubling strategy)
    if n > cap
        new_cap = max(2 * cap, 16)
        d = size(agent.history_X, 1)
        new_X = Matrix{Float64}(undef, d, new_cap)
        new_X[:, 1:cap] .= agent.history_X
        agent.history_X = new_X
        resize!(agent.history_q, new_cap)
    end

    agent.history_X[:, n] .= partner_type
    agent.history_q[n] = q
    return nothing
end

"""Update per-partner sum and count after a match with partner j."""
function update_partner_mean!(agent::Agent, partner_id::Int, q::Float64)
    agent.partner_sum[partner_id] += q
    agent.partner_count[partner_id] += 1
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Directed offers and accepted relationships
# ─────────────────────────────────────────────────────────────────────────────

"""A directed active-search offer in the shared market."""
struct DirectedOffer
    from_id::Int
    to_id::Int
    channel::Symbol
    predicted_value::Float64
end

"""Accepted directed-offer credit attached to a realized relationship."""
struct OfferCredit
    from_id::Int
    to_id::Int
    channel::Symbol
    predicted_value::Float64
    was_connected::Bool
end

"""Accepted relationship record emitted by the shared offer market."""
struct AcceptedMatch
    demander_id::Int
    counterparty_id::Int
    channel::Symbol
    q_realized::Float64
    q_predicted::Float64
    offer1::Union{OfferCredit,Nothing}
    offer2::Union{OfferCredit,Nothing}
end

# ─────────────────────────────────────────────────────────────────────────────
# Broker
# ─────────────────────────────────────────────────────────────────────────────

"""Single intermediary with a standing roster, current clients, cross-agent history, and prediction model."""
Base.@kwdef mutable struct Broker
    node_id::Int                              # permanent node in G (= N + 1)
    roster::Set{Int} = Set{Int}()             # agent IDs on the broker's roster
    current_clients::Set{Int} = Set{Int}()    # agents outsourcing in the current period

    # Experience history: d x capacity matrices (column-major, doubling growth)
    history_party1_types::Matrix{Float64}
    history_party2_types::Matrix{Float64}
    history_q::Vector{Float64}                # realized outputs
    history_count::Int = 0

    # Cumulative history_count at the end of initialization (period 0) and each
    # later period (period-based training window; see Agent.obs_period_marks).
    obs_period_marks::Vector{Int} = Int[]

    # Neural network
    nn::NeuralNet
    nn_grad::NNGradBuffers
    predict_buf::Vector{Float64}              # length broker_hidden_width(params)
    n_new_obs::Int = 0

    # Pre-allocated training matrix for symmetric broker pair features
    train_X::Matrix{Float64}
    train_q::Vector{Float64}

    # Reputation
    last_reputation::Float64 = 0.0
    has_had_clients::Bool = false
end

"""Number of valid history entries for the broker."""
effective_history_size(broker::Broker) = broker.history_count

"""Record two party types and output in the broker's symmetric match history."""
function record_broker_history!(
    broker::Broker,
    party1_type::AbstractVector{Float64},
    party2_type::AbstractVector{Float64},
    q::Float64,
)
    broker.history_count += 1
    broker.n_new_obs += 1
    n = broker.history_count
    cap = size(broker.history_party1_types, 2)

    # Grow buffers if needed
    if n > cap
        d = size(broker.history_party1_types, 1)
        new_cap = max(2 * cap, 32)
        new_party1_types = Matrix{Float64}(undef, d, new_cap)
        new_party1_types[:, 1:cap] .= broker.history_party1_types
        broker.history_party1_types = new_party1_types
        new_party2_types = Matrix{Float64}(undef, d, new_cap)
        new_party2_types[:, 1:cap] .= broker.history_party2_types
        broker.history_party2_types = new_party2_types
        resize!(broker.history_q, new_cap)

        # Also grow broker feature training buffers
        new_train_cap = max(new_cap, 2 * size(broker.train_X, 2))
        d_broker = size(broker.train_X, 1)
        new_train_X = Matrix{Float64}(undef, d_broker, new_train_cap)
        broker.train_X = new_train_X
        resize!(broker.train_q, new_train_cap)
    end

    broker.history_party1_types[:, n] .= party1_type
    broker.history_party2_types[:, n] .= party2_type
    broker.history_q[n] = q
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Matching environment and calibration
# ─────────────────────────────────────────────────────────────────────────────

"""Matching environment: ideal type c, SPD interaction matrix A, symmetric regime
operator B, and noise scale."""
struct MatchingEnv
    d::Int
    rho::Float64
    c::Vector{Float64}       # ideal type vector
    A::Matrix{Float64}       # SPD interaction matrix (M_A'M_A)
    B::Matrix{Float64}       # symmetric regime operator, weighted-orthogonalized against A
    delta::Float64            # gain strength
    sigma_eps::Float64        # match output noise SD
end

"""Output-scale constants derived from Monte Carlo calibration."""
struct CalibrationConstants
    q_cal::Float64     # calibration reference E[q] (scales r, phi, c_s; not used for initialization)
    r::Float64         # outside option (0.60 * q_cal)
    phi::Float64       # successful broker-placement fee
    c_s::Float64       # self-search cost per demanded relationship position
end

"""Prediction quality metrics: R-squared, bias, and rank correlation."""
struct PredictionQuality
    r_squared::Float64
    bias::Float64
    rank_corr::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# Type curve geometry
# ─────────────────────────────────────────────────────────────────────────────

"""Sinusoidal curve on the unit sphere with s active dimensions out of d.
Agent types are drawn at random positions on this curve, then perturbed."""
struct CurveGeometry
    d::Int
    s::Int                        # active dimensions (1..s have nonzero curve amplitude)
    freqs::Vector{Int}            # per-dimension integer frequencies (length s), from U{1,...,5}
    phases::Vector{Float64}       # per-dimension phases (length s), from U[0, 2π)
end

# ─────────────────────────────────────────────────────────────────────────────
# Period accumulators
# ─────────────────────────────────────────────────────────────────────────────

"""Per-period counters and output vectors. All fields reset each tick."""
Base.@kwdef mutable struct PeriodAccumulators
    # Match counts by channel
    n_self_matches::Int = 0
    n_broker_matches::Int = 0

    # Realized output by channel
    q_self::Vector{Float64} = Float64[]
    q_broker::Vector{Float64} = Float64[]

    # Access vs assessment decomposition
    access_count::Int = 0       # counterparty was NOT a neighbor of demander
    assessment_count::Int = 0   # counterparty WAS a neighbor

    # Outsourcing rate and demand
    n_demanders::Int = 0
    n_outsourced::Int = 0           # demanders who chose the broker channel
    outsourced_slots::Int = 0       # requested positions routed through the broker channel
    total_demand::Int = 0           # total requested positions across all demanders

    # Prediction/outcome pairs from actual matches (subject to selection bias)
    agent_predicted::Vector{Float64} = Float64[]
    agent_realized::Vector{Float64} = Float64[]
    broker_predicted::Vector{Float64} = Float64[]
    broker_realized::Vector{Float64} = Float64[]

    # Holdout: per-agent averaged over sampled agents (both agent and broker
    # evaluated on the same per-agent partner sets for comparability)
    agent_holdout_r2::Float64 = NaN
    agent_holdout_bias::Float64 = NaN
    agent_holdout_rank::Float64 = NaN
    agent_holdout_rmse::Float64 = NaN
    broker_holdout_r2::Float64 = NaN
    broker_holdout_bias::Float64 = NaN
    broker_holdout_rank::Float64 = NaN
    broker_holdout_rmse::Float64 = NaN

    # Roster
    roster_size::Int = 0
    broker_access_size::Int = 0

    # Current-period counterparty concentration
    median_counterparties::Float64 = NaN
    max_counterparties::Int = 0

    # Agent-network degree summaries recorded before entry/exit turnover.
    agent_degrees::Vector{Int} = Int[]
    mean_degree::Float64 = NaN
    median_degree::Float64 = NaN
    min_degree::Float64 = NaN
    max_degree::Float64 = NaN
end

"""Zero all per-period fields while preserving vector capacity for reuse."""
function reset_accumulators!(a::PeriodAccumulators)
    a.n_self_matches = 0
    a.n_broker_matches = 0
    empty!(a.q_self)
    empty!(a.q_broker)
    a.access_count = 0
    a.assessment_count = 0
    a.n_demanders = 0
    a.n_outsourced = 0
    a.outsourced_slots = 0
    a.total_demand = 0
    empty!(a.agent_predicted)
    empty!(a.agent_realized)
    empty!(a.broker_predicted)
    empty!(a.broker_realized)
    a.agent_holdout_r2 = NaN
    a.agent_holdout_bias = NaN
    a.agent_holdout_rank = NaN
    a.agent_holdout_rmse = NaN
    a.broker_holdout_r2 = NaN
    a.broker_holdout_bias = NaN
    a.broker_holdout_rank = NaN
    a.broker_holdout_rmse = NaN
    a.roster_size = 0
    a.broker_access_size = 0
    a.median_counterparties = NaN
    a.max_counterparties = 0
    empty!(a.agent_degrees)
    a.mean_degree = NaN
    a.median_degree = NaN
    a.min_degree = NaN
    a.max_degree = NaN
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Network measures cache
# ─────────────────────────────────────────────────────────────────────────────

"""Broker's network position measures, recomputed periodically."""
mutable struct CachedNetworkMeasures
    betweenness::Float64      # standard Freeman betweenness (broker node in G)
    constraint::Float64       # Burt's constraint (broker's ego network)
    effective_size::Float64   # Burt's effective size (non-redundant contacts)
end

CachedNetworkMeasures() = CachedNetworkMeasures(0.0, 1.0, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Model parameters
# ─────────────────────────────────────────────────────────────────────────────

"""Immutable simulation parameters for the unimodal matching model."""
struct ModelParams
    # Population and types
    N::Int                       # agent count (default 1000)
    d::Int                       # type dimensionality (fixed at 8)
    s::Int                       # active dimensions of type curve (default 8; swept {2,4,6,8})

    # Matching function
    rho::Float64                 # quality-interaction mixing weight (default 0.50)
    delta::Float64               # regime gain strength (default 0.5)
    sigma_x::Float64             # type noise scale (default 0.5)
    sigma_eps::Float64           # match output noise SD (default 0.10)

    # Match accounting
    K::Int                       # maximum active demands per period (default 5)
    p_demand::Float64            # per-position demand probability (default 0.50)

    # Network
    k::Int                       # Watts-Strogatz ring lattice degree (default 6)
    p_rewire::Float64            # rewiring probability (default 0.1)

    # Economics
    omega::Float64               # satisfaction recency weight (default 0.2)
    search_cost_rate::Float64    # shared friction rate: phi = c_s = search_cost_rate*q_cal (default 0.05)
    reservation_frac::Float64    # outside option as a fraction of q_cal: r = reservation_frac*q_cal (default 0.60; may exceed 1)

    # Neural network
    eta_lr::Float64              # Adam learning rate (default 0.01)
    E_init::Int                  # initial training steps (default 200)
    train_window_periods::Int    # training look-back horizon, in periods (default 40)
    train_max_obs::Int           # per-call observation cap; window subsampled to this (default 2000)
    train_steps::Int             # min update steps per period; floor of the adaptive schedule (default 100)

    # Search
    roster_frac::Float64         # standing broker roster share (default 0.20)
    n_strangers::Int             # period-level stranger pool size (default 10)
    eta::Float64                 # agent entry/exit rate (default 0.02)
    roster_churn::Float64        # standing-roster exogenous churn probability (default 0.02)

    # Simulation
    network_measure_interval::Int # M (default 20)
    T::Int                       # total periods (default 500)
    seed::Int                    # RNG seed
end

# ─────────────────────────────────────────────────────────────────────────────
# Model state
# ─────────────────────────────────────────────────────────────────────────────

"""Exact symmetric index of current-period agent-agent relationships."""
Base.@kwdef mutable struct CurrentMatchWorkspace
    mask::Matrix{Bool} = Matrix{Bool}(undef, 0, 0)
    touched::Vector{Int} = Int[]
end

"""Reusable buffers for self-search candidate construction and ranking."""
Base.@kwdef mutable struct SearchWorkspace
    neighbor_ids::Vector{Int} = Int[]
    neighbor_evals::Vector{Float64} = Float64[]
    # Neighbor bitset: nbr_mask[j] = true iff j is a neighbor of the current agent.
    # Length N+1 (extra slot for the broker node). Reset after each self-search call.
    nbr_mask::Vector{Bool} = Bool[]
    # Tracks which indices we set in nbr_mask this call, so we can clear only those.
    nbr_marked::Vector{Int} = Int[]
    period_strangers::Vector{Int} = Int[]
    # Sorted greedy: pre-allocated (negated_val, flat_index) pairs, sorted in-place.
    sort_pairs::Vector{Tuple{Float64,Int}} = Tuple{Float64,Int}[]
end

"""Reusable buffers for broker access deduplication, pair scoring, and batch prediction."""
Base.@kwdef mutable struct BrokerPairWorkspace
    access_seen::Vector{Bool} = Bool[]
    access_touched::Vector{Int} = Int[]  # sparse-clear markers for broker access deduplication
    # Batched prediction scratch
    Z_batch::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)  # broker feature input
    H_batch::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)  # h x n_pairs hidden
    Y_batch::Vector{Float64} = Float64[]                      # n_pairs output
    period_broker_demanders::Vector{Int} = Int[]
    period_broker_access_ids::Vector{Int} = Int[]
    broker_pair_scores::Vector{Tuple{Float64,Int,Int}} = Tuple{Float64,Int,Int}[]
    broker_top_counts::Vector{Int} = Int[]
    broker_top_offers::Matrix{Tuple{Float64,Int,Int,Int,Int}} = Matrix{
        Tuple{Float64,Int,Int,Int,Int}
    }(
        undef, 0, 0
    )
    broker_selected_offers::Vector{Tuple{Float64,Int,Int,Int,Int}} = Tuple{
        Float64,Int,Int,Int,Int
    }[]
    broker_demander_mask::Vector{Bool} = Bool[]
    broker_demander_touched::Vector{Int} = Int[]
    broker_access_mask::Vector{Bool} = Bool[]
    broker_access_touched::Vector{Int} = Int[]
end

"""Directed-offer book keyed by ordered pairs and iterated by unordered pairs."""
Base.@kwdef mutable struct OfferBook
    offers::Vector{DirectedOffer} = DirectedOffer[]
    offer_index::Matrix{Int} = Matrix{Int}(undef, 0, 0)
    offer_index_touched::Vector{Int} = Int[]
    offer_pairs::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]
end

"""Reusable period ledger for demand, satisfaction, and match buffers."""
Base.@kwdef mutable struct PeriodLedger
    demand_agent_ids::Vector{Int} = Int[]       # agents with demand
    demand_channels::Vector{Symbol} = Symbol[]  # channel per demander
    demand_counts::Vector{Int} = Int[]          # d_i per demander
    broker_clients_ws::Vector{Int} = Int[]
    demander_q_sum::Vector{Float64} = Float64[]   # realized output by demander id
    broker_match_count::Vector{Int} = Int[]       # successful broker matches by demander id
    accepted_matches::Vector{AcceptedMatch} = AcceptedMatch[]
    offer_remaining::Vector{Int} = Int[]
end

"""Reusable deterministic holdout-diagnostics buffers."""
Base.@kwdef mutable struct HoldoutWorkspace
    z_buf::Vector{Float64} = Float64[]
    agent_preds::Vector{Float64} = Float64[]
    agent_trues::Vector{Float64} = Float64[]
    broker_preds::Vector{Float64} = Float64[]
    agent_ids::Vector{Int} = Int[]
    partner_ids::Vector{Int} = Int[]
    pred_order::Vector{Int} = Int[]
    true_order::Vector{Int} = Int[]
    pred_ranks::Vector{Float64} = Float64[]
    true_ranks::Vector{Float64} = Float64[]
end

"""Reusable buffers for deterministic and stochastic match-output calculations."""
Base.@kwdef mutable struct MatchOutputWorkspace
    Ax_buf::Vector{Float64} = Float64[]
    Bx_buf::Vector{Float64} = Float64[]
end

"""
Pre-allocated per-step scratch buffers reused across agents and calls.
Sub-workspaces keep mutable scratch ownership local to the subsystem that uses it.
"""
Base.@kwdef mutable struct SimWorkspace
    current_matches::CurrentMatchWorkspace = CurrentMatchWorkspace()
    search::SearchWorkspace = SearchWorkspace()
    broker_pairs::BrokerPairWorkspace = BrokerPairWorkspace()
    offer_book::OfferBook = OfferBook()
    ledger::PeriodLedger = PeriodLedger()
    holdout::HoldoutWorkspace = HoldoutWorkspace()
    match_output::MatchOutputWorkspace = MatchOutputWorkspace()
end

"""Complete simulation state: all agents, broker, network, environment, and accumulators."""
Base.@kwdef mutable struct ModelState
    params::ModelParams
    rng::StableRNG
    period::Int = 0
    env::MatchingEnv
    cal::CalibrationConstants
    curve_geo::CurveGeometry
    agents::Vector{Agent}
    broker::Broker
    G::SimpleGraph{Int}                       # N+1 nodes: agents 1:N, broker at N+1
    accum::PeriodAccumulators = PeriodAccumulators()
    cached_network::CachedNetworkMeasures = CachedNetworkMeasures()
    workspace::SimWorkspace = SimWorkspace()
end

"""Clear the current-period match index using sparse touched coordinates."""
function reset_current_match_index!(ws::SimWorkspace, N::Int)
    current = ws.current_matches
    mask = current.mask
    touched = current.touched
    if size(mask, 1) != N || size(mask, 2) != N
        current.mask = falses(N, N)
        empty!(touched)
        return current.mask
    end
    @inbounds for idx in touched
        mask[idx] = false
    end
    empty!(touched)
    return mask
end

"""Mark agents `i` and `j` as current-period counterparties in the workspace index."""
@inline function mark_current_match!(ws::SimWorkspace, i::Int, j::Int)
    current = ws.current_matches
    mask = current.mask
    touched = current.touched
    @inbounds if !mask[i, j]
        mask[i, j] = true
        mask[j, i] = true
        n = size(mask, 1)
        push!(touched, i + (j - 1) * n)
        push!(touched, j + (i - 1) * n)
    end
    return nothing
end

"""Rebuild the workspace current-period match index from agents' active matches."""
function rebuild_current_match_index!(ws::SimWorkspace, agents::Vector{Agent})
    N = length(agents)
    reset_current_match_index!(ws, N)
    @inbounds for i in 1:N
        for am in agents[i].active_matches
            j = am.partner_id
            1 <= j <= N || continue
            mark_current_match!(ws, i, j)
        end
    end
    return nothing
end

"""True if the workspace current-pair index marks agents `i` and `j` as matched."""
@inline function has_current_match(ws::SimWorkspace, i::Int, j::Int)::Bool
    return @inbounds ws.current_matches.mask[i, j]
end
