"""
    simulation.jl

Simulation runner and per-period metric collection.
"""

using DataFrames: DataFrame
using Graphs: degree
using Statistics: mean, quantile
using StatsBase: corspearman

"""Safe mean that returns NaN on empty vectors."""
safe_mean(v) = isempty(v) ? NaN : mean(v)

"""Gini coefficient of a non-empty, non-negative vector. Returns 0 for empty or
constant-zero input. Uses the sorted-order definition:
    G = (2 Σ_{i=1}^n i · y_(i)) / (n · Σ y) − (n + 1) / n
(Dorfman 1979; matches standard inequality-measurement conventions.)"""
function gini(v::AbstractVector{<:Real})
    isempty(v) && return 0.0
    s = sum(v)
    s <= 0.0 && return 0.0
    sorted = sort(collect(v))
    n = length(sorted)
    acc = 0.0
    @inbounds for i in 1:n
        acc += i * sorted[i]
    end
    return (2.0 * acc) / (n * s) - (n + 1) / n
end

"""90th percentile of a non-empty vector; NaN if empty."""
p90(v::AbstractVector{<:Real}) = isempty(v) ? NaN : quantile(v, 0.9)

"""Agent-node degree summary statistics for the current graph `G`, excluding the broker."""
function degree_summary(state::ModelState)
    degrees = degree(state.G)[1:state.params.N]
    sort!(degrees)
    n = length(degrees)
    mid = n ÷ 2
    median_degree = isodd(n) ? Float64(degrees[mid + 1]) :
        (degrees[mid] + degrees[mid + 1]) / 2

    return (
        mean_degree = mean(degrees),
        median_degree = median_degree,
        min_degree = Float64(first(degrees)),
        max_degree = Float64(last(degrees)),
    )
end

"""
    collect_period_metrics(state) -> NamedTuple

Collect all per-period metrics from the state and accumulators.
"""
function collect_period_metrics(state::ModelState)
    p = state.params
    a = state.accum
    agents = state.agents
    broker = state.broker
    N = p.N

    # Prediction quality: holdout is per-agent averaged, computed in step.jl.
    # Selected-sample metrics are pooled over actual matches by channel.
    se = state.env.sigma_eps
    agent_sel = compute_prediction_quality(a.agent_predicted, a.agent_realized; sigma_eps=se)
    broker_sel = compute_prediction_quality(a.broker_predicted, a.broker_realized; sigma_eps=se)
    agent_sel_rmse = isempty(a.agent_predicted) ? NaN :
        sqrt(mean((a.agent_predicted .- a.agent_realized).^2))
    broker_sel_rmse = isempty(a.broker_predicted) ? NaN :
        sqrt(mean((a.broker_predicted .- a.broker_realized).^2))
    broker_sel_mae = isempty(a.broker_predicted) ? NaN :
        mean(abs.(a.broker_predicted .- a.broker_realized))

    # ── Capture outcome and decision quality (§12i) ──
    # Δq_ij = realized position value minus q̄_j for all acquired positions, placed or unplaced.
    n_principal = length(a.q_broker_principal)
    n_capture = length(a.capture_realized)
    capture_delta = n_capture == 0 ? Float64[] :
        a.capture_realized .- a.capture_ask
    capture_surplus_mean = isempty(capture_delta) ? NaN : mean(capture_delta)
    n_loss = count(<(0.0), capture_delta)
    capture_loss_rate = n_capture == 0 ? NaN : n_loss / n_capture
    capture_loss_magnitude = n_loss == 0 ? NaN :
        mean(abs(d) for d in capture_delta if d < 0.0)

    # Capture decision quality: Spearman ρ and RMSE on the principal-mode subset.
    # NaN when fewer than 5 acquired positions to keep the metric comparable to selected_r2.
    if n_capture >= 5
        expected_delta = a.capture_qhat .- a.capture_ask
        capture_decision_rank = corspearman(expected_delta, capture_delta)
        capture_decision_rmse = sqrt(mean((a.capture_qhat .- a.capture_realized) .^ 2))
    else
        capture_decision_rank = NaN
        capture_decision_rmse = NaN
    end

    # Supply scarcity: distinct counterparties acquired in principal mode this period.
    supply_scarcity = length(a.principal_acquired_ids) / N

    # ── Broker dependency D_j (§12i) ──
    # Cumulative: D_j = n_principal_acquired / n_matches_any for agents with ≥ 1 match.
    # Recomputed per-period from current agent state.
    dep = Float64[]
    sizehint!(dep, N)
    @inbounds for ag in agents
        if ag.n_matches_any > 0
            push!(dep, ag.n_principal_acquired / ag.n_matches_any)
        end
    end
    if isempty(dep)
        dep_mean = NaN
        dep_p90 = NaN
        dep_frac_above_half = NaN
        dep_gini = NaN
    else
        dep_mean = mean(dep)
        dep_p90 = p90(dep)
        dep_frac_above_half = count(>(0.5), dep) / length(dep)
        dep_gini = gini(dep)
    end

    # Agent-level stats
    n_available = count(ag -> available_capacity(ag, p.K) > 0, agents)
    mean_sat_self = mean(ag.satisfaction_self for ag in agents)
    mean_sat_broker = mean(ag.satisfaction_broker for ag in agents)
    degree_stats = degree_summary(state)

    return (
        period = state.period,
        # Match counts
        n_self_matches = a.n_self_matches,
        n_broker_standard = a.n_broker_standard,
        n_broker_principal = a.n_broker_principal,
        n_total_matches = a.n_self_matches + a.n_broker_standard + a.n_broker_principal,
        # Match quality
        q_self_mean = safe_mean(a.q_self),
        q_broker_standard_mean = safe_mean(a.q_broker_standard),
        q_broker_principal_mean = safe_mean(a.q_broker_principal),
        # Outsourcing
        n_demanders = a.n_demanders,
        n_outsourced = a.n_outsourced,
        outsourced_slots = a.outsourced_slots,
        total_demand = a.total_demand,
        outsourcing_rate = a.total_demand > 0 ? a.outsourced_slots / a.total_demand : 0.0,
        outsourcing_rate_demanders = a.n_demanders > 0 ? a.n_outsourced / a.n_demanders : 0.0,
        # Access vs assessment
        access_count = a.access_count,
        assessment_count = a.assessment_count,
        # Prediction quality (holdout)
        # Holdout prediction quality (per-agent averaged)
        agent_holdout_r2 = a.agent_holdout_r2,
        agent_holdout_bias = a.agent_holdout_bias,
        agent_holdout_rank = a.agent_holdout_rank,
        agent_holdout_rmse = a.agent_holdout_rmse,
        broker_holdout_r2 = a.broker_holdout_r2,
        broker_holdout_bias = a.broker_holdout_bias,
        broker_holdout_rank = a.broker_holdout_rank,
        broker_holdout_rmse = a.broker_holdout_rmse,
        r2_gap = a.broker_holdout_r2 - a.agent_holdout_r2,
        rank_gap = a.broker_holdout_rank - a.agent_holdout_rank,
        rmse_gap = a.agent_holdout_rmse - a.broker_holdout_rmse,  # positive = broker more accurate
        # Selected-sample prediction quality (pooled over actual matches)
        agent_selected_rank = agent_sel.rank_corr,
        agent_selected_r2 = agent_sel.r_squared,
        agent_selected_rmse = agent_sel_rmse,
        agent_selected_bias = agent_sel.bias,
        broker_selected_rank = broker_sel.rank_corr,
        broker_selected_r2 = broker_sel.r_squared,
        broker_selected_rmse = broker_sel_rmse,
        broker_selected_mae = broker_sel_mae,
        broker_selected_bias = broker_sel.bias,
        broker_confidence_mae = a.broker_confidence_mae,
        # Broker state
        broker_reputation = broker.last_reputation,
        roster_size = a.roster_size,
        broker_access_size = a.broker_access_size,
        broker_history_size = broker.history_count,
        # Capture metrics (Model 1)
        principal_mode_share = (a.n_broker_standard + a.n_broker_principal) > 0 ?
            a.n_broker_principal / (a.n_broker_standard + a.n_broker_principal) : 0.0,
        # Capture outcome (§12i)
        capture_surplus_mean = capture_surplus_mean,
        capture_loss_rate = capture_loss_rate,
        capture_loss_magnitude = capture_loss_magnitude,
        # Capture decision quality (§12i)
        capture_decision_rank = capture_decision_rank,
        capture_decision_rmse = capture_decision_rmse,
        # Supply scarcity (§12i)
        supply_scarcity = supply_scarcity,
        # Broker dependency (§12i): cumulative D_j summary stats
        broker_dependency_mean = dep_mean,
        broker_dependency_p90 = dep_p90,
        broker_dependency_frac_above_half = dep_frac_above_half,
        broker_dependency_gini = dep_gini,
        # Satisfaction
        mean_satisfaction_self = mean_sat_self,
        mean_satisfaction_broker = mean_sat_broker,
        # Market state
        n_available = n_available,
        # Whole-network degree summaries
        mean_degree = degree_stats.mean_degree,
        median_degree = degree_stats.median_degree,
        min_degree = degree_stats.min_degree,
        max_degree = degree_stats.max_degree,
        # Network measures
        betweenness = state.cached_network.betweenness,
        constraint = state.cached_network.constraint,
        effective_size = state.cached_network.effective_size,
    )
end

"""
    run_simulation(params; verify=false, sort_by_pc1=false) -> (ModelState, DataFrame)

Initialize the model and run for T periods. Returns final state and metrics DataFrame.
"""
function run_simulation(params::ModelParams; verify::Bool = false, sort_by_pc1::Bool = false)
    state = initialize_model(params; sort_by_pc1=sort_by_pc1)
    rows = NamedTuple[]
    sizehint!(rows, params.T)

    for t in 1:params.T
        step_period!(state)
        verify && verify_invariants(state)
        push!(rows, collect_period_metrics(state))
    end

    df = DataFrame(rows)
    return (state, df)
end
