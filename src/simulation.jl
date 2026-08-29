"""
    simulation.jl

Simulation runner and per-period metric collection.
"""

using DataFrames: DataFrame
using Graphs: degree
using Statistics: mean

"""Safe mean that returns NaN on empty vectors."""
safe_mean(v) = isempty(v) ? NaN : mean(v)

"""Fill `degrees` with sorted agent-node degrees, excluding the broker node."""
function fill_agent_node_degrees!(degrees::Vector{Int}, state::ModelState)
    N = state.params.N
    length(degrees) == N || resize!(degrees, N)
    @inbounds for agent in state.agents
        degrees[agent.id] = degree(state.G, agent.id)
    end
    sort!(degrees)
    return degrees
end

"""Sorted agent-node degrees for the current graph `G`, excluding the broker node."""
function agent_node_degrees(state::ModelState)
    return fill_agent_node_degrees!(Vector{Int}(undef, state.params.N), state)
end

"""Summary statistics for an already sorted agent-degree vector."""
function summarize_sorted_degrees(degrees::Vector{Int})
    n = length(degrees)
    mid = n ÷ 2
    median_degree =
        isodd(n) ? Float64(degrees[mid + 1]) : (degrees[mid] + degrees[mid + 1]) / 2

    return (
        mean_degree=mean(degrees),
        median_degree=median_degree,
        min_degree=Float64(first(degrees)),
        max_degree=Float64(last(degrees)),
    )
end

"""
    degree_summary(state) -> NamedTuple

Agent-node degree summary statistics for the current graph `G`, excluding the
broker node from the distribution used for the median and other summaries.
"""
function degree_summary(state::ModelState)
    degrees = agent_node_degrees(state)
    return summarize_sorted_degrees(degrees)
end

"""Record pre-turnover agent-degree distribution and summaries in period accumulators."""
function record_agent_degree_summary!(state::ModelState)
    a = state.accum
    fill_agent_node_degrees!(a.agent_degrees, state)
    stats = summarize_sorted_degrees(a.agent_degrees)
    a.mean_degree = stats.mean_degree
    a.median_degree = stats.median_degree
    a.min_degree = stats.min_degree
    a.max_degree = stats.max_degree
    return stats
end

"""
    collect_period_metrics(state) -> NamedTuple

Collect the current period's match, prediction, satisfaction, and network
metrics from the simulation state.
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
    selected_rank_rng = diagnostics_rng(p.seed, state.period, 0x4a8f3c21)
    agent_sel = compute_prediction_quality(
        a.agent_predicted, a.agent_realized, selected_rank_rng; sigma_eps=se
    )
    broker_sel = compute_prediction_quality(
        a.broker_predicted, a.broker_realized, selected_rank_rng; sigma_eps=se
    )
    agent_sel_rmse = if isempty(a.agent_predicted)
        NaN
    else
        sqrt(mean((a.agent_predicted .- a.agent_realized) .^ 2))
    end
    broker_sel_rmse = if isempty(a.broker_predicted)
        NaN
    else
        sqrt(mean((a.broker_predicted .- a.broker_realized) .^ 2))
    end

    # Agent-level stats
    mean_sat_self = mean(ag.satisfaction_self for ag in agents)
    mean_sat_broker = mean(ag.satisfaction_broker for ag in agents)
    degree_stats = degree_summary(state)

    return (
        period=state.period,
        # Match counts
        n_self_matches=a.n_self_matches,
        n_broker_matches=a.n_broker_matches,
        n_total_matches=(a.n_self_matches + a.n_broker_matches),
        # Match quality
        q_self_mean=safe_mean(a.q_self),
        q_broker_mean=safe_mean(a.q_broker),
        # Outsourcing
        n_demanders=a.n_demanders,
        n_outsourced=a.n_outsourced,
        outsourced_slots=a.outsourced_slots,
        total_demand=a.total_demand,
        outsourcing_rate=a.total_demand > 0 ? a.outsourced_slots / a.total_demand : 0.0,
        outsourcing_rate_demanders=a.n_demanders > 0 ? a.n_outsourced / a.n_demanders : 0.0,
        # Access vs assessment
        access_count=a.access_count,
        assessment_count=a.assessment_count,
        # Holdout prediction quality (per-agent averaged)
        agent_holdout_r2=a.agent_holdout_r2,
        agent_holdout_bias=a.agent_holdout_bias,
        agent_holdout_rank=a.agent_holdout_rank,
        agent_holdout_rmse=a.agent_holdout_rmse,
        broker_holdout_r2=a.broker_holdout_r2,
        broker_holdout_bias=a.broker_holdout_bias,
        broker_holdout_rank=a.broker_holdout_rank,
        broker_holdout_rmse=a.broker_holdout_rmse,
        r2_gap=(a.broker_holdout_r2 - a.agent_holdout_r2),
        rank_gap=(a.broker_holdout_rank - a.agent_holdout_rank),
        rmse_gap=(a.agent_holdout_rmse - a.broker_holdout_rmse),  # positive = broker more accurate
        # Selected-sample prediction quality (pooled over actual matches)
        agent_selected_rank=agent_sel.rank_corr,
        agent_selected_r2=agent_sel.r_squared,
        agent_selected_rmse=agent_sel_rmse,
        agent_selected_bias=agent_sel.bias,
        broker_selected_rank=broker_sel.rank_corr,
        broker_selected_r2=broker_sel.r_squared,
        broker_selected_rmse=broker_sel_rmse,
        broker_selected_bias=broker_sel.bias,
        # Broker state
        broker_reputation=broker.last_reputation,
        roster_size=a.roster_size,
        broker_access_size=a.broker_access_size,
        broker_history_size=broker.history_count,
        # Satisfaction
        mean_satisfaction_self=mean_sat_self,
        mean_satisfaction_broker=mean_sat_broker,
        # Counterparty concentration
        median_counterparties=a.median_counterparties,
        max_counterparties=a.max_counterparties,
        # Whole-network degree summaries
        mean_degree=degree_stats.mean_degree,
        median_degree=degree_stats.median_degree,
        min_degree=degree_stats.min_degree,
        max_degree=degree_stats.max_degree,
        # Network measures
        betweenness=state.cached_network.betweenness,
        constraint=state.cached_network.constraint,
        effective_size=state.cached_network.effective_size,
    )
end

"""
    run_simulation(params; verify=false) -> (ModelState, DataFrame)

Initialize the model and run for `params.T` periods. Returns the final state and
the per-period metrics DataFrame.
"""
function run_simulation(params::ModelParams; verify::Bool=false)
    state = initialize_model(params)
    rows = NamedTuple[]
    sizehint!(rows, params.T)

    for t in 1:params.T
        metrics = step_period!(state)
        verify && verify_invariants(state)
        push!(rows, metrics)
    end

    df = DataFrame(rows)
    return (state, df)
end
