"""
    diagnostics.jl

State snapshots for debugging and inspection.
"""

using Statistics: mean

"""
    diagnostic_summary(state) -> Dict{String, Any}

Snapshot of key state variables for debugging. Quick-inspect Dict suitable
for printing or logging during development.
"""
function diagnostic_summary(state::ModelState)::Dict{String,Any}
    agents = state.agents
    broker = state.broker
    N = state.params.N
    agent_hist = [a.history_count for a in agents]

    return Dict{String,Any}(
        "period" => state.period,
        "N" => N,
        "n_active_matches" => sum(length(a.active_matches) for a in agents) ÷ 2,
        "median_counterparties" => state.accum.median_counterparties,
        "max_counterparties" => state.accum.max_counterparties,
        "mean_agent_history" => mean(agent_hist),
        "max_agent_history" => maximum(agent_hist),
        "broker_history_size" => broker.history_count,
        "broker_roster_size" => length(broker.roster),
        "broker_access_size" => broker_access_size(broker),
        "broker_reputation" => broker.last_reputation,
        "broker_has_had_clients" => broker.has_had_clients,
        "capture_ready" => state.accum.capture_ready,
        "capture_scaled_mae" => state.accum.capture_scaled_mae,
        "captured_origin_count" => state.accum.captured_origin_count,
        "captured_position_count" => state.accum.captured_position_count,
        "principal_accepted" => state.accum.n_broker_principal,
        "principal_rejected" => state.accum.principal_rejected,
        "mean_satisfaction_self" => mean(a.satisfaction_self for a in agents),
        "mean_satisfaction_broker" => mean(a.satisfaction_broker for a in agents),
        "mean_periods_alive" => mean(a.periods_alive for a in agents),
        "n_on_roster" => length(state.broker.roster),
        "betweenness" => state.cached_network.betweenness,
        "constraint" => state.cached_network.constraint,
        "effective_size" => state.cached_network.effective_size,
        "q_cal" => state.cal.q_cal,
        "r" => state.cal.r,
        "phi" => state.cal.phi,
        "mad_f" => state.cal.mad_f,
    )
end
