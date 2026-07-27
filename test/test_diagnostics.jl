using Test
using BrokerageABM

@testset "Diagnostics" begin
    state = initialize_model(default_params(N=30, T=5, T_burn=1, seed=42))
    step_period!(state)
    summary = diagnostic_summary(state)

    expected_keys = Set([
        "period",
        "N",
        "n_active_matches",
        "median_counterparties",
        "max_counterparties",
        "mean_agent_history",
        "max_agent_history",
        "broker_history_size",
        "broker_roster_size",
        "broker_access_size",
        "broker_reputation",
        "broker_has_had_clients",
        "mean_satisfaction_self",
        "mean_satisfaction_broker",
        "mean_periods_alive",
        "n_on_roster",
        "betweenness",
        "constraint",
        "effective_size",
        "q_cal",
        "r",
        "phi",
    ])

    @test Set(keys(summary)) == expected_keys
    @test summary["period"] == state.period
    @test summary["N"] == state.params.N
    @test summary["broker_roster_size"] == length(state.broker.roster)
    @test summary["broker_access_size"] ==
        BrokerageABM.broker_access_size(state.broker)
    @test summary["n_on_roster"] == length(state.broker.roster)
    expected_types = Dict(
        "period" => Int,
        "broker_has_had_clients" => Bool,
        "median_counterparties" => Float64,
        "max_counterparties" => Int,
        "mean_satisfaction_self" => Float64,
        "mean_satisfaction_broker" => Float64,
        "betweenness" => Float64,
    )
    @test all(isa(summary[key], value_type) for (key, value_type) in expected_types)
end
