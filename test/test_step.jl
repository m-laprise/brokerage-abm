using Test
using BrokerageABM
using BrokerageABM: ActiveMatch
using StableRNGs: StableRNG
using Graphs: SimpleGraph, add_edge!, degree, has_edge

@testset "Step and Simulation" begin
    @testset "step_period! advances period counter" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42)
        state = initialize_model(p)
        @test state.period == 0
        step_period!(state)
        @test state.period == 1
    end

    @testset "Agent retraining schedule alternates by parity" begin
        @test BrokerageABM.agent_retrains_this_period(1, 1)
        @test !BrokerageABM.agent_retrains_this_period(2, 1)
        @test !BrokerageABM.agent_retrains_this_period(1, 2)
        @test BrokerageABM.agent_retrains_this_period(2, 2)
    end

    @testset "Current-period match ledger resets before demand generation" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42)
        state = initialize_model(p)
        push!(state.agents[1].active_matches, ActiveMatch(0, :self))
        step_period!(state)
        @test all(am.partner_id != 0 for a in state.agents for am in a.active_matches)
    end

    @testset "Roster size stays fixed at the target" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42, eta=0.0)
        target = BrokerageABM.roster_target_size(p.N)
        _, df = run_simulation(p)
        @test all(df.roster_size .== target)
    end

    @testset "Roster composition changes under churn" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42, eta=0.0, roster_churn=0.5)
        state = initialize_model(p)
        roster_before = copy(state.broker.roster)
        for _ in 1:3
            step_period!(state)
        end
        @test state.broker.roster != roster_before
    end

    @testset "Current broker clients receive broker edges" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42)
        state = initialize_model(p)
        client_id = first(i for i in 1:p.N if i ∉ state.broker.roster)

        push!(state.broker.current_clients, client_id)
        BrokerageABM.sync_broker_edges!(state.G, state.agents, state.broker)

        @test has_edge(state.G, client_id, state.broker.node_id)
    end

    @testset "Broker access size uses the hybrid union without double counting" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42)
        state = initialize_model(p)
        roster_member = first(state.broker.roster)
        outside_member = first(i for i in 1:p.N if i ∉ state.broker.roster)

        push!(state.broker.current_clients, roster_member)
        push!(state.broker.current_clients, outside_member)

        @test BrokerageABM.broker_access_size(state.broker) ==
            length(state.broker.roster) + 1
    end

    @testset "collect_period_metrics returns valid NamedTuple" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42)
        state = initialize_model(p)
        metrics = step_period!(state)
        @test metrics.period == 1
        @test metrics.n_total_matches >= 0
        @test 0.0 <= metrics.outsourcing_rate <= 1.0
        @test 0.0 <= metrics.outsourcing_rate_demanders <= 1.0
        @test 0 <= metrics.outsourced_slots <= metrics.total_demand
        @test isfinite(metrics.mean_satisfaction_self)
        @test isfinite(metrics.mean_satisfaction_broker)
        @test metrics.median_counterparties >= 0
        @test metrics.max_counterparties >= metrics.median_counterparties
        @test metrics.broker_access_size >= metrics.roster_size
        @test metrics.broker_access_size <= p.N
    end

    @testset "collect_period_metrics reports agent-only degree summaries from G" begin
        p = default_params(N=51, T=10, T_burn=2, seed=42, eta=0.0)
        state = initialize_model(p)
        metrics = step_period!(state)

        degrees = sort(degree(state.G)[1:p.N])
        n = length(degrees)
        expected_median = if isodd(n)
            Float64(degrees[n ÷ 2 + 1])
        else
            (degrees[n ÷ 2] + degrees[n ÷ 2 + 1]) / 2
        end

        @test metrics.mean_degree ≈ sum(degrees) / n
        @test metrics.median_degree ≈ expected_median
        @test metrics.min_degree == first(degrees)
        @test metrics.max_degree == last(degrees)
    end

    @testset "step_period! returns pre-turnover measurements" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42, eta=0.95)
        state = initialize_model(p)
        metrics = step_period!(state)
        post_turnover_degrees = BrokerageABM.degree_summary(state)
        collected_after_turnover = collect_period_metrics(state)

        @test metrics.mean_degree == state.accum.mean_degree
        @test metrics.median_degree == state.accum.median_degree
        @test metrics.min_degree == state.accum.min_degree
        @test metrics.max_degree == state.accum.max_degree
        @test collected_after_turnover.mean_degree == post_turnover_degrees.mean_degree
        @test collected_after_turnover.max_degree == post_turnover_degrees.max_degree
        @test (
            metrics.mean_degree != post_turnover_degrees.mean_degree ||
            metrics.max_degree != post_turnover_degrees.max_degree
        )

        final_state, df = run_simulation(
            default_params(N=50, T=2, T_burn=1, seed=42, eta=0.95)
        )
        final_degrees = BrokerageABM.degree_summary(final_state)
        @test df.mean_degree[end] == final_state.accum.mean_degree
        @test (
            df.mean_degree[end] != final_degrees.mean_degree ||
            df.max_degree[end] != final_degrees.max_degree
        )
    end

    @testset "degree quantile diagnostics exclude the broker node" begin
        p = default_params(N=10, T=2, T_burn=1, seed=42)
        state = initialize_model(p)
        G = SimpleGraph(p.N + 1)
        broker_node = state.broker.node_id
        for i in 1:p.N
            add_edge!(G, i, broker_node)
        end
        add_edge!(G, 6, 7)
        add_edge!(G, 8, 9)
        add_edge!(G, 9, 10)
        state.G = G

        metrics = collect_period_metrics(state)
        expected_degrees = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3]

        @test metrics.mean_degree ≈ sum(expected_degrees) / length(expected_degrees)
        @test metrics.median_degree == 1.5
        @test metrics.min_degree == 1.0
        @test metrics.max_degree == 3.0
        @test degree(state.G, broker_node) == 10
    end

    @testset "Holdout metrics are populated after stepping" begin
        p = default_params(N=80, T=10, T_burn=2, seed=42, eta=0.0)
        state = initialize_model(p)
        for _ in 1:3
            step_period!(state)
        end
        @test isfinite(state.accum.agent_holdout_rank)
        @test isfinite(state.accum.broker_holdout_rank)
        @test isfinite(state.accum.agent_holdout_rmse)
        @test isfinite(state.accum.broker_holdout_rmse)
    end

    @testset "Holdout diagnostics do not consume the simulation RNG" begin
        p = default_params(N=80, T=10, T_burn=2, seed=42, eta=0.0)
        state_with_holdout = initialize_model(p)
        state_reference = initialize_model(p)

        BrokerageABM.update_holdout_metrics!(state_with_holdout)

        @test rand(state_with_holdout.rng) == rand(state_reference.rng)
    end
end
