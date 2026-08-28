using Test
using BrokerageABM

@testset "Integration Tests" begin
    using DataFrames: nrow

    @testset "Simulation is deterministic and internally coherent" begin
        p = default_params(N=50, T=10, seed=42)
        state, df1 = run_simulation(p)
        _, df2 = run_simulation(p)

        @test nrow(df1) == p.T
        @test state.period == p.T
        @test df1.n_total_matches == df2.n_total_matches
        @test df1.outsourcing_rate == df2.outsourcing_rate
        @test df1.agent_holdout_r2 == df2.agent_holdout_r2
        @test df1.broker_holdout_r2 == df2.broker_holdout_r2
        @test all(df1.n_total_matches .== df1.n_self_matches .+ df1.n_broker_matches)
        @test all(0.0 .<= df1.outsourcing_rate .<= 1.0)
        @test all(isfinite(a.satisfaction_self) for a in state.agents)
        @test all(isfinite(a.satisfaction_broker) for a in state.agents)
    end

    @testset "Representative parameter variants complete" begin
        variants = [(K=20,), (K=2,), (rho=0.0,)]
        results = map(variants) do kwargs
            p = default_params(N=50, T=10, seed=42; kwargs...)
            _, df = run_simulation(p)
            df
        end
        @test all(nrow(df) == 10 for df in results)
        @test results[1].n_total_matches[end] > 50
    end
end
