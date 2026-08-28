using Test
using BrokerageABM
using BrokerageABM: ActiveMatch
using DataFrames: nrow

@testset "Invariants" begin
    @testset "verify_invariants passes on valid state" begin
        state = initialize_model(default_params(N=30, T=5, seed=42))
        @test verify_invariants(state) === nothing
    end

    @testset "verify_invariants fails on invalid partner id" begin
        state = initialize_model(default_params(N=30, T=5, seed=99))
        push!(state.agents[1].active_matches, ActiveMatch(state.params.N + 1, :self))
        @test_throws AssertionError verify_invariants(state)
    end

    @testset "verify_invariants fails on duplicate current counterparty" begin
        state = initialize_model(default_params(N=30, T=5, seed=101))
        push!(state.agents[1].active_matches, ActiveMatch(2, :self))
        push!(state.agents[1].active_matches, ActiveMatch(2, :broker))
        push!(state.agents[2].active_matches, ActiveMatch(1, :self))
        @test_throws AssertionError verify_invariants(state)
    end

    @testset "run_simulation verify path executes" begin
        p = default_params(N=30, T=6, seed=42)
        state, df = run_simulation(p; verify=true)
        @test state.period == p.T
        @test nrow(df) == p.T
    end
end
