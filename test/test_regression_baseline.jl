using Test
using TransientBrokerage

# Regression baseline: verify that a fixed-seed simulation produces known-good
# output values. Catches accidental changes to simulation dynamics or RNG stream.
# Baseline refreshed on 2026-06-03 after approved initialization simplification:
# agent seed histories now include all initial non-broker graph edges, and broker
# seed history observes existing roster-roster ties without adding new edges.
@testset "Regression Baseline" begin
    using Statistics: mean

    p = default_params(N=50, T=20, T_burn=5, seed=42)
    _, df = run_simulation(p)
    tail = df[df.period .> 5, :]

    # Match counts
    @test mean(tail.n_total_matches) ≈ 109.8 atol=0.01

    # Outsourcing rate
    @test mean(tail.outsourcing_rate) ≈ 0.3460092276616751 atol=1e-4

    # Prediction quality (per-agent averaged, hc>0 only)
    broker_r2 = mean(filter(!isnan, tail.broker_holdout_r2))
    agent_r2 = mean(filter(!isnan, tail.agent_holdout_r2))
    @test broker_r2 ≈ 0.22010500532829172 atol=1e-4
    @test agent_r2 ≈ -0.025169089184834424 atol=1e-4

    # Match output
    @test mean(filter(!isnan, tail.q_self_mean)) ≈ 1.5371582071862233 atol=1e-4

    # Counterparty concentration diagnostics
    @test mean(tail.median_counterparties) ≈ 4.033333333333333 atol=1e-4
    @test maximum(tail.max_counterparties) == 14

    # Broker state at end
    @test df.betweenness[end] ≈ 0.04092258514539704 atol=1e-6
    @test df.roster_size[end] == 10
end
