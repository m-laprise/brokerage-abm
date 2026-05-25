using Test
using TransientBrokerage

# Regression baseline: verify that a fixed-seed simulation produces known-good
# output values. Catches accidental changes to simulation dynamics or RNG stream.
# Baseline refreshed on 2026-05-25 after approved holdout diagnostics sampling:
# holdout samples now use a deterministic diagnostics RNG and no longer advance
# the simulation RNG, so model-event trajectories changed intentionally.
@testset "Regression Baseline" begin
    using Statistics: mean

    p = default_params(N=50, T=20, T_burn=5, seed=42)
    _, df = run_simulation(p)
    tail = df[df.period .> 5, :]

    # Match counts
    @test mean(tail.n_total_matches) ≈ 104.66666666666667 atol=0.01

    # Outsourcing rate
    @test mean(tail.outsourcing_rate) ≈ 0.2774857551209517 atol=1e-4

    # Prediction quality (per-agent averaged, hc>0 only)
    broker_r2 = mean(filter(!isnan, tail.broker_holdout_r2))
    agent_r2 = mean(filter(!isnan, tail.agent_holdout_r2))
    @test broker_r2 ≈ 0.36911265514117436 atol=1e-4
    @test agent_r2 ≈ -0.18210346648101067 atol=1e-4

    # Match output
    @test mean(filter(!isnan, tail.q_self_mean)) ≈ 1.5592299259049887 atol=1e-4

    # Counterparty concentration diagnostics
    @test mean(tail.median_counterparties) ≈ 3.533333333333333 atol=1e-4
    @test maximum(tail.max_counterparties) == 20

    # Broker state at end
    @test df.betweenness[end] ≈ 0.01953066239937867 atol=1e-6
    @test df.roster_size[end] == 10
end
