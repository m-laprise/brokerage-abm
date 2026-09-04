using Test
using BrokerageABM

# Fixed-seed trajectory fingerprint for detecting unintended changes to model
# behavior or random-number consumption. This is not a performance benchmark.
@testset "Regression Baseline" begin
    using Statistics: mean

    p = default_params(N=50, T=20, seed=42)
    _, df = run_simulation(p)
    tail = df[df.period .> 5, :]

    # Match counts
    @test mean(tail.n_total_matches) ≈ 104.06666666666666 atol=0.01

    # Outsourcing rate
    @test mean(tail.outsourcing_rate) ≈ 0.21002069802713688 atol=1e-4

    # Prediction quality (per-agent averaged, hc>0 only)
    broker_r2 = mean(filter(!isnan, tail.broker_holdout_r2))
    agent_r2 = mean(filter(!isnan, tail.agent_holdout_r2))
    @test broker_r2 ≈ -0.06199639717261532 atol=1e-4
    @test agent_r2 ≈ 0.43006093934433015 atol=1e-4

    # Match output
    @test mean(filter(!isnan, tail.q_self_mean)) ≈ 1.6237246261552427 atol=1e-4

    # Counterparty concentration diagnostics
    @test mean(tail.median_counterparties) ≈ 3.7 atol=1e-4
    @test maximum(tail.max_counterparties) == 13

    # Broker state at end
    @test df.betweenness[end] ≈ 0.007447497038630044 atol=1e-6
    @test df.roster_size[end] == 10
end
