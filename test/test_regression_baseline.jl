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
    @test mean(tail.n_total_matches) ≈ 108.8 atol=0.01

    # Outsourcing rate
    @test mean(tail.outsourcing_rate) ≈ 0.17976810678652858 atol=1e-4

    # Prediction quality (per-agent averaged, hc>0 only)
    broker_r2 = mean(filter(!isnan, tail.broker_holdout_r2))
    agent_r2 = mean(filter(!isnan, tail.agent_holdout_r2))
    @test broker_r2 ≈ 0.2060457606330443 atol=1e-4
    @test agent_r2 ≈ 0.583373090233516 atol=1e-4

    # Match output
    @test mean(filter(!isnan, tail.q_self_mean)) ≈ 4.596721545950681 atol=1e-4

    # Counterparty concentration diagnostics
    @test mean(tail.median_counterparties) ≈ 3.533333333333333 atol=1e-4
    @test maximum(tail.max_counterparties) == 21

    # Broker state at end
    @test df.betweenness[end] ≈ 0.0072986125641915055 atol=1e-6
    @test df.roster_size[end] == 10
end
