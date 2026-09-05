using Test
using BrokerageABM

# Fixed-seed trajectory fingerprint for detecting unintended changes to model
# behavior or random-number consumption. This is not a performance benchmark.
# Values reflect the unit-norm ideal type, calibrated NN learning rates, and
# 100/50 update budgets.
@testset "Regression Baseline" begin
    using Statistics: mean

    p = default_params(N=50, T=20, seed=42)
    _, df = run_simulation(p)
    tail = df[df.period .> 5, :]

    # Match counts
    @test mean(tail.n_total_matches) ≈ 109.66666666666667 atol=0.01

    # Outsourcing rate
    @test mean(tail.outsourcing_rate) ≈ 0.2283998528474888 atol=1e-4

    # Prediction quality (per-agent averaged, hc>0 only)
    broker_r2 = mean(filter(!isnan, tail.broker_holdout_r2))
    agent_r2 = mean(filter(!isnan, tail.agent_holdout_r2))
    @test broker_r2 ≈ 0.24331944719985266 atol=1e-4
    @test agent_r2 ≈ 0.6429427110701353 atol=1e-4

    # Match output
    @test mean(filter(!isnan, tail.q_self_mean)) ≈ 4.614213590160198 atol=1e-4

    # Counterparty concentration diagnostics
    @test mean(tail.median_counterparties) ≈ 3.3666666666666667 atol=1e-4
    @test maximum(tail.max_counterparties) == 18

    # Broker state at end
    @test df.betweenness[end] ≈ 0.012365640102430011 atol=1e-6
    @test df.roster_size[end] == 10
end
