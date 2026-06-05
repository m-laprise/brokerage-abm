using Test
using TransientBrokerage

# Regression baseline: verify that a fixed-seed simulation produces known-good
# output values. Catches accidental changes to simulation dynamics or RNG stream.
# Baseline refreshed on 2026-06-03 after approved learning and initialization
# changes: symmetric broker pair features, hidden widths derived from d
# including agent width 2d, DI/Enzyme gradients, full initial agent neighbor
# history seeding, and broker seed history from existing roster-roster ties
# without adding edges.
# Baseline refreshed on 2026-06-04 after approved correctness fixes: each
# self-search demander samples an independent stranger pool, and period metrics
# are recorded before entry/exit turnover.
@testset "Regression Baseline" begin
    using Statistics: mean

    p = default_params(N=50, T=20, T_burn=5, seed=42)
    _, df = run_simulation(p)
    tail = df[df.period .> 5, :]

    # Match counts
    @test mean(tail.n_total_matches) ≈ 107.4 atol=0.01

    # Outsourcing rate
    @test mean(tail.outsourcing_rate) ≈ 0.30300309510181866 atol=1e-4

    # Prediction quality (per-agent averaged, hc>0 only)
    broker_r2 = mean(filter(!isnan, tail.broker_holdout_r2))
    agent_r2 = mean(filter(!isnan, tail.agent_holdout_r2))
    @test broker_r2 ≈ 0.10204493119306268 atol=1e-4
    @test agent_r2 ≈ 0.09577101671226357 atol=1e-4

    # Match output
    @test mean(filter(!isnan, tail.q_self_mean)) ≈ 1.6261870680317243 atol=1e-4

    # Counterparty concentration diagnostics
    @test mean(tail.median_counterparties) ≈ 3.466666666666667 atol=1e-4
    @test maximum(tail.max_counterparties) == 18

    # Broker state at end
    @test df.betweenness[end] ≈ 0.042926660482601464 atol=1e-6
    @test df.roster_size[end] == 10
end
