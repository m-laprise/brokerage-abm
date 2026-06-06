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
# Baseline refreshed on 2026-06-06 after the approved broker-learning fix: Adam
# optimizer (lr 0.01) in place of vanilla GD, and a period-based training window
# in place of the 500-observation window. Refreshed again the same day after
# tuning the now-parameterized schedule to its cost/benefit knees:
# train_window_periods=40, train_max_obs=2000, train_steps=100 (per-period step
# count decoupled from the window/cap and set over full history).
@testset "Regression Baseline" begin
    using Statistics: mean

    p = default_params(N=50, T=20, T_burn=5, seed=42)
    _, df = run_simulation(p)
    tail = df[df.period .> 5, :]

    # Match counts
    @test mean(tail.n_total_matches) ≈ 97.53333333333333 atol=0.01

    # Outsourcing rate
    @test mean(tail.outsourcing_rate) ≈ 0.2539920307370028 atol=1e-4

    # Prediction quality (per-agent averaged, hc>0 only)
    broker_r2 = mean(filter(!isnan, tail.broker_holdout_r2))
    agent_r2 = mean(filter(!isnan, tail.agent_holdout_r2))
    @test broker_r2 ≈ -0.025860213533228173 atol=1e-4
    @test agent_r2 ≈ 0.27275453888288465 atol=1e-4

    # Match output
    @test mean(filter(!isnan, tail.q_self_mean)) ≈ 1.5965108105858135 atol=1e-4

    # Counterparty concentration diagnostics
    @test mean(tail.median_counterparties) ≈ 3.7333333333333334 atol=1e-4
    @test maximum(tail.max_counterparties) == 12

    # Broker state at end
    @test df.betweenness[end] ≈ 0.01867470558493084 atol=1e-6
    @test df.roster_size[end] == 10
end
