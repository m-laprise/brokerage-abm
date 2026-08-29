using Test
using BrokerageABM

# Deterministic trajectory fingerprint: verify that a fixed-seed simulation
# reproduces the approved output values. This catches accidental changes to
# simulation dynamics or RNG consumption; it is not a scientific performance
# benchmark.
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
# Baseline refreshed on 2026-06-07 after decoupling the frictions and reservation
# from the surplus scale (phi = c_s = lambda_c * q_cal, r = reservation_frac *
# q_cal) and setting lambda_c = 0.05 (5% commission).
# Fingerprint refreshed on 2026-08-28 after approved initialization, rolling-window,
# and unmatched-neighbor evaluation changes.
# Fingerprint refreshed on 2026-08-28 after the approved prediction-ranking
# correction: agents and the broker now order exact prediction ties randomly
# before market selection and rank-correlation assessment.
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
