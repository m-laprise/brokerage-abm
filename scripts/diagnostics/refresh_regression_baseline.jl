"""
    refresh_regression_baseline.jl

Recompute the pinned values for `test/test_regression_baseline.jl` after an
approved model change. Prints the tail-mean metrics for the fixed
(N=50, T=20, seed=42) run; copy them into the test with an update note.

Usage: julia --project --threads=auto scripts/diagnostics/refresh_regression_baseline.jl
"""

using BrokerageABM
using DataFrames
using Statistics: mean

p = default_params(N=50, T=20, seed=42)
_, df = run_simulation(p)
tail = df[df.period .> 5, :]
nz(c) = mean(filter(!isnan, tail[!, c]))

println("n_total_matches       = ", mean(tail.n_total_matches))
println("outsourcing_rate      = ", mean(tail.outsourcing_rate))
println("broker_r2             = ", nz(:broker_holdout_r2))
println("agent_r2              = ", nz(:agent_holdout_r2))
println("q_self_mean           = ", nz(:q_self_mean))
println("median_counterparties = ", mean(tail.median_counterparties))
println("max_counterparties    = ", maximum(tail.max_counterparties))
println("betweenness_end       = ", df.betweenness[end])
println("roster_size_end       = ", df.roster_size[end])
