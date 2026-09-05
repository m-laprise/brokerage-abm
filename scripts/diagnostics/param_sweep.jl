"""
    param_sweep.jl

Run a one-at-a-time sensitivity analysis of the training-window length,
observation cap, and neural-network step count. Reports gain recovery, holdout
rank correlation, the broker-minus-principal rank difference, and runtime.

Usage:
    julia --project --threads=auto scripts/diagnostics/param_sweep.jl
Environment overrides: `SWEEP_QUICK=1`, `SWEEP_T`, and `SWEEP_N`.
"""

include(joinpath(@__DIR__, "broker_learning_common.jl"))
using BrokerageABM: run_simulation, default_params
using DataFrames

const QUICK = haskey(ENV, "SWEEP_QUICK")
const N = parse(Int, get(ENV, "SWEEP_N", QUICK ? "200" : "1000"))
const T = parse(Int, get(ENV, "SWEEP_T", QUICK ? "40" : "80"))
const SEEDS = QUICK ? (42,) : (42, 43, 44)
# Tuples contain observation cap, window periods, and update steps.
const CELLS = QUICK ? [(2000, 20, 50), (2000, 20, 100)] :
    [(2000, 20, 50), (2000, 20, 100), (2000, 20, 200),   # steps
     (1000, 20, 100), (4000, 20, 100),                   # cap
     (2000, 10, 100), (2000, 40, 100), (2000, 80, 100)]  # wp

tailmean(df, c) = mean(filter(!isnan, df[df.period .> T ÷ 2, c]))

function run_cell(cap, wp, steps, seed)
    p = default_params(;
        N=N,
        T=T,
        seed=seed,
        train_window_periods=wp,
        train_max_obs=cap,
        train_steps_agent=steps,
        train_steps_broker=steps,
    )
    wall = @elapsed ((state, df) = run_simulation(p))
    pool = [a.type for a in state.agents]
    Xi, Xj = sample_pairs(state.env, pool, 4000, StableRNG(99); source=:pool)
    m = evaluate(predict_nn_cols(state.broker.nn, broker_features(Xi, Xj)),
                 decompose(state.env, Xi, Xj))
    gap = tailmean(df, :broker_holdout_rank) - tailmean(df, :agent_holdout_rank)
    return (; seed, wall, bg=m.bg, brank=tailmean(df, :broker_holdout_rank), gap)
end

results = Dict{Tuple{Int,Int,Int},Vector}()
for c in CELLS, s in SEEDS
    push!(get!(results, c, []), run_cell(c..., s))
end

agg(rs, f) = (mean(getfield.(rs, f)), minimum(getfield.(rs, f)), maximum(getfield.(rs, f)))
@printf("\n%-20s %8s %8s   %-24s %s\n", "cell(cap,wp,steps)", "wall(s)", "bg", "rank_gap mean[min,max]", "broker_rank")
for c in CELLS
    rs = results[c]; g = agg(rs, :gap); br = agg(rs, :brank); bg = agg(rs, :bg)
    @printf("%-20s %8.1f %+8.3f   %.3f [%+.3f,%+.3f]    %.3f\n",
        string(c), rs[1].wall, bg[1], g[1], g[2], g[3], br[1])
end
