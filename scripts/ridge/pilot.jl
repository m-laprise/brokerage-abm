"""
    pilot.jl

Run one baseline simulation with paired Ridge learning and save the complete
period table plus a compact early/late summary. This is a non-reporting pilot
used before the first commit gate.

Environment overrides:
  BROKERAGE_ABM_RIDGE_LAMBDA     Ridge penalty (default 0.01)
  BROKERAGE_ABM_RIDGE_PILOT_SEED seed (default 1)
  BROKERAGE_ABM_RIDGE_PILOT_T    periods (default 500)
  BROKERAGE_ABM_RIDGE_PILOT_DIR  output directory (required)
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; start Julia with --threads=auto"

using BrokerageABM: RidgeModel, default_params, run_simulation
using DataFrames: DataFrame
using JLD2: jldsave
using LinearAlgebra: norm
using Statistics: mean, median, quantile

const RIDGE_LAMBDA = parse(Float64, get(ENV, "BROKERAGE_ABM_RIDGE_LAMBDA", "0.01"))
const PILOT_SEED = parse(Int, get(ENV, "BROKERAGE_ABM_RIDGE_PILOT_SEED", "1"))
const PILOT_T = parse(Int, get(ENV, "BROKERAGE_ABM_RIDGE_PILOT_T", "500"))

function output_dir()
    return get(ENV, "BROKERAGE_ABM_RIDGE_PILOT_DIR") do
        error("BROKERAGE_ABM_RIDGE_PILOT_DIR is required")
    end
end

finite_mean(values) = mean(filter(isfinite, values))

function window_summary(df::DataFrame, periods)
    rows = df[in.(df.period, Ref(periods)), :]
    return Dict{String,Float64}(
        "matches" => finite_mean(rows.n_total_matches),
        "outsourcing" => finite_mean(rows.outsourcing_rate),
        "agent_rank" => finite_mean(rows.agent_holdout_rank),
        "broker_rank" => finite_mean(rows.broker_holdout_rank),
        "agent_r2" => finite_mean(rows.agent_holdout_r2),
        "broker_r2" => finite_mean(rows.broker_holdout_r2),
        "agent_rmse" => finite_mean(rows.agent_holdout_rmse),
        "broker_rmse" => finite_mean(rows.broker_holdout_rmse),
    )
end

function main()
    params = default_params(;
        learning_model=:ridge,
        ridge_lambda=RIDGE_LAMBDA,
        ridge_broker_variant=:pair,
        seed=PILOT_SEED,
        T=PILOT_T,
    )
    elapsed = @elapsed state, df = run_simulation(params; verify=true)
    early_end = min(PILOT_T, 50)
    late_start = max(1, PILOT_T - 99)
    agent_slope_norms = [
        norm((agent.ridge::RidgeModel).coefficients) for agent in state.agents
    ]
    broker_ridge = state.broker.ridge::RidgeModel
    summary = Dict(
        "early" => window_summary(df, 1:early_end),
        "late" => window_summary(df, late_start:PILOT_T),
        "elapsed_s" => elapsed,
        "final_broker_history" => state.broker.history_count,
        "broker_intercept" => broker_ridge.intercept,
        "broker_slope_norm" => norm(broker_ridge.coefficients),
        "agent_slope_norm_median" => median(agent_slope_norms),
        "agent_slope_norm_p95" => quantile(agent_slope_norms, 0.95),
        "agent_slope_norm_max" => maximum(agent_slope_norms),
    )
    config = Dict(
        "learning_model" => "ridge",
        "ridge_broker_variant" => "pair",
        "ridge_lambda" => RIDGE_LAMBDA,
        "seed" => PILOT_SEED,
        "T" => PILOT_T,
        "N" => params.N,
    )

    outdir = output_dir()
    mkpath(outdir)
    output = joinpath(outdir, "paired_ridge_lambda=$(RIDGE_LAMBDA)_seed=$(PILOT_SEED).jld2")
    jldsave(
        output;
        df=df,
        final_agent_degrees=copy(state.accum.agent_degrees),
        config=config,
        summary=summary,
        final_broker_coefficients=copy(broker_ridge.coefficients),
        final_agent_slope_norms=agent_slope_norms,
    )

    println("Ridge pilot complete: $output")
    println("config = $config")
    println("summary = $summary")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
