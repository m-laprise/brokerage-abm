"""
    run.jl [--smoke] [task_id]

Run one task from a Ridge calibration manifest. Smoke runs use `N=100` and
`T=2` and do not write scientific result shards.
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; use --threads=auto"

using BrokerageABM: RidgeModel, default_params, run_simulation, verify_invariants
using JLD2: jldopen, jldsave

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "provenance.jl"))

const RIDGECAL_SMOKE = "--smoke" in ARGS

function ridgecal_task_id()
    haskey(ENV, "SLURM_ARRAY_TASK_ID") && return parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    for arg in ARGS
        arg == "--smoke" && continue
        return parse(Int, arg)
    end
    error("set SLURM_ARRAY_TASK_ID or pass a task id")
end

function ridgecal_full_config(params)
    return Dict{String,Any}(
        string(field) => getfield(params, field) for field in fieldnames(typeof(params))
    )
end

function ridgecal_shard_current(path, provenance)
    isfile(path) || return false
    try
        return jldopen(path, "r") do file
            all(RIDGECAL_PROVENANCE_KEYS) do key
                name = string(key)
                haskey(file, name) && file[name] == provenance[key]
            end
        end
    catch
        return false
    end
end

function ridgecal_model_finite(model::RidgeModel)
    return all(isfinite, model.coefficients) &&
           isfinite(model.intercept) &&
           isfinite(model.target_mean)
end

function ridgecal_validate_period_table(df, T)
    nrow = size(df, 1)
    nrow == T || error("expected $T period rows, found $nrow")
    df.period == collect(1:T) || error("period table is incomplete or out of order")
    return nothing
end

function ridgecal_run_main()
    stage_dir = get(ENV, "BROKERAGE_ABM_RIDGE_CALIBRATION_STAGE_DIR") do
        error("BROKERAGE_ABM_RIDGE_CALIBRATION_STAGE_DIR is required")
    end
    manifest_path = joinpath(stage_dir, "manifest.jld2")
    isfile(manifest_path) || error("manifest not found: $manifest_path")
    entries, provenance = jldopen(manifest_path, "r") do file
        file["entries"], file["provenance"]
    end
    ridgecal_verify_runtime_provenance(
        provenance;
        require_clean=get(ENV, "BROKERAGE_ABM_ALLOW_DIRTY", "0") != "1",
    )

    task_id = ridgecal_task_id()
    0 <= task_id < length(entries) || error("task id $task_id is out of range")
    entry = entries[task_id + 1]
    entry[:index] == task_id || error("manifest task index mismatch")
    config = entry[:config]
    seed = Int(entry[:seed])

    if RIDGECAL_SMOKE
        N = 100
        T = 2
        outdir = joinpath(stage_dir, "smoke")
    else
        N = RIDGECAL_N
        T = RIDGECAL_T
        outdir = joinpath(stage_dir, "runs", entry[:reldir])
    end
    shard = joinpath(outdir, "seed_$(seed).jld2")
    if !RIDGECAL_SMOKE && ridgecal_shard_current(shard, provenance)
        println("SKIP task=$task_id seed=$seed (current shard exists)")
        return nothing
    end

    resolved = merge(RIDGECAL_BASELINE, (; N, T, seed))
    params = default_params(;
        resolved...,
        learning_model=:ridge,
        ridge_lambda_agent=Float64(config[:lambda_agent]),
        ridge_lambda_broker=Float64(config[:lambda_broker]),
        ridge_broker_variant=:pair,
    )
    println(
        "RUN task=$task_id seed=$seed N=$N T=$T " *
        "lambda_agent=$(params.ridge_lambda_agent) " *
        "lambda_broker=$(params.ridge_lambda_broker)",
    )
    elapsed = @elapsed state, df = run_simulation(params)
    verify_invariants(state)
    ridgecal_validate_period_table(df, T)
    all(ridgecal_model_finite(agent.ridge::RidgeModel) for agent in state.agents) ||
        error("nonfinite principal Ridge parameters")
    ridgecal_model_finite(state.broker.ridge::RidgeModel) ||
        error("nonfinite broker Ridge parameters")

    if !RIDGECAL_SMOKE
        late = df[in.(df.period, Ref(RIDGECAL_LATE_PERIODS)), :]
        size(late, 1) == length(RIDGECAL_LATE_PERIODS) ||
            error("incomplete late calibration window")
        for column in (
            :agent_holdout_rank,
            :broker_holdout_rank,
            :agent_holdout_rmse,
            :broker_holdout_rmse,
            :agent_holdout_bias,
            :broker_holdout_bias,
        )
            all(isfinite, late[!, column]) || error("nonfinite late-period $column")
        end
    end

    mkpath(outdir)
    tmp = shard * ".tmp"
    jldsave(
        tmp;
        df=df,
        task_id=task_id,
        seed=seed,
        config=config,
        resolved_params=ridgecal_full_config(params),
        elapsed_s=elapsed,
        git_commit=provenance[:git_commit],
        julia_version=provenance[:julia_version],
        pkg_manifest_hash=provenance[:pkg_manifest_hash],
        manifest_hash=provenance[:manifest_hash],
        schema_version=provenance[:schema_version],
        smoke=RIDGECAL_SMOKE,
    )
    mv(tmp, shard; force=true)
    println("wrote $shard in $(round(elapsed; digits=1)) seconds")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && ridgecal_run_main()
