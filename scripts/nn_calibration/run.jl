"""
    run.jl [--smoke] [task_id]

Run one task from a neural-network calibration stage manifest. A smoke run uses
`N=100`, `T=2`, and one update step without writing a scientific result shard.
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; start Julia with --threads=auto"

using BrokerageABM: default_params, run_simulation, verify_invariants
using JLD2: jldopen, jldsave

include(joinpath(@__DIR__, "calibration_config.jl"))

const NNCAL_SMOKE = "--smoke" in ARGS

function nncal_task_id()
    haskey(ENV, "SLURM_ARRAY_TASK_ID") && return parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    for arg in ARGS
        arg == "--smoke" && continue
        return parse(Int, arg)
    end
    error("set SLURM_ARRAY_TASK_ID or pass a task id")
end

function nncal_nn_finite(nn)
    return all(isfinite, nn.W1) &&
           all(isfinite, nn.b1) &&
           all(isfinite, nn.w2) &&
           isfinite(nn.b2)
end

function nncal_full_config(params)
    return Dict{String,Any}(
        string(field) => getfield(params, field) for field in fieldnames(typeof(params))
    )
end

function nncal_shard_current(path, provenance)
    isfile(path) || return false
    try
        return jldopen(path, "r") do file
            file["git_commit"] == provenance[:git_commit] &&
                file["pkg_manifest_hash"] == provenance[:pkg_manifest_hash] &&
                file["manifest_hash"] == provenance[:manifest_hash] &&
                file["schema_version"] == provenance[:schema_version]
        end
    catch
        return false
    end
end

function main()
    stage_dir = get(ENV, "BROKERAGE_ABM_NN_CALIBRATION_STAGE_DIR") do
        error("BROKERAGE_ABM_NN_CALIBRATION_STAGE_DIR is required")
    end
    manifest_path = joinpath(stage_dir, "manifest.jld2")
    isfile(manifest_path) || error("manifest not found: $manifest_path")
    entries, provenance = jldopen(manifest_path, "r") do file
        file["entries"], file["provenance"]
    end

    task_id = nncal_task_id()
    0 <= task_id < length(entries) || error("task id $task_id is out of range")
    entry = entries[task_id + 1]
    entry[:index] == task_id || error("manifest task index mismatch")
    config = entry[:config]
    seed = Int(entry[:seed])

    if NNCAL_SMOKE
        N = 100
        T = 2
        initial_agent = 1
        initial_broker = 1
        recurrent_agent = 1
        recurrent_broker = 1
        outdir = joinpath(stage_dir, "smoke")
    else
        N = NNCAL_N
        T = NNCAL_T
        initial_agent = Int(config[:agent_initial_steps])
        initial_broker = Int(config[:broker_initial_steps])
        recurrent_agent = Int(config[:agent_recurrent_steps])
        recurrent_broker = Int(config[:broker_recurrent_steps])
        outdir = joinpath(stage_dir, "runs", entry[:reldir])
    end

    shard = joinpath(outdir, "seed_$(seed).jld2")
    if !NNCAL_SMOKE && nncal_shard_current(shard, provenance)
        println("SKIP task=$task_id seed=$seed (current shard exists)")
        return nothing
    end

    params = default_params(;
        N=N,
        T=T,
        seed=seed,
        learning_model=:nn,
        eta_lr_agent=Float64(config[:agent_eta_lr]),
        eta_lr_broker=Float64(config[:broker_eta_lr]),
        E_init_agent=initial_agent,
        E_init_broker=initial_broker,
        train_steps_agent=recurrent_agent,
        train_steps_broker=recurrent_broker,
    )

    println(
        "RUN task=$task_id seed=$seed N=$N T=$T " *
        "agent=($(params.eta_lr_agent),$(params.E_init_agent),$(params.train_steps_agent)) " *
        "broker=($(params.eta_lr_broker),$(params.E_init_broker),$(params.train_steps_broker))",
    )
    elapsed = @elapsed state, df = run_simulation(params)
    verify_invariants(state)

    all(nncal_nn_finite(agent.nn) for agent in state.agents) ||
        error("nonfinite agent neural-network parameters")
    nncal_nn_finite(state.broker.nn) || error("nonfinite broker neural-network parameters")
    if !NNCAL_SMOKE
        late = df[in.(df.period, Ref(NNCAL_LATE_PERIODS)), :]
        all(isfinite, late.agent_holdout_rank) || error("nonfinite late agent rank")
        all(isfinite, late.broker_holdout_rank) || error("nonfinite late broker rank")
        all(isfinite, late.agent_holdout_rmse) || error("nonfinite late agent RMSE")
        all(isfinite, late.broker_holdout_rmse) || error("nonfinite late broker RMSE")
    end

    mkpath(outdir)
    tmp = shard * ".tmp"
    jldsave(
        tmp;
        df=df,
        task_id=task_id,
        seed=seed,
        config=config,
        resolved_params=nncal_full_config(params),
        elapsed_s=elapsed,
        git_commit=provenance[:git_commit],
        julia_version=provenance[:julia_version],
        pkg_manifest_hash=provenance[:pkg_manifest_hash],
        manifest_hash=provenance[:manifest_hash],
        schema_version=provenance[:schema_version],
        smoke=NNCAL_SMOKE,
    )
    mv(tmp, shard; force=true)
    println("wrote $shard in $(round(elapsed; digits=1)) seconds")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
