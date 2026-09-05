"""
    sweep_run.jl [--smoke] (one array task = one unique (condition, seed))

Map `\$SLURM_ARRAY_TASK_ID` to a manifest entry, run its simulation, and write
the per-seed shard. Plotting runs separately in `sweep_plot.jl`.

Each `seed_<s>.jld2` shard stores all per-period metrics, the final principal
degree vector, the resolved configuration, and provenance hashes.

Idempotent: a task whose shard already exists with matching provenance is
skipped. Pass `--rerun` to force recompute. A smoke run uses two periods and
one initial and recurrent update step, then writes outside the scientific run
directories.

Usage:
  BROKERAGE_ABM_SWEEP_DIR=... SLURM_ARRAY_TASK_ID=<i> julia --project --threads=auto scripts/sweep/sweep_run.jl
  BROKERAGE_ABM_SWEEP_DIR=... julia --project --threads=auto scripts/sweep/sweep_run.jl --smoke <i>
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; start Julia with --threads=auto"

include(joinpath(@__DIR__, "sweep_config.jl"))

using BrokerageABM: default_params, run_simulation, verify_invariants
using DataFrames: DataFrame
using JLD2: jldsave, jldopen

include(joinpath(@__DIR__, "shard_validation.jl"))

const RERUN = "--rerun" in ARGS
const SMOKE = "--smoke" in ARGS

function task_id()
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        return parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    end
    for a in ARGS
        a in ("--rerun", "--smoke") && continue
        return parse(Int, a)
    end
    error("no task id: set SLURM_ARRAY_TASK_ID or pass it as an argument")
end

function main()
    sweepdir = sweep_dir()
    manifest = joinpath(sweepdir, "manifest.jld2")
    isfile(manifest) || error("manifest not found: $manifest (run sweep_manifest.jl first)")

    entries, prov = jldopen(manifest, "r") do f
        f["entries"], f["prov"]
    end

    id = task_id()
    (0 <= id < length(entries)) ||
        error("task id $id out of range 0..$(length(entries) - 1)")
    e = entries[id + 1]
    @assert e[:index] == id "manifest index mismatch: entry $(e[:index]) != task $id"

    reldir = e[:reldir]
    seed = e[:seed]
    outdir = SMOKE ? joinpath(sweepdir, "smoke") : joinpath(sweepdir, reldir)
    shard = if SMOKE
        joinpath(outdir, "task_$(id)_seed_$(seed).jld2")
    else
        joinpath(outdir, "seed_$(seed).jld2")
    end

    if !SMOKE && !RERUN && shard_is_current(shard, prov)
        println("SKIP  [$id] $reldir seed=$seed (shard up to date)")
        return nothing
    end

    # ── Build params ─────────────────────────────────────────────────────────
    params = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in e[:resolved_params])
    p = default_params(;
        seed=seed,
        T=SMOKE ? 2 : SWEEP_T,
        learning_model=SWEEP_LEARNING_MODEL,
        eta_lr_agent=SWEEP_NN_ETA_LR_AGENT,
        eta_lr_broker=SWEEP_NN_ETA_LR_BROKER,
        E_init_agent=SMOKE ? 1 : SWEEP_NN_E_INIT_AGENT,
        E_init_broker=SMOKE ? 1 : SWEEP_NN_E_INIT_BROKER,
        train_steps_agent=SMOKE ? 1 : SWEEP_NN_TRAIN_STEPS_AGENT,
        train_steps_broker=SMOKE ? 1 : SWEEP_NN_TRAIN_STEPS_BROKER,
        ridge_lambda_agent=SWEEP_RIDGE_LAMBDA_AGENT,
        ridge_lambda_broker=SWEEP_RIDGE_LAMBDA_BROKER,
        ridge_broker_variant=SWEEP_RIDGE_BROKER_VARIANT,
        params...,
    )

    println(
        "RUN   [$id] $reldir seed=$seed  (N=$(p.N), rho=$(p.rho), eta=$(p.eta), " *
        "r_frac=$(p.reservation_frac), learner=$(p.learning_model), " *
        "broker_variant=$(p.ridge_broker_variant), " *
        "threads=$(Threads.nthreads()))",
    )

    t0 = time()
    state, df = run_simulation(p)
    SMOKE && verify_invariants(state)
    elapsed = round(time() - t0; digits=1)
    println("      done in $(elapsed)s, $(size(df, 1)) periods x $(size(df, 2)) cols")

    final_agent_degrees = copy(state.accum.agent_degrees)

    # Resolved config (provenance of the actual values used).
    config = Dict{String,Any}(
        "kind" => e[:kind],
        "reldir" => reldir,
        "result_reldir" => reldir,
        "condition_index" => e[:condition_index],
        "seed" => seed,
        "N" => p.N,
        "T" => p.T,
        "T_burn" => SWEEP_T_BURN,
        "rho" => p.rho,
        "eta" => p.eta,
        "delta" => p.delta,
        "k" => p.k,
        "roster_frac" => p.roster_frac,
        "n_strangers" => p.n_strangers,
        "s" => p.s,
        "reservation_frac" => p.reservation_frac,
        "learning_model" => string(p.learning_model),
        "eta_lr_agent" => p.eta_lr_agent,
        "eta_lr_broker" => p.eta_lr_broker,
        "E_init_agent" => p.E_init_agent,
        "E_init_broker" => p.E_init_broker,
        "train_steps_agent" => p.train_steps_agent,
        "train_steps_broker" => p.train_steps_broker,
        "ridge_lambda_agent" => p.ridge_lambda_agent,
        "ridge_lambda_broker" => p.ridge_lambda_broker,
        "ridge_broker_variant" => string(p.ridge_broker_variant),
        "quality_weight" => state.env.quality_weight,
        "interaction_weight" => state.env.interaction_weight,
        "signal_shift" => state.env.signal_shift,
    )
    for k in (:axis, :key, :value, :pair, :xkey, :xval, :xi, :ykey, :yval, :yi)
        haskey(e, k) && (config[string(k)] = e[k])
    end

    mkpath(outdir)
    tmp = shard * ".tmp"   # atomic-ish write: full file then rename
    jldsave(
        tmp;
        df=df,
        final_agent_degrees=final_agent_degrees,
        config=config,
        seed=seed,
        kind=e[:kind],
        reldir=reldir,
        elapsed_s=elapsed,
        git_commit=prov[:git_commit],
        julia_version=prov[:julia_version],
        pkg_manifest_hash=prov[:pkg_manifest_hash],
        manifest_hash=prov[:manifest_hash],
        schema_version=prov[:schema_version],
        smoke=SMOKE,
    )
    mv(tmp, shard; force=true)
    println("      wrote $shard")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
