"""
    sweep_run.jl  (one array task = one (cell, seed))

Map `\$SLURM_ARRAY_TASK_ID` -> entry in `manifest.jld2` -> `default_params(...)`
-> `run_simulation` -> write the per-seed shard. Lean by design: no CairoMakie,
no plotting — figures are a separate dependent job (sweep_plot.jl).

Each shard `seed_<s>.jld2` stores the COMPLETE per-period metrics DataFrame
(every column, every period — the raw source of truth, §4), the final agent
degree vector (for the network-stats histogram), the resolved config, and full
provenance (git commit, julia version, Manifest hash, manifest hash, schema
version).

Idempotent: a task whose shard already exists with a matching git commit +
schema version is skipped. Pass `--rerun` to force recompute.

Usage:
  BROKERAGE_ABM_SWEEP_DIR=... SLURM_ARRAY_TASK_ID=<i> julia --project --threads=auto scripts/sweep/sweep_run.jl
  BROKERAGE_ABM_SWEEP_DIR=... julia --project --threads=auto scripts/sweep/sweep_run.jl <i>   # smoke test
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; start Julia with --threads=auto"

include(joinpath(@__DIR__, "sweep_config.jl"))

using BrokerageABM: default_params, run_simulation
using DataFrames: DataFrame
using JLD2: jldsave, jldopen

const RERUN = "--rerun" in ARGS

function task_id()
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        return parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    end
    for a in ARGS
        a == "--rerun" && continue
        return parse(Int, a)
    end
    error("no task id: set SLURM_ARRAY_TASK_ID or pass it as an argument")
end

"""True if a usable shard already exists for this entry (same commit + schema)."""
function shard_is_current(path, prov)
    isfile(path) || return false
    try
        jldopen(path, "r") do f
            haskey(f, "schema_version") && f["schema_version"] == prov[:schema_version] ||
                return false
            haskey(f, "git_commit") && f["git_commit"] == prov[:git_commit] || return false
            return true
        end
    catch
        return false   # corrupt / partial shard -> recompute
    end
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
    outdir = joinpath(sweepdir, reldir)
    shard = joinpath(outdir, "seed_$(seed).jld2")

    if !RERUN && shard_is_current(shard, prov)
        println("SKIP  [$id] $reldir seed=$seed (shard up to date)")
        return nothing
    end

    # ── Build params ─────────────────────────────────────────────────────────
    params = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in e[:params])
    p = default_params(; seed=seed, T=SWEEP_T, T_burn=SWEEP_T_BURN, params...)

    println(
        "RUN   [$id] $reldir seed=$seed  (N=$(p.N), rho=$(p.rho), eta=$(p.eta), " *
        "r_frac=$(p.reservation_frac), threads=$(Threads.nthreads()))",
    )

    t0 = time()
    state, df = run_simulation(p)
    elapsed = round(time() - t0; digits=1)
    println("      done in $(elapsed)s, $(size(df, 1)) periods x $(size(df, 2)) cols")

    final_agent_degrees = copy(state.accum.agent_degrees)

    # Resolved config (provenance of the actual values used).
    config = Dict{String,Any}(
        "kind" => e[:kind],
        "reldir" => reldir,
        "seed" => seed,
        "N" => p.N,
        "T" => p.T,
        "T_burn" => p.T_burn,
        "rho" => p.rho,
        "eta" => p.eta,
        "delta" => p.delta,
        "s" => p.s,
        "reservation_frac" => p.reservation_frac,
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
    )
    mv(tmp, shard; force=true)
    println("      wrote $shard")
end

main()
