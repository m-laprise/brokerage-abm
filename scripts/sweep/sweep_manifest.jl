"""
    sweep_manifest.jl  (Step 0)

Expand the sweep specification into grid coordinates, effective realizations,
and the ordered list of unique (condition, seed) jobs. Run once before the
compute array.

Writes, under `BROKERAGE_ABM_SWEEP_DIR`:
  * `manifest.json`  — human/report-readable source of truth (spec, provenance,
                       grid references, every entry, every plot job).
  * `manifest.jld2`  — identical structure, read natively by sweep_run.jl /
                       sweep_plot.jl (avoids a JSON dependency at run time).
  * `manifest.sha256`— sha256 of manifest.json (the `manifest_hash` recorded in
                       every shard).

Prints `NRUNS=<n>` and `NPLOT=<n>` on their own lines for the orchestrator.

Usage:
  BROKERAGE_ABM_SWEEP_DIR=/path/to/sweep/<tag> julia --project --threads=auto scripts/sweep/sweep_manifest.jl
"""

include(joinpath(@__DIR__, "sweep_config.jl"))

using JLD2: jldsave
using SHA: sha256
using Dates: Dates

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function git_commit()
    try
        return strip(read(`git -C $REPO_ROOT rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

function git_dirty()
    try
        out = read(`git -C $REPO_ROOT status --porcelain`, String)
        return !isempty(strip(out))
    catch
        return false
    end
end

function file_sha256(path)
    isfile(path) || return "absent"
    return bytes2hex(sha256(read(path)))
end

function main()
    sweepdir = sweep_dir()
    dirty = git_dirty()
    if dirty && get(ENV, "BROKERAGE_ABM_ALLOW_DIRTY", "0") != "1"
        error(
            "refusing to create a scientific sweep manifest from a dirty worktree; " *
            "commit the intended code first, or set BROKERAGE_ABM_ALLOW_DIRTY=1 " *
            "only for non-reporting smoke tests",
        )
    end
    mkpath(sweepdir)

    cells = build_cells()
    conditions = result_cells(cells)
    entries = build_entries(cells)
    plot_jobs = build_plot_jobs(cells)

    nruns = length(entries)
    nplot = length(plot_jobs)
    nconditions = length(conditions)

    # ── Provenance ───────────────────────────────────────────────────────────
    commit = git_commit()
    pkg_manifest_hash = file_sha256(joinpath(REPO_ROOT, "Manifest.toml"))
    meta = Dict{Symbol,Any}(
        :tag => basename(sweepdir),
        :date => string(Dates.today()),
        :git_commit => commit,
        :git_dirty => dirty,
        :julia_version => string(VERSION),
        :pkg_manifest_hash => pkg_manifest_hash,
        :schema_version => SWEEP_SCHEMA_VERSION,
        :n_runs => nruns,
        :n_plot_jobs => nplot,
        :n_conditions => nconditions,
        :n_grid_cells => length(cells),
        :seeds => SWEEP_SEEDS,
        :T => SWEEP_T,
        :T_burn => SWEEP_T_BURN,
    )

    # Spec block for the report agent (what was swept, at a glance).
    spec = Dict{Symbol,Any}(
        :baseline => Dict(pairs(SWEEP_BASELINE)..., :T => SWEEP_T, :T_burn => SWEEP_T_BURN),
        :oat_axes => [
            Dict(:label => a.label, :key => string(a.key), :vals => a.vals) for
            a in OAT_AXES
        ],
        :phase_pairs => [
            Dict(
                :name => p.name,
                :xkey => string(p.xkey),
                :xvals => p.xvals,
                :ykey => string(p.ykey),
                :yvals => p.yvals,
            ) for p in PHASE_PAIRS
        ],
        :seeds => SWEEP_SEEDS,
    )

    manifest = Dict{Symbol,Any}(
        :meta => meta,
        :spec => spec,
        :cells => cells,
        :conditions => conditions,
        :entries => entries,
        :plot_jobs => plot_jobs,
    )

    # ── Write manifest.json (source of truth for humans / report) ────────────
    json_path = joinpath(sweepdir, "manifest.json")
    open(json_path, "w") do io
        to_json(io, manifest; indent=0)
        println(io)
    end
    manifest_hash = file_sha256(json_path)
    write(joinpath(sweepdir, "manifest.sha256"), manifest_hash * "\n")

    # Shell-sourceable counts for the orchestrator (submit.sh).
    write(
        joinpath(sweepdir, "counts.env"),
        "NCONDITIONS=$nconditions\nNGRIDCELLS=$(length(cells))\nNRUNS=$nruns\nNPLOT=$nplot\n",
    )

    # Provenance copied verbatim into every shard for idempotency checks.
    prov = Dict{Symbol,Any}(
        :git_commit => commit,
        :julia_version => string(VERSION),
        :pkg_manifest_hash => pkg_manifest_hash,
        :manifest_hash => manifest_hash,
        :schema_version => SWEEP_SCHEMA_VERSION,
    )

    # ── Write manifest.jld2 (native, read by run/plot) ───────────────────────
    jldsave(
        joinpath(sweepdir, "manifest.jld2");
        cells=cells,
        conditions=conditions,
        entries=entries,
        plot_jobs=plot_jobs,
        meta=meta,
        spec=spec,
        prov=prov,
        manifest_hash=manifest_hash,
        schema_version=SWEEP_SCHEMA_VERSION,
    )

    # ── Report ───────────────────────────────────────────────────────────────
    println("Sweep manifest written to: $sweepdir")
    println(
        "  git commit:        $commit$(meta[:git_dirty] ? "  (DIRTY working tree)" : "")"
    )
    println("  julia version:     $(VERSION)")
    println("  Manifest.toml hash: $pkg_manifest_hash")
    println("  manifest.json hash: $manifest_hash")
    println("  grid coordinates:  $(length(cells))")
    println("  effective results: $nconditions")
    println("NRUNS=$nruns")
    println("NPLOT=$nplot")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
