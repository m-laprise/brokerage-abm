"""
    sweep_manifest.jl  (Step 0)

Expand the §2 sweep specification into the full ordered list of (cell, seed)
jobs and write the manifest. Run ONCE, before the compute array.

Writes, under `TB_SWEEP_DIR`:
  * `manifest.json`  — human/report-readable source of truth (spec, provenance,
                       every entry, every plot job).
  * `manifest.jld2`  — identical structure, read natively by sweep_run.jl /
                       sweep_plot.jl (avoids a JSON dependency at run time).
  * `manifest.sha256`— sha256 of manifest.json (the `manifest_hash` recorded in
                       every shard).

Prints `NRUNS=<n>` and `NPLOT=<n>` on their own lines for the orchestrator.

Usage:
  TB_SWEEP_DIR=/path/to/sweep/<tag> julia --project scripts/sweep/sweep_manifest.jl
"""

include(joinpath(@__DIR__, "sweep_config.jl"))

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
    mkpath(sweepdir)

    cells = build_cells()
    entries = build_entries(cells)
    plot_jobs = build_plot_jobs(cells)

    nruns = length(entries)
    nplot = length(plot_jobs)

    # ── Provenance ───────────────────────────────────────────────────────────
    commit = git_commit()
    pkg_manifest_hash = file_sha256(joinpath(REPO_ROOT, "Manifest.toml"))
    meta = Dict{Symbol,Any}(
        :tag => basename(sweepdir),
        :date => string(Dates.today()),
        :git_commit => commit,
        :git_dirty => git_dirty(),
        :julia_version => string(VERSION),
        :pkg_manifest_hash => pkg_manifest_hash,
        :schema_version => SWEEP_SCHEMA_VERSION,
        :n_runs => nruns,
        :n_plot_jobs => nplot,
        :n_cells => length(cells),
        :seeds => SWEEP_SEEDS,
        :T => SWEEP_T,
        :T_burn => SWEEP_T_BURN,
    )

    # Spec block for the report agent (what was swept, at a glance).
    spec = Dict{Symbol,Any}(
        :baseline => Dict(:rho => 0.5, :eta => 0.02, :N => 1000,
                          :reservation_frac => 0.6, :T => SWEEP_T, :T_burn => SWEEP_T_BURN),
        :oat_axes => [Dict(:label => a.label, :key => string(a.key),
                           :vals => a.vals, :models => string.(a.models)) for a in OAT_AXES],
        :phase_pairs => [Dict(:name => p.name, :xkey => string(p.xkey), :xvals => p.xvals,
                              :ykey => string(p.ykey), :yvals => p.yvals) for p in PHASE_PAIRS],
        :seeds => SWEEP_SEEDS,
    )

    manifest = Dict{Symbol,Any}(
        :meta => meta,
        :spec => spec,
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
    write(joinpath(sweepdir, "counts.env"), "NRUNS=$nruns\nNPLOT=$nplot\n")

    # Provenance copied verbatim into every shard for idempotency checks.
    prov = Dict{Symbol,Any}(
        :git_commit => commit,
        :julia_version => string(VERSION),
        :pkg_manifest_hash => pkg_manifest_hash,
        :manifest_hash => manifest_hash,
        :schema_version => SWEEP_SCHEMA_VERSION,
    )

    # ── Write manifest.jld2 (native, read by run/plot) ───────────────────────
    jldsave(joinpath(sweepdir, "manifest.jld2");
        entries = entries,
        plot_jobs = plot_jobs,
        meta = meta,
        spec = spec,
        prov = prov,
        manifest_hash = manifest_hash,
        schema_version = SWEEP_SCHEMA_VERSION,
    )

    # ── Report ───────────────────────────────────────────────────────────────
    println("Sweep manifest written to: $sweepdir")
    println("  git commit:        $commit$(meta[:git_dirty] ? "  (DIRTY working tree)" : "")")
    println("  julia version:     $(VERSION)")
    println("  Manifest.toml hash: $pkg_manifest_hash")
    println("  manifest.json hash: $manifest_hash")
    println("  cells:             $(length(cells))")
    println("NRUNS=$nruns")
    println("NPLOT=$nplot")
end

main()
