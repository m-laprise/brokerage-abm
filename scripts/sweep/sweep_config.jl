"""
    sweep_config.jl

Single source of truth for the parameter sweep.

Defines the OAT axes, the phase-diagram pairs, the baseline, the seed list, and
the storage layout. Expands
all of that into:

  * `build_cells()`   -> the ordered list of parameter cells,
  * `build_entries()` -> the ordered list of (cell, seed) *jobs* (one per array
                         task), each with a 0-based `index`,
  * `build_plot_jobs()` -> the ordered list of per-cell / per-pair plot jobs.

This file is deliberately dependency-light (no BrokerageABM, no CairoMakie,
only `JLD2` + the `SHA`/`Dates` stdlibs) so it can be `include`d from the manifest
generator, the per-task runner, and the plotting script alike. It also provides a
minimal JSON emitter so `manifest.json` is a real, human/report-readable file
without adding a JSON package to the project. The machine-read mirror is
`manifest.jld2`, written from the identical in-memory structure so the two cannot
drift.
"""

using JLD2: jldsave, jldopen
using SHA: sha256
using Dates: Dates

# ─────────────────────────────────────────────────────────────────────────────
# Versioning
# ─────────────────────────────────────────────────────────────────────────────

"""Bump when the shard / manifest schema changes (invalidates cached shards)."""
const SWEEP_SCHEMA_VERSION = 2

# ─────────────────────────────────────────────────────────────────────────────
# Baseline + sweep specification
# ─────────────────────────────────────────────────────────────────────────────

# Baseline simulation knobs applied to every cell (others come from
# `default_params`). T/T_burn are fixed across the whole sweep.
const SWEEP_T = 200
const SWEEP_T_BURN = 30
const SWEEP_SEEDS = collect(1:5)

# OAT axis levels. `key` is the `default_params` keyword and `label` names the
# output directory.
const RHO_VALS = [0.0, 0.5, 1.0]
# The OAT rho axis is denser than the phase rho axis (extra 0.3, 0.7) for line-plot
# resolution; phase grids keep RHO_VALS so their index-keyed cells are not disturbed.
const RHO_OAT = [0.0, 0.3, 0.5, 0.7, 1.0]
const ETA_VALS = [0.01, 0.02, 0.03]
const N_VALS = [500, 1000, 1500]
const R_VALS = [0.40, 0.60, 0.90, 1.20]   # reservation_frac (lambda_r)

# Matching complexity (delta = regime-gain strength, the operator behind the
# fundamental information gap of §1e) and network density (k_G = initial degree).
# The OAT levels bracket the baseline delta=0.5 / k_G=6; DELTA_VALS keeps that
# midpoint for the rho x delta phase grid.
const DELTA_OAT_VALS = [0.0, 0.75]
const DELTA_VALS = [0.0, 0.50, 0.75]
const K_VALS = [4, 12]

const OAT_AXES = [
    (label="rho", key=:rho, vals=RHO_OAT),
    (label="eta", key=:eta, vals=ETA_VALS),
    (label="N", key=:N, vals=N_VALS),
    (label="reservation_frac", key=:reservation_frac, vals=R_VALS),
    (label="delta", key=:delta, vals=DELTA_OAT_VALS),
    (label="k", key=:k, vals=K_VALS),
]

# Phase-diagram pairs (§2b): all six pairwise combinations of {rho, eta, N, r},
# axis levels reused from the OAT design.
const PHASE_PAIRS = [
    (name="rho_eta", xkey=:rho, xvals=RHO_VALS, ykey=:eta, yvals=ETA_VALS),
    (name="rho_N", xkey=:rho, xvals=RHO_VALS, ykey=:N, yvals=N_VALS),
    (name="rho_r", xkey=:rho, xvals=RHO_VALS, ykey=:reservation_frac, yvals=R_VALS),
    (name="eta_r", xkey=:eta, xvals=ETA_VALS, ykey=:reservation_frac, yvals=R_VALS),
    (name="eta_N", xkey=:eta, xvals=ETA_VALS, ykey=:N, yvals=N_VALS),
    (name="r_N", xkey=:reservation_frac, xvals=R_VALS, ykey=:N, yvals=N_VALS),
    (name="rho_delta", xkey=:rho, xvals=RHO_VALS, ykey=:delta, yvals=DELTA_VALS),
]

const BASELINE_RELDIR = "oat/rho=0.5"

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Stable, filesystem-friendly string for a swept value."""
fmt_val(v::Integer) = string(v)
fmt_val(v::Real) = string(float(v))   # 0.0, 0.3, 0.04, 1.2 ...

"""Directory (relative to the sweep root) holding a sweep root's data root."""
function sweep_dir()
    get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
        error("BROKERAGE_ABM_SWEEP_DIR is not set; the sbatch scripts / orchestrator must export it")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Cell + entry + plot-job construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_cells() -> Vector{Dict{Symbol,Any}}

Ordered list of cells. OAT cells first (in `OAT_AXES` order), then phase cells
(in `PHASE_PAIRS` order). Each cell is run for every seed in `SWEEP_SEEDS`.
"""
function build_cells()
    cells = Dict{Symbol,Any}[]

    # OAT cells
    for ax in OAT_AXES, v in ax.vals
        reldir = "oat/$(ax.label)=$(fmt_val(v))"
        push!(
            cells,
            Dict{Symbol,Any}(
                :kind => "oat",
                :axis => ax.label,
                :key => string(ax.key),
                :value => v,
                :params => Dict{Symbol,Any}(ax.key => v),
                :reldir => reldir,
            ),
        )
    end

    # Phase cells
    for pr in PHASE_PAIRS
        for (xi, xv) in enumerate(pr.xvals), (yi, yv) in enumerate(pr.yvals)
            reldir = "phase/$(pr.name)/cells/$(xi - 1)_$(yi - 1)"
            push!(
                cells,
                Dict{Symbol,Any}(
                    :kind => "phase",
                    :pair => pr.name,
                    :xkey => string(pr.xkey),
                    :xval => xv,
                    :xi => xi,
                    :ykey => string(pr.ykey),
                    :yval => yv,
                    :yi => yi,
                    :params => Dict{Symbol,Any}(pr.xkey => xv, pr.ykey => yv),
                    :reldir => reldir,
                ),
            )
        end
    end

    return cells
end

"""
    build_entries(cells) -> Vector{Dict{Symbol,Any}}

Expand cells into (cell, seed) jobs with a 0-based `index`. This is the array:
`NRUNS = length(entries)`, submitted as `--array=0-(NRUNS-1)`.
"""
function build_entries(cells)
    entries = Dict{Symbol,Any}[]
    idx = 0
    for c in cells, s in SWEEP_SEEDS
        e = Dict{Symbol,Any}(
            :index => idx,
            :seed => s,
            :kind => c[:kind],
            :reldir => c[:reldir],
            :params => c[:params],
        )
        # carry axis/pair metadata through for provenance
        for k in (:axis, :key, :value, :pair, :xkey, :xval, :xi, :ykey, :yval, :yi)
            haskey(c, k) && (e[k] = c[k])
        end
        push!(entries, e)
        idx += 1
    end
    return entries
end

"""
    build_plot_jobs(cells) -> Vector{Dict{Symbol,Any}}

One plot job per OAT cell (loads that cell's seed shards -> figures + data.jld2),
then one plot job per phase pair (loads all its grid-point shards ->
per-grid-point data.jld2 + summary.jld2 + heatmaps). 0-based `index` = plot
array task id.
"""
function build_plot_jobs(cells)
    jobs = Dict{Symbol,Any}[]
    idx = 0

    # OAT plot jobs (one per OAT cell)
    for c in cells
        c[:kind] == "oat" || continue
        job = Dict{Symbol,Any}(
            :index => idx,
            :kind => "oat_cell",
            :reldir => c[:reldir],
            :axis => c[:axis],
            :value => c[:value],
        )
        push!(jobs, job)
        idx += 1
    end

    # Phase plot jobs (one per pair)
    for pr in PHASE_PAIRS
        push!(
            jobs,
            Dict{Symbol,Any}(
                :index => idx,
                :kind => "phase_pair",
                :pair => pr.name,
                :xkey => string(pr.xkey),
                :xvals => collect(pr.xvals),
                :ykey => string(pr.ykey),
                :yvals => collect(pr.yvals),
                :reldir => "phase/$(pr.name)",
            ),
        )
        idx += 1
    end

    return jobs
end

# ─────────────────────────────────────────────────────────────────────────────
# Minimal JSON emitter (emit-only; enough for the manifest structure)
# ─────────────────────────────────────────────────────────────────────────────

function _json_escape(s::AbstractString)
    replace(
        string(s),
        '\\' => "\\\\",
        '"' => "\\\"",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
    )
end

to_json(io::IO, x::Nothing; indent::Int=0) = print(io, "null")
to_json(io::IO, x::Bool; indent::Int=0) = print(io, x ? "true" : "false")
to_json(io::IO, x::Integer; indent::Int=0) = print(io, x)
function to_json(io::IO, x::Real; indent::Int=0)
    (isnan(x) || isinf(x)) ? print(io, "null") : print(io, x)
end
to_json(io::IO, x::AbstractString; indent::Int=0) = print(io, '"', _json_escape(x), '"')
to_json(io::IO, x::Symbol; indent::Int=0) = to_json(io, string(x); indent=indent)

function to_json(io::IO, x::AbstractVector; indent::Int=0)
    isempty(x) && return print(io, "[]")
    pad = "  "^(indent + 1)
    println(io, "[")
    for (i, v) in enumerate(x)
        print(io, pad)
        to_json(io, v; indent=indent + 1)
        println(io, i < length(x) ? "," : "")
    end
    print(io, "  "^indent, "]")
end

function to_json(io::IO, x::AbstractDict; indent::Int=0)
    isempty(x) && return print(io, "{}")
    pad = "  "^(indent + 1)
    ks = collect(keys(x))
    println(io, "{")
    for (i, k) in enumerate(ks)
        print(io, pad, '"', _json_escape(string(k)), "\": ")
        to_json(io, x[k]; indent=indent + 1)
        println(io, i < length(ks) ? "," : "")
    end
    print(io, "  "^indent, "}")
end

function to_json_string(x)
    io = IOBuffer()
    to_json(io, x; indent=0)
    return String(take!(io))
end
