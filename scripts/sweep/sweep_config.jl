"""
    sweep_config.jl

Single source of truth for the parameter sweep.

Defines the OAT axes, the two-parameter grids, the baseline, the seed list, and
the storage layout. Grid coordinates that resolve to the same effective model
realization share one canonical result directory. Expands all of that into:

  * `build_cells()`   -> ordered grid coordinates with canonical result references,
  * `build_entries()` -> ordered unique (condition, seed) *jobs* (one per array
                         task), each with a 0-based `index`,
  * `build_plot_jobs()` -> the ordered list of per-cell and per-grid plot jobs.

This file has no package dependencies, so it can be `include`d from the manifest
generator, the per-task runner, the plotting script, and the isolated test
environment alike. It also provides a minimal JSON emitter so `manifest.json` is
human-readable without adding a JSON package to the project.
The machine-read mirror is `manifest.jld2`, written from the identical in-memory
structure so the two cannot drift.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Versioning
# ─────────────────────────────────────────────────────────────────────────────

"""Bump when the shard / manifest schema changes (invalidates cached shards)."""
const SWEEP_SCHEMA_VERSION = 7

# ─────────────────────────────────────────────────────────────────────────────
# Baseline + sweep specification
# ─────────────────────────────────────────────────────────────────────────────

# Baseline simulation knobs applied to every cell (others come from
# `default_params`). Simulation length and the analysis burn-in are fixed across
# the whole sweep, but only T is a model parameter.
const SWEEP_T = 500
const SWEEP_T_BURN = 30
const SWEEP_SEEDS = collect(1:parse(Int, get(ENV, "BROKERAGE_ABM_N_SEEDS", "20")))
const SWEEP_BASELINE_SEEDS = collect(
    1:parse(Int, get(ENV, "BROKERAGE_ABM_BASELINE_N_SEEDS", string(length(SWEEP_SEEDS))))
)
const SWEEP_LEARNING_MODEL = Symbol(get(ENV, "BROKERAGE_ABM_LEARNING_MODEL", "nn"))
const SWEEP_RIDGE_LAMBDA_AGENT = parse(
    Float64, get(ENV, "BROKERAGE_ABM_RIDGE_LAMBDA_AGENT", "0.001")
)
const SWEEP_RIDGE_LAMBDA_BROKER = parse(
    Float64, get(ENV, "BROKERAGE_ABM_RIDGE_LAMBDA_BROKER", "0.001")
)
const SWEEP_RIDGE_BROKER_VARIANT = Symbol(
    get(ENV, "BROKERAGE_ABM_RIDGE_BROKER_VARIANT", "pair")
)
const SWEEP_SCOPE = Symbol(get(ENV, "BROKERAGE_ABM_SWEEP_SCOPE", "full"))

SWEEP_LEARNING_MODEL in (:nn, :ridge) || error("invalid sweep learning model")
SWEEP_RIDGE_LAMBDA_AGENT > 0.0 || error("agent Ridge penalty must be positive")
SWEEP_RIDGE_LAMBDA_BROKER > 0.0 || error("broker Ridge penalty must be positive")
SWEEP_RIDGE_BROKER_VARIANT in (:pair, :size_matched, :single_principal, :additive) ||
    error("invalid sweep Ridge broker variant")
SWEEP_SCOPE in (:full, :rho_delta) || error("invalid sweep scope")
isempty(SWEEP_SEEDS) && error("sweep must include at least one seed")
length(SWEEP_BASELINE_SEEDS) < length(SWEEP_SEEDS) &&
    error("baseline seed set cannot be smaller than the general seed set")

# Full baseline over every parameter that appears on a sweep axis. A grid
# coordinate's identity is its resolved tuple over these keys, not merely its local
# override dictionary. This is what allows, for example, `oat/eta=0.02` and
# `oat/rho=0.5` to share the same realized baseline results.
const SWEEP_BASELINE = (
    rho=0.5,
    eta=0.02,
    N=1000,
    reservation_frac=0.60,
    delta=0.50,
    k=6,
    roster_frac=0.20,
    n_strangers=10,
)
const SWEEP_KEYS = keys(SWEEP_BASELINE)

# OAT axis levels. `key` is the `default_params` keyword and `label` names the
# output directory.
const RHO_CORE_VALS = [0.0, 0.5, 1.0]
const RHO_EXTENDED_VALS = [0.0, 0.5, 0.85, 1.0]
# The OAT rho axis includes the extra 0.3, 0.7, and 0.85 levels for line-plot
# resolution. The rho x delta grid uses the full OAT axis.
const RHO_OAT = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
# The 0.001 level distinguishes limited positive turnover from the qualitatively
# fixed population at zero; it is included in every grid that uses eta.
const ETA_VALS = [0.0, 0.001, 0.01, 0.02, 0.03]
const N_VALS = [500, 1000, 1500]
const R_VALS = [0.40, 0.60, 0.90, 1.20]   # reservation_frac (lambda_r)

# Matching difficulty (`delta`, the regime-gain strength) and network density
# (`k`, the initial degree).
# The matching-problem grid uses the same six rho levels as the OAT sweep and
# five delta levels over its full allowed range. At rho=1, delta drops out of
# match output exactly; those grid coordinates therefore share one realized result.
const DELTA_VALS = [0.0, 0.25, 0.50, 0.75, 1.0]
const K_VALS = [4, 12]
const ROSTER_FRAC_VALS = [0.10, 0.20, 0.40]
const N_STRANGERS_VALS = [0, 10, 50]

# Focused grid at high rho and high reservation thresholds. It is intentionally
# local rather than a global refinement of both parameters.
const RHO_R_CORNER_VALS = [0.70, 0.85, 1.0]
const R_CORNER_VALS = [0.90, 1.05, 1.20]

const OAT_AXES = [
    (label="rho", key=:rho, vals=RHO_OAT),
    (label="eta", key=:eta, vals=ETA_VALS),
    (label="N", key=:N, vals=N_VALS),
    (label="reservation_frac", key=:reservation_frac, vals=R_VALS),
    (label="delta", key=:delta, vals=DELTA_VALS),
    (label="k", key=:k, vals=K_VALS),
    (label="roster_frac", key=:roster_frac, vals=ROSTER_FRAC_VALS),
    (label="n_strangers", key=:n_strangers, vals=N_STRANGERS_VALS),
]

# Two-parameter grids: the first six are all pairwise combinations of
# {rho, eta, N, r}; the last two are targeted refinements.
const PHASE_PAIRS = [
    (name="rho_eta", xkey=:rho, xvals=RHO_EXTENDED_VALS, ykey=:eta, yvals=ETA_VALS),
    (name="rho_N", xkey=:rho, xvals=RHO_EXTENDED_VALS, ykey=:N, yvals=N_VALS),
    (name="rho_r", xkey=:rho, xvals=RHO_CORE_VALS, ykey=:reservation_frac, yvals=R_VALS),
    (name="eta_r", xkey=:eta, xvals=ETA_VALS, ykey=:reservation_frac, yvals=R_VALS),
    (name="eta_N", xkey=:eta, xvals=ETA_VALS, ykey=:N, yvals=N_VALS),
    (name="r_N", xkey=:reservation_frac, xvals=R_VALS, ykey=:N, yvals=N_VALS),
    (name="rho_delta", xkey=:rho, xvals=RHO_OAT, ykey=:delta, yvals=DELTA_VALS),
    (
        name="rho_r_corner",
        xkey=:rho,
        xvals=RHO_R_CORNER_VALS,
        ykey=:reservation_frac,
        yvals=R_CORNER_VALS,
    ),
]

const BASELINE_RELDIR = "oat/rho=0.5"

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Stable, filesystem-friendly string for a swept value."""
fmt_val(v::Integer) = string(v)
fmt_val(v::Real) = string(float(v))   # 0.0, 0.3, 0.04, 1.2 ...

"""Full resolved sweep-axis parameter dictionary for provenance and tests."""
function resolved_sweep_params(params::AbstractDict)
    Dict{Symbol,Any}(key => get(params, key, SWEEP_BASELINE[key]) for key in SWEEP_KEYS)
end

"""
Full model-realization key after applying a cell's local overrides.

When `rho == 1`, the interaction component is multiplied by zero, so `delta`
cannot affect any model event. Canonicalizing it prevents redundant simulations
while grid coordinates retain their requested `delta` metadata.
"""
function condition_key(params::AbstractDict)
    resolved = resolved_sweep_params(params)
    resolved[:rho] == 1.0 && (resolved[:delta] = SWEEP_BASELINE.delta)
    return Tuple(resolved[key] for key in SWEEP_KEYS)
end

"""True when a resolved cell is the baseline effective realization."""
is_baseline_condition(cell) = condition_key(cell[:params]) == Tuple(SWEEP_BASELINE)

"""Planned seeds for an effective condition."""
condition_seeds(cell) = is_baseline_condition(cell) ? SWEEP_BASELINE_SEEDS : SWEEP_SEEDS

"""
Annotate grid coordinates with the first result directory realizing each condition.

The first occurrence owns the simulation shards. Later occurrences retain their
grid output paths but point `:result_reldir` to that canonical directory.
"""
function link_result_cells!(cells)
    first_result = Dict{Tuple,String}()
    result_index = Dict{Tuple,Int}()
    for (grid_index, cell) in enumerate(cells)
        key = condition_key(cell[:params])
        if !haskey(first_result, key)
            first_result[key] = cell[:reldir]
            result_index[key] = length(result_index)
        end
        cell[:grid_index] = grid_index - 1
        cell[:condition_index] = result_index[key]
        cell[:result_reldir] = first_result[key]
        cell[:is_canonical] = cell[:reldir] == cell[:result_reldir]
        cell[:resolved_params] = resolved_sweep_params(cell[:params])
        cell[:seeds] = condition_seeds(cell)
    end
    return cells
end

"""Canonical grid coordinates, one for each effective model realization."""
result_cells(cells) = filter(cell -> cell[:is_canonical], cells)

"""Directory (relative to the sweep root) holding a sweep root's data root."""
function sweep_dir()
    get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
        error(
            "BROKERAGE_ABM_SWEEP_DIR is not set; the sbatch scripts / orchestrator must export it",
        )
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Cell + entry + plot-job construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_cells() -> Vector{Dict{Symbol,Any}}

Ordered list of grid coordinates. OAT coordinates come first (in `OAT_AXES` order), then
phase cells (in `PHASE_PAIRS` order). Each cell points to the canonical result
directory for its model realization.
"""
function build_cells()
    cells = Dict{Symbol,Any}[]

    if SWEEP_SCOPE == :rho_delta
        push!(
            cells,
            Dict{Symbol,Any}(
                :kind => "oat",
                :axis => "rho",
                :key => "rho",
                :value => SWEEP_BASELINE.rho,
                :params => Dict{Symbol,Any}(),
                :reldir => BASELINE_RELDIR,
            ),
        )
        pr = only(pair for pair in PHASE_PAIRS if pair.name == "rho_delta")
        for (xi, xv) in enumerate(pr.xvals), (yi, yv) in enumerate(pr.yvals)
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
                    :reldir => "phase/$(pr.name)/cells/$(xi - 1)_$(yi - 1)",
                ),
            )
        end
        return link_result_cells!(cells)
    end

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

    return link_result_cells!(cells)
end

"""
    build_entries(cells) -> Vector{Dict{Symbol,Any}}

Expand only canonical result cells into (condition, seed) jobs with a 0-based
`index`. This is the array: `NRUNS = length(entries)`, submitted as
`--array=0-(NRUNS-1)`.
"""
function build_entries(cells)
    entries = Dict{Symbol,Any}[]
    idx = 0
    for c in result_cells(cells)
        for s in c[:seeds]
            e = Dict{Symbol,Any}(
                :index => idx,
                :seed => s,
                :kind => c[:kind],
                :condition_index => c[:condition_index],
                :reldir => c[:result_reldir],
                :params => c[:params],
                :resolved_params => c[:resolved_params],
            )
            # carry axis/pair metadata through for provenance
            for k in (:axis, :key, :value, :pair, :xkey, :xval, :xi, :ykey, :yval, :yi)
                haskey(c, k) && (e[k] = c[k])
            end
            push!(entries, e)
            idx += 1
        end
    end
    return entries
end

"""
    build_plot_jobs(cells) -> Vector{Dict{Symbol,Any}}

One plot job per OAT grid coordinate (loads its referenced result shards -> figures
and data.jld2), then one plot job per phase pair (loads referenced result shards
for all grid points -> per-grid-point data.jld2 + summary.jld2 + heatmaps).
0-based `index` = plot array task id.
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
            :result_reldir => c[:result_reldir],
            :condition_index => c[:condition_index],
            :resolved_params => c[:resolved_params],
            :seeds => c[:seeds],
            :axis => c[:axis],
            :key => c[:key],
            :value => c[:value],
        )
        push!(jobs, job)
        idx += 1
    end

    # Phase plot jobs (one per pair)
    for pr in PHASE_PAIRS
        refs = [
            Dict{Symbol,Any}(
                :reldir => c[:reldir],
                :result_reldir => c[:result_reldir],
                :condition_index => c[:condition_index],
                :resolved_params => c[:resolved_params],
                :seeds => c[:seeds],
                :pair => c[:pair],
                :xkey => c[:xkey],
                :xval => c[:xval],
                :xi => c[:xi],
                :ykey => c[:ykey],
                :yval => c[:yval],
                :yi => c[:yi],
            ) for c in cells if c[:kind] == "phase" && c[:pair] == pr.name
        ]
        isempty(refs) && continue
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
                :cell_refs => refs,
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
