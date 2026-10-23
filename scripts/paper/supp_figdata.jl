"""Extract supplementary structural measures and settings from a completed sweep.
Set BROKERAGE_ABM_SWEEP_DIR."""

using JLD2, DataFrames, Statistics, Dates

isdefined(@__MODULE__, :AnalysisSettings) || include(joinpath(@__DIR__, "..", "analysis_settings.jl"))
isdefined(@__MODULE__, :ReportingSettings) || include(joinpath(@__DIR__, "..", "reporting_settings.jl"))

include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUTFILE = normpath(
    joinpath(@__DIR__, "..", "..", "output", "supplement", "structural_figure_data.jld2")
)
const REPORTING_PROVENANCE = reporting_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."));
    sources=("scripts/analysis_settings.jl", "scripts/reporting_settings.jl",
        @__FILE__,
        "scripts/sweep/sweep_results.jl",
    ),
)
const LATE_WIDTH = AnalysisSettings.LATE_WIDTH
const SWEEP = load_sweep_dataset(ROOT)
const REPORTING_SETTINGS = merge(
    ReportingSettings.sweep_metadata(REPORTING_PROVENANCE.root, ReportingSettings.manifest(ROOT)),
    AnalysisSettings.metadata(Int(SWEEP.meta[:T])),
)
const BASELINE_REL = REPORTING_SETTINGS["baseline_reldir"]

# ── shared reducers (kept local so this extract does not depend on figdata.jl) ──
nanmean(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
late_mask(df) = df.period .>= maximum(df.period) - LATE_WIDTH + 1
tailmean(df, col) = nanmean(df[late_mask(df), col])
seed_values(mdfs, col) = Float64[tailmean(d, col) for d in mdfs]
function accessf(df)
    (
        t=(df.access_count .+ df.assessment_count);
        [t[i] > 0 ? df.access_count[i] / t[i] : NaN for i in eachindex(t)]
    )
end
access_tail(df) = nanmean(accessf(df)[late_mask(df)])
access_values(mdfs) = Float64[access_tail(d) for d in mdfs]
# broker-minus-self gaps, the y-axes of S3 (identical definitions to figdata.jl)
function gap_values(mdfs, broker_col, agent_col)
    seed_values(mdfs, broker_col) .- seed_values(mdfs, agent_col)
end
qgap_values(mdfs) = gap_values(mdfs, :q_broker_mean, :q_self_mean)
rankgap_values(mdfs) = gap_values(mdfs, :broker_holdout_rank, :agent_holdout_rank)
seed_series(mdfs, f) = reduce(hcat, (Float64.(f(df)) for df in mdfs))
function ensemble_mean(values::AbstractMatrix)
    Float64[nanmean(view(values, period_index, :)) for period_index in axes(values, 1)]
end
load_mdfs(rel) = grid_result(SWEEP, rel).mdfs

fd = Dict{String,Any}()

# ── baseline per-period ensemble series (S2 left) ──
baseline = load_mdfs(BASELINE_REL)
fd["period"] = collect(baseline[1].period)
function series_seed_values(m)
    Dict{String,Matrix{Float64}}(
        "constraint" => seed_series(m, d -> d.constraint),
        "effective_size" => seed_series(m, d -> d.effective_size),
    )
end
fd["baseline_seeds"] = copy(SWEEP.result_by_rel[BASELINE_REL].seeds)
fd["series_seed_values"] = series_seed_values(baseline)
fd["series"] = Dict(
    key => ensemble_mean(values) for (key, values) in fd["series_seed_values"]
)

# ── one-at-a-time cells (S2 right scatter) ──
# same regime list as figdata.jl's oat_cells, so the supplement scatter spans the
# same regimes as the main-text position analysis.
fd["oat_cells"] = let out = Dict{String,Any}[]
    seen = Set{String}()
    for grid in SWEEP.grid_cells
        grid[:kind] == "oat" || continue
        result = SWEEP.result_by_rel[grid[:result_reldir]]
        result.rel in seen && continue
        push!(seen, result.rel)
        m = result.mdfs
        seed_data = Dict(
            "access" => access_values(m),
            "constraint" => seed_values(m, :constraint),
            "effsize" => seed_values(m, :effective_size),
        )
        a = nanmean(seed_data["access"])
        c = nanmean(seed_data["constraint"])
        e = nanmean(seed_data["effsize"])
        (isnan(a) || isnan(c) || isnan(e)) && continue
        push!(
            out,
            Dict(
                "rel" => result.rel,
                "seeds" => copy(result.seeds),
                "access" => a,
                "constraint" => c,
                "effsize" => e,
                "seed_values" => seed_data,
            ),
        )
    end
    out
end

# ── rho x delta grid cells + OAT rho refiners (S1) ──
# identical cell set to figdata.jl's grid_cells, so the supplement S1 lines span
# the same grid as the main-text matching-grid analysis.
fd["grid_cells"] = let out = Dict{String,Any}[]
    for grid in SWEEP.grid_cells
        get(grid, :pair, nothing) == "rho_delta" || continue
        m = SWEEP.result_by_rel[grid[:result_reldir]].mdfs
        result = SWEEP.result_by_rel[grid[:result_reldir]]
        cfg = grid[:resolved_params]
        constraint_values = seed_values(m, :constraint)
        effective_size_values = seed_values(m, :effective_size)
        push!(
            out,
            Dict(
                "rel" => result.rel,
                "seeds" => copy(result.seeds),
                "rho" => Float64(cfg[:rho]),
                "delta" => Float64(cfg[:delta]),
                "constraint" => nanmean(constraint_values),
                "effsize" => nanmean(effective_size_values),
                "seed_values" => Dict(
                    "constraint" => constraint_values,
                    "effsize" => effective_size_values,
                ),
            ),
        )
    end
    out
end

# ── every effective realization (S3) ──
fd["regime_cells"] = let out = Dict{String,Any}[]
    for result in SWEEP.results
        m, cfg = result.mdfs, result.cfg
        seed_data = Dict(
            "constraint" => seed_values(m, :constraint),
            "effsize" => seed_values(m, :effective_size),
            "rankgap" => rankgap_values(m),
            "qgap" => qgap_values(m),
        )
        c = nanmean(seed_data["constraint"])
        isnan(c) && continue
        push!(
            out,
            Dict(
                "rel" => result.rel,
                "seeds" => copy(result.seeds),
                "rho" => Float64(cfg["rho"]),
                "delta" => Float64(cfg["delta"]),
                "constraint" => c,
                "effsize" => nanmean(seed_data["effsize"]),
                "rankgap" => nanmean(seed_data["rankgap"]),
                "qgap" => nanmean(seed_data["qgap"]),
                "seed_values" => seed_data,
            ),
        )
    end
    out
end

fd["meta"] = Dict(
    "reporting_settings" => REPORTING_SETTINGS,
    "sweep" => basename(ROOT),
    "manifest_hash" => SWEEP.manifest_hash,
    "schema_version" => SWEEP.schema_version,
    "n_runs" => sum(length(result.seeds) for result in SWEEP.results),
    "baseline_n_seeds" => length(SWEEP.result_by_rel[BASELINE_REL].seeds),
    "condition_seed_counts" =>
        Dict(result.rel => length(result.seeds) for result in SWEEP.results),
    "generated" => string(now()),
    "source" => "scripts/paper/supp_figdata.jl",
    "analysis_git_commit" => REPORTING_PROVENANCE.commit,
    "analysis_source_clean" => REPORTING_PROVENANCE.source_clean,
)

mkpath(dirname(OUTFILE))
jldsave(OUTFILE; figdata=fd)
println(
    "wrote $OUTFILE ($(round(filesize(OUTFILE) / 1024; digits=1)) KB; ",
    length(fd["regime_cells"]),
    " effective realizations, ",
    length(fd["oat_cells"]),
    " OAT cells, ",
    length(fd["grid_cells"]),
    " grid cells)",
)
