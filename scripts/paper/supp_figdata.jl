"""Extract supplementary structural measures and settings from a completed sweep.
Set BROKERAGE_ABM_SWEEP_DIR."""

using JLD2, DataFrames, Statistics, Dates

isdefined(@__MODULE__, :AnalysisSettings) ||
    include(joinpath(@__DIR__, "..", "analysis_settings.jl"))
isdefined(@__MODULE__, :ReportingSettings) ||
    include(joinpath(@__DIR__, "..", "reporting_settings.jl"))

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
    sources=(
        "scripts/analysis_settings.jl",
        "scripts/reporting_settings.jl",
        @__FILE__,
        "scripts/sweep/sweep_results.jl",
    ),
)
const LATE_WIDTH = AnalysisSettings.LATE_WIDTH
const SWEEP = load_sweep_dataset(ROOT)
const REPORTING_SETTINGS = merge(
    ReportingSettings.sweep_metadata(
        REPORTING_PROVENANCE.root, ReportingSettings.manifest(ROOT)
    ),
    AnalysisSettings.metadata(Int(SWEEP.meta[:T])),
)
const BASELINE_REL = REPORTING_SETTINGS["baseline_reldir"]

# ── shared reducers (kept local so this extract does not depend on figdata.jl) ──
nanmean(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
late_mask(df) = df.period .>= maximum(df.period) - LATE_WIDTH + 1
tailmean(df, col) = nanmean(df[late_mask(df), col])
seed_values(mdfs, col) = Float64[tailmean(d, col) for d in mdfs]
baseline = grid_result(SWEEP, BASELINE_REL)
fd = Dict{String,Any}(
    "period" => collect(first(baseline.mdfs).period),
    "baseline_seeds" => copy(baseline.seeds),
)

fd["regime_cells"] = let out = Dict{String,Any}[]
    for result in SWEEP.results
        m, cfg = result.mdfs, result.cfg
        seed_data = Dict(
            "constraint" => seed_values(m, :constraint),
            "effsize" => seed_values(m, :effective_size),
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
println("wrote $OUTFILE ($(length(fd["regime_cells"])) regimes)")
