"""
    scripts/paper/figdata.jl

Cluster-side extract for the figure pipeline. Reads the full sweep
(`BROKERAGE_ABM_SWEEP_DIR`) and writes the small derived dataset from which
the figure renderer operates. The default output is `output/main/figdata.jld2`;
set `BROKERAGE_ABM_FIGDATA_PATH` to generate a counterpart for another sweep.
The renderer then works locally with no access to the sweep. No hard-coded
results: every stored value is computed from the saved sweep data at run time.

Rerun only when the underlying numbers change (a new sweep, or a figure needing a
metric not yet extracted); styling iteration needs only figures.jl.

Contents of figdata.jld2 (single key "figdata", a Dict):
  period         per-period time axis of the baseline runs
  series         per-period ensemble means at the baseline, retained for
                 convenience
  series_seed_values
                 period-by-seed matrices at the baseline:
                 betweenness, access, mean_degree, median_degree, mpa
                 (matches per agent, both sides), outsourcing
  oat_cells      late means and seed values per one-at-a-time regime
  grid_cells     rho, delta, and the nine named outcome late means per rho x delta
                 grid coordinate, plus the underlying seed values
  regime_cells   rho, delta, late means, and seed values per effective realization
  meta           sweep id, generation time, generating script

Usage: BROKERAGE_ABM_SWEEP_DIR=<sweep root> julia --project scripts/paper/figdata.jl
"""

using JLD2, DataFrames, Statistics, Dates

include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const DEFAULT_OUTFILE = normpath(
    joinpath(@__DIR__, "..", "..", "output", "main", "figdata.jld2")
)
const OUTFILE = normpath(get(ENV, "BROKERAGE_ABM_FIGDATA_PATH", DEFAULT_OUTFILE))
const REPORTING_PROVENANCE = reporting_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."))
)
const LATE_WIDTH = 20   # final-period window, the headline statistic
const SWEEP = load_sweep_dataset(ROOT)
const BASELINE_REL = "oat/rho=0.5"
length(SWEEP.result_by_rel[BASELINE_REL].seeds) == 50 || error("expected 50 baseline seeds")
all(result.rel == BASELINE_REL || length(result.seeds) == 20 for result in SWEEP.results) ||
    error("expected 20 seeds outside the baseline")

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
load_mdfs(rel) = grid_result(SWEEP, rel).mdfs
load_cfg(rel) = grid_result(SWEEP, rel).cfg
function gap_values(mdfs, broker_col, agent_col)
    seed_values(mdfs, broker_col) .- seed_values(mdfs, agent_col)
end
qgap_values(mdfs) = gap_values(mdfs, :q_broker_mean, :q_self_mean)
rankgap_values(mdfs) = gap_values(mdfs, :broker_holdout_rank, :agent_holdout_rank)
r2gap_values(mdfs) = gap_values(mdfs, :broker_holdout_r2, :agent_holdout_r2)
seed_series(mdfs, f) = reduce(hcat, (Float64.(f(df)) for df in mdfs))
function ensemble_mean(values::AbstractMatrix)
    Float64[nanmean(view(values, period_index, :)) for period_index in axes(values, 1)]
end

# the nine outcomes shared by figures 2 and 3; names must match figures.jl panels
function outcome_seed_values(m)
    Dict{String,Vector{Float64}}(
        "Betweenness centrality" => seed_values(m, :betweenness),
        "Access fraction" => access_values(m),
        "Broker prediction R²" => seed_values(m, :broker_holdout_r2),
        "Prediction R² gap" => r2gap_values(m),
        "Broker rank correlation" => seed_values(m, :broker_holdout_rank),
        "Rank correlation gap" => rankgap_values(m),
        "Broker output q" => seed_values(m, :q_broker_mean),
        "Output gap q" => qgap_values(m),
        "Outsourcing rate" => seed_values(m, :outsourcing_rate),
    )
end
outcome_means(values) = Dict(key => nanmean(seed_values) for (key, seed_values) in values)

fd = Dict{String,Any}()

# ── baseline per-period ensemble series ──
baseline_rel = BASELINE_REL
baseline = load_mdfs(baseline_rel)
N = load_cfg(baseline_rel)["N"]
fd["period"] = collect(baseline[1].period)
function series_seed_values(m, N)
    Dict{String,Matrix{Float64}}(
        "betweenness" => seed_series(m, d -> d.betweenness),
        "access" => seed_series(m, accessf),
        "mean_degree" => seed_series(m, d -> d.mean_degree),
        "median_degree" => seed_series(m, d -> d.median_degree),
        "mpa" => seed_series(m, d -> 2 .* d.n_total_matches ./ N),
        "outsourcing" => seed_series(m, d -> d.outsourcing_rate),
    )
end
fd["baseline_seeds"] = copy(SWEEP.result_by_rel[baseline_rel].seeds)
fd["series_seed_values"] = series_seed_values(baseline, N)
fd["series"] = Dict(
    key => ensemble_mean(values) for (key, values) in fd["series_seed_values"]
)

# ── one-at-a-time cells (figure 1 scatter) ──
fd["oat_cells"] = let out = Dict{String,Any}[]
    seen = Set{String}()
    for grid in SWEEP.grid_cells
        grid[:kind] == "oat" || continue
        result = SWEEP.result_by_rel[grid[:result_reldir]]
        result.rel in seen && continue
        push!(seen, result.rel)
        m = result.mdfs
        seed_data = Dict(
            "betw" => seed_values(m, :betweenness),
            "access" => access_values(m),
            "qgap" => qgap_values(m),
        )
        b = nanmean(seed_data["betw"])
        a = nanmean(seed_data["access"])
        (isnan(b) || isnan(a)) && continue
        push!(
            out,
            Dict(
                "rel" => result.rel,
                "seeds" => copy(result.seeds),
                "betw" => b,
                "access" => a,
                "qgap" => nanmean(seed_data["qgap"]),
                "seed_values" => seed_data,
            ),
        )
    end
    out
end

# ── rho x delta grid cells + OAT rho refiners (figures 2 and 3) ──
fd["grid_cells"] = let out = Dict{String,Any}[]
    for grid in SWEEP.grid_cells
        get(grid, :pair, nothing) == "rho_delta" || continue
        m = SWEEP.result_by_rel[grid[:result_reldir]].mdfs
        result = SWEEP.result_by_rel[grid[:result_reldir]]
        cfg = grid[:resolved_params]
        seed_data = outcome_seed_values(m)
        push!(
            out,
            Dict(
                "rel" => result.rel,
                "seeds" => copy(result.seeds),
                "rho" => Float64(cfg[:rho]),
                "delta" => Float64(cfg[:delta]),
                "outcomes" => outcome_means(seed_data),
                "outcome_seed_values" => seed_data,
            ),
        )
    end
    out
end

# ── every effective realization (figure 3) ──
fd["regime_cells"] = let out = Dict{String,Any}[]
    for result in SWEEP.results
        m, cfg = result.mdfs, result.cfg
        seed_data = Dict(
            "betw" => seed_values(m, :betweenness),
            "access" => access_values(m),
            "rankgap" => rankgap_values(m),
            "qgap" => qgap_values(m),
        )
        b = nanmean(seed_data["betw"])
        isnan(b) && continue
        push!(
            out,
            Dict(
                "rel" => result.rel,
                "seeds" => copy(result.seeds),
                "rho" => Float64(cfg["rho"]),
                "delta" => Float64(cfg["delta"]),
                "betw" => b,
                "access" => nanmean(seed_data["access"]),
                "rankgap" => nanmean(seed_data["rankgap"]),
                "qgap" => nanmean(seed_data["qgap"]),
                "seed_values" => seed_data,
            ),
        )
    end
    out
end

fd["meta"] = Dict(
    "sweep" => basename(ROOT),
    "manifest_hash" => SWEEP.manifest_hash,
    "schema_version" => SWEEP.schema_version,
    "n_runs" => sum(length(result.seeds) for result in SWEEP.results),
    "baseline_n_seeds" => length(SWEEP.result_by_rel[baseline_rel].seeds),
    "learning_model" => String(get(load_cfg(baseline_rel), "learning_model", "nn")),
    "condition_seed_counts" =>
        Dict(result.rel => length(result.seeds) for result in SWEEP.results),
    "generated" => string(now()),
    "source" => "scripts/paper/figdata.jl",
    "analysis_git_commit" => REPORTING_PROVENANCE.commit,
    "analysis_source_clean" => REPORTING_PROVENANCE.source_clean,
)

mkpath(dirname(OUTFILE))
jldsave(OUTFILE; figdata=fd)
println(
    "wrote $OUTFILE ($(round(filesize(OUTFILE) / 1024; digits=1)) KB; ",
    length(fd["regime_cells"]),
    " effective realizations)",
)
