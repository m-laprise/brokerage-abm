"""
    scripts/paper/ridge_supplement.jl

Emit the Ridge and NN comparison values quoted in the paper. Inputs are retained
figure data under `output/{main,ridge/paired}/` and the paired condition comparison
under `output/ridge/paired/analysis/`.

Outputs:

  * `output/ridge/paired/analysis/paper_values.tex`

No raw sweep access is required. Every estimate and count is computed from
retained data. The Ridge input and output paths can be overridden
with `BROKERAGE_ABM_RIDGE_FIGDATA_PATH` and
`BROKERAGE_ABM_RIDGE_PAPER_VALUES_PATH`.

Usage: julia --project --threads=auto scripts/paper/ridge_supplement.jl
"""

include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

using JLD2
using Printf: @sprintf
using Statistics: mean, median

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = get(
    ENV,
    "BROKERAGE_ABM_RIDGE_FIGDATA_PATH",
    joinpath(REPO, "output", "ridge", "paired", "figure_data.jld2"),
)
const VALUES = get(
    ENV,
    "BROKERAGE_ABM_RIDGE_PAPER_VALUES_PATH",
    joinpath(REPO, "output", "ridge", "paired", "analysis", "paper_values.tex"),
)
const BASELINE_REL = "oat/rho=0.5"

const PROVENANCE = manuscript_git_provenance(
    REPO;
    sources=(
        @__FILE__,
        "scripts/monte_carlo.jl",
    ),
)
isfile(INPUT) || error("missing base Ridge figure data: $INPUT")
const FD = JLD2.load(INPUT)["figdata"]
const META = FD["meta"]
const DATA_ANALYSIS_COMMIT = validate_analysis_commit(
    PROVENANCE,
    META["analysis_git_commit"];
    artifact="base Ridge figure data",
)
META["analysis_source_clean"] == true ||
    error("base Ridge figure data were extracted from dirty analysis sources")
META["learning_model"] == "ridge" || error("expected base Ridge figure data")

const REGIMES = FD["regime_cells"]
length(REGIMES) == length(META["condition_seed_counts"]) ||
    error("base Ridge effective-realization count does not match its metadata")
all(length(cell["seeds"]) == (cell["rel"] == BASELINE_REL ? 50 : 20) for cell in REGIMES) ||
    error("unexpected base Ridge seed plan")

rank_interval(cell) = monte_carlo_interval(cell["seed_values"]["rankgap"])

const REGIME_POINTS = sort(
    [
        let interval = rank_interval(cell)
            (
                rel=cell["rel"],
                mean=interval.mean,
                lower=interval.lower,
                upper=interval.upper,
            )
        end for cell in REGIMES
    ];
    by=point -> point.mean,
)
const BASELINE = only(point for point in REGIME_POINTS if point.rel == BASELINE_REL)
const RIDGE_MEDIAN = median(point.mean for point in REGIME_POINTS)
const POSITIVE_N = count(point -> point.mean > 0.0, REGIME_POINTS)

"""Validate retained NN-Ridge comparisons and compute subsection-1 values."""
function assessment_values()
    nn = JLD2.load(joinpath(REPO, "output", "main", "figure_data.jld2"))["figdata"]
    nn_meta = nn["meta"]
    nn_meta["analysis_source_clean"] == true || error("NN figure data used dirty sources")
    nn_meta["learning_model"] == "nn" || error("expected NN figure data")
    nn_commit = validate_analysis_commit(
        PROVENANCE, nn_meta["analysis_git_commit"]; artifact="NN figure data"
    )
    root = joinpath(dirname(INPUT), "analysis")
    provenance = Dict(Tuple(split(line, '='; limit=2)) for
        line in readlines(joinpath(root, "provenance.txt")))
    comparison_commit = validate_analysis_commit(
        PROVENANCE, provenance["analysis_commit"]; artifact="NN-Ridge comparison"
    )
    provenance["analysis_source_clean"] == "true" || error("paired comparison used dirty sources")
    early = (51, 70)
    late = (maximum(nn["period"]) - 19, maximum(nn["period"]))
    provenance["late_periods"] == "$(late[1]):$(late[2])" || error("comparison window differs")
    nn["period"] == FD["period"] || error("NN and Ridge periods differ")
    for (prefix, data) in (("nn", nn), ("ridge", FD))
        provenance["$(prefix)_manifest"] == data["meta"]["manifest_hash"] ||
            error("$prefix comparison manifest differs")
        parse(Int, provenance["$(prefix)_schema"]) == data["meta"]["schema_version"] ||
            error("$prefix comparison schema differs")
        for cell in data["regime_cells"]
            key = cell["rel"] == BASELINE_REL ? "baseline_comparison_seeds" : "other_comparison_seeds"
            parse.(Int, split(provenance[key], ',')) == cell["seeds"] ||
                error("$prefix comparison seed plan differs")
        end
    end
    lines = readlines(joinpath(root, "condition_comparison.tsv"))
    columns = split(first(lines), '\t')
    rows = [Dict(zip(columns, split(line, '\t'))) for line in lines[2:end]]
    by_rel = Dict(row["result_reldir"] => row for row in rows)
    length(by_rel) == length(rows) || error("duplicate comparison rows")
    value(row, key) = parse(Float64, row[key])
    for (prefix, data) in (("nn", nn), ("ridge", FD))
        Set(keys(by_rel)) == Set(cell["rel"] for cell in data["regime_cells"]) ||
            error("$prefix comparison regime coverage differs")
        for cell in data["regime_cells"]
            row = by_rel[cell["rel"]]
            for metric in ("outsourcing_rate", "access_fraction", "agent_holdout_rank", "rank_gap")
                parse(Int, row["delta_n_$metric"]) == length(cell["seeds"]) ||
                    error("comparison has missing seed observations")
                isfinite(value(row, "$(prefix)_$metric")) || error("nonfinite comparison value")
            end
            for (metric, key) in (("access_fraction", "access"), ("rank_gap", "rankgap"))
                isapprox(value(row, "$(prefix)_$metric"), mean(cell["seed_values"][key]);
                    atol=1e-12, rtol=1e-10) || error("comparison does not reproduce $prefix $metric")
            end
            isapprox(value(row, "$(prefix)_broker_holdout_rank") - value(row, "$(prefix)_agent_holdout_rank"),
                value(row, "$(prefix)_rank_gap"); atol=1e-12, rtol=1e-10) ||
                error("ranking means do not reproduce $prefix gap")
        end
        mask = late[1] .<= data["period"] .<= late[2]
        isapprox(value(by_rel[BASELINE_REL], "$(prefix)_outsourcing_rate"),
            mean(data["series_seed_values"]["outsourcing"][mask, :]);
            atol=1e-12, rtol=1e-10) || error("baseline outsourcing differs")
    end
    values = Dict{String,String}()
    for (name, key) in (("out", "outsourcing"), ("access", "access")),
        (label, window) in (("Early", early), ("Late", late))
        mask = window[1] .<= nn["period"] .<= window[2]
        seed_means = vec(mean(nn["series_seed_values"][key][mask, :]; dims=1))
        all(isfinite, seed_means) || error("missing baseline $key observations")
        values["$(name)Baseline$(label)Percent"] = @sprintf("%.0f", 100 * mean(seed_means))
    end
    baseline = only(cell for cell in nn["regime_cells"] if cell["rel"] == BASELINE_REL)
    gap = rank_interval(baseline)
    values["rankGapBaselineLower"] = @sprintf("%.2f", gap.lower)
    values["rankGapBaselineUpper"] = @sprintf("%.2f", gap.upper)
    # This is the descriptive threshold in the approved prose, not an estimate.
    assessment_floor = 0.8
    all(1 - value(row, "$(prefix)_access_fraction") > assessment_floor
        for row in rows for prefix in ("nn", "ridge")) ||
        error("the stated assessment-share bound no longer holds")
    all(rank_interval(cell).lower > 0 for cell in nn["regime_cells"]) ||
        error("the stated positive ranking intervals no longer hold")
    higher = count(row -> value(row, "nn_agent_holdout_rank") > value(row, "ridge_agent_holdout_rank"), rows)
    higher == length(rows) || error("NN does not raise principal accuracy in every regime")
    values["assessmentShareFloorPercent"] = @sprintf("%.0f", 100 * assessment_floor)
    values["ridgeOutDominantN"] = string(count(row -> value(row, "ridge_outsourcing_rate") > 0.5, rows))
    values["nnPrincipalRankHigherN"] = string(higher)
    values["nnRankGapNarrowerN"] = string(count(row -> value(row, "nn_rank_gap") < value(row, "ridge_rank_gap"), rows))
    return (; values, comparison_commit, nn_commit)
end

function write_values()
    assessment = assessment_values()
    mkpath(dirname(VALUES))
    open(VALUES, "w") do io
        println(io, "% Generated by scripts/paper/ridge_supplement.jl.")
        println(io, "% Data analysis commit: $DATA_ANALYSIS_COMMIT")
        println(io, "% NN analysis commit: $(assessment.nn_commit)")
        println(io, "% NN-Ridge comparison analysis commit: $(assessment.comparison_commit)")
        println(io, "% Reporting commit: $(PROVENANCE.commit)")
        println(io, "% Reporting source clean: $(PROVENANCE.source_clean)")
        write_source_provenance(io, PROVENANCE)
        println(io, "% Ridge sweep: ", META["sweep"])
        println(io, "\\pvDefine{ridgeNConditions}{$(length(REGIME_POINTS))}")
        println(io, "\\pvDefine{ridgePositiveN}{$POSITIVE_N}")
        println(io, "\\pvDefine{ridgeBaselineGap}{", @sprintf("%.3f", BASELINE.mean), "}")
        println(io, "\\pvDefine{ridgeMedianGap}{", @sprintf("%.3f", RIDGE_MEDIAN), "}")
        for key in sort(collect(keys(assessment.values)))
            println(io, "\\pvDefine{$key}{$(assessment.values[key])}")
        end
    end
    println("wrote $VALUES")
    return nothing
end

write_values()
