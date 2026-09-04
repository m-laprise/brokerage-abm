"""
    scripts/paper/audit_convergence.jl

Reproducible Monte Carlo convergence audit for the canonical reporting sweep.
Each effective realization is evaluated once. For every reported figure
outcome, the script computes a Student-t 95% interval from seed-level
late-window means and compares a nested seed estimate with the full estimate:
the first 10 versus all 20 seeds generally, and the first 20 versus all 50 at
the baseline.

Relative precision uses the conventional ratio of interval half-width or
nested-seed change to the absolute full-sample estimate. Relative precision is
not reported when the 95% interval contains zero; those cells retain their
absolute intervals in the audit table. R² outcomes remain in the audit tables
but are excluded from the cross-outcome ranges written for the methods section.

Required environment:

  BROKERAGE_ABM_SWEEP_DIR

Optional environment:

  BROKERAGE_ABM_CONVERGENCE_DIR

The default output directory is `output/main/convergence/`.

Usage:
  BROKERAGE_ABM_SWEEP_DIR=/path/to/sweep \
    julia --project --threads=auto scripts/paper/audit_convergence.jl
"""

using DataFrames: DataFrame
using Dates: now
using Printf: @sprintf
using Statistics: mean, quantile

include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUT_DIR = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_CONVERGENCE_DIR",
        joinpath(@__DIR__, "..", "..", "output", "main", "convergence"),
    ),
)
const BASELINE_REL = "oat/rho=0.5"
const LATE_WIDTH = 20
const R2_OUTCOMES = Set((:agent_holdout_r2, :broker_holdout_r2, :r2_gap))
const OUTCOMES = (
    :matches_per_agent,
    :mean_degree,
    :median_degree,
    :betweenness,
    :constraint,
    :effective_size,
    :access_fraction,
    :outsourcing_rate,
    :agent_holdout_rank,
    :broker_holdout_rank,
    :rank_gap,
    :q_self_mean,
    :q_broker_mean,
    :output_gap,
    :agent_holdout_r2,
    :broker_holdout_r2,
    :r2_gap,
)

nanmean(values) = begin
    kept = filter(isfinite, Float64.(collect(values)))
    isempty(kept) ? NaN : mean(kept)
end

function metric_series(df::DataFrame, metric::Symbol, config)
    if metric == :matches_per_agent
        return 2.0 .* df.n_total_matches ./ Int(config["N"])
    elseif metric == :access_fraction
        total = df.access_count .+ df.assessment_count
        return [
            total[index] > 0 ? df.access_count[index] / total[index] : NaN for
            index in eachindex(total)
        ]
    elseif metric == :rank_gap
        return df.broker_holdout_rank .- df.agent_holdout_rank
    elseif metric == :output_gap
        return df.q_broker_mean .- df.q_self_mean
    elseif metric == :r2_gap
        return df.broker_holdout_r2 .- df.agent_holdout_r2
    end
    return df[!, metric]
end

function seed_late(df::DataFrame, metric::Symbol, config)::Float64
    first_period = maximum(df.period) - LATE_WIDTH + 1
    mask = df.period .>= first_period
    return nanmean(metric_series(df, metric, config)[mask])
end

function seed_values(result, metric)
    Float64[seed_late(df, metric, result.cfg) for df in result.mdfs]
end

function write_tsv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, '\t'))
        for row in rows
            println(io, join(row, '\t'))
        end
    end
    return nothing
end

function main()
    provenance = reporting_git_provenance(normpath(joinpath(@__DIR__, "..", "..")))
    mkpath(OUT_DIR)
    sweep = load_sweep_dataset(ROOT)
    length(sweep.results) == 98 || error("expected 98 effective realizations")
    baseline = sweep.result_by_rel[BASELINE_REL]
    length(baseline.seeds) == 50 || error("expected 50 baseline seeds")
    all(
        result.rel == BASELINE_REL || length(result.seeds) == 20 for result in sweep.results
    ) || error("expected 20 seeds outside the baseline")

    rows = NamedTuple[]
    for metric in OUTCOMES
        metric_rows = NamedTuple[]
        for result in sort(sweep.results; by=result -> result.rel)
            values = seed_values(result, metric)
            full = monte_carlo_interval(values)
            nested_n = result.rel == BASELINE_REL ? 20 : 10
            nested = monte_carlo_interval(view(values, 1:nested_n))
            push!(
                metric_rows,
                (;
                    outcome=metric,
                    rel=result.rel,
                    n_seeds=full.n,
                    nested_n,
                    estimate=full.mean,
                    ci_lower=full.lower,
                    ci_upper=full.upper,
                    ci_half_width=(full.upper - full.lower) / 2.0,
                    nested_estimate=nested.mean,
                    nested_change=abs(full.mean - nested.mean),
                ),
            )
        end
        append!(
            rows,
            [
                merge(
                    row,
                    let
                        relative_defined =
                            isfinite(row.estimate) &&
                            isfinite(row.ci_lower) &&
                            isfinite(row.ci_upper) &&
                            (row.ci_lower > 0.0 || row.ci_upper < 0.0)
                        scale = abs(row.estimate)
                        (;
                            relative_precision_defined=relative_defined,
                            relative_ci_half_width=(
                                relative_defined ? row.ci_half_width / scale : NaN
                            ),
                            relative_nested_change=(
                                relative_defined ? row.nested_change / scale : NaN
                            ),
                        )
                    end,
                ) for row in metric_rows
            ],
        )
    end

    detail_header = (
        "outcome",
        "result_reldir",
        "n_seeds",
        "nested_n",
        "estimate",
        "ci_lower",
        "ci_upper",
        "ci_half_width",
        "nested_estimate",
        "nested_change",
        "relative_precision_defined",
        "relative_ci_half_width",
        "relative_nested_change",
    )
    detail_rows = [
        (
            row.outcome,
            row.rel,
            row.n_seeds,
            row.nested_n,
            row.estimate,
            row.ci_lower,
            row.ci_upper,
            row.ci_half_width,
            row.nested_estimate,
            row.nested_change,
            row.relative_precision_defined,
            row.relative_ci_half_width,
            row.relative_nested_change,
        ) for row in rows
    ]
    write_tsv(joinpath(OUT_DIR, "condition_audit.tsv"), detail_header, detail_rows)

    summaries = NamedTuple[]
    for metric in OUTCOMES
        metric_rows = filter(row -> row.outcome == metric, rows)
        ci = filter(isfinite, [row.relative_ci_half_width for row in metric_rows])
        change = filter(isfinite, [row.relative_nested_change for row in metric_rows])
        push!(
            summaries,
            (;
                outcome=metric,
                includes_r2=metric in R2_OUTCOMES,
                n_conditions=length(metric_rows),
                n_relative_precision=length(ci),
                p50_relative_ci_half_width=quantile(ci, 0.50),
                p90_relative_ci_half_width=quantile(ci, 0.90),
                max_relative_ci_half_width=maximum(ci),
                p50_relative_nested_change=quantile(change, 0.50),
                p90_relative_nested_change=quantile(change, 0.90),
                max_relative_nested_change=maximum(change),
            ),
        )
    end
    summary_header = propertynames(first(summaries))
    summary_rows = [Tuple(summary) for summary in summaries]
    write_tsv(joinpath(OUT_DIR, "outcome_summary.tsv"), summary_header, summary_rows)

    non_r2 = filter(summary -> !summary.includes_r2, summaries)
    ci_median = [summary.p50_relative_ci_half_width for summary in non_r2]
    change_median = [summary.p50_relative_nested_change for summary in non_r2]
    percent(value) = @sprintf("%.1f", 100.0 * value)
    values = (
        "mcOutcomeCount" => string(length(non_r2)),
        "mcCiMedianMinPercent" => percent(minimum(ci_median)),
        "mcCiMedianMaxPercent" => percent(maximum(ci_median)),
        "mcNestedMedianMinPercent" => percent(minimum(change_median)),
        "mcNestedMedianMaxPercent" => percent(maximum(change_median)),
    )
    open(joinpath(OUT_DIR, "values.tex"), "w") do io
        println(io, "% Generated by scripts/paper/audit_convergence.jl on $(now()).")
        println(io, "% Analysis commit: $(provenance.commit)")
        println(io, "% Analysis source clean: $(provenance.source_clean)")
        println(io, "% R2 outcomes are excluded from these cross-outcome ranges.")
        for (key, value) in values
            println(io, "\\pvDefine{$key}{$value}")
        end
    end
    open(joinpath(OUT_DIR, "summary.txt"), "w") do io
        println(io, "generated=$(now())")
        println(io, "sweep=$(basename(normpath(ROOT)))")
        println(io, "manifest=$(sweep.manifest_hash)")
        println(io, "analysis_commit=$(provenance.commit)")
        println(io, "analysis_source_clean=$(provenance.source_clean)")
        println(io, "effective_realizations=$(length(sweep.results))")
        println(io, "general_nested_comparison=first 10 versus all 20 seeds")
        println(io, "baseline_nested_comparison=first 20 versus all 50 seeds")
        println(io, "normalization=absolute full-sample condition estimate")
        println(io, "relative_precision_exclusion=95% interval contains zero")
        println(io, "non_r2_outcomes=$(length(non_r2))")
        println(
            io,
            "median_relative_ci_half_width_percent=$(percent(minimum(ci_median))):$(percent(maximum(ci_median)))",
        )
        println(
            io,
            "median_relative_nested_change_percent=$(percent(minimum(change_median))):$(percent(maximum(change_median)))",
        )
    end

    println("wrote convergence audit to $OUT_DIR")
    println(
        "non-R2 median relative CI half-width range: ",
        percent(minimum(ci_median)),
        "% to ",
        percent(maximum(ci_median)),
        "%",
    )
    println(
        "non-R2 median relative nested-change range: ",
        percent(minimum(change_median)),
        "% to ",
        percent(maximum(change_median)),
        "%",
    )
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
