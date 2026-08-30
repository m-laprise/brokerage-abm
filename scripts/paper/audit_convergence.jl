"""
    scripts/paper/audit_convergence.jl

Reproducible Monte Carlo convergence audit for the canonical reporting sweep.
Each effective realization is evaluated once. For every reported figure
outcome, the script computes a Student-t 95% interval from seed-level
late-window means and compares a nested seed estimate with the full estimate:
the first 10 versus all 20 seeds generally, and the first 20 versus all 50 at
the baseline.

To compare outcomes with different units, interval half-widths and nested-seed
changes are divided by that outcome's P90-minus-P10 span across the 98 effective
realizations. R² outcomes remain in the audit tables but are excluded from the
cross-outcome ranges written for the methods section.

Required environment:

  BROKERAGE_ABM_SWEEP_DIR

Optional environment:

  BROKERAGE_ABM_CONVERGENCE_DIR

The default output directory is `output/main/convergence/`.
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
        estimates = filter(isfinite, [row.estimate for row in metric_rows])
        length(estimates) > 1 || error("too few finite estimates for $metric")
        scale = quantile(estimates, 0.90) - quantile(estimates, 0.10)
        scale > 0.0 || error("zero P90-P10 scale for $metric")
        append!(
            rows,
            [
                merge(
                    row,
                    (;
                        outcome_span=scale,
                        relative_ci_half_width=(row.ci_half_width / scale),
                        relative_nested_change=(row.nested_change / scale),
                    ),
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
        "outcome_p90_minus_p10",
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
            row.outcome_span,
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
    ci_p90 = [summary.p90_relative_ci_half_width for summary in non_r2]
    change_p90 = [summary.p90_relative_nested_change for summary in non_r2]
    percent(value) = @sprintf("%.1f", 100.0 * value)
    values = (
        "mcOutcomeCount" => string(length(non_r2)),
        "mcCiP90MinPercent" => percent(minimum(ci_p90)),
        "mcCiP90MaxPercent" => percent(maximum(ci_p90)),
        "mcNestedP90MinPercent" => percent(minimum(change_p90)),
        "mcNestedP90MaxPercent" => percent(maximum(change_p90)),
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
        println(io, "normalization=outcome P90 minus P10 across effective realizations")
        println(io, "non_r2_outcomes=$(length(non_r2))")
        println(
            io,
            "p90_relative_ci_half_width_percent=$(percent(minimum(ci_p90))):$(percent(maximum(ci_p90)))",
        )
        println(
            io,
            "p90_relative_nested_change_percent=$(percent(minimum(change_p90))):$(percent(maximum(change_p90)))",
        )
    end

    println("wrote convergence audit to $OUT_DIR")
    println(
        "non-R2 90th-percentile relative CI half-width range: ",
        percent(minimum(ci_p90)),
        "% to ",
        percent(maximum(ci_p90)),
        "%",
    )
    println(
        "non-R2 90th-percentile relative nested-change range: ",
        percent(minimum(change_p90)),
        "% to ",
        percent(maximum(change_p90)),
        "%",
    )
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
