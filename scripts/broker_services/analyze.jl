"""
Analyze the broker-service experiment with seed-paired Monte Carlo intervals.

Set `BROKERAGE_ABM_BROKER_SERVICE_SWEEP_DIR` to the completed sweep root.
Outputs are written to `output/broker_services/`.

Usage: julia --project --threads=auto scripts/broker_services/analyze.jl
"""

using CairoMakie
using DataFrames: DataFrame
using JLD2: jldsave
using Printf: @sprintf
using Statistics: mean

include(normpath(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl")))
include(normpath(joinpath(@__DIR__, "..", "monte_carlo.jl")))
include(normpath(joinpath(@__DIR__, "..", "reporting_provenance.jl")))

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ANALYSIS_PROVENANCE = reporting_git_provenance(REPO_ROOT)
const SWEEP_ROOT = get(ENV, "BROKERAGE_ABM_BROKER_SERVICE_SWEEP_DIR") do
    error("BROKERAGE_ABM_BROKER_SERVICE_SWEEP_DIR is required")
end
const OUT_DIR = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_BROKER_SERVICE_OUTPUT_DIR",
        joinpath(@__DIR__, "..", "..", "output", "broker_services"),
    ),
)
const FIGURE_DIR = joinpath(OUT_DIR, "figures")
const LATE_WIDTH = 20
const LEVEL = 0.95
const MODES = (:full, :assessment_only, :access_only)
const GRID_CELLS = [
    (rho, eta) for rho in (0.0, 0.5, 1.0) for eta in (0.0, 0.01, 0.03)
]
const CONTRAST_CELLS = vcat([(0.5, 0.02)], GRID_CELLS)
const MODE_LABELS = Dict(
    :full => "Full service",
    :assessment_only => "Assessment only",
    :access_only => "Access only",
)
const MODE_COLORS = Dict(
    :full => :black, :assessment_only => :steelblue, :access_only => :darkorange
)
const METRICS = (
    :agent_holdout_rank,
    :broker_holdout_rank,
    :rank_gap,
    :broker_selected_rank,
    :outsourcing_rate,
    :self_fill_rate,
    :broker_fill_rate,
    :self_net_output_per_requested_position,
    :broker_net_output_per_requested_position,
    :net_output_per_requested_position,
    :q_gap,
    :access_fraction,
    :betweenness,
    :mean_degree,
)

finite_mean(values) = begin
    kept = filter(isfinite, Float64.(collect(values)))
    isempty(kept) ? NaN : mean(kept)
end

function metric_series(df::DataFrame, metric::Symbol)
    if metric == :rank_gap
        return df.broker_holdout_rank .- df.agent_holdout_rank
    elseif metric == :q_gap
        return df.q_broker_mean .- df.q_self_mean
    elseif metric == :access_fraction
        total = df.access_count .+ df.assessment_count
        return [total[i] > 0 ? df.access_count[i] / total[i] : NaN for i in eachindex(total)]
    end
    return df[!, metric]
end

function late_value(df::DataFrame, metric::Symbol)
    first_period = maximum(df.period) - LATE_WIDTH + 1
    return finite_mean(metric_series(df, metric)[df.period .>= first_period])
end

function condition_key(result::SweepResult)
    return (
        Symbol(result.cfg["broker_service"]),
        Float64(result.cfg["rho"]),
        Float64(result.cfg["eta"]),
    )
end

function result_index(dataset::SweepDataset)
    index = Dict{Tuple{Symbol,Float64,Float64},SweepResult}()
    for result in dataset.results
        key = condition_key(result)
        haskey(index, key) && error("duplicate broker-service condition: $key")
        index[key] = result
    end
    return index
end

function seed_values(result::SweepResult, metric::Symbol)
    return Dict(
        seed => late_value(result.mdfs[idx], metric) for
        (idx, seed) in enumerate(result.seeds)
    )
end

function paired_interval(reference::SweepResult, comparison::SweepResult, metric::Symbol)
    reference_values = seed_values(reference, metric)
    comparison_values = seed_values(comparison, metric)
    seeds = sort!(collect(intersect(keys(reference_values), keys(comparison_values))))
    isempty(seeds) && error("no common seeds for paired contrast")
    return paired_monte_carlo_interval(
        [reference_values[seed] for seed in seeds],
        [comparison_values[seed] for seed in seeds];
        level=LEVEL,
    )
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

function ensemble_series(result::SweepResult, metric::Symbol)
    periods = result.mdfs[1].period
    values = reduce(hcat, (Float64.(metric_series(df, metric)) for df in result.mdfs))
    intervals = [
        monte_carlo_interval(view(values, period_index, :); level=LEVEL) for
        period_index in axes(values, 1)
    ]
    return (
        periods=periods,
        estimate=[interval.mean for interval in intervals],
        lower=[interval.lower for interval in intervals],
        upper=[interval.upper for interval in intervals],
    )
end

function baseline_figure(index)
    fig = Figure(; size=(1050, 430), fontsize=16)
    panels = (
        (
            metric=:broker_net_output_per_requested_position,
            title="A. Brokered net output per requested position",
            ylabel="Net output",
        ),
        (metric=:outsourcing_rate, title="B. Outsourcing", ylabel="Share of demand"),
    )
    for (column, panel) in enumerate(panels)
        axis = Axis(fig[1, column]; title=panel.title, xlabel="Period", ylabel=panel.ylabel)
        for mode in MODES
            series = ensemble_series(index[(mode, 0.5, 0.02)], panel.metric)
            color = MODE_COLORS[mode]
            band!(axis, series.periods, series.lower, series.upper; color=(color, 0.15))
            lines!(
                axis,
                series.periods,
                series.estimate;
                color,
                linewidth=2.3,
                label=MODE_LABELS[mode],
            )
        end
        column == 2 && axislegend(axis; position=:rb, framevisible=false)
    end
    save(joinpath(FIGURE_DIR, "baseline_dynamics.png"), fig; px_per_unit=2)
    return fig
end

function contrast_figure(index, metric::Symbol, filename::String, ylabel::String)
    contributions = (
        (
            label="Assessment contribution",
            reference=:access_only,
            comparison=:full,
        ),
        (
            label="Access contribution",
            reference=:assessment_only,
            comparison=:full,
        ),
    )
    eta_values = (0.0, 0.01, 0.03)
    colors = (:steelblue, :goldenrod, :firebrick)
    fig = Figure(; size=(1050, 430), fontsize=16)
    for (column, contribution) in enumerate(contributions)
        axis = Axis(
            fig[1, column];
            title="$(Char(Int('A') + column - 1)). $(contribution.label)",
            xlabel="General-quality share (ρ)",
            ylabel,
            xticks=([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        )
        hlines!(axis, [0.0]; color=:gray65, linewidth=1)
        for (eta_index, eta) in enumerate(eta_values)
            intervals = [
                paired_interval(
                    index[(contribution.reference, rho, eta)],
                    index[(contribution.comparison, rho, eta)],
                    metric,
                ) for rho in (0.0, 0.5, 1.0)
            ]
            estimates = [interval.mean for interval in intervals]
            lower = estimates .- [interval.lower for interval in intervals]
            upper = [interval.upper for interval in intervals] .- estimates
            color = colors[eta_index]
            errorbars!(axis, [0.0, 0.5, 1.0], estimates, lower, upper; color)
            scatterlines!(
                axis,
                [0.0, 0.5, 1.0],
                estimates;
                color,
                linewidth=2,
                markersize=8,
                label="Turnover = $eta",
            )
        end
        column == 2 && axislegend(axis; position=:rt, framevisible=false)
    end
    save(joinpath(FIGURE_DIR, filename), fig; px_per_unit=2)
    return fig
end

function main()
    dataset = load_sweep_dataset(SWEEP_ROOT)
    dataset.meta[:scope] == :broker_services || error("unexpected sweep scope")
    dataset.meta[:git_commit] == ANALYSIS_PROVENANCE.commit ||
        error("sweep and analysis commits differ")
    length(dataset.results) == 30 || error("expected 30 broker-service conditions")
    index = result_index(dataset)
    expected_keys = Set(
        (mode, rho, eta) for mode in MODES for (rho, eta) in CONTRAST_CELLS
    )
    Set(keys(index)) == expected_keys || error("broker-service design mismatch")

    mkpath(FIGURE_DIR)
    seed_rows = Vector{Vector{Any}}()
    summary_rows = Vector{Vector{Any}}()
    for key in sort!(collect(keys(index)); by=string)
        mode, rho, eta = key
        result = index[key]
        for metric in METRICS
            seed_metric_values = seed_values(result, metric)
            for seed in sort!(collect(keys(seed_metric_values)))
                push!(
                    seed_rows,
                    Any[mode, rho, eta, seed, metric, seed_metric_values[seed]],
                )
            end
            interval = monte_carlo_interval(collect(values(seed_metric_values)); level=LEVEL)
            push!(
                summary_rows,
                Any[
                    mode,
                    rho,
                    eta,
                    metric,
                    interval.mean,
                    interval.se,
                    interval.lower,
                    interval.upper,
                    interval.n,
                ],
            )
        end
    end
    write_tsv(
        joinpath(OUT_DIR, "seed_level.tsv"),
        ["broker_service", "rho", "eta", "seed", "metric", "late_value"],
        seed_rows,
    )
    write_tsv(
        joinpath(OUT_DIR, "condition_summary.tsv"),
        ["broker_service", "rho", "eta", "metric", "estimate", "se", "lower", "upper", "n"],
        summary_rows,
    )

    contrast_rows = Vector{Vector{Any}}()
    for (rho, eta) in CONTRAST_CELLS, metric in METRICS
        for (label, reference) in
            (("assessment", :access_only), ("access", :assessment_only))
            interval = paired_interval(
                index[(reference, rho, eta)], index[(:full, rho, eta)], metric
            )
            push!(
                contrast_rows,
                Any[
                    label,
                    rho,
                    eta,
                    metric,
                    interval.mean,
                    interval.se,
                    interval.lower,
                    interval.upper,
                    interval.n,
                ],
            )
        end
    end
    write_tsv(
        joinpath(OUT_DIR, "paired_contrasts.tsv"),
        ["contribution", "rho", "eta", "metric", "estimate", "se", "lower", "upper", "n"],
        contrast_rows,
    )

    baseline_figure(index)
    contrast_figure(
        index,
        :net_output_per_requested_position,
        "net_output_contributions.png",
        "Full service minus restricted service",
    )
    jldsave(
        joinpath(OUT_DIR, "figure_data.jld2");
        seed_rows,
        summary_rows,
        contrast_rows,
        manifest_hash=dataset.manifest_hash,
        schema_version=dataset.schema_version,
        analysis_git_commit=ANALYSIS_PROVENANCE.commit,
        analysis_source_clean=ANALYSIS_PROVENANCE.source_clean,
        source_root=abspath(SWEEP_ROOT),
        late_width=LATE_WIDTH,
        interval_level=LEVEL,
    )
    open(joinpath(OUT_DIR, "provenance.txt"), "w") do io
        println(io, "analysis_git_commit=$(ANALYSIS_PROVENANCE.commit)")
        println(io, "analysis_source_clean=$(ANALYSIS_PROVENANCE.source_clean)")
        println(io, "sweep_git_commit=$(dataset.meta[:git_commit])")
        println(io, "manifest_hash=$(dataset.manifest_hash)")
        println(io, "schema_version=$(dataset.schema_version)")
        println(io, "sweep_root=$(abspath(SWEEP_ROOT))")
    end
    println("Broker-service analysis written to $OUT_DIR")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
