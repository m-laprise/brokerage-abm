"""
Analyze the broker assessment vs access experiment with seed-paired Monte Carlo
intervals.

Set `BROKERAGE_ABM_ASSESSMENT_ACCESS_SWEEP_DIR` to the completed sweep root.
Outputs are written to `output/assessment_access/`.
After analyzing the base sweep, run `centrality_data.jl`, then `merge_supplement.jl`.

Usage: julia --project --threads=auto scripts/assessment_access/analyze.jl
"""

using CairoMakie
using DataFrames: DataFrame
using JLD2: jldsave
using Statistics: mean

isdefined(@__MODULE__, :ReportingSettings) || include(joinpath(@__DIR__, "..", "reporting_settings.jl"))

include(normpath(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl")))
include(normpath(joinpath(@__DIR__, "..", "monte_carlo.jl")))
include(normpath(joinpath(@__DIR__, "..", "reporting_provenance.jl")))
include(normpath(joinpath(@__DIR__, "..", "net_output_data.jl")))

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ANALYSIS_PROVENANCE = reporting_git_provenance(
    REPO_ROOT;
    sources=("scripts/analysis_settings.jl", "scripts/reporting_settings.jl",
        @__FILE__,
        "scripts/sweep/sweep_results.jl",
        "scripts/monte_carlo.jl",
        "scripts/net_output_data.jl",
        "scripts/net_output.jl",
    ),
)
const SWEEP_ROOT = get(ENV, "BROKERAGE_ABM_ASSESSMENT_ACCESS_SWEEP_DIR") do
    error("BROKERAGE_ABM_ASSESSMENT_ACCESS_SWEEP_DIR is required")
end
const OUT_DIR = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_ASSESSMENT_ACCESS_OUTPUT_DIR",
        joinpath(@__DIR__, "..", "..", "output", "assessment_access"),
    ),
)
const FIGURE_DIR = joinpath(OUT_DIR, "figures")
const LATE_WIDTH = AnalysisSettings.LATE_WIDTH
const NETWORK_INTERVAL = ReportingSettings.network_interval(SWEEP_ROOT)
const LEVEL = AnalysisSettings.INTERVAL_LEVEL
const MODES = (:full, :assessment_only, :access_only)
const MANIFEST = ReportingSettings.manifest(SWEEP_ROOT)
const BASELINE_CONFIG = ReportingSettings.baseline_condition(MANIFEST; overrides=Dict(:broker_service => :full))[:resolved_params]
const CONTRAST_CELLS = sort(unique((c[:resolved_params][:rho], c[:resolved_params][:eta]) for c in MANIFEST["conditions"]))
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
    :gross_match_output,
    :total_search_cost,
    :total_broker_fees,
    :net_output_per_principal,
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
        return [
            total[i] > 0 ? df.access_count[i] / total[i] : NaN for i in eachindex(total)
        ]
    end
    return df[!, metric]
end

function late_value(df::DataFrame, metric::Symbol)
    window = (maximum(df.period) - LATE_WIDTH + 1, maximum(df.period))
    indices = AnalysisSettings.reporting_indices(
        df.period, metric, window; network_interval=NETWORK_INTERVAL
    )
    return finite_mean(metric_series(df, metric)[indices])
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
        haskey(index, key) && error("duplicate assessment vs access condition: $key")
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
            metric=:net_output_per_principal,
            title="A. Net output to principals",
            ylabel="Net output per principal",
        ),
        (metric=:outsourcing_rate, title="B. Outsourcing", ylabel="Share of demand"),
    )
    for (column, panel) in enumerate(panels)
        axis = Axis(fig[1, column]; title=panel.title, xlabel="Period", ylabel=panel.ylabel)
        for mode in MODES
            series = ensemble_series(index[(mode, BASELINE_CONFIG[:rho], BASELINE_CONFIG[:eta])], panel.metric)
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
        (label="Assessment contribution", reference=:access_only, comparison=:full),
        (label="Access contribution", reference=:assessment_only, comparison=:full),
    )
    rho_values = sort(unique(first.(CONTRAST_CELLS)))
    eta_values = [eta for eta in sort(unique(last.(CONTRAST_CELLS))) if all((rho, eta) in CONTRAST_CELLS for rho in rho_values)]
    colors = (:steelblue, :goldenrod, :firebrick)
    fig = Figure(; size=(1050, 430), fontsize=16)
    for (column, contribution) in enumerate(contributions)
        axis = Axis(
            fig[1, column];
            title="$(Char(Int('A') + column - 1)). $(contribution.label)",
            xlabel="Individual-quality share (ρ)",
            ylabel,
            xticks=(rho_values, string.(rho_values)),
        )
        hlines!(axis, [0.0]; color=:gray65, linewidth=1)
        for (eta_index, eta) in enumerate(eta_values)
            intervals = [
                paired_interval(
                    index[(contribution.reference, rho, eta)],
                    index[(contribution.comparison, rho, eta)],
                    metric,
                ) for rho in rho_values
            ]
            estimates = [interval.mean for interval in intervals]
            lower = estimates .- [interval.lower for interval in intervals]
            upper = [interval.upper for interval in intervals] .- estimates
            color = colors[mod1(eta_index, length(colors))]
            errorbars!(axis, rho_values, estimates, lower, upper; color)
            scatterlines!(
                axis,
                rho_values,
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
    output_accounting = NetOutputData.reconstruct_dataset!(dataset, SWEEP_ROOT, REPO_ROOT)
    dataset.meta[:scope] in (:assessment_access, :broker_services) ||
        error("unexpected sweep scope")
    validate_analysis_commit(
        ANALYSIS_PROVENANCE,
        dataset.meta[:git_commit];
        artifact="broker assessment vs access sweep",
    )
    index = result_index(dataset)
    complete_keys = Set((mode, rho, eta) for mode in MODES for (rho, eta) in CONTRAST_CELLS)
    Set(keys(index)) == complete_keys || error("assessment vs access design mismatch")
    contrast_cells = sort!(unique((key[2], key[3]) for key in keys(index)))

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
                    seed_rows, Any[mode, rho, eta, seed, metric, seed_metric_values[seed]]
                )
            end
            interval = monte_carlo_interval(
                collect(values(seed_metric_values)); level=LEVEL
            )
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
    for (rho, eta) in contrast_cells, metric in METRICS
        for (label, reference, comparison) in (
            ("assessment", :access_only, :full),
            ("access", :assessment_only, :full),
            ("assessment_vs_access", :access_only, :assessment_only),
        )
            interval = paired_interval(
                index[(reference, rho, eta)], index[(comparison, rho, eta)], metric
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
        :net_output_per_principal,
        "net_output_contributions.png",
        "Change in net output per principal",
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
        reporting_settings=merge(ReportingSettings.sweep_metadata(REPO_ROOT, MANIFEST; baseline_overrides=Dict(:broker_service => :full)), AnalysisSettings.metadata(Int(dataset.meta[:T]))),
        late_width=LATE_WIDTH,
        interval_level=LEVEL,
        net_output_definition="net_output_per_principal",
        output_accounting,
    )
    open(joinpath(OUT_DIR, "provenance.txt"), "w") do io
        println(io, "analysis_git_commit=$(ANALYSIS_PROVENANCE.commit)")
        println(io, "analysis_source_clean=$(ANALYSIS_PROVENANCE.source_clean)")
        println(io, "sweep_git_commit=$(dataset.meta[:git_commit])")
        println(io, "manifest_hash=$(dataset.manifest_hash)")
        println(io, "schema_version=$(dataset.schema_version)")
        println(io, "sweep_root=$(abspath(SWEEP_ROOT))")
    end
    println("Broker assessment vs access analysis written to $OUT_DIR")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
