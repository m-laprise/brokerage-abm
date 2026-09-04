"""
    scripts/ridge/analyze_ablations.jl

Compare the three Ridge broker ablations with the base Ridge reference over
the baseline and rho-by-delta design. Each effective model realization receives
one observation. Baseline seed contrasts use all 50 common seeds, and all other
condition contrasts use their 20 common seeds.

Required environment:

  BROKERAGE_ABM_RIDGE_PAIR_SWEEP_DIR
  BROKERAGE_ABM_RIDGE_SIZE_MATCHED_SWEEP_DIR
  BROKERAGE_ABM_RIDGE_SINGLE_PRINCIPAL_SWEEP_DIR
  BROKERAGE_ABM_RIDGE_ADDITIVE_SWEEP_DIR

Detailed outputs are written to `output/ridge/ablations/results/`. Seed-level
inputs for the compact main-text figure are written to
`output/ridge/ablations/figdata.jld2`.

Usage: julia --project --threads=auto scripts/ridge/analyze_ablations.jl
"""

using CairoMakie
using DataFrames: DataFrame
using Dates: now
using JLD2: jldsave
using Printf: @sprintf
using Statistics: cor, mean, median, quantile, std

include(normpath(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl")))
include(normpath(joinpath(@__DIR__, "..", "monte_carlo.jl")))
include(normpath(joinpath(@__DIR__, "..", "reporting_provenance.jl")))

const PAIR_ROOT = get(ENV, "BROKERAGE_ABM_RIDGE_PAIR_SWEEP_DIR") do
    error("BROKERAGE_ABM_RIDGE_PAIR_SWEEP_DIR is required")
end
const VARIANTS = (
    (
        key=:size_matched,
        label="Size-matched",
        root=get(ENV, "BROKERAGE_ABM_RIDGE_SIZE_MATCHED_SWEEP_DIR") do
            error("BROKERAGE_ABM_RIDGE_SIZE_MATCHED_SWEEP_DIR is required")
        end,
    ),
    (
        key=:single_principal,
        label="Single-principal",
        root=get(ENV, "BROKERAGE_ABM_RIDGE_SINGLE_PRINCIPAL_SWEEP_DIR") do
            error("BROKERAGE_ABM_RIDGE_SINGLE_PRINCIPAL_SWEEP_DIR is required")
        end,
    ),
    (
        key=:additive,
        label="Additive",
        root=get(ENV, "BROKERAGE_ABM_RIDGE_ADDITIVE_SWEEP_DIR") do
            error("BROKERAGE_ABM_RIDGE_ADDITIVE_SWEEP_DIR is required")
        end,
    ),
)
const OUT_DIR = normpath(
    joinpath(@__DIR__, "..", "..", "output", "ridge", "ablations", "results")
)
const PAPER_VALUES = joinpath(OUT_DIR, "paper_values.tex")
const FIGDATA = normpath(
    joinpath(@__DIR__, "..", "..", "output", "ridge", "ablations", "figdata.jld2")
)
const BASELINE_REL = "oat/rho=0.5"
const LATE_WIDTH = 20
const METRICS = (
    :agent_holdout_rank,
    :broker_holdout_rank,
    :rank_gap,
    :agent_holdout_r2,
    :broker_holdout_r2,
    :r2_gap,
    :output_gap,
    :outsourcing_rate,
    :access_fraction,
    :betweenness,
    :mean_degree,
)
const DESIGN_KEYS = (
    "N",
    "T",
    "T_burn",
    "rho",
    "eta",
    "delta",
    "k",
    "roster_frac",
    "n_strangers",
    "s",
    "reservation_frac",
    "learning_model",
    "ridge_lambda_agent",
    "ridge_lambda_broker",
)

nanmean(values) = begin
    kept = filter(!isnan, Float64.(collect(values)))
    isempty(kept) ? NaN : mean(kept)
end

function metric_series(df::DataFrame, metric::Symbol)
    if metric == :rank_gap
        return df.broker_holdout_rank .- df.agent_holdout_rank
    elseif metric == :r2_gap
        return df.broker_holdout_r2 .- df.agent_holdout_r2
    elseif metric == :output_gap
        return df.q_broker_mean .- df.q_self_mean
    elseif metric == :access_fraction
        total = df.access_count .+ df.assessment_count
        return [
            total[i] > 0 ? df.access_count[i] / total[i] : NaN for i in eachindex(total)
        ]
    end
    return df[!, metric]
end

function seed_late(df::DataFrame, metric::Symbol)::Float64
    first_period = maximum(df.period) - LATE_WIDTH + 1
    mask = df.period .>= first_period
    return nanmean(metric_series(df, metric)[mask])
end

function seed_values(result::SweepResult, metric::Symbol; seeds=result.seeds)
    index = Dict(seed => idx for (idx, seed) in enumerate(result.seeds))
    all(haskey(index, seed) for seed in seeds) ||
        error("requested seeds absent from $(result.rel)")
    return [seed_late(result.mdfs[index[seed]], metric) for seed in seeds]
end

function condition_mean(result::SweepResult, metric::Symbol; seeds=result.seeds)
    nanmean(seed_values(result, metric; seeds=seeds))
end

function ensemble_series(result::SweepResult, metric::Symbol; seeds=result.seeds)
    index = Dict(seed => idx for (idx, seed) in enumerate(result.seeds))
    periods = result.mdfs[index[first(seeds)]].period
    seed_series = reduce(
        hcat, (Float64.(metric_series(result.mdfs[index[seed]], metric)) for seed in seeds)
    )
    summaries = [
        monte_carlo_interval(view(seed_series, period_index, :)) for
        period_index in axes(seed_series, 1)
    ]
    return (
        periods,
        [summary.mean for summary in summaries],
        [summary.lower for summary in summaries],
        [summary.upper for summary in summaries],
    )
end

function monte_carlo_se(values)::Float64
    return monte_carlo_interval(values).se
end

f3(value) = @sprintf("%.3f", value)
signed3(value) = @sprintf("%+.3f", value)
fint(value) = string(Int(value))
rho_tag(value) = replace(@sprintf("%.3g", value), "." => "p")

function write_tsv(path::AbstractString, header, rows)
    open(path, "w") do io
        println(io, join(header, '\t'))
        for row in rows
            println(io, join(row, '\t'))
        end
    end
    return nothing
end

function design_value(result::SweepResult, parameter, baseline_cfg)
    return if parameter == "delta" && result.cfg["rho"] == 1.0
        baseline_cfg["delta"]
    else
        result.cfg[parameter]
    end
end

function design_key(result::SweepResult, baseline_cfg)
    return Tuple(design_value(result, parameter, baseline_cfg) for parameter in DESIGN_KEYS)
end

function results_by_design(dataset::SweepDataset)
    baseline_cfg = dataset.result_by_rel[BASELINE_REL].cfg
    index = Dict{Any,SweepResult}()
    for result in dataset.results
        key = design_key(result, baseline_cfg)
        haskey(index, key) && error("duplicate effective realization: $(result.rel)")
        index[key] = result
    end
    return index
end

function validate_design(pair::SweepDataset, datasets)
    length(pair.results) == 98 || error("expected 98 base Ridge effective realizations")
    pair_by_design = results_by_design(pair)
    datasets_by_design = Dict(
        variant.key => results_by_design(datasets[variant.key]) for variant in VARIANTS
    )
    reference_keys = Set(keys(datasets_by_design[first(VARIANTS).key]))
    length(reference_keys) == 26 || error("expected 26 ablation effective realizations")
    pair_baseline_cfg = pair.result_by_rel[BASELINE_REL].cfg
    for variant in VARIANTS
        dataset = datasets[variant.key]
        by_design = datasets_by_design[variant.key]
        variant_baseline_cfg = dataset.result_by_rel[BASELINE_REL].cfg
        length(dataset.results) == 26 ||
            error("$(variant.label) does not have 26 realizations")
        length(dataset.grid_cells) == 31 ||
            error("$(variant.label) does not have 31 grid coordinates")
        Set(keys(by_design)) == reference_keys ||
            error("$(variant.label) effective design differs")
        for realization_key in reference_keys
            haskey(pair_by_design, realization_key) ||
                error("base Ridge reference lacks an ablation realization")
            pair_result = pair_by_design[realization_key]
            result = by_design[realization_key]
            for parameter in DESIGN_KEYS
                design_value(pair_result, parameter, pair_baseline_cfg) ==
                design_value(result, parameter, variant_baseline_cfg) ||
                    error("$(variant.label) design mismatch for $(result.rel): $parameter")
            end
            expected_seeds = result.rel == BASELINE_REL ? collect(1:50) : collect(1:20)
            result.seeds == expected_seeds ||
                error("unexpected $(variant.label) seed set for $(result.rel)")
            all(seed in pair_result.seeds for seed in result.seeds) ||
                error("base Ridge reference lacks common seeds for $(result.rel)")
        end
    end
    reference_dataset = datasets[first(VARIANTS).key]
    baseline_result = reference_dataset.result_by_rel[BASELINE_REL]
    baseline_key = design_key(baseline_result, baseline_result.cfg)
    baseline_key in reference_keys || error("ablation design lacks the baseline")
    ordered_keys = sort!(
        collect(reference_keys); by=key -> datasets_by_design[first(VARIANTS).key][key].rel
    )
    return (; ordered_keys, baseline_key, pair_by_design, datasets_by_design)
end

function ablation_figure(pair_by_design, datasets_by_design, design_keys, baseline_key)
    colors = Dict(
        :pair => :black,
        :size_matched => :darkorange,
        :single_principal => :seagreen,
        :additive => :steelblue,
    )
    fig = Figure(; size=(1850, 540), fontsize=18)
    ax1 = Axis(
        fig[1, 1];
        title="A. Baseline ranking advantage",
        xlabel="Period",
        ylabel="Difference in holdout rank correlation\n(broker minus principal)",
    )
    pair_series = ensemble_series(pair_by_design[baseline_key], :rank_gap)
    function interval_line!(axis, series; color, label)
        period, estimate, lower, upper = series
        band!(axis, period, lower, upper; color=(color, 0.14))
        lines!(axis, period, estimate; color, linewidth=2.5, label)
    end
    interval_line!(ax1, pair_series; color=colors[:pair], label="Pair")
    for variant in VARIANTS
        series = ensemble_series(datasets_by_design[variant.key][baseline_key], :rank_gap)
        interval_line!(ax1, series; color=colors[variant.key], label=variant.label)
    end
    Legend(fig[0, 1:4], ax1; orientation=:horizontal, framevisible=false, tellwidth=false)

    for (column, variant) in enumerate(VARIANTS)
        points = [
            let pair_result = pair_by_design[key]
                variant_result = datasets_by_design[variant.key][key]
                seeds = variant_result.seeds
                pair_seed_values = seed_values(pair_result, :rank_gap; seeds)
                variant_seed_values = seed_values(variant_result, :rank_gap; seeds)
                interval = paired_monte_carlo_interval(
                    pair_seed_values, variant_seed_values
                )
                (
                    reference=nanmean(pair_seed_values),
                    difference=interval.mean,
                    lower=interval.lower,
                    upper=interval.upper,
                )
            end for key in design_keys
        ]
        x = [point.reference for point in points]
        y = [point.difference for point in points]
        lower = [point.lower for point in points]
        upper = [point.upper for point in points]
        ylo, yhi = extrema(filter(isfinite, vcat(lower, upper)))
        ypad = max(0.02, 0.05 * (yhi - ylo))
        panel = Char(Int('B') + column - 1)
        ax = Axis(
            fig[1, column + 1];
            title="$panel. $(variant.label)",
            xlabel="Pair Ridge difference in holdout rank correlation",
            ylabel="Ablation - Pair Ridge",
            limits=(nothing, (ylo - ypad, yhi + ypad)),
        )
        hlines!(ax, [0.0]; color=:gray55, linestyle=:dash, linewidth=1.5)
        rangebars!(
            ax,
            x,
            lower,
            upper;
            color=(colors[variant.key], 0.55),
            linewidth=1.0,
            whiskerwidth=5,
        )
        scatter!(ax, x, y; color=(colors[variant.key], 0.78), markersize=11)
        text!(
            ax,
            minimum(x),
            yhi;
            text="26 effective realizations",
            align=(:left, :top),
            fontsize=14,
        )
    end
    save(joinpath(OUT_DIR, "ridge_ablations.png"), fig; px_per_unit=2)
    return nothing
end

function ablation_grid_figure(pair::SweepDataset, datasets)
    model_datasets = (
        (key=:pair, label="Pair", dataset=pair),
        (key=:size_matched, label="Size-matched", dataset=datasets[:size_matched]),
        (
            key=:single_principal,
            label="Single-principal",
            dataset=datasets[:single_principal],
        ),
        (key=:additive, label="Additive", dataset=datasets[:additive]),
    )
    delta_colors = Dict(
        0.0 => :steelblue,
        0.25 => :cadetblue,
        0.5 => :goldenrod,
        0.75 => :darkorange,
        1.0 => :firebrick,
    )
    function grid_points(dataset)
        cells = filter(cell -> get(cell, :pair, nothing) == "rho_delta", dataset.grid_cells)
        return [
            (
                rho=Float64(cell[:xval]),
                delta=Float64(cell[:yval]),
                result_rel=String(cell[:result_reldir]),
                interval=monte_carlo_interval(
                    seed_values(dataset.result_by_rel[cell[:result_reldir]], :rank_gap)
                ),
            ) for cell in cells
        ]
    end
    points = Dict(model.key => grid_points(model.dataset) for model in model_datasets)
    bounds = [
        bound for model_points in values(points) for point in model_points for
        bound in (point.interval.lower, point.interval.upper)
    ]
    lo, hi = extrema(filter(isfinite, bounds))
    pad = 0.06 * (hi - lo + eps())
    ylimits = (lo - pad, hi + pad)
    deltas = sort(unique(point.delta for point in points[:pair]))

    fig = Figure(; size=(1180, 850), fontsize=18)
    for (index, model) in enumerate(model_datasets)
        row, column = fldmod1(index, 2)
        axis = Axis(
            fig[row, column];
            title=model.label,
            xlabel=row == 2 ? "ρ (complementarity vs quality)" : "",
            ylabel=if column == 1
                "Difference in holdout rank correlation\n(broker minus principal)"
            else
                ""
            end,
            xticks=[0, 0.3, 0.5, 0.7, 0.85, 1],
            limits=(nothing, ylimits),
        )
        hlines!(axis, [0.0]; color=:gray55, linestyle=:dash, linewidth=1.5)
        for delta in deltas
            line_points = sort(
                filter(point -> point.delta == delta && point.rho < 1.0, points[model.key]);
                by=x -> x.rho,
            )
            rangebars!(
                axis,
                [point.rho for point in line_points],
                [point.interval.lower for point in line_points],
                [point.interval.upper for point in line_points];
                color=(delta_colors[delta], 0.72),
                linewidth=1.1,
                whiskerwidth=7,
            )
            scatterlines!(
                axis,
                [point.rho for point in line_points],
                [point.interval.mean for point in line_points];
                color=delta_colors[delta],
                linewidth=2.2,
                markersize=9,
                strokewidth=0.4,
                strokecolor=:gray30,
                label="δ = $delta",
            )
        end
        boundary_points = filter(point -> point.rho == 1.0, points[model.key])
        length(unique(point.result_rel for point in boundary_points)) == 1 ||
            error("$(model.label) rho = 1 coordinates do not share one realization")
        boundary = first(boundary_points)
        rangebars!(
            axis,
            [1.0],
            [boundary.interval.lower],
            [boundary.interval.upper];
            color=(:gray25, 0.8),
            linewidth=1.1,
            whiskerwidth=7,
        )
        scatter!(
            axis,
            [1.0],
            [boundary.interval.mean];
            color=:gray25,
            marker=:diamond,
            markersize=10,
            strokewidth=0.4,
            strokecolor=:gray20,
            label="ρ = 1 boundary",
        )
        index == 1 && axislegend(axis, "Difficulty"; position=:lt, framevisible=true)
    end
    colgap!(fig.layout, 16)
    rowgap!(fig.layout, 12)
    save(joinpath(OUT_DIR, "ridge_ablation_grid.png"), fig; px_per_unit=2)
    return nothing
end

function main()
    provenance = reporting_git_provenance(normpath(joinpath(@__DIR__, "..", "..")))
    mkpath(OUT_DIR)
    pair = load_sweep_dataset(PAIR_ROOT)
    datasets = Dict(variant.key => load_sweep_dataset(variant.root) for variant in VARIANTS)
    design = validate_design(pair, datasets)
    design_keys = design.ordered_keys
    baseline_key = design.baseline_key
    pair_by_design = design.pair_by_design
    datasets_by_design = design.datasets_by_design

    all_keys = (:pair, (variant.key for variant in VARIANTS)...)
    all_labels = Dict(
        :pair => "Pair", (variant.key => variant.label for variant in VARIANTS)...
    )
    result_for(key, design_key) =
        key == :pair ? pair_by_design[design_key] : datasets_by_design[key][design_key]

    condition_header = Any[
        "ablation_result_reldir", "pair_result_reldir", "rho", "delta", "n_seeds"
    ]
    append!(
        condition_header, ["$(key)_$(metric)" for key in all_keys for metric in METRICS]
    )
    append!(
        condition_header,
        [
            "$(variant.key)_minus_pair_$(metric)" for variant in VARIANTS for
            metric in METRICS
        ],
    )
    for statistic in ("se", "ci_lower", "ci_upper", "n")
        append!(
            condition_header,
            [
                "$(variant.key)_minus_pair_$(metric)_$(statistic)" for variant in VARIANTS
                for metric in METRICS
            ],
        )
    end
    condition_rows = Vector{Any}[]
    for design_key in design_keys
        reference = pair_by_design[design_key]
        ablation_reference = datasets_by_design[:size_matched][design_key]
        seeds = ablation_reference.seeds
        row = Any[
            ablation_reference.rel,
            reference.rel,
            reference.cfg["rho"],
            reference.cfg["delta"],
            length(seeds),
        ]
        estimates = Dict{Tuple{Symbol,Symbol},Float64}()
        for key in all_keys, metric in METRICS
            estimates[(key, metric)] = condition_mean(
                result_for(key, design_key), metric; seeds=seeds
            )
            push!(row, estimates[(key, metric)])
        end
        for variant in VARIANTS, metric in METRICS
            push!(row, estimates[(variant.key, metric)] - estimates[(:pair, metric)])
        end
        for field in (:se, :lower, :upper, :n), variant in VARIANTS, metric in METRICS
            pair_seed_values = seed_values(reference, metric; seeds)
            variant_seed_values = seed_values(
                datasets_by_design[variant.key][design_key], metric; seeds
            )
            interval = paired_monte_carlo_interval(pair_seed_values, variant_seed_values)
            push!(row, getproperty(interval, field))
        end
        push!(condition_rows, row)
    end
    write_tsv(
        joinpath(OUT_DIR, "condition_comparison.tsv"), condition_header, condition_rows
    )

    baseline_seeds = datasets_by_design[:size_matched][baseline_key].seeds
    baseline_header = Any["seed"]
    append!(baseline_header, ["$(key)_$(metric)" for key in all_keys for metric in METRICS])
    append!(
        baseline_header,
        [
            "$(variant.key)_minus_pair_$(metric)" for variant in VARIANTS for
            metric in METRICS
        ],
    )
    baseline_rows = Vector{Any}[]
    for seed in baseline_seeds
        row = Any[seed]
        estimates = Dict{Tuple{Symbol,Symbol},Float64}()
        for key in all_keys, metric in METRICS
            estimates[(key, metric)] = only(
                seed_values(result_for(key, baseline_key), metric; seeds=[seed])
            )
            push!(row, estimates[(key, metric)])
        end
        for variant in VARIANTS, metric in METRICS
            push!(row, estimates[(variant.key, metric)] - estimates[(:pair, metric)])
        end
        push!(baseline_rows, row)
    end
    write_tsv(
        joinpath(OUT_DIR, "baseline_common_seeds.tsv"), baseline_header, baseline_rows
    )

    figure_conditions = Dict{String,Any}[]
    for design_key in design_keys
        reference = datasets_by_design[:size_matched][design_key]
        seeds = reference.seeds
        rank_gaps = Dict(
            String(key) => seed_values(result_for(key, design_key), :rank_gap; seeds) for
            key in all_keys
        )
        push!(
            figure_conditions,
            Dict{String,Any}(
                "result_reldir" => reference.rel,
                "rho" => reference.cfg["rho"],
                "delta" => reference.cfg["delta"],
                "seeds" => seeds,
                "rank_gaps" => rank_gaps,
            ),
        )
    end
    figure_data = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "analysis_git_commit" => provenance.commit,
            "analysis_source_clean" => provenance.source_clean,
            "n_conditions" => length(design_keys),
            "baseline_reldir" => datasets_by_design[:size_matched][baseline_key].rel,
            "late_width" => LATE_WIDTH,
        ),
        "conditions" => figure_conditions,
    )
    mkpath(dirname(FIGDATA))
    jldsave(FIGDATA; figdata=figure_data)

    values = Pair{String,String}[]
    add(key, value; formatter=f3) = push!(values, key => formatter(value))
    add("nConditions", length(design_keys); formatter=fint)
    add("nBaselineSeeds", length(baseline_seeds); formatter=fint)
    add("nOtherSeeds", 20; formatter=fint)

    for key in all_keys, metric in METRICS
        baseline = condition_mean(
            result_for(key, baseline_key), metric; seeds=baseline_seeds
        )
        condition_values = [
            condition_mean(
                result_for(key, design_key),
                metric;
                seeds=datasets_by_design[:size_matched][design_key].seeds,
            ) for design_key in design_keys
        ]
        add("$(key)Baseline_$(metric)", baseline)
        add("$(key)Median_$(metric)", median(condition_values))
        add("$(key)PositiveN_$(metric)", count(>(0.0), condition_values); formatter=fint)
    end

    for variant in VARIANTS, metric in METRICS
        paired_baseline =
            seed_values(
                datasets_by_design[variant.key][baseline_key], metric; seeds=baseline_seeds
            ) .- seed_values(pair_by_design[baseline_key], metric; seeds=baseline_seeds)
        paired_baseline_interval = monte_carlo_interval(paired_baseline)
        pair_values = [
            condition_mean(
                pair_by_design[design_key],
                metric;
                seeds=datasets_by_design[variant.key][design_key].seeds,
            ) for design_key in design_keys
        ]
        variant_values = [
            condition_mean(datasets_by_design[variant.key][design_key], metric) for
            design_key in design_keys
        ]
        differences = variant_values .- pair_values
        add(
            "$(variant.key)BaselineDelta_$(metric)",
            nanmean(paired_baseline);
            formatter=signed3,
        )
        add("$(variant.key)BaselineDeltaSE_$(metric)", monte_carlo_se(paired_baseline))
        add(
            "$(variant.key)BaselineDeltaLower_$(metric)",
            paired_baseline_interval.lower;
            formatter=signed3,
        )
        add(
            "$(variant.key)BaselineDeltaUpper_$(metric)",
            paired_baseline_interval.upper;
            formatter=signed3,
        )
        add("$(variant.key)DeltaMean_$(metric)", mean(differences); formatter=signed3)
        add("$(variant.key)DeltaMedian_$(metric)", median(differences); formatter=signed3)
        add(
            "$(variant.key)DeltaQ10_$(metric)",
            quantile(differences, 0.10);
            formatter=signed3,
        )
        add(
            "$(variant.key)DeltaQ90_$(metric)",
            quantile(differences, 0.90);
            formatter=signed3,
        )
        add("$(variant.key)GreaterN_$(metric)", count(>(0.0), differences); formatter=fint)
        add(
            "$(variant.key)ConditionCorrelation_$(metric)", cor(pair_values, variant_values)
        )
    end

    single_baseline = seed_values(
        datasets_by_design[:single_principal][baseline_key], :rank_gap; seeds=baseline_seeds
    )
    additive_baseline = seed_values(
        datasets_by_design[:additive][baseline_key], :rank_gap; seeds=baseline_seeds
    )
    baseline_single_minus_additive = single_baseline .- additive_baseline
    single_minus_additive_interval = monte_carlo_interval(baseline_single_minus_additive)
    add(
        "singleMinusAdditiveBaseline_rank_gap",
        nanmean(baseline_single_minus_additive);
        formatter=signed3,
    )
    add(
        "singleMinusAdditiveBaselineSE_rank_gap",
        monte_carlo_se(baseline_single_minus_additive),
    )
    add(
        "singleMinusAdditiveBaselineLower_rank_gap",
        single_minus_additive_interval.lower;
        formatter=signed3,
    )
    add(
        "singleMinusAdditiveBaselineUpper_rank_gap",
        single_minus_additive_interval.upper;
        formatter=signed3,
    )
    single_condition = [
        condition_mean(datasets_by_design[:single_principal][key], :rank_gap) for
        key in design_keys
    ]
    additive_condition = [
        condition_mean(datasets_by_design[:additive][key], :rank_gap) for key in design_keys
    ]
    single_minus_additive = single_condition .- additive_condition
    add(
        "singleMinusAdditiveMedian_rank_gap",
        median(single_minus_additive);
        formatter=signed3,
    )
    add(
        "singleMinusAdditiveQ10_rank_gap",
        quantile(single_minus_additive, 0.10);
        formatter=signed3,
    )
    add(
        "singleMinusAdditiveQ90_rank_gap",
        quantile(single_minus_additive, 0.90);
        formatter=signed3,
    )
    add(
        "singleMinusAdditiveGreaterN_rank_gap",
        count(>(0.0), single_minus_additive);
        formatter=fint,
    )

    rho_values = sort!(
        unique(Float64(pair_by_design[key].cfg["rho"]) for key in design_keys)
    )
    for rho in rho_values
        rho_keys = filter(
            key -> Float64(pair_by_design[key].cfg["rho"]) == rho, design_keys
        )
        for key in all_keys
            gaps = [
                condition_mean(result_for(key, rho_key), :rank_gap) for rho_key in rho_keys
            ]
            add("$(key)Rho$(rho_tag(rho))_rank_gap", median(gaps))
        end
    end

    open(joinpath(OUT_DIR, "values.tex"), "w") do io
        println(io, "% Generated by scripts/ridge/analyze_ablations.jl on $(now()).")
        println(io, "% Analysis commit: $(provenance.commit)")
        println(io, "% Analysis source clean: $(provenance.source_clean)")
        println(io, "% Effective realizations receive equal weight.")
        println(
            io,
            raw"\newcommand{\ravDefine}[2]{\expandafter\newcommand\csname rav@#1\endcsname{#2}}",
        )
        println(io, raw"\newcommand{\rav}[1]{\csname rav@#1\endcsname}")
        for (key, value) in values
            println(io, "\\ravDefine{$key}{$value}")
        end
    end

    value_by_key = Dict(values)
    paper_values = (
        "ablationConditionN" => value_by_key["nConditions"],
        "ablationPairMedianGap" => value_by_key["pairMedian_rank_gap"],
        "ablationSizeMedianGap" => value_by_key["size_matchedMedian_rank_gap"],
        "ablationSizePositiveN" => value_by_key["size_matchedPositiveN_rank_gap"],
        "ablationSingleMedianGap" => value_by_key["single_principalMedian_rank_gap"],
        "ablationSinglePositiveN" => value_by_key["single_principalPositiveN_rank_gap"],
        "ablationAdditiveMedianGap" => value_by_key["additiveMedian_rank_gap"],
        "ablationAdditivePositiveN" => value_by_key["additivePositiveN_rank_gap"],
    )
    open(PAPER_VALUES, "w") do io
        println(io, "% Generated by scripts/ridge/analyze_ablations.jl on $(now()).")
        println(io, "% Analysis commit: $(provenance.commit)")
        println(io, "% Analysis source clean: $(provenance.source_clean)")
        println(io, "% Effective realizations receive equal weight.")
        for (key, value) in paper_values
            println(io, "\\pvDefine{$key}{$value}")
        end
    end

    open(joinpath(OUT_DIR, "provenance.txt"), "w") do io
        println(io, "generated=$(now())")
        println(io, "analysis_commit=$(provenance.commit)")
        println(io, "analysis_source_clean=$(provenance.source_clean)")
        println(io, "pair_sweep=$(basename(normpath(PAIR_ROOT)))")
        println(io, "pair_manifest=$(pair.manifest_hash)")
        for variant in VARIANTS
            dataset = datasets[variant.key]
            println(io, "$(variant.key)_sweep=$(basename(normpath(variant.root)))")
            println(io, "$(variant.key)_manifest=$(dataset.manifest_hash)")
        end
        println(io, "late_periods=$(Int(pair.meta[:T]) - LATE_WIDTH + 1):$(pair.meta[:T])")
        println(io, "weighting=one observation per effective realization")
        println(
            io,
            "realization_matching=resolved parameters in DESIGN_KEYS, with delta canonicalized to baseline when rho=1",
        )
        println(io, "baseline_seeds=$(join(baseline_seeds, ','))")
        println(io, "other_seeds=$(join(1:20, ','))")
    end

    println("Baseline late-period differences in holdout rank correlation:")
    for key in all_keys
        value = condition_mean(
            result_for(key, baseline_key), :rank_gap; seeds=baseline_seeds
        )
        println("  ", rpad(all_labels[key], 18), f3(value))
    end
    println(
        "Median late-period differences in holdout rank correlation across $(length(design_keys)) effective realizations:",
    )
    for key in all_keys
        vals = [
            condition_mean(
                result_for(key, design_key),
                :rank_gap;
                seeds=datasets_by_design[:size_matched][design_key].seeds,
            ) for design_key in design_keys
        ]
        println("  ", rpad(all_labels[key], 18), f3(median(vals)))
    end

    ablation_figure(pair_by_design, datasets_by_design, design_keys, baseline_key)
    ablation_grid_figure(pair, datasets)
    println("wrote Ridge ablation outputs to $OUT_DIR")
    println("wrote seed-level main-figure data to $FIGDATA")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
