"""
    scripts/ridge/paired_figures.jl

Render the four comparative figure counterparts for the paired-Ridge
supplement. The figures reproduce the content and conventions of Main Figures
1--4, but place the NN and paired-Ridge results in the same asset with shared
axes. This makes visual comparisons meaningful without changing the main
figures.

Inputs default to:

  output/main/figdata.jld2
  output/ridge/paired/figdata.jld2

Set `BROKERAGE_ABM_NN_FIGDATA` or `BROKERAGE_ABM_RIDGE_FIGDATA` to override an
input. Set `BROKERAGE_ABM_RIDGE_FIGURE_DIR` to override the output directory.

Usage: julia --project --threads=auto scripts/ridge/paired_figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

using Dates: now
using JLD2: load

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const NN_PATH = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_NN_FIGDATA",
        joinpath(REPO_ROOT, "output", "main", "figdata.jld2"),
    ),
)
const RIDGE_PATH = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_RIDGE_FIGDATA",
        joinpath(REPO_ROOT, "output", "ridge", "paired", "figdata.jld2"),
    ),
)
const OUT_DIR = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_RIDGE_FIGURE_DIR",
        joinpath(REPO_ROOT, "output", "ridge", "paired", "figures"),
    ),
)
const REPORTING_PROVENANCE = reporting_git_provenance(REPO_ROOT)

const NN = load(NN_PATH)["figdata"]
const RIDGE = load(RIDGE_PATH)["figdata"]
const MODELS = ((label="NN", data=NN), (label="Paired Ridge", data=RIDGE))
const ROLLW = 5
const BETWINT = 20
const TSTART = 30
const PXU = 2.0
const RHO_COLORS = Dict(
    0.0 => :seagreen,
    0.3 => :mediumaquamarine,
    0.5 => :goldenrod,
    0.7 => :darkorange,
    0.85 => :orangered,
    1.0 => :firebrick,
)
const DELTA_COLORS = Dict(
    0.0 => :steelblue,
    0.25 => :cadetblue,
    0.5 => :goldenrod,
    0.75 => :darkorange,
    1.0 => :firebrick,
)

function validate_inputs()
    for (label, data, expected_model) in (("NN", NN, "nn"), ("Ridge", RIDGE, "ridge"))
        meta = data["meta"]
        counts = meta["condition_seed_counts"]
        length(counts) == 98 || error("expected 98 $label effective realizations")
        meta["n_runs"] == 1990 || error("expected 1,990 $label runs")
        meta["baseline_n_seeds"] == 50 || error("expected 50 $label baseline seeds")
        all(rel == "oat/rho=0.5" ? n == 50 : n == 20 for (rel, n) in counts) ||
            error("unexpected $label seed plan")
        lowercase(String(meta["learning_model"])) == expected_model ||
            error("unexpected $label learning model")
        meta["analysis_git_commit"] == REPORTING_PROVENANCE.commit ||
            error("$label figure data were extracted by a different analysis commit")
        meta["analysis_source_clean"] == true ||
            error("$label figure data were extracted from dirty analysis sources")
    end
    NN["period"] == RIDGE["period"] || error("NN and Ridge periods differ")
    Set(keys(NN["meta"]["condition_seed_counts"])) ==
    Set(keys(RIDGE["meta"]["condition_seed_counts"])) ||
        error("NN and Ridge effective designs differ")
    nn_grid = Set((cell["rho"], cell["delta"]) for cell in NN["grid_cells"])
    ridge_grid = Set((cell["rho"], cell["delta"]) for cell in RIDGE["grid_cells"])
    nn_grid == ridge_grid || error("NN and Ridge rho-by-delta grids differ")
    return nothing
end

function padded(values; lower=nothing, upper=nothing, fraction=0.06)
    kept = filter(!isnan, Float64.(collect(values)))
    isempty(kept) && error("cannot scale an empty series")
    lo, hi = extrema(kept)
    pad = fraction * (hi - lo + eps())
    return (isnothing(lower) ? lo - pad : lower, isnothing(upper) ? hi + pad : upper)
end

function save_figure(name, figure)
    save(joinpath(OUT_DIR, name), figure; px_per_unit=PXU)
    println("  $name done")
    return nothing
end

function measured_series(data, key)
    period = data["period"]
    indices = if key == "betweenness"
        [index for index in eachindex(period) if period[index] % BETWINT == 0]
    else
        collect(eachindex(period))
    end
    raw = data["series_seed_values"][key][indices, :]
    smoothed = reduce(
        hcat, (rolling_mean(view(raw, :, seed_index), ROLLW) for seed_index in axes(raw, 2))
    )
    summaries = [
        monte_carlo_interval(view(smoothed, period_index, :)) for
        period_index in axes(smoothed, 1)
    ]
    return (
        period[indices],
        [summary.mean for summary in summaries],
        [summary.lower for summary in summaries],
        [summary.upper for summary in summaries],
    )
end

function displayed_values(series...)
    values = Float64[]
    for (period, _, lower, upper) in series
        for bound in (lower, upper)
            append!(values, bound[(period .>= TSTART) .& .!isnan.(bound)])
        end
    end
    return values
end

function draw_interval_series!(
    axis, series, color; label=nothing, linestyle=:solid, points=false
)
    period, estimate, lower, upper = series
    band!(axis, period, lower, upper; color=(color, 0.16))
    keywords = isnothing(label) ? (;) : (; label)
    if points
        scatterlines!(
            axis,
            period,
            estimate;
            color,
            linewidth=2.2,
            markersize=5,
            linestyle,
            keywords...,
        )
    else
        lines!(axis, period, estimate; color, linewidth=2.2, linestyle, keywords...)
    end
    return nothing
end

function figure_r1()
    period = NN["period"]
    period == RIDGE["period"] || error("period mismatch")
    tend = maximum(period)
    series = Dict(
        (model.label, key) => measured_series(model.data, key) for model in MODELS for
        key in
        ("mpa", "mean_degree", "median_degree", "betweenness", "access", "outsourcing")
    )
    ylimits = Dict(
        "mpa" => padded(
            displayed_values((series[(model.label, "mpa")] for model in MODELS)...);
            lower=0.0,
        ),
        "degree" => padded(
            displayed_values(
                (
                    series[(model.label, key)] for model in MODELS for
                    key in ("mean_degree", "median_degree")
                )...,
            );
            lower=0.0,
        ),
        "betweenness" => padded(
            displayed_values((series[(model.label, "betweenness")] for model in MODELS)...);
            lower=0.0,
        ),
        "access" => padded(
            displayed_values((series[(model.label, "access")] for model in MODELS)...);
            lower=0.0,
        ),
        "outsourcing" => (0.0, 1.02),
    )
    panels = (
        (title="Matches per agent", limit="mpa"),
        (title="Network degree", limit="degree"),
        (title="Broker betweenness", limit="betweenness"),
        (title="Access fraction", limit="access"),
        (title="Outsourcing rate", limit="outsourcing"),
    )
    fig = Figure(; size=(1580, 650), fontsize=16)
    for (row, model) in enumerate(MODELS)
        Label(fig[row, 0], model.label; rotation=pi / 2, fontsize=20, font=:bold)
        for (column, panel) in enumerate(panels)
            axis = Axis(
                fig[row, column];
                title=row == 1 ? panel.title : "",
                xlabel=row == length(MODELS) ? "period" : "",
                xticks=100:100:tend,
                limits=((TSTART, tend + 1), ylimits[panel.limit]),
            )
            row == 1 && hidexdecorations!(axis; grid=false)
            if column == 1
                draw_interval_series!(axis, series[(model.label, "mpa")], COL_DIAG)
            elseif column == 2
                draw_interval_series!(
                    axis, series[(model.label, "mean_degree")], COL_AGENT; label="mean"
                )
                draw_interval_series!(
                    axis,
                    series[(model.label, "median_degree")],
                    COL_REFERENCE;
                    linestyle=:dot,
                    label="median",
                )
                row == 1 && axislegend(axis; position=:rb, LEG_KW...)
            elseif column == 3
                draw_interval_series!(
                    axis, series[(model.label, "betweenness")], COL_GAP; points=true
                )
            elseif column == 4
                draw_interval_series!(axis, series[(model.label, "access")], COL_ACCESS)
            else
                draw_interval_series!(
                    axis, series[(model.label, "outsourcing")], COL_BROKER
                )
            end
        end
    end
    rowsize!(fig.layout, 1, Relative(0.5))
    rowsize!(fig.layout, 2, Relative(0.5))
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)
    save_figure("figR1_dynamics.png", fig)
end

const GRID_LAYOUT = [
    "Betweenness centrality" "Broker rank correlation" "Rank correlation gap";
    "Access fraction" "Principal rank correlation" "Output gap q"
]

function grid_outcomes(data)
    Dict((cell["rho"], cell["delta"]) => cell for cell in data["grid_cells"])
end

function figure_r2()
    model_cells = Dict(model.label => grid_outcomes(model.data) for model in MODELS)
    boundary_cells = Dict(
        model.label =>
            let candidates = [
                    cell for cell in model.data["grid_cells"] if cell["rho"] == 1.0
                ]
                length(unique(cell["rel"] for cell in candidates)) == 1 || error(
                    "$(model.label) rho = 1 coordinates do not share one realization"
                )
                first(candidates)
            end for model in MODELS
    )
    deltas = sort(unique(cell["delta"] for cell in NN["grid_cells"]))
    ylimits = Dict{String,Tuple{Float64,Float64}}()
    for title in GRID_LAYOUT
        outcome_values = [
            cell["outcomes"][title] for cells in values(model_cells) for
            cell in values(cells)
        ]
        ylimits[title] = if title in ("Betweenness centrality", "Access fraction")
            (0.0, 1.02)
        else
            padded(outcome_values)
        end
    end

    fig = Figure(; size=(1320, 1280), fontsize=16)
    for (model_index, model) in enumerate(MODELS), local_row in 1:2, column in 1:3
        row = 2 * (model_index - 1) + local_row
        title = GRID_LAYOUT[local_row, column]
        axis = Axis(
            fig[row, column];
            title=title,
            xlabel=local_row == 2 ? "ρ (complementarity vs quality)" : "",
            xticks=[0, 0.3, 0.5, 0.7, 0.85, 1],
            limits=(nothing, ylimits[title]),
        )
        for delta in deltas
            points = sort(
                [
                    let interval = monte_carlo_interval(cell["outcome_seed_values"][title])
                        (
                            rho=rho,
                            mean=interval.mean,
                            lower=interval.lower,
                            upper=interval.upper,
                        )
                    end for
                    ((rho, d), cell) in model_cells[model.label] if d == delta && rho < 1.0
                ];
                by=point -> point.rho,
            )
            rangebars!(
                axis,
                [point.rho for point in points],
                [point.lower for point in points],
                [point.upper for point in points];
                color=(DELTA_COLORS[delta], 0.72),
                linewidth=1.1,
                whiskerwidth=7,
            )
            scatterlines!(
                axis,
                [point.rho for point in points],
                [point.mean for point in points];
                color=DELTA_COLORS[delta],
                linewidth=2.0,
                markersize=8,
                strokewidth=0.4,
                strokecolor=:gray30,
                label="δ = $delta",
            )
        end
        boundary_interval = monte_carlo_interval(
            boundary_cells[model.label]["outcome_seed_values"][title]
        )
        rangebars!(
            axis,
            [1.0],
            [boundary_interval.lower],
            [boundary_interval.upper];
            color=(:gray25, 0.8),
            linewidth=1.1,
            whiskerwidth=7,
        )
        scatter!(
            axis,
            [1.0],
            [boundary_interval.mean];
            color=:gray25,
            marker=:diamond,
            markersize=9,
            strokewidth=0.4,
            strokecolor=:gray20,
            label="ρ = 1 boundary",
        )
        model_index == 1 &&
            local_row == 1 &&
            column == 1 &&
            axislegend(axis, "Difficulty"; position=:lb, LEG_KW...)
    end
    Label(fig[1:2, 0], "NN"; rotation=pi / 2, fontsize=20, font=:bold)
    Label(fig[3:4, 0], "Paired Ridge"; rotation=pi / 2, fontsize=20, font=:bold)
    colgap!(fig.layout, 14)
    rowgap!(fig.layout, 10)
    save_figure("figR2_grid_lines.png", fig)
end

function figure_r3()
    period = NN["period"]
    tend = maximum(period)
    all_access = [cell["access"] for model in MODELS for cell in model.data["oat_cells"]]
    access_limit = padded(all_access; lower=0.0)
    fig = Figure(; size=(1180, 780), fontsize=17)
    for (row, model) in enumerate(MODELS)
        Label(fig[row, 0], model.label; rotation=pi / 2, fontsize=20, font=:bold)
        time_axis = Axis(
            fig[row, 1];
            title=row == 1 ? "Over time, at baseline" : "",
            xlabel=row == 2 ? "period" : "",
            limits=((TSTART, tend + 1), (0.0, 1.0)),
        )
        betweenness = measured_series(model.data, "betweenness")
        access = measured_series(model.data, "access")
        draw_interval_series!(
            time_axis,
            betweenness,
            COL_GAP;
            points=true,
            label="broker betweenness centrality",
        )
        draw_interval_series!(time_axis, access, COL_ACCESS; label="access fraction")
        row == 1 && axislegend(time_axis; position=:rc, LEG_KW...)
        row == 1 && hidexdecorations!(time_axis; grid=false)

        scatter_axis = Axis(
            fig[row, 2];
            title=row == 1 ? "Across OAT regimes" : "",
            xlabel=row == 2 ? "access fraction" : "",
            ylabel="broker betweenness centrality",
            limits=(access_limit, (0.0, 1.0)),
        )
        cells = model.data["oat_cells"]
        scatter!(
            scatter_axis,
            [cell["access"] for cell in cells],
            [cell["betw"] for cell in cells];
            color=:black,
            markersize=11,
        )
        row == 1 && hidexdecorations!(scatter_axis; grid=false)
    end
    rowsize!(fig.layout, 1, Relative(0.5))
    rowsize!(fig.layout, 2, Relative(0.5))
    colgap!(fig.layout, 16)
    rowgap!(fig.layout, 12)
    save_figure("figR3_position_work.png", fig)
end

function figure_r4()
    outcomes = (
        (label="Rank correlation gap", key="rankgap"), (label="Output gap q", key="qgap")
    )
    predictors = (
        (label="Broker betweenness centrality", key="betw"),
        (label="Access fraction", key="access"),
    )
    ylimits = Dict(
        outcome.key => padded(
            cell[outcome.key] for model in MODELS for cell in model.data["regime_cells"]
        ) for outcome in outcomes
    )
    xlimits = Dict(
        predictor.key => padded(
            cell[predictor.key] for model in MODELS for cell in model.data["regime_cells"]
        ) for predictor in predictors
    )

    fig = Figure(; size=(1580, 820), fontsize=16)
    for (row, outcome) in enumerate(outcomes), predictor_index in 1:2, model_index in 1:2
        model = MODELS[model_index]
        predictor = predictors[predictor_index]
        column = 2 * (predictor_index - 1) + model_index
        axis = Axis(
            fig[row, column];
            title=row == 1 ? "$(model.label): $(predictor.label)" : "",
            xlabel=row == 2 ? predictor.label : "",
            ylabel=column == 1 ? outcome.label : "",
            limits=(xlimits[predictor.key], ylimits[outcome.key]),
        )
        cells = model.data["regime_cells"]
        rho = [cell["rho"] for cell in cells]
        x = [cell[predictor.key] for cell in cells]
        y = [cell[outcome.key] for cell in cells]
        for rho_value in sort(unique(rho))
            mask = rho .== rho_value
            scatter!(
                axis,
                x[mask],
                y[mask];
                color=(RHO_COLORS[rho_value], 0.8),
                markersize=8,
                strokewidth=0.3,
                strokecolor=:gray30,
            )
        end
    end
    rho_values = sort(collect(keys(RHO_COLORS)))
    elements = [
        MarkerElement(; marker=:circle, color=RHO_COLORS[value], markersize=10) for
        value in rho_values
    ]
    Legend(
        fig[0, 1:4],
        elements,
        ["ρ = $value" for value in rho_values];
        orientation=:horizontal,
        framevisible=false,
        tellwidth=false,
    )
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)
    save_figure("figR4_advantage.png", fig)
end

function write_provenance()
    open(joinpath(OUT_DIR, "provenance.txt"), "w") do io
        println(io, "generated=$(now())")
        println(io, "source=scripts/ridge/paired_figures.jl")
        println(io, "analysis_commit=$(REPORTING_PROVENANCE.commit)")
        println(io, "analysis_source_clean=$(REPORTING_PROVENANCE.source_clean)")
        println(io, "nn_sweep=$(NN["meta"]["sweep"])")
        println(io, "nn_manifest=$(NN["meta"]["manifest_hash"])")
        println(io, "ridge_sweep=$(RIDGE["meta"]["sweep"])")
        println(io, "ridge_manifest=$(RIDGE["meta"]["manifest_hash"])")
        println(io, "effective_realizations=98")
        println(io, "general_seeds=20")
        println(io, "baseline_seeds=50")
        println(io, "rolling_window=$ROLLW")
        println(io, "network_measure_interval=$BETWINT")
        println(io, "display_start=$TSTART")
        println(io, "axes=shared between NN and paired Ridge within each figure")
    end
    return nothing
end

function main()
    validate_inputs()
    mkpath(OUT_DIR)
    figure_r1()
    figure_r2()
    figure_r3()
    figure_r4()
    write_provenance()
    println("wrote paired-Ridge figure set to $OUT_DIR")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
