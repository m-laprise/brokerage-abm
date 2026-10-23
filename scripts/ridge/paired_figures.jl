"""Render paired NN–Ridge figures from retained analyses."""

isdefined(@__MODULE__, :ReportingSettings) || include(joinpath(@__DIR__, "..", "reporting_settings.jl"))

include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

using JLD2: load

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const NN_PATH = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_NN_FIGDATA",
        joinpath(REPO_ROOT, "output", "main", "figure_data.jld2"),
    ),
)
const RIDGE_PATH = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_RIDGE_FIGDATA",
        joinpath(REPO_ROOT, "output", "ridge", "paired", "figure_data.jld2"),
    ),
)
const OUT_DIR = normpath(
    get(
        ENV,
        "BROKERAGE_ABM_RIDGE_FIGURE_DIR",
        joinpath(REPO_ROOT, "output", "ridge", "paired", "figures"),
    ),
)
const REPORTING_PROVENANCE = manuscript_git_provenance(
    REPO_ROOT;
    sources=("scripts/analysis_settings.jl", "scripts/reporting_settings.jl",
        @__FILE__,
        "scripts/monte_carlo.jl",
        "scripts/figure_style.jl",
    ),
)

const NN = load(NN_PATH)["figdata"]
const RIDGE = load(RIDGE_PATH)["figdata"]
const NN_SETTINGS = ReportingSettings.validate_figure(NN, ReportingSettings.retained(NN_PATH, NN))
const RIDGE_SETTINGS = ReportingSettings.validate_figure(RIDGE, ReportingSettings.retained(RIDGE_PATH, RIDGE))
const BASELINE_REL = NN_SETTINGS["baseline_reldir"]
comparison_interval(values) = monte_carlo_interval(values; level=Float64(NN_SETTINGS["interval_level"]))
for key in ("baseline_reldir", "network_measure_interval", "late_window", "early_window", "interval_level")
    NN_SETTINGS[key] == RIDGE_SETTINGS[key] || error("NN and Ridge reporting settings differ: $key")
end
const MODELS = ((label="NN", data=NN), (label="Base Ridge", data=RIDGE))
const ROLLW = AnalysisSettings.ROLLING_WIDTH
const BETWINT = Int(NN_SETTINGS["network_measure_interval"])
const TSTART = AnalysisSettings.AXIS_START
const PXU = 2.0
const FIGURE_VALUES = joinpath(dirname(OUT_DIR), "figure_values.tex")
const RHO_COLORS = Dict(
    0.0 => :seagreen,
    0.15 => :royalblue,
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
        length(counts) == length(data["regime_cells"]) ||
            error("$label effective-realization count does not match its metadata")
        meta["n_runs"] == sum(values(counts)) ||
            error("$label run count does not match its condition seed counts")
        lowercase(String(meta["learning_model"])) == expected_model ||
            error("unexpected $label learning model")
        validate_analysis_commit(
            REPORTING_PROVENANCE, meta["analysis_git_commit"]; artifact="$label figure data",
        )
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
        comparison_interval(view(smoothed, period_index, :)) for
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
    save_figure("baseline_dynamics.png", fig)
end

const GRID_KEYS = [
    "Betweenness centrality" "Broker rank correlation" "Rank correlation gap";
    "Access fraction" "Principal rank correlation" "Output gap q"
]
const GRID_LABELS = [
    "Betweenness centrality" "Broker rank correlation" "Rank-correlation difference";
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
    for key in GRID_KEYS
        outcome_values = [
            cell["outcomes"][key] for cells in values(model_cells) for cell in values(cells)
        ]
        ylimits[key] = if key in ("Betweenness centrality", "Access fraction")
            (0.0, 1.02)
        else
            padded(outcome_values)
        end
    end

    fig = Figure(; size=(1320, 1280), fontsize=16)
    for (model_index, model) in enumerate(MODELS), local_row in 1:2, column in 1:3
        row = 2 * (model_index - 1) + local_row
        key = GRID_KEYS[local_row, column]
        axis = Axis(
            fig[row, column];
            title=GRID_LABELS[local_row, column],
            xlabel=local_row == 2 ? "ρ (complementarity vs quality)" : "",
            xticks=[0, 0.3, 0.5, 0.7, 0.85, 1],
            limits=(nothing, ylimits[key]),
        )
        for delta in deltas
            points = sort(
                [
                    let interval = comparison_interval(cell["outcome_seed_values"][key])
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
        boundary_interval = comparison_interval(
            boundary_cells[model.label]["outcome_seed_values"][key]
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
    Label(fig[3:4, 0], "Base Ridge"; rotation=pi / 2, fontsize=20, font=:bold)
    colgap!(fig.layout, 14)
    rowgap!(fig.layout, 10)
    save_figure("matching_grid.png", fig)
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
    save_figure("centrality_and_access.png", fig)
end

function figure_r4()
    outcomes = (
        (label="Rank-correlation difference", key="rankgap"),
        (label="Output gap q", key="qgap"),
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
    save_figure("structural_advantage.png", fig)
end

function write_provenance()
    nn_counts = NN["meta"]["condition_seed_counts"]
    nonbaseline_seed_counts = unique(
        n for (rel, n) in nn_counts if rel != BASELINE_REL
    )
    length(nonbaseline_seed_counts) == 1 || error("NN nonbaseline seed counts differ")
    record_publication_stage(REPORTING_PROVENANCE;
        stage="scripts/ridge/paired_figures.jl", inputs=(ReportingSettings.retained_dependencies(NN)..., ReportingSettings.retained_dependencies(RIDGE)..., NN_PATH, RIDGE_PATH),
        outputs=(FIGURE_VALUES, joinpath.(OUT_DIR, ("baseline_dynamics.png",
            "matching_grid.png", "centrality_and_access.png", "structural_advantage.png"))...),
        analysis_commits=Dict("nn analysis commit" => NN["meta"]["analysis_git_commit"],
            "ridge analysis commit" => RIDGE["meta"]["analysis_git_commit"]),
        settings=Dict("rolling_window" => ROLLW, "network_measure_interval" => BETWINT,
            "display_start" => TSTART),
    )
    return nothing
end

function write_figure_values()
    content = sprint() do io
        println(io, "% Generated by scripts/ridge/paired_figures.jl; do not edit.")
        println(
            io,
            raw"\newcommand{\rfvDefine}[2]{\expandafter\newcommand\csname rfv@#1\endcsname{#2}}",
        )
        println(io, raw"\newcommand{\rfv}[1]{\csname rfv@#1\endcsname}")
        println(io, "\\rfvDefine{rollingWindow}{$ROLLW}")
        println(io, "\\rfvDefine{networkMeasureInterval}{$BETWINT}")
        println(io, "\\rfvDefine{displayStart}{$TSTART}")
    end
    write_if_changed(FIGURE_VALUES, content)
    return nothing
end

function main()
    validate_inputs()
    mkpath(OUT_DIR)
    figure_r1()
    figure_r2()
    figure_r3()
    figure_r4()
    write_figure_values()
    write_provenance()
    println("wrote base Ridge figure set to $OUT_DIR")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
