"""
Render the main-text review figure for the broker assessment vs access experiment.

The script reads the retained condition summaries and writes
`output/main/figures/assessment_access.png`.

Usage: julia --project --threads=auto scripts/assessment_access/main_figure.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))
using JLD2

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIGURE_DATA = joinpath(REPO_ROOT, "output", "assessment_access", "figure_data.jld2")
const FIGURE_PATH = joinpath(
    REPO_ROOT, "output", "main", "figures", "assessment_access.png"
)
const RHO = 0.5
const ETAS = (0.0, 0.001, 0.01, 0.02, 0.03)
const POSITIVE_ETAS = ETAS[2:end]
const MODES = ("full", "assessment_only", "access_only")
const MODE_LABELS = Dict(
    "full" => "Assessment and access (baseline)",
    "assessment_only" => "Assessment only",
    "access_only" => "Access only",
)
const MODE_COLORS = Dict(
    "full" => :black,
    "assessment_only" => Makie.to_color("#4477AA"),
    "access_only" => Makie.to_color("#EE7733"),
)

function summary_index(rows)
    index = Dict{Tuple{String,Float64,Float64,String},NamedTuple}()
    for row in rows
        key = (String(row[1]), Float64(row[2]), Float64(row[3]), String(row[4]))
        haskey(index, key) && error("duplicate condition summary: $key")
        index[key] = (;
            estimate=Float64(row[5]),
            se=Float64(row[6]),
            lower=Float64(row[7]),
            upper=Float64(row[8]),
            n=Int(row[9]),
        )
    end
    return index
end

function interval(index, mode, eta, metric)
    key = (mode, RHO, eta, metric)
    haskey(index, key) || error("missing condition summary: $key")
    return index[key]
end

function contrast_index(rows)
    index = Dict{Tuple{String,Float64,Float64,String},NamedTuple}()
    for row in rows
        key = (String(row[1]), Float64(row[2]), Float64(row[3]), String(row[4]))
        haskey(index, key) && error("duplicate paired contrast: $key")
        index[key] = (;
            estimate=Float64(row[5]),
            se=Float64(row[6]),
            lower=Float64(row[7]),
            upper=Float64(row[8]),
            n=Int(row[9]),
        )
    end
    return index
end

function relative_interval(index, mode, eta)
    contribution = mode == "assessment_only" ? "access" : "assessment"
    key = (contribution, RHO, eta, "net_output_per_requested_position")
    haskey(index, key) || error("missing paired contrast: $key")
    value = index[key]
    return (;
        estimate=(-value.estimate),
        se=value.se,
        lower=(-value.upper),
        upper=(-value.lower),
        n=value.n,
    )
end

function draw_series!(axis, index, mode, metric)
    values = [interval(index, mode, eta, metric) for eta in ETAS]
    color = MODE_COLORS[mode]
    rangebars!(
        axis,
        collect(ETAS),
        [value.lower for value in values],
        [value.upper for value in values];
        color,
        linewidth=1.3,
        whiskerwidth=7,
    )
    scatter!(
        axis,
        collect(ETAS),
        [value.estimate for value in values];
        color,
        markersize=9,
        strokecolor=:gray25,
        strokewidth=0.4,
    )
    lines!(
        axis,
        collect(POSITIVE_ETAS),
        [interval(index, mode, eta, metric).estimate for eta in POSITIVE_ETAS];
        color,
        linewidth=2.2,
    )
    return nothing
end

function draw_relative_output!(axis, index, mode)
    values = [relative_interval(index, mode, eta) for eta in ETAS]
    color = MODE_COLORS[mode]
    rangebars!(
        axis,
        collect(ETAS),
        [value.lower for value in values],
        [value.upper for value in values];
        color,
        linewidth=1.3,
        whiskerwidth=7,
    )
    scatter!(
        axis,
        collect(ETAS),
        [value.estimate for value in values];
        color,
        markersize=9,
        strokecolor=:gray25,
        strokewidth=0.4,
    )
    lines!(
        axis,
        collect(POSITIVE_ETAS),
        [relative_interval(index, mode, eta).estimate for eta in POSITIVE_ETAS];
        color,
        linewidth=2.2,
    )
    return nothing
end

function main()
    isfile(FIGURE_DATA) || error("missing figure data: $FIGURE_DATA")
    data = JLD2.load(FIGURE_DATA)
    data["analysis_source_clean"] == true ||
        error("broker assessment vs access data were produced from dirty sources")
    data["interval_level"] == 0.95 || error("expected 95% Monte Carlo intervals")
    index = summary_index(data["summary_rows"])
    contrasts = contrast_index(data["contrast_rows"])

    panel_specs = (
        (
            title="A. Market performance",
            metric="net_output_per_requested_position",
            ylabel="Net output difference from baseline",
            ylimits=(-0.45, 0.10),
        ),
        (
            title="B. Reliance on the broker",
            metric="outsourcing_rate",
            ylabel="Share of demand outsourced",
            ylimits=(0.0, 1.0),
        ),
        (
            title="C. Principal network connectivity",
            metric="mean_degree",
            ylabel="Direct ties per principal",
            ylimits=(0.0, 85.0),
        ),
        (
            title="D. Broker's structural position",
            metric="betweenness",
            ylabel="Betweenness centrality",
            ylimits=(0.0, 1.0),
        ),
    )

    for panel in panel_specs, mode in MODES, eta in ETAS
        expected_n = eta == 0.02 ? 50 : 20
        interval(index, mode, eta, panel.metric).n == expected_n ||
            error("unexpected seed count")
    end
    for mode in ("assessment_only", "access_only"), eta in ETAS
        expected_n = eta == 0.02 ? 50 : 20
        relative_interval(contrasts, mode, eta).n == expected_n ||
            error("unexpected paired-contrast seed count")
    end

    fig = Figure(; size=(1200, 820))
    for (panel_index, panel) in enumerate(panel_specs)
        row = panel_index <= 2 ? 1 : 2
        column = isodd(panel_index) ? 1 : 2
        axis = Axis(
            fig[row, column];
            title=panel.title,
            xlabel="Turnover rate (η)",
            ylabel=panel.ylabel,
            limits=(nothing, panel.ylimits),
            xticks=(collect(ETAS), ["0", "0.001", "0.01", "0.02", "0.03"]),
            xticklabelrotation=pi / 4,
            titlesize=TITLE_FS - 4,
            xlabelsize=LABEL_FS - 2,
            ylabelsize=LABEL_FS - 2,
            xticklabelsize=TICK_FS - 2,
            yticklabelsize=TICK_FS - 2,
        )
        if panel_index == 1
            lines!(
                axis,
                [first(POSITIVE_ETAS), last(POSITIVE_ETAS)],
                [0.0, 0.0];
                color=MODE_COLORS["full"],
                linewidth=2.2,
            )
            scatter!(
                axis,
                [first(ETAS)],
                [0.0];
                color=MODE_COLORS["full"],
                markersize=9,
                strokecolor=:gray25,
                strokewidth=0.4,
            )
            for mode in ("assessment_only", "access_only")
                draw_relative_output!(axis, contrasts, mode)
            end
        else
            for mode in MODES
                draw_series!(axis, index, mode, panel.metric)
            end
        end
    end

    legend_elements = [
        LineElement(; color=MODE_COLORS[mode], linewidth=2.2, marker=:circle) for
        mode in MODES
    ]
    Legend(
        fig[0, 1:2],
        legend_elements,
        [MODE_LABELS[mode] for mode in MODES];
        orientation=:horizontal,
        framevisible=false,
        labelsize=TICK_FS - 2,
        tellheight=true,
    )

    rowgap!(fig.layout, 15)
    colgap!(fig.layout, 24)
    mkpath(dirname(FIGURE_PATH))
    save(FIGURE_PATH, fig; px_per_unit=2)
    println("Wrote $FIGURE_PATH")
end

main()
