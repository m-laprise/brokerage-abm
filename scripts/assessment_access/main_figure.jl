"""
Render the retained main-figure alternative and the complementarity supplement.

Turnover is horizontal and outcomes are vertical. Zero turnover is displayed in
a separate strip; positive turnover uses a linear axis. Service modes are offset
horizontally for visibility. Net output uses the saved seed-paired differences
from full service; other panels show condition means.

The --complementarity option renders the supplementary comparison of adding
assessment and adding access across matching composition. It uses the common
turnover grid and checks saved contrasts against retained seed-level data.
Its caption is in paper/supplement.tex. Neither option changes simulations.

Usage: julia --project --threads=auto scripts/assessment_access/main_figure.jl
Outputs: output/main/figures/assessment_access.{png,pdf}

Usage: julia --project --threads=auto scripts/assessment_access/main_figure.jl --complementarity
Output: output/supplement/figures/complementarity_contributions.png
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
using JLD2
using Printf: @sprintf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIGURE_DATA = joinpath(REPO_ROOT, "output", "assessment_access", "figure_data.jld2")
const TRAJECTORY_DATA = joinpath(
    REPO_ROOT, "output", "assessment_access", "centrality_trajectories.jld2"
)
const FIGURE_BASE = joinpath(REPO_ROOT, "output", "main", "figures", "assessment_access")
const RHO = 0.5
const ETAS = (0.0, 0.001, 0.01, 0.02, 0.03)
const POSITIVE_ETAS = ETAS[2:end]
const MODES = ("full", "assessment_only", "access_only")
const MODE_LABELS = ("Full service", "Assessment only", "Access only")
const MODE_STYLES = Dict(
    "full" =>
        (color=Makie.to_color("#333A40"), marker=:rect, linestyle=:dot, offset=-1),
    "assessment_only" =>
        (color=Makie.to_color("#237A93"), marker=:circle, linestyle=:solid, offset=0),
    "access_only" =>
        (color=Makie.to_color("#BA5A3A"), marker=:utriangle, linestyle=:dash, offset=1),
)
const PANEL_SPECS = (
    (
        title="A. Outsourcing",
        metric="outsourcing_rate",
        ylabel="Share of requested positions",
        ylimits=(0.0, 1.03),
        yticks=0:0.25:1,
        relative=false,
    ),
    (
        title="B. Net output to principals",
        metric="net_output_per_requested_position",
        ylabel="Difference from full service\n(per requested position)",
        ylimits=(-0.45, 0.10),
        yticks=-0.4:0.1:0.1,
        relative=true,
    ),
    (
        title="C. Principal connectivity",
        metric="mean_degree",
        ylabel="Mean principal degree",
        ylimits=(0.0, 85.0),
        yticks=0:20:80,
        relative=false,
    ),
    (
        title="D. Broker centrality",
        metric="betweenness",
        ylabel="Betweenness centrality",
        ylimits=(0.0, 1.0),
        yticks=0:0.25:1,
        relative=false,
    ),
)

"""Index retained estimates without changing their values or interval method."""
function interval_index(rows)
    index = Dict{Tuple{String,Float64,Float64,String},NamedTuple}()
    for row in rows
        key = (String(row[1]), Float64(row[2]), Float64(row[3]), String(row[4]))
        haskey(index, key) && error("duplicate retained estimate: $key")
        value = (;
            estimate=Float64(row[5]),
            se=Float64(row[6]),
            lower=Float64(row[7]),
            upper=Float64(row[8]),
            n=Int(row[9]),
        )
        all(isfinite, (value.estimate, value.se, value.lower, value.upper)) ||
            error("nonfinite retained estimate: $key")
        value.lower <= value.estimate <= value.upper ||
            error("invalid retained interval: $key")
        index[key] = value
    end
    return index
end

"""Return a condition mean or a seed-paired restricted-minus-full difference."""
function displayed_interval(summaries, contrasts, mode, eta, panel)
    expected_n = summaries[("full", RHO, eta, panel.metric)].n
    if panel.relative
        mode in ("assessment_only", "access_only") ||
            error("full service is the zero reference, not an estimated contrast")
        contribution = mode == "assessment_only" ? "access" : "assessment"
        key = (contribution, RHO, eta, panel.metric)
        haskey(contrasts, key) || error("missing paired contrast: $key")
        value = contrasts[key]
        # Retained contrasts are full minus restricted. Reverse both endpoints.
        result = (;
            estimate=(-value.estimate),
            se=value.se,
            lower=(-value.upper),
            upper=(-value.lower),
            n=value.n,
        )
    else
        key = (mode, RHO, eta, panel.metric)
        haskey(summaries, key) || error("missing condition summary: $key")
        result = summaries[key]
    end
    result.n == expected_n || error("unexpected seed count for $mode, eta=$eta")
    panel.ylimits[1] <= result.lower <= result.upper <= panel.ylimits[2] ||
        error("display limits would clip an interval for $mode, eta=$eta")
    return result
end

"""Draw one service mode, preserving its estimates and interval endpoints."""
function draw_series!(axis, summaries, contrasts, mode, panel; zero=false)
    etas = zero ? (0.0,) : POSITIVE_ETAS
    values = [displayed_interval(summaries, contrasts, mode, eta, panel) for eta in etas]
    style = MODE_STYLES[mode]
    positions = collect(etas) .+ style.offset * (zero ? 0.27 : 0.0005)
    if !zero
        lines!(
            axis,
            positions,
            [value.estimate for value in values];
            color=style.color,
            linestyle=style.linestyle,
            linewidth=2.1,
        )
    end
    rangebars!(
        axis,
        positions,
        [value.lower for value in values],
        [value.upper for value in values];
        direction=:y,
        color=style.color,
        linewidth=1.7,
        whiskerwidth=7,
    )
    scatter!(
        axis,
        positions,
        [value.estimate for value in values];
        color=style.color,
        marker=style.marker,
        markersize=9,
    )
    return nothing
end

"""Validate the overview's design against retained simulation and analysis metadata."""
function overview_design(data)
    trajectories = JLD2.load(TRAJECTORY_DATA)
    trajectories["manifest_hash"] == data["manifest_hash"] || error("manifest mismatch")
    for commit in (trajectories["retained_analysis_git_commit"], data["analysis_git_commit"])
        validate_analysis_commit((; root=REPO_ROOT), commit; artifact="assessment-access analysis")
    end
    config = trajectories["configs"]["full"]
    config["rho"] == RHO || error("overview does not select the reporting baseline")
    summaries = interval_index(data["summary_rows"])
    seed_counts = Dict(
        eta => summaries[("full", RHO, eta, "outsourcing_rate")].n for eta in ETAS
    )
    horizon = Int(config["T"])
    late_width = Int(data["late_width"])
    0 < late_width <= horizon || error("invalid late window")
    level = Float64(data["interval_level"])
    0 < level < 1 || error("invalid interval level")
    return (;
        rho=Float64(config["rho"]),
        delta=Float64(config["delta"]),
        eta=Float64(config["eta"]),
        horizon,
        late_start=horizon - late_width + 1,
        interval_percent=@sprintf("%.0f", 100 * level),
        baseline_seeds=seed_counts[config["eta"]],
        other_seeds=only(unique(n for (eta, n) in seed_counts if eta != config["eta"])),
    )
end

"""Build the four-panel figure from the retained, clean-source analysis artifact."""
function make_figure(data)
    data["analysis_source_clean"] == true || error("analysis sources were dirty")
    design = overview_design(data)
    summaries = interval_index(data["summary_rows"])
    contrasts = interval_index(data["contrast_rows"])
    publication_theme!()
    fig = Figure(; size=(1160, 760), fontsize=19, figure_padding=24)

    for (panel_index, panel) in enumerate(PANEL_SPECS)
        row, column = divrem(panel_index - 1, 2) .+ (1, 1)
        grid = GridLayout(fig[row, column])
        Label(
            grid[0, 1:2],
            panel.title;
            fontsize=24,
            font=:bold,
            halign=:left,
            tellwidth=false,
        )
        for (axis_column, zero) in enumerate((true, false))
            axis = Axis(
                grid[1, axis_column];
                ylabel=zero ? panel.ylabel : "",
                ylabelsize=20,
                xticklabelsize=18,
                yticklabelsize=18,
                limits=(zero ? (-0.7, 0.7) : (-0.001, 0.032), panel.ylimits),
                xticks=if zero
                    ([0.0], ["0"])
                else
                    (collect(POSITIVE_ETAS), string.(collect(POSITIVE_ETAS)))
                end,
                yticks=panel.yticks,
                yticklabelsvisible=zero,
                yticksvisible=zero,
                leftspinevisible=zero,
                topspinevisible=false,
                rightspinevisible=false,
                xgridvisible=false,
                ygridcolor=(:black, 0.075),
                backgroundcolor=zero ? (:black, 0.025) : :white,
            )
            if !zero
                half_width = minimum(diff(collect(ETAS)))
                vspan!(
                    axis, design.eta - half_width, design.eta + half_width;
                    color=(:black, 0.045),
                )
            end
            if panel.relative
                hlines!(axis, [0.0]; color=:gray45, linestyle=:dash, linewidth=1.2)
            end
            for mode in MODES
                panel.relative && mode == "full" && continue
                draw_series!(axis, summaries, contrasts, mode, panel; zero)
            end
        end
        Label(grid[2, 1:2], "Turnover rate (η)"; fontsize=21, tellwidth=false)
        colsize!(grid, 1, Fixed(60))
        colgap!(grid, 17)
        rowgap!(grid, 8)
    end

    Legend(
        fig[0, 1:2],
        [
            [
                LineElement(;
                    color=MODE_STYLES[mode].color,
                    linestyle=MODE_STYLES[mode].linestyle,
                    linewidth=2.1,
                ),
                MarkerElement(;
                    color=MODE_STYLES[mode].color,
                    marker=MODE_STYLES[mode].marker,
                    markersize=10,
                ),
            ] for mode in MODES
        ],
        collect(MODE_LABELS);
        orientation=:horizontal,
        framevisible=false,
        labelsize=21,
        colgap=30,
        padding=(0, 0, 5, 0),
    )
    rowgap!(fig.layout, 22)
    colgap!(fig.layout, 30)
    return fig
end

"""Save a print-resolution PNG and a vector PDF without altering retained data."""
function main(; output_base=FIGURE_BASE, formats=("png", "pdf"))
    isfile(FIGURE_DATA) || error("missing figure data: $FIGURE_DATA")
    data = JLD2.load(FIGURE_DATA)
    provenance = manuscript_git_provenance(
        REPO_ROOT;
        sources=(
            @__FILE__,
            "scripts/monte_carlo.jl",
            "scripts/figure_style.jl",
        ),
    )
    validate_analysis_commit(
        provenance, data["analysis_git_commit"]; artifact="assessment-access figure data"
    )
    fig = make_figure(data)
    mkpath(dirname(output_base))
    for format in formats
        path = "$output_base.$format"
        if format == "png"
            save(path, fig; px_per_unit=2)
        elseif format == "pdf"
            save(path, fig; pt_per_unit=0.5)
        else
            error("unsupported figure format: $format")
        end
        println("Wrote $path")
    end
    return nothing
end

"""Select the common composition-turnover grid and verify every displayed contrast."""
function complementarity_estimates(data)
    data["analysis_source_clean"] === true || error("analysis sources were dirty")
    summaries = interval_index(data["summary_rows"])
    contrasts = interval_index(data["contrast_rows"])
    metric = "net_output_per_requested_position"
    rhos = sort!(unique(key[2] for key in keys(summaries) if key[4] == metric))
    available = [
        Set(
            key[3] for
            key in keys(summaries) if key[1:2] == ("full", rho) && key[4] == metric
        ) for rho in rhos
    ]
    etas = sort!(collect(reduce(intersect, available)))
    isempty(etas) && error("no common turnover grid")
    level = Float64(data["interval_level"])
    seed_values = Dict{Tuple{String,Float64,Float64},Dict{Int,Float64}}()
    for row in data["seed_rows"]
        String(row[5]) == metric || continue
        key = (String(row[1]), Float64(row[2]), Float64(row[3]))
        seed_map = get!(seed_values, key, Dict{Int,Float64}())
        seed = Int(row[4])
        haskey(seed_map, seed) && error("duplicate retained seed: $key, $seed")
        seed_map[seed] = Float64(row[6])
    end
    panels = (
        (title="A  Adding assessment", contrast="assessment", reference="access_only"),
        (title="B  Adding access", contrast="access", reference="assessment_only"),
    )
    shown = Dict{Tuple{String,Float64,Float64},NamedTuple}()
    for panel in panels, eta in etas, rho in rhos
        full = seed_values[("full", rho, eta)]
        reference = seed_values[(panel.reference, rho, eta)]
        Set(keys(full)) == Set(keys(reference)) || error("paired seed sets differ")
        seeds = sort!(collect(keys(full)))
        all(isfinite, values(full)) && all(isfinite, values(reference)) ||
            error("nonfinite retained output")
        length(seeds) ==
        summaries[("full", rho, eta, metric)].n ==
        summaries[(panel.reference, rho, eta, metric)].n || error("incomplete seeds")
        expected = monte_carlo_interval([full[s] - reference[s] for s in seeds]; level)
        saved = contrasts[(panel.contrast, rho, eta, metric)]
        all(
            isapprox(a, b; atol=1e-12) for (a, b) in zip(
                (saved.estimate, saved.se, saved.lower, saved.upper, saved.n),
                (expected.mean, expected.se, expected.lower, expected.upper, expected.n),
            )
        ) || error("retained contrast disagrees with seeds: $(panel.contrast), $rho, $eta")
        shown[(panel.contrast, rho, eta)] = saved
    end
    return (; rhos, etas, panels, shown, level)
end

"""Draw the supplementary comparison with shared scales and no embedded caption."""
function make_complementarity_figure(data)
    publication_theme!()
    selected = complementarity_estimates(data)
    styles = (
        (color=PUB_ETA_COLORS[0.0], marker=:circle, linestyle=:dot),
        (color=PUB_ETA_COLORS[0.01], marker=:rect, linestyle=:dash),
        (color=PUB_ETA_COLORS[0.03], marker=:utriangle, linestyle=:solid),
    )
    length(selected.etas) == length(styles) || error("turnover legend needs updating")
    bounds = extrema(
        v for value in values(selected.shown) for v in (value.lower, value.upper)
    )
    tick_step = 0.1
    ylimits = (
        floor(bounds[1] / tick_step) * tick_step,
        ceil(bounds[2] / tick_step) * tick_step + 0.025,
    )
    ticks = collect(first(ylimits):tick_step:(floor(last(ylimits) / tick_step) * tick_step))
    ticklabels = [@sprintf("%.1f", abs(value) < eps() ? 0.0 : value) for value in ticks]
    rho_labels = [
        if iszero(rho)
            "$(Int(rho))\nComplementarity"
        elseif isone(rho)
            "$(Int(rho))\nGeneral quality"
        else
            string(rho)
        end for rho in selected.rhos
    ]
    fig = Figure(; size=(1320, 620), fontsize=TICK_FS, figure_padding=(24, 28, 18, 22))
    for (column, panel) in enumerate(selected.panels)
        grid = GridLayout(fig[1, column])
        Label(
            grid[1, 1],
            panel.title;
            fontsize=TITLE_FS,
            font=:bold,
            halign=:left,
            tellwidth=false,
        )
        axis = Axis(
            grid[2, 1];
            xlabel="General-quality share (ρ)",
            ylabel=if column == 1
                "Change in net output to principals\nper requested position"
            else
                ""
            end,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticks=(selected.rhos, rho_labels),
            xticklabelsize=TICK_FS,
            yticks=(ticks, ticklabels),
            yticklabelsize=TICK_FS,
            limits=((first(selected.rhos) - 0.13, last(selected.rhos) + 0.13), ylimits),
            xgridvisible=false,
            ygridcolor=(:black, 0.065),
            topspinevisible=false,
            rightspinevisible=false,
            yticklabelsvisible=column == 1,
            yticksvisible=column == 1,
            leftspinevisible=column == 1,
        )
        hlines!(axis, [0.0]; color=:gray45, linewidth=1.3)
        for (eta, style) in zip(selected.etas, styles)
            points = [selected.shown[(panel.contrast, rho, eta)] for rho in selected.rhos]
            all(ylimits[1] <= p.lower <= p.upper <= ylimits[2] for p in points) ||
                error("display limits clip an interval")
            lines!(
                axis,
                selected.rhos,
                [p.estimate for p in points];
                color=style.color,
                linestyle=style.linestyle,
                linewidth=2.6,
            )
            rangebars!(
                axis,
                selected.rhos,
                [p.lower for p in points],
                [p.upper for p in points];
                color=style.color,
                linewidth=2,
                whiskerwidth=12,
            )
            scatter!(
                axis,
                selected.rhos,
                [p.estimate for p in points];
                color=style.color,
                marker=style.marker,
                markersize=13,
                strokecolor=:white,
                strokewidth=0.7,
            )
        end
        rowgap!(grid, 1, 14)
    end
    Legend(
        fig[0, 1:2],
        [
            [
                LineElement(; color=s.color, linestyle=s.linestyle, linewidth=2.6),
                MarkerElement(; color=s.color, marker=s.marker, markersize=12),
            ] for s in styles
        ],
        [iszero(eta) ? "No turnover (η = 0)" : "η = $eta" for eta in selected.etas],
        "Turnover";
        orientation=:horizontal,
        titleposition=:top,
        framevisible=false,
        labelsize=TICK_FS,
        titlesize=TICK_FS,
        colgap=32,
        patchsize=(38, 18),
        tellwidth=false,
    )
    colgap!(fig.layout, 40)
    rowgap!(fig.layout, 22)
    return fig
end

"""Save the complementarity figure from validated retained analysis inputs."""
function complementarity_figure()
    data = JLD2.load(FIGURE_DATA)
    provenance = manuscript_git_provenance(
        REPO_ROOT;
        sources=(
            @__FILE__,
            "scripts/monte_carlo.jl",
            "scripts/figure_style.jl",
        ),
    )
    validate_analysis_commit(
        provenance, data["analysis_git_commit"]; artifact="assessment-access figure data"
    )
    fig = make_complementarity_figure(data)
    output = joinpath(
        REPO_ROOT, "output", "supplement", "figures", "complementarity_contributions.png"
    )
    mkpath(dirname(output))
    save(output, fig; px_per_unit=2)
    println("Verified all displayed contrasts against retained seed-level data.")
    println("Retained analysis commit: ", data["analysis_git_commit"])
    println("Wrote $output")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    if isempty(ARGS)
        main()
    elseif ARGS == ["--complementarity"]
        complementarity_figure()
    else
        error("usage: main_figure.jl [--complementarity]")
    end
end
