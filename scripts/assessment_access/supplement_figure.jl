"""
Render the supplementary assessment-access comparison across matching composition.

Saved contrasts are checked against retained seed-level observations.
The caption is in paper/supplement.tex.

Usage: julia --project --threads=auto scripts/assessment_access/supplement_figure.jl
Output: output/supplement/figures/complementarity_contributions.png
"""


include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
using JLD2

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIGURE_DATA = joinpath(REPO_ROOT, "output", "assessment_access", "figure_data.jld2")
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

"""Select the common composition-turnover grid and verify every displayed contrast."""
function complementarity_estimates(data)
    data["analysis_source_clean"] === true || error("analysis sources were dirty")
    summaries = interval_index(data["summary_rows"])
    contrasts = interval_index(data["contrast_rows"])
    metric = "net_output_per_principal"
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
    display = interval_axis(bounds; upper_padding=0.25)
    ylimits = display.limits
    ticks = collect(display.ticks)
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
                "Change in net output to principals\nper principal"
            else
                ""
            end,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticks=(selected.rhos, rho_labels),
            xticklabelsize=TICK_FS,
            yticks=ticks,
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
    isempty(ARGS) || error("usage: supplement_figure.jl")
    complementarity_figure()
end
