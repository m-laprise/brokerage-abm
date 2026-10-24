"""Render supplementary figures and display values from retained inputs."""

isdefined(@__MODULE__, :ReportingSettings) ||
    include(joinpath(@__DIR__, "..", "reporting_settings.jl"))

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
include(joinpath(@__DIR__, "supp_robustness.jl"))
include(joinpath(@__DIR__, "information_sources.jl"))
using JLD2

publication_theme!()

const OUT = normpath(joinpath(@__DIR__, "..", "..", "output", "supplement", "figures"))
mkpath(OUT)
const PXU = 2.0                       # px_per_unit: ~330+ dpi at printed full-page width
const SPECTRUM_COMPONENTS = AnalysisSettings.SPECTRUM_COMPONENTS
const RHO_COLORS = PUB_RHO_COLORS
const DELTA_COLORS = PUB_DELTA_COLORS
const CENTRALITY_PATH = SupplementRobustness.CENTRALITY
const COMPARISONS_PATH = SupplementRobustness.INPUT
const RIDGE_PATH = normpath(
    joinpath(@__DIR__, "..", "..", "output", "ridge", "ablations", "figure_data.jld2")
)
const CENTRALITY_DATA = JLD2.load(CENTRALITY_PATH)["data"]
const COMPARISONS_DATA = JLD2.load(COMPARISONS_PATH)["data"]
const RIDGE_DATA = JLD2.load(RIDGE_PATH)["figdata"]
COMPARISONS_DATA.metadata["summary_sha256"] == CENTRALITY_DATA.metadata["summary_sha256"] ||
    error("supplement and main structural summaries differ")

const STRUCTURAL_FD = JLD2.load(
    normpath(
        joinpath(
            @__DIR__, "..", "..", "output", "supplement", "structural_figure_data.jld2"
        ),
    ),
)["figdata"]
const DGP_FD = JLD2.load(
    normpath(joinpath(@__DIR__, "..", "..", "output", "supplement", "dgp_figure_data.jld2"))
)["figdata"]
const STRUCTURAL_PATH = normpath(
    joinpath(@__DIR__, "..", "..", "output", "supplement", "structural_figure_data.jld2")
)
const STRUCTURAL_SETTINGS = ReportingSettings.validate_figure(
    STRUCTURAL_FD, ReportingSettings.retained(STRUCTURAL_PATH, STRUCTURAL_FD)
)
function supp_interval(values)
    monte_carlo_interval(values; level=Float64(STRUCTURAL_SETTINGS["interval_level"]))
end
const ENERGY_PERCENT = 100 * DGP_FD["design"]["effective_rank_energy_fraction"]
const ENERGY_LABEL =
    isinteger(ENERGY_PERCENT) ? string(Int(ENERGY_PERCENT)) : string(ENERGY_PERCENT)
const EXPLORATORY = "--exploratory" in ARGS
const REPORTING_PROVENANCE = manuscript_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."));
    sources=(
        "scripts/analysis_settings.jl",
        "scripts/reporting_settings.jl",
        @__FILE__,
        "scripts/paper/supp_robustness.jl",
        "scripts/paper/information_sources.jl",
        "scripts/monte_carlo.jl",
        "scripts/figure_style.jl",
    ),
)
const DGP_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE, DGP_FD["meta"]["analysis_git_commit"]; artifact="DGP figure data"
)
const STRUCTURAL_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE,
    STRUCTURAL_FD["meta"]["analysis_git_commit"];
    artifact="structural figure data",
)
EXPLORATORY ||
    all(
        dataset["meta"]["analysis_source_clean"] == true for
        dataset in (DGP_FD, STRUCTURAL_FD)
    ) ||
    error("supplement figure data were generated from dirty analysis sources")
const COMPARISONS_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE, COMPARISONS_DATA.metadata["analysis_git_commit"]
)
const RIDGE_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE, RIDGE_DATA["meta"]["analysis_git_commit"]
)
RIDGE_DATA["meta"]["analysis_source_clean"] === true || error("Ridge analysis was dirty")
function savefig(fname, fig)
    save(joinpath(OUT, fname), fig; px_per_unit=PXU)
    println("  $fname done")
end

# ── S1: realized types around the latent type curve ──
function type_geometry()
    geometry = DGP_FD["type_geometry"]
    curve = geometry["curve_projection"]
    types = geometry["type_projection"]
    curve_parameter = geometry["curve_parameter"]
    projections = ((1, 2), (1, 3), (2, 3))
    fig = Figure(; size=(1200, 420))
    Legend(
        fig[0, 1:3],
        [
            MarkerElement(; color=:gray55, marker=:circle, markersize=8),
            LineElement(; color=PUB_PRINCIPAL, linewidth=2.4),
        ],
        ["Principal types", "Latent curve"];
        PUB_LEGEND...,
    )
    for (column, (horizontal, vertical)) in enumerate(projections)
        axis = Axis(
            fig[1, column];
            title="$(('A':'C')[column]). Components $horizontal and $vertical",
            xlabel="Component $horizontal",
            ylabel="Component $vertical",
            xticksvisible=false,
            xticklabelsvisible=false,
            yticksvisible=false,
            yticklabelsvisible=false,
            xgridvisible=false,
            ygridvisible=false,
            aspect=DataAspect(),
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
        )
        scatter!(
            axis,
            view(types, horizontal, :),
            view(types, vertical, :);
            color=(:gray35, 0.24),
            markersize=4,
            strokewidth=0,
            label=column == 1 ? "principal types" : nothing,
        )
        lines!(
            axis,
            view(curve, horizontal, :),
            view(curve, vertical, :);
            color=curve_parameter,
            colormap=:viridis,
            colorrange=(0.0, 1.0),
            linewidth=2.4,
            label=column == 1 ? "latent curve" : nothing,
        )
    end
    Colorbar(
        fig[1, 4];
        colormap=:viridis,
        limits=(0.0, 1.0),
        label="Position along latent curve",
        labelsize=LABEL_FS,
        ticklabelsize=TICK_FS,
        width=16,
    )
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 16)
    savefig("type_geometry.png", fig)
end

# ── S2: underlying match-value surfaces ──
function match_value_surfaces()
    heatmaps = DGP_FD["heatmaps"]
    conditions = DGP_FD["heatmap_conditions"]
    matrices = [
        heatmaps["rho=$(condition.rho)|delta=$(condition.delta)"] for
        condition in conditions
    ]
    display_matrices = map(matrices) do matrix
        displayed = copy(matrix)
        for column in axes(displayed, 2), row in axes(displayed, 1)
            row >= column && (displayed[row, column] = NaN32)
        end
        displayed
    end
    color_limit = maximum(
        abs(value) for matrix in display_matrices for value in matrix if isfinite(value)
    )
    fig = Figure(; size=(1200, 850))
    panels = fig[1, 1] = GridLayout()
    for (column, title) in
        enumerate(("Complementarity\nρ = 0", "Mixed\nρ = 0.5", "Individual quality\nρ = 1"))
        Label(panels[1, column + 2], title; fontsize=LABEL_FS, tellwidth=false)
    end
    Label(
        panels[2:3, 1],
        "Individual quality of principal i";
        rotation=pi / 2,
        fontsize=LABEL_FS,
        tellheight=false,
    )
    Label(
        panels[2, 2],
        "Low difficulty (δ = 0)";
        rotation=pi / 2,
        fontsize=LABEL_FS,
        tellheight=false,
    )
    Label(
        panels[3, 2],
        "High difficulty (δ = 1)";
        rotation=pi / 2,
        fontsize=LABEL_FS,
        tellheight=false,
    )
    Label(
        panels[4, 3:5],
        "Individual quality of principal j";
        fontsize=LABEL_FS,
        tellwidth=false,
    )

    panel_positions = ((2, 3), (2, 4), (2, 5), (3, 3), (3, 4))
    for (index, ((row, column), matrix)) in
        enumerate(zip(panel_positions, display_matrices))
        axis = Axis(
            panels[row, column];
            title=string(('A':'E')[index]),
            titlesize=TITLE_FS,
            titlegap=4,
            xticksvisible=false,
            xticklabelsvisible=false,
            yticksvisible=false,
            yticklabelsvisible=false,
            xgridvisible=false,
            ygridvisible=false,
            leftspinevisible=false,
            bottomspinevisible=false,
            halign=:left,
            width=320,
            height=320,
            aspect=DataAspect(),
        )
        heatmap!(
            axis,
            matrix;
            colormap=:vik,
            colorrange=(-color_limit, color_limit),
            nan_color=:transparent,
            rasterize=true,
        )
    end
    Label(
        panels[3, 5],
        "Difficulty has\nno effect at ρ = 1";
        fontsize=LABEL_FS,
        color=:gray35,
        tellwidth=false,
    )
    Colorbar(
        panels[2:3, 6];
        colormap=:vik,
        limits=(-color_limit, color_limit),
        label="Underlying match value (centered)",
        labelsize=LABEL_FS,
        ticklabelsize=TICK_FS,
        width=18,
    )
    for column in 3:5
        colsize!(panels, column, Auto(1))
    end
    rowsize!(panels, 2, Auto(1))
    rowsize!(panels, 3, Auto(1))
    colgap!(panels, 14)
    colgap!(panels, 1, 6)
    colgap!(panels, 2, 8)
    rowgap!(panels, 12)
    resize_to_layout!(fig)
    savefig("match_value_surfaces.png", fig)
end

function spectrum_interval(matrix, component)
    interval = supp_interval(view(matrix, component, :))
    return (
        mean=max(interval.mean, 0.0),
        lower=max(interval.lower, 0.0),
        upper=max(interval.upper, 0.0),
    )
end

# ── S3: singular spectra and effective dimension ──
function effective_dimensionality()
    displayed_rhos = [0.0, 0.15, 0.5, 0.85, 1.0]
    components = 1:SPECTRUM_COMPONENTS
    fig = Figure(; size=(1200, 520))
    spectrum_axis = Axis(
        fig[1, 1];
        title="A. Singular-value spectrum",
        xlabel="Component rank (largest first)",
        ylabel="Relative magnitude (σₖ / σ₁)",
        yticks=0.0:0.25:1.0,
        limits=((0.5, SPECTRUM_COMPONENTS + 0.5), (-0.015, 1.05)),
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
    )
    for rho in displayed_rhos
        spectrum = DGP_FD["spectra"][string(rho)]
        summaries = [spectrum_interval(spectrum, component) for component in components]
        band!(
            spectrum_axis,
            components,
            [summary.lower for summary in summaries],
            [summary.upper for summary in summaries];
            color=(RHO_COLORS[rho], 0.16),
        )
        scatterlines!(
            spectrum_axis,
            components,
            [summary.mean for summary in summaries];
            color=RHO_COLORS[rho],
            marker=PUB_RHO_MARKERS[rho],
            linewidth=2.2,
            markersize=5,
            label="ρ = $rho",
        )
    end
    composition_legend!(fig[2, 1], displayed_rhos; nbanks=3)

    rank_axis = Axis(
        fig[1, 2];
        title="B. Effective dimensionality",
        xlabel=PUB_RHO_LABEL,
        ylabel="Components explaining\n$ENERGY_LABEL% of variation",
        xticks=[0, 0.5, 1],
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
    )
    rows = DGP_FD["conditions"]
    for delta in DGP_FD["delta_values"]
        points = sort(
            [
                let interval = supp_interval(row["rank90_seed_values"])
                    (
                        rho=row["rho"],
                        mean=interval.mean,
                        lower=interval.lower,
                        upper=interval.upper,
                    )
                end for row in rows if row["delta"] == delta && row["rho"] < 1.0
            ];
            by=point -> point.rho,
        )
        rangebars!(
            rank_axis,
            [point.rho for point in points],
            [point.lower for point in points],
            [point.upper for point in points];
            color=(DELTA_COLORS[delta], 0.72),
            linewidth=1.2,
            whiskerwidth=8,
        )
        scatterlines!(
            rank_axis,
            [point.rho for point in points],
            [point.mean for point in points];
            color=DELTA_COLORS[delta],
            marker=PUB_DELTA_MARKERS[delta],
            linewidth=2.0,
            markersize=9,
            label="δ = $delta",
        )
    end
    boundary = only(row for row in rows if row["rho"] == 1.0)
    boundary_interval = supp_interval(boundary["rank90_seed_values"])
    rangebars!(
        rank_axis,
        [1.0],
        [boundary_interval.lower],
        [boundary_interval.upper];
        color=:gray25,
        linewidth=1.2,
        whiskerwidth=8,
    )
    scatter!(
        rank_axis,
        [1.0],
        [boundary_interval.mean];
        color=:gray25,
        marker=:diamond,
        markersize=11,
        label="ρ = 1 boundary",
    )
    difficulty_legend!(fig[2, 2], DGP_FD["delta_values"]; nbanks=3)
    colgap!(fig.layout, 38)
    rowgap!(fig.layout, 22)
    savefig("effective_dimensionality.png", fig)
end

foreach(f -> f(), (type_geometry, match_value_surfaces, effective_dimensionality))
SupplementRobustness.render_conditions(
    STRUCTURAL_FD,
    CENTRALITY_DATA,
    joinpath(OUT, "alternative_measures_composition_turnover.png"),
)
SupplementRobustness.render_experiments(
    COMPARISONS_DATA, joinpath(OUT, "alternative_measures_experiments.png")
)
InformationSources.render_supplement(
    RIDGE_DATA, joinpath(OUT, "learning_across_difficulties.png")
)
# emit the display-convention keys quoted by the supplement captions
content = sprint() do io
    println(
        io,
        "% supp_figmeta.tex: generated by scripts/paper/supp_figures.jl. Do not edit by hand.",
    )
    println(io, "\\pvDefine{suppDgpSeeds}{$(length(DGP_FD["seeds"]))}")
    println(io, "\\pvDefine{suppDgpN}{$(DGP_FD["design"]["N"])}")
    println(io, "\\pvDefine{suppDgpHeatmapN}{$(DGP_FD["design"]["heatmap_display_N"])}")
    println(io, "\\pvDefine{suppDgpHeatmapSeed}{$(DGP_FD["heatmap_seed"])}")
    println(io, "\\pvDefine{suppDgpDelta}{$(DGP_FD["baseline_delta"])}")
    println(io, "\\pvDefine{suppDgpSpectrumComponents}{$SPECTRUM_COMPONENTS}")
    println(io, "\\pvDefine{suppDgpRankEnergyPercent}{$ENERGY_LABEL}")
end
write_if_changed(joinpath(dirname(OUT), "figmeta.tex"), content)
record_publication_stage(
    REPORTING_PROVENANCE;
    stage="scripts/paper/supp_figures.jl",
    inputs=(
        CENTRALITY_PATH,
        COMPARISONS_PATH,
        RIDGE_PATH,
        SupplementRobustness.SERVICES,
        ReportingSettings.retained_dependencies(STRUCTURAL_FD)...,
        joinpath.(dirname(OUT), ("structural_figure_data.jld2", "dgp_figure_data.jld2"))...,
    ),
    outputs=(
        joinpath(dirname(OUT), "figmeta.tex"),
        joinpath.(
            OUT,
            (
                "type_geometry.png",
                "match_value_surfaces.png",
                "effective_dimensionality.png",
                "alternative_measures_composition_turnover.png",
                "alternative_measures_experiments.png",
                "learning_across_difficulties.png",
            ),
        )...,
    ),
    analysis_commits=Dict(
        "dgp data analysis commit" => DGP_ANALYSIS_COMMIT,
        "structural data analysis commit" => STRUCTURAL_ANALYSIS_COMMIT,
        "experimental data analysis commit" => COMPARISONS_ANALYSIS_COMMIT,
        "ridge data analysis commit" => RIDGE_ANALYSIS_COMMIT,
    ),
    settings=Dict(
        "exploratory" => EXPLORATORY,
        "interval_level" => STRUCTURAL_SETTINGS["interval_level"],
    ),
)
println("supplement figures done (+ supp_figmeta.tex)")
