"""
    scripts/paper/supp_figures.jl

Render Supplementary Figures S1--S3 and S5--S7 from retained figure-input datasets.
Figures S1--S3 describe the matching-function data-generating process. Figures S5--S7
reproduce the main structural analyses with Burt's aggregate constraint and
effective size:

  S1  realized principal types and their latent curve in three dimensions
  S2  conditional match-value surfaces ordered by realized general quality
  S3  normalized singular spectra and 90%-energy effective dimension
  S5  constraint and effective size across the rho x delta grid, line per delta
      (the matching-grid structural panel, for each alternative measure)
  S6  each measure over time at baseline (left) and against access fraction
      across regimes (right); constraint top, effective size bottom (the
      position analysis, without the access-fraction time series)
  S7  rank-correlation difference and output gap against each measure, colored by rho
      (the advantage analysis, with the alternative measures in place of betweenness)

Figure S4 is rendered by scripts/assessment_access/supplement_figure.jl.
This script reads only `output/supplement/dgp_figure_data.jld2` and
`output/supplement/structural_figure_data.jld2`. It performs no simulation and writes
print-resolution PNGs plus the display-convention keys used by the captions.

Usage: julia --project --threads=auto scripts/paper/supp_figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
using JLD2

publication_theme!()

const OUT = normpath(joinpath(@__DIR__, "..", "..", "output", "supplement", "figures"))
mkpath(OUT)
const PXU = 2.0                       # px_per_unit: ~330+ dpi at printed full-page width
# Display conventions, quoted in the supplement captions via the keys emitted to
# output/supplement/figmeta.tex. Identical to figures.jl: MEASINT matches the
# generation-time network_measure_interval; TSTART is the end of burn-in.
const ROLLW = 5                       # rolling-mean window, in observations
const MEASINT = 20                    # network-measure interval, periods
const TSTART = 30                     # displayed axes start here; data never cut
const SPECTRUM_COMPONENTS = 25
const RHO_COLORS = PUB_RHO_COLORS
const DELTA_COLORS = PUB_DELTA_COLORS
# the two alternative structural measures, with the keys used in the extract and
# the labels printed on the panels
const CONSTR = ("constraint", "Broker constraint")
const EFFS = ("effective_size", "Broker effective size")
# the cell-level keys differ from the series keys (extract names them shorter)
const CELLKEY = Dict("constraint" => "constraint", "effective_size" => "effsize")

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
const EXPLORATORY = "--exploratory" in ARGS
const REPORTING_PROVENANCE = manuscript_git_provenance(
    normpath(joinpath(@__DIR__, "..", "..")); sources=(@__FILE__, "scripts/monte_carlo.jl", "scripts/figure_style.jl",),
)
const DGP_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE, DGP_FD["meta"]["analysis_git_commit"]; artifact="DGP figure data"
)
const STRUCTURAL_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE,
    STRUCTURAL_FD["meta"]["analysis_git_commit"];
    artifact="structural figure data",
)
EXPLORATORY || all(
    dataset["meta"]["analysis_source_clean"] == true for dataset in (DGP_FD, STRUCTURAL_FD)
) || error("supplement figure data were generated from dirty analysis sources")
const PER = STRUCTURAL_FD["period"]
const SER = STRUCTURAL_FD["series"]
const SER_SEEDS = STRUCTURAL_FD["series_seed_values"]
const TEND = maximum(PER)
const TIME_TICK_STEP = TEND <= 250 ? 50 : 100
const TIME_TICKS = TIME_TICK_STEP:TIME_TICK_STEP:TEND
const MARKER_SIZE = 10
function savefig(fname, fig)
    (save(joinpath(OUT, fname), fig; px_per_unit=PXU); println("  $fname done"))
end

# Measurement-period indices and the pointwise Monte Carlo summary of `key`.
# Each seed is smoothed before summarizing across seeds.
measidx() = [i for i in eachindex(PER) if PER[i] % MEASINT == 0]
function measseries(key)
    indices = measidx()
    raw = SER_SEEDS[key][indices, :]
    smoothed = reduce(
        hcat, (rolling_mean(view(raw, :, seed_index), ROLLW) for seed_index in axes(raw, 2))
    )
    summaries = [
        monte_carlo_interval(view(smoothed, period_index, :)) for
        period_index in axes(smoothed, 1)
    ]
    return (
        PER[indices],
        [summary.mean for summary in summaries],
        [summary.lower for summary in summaries],
        [summary.upper for summary in summaries],
    )
end
# Autoscaled y-limits over the displayed window only, padded at both ends.
function ywin(curves...)
    v = Float64[]
    for (xs, _, lower, upper) in curves
        for bound in (lower, upper)
            append!(
                v,
                [bound[i] for i in eachindex(bound) if xs[i] >= TSTART && !isnan(bound[i])],
            )
        end
    end
    isempty(v) && return nothing
    lo, hi = minimum(v), maximum(v);
    pad = 0.06 * (hi - lo + eps())
    (lo - pad, hi + pad)
end

function draw_interval_series!(axis, series; color=PUB_CENTRALITY)
    x, estimate, lower, upper = series
    band!(axis, x, lower, upper; color=(color, 0.16))
    scatterlines!(axis, x, estimate; color, linewidth=2.2, markersize=6)
    return nothing
end

# ── S1: realized types around the latent type curve ──
function type_geometry()
    geometry = DGP_FD["type_geometry"]
    curve = geometry["curve_projection"]
    types = geometry["type_projection"]
    curve_parameter = geometry["curve_parameter"]
    projections = ((1, 2), (1, 3), (2, 3))
    fig = Figure(; size=(1200, 420))
    Legend(fig[0, 1:3],
        [MarkerElement(; color=:gray55, marker=:circle, markersize=8),
         LineElement(; color=PUB_PRINCIPAL, linewidth=2.4)],
        ["Principal types", "Latent curve"]; PUB_LEGEND...)
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

# ── S2: conditional match-value surfaces ──
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
        enumerate(("Complementarity\nρ = 0", "Mixed\nρ = 0.5", "General quality\nρ = 1"))
        Label(panels[1, column + 2], title; fontsize=LABEL_FS, tellwidth=false)
    end
    Label(panels[2:3, 1], "General quality of principal i";
        rotation=pi / 2, fontsize=LABEL_FS, tellheight=false)
    Label(panels[2, 2], "Low difficulty (δ = 0)";
        rotation=pi / 2, fontsize=LABEL_FS, tellheight=false)
    Label(panels[3, 2], "High difficulty (δ = 1)";
        rotation=pi / 2, fontsize=LABEL_FS, tellheight=false)
    Label(panels[4, 3:5], "General quality of principal j";
        fontsize=LABEL_FS, tellwidth=false)

    panel_positions = ((2, 3), (2, 4), (2, 5), (3, 3), (3, 4))
    for (index, ((row, column), matrix)) in enumerate(zip(panel_positions, display_matrices))
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
    Label(panels[3, 5], "Difficulty has\nno effect at ρ = 1";
        fontsize=LABEL_FS, color=:gray35, tellwidth=false)
    Colorbar(
        panels[2:3, 6];
        colormap=:vik,
        limits=(-color_limit, color_limit),
        label="Expected match output (centered)",
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
    interval = monte_carlo_interval(view(matrix, component, :))
    return (
        mean=max(interval.mean, 0.0),
        lower=max(interval.lower, 0.0),
        upper=max(interval.upper, 0.0),
    )
end

# ── S3: singular spectra and 90%-energy effective dimension ──
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
        ylabel="Components explaining\n90% of variation",
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
                let interval = monte_carlo_interval(row["rank90_seed_values"])
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
    boundary_interval = monte_carlo_interval(boundary["rank90_seed_values"])
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

# ── S5: each measure vs rho across the grid, one line per delta ──
function alternative_measures_grid()
    gc = STRUCTURAL_FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gc]))
    boundary_cells = [c for c in gc if c["rho"] == 1.0]
    length(unique(c["rel"] for c in boundary_cells)) == 1 ||
        error("rho = 1 grid coordinates do not share one effective realization")
    boundary_cell = first(boundary_cells)
    fig = Figure(; size=(1200, 500))
    difficulty_legend!(fig[0, 1:2], dls)
    for (ci, (key, lab)) in enumerate((CONSTR, EFFS))
        ck = CELLKEY[key]
        ax = Axis(
            fig[1, ci];
            title="$(('A':'B')[ci]). $lab",
            xlabel=PUB_RHO_LABEL,
            xticks=[0, 0.5, 1],
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
        )
        for d in dls
            pts = sort(
                [
                    let interval = monte_carlo_interval(c["seed_values"][ck])
                        (
                            rho=c["rho"],
                            mean=interval.mean,
                            lower=interval.lower,
                            upper=interval.upper,
                        )
                    end for c in gc if c["delta"] == d && c["rho"] < 1.0
                ];
                by=point -> point.rho,
            )
            rangebars!(
                ax,
                [point.rho for point in pts],
                [point.lower for point in pts],
                [point.upper for point in pts];
                color=(DELTA_COLORS[d], 0.72),
                linewidth=1.2,
                whiskerwidth=8,
            )
            scatterlines!(
                ax,
                [point.rho for point in pts],
                [point.mean for point in pts];
                color=DELTA_COLORS[d],
                marker=PUB_DELTA_MARKERS[d],
                linewidth=2.0,
                markersize=10,
                strokewidth=0.4,
                strokecolor=:gray30,
                label="δ = $d",
            )
        end
        boundary_interval = monte_carlo_interval(boundary_cell["seed_values"][ck])
        rangebars!(
            ax,
            [1.0],
            [boundary_interval.lower],
            [boundary_interval.upper];
            color=(:gray25, 0.8),
            linewidth=1.2,
            whiskerwidth=8,
        )
        scatter!(
            ax,
            [1.0],
            [boundary_interval.mean];
            color=:gray25,
            marker=:diamond,
            markersize=11,
            strokewidth=0.4,
            strokecolor=:gray20,
            label="ρ = 1 boundary",
        )
    end
    colgap!(fig.layout, 40)
    rowgap!(fig.layout, 24)
    savefig("alternative_measures_grid.png", fig)
end

# ── S6: measure over time at baseline (left) + vs access fraction across regimes
#    (right); constraint (top), effective size (bottom) ──
function alternative_measures_position()
    cells = STRUCTURAL_FD["oat_cells"]
    ac = [c["access"] for c in cells]
    fig = Figure(; size=(1200, 650))
    for (ri, (key, lab)) in enumerate(((CONSTR), (EFFS)))
        # left: the measure over time at baseline (no access-fraction series)
        series = measseries(key)
        axl = Axis(
            fig[ri, 1];
            title="$(ri == 1 ? 'A' : 'C'). Baseline dynamics",
            xlabel="Period",
            ylabel=lab,
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
            xticks=TIME_TICKS,
            limits=((TSTART, TEND + 1), ywin(series)),
        )
        draw_interval_series!(axl, series)
        # right: across regimes, access fraction (x) vs the measure (y)
        yv = [c[CELLKEY[key]] for c in cells]
        axr = Axis(
            fig[ri, 2];
            title="$(ri == 1 ? 'B' : 'D'). Across regimes",
            xlabel="Access fraction",
            ylabel=lab,
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
        )
        scatter!(axr, ac, yv; color=(PUB_CENTRALITY, 0.85), markersize=11,
            strokecolor=:white, strokewidth=0.6)
    end
    colgap!(fig.layout, 34)
    rowgap!(fig.layout, 26)
    savefig("alternative_measures_position.png", fig)
end

# ── S7: rank-correlation difference and output gap against each measure ──
function alternative_measures_advantage()
    bc = STRUCTURAL_FD["regime_cells"]
    rho = [c["rho"] for c in bc]
    xs = [(CONSTR[2], [c["constraint"] for c in bc]), (EFFS[2], [c["effsize"] for c in bc])]
    ys = [
        ("Rank correlation:\nbroker minus principal", [c["rankgap"] for c in bc]),
        ("Match output:\nbroker minus self-search", [c["qgap"] for c in bc]),
    ]
    titles = ["A. Ranking advantage" "B. Ranking advantage";
              "C. Match-output difference" "D. Match-output difference"]
    fig = Figure(; size=(1200, 860))
    composition_legend!(fig[0, 1:2], sort(unique(rho)))
    for (ri, (ylab, yv)) in enumerate(ys), (ci, (xlab, xv)) in enumerate(xs)
        ax = Axis(
            fig[ri, ci];
            xlabel=xlab,
            ylabel=ci == 1 ? ylab : "",
            title=titles[ri, ci],
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
        )
        for rv in sort(unique(rho))
            mm = rho .== rv
            scatter!(
                ax,
                xv[mm],
                yv[mm];
                color=(RHO_COLORS[rv], 0.8),
                marker=PUB_RHO_MARKERS[rv],
                markersize=MARKER_SIZE,
                strokewidth=0.3,
                strokecolor=:gray30,
            )
        end
    end
    colgap!(fig.layout, 34)
    rowgap!(fig.layout, 24)
    savefig("alternative_measures_advantage.png", fig)
end

foreach(
    function_name -> function_name(),
    (
        type_geometry,
        match_value_surfaces,
        effective_dimensionality,
        alternative_measures_grid,
        alternative_measures_position,
        alternative_measures_advantage,
    ),
)
# emit the display-convention keys quoted by the supplement captions
open(
    normpath(joinpath(@__DIR__, "..", "..", "output", "supplement", "figmeta.tex")), "w"
) do io
    println(
        io,
        "% supp_figmeta.tex: generated by scripts/paper/supp_figures.jl. Do not edit by hand.",
    )
    println(io, "% DGP data analysis commit: $DGP_ANALYSIS_COMMIT")
    println(io, "% Structural data analysis commit: $STRUCTURAL_ANALYSIS_COMMIT")
    println(io, "% Rendering commit: $(REPORTING_PROVENANCE.commit)")
    write_source_provenance(io, REPORTING_PROVENANCE)
    println(io, "% Exploratory DGP artifact: $(get(DGP_FD["meta"], "exploratory", false))")
    println(io, "% Exploratory rendering: $EXPLORATORY")
    println(
        io,
        "% Display conventions used to render output/supplement/figures/, quoted in captions via \\pv keys.",
    )
    println(io, "\\pvDefine{suppRollWin}{$ROLLW}")
    println(io, "\\pvDefine{suppMeasInterval}{$MEASINT}")
    println(io, "\\pvDefine{suppAxisStart}{$TSTART}")
    println(io, "\\pvDefine{suppRegimeN}{$(length(STRUCTURAL_FD["regime_cells"]))}")
    println(io, "\\pvDefine{suppOatN}{$(length(STRUCTURAL_FD["oat_cells"]))}")
    println(io, "\\pvDefine{suppDgpSeeds}{$(length(DGP_FD["seeds"]))}")
    println(io, "\\pvDefine{suppDgpN}{$(DGP_FD["design"]["N"])}")
    println(io, "\\pvDefine{suppDgpHeatmapN}{$(DGP_FD["design"]["heatmap_display_N"])}")
    println(io, "\\pvDefine{suppDgpHeatmapSeed}{$(DGP_FD["heatmap_seed"])}")
    println(io, "\\pvDefine{suppDgpDelta}{$(DGP_FD["baseline_delta"])}")
    println(io, "\\pvDefine{suppDgpSpectrumComponents}{$SPECTRUM_COMPONENTS}")
    println(io, "\\pvDefine{suppDgpRankEnergyPercent}{90}")
end
println("supplement figures done (+ supp_figmeta.tex)")
