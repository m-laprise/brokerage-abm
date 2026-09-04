"""
    scripts/paper/supp_figures.jl

Supplementary figures (S1-S3). Standalone twin of scripts/paper/figures.jl:
reads ONLY output/supplement/figdata.jld2 (written by scripts/paper/supp_figdata.jl on
the cluster), so the supplement renders locally with no access to the sweep, and
independently of the results-section figures. CairoMakie only; no simulation; no
hard-coded results (literal constants are display conventions only). Outputs
print-resolution PNGs to output/supplement/figs/ and the display-convention keys to
output/supplement/figmeta.tex.

The main results use broker betweenness centrality. These supplementary figures
reproduce the same analyses with the broker's two other ego-network measures,
Burt's aggregate constraint and Burt's effective size:

  S1  constraint and effective size across the rho x delta grid, line per delta
      (the matching-grid structural panel, for each alternative measure)
  S2  each measure over time at baseline (left) and against access fraction
      across regimes (right); constraint top, effective size bottom (the
      position analysis, without the access-fraction time series)
  S3  rank-correlation difference and output gap against each measure, colored by rho
      (the advantage analysis, with the alternative measures in place of betweenness)

Like betweenness, both measures are recomputed only every network_measure_interval
(20) periods, so the time panels in S2 plot the measurement periods.

Usage: julia --project --threads=auto scripts/paper/supp_figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
using JLD2
using Statistics: mean

const OUT = normpath(joinpath(@__DIR__, "..", "..", "output", "supplement", "figs"))
mkpath(OUT)
const PXU = 2.0                       # px_per_unit: ~330+ dpi at printed full-page width
# Display conventions, quoted in the supplement captions via the keys emitted to
# output/supplement/figmeta.tex. Identical to figures.jl: MEASINT matches the
# generation-time network_measure_interval; TSTART is the end of burn-in.
const ROLLW = 5                       # rolling-mean window, in observations
const MEASINT = 20                    # network-measure interval, periods
const TSTART = 30                     # displayed axes start here; data never cut
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
# the two alternative structural measures, with the keys used in the extract and
# the labels printed on the panels
const CONSTR = ("constraint", "Broker Burt constraint")
const EFFS = ("effective_size", "Broker effective size")
# the cell-level keys differ from the series keys (extract names them shorter)
const CELLKEY = Dict("constraint" => "constraint", "effective_size" => "effsize")

const FD = JLD2.load(
    normpath(joinpath(@__DIR__, "..", "..", "output", "supplement", "figdata.jld2"))
)["figdata"]
const REPORTING_PROVENANCE = reporting_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."))
)
const DATA_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE,
    FD["meta"]["analysis_git_commit"];
    artifact="supplement figure data",
)
FD["meta"]["analysis_source_clean"] == true ||
    error("supplement figure data were extracted from dirty analysis sources")
const PER = FD["period"]
const SER = FD["series"]
const SER_SEEDS = FD["series_seed_values"]
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

function draw_interval_series!(axis, series; color=COL_GAP)
    x, estimate, lower, upper = series
    band!(axis, x, lower, upper; color=(color, 0.16))
    scatterlines!(axis, x, estimate; color, linewidth=2.2, markersize=6)
    return nothing
end

# ── S1: each measure vs rho across the grid, one line per delta ──
function supp_S1_grid_lines()
    gc = FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gc]))
    boundary_cells = [c for c in gc if c["rho"] == 1.0]
    length(unique(c["rel"] for c in boundary_cells)) == 1 ||
        error("rho = 1 grid coordinates do not share one effective realization")
    boundary_cell = first(boundary_cells)
    fig = Figure(; size=(1180, 470))
    for (ci, (key, lab)) in enumerate((CONSTR, EFFS))
        ck = CELLKEY[key]
        ax = Axis(
            fig[1, ci];
            title=lab,
            xlabel="ρ (complementarity vs quality)",
            ylabel=ci == 1 ? "late-window mean" : "",
            xticks=[0, 0.3, 0.5, 0.7, 0.85, 1],
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
        ci == 1 && axislegend(ax, "Difficulty"; position=:rt, LEG_KW...)
    end
    colgap!(fig.layout, 16)
    savefig("supp_S1_grid_lines.png", fig)
end

# ── S2: measure over time at baseline (left) + vs access fraction across regimes
#    (right); constraint (top), effective size (bottom) ──
function supp_S2_position()
    cells = FD["oat_cells"]
    ac = [c["access"] for c in cells]
    fig = Figure(; size=(1180, 860))
    for (ri, (key, lab)) in enumerate(((CONSTR), (EFFS)))
        # left: the measure over time at baseline (no access-fraction series)
        series = measseries(key)
        x = series[1]
        axl = Axis(
            fig[ri, 1];
            title=ri == 1 ? "Over time, at baseline" : "",
            xlabel=ri == 2 ? "period" : "",
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
            title=ri == 1 ? "Across regimes" : "",
            xlabel=ri == 2 ? "access fraction" : "",
            ylabel=lab,
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
        )
        scatter!(axr, ac, yv; color=:black, markersize=13)
    end
    colgap!(fig.layout, 16);
    rowgap!(fig.layout, 12)
    savefig("supp_S2_position.png", fig)
end

# ── S3: rank-correlation difference and output gap against each measure ──
function supp_S3_advantage()
    bc = FD["regime_cells"]
    rho = [c["rho"] for c in bc]
    xs = [(CONSTR[2], [c["constraint"] for c in bc]), (EFFS[2], [c["effsize"] for c in bc])]
    ys = [
        ("Rank-correlation difference", [c["rankgap"] for c in bc]),
        ("Output gap q", [c["qgap"] for c in bc]),
    ]
    fig = Figure(; size=(1150, 940))
    for (ri, (ylab, yv)) in enumerate(ys), (ci, (xlab, xv)) in enumerate(xs)
        ax = Axis(
            fig[ri, ci];
            xlabel=ri == 2 ? xlab : "",
            ylabel=ci == 1 ? ylab : "",
            title=ri == 1 ? xlab : "",
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
                markersize=MARKER_SIZE,
                strokewidth=0.3,
                strokecolor=:gray30,
            )
        end
        if ri == 1 && ci == 2
            rvs = sort(unique(rho))
            els = [
                MarkerElement(;
                    marker=:circle, color=RHO_COLORS[v], markersize=MARKER_SIZE
                ) for v in rvs
            ]
            axislegend(ax, els, ["ρ = $v" for v in rvs]; position=:rt, LEG_KW...)
        end
    end
    colgap!(fig.layout, 16);
    rowgap!(fig.layout, 12)
    savefig("supp_S3_advantage.png", fig)
end

foreach(
    function_name -> function_name(),
    (supp_S1_grid_lines, supp_S2_position, supp_S3_advantage),
)
# emit the display-convention keys quoted by the supplement captions
open(
    normpath(joinpath(@__DIR__, "..", "..", "output", "supplement", "figmeta.tex")), "w"
) do io
    println(
        io,
        "% supp_figmeta.tex: generated by scripts/paper/supp_figures.jl. Do not edit by hand.",
    )
    println(io, "% Data analysis commit: $DATA_ANALYSIS_COMMIT")
    println(io, "% Rendering commit: $(REPORTING_PROVENANCE.commit)")
    println(
        io,
        "% Display conventions used to render output/supplement/figs/, quoted in captions via \\pv keys.",
    )
    println(io, "\\pvDefine{suppRollWin}{$ROLLW}")
    println(io, "\\pvDefine{suppMeasInterval}{$MEASINT}")
    println(io, "\\pvDefine{suppAxisStart}{$TSTART}")
end
println("supplement figures done (+ supp_figmeta.tex)")
