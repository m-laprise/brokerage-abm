"""
    scripts/paper/supp_figures.jl

Supplementary figures (S1-S4). Standalone twin of scripts/paper/figures.jl:
reads ONLY paper/supp_figdata.jld2 (written by scripts/paper/supp_figdata.jl on
the cluster), so the supplement renders locally with no access to the sweep, and
independently of the results-section figures. CairoMakie only; no simulation; no
hard-coded results (literal constants are display conventions only). Outputs
print-resolution PNGs to paper/supp_figs/ and the display-convention keys to
paper/supp_figmeta.tex.

The main results use broker betweenness centrality. These supplementary figures
reproduce the same analyses with the broker's two other ego-network measures,
Burt's aggregate constraint and Burt's effective size:

  S1  constraint and effective size across the rho x delta grid, line per delta
      (the structural panel of Figure 1, for each alternative measure)
  S2  each measure over time at baseline (left) and against access fraction
      across regimes (right); constraint top, effective size bottom (Figure 2,
      without the access-fraction series on the time panels)
  S3  rank-correlation gap and output gap against each measure, colored by rho
      (Figure 3, with constraint and effective size in place of betweenness)
  S4  each measure over time at baseline (the betweenness panel of Figure 4,
      one panel per measure)

Like betweenness, both measures are recomputed only every network_measure_interval
(20) periods, so the time panels (S2 left, S4) plot the measurement periods.

Usage: julia --project scripts/paper/supp_figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
using JLD2
using Statistics: mean

const OUT = normpath(joinpath(@__DIR__, "..", "..", "paper", "supp_figs"))
mkpath(OUT)
const PXU = 2.0                       # px_per_unit: ~330+ dpi at printed full-page width
# Display conventions, quoted in the supplement captions via the keys emitted to
# paper/supp_figmeta.tex. Identical to figures.jl: MEASINT matches the
# generation-time network_measure_interval; TSTART is the end of burn-in.
const ROLLW = 5                       # rolling-mean window, in observations
const MEASINT = 20                    # network-measure interval, periods
const TSTART = 30                     # displayed axes start here; data never cut
const RHO_COLORS = Dict(
    0.0 => :seagreen,
    0.3 => :mediumaquamarine,
    0.5 => :goldenrod,
    0.7 => :darkorange,
    1.0 => :firebrick,
)
const DELTA_COLORS = Dict(0.0 => :steelblue, 0.5 => :goldenrod, 0.75 => :firebrick)
# the two alternative structural measures, with the keys used in the extract and
# the labels printed on the panels
const CONSTR = ("constraint", "Broker Burt constraint")
const EFFS = ("effective_size", "Broker effective size")
# the cell-level keys differ from the series keys (extract names them shorter)
const CELLKEY = Dict("constraint" => "constraint", "effective_size" => "effsize")

const FD = JLD2.load(
    normpath(joinpath(@__DIR__, "..", "..", "paper", "supp_figdata.jld2"))
)["figdata"]
const PER = FD["period"]
const SER = FD["series"]
const MARKER_SIZE = 10
function savefig(fname, fig)
    (save(joinpath(OUT, fname), fig; px_per_unit=PXU); println("  $fname done"))
end

# measurement-period indices and the smoothed series of `key`
measidx() = [i for i in eachindex(PER) if PER[i] % MEASINT == 0]
measseries(key) = (mi=measidx(); (PER[mi], rolling_mean(SER[key][mi], ROLLW)))
# autoscaled y-limits over the DISPLAYED window only (x >= TSTART), padded both ends
function ywin(curves...)
    v = Float64[]
    for (xs, ys) in curves
        append!(v, [ys[i] for i in eachindex(ys) if xs[i] >= TSTART && !isnan(ys[i])])
    end
    isempty(v) && return nothing
    lo, hi = minimum(v), maximum(v);
    pad = 0.06 * (hi - lo + eps())
    (lo - pad, hi + pad)
end

# ── S1: each measure vs rho across the grid, one line per delta ──
function supp_S1_grid_lines()
    gc = FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gc]))
    fig = Figure(; size=(1180, 470))
    for (ci, (key, lab)) in enumerate((CONSTR, EFFS))
        ck = CELLKEY[key]
        ax = Axis(
            fig[1, ci];
            title=lab,
            xlabel="ρ (complementarity vs quality)",
            ylabel=ci == 1 ? "late-window mean" : "",
            xticks=[0, 0.3, 0.5, 0.7, 1],
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
        )
        for d in dls
            pts = sort([(c["rho"], c[ck]) for c in gc if c["delta"] == d]; by=first)
            scatterlines!(
                ax,
                first.(pts),
                last.(pts);
                color=DELTA_COLORS[d],
                linewidth=2.0,
                markersize=10,
                strokewidth=0.4,
                strokecolor=:gray30,
                label="δ = $d",
            )
        end
        ci == 1 && axislegend(ax, "Regime gain"; position=:rt, LEG_KW...)
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
        x, y = measseries(key)
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
            limits=((TSTART, 201), ywin((x, y))),
        )
        scatterlines!(axl, x, y; color=COL_GAP, linewidth=2.2, markersize=6)
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

# ── S3: rank gap and output gap against each measure, colored by rho ──
function supp_S3_advantage()
    bc = FD["regime_cells"]
    rho = [c["rho"] for c in bc]
    xs = [(CONSTR[2], [c["constraint"] for c in bc]), (EFFS[2], [c["effsize"] for c in bc])]
    ys = [
        ("Rank correlation gap", [c["rankgap"] for c in bc]),
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

# ── S4: each measure over time at baseline ──
function supp_S4_network_dynamics()
    fig = Figure(; size=(1180, 460))
    for (ci, (key, lab)) in enumerate((CONSTR, EFFS))
        x, y = measseries(key)
        ax = Axis(
            fig[1, ci];
            title=lab,
            xlabel="period",
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
            xticks=50:50:200,
            limits=((TSTART, 201), ywin((x, y))),
        )
        scatterlines!(ax, x, y; color=COL_GAP, linewidth=2.2, markersize=6)
    end
    colgap!(fig.layout, 16)
    savefig("supp_S4_network_dynamics.png", fig)
end

for (name, f) in (
    ("S1", supp_S1_grid_lines),
    ("S2", supp_S2_position),
    ("S3", supp_S3_advantage),
    ("S4", supp_S4_network_dynamics),
)
    try
        f()
    catch e
        println("  $name FAILED: ", sprint(showerror, e)[1:min(end, 400)])
    end
end
# emit the display-convention keys quoted by the supplement captions
open(normpath(joinpath(@__DIR__, "..", "..", "paper", "supp_figmeta.tex")), "w") do io
    println(
        io,
        "% supp_figmeta.tex: generated by scripts/paper/supp_figures.jl. Do not edit by hand.",
    )
    println(
        io,
        "% Display conventions used to render paper/supp_figs/, quoted in captions via \\pv keys.",
    )
    println(io, "\\pvDefine{suppRollWin}{$ROLLW}")
    println(io, "\\pvDefine{suppMeasInterval}{$MEASINT}")
    println(io, "\\pvDefine{suppAxisStart}{$TSTART}")
end
println("supplement figures done (+ supp_figmeta.tex)")
