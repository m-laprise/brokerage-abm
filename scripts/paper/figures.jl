"""
    scripts/paper/figures.jl

Figure assets for the paper's results section. TeX figure numbers follow their
placement in `paper/section_source.tex`; the asset filenames remain stable.
Reads ONLY output/main/figdata.jld2 (the small derived dataset written by
scripts/paper/figdata.jl on the cluster), so figures render locally with no
access to the sweep. CairoMakie only; no simulation; no hard-coded results
(literal constants are display conventions only). Outputs print-resolution PNGs
to output/main/figs/ and the display-convention keys to output/main/figmeta.tex.

  fig1_dynamics       baseline dynamics
  fig2_grid_lines     six outcomes across the rho x delta grid, lines per delta
  fig3_position_work  betweenness & access over time at baseline + cross-regime scatter
  fig4_advantage      structural measures vs informational/output gaps

Usage: julia --project scripts/paper/figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
using JLD2
using Statistics: mean

const OUT = normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figs"))
mkpath(OUT)
const PXU = 2.0                       # px_per_unit: ~330+ dpi at printed full-page width
# Display conventions, quoted in the captions (paper/captions.tex) via the keys
# emitted to output/main/figmeta.tex. BETWINT matches the generation-time
# network_measure_interval; TSTART is the end of burn-in (display trim only).
const ROLLW = 5                       # rolling-mean window, in observations
const BETWINT = 20                    # betweenness measurement interval, periods
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

const FD = JLD2.load(
    normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figdata.jld2"))
)["figdata"]
const PER = FD["period"]
const SER = FD["series"]
const TEND = maximum(PER)
const TIME_TICK_STEP = TEND <= 250 ? 50 : 100
const TIME_TICKS = TIME_TICK_STEP:TIME_TICK_STEP:TEND
const ADV_MARKER_SIZE = 10
function savefig(fname, fig)
    (save(joinpath(OUT, fname), fig; px_per_unit=PXU); println("  $fname done"))
end

# ── Position: betweenness & access over time at baseline + cross-regime scatter ──
function fig3_position_work()
    fig = Figure(; size=(1180, 470))
    # left: broker betweenness and access fraction over time at baseline
    axa = Axis(
        fig[1, 1];
        title="Over time, at baseline",
        xlabel="period",
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
        limits=((TSTART, TEND + 1), (0, 1.0)),
    )
    mi = [i for i in eachindex(PER) if PER[i] % BETWINT == 0]
    bw = rolling_mean(SER["betweenness"][mi], ROLLW)   # measured every BETWINT periods
    scatterlines!(
        axa,
        PER[mi],
        bw;
        color=COL_GAP,
        linewidth=2.2,
        markersize=6,
        label="broker betweenness centrality",
    )
    ac = rolling_mean(SER["access"], ROLLW)            # per-period
    lines!(axa, PER, ac; color=COL_ACCESS, linewidth=2.2, label="access fraction")
    axislegend(axa; position=:rc, LEG_KW...)
    # right: cross-regime scatter, access (x) vs betweenness (y)
    cells = FD["oat_cells"]
    bx = [c["betw"] for c in cells];
    ay = [c["access"] for c in cells]
    axc = Axis(
        fig[1, 2];
        title="Across regimes",
        xlabel="access fraction",
        ylabel="broker betweenness centrality",
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
        limits=(nothing, (0, 1)),
    )
    scatter!(axc, ay, bx; color=:black, markersize=13)
    colgap!(fig.layout, 14)
    savefig("fig3_position_work.png", fig)
end

# ── Matching grid: six outcomes vs rho, one line per delta (rho = 1 included) ──
function fig2_grid_lines()
    gcells = FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gcells]))
    cells = Dict((c["rho"], c["delta"]) => c["outcomes"] for c in gcells)
    # 2x3: column 1 = the [0,1]-bounded structural quantities (absolute 0-1 axis);
    # columns 2-3 = the prediction and output outcomes (each panel autoscaled).
    layout = [
        "Betweenness centrality" "Broker rank correlation" "Rank correlation gap";
        "Access fraction" "Broker prediction R²" "Output gap q"
    ]
    fig = Figure(; size=(1280, 700))
    for rr in 1:2, cc in 1:3
        ttl = layout[rr, cc]
        ax = Axis(
            fig[rr, cc];
            title=ttl,
            xlabel=rr == 2 ? "ρ (complementarity vs quality)" : "",
            xticks=[0, 0.3, 0.5, 0.7, 0.85, 1],
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
            limits=cc == 1 ? (nothing, (0, 1.02)) : (nothing, nothing),
        )
        for d in dls
            pts = sort([(r, o[ttl]) for ((r, dd), o) in cells if dd == d]; by=first)
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
        rr == 1 && cc == 1 && axislegend(ax, "Regime gain"; position=:lb, LEG_KW...)
    end
    colgap!(fig.layout, 16);
    rowgap!(fig.layout, 12)
    savefig("fig2_grid_lines.png", fig)
end

# ── Advantage: structural measures vs informational/output gaps (4 panels) ──
function fig4_advantage()
    bc = FD["regime_cells"]
    rho = [c["rho"] for c in bc]
    bw = [c["betw"] for c in bc];
    ac = [c["access"] for c in bc]
    rg = [c["rankgap"] for c in bc];
    qg = [c["qgap"] for c in bc]
    xs = [("Broker betweenness centrality", bw), ("Access fraction", ac)]
    ys = [("Rank correlation gap", rg), ("Output gap q", qg)]
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
                markersize=ADV_MARKER_SIZE,
                strokewidth=0.3,
                strokecolor=:gray30,
            )
        end
        if ri == 1 && ci == 2
            rvs = sort(unique(rho))
            els = [
                MarkerElement(;
                    marker=:circle, color=RHO_COLORS[v], markersize=ADV_MARKER_SIZE
                ) for v in rvs
            ]
            axislegend(ax, els, ["ρ = $v" for v in rvs]; position=:rb, LEG_KW...)
        end
    end
    colgap!(fig.layout, 16);
    rowgap!(fig.layout, 12)
    savefig("fig4_advantage.png", fig)
end

# ── Baseline dynamics, placed first in the results section ──
function fig1_dynamics()
    # ensemble mean, ROLLW-rolling over the full series; display trimming is axis-only
    ot(key) = (PER, rolling_mean(SER[key], ROLLW))
    function otb()    # betweenness: rolling over the BETWINT-period measurements
        mi = [i for i in eachindex(PER) if PER[i] % BETWINT == 0]
        (PER[mi], rolling_mean(SER["betweenness"][mi], ROLLW))
    end
    yrange(ss...) = (
        v=filter(!isnan, vcat((s[2][s[1] .>= TSTART] for s in ss)...));   # displayed window only
        hi=maximum(v);
        (0, hi + 0.06 * (hi + eps()))
    )   # all y-axes start at zero
    mpa = ot("mpa")
    dmean, dmedian = ot("mean_degree"), ot("median_degree")
    betweenness = otb()
    access = ot("access")
    outsourcing = ot("outsourcing")
    yl = (
        yrange(mpa), yrange(dmean, dmedian), yrange(betweenness), yrange(access), (0, 1.02)
    )

    fig = Figure(; size=(1220, 580))
    T4, L4, K4, G4 = TITLE_FS - 4, LABEL_FS - 2, TICK_FS - 2, 16
    mk(row, cols, ttl, ylim) = Axis(
        fig[row, cols];
        title=ttl,
        xlabel=row == 2 ? "period" : "",
        xticks=TIME_TICKS,
        titlesize=T4,
        xlabelsize=L4,
        xticklabelsize=K4,
        yticklabelsize=K4,
        limits=((TSTART, TEND + 1), ylim),
    )
    # a `label` keyword is only passed when a label is requested: a plot with
    # label="" would still register a (blank) legend entry
    function drw!(ax, s, col; lbl=nothing, ls=:solid, pts=false, lw=2.2)
        kw = isnothing(lbl) ? (;) : (; label=lbl)
        if pts
            scatterlines!(
                ax, s[1], s[2]; color=col, linewidth=lw, markersize=6, linestyle=ls, kw...
            )
        else
            lines!(ax, s[1], s[2]; color=col, linewidth=lw, linestyle=ls, kw...)
        end
    end
    let a = mk(1, 1:2, "Matches per agent", yl[1])
        drw!(a, mpa, COL_DIAG)
    end
    let a = mk(1, 3:4, "Network degree", yl[2])
        drw!(a, dmean, COL_AGENT; lbl="mean")
        drw!(a, dmedian, COL_REFERENCE; lbl="median", ls=:dot)
        axislegend(a; position=:rb, LEG_KW..., labelsize=G4, patchsize=(15, 11))
    end
    let a = mk(1, 5:6, "Broker betweenness centrality", yl[3])
        drw!(a, betweenness, COL_GAP; pts=true)
    end
    let a = mk(2, 1:3, "Access fraction", yl[4])
        drw!(a, access, COL_ACCESS)
    end
    let a = mk(2, 4:6, "Outsourcing rate", yl[5])
        drw!(a, outsourcing, COL_BROKER)
    end
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 12)
    savefig("fig1_dynamics.png", fig)
end

for (name, f) in (
    ("fig1", fig1_dynamics),
    ("fig2", fig2_grid_lines),
    ("fig3", fig3_position_work),
    ("fig4", fig4_advantage),
)
    try
        f()
    catch e
        println("  $name FAILED: ", sprint(showerror, e)[1:min(end, 400)])
    end
end
# emit the display-convention keys quoted by the captions (paper/captions.tex)
open(normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figmeta.tex")), "w") do io
    println(
        io, "% figmeta.tex: generated by scripts/paper/figures.jl. Do not edit by hand."
    )
    println(
        io,
        "% Display conventions used to render output/main/figs/, quoted in captions via \\pv keys.",
    )
    println(io, "\\pvDefine{rollWin}{$ROLLW}")
    println(io, "\\pvDefine{betwInterval}{$BETWINT}")
    println(io, "\\pvDefine{axisStart}{$TSTART}")
end
println("main figures done (+ figmeta.tex)")
