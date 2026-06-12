"""
    scripts/paper/figures.jl

Figures for the paper's results section, numbered in order of first citation in
the prose. Reads ONLY paper/figdata.jld2 (the small derived dataset written by
scripts/paper/figdata.jl on the cluster), so figures render locally with no
access to the sweep. CairoMakie only; no simulation; no hard-coded results
(literal constants are display conventions only). Outputs print-resolution PNGs
to paper/figs/ and the display-convention keys to paper/figmeta.tex.

  fig1_grid_lines     six outcomes across the rho x delta grid, lines per delta
  fig2_position_work  betweenness & access over time at baseline + cross-regime scatter
  fig3_advantage      structural measures vs informational/output gaps
  fig4_capture        captured share across three sweeps and vs the output gap
  fig5_dynamics       baseline dynamics, no-capture (top) vs capture (bottom); each row
                      overlays the other row's series in pale gray for comparison

Usage: julia --project scripts/paper/figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
using JLD2
using Statistics: mean

const OUT = normpath(joinpath(@__DIR__, "..", "..", "paper", "figs"))
mkpath(OUT)
const PXU = 2.0                       # px_per_unit: ~330+ dpi at printed full-page width
# Display conventions, quoted in the captions (paper/captions.tex) via the keys
# emitted to paper/figmeta.tex. BETWINT matches the generation-time
# network_measure_interval; TSTART is the end of burn-in (display trim only).
const ROLLW = 5                       # rolling-mean window, in observations
const BETWINT = 20                    # betweenness measurement interval, periods
const TSTART = 30                     # displayed axes start here; data never cut
const RHO_COLORS = Dict(0.0 => :seagreen, 0.3 => :mediumaquamarine, 0.5 => :goldenrod,
                        0.7 => :darkorange, 1.0 => :firebrick)
const DELTA_COLORS = Dict(0.0 => :steelblue, 0.5 => :goldenrod, 0.75 => :firebrick)
const FR_COLORS = Dict(0.4 => :black, 0.6 => :deepskyblue, 0.9 => :darkorange, 1.2 => :firebrick)
const COL_OVERLAY = :gray72           # pale-gray cross-overlay in fig5

const FD = JLD2.load(normpath(joinpath(@__DIR__, "..", "..", "paper", "figdata.jld2")))["figdata"]
const PER = FD["period"]
const SER = FD["series"]
r90(rho, dl) = FD["r90"][(Float64(rho), Float64(dl))]
const RKLO, RKHI = extrema(values(FD["r90"]))          # display scales derived from the artifact
msz(k) = 6 + 12 * (k - RKLO) / (RKHI - RKLO)           # marker size from effective rank
savefig(fname, fig) = (save(joinpath(OUT, fname), fig; px_per_unit=PXU); println("  $fname done"))

# ── Figure 2: betweenness & access over time at baseline + cross-regime scatter ──
function fig2_position_work()
    fig = Figure(size=(1180, 470))
    # left: broker betweenness and access fraction over time, no-capture baseline only
    axa = Axis(fig[1, 1]; title="Over time, at baseline", xlabel="period",
        titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS, limits=((TSTART, 201), (0, 1.0)))
    mi = [i for i in eachindex(PER) if PER[i] % BETWINT == 0]
    bw = rolling_mean(SER["base"]["betweenness"][mi], ROLLW)   # measured every BETWINT periods
    scatterlines!(axa, PER[mi], bw; color=COL_GAP, linewidth=2.2, markersize=6,
        label="broker betweenness centrality")
    ac = rolling_mean(SER["base"]["access"], ROLLW)            # per-period
    lines!(axa, PER, ac; color=COL_ACCESS, linewidth=2.2, label="access fraction")
    axislegend(axa; position=:rc, LEG_KW...)
    # right: cross-regime scatter (base OAT cells), access (x) vs betweenness (y)
    cells = FD["oat_cells"]
    bx = [c["betw"] for c in cells]; ay = [c["access"] for c in cells]
    axc = Axis(fig[1, 2]; title="Across regimes",
        xlabel="access fraction", ylabel="broker betweenness centrality", titlesize=TITLE_FS,
        xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
        limits=(nothing, (0, 1)))
    scatter!(axc, ay, bx; color=:black, markersize=13)
    colgap!(fig.layout, 14)
    savefig("fig2_position_work.png", fig)
end

# ── Figure 1: six outcomes vs rho across the grid, one line per delta (rho = 1 included) ──
function fig1_grid_lines()
    gcells = FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gcells]))
    cells = Dict((c["rho"], c["delta"]) => c["outcomes"] for c in gcells)
    # 2x3: column 1 = the [0,1]-bounded structural quantities (absolute 0-1 axis);
    # columns 2-3 = the prediction and output outcomes (each panel autoscaled).
    layout = ["Betweenness centrality" "Broker rank correlation" "Rank correlation gap";
              "Access fraction"        "Broker prediction R²"    "Output gap q"]
    fig = Figure(size=(1280, 700))
    for rr in 1:2, cc in 1:3
        ttl = layout[rr, cc]
        ax = Axis(fig[rr, cc]; title=ttl, xlabel = rr == 2 ? "ρ (complementarity vs quality)" : "",
            xticks=[0, 0.3, 0.5, 0.7, 1], titlesize=TITLE_FS, xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
            limits = cc == 1 ? (nothing, (0, 1.02)) : (nothing, nothing))
        for d in dls
            pts = sort([(r, o[ttl]) for ((r, dd), o) in cells if dd == d]; by=first)
            scatterlines!(ax, first.(pts), last.(pts); color=DELTA_COLORS[d], linewidth=2.0,
                markersize=10, strokewidth=0.4, strokecolor=:gray30, label="δ = $d")
        end
        rr == 1 && cc == 1 && axislegend(ax, "Regime gain"; position=:lb, LEG_KW...)
    end
    colgap!(fig.layout, 16); rowgap!(fig.layout, 12)
    savefig("fig1_grid_lines.png", fig)
end

# ── Figure 3: structural measures vs informational/output gaps (4 panels) ──
function fig3_advantage()
    bc = FD["base_cells"]   # every saved no-capture regime
    rho = [c["rho"] for c in bc]; dlt = [c["delta"] for c in bc]
    bw = [c["betw"] for c in bc]; ac = [c["access"] for c in bc]
    rg = [c["rankgap"] for c in bc]; qg = [c["qgap"] for c in bc]
    rks = [r90(r, d) for (r, d) in zip(rho, dlt)]
    xs = [("Broker betweenness centrality", bw), ("Access fraction", ac)]
    ys = [("Rank correlation gap", rg), ("Output gap q", qg)]
    fig = Figure(size=(1150, 940))
    for (ri, (ylab, yv)) in enumerate(ys), (ci, (xlab, xv)) in enumerate(xs)
        ax = Axis(fig[ri, ci]; xlabel = ri == 2 ? xlab : "", ylabel = ci == 1 ? ylab : "",
            title = ri == 1 ? xlab : "", titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
        for rv in sort(unique(rho))
            mm = rho .== rv
            scatter!(ax, xv[mm], yv[mm]; color=(RHO_COLORS[rv], 0.8), markersize=msz.(rks[mm]),
                strokewidth=0.3, strokecolor=:gray30)
        end
        # legends in panel corners the data leaves empty (fixed-size swatches:
        # plot markers have data-dependent sizes)
        if ri == 1 && ci == 2
            rvs = sort(unique(rho))
            els = [MarkerElement(marker=:circle, color=RHO_COLORS[v], markersize=12) for v in rvs]
            axislegend(ax, els, ["ρ = $v" for v in rvs]; position=:rb, LEG_KW...)
        end
        if ri == 2 && ci == 1
            ks = round.((RKLO, (RKLO + RKHI) / 2, RKHI); digits=0)
            els = [MarkerElement(marker=:circle, color=:gray55, markersize=msz(k)) for k in ks]
            axislegend(ax, els, ["r₉₀ = $(Int(k))" for k in ks], "Effective rank"; position=:lt, LEG_KW...)
        end
    end
    colgap!(fig.layout, 16); rowgap!(fig.layout, 12)
    savefig("fig3_advantage.png", fig)
end

# ── Figure 4: captured share across three sweeps + captured share vs the output gap ──
function fig4_capture()
    sw = FD["capture_sweeps"]
    sweeps = [("Matching problem", "rho", "ρ"),
              ("Reservation", "fr", "f_r"), ("Turnover", "eta", "η")]
    fig = Figure(size=(1340, 440))
    for (ci, (name, key, xl)) in enumerate(sweeps)
        s = sw[key]; vals = s["labels"]; mu = s["mean"]; sd = s["sd"]
        ax = Axis(fig[1, ci]; title=name, xlabel=xl,
            ylabel = ci == 1 ? "captured share of\noutsourced demand" : "",
            xticks=(1:length(vals), vals), titlesize=TITLE_FS, xlabelsize=LABEL_FS - 1,
            ylabelsize=LABEL_FS - 1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
            limits=(nothing, (0, 1.02)))
        x = 1:length(vals)
        band!(ax, x, mu .- sd, mu .+ sd; color=(COL_CAPTURE, 0.15))
        scatterlines!(ax, x, mu; color=COL_CAPTURE, markersize=9)
    end
    # final panel: every capture cell (f_r = 1.2 included), captured share (x) vs output gap (y)
    cc = FD["capture_cells"]
    cs = [c["capshare"] for c in cc]; qg = [c["qgap"] for c in cc]; fr = [c["fr"] for c in cc]
    ax = Axis(fig[1, 4]; title="Output gap q", xlabel="captured share of\noutsourced demand",
        ylabel="output gap q", titlesize=TITLE_FS, xlabelsize=LABEL_FS - 1, ylabelsize=LABEL_FS - 1,
        xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    for v in (0.4, 0.6, 0.9, 1.2)
        mm = fr .== v
        scatter!(ax, cs[mm], qg[mm]; color=FR_COLORS[v], markersize=9,
            strokewidth=0.3, strokecolor=:gray30, label="f_r = $v")
    end
    axislegend(ax, "Reservation"; position=:rt, LEG_KW...)
    colgap!(fig.layout, 16)
    savefig("fig4_capture.png", fig)
end

# ── Figure 5: baseline dynamics, no-capture (top) vs capture (bottom); each row also
#    shows the other row's series in pale gray for direct comparison ──
function fig5_dynamics()
    # ensemble mean, ROLLW-rolling over the full series; display trimming is axis-only
    ot(model, key) = (PER, rolling_mean(SER[model][key], ROLLW))
    function otb(model)    # betweenness: rolling over the BETWINT-period measurements
        mi = [i for i in eachindex(PER) if PER[i] % BETWINT == 0]
        (PER[mi], rolling_mean(SER[model]["betweenness"][mi], ROLLW))
    end
    yrange(ss...) = (v = filter(!isnan, vcat((s[2][s[1] .>= TSTART] for s in ss)...));   # displayed window only
        hi = maximum(v); (0, hi + 0.06 * (hi + eps())))   # all y-axes start at zero
    mpaB, mpaC = ot("base", "mpa"), ot("capture", "mpa")
    dmnB, dmdB = ot("base", "mean_degree"), ot("base", "median_degree")
    dmnC, dmdC = ot("capture", "mean_degree"), ot("capture", "median_degree")
    bwB, bwC = otb("base"), otb("capture")
    acB, acC = ot("base", "access"), ot("capture", "access")
    osB, osC = ot("base", "outsourcing"), ot("capture", "outsourcing")
    capC = ot("capture", "capshare_total")   # captured share of TOTAL demand
    yl = (yrange(mpaB, mpaC), yrange(dmnB, dmdB, dmnC, dmdC), yrange(bwB, bwC), yrange(acB, acC), (0, 1.02))

    fig = Figure(size=(1220, 560))
    # all fonts slightly smaller than the shared sizes: this figure has 10 panels
    T6, L6, K6, G6 = TITLE_FS - 6, LABEL_FS - 3, TICK_FS - 3, 15
    mk(r, c, ttl, ylim) = Axis(fig[r, c]; title=ttl, xlabel = r == 2 ? "period" : "",
        xticks=50:50:200, titlesize=T6, xlabelsize=L6, xticklabelsize=K6,
        yticklabelsize=K6, limits=((TSTART, 201), ylim))
    # a `label` keyword is only passed when a label is requested: a plot with
    # label="" would still register a (blank) legend entry
    function drw!(ax, s, col; lbl=nothing, ls=:solid, pts=false, lw=2.2)
        kw = isnothing(lbl) ? (;) : (; label=lbl)
        pts ? scatterlines!(ax, s[1], s[2]; color=col, linewidth=lw, markersize=6, linestyle=ls, kw...) :
              lines!(ax, s[1], s[2]; color=col, linewidth=lw, linestyle=ls, kw...)
    end
    # pale-gray cross-overlays first, so each row's own series draws on top; never labeled
    gry!(ax, s; ls=:solid, pts=false) = drw!(ax, s, COL_OVERLAY; ls=ls, pts=pts, lw=1.6)
    # top row: no capture (gray = capture counterparts; explained in the caption)
    let a = mk(1, 1, "Matches per agent", yl[1])
        gry!(a, mpaC); drw!(a, mpaB, COL_DIAG)
    end
    let a = mk(1, 2, "Network degree", yl[2])
        gry!(a, dmnC); gry!(a, dmdC; ls=:dot)
        drw!(a, dmnB, COL_AGENT; lbl="mean"); drw!(a, dmdB, COL_AGENT; lbl="median", ls=:dot)
        axislegend(a; position=:rb, LEG_KW..., labelsize=G6, patchsize=(15, 11))
    end
    let a = mk(1, 3, "Broker betweenness\ncentrality", yl[3])
        gry!(a, bwC; pts=true); drw!(a, bwB, COL_GAP; pts=true)
    end
    let a = mk(1, 4, "Access fraction", yl[4])
        gry!(a, acC); drw!(a, acB, COL_ACCESS)
    end
    let a = mk(1, 5, "Outsourcing rate", yl[5])
        gry!(a, osC); gry!(a, capC; ls=:dash); drw!(a, osB, COL_BROKER)
    end
    # bottom row: capture (gray = no-capture counterparts)
    let a = mk(2, 1, "", yl[1])
        gry!(a, mpaB); drw!(a, mpaC, COL_DIAG)
    end
    let a = mk(2, 2, "", yl[2])
        gry!(a, dmnB); gry!(a, dmdB; ls=:dot)
        drw!(a, dmnC, COL_AGENT); drw!(a, dmdC, COL_AGENT; ls=:dot)
    end
    let a = mk(2, 3, "", yl[3])
        gry!(a, bwB; pts=true); drw!(a, bwC, COL_GAP; pts=true)
    end
    let a = mk(2, 4, "", yl[4])
        gry!(a, acB); drw!(a, acC, COL_ACCESS)
    end
    let a = mk(2, 5, "", yl[5])
        gry!(a, osB)
        drw!(a, osC, COL_BROKER; lbl="outsourcing rate")
        drw!(a, capC, COL_CAPTURE; lbl="capture share", ls=:dash)
        axislegend(a; position=:rb, LEG_KW..., labelsize=G6, patchsize=(15, 11))
    end
    Label(fig[1, 0], "No capture"; rotation=π/2, font=:bold, fontsize=L6, tellheight=false)
    Label(fig[2, 0], "Capture"; rotation=π/2, font=:bold, fontsize=L6, tellheight=false)
    colsize!(fig.layout, 0, Fixed(30))
    colgap!(fig.layout, 12); rowgap!(fig.layout, 12)
    savefig("fig5_dynamics.png", fig)
end

for (name, f) in (("fig1", fig1_grid_lines), ("fig2", fig2_position_work),
                  ("fig3", fig3_advantage), ("fig4", fig4_capture), ("fig5", fig5_dynamics))
    try
        f()
    catch e
        println("  $name FAILED: ", sprint(showerror, e)[1:min(end, 400)])
    end
end
# emit the display-convention keys quoted by the captions (paper/captions.tex)
open(normpath(joinpath(@__DIR__, "..", "..", "paper", "figmeta.tex")), "w") do io
    println(io, "% figmeta.tex: generated by scripts/paper/figures.jl. Do not edit by hand.")
    println(io, "% Display conventions used to render paper/figs/, quoted in captions via \\pv keys.")
    println(io, "\\pvDefine{rollWin}{$ROLLW}")
    println(io, "\\pvDefine{betwInterval}{$BETWINT}")
    println(io, "\\pvDefine{axisStart}{$TSTART}")
end
println("paper figures done (+ figmeta.tex)")
