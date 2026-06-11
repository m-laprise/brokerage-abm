"""
    scripts/paper/figures.jl

Figures for the paper's results section, numbered in order of first citation in
the prose. Reads ONLY paper/figdata.jld2 (the small derived dataset written by
scripts/paper/figdata.jl on the cluster), so figures render locally with no
access to the sweep. CairoMakie only; no simulation; no hard-coded results
(literal constants are display conventions only). Outputs print-resolution PNGs
to paper/figs/ and the display-convention keys to paper/figmeta.tex.

  fig1_position_work  betweenness + access over time (both baselines) and cross-regime scatter
  fig2_rank_lines     four outcomes vs the effective rank of the matching function
  fig3_grid_lines     nine outcomes across the rho x delta grid, lines per delta
  fig4_advantage      structural measures vs informational/output gaps
  fig5_capture        captured share across sweeps and vs advantage measures
  fig6_dynamics       baseline dynamics, no-capture (top) vs capture (bottom); each row
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
const COL_OVERLAY = :gray72           # pale-gray cross-overlay in fig6

const FD = JLD2.load(normpath(joinpath(@__DIR__, "..", "..", "paper", "figdata.jld2")))["figdata"]
const PER = FD["period"]
const SER = FD["series"]
r90(rho, dl) = FD["r90"][(Float64(rho), Float64(dl))]
const RKLO, RKHI = extrema(values(FD["r90"]))          # display scales derived from the artifact
msz(k) = 6 + 12 * (k - RKLO) / (RKHI - RKLO)           # marker size from effective rank
savefig(fname, fig) = (save(joinpath(OUT, fname), fig; px_per_unit=PXU); println("  $fname done"))

# ── Figure 1: betweenness and access over time (both baselines) + cross-regime scatter ──
function fig1_position_work()
    fig = Figure(size=(1150, 860))
    axa = Axis(fig[1, 1]; title="Betweenness centrality over time", xlabel="period",
        ylabel="broker betweenness centrality",
        titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS, limits=((TSTART, 201), (0, nothing)))
    axb = Axis(fig[1, 2]; title="Access fraction over time", xlabel="period", ylabel="access fraction",
        titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS, limits=((TSTART, 201), (0, nothing)))
    for (model, col, ls, lbl) in (("base", COL_AGENT, :solid, "no capture"),
                                  ("capture", COL_CAPTURE, :solid, "capture"))
        bw = SER[model]["betweenness"]                          # measured every BETWINT periods
        mi = [i for i in eachindex(PER) if PER[i] % BETWINT == 0]
        sm = rolling_mean(bw[mi], ROLLW)
        scatterlines!(axa, PER[mi], sm; color=col, linestyle=ls, linewidth=2.2,
            markersize=6, label=lbl)
        sm2 = rolling_mean(SER[model]["access"], ROLLW)         # per-period
        lines!(axb, PER, sm2; color=col, linestyle=ls, linewidth=2.2, label=lbl)
    end
    axislegend(axa; position=:rb, LEG_KW...); axislegend(axb; position=:rt, LEG_KW...)
    # row 2: cross-regime scatter (base OAT cells), coloured by output gap
    cells = FD["oat_cells"]
    bx = [c["betw"] for c in cells]; ay = [c["access"] for c in cells]
    og = [c["qgap"] for c in cells]
    axc = Axis(fig[2, 1:2]; title="Across regimes (one dot per regime, late-window means)",
        xlabel="broker betweenness centrality", ylabel="access fraction", titlesize=TITLE_FS,
        xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    sc = scatter!(axc, bx, ay; color=og, colormap=Reverse(:plasma), colorrange=extrema(og), markersize=13,
        strokewidth=0.4, strokecolor=:gray30)
    Colorbar(fig[2, 3], sc; label="output gap q", labelsize=LABEL_FS,
        ticklabelsize=TICK_FS)
    colgap!(fig.layout, 14); rowgap!(fig.layout, 14)
    savefig("fig1_position_work.png", fig)
end

# ── Figure 2: four outcomes against the effective rank of the matching function ──
function fig2_rank_lines()
    pts = [(r90(c["rho"], c["delta"]), c["rho"], c["delta"], c["outcomes"])
           for c in FD["grid_cells"] if c["rho"] != 1.0]
    shown = ("Broker rank correlation", "Rank correlation gap", "Broker output q", "Output gap q")
    fig = Figure(size=(980, 760))
    for (k, ttl) in enumerate(shown)
        rr = div(k - 1, 2) + 1; cc = mod(k - 1, 2) + 1
        ax = Axis(fig[rr, cc]; title=ttl, xlabel = rr == 2 ? "effective rank r₉₀" : "",
            titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
        for d in (0.0, 0.5, 0.75)
            grp = sort([(p[1], p[4][ttl]) for p in pts if p[3] == d]; by=first)
            isempty(grp) && continue
            scatterlines!(ax, first.(grp), last.(grp); color=DELTA_COLORS[d], linewidth=1.6,
                markersize=11, strokewidth=0.4, strokecolor=:gray30)
        end
        if k == 1
            els = [MarkerElement(marker=:circle, color=DELTA_COLORS[d], markersize=11) for d in (0.0, 0.5, 0.75)]
            axislegend(ax, els, ["δ = 0", "δ = 0.5", "δ = 0.75"]; position=:rt, LEG_KW...)
        end
    end
    colgap!(fig.layout, 14); rowgap!(fig.layout, 10)
    savefig("fig2_rank_lines.png", fig)
end

# ── Figure 3: nine outcomes vs rho, one line per delta (rho = 1 dropped) ──
function fig3_grid_lines()
    gcells = FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gcells]))
    cells = Dict((c["rho"], c["delta"]) => c["outcomes"] for c in gcells if c["rho"] != 1.0)
    # 3x3: first column = the [0,1]-bounded quantities (absolute 0-1 axis); columns
    # 2-3 = the rank-correlation, R², and output pairs (shared y-axis per row).
    layout = ["Betweenness centrality" "Broker rank correlation" "Rank correlation gap";
              "Access fraction"        "Broker prediction R²"    "Prediction R² gap";
              "Outsourcing rate"       "Broker output q"         "Output gap q"]
    fig = Figure(size=(1280, 980))
    axs = Matrix{Axis}(undef, 3, 3)
    for rr in 1:3, cc in 1:3
        ttl = layout[rr, cc]
        ax = Axis(fig[rr, cc]; title=ttl, xlabel = rr == 3 ? "ρ (complementarity vs quality)" : "",
            xticks=[0, 0.3, 0.5, 0.7], titlesize=TITLE_FS, xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
            limits = cc == 1 ? (nothing, (0, 1.02)) : (nothing, nothing))
        for d in dls
            pts = sort([(r, o[ttl]) for ((r, dd), o) in cells if dd == d]; by=first)
            scatterlines!(ax, first.(pts), last.(pts); color=DELTA_COLORS[d], linewidth=2.0,
                markersize=10, strokewidth=0.4, strokecolor=:gray30, label="δ = $d")
        end
        rr == 1 && cc == 1 && axislegend(ax, "Regime gain"; position=:rb, LEG_KW...)
        axs[rr, cc] = ax
    end
    for rr in 1:3; linkyaxes!(axs[rr, 2], axs[rr, 3]) end   # pairs share a y-axis
    colgap!(fig.layout, 16); rowgap!(fig.layout, 12)
    savefig("fig3_grid_lines.png", fig)
end

# ── Figure 4: structural measures vs informational/output gaps (4 panels) ──
function fig4_advantage()
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
    savefig("fig4_advantage.png", fig)
end

# ── Figure 5: captured share across sweeps (row 1) and vs the advantage measures (row 2) ──
function fig5_capture()
    sw = FD["capture_sweeps"]
    sweeps = [("ρ (complementarity vs quality)", "rho"), ("N (market size)", "N"),
              ("f_r (reservation threshold)", "fr"), ("η (turnover)", "eta")]
    fig = Figure(size=(1340, 760))
    for (ci, (name, key)) in enumerate(sweeps)
        s = sw[key]; vals = s["labels"]; mu = s["mean"]; sd = s["sd"]
        ax = Axis(fig[1, ci]; title=name, ylabel = ci == 1 ? "captured share of\noutsourced demand" : "",
            xticks=(1:length(vals), vals), titlesize=TITLE_FS, ylabelsize=LABEL_FS - 1,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, 1.02)))
        x = 1:length(vals)
        band!(ax, x, mu .- sd, mu .+ sd; color=(COL_CAPTURE, 0.15))
        scatterlines!(ax, x, mu; color=COL_CAPTURE, markersize=9)
    end
    # row 2: every capture cell (f_r = 1.2 included), coloured by f_r
    cc = FD["capture_cells"]
    cs = [c["capshare"] for c in cc]; bw = [c["betw"] for c in cc]
    ac = [c["access"] for c in cc]; rg = [c["rankgap"] for c in cc]
    qg = [c["qgap"] for c in cc]; fr = [c["fr"] for c in cc]
    panels = [("Betweenness centrality", bw), ("Access fraction", ac), ("Rank correlation gap", rg), ("Output gap q", qg)]
    for (ci, (xlab, xv)) in enumerate(panels)
        ax = Axis(fig[2, ci]; title=xlab, ylabel = ci == 1 ? "captured share of\noutsourced demand" : "",
            titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS - 1,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, 1.05)))
        for v in (0.4, 0.6, 0.9, 1.2)
            mm = fr .== v
            scatter!(ax, xv[mm], cs[mm]; color=FR_COLORS[v], markersize=9,
                strokewidth=0.3, strokecolor=:gray30, label="f_r = $v")
        end
        ci == 4 && axislegend(ax, "Reservation\nthreshold"; position=:lb, LEG_KW...)
    end
    colgap!(fig.layout, 12); rowgap!(fig.layout, 14)
    savefig("fig5_capture.png", fig)
end

# ── Figure 6: baseline dynamics, no-capture (top) vs capture (bottom); each row also
#    shows the other row's series in pale gray for direct comparison ──
function fig6_dynamics()
    # ensemble mean, ROLLW-rolling over the full series; display trimming is axis-only
    ot(model, key) = (PER, rolling_mean(SER[model][key], ROLLW))
    function otb(model)    # betweenness: rolling over the BETWINT-period measurements
        mi = [i for i in eachindex(PER) if PER[i] % BETWINT == 0]
        (PER[mi], rolling_mean(SER[model]["betweenness"][mi], ROLLW))
    end
    yrange(ss...) = (v = filter(!isnan, vcat((s[2][s[1] .>= TSTART] for s in ss)...));   # displayed window only
        lo = minimum(v); hi = maximum(v); pad = 0.06 * (hi - lo + eps()); (lo - pad, hi + pad))
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
    savefig("fig6_dynamics.png", fig)
end

for (name, f) in (("fig1", fig1_position_work), ("fig2", fig2_rank_lines), ("fig3", fig3_grid_lines),
                  ("fig4", fig4_advantage), ("fig5", fig5_capture), ("fig6", fig6_dynamics))
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
