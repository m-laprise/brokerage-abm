"""
    summary_figures.jl

Figures for the results-summary report (results_report.tex). Reads only saved sweep
data plus the initialization-only DGP rank grid (_results/dgp_rank_grid.jld2).
CairoMakie only — no simulation. Figure titles are brief and descriptive; captions
(with derivations) live in the LaTeX.

Usage: julia --project scripts/diagnostics/summary_figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean, access_fraction
using JLD2
using Statistics: mean, std, cor, cov, var

const ROOT = "/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438"
const OUT = joinpath(ROOT, "report")
const RHO5 = [0.0, 0.3, 0.5, 0.7, 1.0]
const RHO_COLORS = Dict(0.0 => :seagreen, 0.3 => :mediumaquamarine, 0.5 => :goldenrod,
                        0.7 => :darkorange, 1.0 => :firebrick)
const DELTA_COLORS = Dict(0.0 => :steelblue, 0.5 => :goldenrod, 0.75 => :firebrick)
const FR_COLORS = Dict(0.4 => :black, 0.6 => :deepskyblue, 0.9 => :darkorange, 1.2 => :firebrick)

nanmean(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
tailmean(df, col) = nanmean(df[(df.period .>= 181) .& (df.period .<= 200), col])   # late-window mean [181,200], the headline statistic
seedstat(mdfs, col) = (vs = filter(!isnan, [tailmean(d, col) for d in mdfs]); (mean(vs), std(vs)))
load_mdfs(rel) = jldopen(joinpath(ROOT, rel, "data.jld2"), "r") do f; f["mdfs"] end
load_cfg(rel) = jldopen(joinpath(ROOT, rel, "data.jld2"), "r") do f; f["config"] end
access_tail(df) = nanmean(access_fraction(df)[(df.period .>= 181) .& (df.period .<= 200)])
cell_access(mdfs) = nanmean([access_tail(d) for d in mdfs])
period_ens(mdfs, f) = (per = mdfs[1].period; (per, [nanmean(Float64[f(d)[t] for d in mdfs]) for t in eachindex(per)]))

RG = JLD2.load(joinpath(@__DIR__, "_results", "dgp_rank_grid.jld2"))
r90(rho, dl) = RG["r90"][(Float64(rho), Float64(dl))]

# the nine outcomes shared by Figures 1 and 1-alt
qgap(m) = seedstat(m, :q_broker_standard_mean)[1] - seedstat(m, :q_self_mean)[1]
const OUTCOMES = [
    ("Betweenness",            m -> seedstat(m, :betweenness)[1]),
    ("Access fraction",        cell_access),
    ("Broker prediction R²",   m -> seedstat(m, :broker_holdout_r2)[1]),
    ("Prediction R² gap",      m -> seedstat(m, :broker_holdout_r2)[1] - seedstat(m, :agent_holdout_r2)[1]),
    ("Broker rank correlation", m -> seedstat(m, :broker_holdout_rank)[1]),
    ("Rank correlation gap",   m -> seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1]),
    ("Broker output q",        m -> seedstat(m, :q_broker_standard_mean)[1]),
    ("Output gap q",           qgap),
    ("Outsourcing rate",       m -> seedstat(m, :outsourcing_rate)[1]),
]

# ── Figure 1: outcomes vs rho, one line per delta (rho = 1 dropped) ──
function fig1_grid_lines()
    s = jldopen(joinpath(ROOT, "phase/rho_delta/base/summary.jld2"), "r") do f; (xv=f["xvals"], yv=f["yvals"]) end
    dls = collect(s.yv)
    cells = Dict{Tuple{Float64,Float64},Vector{DataFrame}}()
    for (xi, r) in enumerate(s.xv), (yi, d) in enumerate(s.yv)
        r == 1.0 && continue
        cells[(r, d)] = load_mdfs("phase/rho_delta/cells/$(xi-1)_$(yi-1)/base")
    end
    for r in (0.3, 0.7)   # OAT rho cells refine the delta = 0.5 line
        cells[(r, 0.5)] = load_mdfs("oat/rho=$r/base")
    end
    # 3x3: first column = the [0,1]-bounded quantities (absolute 0-1 axis); columns
    # 2-3 = the rank-correlation, R², and output pairs (shared y-axis per row).
    byname(n) = OUTCOMES[findfirst(o -> o[1] == n, OUTCOMES)]
    layout = [byname("Betweenness")      byname("Broker rank correlation") byname("Rank correlation gap");
              byname("Access fraction")  byname("Broker prediction R²")    byname("Prediction R² gap");
              byname("Outsourcing rate") byname("Broker output q")         byname("Output gap q")]
    fig = Figure(size=(1280, 980))
    Label(fig[0, 1:3], "Late-window means across the matching-function grid";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    axs = Matrix{Axis}(undef, 3, 3)
    for rr in 1:3, cc in 1:3
        ttl, f = layout[rr, cc]
        ax = Axis(fig[rr, cc]; title=ttl, xlabel = rr == 3 ? "ρ (channel mix)" : "",
            xticks=[0, 0.3, 0.5, 0.7], titlesize=TITLE_FS, xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
            limits = cc == 1 ? (nothing, (0, 1.02)) : (nothing, nothing))
        for d in dls
            pts = sort([(r, f(m)) for ((r, dd), m) in cells if dd == d]; by=first)
            scatterlines!(ax, first.(pts), last.(pts); color=DELTA_COLORS[d], linewidth=2.0,
                markersize=10, strokewidth=0.4, strokecolor=:gray30, label="δ = $d")
        end
        rr == 1 && cc == 1 && axislegend(ax, "Regime gain"; position=:rb, LEG_KW...)
        axs[rr, cc] = ax
    end
    for rr in 1:3; linkyaxes!(axs[rr, 2], axs[rr, 3]) end   # pairs share a y-axis
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 16); rowgap!(fig.layout, 12)
    save(joinpath(OUT, "rs1_grid_lines.png"), fig); println("  rs1 done")
end

# ── Figure 1-alt: the same outcomes against the effective rank of the DGP ──
function fig1_rank_lines()
    combos = Tuple{Float64,Float64,String}[]
    s = jldopen(joinpath(ROOT, "phase/rho_delta/base/summary.jld2"), "r") do f; (xv=f["xvals"], yv=f["yvals"]) end
    for (xi, r) in enumerate(s.xv), (yi, d) in enumerate(s.yv)
        r == 1.0 && continue
        push!(combos, (r, d, "phase/rho_delta/cells/$(xi-1)_$(yi-1)/base"))
    end
    for r in (0.3, 0.7)   # OAT rho cells at delta = 0.5
        push!(combos, (r, 0.5, "oat/rho=$r/base"))
    end
    pts = [(r90(r, d), r, d, load_mdfs(rel)) for (r, d, rel) in combos]
    shown = [o for o in OUTCOMES if o[1] in
        ("Broker rank correlation", "Rank correlation gap", "Broker output q", "Output gap q")]
    fig = Figure(size=(980, 760))
    Label(fig[0, 1:2], "Late-window means vs. effective rank";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    for (k, (ttl, f)) in enumerate(shown)
        rr = div(k - 1, 2) + 1; cc = mod(k - 1, 2) + 1
        ax = Axis(fig[rr, cc]; title=ttl, xlabel = rr == 2 ? "effective rank r₉₀" : "",
            titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
        for d in (0.0, 0.5, 0.75)
            grp = sort([(x, f(m)) for (x, dd, _, m) in [(p[1], p[3], p[2], p[4]) for p in pts] if dd == d]; by=first)
            isempty(grp) && continue
            scatterlines!(ax, first.(grp), last.(grp); color=DELTA_COLORS[d], linewidth=1.6,
                markersize=11, strokewidth=0.4, strokecolor=:gray30)
        end
        if k == 1
            els = [MarkerElement(marker=:circle, color=DELTA_COLORS[d], markersize=11) for d in (0.0, 0.5, 0.75)]
            axislegend(ax, els, ["δ = 0", "δ = 0.5", "δ = 0.75"]; position=:rt, LEG_KW...)
        end
    end
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 14); rowgap!(fig.layout, 10)
    save(joinpath(OUT, "rs1_rank_lines.png"), fig); println("  rs1alt done")
end

# ── Figure 2: betweenness and access over time (both baselines) + cross-regime scatter ──
function fig2_position_work()
    mb = load_mdfs("oat/rho=0.5/base"); mc = load_mdfs("oat/rho=0.5/capture")
    fig = Figure(size=(1150, 860))
    Label(fig[0, 1:2], "Broker betweenness and access fraction"; fontsize=SUPTITLE_FS,
        font=:bold, tellwidth=false)
    # row 1: time series, from t=20 (first betweenness measurement), 5-obs rolling means
    axa = Axis(fig[1, 1]; title="Betweenness over time", xlabel="period", ylabel="broker betweenness",
        titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS, limits=((30, 201), (0, nothing)))
    axb = Axis(fig[1, 2]; title="Access fraction over time", xlabel="period", ylabel="access fraction",
        titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS, limits=((30, 201), (0, nothing)))
    for (m, col, ls, lbl) in ((mb, COL_BASE_REF, :dash, "no capture"), (mc, COL_CAPTURE, :solid, "capture"))
        per, bw = period_ens(m, d -> d.betweenness)             # measured every 20 periods
        mi = [i for i in eachindex(per) if per[i] % 20 == 0]
        sm = rolling_mean(bw[mi], 5)
        scatterlines!(axa, per[mi], sm; color=col, linestyle=ls, linewidth=2.2,
            markersize=6, label=lbl)
        pa, af = period_ens(m, access_fraction)                 # per-period
        sm2 = rolling_mean(af, 5)
        lines!(axb, pa, sm2; color=col, linestyle=ls, linewidth=2.2, label=lbl)
    end
    axislegend(axa; position=:rb, LEG_KW...); axislegend(axb; position=:rt, LEG_KW...)
    # row 2: cross-regime scatter (base OAT cells), coloured by effective rank
    cellspec = vcat(["oat/rho=$(r)/base" for r in RHO5],
        ["oat/$a/base" for a in ("eta=0.01", "eta=0.02", "eta=0.03", "N=500", "N=1000", "N=1500",
            "reservation_frac=0.4", "reservation_frac=0.6", "reservation_frac=0.9",
            "reservation_frac=1.2", "delta=0.0", "delta=0.75", "k=4", "k=12")])
    bx = Float64[]; ay = Float64[]; rk = Float64[]
    for rel in cellspec
        isfile(joinpath(ROOT, rel, "data.jld2")) || continue
        m = load_mdfs(rel); cfg = load_cfg(rel)
        b = seedstat(m, :betweenness)[1]; a = cell_access(m)
        (isnan(b) || isnan(a)) && continue
        push!(bx, b); push!(ay, a); push!(rk, r90(cfg["rho"], cfg["delta"]))
    end
    axc = Axis(fig[2, 1:2]; title="Across regimes (one dot per regime, late-window means)",
        xlabel="broker betweenness", ylabel="access fraction", titlesize=TITLE_FS,
        xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    sc = scatter!(axc, bx, ay; color=rk, colormap=:viridis, colorrange=(2, 10.5), markersize=13,
        strokewidth=0.4, strokecolor=:gray30)
    Colorbar(fig[2, 3], sc; label="effective rank r₉₀", labelsize=LABEL_FS, ticklabelsize=TICK_FS)
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 14); rowgap!(fig.layout, 14)
    save(joinpath(OUT, "rs2_position_work.png"), fig); println("  rs2 done")
end

# ── Figure 3: structural measures vs informational/output gaps (4 panels) ──
function fig3_advantage()
    # every saved base cell; rho/delta read from each cell's config
    rho = Float64[]; dlt = Float64[]; bw = Float64[]; ac = Float64[]; rg = Float64[]; qg = Float64[]
    for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
        (endswith(root, "base") && "data.jld2" in files) || continue
        m, cfg = jldopen(joinpath(root, "data.jld2"), "r") do f; (f["mdfs"], f["config"]) end
        b = seedstat(m, :betweenness)[1]; isnan(b) && continue
        push!(rho, Float64(cfg["rho"])); push!(dlt, Float64(cfg["delta"]))
        push!(bw, b); push!(ac, cell_access(m))
        push!(rg, seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1])
        push!(qg, qgap(m))
    end
    rks = [r90(r, d) for (r, d) in zip(rho, dlt)]
    msz(k) = 6 + 12 * (k - 2) / 8.5      # r90 in [2, 10.5] -> size 6..18
    xs = [("Broker betweenness", bw), ("Access fraction", ac)]
    ys = [("Rank correlation gap (broker − agent)", rg), ("Output gap q (broker − self)", qg)]
    fig = Figure(size=(1150, 940))
    Label(fig[0, 1:2], "Structural measures vs. informational and output gaps";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    for (ri, (ylab, yv)) in enumerate(ys), (ci, (xlab, xv)) in enumerate(xs)
        ax = Axis(fig[ri, ci]; xlabel = ri == 2 ? xlab : "", ylabel = ci == 1 ? ylab : "",
            title = ri == 1 ? xlab : "", titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
        for rv in sort(unique(rho))
            mm = rho .== rv
            scatter!(ax, xv[mm], yv[mm]; color=(RHO_COLORS[rv], 0.8), markersize=msz.(rks[mm]),
                strokewidth=0.3, strokecolor=:gray30, label="ρ = $rv")
        end
        ri == 1 && ci == 2 && axislegend(ax, "Channel mix"; position=:rt, LEG_KW...)
        if ri == 2 && ci == 1
            els = [MarkerElement(marker=:circle, color=:gray55, markersize=msz(k)) for k in (2, 6, 10)]
            axislegend(ax, els, ["r₉₀ = 2", "r₉₀ = 6", "r₉₀ = 10"], "Effective rank"; position=:rt, LEG_KW...)
        end
    end
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 16); rowgap!(fig.layout, 12)
    save(joinpath(OUT, "rs3_advantage.png"), fig); println("  rs3 done")
end

# ── Figure 4: captured share across sweeps (row 1) and vs the advantage measures (row 2) ──
function fig4_capture()
    capshare(d) = tailmean(d, :principal_mode_share)
    seedstat_f(mdfs, f) = (vs = filter(!isnan, [f(d) for d in mdfs]); isempty(vs) ? (NaN, 0.0) : (mean(vs), std(vs)))
    sweeps = [("ρ (channel mix)", string.(RHO5), ["oat/rho=$(r)/capture" for r in RHO5]),
              ("N (market size)", ["500", "1000", "1500"], ["oat/N=$(n)/capture" for n in (500, 1000, 1500)]),
              ("λ_r (outside option)", ["0.4", "0.6", "0.9"], ["oat/reservation_frac=$(r)/capture" for r in (0.4, 0.6, 0.9)]),
              ("η (turnover)", ["0.01", "0.02", "0.03"], ["oat/eta=$(e)/capture" for e in (0.01, 0.02, 0.03)])]
    fig = Figure(size=(1340, 760))
    Label(fig[0, 1:4], "Captured demand share across regimes"; fontsize=SUPTITLE_FS,
        font=:bold, tellwidth=false)
    for (ci, (name, vals, cells)) in enumerate(sweeps)
        st = [seedstat_f(load_mdfs(c), capshare) for c in cells]
        ax = Axis(fig[1, ci]; title=name, ylabel = ci == 1 ? "captured share of outsourced demand" : "",
            xticks=(1:length(vals), vals), titlesize=TITLE_FS, ylabelsize=LABEL_FS - 1,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, 1.02)))
        x = 1:length(vals)
        band!(ax, x, first.(st) .- last.(st), first.(st) .+ last.(st); color=(COL_CAPTURE, 0.15))
        scatterlines!(ax, x, first.(st); color=COL_CAPTURE, markersize=9)
    end
    # row 2: every capture cell (f_r = 1.2 included), coloured by f_r
    caps = String[]
    for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
        endswith(root, "capture") && ("data.jld2" in files) && push!(caps, root)
    end
    cs = Float64[]; bw = Float64[]; ac = Float64[]; rg = Float64[]; qg = Float64[]; fr = Float64[]
    for d in caps
        m, cfg = jldopen(joinpath(d, "data.jld2"), "r") do f; (f["mdfs"], f["config"]) end
        c = seedstat(m, :principal_mode_share)[1]; isnan(c) && continue
        push!(cs, c); push!(bw, seedstat(m, :betweenness)[1]); push!(ac, cell_access(m))
        push!(rg, seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1])
        push!(qg, qgap(m)); push!(fr, Float64(cfg["reservation_frac"]))
    end
    panels = [("Betweenness", bw), ("Access fraction", ac), ("Rank correlation gap", rg), ("Output gap q", qg)]
    for (ci, (xlab, xv)) in enumerate(panels)
        ax = Axis(fig[2, ci]; title=xlab, ylabel = ci == 1 ? "captured share of outsourced demand" : "",
            titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS - 1,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, 1.05)))
        for v in (0.4, 0.6, 0.9, 1.2)
            mm = fr .== v
            scatter!(ax, xv[mm], cs[mm]; color=FR_COLORS[v], markersize=9,
                strokewidth=0.3, strokecolor=:gray30, label="λ_r = $v")
        end
        ci == 4 && axislegend(ax, "Outside option"; position=:lb, LEG_KW...)
    end
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 12); rowgap!(fig.layout, 14)
    save(joinpath(OUT, "rs4_capture.png"), fig); println("  rs4 done")
end

# ── Figure 5: baseline dynamics, no-capture vs capture, outsourcing in the last column ──
function fig5_dynamics()
    N = 1000
    mb = load_mdfs("oat/rho=0.5/base"); mc = load_mdfs("oat/rho=0.5/capture")
    function ot(mdfs, f)   # ensemble mean, 5-period rolling over full series, display trimming is axis-only
        per = mdfs[1].period
        sm = rolling_mean([nanmean(Float64[f(d)[t] for d in mdfs]) for t in eachindex(per)], 5)
        (per, sm)
    end
    function otb(mdfs)     # betweenness: rolling over the 20-period measurements
        per = mdfs[1].period
        raw = [nanmean(Float64[d.betweenness[t] for d in mdfs]) for t in eachindex(per)]
        mi = [i for i in eachindex(per) if per[i] % 20 == 0]
        (per[mi], rolling_mean(raw[mi], 5))
    end
    yrange(ss...) = (v = filter(!isnan, vcat((s[2][s[1] .>= 30] for s in ss)...));   # displayed window only
        lo = minimum(v); hi = maximum(v); pad = 0.06 * (hi - lo + eps()); (lo - pad, hi + pad))
    mpa(d) = 2 .* d.n_total_matches ./ N
    mpaB, mpaC = ot(mb, mpa), ot(mc, mpa)
    dmnB, dmdB = ot(mb, d -> d.mean_degree), ot(mb, d -> d.median_degree)
    dmnC, dmdC = ot(mc, d -> d.mean_degree), ot(mc, d -> d.median_degree)
    bwB, bwC = otb(mb), otb(mc)
    acB, acC = ot(mb, access_fraction), ot(mc, access_fraction)
    osB, osC = ot(mb, d -> d.outsourcing_rate), ot(mc, d -> d.outsourcing_rate)
    capC = ot(mc, d -> d.principal_mode_share .* d.outsourcing_rate)   # captured share of TOTAL demand
    yl = (yrange(mpaB, mpaC), yrange(dmnB, dmdB, dmnC, dmdC), yrange(bwB, bwC), yrange(acB, acC), (0, 1.02))

    fig = Figure(size=(1220, 560))
    Label(fig[0, 1:5], "Baseline dynamics over time"; fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    mk(r, c, ttl, ylim) = Axis(fig[r, c]; title=ttl, xlabel = r == 2 ? "period" : "",
        xticks=50:50:200, titlesize=TITLE_FS, xlabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS, limits=((30, 201), ylim))
    drw!(ax, s, col; lbl="", ls=:solid, pts=false) = pts ?
        scatterlines!(ax, s[1], s[2]; color=col, linewidth=2.2, markersize=6, linestyle=ls, label=lbl) :
        lines!(ax, s[1], s[2]; color=col, linewidth=2.2, linestyle=ls, label=lbl)
    # top row: no capture
    drw!(mk(1, 1, "Matches per agent", yl[1]), mpaB, COL_DIAG)
    let a = mk(1, 2, "Network degree", yl[2])
        drw!(a, dmnB, COL_BROKER; lbl="mean"); drw!(a, dmdB, COL_AGENT; lbl="median")
        axislegend(a; position=:rb, LEG_KW...)
    end
    drw!(mk(1, 3, "Broker betweenness", yl[3]), bwB, COL_GAP; pts=true)
    drw!(mk(1, 4, "Access fraction", yl[4]), acB, COL_ACCESS)
    drw!(mk(1, 5, "Outsourcing rate", yl[5]), osB, COL_BROKER)
    # bottom row: capture
    drw!(mk(2, 1, "", yl[1]), mpaC, COL_DIAG)
    let a = mk(2, 2, "", yl[2]); drw!(a, dmnC, COL_BROKER); drw!(a, dmdC, COL_AGENT) end
    drw!(mk(2, 3, "", yl[3]), bwC, COL_GAP; pts=true)
    drw!(mk(2, 4, "", yl[4]), acC, COL_ACCESS)
    let a = mk(2, 5, "", yl[5])
        drw!(a, osC, COL_BROKER; lbl="outsourcing rate")
        drw!(a, capC, COL_CAPTURE; lbl="captured share\nof total demand", ls=:dash)
        axislegend(a; position=:rb, LEG_KW...)
    end
    Label(fig[1, 0], "No capture"; rotation=π/2, font=:bold, fontsize=LABEL_FS, tellheight=false)
    Label(fig[2, 0], "Capture"; rotation=π/2, font=:bold, fontsize=LABEL_FS, tellheight=false)
    rowsize!(fig.layout, 0, Fixed(28)); colsize!(fig.layout, 0, Fixed(22))
    colgap!(fig.layout, 12); rowgap!(fig.layout, 12)
    save(joinpath(OUT, "rs5_dynamics.png"), fig); println("  rs5 done")
end

# ── Figure 6: agent degree and broker betweenness over time, no-capture vs capture ──
function fig6_capture_topology()
    mb = load_mdfs("oat/rho=0.5/base"); mc = load_mdfs("oat/rho=0.5/capture")
    roll(per, v) = (per, rolling_mean(v, 5))
    fig = Figure(size=(820, 640))
    Label(fig[0, 1], "Network topology at baseline: no capture vs. capture";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    ax1 = Axis(fig[1, 1]; ylabel="agent degree", ylabelsize=LABEL_FS + 1,
        xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=((30, 201), (0, nothing)))
    for (m, col, ls, lbl) in ((mb, COL_BASE_REF, :dash, "no capture"), (mc, COL_CAPTURE, :solid, "capture"))
        p1, v1 = period_ens(m, d -> d.mean_degree); lines!(ax1, roll(p1, v1)...; color=col, linestyle=ls, linewidth=2.4, label="$lbl, mean")
        p2, v2 = period_ens(m, d -> d.median_degree); lines!(ax1, roll(p2, v2)...; color=(col, 0.55), linestyle=ls, linewidth=1.6, label="$lbl, median")
    end
    axislegend(ax1; position=:rt, LEG_KW...)
    ax2 = Axis(fig[2, 1]; xlabel="period", ylabel="broker betweenness", xlabelsize=LABEL_FS + 1,
        ylabelsize=LABEL_FS + 1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
        limits=((30, 201), (0, nothing)))
    for (m, col, ls, lbl) in ((mb, COL_BASE_REF, :dash, "no capture"), (mc, COL_CAPTURE, :solid, "capture"))
        per, bw = period_ens(m, d -> d.betweenness)
        mi = [i for i in eachindex(per) if per[i] % 20 == 0]
        sm = rolling_mean(bw[mi], 5)
        scatterlines!(ax2, per[mi], sm; color=col, linestyle=ls, linewidth=2.2,
            markersize=6, label=lbl)
    end
    axislegend(ax2; position=:rb, LEG_KW...)
    rowsize!(fig.layout, 0, Fixed(28)); rowgap!(fig.layout, 10)
    save(joinpath(OUT, "rs6_capture_topology.png"), fig); println("  rs6 done")
end

for (name, f) in (("rs1", fig1_grid_lines), ("rs1alt", fig1_rank_lines), ("rs2", fig2_position_work),
                  ("rs3", fig3_advantage), ("rs4", fig4_capture), ("rs5", fig5_dynamics),
                  ("rs6", fig6_capture_topology))
    try
        f()
    catch e
        println("  $name FAILED: ", sprint(showerror, e)[1:min(end, 400)])
    end
end
println("summary figures done")
