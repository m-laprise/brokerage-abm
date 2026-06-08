"""
    questions_figures.jl

Build the purpose-made figures for the questions report. Reads only saved sweep
data (phase summary tensors, OAT cell aggregates, capture base-vs-M1 cells) plus
the adverse-selection re-run's per-agent quality arrays. CairoMakie only (no
TransientBrokerage / Enzyme). Each figure has axis labels, a legend, and a title;
captions live in the LaTeX.
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # using CairoMakie + COL_* + helpers
using JLD2
using Statistics: mean, std, cor, cov, var

const ROOT = "/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438"
const OUT = joinpath(ROOT, "report")
const RHO_COLORS = Dict(0.0 => :seagreen, 0.3 => :mediumaquamarine, 0.5 => :goldenrod,
                        0.7 => :darkorange, 1.0 => :firebrick)
const RHO5 = [0.0, 0.3, 0.5, 0.7, 1.0]   # OAT rho axis (denser than the phase rho grids)

nanmean(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
tailmean(df, col; t0=30) = nanmean(df[df.period .> t0, col])
seedstat(mdfs, col) = (vs = filter(!isnan, [tailmean(d, col) for d in mdfs]); (mean(vs), std(vs)))
function period_avg(mdfs, col)
    per = mdfs[1].period
    return per, [nanmean(Float64[d[t, col] for d in mdfs]) for t in eachindex(per)]
end
load_summary(pair, model) = jldopen(joinpath(ROOT, "phase", pair, model, "summary.jld2"), "r") do f
    (t=f["tensors"], xv=f["xvals"], yv=f["yvals"], xk=f["xkey"], yk=f["ykey"])
end
load_mdfs(reldir) = jldopen(joinpath(ROOT, reldir, "data.jld2"), "r") do f; f["mdfs"]; end

# ── Fig 1: structural vs informational advantage, anti-correlated across rho ──
function fig_edges_scatter()
    bx = Float64[]; gy = Float64[]; rh = Float64[]; qb = Float64[]
    for pair in ("rho_eta", "rho_N", "rho_r", "rho_delta")
        s = load_summary(pair, "base")
        s.xk == "rho" || continue
        betw = s.t[:betweenness]; rg = s.t[:rank_gap]; qq = s.t[:q_broker]
        for i in axes(betw, 1), j in axes(betw, 2)
            (isnan(betw[i, j]) || isnan(rg[i, j]) || isnan(qq[i, j])) && continue
            push!(bx, betw[i, j]); push!(gy, rg[i, j]); push!(rh, Float64(s.xv[i]))
            push!(qb, qq[i, j])
        end
    end
    # The intermediate channel-mix levels (0.3, 0.7) live only on the OAT axis; add
    # them as single baseline points so the trend is visible at all five levels.
    for rv in (0.3, 0.7)
        mdfs = load_mdfs("oat/rho=$(rv)/base")
        bw = seedstat(mdfs, :betweenness)[1]
        rg = seedstat(mdfs, :broker_holdout_rank)[1] - seedstat(mdfs, :agent_holdout_rank)[1]
        qq = seedstat(mdfs, :q_broker_standard_mean)[1]
        (isnan(bw) || isnan(rg) || isnan(qq)) && continue
        push!(bx, bw); push!(gy, rg); push!(rh, rv); push!(qb, qq)
    end
    r = cor(bx, gy)
    qmin, qmax = extrema(qb)
    msz(q) = 7 + 16 * (q - qmin) / (qmax - qmin + 1e-9)   # marker size = broker realized output
    fig = Figure(size=(760, 540))
    ax = Axis(fig[1, 1];
        title="Structural and informational advantage are anti-correlated across the channel mix",
        xlabel="Broker betweenness  (structural advantage)",
        ylabel="Holdout rank gap, broker − agent  (informational advantage)",
        titlesize=TITLE_FS+1, xlabelsize=LABEL_FS+1, ylabelsize=LABEL_FS+1,
        xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    for rv in RHO5
        m = rh .== rv
        scatter!(ax, bx[m], gy[m]; color=(RHO_COLORS[rv], 0.75), markersize=msz.(qb[m]),
            strokewidth=0.4, strokecolor=:gray30, label="ρ = $(rv)")
    end
    text!(ax, 0.02, 0.02;
        text="Pearson r = $(round(r; digits=2))  (each point = one base sweep cell);  marker size = broker realized output q",
        space=:relative, align=(:left, :bottom), fontsize=TICK_FS, color=:gray30)
    axislegend(ax, "Channel mix"; position=:lt, LEG_KW...)
    qref = [round(qmin; digits=1), round((qmin + qmax) / 2; digits=1), round(qmax; digits=1)]
    Legend(fig[1, 1],
        [MarkerElement(marker=:circle, color=:gray55, markersize=msz(q), strokewidth=0.4,
            strokecolor=:gray30) for q in qref],
        ["q = $(q)" for q in qref], "Broker realized output";
        tellwidth=false, tellheight=false, halign=:right, valign=:top, framevisible=true,
        labelsize=9, titlesize=10, patchsize=(26, 22), rowgap=4)
    save(joinpath(OUT, "q1_edges_scatter.png"), fig); println("  q1 done")
end

# ── Fig 2: realized output by channel + broker channel share, vs rho ──
function fig_realized_channel()
    rhos = RHO5
    n = length(rhos)
    qs = Tuple{Float64,Float64}[]; qb = Tuple{Float64,Float64}[]; share = Float64[]
    for r in rhos
        mdfs = load_mdfs("oat/rho=$(r)/base")
        push!(qs, seedstat(mdfs, :q_self_mean))
        push!(qb, seedstat(mdfs, :q_broker_standard_mean))
        push!(share, seedstat(mdfs, :outsourcing_rate)[1])
    end
    fig = Figure(size=(720, 560))
    ax = Axis(fig[1, 1]; title="Realized match output is higher through the broker, except at ρ = 1",
        ylabel="Mean realized match output  q", xticks=(1:n, string.(rhos)),
        titlesize=TITLE_FS+1, ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    errorbars!(ax, 1:n, first.(qs), last.(qs); color=COL_AGENT, whiskerwidth=8)
    errorbars!(ax, 1:n, first.(qb), last.(qb); color=COL_BROKER, whiskerwidth=8)
    scatterlines!(ax, 1:n, first.(qs); color=COL_AGENT, markersize=11, label="Self-search")
    scatterlines!(ax, 1:n, first.(qb); color=COL_BROKER, markersize=11, label="Broker")
    axislegend(ax; position=:lb, LEG_KW...)
    ax2 = Axis(fig[2, 1]; xlabel="Channel mix  ρ  (0 = relational … 1 = quality)",
        ylabel="Broker channel share", xticks=(1:n, string.(rhos)), limits=(nothing, (0, 1)),
        xlabelsize=LABEL_FS+1, ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    barplot!(ax2, 1:n, share; color=(COL_BROKER, 0.55), strokewidth=0.5)
    for (k, v) in enumerate(share)
        text!(ax2, k, v + 0.03; text="$(round(Int, 100v))%", align=(:center, :bottom), fontsize=TICK_FS)
    end
    rowsize!(fig.layout, 2, Relative(0.28)); rowgap!(fig.layout, 6)
    save(joinpath(OUT, "q2_realized_channel.png"), fig); println("  q2 done")
end

# ── Fig 4: prediction rank correlation, R², bias, broker vs agent, across ρ ──
function fig_ordering_level()
    rhos = RHO5
    M = [load_mdfs("oat/rho=$(r)/base") for r in rhos]
    bv(c) = [seedstat(m, c) for m in M]
    x = 1:length(rhos)
    # (title, broker col, partner col, partner label, ylim, draw 0-line, draw bands)
    panels = [
        ("Prediction rank correlation\n(which match is better)",
            :broker_holdout_rank, :agent_holdout_rank, "Agent", (0, 1.0), false, true),
        ("Prediction R²\n(how good the match will be)",
            :broker_holdout_r2, :agent_holdout_r2, "Agent", (-5.5, 1.1), true, false),
        ("Prediction bias\n(systematic over/under-prediction)",
            :broker_holdout_bias, :agent_holdout_bias, "Agent", nothing, true, true),
        ("Realized output q\n(by channel)",
            :q_broker_standard_mean, :q_self_mean, "Self-search", (0, nothing), false, true),
    ]
    fig = Figure(size=(1120, 780))
    Label(fig[0, 1:2],
        "The broker ranks matches well, but R² is weak";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    for (k, (ttl, bc, ac, alab, ylim, zero, bands)) in enumerate(panels)
        rr = div(k - 1, 2) + 1; cc = mod(k - 1, 2) + 1
        ax = Axis(fig[rr, cc]; title=ttl, xlabel = rr == 2 ? "ρ (channel mix)" : "",
            xticks=(x, string.(rhos)), titlesize=TITLE_FS, xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, ylim))
        zero && hlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=1)
        b = bv(bc); a = bv(ac)
        if bands
            band!(ax, x, first.(b) .- last.(b), first.(b) .+ last.(b); color=(COL_BROKER, 0.12))
            band!(ax, x, first.(a) .- last.(a), first.(a) .+ last.(a); color=(COL_AGENT, 0.12))
        end
        scatterlines!(ax, x, first.(b); color=COL_BROKER, markersize=9, label="Broker")
        scatterlines!(ax, x, first.(a); color=COL_AGENT, markersize=9, label=alab)
        k == 2 && text!(ax, 0.5, 0.03; text="R² unstable at ρ ≤ 0.3 (seed σ up to 9); bands omitted",
            space=:relative, align=(:center, :bottom), fontsize=TICK_FS-1, color=:gray35)
        k == 1 && axislegend(ax; position=:rc, LEG_KW...)
    end
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 16); rowgap!(fig.layout, 12)
    save(joinpath(OUT, "q4_ordering_level.png"), fig); println("  q4 done")
end

# ── Fig 5a: capture gated by the outside option λ_r ──
function fig_capture_gate()
    s = load_summary("r_N", "capture")   # xkey = reservation_frac, ykey = N
    lr = Float64.(s.xv)
    share = [nanmean(s.t[:principal_share][i, :]) for i in eachindex(lr)]
    surp = [nanmean(s.t[:capture_surplus][i, :]) for i in eachindex(lr)]
    fig = Figure(size=(720, 480))
    ax = Axis(fig[1, 1];
        title="Capture is gated by the clients' outside option λ_r",
        xlabel="Outside option  λ_r", ylabel="Captured-demand share",
        xticks=(1:length(lr), string.(lr)), limits=(nothing, (0, 1.02)),
        titlesize=TITLE_FS+1, xlabelsize=LABEL_FS+1, ylabelsize=LABEL_FS+1,
        xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, ytickcolor=COL_CAPTURE, yticklabelcolor=COL_CAPTURE)
    ax2 = Axis(fig[1, 1]; ylabel="Per-unit capture surplus", yaxisposition=:right,
        ylabelcolor=COL_GAP, yticklabelcolor=COL_GAP, ytickcolor=COL_GAP,
        ylabelsize=LABEL_FS+1, yticklabelsize=TICK_FS)
    hidespines!(ax2); hidexdecorations!(ax2); linkxaxes!(ax, ax2)
    l1 = scatterlines!(ax, 1:length(lr), share; color=COL_CAPTURE, markersize=11)
    hlines!(ax2, [0.0]; color=:gray55, linestyle=:dash, linewidth=1)
    l2 = scatterlines!(ax2, 1:length(lr), surp; color=COL_GAP, markersize=11, marker=:rect)
    axislegend(ax, [l1, l2], ["Captured-demand share (left)", "Capture surplus (right)"];
        position=:rc, LEG_KW...)
    save(joinpath(OUT, "q5a_capture_gate.png"), fig); println("  q5a done")
end

# ── Fig 5b: capture lock-in (open-market degree + broker betweenness over time) ──
function fig_lockin()
    base = load_mdfs("oat/rho=0.5/base"); cap = load_mdfs("oat/rho=0.5/capture")
    fig = Figure(size=(720, 560))
    pb, db = period_avg(base, :mean_degree); pc, dc = period_avg(cap, :mean_degree)
    ax1 = Axis(fig[1, 1]; title="Capture is lock-in: it halves the open market and sustains broker centrality",
        ylabel="Open-market mean degree", titlesize=TITLE_FS+1, ylabelsize=LABEL_FS+1,
        xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, nothing)))
    lines!(ax1, pb, db; color=COL_BASE_REF, linewidth=2, linestyle=:dash, label="Base model")
    lines!(ax1, pc, dc; color=COL_CAPTURE, linewidth=2.2, label="With capture (Model 1)")
    vlines!(ax1, [30]; color=:gray70, linestyle=:dot); axislegend(ax1; position=:rt, LEG_KW...)
    pb2, bb = period_avg(base, :betweenness); pc2, bc = period_avg(cap, :betweenness)
    ax2 = Axis(fig[2, 1]; xlabel="Period", ylabel="Broker betweenness",
        xlabelsize=LABEL_FS+1, ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
        limits=(nothing, (0, nothing)))
    lines!(ax2, pb2, bb; color=COL_BASE_REF, linewidth=2, linestyle=:dash, label="Base model")
    lines!(ax2, pc2, bc; color=COL_CAPTURE, linewidth=2.2, label="With capture (Model 1)")
    vlines!(ax2, [30]; color=:gray70, linestyle=:dot)
    rowgap!(fig.layout, 6)
    save(joinpath(OUT, "q5b_lockin.png"), fig); println("  q5b done")
end

# ── Fig 6: broker vs agent performance and gap across the relevant regimes ──
function fig_performance_grid()
    regimes = [
        ("ρ (channel mix)",  ["0.0", "0.3", "0.5", "0.7", "1.0"],
            ["oat/rho=$(r)/base" for r in RHO5]),
        ("N (market size)",  ["500", "1000", "1500"],
            ["oat/N=500/base", "oat/N=1000/base", "oat/N=1500/base"]),
        ("η (turnover)",     ["0.01", "0.02", "0.03"],
            ["oat/eta=0.01/base", "oat/eta=0.02/base", "oat/eta=0.03/base"]),
        ("k_G (density)",    ["4", "6", "12"],
            ["oat/k=4/base", "oat/rho=0.5/base", "oat/k=12/base"]),
    ]
    rows = [  # (row label, broker col, partner col, partner legend, ylims)
        ("Holdout rank", :broker_holdout_rank, :agent_holdout_rank, "Agent", (0, 1.0)),
        ("Holdout RMSE", :broker_holdout_rmse, :agent_holdout_rmse, "Agent", (0, nothing)),
        ("Realized output q", :q_broker_standard_mean, :q_self_mean, "Self-search", (nothing, nothing)),
    ]
    fig = Figure(size=(1400, 980))
    Label(fig[0, 1:4], "Broker vs. agent performance across regimes";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    println("\n=== performance gaps (broker advantage), tail means ===")
    for (ri, row) in enumerate(rows), (ci, reg) in enumerate(regimes)
        rlab, bcol, pcol, plab, ylim = row
        name, vals, cells = reg
        bv = Tuple{Float64,Float64}[]; pv = Tuple{Float64,Float64}[]
        for cell in cells
            mdfs = load_mdfs(cell)
            push!(bv, seedstat(mdfs, bcol)); push!(pv, seedstat(mdfs, pcol))
        end
        ax = Axis(fig[ri, ci]; xticks=(1:length(vals), vals),
            title = ri == 1 ? name : "", titlesize=TITLE_FS,
            ylabel = ci == 1 ? rlab : "", ylabelsize=LABEL_FS+1,
            xlabel = ri == 3 ? "level" : "", xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, ylim))
        x = 1:length(vals)
        band!(ax, x, first.(bv) .- last.(bv), first.(bv) .+ last.(bv); color=(COL_BROKER, 0.12))
        band!(ax, x, first.(pv) .- last.(pv), first.(pv) .+ last.(pv); color=(COL_AGENT, 0.12))
        scatterlines!(ax, x, first.(bv); color=COL_BROKER, markersize=9, label="Broker")
        scatterlines!(ax, x, first.(pv); color=COL_AGENT, markersize=9, label=plab)
        gap = ri == 2 ? first.(pv) .- first.(bv) : first.(bv) .- first.(pv)  # positive = broker better
        println("  $(rpad(rlab,18)) $(rpad(name,18)) gap(broker-better) = ",
            join(round.(gap; digits=3), ", "))
    end
    Legend(fig[4, 1:4],
        [LineElement(color=COL_BROKER), LineElement(color=COL_AGENT)],
        ["Broker", "Agent (rows 1--2) / Self-search (row 3)"];
        orientation=:horizontal, framevisible=false, labelsize=10)
    rowgap!(fig.layout, 8); colgap!(fig.layout, 12)
    rowsize!(fig.layout, 0, Fixed(30)); rowsize!(fig.layout, 4, Fixed(20))
    save(joinpath(OUT, "q6_performance_grid.png"), fig); println("  q6 done")
end

# ── Stage-setting: market structure, capture, and access fraction across regimes ──
seedstat_f(mdfs, f) = (vs = filter(!isnan, [f(d) for d in mdfs]);
    isempty(vs) ? (NaN, 0.0) : (mean(vs), std(vs)))
function tail_overall_q(df; t0=30)  # count-weighted realized output over both channels
    s = df[df.period .> t0, :]
    tot = 0.0; cnt = 0.0
    @inbounds for i in 1:size(s, 1)
        ns = Float64(s.n_self_matches[i]); nb = Float64(s.n_broker_standard[i])
        (ns > 0 && !isnan(s.q_self_mean[i])) && (tot += ns * s.q_self_mean[i])
        (nb > 0 && !isnan(s.q_broker_standard_mean[i])) && (tot += nb * s.q_broker_standard_mean[i])
        cnt += ns + nb
    end
    cnt > 0 ? tot / cnt : NaN
end
access_tail(df; t0=30) = (s = df[df.period .> t0, :];
    nanmean([(t = s.access_count[i] + s.assessment_count[i]; t > 0 ? s.access_count[i] / t : NaN)
             for i in 1:size(s, 1)]))
ax_rho(m) = ("ρ (channel mix)", string.(RHO5), ["oat/rho=$(r)/$(m)" for r in RHO5])
ax_N(m)   = ("N (market size)", ["500", "1000", "1500"], ["oat/N=$(n)/$(m)" for n in (500, 1000, 1500)])
ax_eta(m) = ("η (turnover)", ["0.01", "0.02", "0.03"], ["oat/eta=$(e)/$(m)" for e in (0.01, 0.02, 0.03)])
ax_lr(m)  = ("λ_r (outside option)", ["0.4", "0.6", "0.9", "1.2"], ["oat/reservation_frac=$(r)/$(m)" for r in (0.4, 0.6, 0.9, 1.2)])
ax_k(m)   = ("k_G (density)", ["4", "6", "12"], ["oat/k=4/$(m)", "oat/rho=0.5/$(m)", "oat/k=12/$(m)"])
ax_kappa() = ("κ_max (confidence)", ["0.4", "0.5", "0.65"], ["oat/kappa_max=$(k)/capture" for k in (0.4, 0.5, 0.65)])

function panel_row!(fig, row, axlist, f, ylabel, ylim; color, header=false)
    for (ci, (name, vals, cells)) in enumerate(axlist)
        st = [seedstat_f(load_mdfs(c), f) for c in cells]
        ax = Axis(fig[row, ci]; ylabel = ci == 1 ? ylabel : "", title = header ? name : "",
            xticks=(1:length(vals), vals), titlesize=TITLE_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, ylim))
        x = 1:length(vals)
        band!(ax, x, first.(st) .- last.(st), first.(st) .+ last.(st); color=(color, 0.15))
        scatterlines!(ax, x, first.(st); color=color, markersize=9)
    end
end

# ── Baseline dynamics over time: base (top) vs capture (bottom), capture-share extra ──
function fig_market_dynamics()
    N = 1000
    mb = load_mdfs("oat/rho=0.5/base"); mc = load_mdfs("oat/rho=0.5/capture")
    # ensemble mean per period, smoothed by a 5-period rolling mean over the FULL series
    # (so t=30 uses the burn-in tail), then sliced to the post-burn-in window t>=30.
    function ot(mdfs, f)
        per = mdfs[1].period
        raw = [nanmean(Float64[f(d)[t] for d in mdfs]) for t in eachindex(per)]
        sm = rolling_mean(raw, 5)
        idx = findall(>=(30), per)
        (per[idx], sm[idx])
    end
    # betweenness is a cached network measure (recomputed every 20 periods and held in
    # between), so smooth over the last 5 MEASUREMENTS, not the last 5 periods.
    function otb(mdfs)
        per = mdfs[1].period
        raw = [nanmean(Float64[d.betweenness[t] for d in mdfs]) for t in eachindex(per)]
        mi = [i for i in eachindex(per) if per[i] % 20 == 0]   # measurement periods (M=20)
        sm = rolling_mean(raw[mi], 5)
        keep = per[mi] .>= 30
        (per[mi][keep], sm[keep])
    end
    yrange(ss...) = (v = filter(!isnan, vcat((s[2] for s in ss)...));
        lo = minimum(v); hi = maximum(v); pad = 0.06 * (hi - lo + eps()); (lo - pad, hi + pad))
    mpa(d) = 2 .* d.n_total_matches ./ N
    degmn(d) = d.mean_degree; degmd(d) = d.median_degree
    capsh(d) = d.principal_mode_share
    # series (top = base, bottom = capture)
    mpaB, mpaC = ot(mb, mpa), ot(mc, mpa)
    dmnB, dmdB, dmnC, dmdC = ot(mb, degmn), ot(mb, degmd), ot(mc, degmn), ot(mc, degmd)
    bwB, bwC = otb(mb), otb(mc)
    acB, acC = ot(mb, access_fraction), ot(mc, access_fraction)
    capC = ot(mc, capsh)
    # shared y-range per column (cols 1-4 pair base/capture; col 5 capture-only)
    yl = (yrange(mpaB, mpaC), yrange(dmnB, dmdB, dmnC, dmdC), yrange(bwB, bwC), yrange(acB, acC))

    fig = Figure(size=(1220, 560))
    Label(fig[0, 1:5], "Baseline dynamics over time"; fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    mk(r, c, ttl, ylim) = Axis(fig[r, c]; title=ttl, xlabel = r == 2 ? "period" : "",
        xticks=50:50:200, titlesize=TITLE_FS, xlabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS, limits=((30, 201), ylim))
    drw!(ax, s, col; lbl="", pts=false) = pts ?
        scatterlines!(ax, s[1], s[2]; color=col, linewidth=2.2, markersize=6, label=lbl) :
        lines!(ax, s[1], s[2]; color=col, linewidth=2.2, label=lbl)
    # top row: base model
    drw!(mk(1, 1, "Matches per agent", yl[1]), mpaB, COL_DIAG)
    let a = mk(1, 2, "Network degree", yl[2])
        drw!(a, dmnB, COL_BROKER; lbl="mean"); drw!(a, dmdB, COL_AGENT; lbl="median")
        axislegend(a; position=:rb, LEG_KW...)
    end
    drw!(mk(1, 3, "Broker betweenness", yl[3]), bwB, COL_GAP; pts=true)
    drw!(mk(1, 4, "Access fraction", yl[4]), acB, COL_ACCESS)
    Label(fig[1, 5], "(capture only)"; fontsize=TICK_FS, color=:gray70, tellwidth=false, tellheight=false)
    # bottom row: with capture (Model 1)
    drw!(mk(2, 1, "", yl[1]), mpaC, COL_DIAG)
    let a = mk(2, 2, "", yl[2]); drw!(a, dmnC, COL_BROKER); drw!(a, dmdC, COL_AGENT) end
    drw!(mk(2, 3, "", yl[3]), bwC, COL_GAP; pts=true)
    drw!(mk(2, 4, "", yl[4]), acC, COL_ACCESS)
    drw!(mk(2, 5, "Captured-demand share", (0, 1.02)), capC, COL_CAPTURE)
    Label(fig[1, 0], "Base model"; rotation=π/2, font=:bold, fontsize=LABEL_FS, tellheight=false)
    Label(fig[2, 0], "With capture"; rotation=π/2, font=:bold, fontsize=LABEL_FS, tellheight=false)
    rowsize!(fig.layout, 0, Fixed(28)); colsize!(fig.layout, 0, Fixed(22))
    colgap!(fig.layout, 12); rowgap!(fig.layout, 12)
    save(joinpath(OUT, "qM0_market_dynamics.png"), fig); println("  qM0 done")
end

# ── Co-variation with capture across regimes: structural vs. informational (bars) ──
function fig_capture_covary()
    nm(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
    cv(ms, c) = (vs = filter(!isnan, [tailmean(d, c) for d in ms]); isempty(vs) ? NaN : mean(vs))
    acc(ms) = (vs = filter(!isnan, [nm(access_fraction(d)[d.period .> 30]) for d in ms]); isempty(vs) ? NaN : mean(vs))
    caps = String[]
    for sub in ("oat", "phase")
        b = joinpath(ROOT, sub); isdir(b) || continue
        for (r, _, fs) in walkdir(b); endswith(r, "capture") && ("data.jld2" in fs) && push!(caps, r) end
    end
    cap=Float64[]; out=Float64[]; los=Float64[]; brk=Float64[]; gap=Float64[]; qb=Float64[]; qs=Float64[]; bw=Float64[]; ac=Float64[]; lr=Float64[]
    for d in caps
        ms, cfg = jldopen(joinpath(d, "data.jld2"), "r") do f; (f["mdfs"], f["config"]) end
        push!(cap, cv(ms,:principal_mode_share)); push!(out, cv(ms,:outsourcing_rate)); push!(los, cv(ms,:capture_loss_rate))
        push!(brk, cv(ms,:broker_holdout_rank)); push!(gap, cv(ms,:rank_gap)); push!(qb, cv(ms,:q_broker_standard_mean))
        push!(qs, cv(ms,:q_self_mean)); push!(bw, cv(ms,:betweenness)); push!(ac, acc(ms))
        push!(lr, Float64(cfg["reservation_frac"]))
    end
    sprho(x, y) = (k = findall(i -> !isnan(x[i]) && !isnan(y[i]), eachindex(x));
        length(k) < 4 ? NaN : cor(Float64.(sortperm(sortperm(x[k]))), Float64.(sortperm(sortperm(y[k])))))
    outgap = qb .- qs
    ratevecs = [cap, out]
    ratenames = ["captured-demand share", "outsourcing rate"]
    ratecols = [COL_CAPTURE, COL_BROKER]
    function barpanel!(ax, items; legend=false)   # items = [(label, vec), ...]
        n = length(items)
        for (ri, rv) in enumerate(ratevecs)
            barplot!(ax, 1:n, [sprho(rv, v) for (_, v) in items]; dodge=fill(ri, n),
                n_dodge=length(ratevecs), color=ratecols[ri], direction=:x, dodge_gap=0.06,
                label=ratenames[ri])
        end
        vlines!(ax, [0]; color=:gray40, linewidth=1.2)
        ax.yticks = (1:n, [l for (l, _) in items]); ax.yreversed = true; xlims!(ax, -1, 1)
        legend && axislegend(ax, "vs. capture item"; position=:lt, LEG_KW...)
    end
    fig = Figure(size=(1320, 1120))
    Label(fig[0, 1:4], "Co-variation with capture across regimes (94 cells)";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    # ── Row 1: Spearman-ρ summary bars ──
    axS = Axis(fig[1, 1:2]; title="Structural advantage", xlabel="Spearman ρ", titlesize=TITLE_FS,
        xlabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    barpanel!(axS, [("betweenness", bw), ("access fraction", ac)]; legend=true)
    axI = Axis(fig[1, 3:4]; title="Informational advantage", xlabel="Spearman ρ", titlesize=TITLE_FS,
        xlabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    barpanel!(axI, [("broker rank", brk), ("rank gap", gap), ("output: broker", qb), ("output gap", outgap)])
    # ── Rows 2-3: scatter of each key measure (x) vs. each rate (y), coloured by λ_r ──
    scatmetrics = [("access fraction", ac), ("betweenness", bw), ("rank gap", gap), ("output gap", outgap)]
    raterows = [("outsourcing rate", out), ("captured-demand share", cap)]
    row2 = Axis[]; row3 = Axis[]; local sc
    for (ci, (mlab, mv)) in enumerate(scatmetrics)
        col = Axis[]
        for (ri, (rlab, rv)) in enumerate(raterows)
            ax = Axis(fig[ri + 1, ci]; title = ri == 1 ? mlab : "", ylabel = ci == 1 ? rlab : "",
                titlesize=TITLE_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
            k = findall(i -> !isnan(mv[i]) && !isnan(rv[i]) && !isnan(lr[i]), eachindex(mv))
            x = mv[k]; y = rv[k]
            sc = scatter!(ax, x, y; color=lr[k], colormap=:viridis, colorrange=(0.4, 1.2), markersize=8)
            b = cov(x, y) / var(x); intc = mean(y) - b * mean(x); xr = [minimum(x), maximum(x)]
            lines!(ax, xr, intc .+ b .* xr; color=:black, linewidth=2)               # OLS fit
            text!(ax, 0.04, 0.96; text="ρ=$(round(sprho(mv, rv); digits=2))", space=:relative,
                align=(:left, :top), color=:gray20, fontsize=TICK_FS, font=:bold)
            push!(col, ax); ri == 1 ? push!(row2, ax) : push!(row3, ax)
        end
        linkxaxes!(col...)
    end
    linkyaxes!(row2...); linkyaxes!(row3...)
    foreach(ax -> hidexdecorations!(ax; grid=false), row2)
    foreach(ci -> (hideydecorations!(row2[ci]; grid=false); hideydecorations!(row3[ci]; grid=false)), 2:4)
    Colorbar(fig[4, 1:4], sc; label="client outside option  λ_r", vertical=false,
        labelsize=LABEL_FS, ticklabelsize=TICK_FS, height=12, ticks=[0.4, 0.6, 0.9, 1.2])
    rowsize!(fig.layout, 0, Fixed(28)); rowsize!(fig.layout, 4, Fixed(44))
    rowgap!(fig.layout, 12); colgap!(fig.layout, 14)
    save(joinpath(OUT, "qM5_capture_covary.png"), fig); println("  qM5 done")
end

# ── Over-time co-variation with capture: stacked time series, baseline capture run ──
function fig_capture_covary_time()
    nmf(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
    ms = load_mdfs("oat/rho=0.5/capture"); per = ms[1].period
    # ensemble mean per period, then a 5-period rolling mean for legibility (as in qM0)
    ens(c) = rolling_mean([nmf([d[t, c] for d in ms]) for t in eachindex(per)], 5)
    ensacc() = rolling_mean([nmf([access_fraction(d)[t] for d in ms]) for t in eachindex(per)], 5)
    cap = ens(:principal_mode_share); brk = ens(:broker_holdout_rank); gap = ens(:rank_gap)
    qb = ens(:q_broker_standard_mean); qs = ens(:q_self_mean); bw = ens(:betweenness); ac = ensacc()
    k = per .>= 30; p = per[k]
    fig = Figure(size=(860, 880))
    Label(fig[0, 1], "The broker's advantage and capture, over time"; fontsize=SUPTITLE_FS,
        font=:bold, tellwidth=false)
    capref!(ax) = lines!(ax, p, cap[k]; color=:gray45, linestyle=:dash, linewidth=2.5,
        label="captured-demand share")
    ax1 = Axis(fig[1, 1]; title="Structural advantage", ylabel="value", titlesize=TITLE_FS,
        ylabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    lines!(ax1, p, bw[k]; color=COL_GAP, linewidth=2.2, label="betweenness")
    lines!(ax1, p, ac[k]; color=COL_ACCESS, linewidth=2.2, label="access fraction"); capref!(ax1)
    axislegend(ax1; position=:rb, LEG_KW...)
    ax2 = Axis(fig[2, 1]; title="Informational advantage — prediction ordering",
        ylabel="rank correlation", titlesize=TITLE_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS)
    lines!(ax2, p, brk[k]; color=COL_BROKER, linewidth=2.2, label="broker rank")
    lines!(ax2, p, gap[k]; color=COL_DIAG, linewidth=2.2, label="broker−agent gap"); capref!(ax2)
    axislegend(ax2; position=:lc, LEG_KW...)
    ax3 = Axis(fig[3, 1]; title="Informational advantage — mean output by channel",
        xlabel="period", ylabel="output q", titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    lines!(ax3, p, qb[k]; color=COL_BROKER, linewidth=2.2, label="broker")
    lines!(ax3, p, qs[k]; color=COL_AGENT, linewidth=2.2, label="self")
    ax3b = Axis(fig[3, 1]; yaxisposition=:right, ylabel="captured-demand share", ylabelcolor=:gray40,
        yticklabelcolor=:gray40, ylabelsize=LABEL_FS, yticklabelsize=TICK_FS)
    lines!(ax3b, p, cap[k]; color=:gray45, linestyle=:dash, linewidth=2.5)
    hidespines!(ax3b); hidexdecorations!(ax3b)
    linkxaxes!(ax1, ax2, ax3, ax3b); hidexdecorations!(ax1; grid=false); hidexdecorations!(ax2; grid=false)
    axislegend(ax3; position=:lt, LEG_KW...)
    rowsize!(fig.layout, 0, Fixed(28)); rowgap!(fig.layout, 12)
    save(joinpath(OUT, "qM6_capture_covary_time.png"), fig); println("  qM6 done")
end

function fig_market_base()
    fig = Figure(size=(1320, 1120))
    Label(fig[0, 1:4], "Market structure across regimes (base model)";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    axl = [ax_rho("base"), ax_N("base"), ax_eta("base"), ax_lr("base")]
    panel_row!(fig, 1, axl, d -> tailmean(d, :outsourcing_rate), "Outsourcing rate", (0, 1.02);
        color=COL_BROKER, header=true)
    panel_row!(fig, 2, axl, tail_overall_q, "Realized match output q\n(all matches)", (0, nothing);
        color=COL_DIAG)

    # Row 3: mean matches per agent per period (2 * total matches / N), shared y-range
    cellN(rel) = (mm = match(r"N=(\d+)", rel); isnothing(mm) ? 1000 : parse(Int, mm.captures[1]))
    r3 = [(vals, [seedstat_f(load_mdfs(c), d -> 2 * tailmean(d, :n_total_matches) / cellN(c)) for c in cells])
          for (_, vals, cells) in axl]
    ymax3 = maximum(filter(!isnan, vcat([first.(st) .+ last.(st) for (_, st) in r3]...)))
    for (ci, (vals, st)) in enumerate(r3)
        ax = Axis(fig[3, ci]; ylabel = ci == 1 ? "Matches per agent\n(per period, steady state)" : "",
            xticks=(1:length(vals), vals), titlesize=TITLE_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, ymax3 * 1.08)))
        x = 1:length(vals)
        band!(ax, x, first.(st) .- last.(st), first.(st) .+ last.(st); color=(COL_ACCESS, 0.15))
        scatterlines!(ax, x, first.(st); color=COL_ACCESS, markersize=9)
    end

    # Row 4: steady-state (tail-mean) of the mean and median agent degree (both in raw data)
    tdeg(mdfs, col) = nanmean([tailmean(d, col) for d in mdfs])
    r4 = [(vals, [tdeg(load_mdfs(c), :mean_degree) for c in cells],
                 [tdeg(load_mdfs(c), :median_degree) for c in cells]) for (_, vals, cells) in axl]
    ymax4 = maximum(filter(!isnan, vcat([vcat(me, md) for (_, me, md) in r4]...)))
    for (ci, (vals, me, md)) in enumerate(r4)
        ax = Axis(fig[4, ci]; ylabel = ci == 1 ? "Agent degree\n(steady state)" : "",
            xticks=(1:length(vals), vals), titlesize=TITLE_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, ymax4 * 1.08)))
        x = 1:length(vals)
        scatterlines!(ax, x, me; color=COL_BROKER, markersize=9, label="mean")
        scatterlines!(ax, x, md; color=COL_AGENT, markersize=9, label="median")
        ci == 1 && axislegend(ax; position=:lt, LEG_KW...)
    end

    rowsize!(fig.layout, 0, Fixed(30)); rowgap!(fig.layout, 8); colgap!(fig.layout, 12)
    save(joinpath(OUT, "qM1_market.png"), fig); println("  qM1 done")
end

function fig_capture_fraction()
    fig = Figure(size=(1320, 330))
    Label(fig[0, 1:4], "Capture fraction across regimes (Model 1)";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    axl = [ax_rho("capture"), ax_N("capture"), ax_lr("capture"), ax_kappa()]
    panel_row!(fig, 1, axl, d -> tailmean(d, :principal_mode_share), "Captured-demand share",
        (0, 1.02); color=COL_CAPTURE, header=true)
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 12)
    save(joinpath(OUT, "qM2_capture.png"), fig); println("  qM2 done")
end

function fig_access_fraction()
    fig = Figure(size=(1320, 340))
    Label(fig[0, 1:4], "Broker access fraction across regimes (base model)";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    axl = [ax_rho("base"), ax_N("base"), ax_eta("base"), ax_k("base")]
    panel_row!(fig, 1, axl, access_tail, "Access fraction", (0, 1.02); color=COL_ACCESS, header=true)
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 12)
    save(joinpath(OUT, "qM3_access.png"), fig); println("  qM3 done")
end

# ── Position vs work: betweenness (opportunity) is not access (bridging done) ──
function period_access(mdfs)
    per = mdfs[1].period
    af = [nanmean(Float64[(a = d[t, :access_count] + d[t, :assessment_count];
                           a > 0 ? d[t, :access_count] / a : NaN) for d in mdfs])
          for t in eachindex(per)]
    return per, af
end
cell_access(mdfs) = nanmean([access_tail(d) for d in mdfs])

function fig_position_vs_work()
    rhos = [0.0, 0.5, 1.0]
    fig = Figure(size=(1500, 440))
    Label(fig[0, 1:3],
        "Brokerage opportunity is not the work of brokering";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    axa = Axis(fig[1, 1]; title="Position is built up over time", xlabel="Period",
        ylabel="Broker betweenness  (could bridge)", titlesize=TITLE_FS, xlabelsize=LABEL_FS+1,
        ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, nothing)))
    axb = Axis(fig[1, 2]; title="Bridging work is liquidated over time", xlabel="Period",
        ylabel="Access fraction  (does bridge)", titlesize=TITLE_FS, xlabelsize=LABEL_FS+1,
        ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, (0, nothing)))
    for r in rhos
        m = load_mdfs("oat/rho=$(r)/base")
        pa, bb = period_avg(m, :betweenness); lines!(axa, pa, bb; color=RHO_COLORS[r], linewidth=2.2, label="ρ = $(r)")
        pb, aa = period_access(m); lines!(axb, pb, aa; color=RHO_COLORS[r], linewidth=2.2)
    end
    vlines!(axa, [30]; color=:gray70, linestyle=:dot); vlines!(axb, [30]; color=:gray70, linestyle=:dot)
    axislegend(axa, "Channel mix"; position=:lt, LEG_KW...)
    # (c) cross-regime: betweenness vs access fraction, all base OAT cells
    cellspec = [("oat/rho=$(r)/base", get(RHO_COLORS, r, :gray55)) for r in RHO5]
    append!(cellspec, [("oat/$a/base", :gray60) for a in
        ("eta=0.01", "eta=0.02", "eta=0.03", "N=500", "N=1000", "N=1500",
         "reservation_frac=0.4", "reservation_frac=0.6", "reservation_frac=0.9",
         "reservation_frac=1.2", "delta=0.0", "delta=0.75", "k=4", "k=12")])
    bx = Float64[]; ay = Float64[]; cc = Any[]
    for (rel, col) in cellspec
        isfile(joinpath(ROOT, rel, "data.jld2")) || continue
        m = load_mdfs(rel); b = seedstat(m, :betweenness)[1]; a = cell_access(m)
        (isnan(b) || isnan(a)) && continue
        push!(bx, b); push!(ay, a); push!(cc, col)
    end
    rr = cor(bx, ay)
    axc = Axis(fig[1, 3]; title="Across regimes they are decoupled", xlabel="Broker betweenness  (position)",
        ylabel="Access fraction  (work)", titlesize=TITLE_FS, xlabelsize=LABEL_FS+1, ylabelsize=LABEL_FS+1,
        xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    # least-squares guide line
    β = cov(bx, ay) / var(bx); α = mean(ay) - β * mean(bx)
    xr = [minimum(bx), maximum(bx)]; lines!(axc, xr, α .+ β .* xr; color=:gray55, linestyle=:dash, linewidth=1.5)
    scatter!(axc, bx, ay; color=cc, markersize=12, strokewidth=0.4, strokecolor=:gray30)
    text!(axc, 0.97, 0.97; text="Pearson r = $(round(rr; digits=2))\ngray = non-ρ regimes",
        space=:relative, align=(:right, :top), fontsize=TICK_FS, color=:gray25)
    colgap!(fig.layout, 16); rowsize!(fig.layout, 0, Fixed(30))
    println("  [position-vs-work] cross-cell cor(betweenness, access) = ", round(rr; digits=3), "  (n=$(length(bx)))")
    save(joinpath(OUT, "qM4_position_vs_work.png"), fig); println("  qM4 done")
end

# ── Supplementary: other advantage measures in the Fig-2 (channel/predictor) style ──
function fig_suppl_predict(kind)
    rhos = RHO5
    mdfs_all = [load_mdfs("oat/rho=$(r)/base") for r in rhos]
    pre = kind == :holdout ? "broker_holdout_" : "broker_selected_"
    apre = kind == :holdout ? "agent_holdout_" : "agent_selected_"
    word = kind == :holdout ? "Holdout" : "Selected-sample"
    metrics = [("Rank correlation", "rank", (0, 1.0)), ("R²", "r2", nothing),
               ("RMSE  (lower = better)", "rmse", (0, nothing)), ("Bias", "bias", nothing)]
    fig = Figure(size=(900, 660))
    Label(fig[0, 1:2], "$word prediction quality: broker vs. agent across ρ";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    for (k, (mlab, msuf, ylim)) in enumerate(metrics)
        rr = div(k - 1, 2) + 1; cc = mod(k - 1, 2) + 1
        bcol = Symbol(pre * msuf); acol = Symbol(apre * msuf)
        bv = [seedstat(m, bcol) for m in mdfs_all]; av = [seedstat(m, acol) for m in mdfs_all]
        ax = Axis(fig[rr, cc]; ylabel=mlab, xticks=(1:length(rhos), string.(rhos)),
            xlabel = rr == 2 ? "ρ (channel mix)" : "", titlesize=TITLE_FS,
            ylabelsize=LABEL_FS+1, xlabelsize=LABEL_FS, xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS, limits=(nothing, ylim))
        x = 1:length(rhos)
        band!(ax, x, first.(bv) .- last.(bv), first.(bv) .+ last.(bv); color=(COL_BROKER, 0.12))
        band!(ax, x, first.(av) .- last.(av), first.(av) .+ last.(av); color=(COL_AGENT, 0.12))
        (msuf in ("r2", "bias")) && hlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=1)
        scatterlines!(ax, x, first.(bv); color=COL_BROKER, markersize=9, label="Broker")
        scatterlines!(ax, x, first.(av); color=COL_AGENT, markersize=9, label="Agent")
        k == 1 && axislegend(ax; position=:rc, LEG_KW...)
    end
    rowgap!(fig.layout, 8); colgap!(fig.layout, 14); rowsize!(fig.layout, 0, Fixed(30))
    save(joinpath(OUT, kind == :holdout ? "qS1_holdout.png" : "qS2_selected.png"), fig)
    println("  suppl $kind done")
end

function fig_suppl_netvalue()
    rhos = RHO5
    mdfs_all = [load_mdfs("oat/rho=$(r)/base") for r in rhos]
    ss = [seedstat(m, :mean_satisfaction_self) for m in mdfs_all]
    sb = [seedstat(m, :mean_satisfaction_broker) for m in mdfs_all]
    rp = [seedstat(m, :broker_reputation) for m in mdfs_all]
    fig = Figure(size=(720, 470))
    ax = Axis(fig[1, 1];
        title="Net value delivered to agents by channel (satisfaction, net of fees and search costs)",
        xlabel="ρ (channel mix)", ylabel="Mean satisfaction  (EWMA net match value)",
        xticks=(1:length(rhos), string.(rhos)), titlesize=TITLE_FS, xlabelsize=LABEL_FS+1,
        ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
    x = 1:length(rhos)
    for (vv, col, lab, ls) in ((ss, COL_AGENT, "Self-search channel", :solid),
                               (sb, COL_BROKER, "Broker channel", :solid),
                               (rp, COL_REPUTATION, "Broker reputation", :dash))
        band!(ax, x, first.(vv) .- last.(vv), first.(vv) .+ last.(vv); color=(col, 0.1))
        scatterlines!(ax, x, first.(vv); color=col, markersize=9, linestyle=ls, label=lab)
    end
    axislegend(ax; position=:rt, LEG_KW...)
    save(joinpath(OUT, "qS3_netvalue.png"), fig); println("  suppl netvalue done")
end

# ── Structural position vs. the broker's two advantages (betweenness scatter) ──
function fig_suppl_scatter()
    rho = Float64[]; betw = Float64[]; g_hrank = Float64[]; g_q = Float64[]
    for pair in ("rho_eta", "rho_N", "rho_r", "rho_delta")
        s = load_summary(pair, "base"); s.xk == "rho" || continue
        for xi in 0:length(s.xv)-1, yi in 0:length(s.yv)-1
            rel = "phase/$(pair)/cells/$(xi)_$(yi)/base"
            isfile(joinpath(ROOT, rel, "data.jld2")) || continue
            m = load_mdfs(rel); b = seedstat(m, :betweenness)[1]; isnan(b) && continue
            push!(rho, Float64(s.xv[xi+1])); push!(betw, b)
            push!(g_hrank, seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1])
            push!(g_q, seedstat(m, :q_broker_standard_mean)[1] - seedstat(m, :q_self_mean)[1])
        end
    end
    panels = [("Holdout rank-correlation gap  (broker − agent)", g_hrank),
              ("Realized output gap  q  (broker − self)", g_q)]
    fig = Figure(size=(1100, 470))
    Label(fig[0, 1:2], "Structural position vs. the broker's advantage";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    for (k, (ylab, yv)) in enumerate(panels)
        ax = Axis(fig[1, k]; xlabel="Broker betweenness  (structural advantage)", ylabel=ylab,
            titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
        ok = .!isnan.(yv)
        for rv in (0.0, 0.5, 1.0)
            mm = ok .& (rho .== rv)
            scatter!(ax, betw[mm], yv[mm]; color=(RHO_COLORS[rv], 0.75), markersize=10,
                strokewidth=0.3, strokecolor=:gray30, label="ρ = $(rv)")
        end
        rok = cor(betw[ok], yv[ok])
        text!(ax, 0.03, 0.97; text="r = $(round(rok; digits=2))", space=:relative,
            align=(:left, :top), fontsize=TICK_FS, color=:gray25)
        k == 1 && axislegend(ax, "Channel mix"; position=:rt, LEG_KW...)
    end
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 16)
    save(joinpath(OUT, "qS4_advantage_scatter.png"), fig); println("  qS4 done")
end

# ── Access fraction vs. the broker's two advantages (identical to fig:advscatter, betweenness → access) ──
function fig_access_scatter()
    nmf(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
    accell(m) = (vs = filter(!isnan, [nmf(access_fraction(d)[d.period .> 30]) for d in m]); isempty(vs) ? NaN : mean(vs))
    rho = Float64[]; acc = Float64[]; g_hrank = Float64[]; g_q = Float64[]
    for pair in ("rho_eta", "rho_N", "rho_r", "rho_delta")
        s = load_summary(pair, "base"); s.xk == "rho" || continue
        for xi in 0:length(s.xv)-1, yi in 0:length(s.yv)-1
            rel = "phase/$(pair)/cells/$(xi)_$(yi)/base"
            isfile(joinpath(ROOT, rel, "data.jld2")) || continue
            m = load_mdfs(rel); a = accell(m); isnan(a) && continue
            push!(rho, Float64(s.xv[xi+1])); push!(acc, a)
            push!(g_hrank, seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1])
            push!(g_q, seedstat(m, :q_broker_standard_mean)[1] - seedstat(m, :q_self_mean)[1])
        end
    end
    panels = [("Holdout rank-correlation gap  (broker − agent)", g_hrank),
              ("Realized output gap  q  (broker − self)", g_q)]
    fig = Figure(size=(1100, 470))
    Label(fig[0, 1:2], "Access fraction vs. the broker's advantage";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    for (k, (ylab, yv)) in enumerate(panels)
        ax = Axis(fig[1, k]; xlabel="Broker access fraction", ylabel=ylab,
            titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
        ok = .!isnan.(yv)
        for rv in (0.0, 0.5, 1.0)
            mm = ok .& (rho .== rv)
            scatter!(ax, acc[mm], yv[mm]; color=(RHO_COLORS[rv], 0.75), markersize=10,
                strokewidth=0.3, strokecolor=:gray30, label="ρ = $(rv)")
        end
        rok = cor(acc[ok], yv[ok])
        text!(ax, 0.03, 0.97; text="r = $(round(rok; digits=2))", space=:relative,
            align=(:left, :top), fontsize=TICK_FS, color=:gray25)
        k == 1 && axislegend(ax, "Channel mix"; position=:rt, LEG_KW...)
    end
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 16)
    save(joinpath(OUT, "qS5_access_scatter.png"), fig); println("  qS5 done")
end

# ── Disentangle rank (delta) from direction (rho) on the rho x delta phase grid ──
function fig_disentangle()
    rhos = [0.0, 0.5, 1.0]; deltas = [0.0, 0.5, 0.75]
    dcol = Dict(0.0 => :goldenrod, 0.5 => :darkorange, 0.75 => :firebrick)
    cell(xi, yi) = load_mdfs("phase/rho_delta/cells/$(xi)_$(yi)/base")
    val(f) = [[f(cell(xi, yi)) for xi in 0:2] for yi in 0:2]   # [delta][rho]
    betw = val(m -> seedstat(m, :betweenness)[1])
    r2   = val(m -> seedstat(m, :broker_holdout_r2)[1])
    rg   = val(m -> seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1])
    qg   = val(m -> seedstat(m, :q_broker_standard_mean)[1] - seedstat(m, :q_self_mean)[1])
    panels = [("Betweenness — structural\n(δ-lines overlap ⇒ direction-driven)", betw, (0, 0.85), false),
              ("Broker holdout R²\n(δ-lines spread ⇒ rank-driven)", r2, nothing, true),
              ("Holdout rank-correlation gap\n(δ-lines spread at low ρ ⇒ rank-driven)", rg, (0, 0.45), false),
              ("Realized output gap q\n(varies with both ρ and δ)", qg, nothing, true)]
    fig = Figure(size=(1080, 800))
    Label(fig[0, 1:2], "Disentangling rank (δ) from direction (ρ)";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    for (k, (ttl, V, ylim, zero)) in enumerate(panels)
        rr = div(k - 1, 2) + 1; cc = mod(k - 1, 2) + 1
        ax = Axis(fig[rr, cc]; title=ttl, xlabel = rr == 2 ? "ρ (quality ↔ complementarity)" : "",
            xticks=(1:3, string.(rhos)), titlesize=TITLE_FS, xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS, yticklabelsize=TICK_FS, limits=(nothing, ylim))
        zero && hlines!(ax, [0.0]; color=:gray60, linestyle=:dash, linewidth=1)
        for (yi, dl) in enumerate(deltas)
            scatterlines!(ax, 1:3, V[yi]; color=dcol[dl], markersize=8, label="δ = $(dl)")
        end
        k == 1 && axislegend(ax, "rank knob"; position=:lb, LEG_KW...)
    end
    rowsize!(fig.layout, 0, Fixed(34)); colgap!(fig.layout, 18); rowgap!(fig.layout, 12)
    save(joinpath(OUT, "qD3_disentangle.png"), fig); println("  qD3 done")
end

for (name, f) in (("qM0", fig_market_dynamics), ("qM1", fig_market_base), ("qM2", fig_capture_fraction),
                  ("qM3", fig_access_fraction), ("qM4", fig_position_vs_work),
                  ("qM5", fig_capture_covary), ("qM6", fig_capture_covary_time),
                  ("qD3", fig_disentangle),
                  ("q1", fig_edges_scatter), ("q2", fig_realized_channel),
                  ("q6", fig_performance_grid),
                  ("qS1", () -> fig_suppl_predict(:holdout)),
                  ("qS2", () -> fig_suppl_predict(:selected)),
                  ("qS3", fig_suppl_netvalue), ("qS4", fig_suppl_scatter),
                  ("qS5", fig_access_scatter),
                  ("q4", fig_ordering_level),
                  ("q5a", fig_capture_gate), ("q5b", fig_lockin))
    try
        f()
    catch err
        println("FAILED $name: ", err)
    end
end
println("DONE")
