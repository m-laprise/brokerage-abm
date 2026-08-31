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
                      with 95% Monte Carlo interval whiskers
  fig3_position_work  betweenness & access over time at baseline + cross-regime scatter
  fig4_advantage      structural measures vs informational/output gaps

Usage: julia --project scripts/paper/figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
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
const ETA_PALETTE = Makie.to_color.(
    ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677"]
)
const RHO_MARKERS = Dict(
    0.0 => :circle, 0.5 => :rect, 0.85 => :diamond, 1.0 => :utriangle
)

const FD = JLD2.load(
    normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figdata.jld2"))
)["figdata"]
const REPORTING_PROVENANCE = reporting_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."))
)
FD["meta"]["analysis_git_commit"] == REPORTING_PROVENANCE.commit ||
    error("main figure data were extracted by a different analysis commit")
FD["meta"]["analysis_source_clean"] == true ||
    error("main figure data were extracted from dirty analysis sources")
const PER = FD["period"]
const SER = FD["series"]
const SER_SEEDS = FD["series_seed_values"]
const TEND = maximum(PER)
const TIME_TICK_STEP = TEND <= 250 ? 50 : 100
const TIME_TICKS = TIME_TICK_STEP:TIME_TICK_STEP:TEND
const ADV_MARKER_SIZE = 10
function savefig(fname, fig)
    (save(joinpath(OUT, fname), fig; px_per_unit=PXU); println("  $fname done"))
end

function summarized_series(key; measured=false)
    indices = if measured
        [index for index in eachindex(PER) if PER[index] % BETWINT == 0]
    else
        collect(eachindex(PER))
    end
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

function draw_interval_series!(
    axis, series, color; label=nothing, points=false, linestyle=:solid, linewidth=2.2
)
    x, estimate, lower, upper = series
    band!(axis, x, lower, upper; color=(color, 0.16))
    keywords = isnothing(label) ? (;) : (; label)
    if points
        scatterlines!(
            axis, x, estimate; color, linewidth, markersize=6, linestyle, keywords...
        )
    else
        lines!(axis, x, estimate; color, linewidth, linestyle, keywords...)
    end
    return nothing
end

# ── Position: betweenness & access over time at baseline + cross-regime scatter ──
function fig3_position_work()
    fig = Figure(; size=(1230, 470))
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
    draw_interval_series!(
        axa,
        summarized_series("betweenness"; measured=true),
        COL_GAP;
        points=true,
        label="broker betweenness centrality",
    )
    draw_interval_series!(
        axa, summarized_series("access"), COL_ACCESS; label="access fraction"
    )
    axislegend(axa; position=:rc, LEG_KW...)
    # right: rho x eta regimes, access (x) vs betweenness (y). Lines connect
    # rho < 1 within eta; rho = 1 is an unconnected pure-quality boundary.
    cells = FD["rho_eta_cells"]
    betweenness = [c["betw"] for c in cells]
    access = [c["access"] for c in cells]
    rho = Float64[c["rho"] for c in cells]
    eta = Float64[c["eta"] for c in cells]
    eta_values = sort(unique(eta))
    length(eta_values) <= length(ETA_PALETTE) || error("turnover palette is too short")
    eta_colors = Dict(
        value => ETA_PALETTE[index] for (index, value) in enumerate(eta_values)
    )
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
    for value in eta_values
        mask = eta .== value
        indices = sort(findall(mask); by=index -> rho[index])
        xs = access[indices]
        ys = betweenness[indices]
        rhos = rho[indices]
        xintervals = [
            monte_carlo_interval(cells[index]["seed_values"]["access"]) for index in
            indices
        ]
        yintervals = [
            monte_carlo_interval(cells[index]["seed_values"]["betw"]) for index in
            indices
        ]
        errorbars!(
            axc,
            xs,
            ys,
            xs .- [interval.lower for interval in xintervals],
            [interval.upper for interval in xintervals] .- xs;
            direction=:x,
            color=(eta_colors[value], 0.45),
            whiskerwidth=5,
        )
        errorbars!(
            axc,
            xs,
            ys,
            ys .- [interval.lower for interval in yintervals],
            [interval.upper for interval in yintervals] .- ys;
            color=(eta_colors[value], 0.45),
            whiskerwidth=5,
        )
        interior = findall(rhos .< 1.0)
        lines!(
            axc, xs[interior], ys[interior]; color=eta_colors[value], linewidth=2.2
        )
        for (r, x, y) in zip(rhos, xs, ys)
            scatter!(
                axc,
                [x],
                [y];
                marker=RHO_MARKERS[r],
                color=eta_colors[value],
                markersize=13,
                strokecolor=:black,
                strokewidth=0.5,
            )
        end
    end
    eta_elements = [
        LineElement(; color=eta_colors[value], linewidth=3) for
        value in eta_values
    ]
    rho_values = sort(unique(rho))
    rho_elements = [
        MarkerElement(;
            marker=RHO_MARKERS[value],
            color=:gray60,
            strokecolor=:black,
            strokewidth=0.5,
            markersize=11,
        ) for value in rho_values
    ]
    Legend(
        fig[1, 3],
        [eta_elements, rho_elements],
        [
            ["η = $value" for value in eta_values],
            ["ρ = $value" for value in rho_values],
        ],
        ["turnover rate", "matching composition"],
        labelsize=LABEL_FS,
        titlesize=LABEL_FS,
        patchsize=(18, 13),
        rowgap=4,
        framevisible=false,
    )
    colgap!(fig.layout, 14)
    savefig("fig3_position_work.png", fig)
end

# ── Matching grid: six outcomes vs rho, one line per delta (rho = 1 included) ──
function fig2_grid_lines()
    gcells = FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gcells]))
    cells = Dict((c["rho"], c["delta"]) => c for c in gcells)
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
            pts = sort(
                [
                    let interval = monte_carlo_interval(c["outcome_seed_values"][ttl])
                        (
                            rho=r,
                            mean=interval.mean,
                            lower=interval.lower,
                            upper=interval.upper,
                        )
                    end for ((r, dd), c) in cells if dd == d
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
    # Each seed is smoothed first. The line and pointwise interval are then
    # computed across seeds; display trimming remains axis-only.
    ot(key) = summarized_series(key)
    otb() = summarized_series("betweenness"; measured=true)
    yrange(ss...) = (
        v=filter(
            !isnan,
            vcat((vcat(s[3][s[1] .>= TSTART], s[4][s[1] .>= TSTART]) for s in ss)...),
        );   # displayed window only
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
    drw!(ax, s, col; lbl=nothing, ls=:solid, pts=false, lw=2.2) = draw_interval_series!(
        ax, s, col; label=lbl, points=pts, linestyle=ls, linewidth=lw
    )
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

foreach(
    function_name -> function_name(),
    (fig1_dynamics, fig2_grid_lines, fig3_position_work, fig4_advantage),
)
# emit the display-convention keys quoted by the captions (paper/captions.tex)
open(normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figmeta.tex")), "w") do io
    println(
        io, "% figmeta.tex: generated by scripts/paper/figures.jl. Do not edit by hand."
    )
    println(io, "% Analysis commit: $(REPORTING_PROVENANCE.commit)")
    println(
        io,
        "% Display conventions used to render output/main/figs/, quoted in captions via \\pv keys.",
    )
    println(io, "\\pvDefine{rollWin}{$ROLLW}")
    println(io, "\\pvDefine{betwInterval}{$BETWINT}")
    println(io, "\\pvDefine{axisStart}{$TSTART}")
end
println("main figures done (+ figmeta.tex)")
