"""
    summary_figures.jl

Render diagnostic figures from a completed single-model sweep. This script reads
saved data only and does not run simulations.

Usage: BROKERAGE_ABM_SWEEP_DIR=<sweep root> julia --project scripts/diagnostics/summary_figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))
using JLD2
using Statistics: mean, std

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUT = joinpath(ROOT, "report")
const RHO_COLORS = Dict(
    0.0 => :seagreen,
    0.3 => :mediumaquamarine,
    0.5 => :goldenrod,
    0.7 => :darkorange,
    1.0 => :firebrick,
)
const DELTA_COLORS = Dict(0.0 => :steelblue, 0.5 => :goldenrod, 0.75 => :firebrick)

mkpath(OUT)
nanmean(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
tailmean(df, col) = nanmean(df[(df.period .>= 181) .& (df.period .<= 200), col])
function seedstat(mdfs, col)
    (vs=filter(!isnan, [tailmean(d, col) for d in mdfs]); (mean(vs), std(vs)))
end
load_mdfs(rel) =
    jldopen(joinpath(ROOT, rel, "data.jld2"), "r") do f
        f["mdfs"]
    end
load_cfg(rel) =
    jldopen(joinpath(ROOT, rel, "data.jld2"), "r") do f
        f["config"]
    end
access_tail(df) = nanmean(access_fraction(df)[(df.period .>= 181) .& (df.period .<= 200)])
cell_access(mdfs) = nanmean([access_tail(d) for d in mdfs])
function period_ens(mdfs, f)
    (
        per=mdfs[1].period;
        (per, [nanmean(Float64[f(d)[t] for d in mdfs]) for t in eachindex(per)])
    )
end

RG = JLD2.load(joinpath(@__DIR__, "_results", "dgp_rank_grid.jld2"))
r90(rho, delta) = RG["r90"][(Float64(rho), Float64(delta))]
qgap(m) = seedstat(m, :q_broker_mean)[1] - seedstat(m, :q_self_mean)[1]
const OUTCOMES = [
    ("Betweenness", m -> seedstat(m, :betweenness)[1]),
    ("Access fraction", cell_access),
    ("Broker prediction R²", m -> seedstat(m, :broker_holdout_r2)[1]),
    (
        "Prediction R² gap",
        m -> seedstat(m, :broker_holdout_r2)[1] - seedstat(m, :agent_holdout_r2)[1],
    ),
    ("Broker rank correlation", m -> seedstat(m, :broker_holdout_rank)[1]),
    (
        "Rank correlation gap",
        m -> seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1],
    ),
    ("Broker output q", m -> seedstat(m, :q_broker_mean)[1]),
    ("Output gap q", qgap),
    ("Outsourcing rate", m -> seedstat(m, :outsourcing_rate)[1]),
]

function grid_cells(; drop_rho_one::Bool)
    summary = jldopen(joinpath(ROOT, "phase/rho_delta/summary.jld2"), "r") do f
        (xv=f["xvals"], yv=f["yvals"])
    end
    cells = Dict{Tuple{Float64,Float64},Vector{DataFrame}}()
    for (xi, rho) in enumerate(summary.xv), (yi, delta) in enumerate(summary.yv)
        drop_rho_one && rho == 1.0 && continue
        cells[(rho, delta)] = load_mdfs("phase/rho_delta/cells/$(xi - 1)_$(yi - 1)")
    end
    for rho in (0.3, 0.7)
        cells[(rho, 0.5)] = load_mdfs("oat/rho=$rho")
    end
    return cells
end

function fig1_grid_lines()
    cells = grid_cells(; drop_rho_one=true)
    deltas = sort(unique(last.(keys(cells))))
    byname(name) = OUTCOMES[findfirst(o -> o[1] == name, OUTCOMES)]
    layout = [
        byname("Betweenness") byname("Broker rank correlation") byname("Rank correlation gap")
        byname("Access fraction") byname("Broker prediction R²") byname("Prediction R² gap")
        byname("Outsourcing rate") byname("Broker output q") byname("Output gap q")
    ]
    fig = Figure(; size=(1280, 980))
    Label(
        fig[0, 1:3],
        "Late-window means across the matching-function grid";
        fontsize=SUPTITLE_FS,
        font=:bold,
        tellwidth=false,
    )
    axs = Matrix{Axis}(undef, 3, 3)
    for row in 1:3, col in 1:3
        title, metric = layout[row, col]
        ax = Axis(
            fig[row, col];
            title=title,
            xlabel=row == 3 ? "ρ (channel mix)" : "",
            xticks=[0, 0.3, 0.5, 0.7],
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
            limits=col == 1 ? (nothing, (0, 1.02)) : (nothing, nothing),
        )
        for delta in deltas
            points = sort(
                [(rho, metric(mdfs)) for ((rho, d), mdfs) in cells if d == delta]; by=first
            )
            scatterlines!(
                ax,
                first.(points),
                last.(points);
                color=DELTA_COLORS[delta],
                linewidth=2.0,
                markersize=10,
                strokewidth=0.4,
                strokecolor=:gray30,
                label="δ = $delta",
            )
        end
        row == 1 && col == 1 && axislegend(ax, "Regime gain"; position=:rb, LEG_KW...)
        axs[row, col] = ax
    end
    for row in 1:3
        linkyaxes!(axs[row, 2], axs[row, 3])
    end
    rowsize!(fig.layout, 0, Fixed(30))
    colgap!(fig.layout, 16)
    rowgap!(fig.layout, 12)
    save(joinpath(OUT, "rs1_grid_lines.png"), fig)
end

function fig1_rank_lines()
    points = [
        (r90(rho, delta), rho, delta, mdfs) for
        ((rho, delta), mdfs) in grid_cells(; drop_rho_one=true)
    ]
    shown = [
        o for o in OUTCOMES if o[1] in (
            "Broker rank correlation",
            "Rank correlation gap",
            "Broker output q",
            "Output gap q",
        )
    ]
    fig = Figure(; size=(980, 760))
    for (index, (title, metric)) in enumerate(shown)
        row, col = div(index - 1, 2) + 1, mod(index - 1, 2) + 1
        ax = Axis(
            fig[row, col];
            title=title,
            xlabel=row == 2 ? "effective rank r₉₀" : "",
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
        )
        for delta in (0.0, 0.5, 0.75)
            group = sort(
                [(rank, metric(mdfs)) for (rank, _, d, mdfs) in points if d == delta];
                by=first,
            )
            isempty(group) && continue
            scatterlines!(
                ax,
                first.(group),
                last.(group);
                color=DELTA_COLORS[delta],
                linewidth=1.6,
                markersize=11,
                strokewidth=0.4,
                strokecolor=:gray30,
            )
        end
    end
    colgap!(fig.layout, 14)
    rowgap!(fig.layout, 10)
    save(joinpath(OUT, "rs1_rank_lines.png"), fig)
end

function fig2_position_work()
    mdfs = load_mdfs("oat/rho=0.5")
    fig = Figure(; size=(1150, 860))
    Label(
        fig[0, 1:2],
        "Broker betweenness and access fraction";
        fontsize=SUPTITLE_FS,
        font=:bold,
        tellwidth=false,
    )
    axa = Axis(
        fig[1, 1];
        title="Betweenness over time",
        xlabel="period",
        ylabel="broker betweenness",
        limits=((30, 201), (0, nothing)),
    )
    per, betweenness = period_ens(mdfs, d -> d.betweenness)
    measured = [i for i in eachindex(per) if per[i] % 20 == 0]
    scatterlines!(
        axa,
        per[measured],
        rolling_mean(betweenness[measured], 5);
        color=COL_GAP,
        linewidth=2.2,
        markersize=6,
    )
    axb = Axis(
        fig[1, 2];
        title="Access fraction over time",
        xlabel="period",
        ylabel="access fraction",
        limits=((30, 201), (0, nothing)),
    )
    pa, access = period_ens(mdfs, access_fraction)
    lines!(axb, pa, rolling_mean(access, 5); color=COL_ACCESS, linewidth=2.2)

    cellrels = vcat(
        ["oat/rho=$rho" for rho in (0.0, 0.3, 0.5, 0.7, 1.0)],
        [
            "oat/$axis" for axis in (
                "eta=0.01",
                "eta=0.02",
                "eta=0.03",
                "N=500",
                "N=1000",
                "N=1500",
                "reservation_frac=0.4",
                "reservation_frac=0.6",
                "reservation_frac=0.9",
                "reservation_frac=1.2",
                "delta=0.0",
                "delta=0.75",
                "k=4",
                "k=12",
            )
        ],
    )
    betw, access_means, ranks = Float64[], Float64[], Float64[]
    for rel in cellrels
        isfile(joinpath(ROOT, rel, "data.jld2")) || continue
        cell = load_mdfs(rel)
        cfg = load_cfg(rel)
        push!(betw, seedstat(cell, :betweenness)[1])
        push!(access_means, cell_access(cell))
        push!(ranks, r90(cfg["rho"], cfg["delta"]))
    end
    axc = Axis(
        fig[2, 1:2];
        title="Across regimes",
        xlabel="broker betweenness",
        ylabel="access fraction",
    )
    scatter!(
        axc,
        betw,
        access_means;
        color=ranks,
        colormap=:viridis,
        colorrange=(2, 10.5),
        markersize=13,
    )
    rowgap!(fig.layout, 14)
    save(joinpath(OUT, "rs2_position_work.png"), fig)
end

function regime_cells()
    out = NamedTuple[]
    for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
        "data.jld2" in files || continue
        mdfs, cfg = jldopen(joinpath(root, "data.jld2"), "r") do f
            f["mdfs"], f["config"]
        end
        push!(out, (mdfs=mdfs, cfg=cfg))
    end
    return out
end

function fig3_advantage()
    cells = regime_cells()
    rho = Float64[c.cfg["rho"] for c in cells]
    betw = [seedstat(c.mdfs, :betweenness)[1] for c in cells]
    access = [cell_access(c.mdfs) for c in cells]
    rank_gap = [
        seedstat(c.mdfs, :broker_holdout_rank)[1] -
        seedstat(c.mdfs, :agent_holdout_rank)[1] for c in cells
    ]
    output_gap = [qgap(c.mdfs) for c in cells]
    xs = [("Broker betweenness", betw), ("Access fraction", access)]
    ys = [
        ("Rank correlation gap (broker − agent)", rank_gap),
        ("Output gap q (broker − self)", output_gap),
    ]
    fig = Figure(; size=(1150, 940))
    for (row, (ylabel, y)) in enumerate(ys), (col, (xlabel, x)) in enumerate(xs)
        ax = Axis(
            fig[row, col];
            xlabel=row == 2 ? xlabel : "",
            ylabel=col == 1 ? ylabel : "",
            title=row == 1 ? xlabel : "",
        )
        for value in sort(unique(rho))
            selected = rho .== value
            scatter!(
                ax, x[selected], y[selected]; color=(RHO_COLORS[value], 0.8), markersize=10
            )
        end
    end
    save(joinpath(OUT, "rs3_advantage.png"), fig)
end

function fig4_dynamics()
    mdfs = load_mdfs("oat/rho=0.5")
    N = load_cfg("oat/rho=0.5")["N"]
    smooth(f) = (
        per=mdfs[1].period;
        (
            per,
            rolling_mean(
                [nanmean(Float64[f(d)[t] for d in mdfs]) for t in eachindex(per)], 5
            ),
        )
    )
    smooth_betweenness() = (
        per,
        raw=period_ens(mdfs, d -> d.betweenness);
        measured=[i for i in eachindex(per) if per[i] % 20 == 0];
        (per[measured], rolling_mean(raw[measured], 5)),
    )
    series = (
        smooth(d -> 2 .* d.n_total_matches ./ N),
        smooth(d -> d.mean_degree),
        smooth(d -> d.median_degree),
        smooth_betweenness(),
        smooth(access_fraction),
        smooth(d -> d.outsourcing_rate),
    )
    fig = Figure(; size=(1220, 310))
    titles = (
        "Matches per agent",
        "Network degree",
        "Broker betweenness",
        "Access fraction",
        "Outsourcing rate",
    )
    for col in 1:5
        ax = Axis(
            fig[1, col]; title=titles[col], xlabel="period", limits=((30, 201), nothing)
        )
        if col == 2
            lines!(ax, series[2]...; color=COL_AGENT, linewidth=2.2, label="mean")
            lines!(
                ax,
                series[3]...;
                color=COL_REFERENCE,
                linestyle=:dot,
                linewidth=2.2,
                label="median",
            )
            axislegend(ax; position=:rb, LEG_KW...)
        else
            s = series[col == 1 ? 1 : col + 1]
            color = (COL_DIAG, COL_GAP, COL_ACCESS, COL_BROKER)[col == 1 ? 1 : col - 1]
            if col == 3
                scatterlines!(ax, s...; color=color, linewidth=2.2, markersize=6)
            else
                lines!(ax, s...; color=color, linewidth=2.2)
            end
        end
    end
    save(joinpath(OUT, "rs4_dynamics.png"), fig)
end

for (name, render) in (
    ("rs1", fig1_grid_lines),
    ("rs1alt", fig1_rank_lines),
    ("rs2", fig2_position_work),
    ("rs3", fig3_advantage),
    ("rs4", fig4_dynamics),
)
    try
        render()
        println("  $name done")
    catch err
        println("  $name FAILED: ", sprint(showerror, err)[1:min(end, 400)])
    end
end
println("summary figures done")
