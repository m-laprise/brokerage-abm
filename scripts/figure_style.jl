"""
    figure_style.jl

Shared plotting helpers and an opt-in publication theme. Exploration scripts
retain their defaults unless they call `publication_theme!`.

Include this file from any script that produces dynamics panels:
    include(joinpath(@__DIR__, "figure_style.jl"))
"""

using CairoMakie
using DataFrames
using Statistics: mean

# ─────────────────────────────────────────────────────────────────────────────
# Color palette (consistent across all figures)
# ─────────────────────────────────────────────────────────────────────────────

const COL_BROKER = :crimson       # broker-related metrics
const COL_AGENT = :steelblue     # agent/self-search metrics
const COL_GAP = :purple        # broker-agent difference
const COL_ACCESS = :goldenrod     # access fraction
const COL_REPUTATION = :darkred       # broker reputation
const COL_DIAG = :teal          # diagnostic metrics
const COL_REFERENCE = :gray60        # secondary/reference series (dashed)

# ─────────────────────────────────────────────────────────────────────────────
# Font sizes
# ─────────────────────────────────────────────────────────────────────────────

# Sizes are chosen so that, after the figures are downscaled to \linewidth in the
# paper (canvases of 1000-1350 px at roughly 190-210 px per printed inch), tick
# labels print at >= 7 pt and axis labels at >= 8 pt.
const SUPTITLE_FS = 28
const TITLE_FS = 24
const LABEL_FS = 22
const TICK_FS = 19
const ROW_LABEL_FS = 22
const FOOTER_FS = 18

# Ordered parameter colors and redundant symbols, shared by the publication
# figures. Service-mode colors are separate from these parameter encodings.
const PUB_RHO_COLORS = Dict(
    zip(
        (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0),
        ("#46327E", "#365C8D", "#277F8E", "#1F9D8A", "#6CAE75", "#A8BE50", "#C5A532"),
    ),
)
const PUB_RHO_MARKERS = Dict(
    zip(
        (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0),
        (:circle, :rect, :utriangle, :diamond, :dtriangle, :cross, :hexagon),
    ),
)
const PUB_DELTA_COLORS = Dict(
    zip(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        ("#0072B2", "#D55E00", "#009E73", "#A845A0", "#111111"),
    ),
)
const PUB_DELTA_MARKERS = Dict(
    zip((0.0, 0.25, 0.5, 0.75, 1.0), (:circle, :rect, :diamond, :utriangle, :dtriangle)),
)
const PUB_ETA_COLORS = Dict(
    zip(
        (0.0, 0.001, 0.01, 0.02, 0.03),
        ("#111111", "#A845A0", "#0072B2", "#D55E00", "#009E73"),
    ),
)
const PUB_CENTRALITY = "#756291"
const PUB_ACCESS = "#B38232"
const PUB_BROKER = "#BA5A3A"
const PUB_PRINCIPAL = "#237A93"
const PUB_RHO_LABEL = "General-quality share (ρ)"
const PUB_LEGEND = (;
    labelsize=19,
    titlesize=19,
    framevisible=false,
    orientation=:horizontal,
    titleposition=:top,
    patchsize=(28, 16),
    colgap=20,
    rowgap=6,
    padding=(0, 0, 0, 0),
    tellwidth=false,
)

"""Apply the publication theme without changing exploration defaults or data."""
function publication_theme!()
    set_theme!(;
        fontsize=TICK_FS,
        figure_padding=(24, 26, 20, 20),
        Axis=(;
            titlealign=:left,
            titlegap=14,
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
            topspinevisible=false,
            rightspinevisible=false,
            xgridvisible=false,
            ygridcolor=(:black, 0.065),
            ygridwidth=0.8,
            spinewidth=0.8,
            bottomspinecolor=:gray45,
            leftspinecolor=:gray45,
            xtickcolor=:gray45,
            ytickcolor=:gray45,
        ),
    )
    return nothing
end

"""Separate difficulty levels from the general-quality boundary in a compact legend."""
function difficulty_legend!(slot, values; nbanks=2)
    elements = [
        [
            LineElement(; color=PUB_DELTA_COLORS[v], linewidth=2),
            MarkerElement(;
                color=PUB_DELTA_COLORS[v], marker=PUB_DELTA_MARKERS[v], markersize=9
            ),
        ] for v in values
    ]
    boundary = [MarkerElement(; color=:gray25, marker=:diamond, markersize=10)]
    return Legend(
        slot, [elements, boundary], [string.(values), ["ρ = 1"]],
        ["Difficulty (δ)", "General quality only\n(δ has no effect)"];
        PUB_LEGEND..., nbanks, groupgap=40,
    )
end

"""Add an external legend for the general-quality share, with redundant symbols."""
function composition_legend!(slot, values; nbanks=2)
    elements = [
        MarkerElement(;
            color=PUB_RHO_COLORS[v], marker=PUB_RHO_MARKERS[v], markersize=11
        ) for v in values
    ]
    return Legend(slot, elements, string.(values), PUB_RHO_LABEL; PUB_LEGEND..., nbanks)
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared kwargs
# ─────────────────────────────────────────────────────────────────────────────

"""Axis keyword arguments for consistent styling. `T` is the final period."""
function ax_kw(T::Int)
    step = if T <= 50
        10
    elseif T <= 100
        20
    else
        50
    end
    return (;
        titlesize=TITLE_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
        xticks=0:step:T,
    )
end

"""Compact legend style shared across all legends. The semi-transparent
background keeps data visible wherever a legend must overlap the plot area."""
const LEG_KW = (;
    labelsize=18,
    titlesize=19,
    patchsize=(18, 13),
    padding=(6, 6, 4, 4),
    rowgap=1,
    patchlabelgap=5,
    framewidth=0.5,
    backgroundcolor=(:white, 0.72),
)

# ─────────────────────────────────────────────────────────────────────────────
# Time-series helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Rolling mean with window. NaN-safe: skips NaN values in the window."""
function rolling_mean(v::AbstractVector, window::Int)
    n = length(v)
    out = fill(NaN, n)
    for i in 1:n
        isnan(v[i]) && continue
        lo = max(1, i - window + 1)
        vals = filter(!isnan, @view v[lo:i])
        !isempty(vals) && (out[i] = mean(vals))
    end
    return out
end

"""Mean of the finite values in `v`, or NaN when no finite values are present."""
function nanmean_or_nan(v)
    total = 0.0
    n = 0
    for x in v
        isnan(x) && continue
        total += x
        n += 1
    end
    return n == 0 ? NaN : total / n
end

"""Access fraction = access / (access + assessment), or NaN."""
function access_fraction(mdf::DataFrame)
    total = mdf.access_count .+ mdf.assessment_count
    return [t > 0 ? mdf.access_count[i] / t : NaN for (i, t) in enumerate(total)]
end

"""
    plot_metric!(ax, periods, mdfs, metric_fn; label, color, window, line_kw...)

Plot thin per-seed lines (alpha=0.45) and a thick ensemble mean (linewidth=2.5).
Additional line keywords, such as `linestyle`, are forwarded to both the seed
and ensemble layers. The ensemble mean is NaN when fewer than half the seeds
have valid data.
"""
function plot_metric!(
    ax,
    periods,
    mdfs::Vector{DataFrame},
    metric_fn;
    label::String="",
    color=COL_AGENT,
    window::Int=20,
    line_kw...,
)
    n_seeds = length(mdfs)
    seed_vals = [rolling_mean(metric_fn(mdf), window) for mdf in mdfs]
    for sv in seed_vals
        lines!(ax, periods, sv; color=(color, 0.45), linewidth=0.8, line_kw...)
    end
    ensemble = [
        let vs = [sv[t] for sv in seed_vals]
            nv = count(!isnan, vs)
            nv > n_seeds / 2 ? mean(v for v in vs if !isnan(v)) : NaN
        end for t in eachindex(periods)
    ]
    lines!(ax, periods, ensemble; color=color, linewidth=2.5, label=label, line_kw...)
end

"""Add a vertical dashed line at the burn-in period."""
function add_burnin!(ax, T_burn::Int)
    vlines!(ax, [T_burn]; color=:gray30, linestyle=:dash, linewidth=1.5)
end

"""Add an explanatory footer caption spanning `cols` columns at `row`."""
function add_footer!(fig, row::Int, cols; n_seeds::Int, window::Int, T_burn::Int)
    txt =
        "Thin lines: individual seeds ($n_seeds). " *
        "Thick: ensemble mean (shown when majority of seeds have data). " *
        "Dashed vertical: burn-in (t=$T_burn). " *
        "Smoothing: $window-period rolling mean."
    Label(
        fig[row, cols],
        txt;
        fontsize=FOOTER_FS,
        color=:gray30,
        halign=:center,
        tellwidth=false,
    )
end

"""Standard panel layout sizing."""
function apply_layout!(
    fig; n_panel_rows::Int=5, n_panel_cols::Int=4, suptitle_row::Int=0, footer_row::Int=-1
)
    colsize!(fig.layout, 0, Fixed(30))
    for r in 1:n_panel_rows
        rowsize!(fig.layout, r, Auto(1))
    end
    rowsize!(fig.layout, suptitle_row, Fixed(22))
    if footer_row > 0
        rowsize!(fig.layout, footer_row, Fixed(30))
    end
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 10)
end
