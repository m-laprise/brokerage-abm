"""
    explore_capture.jl

Run Model 1 (client-origin resource capture) across parameter configurations,
with the base model as a dashed-gray reference. Produces dense dynamics panels.

Data cached as JLD2; pass --rerun to force re-simulation.

Usage: julia --project --threads=auto scripts/explore_capture.jl
       julia --project --threads=auto scripts/explore_capture.jl --baseline
       julia --project --threads=auto scripts/explore_capture.jl --rerun
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; start Julia with --threads=auto"

using TransientBrokerage
using DataFrames: DataFrame, nrow
using JLD2
using Statistics: mean

include(joinpath(@__DIR__, "exploration_common.jl"))
include(joinpath(@__DIR__, "figure_style.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "data", "figures", "capture")
const DATADIR = joinpath(@__DIR__, "..", "data", "sims", "capture")
const BASE_DATADIR = joinpath(@__DIR__, "..", "data", "sims", "exploration")
const CAPTURE_OUTPUT_SCHEMA_VERSION = 2
mkpath(OUTDIR)
mkpath(DATADIR)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

const CAPTURE_REQUIRED_FIELDS = (
    :period,
    :outsourcing_rate,
    :n_self_matches,
    :n_broker_standard,
    :n_broker_principal,
    :n_total_matches,
    :total_demand,
    :q_self_mean,
    :q_broker_standard_mean,
    :q_broker_principal_mean,
    :broker_access_size,
    :roster_size,
    :agent_selected_rank,
    :broker_selected_rank,
    :agent_selected_r2,
    :broker_selected_r2,
    :agent_selected_rmse,
    :broker_selected_rmse,
    :agent_selected_bias,
    :broker_selected_bias,
    :agent_holdout_rank,
    :broker_holdout_rank,
    :agent_holdout_r2,
    :broker_holdout_r2,
    :agent_holdout_rmse,
    :broker_holdout_rmse,
    :agent_holdout_bias,
    :broker_holdout_bias,
    :rank_gap,
    :r2_gap,
    :rmse_gap,
    :access_count,
    :assessment_count,
    :betweenness,
    :mean_satisfaction_self,
    :mean_satisfaction_broker,
    :broker_reputation,
    :principal_mode_share,
    :captured_origin_count,
    :captured_position_count,
    :principal_acceptance_rate,
    :capture_scaled_mae,
    :capture_ready,
    :capture_surplus_mean,
    :capture_loss_rate,
    :principal_accepted,
    :principal_rejected,
    :capture_decision_rmse,
)

function has_fields(mdfs::AbstractVector, fields)::Bool
    isempty(mdfs) && return false
    for mdf in mdfs
        cols = Set(Symbol.(names(mdf)))
        all(field -> field in cols, fields) || return false
    end
    return true
end

function capture_cache_current(saved, mdfs::AbstractVector)::Bool
    get(saved, "capture_output_schema_version", 0) == CAPTURE_OUTPUT_SCHEMA_VERSION ||
        return false
    return has_fields(mdfs, CAPTURE_REQUIRED_FIELDS)
end

function run_capture_config(c, datafile::String, T::Int, N_SIM::Int, N_SEEDS::Int)
    mdfs = run_ensemble(;
        base_kwargs=c.kwargs, T=T, N=N_SIM, n_seeds=N_SEEDS, enable_principal=true
    )
    jldsave(datafile; mdfs=mdfs, capture_output_schema_version=CAPTURE_OUTPUT_SCHEMA_VERSION)
    println("  Saved M1 data: $datafile")
    return mdfs
end

# ─────────────────────────────────────────────────────────────────────────────
# Dynamics figure: M1 (solid colors) + base (dashed gray reference)
# ─────────────────────────────────────────────────────────────────────────────

function plot_capture_ensemble(
    m1_mdfs::Vector{DataFrame},
    base_mdfs::Union{Vector{DataFrame},Nothing},
    suptitle::String,
    filename::String;
    T_burn::Int=30,
    window::Int=20,
)
    n_seeds = length(m1_mdfs)
    periods = m1_mdfs[1].period
    T = last(periods)
    xlims = (first(periods), T)
    akw = ax_kw(T)

    pm!(ax, fn; kw...) = plot_metric!(ax, periods, m1_mdfs, fn; window=window, kw...)

    function base_ref!(ax, fn)
        base_mdfs === nothing && return nothing
        seed_vals = [rolling_mean(fn(mdf), window) for mdf in base_mdfs]
        ensemble = [
            let vs = [sv[t] for sv in seed_vals]
                nv = count(!isnan, vs)
                nv > n_seeds / 2 ? mean(v for v in vs if !isnan(v)) : NaN
            end for t in eachindex(periods)
        ]
        lines!(
            ax,
            periods,
            ensemble;
            color=COL_BASE_REF,
            linewidth=1.5,
            linestyle=:dash,
            label="Base",
        )
    end

    fig = Figure(; size=(1500, 1100), figure_padding=(5, 15, 5, 5))
    all_axes = Axis[]
    newax(pos; kw...) = (a=Axis(pos; kw...); push!(all_axes, a); a)

    Label(
        fig[0, 1:4],
        suptitle;
        fontsize=SUPTITLE_FS,
        font=:bold,
        halign=:center,
        tellwidth=false,
    )

    AGT_SEL = "Agents (pooled)"
    BRK_SEL = "Broker (pooled)"
    AGT_HLD = "Agents (mean)"
    BRK_HLD = "Broker (mean)"

    # ── Row 1: Market ──
    Label(
        fig[1, 0],
        "Market";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )

    ax = newax(
        fig[1, 1];
        title="Outsourcing rate (positions)",
        ylabel="Rate",
        limits=(xlims, (-0.02, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.outsourcing_rate; label="M1", color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.outsourcing_rate)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[1, 2];
        title="Matches by channel",
        ylabel="Count",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> Float64.(mdf.n_self_matches); label="Self", color=COL_AGENT)
    pm!(ax, mdf -> Float64.(mdf.n_broker_standard); label="Broker (std)", color=COL_BROKER)
    pm!(
        ax,
        mdf -> Float64.(mdf.n_broker_principal);
        label="Broker (principal)",
        color=COL_CAPTURE,
    )
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[1, 3];
        title="Total demand & matches",
        ylabel="Count",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> Float64.(mdf.total_demand); label="Demand (positions)", color=COL_DIAG)
    pm!(ax, mdf -> Float64.(mdf.n_total_matches); label="Matches", color=COL_AGENT)
    base_ref!(ax, mdf -> Float64.(mdf.n_total_matches))
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[1, 4];
        title="Broker access & capture volume",
        ylabel="Count",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(
        ax,
        mdf -> Float64.(mdf.broker_access_size);
        label="Broker access set (M1)",
        color=COL_BROKER,
    )
    pm!(
        ax,
        mdf -> Float64.(mdf.roster_size);
        label="Standing roster (M1)",
        color=COL_BASE_REF,
        linestyle=:dash,
    )
    pm!(
        ax,
        mdf -> Float64.(mdf.captured_position_count);
        label="Captured positions",
        color=COL_CAPTURE,
    )
    pm!(
        ax,
        mdf -> Float64.(mdf.captured_origin_count);
        label="Captured origins",
        color=COL_DIAG,
    )
    axislegend(ax; position=:rb, LEG_KW...)

    # ── Row 2: Selected ──
    Label(
        fig[2, 0],
        "Selected";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )

    ax = newax(
        fig[2, 1];
        title="Selected rank corr.",
        ylabel="Spearman ρ",
        limits=(xlims, (0, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.agent_selected_rank; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_rank; label=BRK_SEL, color=COL_BROKER)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[2, 2]; title="Selected R²", ylabel="R²", limits=(xlims, (nothing, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_selected_r2; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_r2; label=BRK_SEL, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[2, 3]; title="Selected RMSE", ylabel="RMSE", limits=(xlims, (0, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_selected_rmse; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_rmse; label=BRK_SEL, color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[2, 4]; title="Selected bias", ylabel="Bias", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.agent_selected_bias; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_bias; label=BRK_SEL, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rt, LEG_KW...)

    # ── Row 3: Holdout ──
    Label(
        fig[3, 0],
        "Holdout";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )

    ax = newax(
        fig[3, 1];
        title="Holdout rank corr.",
        ylabel="Spearman ρ",
        limits=(xlims, (0, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.agent_holdout_rank; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_rank; label=BRK_HLD, color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.broker_holdout_rank)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[3, 2]; title="Holdout R²", ylabel="R²", limits=(xlims, (nothing, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_holdout_r2; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_r2; label=BRK_HLD, color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.broker_holdout_r2)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[3, 3]; title="Holdout RMSE", ylabel="RMSE", limits=(xlims, (0, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_holdout_rmse; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_rmse; label=BRK_HLD, color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.broker_holdout_rmse)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[3, 4]; title="Holdout bias", ylabel="Bias", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.agent_holdout_bias; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_bias; label=BRK_HLD, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rt, LEG_KW...)

    # ── Row 4: Advantage ──
    Label(
        fig[4, 0],
        "Advantage";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )

    ax = newax(
        fig[4, 1]; title="Holdout rank gap", ylabel="Δ ρ", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.rank_gap; label="M1", color=COL_GAP)
    base_ref!(ax, mdf -> mdf.rank_gap)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[4, 2]; title="Holdout R² gap", ylabel="Δ R²", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.r2_gap; label="M1", color=COL_GAP)
    base_ref!(ax, mdf -> mdf.r2_gap)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[4, 3];
        title="Holdout RMSE gap",
        ylabel="Δ RMSE",
        limits=(xlims, nothing),
        akw...,
    )
    pm!(ax, mdf -> mdf.rmse_gap; label="M1", color=COL_GAP)
    base_ref!(ax, mdf -> mdf.rmse_gap)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[4, 4];
        title="Access fraction",
        ylabel="Fraction",
        limits=(xlims, (-0.02, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> access_fraction(mdf); label="M1", color=COL_ACCESS)
    base_ref!(ax, mdf -> access_fraction(mdf))
    axislegend(ax; position=:rb, LEG_KW...)

    # ── Row 5: Dynamics ──
    Label(
        fig[5, 0],
        "Dynamics";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )

    ax = newax(
        fig[5, 1];
        title="Broker betweenness",
        xlabel="Period",
        ylabel="Betweenness centrality",
        limits=(xlims, (0, 1.02)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.betweenness; label="M1", color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.betweenness)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[5, 2];
        title="Mean output by channel",
        xlabel="Period",
        ylabel="Output",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.q_self_mean; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.q_broker_standard_mean; label="Broker (std)", color=COL_BROKER)
    pm!(
        ax,
        mdf -> mdf.q_broker_principal_mean;
        label="Broker (principal)",
        color=COL_CAPTURE,
    )
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[5, 3];
        title="Satisfaction + reputation",
        xlabel="Period",
        ylabel="Satisfaction",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.mean_satisfaction_self; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.mean_satisfaction_broker; label="Broker", color=COL_BROKER)
    pm!(ax, mdf -> mdf.broker_reputation; label="Reputation", color=COL_REPUTATION)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(
        fig[5, 4];
        title="Captured broker demand share",
        xlabel="Period",
        ylabel="P^t",
        limits=(xlims, (-0.02, 1.02)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.principal_mode_share; color=COL_CAPTURE)
    hlines!(ax, [0.0, 1.0]; color=:gray80, linestyle=:dot, linewidth=0.8)

    # ── Burn-in lines ──
    for a in all_axes
        ;
        add_burnin!(a, T_burn);
    end

    # ── Footer ──
    base_note = base_mdfs === nothing ? "" : " Dashed gray: base model reference."
    txt =
        "Thin lines: individual seeds ($n_seeds). " *
        "Thick: ensemble mean. " *
        "Dashed vertical: burn-in (t=$T_burn). " *
        "Smoothing: $window-period rolling mean." *
        base_note
    Label(
        fig[6, 1:4], txt; fontsize=FOOTER_FS, color=:gray30, halign=:center, tellwidth=false
    )

    # ── Layout ──
    apply_layout!(fig; n_panel_rows=5, n_panel_cols=4, suptitle_row=0, footer_row=6)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved: $filename")
end

# ─────────────────────────────────────────────────────────────────────────────
# Supplementary capture figure: capture outcomes and mechanics (§12)
# ─────────────────────────────────────────────────────────────────────────────
#
# Two rows by five columns. Row 1 covers capture outcome and decision quality;
# row 2 covers client-origin lot capture volume plus a histogram of per-period
# mean capture surplus in the last `hist_window` periods of the simulation.

function plot_capture_suppl(
    m1_mdfs::Vector{DataFrame},
    suptitle::String,
    filename::String;
    T_burn::Int=30,
    window::Int=20,
    hist_window::Int=20,
    capture_error_threshold::Float64=default_params().capture_error_threshold,
)
    n_seeds = length(m1_mdfs)
    periods = m1_mdfs[1].period
    T = last(periods)
    xlims = (first(periods), T)
    akw = ax_kw(T)

    pm!(ax, fn; kw...) = plot_metric!(ax, periods, m1_mdfs, fn; window=window, kw...)

    fig = Figure(; size=(1800, 600), figure_padding=(5, 15, 5, 5))
    all_axes = Axis[]
    newax(pos; kw...) = (a=Axis(pos; kw...); push!(all_axes, a); a)

    Label(
        fig[0, 1:5],
        suptitle;
        fontsize=SUPTITLE_FS,
        font=:bold,
        halign=:center,
        tellwidth=false,
    )

    # ── Row 1: Capture outcome and decision quality ──
    Label(
        fig[1, 0],
        "Outcome & decision";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )

    ax = newax(
        fig[1, 1];
        title="Mean principal surplus",
        ylabel="q_ij - ask_i",
        limits=(xlims, nothing),
        akw...,
    )
    pm!(ax, mdf -> mdf.capture_surplus_mean; label="M1", color=COL_CAPTURE)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[1, 2];
        title="Capture loss rate",
        ylabel="share with Δq < 0",
        limits=(xlims, (-0.02, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.capture_loss_rate; label="M1", color=COL_CAPTURE)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[1, 3];
        title="Principal acceptance rate",
        ylabel="Accepted / captured positions",
        limits=(xlims, (-0.02, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.principal_acceptance_rate; label="M1", color=COL_CAPTURE)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[1, 4];
        title="Capture decision RMSE",
        ylabel="RMSE(q̂_b, q_ij) | principal",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> mdf.capture_decision_rmse; label="M1", color=COL_GAP)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[1, 5];
        title="Capture readiness",
        ylabel="Scaled MAE / ready",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> mdf.capture_scaled_mae; label="Scaled MAE", color=COL_GAP)
    pm!(ax, mdf -> Float64.(mdf.capture_ready); label="Ready", color=COL_CAPTURE)
    hlines!(ax, [capture_error_threshold]; color=:gray50, linewidth=0.8, linestyle=:dot)
    axislegend(ax; position=:rt, LEG_KW...)

    # ── Row 2: Capture mechanics + histogram ──
    Label(
        fig[2, 0],
        "Mechanics";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )

    ax = newax(
        fig[2, 1];
        title="Captured origins",
        ylabel="Origins",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabel="Period",
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> Float64.(mdf.captured_origin_count); label="M1", color=COL_CAPTURE)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[2, 2];
        title="Captured positions",
        ylabel="Positions",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabel="Period",
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> Float64.(mdf.captured_position_count); label="M1", color=COL_CAPTURE)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[2, 3];
        title="Accepted and rejected principal slots",
        ylabel="Positions",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabel="Period",
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> Float64.(mdf.principal_accepted); label="Accepted", color=COL_CAPTURE)
    pm!(ax, mdf -> Float64.(mdf.principal_rejected); label="Rejected", color=COL_DIAG)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(
        fig[2, 4];
        title="Derived lock-in effects",
        ylabel="Count",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabel="Period",
        xlabelsize=LABEL_FS,
    )
    pm!(
        ax,
        mdf -> Float64.(mdf.principal_accepted);
        label="Direct ties not formed",
        color=COL_BROKER,
    )
    pm!(
        ax,
        mdf -> 2.0 .* Float64.(mdf.principal_accepted);
        label="Agent observations not recorded",
        color=COL_DIAG,
    )
    axislegend(ax; position=:rt, LEG_KW...)

    # Histogram: per-period mean capture surplus pooled over the last
    # `hist_window` periods across all seeds. A steady-state distribution.
    hist_ax = Axis(
        fig[2, 5];
        title="Mean surplus distribution (last $hist_window periods)",
        ylabel="density",
        xlabel="mean q_ij - ask_i",
        titlesize=TITLE_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
        xlabelsize=LABEL_FS,
    )
    push!(all_axes, hist_ax)
    pooled = Float64[]
    for mdf in m1_mdfs
        n = nrow(mdf)
        lo = max(1, n - hist_window + 1)
        for v in mdf.capture_surplus_mean[lo:n]
            isnan(v) || push!(pooled, v)
        end
    end
    if !isempty(pooled)
        hist!(
            hist_ax,
            pooled;
            bins=min(20, max(5, length(pooled) ÷ 5)),
            color=(COL_CAPTURE, 0.6),
            strokecolor=COL_CAPTURE,
            strokewidth=0.8,
            normalization=:pdf,
        )
        vlines!(
            hist_ax,
            [mean(pooled)];
            color=COL_CAPTURE,
            linewidth=1.5,
            linestyle=:dash,
            label="mean = $(round(mean(pooled); digits=3))",
        )
        vlines!(hist_ax, [0.0]; color=:gray50, linewidth=0.8)
        axislegend(hist_ax; position=:rt, LEG_KW...)
    else
        text!(
            hist_ax,
            0.5,
            0.5;
            text="no accepted principal matches",
            align=(:center, :center),
            fontsize=TICK_FS,
            color=:gray30,
            space=:relative,
        )
    end

    # ── Burn-in lines on time-series only (skip histogram) ──
    for a in all_axes
        a === hist_ax && continue
        add_burnin!(a, T_burn)
    end

    # ── Footer ──
    txt =
        "Thin lines: individual seeds ($n_seeds). " *
        "Thick: ensemble mean. " *
        "Dashed vertical: burn-in (t=$T_burn). " *
        "Smoothing: $window-period rolling mean. " *
        "Histogram: per-period mean surplus pooled across seeds over final $hist_window periods."
    Label(
        fig[3, 1:5], txt; fontsize=FOOTER_FS, color=:gray30, halign=:center, tellwidth=false
    )

    # ── Layout ──
    colsize!(fig.layout, 0, Fixed(30))
    rowsize!(fig.layout, 0, Fixed(22))
    rowsize!(fig.layout, 1, Auto(1))
    rowsize!(fig.layout, 2, Auto(1))
    rowsize!(fig.layout, 3, Fixed(30))
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 10)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved: $filename")
end

# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────

configs = [
    (tag="baseline", label="Baseline (M1)", kwargs=(;)),
    (tag="rho00_pureinteraction", label="Pure interaction (M1, ρ=0.0)", kwargs=(rho=0.0,)),
    (
        tag="rho30_mildinteraction",
        label="Mild interaction (M1, ρ=0.30)",
        kwargs=(rho=0.30,),
    ),
    (tag="rho70_mildquality", label="Mild quality (M1, ρ=0.70)", kwargs=(rho=0.70,)),
    (tag="rho100_purequality", label="Pure quality (M1, ρ=1.0)", kwargs=(rho=1.0,)),
    (tag="delta00_noregime", label="No regime (M1, δ=0.0)", kwargs=(delta=0.0,)),
    (tag="delta75_strongregime", label="Strong regime (M1, δ=0.75)", kwargs=(delta=0.75,)),
    (tag="s2_lowdim", label="Low-dim curve (M1, s=2)", kwargs=(s=2,)),
    (tag="eta01_stable", label="Stable market (M1, η=0.01)", kwargs=(eta=0.01,)),
    (tag="eta05_volatile", label="Volatile market (M1, η=0.05)", kwargs=(eta=0.05,)),
]

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

T = 200
N_SIM = 1000
N_SEEDS = 5
RERUN = "--rerun" in ARGS
BASELINE_ONLY = "--baseline" in ARGS

if BASELINE_ONLY
    configs = filter(c -> c.tag == "baseline", configs)
end

println("Capture exploration: $(length(configs)) configs, $N_SEEDS seeds, N=$N_SIM, T=$T")
RERUN && println("  --rerun: forcing re-simulation")
BASELINE_ONLY && println("  --baseline: running baseline only")
println()

for (idx, c) in enumerate(configs)
    println("[$idx/$(length(configs))] $(c.label)")
    datafile = joinpath(DATADIR, "$(c.tag).jld2")

    if !RERUN && isfile(datafile)
        println("  Loading cached M1 data")
        saved = JLD2.load(datafile)
        m1_mdfs = get(saved, "mdfs", DataFrame[])
        if !capture_cache_current(saved, m1_mdfs)
            println("  Cache schema is stale, re-simulating")
            m1_mdfs = run_capture_config(c, datafile, T, N_SIM, N_SEEDS)
        end
    else
        m1_mdfs = run_capture_config(c, datafile, T, N_SIM, N_SEEDS)
    end

    # Load base model reference if available
    base_file = joinpath(BASE_DATADIR, "$(c.tag).jld2")
    base_mdfs = if isfile(base_file)
        println("  Loading base reference from $base_file")
        ref = JLD2.load(base_file)["mdfs"]
        has_fields(ref, CAPTURE_REQUIRED_FIELDS) ? ref : begin
            println("  Base reference cache is stale; skipping dashed reference")
            nothing
        end
    else
        println("  No base reference found (run explore_base_model.jl first)")
        nothing
    end

    plot_capture_ensemble(
        m1_mdfs, base_mdfs, "$(c.label) [N=$N_SIM, T=$T]", "$(c.tag)_capture.png"
    )
    threshold = default_params(; c.kwargs...).capture_error_threshold
    plot_capture_suppl(
        m1_mdfs,
        "$(c.label) [N=$N_SIM, T=$T] capture supplement",
        "$(c.tag)_capture_suppl.png";
        T_burn=30,
        window=20,
        hist_window=20,
        capture_error_threshold=threshold,
    )

    # Summary
    tails = [mdf[max(1, end - 49):end, :] for mdf in m1_mdfs]
    combined = vcat(tails...)
    println("  Summary (last 50 periods):")
    println(
        "    Captured broker demand share: $(round(mean(combined.principal_mode_share), digits=3))",
    )
    println(
        "    Outsourcing (position share): $(round(mean(combined.outsourcing_rate), digits=3))",
    )
    println("    R² gap: $(round(nanmean_or_nan(combined.r2_gap), digits=3))")
    println(
        "    Mean capture surplus: $(round(nanmean_or_nan(combined.capture_surplus_mean), digits=3))",
    )
    println(
        "    Principal acceptance rate: $(round(nanmean_or_nan(combined.principal_acceptance_rate), digits=3))",
    )
    println(
        "    Captured positions: $(round(mean(combined.captured_position_count), digits=1))"
    )
    println()
end

println("Figures: $OUTDIR")
println("Data: $DATADIR")
println("Done.")
