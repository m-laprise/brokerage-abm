"""
    explore_capture.jl

Run Model 1 (principal mode / resource capture) across parameter configurations,
with the base model as a dashed-gray reference. Produces dense 5x3 dynamics panels.

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
mkpath(OUTDIR)
mkpath(DATADIR)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dynamics figure: M1 (solid colors) + base (dashed gray reference)
# ─────────────────────────────────────────────────────────────────────────────

function plot_capture_ensemble(m1_mdfs::Vector{DataFrame},
                               base_mdfs::Union{Vector{DataFrame}, Nothing},
                               suptitle::String, filename::String;
                               T_burn::Int=30, window::Int=20)
    n_seeds = length(m1_mdfs)
    periods = m1_mdfs[1].period
    T = last(periods)
    xlims = (first(periods), T)
    akw = ax_kw(T)

    pm!(ax, fn; kw...) = plot_metric!(ax, periods, m1_mdfs, fn; window=window, kw...)

    function base_ref!(ax, fn)
        base_mdfs === nothing && return
        seed_vals = [rolling_mean(fn(mdf), window) for mdf in base_mdfs]
        ensemble = [let vs = [sv[t] for sv in seed_vals]
            nv = count(!isnan, vs)
            nv > n_seeds / 2 ? mean(v for v in vs if !isnan(v)) : NaN
        end for t in eachindex(periods)]
        lines!(ax, periods, ensemble; color=COL_BASE_REF, linewidth=1.5,
               linestyle=:dash, label="Base")
    end

    fig = Figure(; size=(1500, 1100), figure_padding=(5, 15, 5, 5))
    all_axes = Axis[]
    newax(pos; kw...) = (a = Axis(pos; kw...); push!(all_axes, a); a)

    Label(fig[0, 1:4], suptitle; fontsize=SUPTITLE_FS, font=:bold,
          halign=:center, tellwidth=false)

    AGT_SEL = "Agents (pooled)"
    BRK_SEL = "Broker (pooled)"
    AGT_HLD = "Agents (mean)"
    BRK_HLD = "Broker (mean)"

    # ── Row 1: Market ──
    Label(fig[1, 0], "Market"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[1, 1]; title="Outsourcing rate (positions)", ylabel="Rate",
              limits=(xlims, (-0.02, 1.02)), akw...)
    pm!(ax, mdf -> mdf.outsourcing_rate; label="M1", color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.outsourcing_rate)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[1, 2]; title="Matches by channel", ylabel="Count",
              limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> Float64.(mdf.n_self_matches); label="Self", color=COL_AGENT)
    pm!(ax, mdf -> Float64.(mdf.n_broker_standard); label="Broker (std)", color=COL_BROKER)
    pm!(ax, mdf -> Float64.(mdf.n_broker_principal); label="Broker (principal)", color=COL_CAPTURE)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[1, 3]; title="Total demand & matches", ylabel="Count",
              limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> Float64.(mdf.total_demand); label="Demand (positions)", color=COL_DIAG)
    pm!(ax, mdf -> Float64.(mdf.n_total_matches); label="Matches", color=COL_AGENT)
    base_ref!(ax, mdf -> Float64.(mdf.n_total_matches))
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[1, 4]; title="Available agents & broker access", ylabel="Count",
              limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> Float64.(mdf.n_available); label="Available (M1)", color=COL_DIAG)
    pm!(ax, mdf -> Float64.(mdf.broker_access_size); label="Broker access set (M1)",
        color=COL_BROKER)
    pm!(ax, mdf -> Float64.(mdf.roster_size); label="Standing roster (M1)",
        color=COL_BASE_REF, linestyle=:dash)
    base_ref!(ax, mdf -> Float64.(mdf.n_available))
    axislegend(ax; position=:rb, LEG_KW...)

    # ── Row 2: Selected ──
    Label(fig[2, 0], "Selected"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[2, 1]; title="Selected rank corr.", ylabel="Spearman ρ",
              limits=(xlims, (0, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_selected_rank; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_rank; label=BRK_SEL, color=COL_BROKER)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[2, 2]; title="Selected R²", ylabel="R²",
              limits=(xlims, (nothing, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_selected_r2; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_r2; label=BRK_SEL, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[2, 3]; title="Selected RMSE", ylabel="RMSE",
              limits=(xlims, (0, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_selected_rmse; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_rmse; label=BRK_SEL, color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[2, 4]; title="Selected bias", ylabel="Bias",
              limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.agent_selected_bias; label=AGT_SEL, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_bias; label=BRK_SEL, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rt, LEG_KW...)

    # ── Row 3: Holdout ──
    Label(fig[3, 0], "Holdout"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[3, 1]; title="Holdout rank corr.", ylabel="Spearman ρ",
              limits=(xlims, (0, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_holdout_rank; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_rank; label=BRK_HLD, color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.broker_holdout_rank)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[3, 2]; title="Holdout R²", ylabel="R²",
              limits=(xlims, (nothing, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_holdout_r2; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_r2; label=BRK_HLD, color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.broker_holdout_r2)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[3, 3]; title="Holdout RMSE", ylabel="RMSE",
              limits=(xlims, (0, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_holdout_rmse; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_rmse; label=BRK_HLD, color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.broker_holdout_rmse)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[3, 4]; title="Holdout bias", ylabel="Bias",
              limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.agent_holdout_bias; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_bias; label=BRK_HLD, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rt, LEG_KW...)

    # ── Row 4: Advantage ──
    Label(fig[4, 0], "Advantage"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[4, 1]; title="Holdout rank gap", ylabel="Δ ρ",
              limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.rank_gap; label="M1", color=COL_GAP)
    base_ref!(ax, mdf -> mdf.rank_gap)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[4, 2]; title="Holdout R² gap", ylabel="Δ R²",
              limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.r2_gap; label="M1", color=COL_GAP)
    base_ref!(ax, mdf -> mdf.r2_gap)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[4, 3]; title="Holdout RMSE gap", ylabel="Δ RMSE",
              limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.rmse_gap; label="M1", color=COL_GAP)
    base_ref!(ax, mdf -> mdf.rmse_gap)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[4, 4]; title="Access fraction", ylabel="Fraction",
              limits=(xlims, (-0.02, 1.02)), akw...)
    pm!(ax, mdf -> access_fraction(mdf); label="M1", color=COL_ACCESS)
    base_ref!(ax, mdf -> access_fraction(mdf))
    axislegend(ax; position=:rb, LEG_KW...)

    # ── Row 5: Dynamics ──
    Label(fig[5, 0], "Dynamics"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[5, 1]; title="Broker betweenness", xlabel="Period",
              ylabel="Betweenness centrality", limits=(xlims, (0, 1.02)), akw...,
              xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.betweenness; label="M1", color=COL_BROKER)
    base_ref!(ax, mdf -> mdf.betweenness)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[5, 2]; title="Mean output by channel", xlabel="Period",
              ylabel="Output", limits=(xlims, (0, nothing)), akw...,
              xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.q_self_mean; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.q_broker_standard_mean; label="Broker (std)", color=COL_BROKER)
    pm!(ax, mdf -> mdf.q_broker_principal_mean; label="Broker (principal)", color=COL_CAPTURE)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[5, 3]; title="Satisfaction + reputation", xlabel="Period",
              ylabel="Satisfaction", limits=(xlims, (0, nothing)), akw...,
              xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.mean_satisfaction_self; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.mean_satisfaction_broker; label="Broker", color=COL_BROKER)
    pm!(ax, mdf -> mdf.broker_reputation; label="Reputation", color=COL_REPUTATION)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[5, 4]; title="Principal-mode share", xlabel="Period",
              ylabel="P^t", limits=(xlims, (-0.02, 1.02)), akw...,
              xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.principal_mode_share; color=COL_CAPTURE)
    hlines!(ax, [0.0, 1.0]; color=:gray80, linestyle=:dot, linewidth=0.8)

    # ── Burn-in lines ──
    for a in all_axes; add_burnin!(a, T_burn); end

    # ── Footer ──
    base_note = base_mdfs === nothing ? "" : " Dashed gray: base model reference."
    txt = "Thin lines: individual seeds ($n_seeds). " *
          "Thick: ensemble mean. " *
          "Dashed vertical: burn-in (t=$T_burn). " *
          "Smoothing: $window-period rolling mean." * base_note
    Label(fig[6, 1:4], txt; fontsize=FOOTER_FS, color=:gray30,
          halign=:center, tellwidth=false)

    # ── Layout ──
    apply_layout!(fig; n_panel_rows=5, n_panel_cols=4, suptitle_row=0, footer_row=6)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved: $filename")
end

# ─────────────────────────────────────────────────────────────────────────────
# Supplementary capture figure: outcome, decision quality, dependency (§12i)
# ─────────────────────────────────────────────────────────────────────────────
#
# Two rows × five columns. Row 1 covers capture outcome and decision quality;
# row 2 covers broker dependency across agents plus a histogram of per-period
# mean capture surplus in the last `hist_window` periods of the simulation
# (end-steady-state distribution, pooled across seeds).

function plot_capture_suppl(m1_mdfs::Vector{DataFrame},
                            suptitle::String, filename::String;
                            T_burn::Int=30, window::Int=20, hist_window::Int=20)
    n_seeds = length(m1_mdfs)
    periods = m1_mdfs[1].period
    T = last(periods)
    xlims = (first(periods), T)
    akw = ax_kw(T)

    pm!(ax, fn; kw...) = plot_metric!(ax, periods, m1_mdfs, fn; window=window, kw...)

    fig = Figure(; size=(1800, 600), figure_padding=(5, 15, 5, 5))
    all_axes = Axis[]
    newax(pos; kw...) = (a = Axis(pos; kw...); push!(all_axes, a); a)

    Label(fig[0, 1:5], suptitle; fontsize=SUPTITLE_FS, font=:bold,
          halign=:center, tellwidth=false)

    # ── Row 1: Capture outcome and decision quality ──
    Label(fig[1, 0], "Outcome & decision"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[1, 1]; title="Mean capture surplus Δq̄",
              ylabel="q_ij − q̄_j", limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.capture_surplus_mean; label="M1", color=COL_CAPTURE)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[1, 2]; title="Capture loss rate",
              ylabel="share with Δq < 0", limits=(xlims, (-0.02, 1.02)), akw...)
    pm!(ax, mdf -> mdf.capture_loss_rate; label="M1", color=COL_CAPTURE)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[1, 3]; title="Capture loss magnitude",
              ylabel="mean |Δq| | Δq<0", limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> mdf.capture_loss_magnitude; label="M1", color=COL_CAPTURE)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[1, 4]; title="Capture decision rank corr.",
              ylabel="Spearman ρ(Δq̂, Δq)", limits=(xlims, (-1.02, 1.02)), akw...)
    pm!(ax, mdf -> mdf.capture_decision_rank; label="M1", color=COL_GAP)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[1, 5]; title="Capture decision RMSE",
              ylabel="RMSE(q̂_b, q_ij) | principal", limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> mdf.capture_decision_rmse; label="M1", color=COL_GAP)
    axislegend(ax; position=:rt, LEG_KW...)

    # ── Row 2: Broker dependency across agents + histogram ──
    Label(fig[2, 0], "Dependency"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[2, 1]; title="Mean broker dependency D_j",
              ylabel="mean D_j", limits=(xlims, (-0.02, 1.02)), akw...,
              xlabel="Period", xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.broker_dependency_mean; label="M1", color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[2, 2]; title="D_j 90th percentile",
              ylabel="D_j p90", limits=(xlims, (-0.02, 1.02)), akw...,
              xlabel="Period", xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.broker_dependency_p90; label="M1", color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[2, 3]; title="Fraction of agents with D_j > 0.5",
              ylabel="share", limits=(xlims, (-0.02, 1.02)), akw...,
              xlabel="Period", xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.broker_dependency_frac_above_half; label="M1", color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[2, 4]; title="Gini coefficient of D_j",
              ylabel="Gini", limits=(xlims, (-0.02, 1.02)), akw...,
              xlabel="Period", xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.broker_dependency_gini; label="M1", color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)

    # Histogram: per-period mean capture surplus pooled over the last
    # `hist_window` periods across all seeds. A steady-state distribution.
    hist_ax = Axis(fig[2, 5]; title="Mean Δq̄ distribution (last $hist_window periods)",
                   ylabel="density", xlabel="mean Δq̄",
                   titlesize=TITLE_FS, ylabelsize=LABEL_FS,
                   xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
                   xlabelsize=LABEL_FS)
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
        hist!(hist_ax, pooled; bins=min(20, max(5, length(pooled) ÷ 5)),
              color=(COL_CAPTURE, 0.6), strokecolor=COL_CAPTURE,
              strokewidth=0.8, normalization=:pdf)
        vlines!(hist_ax, [mean(pooled)]; color=COL_CAPTURE, linewidth=1.5,
                linestyle=:dash, label="mean = $(round(mean(pooled); digits=3))")
        vlines!(hist_ax, [0.0]; color=:gray50, linewidth=0.8)
        axislegend(hist_ax; position=:rt, LEG_KW...)
    else
        text!(hist_ax, 0.5, 0.5; text="no principal-mode matches", align=(:center, :center),
              fontsize=TICK_FS, color=:gray30, space=:relative)
    end

    # ── Burn-in lines on time-series only (skip histogram) ──
    for a in all_axes
        a === hist_ax && continue
        add_burnin!(a, T_burn)
    end

    # ── Footer ──
    txt = "Thin lines: individual seeds ($n_seeds). " *
          "Thick: ensemble mean. " *
          "Dashed vertical: burn-in (t=$T_burn). " *
          "Smoothing: $window-period rolling mean. " *
          "Histogram: per-period mean Δq pooled across seeds over final $hist_window periods."
    Label(fig[3, 1:5], txt; fontsize=FOOTER_FS, color=:gray30,
          halign=:center, tellwidth=false)

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
    (tag="baseline",               label="Baseline (M1)", kwargs=(;)),
    (tag="rho00_pureinteraction",  label="Pure interaction (M1, ρ=0.0)", kwargs=(rho=0.0,)),
    (tag="rho30_mildinteraction",  label="Mild interaction (M1, ρ=0.30)", kwargs=(rho=0.30,)),
    (tag="rho70_mildquality",      label="Mild quality (M1, ρ=0.70)", kwargs=(rho=0.70,)),
    (tag="rho100_purequality",     label="Pure quality (M1, ρ=1.0)", kwargs=(rho=1.0,)),
    (tag="delta00_noregime",       label="No regime (M1, δ=0.0)", kwargs=(delta=0.0,)),
    (tag="delta75_strongregime",   label="Strong regime (M1, δ=0.75)", kwargs=(delta=0.75,)),
    (tag="s2_lowdim",              label="Low-dim curve (M1, s=2)", kwargs=(s=2,)),
    (tag="eta01_stable",           label="Stable market (M1, η=0.01)", kwargs=(eta=0.01,)),
    (tag="eta05_volatile",         label="Volatile market (M1, η=0.05)", kwargs=(eta=0.05,)),
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
        m1_mdfs = saved["mdfs"]
    else
        m1_mdfs = run_ensemble(; base_kwargs=c.kwargs, T=T, N=N_SIM,
                               n_seeds=N_SEEDS, enable_principal=true)
        jldsave(datafile; mdfs=m1_mdfs)
        println("  Saved M1 data")
    end

    # Load base model reference if available
    base_file = joinpath(BASE_DATADIR, "$(c.tag).jld2")
    base_mdfs = if isfile(base_file)
        println("  Loading base reference from $base_file")
        JLD2.load(base_file)["mdfs"]
    else
        println("  No base reference found (run explore_base_model.jl first)")
        nothing
    end

    plot_capture_ensemble(m1_mdfs, base_mdfs, "$(c.label) [N=$N_SIM, T=$T]", "$(c.tag)_capture.png")
    plot_capture_suppl(m1_mdfs, "$(c.label) [N=$N_SIM, T=$T] — capture supplement",
                       "$(c.tag)_capture_suppl.png"; T_burn=30, window=20, hist_window=20)

    # Summary
    tails = [mdf[max(1, end-49):end, :] for mdf in m1_mdfs]
    combined = vcat(tails...)
    println("  Summary (last 50 periods):")
    println("    Principal share: $(round(mean(combined.principal_mode_share), digits=3))")
    println("    Outsourcing (position share): $(round(mean(combined.outsourcing_rate), digits=3))")
    println("    R² gap: $(round(nanmean_or_nan(combined.r2_gap), digits=3))")
    println("    Mean capture surplus: $(round(nanmean_or_nan(combined.capture_surplus_mean), digits=3))")
    println("    Mean broker dependency: $(round(nanmean_or_nan(combined.broker_dependency_mean), digits=3))")
    println()
end

println("Figures: $OUTDIR")
println("Data: $DATADIR")
println("Done.")
