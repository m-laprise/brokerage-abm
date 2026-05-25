"""
    explore_base_model.jl

Run the base model (no capture) across parameter configurations that control
matching difficulty (rho, delta, s, eta) with multiple seeds per config.
Produces per-config 5x4 dynamics panels, a network-statistics figure, and DGP
visualization (matrix, SVD).

Data is cached as JLD2; pass --rerun to force re-simulation.

Usage: julia --project --threads=auto scripts/explore_base_model.jl
       julia --project --threads=auto scripts/explore_base_model.jl --baseline
       julia --project --threads=auto scripts/explore_base_model.jl --rerun
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; start Julia with --threads=auto"

using TransientBrokerage
using TransientBrokerage: generate_matching_env, generate_curve_geometry,
                          generate_agent_types
using DataFrames: DataFrame
using LinearAlgebra: svdvals
using StableRNGs: StableRNG
using JLD2
using Statistics: mean

include(joinpath(@__DIR__, "exploration_common.jl"))
include(joinpath(@__DIR__, "figure_style.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "data", "figures", "exploration")
const DATADIR = joinpath(@__DIR__, "..", "data", "sims", "exploration")
const NETWORK_DEGREE_STATS_VERSION = 2
mkpath(OUTDIR)
mkpath(DATADIR)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

const NETWORK_DEGREE_FIELDS = (:mean_degree, :median_degree, :min_degree, :max_degree)

function has_network_degree_stats(mdfs::Vector{DataFrame})
    isempty(mdfs) && return false
    cols = Set(Symbol.(names(mdfs[1])))
    return all(field -> field in cols, NETWORK_DEGREE_FIELDS)
end

cache_degree_stats_version(saved) = get(saved, "network_degree_stats_version", 0)

# ─────────────────────────────────────────────────────────────────────────────
# Dynamics figure (5×4 panel, matching v0.1 quality)
# ─────────────────────────────────────────────────────────────────────────────

function plot_ensemble(mdfs::Vector{DataFrame}, suptitle::String, filename::String;
                       T_burn::Int=30, window::Int=20)
    n_seeds = length(mdfs)
    periods = mdfs[1].period
    T = last(periods)
    xlims = (first(periods), T)
    akw = ax_kw(T)
    pm!(ax, fn; kw...) = plot_metric!(ax, periods, mdfs, fn; window=window, kw...)

    fig = Figure(; size=(1500, 1100), figure_padding=(5, 15, 5, 5))
    all_axes = Axis[]
    newax(pos; kw...) = (a = Axis(pos; kw...); push!(all_axes, a); a)

    Label(fig[0, 1:4], suptitle; fontsize=SUPTITLE_FS, font=:bold,
          halign=:center, tellwidth=false)

    # Labels for selected (pooled) vs holdout (per-agent mean)
    AGT_SEL = "Agents (pooled)"
    BRK_SEL = "Broker (pooled)"
    AGT_HLD = "Agents (mean)"
    BRK_HLD = "Broker (mean)"

    # ── Row 1: Market ──
    Label(fig[1, 0], "Market"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[1, 1]; title="Outsourcing rate (positions)", ylabel="Rate",
              limits=(xlims, (-0.02, 1.02)), akw...)
    pm!(ax, mdf -> mdf.outsourcing_rate; color=COL_BROKER)

    ax = newax(fig[1, 2]; title="Matches by channel", ylabel="Count",
              limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> Float64.(mdf.n_self_matches); label="Self", color=COL_AGENT)
    pm!(ax, mdf -> Float64.(mdf.n_broker_standard); label="Broker", color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[1, 3]; title="Total demand & matches", ylabel="Count",
              limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> Float64.(mdf.total_demand); label="Demand (positions)", color=COL_DIAG)
    pm!(ax, mdf -> Float64.(mdf.n_total_matches); label="Matches", color=COL_AGENT)
    axislegend(ax; position=:rt, LEG_KW...)

    ax = newax(fig[1, 4]; title="Broker access and roster", ylabel="Count",
              limits=(xlims, (0, nothing)), akw...)
    pm!(ax, mdf -> Float64.(mdf.broker_access_size); label="Broker access set", color=COL_BROKER)
    pm!(ax, mdf -> Float64.(mdf.roster_size); label="Standing roster", color=COL_BASE_REF,
        linestyle=:dash)
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
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[3, 2]; title="Holdout R²", ylabel="R²",
              limits=(xlims, (nothing, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_holdout_r2; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_r2; label=BRK_HLD, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[3, 3]; title="Holdout RMSE", ylabel="RMSE",
              limits=(xlims, (0, 1.02)), akw...)
    pm!(ax, mdf -> mdf.agent_holdout_rmse; label=AGT_HLD, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_rmse; label=BRK_HLD, color=COL_BROKER)
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
    pm!(ax, mdf -> mdf.rank_gap; color=COL_GAP)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)

    ax = newax(fig[4, 2]; title="Holdout R² gap", ylabel="Δ R²",
              limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.r2_gap; color=COL_GAP)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)

    ax = newax(fig[4, 3]; title="Holdout RMSE gap", ylabel="Δ RMSE",
              limits=(xlims, nothing), akw...)
    pm!(ax, mdf -> mdf.rmse_gap; color=COL_GAP)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)

    ax = newax(fig[4, 4]; title="Access fraction", ylabel="Fraction",
              limits=(xlims, (-0.02, 1.02)), akw...)
    pm!(ax, mdf -> access_fraction(mdf); color=COL_ACCESS)

    # ── Row 5: Dynamics ──
    Label(fig[5, 0], "Dynamics"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    ax = newax(fig[5, 1]; title="Broker betweenness", xlabel="Period",
              ylabel="Betweenness centrality", limits=(xlims, (0, 1.02)), akw...,
              xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.betweenness; color=COL_BROKER)

    ax = newax(fig[5, 2]; title="Mean output by channel", xlabel="Period",
              ylabel="Output", limits=(xlims, (0, nothing)), akw...,
              xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.q_self_mean; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.q_broker_standard_mean; label="Broker", color=COL_BROKER)
    axislegend(ax; position=:rb, LEG_KW...)

    ax = newax(fig[5, 3]; title="Satisfaction + reputation", xlabel="Period",
              ylabel="Satisfaction", limits=(xlims, (0, nothing)), akw...,
              xlabelsize=LABEL_FS)
    pm!(ax, mdf -> mdf.mean_satisfaction_self; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.mean_satisfaction_broker; label="Broker", color=COL_BROKER)
    pm!(ax, mdf -> mdf.broker_reputation; label="Reputation", color=COL_REPUTATION)
    axislegend(ax; position=:rb, LEG_KW...)

    # Row 5, col 4: empty in base model

    # ── Burn-in lines ──
    for a in all_axes; add_burnin!(a, T_burn); end

    # ── Footer ──
    add_footer!(fig, 6, 1:4; n_seeds=n_seeds, window=window, T_burn=T_burn)

    # ── Layout ──
    apply_layout!(fig; n_panel_rows=5, n_panel_cols=4, suptitle_row=0, footer_row=6)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved dynamics: $filename")
end

function plot_network_stats(mdfs::Vector{DataFrame}, suptitle::String, filename::String;
                            T_burn::Int=30, window::Int=20)
    n_seeds = length(mdfs)
    periods = mdfs[1].period
    T = last(periods)
    xlims = (first(periods), T)
    akw = ax_kw(T)
    pm!(ax, fn; kw...) = plot_metric!(ax, periods, mdfs, fn; window=window, kw...)

    fig = Figure(; size=(1500, 360), figure_padding=(5, 15, 5, 5))
    all_axes = Axis[]
    newax(pos; kw...) = (a = Axis(pos; kw...); push!(all_axes, a); a)

    Label(fig[0, 1:4], "Agent network statistics: $suptitle"; fontsize=SUPTITLE_FS, font=:bold,
          halign=:center, tellwidth=false)
    Label(fig[1, 0], "Agents"; fontsize=ROW_LABEL_FS, font=:bold,
          rotation=π/2, tellheight=false)

    panels = [
        (title="Mean degree", field=:mean_degree),
        (title="Median degree", field=:median_degree),
        (title="Min degree", field=:min_degree),
        (title="Max degree", field=:max_degree),
    ]

    for (col, panel) in enumerate(panels)
        ax = newax(fig[1, col];
                   title=panel.title,
                   xlabel="Period",
                   ylabel="Degree",
                   limits=(xlims, (0, nothing)),
                   akw...,
                   xlabelsize=LABEL_FS)
        pm!(ax, mdf -> mdf[!, panel.field]; color=COL_DIAG)
    end

    for ax in all_axes
        add_burnin!(ax, T_burn)
    end

    add_footer!(fig, 2, 1:4; n_seeds=n_seeds, window=window, T_burn=T_burn)
    apply_layout!(fig; n_panel_rows=1, n_panel_cols=4, suptitle_row=0, footer_row=2)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved network stats: $filename")
end

# ─────────────────────────────────────────────────────────────────────────────
# DGP figures (matrix heatmap + histogram, SVD spectrum)
# ─────────────────────────────────────────────────────────────────────────────

function plot_dgp_matrix(F; tag, rho, delta, s, N)
    crange = let m = maximum(abs, F .- mean(F)); (mean(F) - m, mean(F) + m) end
    fig = Figure(; size=(800, 650))
    Label(fig[0, 1:2], "Output matrix (N=$N, ρ=$rho, δ=$delta, s=$s)";
          fontsize=SUPTITLE_FS-2, font=:bold, halign=:center, tellwidth=false)

    ax1 = Axis(fig[1, 1]; xlabel="Agent j (PC1)", ylabel="Agent i (PC1)",
               aspect=1, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS)
    hm = heatmap!(ax1, 1:N, 1:N, F; colormap=:RdBu, colorrange=crange)
    Colorbar(fig[1, 2], hm; label="q", labelsize=LABEL_FS, ticklabelsize=TICK_FS)

    vals = [F[i, j] for j in 2:N for i in 1:(j-1)]
    ax2 = Axis(fig[2, 1]; xlabel="q", ylabel="Count",
               xlabelsize=LABEL_FS, ylabelsize=LABEL_FS)
    hist!(ax2, vals; bins=80, color=(COL_AGENT, 0.7), strokewidth=0.5, strokecolor=:gray40)
    vlines!(ax2, [mean(vals)]; color=COL_BROKER, linestyle=:dash)

    rowsize!(fig.layout, 0, Fixed(22))
    rowsize!(fig.layout, 1, Relative(0.65))
    colsize!(fig.layout, 2, Fixed(30))
    save(joinpath(OUTDIR, "$(tag)_matrix.png"), fig)
    println("  Saved: $(tag)_matrix.png")
end

function plot_dgp_svd(F; tag, rho, delta, s, N)
    svals = svdvals(F)
    cumvar = cumsum(svals .^ 2) ./ sum(svals .^ 2)
    n_show = min(50, length(svals))

    fig = Figure(; size=(700, 280))
    Label(fig[0, 1:2], "SVD of output matrix (ρ=$rho, δ=$delta, s=$s)";
          fontsize=SUPTITLE_FS-2, font=:bold, halign=:center, tellwidth=false)

    ax1 = Axis(fig[1, 1]; title="Singular values (normalized)",
               xlabel="Component", ylabel="σ/σ₁",
               titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS)
    sn = svals ./ svals[1]
    scatterlines!(ax1, 1:n_show, sn[1:n_show]; markersize=4, color=COL_AGENT)

    ax2 = Axis(fig[1, 2]; title="Cumulative variance explained",
               xlabel="Components", ylabel="Fraction",
               titlesize=TITLE_FS, xlabelsize=LABEL_FS, ylabelsize=LABEL_FS,
               limits=(nothing, (0, 1.05)))
    scatterlines!(ax2, 0:n_show, vcat(0.0, cumvar[1:n_show]); markersize=4, color=COL_AGENT)
    hlines!(ax2, [0.90, 0.95]; color=:gray60, linestyle=:dash, linewidth=0.8)
    text!(ax2, n_show * 0.75, 0.90; text="90%", fontsize=TICK_FS, color=:gray40,
          align=(:left, :bottom))
    text!(ax2, n_show * 0.75, 0.95; text="95%", fontsize=TICK_FS, color=:gray40,
          align=(:left, :bottom))

    rowsize!(fig.layout, 0, Fixed(22))
    save(joinpath(OUTDIR, "$(tag)_svd.png"), fig)
    println("  Saved: $(tag)_svd.png")
end

# ─────────────────────────────────────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────────────────────────────────────

configs = [
    (tag="baseline",
     label="Baseline (ρ=0.50, δ=0.50, s=8, η=0.02)",
     kwargs=(;)),
    # rho sweep
    (tag="rho00_pureinteraction",
     label="Pure interaction (ρ=0.0)",
     kwargs=(rho=0.0,)),
    (tag="rho10_weakquality",
     label="Weak quality (ρ=0.10)",
     kwargs=(rho=0.10,)),
    (tag="rho30_mildinteraction",
     label="Mild interaction (ρ=0.30)",
     kwargs=(rho=0.30,)),
    (tag="rho70_mildquality",
     label="Mild quality (ρ=0.70)",
     kwargs=(rho=0.70,)),
    (tag="rho90_strongquality",
     label="Strong quality (ρ=0.90)",
     kwargs=(rho=0.90,)),
    (tag="rho100_purequality",
     label="Pure quality (ρ=1.0)",
     kwargs=(rho=1.0,)),
    # delta sweep
    (tag="delta00_noregime",
     label="No regime effect (δ=0.0)",
     kwargs=(delta=0.0,)),
    (tag="delta25_weakregime",
     label="Weak regime (δ=0.25)",
     kwargs=(delta=0.25,)),
    (tag="delta75_strongregime",
     label="Strong regime (δ=0.75)",
     kwargs=(delta=0.75,)),
    # s sweep
    (tag="s2_lowdim",
     label="Low-dim curve (s=2)",
     kwargs=(s=2,)),
    (tag="s4_middim",
     label="Mid-dim curve (s=4)",
     kwargs=(s=4,)),
    # eta sweep
    (tag="eta01_stable",
     label="Stable market (η=0.01)",
     kwargs=(eta=0.01,)),
    (tag="eta05_volatile",
     label="Volatile market (η=0.05)",
     kwargs=(eta=0.05,)),
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

println("Base model exploration: $(length(configs)) configs, $N_SEEDS seeds, N=$N_SIM, T=$T")
RERUN && println("  --rerun: forcing re-simulation")
BASELINE_ONLY && println("  --baseline: running baseline only")
println()

for (idx, c) in enumerate(configs)
    println("[$idx/$(length(configs))] $(c.label)")
    datafile = joinpath(DATADIR, "$(c.tag).jld2")

    if !RERUN && isfile(datafile)
        println("  Loading cached data: $datafile")
        saved = JLD2.load(datafile)
        mdfs = saved["mdfs"]
        if cache_degree_stats_version(saved) != NETWORK_DEGREE_STATS_VERSION ||
           !has_network_degree_stats(mdfs)
            println("  Cache has stale degree summaries, re-simulating")
            mdfs = run_ensemble(; base_kwargs=c.kwargs, T=T, N=N_SIM, n_seeds=N_SEEDS)
            jldsave(datafile; mdfs=mdfs,
                    network_degree_stats_version=NETWORK_DEGREE_STATS_VERSION)
            println("  Saved data: $datafile")
        end
    else
        mdfs = run_ensemble(; base_kwargs=c.kwargs, T=T, N=N_SIM, n_seeds=N_SEEDS)
        jldsave(datafile; mdfs=mdfs,
                network_degree_stats_version=NETWORK_DEGREE_STATS_VERSION)
        println("  Saved data: $datafile")
    end

    # Dynamics panel
    plot_ensemble(mdfs, "$(c.label) [N=$N_SIM, T=$T]", "$(c.tag)_dynamics.png")
    plot_network_stats(mdfs, "$(c.label) [N=$N_SIM, T=$T]", "$(c.tag)_network_stats.png")

    # DGP figures
    p = default_params(; N=N_SIM, seed=42, c.kwargs...)
    rng = StableRNG(p.seed)
    geo = generate_curve_geometry(p.d, p.s, rng)
    types, _ = generate_agent_types(p.N, geo, p.sigma_x, rng)
    env = generate_matching_env(p.d, p.rho, p.delta, p.sigma_eps, types, rng;
                                sigma_x=p.sigma_x, curve_geo=geo)
    ordered = build_ordered_output_matrix(types, env)
    plot_dgp_matrix(ordered.F; tag=c.tag, rho=p.rho, delta=p.delta, s=p.s, N=p.N)
    plot_dgp_svd(ordered.F; tag=c.tag, rho=p.rho, delta=p.delta, s=p.s, N=p.N)

    # Summary stats
    tail_dfs = [mdf[max(1, end-49):end, :] for mdf in mdfs]
    combined = vcat(tail_dfs...)
    println("  Summary (last 50 periods, pooled):")
    println("    Outsourcing (position share): $(round(mean(combined.outsourcing_rate), digits=3))")
    println("    Broker R²: $(round(nanmean_or_nan(combined.broker_holdout_r2), digits=3))")
    println("    Agent R²: $(round(nanmean_or_nan(combined.agent_holdout_r2), digits=3))")
    println("    R² gap: $(round(nanmean_or_nan(combined.r2_gap), digits=3))")
    println("    Betweenness: $(round(mean(combined.betweenness), digits=4))")
    println()
end

println("Figures: $OUTDIR")
println("Data: $DATADIR")
println("Done.")
