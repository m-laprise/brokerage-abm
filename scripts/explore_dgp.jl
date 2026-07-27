"""
    explore_dgp.jl

Visualize the data-generating process across parameter configurations.
For each config, produces:
- Output matrix heatmap (agents sorted by PC1): {tag}_matrix.png
- SVD spectrum + cumulative variance: {tag}_svd.png
- Regime gain map: {tag}_regime.png

Configs sweep the parameters controlling matching difficulty (rho, delta, s)
at a shared environment seed. Data is saved to JLD2 for later analysis.

Usage: julia --project --threads=auto scripts/explore_dgp.jl
"""

using BrokerageABM
using BrokerageABM: generate_matching_env, generate_curve_geometry,
                          generate_agent_types
using CairoMakie
using LinearAlgebra: svdvals, norm
using Statistics: mean, std
using StableRNGs: StableRNG
using JLD2

include(joinpath(@__DIR__, "exploration_common.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "data", "figures", "dgp")
const DATADIR = joinpath(@__DIR__, "..", "data", "dgp")
mkpath(OUTDIR)
mkpath(DATADIR)

# ─────────────────────────────────────────────────────────────────────────────
# Configurations: (tag, kwargs)
# ─────────────────────────────────────────────────────────────────────────────

configs = [
    (tag="baseline",               kwargs=(;)),
    # rho sweep
    (tag="rho00_pureinteraction",  kwargs=(rho=0.0,)),
    (tag="rho10_weakquality",      kwargs=(rho=0.10,)),
    (tag="rho30_mildinteraction",  kwargs=(rho=0.30,)),
    (tag="rho70_mildquality",      kwargs=(rho=0.70,)),
    (tag="rho90_strongquality",    kwargs=(rho=0.90,)),
    (tag="rho100_purequality",     kwargs=(rho=1.0,)),
    # delta sweep
    (tag="delta00_noregime",       kwargs=(delta=0.0,)),
    (tag="delta25_weakregime",     kwargs=(delta=0.25,)),
    (tag="delta75_strongregime",   kwargs=(delta=0.75,)),
    # s sweep
    (tag="s2_lowdim",              kwargs=(s=2,)),
    (tag="s4_middim",              kwargs=(s=4,)),
]

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

function plot_matrix(F, G; tag, d, rho, delta, s, N)
    crange = let m = maximum(abs, F .- mean(F)); (mean(F) - m, mean(F) + m) end

    fig = Figure(; size=(800, 650), figure_padding=(10, 15, 5, 5))
    Label(fig[0, 1:2],
          "Noiseless output matrix (N=$N, d=$d, rho=$rho, delta=$delta, s=$s)";
          fontsize=13, font=:bold, halign=:center, tellwidth=false)

    # Top: output heatmap
    ax1 = Axis(fig[1, 1]; xlabel="Agent j (PC1 order)", ylabel="Agent i (PC1 order)",
               titlesize=11, xlabelsize=10, ylabelsize=10, aspect=1)
    hm = heatmap!(ax1, 1:N, 1:N, F; colormap=:RdBu, colorrange=crange)
    Colorbar(fig[1, 2], hm; label="q = Q + signal", labelsize=10, ticklabelsize=9)

    # Bottom: histogram
    ax2 = Axis(fig[2, 1]; xlabel="q", ylabel="Count",
               titlesize=11, xlabelsize=10, ylabelsize=10)
    # Upper triangle only (unique pairs)
    vals = [F[i, j] for j in 2:N for i in 1:(j-1)]
    hist!(ax2, vals; bins=80, color=(:steelblue, 0.7), strokewidth=0.5, strokecolor=:gray40)
    vlines!(ax2, [mean(vals)]; color=:crimson, linestyle=:dash, linewidth=1)

    rowsize!(fig.layout, 0, Fixed(22))
    rowsize!(fig.layout, 1, Relative(0.65))
    colsize!(fig.layout, 2, Fixed(30))
    save(joinpath(OUTDIR, "$(tag)_matrix.png"), fig)
end

function plot_svd(F; tag, d, rho, delta, s, N)
    svals = svdvals(F)
    svals_norm = svals ./ svals[1]
    cumvar = cumsum(svals .^ 2) ./ sum(svals .^ 2)

    fig = Figure(; size=(700, 280), figure_padding=(10, 15, 5, 5))
    Label(fig[0, 1:2],
          "SVD of output matrix (N=$N, d=$d, rho=$rho, delta=$delta, s=$s)";
          fontsize=13, font=:bold, halign=:center, tellwidth=false)

    n_show = min(50, length(svals_norm))
    ax1 = Axis(fig[1, 1]; title="Singular values (normalized)",
               xlabel="Component", ylabel="sigma_k / sigma_1",
               titlesize=11, xlabelsize=10, ylabelsize=10)
    scatterlines!(ax1, 1:n_show, svals_norm[1:n_show]; markersize=4, color=:steelblue)

    ax2 = Axis(fig[1, 2]; title="Cumulative variance explained",
               xlabel="Number of components", ylabel="Fraction",
               titlesize=11, xlabelsize=10, ylabelsize=10,
               limits=(nothing, (0, 1.05)))
    cumvar_plot = vcat(0.0, cumvar[1:n_show])
    scatterlines!(ax2, 0:n_show, cumvar_plot; markersize=4, color=:steelblue)
    hlines!(ax2, [0.90, 0.95]; color=:gray60, linestyle=:dash, linewidth=0.8)
    text!(ax2, n_show * 0.75, 0.90; text="90%", fontsize=9, color=:gray40,
          align=(:left, :bottom))
    text!(ax2, n_show * 0.75, 0.95; text="95%", fontsize=9, color=:gray40,
          align=(:left, :bottom))

    rowsize!(fig.layout, 0, Fixed(22))
    save(joinpath(OUTDIR, "$(tag)_svd.png"), fig)
end

function plot_regime(G; tag, d, rho, delta, s, N)
    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1]; title="Regime gain g(x_i, x_j) (delta=$delta)",
              xlabel="Agent j (PC1 order)", ylabel="Agent i (PC1 order)", aspect=1)
    hm = heatmap!(ax, 1:N, 1:N, G; colormap=:RdYlGn,
                  colorrange=(1 - delta - 0.05, 1 + delta + 0.05))
    Colorbar(fig[1, 2], hm; label="Gain")
    save(joinpath(OUTDIR, "$(tag)_regime.png"), fig)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

println("DGP exploration: $(length(configs)) configurations\n")

for c in configs
    p = default_params(; seed=42, c.kwargs...)
    rng = StableRNG(p.seed)

    geo = generate_curve_geometry(p.d, p.s, rng)
    types, _ = generate_agent_types(p.N, geo, p.sigma_x, rng)
    env = generate_matching_env(p.d, p.rho, p.delta, p.sigma_eps, types, rng;
                                sigma_x=p.sigma_x, curve_geo=geo)

    print("  $(c.tag) (rho=$(p.rho), delta=$(p.delta), s=$(p.s)) ... ")
    ordered = build_ordered_output_matrix(types, env; include_regime=true)
    F = ordered.F
    G = ordered.G_regime
    pc1 = ordered.pc1_sorted

    # Save data
    jldsave(joinpath(DATADIR, "$(c.tag).jld2");
            F=F, G_regime=G, pc1=pc1,
            A=env.A, B=env.B, c=env.c,
            rho=p.rho, delta=p.delta, sigma_eps=p.sigma_eps,
            d=p.d, s=p.s, N=p.N)

    meta = (tag=c.tag, d=p.d, rho=p.rho, delta=p.delta, s=p.s, N=p.N)
    plot_matrix(F, G; meta...)
    plot_svd(F; meta...)
    if p.delta > 0
        plot_regime(G; meta...)
    end

    # Quick stats
    vals = [F[i, j] for j in 2:p.N for i in 1:(j-1)]
    svals = svdvals(F)
    cumvar = cumsum(svals .^ 2) ./ sum(svals .^ 2)
    println("mean=$(round(mean(vals), digits=3)), std=$(round(std(vals), digits=3)), rank90=$(findfirst(>=(0.90), cumvar))")
end

println("\nFigures: $OUTDIR")
println("Data: $DATADIR")
println("Done.")
