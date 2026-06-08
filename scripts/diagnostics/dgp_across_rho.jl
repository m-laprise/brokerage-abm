"""
    dgp_across_rho.jl

Characterize how the matching function (the DGP) changes with the channel mix rho,
using the initialization stage only (no simulation). For a fixed seed the geometry,
agent types, and the matrices A, B, c are RNG-identical across rho; rho only
re-weights the additive-quality term and the bilinear interaction term in

    signal(x_i, x_j) = rho * 1/2 (x_i'c + x_j'c) + (1-rho) * g(x_i,x_j) * x_i'A x_j,
    g = 1 + delta * sign(x_i'B x_j).

For each rho we form the noiseless N x N output matrix F (agents PC1-sorted) and
report its structure: heatmap, distribution of pairwise values, and the singular
spectrum / effective rank. Writes report figures qD1, qD2 and a stats line.
"""

using TransientBrokerage
using TransientBrokerage: generate_matching_env, generate_curve_geometry, generate_agent_types
using LinearAlgebra: svdvals
using Statistics: mean, std
using StableRNGs: StableRNG
using JLD2

include(joinpath(@__DIR__, "..", "exploration_common.jl"))   # build_ordered_output_matrix
include(joinpath(@__DIR__, "..", "figure_style.jl"))         # CairoMakie + COL_*/FS

const OUT = "/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438/report"
const RHO5 = [0.0, 0.3, 0.5, 0.7, 1.0]
const RHO_COLORS = Dict(0.0 => :seagreen, 0.3 => :mediumaquamarine, 0.5 => :goldenrod,
                        0.7 => :darkorange, 1.0 => :firebrick)
const SEEDS = 1:3

function env_for(rho, seed)
    p = default_params(; rho=rho, seed=seed)
    rng = StableRNG(p.seed)                     # reset per call -> A,B,c,types identical across rho
    geo = generate_curve_geometry(p.d, p.s, rng)
    types, _ = generate_agent_types(p.N, geo, p.sigma_x, rng)
    env = generate_matching_env(p.d, p.rho, p.delta, p.sigma_eps, types, rng;
                                sigma_x=p.sigma_x, curve_geo=geo)
    return types, env, p
end

# effective rank: number of components for 90% of (centered) spectral energy
rank90(svals) = (cv = cumsum(svals .^ 2) ./ sum(svals .^ 2); findfirst(>=(0.90), cv))

# ── accumulate effective rank over seeds; keep seed-1 matrices/spectra for plots ──
r90 = Dict(r => Int[] for r in RHO5)
Fkeep = Dict{Float64,Matrix{Float64}}()
svkeep = Dict{Float64,Vector{Float64}}()
histkeep = Dict{Float64,Vector{Float64}}()

for seed in SEEDS, r in RHO5
    types, env, p = env_for(r, seed)
    F = build_ordered_output_matrix(types, env).F
    Fc = F .- mean(F)                           # drop the constant offset; measure structure
    sv = svdvals(Fc)
    push!(r90[r], rank90(sv))
    if seed == 1
        Fkeep[r] = F
        svkeep[r] = sv ./ sv[1]
        histkeep[r] = [F[i, j] for j in 2:p.N for i in 1:(j-1)]
    end
end

println("=== effective rank (components for 90% of centered spectral energy), mean over seeds ===")
for r in RHO5
    println("  rho=$r  r90 = $(round(mean(r90[r]); digits=1))  (seeds: $(r90[r]))")
end

# ── qD1: output-matrix heatmaps across rho (PC1-sorted, centered) ──
let
    fig = Figure(size=(1320, 470))
    Label(fig[0, 1:3], "Matching-function structure across the channel mix ρ";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    showr = [0.0, 0.5, 1.0]
    titles = ["ρ = 0  (pure complementarity)", "ρ = 0.5  (mixed)", "ρ = 1  (pure quality)"]
    for (ci, r) in enumerate(showr)
        F = Fkeep[r]; N = size(F, 1); Fc = F .- mean(F)
        m = maximum(abs, Fc)
        ax = Axis(fig[1, ci]; title=titles[ci], titlesize=TITLE_FS, aspect=1,
            xlabel="agent j (quality order)", ylabel = ci == 1 ? "agent i (quality order)" : "",
            xlabelsize=LABEL_FS, ylabelsize=LABEL_FS, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS)
        hm = heatmap!(ax, 1:N, 1:N, Fc; colormap=:RdBu, colorrange=(-m, m))
        ci == 3 && Colorbar(fig[1, 4], hm; label="output, centered", labelsize=LABEL_FS, ticklabelsize=TICK_FS)
    end
    rowsize!(fig.layout, 0, Fixed(30)); colgap!(fig.layout, 10); colsize!(fig.layout, 4, Fixed(14))
    save(joinpath(OUT, "qD1_dgp_matrix.png"), fig); println("  qD1 done")
end

# ── qD2: singular spectrum (effective rank) + value distribution across rho ──
let
    shown = filter(r -> r != 1.0, RHO5)   # rho=1 is the rank-2 additive limit; omit it
    fig = Figure(size=(1180, 460))
    Label(fig[0, 1:2], "Matching-function spectrum and value distribution (ρ < 1)";
        fontsize=SUPTITLE_FS, font=:bold, tellwidth=false)
    axa = Axis(fig[1, 1]; title="Singular spectrum (centered output matrix)",
        xlabel="component k", ylabel="σ_k / σ_1", yscale=log10, titlesize=TITLE_FS,
        xlabelsize=LABEL_FS+1, ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
        limits=((0, 25), (8e-3, 1.3)))
    for r in shown
        sv = max.(svkeep[r], 1e-4)
        scatterlines!(axa, 1:min(25, length(sv)), sv[1:min(25, length(sv))];
            color=RHO_COLORS[r], markersize=5, label="ρ=$(r)  (r₉₀=$(round(Int, mean(r90[r]))))")
    end
    axislegend(axa, "Channel mix"; position=:rt, LEG_KW...)
    # right panel: density on a log y-axis. Use a manual histogram (density!'s fill to a
    # zero baseline is undefined under log10), plotting only the non-empty bins.
    axb = Axis(fig[1, 2]; title="Distribution of pairwise match output",
        xlabel="noiseless output  q", ylabel="density (log scale)", yscale=log10, titlesize=TITLE_FS,
        xlabelsize=LABEL_FS+1, ylabelsize=LABEL_FS+1, xticklabelsize=TICK_FS, yticklabelsize=TICK_FS,
        limits=(nothing, (1e-3, 2.0)))
    allv = vcat([histkeep[r] for r in shown]...)
    lo, hi = extrema(allv); nb = 70; bw = (hi - lo) / nb
    centers = [lo + (i - 0.5) * bw for i in 1:nb]
    for r in shown
        counts = zeros(Int, nb)
        for x in histkeep[r]
            b = clamp(floor(Int, (x - lo) / bw) + 1, 1, nb); counts[b] += 1
        end
        dens = counts ./ (length(histkeep[r]) * bw)
        keep = dens .> 0
        lines!(axb, centers[keep], dens[keep]; color=RHO_COLORS[r], linewidth=2, label="ρ=$(r)")
    end
    colgap!(fig.layout, 18); rowsize!(fig.layout, 0, Fixed(30))
    save(joinpath(OUT, "qD2_dgp_spectrum.png"), fig); println("  qD2 done")
end

jldsave(joinpath(OUT, "..", "dgp_across_rho.jld2");
    rho=RHO5, r90=Dict(r => r90[r] for r in RHO5), svals=svkeep)
println("DONE")
