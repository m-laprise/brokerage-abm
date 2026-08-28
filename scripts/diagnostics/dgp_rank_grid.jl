"""
    dgp_rank_grid.jl

Effective rank r90 of the matching function for every (rho, delta) combination used in
the sweep, from the initialization stage only (no simulation). For a fixed seed the
geometry, types, and A, B, c are RNG-identical across (rho, delta); the knobs only
re-weight / re-gate the terms. r90 = number of singular components carrying 90% of the
centered spectral energy of the noiseless pairwise output matrix.

Covers the complete rho x delta sweep grid under 3 seeds. Saves a Dict
(rho, delta) => mean r90 to _results/dgp_rank_grid.jld2 for the diagnostic figures.

Usage: julia --project --threads=auto scripts/diagnostics/dgp_rank_grid.jl
"""

using BrokerageABM
using BrokerageABM: generate_matching_env, generate_curve_geometry, generate_agent_types
using LinearAlgebra: svdvals
using Statistics: mean
using StableRNGs: StableRNG
using JLD2
include(joinpath(@__DIR__, "..", "exploration_common.jl"))   # build_ordered_output_matrix
include(joinpath(@__DIR__, "..", "sweep", "sweep_config.jl"))

const RHOS = copy(RHO_OAT)
const DELTAS = copy(DELTA_VALS)
const SEEDS = 1:3

rank90(sv) = (cv=cumsum(sv .^ 2) ./ sum(sv .^ 2); findfirst(>=(0.90), cv))

acc = Dict((r, d) => Float64[] for r in RHOS, d in DELTAS)
for seed in SEEDS
    p = default_params(; seed=seed)
    for dl in DELTAS, r in RHOS
        rng = StableRNG(p.seed)                  # reset -> geometry/A/B/c identical across knobs
        geo = generate_curve_geometry(p.d, p.s, rng)
        types, _ = generate_agent_types(p.N, geo, p.sigma_x, rng)
        env = generate_matching_env(
            p.d, r, dl, p.sigma_eps, types, rng; sigma_x=p.sigma_x, curve_geo=geo
        )
        F = build_ordered_output_matrix(types, env).F
        push!(acc[(r, dl)], rank90(svdvals(F .- mean(F))))
    end
    println("seed $seed done")
end

r90 = Dict((r, d) => mean(acc[(r, d)]) for r in RHOS, d in DELTAS)
println("=== r90 (mean over $(length(SEEDS)) seeds) ===")
for dl in DELTAS
    println(
        "  delta=$dl:  ",
        join(["rho=$r -> $(round(r90[(r, dl)]; digits=1))" for r in RHOS], "  "),
    )
end

mkpath(joinpath(@__DIR__, "_results"))
jldsave(
    joinpath(@__DIR__, "_results", "dgp_rank_grid.jld2");
    rhos=RHOS,
    deltas=DELTAS,
    r90=r90,
    seeds=collect(SEEDS),
)
println("DONE")
