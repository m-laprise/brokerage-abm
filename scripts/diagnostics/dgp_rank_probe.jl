"""
    dgp_rank_probe.jl

Probe whether rank and quality/complementarity direction can be varied independently
within the existing model, using initialization only. The rho knob bundles them; this
checks two candidate disentanglers at fixed direction (pure complementarity, rho=0):
  (a) delta (the sign-gated regime restructures the interaction term), and
  (b) rank-truncating the interaction matrix A (its SPD eigU-decomposition),
both holding the complementarity direction fixed. Reports the 90%-energy effective
rank of the centered noiseless output matrix.
"""

using TransientBrokerage
using TransientBrokerage: generate_matching_env, generate_curve_geometry, generate_agent_types
using LinearAlgebra: svdvals, eigen, Symmetric, tr, Diagonal
using Statistics: mean
using StableRNGs: StableRNG
include(joinpath(@__DIR__, "..", "exploration_common.jl"))

rank90(sv) = (cv = cumsum(sv .^ 2) ./ sum(sv .^ 2); findfirst(>=(0.90), cv))
r90_of(types, env) = rank90(svdvals(build_ordered_output_matrix(types, env).F .- mean(build_ordered_output_matrix(types, env).F)))

function trunc_A(A, r)
    E = eigen(Symmetric(A))
    idx = sortperm(E.values; rev=true)[1:r]
    Ar = E.vectors[:, idx] * Diagonal(E.values[idx]) * E.vectors[:, idx]'
    return Ar .* (size(A, 1) / tr(Ar))     # renormalize E[x'Ax]≈1, matching the model
end

mk(base, p; rho, A, delta) = TransientBrokerage.MatchingEnv(p.d, rho, base.c, A, base.B, delta, p.sigma_eps)

deltas = [0.0, 0.5, 0.75]
ranks = [2, 4, 6, 8]
rd = Dict(d => Float64[] for d in deltas)
rr = Dict(r => Float64[] for r in ranks)
rho1 = Float64[]

for seed in 1:3
    p = default_params(; seed=seed)
    rng = StableRNG(p.seed)
    geo = generate_curve_geometry(p.d, p.s, rng)
    types, _ = generate_agent_types(p.N, geo, p.sigma_x, rng)
    base = generate_matching_env(p.d, 0.5, 0.5, p.sigma_eps, types, rng; sigma_x=p.sigma_x, curve_geo=geo)
    # (a) delta at rho=0 (pure complementarity, full A): does the regime gain move rank?
    for dl in deltas
        push!(rd[dl], r90_of(types, mk(base, p; rho=0.0, A=base.A, delta=dl)))
    end
    # (b) rank-truncated A at rho=0, delta=0 (pure bilinear complementarity): clean rank control
    for r in ranks
        push!(rr[r], r90_of(types, mk(base, p; rho=0.0, A=trunc_A(base.A, r), delta=0.0)))
    end
    push!(rho1, r90_of(types, mk(base, p; rho=1.0, A=base.A, delta=0.5)))   # quality reference
end

println("=== effective rank r90 (mean over 3 seeds), direction held at pure complementarity (rho=0) ===")
println("(a) varying the sign-gain delta (interaction restructured, mix unchanged):")
for dl in deltas; println("    delta=$dl  r90 = $(round(mean(rd[dl]); digits=1))  $(Int.(rd[dl]))"); end
println("(b) rank-truncating A (delta=0, pure bilinear):")
for r in ranks; println("    A-rank=$r  r90 = $(round(mean(rr[r]); digits=1))  $(Int.(rr[r]))"); end
println("reference: rho=1 (pure quality, additive) r90 = $(round(mean(rho1); digits=1))")
