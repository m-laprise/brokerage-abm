"""
    scripts/calibrate_match_component_moments.jl

Estimate the fixed population moments used to put general quality and
complementarity on common scales. The calibration averages within-market
moments across 50 independently generated markets of 1,000 principals. It uses
seeds 10,001 through 10,050, which are separate from the simulation sweep.

Usage: julia --project --threads=auto scripts/calibrate_match_component_moments.jl
"""

using BrokerageABM
using BrokerageABM: generate_matching_dgp
using LinearAlgebra: dot, mul!
using StableRNGs: StableRNG
using Statistics: mean

const CALIBRATION_N = 1000
const CALIBRATION_SEEDS = 10_001:10_050
function within_market_moments(types, env)
    n = length(types)
    n_pairs = n * (n - 1) ÷ 2
    quality_scores = [dot(x, env.c) for x in types]
    Ax = Vector{Float64}(undef, env.d)
    Bx = similar(Ax)
    sums = zeros(9)

    @inbounds for j in 2:n
        xj = types[j]
        mul!(Ax, env.A, xj)
        mul!(Bx, env.B, xj)
        for i in 1:(j - 1)
            x = 0.5 * (quality_scores[i] + quality_scores[j])
            v = dot(types[i], Ax)
            h = sign(dot(types[i], Bx)) * v
            sums[1] += x
            sums[2] += v
            sums[3] += h
            sums[4] += x^2
            sums[5] += v^2
            sums[6] += h^2
            sums[7] += x * v
            sums[8] += x * h
            sums[9] += v * h
        end
    end

    means = sums[1:3] ./ n_pairs
    variances = sums[4:6] ./ n_pairs .- means .^ 2
    covariances = sums[7:9] ./ n_pairs .-
                  (means[1] * means[2], means[1] * means[3], means[2] * means[3])
    return (; means, variances, covariances)
end

function main()
    rows = map(CALIBRATION_SEEDS) do seed
        params = default_params(; N=CALIBRATION_N, seed)
        dgp = generate_matching_dgp(params, StableRNG(seed))
        within_market_moments(dgp.agent_types, dgp.env)
    end

    component_means = mean(row.means for row in rows)
    component_variances = mean(row.variances for row in rows)
    component_covariances = mean(row.covariances for row in rows)
    println("Calibration N = $CALIBRATION_N")
    println("Calibration seeds = $(first(CALIBRATION_SEEDS)):$(last(CALIBRATION_SEEDS))")
    for (name, value) in zip(("mean_u", "mean_v", "mean_h"), component_means)
        println(name, " = ", repr(value))
    end
    for (name, value) in zip(("var_u", "var_v", "var_h"), component_variances)
        println(name, " = ", repr(value))
    end
    for (name, value) in zip(("cov_uv", "cov_uh", "cov_vh"), component_covariances)
        println(name, " = ", repr(value))
    end
    return nothing
end

main()
