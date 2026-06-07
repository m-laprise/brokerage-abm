"""
    matching_function.jl

Gain-modulated matching function:
    f(x_i, x_j) = ρ · ½(x_i'c + x_j'c) + (1-ρ) · g(x_i,x_j) · x_i'Ax_j
    g(x_i, x_j) = 1 + δ · sign(x_i'Bx_j)

A is a symmetric positive definite (SPD) interaction matrix drawn at
initialization. B is a symmetric regime operator constructed to be weakly
aligned with A under the realized type distribution. All types are on the unit
sphere. Quality is a dot product with ideal type c. Interaction is a bilinear
form through A, modulated by a regime-dependent gain determined by B. The gain
creates two regimes (high-gain 1+δ, low-gain 1-δ) that produce a genuine
informational gap between single-agent and cross-agent data.

Observable output: q = Q + f(x_i, x_j) + ε, where Q is a constant offset and
ε ~ N(0, σ_ε²) is match noise.
"""

using LinearAlgebra: dot, mul!, norm, tr
using Random: AbstractRNG

"""
    type_second_moment(agent_types) -> Matrix{Float64}

Empirical second-moment matrix S = N^{-1} Σ_i x_i x_i' for the realized type
draws. Used to define the weighted overlap between the payoff geometry A and
the regime operator B under the realized type distribution.
"""
function type_second_moment(agent_types::Vector{Vector{Float64}})::Matrix{Float64}
    d = length(agent_types[1])
    S = zeros(d, d)
    n = length(agent_types)
    @inbounds for x in agent_types
        for j in 1:d, i in 1:d
            S[i, j] += x[i] * x[j]
        end
    end
    S ./= n
    return S
end

"""Weighted matrix inner product tr(S M S N)."""
weighted_matrix_inner(M::AbstractMatrix, N::AbstractMatrix, S::AbstractMatrix) = tr(S * M * S * N)

"""
    weighted_regime_overlap(A, B, agent_types) -> Float64

Normalized weighted overlap between payoff matrix A and regime operator B under
the empirical type second moment S. Returns 0 when the two are orthogonal in
the weighted metric.
"""
function weighted_regime_overlap(A::AbstractMatrix, B::AbstractMatrix,
                                 agent_types::Vector{Vector{Float64}})::Float64
    S = type_second_moment(agent_types)
    denom = sqrt(weighted_matrix_inner(A, A, S) * weighted_matrix_inner(B, B, S))
    denom <= 0.0 && return 0.0
    return weighted_matrix_inner(A, B, S) / denom
end

"""
    construct_regime_operator(A, agent_types, rng) -> Matrix{Float64}

Draw a symmetric Gaussian regime operator H, remove its weighted projection onto
A under the empirical second moment of realized types, then normalize the
result to unit Frobenius norm. The sign of x_i' B x_j determines the latent
regime, so only the orientation of B matters.
"""
function construct_regime_operator(A::Matrix{Float64},
                                   agent_types::Vector{Vector{Float64}},
                                   rng::AbstractRNG)::Matrix{Float64}
    d = size(A, 1)
    S = type_second_moment(agent_types)
    denom = weighted_matrix_inner(A, A, S)
    denom > 0.0 || error("Weighted overlap denominator must be positive")

    for _ in 1:16
        G = randn(rng, d, d)
        H = 0.5 .* (G .+ G')
        shift = tr(H) / d
        @inbounds for k in 1:d
            H[k, k] -= shift
        end

        α = weighted_matrix_inner(H, A, S) / denom
        B = H .- α .* A
        B = 0.5 .* (B .+ B')
        nrm = norm(B)
        nrm <= sqrt(eps(Float64)) && continue
        B ./= nrm
        return B
    end

    error("Could not construct a nondegenerate regime operator after repeated draws")
end

"""
    generate_matching_env(d, rho, delta, sigma_eps, agent_types, rng; sigma_x, curve_geo) -> MatchingEnv

Build the matching environment:
- Ideal type `c` drawn as a perturbation of a fresh random curve position from
  the supplied curve geometry
- A = M_A'M_A (SPD interaction matrix)
- B = symmetric regime operator, orthogonalized against A under the empirical
  type second moment
"""
function generate_matching_env(d::Int, rho::Float64, delta::Float64, sigma_eps::Float64,
                                agent_types::Vector{Vector{Float64}},
                                rng::AbstractRNG;
                                sigma_x::Float64 = 0.5,
                                curve_geo::CurveGeometry)::MatchingEnv
    sigma_per_dim = sigma_x / sqrt(d)

    # Ideal type c: perturbation of a fresh random curve position per spec.
    @assert curve_geo.d == d "curve_geo.d must equal d"
    ref = curve_point(rand(rng), curve_geo)
    c = ref .+ sigma_per_dim .* randn(rng, d)

    # SPD interaction matrix: A = M_A'M_A, normalized so E[x'Ax] ≈ 1 for unit vectors
    # For unit vectors, E[x'Ax] = trace(A)/d. Dividing by trace(A)/d normalizes to unit scale.
    M_A = randn(rng, d, d)
    A_raw = M_A' * M_A
    A = A_raw .* (d / tr(A_raw))

    # Symmetric regime operator: draw H, then remove its weighted projection
    # onto A under the realized type distribution. This keeps the regime
    # boundary weakly aligned with the payoff geometry by construction.
    B = construct_regime_operator(A, agent_types, rng)

    return MatchingEnv(d, rho, c, A, B, delta, sigma_eps)
end

# ─────────────────────────────────────────────────────────────────────────────
# Regime gain
# ─────────────────────────────────────────────────────────────────────────────

"""
    regime_gain(xi, xj, env) -> Float64

Compute the regime-dependent gain g(x_i, x_j) = 1 + δ · sign(x_i'Bx_j).
Returns (1 + δ) for high-gain regime, (1 - δ) for low-gain regime.
"""
function regime_gain(xi::AbstractVector, xj::AbstractVector, env::MatchingEnv)::Float64
    bxj = dot(xi, env.B * xj)
    return 1.0 + env.delta * sign(bxj)
end

"""In-place `regime_gain` using pre-allocated buffer for Bx_j."""
function regime_gain!(Bx_buf::Vector{Float64}, xi::AbstractVector, xj::AbstractVector, env::MatchingEnv)::Float64
    mul!(Bx_buf, env.B, xj)
    bxj = dot(xi, Bx_buf)
    return 1.0 + env.delta * sign(bxj)
end

# ─────────────────────────────────────────────────────────────────────────────
# Match signal and output
# ─────────────────────────────────────────────────────────────────────────────

"""
    match_signal(xi, xj, env) -> Float64

Deterministic matching function:
    f(x_i, x_j) = ρ · ½(x_i'c + x_j'c) + (1-ρ) · g(x_i,x_j) · x_i'Ax_j

Does not include the offset Q or noise ε. Used for holdout evaluation and diagnostics.
"""
function match_signal(xi::AbstractVector, xj::AbstractVector, env::MatchingEnv)::Float64
    quality = env.rho * 0.5 * (dot(xi, env.c) + dot(xj, env.c))
    base_interaction = dot(xi, env.A * xj)
    g = regime_gain(xi, xj, env)
    interaction = (1.0 - env.rho) * g * base_interaction
    return quality + interaction
end

"""In-place `match_signal` using pre-allocated buffers for Ax_j and Bx_j."""
function match_signal!(Ax_buf::Vector{Float64}, Bx_buf::Vector{Float64},
                       xi::AbstractVector, xj::AbstractVector, env::MatchingEnv)::Float64
    quality = env.rho * 0.5 * (dot(xi, env.c) + dot(xj, env.c))
    mul!(Ax_buf, env.A, xj)
    base_interaction = dot(xi, Ax_buf)
    g = regime_gain!(Bx_buf, xi, xj, env)
    interaction = (1.0 - env.rho) * g * base_interaction
    return quality + interaction
end

"""
    match_output(xi, xj, env, rng) -> Float64

Stochastic observable output: q = Q + f(x_i, x_j) + ε, where ε ~ N(0, σ_ε²).
"""
function match_output(xi::AbstractVector, xj::AbstractVector,
                      env::MatchingEnv, rng::AbstractRNG)::Float64
    return Q_OFFSET + match_signal(xi, xj, env) + env.sigma_eps * randn(rng)
end

"""In-place `match_output` using pre-allocated buffers."""
function match_output!(Ax_buf::Vector{Float64}, Bx_buf::Vector{Float64},
                       xi::AbstractVector, xj::AbstractVector,
                       env::MatchingEnv, rng::AbstractRNG)::Float64
    return Q_OFFSET + match_signal!(Ax_buf, Bx_buf, xi, xj, env) + env.sigma_eps * randn(rng)
end

# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

"""
    calibrate(env, agent_types, params, rng; n_samples=10_000) -> CalibrationConstants

Monte Carlo calibration from random agent pairs. Returns q_cal (mean output),
r, phi, c_s, and mad_f (mean absolute deviation of the signal f, the
forecast-error scale used by the capture-readiness gate).
"""
function calibrate(env::MatchingEnv,
                   agent_types::Vector{Vector{Float64}},
                   params::ModelParams,
                   rng::AbstractRNG;
                   n_samples::Int = 10_000)::CalibrationConstants
    n_agents = length(agent_types)
    d = env.d
    Ax_buf = Vector{Float64}(undef, d)
    Bx_buf = Vector{Float64}(undef, d)
    samples = Vector{Float64}(undef, n_samples)
    for k in 1:n_samples
        i = rand(rng, 1:n_agents)
        j = rand(rng, 1:n_agents)
        samples[k] = Q_OFFSET + match_signal!(Ax_buf, Bx_buf, agent_types[i], agent_types[j], env)
    end
    q_cal = sum(samples) / n_samples
    # MAD of the signal: the constant Q_OFFSET drops out, so this is MAD(f).
    mad_acc = 0.0
    @inbounds for k in 1:n_samples
        mad_acc += abs(samples[k] - q_cal)
    end
    mad_f = mad_acc / n_samples
    # r, phi, c_s are independent fractions of q_cal. Decoupling phi/c_s from the
    # reservation lets r be swept freely (including r >= q_cal) without the
    # frictions vanishing or turning negative.
    r = params.reservation_frac * q_cal
    phi = params.search_cost_rate * q_cal
    c_s = params.search_cost_rate * q_cal
    return CalibrationConstants(q_cal, r, phi, c_s, mad_f)
end
