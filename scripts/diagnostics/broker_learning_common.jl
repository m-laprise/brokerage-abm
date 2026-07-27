"""
    broker_learning_common.jl

Shared harness for the staged broker-learning diagnostics. The goal across the
stages is to separate three questions about whether the broker can learn the
*interaction* term of the matching function:

    f(x_i,x_j) = ρ·½(x_i'c + x_j'c)              ← quality   (linear, additive)
               + (1-ρ)·x_i'A x_j                  ← core      (symmetric bilinear)
               + (1-ρ)·δ·sign(x_i'B x_j)·x_i'A x_j ← gain      (regime-gated, discontinuous)

The broker's symmetric feature map spans `quality` and `core` exactly (a linear
readout recovers them). The *gain* term is a product of `sign(quadratic)` and a
`quadratic`; it is discontinuous across the regime boundary `x_i'B x_j = 0` and
is NOT in the linear span of the features. It is also the only term where the
broker's cross-agent data should give it an edge over single-agent learners.

So the headline question is operationalized as: **how much of the `gain`
component does a fitted predictor reproduce?** We measure this with a component
regression (regress predictions on the true components; the coefficient on
`gain` is ~1 if recovered, ~0 if ignored) plus per-regime bias and overall R².

This file only defines reusable pieces. Each stage script includes it.
"""

using BrokerageABM
const ABM = BrokerageABM

using BrokerageABM:
    generate_curve_geometry,
    generate_agent_types,
    generate_matching_env,
    fill_broker_pair_features!,
    broker_pair_feature_dim,
    broker_hidden_width,
    NeuralNet,
    NNGradBuffers,
    init_neural_net,
    predict_nn!,
    train_nn!,
    Q_OFFSET,
    MatchingEnv,
    CurveGeometry

using LinearAlgebra
using Random
using StableRNGs: StableRNG
using Statistics: mean, var, std, cor, quantile
using Printf: @printf, @sprintf
using DifferentiationInterface: gradient!, Constant

# ─────────────────────────────────────────────────────────────────────────────
# Environment + type pool construction (standalone, no full ABM)
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_env(; d, s, rho, delta, sigma_eps, sigma_x, seed, n_pool) -> (env, pool, geo)

Build a matching environment and a pool of `n_pool` agent types drawn from the
same sinusoidal-curve distribution the live model uses. Sampling pairs from
`pool` reproduces the support the broker would actually encounter; for full
sphere coverage, draw uniform unit vectors instead (see `sample_pairs`).
"""
function make_env(;
    d::Int=8,
    s::Int=8,
    rho::Float64=0.5,
    delta::Float64=0.5,
    sigma_eps::Float64=0.10,
    sigma_x::Float64=0.5,
    seed::Int=42,
    n_pool::Int=4000,
)
    rng = StableRNG(seed)
    geo = generate_curve_geometry(d, s, rng)
    pool, _ = generate_agent_types(n_pool, geo, sigma_x, rng)
    env = generate_matching_env(
        d, rho, delta, sigma_eps, pool, rng; sigma_x=sigma_x, curve_geo=geo
    )
    return env, pool, geo
end

"""Draw a uniform random unit vector in R^d."""
function rand_sphere(rng::AbstractRNG, d::Int)::Vector{Float64}
    v = randn(rng, d)
    return v ./ norm(v)
end

"""
    sample_pairs(env, pool, n, rng; source) -> (Xi, Xj)

Sample `n` ordered type pairs as two `d×n` matrices. `source`:
- `:pool`    — both endpoints drawn from the realized type pool (model support)
- `:sphere`  — both endpoints uniform on the unit sphere (full coverage)
"""
function sample_pairs(
    env::MatchingEnv, pool::Vector{Vector{Float64}}, n::Int, rng::AbstractRNG; source::Symbol=:pool
)
    d = env.d
    Xi = Matrix{Float64}(undef, d, n)
    Xj = Matrix{Float64}(undef, d, n)
    np = length(pool)
    for k in 1:n
        if source === :pool
            xi = pool[rand(rng, 1:np)]
            xj = pool[rand(rng, 1:np)]
        elseif source === :sphere
            xi = rand_sphere(rng, d)
            xj = rand_sphere(rng, d)
        else
            error("unknown source $source")
        end
        @views Xi[:, k] .= xi
        @views Xj[:, k] .= xj
    end
    return Xi, Xj
end

# ─────────────────────────────────────────────────────────────────────────────
# Component decomposition of the matching signal
# ─────────────────────────────────────────────────────────────────────────────

"""
    decompose(env, Xi, Xj) -> NamedTuple of length-n vectors

Decompose the noiseless target `Q_OFFSET + f` into its additive components:
`quality`, `core` (δ=0 bilinear interaction), `gain` (regime-gated modulation).
`target = Q_OFFSET + quality + core + gain`. Also returns `bxj = x_i'B x_j`
(the latent regime score) and the regime label `sign(bxj)`.
"""
function decompose(env::MatchingEnv, Xi::Matrix{Float64}, Xj::Matrix{Float64})
    n = size(Xi, 2)
    quality = Vector{Float64}(undef, n)
    core = Vector{Float64}(undef, n)
    gain = Vector{Float64}(undef, n)
    bxj = Vector{Float64}(undef, n)
    regime = Vector{Int}(undef, n)
    Ax = Vector{Float64}(undef, env.d)
    Bx = Vector{Float64}(undef, env.d)
    for k in 1:n
        xi = @view Xi[:, k]
        xj = @view Xj[:, k]
        mul!(Ax, env.A, xj)
        mul!(Bx, env.B, xj)
        base = dot(xi, Ax)
        b = dot(xi, Bx)
        s = sign(b)
        quality[k] = env.rho * 0.5 * (dot(xi, env.c) + dot(xj, env.c))
        core[k] = (1 - env.rho) * base
        gain[k] = (1 - env.rho) * env.delta * s * base
        bxj[k] = b
        regime[k] = s >= 0 ? 1 : -1
    end
    target = Q_OFFSET .+ quality .+ core .+ gain
    return (; quality, core, gain, target, bxj, regime)
end

# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────

"""Build the `d_broker × n` symmetric broker feature matrix for pairs (Xi, Xj)."""
function broker_features(Xi::Matrix{Float64}, Xj::Matrix{Float64})
    d = size(Xi, 1)
    n = size(Xi, 2)
    db = broker_pair_feature_dim(d)
    Z = Matrix{Float64}(undef, db, n)
    for k in 1:n
        fill_broker_pair_features!(Z, k, view(Xi, :, k), view(Xj, :, k))
    end
    return Z
end

# ─────────────────────────────────────────────────────────────────────────────
# Metrics: the headline is gain recovery via component regression
# ─────────────────────────────────────────────────────────────────────────────

r2(pred, truth) = 1 - sum(abs2, pred .- truth) / sum(abs2, truth .- mean(truth))

"""
    component_regression(pred, comps) -> (; b0, bq, bc, bg, resid_frac_gain)

Regress predictions on `[1, quality, core, gain]`. A predictor that perfectly
reproduces a component has coefficient ~1 on it. `bg` (the gain coefficient) is
the headline: ~1 = gain recovered, ~0 = gain ignored. `resid_frac_gain` is the
fraction of the prediction-residual variance (pred - target) explained by gain,
i.e. how much of the model's error is structured gain it failed to track.
"""
function component_regression(pred::Vector{Float64}, comps)
    n = length(pred)
    D = hcat(ones(n), comps.quality, comps.core, comps.gain)
    β = D \ pred
    resid = pred .- comps.target
    # how much of resid is explained by the gain direction
    g = comps.gain .- mean(comps.gain)
    rr = resid .- mean(resid)
    denom = sum(abs2, rr)
    frac = denom > 0 ? (dot(g, rr)^2 / (sum(abs2, g) * denom)) : 0.0
    return (; b0=β[1], bq=β[2], bc=β[3], bg=β[4], resid_frac_gain=frac)
end

"""Per-regime mean signed residual (bias) and R² of pred vs target."""
function regime_metrics(pred::Vector{Float64}, comps)
    out = Dict{Int,NamedTuple}()
    for s in (1, -1)
        idx = findall(==(s), comps.regime)
        isempty(idx) && continue
        p = pred[idx]
        t = comps.target[idx]
        out[s] = (; n=length(idx), bias=mean(p .- t), r2=r2(p, t))
    end
    return out
end

"""Fraction of target variance carried by each component (and its cov share)."""
function variance_shares(comps)
    vt = var(comps.target)
    return (;
        var_target=vt,
        frac_quality=var(comps.quality) / vt,
        frac_core=var(comps.core) / vt,
        frac_gain=var(comps.gain) / vt,
    )
end

"""
    evaluate(pred, comps) -> NamedTuple

Bundle the full metric set for a prediction vector against the decomposed
target on a holdout sample.
"""
function evaluate(pred::Vector{Float64}, comps)
    cr = component_regression(pred, comps)
    rm = regime_metrics(pred, comps)
    return (;
        r2=r2(pred, comps.target),
        bg=cr.bg,
        bc=cr.bc,
        bq=cr.bq,
        resid_frac_gain=cr.resid_frac_gain,
        bias_hi=get(rm, 1, (; bias=NaN)).bias,
        bias_lo=get(rm, -1, (; bias=NaN)).bias,
        r2_hi=get(rm, 1, (; r2=NaN)).r2,
        r2_lo=get(rm, -1, (; r2=NaN)).r2,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Predictors
# ─────────────────────────────────────────────────────────────────────────────

"""Closed-form ridge/least-squares linear readout on broker features Z (db×n)."""
function fit_linear_readout(Z::Matrix{Float64}, y::Vector{Float64}; ridge::Float64=1e-8)
    db = size(Z, 1)
    A = Z * Z' + ridge * I
    b = Z * y
    w = A \ b
    # intercept: fit on residual mean
    μ = mean(y) - mean(Z' * w)
    return w, μ
end

predict_linear(Z, w, μ) = vec(Z' * w) .+ μ

"""Run the broker NN forward over feature columns of Z, returning a length-n vector."""
function predict_nn_cols(nn::NeuralNet, Z::Matrix{Float64})
    n = size(Z, 2)
    h = size(nn.W1, 1)
    buf = zeros(h)
    out = Vector{Float64}(undef, n)
    for k in 1:n
        out[k] = predict_nn!(nn, buf, view(Z, :, k))
    end
    return out
end

"""Pretty-print one labeled metric row."""
function print_row(label, m)
    @printf(
        "%-28s  R²=%6.3f  βg=%6.3f  βc=%5.2f  βq=%5.2f  bias(hi/lo)=%+5.3f/%+5.3f  R²(hi/lo)=%5.3f/%5.3f\n",
        label, m.r2, m.bg, m.bc, m.bq, m.bias_hi, m.bias_lo, m.r2_hi, m.r2_lo,
    )
    flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────────────
# Strong optimizer (Adam) — representational ceiling probe
#
# Stage 1's core risk is conflating REPRESENTATION ("can the function class
# express the gain term?") with OPTIMIZATION ("can vanilla GD find it?"). Vanilla
# full-batch GD at a fixed lr is a weak optimizer for sharpening the steep ramp a
# ReLU net needs to approximate the discontinuous gain term, so a low βg under
# the model's own GD does NOT prove a representational gap.
#
# `train_nn_adam!` reuses the model's EXACT Enzyme gradient of the EXACT loss
# (`nn_loss_theta`) and the same parameter packing, swapping only the update rule
# for Adam. So it is an apples-to-apples ceiling for the *architecture + feature
# map*: if Adam (long) drives βg→1, the representation is fine and the live
# shortfall is optimizer/schedule (Stage 2) or data (Stage 3); if Adam also
# plateaus well below 1, the gap is representational.
# ─────────────────────────────────────────────────────────────────────────────

"""
    train_nn_adam!(nn, grad, X, q, n_steps, lr; β1, β2, ϵ) -> Nothing

Full-batch Adam on the same packed-parameter loss the model trains with. The raw
gradient is computed via the model's `gradient!`/Enzyme path into `grad.dtheta`;
we overwrite `grad.dtheta` in place with the Adam step and reuse the model's
`apply_nn_gradient!` to write it back, so the only difference from the live
`train_step_prefix!` is the optimizer.
"""
function train_nn_adam!(
    nn::NeuralNet,
    grad::NNGradBuffers,
    X::Matrix{Float64},
    q::Vector{Float64},
    n_steps::Int,
    lr::Float64;
    β1::Float64=0.9,
    β2::Float64=0.999,
    ϵ::Float64=1e-8,
)
    h = size(nn.W1, 1)
    d_in = size(nn.W1, 2)
    ABM.ensure_nn_param_buffers!(grad, h, d_in)
    n = length(q)
    np = length(grad.dtheta)
    m = zeros(np)
    v = zeros(np)
    for t in 1:n_steps
        ABM.pack_nn_params!(grad.theta, nn)
        gradient!(
            ABM.nn_loss_theta,
            grad.dtheta,
            ABM.NN_AD_BACKEND,
            grad.theta,
            Constant(X),
            Constant(q),
            Constant(n),
            Constant(h),
            Constant(d_in),
        )
        bc1 = 1 - β1^t
        bc2 = 1 - β2^t
        @inbounds for i in 1:np
            g = grad.dtheta[i]
            mi = β1 * m[i] + (1 - β1) * g
            vi = β2 * v[i] + (1 - β2) * g * g
            m[i] = mi
            v[i] = vi
            grad.dtheta[i] = (mi / bc1) / (sqrt(vi / bc2) + ϵ)
        end
        ABM.apply_nn_gradient!(nn, grad, lr)
    end
    return nothing
end

"""Fit a broker NN on features `Z`→`y` with Adam (strong-optimizer ceiling)."""
function fit_broker_adam(
    Z, y, h::Int; steps::Int=4000, lr::Float64=0.01, seed::Int=1, b2_init::Float64=Q_OFFSET
)
    db = size(Z, 1)
    nn = init_neural_net(db, h, StableRNG(seed); b2_init=b2_init)
    grad = NNGradBuffers(nn)
    train_nn_adam!(nn, grad, Matrix(Z), Vector(y), steps, lr)
    return nn
end

# ─────────────────────────────────────────────────────────────────────────────
# Decisive representation diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""
    residual_by_boundary(pred, comps; nbins) -> Vector{NamedTuple}

Bin holdout points by distance to the regime boundary |x_iʹB x_j| and report mean
|residual| per bin. The discontinuity in the gain term lives at x_iʹB x_j = 0, so
if the predictor's error is *concentrated in the smallest-|bxj| bin* and falls
away as |bxj| grows, the error is the regime jump itself (a representational
signature). If |residual| is roughly flat across bins, the shortfall is global
underfitting (optimization), not the discontinuity.
"""
function residual_by_boundary(pred::Vector{Float64}, comps; nbins::Int=6)
    a = abs.(comps.bxj)
    resid = abs.(pred .- comps.target)
    edges = quantile(a, range(0.0, 1.0; length=nbins + 1))
    out = NamedTuple[]
    for b in 1:nbins
        lo = edges[b]
        hi = edges[b + 1]
        idx = if b < nbins
            findall(x -> lo <= x < hi, a)
        else
            findall(x -> lo <= x <= hi, a)
        end
        isempty(idx) && continue
        push!(out, (; bin=b, lo, hi, n=length(idx), mae=mean(resid[idx]),
                     mae_gain=mean(abs.(comps.gain[idx]))))
    end
    return out
end

"""Print a residual-by-boundary table (closest-to-boundary bin first)."""
function print_boundary_table(label, rows)
    println("  $label — mean|residual| by distance to regime boundary |x'Bx|:")
    @printf("    %-8s %10s %10s %8s %12s %12s\n", "bin", "lo|bxj|", "hi|bxj|", "n", "mae", "|gain|")
    for r in rows
        @printf("    %-8d %10.4f %10.4f %8d %12.4f %12.4f\n", r.bin, r.lo, r.hi, r.n, r.mae, r.mae_gain)
    end
    flush(stdout)
end

"""R² of a prediction against the gain component alone (after removing the part
of the prediction explained by [1, quality, core] via least squares). Measures
pure recovery of the discontinuous gain signal, immune to quality/core scale."""
function gain_only_r2(pred::Vector{Float64}, comps)
    n = length(pred)
    D = hcat(ones(n), comps.quality, comps.core)
    resid_pred = pred .- D * (D \ pred)        # part of pred orthogonal to {1,quality,core}
    g = comps.gain
    resid_g = g .- D * (D \ g)                 # part of gain orthogonal to {1,quality,core}
    denom_g = sum(abs2, resid_g)
    denom_g <= 0 && return (; beta_gain=NaN, frac_pred_var_on_gain=NaN)
    # regress resid_pred on resid_g, report R² of that fit (how much of the
    # orthogonalized gain direction the prediction reproduces)
    β = dot(resid_g, resid_pred) / denom_g
    ss_res = sum(abs2, resid_pred .- β .* resid_g)
    ss_pred = sum(abs2, resid_pred .- mean(resid_pred))
    return (; beta_gain=β, frac_pred_var_on_gain=ss_pred > 0 ? 1 - ss_res / ss_pred : NaN)
end

"""Sample-level collinearity between core and gain components (validates that the
component-regression βg is interpretable; |cor|≈0 ⇒ clean separation)."""
core_gain_cor(comps) = cor(comps.core, comps.gain)
