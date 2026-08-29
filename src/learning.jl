"""
    learning.jl

Prediction models for agents and broker. The default model trains one-hidden-layer
ReLU networks with full-batch Adam. The alternate model fits Ridge regressions on
the same histories, windows, and observation caps. Vanilla-gradient helpers are
retained for gradient tests. Gradients are computed through
DifferentiationInterface with Enzyme.

Agent: input x_j (d features), with width derived from d.
Broker: symmetric additive and bilinear pair features, with width derived from d.
"""

using LinearAlgebra: mul!, dot, BLAS, Symmetric, cholesky!, ldiv!
using Random: AbstractRNG
using DifferentiationInterface: AutoEnzyme, Constant, gradient!
using Enzyme: Enzyme

const NN_AD_BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

# Adam optimizer hyperparameters (standard defaults).
const ADAM_BETA1 = 0.9
const ADAM_BETA2 = 0.999
const ADAM_EPS = 1e-8

"""Number of symmetric broker pair features for d-dimensional types."""
@inline broker_pair_feature_dim(d::Int)::Int = d + (d * (d + 1)) ÷ 2

"""
    fill_broker_pair_features!(z, xi, xj) -> Nothing

Fill `z` with symmetric broker pair features:
`xi + xj` followed by the lower-triangular half-vectorization of
`(xi*xj' + xj*xi') / 2`.
"""
function fill_broker_pair_features!(
    z::AbstractVector{Float64}, xi::AbstractVector{Float64}, xj::AbstractVector{Float64}
)
    d = length(xi)
    @assert length(xj) == d
    @assert length(z) >= broker_pair_feature_dim(d)

    @inbounds for k in 1:d
        z[k] = xi[k] + xj[k]
    end

    pos = d + 1
    @inbounds for col in 1:d
        for row in col:d
            z[pos] = if row == col
                xi[row] * xj[col]
            else
                0.5 * (xi[row] * xj[col] + xj[row] * xi[col])
            end
            pos += 1
        end
    end
    return nothing
end

"""Matrix-column variant of `fill_broker_pair_features!` for batched scoring."""
function fill_broker_pair_features!(
    Z::AbstractMatrix{Float64},
    colidx::Int,
    xi::AbstractVector{Float64},
    xj::AbstractVector{Float64},
)
    d = length(xi)
    @assert length(xj) == d
    @assert size(Z, 1) >= broker_pair_feature_dim(d)

    @inbounds for k in 1:d
        Z[k, colidx] = xi[k] + xj[k]
    end

    pos = d + 1
    @inbounds for col in 1:d
        for row in col:d
            Z[pos, colidx] = if row == col
                xi[row] * xj[col]
            else
                0.5 * (xi[row] * xj[col] + xj[row] * xi[col])
            end
            pos += 1
        end
    end
    return nothing
end

"""Matrix-column source variant for broker history buffers."""
function fill_broker_pair_features!(
    Z::AbstractMatrix{Float64},
    colidx::Int,
    Xi::AbstractMatrix{Float64},
    xi_col::Int,
    Xj::AbstractMatrix{Float64},
    xj_col::Int,
)
    d = size(Xi, 1)
    @assert size(Xj, 1) == d
    @assert size(Z, 1) >= broker_pair_feature_dim(d)

    @inbounds for k in 1:d
        Z[k, colidx] = Xi[k, xi_col] + Xj[k, xj_col]
    end

    pos = d + 1
    @inbounds for col in 1:d
        for row in col:d
            Z[pos, colidx] = if row == col
                Xi[row, xi_col] * Xj[col, xj_col]
            else
                0.5 * (Xi[row, xi_col] * Xj[col, xj_col] + Xj[row, xj_col] * Xi[col, xi_col])
            end
            pos += 1
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    init_neural_net(d_in, h, rng; b2_init=Q_OFFSET) -> NeuralNet

Initialize a one-hidden-layer ReLU network with He-initialized hidden weights,
zero hidden bias, zero output weights, and output bias `b2_init` (default
`Q_OFFSET`). The untrained network therefore returns exactly `b2_init` for every
input. This gives fresh entrants a neutral constant prior until they acquire
training data. Subsequent training updates every parameter.
"""
function init_neural_net(
    d_in::Int, h::Int, rng::AbstractRNG; b2_init::Float64=Q_OFFSET
)::NeuralNet
    # He initialization: scale = sqrt(2 / fan_in)
    scale_1 = sqrt(2.0 / d_in)
    W1 = scale_1 .* randn(rng, h, d_in)
    b1 = zeros(h)
    w2 = zeros(h)
    b2 = b2_init
    return NeuralNet(W1, b1, w2, b2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Prediction (zero-allocation hot path)
# ─────────────────────────────────────────────────────────────────────────────

"""
    predict_nn!(nn, hidden_buf, z) -> Float64

Zero-allocation forward pass: y = w2' * relu(W1 * z + b1) + b2.
`hidden_buf` is a pre-allocated vector of length h.
"""
function predict_nn!(
    nn::NeuralNet, hidden_buf::Vector{Float64}, z::AbstractVector{Float64}
)::Float64
    mul!(hidden_buf, nn.W1, z)
    hidden_buf .+= nn.b1
    # ReLU in place
    @inbounds for i in eachindex(hidden_buf)
        hidden_buf[i] = max(hidden_buf[i], 0.0)
    end
    return dot(nn.w2, hidden_buf) + nn.b2
end

"""
    predict_nn_batch!(nn, H_buf, Y_out, Z_buf, n)

Batched forward pass for `n` input columns using BLAS gemm/gemv.
Buffers must be pre-allocated: Z_buf (d_in x cap), H_buf (h x cap), Y_out (cap).
"""
function predict_nn_batch!(
    nn::NeuralNet,
    H_buf::Matrix{Float64},
    Y_out::Vector{Float64},
    Z_buf::Matrix{Float64},
    n::Int,
)
    h = size(nn.W1, 1)
    b1 = nn.b1;
    w2 = nn.w2;
    b2 = nn.b2

    # H[:,1:n] = W1 * Z[:,1:n]  — use gemm on contiguous column block
    # BLAS gemm: C = alpha*A*B + beta*C.  A is h x d_in, B is d_in x n.
    # We call gemm! directly to avoid SubArray overhead from views.
    BLAS.gemm!('N', 'N', 1.0, nn.W1, view(Z_buf, :, 1:n), 0.0, view(H_buf, :, 1:n))

    # H += b1 (broadcast), then ReLU in place
    @inbounds for j in 1:n, i in 1:h
        v = H_buf[i, j] + b1[i]
        H_buf[i, j] = v > 0.0 ? v : 0.0
    end

    # Y[1:n] = H[:,1:n]' * w2 + b2
    # gemv: y = alpha * A' * x + beta * y
    BLAS.gemv!('T', 1.0, view(H_buf, :, 1:n), w2, 0.0, view(Y_out, 1:n))
    @inbounds for j in 1:n
        Y_out[j] += b2
    end

    return nothing
end

"""Evaluate a fitted Ridge model without allocating."""
@inline function predict_ridge(model::RidgeModel, z::AbstractVector{Float64})::Float64
    return model.intercept + dot(model.coefficients, z)
end

@inline function predict_ridge_column(
    model::RidgeModel, Z::AbstractMatrix{Float64}, col::Int
)::Float64
    score = model.intercept
    @inbounds for row in eachindex(model.coefficients)
        score += model.coefficients[row] * Z[row, col]
    end
    return score
end

@inline function predict_additive_ridge(
    model::RidgeModel,
    party1_type::AbstractVector{Float64},
    party2_type::AbstractVector{Float64},
    subtract_target_mean::Bool,
)::Float64
    score =
        subtract_target_mean ? 2.0 * model.intercept - model.target_mean : model.intercept
    @inbounds for k in eachindex(model.coefficients)
        score += model.coefficients[k] * (party1_type[k] + party2_type[k])
    end
    return score
end

"""Evaluate an agent's selected prediction model."""
@inline function predict_agent(
    agent::Agent, partner_type::AbstractVector{Float64}, params::ModelParams
)::Float64
    if params.learning_model == :nn
        return predict_nn!(agent.nn, agent.predict_buf, partner_type)
    end
    return predict_ridge(agent.ridge::RidgeModel, partner_type)
end

@inline predict_agent(agent::Agent, partner_type::AbstractVector{Float64}, ::Nothing) = predict_nn!(
    agent.nn, agent.predict_buf, partner_type
)

"""Evaluate the broker's selected prediction model for an unordered pair."""
function predict_broker!(
    broker::Broker,
    feature_buf::AbstractVector{Float64},
    party1_type::AbstractVector{Float64},
    party2_type::AbstractVector{Float64},
    params::ModelParams,
)::Float64
    if params.learning_model == :nn
        fill_broker_pair_features!(feature_buf, party1_type, party2_type)
        return predict_nn!(broker.nn, broker.predict_buf, feature_buf)
    end

    ridge = broker.ridge::RidgeModel
    variant = params.ridge_broker_variant
    if variant in (:pair, :size_matched)
        fill_broker_pair_features!(feature_buf, party1_type, party2_type)
        return predict_ridge(ridge, feature_buf)
    elseif variant == :additive
        return predict_additive_ridge(ridge, party1_type, party2_type, false)
    end

    # The single-principal model g_b is fit to one randomly retained endpoint
    # per broker observation. Pair scores combine both endpoint evaluations.
    return predict_additive_ridge(ridge, party1_type, party2_type, true)
end

"""
    fit_ridge!(model, X, y, n, lambda)

Fit slopes to mean squared error plus `lambda * sum(abs2, slopes)`. The
intercept is unpenalized. Inputs are used on their raw DGP scale.
"""
function fit_ridge!(
    model::RidgeModel, X::Matrix{Float64}, y::Vector{Float64}, n::Int, lambda::Float64
)
    n > 0 || return nothing
    p = length(model.coefficients)
    @assert size(X, 1) == p
    @assert n <= size(X, 2) && n <= length(y)

    xbar = model.feature_mean
    fill!(xbar, 0.0)
    ybar = 0.0
    @inbounds for obs in 1:n
        ybar += y[obs]
        for feature in 1:p
            xbar[feature] += X[feature, obs]
        end
    end
    inv_n = 1.0 / n
    ybar *= inv_n
    @inbounds for feature in 1:p
        xbar[feature] *= inv_n
    end

    gram = model.gram
    rhs = model.rhs
    fill!(gram, 0.0)
    fill!(rhs, 0.0)
    @inbounds for obs in 1:n
        yc = y[obs] - ybar
        for col in 1:p
            xc = X[col, obs] - xbar[col]
            rhs[col] += xc * yc
            for row in col:p
                gram[row, col] += (X[row, obs] - xbar[row]) * xc
            end
        end
    end
    @inbounds for col in 1:p
        rhs[col] *= inv_n
        for row in col:p
            value = gram[row, col] * inv_n
            gram[row, col] = value
            gram[col, row] = value
        end
        gram[col, col] += lambda
    end

    factor = cholesky!(Symmetric(gram, :L))
    copyto!(model.coefficients, rhs)
    ldiv!(factor, model.coefficients)
    model.target_mean = ybar
    model.intercept = ybar - dot(model.coefficients, xbar)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Loss function (pure functional, used for numerical-gradient testing)
# ─────────────────────────────────────────────────────────────────────────────

"""
    nn_loss(W1, b1, w2, b2_ref, X, q) -> Float64

Unregularized MSE loss for a one-hidden-layer ReLU network. Retained for
numerical-gradient testing; not on the hot training path.

X is d_in x n (column-major batch), q is length-n target vector.
b2_ref wraps the scalar output bias.
"""
function nn_loss(
    W1::Matrix{Float64},
    b1::Vector{Float64},
    w2::Vector{Float64},
    b2_ref::Base.RefValue{Float64},
    X::AbstractMatrix{Float64},
    q::AbstractVector{Float64},
)::Float64
    b2 = b2_ref[]
    n = length(q)
    d_in, _ = size(X)
    h = length(b1)

    total_mse = 0.0
    @inbounds for j in 1:n
        y_j = b2
        for i in 1:h
            act = b1[i]
            for k in 1:d_in
                act += W1[i, k] * X[k, j]
            end
            act = max(act, 0.0)
            y_j += w2[i] * act
        end
        total_mse += (y_j - q[j])^2
    end
    return total_mse / n
end

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@inline nn_param_count(h::Int, d_in::Int)::Int = h * d_in + 2 * h + 1

function ensure_nn_param_buffers!(grad::NNGradBuffers, h::Int, d_in::Int)
    n_params = nn_param_count(h, d_in)
    length(grad.theta) == n_params || resize!(grad.theta, n_params)
    length(grad.dtheta) == n_params || resize!(grad.dtheta, n_params)
    # Adam moments must start at zero whenever (re)sized; a fresh size also resets
    # the Adam timestep so bias correction restarts with the new parameter vector.
    if length(grad.m) != n_params
        grad.m = zeros(n_params)
        grad.v = zeros(n_params)
        grad.adam_t[] = 0
    end
    return nothing
end

function pack_nn_params!(theta::Vector{Float64}, nn::NeuralNet)
    h, d_in = size(nn.W1)
    @assert length(theta) == nn_param_count(h, d_in)
    pos = 1
    @inbounds for k in 1:d_in, i in 1:h
        theta[pos] = nn.W1[i, k]
        pos += 1
    end
    @inbounds for i in 1:h
        theta[pos] = nn.b1[i]
        pos += 1
    end
    @inbounds for i in 1:h
        theta[pos] = nn.w2[i]
        pos += 1
    end
    theta[pos] = nn.b2
    return nothing
end

function unpack_nn_grad!(grad::NNGradBuffers, h::Int, d_in::Int)
    theta_grad = grad.dtheta
    @assert length(theta_grad) == nn_param_count(h, d_in)
    pos = 1
    @inbounds for k in 1:d_in, i in 1:h
        grad.dW1[i, k] = theta_grad[pos]
        pos += 1
    end
    @inbounds for i in 1:h
        grad.db1[i] = theta_grad[pos]
        pos += 1
    end
    @inbounds for i in 1:h
        grad.dw2[i] = theta_grad[pos]
        pos += 1
    end
    grad.db2[] = theta_grad[pos]
    return nothing
end

function apply_nn_gradient!(nn::NeuralNet, grad::NNGradBuffers, lr::Float64)
    h, d_in = size(nn.W1)
    theta_grad = grad.dtheta
    pos = 1
    @inbounds for k in 1:d_in, i in 1:h
        nn.W1[i, k] -= lr * theta_grad[pos]
        pos += 1
    end
    @inbounds for i in 1:h
        nn.b1[i] -= lr * theta_grad[pos]
        pos += 1
    end
    @inbounds for i in 1:h
        nn.w2[i] -= lr * theta_grad[pos]
        pos += 1
    end
    nn.b2 -= lr * theta_grad[pos]
    return nothing
end

"""
    apply_nn_adam!(nn, grad, lr)

Apply one Adam update. `grad.dtheta` must already hold the raw packed gradient
(as produced by `gradient!`). This updates the persistent first/second moment
buffers in place, overwrites `grad.dtheta` with the bias-corrected Adam step
`m̂ / (√v̂ + ϵ)`, and applies it through `apply_nn_gradient!`. The Adam timestep
`grad.adam_t` advances once per call and persists across periods, so warm-started
weights carry warm-started moments.
"""
function apply_nn_adam!(nn::NeuralNet, grad::NNGradBuffers, lr::Float64)
    grad.adam_t[] += 1
    t = grad.adam_t[]
    bc1 = 1.0 - ADAM_BETA1^t
    bc2 = 1.0 - ADAM_BETA2^t
    m = grad.m
    v = grad.v
    g = grad.dtheta
    @inbounds for i in eachindex(g)
        gi = g[i]
        mi = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * gi
        vi = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * gi * gi
        m[i] = mi
        v[i] = vi
        g[i] = (mi / bc1) / (sqrt(vi / bc2) + ADAM_EPS)
    end
    apply_nn_gradient!(nn, grad, lr)
    return nothing
end

function nn_loss_theta(
    theta::Vector{Float64},
    X::Matrix{Float64},
    q::Vector{Float64},
    n::Int,
    h::Int,
    d_in::Int,
)::Float64
    w1_stop = h * d_in
    b1_start = w1_stop + 1
    w2_start = b1_start + h
    b2 = theta[w2_start + h]

    total_mse = 0.0
    @inbounds for j in 1:n
        y_j = b2
        for i in 1:h
            act = theta[b1_start + i - 1]
            for k in 1:d_in
                act += theta[(k - 1) * h + i] * X[k, j]
            end
            if act > 0.0
                y_j += theta[w2_start + i - 1] * act
            end
        end
        err = y_j - q[j]
        total_mse += err * err
    end
    return total_mse / n
end

"""
    train_step!(nn, grad, X, q, lr)

One vanilla-GD step. Gradients are computed through DifferentiationInterface
with the Enzyme backend, then copied into `grad` and applied to `nn`.
"""
function train_step!(
    nn::NeuralNet, grad::NNGradBuffers, X::Matrix{Float64}, q::Vector{Float64}, lr::Float64
)
    train_step_prefix!(nn, grad, X, q, length(q), lr)
    return nothing
end

"""
    train_step_prefix!(nn, grad, X, q, n, lr)

One vanilla-GD step on the first `n` columns/elements of contiguous training
buffers `X` and `q`. This supports broker training directly on the active prefix
of the preallocated broker feature buffer without recopying it.
"""
function train_step_prefix!(
    nn::NeuralNet,
    grad::NNGradBuffers,
    X::Matrix{Float64},
    q::Vector{Float64},
    n::Int,
    lr::Float64,
)
    h = size(nn.W1, 1)
    d_in = size(nn.W1, 2)
    ensure_nn_param_buffers!(grad, h, d_in)
    pack_nn_params!(grad.theta, nn)
    gradient!(
        nn_loss_theta,
        grad.dtheta,
        NN_AD_BACKEND,
        grad.theta,
        Constant(X),
        Constant(q),
        Constant(n),
        Constant(h),
        Constant(d_in),
    )
    unpack_nn_grad!(grad, h, d_in)
    apply_nn_gradient!(nn, grad, lr)

    return nothing
end

"""
    train_step_prefix_adam!(nn, grad, X, q, n, lr)

One Adam step on the first `n` columns/elements of contiguous training buffers
`X` and `q`. This is the live-model optimizer for both agents and the broker; it
shares the exact loss (`nn_loss_theta`) and Enzyme gradient path with the vanilla
GD step, swapping only the parameter update rule (`apply_nn_adam!`).
"""
function train_step_prefix_adam!(
    nn::NeuralNet,
    grad::NNGradBuffers,
    X::Matrix{Float64},
    q::Vector{Float64},
    n::Int,
    lr::Float64,
)
    h = size(nn.W1, 1)
    d_in = size(nn.W1, 2)
    ensure_nn_param_buffers!(grad, h, d_in)
    pack_nn_params!(grad.theta, nn)
    gradient!(
        nn_loss_theta,
        grad.dtheta,
        NN_AD_BACKEND,
        grad.theta,
        Constant(X),
        Constant(q),
        Constant(n),
        Constant(h),
        Constant(d_in),
    )
    apply_nn_adam!(nn, grad, lr)
    return nothing
end

"""
    train_nn_prefix_adam!(nn, grad, X, q, n_active, n_steps, lr)

Adam analogue of `train_nn_prefix!`: run `n_steps` Adam steps on the first
`n_active` observations of contiguous training buffers.
"""
function train_nn_prefix_adam!(
    nn::NeuralNet,
    grad::NNGradBuffers,
    X::Matrix{Float64},
    q::Vector{Float64},
    n_active::Int,
    n_steps::Int,
    lr::Float64,
)
    @assert 1 <= n_active <= size(X, 2) "train_nn_prefix_adam! requires 1 <= n_active <= size(X, 2)"
    @assert n_active <= length(q) "train_nn_prefix_adam! requires n_active <= length(q)"
    for _ in 1:n_steps
        train_step_prefix_adam!(nn, grad, X, q, n_active, lr)
    end
    return nothing
end

"""
    compute_adaptive_steps(E_init, n_new, n_total; min_steps) -> Int

Adaptive training schedule: more steps when data is new (`n_new` large relative to
total history `n_total`), settling to `min_steps` once history dominates. `n_total`
is the full history size (not the training window), so the step count is
independent of the window/cap. `min_steps` is the per-period step floor.
"""
function compute_adaptive_steps(
    E_init::Int, n_new::Int, n_total::Int; min_steps::Int=ADAPTIVE_FLOOR
)::Int
    n_total <= 0 && return E_init
    return max(min_steps, ceil(Int, E_init * n_new / n_total))
end

"""
    train_nn!(nn, grad, X, q, n_steps, lr)

Train the network for n_steps of vanilla GD on the full batch (X, q).
"""
function train_nn!(
    nn::NeuralNet,
    grad::NNGradBuffers,
    X::Matrix{Float64},
    q::Vector{Float64},
    n_steps::Int,
    lr::Float64,
)
    for _ in 1:n_steps
        train_step!(nn, grad, X, q, lr)
    end
    return nothing
end

function train_nn!(
    nn::NeuralNet,
    grad::NNGradBuffers,
    X::AbstractMatrix{Float64},
    q::AbstractVector{Float64},
    n_steps::Int,
    lr::Float64,
)
    # BLAS mul! on SubArray hits a slow dispatch path (~5 MB allocs/call).
    # Materialize the training window into contiguous Matrix/Vector once,
    # then run the tight train_step! loop on those (zero-alloc per step).
    Xc = Matrix{Float64}(X)
    qc = Vector{Float64}(q)
    train_nn!(nn, grad, Xc, qc, n_steps, lr)
    return nothing
end

"""
    train_nn_prefix!(nn, grad, X, q, n_active, n_steps, lr)

Train on the first `n_active` observations in contiguous training buffers.
Used after agent and broker histories have been copied into reusable scratch space.
"""
function train_nn_prefix!(
    nn::NeuralNet,
    grad::NNGradBuffers,
    X::Matrix{Float64},
    q::Vector{Float64},
    n_active::Int,
    n_steps::Int,
    lr::Float64,
)
    @assert 1 <= n_active <= size(X, 2) "train_nn_prefix! requires 1 <= n_active <= size(X, 2)"
    @assert n_active <= length(q) "train_nn_prefix! requires n_active <= length(q)"

    for _ in 1:n_steps
        train_step_prefix!(nn, grad, X, q, n_active, lr)
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Agent training
# ─────────────────────────────────────────────────────────────────────────────

"""Minimum GD steps per training period."""
const ADAPTIVE_FLOOR = 50

"""Minimum capacity jump for agent training scratch after initialization."""
const AGENT_TRAIN_BUFFER_FLOOR = 128

"""
    period_training_window(marks, n, window_periods, max_obs) -> (start_idx, window, count)

Resolve the period horizon to history columns. `marks[1]` is the cumulative
history count at the end of initialization (period 0), and later entries are the
counts at the ends of simulation periods 1, 2, and so on. The most recent
`window_periods` simulation periods span the `window`-wide block `start_idx:n`.
Before that many periods exist, initialization remains in the window. `count =
min(window, max_obs)` observations are taken from the block, evenly spaced via
`windowed_index`, using the full cap when necessary and every observation
otherwise. A capped sample includes both window endpoints when `count >= 2` and
the newest observation when `count == 1`.
"""
function period_training_window(
    marks::Vector{Int}, n::Int, window_periods::Int, max_obs::Int
)
    P = length(marks)
    start_idx = P <= window_periods ? 1 : min(marks[P - window_periods] + 1, n)
    window = n - start_idx + 1
    count = min(window, max_obs)
    return start_idx, window, count
end

"""
    windowed_index(start_idx, window, count, k) -> Int

0-based `k`-th source column when taking `count` observations evenly across the
`window`-wide block beginning at `start_idx`. A multi-observation sample includes
the oldest and newest columns. A one-observation sample uses the newest column.
Integer arithmetic, no allocation; reduces to consecutive indices when
`count == window`.
"""
@inline windowed_index(start_idx::Int, window::Int, count::Int, k::Int)::Int =
    count == 1 ? start_idx + window - 1 : start_idx + (k * (window - 1)) ÷ (count - 1)

"""
    ensure_agent_train_buffers!(agent, d, n, max_obs)

Ensure the agent's contiguous training scratch can hold `n` observations.
Growth jumps past tiny capacities to avoid repeated hot-path reallocations as
histories lengthen, while still capping the scratch at `max_obs`.
"""
function ensure_agent_train_buffers!(agent::Agent, d::Int, n::Int, max_obs::Int)
    if size(agent.train_X, 1) != d || size(agent.train_X, 2) < n
        current_cap = size(agent.train_X, 2)
        new_cap = min(max_obs, max(n, 2 * current_cap, AGENT_TRAIN_BUFFER_FLOOR))
        agent.train_X = Matrix{Float64}(undef, d, new_cap)
        resize!(agent.train_q, new_cap)
    end
    return nothing
end

function prepare_agent_training!(agent::Agent, params::ModelParams)::Int
    n = agent.history_count
    n <= 0 && return 0

    start_idx, window, count = period_training_window(
        agent.obs_period_marks, n, params.train_window_periods, params.train_max_obs
    )
    d = params.d
    ensure_agent_train_buffers!(agent, d, count, params.train_max_obs)

    history_X = agent.history_X
    train_X = agent.train_X
    train_q = agent.train_q
    @inbounds for k in 0:(count - 1)
        j = windowed_index(start_idx, window, count, k)
        col = k + 1
        for row in 1:d
            train_X[row, col] = history_X[row, j]
        end
        train_q[col] = agent.history_q[j]
    end
    return count
end

function train_agent_nn_impl!(agent::Agent, params::ModelParams)
    n = agent.history_count
    n <= 0 && return nothing
    n_steps = compute_adaptive_steps(
        params.E_init, agent.n_new_obs, n; min_steps=params.train_steps
    )
    count = prepare_agent_training!(agent, params)
    agent.n_new_obs = 0
    train_nn_prefix_adam!(
        agent.nn, agent.nn_grad, agent.train_X, agent.train_q, count, n_steps, params.eta_lr
    )
    return nothing
end

"""
    train_agent_nn!(agent, params)

Train the agent's neural network on the observations recorded in the most recent
`train_window_periods` periods (see `period_training_window`), with an adaptive
step count and warm-started Adam state.
"""
function train_agent_nn!(agent::Agent, params::ModelParams)
    train_agent_nn_impl!(agent, params)
    return nothing
end

"""Fit an agent's Ridge model on the same training sample used by the NN."""
function train_agent_ridge!(agent::Agent, params::ModelParams)
    agent.history_count <= 0 && return nothing
    count = prepare_agent_training!(agent, params)
    agent.n_new_obs = 0
    fit_ridge!(
        agent.ridge::RidgeModel, agent.train_X, agent.train_q, count, params.ridge_lambda
    )
    return nothing
end

"""Fit the agent's selected prediction model."""
function train_agent_predictor!(agent::Agent, params::ModelParams)
    if params.learning_model == :nn
        train_agent_nn!(agent, params)
    else
        train_agent_ridge!(agent, params)
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Broker training
# ─────────────────────────────────────────────────────────────────────────────

"""
    train_broker_nn!(broker, params)

Train the broker on symmetric pair features from the period-based window
(`period_training_window`), subsampled across its span to `train_max_obs`.
"""
function train_broker_nn!(broker::Broker, params::ModelParams)
    n = broker.history_count
    n <= 0 && return nothing
    count = prepare_broker_training!(broker, nothing, params, nothing; variant=:pair)

    # Step count is set by the new-data ratio over full history (independent of the
    # window/cap), settling to the params.train_steps floor once history dominates.
    n_steps = compute_adaptive_steps(
        params.E_init, broker.n_new_obs, n; min_steps=params.train_steps
    )
    broker.n_new_obs = 0

    train_nn_prefix_adam!(
        broker.nn,
        broker.nn_grad,
        broker.train_X,
        broker.train_q,
        count,
        n_steps,
        params.eta_lr,
    )
    return nothing
end

function ensure_broker_train_buffers!(broker::Broker, n_features::Int, count::Int)
    if size(broker.train_X, 1) != n_features || size(broker.train_X, 2) < count
        current_cap = size(broker.train_X, 2)
        new_cap = max(count, 2 * current_cap, 128)
        broker.train_X = Matrix{Float64}(undef, n_features, new_cap)
        resize!(broker.train_q, new_cap)
    end
    return nothing
end

function typical_agent_training_count!(
    broker::Broker, agents::Vector{Agent}, params::ModelParams
)::Int
    counts = broker.agent_train_counts
    empty!(counts)
    @inbounds for agent in agents
        n = agent.history_count
        n <= 0 && continue
        _, _, count = period_training_window(
            agent.obs_period_marks, n, params.train_window_periods, params.train_max_obs
        )
        push!(counts, count)
    end
    isempty(counts) && return 0
    sort!(counts)
    m = length(counts)
    return if isodd(m)
        counts[(m + 1) ÷ 2]
    else
        round(Int, (counts[m ÷ 2] + counts[m ÷ 2 + 1]) / 2, RoundNearestTiesUp)
    end
end

function choose_size_matched_indices!(
    broker::Broker, available::Int, selected::Int, rng::AbstractRNG
)
    indices = broker.sample_indices
    length(indices) < available && resize!(indices, available)
    @inbounds for idx in 1:available
        indices[idx] = idx
    end
    @inbounds for idx in 1:selected
        swap_idx = rand(rng, idx:available)
        indices[idx], indices[swap_idx] = indices[swap_idx], indices[idx]
    end
    return nothing
end

function fill_broker_training_column!(
    broker::Broker, col::Int, history_idx::Int, variant::Symbol, d::Int
)
    if variant in (:pair, :size_matched)
        fill_broker_pair_features!(
            broker.train_X,
            col,
            broker.history_party1_types,
            history_idx,
            broker.history_party2_types,
            history_idx,
        )
    elseif variant == :additive
        @inbounds for row in 1:d
            broker.train_X[row, col] =
                broker.history_party1_types[row, history_idx] +
                broker.history_party2_types[row, history_idx]
        end
    else
        retained = broker.history_retained_party[history_idx]
        @assert retained in (UInt8(1), UInt8(2))
        source =
            retained == UInt8(1) ? broker.history_party1_types : broker.history_party2_types
        @inbounds for row in 1:d
            broker.train_X[row, col] = source[row, history_idx]
        end
    end
    broker.train_q[col] = broker.history_q[history_idx]
    return nothing
end

function prepare_broker_training!(
    broker::Broker,
    agents::Union{Vector{Agent},Nothing},
    params::ModelParams,
    rng::Union{AbstractRNG,Nothing};
    variant::Symbol=params.ridge_broker_variant,
)::Int
    n = broker.history_count
    n <= 0 && return 0
    start_idx, window, available = period_training_window(
        broker.obs_period_marks, n, params.train_window_periods, params.train_max_obs
    )
    count = available
    if variant == :size_matched
        isnothing(rng) && error("size-matched broker training requires an RNG")
        isnothing(agents) && error("size-matched broker training requires agents")
        count = min(
            available, typical_agent_training_count!(broker, agents::Vector{Agent}, params)
        )
        count <= 0 && return 0
        choose_size_matched_indices!(broker, available, count, rng)
    end

    n_features = if variant in (:additive, :single_principal)
        params.d
    else
        broker_pair_feature_dim(params.d)
    end
    ensure_broker_train_buffers!(broker, n_features, count)
    @inbounds for col in 1:count
        window_pos = variant == :size_matched ? broker.sample_indices[col] - 1 : col - 1
        history_idx = windowed_index(start_idx, window, available, window_pos)
        fill_broker_training_column!(broker, col, history_idx, variant, params.d)
    end
    return count
end

"""Fit the selected Ridge broker variant."""
function train_broker_ridge!(
    broker::Broker, agents::Vector{Agent}, params::ModelParams, rng::AbstractRNG
)
    broker.history_count <= 0 && return nothing
    count = prepare_broker_training!(broker, agents, params, rng)
    count <= 0 && return nothing
    broker.n_new_obs = 0
    fit_ridge!(
        broker.ridge::RidgeModel, broker.train_X, broker.train_q, count, params.ridge_lambda
    )
    return nothing
end

"""Fit the broker's selected prediction model."""
function train_broker_predictor!(
    broker::Broker, agents::Vector{Agent}, params::ModelParams, rng::AbstractRNG
)
    if params.learning_model == :nn
        train_broker_nn!(broker, params)
    else
        train_broker_ridge!(broker, agents, params, rng)
    end
    return nothing
end
