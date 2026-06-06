"""
    learning.jl

Neural network prediction models for agents and broker.
One-hidden-layer ReLU networks trained by vanilla full-batch gradient descent.
Gradients are computed through DifferentiationInterface with Enzyme.

Agent: input x_j (d features), with width derived from d.
Broker: symmetric additive and bilinear pair features, with width derived from d.
"""

using LinearAlgebra: mul!, dot, BLAS
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

Initialize a one-hidden-layer ReLU network with Kaiming (He) initialization.
The output bias `b2` is initialized to `b2_init` (default `Q_OFFSET`) so that an
untrained network outputs approximately the population mean match quality,
rather than zero. This avoids a large negative-bias artifact for fresh entrants
whose NN has not yet been trained, without changing the behavior of mature NNs
(the first training step on any data shifts `b2` to its fitted value).
"""
function init_neural_net(
    d_in::Int, h::Int, rng::AbstractRNG; b2_init::Float64=Q_OFFSET
)::NeuralNet
    # He initialization: scale = sqrt(2 / fan_in)
    scale_1 = sqrt(2.0 / d_in)
    W1 = scale_1 .* randn(rng, h, d_in)
    b1 = zeros(h)
    # Output layer: Xavier scale
    scale_2 = sqrt(1.0 / h)
    w2 = scale_2 .* randn(rng, h)
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

Resolve the period horizon to history columns. `marks[k]` is the cumulative
`history_count` at the end of the learner's k-th period, so the most recent
`window_periods` periods span the `window`-wide block `start_idx:n` (start is index
1 if fewer periods exist). `count = min(window, max_obs)` observations are taken
from that block, evenly spaced via `windowed_index` — using the full budget exactly
when the window exceeds the cap, and every observation otherwise.
"""
function period_training_window(marks::Vector{Int}, n::Int, window_periods::Int, max_obs::Int)
    P = length(marks)
    start_idx = P <= window_periods ? 1 : min(marks[P - window_periods] + 1, n)
    window = n - start_idx + 1
    count = min(window, max_obs)
    return start_idx, window, count
end

"""
    windowed_index(start_idx, window, count, k) -> Int

0-based `k`-th source column when taking `count` observations evenly spaced across
the `window`-wide block beginning at `start_idx`. Integer arithmetic, no
allocation; reduces to consecutive indices when `count == window`.
"""
@inline windowed_index(start_idx::Int, window::Int, count::Int, k::Int)::Int =
    start_idx + (k * window) ÷ count

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

function train_agent_nn_impl!(agent::Agent, params::ModelParams)
    n = agent.history_count
    n <= 0 && return nothing

    # Period-based window: observations from the most recent train_window_periods
    # periods, evenly subsampled to the compute cap when the window exceeds it.
    start_idx, window, count = period_training_window(
        agent.obs_period_marks, n, params.train_window_periods, params.train_max_obs
    )

    # Step count is set by the new-data ratio over full history (independent of the
    # window/cap), settling to the params.train_steps floor once history dominates.
    n_steps = compute_adaptive_steps(params.E_init, agent.n_new_obs, n; min_steps=params.train_steps)
    agent.n_new_obs = 0

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

    d = params.d
    d_broker = broker_pair_feature_dim(d)

    # Period-based window, evenly subsampled to the cap when it exceeds it.
    start_idx, window, count = period_training_window(
        broker.obs_period_marks, n, params.train_window_periods, params.train_max_obs
    )

    # Ensure training buffers are large enough
    if size(broker.train_X, 1) != d_broker || size(broker.train_X, 2) < count
        new_cap = max(count, 2 * size(broker.train_X, 2), 128)
        broker.train_X = Matrix{Float64}(undef, d_broker, new_cap)
        resize!(broker.train_q, new_cap)
    end

    # Build symmetric-feature training data from the evenly-spaced window sample
    @inbounds for k in 0:(count - 1)
        j = windowed_index(start_idx, window, count, k)
        col = k + 1
        fill_broker_pair_features!(
            broker.train_X, col, broker.history_Xi, j, broker.history_Xj, j
        )
        broker.train_q[col] = broker.history_q[j]
    end

    # Step count is set by the new-data ratio over full history (independent of the
    # window/cap), settling to the params.train_steps floor once history dominates.
    n_steps = compute_adaptive_steps(params.E_init, broker.n_new_obs, n; min_steps=params.train_steps)
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
