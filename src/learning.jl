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
    compute_adaptive_steps(E_init, n_new, n_total) -> Int

Adaptive training schedule: more steps when data is new, fewer when history is large.
Floor of 50 steps ensures meaningful updates even with large histories.
"""
function compute_adaptive_steps(E_init::Int, n_new::Int, n_total::Int)::Int
    n_total <= 0 && return E_init
    return max(ADAPTIVE_FLOOR, ceil(Int, E_init * n_new / n_total))
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

"""Maximum training window: train on at most this many recent observations.
The warm start preserves what was learned from older data."""
const TRAIN_WINDOW = 500

"""Minimum GD steps per training period."""
const ADAPTIVE_FLOOR = 50

"""Minimum capacity jump for agent training scratch after initialization."""
const AGENT_TRAIN_BUFFER_FLOOR = 128

"""Small windows are cheaper to materialize directly than to route through the
agent-owned training scratch. This preserves the hot-path win on larger windows
without penalizing tiny seeded histories during initialization."""
const AGENT_TRAIN_DIRECT_COPY_THRESHOLD = 8

"""
    ensure_agent_train_buffers!(agent, d, n)

Ensure the agent's contiguous training scratch can hold `n` observations.
Growth jumps past tiny capacities to avoid repeated hot-path reallocations as
histories lengthen, while still capping the scratch at the training window.
"""
function ensure_agent_train_buffers!(agent::Agent, d::Int, n::Int)
    if size(agent.train_X, 1) != d || size(agent.train_X, 2) < n
        current_cap = size(agent.train_X, 2)
        new_cap = min(TRAIN_WINDOW, max(n, 2 * current_cap, AGENT_TRAIN_BUFFER_FLOOR))
        agent.train_X = Matrix{Float64}(undef, d, new_cap)
        resize!(agent.train_q, new_cap)
    end
    return nothing
end

function train_agent_nn_impl!(agent::Agent, params::ModelParams, direct_copy_small::Bool)
    n = agent.history_count
    n <= 0 && return nothing

    n_use = min(n, TRAIN_WINDOW)
    start_idx = n - n_use + 1

    # Adaptive steps
    n_steps = compute_adaptive_steps(params.E_init, agent.n_new_obs, n)
    agent.n_new_obs = 0

    if direct_copy_small && n_use <= AGENT_TRAIN_DIRECT_COPY_THRESHOLD
        X = view(agent.history_X, :, start_idx:n)
        q = view(agent.history_q, start_idx:n)
        train_nn!(agent.nn, agent.nn_grad, X, q, n_steps, params.eta_lr)
    else
        d = params.d
        ensure_agent_train_buffers!(agent, d, n_use)

        history_X = agent.history_X
        train_X = agent.train_X
        @inbounds for col in 1:n_use, row in 1:d
            train_X[row, col] = history_X[row, start_idx + col - 1]
        end
        copyto!(agent.train_q, 1, agent.history_q, start_idx, n_use)
        train_nn_prefix!(
            agent.nn,
            agent.nn_grad,
            agent.train_X,
            agent.train_q,
            n_use,
            n_steps,
            params.eta_lr,
        )
    end
    return nothing
end

"""
    train_agent_nn!(agent, params)

Train the agent's neural network on recent history with adaptive step count.
Uses a sliding window of at most `TRAIN_WINDOW` observations.
"""
function train_agent_nn!(agent::Agent, params::ModelParams)
    train_agent_nn_impl!(agent, params, false)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Broker training
# ─────────────────────────────────────────────────────────────────────────────

"""
    train_broker_nn!(broker, params)

Train the broker's neural network on symmetric pair features from recent history.
Uses a sliding window of the most recent TRAIN_WINDOW observations.
"""
function train_broker_nn!(broker::Broker, params::ModelParams)
    n = broker.history_count
    n <= 0 && return nothing

    d = params.d
    d_broker = broker_pair_feature_dim(d)

    # Sliding window
    n_use = min(n, TRAIN_WINDOW)
    start_idx = n - n_use + 1

    # Ensure training buffers are large enough
    if size(broker.train_X, 1) != d_broker || size(broker.train_X, 2) < n_use
        new_cap = max(n_use, 2 * size(broker.train_X, 2), 128)
        broker.train_X = Matrix{Float64}(undef, d_broker, new_cap)
        resize!(broker.train_q, new_cap)
    end

    # Build symmetric-feature training data from window
    @inbounds for (idx, j) in enumerate(start_idx:n)
        fill_broker_pair_features!(
            broker.train_X, idx, broker.history_Xi, j, broker.history_Xj, j
        )
        broker.train_q[idx] = broker.history_q[j]
    end

    # Adaptive steps
    n_steps = compute_adaptive_steps(params.E_init, broker.n_new_obs, n)
    broker.n_new_obs = 0

    train_nn_prefix!(
        broker.nn,
        broker.nn_grad,
        broker.train_X,
        broker.train_q,
        n_use,
        n_steps,
        params.eta_lr,
    )
    return nothing
end
