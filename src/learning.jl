"""
    learning.jl

Neural network prediction models for agents and broker.
One-hidden-layer ReLU networks trained by vanilla full-batch gradient descent
with hand-written forward and backward passes using BLAS `mul!`.

Agent: input x_j (d features), h_a hidden units.
Broker: input [x_i; x_j] (2d features), h_b hidden units.
No hand-crafted features; both receive raw type vectors.
"""

using LinearAlgebra: mul!, dot, BLAS
using Random: AbstractRNG

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

"""
    train_step!(nn, grad, X, q, lr)

One vanilla-GD step with hand-written forward + backward, using BLAS `mul!`
for all three matmuls. `grad` owns pre-allocated activation and gradient
buffers; resized once on first contact, then reused for the lifetime of the NN.

Forward:
    Z1 = W1 * X .+ b1
    A  = relu(Z1)
    Y  = A' * w2 .+ b2

Backward (MSE on full batch, no regularization):
    r  = (Y - q)                            shape n
    dw2 = (2/n) * A * r                     shape h
    db2 = (2/n) * sum(r)                    scalar
    dZ1[i,j] = (2/n) * w2[i] * r[j] * (Z1[i,j] > 0)
    dW1 = dZ1 * X'                          shape h x d_in
    db1 = sum(dZ1; dims=2)                  shape h
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
of the preallocated symmetry-augmented buffer without recopying it.
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
    ensure_nn_buffers!(grad, h, n)

    Z1 = grad.Z1;
    A = grad.A;
    dZ1 = grad.dZ1;
    Y = grad.Y
    b1 = nn.b1;
    w2 = nn.w2;
    b2 = nn.b2

    Xv = view(X, :, 1:n)
    qv = view(q, 1:n)
    Z1v = view(Z1, :, 1:n)
    Av = view(A, :, 1:n)
    dZ1v = view(dZ1, :, 1:n)
    Yv = view(Y, 1:n)

    BLAS.gemm!('N', 'N', 1.0, nn.W1, Xv, 0.0, Z1v)

    # Z1 += b1 (broadcast along columns), A = relu(Z1)
    @inbounds for j in 1:n, i in 1:h
        z = Z1v[i, j] + b1[i]
        Z1v[i, j] = z
        Av[i, j] = z > 0.0 ? z : 0.0
    end

    # Y = A' * w2 + b2
    BLAS.gemv!('T', 1.0, Av, w2, 0.0, Yv)
    @inbounds for j in 1:n
        Yv[j] += b2
    end

    # ── Backward ─────────────────────────────────────────────────────────────
    inv_n = 1.0 / n
    two_over_n = 2.0 * inv_n

    # r = Y - q   (store in Yv; reused below)
    sum_r = 0.0
    @inbounds for j in 1:n
        rj = Yv[j] - qv[j]
        Yv[j] = rj
        sum_r += rj
    end

    # dw2 = (2/n) * A * r
    BLAS.gemv!('N', two_over_n, Av, Yv, 0.0, grad.dw2)

    # db2 = (2/n) * sum(r)
    grad.db2[] = two_over_n * sum_r

    # dZ1[i,j] = (2/n) * w2[i] * r[j] * (Z1[i,j] > 0)
    # db1[i]   = sum_j dZ1[i,j]
    fill!(grad.db1, 0.0)
    @inbounds for j in 1:n
        rj_scaled = two_over_n * Yv[j]
        for i in 1:h
            if Z1v[i, j] > 0.0
                g = w2[i] * rj_scaled
                dZ1v[i, j] = g
                grad.db1[i] += g
            else
                dZ1v[i, j] = 0.0
            end
        end
    end

    # dW1 = dZ1 * X'   (h x d_in)
    BLAS.gemm!('N', 'T', 1.0, dZ1v, Xv, 0.0, grad.dW1)

    # ── Weight update ────────────────────────────────────────────────────────
    @inbounds @simd for idx in eachindex(nn.W1)
        nn.W1[idx] -= lr * grad.dW1[idx]
    end
    @inbounds @simd for i in 1:h
        nn.b1[i] -= lr * grad.db1[i]
        nn.w2[i] -= lr * grad.dw2[i]
    end
    nn.b2 -= lr * grad.db2[]

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
# Broker training (with symmetry augmentation)
# ─────────────────────────────────────────────────────────────────────────────

"""
    train_broker_nn!(broker, params)

Train the broker's neural network on symmetry-augmented recent history.
Uses a sliding window of the most recent TRAIN_WINDOW observations.
Each observation produces two training examples (symmetry augmentation).
"""
function train_broker_nn!(broker::Broker, params::ModelParams)
    n = broker.history_count
    n <= 0 && return nothing

    d = params.d

    # Sliding window
    n_use = min(n, TRAIN_WINDOW)
    start_idx = n - n_use + 1
    n_aug = 2 * n_use

    # Ensure training buffers are large enough
    if size(broker.train_X, 2) < n_aug
        new_cap = max(n_aug, 2 * size(broker.train_X, 2))
        broker.train_X = Matrix{Float64}(undef, 2 * d, new_cap)
        resize!(broker.train_q, new_cap)
    end

    # Build symmetry-augmented training data from window
    @inbounds for (idx, j) in enumerate(start_idx:n)
        for k in 1:d
            broker.train_X[k, idx] = broker.history_Xi[k, j]
            broker.train_X[d + k, idx] = broker.history_Xj[k, j]
        end
        broker.train_q[idx] = broker.history_q[j]

        for k in 1:d
            broker.train_X[k, n_use + idx] = broker.history_Xj[k, j]
            broker.train_X[d + k, n_use + idx] = broker.history_Xi[k, j]
        end
        broker.train_q[n_use + idx] = broker.history_q[j]
    end

    # Adaptive steps
    n_steps = compute_adaptive_steps(params.E_init, broker.n_new_obs, n)
    broker.n_new_obs = 0

    train_nn_prefix!(
        broker.nn,
        broker.nn_grad,
        broker.train_X,
        broker.train_q,
        n_aug,
        n_steps,
        params.eta_lr,
    )
    return nothing
end
