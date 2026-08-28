"""
synthetic_nn_calibration.jl

Isolate NN learning from simulation dynamics. Generate synthetic (x_i, x_j, q)
data from the DGP at the default env, train broker-style NNs with varying
hyperparameters, measure held-out R². This establishes the achievable R² as
a function of data size, width, activation, learning rate, and training steps.
"""

using Random, Statistics, LinearAlgebra, Printf
using StableRNGs: StableRNG
using BrokerageABM
using BrokerageABM: MatchingEnv, generate_matching_env, match_signal, Q_OFFSET,
                          default_params, generate_curve_geometry, generate_agent_types,
                          broker_hidden_width, broker_pair_feature_dim,
                          fill_broker_pair_features!

# ── Minimal NN with pluggable activation ─────────────────────────────────────
struct MiniNN
    W1::Matrix{Float64}   # h x d_in
    b1::Vector{Float64}   # h
    w2::Vector{Float64}   # h
    b2::Base.RefValue{Float64}
end

function init_mini(d_in, h, rng; b2_init=Q_OFFSET)
    scale_1 = sqrt(2.0 / d_in)
    scale_2 = sqrt(1.0 / h)
    MiniNN(scale_1 .* randn(rng, h, d_in),
           zeros(h),
           scale_2 .* randn(rng, h),
           Ref(b2_init))
end

# Activations and their derivatives
relu(x) = max(0.0, x); d_relu(x) = x > 0 ? 1.0 : 0.0
tanh_(x) = tanh(x); d_tanh(x) = 1.0 - tanh(x)^2
gelu(x) = 0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
function d_gelu(x)
    c = sqrt(2/π)
    u = c * (x + 0.044715 * x^3)
    t = tanh(u)
    0.5 * (1 + t) + 0.5 * x * (1 - t^2) * c * (1 + 3 * 0.044715 * x^2)
end

const ACT_FNS = Dict(
    :relu => (relu, d_relu),
    :tanh => (tanh_, d_tanh),
    :gelu => (gelu, d_gelu),
)

# Forward pass (batch)
function forward_batch(nn::MiniNN, X::Matrix{Float64}, act_fn)
    z1 = nn.W1 * X .+ nn.b1         # h x n
    h = act_fn.(z1)                  # h x n
    y = nn.w2' * h .+ nn.b2[]        # 1 x n
    (z1, h, vec(y))
end

# One GD step with explicit gradients
function train_step_mini!(nn::MiniNN, X::Matrix{Float64}, q::Vector{Float64},
                          lr::Float64, lambda_over_n::Float64, act_fn, dact_fn)
    n = length(q)
    z1, h_pre, y = forward_batch(nn, X, act_fn)
    resid = (y .- q) ./ n          # 1 x n (dL/dy for MSE)

    # d w2 = h * resid'  ; shape h
    dw2 = h_pre * resid
    db2 = sum(resid)

    # dh = w2 * resid'     h x n
    dh = nn.w2 .* resid'
    dz1 = dh .* dact_fn.(z1)        # h x n
    dW1 = dz1 * X'                  # h x d
    db1 = vec(sum(dz1, dims=2))     # h

    # Weight decay
    dW1 .+= 2 * lambda_over_n .* nn.W1
    db1 .+= 2 * lambda_over_n .* nn.b1
    dw2 .+= 2 * lambda_over_n .* nn.w2
    db2 += 2 * lambda_over_n * nn.b2[]

    nn.W1 .-= lr .* dW1
    nn.b1 .-= lr .* db1
    nn.w2 .-= lr .* dw2
    nn.b2[] -= lr * db2
end

function predict_batch(nn::MiniNN, X::Matrix{Float64}, act_fn)
    _, _, y = forward_batch(nn, X, act_fn)
    y
end

# ── Synthetic DGP data ──────────────────────────────────────────────────────
"""Generate symmetric broker pair features and match signals."""
function gen_data(n, env::MatchingEnv, types::Vector{Vector{Float64}}, rng)
    d = env.d
    N = length(types)
    X = Matrix{Float64}(undef, broker_pair_feature_dim(d), n)
    q = Vector{Float64}(undef, n)
    for k in 1:n
        i = rand(rng, 1:N); j = rand(rng, 1:N)
        while j == i
            j = rand(rng, 1:N)
        end
        fill_broker_pair_features!(X, k, types[i], types[j])
        q[k] = Q_OFFSET + match_signal(types[i], types[j], env) + env.sigma_eps * randn(rng)
    end
    X, q
end

# ── Scan core ───────────────────────────────────────────────────────────────
function eval_config(; n_train, n_test=5000, h, activation, lr, lambda_wd,
                     n_steps, d=8, seed=42)
    rng = StableRNG(seed)
    # Build env and types
    p = default_params(seed=seed)
    geo = generate_curve_geometry(d, p.s, rng)
    types = generate_agent_types(p.N, geo, p.sigma_x, rng)
    env = generate_matching_env(d, p.rho, p.delta, p.sigma_eps, types, rng;
                                sigma_x=p.sigma_x, curve_geo=geo)

    # Data
    X_tr, q_tr = gen_data(n_train, env, types, rng)
    X_te, q_te = gen_data(n_test, env, types, rng)

    # Train
    act_fn, dact_fn = ACT_FNS[activation]
    nn = init_mini(size(X_tr, 1), h, rng)
    lambda_over_n = lambda_wd / max(size(X_tr, 2), 1)
    for _ in 1:n_steps
        train_step_mini!(nn, X_tr, q_tr, lr, lambda_over_n, act_fn, dact_fn)
    end

    # Evaluate
    y_pred = predict_batch(nn, X_te, act_fn)
    mse = mean((y_pred .- q_te).^2)
    r2 = 1 - mse / var(q_te)
    (r2=r2, mse=mse, bias=mean(y_pred .- q_te))
end

# ── Scan 1: data size x width x activation, at "good" lr/steps ──────────────
function scan_data_width_act()
    println("\n══ SCAN 1: n_train × width × activation (lr=0.01, 5000 steps, λ=0.01) ══")
    println("Q: how much do width and activation matter with symmetric broker features?")
    @printf "%-8s %-6s %-5s %8s %8s %8s\n" "n_train" "width" "act" "R²" "bias" "MSE"
    println(repeat("-", 55))
    for n in [100, 200, 500, 2000, 10000]
        for h in [32, broker_hidden_width(8), 128]
            for a in [:relu, :tanh, :gelu]
                r = eval_config(n_train=n, h=h, activation=a, lr=0.01,
                               lambda_wd=0.01, n_steps=5000)
                @printf "%8d %6d %5s %+8.3f %+8.3f %8.3f\n" n h string(a) r.r2 r.bias r.mse
            end
        end
    end
end

# ── Scan 2: lr × steps at fixed "good" setup ────────────────────────────────
function scan_lr_steps()
    h = broker_hidden_width(8)
    println("\n══ SCAN 2: lr × n_steps (n_train=2000, h=$(h), ReLU, λ=0.01) ══")
    println("Q: is 200 steps/period enough at lr=0.01; should we go higher?")
    @printf "%-8s %-10s %8s %8s\n" "lr" "n_steps" "R²" "bias"
    println(repeat("-", 38))
    for lr in [0.003, 0.01, 0.03, 0.1]
        for ns in [200, 1000, 5000, 20000]
            r = eval_config(n_train=2000, h=h, activation=:relu, lr=lr,
                           lambda_wd=0.01, n_steps=ns)
            @printf "%8.3f %10d %+8.3f %+8.3f\n" lr ns r.r2 r.bias
        end
    end
end

# ── Scan 3: weight decay ────────────────────────────────────────────────────
function scan_lambda()
    h = broker_hidden_width(8)
    println("\n══ SCAN 3: λ_nn at n_train=2000, h=$(h), lr=0.03, 5000 steps ══")
    @printf "%-10s %8s %8s\n" "λ_nn" "R²" "bias"
    println(repeat("-", 30))
    for λ in [0.0, 0.001, 0.01, 0.1, 1.0]
        r = eval_config(n_train=2000, h=h, activation=:relu, lr=0.03,
                       lambda_wd=λ, n_steps=5000)
        @printf "%10.4f %+8.3f %+8.3f\n" λ r.r2 r.bias
    end
end

# ── Scan 4: best combo at various n ─────────────────────────────────────────
function scan_best_combo()
    println("\n══ SCAN 4: best combos at increasing n_train ══")
    println("Probe if R² continues improving with more data at good hyper")
    @printf "%-8s %-6s %-5s %-8s %-6s %8s\n" "n_train" "h" "act" "lr" "steps" "R²"
    println(repeat("-", 50))
    configs = [
        (h=broker_hidden_width(8), act=:relu, lr=0.03, steps=5000),
        (h=64, act=:tanh, lr=0.03, steps=5000),
        (h=64, act=:gelu, lr=0.03, steps=5000),
        (h=128, act=:tanh, lr=0.03, steps=5000),
    ]
    for n in [500, 2000, 10000, 30000]
        for c in configs
            r = eval_config(n_train=n, h=c.h, activation=c.act, lr=c.lr,
                           lambda_wd=0.01, n_steps=c.steps)
            @printf "%8d %6d %5s %8.3f %6d %+8.3f\n" n c.h string(c.act) c.lr c.steps r.r2
        end
    end
end

scan_data_width_act()
scan_lr_steps()
scan_lambda()
scan_best_combo()
