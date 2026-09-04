"""
    stage2_training.jl

Compare full-batch vanilla gradient descent and Adam on fixed, broadly sampled
training data. Step-budget, learning-rate, sample-size, and noise comparisons
measure optimizer sensitivity while holding endogenous selection outside the
experiment. Production neural-network training uses persistent Adam state.

Usage: julia --project --threads=auto scripts/diagnostics/stage2_training.jl
"""

include("broker_learning_common.jl")

const SEED = 42
const D = 8
const N = 500              # clean-data batch size for the optimizer comparison
const N_TEST = 4000

"""Fit with `opt` (`:gd` or `:adam`) for a fixed budget and report component recovery."""
function recover(env, pool; n=N, steps=50, lr=0.03, sigma_eps=0.0, seed=1, h=broker_hidden_width(D), opt=:gd)
    rng = StableRNG(seed)
    Xi, Xj = sample_pairs(env, pool, n, rng; source=:pool)
    c = decompose(env, Xi, Xj)
    y = copy(c.target)
    if sigma_eps > 0
        y .+= sigma_eps .* randn(rng, n)
    end
    Z = broker_features(Xi, Xj)
    db = size(Z, 1)
    nn = init_neural_net(db, h, StableRNG(seed + 7); b2_init=Q_OFFSET)
    grad = NNGradBuffers(nn)
    if opt === :gd
        train_nn!(nn, grad, Matrix(Z), Vector(y), steps, lr)
    elseif opt === :adam
        train_nn_adam!(nn, grad, Matrix(Z), Vector(y), steps, lr)
    else
        error("unknown opt $opt")
    end

    Xi_te, Xj_te = sample_pairs(env, pool, N_TEST, StableRNG(seed + 999); source=:pool)
    cte = decompose(env, Xi_te, Xj_te)
    Zte = broker_features(Xi_te, Xj_te)
    return evaluate(predict_nn_cols(nn, Zte), cte)
end

function run_stage2()
    println("="^110)
    println("STAGE 2: Optimizer comparison on clean data (batch n=$N)")
    println("="^110)

    env, pool, _ = make_env(; d=D, delta=0.5, sigma_eps=0.0, seed=SEED)

    println("\n### Gain recovery by GD step budget   (lr=0.03, noiseless)")
    for steps in (50, 100, 200, 500, 1000, 2000, 5000)
        m = recover(env, pool; steps=steps, lr=0.03)
        print_row("  steps=$steps", m)
    end

    println("\n### Adam and GD at the same budget   (lr=0.01, noiseless)")
    for steps in (50, 100, 200, 500, 2000)
        mg = recover(env, pool; steps=steps, lr=0.01, opt=:gd)
        ma = recover(env, pool; steps=steps, lr=0.01, opt=:adam)
        print_row("  GD   steps=$steps", mg)
        print_row("  Adam steps=$steps", ma)
    end

    println("\n### Sensitivity to learning rate   (steps=2000, noiseless)")
    for lr in (0.01, 0.03, 0.1, 0.3)
        m = recover(env, pool; steps=2000, lr=lr)
        print_row("  lr=$lr", m)
    end

    println("\n### With model noise σ_ε=0.10   (steps=2000, lr=0.03)")
    m = recover(env, pool; steps=2000, lr=0.03, sigma_eps=0.10)
    print_row("  σ_ε=0.10", m)

    println("\n### Effect of training-window size n   (steps=2000, lr=0.03, noiseless)")
    for n in (200, 500, 1000, 4000)
        m = recover(env, pool; n=n, steps=2000, lr=0.03)
        print_row("  n=$n", m)
    end

    println("\n" * "="^110)
    println("INTERPRETATION: differences between Adam and GD at a common budget measure")
    println("optimizer sensitivity. Compare long-budget recovery across both optimizers before")
    println("drawing conclusions about network capacity.")
    println("="^110)
end

run_stage2()
