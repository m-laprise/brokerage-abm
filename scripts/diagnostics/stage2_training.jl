"""
    stage2_training.jl

STAGE 2 — Can the TRAINING PROCEDURE recover the interaction from clean data?

Stage 1 asks what the function class can represent given unlimited optimization.
Stage 2 holds the data clean and well-spread and asks whether the OPTIMIZER — not
the representation — is what loses the interaction: it compares vanilla GD against
Adam across a range of step budgets and learning rates on a fixed clean batch.

The finding that drove the fix: vanilla GD with a single global step size learns
the high-variance quality/core directions fast but starves the low-curvature gain
direction, so βg stays low even with many steps; Adam's per-parameter scaling
recovers the gain at the same budget. (This is why the live model now uses Adam.)

We train on i.i.d. clean draws (optionally with the model's σ_ε noise) so any
shortfall is attributable to optimization, not to data selection (Stage 3).

Usage: julia --project scripts/diagnostics/stage2_training.jl
"""

include("broker_learning_common.jl")

const SEED = 42
const D = 8
const N = 500              # clean-data batch size for the optimizer comparison
const N_TEST = 4000

"""Train for a fixed step budget; report gain recovery. `opt` selects the
optimizer: :gd is the model's live vanilla GD (the thing under test); :adam is
the strong-optimizer control at the SAME budget — if Adam recovers gain at the
live per-period budget where GD does not, the optimizer is the fixable bottleneck."""
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
    println("STAGE 2: Training recovery from clean data (real train_nn!, window-sized batch n=$N)")
    println("="^110)

    env, pool, _ = make_env(; d=D, delta=0.5, sigma_eps=0.0, seed=SEED)

    println("\n### Gain recovery vs GD step budget   (lr=0.03, noiseless)")
    println("  Vanilla GD: βg climbs only slowly with steps and lags Adam at every budget below.")
    for steps in (50, 100, 200, 500, 1000, 2000, 5000)
        m = recover(env, pool; steps=steps, lr=0.03)
        print_row("  steps=$steps", m)
    end

    println("\n### Same budget, Adam vs GD   (lr=0.01, noiseless) — is the OPTIMIZER the bottleneck?")
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
    println("READING: if vanilla-GD βg stays low across step budgets while Adam recovers it at the")
    println("same budget, the optimizer — not the representation — is the bottleneck. If βg stayed")
    println("low for BOTH optimizers even at 5000 steps, Stage 1's representational gap would bind.")
    println("="^110)
end

run_stage2()
