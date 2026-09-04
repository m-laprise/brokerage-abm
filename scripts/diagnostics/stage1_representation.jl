"""
    stage1_representation.jl

Measure gain-component recovery on noiseless, broadly sampled pairs. The script
compares a linear readout with one-hidden-layer broker networks trained by
full-batch vanilla gradient descent across gain strengths and network widths.
`stage1b_representation_ceiling.jl` provides the Adam comparison.

Usage: julia --project --threads=auto scripts/diagnostics/stage1_representation.jl
"""

include("broker_learning_common.jl")

const SEED = 42
const D = 8
const N_TRAIN = 1200
const N_TEST = 3000
const LR = 0.03
const STEPS = 2000

function fit_broker_long(Z, y, h; steps=STEPS, lr=LR, seed=1)
    db = size(Z, 1)
    nn = init_neural_net(db, h, StableRNG(seed); b2_init=Q_OFFSET)
    grad = NNGradBuffers(nn)
    train_nn!(nn, grad, Matrix(Z), Vector(y), steps, lr)
    return nn
end

"""Build train/test features + decompositions for one (δ, source)."""
function setup(δ, source)
    env, pool, _ = make_env(; d=D, delta=δ, sigma_eps=0.0, seed=SEED)
    rng = StableRNG(SEED + 1)
    Xi, Xj = sample_pairs(env, pool, N_TRAIN, rng; source=source)
    Xi_te, Xj_te = sample_pairs(env, pool, N_TEST, rng; source=source)
    return (
        Ztr=broker_features(Xi, Xj),
        Zte=broker_features(Xi_te, Xj_te),
        ctr=decompose(env, Xi, Xj),
        cte=decompose(env, Xi_te, Xj_te),
    )
end

function eval_cell(δ, source; widths=(8, 16))
    s = setup(δ, source)
    vs = variance_shares(s.cte)
    @printf(
        "\nδ=%.2f [%s] | variance ratios: quality=%.2f core=%.2f gain=%.2f\n",
        δ, source, vs.frac_quality, vs.frac_core, vs.frac_gain,
    )
    flush(stdout)
    w, μ = fit_linear_readout(s.Ztr, s.ctr.target)
    print_row("  linear readout", evaluate(predict_linear(s.Zte, w, μ), s.cte))
    for mult in widths
        h = mult * D
        nn = fit_broker_long(s.Ztr, s.ctr.target, h)
        print_row("  broker NN (h=$(h)=$(mult)d)", evaluate(predict_nn_cols(nn, s.Zte), s.cte))
    end
end

function run_stage1()
    println("="^115)
    println("STAGE 1: Vanilla-GD recovery (noiseless, long training, no window)")
    println("  d=$D, n_train=$N_TRAIN, n_test=$N_TEST, GD steps=$STEPS, lr=$LR")
    println("="^115)
    flush(stdout)

    # Compare the model width with a wider network.
    for source in (:pool, :sphere)
        for δ in (0.0, 0.5)
            eval_cell(δ, source; widths=(8, 16))
        end
    end
    # Additional gain-strength value on realized-type support.
    eval_cell(0.75, :pool; widths=(8, 16))

    # Run the widest network once because its per-step cost is substantially higher.
    println("\n### Wider-network comparison (δ=0.5, pool, h=32d, steps=2500)")
    flush(stdout)
    s = setup(0.5, :pool)
    nn = fit_broker_long(s.Ztr, s.ctr.target, 32 * D; steps=2500)
    print_row("  broker NN (h=$(32D)=32d)", evaluate(predict_nn_cols(nn, s.Zte), s.cte))

    println("\n" * "="^115)
    println("INTERPRETATION: βg near one indicates recovery of the gain component. Compare")
    println("recovery across widths and with the Adam fits in Stage 1b before attributing")
    println("a shortfall to network capacity.")
    println("="^115)
    flush(stdout)
end

run_stage1()
