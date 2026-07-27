"""
    stage1_representation.jl

STAGE 1 — Can the broker's prediction function REPRESENT the interaction term?

Pure function-approximation test with every live-model confound removed (no
noise, no sliding window, no adaptive schedule, no endogenous selection): fit
the broker network to the *noiseless* target on a well-spread sample, training
long, and ask what the best achievable fit is — specifically how much of the
regime-gated `gain` component the network can reproduce.

Baselines on the SAME features:
  • linear readout  — closed-form least squares; recovers quality+core exactly,
    so its gain coefficient βg is the "features-alone, no gating" floor (~0).
  • broker NN       — the actual one-hidden-layer ReLU net at the model width
    (h=8d) and a much wider net (h=32d), to see whether extra capacity closes
    the gap (capacity-limited) or not (true representational gap).

Sweeps δ (δ=0 ⇒ no gain ⇒ everything ~perfect; larger δ ⇒ more target mass in
the discontinuous gain term). Headline metric is βg (gain recovery: 1=recovered,
0=ignored); gating failure also shows as opposite-signed per-regime bias.

Usage: julia --project scripts/diagnostics/stage1_representation.jl
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
        "\nδ=%.2f [%s] | variance share: quality=%.2f core=%.2f gain=%.2f  (gain = ceiling on what gating buys)\n",
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
    println("STAGE 1: Representational capacity (noiseless, long training, no window)")
    println("  d=$D, n_train=$N_TRAIN, n_test=$N_TEST, GD steps=$STEPS, lr=$LR")
    println("="^115)
    flush(stdout)

    # Main sweep at the model width (8d) and double (16d)
    for source in (:pool, :sphere)
        for δ in (0.0, 0.5)
            eval_cell(δ, source; widths=(8, 16))
        end
    end
    # δ trend on the realized-type support
    eval_cell(0.75, :pool; widths=(8, 16))

    # High-capacity ceiling probe: can a much wider net + more steps represent
    # the gain at all? (h=32d is ~10x the per-step cost, so run it once.)
    println("\n### High-capacity ceiling probe (δ=0.5, pool, h=32d, steps=2500)")
    flush(stdout)
    s = setup(0.5, :pool)
    nn = fit_broker_long(s.Ztr, s.ctr.target, 32 * D; steps=2500)
    print_row("  broker NN (h=$(32D)=32d)", evaluate(predict_nn_cols(nn, s.Zte), s.cte))

    println("\n" * "="^115)
    println("READING: βg≈0 with large opposite-signed per-regime bias ⇒ the function class cannot")
    println("represent the regime-gated gain (representational gap). βg climbing toward 1 as width")
    println("grows ⇒ capacity-limited but representable. δ=0 rows should be ~perfect for all models.")
    println("="^115)
    flush(stdout)
end

run_stage1()
