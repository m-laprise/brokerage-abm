"""
    stage1b_representation_ceiling.jl

Compare full-batch vanilla gradient descent and Adam on the same noiseless pair
features and loss. Fits across network widths and step budgets separate
optimizer sensitivity from network capacity. Boundary-binned residuals show
whether errors concentrate near the discontinuity in the gain term.

Usage: julia --project --threads=auto scripts/diagnostics/stage1b_representation_ceiling.jl
"""

include("broker_learning_common.jl")

const SEED = 42
const D = 8
const N_TRAIN = 3000
const N_TEST = 4000
const DELTA = 0.5

function setup(source)
    env, pool, _ = make_env(; d=D, delta=DELTA, sigma_eps=0.0, seed=SEED)
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

"""Fit a broker network with full-batch vanilla gradient descent."""
function fit_gd(Z, y, h; steps, lr=0.03, seed=1)
    nn = init_neural_net(size(Z, 1), h, StableRNG(seed); b2_init=Q_OFFSET)
    grad = NNGradBuffers(nn)
    train_nn!(nn, grad, Matrix(Z), Vector(y), steps, lr)
    return nn
end

function run_stage1b()
    println("="^118)
    println("STAGE 1b: Adam and vanilla-GD recovery, d=$D, δ=$DELTA, noiseless")
    println("  n_train=$N_TRAIN, n_test=$N_TEST")
    println("="^118)
    flush(stdout)

    for source in (:pool, :sphere)
        s = setup(source)
        vs = variance_shares(s.cte)
        cg = core_gain_cor(s.ctr)
        @printf(
            "\n[%s] variance share: quality=%.2f core=%.2f gain=%.2f | cor(core,gain)=%+.3f (≈0 ⇒ βg clean)\n",
            source, vs.frac_quality, vs.frac_core, vs.frac_gain, cg,
        )
        flush(stdout)

        # Linear comparison on the same pair features.
        w, μ = fit_linear_readout(s.Ztr, s.ctr.target)
        print_row("  linear readout", evaluate(predict_linear(s.Zte, w, μ), s.cte))

        # Vanilla gradient descent at increasing budgets and the model width.
        h0 = broker_hidden_width(D)
        for steps in (50, 200, 2000)
            nn = fit_gd(s.Ztr, s.ctr.target, h0; steps=steps, lr=0.03)
            print_row("  GD h=$(h0) steps=$steps", evaluate(predict_nn_cols(nn, s.Zte), s.cte))
        end

        # Full-batch Adam at increasing widths.
        for mult in (8, 16, 32)
            h = mult * D
            nn = fit_broker_adam(s.Ztr, s.ctr.target, h; steps=6000, lr=0.01)
            m = evaluate(predict_nn_cols(nn, s.Zte), s.cte)
            go = gain_only_r2(predict_nn_cols(nn, s.Zte), s.cte)
            print_row("  Adam h=$(h)=$(mult)d steps=6000", m)
            @printf("      └ gain-only: βgain=%.3f, frac pred-var on gain dir=%.3f\n",
                    go.beta_gain, go.frac_pred_var_on_gain)
            flush(stdout)
        end

        # Boundary-binned residuals for the widest Adam fit.
        nn = fit_broker_adam(s.Ztr, s.ctr.target, 32 * D; steps=6000, lr=0.01)
        rows = residual_by_boundary(predict_nn_cols(nn, s.Zte), s.cte; nbins=6)
        print_boundary_table("[$source] Adam h=$(32D)", rows)
    end

    println("\n" * "="^118)
    println("INTERPRETATION:")
    println(" • Higher Adam than GD recovery indicates optimizer sensitivity at the tested budgets.")
    println(" • Residual concentration near |x'Bx|=0 indicates error near the regime boundary.")
    println("="^118)
    flush(stdout)
end

run_stage1b()
