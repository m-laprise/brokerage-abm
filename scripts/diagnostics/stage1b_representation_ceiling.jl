"""
    stage1b_representation_ceiling.jl

STAGE 1b — Does the broker function class REPRESENT the gain term, separated
cleanly from whether vanilla GD can FIND it?

Stage 1 trains with the model's own vanilla full-batch GD. That conflates two
questions: (i) representation — can the architecture + symmetric quadratic
feature map express the regime-gated gain term at all? and (ii) optimization —
can vanilla GD at a fixed lr sharpen the steep ramp needed to approximate the
discontinuity in a reasonable budget? A low βg under vanilla GD could be either.

This script pins down (i) by fitting the SAME network (same features, same loss,
same Enzyme gradient) with Adam — a strong optimizer — for a large budget, and
contrasts:
  • linear readout         — βg floor (~0): features-alone, no gating.
  • vanilla GD (model)     — βg trajectory at the live-ish step counts.
  • Adam ceiling           — βg at h=8d/16d/32d, long budget: the representational
                             ceiling of the class.

Decisive reads:
  • If Adam drives βg→1 while vanilla GD stalls low ⇒ representation is FINE; the
    live failure is the OPTIMIZER/SCHEDULE (Stage 2) or DATA (Stage 3).
  • If Adam ALSO plateaus well below 1, AND the residual concentrates in the
    smallest-|x'Bx| bin (regime boundary) ⇒ a genuine REPRESENTATIONAL GAP: a
    continuous ReLU net cannot match the discontinuous gain jump.

Usage: julia --project scripts/diagnostics/stage1b_representation_ceiling.jl
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

"""Fit with the model's own vanilla GD (the optimizer under test)."""
function fit_gd(Z, y, h; steps, lr=0.03, seed=1)
    nn = init_neural_net(size(Z, 1), h, StableRNG(seed); b2_init=Q_OFFSET)
    grad = NNGradBuffers(nn)
    train_nn!(nn, grad, Matrix(Z), Vector(y), steps, lr)
    return nn
end

function run_stage1b()
    println("="^118)
    println("STAGE 1b: Representation ceiling (Adam) vs vanilla-GD optimization — d=$D, δ=$DELTA, noiseless")
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

        # Floor: linear readout (no gating possible)
        w, μ = fit_linear_readout(s.Ztr, s.ctr.target)
        print_row("  linear readout (floor)", evaluate(predict_linear(s.Zte, w, μ), s.cte))

        # Model's own vanilla GD at increasing budgets, model width h=8d
        h0 = broker_hidden_width(D)
        for steps in (50, 200, 2000)
            nn = fit_gd(s.Ztr, s.ctr.target, h0; steps=steps, lr=0.03)
            print_row("  GD h=$(h0) steps=$steps", evaluate(predict_nn_cols(nn, s.Zte), s.cte))
        end

        # Adam representational ceiling at increasing width
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

        # Boundary-residual signature for the strongest Adam fit (h=32d)
        nn = fit_broker_adam(s.Ztr, s.ctr.target, 32 * D; steps=6000, lr=0.01)
        rows = residual_by_boundary(predict_nn_cols(nn, s.Zte), s.cte; nbins=6)
        print_boundary_table("[$source] Adam h=$(32D)", rows)
    end

    println("\n" * "="^118)
    println("READING:")
    println(" • Adam βg→1 but GD βg low  ⇒ representation OK, live failure is optimizer/schedule or data.")
    println(" • Adam βg plateaus <1 AND mae largest in smallest-|x'Bx| bin ⇒ representational gap (the")
    println("   discontinuous gain jump is not in a continuous ReLU net's class on these features).")
    println(" • If mae is flat across |x'Bx| bins, the shortfall is global underfitting, not the boundary.")
    println("="^118)
    flush(stdout)
end

run_stage1b()
