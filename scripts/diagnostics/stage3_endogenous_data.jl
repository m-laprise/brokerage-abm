"""
    stage3_endogenous_data.jl

STAGE 3 — Does the LIVE broker receive enough useful data to learn the
interaction endogenously?

Stages 1–2 use clean, well-spread pairs. The live broker instead sees only the
pairs that were actually matched through it — an endogenously *selected* sample
(offers it predicted to be good and that were accepted). Selection can starve
the interaction signal in two ways:
  1. Coverage — matched pairs may cluster in type space, collapsing the
     effective rank of the feature second moment, so A / the regime structure is
     under-identified.
  2. Regime balance — if matches concentrate on one side of x_i'B x_j > 0, the
     broker rarely sees the contrast that distinguishes the two gain regimes,
     so the gating term is unidentifiable from its data alone.

We run the real ABM, snapshot the broker's recorded history (party 1 types,
party 2 types, and output),
and (a) characterize that sample vs a uniform pool draw, and (b) train a broker
net on the endogenous history vs an equal-sized uniform sample under identical
optimization, comparing gain recovery (βg). If uniform ≫ endogenous, the deficit
is data/selection, not the learning function.

Usage: julia --project scripts/diagnostics/stage3_endogenous_data.jl
"""

include("broker_learning_common.jl")

using BrokerageABM: initialize_model, step_period!

const SEED = 42
const T = 200

"""Effective rank of a Gram/second-moment matrix via the participation ratio
(Σλ)² / Σλ²  — a continuous count of dominant directions."""
function effective_rank(M::Matrix{Float64})
    λ = eigvals(Symmetric(M))
    λ = max.(λ, 0.0)
    s = sum(λ)
    s <= 0 && return 0.0
    return s^2 / sum(abs2, λ)
end

"""Snapshot the broker's endogenous history as (party1, party2, q)."""
function broker_history_snapshot(state)
    b = state.broker
    n = b.history_count
    Xi = b.history_party1_types[:, 1:n]
    Xj = b.history_party2_types[:, 1:n]
    q = b.history_q[1:n]
    return Matrix(Xi), Matrix(Xj), Vector(q)
end

function run_stage3()
    println("="^110)
    println("STAGE 3: Endogenous data sufficiency (live ABM, T=$T, default params)")
    println("="^110)

    p = default_params(; seed=SEED, T=T)
    state = initialize_model(p)
    for _ in 1:T
        step_period!(state)
    end
    env = state.env

    Xi_e, Xj_e, q_e = broker_history_snapshot(state)
    n_e = size(Xi_e, 2)
    println("\nBroker endogenous history: $n_e observations after $T periods.")

    # Reference: uniform draw from the realized type pool, same env
    rng = StableRNG(SEED + 3)
    pool = [state.agents[i].type for i in 1:p.N]
    Xi_u, Xj_u = sample_pairs(env, pool, n_e, rng; source=:pool)

    # ── (a) Distributional comparison ──
    ce = decompose(env, Xi_e, Xj_e)
    cu = decompose(env, Xi_u, Xj_u)
    Ze = broker_features(Xi_e, Xj_e)
    Zu = broker_features(Xi_u, Xj_u)

    frac_hi_e = mean(ce.regime .== 1)
    frac_hi_u = mean(cu.regime .== 1)
    er_e = effective_rank(Ze * Ze' ./ n_e)
    er_u = effective_rank(Zu * Zu' ./ n_e)
    db = size(Ze, 1)

    @printf("\n%-34s %12s %12s\n", "metric", "endogenous", "uniform")
    @printf("%-34s %12.3f %12.3f\n", "P(high-gain regime, x'Bx>0)", frac_hi_e, frac_hi_u)
    @printf("%-34s %12.3f %12.3f\n", "effective rank of features (/$db)", er_e, er_u)
    @printf("%-34s %12.3f %12.3f\n", "var(core) on sample", var(ce.core), var(cu.core))
    @printf("%-34s %12.3f %12.3f\n", "var(gain) on sample", var(ce.gain), var(cu.gain))
    @printf("%-34s %12.3f %12.3f\n", "mean x'Bx", mean(ce.bxj), mean(cu.bxj))
    @printf("%-34s %12.3f %12.3f\n", "std  x'Bx", std(ce.bxj), std(cu.bxj))

    # ── (b) Train on each sample under identical optimization, score on a
    #        common uniform holdout ──
    Xi_te, Xj_te = sample_pairs(env, pool, 4000, StableRNG(SEED + 99); source=:pool)
    cte = decompose(env, Xi_te, Xj_te)
    Zte = broker_features(Xi_te, Xj_te)
    h = broker_hidden_width(p.d)

    # ── (a′) MOST DIRECT TEST: evaluate the ACTUAL live broker net ──
    # state.broker.nn is what the live model trained endogenously over T periods
    # (per-period adaptive-floor GD, sliding window, warm-started). Score it on
    # the common uniform holdout. This is the headline answer to "does the live
    # broker actually learn the gain?" — no retraining, no idealization.
    println("\n### Live broker net (state.broker.nn) on uniform holdout — the actual endogenous learner")
    live_pred = predict_nn_cols(state.broker.nn, Zte)
    m_live = evaluate(live_pred, cte)
    print_row("  LIVE broker.nn", m_live)
    go_live = gain_only_r2(live_pred, cte)
    @printf("      └ gain-only: βgain=%.3f, frac pred-var on gain dir=%.3f | cor(core,gain)=%+.3f\n",
            go_live.beta_gain, go_live.frac_pred_var_on_gain, core_gain_cor(cte))
    rows_live = residual_by_boundary(live_pred, cte; nbins=6)
    print_boundary_table("LIVE broker.nn", rows_live)

    function train_on(Z, y; steps=2000, lr=p.eta_lr)
        nn = init_neural_net(db, h, StableRNG(SEED + 5); b2_init=Q_OFFSET)
        grad = NNGradBuffers(nn)
        train_nn!(nn, grad, Matrix(Z), Vector(y), steps, lr)
        return evaluate(predict_nn_cols(nn, Zte), cte)
    end

    println("\n### Gain recovery: trained on endogenous vs uniform data, identical optimization")
    println("    (steps=2000, lr=$(p.eta_lr); scored on common uniform holdout)")
    print_row("  endogenous history (q w/ noise)", train_on(Ze, q_e))
    print_row("  uniform sample (q w/ noise)", train_on(Zu, cu.target .+ env.sigma_eps .* randn(StableRNG(SEED + 11), n_e)))
    print_row("  uniform sample (noiseless)", train_on(Zu, cu.target))

    println("\n" * "="^110)
    println("READING: a low high-gain fraction far from 0.5, or much lower var(gain)/effective rank")
    println("on the endogenous sample, means selection starves the gating signal. If uniform data")
    println("recovers βg≈1 but endogenous data does not at the SAME size+optimization, the binding")
    println("constraint is data selection — not the learning function or the schedule.")
    println("="^110)
end

run_stage3()
