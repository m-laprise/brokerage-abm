"""
    stage3_endogenous_data.jl

Compare the broker's endogenously selected match history with an equal-sized
uniform pair sample. The script evaluates the production-trained broker on a
uniform holdout, describes coverage and regime balance, and compares fresh
vanilla-GD fits while holding sample size and optimization fixed.

Usage: julia --project --threads=auto scripts/diagnostics/stage3_endogenous_data.jl
"""

include("broker_learning_common.jl")

using BrokerageABM: initialize_model, step_period!

const SEED = 42
const T = 200

"""Effective rank of a second-moment matrix, `(Σλ)² / Σλ²`."""
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
    println("STAGE 3: Endogenous training-data comparison (T=$T, baseline parameters)")
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

    # Uniform reference sample from the realized type pool and matching environment.
    rng = StableRNG(SEED + 3)
    pool = [state.agents[i].type for i in 1:p.N]
    Xi_u, Xj_u = sample_pairs(env, pool, n_e, rng; source=:pool)

    # ── Distributional comparison ──
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

    # ── Fit each sample identically and score on a common uniform holdout ──
    Xi_te, Xj_te = sample_pairs(env, pool, 4000, StableRNG(SEED + 99); source=:pool)
    cte = decompose(env, Xi_te, Xj_te)
    Zte = broker_features(Xi_te, Xj_te)
    h = broker_hidden_width(p.d)

    # Evaluate the production-trained broker, which uses persistent Adam state
    # and the period-based training window, on the common uniform holdout.
    println("\n### Production-trained broker on the uniform holdout")
    live_pred = predict_nn_cols(state.broker.nn, Zte)
    m_live = evaluate(live_pred, cte)
    print_row("  production broker", m_live)
    go_live = gain_only_r2(live_pred, cte)
    @printf("      └ gain-only: βgain=%.3f, frac pred-var on gain dir=%.3f | cor(core,gain)=%+.3f\n",
            go_live.beta_gain, go_live.frac_pred_var_on_gain, core_gain_cor(cte))
    rows_live = residual_by_boundary(live_pred, cte; nbins=6)
    print_boundary_table("production broker", rows_live)

    function train_on(Z, y; steps=2000, lr=p.eta_lr_broker)
        nn = init_neural_net(db, h, StableRNG(SEED + 5); b2_init=Q_OFFSET)
        grad = NNGradBuffers(nn)
        train_nn!(nn, grad, Matrix(Z), Vector(y), steps, lr)
        return evaluate(predict_nn_cols(nn, Zte), cte)
    end

    println("\n### Gain recovery: trained on endogenous vs uniform data, identical optimization")
    println("    (steps=2000, lr=$(p.eta_lr_broker); scored on common uniform holdout)")
    print_row("  endogenous history (q w/ noise)", train_on(Ze, q_e))
    print_row("  uniform sample (q w/ noise)", train_on(Zu, cu.target .+ env.sigma_eps .* randn(StableRNG(SEED + 11), n_e)))
    print_row("  uniform sample (noiseless)", train_on(Zu, cu.target))

    println("\n" * "="^110)
    println("INTERPRETATION: differences between the fresh fits isolate the effect of the")
    println("training sample under the tested optimizer. The production-trained fit provides")
    println("the corresponding result under the simulation training procedure.")
    println("="^110)
end

run_stage3()
