using Test
using BrokerageABM
using BrokerageABM: Agent, NNGradBuffers, NeuralNet, RidgeModel, agent_hidden_width
using BrokerageABM: compute_adaptive_steps
using BrokerageABM: fit_ridge!, init_neural_net, nn_loss, predict_nn!, predict_nn_batch!
using BrokerageABM: predict_agent, predict_broker!, predict_ridge
using BrokerageABM: period_training_window, windowed_index
using BrokerageABM: record_broker_history!
using BrokerageABM: train_agent_nn!, train_agent_predictor!, train_broker_nn!
using BrokerageABM: train_broker_predictor!, train_nn!, train_step!
using StableRNGs: StableRNG
using LinearAlgebra: I, dot, normalize
using Statistics: mean

@testset "Prediction Learning" begin
    @testset "predict_nn! produces finite output" begin
        rng = StableRNG(42)
        nn = init_neural_net(8, 16, rng)
        buf = zeros(16)
        z = randn(rng, 8)
        y = predict_nn!(nn, buf, z)
        @test isfinite(y)
    end

    @testset "untrained network has a constant Q prior" begin
        rng = StableRNG(41)
        nn = init_neural_net(8, 16, rng)
        buf = zeros(16)
        X = randn(rng, 8, 12)

        @test all(iszero, nn.w2)
        @test [predict_nn!(nn, buf, X[:, j]) for j in axes(X, 2)] == fill(BrokerageABM.Q_OFFSET, size(X, 2))
    end

    @testset "period-zero boundary leaves only completed simulation periods" begin
        # Seed history has columns 1:3. Periods 1 and 2 add columns 4:5 and 6:7.
        start_idx, window, count = period_training_window([3, 5, 7], 7, 2, 20)
        @test (start_idx, window, count) == (4, 4, 4)

        # Before W_p completed periods exist, period 0 remains in the window.
        @test period_training_window([3, 5], 5, 2, 20) == (1, 5, 5)

        # The observation cap changes the training sample, not the window boundary.
        @test period_training_window([3, 5, 7], 7, 2, 2) == (4, 4, 2)

        # Capped sampling always includes the newest observation.
        @test [windowed_index(4, 4, 2, k) for k in 0:1] == [4, 7]
        @test [windowed_index(4, 4, 3, k) for k in 0:2] == [4, 5, 7]
        @test windowed_index(4, 4, 1, 0) == 7
        @test [windowed_index(4, 4, 4, k) for k in 0:3] == collect(4:7)
    end

    @testset "predict_nn! is deterministic" begin
        rng = StableRNG(42)
        nn = init_neural_net(8, 16, rng)
        buf = zeros(16)
        z = randn(StableRNG(1), 8)
        y1 = predict_nn!(nn, buf, z)
        y2 = predict_nn!(nn, buf, z)
        @test y1 == y2
    end

    @testset "predict_nn_batch! matches scalar predict_nn!" begin
        rng = StableRNG(202)
        nn = init_neural_net(8, 16, rng)
        n = 12
        cap = 16
        Z = randn(rng, 8, cap)
        H = zeros(16, cap)
        Y = zeros(cap)
        predict_nn_batch!(nn, H, Y, Z, n)

        buf = zeros(16)
        y_scalar = [predict_nn!(nn, buf, Z[:, j]) for j in 1:n]
        @test all(isapprox.(Y[1:n], y_scalar; atol=1e-12))
    end

    @testset "nn_loss is finite and positive" begin
        rng = StableRNG(42)
        nn = init_neural_net(8, 16, rng)
        X = randn(rng, 8, 10)
        q = randn(rng, 10)
        loss = nn_loss(nn.W1, nn.b1, nn.w2, Ref(nn.b2), X, q)
        @test isfinite(loss)
        @test loss > 0.0
    end

    @testset "Training reduces loss" begin
        rng = StableRNG(42)
        nn = init_neural_net(8, 16, rng)
        grad = NNGradBuffers(nn)
        X = randn(StableRNG(1), 8, 20)
        q = randn(StableRNG(2), 20)

        loss_before = nn_loss(nn.W1, nn.b1, nn.w2, Ref(nn.b2), X, q)
        train_nn!(nn, grad, X, q, 50, 0.01)
        loss_after = nn_loss(nn.W1, nn.b1, nn.w2, Ref(nn.b2), X, q)

        @test loss_after < loss_before
    end

    @testset "train_step! matches one-step train_nn!" begin
        rng = StableRNG(303)
        nn0 = init_neural_net(8, 16, rng)
        nn_step = NeuralNet(copy(nn0.W1), copy(nn0.b1), copy(nn0.w2), nn0.b2)
        nn_loop = NeuralNet(copy(nn0.W1), copy(nn0.b1), copy(nn0.w2), nn0.b2)
        grad_step = NNGradBuffers(nn_step)
        grad_loop = NNGradBuffers(nn_loop)
        X = randn(rng, 8, 20)
        q = randn(rng, 20)
        lr = 0.01

        train_step!(nn_step, grad_step, X, q, lr)
        train_nn!(nn_loop, grad_loop, X, q, 1, lr)

        @test nn_step.W1 == nn_loop.W1
        @test nn_step.b1 == nn_loop.b1
        @test nn_step.w2 == nn_loop.w2
        @test nn_step.b2 == nn_loop.b2
    end

    @testset "train_nn_prefix! matches copied train_nn!" begin
        rng = StableRNG(304)
        nn0 = init_neural_net(8, 16, rng)
        nn_prefix = NeuralNet(copy(nn0.W1), copy(nn0.b1), copy(nn0.w2), nn0.b2)
        nn_copied = NeuralNet(copy(nn0.W1), copy(nn0.b1), copy(nn0.w2), nn0.b2)
        grad_prefix = NNGradBuffers(nn_prefix)
        grad_copied = NNGradBuffers(nn_copied)
        X_full = randn(rng, 8, 32)
        q_full = randn(rng, 32)
        n_active = 20
        lr = 0.01

        BrokerageABM.train_nn_prefix!(
            nn_prefix, grad_prefix, X_full, q_full, n_active, 5, lr
        )
        train_nn!(
            nn_copied,
            grad_copied,
            Matrix(X_full[:, 1:n_active]),
            Vector(q_full[1:n_active]),
            5,
            lr,
        )

        @test nn_prefix.W1 == nn_copied.W1
        @test nn_prefix.b1 == nn_copied.b1
        @test nn_prefix.w2 == nn_copied.w2
        @test nn_prefix.b2 == nn_copied.b2
    end

    @testset "train_nn_prefix! reuses parameter buffers after warmup" begin
        rng = StableRNG(305)
        nn = init_neural_net(8, 16, rng)
        grad = NNGradBuffers(nn)
        X_full = randn(rng, 8, 32)
        q_full = randn(rng, 32)
        n_active = 20

        BrokerageABM.train_nn_prefix!(nn, grad, X_full, q_full, n_active, 1, 0.01)
        theta = grad.theta
        dtheta = grad.dtheta
        BrokerageABM.train_nn_prefix!(nn, grad, X_full, q_full, n_active, 1, 0.01)
        @test grad.theta === theta
        @test grad.dtheta === dtheta
    end

    @testset "DI/Enzyme gradient populates finite buffers" begin
        rng = StableRNG(404)
        nn = init_neural_net(8, 16, rng)
        grad = NNGradBuffers(nn)
        X = randn(rng, 8, 20)
        q = randn(rng, 20)

        train_step!(nn, grad, X, q, 0.0)

        @test length(grad.theta) == 16 * 8 + 2 * 16 + 1
        @test length(grad.dtheta) == length(grad.theta)
        @test all(isfinite, grad.dW1)
        @test all(isfinite, grad.db1)
        @test all(isfinite, grad.dw2)
        @test isfinite(grad.db2[])
        @test any(!iszero, grad.dtheta)
    end

    @testset "NN can learn a linear function" begin
        rng = StableRNG(42)
        d = 8
        nn = init_neural_net(d, 16, rng)
        grad = NNGradBuffers(nn)

        n = 100
        X = randn(StableRNG(1), d, n)
        q = [2.0 * X[1, j] + 0.5 * X[2, j] + 1.0 for j in 1:n]

        train_nn!(nn, grad, X, Vector{Float64}(q), 200, 0.01)

        X_test = randn(StableRNG(99), d, 20)
        q_test = [2.0 * X_test[1, j] + 0.5 * X_test[2, j] + 1.0 for j in 1:20]
        buf = zeros(16)
        preds = [predict_nn!(nn, buf, X_test[:, j]) for j in 1:20]
        mse = sum((preds .- q_test) .^ 2) / 20
        @test mse < 0.5
    end

    @testset "Ridge matches the centered closed-form estimator" begin
        rng = StableRNG(501)
        X = randn(rng, 3, 30)
        y = 1.7 .+ vec([0.8, -0.4, 0.2]' * X) .+ 0.1 .* randn(rng, 30)
        lambda = 0.07
        model = RidgeModel(3, BrokerageABM.Q_OFFSET)

        fit_ridge!(model, X, y, length(y), lambda)

        xbar = vec(mean(X; dims=2))
        ybar = mean(y)
        Xc = X .- xbar
        expected_beta = (Xc * Xc' / length(y) + lambda * I) \ (Xc * (y .- ybar) / length(y))
        expected_intercept = ybar - expected_beta' * xbar
        @test model.coefficients ≈ expected_beta atol=1e-12
        @test model.intercept ≈ expected_intercept atol=1e-12
        @test model.target_mean ≈ ybar atol=1e-12
        @test predict_ridge(model, X[:, 1]) ≈ expected_intercept + expected_beta' * X[:, 1]
    end

    @testset "Ridge leaves the intercept unpenalized" begin
        X = [1.0 2.0 3.0; -2.0 0.0 2.0]
        y = fill(4.25, 3)
        model = RidgeModel(2, BrokerageABM.Q_OFFSET)
        fit_ridge!(model, X, y, 3, 100.0)
        @test model.intercept ≈ 4.25 atol=1e-12
        @test all(isapprox.(model.coefficients, 0.0; atol=1e-12))
        @test_throws AssertionError fit_ridge!(model, randn(3, 3), y, 3, 0.1)
    end

    @testset "Ridge agent uses the NN training window and raw features" begin
        p = default_params(
            N=20,
            learning_model=:ridge,
            ridge_lambda_agent=0.03,
            train_window_periods=2,
            train_max_obs=3,
        )
        state = initialize_model(p)
        agent = state.agents[1]
        agent.history_X = reshape(collect(1.0:56.0), p.d, 7)
        agent.history_q = collect(1.0:7.0)
        agent.history_count = 7
        agent.obs_period_marks = [3, 5, 7]
        agent.n_new_obs = 2
        reference = RidgeModel(p.d, BrokerageABM.Q_OFFSET)
        selected = [4, 5, 7]
        fit_ridge!(
            reference,
            Matrix(agent.history_X[:, selected]),
            Vector(agent.history_q[selected]),
            length(selected),
            p.ridge_lambda_agent,
        )

        train_agent_predictor!(agent, p)

        @test agent.train_X[:, 1:3] == agent.history_X[:, selected]
        @test agent.train_q[1:3] == agent.history_q[selected]
        @test (agent.ridge::RidgeModel).coefficients ≈ reference.coefficients
        @test (agent.ridge::RidgeModel).intercept ≈ reference.intercept
        @test agent.n_new_obs == 0
        @test predict_agent(agent, agent.type, p) == predict_ridge(agent.ridge, agent.type)
    end

    @testset "Ridge broker uses its own penalty" begin
        p = default_params(
            N=10,
            d=2,
            s=2,
            k=2,
            learning_model=:ridge,
            ridge_lambda_agent=0.7,
            ridge_lambda_broker=0.04,
            ridge_broker_variant=:additive,
        )
        state = initialize_model(p)
        broker = state.broker
        broker.history_count = 3
        broker.obs_period_marks = [3]
        broker.n_new_obs = 3
        broker.history_party1_types[:, 1:3] .= [1.0 2.0 3.0; 0.0 1.0 0.0]
        broker.history_party2_types[:, 1:3] .= [0.5 1.0 0.0; 1.0 0.0 2.0]
        broker.history_q[1:3] .= [0.4, 1.1, 0.8]

        reference = RidgeModel(p.d, BrokerageABM.Q_OFFSET)
        inputs = broker.history_party1_types[:, 1:3] .+ broker.history_party2_types[:, 1:3]
        fit_ridge!(reference, inputs, broker.history_q, 3, p.ridge_lambda_broker)

        train_broker_predictor!(broker, state.agents, p, StableRNG(501))

        @test broker.train_X[:, 1:3] == inputs
        @test (broker.ridge::RidgeModel).coefficients ≈ reference.coefficients
        @test (broker.ridge::RidgeModel).intercept ≈ reference.intercept
        @test broker.n_new_obs == 0
    end

    @testset "Broker Ridge variants implement their stated feature maps" begin
        xi = [1.0, 2.0]
        xj = [3.0, 5.0]
        feature_buf = zeros(BrokerageABM.broker_pair_feature_dim(2))

        for (variant, expected) in (
            (:additive, 0.5 + dot([2.0, -1.0], xi + xj)),
            (:single_principal, 1.0 + dot([2.0, -1.0], xi + xj)),
        )
            p = default_params(
                N=10, d=2, s=2, k=2, learning_model=:ridge, ridge_broker_variant=variant
            )
            state = initialize_model(p)
            ridge = state.broker.ridge::RidgeModel
            ridge.coefficients .= [2.0, -1.0]
            ridge.intercept = 0.5
            ridge.target_mean = 0.0
            @test predict_broker!(state.broker, feature_buf, xi, xj, p) ≈ expected
            @test predict_broker!(state.broker, feature_buf, xj, xi, p) ≈ expected
        end
    end

    @testset "Size-matched broker uses the median agent sample size" begin
        p = default_params(
            N=10,
            d=2,
            s=2,
            k=2,
            learning_model=:ridge,
            ridge_broker_variant=:size_matched,
            train_max_obs=20,
        )
        state = initialize_model(p)
        for (idx, count) in enumerate(1:10)
            agent = state.agents[idx]
            agent.history_count = count
            agent.obs_period_marks = [count]
        end
        broker = state.broker
        broker.history_count = 10
        broker.obs_period_marks = [10]
        @inbounds for obs in 1:10
            broker.history_party1_types[:, obs] .= state.agents[obs].type
            broker.history_party2_types[:, obs] .= state.agents[mod1(obs + 1, 10)].type
            broker.history_q[obs] = obs / 10
        end
        broker.n_new_obs = 1
        train_broker_predictor!(broker, state.agents, p, StableRNG(502))

        @test length(unique(broker.sample_indices[1:6])) == 6
        @test all(in(1:10), broker.sample_indices[1:6])
        @test broker.n_new_obs == 0
    end

    @testset "Single-principal fitting discards the unretained endpoint" begin
        p = default_params(
            N=10,
            d=2,
            s=2,
            k=2,
            learning_model=:ridge,
            ridge_broker_variant=:single_principal,
        )
        state = initialize_model(p)
        broker = state.broker
        broker.history_count = 2
        broker.obs_period_marks = [2]
        broker.n_new_obs = 1
        broker.history_party1_types[:, 1:2] .= [1.0 2.0; 3.0 4.0]
        broker.history_party2_types[:, 1:2] .= [10.0 20.0; 30.0 40.0]
        broker.history_retained_party[1:2] .= [UInt8(1), UInt8(2)]
        broker.history_q[1:2] .= [0.5, 1.5]

        train_broker_predictor!(broker, state.agents, p, StableRNG(503))

        @test broker.train_X[:, 1] == [1.0, 3.0]
        @test broker.train_X[:, 2] == [20.0, 40.0]
        @test broker.train_q[1:2] == [0.5, 1.5]
    end

    @testset "Ridge simulation is deterministic and verifies invariants" begin
        p = default_params(N=20, T=2, seed=504, learning_model=:ridge)
        state1, df1 = run_simulation(p; verify=true)
        state2, df2 = run_simulation(p; verify=true)
        @test isequal(df1, df2)
        @test state1.broker.history_count == state2.broker.history_count
    end

    @testset "Adaptive step schedule" begin
        @test compute_adaptive_steps(100, 5, 5) == 100     # all new
        @test compute_adaptive_steps(100, 3, 5) == 60      # 3 of 5 new (above floor)
        @test compute_adaptive_steps(100, 1, 50) == 50     # floor at ADAPTIVE_FLOOR=50
        @test compute_adaptive_steps(100, 1, 200) == 50    # floor
        @test compute_adaptive_steps(100, 0, 100) >= 50    # floor
        @test compute_adaptive_steps(100, 1, 0) == 100     # empty history
    end

    @testset "train_agent_nn! resets n_new_obs" begin
        rng = StableRNG(42)
        p = default_params(N=20)
        h_agent = agent_hidden_width(p)
        nn = init_neural_net(p.d, h_agent, rng)
        agent = Agent(
            id=1,
            type=normalize(randn(rng, p.d)),
            history_X=randn(rng, p.d, 16),
            history_q=randn(rng, 16),
            history_count=5,
            n_new_obs=5,
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
            partner_sum=zeros(20),
            partner_count=zeros(Int, 20),
        )
        train_agent_nn!(agent, p)
        @test agent.n_new_obs == 0
    end

    @testset "train_agent_nn! matches contiguous copied training" begin
        rng = StableRNG(314)
        p = default_params(N=20, E_init=7)
        history_X = randn(rng, p.d, 24)
        history_q = randn(rng, 24)
        h_agent = agent_hidden_width(p)
        nn0 = init_neural_net(p.d, h_agent, rng)
        nn_agent = NeuralNet(copy(nn0.W1), copy(nn0.b1), copy(nn0.w2), nn0.b2)
        nn_ref = NeuralNet(copy(nn0.W1), copy(nn0.b1), copy(nn0.w2), nn0.b2)
        grad_agent = NNGradBuffers(nn_agent)
        grad_ref = NNGradBuffers(nn_ref)
        n_new = 6
        agent = Agent(
            id=1,
            type=normalize(randn(rng, p.d)),
            history_X=copy(history_X),
            history_q=copy(history_q),
            history_count=24,
            n_new_obs=n_new,
            nn=nn_agent,
            nn_grad=grad_agent,
            predict_buf=zeros(h_agent),
            partner_sum=zeros(20),
            partner_count=zeros(Int, 20),
            train_X=Matrix{Float64}(undef, p.d, 8),
            train_q=Vector{Float64}(undef, 8),
        )

        n_steps = compute_adaptive_steps(
            p.E_init, n_new, agent.history_count; min_steps=p.train_steps
        )
        train_agent_nn!(agent, p)
        # Live agent training uses Adam; with no period marks the window spans the
        # full history, so the reference runs the same Adam over all observations.
        nref = agent.history_count
        BrokerageABM.train_nn_prefix_adam!(
            nn_ref,
            grad_ref,
            Matrix(history_X[:, 1:nref]),
            Vector(history_q[1:nref]),
            nref,
            n_steps,
            p.eta_lr,
        )

        @test agent.nn.W1 == nn_ref.W1
        @test agent.nn.b1 == nn_ref.b1
        @test agent.nn.w2 == nn_ref.w2
        @test agent.nn.b2 == nn_ref.b2
    end

    @testset "train_agent_nn! is allocation-light after warmup" begin
        rng = StableRNG(315)
        p = default_params(N=20, E_init=1)
        h_agent = agent_hidden_width(p)
        nn = init_neural_net(p.d, h_agent, rng)
        agent = Agent(
            id=1,
            type=normalize(randn(rng, p.d)),
            history_X=randn(rng, p.d, 24),
            history_q=randn(rng, 24),
            history_count=24,
            n_new_obs=1,
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
            partner_sum=zeros(20),
            partner_count=zeros(Int, 20),
            train_X=Matrix{Float64}(undef, p.d, 4),
            train_q=Vector{Float64}(undef, 4),
        )

        train_agent_nn!(agent, p)
        agent.n_new_obs = 1
        train_X = agent.train_X
        train_q = agent.train_q
        theta = agent.nn_grad.theta
        dtheta = agent.nn_grad.dtheta
        train_agent_nn!(agent, p)
        @test agent.train_X === train_X
        @test agent.train_q === train_q
        @test agent.nn_grad.theta === theta
        @test agent.nn_grad.dtheta === dtheta
    end

    @testset "train_agent_nn! with empty history is no-op" begin
        rng = StableRNG(42)
        p = default_params(N=20)
        h_agent = agent_hidden_width(p)
        nn = init_neural_net(p.d, h_agent, rng)
        w1_before = copy(nn.W1)
        agent = Agent(
            id=1,
            type=normalize(randn(rng, p.d)),
            history_X=Matrix{Float64}(undef, p.d, 16),
            history_q=Vector{Float64}(undef, 16),
            history_count=0,
            n_new_obs=0,
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
            partner_sum=zeros(20),
            partner_count=zeros(Int, 20),
        )
        train_agent_nn!(agent, p)
        @test agent.nn.W1 == w1_before
    end

    @testset "train_broker_nn! resets n_new_obs and updates weights" begin
        p = default_params(N=30, seed=42)
        state = initialize_model(p)
        broker = state.broker
        w1_before = copy(broker.nn.W1)

        record_broker_history!(broker, state.agents[1].type, state.agents[2].type, 1.2)
        record_broker_history!(broker, state.agents[2].type, state.agents[3].type, 1.4)
        @test broker.n_new_obs == 2

        train_broker_nn!(broker, p)
        @test broker.n_new_obs == 0
        @test broker.nn.W1 != w1_before
    end

    # Weight decay removed: the NN has no explicit L2 regularization. With MSE
    # targets of zero the weights still trend toward zero via pure gradient
    # descent, but that's a property of the optimization target, not of a
    # separate decay term. See scan results showing λ had no measurable effect
    # at tested scales; decay removed for simplicity.
end
