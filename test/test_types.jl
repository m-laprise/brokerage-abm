using Test
using BrokerageABM
using BrokerageABM: ActiveMatch, Agent, CachedNetworkMeasures, CalibrationConstants
using BrokerageABM: CurveGeometry, MatchingEnv, NNGradBuffers, PeriodAccumulators
using BrokerageABM: MATCH_NOISE_SD_BASE, Q_OFFSET, effective_history_size
using BrokerageABM: agent_hidden_width, broker_hidden_width, broker_pair_feature_dim
using BrokerageABM: has_current_match, init_neural_net, partner_mean
using BrokerageABM: record_agent_history!, record_broker_history!
using BrokerageABM: reset_accumulators!
using BrokerageABM: update_partner_mean!
using StableRNGs: StableRNG
using LinearAlgebra: norm

@testset "Types and Parameters" begin
    @testset "default_params construction" begin
        p = default_params()
        @test p isa ModelParams
        @test p.N == 1000
        @test p.d == 8
        @test p.K == 5
        @test p.p_demand == 0.50
        @test Q_OFFSET == 2.840698029863751
        @test p.sigma_eps == MATCH_NOISE_SD_BASE == 0.28406980298637513
        @test p.omega == 0.20
        @test p.search_cost_rate == 0.05
        @test p.roster_frac == 0.20
        @test p.n_strangers == 10
        @test p.roster_churn == 0.02
        @test p.learning_model == :nn
        @test p.ridge_lambda_agent == 0.001
        @test p.ridge_lambda_broker == 0.001
        @test p.ridge_broker_variant == :pair
        @test p.network_measure_interval == 20
        @test p.T == 500
    end

    @testset "public export surface keeps internals explicit" begin
        exported = Set(names(BrokerageABM))
        expected_exports = Set([
            :ModelParams,
            :ModelState,
            :PredictionQuality,
            :default_params,
            :validate_params,
            :initialize_model,
            :step_period!,
            :collect_period_metrics,
            :run_simulation,
            :verify_invariants,
            :diagnostic_summary,
            :compute_prediction_quality,
            :compute_betweenness,
            :compute_burt_constraint,
            :compute_effective_size,
        ])

        @test all(name -> name in exported, expected_exports)
        @test !(:Agent in exported)
        @test !(:NeuralNet in exported)
        @test !(:run_offer_market! in exported)
    end

    @testset "default_params with overrides" begin
        p = default_params(; seed=99, N=200, K=10, delta=0.75, roster_frac=0.40)
        @test p.seed == 99
        @test p.N == 200
        @test p.K == 10
        @test p.delta == 0.75
        @test p.roster_frac == 0.40
    end

    @testset "default_params rejects unknown kwargs" begin
        @test_throws ErrorException default_params(; bogus_param=42)
        @test_throws ErrorException default_params(; tau=1)
        @test_throws ErrorException default_params(; T_burn=30)
    end

    @testset "validate_params catches invalid values" begin
        @test_throws AssertionError default_params(d=1)
        @test_throws AssertionError default_params(N=5)
        @test_throws AssertionError default_params(rho=-0.1)
        @test_throws AssertionError default_params(rho=1.5)
        @test_throws AssertionError default_params(K=0)
        @test_throws AssertionError default_params(search_cost_rate=-0.01)
        @test_throws AssertionError default_params(search_cost_rate=1.01)
        @test_throws AssertionError default_params(eta=-0.1)
        @test_throws AssertionError default_params(roster_frac=-0.1)
        @test_throws AssertionError default_params(roster_frac=1.1)
        @test_throws AssertionError default_params(roster_churn=-0.1)
        @test_throws AssertionError default_params(roster_churn=1.1)
        @test_throws AssertionError default_params(learning_model=:linear)
        @test_throws AssertionError default_params(ridge_lambda_agent=0.0)
        @test_throws AssertionError default_params(ridge_lambda_broker=0.0)
        @test_throws AssertionError default_params(ridge_broker_variant=:unknown)
    end

    @testset "NeuralNet and NNGradBuffers" begin
        rng = StableRNG(42)
        nn = init_neural_net(8, 16, rng)
        @test size(nn.W1) == (16, 8)
        @test length(nn.b1) == 16
        @test length(nn.w2) == 16
        @test all(iszero, nn.b1)
        @test all(iszero, nn.w2)
        @test nn.b2 == Q_OFFSET

        grad = NNGradBuffers(nn)
        @test size(grad.dW1) == (16, 8)
        @test length(grad.db1) == 16
        @test length(grad.dw2) == 16
    end

    @testset "NeuralNet parameter counts" begin
        rng = StableRNG(42)
        p = default_params()
        h_agent = agent_hidden_width(p)
        h_broker = broker_hidden_width(p)
        d_broker = broker_pair_feature_dim(p.d)
        nn_a = init_neural_net(p.d, h_agent, rng)
        n_params_a = length(nn_a.W1) + length(nn_a.b1) + length(nn_a.w2) + 1
        @test n_params_a == h_agent * (p.d + 2) + 1

        nn_b = init_neural_net(d_broker, h_broker, rng)
        n_params_b = length(nn_b.W1) + length(nn_b.b1) + length(nn_b.w2) + 1
        @test n_params_b == h_broker * (d_broker + 2) + 1
    end

    @testset "ActiveMatch construction" begin
        am = ActiveMatch(5, :self)
        @test am.partner_id == 5
        @test am.channel == :self
    end

    @testset "reset_accumulators!" begin
        accum = PeriodAccumulators()
        accum.n_self_matches = 10
        accum.n_broker_matches = 5
        push!(accum.q_self, 1.0, 2.0)
        push!(accum.q_broker, 3.0, 4.0)
        accum.n_demanders = 7
        accum.n_outsourced = 2
        accum.outsourced_slots = 9
        accum.roster_size = 42
        accum.broker_access_size = 45

        reset_accumulators!(accum)

        @test accum.n_self_matches == 0
        @test accum.n_broker_matches == 0
        @test isempty(accum.q_self)
        @test isempty(accum.q_broker)
        @test accum.n_demanders == 0
        @test accum.n_outsourced == 0
        @test accum.outsourced_slots == 0
        @test accum.roster_size == 0
        @test accum.broker_access_size == 0
    end

    @testset "Agent history recording and growth" begin
        rng = StableRNG(42)
        p = default_params(N=20)
        h_agent = agent_hidden_width(p)
        nn = init_neural_net(p.d, h_agent, rng)
        agent = Agent(
            id=1,
            type=randn(rng, p.d),
            history_X=Matrix{Float64}(undef, p.d, 4),  # small initial capacity
            history_q=Vector{Float64}(undef, 4),
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
            partner_sum=zeros(20),
            partner_count=zeros(Int, 20),
        )

        # Record 4 observations (fills initial capacity)
        for i in 1:4
            record_agent_history!(agent, randn(rng, p.d), Float64(i))
        end
        @test agent.history_count == 4
        @test agent.n_new_obs == 4

        # Record 5th observation (triggers doubling growth)
        record_agent_history!(agent, randn(rng, p.d), 5.0)
        @test agent.history_count == 5
        @test size(agent.history_X, 2) >= 8  # doubled from 4
        @test agent.history_q[5] == 5.0
    end

    @testset "effective_history_size for agent and broker" begin
        state = initialize_model(default_params(N=20, seed=17))
        state.agents[1].history_count = 7
        state.broker.history_count = 11
        @test effective_history_size(state.agents[1]) == 7
        @test effective_history_size(state.broker) == 11
    end

    @testset "record_broker_history! records and grows buffers" begin
        rng = StableRNG(123)
        p = default_params(N=20, seed=123)
        state = initialize_model(p)
        broker = state.broker
        d = p.d

        broker.history_party1_types = Matrix{Float64}(undef, d, 2)
        broker.history_party2_types = Matrix{Float64}(undef, d, 2)
        broker.history_q = Vector{Float64}(undef, 2)
        broker.train_X = Matrix{Float64}(undef, broker_pair_feature_dim(d), 4)
        broker.train_q = Vector{Float64}(undef, 4)
        broker.history_count = 0
        broker.n_new_obs = 0

        party1_type1 = randn(rng, d)
        party2_type1 = randn(rng, d)
        party1_type2 = randn(rng, d)
        party2_type2 = randn(rng, d)
        party1_type3 = randn(rng, d)
        party2_type3 = randn(rng, d)
        record_broker_history!(broker, party1_type1, party2_type1, 1.0)
        record_broker_history!(broker, party1_type2, party2_type2, 2.0)
        record_broker_history!(broker, party1_type3, party2_type3, 3.0)  # triggers growth

        @test broker.history_count == 3
        @test broker.n_new_obs == 3
        @test size(broker.history_party1_types, 2) >= 3
        @test size(broker.history_party2_types, 2) >= 3
        @test size(broker.train_X, 1) == broker_pair_feature_dim(d)
        @test size(broker.train_X, 2) >= 3
        @test broker.history_party1_types[:, 3] == party1_type3
        @test broker.history_party2_types[:, 3] == party2_type3
        @test broker.history_q[3] == 3.0
    end

    @testset "single-principal endpoint is retained once per observation" begin
        rng = StableRNG(124)
        p = default_params(
            N=20, seed=124, learning_model=:ridge, ridge_broker_variant=:single_principal
        )
        broker = initialize_model(p).broker
        n_before = broker.history_count
        party1 = fill(1.0, p.d)
        party2 = fill(2.0, p.d)
        record_broker_history!(broker, party1, party2, 3.0; rng=rng)

        retained = broker.history_retained_party[n_before + 1]
        @test retained in (UInt8(1), UInt8(2))
        @test broker.history_retained_party[n_before + 1] == retained
        count_before_failure = broker.history_count
        @test_throws ErrorException record_broker_history!(broker, party1, party2, 4.0)
        @test broker.history_count == count_before_failure
    end

    @testset "Partner mean tracking" begin
        rng = StableRNG(42)
        p = default_params(N=10)
        h_agent = agent_hidden_width(p)
        nn = init_neural_net(p.d, h_agent, rng)
        agent = Agent(
            id=1,
            type=randn(rng, p.d),
            history_X=Matrix{Float64}(undef, p.d, 16),
            history_q=Vector{Float64}(undef, 16),
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
            partner_sum=zeros(10),
            partner_count=zeros(Int, 10),
        )

        # No history with partner 3
        @test isnan(partner_mean(agent, 3))

        # Add two observations with partner 3
        update_partner_mean!(agent, 3, 2.0)
        update_partner_mean!(agent, 3, 4.0)
        @test partner_mean(agent, 3) ≈ 3.0
        @test agent.partner_count[3] == 2
    end

    @testset "Current matches are tracked independently of K" begin
        rng = StableRNG(42)
        p = default_params(N=10, K=3)
        h_agent = agent_hidden_width(p)
        nn = init_neural_net(p.d, h_agent, rng)
        agent = Agent(
            id=1,
            type=randn(rng, p.d),
            history_X=Matrix{Float64}(undef, p.d, 16),
            history_q=Vector{Float64}(undef, 16),
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(h_agent),
            partner_sum=zeros(10),
            partner_count=zeros(Int, 10),
        )

        @test !has_current_match(agent, 2)
        push!(agent.active_matches, ActiveMatch(2, :self))
        @test has_current_match(agent, 2)
        @test !has_current_match(agent, 3)
        push!(agent.active_matches, ActiveMatch(3, :broker))
        push!(agent.active_matches, ActiveMatch(4, :broker))
        @test length(agent.active_matches) == 3
    end

    @testset "Current-match workspace index" begin
        state = initialize_model(default_params(N=10, K=3, seed=101))
        ws = BrokerageABM.SimWorkspace()
        agents = state.agents

        push!(agents[1].active_matches, ActiveMatch(2, :self))
        BrokerageABM.rebuild_current_match_index!(ws, agents)
        @test has_current_match(ws, 1, 2)
        @test has_current_match(ws, 2, 1)
        @test !has_current_match(ws, 1, 3)

        BrokerageABM.mark_current_match!(ws, 2, 3)
        @test has_current_match(ws, 2, 3)
        @test has_current_match(ws, 3, 2)

        BrokerageABM.reset_current_match_index!(ws, length(agents))
        @test !has_current_match(ws, 1, 2)
        @test !has_current_match(ws, 2, 3)
    end

    @testset "CurveGeometry" begin
        geo = CurveGeometry(8, 6, [1, 2, 3, 4, 5, 1], rand(6))
        @test geo.d == 8
        @test geo.s == 6
        @test length(geo.freqs) == 6
        @test length(geo.phases) == 6
    end

    @testset "CachedNetworkMeasures default" begin
        cnm = CachedNetworkMeasures()
        @test cnm.betweenness == 0.0
        @test cnm.constraint == 1.0
        @test cnm.effective_size == 0.0
    end
end
