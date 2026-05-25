using Test
using TransientBrokerage
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
        @test p.omega == 0.20
        @test p.search_cost_rate == 0.15
        @test p.n_strangers == 10
        @test p.roster_churn == 0.02
        @test p.network_measure_interval == 20
        @test p.enable_principal == false
        @test p.capture_min_error_obs == 100
        @test p.capture_error_threshold == 0.65
    end

    @testset "default_params with overrides" begin
        p = default_params(; seed=99, N=200, K=10, delta=0.75)
        @test p.seed == 99
        @test p.N == 200
        @test p.K == 10
        @test p.delta == 0.75
    end

    @testset "default_params rejects unknown kwargs" begin
        @test_throws ErrorException default_params(; bogus_param=42)
        @test_throws ErrorException default_params(; tau=1)
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
        @test_throws AssertionError default_params(roster_churn=-0.1)
        @test_throws AssertionError default_params(roster_churn=1.1)
        @test_throws AssertionError default_params(capture_min_error_obs=-1)
        @test_throws AssertionError default_params(capture_error_threshold=-0.1)
        @test_throws AssertionError default_params(T=10, T_burn=15)
    end

    @testset "NeuralNet and NNGradBuffers" begin
        rng = StableRNG(42)
        nn = init_neural_net(8, 16, rng)
        @test size(nn.W1) == (16, 8)
        @test length(nn.b1) == 16
        @test length(nn.w2) == 16
        @test nn.b2 == Q_OFFSET  # b2 initialized to population-mean prior

        grad = NNGradBuffers(nn)
        @test size(grad.dW1) == (16, 8)
        @test length(grad.db1) == 16
        @test length(grad.dw2) == 16
    end

    @testset "NeuralNet parameter counts" begin
        rng = StableRNG(42)
        p = default_params()
        nn_a = init_neural_net(p.d, p.h_a, rng)
        n_params_a = length(nn_a.W1) + length(nn_a.b1) + length(nn_a.w2) + 1
        @test n_params_a == 161  # 16*8 + 16 + 16 + 1

        nn_b = init_neural_net(2*p.d, p.h_b, rng)
        n_params_b = length(nn_b.W1) + length(nn_b.b1) + length(nn_b.w2) + 1
        @test n_params_b == 577  # 32*16 + 32 + 32 + 1
    end

    @testset "ActiveMatch construction" begin
        am = ActiveMatch(5, false, :self)
        @test am.partner_id == 5
        @test am.is_principal == false
        @test am.channel == :self
    end

    @testset "reset_accumulators!" begin
        accum = PeriodAccumulators()
        accum.n_self_matches = 10
        accum.n_broker_standard = 5
        accum.n_broker_principal = 3
        push!(accum.q_self, 1.0, 2.0)
        accum.n_demanders = 7
        accum.n_outsourced = 2
        accum.outsourced_slots = 9
        accum.roster_size = 42
        accum.broker_access_size = 45
        accum.broker_error_abs_sum = 4.0
        accum.broker_error_count = 3
        accum.broker_confidence_mae = 1.5
        accum.captured_origin_count = 2
        accum.captured_position_count = 4
        accum.principal_rejected = 1
        accum.capture_ready = true
        accum.capture_scaled_mae = 0.2

        reset_accumulators!(accum)

        @test accum.n_self_matches == 0
        @test accum.n_broker_standard == 0
        @test accum.n_broker_principal == 0
        @test isempty(accum.q_self)
        @test accum.n_demanders == 0
        @test accum.n_outsourced == 0
        @test accum.outsourced_slots == 0
        @test accum.roster_size == 0
        @test accum.broker_access_size == 0
        @test accum.broker_error_abs_sum == 0.0
        @test accum.broker_error_count == 0
        @test isnan(accum.broker_confidence_mae)
        @test accum.captured_origin_count == 0
        @test accum.captured_position_count == 0
        @test accum.principal_rejected == 0
        @test !accum.capture_ready
        @test isnan(accum.capture_scaled_mae)
    end

    @testset "Agent history recording and growth" begin
        rng = StableRNG(42)
        p = default_params(N=20)
        nn = init_neural_net(p.d, p.h_a, rng)
        agent = Agent(
            id=1,
            type=randn(rng, p.d),
            history_X=Matrix{Float64}(undef, p.d, 4),  # small initial capacity
            history_q=Vector{Float64}(undef, 4),
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(p.h_a),
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

        broker.history_Xi = Matrix{Float64}(undef, d, 2)
        broker.history_Xj = Matrix{Float64}(undef, d, 2)
        broker.history_q = Vector{Float64}(undef, 2)
        broker.train_X = Matrix{Float64}(undef, 2 * d, 4)
        broker.train_q = Vector{Float64}(undef, 4)
        broker.history_count = 0
        broker.n_new_obs = 0

        xi1 = randn(rng, d);
        xj1 = randn(rng, d)
        xi2 = randn(rng, d);
        xj2 = randn(rng, d)
        xi3 = randn(rng, d);
        xj3 = randn(rng, d)
        record_broker_history!(broker, xi1, xj1, 1.0)
        record_broker_history!(broker, xi2, xj2, 2.0)
        record_broker_history!(broker, xi3, xj3, 3.0)  # triggers growth

        @test broker.history_count == 3
        @test broker.n_new_obs == 3
        @test size(broker.history_Xi, 2) >= 3
        @test size(broker.history_Xj, 2) >= 3
        @test size(broker.train_X, 2) >= 6
        @test broker.history_q[3] == 3.0
    end

    @testset "Partner mean tracking" begin
        rng = StableRNG(42)
        p = default_params(N=10)
        nn = init_neural_net(p.d, p.h_a, rng)
        agent = Agent(
            id=1,
            type=randn(rng, p.d),
            history_X=Matrix{Float64}(undef, p.d, 16),
            history_q=Vector{Float64}(undef, 16),
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(p.h_a),
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
        nn = init_neural_net(p.d, p.h_a, rng)
        agent = Agent(
            id=1,
            type=randn(rng, p.d),
            history_X=Matrix{Float64}(undef, p.d, 16),
            history_q=Vector{Float64}(undef, 16),
            nn=nn,
            nn_grad=NNGradBuffers(nn),
            predict_buf=zeros(p.h_a),
            partner_sum=zeros(10),
            partner_count=zeros(Int, 10),
        )

        @test !has_current_match(agent, 2)
        push!(agent.active_matches, ActiveMatch(2, false, :self))
        @test has_current_match(agent, 2)
        @test !has_current_match(agent, 3)
        push!(agent.active_matches, ActiveMatch(3, false, :broker))
        push!(agent.active_matches, ActiveMatch(4, false, :broker))
        @test length(agent.active_matches) == 3
    end

    @testset "Current-match workspace index" begin
        state = initialize_model(default_params(N=10, K=3, seed=101))
        ws = TransientBrokerage.SimWorkspace()
        agents = state.agents

        push!(agents[1].active_matches, ActiveMatch(2, false, :self))
        TransientBrokerage.rebuild_current_match_index!(ws, agents)
        @test has_current_match(ws, 1, 2)
        @test has_current_match(ws, 2, 1)
        @test !has_current_match(ws, 1, 3)

        TransientBrokerage.mark_current_match!(ws, 2, 3)
        @test has_current_match(ws, 2, 3)
        @test has_current_match(ws, 3, 2)

        TransientBrokerage.reset_current_match_index!(ws, length(agents))
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
