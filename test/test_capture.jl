using Test
using TransientBrokerage
using Graphs: has_edge

function constant_prediction!(nn::NeuralNet, value::Float64)
    fill!(nn.W1, 0.0)
    fill!(nn.b1, 0.0)
    fill!(nn.w2, 0.0)
    nn.b2 = value
    return nothing
end

function capture_fixture(; receiver_value=10.0, ask_offset=1.0, n_recipients=1)
    p = default_params(;
        N=12,
        T=5,
        T_burn=1,
        n_strangers=0,
        seed=91,
        enable_principal=true,
        capture_min_error_obs=1,
        capture_error_threshold=999.0,
    )
    state = initialize_model(p)
    state.broker.capture_confidence_ready = true
    state.broker.capture_confidence_mae = 0.0
    state.broker.capture_error_count = 1
    constant_prediction!(state.broker.nn, state.cal.r + 10.0)
    for id in 1:(n_recipients + 1)
        constant_prediction!(state.agents[id].nn, receiver_value)
        remove_agent_edges!(state.G, id)
    end
    state.broker.roster = Set(2:(n_recipients + 1))
    state.broker.current_clients = Set([1])
    state.agents[1].history_count = 1
    state.agents[1].history_q[1] = state.cal.r + ask_offset
    return state
end

function run_capture_round!(state, counts=[1])
    return TransientBrokerage.run_offer_market!(
        [1],
        [:broker],
        counts,
        state.agents,
        state.broker,
        state.env,
        state.G,
        state.params,
        state.cal,
        state.rng;
        ws=state.workspace,
        accum=state.accum,
    )
end

@testset "Resource Capture" begin
    @testset "counterparty ask uses history mean or calibration fallback" begin
        state = initialize_model(default_params(N=20, seed=42))
        agent = state.agents[1]
        agent.history_count = 0
        @test counterparty_ask(agent, state.cal.q_cal) == state.cal.q_cal

        agent.history_count = 3
        agent.history_q[1:3] .= [1.0, 2.0, 4.0]
        @test counterparty_ask(agent, state.cal.q_cal) ≈ 7.0 / 3
    end

    @testset "capture surplus remains a simple spread helper" begin
        @test capture_surplus(5.0, 3.0) == 2.0
        @test capture_surplus(2.0, 3.5) == -1.5
    end

    @testset "capture confidence update tracks cumulative errors" begin
        state = initialize_model(default_params(N=20, seed=43))
        broker = state.broker
        @test !broker.capture_confidence_ready

        TransientBrokerage.update_capture_confidence_mae!(broker, 6.0, 3, 0.2)
        @test broker.capture_confidence_ready
        @test broker.capture_confidence_mae == 2.0
        @test broker.capture_error_count == 3

        TransientBrokerage.update_capture_confidence_mae!(broker, 3.0, 3, 0.2)
        @test broker.capture_confidence_mae ≈ 1.8
        @test broker.capture_error_count == 6
    end

    @testset "readiness gate controls principal mode" begin
        state = capture_fixture()
        state.broker.capture_confidence_ready = false
        accepted = run_capture_round!(state)
        @test state.accum.captured_position_count == 0

        state = capture_fixture()
        state.broker.capture_confidence_mae = 1.0e9
        accepted = run_capture_round!(state)
        @test state.accum.captured_position_count == 0
    end

    @testset "ask below reservation makes lot ineligible" begin
        state = capture_fixture(ask_offset=-0.1)
        accepted = run_capture_round!(state)
        @test state.accum.captured_position_count == 0
    end

    @testset "whole-lot feasibility requires enough planned recipients" begin
        state = capture_fixture(n_recipients=1)
        accepted = run_capture_round!(state, [2])
        @test state.accum.captured_position_count == 0
    end

    @testset "recipient rejection realizes zero-output exposure" begin
        state = capture_fixture(receiver_value=0.0)
        accepted = run_capture_round!(state)
        ask = counterparty_ask(state.agents[1], state.cal.q_cal)

        @test isempty(accepted)
        @test state.accum.captured_origin_count == 1
        @test state.accum.captured_position_count == 1
        @test state.accum.principal_rejected == 1
        @test state.accum.capture_realized == [0.0]
        @test state.accum.capture_ask == [ask]
        @test state.workspace.principal_payment[1] == ask
        @test state.accum.broker_error_count == 1
    end

    @testset "accepted principal positions lock in learning and structure" begin
        state = capture_fixture(receiver_value=10.0)
        constant_prediction!(state.agents[2].nn, state.cal.r + 1.0)
        h1 = state.agents[1].history_count
        h2 = state.agents[2].history_count
        hb = state.broker.history_count

        accepted = run_capture_round!(state)

        @test length(accepted) == 1
        @test accepted[1].is_principal
        @test state.agents[1].history_count == h1
        @test state.agents[2].history_count == h2
        @test state.broker.history_count == hb + 1
        @test !has_edge(state.G, 1, 2)
        @test has_current_match(state.agents[1], 2)
        @test has_current_match(state.agents[2], 1)
        @test all(am -> am.is_principal, state.agents[1].active_matches)
    end

    @testset "captured origin satisfaction uses ask without broker fee" begin
        state = capture_fixture(receiver_value=0.0)
        accepted = run_capture_round!(state)
        sat0 = state.agents[1].satisfaction_broker
        ask = state.workspace.principal_payment[1]

        update_satisfaction!(
            state.agents,
            accepted,
            [1],
            [:broker],
            [1],
            state.cal,
            state.params;
            principal_payment=state.workspace.principal_payment,
        )

        expected = (1 - state.params.omega) * sat0 + state.params.omega * ask
        @test state.agents[1].satisfaction_broker ≈ expected
    end

    @testset "captured origins are excluded from standard self-search offers" begin
        state = capture_fixture(receiver_value=10.0, n_recipients=1)
        constant_prediction!(state.agents[2].nn, state.cal.r + 1.0)
        remove_agent_edges!(state.G, 3)
        add_match_edge!(state.G, 3, 1)
        add_match_edge!(state.G, 3, 2)
        update_partner_mean!(state.agents[3], 1, state.cal.r + 10.0)
        update_partner_mean!(state.agents[3], 2, state.cal.r + 5.0)
        update_partner_mean!(state.agents[2], 3, state.cal.r + 1.0)

        TransientBrokerage.run_offer_market!(
            [1, 3],
            [:broker, :self],
            [1, 1],
            state.agents,
            state.broker,
            state.env,
            state.G,
            state.params,
            state.cal,
            state.rng;
            ws=state.workspace,
            accum=state.accum,
        )

        self_offers = [o for o in state.workspace.offers if o.from_id == 3]
        @test length(self_offers) == 1
        @test self_offers[1].to_id == 2
        @test all(o -> o.to_id != 1, state.workspace.offers)
    end

    @testset "captured origins are excluded from standard broker offers" begin
        state = capture_fixture(receiver_value=10.0, n_recipients=1)
        state.broker.current_clients = Set([1, 3])
        push!(state.broker.roster, 1)
        push!(state.broker.roster, 2)
        constant_prediction!(state.agents[2].nn, state.cal.r + 1.0)
        state.agents[3].history_count = 1
        state.agents[3].history_q[1] = state.cal.r - 0.1

        TransientBrokerage.run_offer_market!(
            [1, 3],
            [:broker, :broker],
            [1, 1],
            state.agents,
            state.broker,
            state.env,
            state.G,
            state.params,
            state.cal,
            state.rng;
            ws=state.workspace,
            accum=state.accum,
        )

        broker_offers = [o for o in state.workspace.offers if o.from_id == 3]
        @test length(broker_offers) == 1
        @test broker_offers[1].to_id == 2
        @test all(o -> o.to_id != 1, state.workspace.offers)
    end
end
