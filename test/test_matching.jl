using Test
using BrokerageABM
using BrokerageABM: AcceptedMatch, OfferCredit, add_match_edge!
using BrokerageABM: has_current_match, outsourcing_decision, remove_agent_edges!
using BrokerageABM: update_broker_reputation!, update_partner_mean!
using BrokerageABM: update_satisfaction!
using Graphs: has_edge
using StableRNGs: StableRNG

@testset "Match Formation and Outsourcing" begin
    p = default_params(N=30, T=5, T_burn=1, K=3, seed=42)

    @testset "One-sided offers require receiver acceptance" begin
        state = initialize_model(
            default_params(N=12, T=5, T_burn=1, n_strangers=0, seed=123)
        )
        agents = state.agents
        G = state.G

        remove_agent_edges!(G, 1)
        remove_agent_edges!(G, 2)
        add_match_edge!(G, 1, 2)
        agents[1].partner_sum[2] = 10.0
        agents[1].partner_count[2] = 1
        agents[2].partner_sum[1] = 0.0
        agents[2].partner_count[1] = 1

        accepted = BrokerageABM.run_offer_market!(
            [1],
            [:self],
            [1],
            agents,
            state.broker,
            state.env,
            G,
            state.params,
            state.cal,
            StableRNG(17);
            ws=state.workspace,
        )

        @test isempty(accepted)

        agents[2].partner_sum[1] = 10.0
        accepted = BrokerageABM.run_offer_market!(
            [1],
            [:self],
            [1],
            agents,
            state.broker,
            state.env,
            G,
            state.params,
            state.cal,
            StableRNG(18);
            ws=state.workspace,
        )

        @test length(accepted) == 1
        @test accepted[1].offer1.from_id == 1
        @test accepted[1].offer1.to_id == 2
        @test accepted[1].offer2 === nothing
    end

    @testset "Reciprocal offers accept without receiver evaluation" begin
        state = initialize_model(default_params(N=12, T=5, T_burn=1, seed=124))
        ws = state.workspace
        agents = state.agents

        BrokerageABM.rebuild_current_match_index!(ws, agents)
        BrokerageABM.reset_offer_book!(ws, state.params.N)
        @test BrokerageABM.add_offer!(ws, 1, 2, :self, 10.0)
        @test BrokerageABM.add_offer!(ws, 2, 1, :broker, 10.0)

        agents[1].partner_sum[2] = -10.0
        agents[1].partner_count[2] = 1
        agents[2].partner_sum[1] = -10.0
        agents[2].partner_count[1] = 1

        accepted = BrokerageABM.AcceptedMatch[]
        match_output = ws.match_output
        if length(match_output.Ax_buf) != state.params.d
            match_output.Ax_buf = Vector{Float64}(undef, state.params.d)
            match_output.Bx_buf = Vector{Float64}(undef, state.params.d)
        end
        BrokerageABM.accept_offer_pair!(
            accepted,
            1,
            2,
            agents,
            state.broker,
            state.env,
            state.G,
            state.cal,
            StableRNG(19);
            Ax_buf=match_output.Ax_buf,
            Bx_buf=match_output.Bx_buf,
            ws=ws,
        )

        @test length(accepted) == 1
        @test accepted[1].offer1.channel == :self
        @test accepted[1].offer2.channel == :broker
        @test has_current_match(agents[1], 2)
        @test has_current_match(agents[2], 1)
    end

    @testset "No receiver-side K cap rejects acceptable offers" begin
        p_low_k = default_params(N=12, T=5, T_burn=1, K=1, n_strangers=0, seed=125)
        state = initialize_model(p_low_k)
        agents = state.agents
        G = state.G

        for agent_id in 1:3
            remove_agent_edges!(G, agent_id)
        end
        add_match_edge!(G, 1, 3)
        add_match_edge!(G, 2, 3)
        for i in 1:2
            agents[i].partner_sum[3] = 10.0
            agents[i].partner_count[3] = 1
            agents[3].partner_sum[i] = 10.0
            agents[3].partner_count[i] = 1
        end

        accepted = BrokerageABM.run_offer_market!(
            [1, 2],
            [:self, :self],
            [1, 1],
            agents,
            state.broker,
            state.env,
            G,
            p_low_k,
            state.cal,
            StableRNG(20);
            ws=state.workspace,
        )

        @test length(accepted) == 2
        @test length(agents[3].active_matches) == 2
        @test Set(m.counterparty_id for m in accepted) == Set([3])
    end

    @testset "Accepted matches create edges and update histories once" begin
        state = initialize_model(p)
        a1, a2 = state.agents[1], state.agents[2]
        h1_before = a1.history_count
        h2_before = a2.history_count

        remove_agent_edges!(state.G, 1)
        remove_agent_edges!(state.G, 2)
        add_match_edge!(state.G, 1, 2)
        update_partner_mean!(a1, 2, 6.0)
        update_partner_mean!(a2, 1, 5.0)

        accepted = BrokerageABM.run_offer_market!(
            [1],
            [:self],
            [1],
            state.agents,
            state.broker,
            state.env,
            state.G,
            p,
            state.cal,
            StableRNG(55);
            ws=state.workspace,
        )

        @test length(accepted) == 1
        @test a1.history_count == h1_before + 1
        @test a2.history_count == h2_before + 1
        @test has_edge(state.G, 1, 2)
    end

    @testset "Satisfaction credits directed offer channels" begin
        state = initialize_model(p)
        omega = p.omega
        c_s = state.cal.c_s
        phi = state.cal.phi
        state.agents[1].satisfaction_self = 1.0
        state.agents[2].satisfaction_broker = 2.0
        sat1_before = state.agents[1].satisfaction_self
        sat2_before = state.agents[2].satisfaction_broker

        accepted = [
            AcceptedMatch(
                1,
                2,
                :broker,
                4.0,
                4.0,
                OfferCredit(1, 2, :self, 4.0, false),
                OfferCredit(2, 1, :broker, 4.0, false),
            ),
        ]

        update_satisfaction!(
            state.agents, accepted, [1, 2], [:self, :broker], [1, 1], state.cal, p
        )

        @test state.agents[1].satisfaction_self ≈
            (1 - omega) * sat1_before + omega * (4.0 - c_s)
        @test state.agents[2].satisfaction_broker ≈
            (1 - omega) * sat2_before + omega * (4.0 - phi)
    end

    @testset "No-match satisfaction updates only active channel" begin
        state = initialize_model(p)
        omega = p.omega
        sat_self_before = state.agents[1].satisfaction_self
        sat_broker_before = state.agents[1].satisfaction_broker
        accepted = AcceptedMatch[]

        update_satisfaction!(state.agents, accepted, [1], [:self], [2], state.cal, p)

        @test state.agents[1].satisfaction_self ≈
            (1 - omega) * sat_self_before - omega * state.cal.c_s
        @test state.agents[1].satisfaction_broker == sat_broker_before
    end

    @testset "Outsourcing decision follows satisfaction and reputation" begin
        state = initialize_model(p)
        agent = state.agents[1]
        agent.satisfaction_self = 5.0
        agent.satisfaction_broker = 1.0
        agent.tried_broker = true
        @test outsourcing_decision(agent, 0.0, StableRNG(1)) == :self

        agent.satisfaction_self = 1.0
        agent.satisfaction_broker = 50.0
        @test outsourcing_decision(agent, 0.0, StableRNG(1)) == :broker

        agent.tried_broker = false
        agent.satisfaction_self = 0.0
        @test outsourcing_decision(agent, 10.0, StableRNG(1)) == :broker
    end

    @testset "Broker reputation update" begin
        state = initialize_model(p)
        state.agents[1].satisfaction_broker = 3.0
        state.agents[2].satisfaction_broker = 5.0
        update_broker_reputation!(state.broker, state.agents, [1, 2])
        @test state.broker.last_reputation ≈ 4.0
        @test state.broker.has_had_clients

        state.broker.last_reputation = 3.5
        update_broker_reputation!(state.broker, state.agents, Int[])
        @test state.broker.last_reputation == 3.5
    end
end
