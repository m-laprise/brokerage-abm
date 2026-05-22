using Test
using TransientBrokerage
using Graphs: has_edge, neighbors
using StableRNGs: StableRNG

@testset "Match Formation and Outsourcing" begin
    p = default_params(N=30, T=5, T_burn=1, K=3, seed=42)
    state = initialize_model(p)
    @testset "Round matching falls back within round after rejection" begin
        p_round = default_params(N=12, T=5, T_burn=1, K=1, n_strangers=0, seed=123)
        state_round = initialize_model(p_round)
        agents_r = state_round.agents
        G_r = state_round.G

        for agent_id in 1:4
            remove_agent_edges!(G_r, agent_id)
        end
        add_match_edge!(G_r, 1, 3)
        add_match_edge!(G_r, 1, 4)
        add_match_edge!(G_r, 2, 3)
        add_match_edge!(G_r, 2, 4)

        agents_r[1].partner_sum[3] = 10.0;
        agents_r[1].partner_count[3] = 1
        agents_r[1].partner_sum[4] = 5.0;
        agents_r[1].partner_count[4] = 1
        agents_r[2].partner_sum[3] = 9.0;
        agents_r[2].partner_count[3] = 1
        agents_r[2].partner_sum[4] = 4.0;
        agents_r[2].partner_count[4] = 1

        agents_r[3].partner_sum[1] = 8.0;
        agents_r[3].partner_count[1] = 1
        agents_r[3].partner_sum[2] = 6.0;
        agents_r[3].partner_count[2] = 1
        agents_r[4].partner_sum[1] = 3.0;
        agents_r[4].partner_count[1] = 1
        agents_r[4].partner_sum[2] = 7.0;
        agents_r[4].partner_count[2] = 1
        TransientBrokerage.reset_principal_inventory!(state_round.workspace, p_round.N)

        accepted = TransientBrokerage.round_match_formation!(
            [1, 2],
            [:self, :self],
            [1, 1],
            agents_r,
            state_round.broker,
            state_round.env,
            G_r,
            p_round,
            state_round.cal,
            StableRNG(17);
            ws=state_round.workspace,
        )

        accepted_pairs = Set((m.demander_id, m.counterparty_id) for m in accepted)
        @test length(accepted) == 2
        @test accepted_pairs == Set([(1, 3), (2, 4)])
    end

    @testset "Round matching fills one relationship position per round" begin
        p_round2 = default_params(N=12, T=5, T_burn=1, K=2, n_strangers=0, seed=321)
        state_round2 = initialize_model(p_round2)
        agents_r2 = state_round2.agents
        G_r2 = state_round2.G

        for agent_id in 1:3
            remove_agent_edges!(G_r2, agent_id)
        end
        add_match_edge!(G_r2, 1, 2)
        add_match_edge!(G_r2, 1, 3)

        agents_r2[1].partner_sum[2] = 9.0;
        agents_r2[1].partner_count[2] = 1
        agents_r2[1].partner_sum[3] = 8.0;
        agents_r2[1].partner_count[3] = 1
        agents_r2[2].partner_sum[1] = 7.0;
        agents_r2[2].partner_count[1] = 1
        agents_r2[3].partner_sum[1] = 6.0;
        agents_r2[3].partner_count[1] = 1
        TransientBrokerage.reset_principal_inventory!(state_round2.workspace, p_round2.N)

        accepted = TransientBrokerage.round_match_formation!(
            [1],
            [:self],
            [2],
            agents_r2,
            state_round2.broker,
            state_round2.env,
            G_r2,
            p_round2,
            state_round2.cal,
            StableRNG(23);
            ws=state_round2.workspace,
        )

        @test length(accepted) == 2
        @test Set(m.counterparty_id for m in accepted) == Set([2, 3])
        @test length(state_round2.agents[1].active_matches) == 2
    end

    @testset "Round matching does not duplicate reciprocal current relationships" begin
        p_recip = default_params(N=12, T=5, T_burn=1, K=2, n_strangers=0, seed=322)
        state_recip = initialize_model(p_recip)
        agents_recip = state_recip.agents
        G_recip = state_recip.G

        for agent_id in 1:2
            remove_agent_edges!(G_recip, agent_id)
        end
        add_match_edge!(G_recip, 1, 2)
        agents_recip[1].partner_sum[2] = 9.0
        agents_recip[1].partner_count[2] = 1
        agents_recip[2].partner_sum[1] = 9.0
        agents_recip[2].partner_count[1] = 1
        TransientBrokerage.reset_principal_inventory!(state_recip.workspace, p_recip.N)

        accepted = TransientBrokerage.round_match_formation!(
            [1, 2],
            [:self, :self],
            [1, 1],
            agents_recip,
            state_recip.broker,
            state_recip.env,
            G_recip,
            p_recip,
            state_recip.cal,
            StableRNG(24);
            ws=state_recip.workspace,
        )

        @test length(accepted) == 1
        @test Set([accepted[1].demander_id, accepted[1].counterparty_id]) == Set([1, 2])
        @test length(agents_recip[1].active_matches) == 1
        @test length(agents_recip[2].active_matches) == 1
        @test has_current_match(agents_recip[1], 2)
        @test has_current_match(agents_recip[2], 1)
    end

    @testset "Round matching creates edges and updates histories for standard matches" begin
        state3 = initialize_model(p)
        a1, a2 = state3.agents[1], state3.agents[2]
        h1_before = a1.history_count
        h2_before = a2.history_count

        remove_agent_edges!(state3.G, 1)
        add_match_edge!(state3.G, 1, 2)
        update_partner_mean!(a1, 2, 6.0)
        update_partner_mean!(a2, 1, 5.0)
        TransientBrokerage.reset_principal_inventory!(state3.workspace, p.N)

        accepted = TransientBrokerage.round_match_formation!(
            [1],
            [:self],
            [1],
            state3.agents,
            state3.broker,
            state3.env,
            state3.G,
            p,
            state3.cal,
            StableRNG(55);
            ws=state3.workspace,
        )

        @test length(accepted) == 1
        @test a1.history_count == h1_before + 1
        @test a2.history_count == h2_before + 1
        @test has_edge(state3.G, 1, 2)
    end

    @testset "Satisfaction EWMA update" begin
        state5 = initialize_model(p)
        omega = p.omega
        phi = state5.cal.phi
        c_s = state5.cal.c_s

        # Agent 1 self-searches for two positions, gets one match. Self-search cost
        # is charged per demanded position, regardless of fill.
        d_ids = [1];
        d_chs = [:self];
        d_cnts = [2]
        accepted = [(
            demander_id=1,
            counterparty_id=5,
            channel=:self,
            is_principal=false,
            q_realized=2.0,
            q_predicted=1.5,
            ask_j=NaN,
            capture_qhat=NaN,
        )]
        sat_before = state5.agents[1].satisfaction_self
        update_satisfaction!(state5.agents, accepted, d_ids, d_chs, d_cnts, state5.cal, p)
        expected = (1 - omega) * sat_before + omega * (2.0 / 2 - c_s)
        @test state5.agents[1].satisfaction_self ≈ expected
    end

    @testset "Broker no-match decays satisfaction" begin
        state6 = initialize_model(p)
        omega = p.omega
        sat_before = state6.agents[1].satisfaction_broker
        d_ids = [1];
        d_chs = [:broker];
        d_cnts = [2]
        accepted = NamedTuple{
            (
                :demander_id,
                :counterparty_id,
                :channel,
                :is_principal,
                :q_realized,
                :q_predicted,
                :ask_j,
                :capture_qhat,
            ),
            Tuple{Int,Int,Symbol,Bool,Float64,Float64,Float64,Float64},
        }[]
        update_satisfaction!(state6.agents, accepted, d_ids, d_chs, d_cnts, state6.cal, p)
        @test state6.agents[1].satisfaction_broker ≈ (1 - omega) * sat_before
    end

    @testset "Self-search failure pays per-position search cost" begin
        state6b = initialize_model(p)
        omega = p.omega
        sat_before = state6b.agents[1].satisfaction_self
        d_ids = [1];
        d_chs = [:self];
        d_cnts = [2]
        accepted = NamedTuple{
            (
                :demander_id,
                :counterparty_id,
                :channel,
                :is_principal,
                :q_realized,
                :q_predicted,
                :ask_j,
                :capture_qhat,
            ),
            Tuple{Int,Int,Symbol,Bool,Float64,Float64,Float64,Float64},
        }[]
        update_satisfaction!(state6b.agents, accepted, d_ids, d_chs, d_cnts, state6b.cal, p)
        expected = (1 - omega) * sat_before - omega * state6b.cal.c_s
        @test state6b.agents[1].satisfaction_self ≈ expected
    end

    @testset "Standard broker fee is charged only on successful placements" begin
        state6c = initialize_model(p)
        omega = p.omega
        sat_before = state6c.agents[1].satisfaction_broker
        d_ids = [1];
        d_chs = [:broker];
        d_cnts = [2]
        accepted = [(
            demander_id=1,
            counterparty_id=5,
            channel=:broker,
            is_principal=false,
            q_realized=3.0,
            q_predicted=2.5,
            ask_j=NaN,
            capture_qhat=NaN,
        )]
        update_satisfaction!(state6c.agents, accepted, d_ids, d_chs, d_cnts, state6c.cal, p)
        expected = (1 - omega) * sat_before + omega * ((3.0 - state6c.cal.phi) / 2)
        @test state6c.agents[1].satisfaction_broker ≈ expected
    end

    @testset "Principal-mode satisfaction: no fee deducted" begin
        state7 = initialize_model(
            default_params(N=30, T=5, T_burn=1, K=3, seed=77, enable_principal=true)
        )
        omega = p.omega
        sat_before = state7.agents[1].satisfaction_broker
        d_ids = [1];
        d_chs = [:broker];
        d_cnts = [2]
        accepted = [(
            demander_id=1,
            counterparty_id=5,
            channel=:broker,
            is_principal=true,
            q_realized=3.0,
            q_predicted=2.5,
            ask_j=1.0,
            capture_qhat=2.5,
        )]
        update_satisfaction!(
            state7.agents, accepted, d_ids, d_chs, d_cnts, state7.cal, state7.params
        )
        # No fee for principal mode: cost = 0
        expected = (1 - omega) * sat_before + omega * (3.0 / 2)
        @test state7.agents[1].satisfaction_broker ≈ expected
    end

    @testset "Outsourcing decision follows satisfaction" begin
        state8 = initialize_model(p)
        agent = state8.agents[1]
        # High self-satisfaction, low broker satisfaction
        agent.satisfaction_self = 5.0
        agent.satisfaction_broker = 1.0
        agent.tried_broker = true
        @test outsourcing_decision(agent, 0.0, StableRNG(1)) == :self

        # Reverse: broker satisfaction much higher than self
        agent.satisfaction_self = 1.0
        agent.satisfaction_broker = 50.0
        @test outsourcing_decision(agent, 0.0, StableRNG(1)) == :broker
    end

    @testset "Outsourcing ignores known-partner means as a separate hurdle" begin
        state8b = initialize_model(p)
        G8b = state8b.G
        bn8b = state8b.broker.node_id
        agent = state8b.agents[1]
        nbr = first(filter(n -> n != bn8b, neighbors(G8b, agent.id)))

        agent.satisfaction_self = 1.0
        agent.satisfaction_broker = 10.0
        agent.tried_broker = true
        agent.partner_sum[nbr] = 100.0
        agent.partner_count[nbr] = 1

        @test outsourcing_decision(agent, 0.0, StableRNG(1)) == :broker
    end

    @testset "Untried broker uses reputation" begin
        state9 = initialize_model(p)
        agent = state9.agents[1]
        agent.tried_broker = false
        agent.satisfaction_self = 0.0
        # High broker reputation should make agent outsource
        @test outsourcing_decision(agent, 10.0, StableRNG(1)) == :broker
    end

    @testset "Broker reputation update" begin
        state10 = initialize_model(p)
        state10.agents[1].satisfaction_broker = 3.0
        state10.agents[2].satisfaction_broker = 5.0
        client_ids = [1, 2]
        update_broker_reputation!(state10.broker, state10.agents, client_ids)
        @test state10.broker.last_reputation ≈ 4.0
        @test state10.broker.has_had_clients == true
    end

    @testset "Broker reputation is sticky with no clients" begin
        state11 = initialize_model(p)
        state11.broker.last_reputation = 3.5
        state11.broker.has_had_clients = true
        update_broker_reputation!(state11.broker, state11.agents, Int[])
        @test state11.broker.last_reputation == 3.5  # unchanged
    end
end
