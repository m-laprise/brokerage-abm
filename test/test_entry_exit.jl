using Test
using BrokerageABM
using BrokerageABM: ActiveMatch, Q_OFFSET, add_match_edge!, enter_agent!
using BrokerageABM: exit_agent!, process_entry_exit!, remove_agent_edges!

@testset "Entry/Exit" begin
    using Graphs: degree, has_edge, neighbors, nv

    @testset "exit_agent! clears edges, broker access sets, and partner tracking" begin
        state1 = initialize_model(default_params(N=20, seed=42))
        target = 1
        partner = 2

        empty!(state1.agents[target].active_matches)
        empty!(state1.agents[partner].active_matches)
        push!(state1.agents[target].active_matches, ActiveMatch(partner, :self))
        push!(state1.agents[partner].active_matches, ActiveMatch(target, :self))
        add_match_edge!(state1.G, target, partner)

        push!(state1.broker.roster, target)
        push!(state1.broker.current_clients, target)
        state1.agents[partner].partner_sum[target] = 5.0
        state1.agents[partner].partner_count[target] = 2

        @test degree(state1.G, target) > 0
        exit_agent!(state1, target)

        @test degree(state1.G, target) == 0
        @test isempty(state1.agents[target].active_matches)
        @test !(target in state1.broker.roster)
        @test !(target in state1.broker.current_clients)
        @test !any(m -> m.partner_id == target, state1.agents[partner].active_matches)
        @test state1.agents[partner].partner_sum[target] == 0.0
        @test state1.agents[partner].partner_count[target] == 0
    end

    @testset "enter_agent! resets all fields to fresh state" begin
        state2 = initialize_model(default_params(N=50, seed=99))
        rng = state2.rng
        agent_id = 1

        # Dirty the agent first
        a = state2.agents[agent_id]
        a.history_count = 10
        a.n_new_obs = 5
        a.satisfaction_self = 999.0
        a.satisfaction_broker = 999.0
        a.tried_broker = true
        a.periods_alive = 100
        push!(a.active_matches, ActiveMatch(2, :self))

        enter_agent!(state2, agent_id, rng)
        a = state2.agents[agent_id]

        @test a.history_count == 0
        @test a.n_new_obs == 0
        @test a.obs_period_marks == [0]
        @test isfinite(a.satisfaction_self)     # set from neighbors' satisfaction
        @test isfinite(a.satisfaction_broker)  # set to broker reputation (market prior)
        @test a.tried_broker == false
        @test a.periods_alive == 0
        @test isempty(a.active_matches)
        @test all(==(0), a.partner_count)
        @test all(==(0.0), a.partner_sum)
    end

    @testset "enter_agent! produces unit-norm type on the sphere" begin
        state3 = initialize_model(default_params(N=50, seed=77))
        rng = state3.rng
        for i in 1:5
            enter_agent!(state3, i, rng)
            t = state3.agents[i].type
            @test all(isfinite, t)
            @test length(t) == state3.params.d
            @test isapprox(sqrt(sum(t .^ 2)), 1.0; atol=1e-10)
        end
    end

    @testset "enter_agent! creates edges to type-similar neighbors" begin
        state4 = initialize_model(default_params(N=50, k=6, seed=33))
        rng = state4.rng
        remove_agent_edges!(state4.G, 1)
        @test degree(state4.G, 1) == 0

        enter_agent!(state4, 1, rng)
        expected_edges = state4.params.k ÷ 2
        @test degree(state4.G, 1) == expected_edges
        @test !has_edge(state4.G, 1, 1)  # no self-edge
    end

    @testset "enter_agent! initializes NN with b2 = Q_OFFSET" begin
        state5 = initialize_model(default_params(N=20, seed=55))
        enter_agent!(state5, 1, state5.rng)
        @test state5.agents[1].nn.b2 == Q_OFFSET
        @test all(iszero, state5.agents[1].nn.w2)
        @test all(isfinite, state5.agents[1].nn.W1)
    end

    @testset "process_entry_exit! conserves population count" begin
        p2 = default_params(N=100, T=1, seed=42, eta=0.20)
        state6 = initialize_model(p2)
        n_before = length(state6.agents)
        process_entry_exit!(state6, state6.rng)
        @test length(state6.agents) == n_before
        @test nv(state6.G) == p2.N + 1  # N agents + 1 broker
    end

    @testset "process_entry_exit! turnover rate approximates eta" begin
        p3 = default_params(N=200, T=1, seed=42, eta=0.10)
        state7 = initialize_model(p3)
        old_types = [copy(a.type) for a in state7.agents]
        process_entry_exit!(state7, state7.rng)
        n_changed = count(i -> state7.agents[i].type != old_types[i], 1:p3.N)
        # With N=200, eta=0.10: expected ~20 exits, stdev ~4.2
        @test 5 <= n_changed <= 45
    end

    @testset "process_entry_exit! with eta=0 is a no-op" begin
        p4 = default_params(N=50, T=1, seed=42, eta=0.0)
        state8 = initialize_model(p4)
        old_types = [copy(a.type) for a in state8.agents]
        process_entry_exit!(state8, state8.rng)
        @test all(state8.agents[i].type == old_types[i] for i in 1:p4.N)
    end

    @testset "exit + enter preserves graph symmetry" begin
        state9 = initialize_model(default_params(N=50, seed=11))
        exit_agent!(state9, 5)
        enter_agent!(state9, 5, state9.rng)
        for nbr in neighbors(state9.G, 5)
            @test has_edge(state9.G, nbr, 5)
        end
    end
end
