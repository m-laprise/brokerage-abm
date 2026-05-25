using Test
using TransientBrokerage
using StableRNGs: StableRNG

@testset "Search" begin
    @testset "Self-search offers rank known neighbors by value" begin
        p = default_params(N=20, T=5, T_burn=1, n_strangers=0, seed=7)
        state = initialize_model(p)
        agents = state.agents
        G = state.G
        ws = state.workspace

        remove_agent_edges!(G, 1)
        add_match_edge!(G, 1, 2)
        add_match_edge!(G, 1, 3)
        agents[1].partner_sum[2] = 9.0
        agents[1].partner_count[2] = 1
        agents[1].partner_sum[3] = 7.0
        agents[1].partner_count[3] = 1

        TransientBrokerage.rebuild_current_match_index!(ws, agents)
        TransientBrokerage.reset_offer_book!(ws, p.N)
        sent = TransientBrokerage.append_self_search_offers!(
            ws,
            agents[1],
            2,
            agents,
            G,
            state.broker.node_id,
            Int[],
            -1e9;
            current_match_index_ready=true,
        )

        @test sent == 2
        @test [(o.to_id, o.predicted_value) for o in ws.offers] == [(2, 9.0), (3, 7.0)]
    end

    @testset "Self-search no longer filters candidates by K capacity" begin
        p = default_params(N=20, T=5, T_burn=1, K=1, n_strangers=0, seed=11)
        state = initialize_model(p)
        agents = state.agents
        G = state.G
        ws = state.workspace

        remove_agent_edges!(G, 1)
        add_match_edge!(G, 1, 2)
        agents[1].partner_sum[2] = 100.0
        agents[1].partner_count[2] = 1
        push!(agents[2].active_matches, ActiveMatch(8, false, :self))

        TransientBrokerage.rebuild_current_match_index!(ws, agents)
        TransientBrokerage.reset_offer_book!(ws, p.N)
        sent = TransientBrokerage.append_self_search_offers!(
            ws,
            agents[1],
            1,
            agents,
            G,
            state.broker.node_id,
            Int[],
            -1e9;
            current_match_index_ready=true,
        )

        @test sent == 1
        @test ws.offers[1].to_id == 2
    end

    @testset "Period stranger pool has fixed default size" begin
        p = default_params(N=80, T=5, T_burn=1, seed=13)
        state = initialize_model(p)
        strangers = state.workspace.period_strangers

        TransientBrokerage.sample_period_strangers!(
            strangers, p.N, p.n_strangers, StableRNG(701)
        )

        @test p.n_strangers == 10
        @test length(strangers) == 10
        @test length(unique(strangers)) == 10
        @test all(x -> 1 <= x <= p.N, strangers)
    end

    @testset "Offer book stores one unordered pair for reciprocal offers" begin
        ws = TransientBrokerage.SimWorkspace()
        TransientBrokerage.reset_offer_book!(ws, 5)

        @test TransientBrokerage.add_offer!(ws, 1, 2, :self, 3.0)
        @test TransientBrokerage.add_offer!(ws, 2, 1, :broker, 4.0)
        @test !TransientBrokerage.add_offer!(ws, 1, 2, :self, 5.0)
        @test length(ws.offers) == 2
        @test ws.offer_pairs == [(1, 2)]
    end

    @testset "Broker offers use one shared unordered-pair ranking" begin
        p = default_params(N=20, T=5, T_burn=1, K=1, seed=17)
        state = initialize_model(p)
        broker = state.broker
        ws = state.workspace

        empty!(broker.roster)
        empty!(broker.current_clients)
        union!(broker.current_clients, [1, 2])
        TransientBrokerage.reset_offer_book!(ws, p.N)

        sent = TransientBrokerage.append_broker_offers!(
            ws, [1, 2], [:broker, :broker], [1, 1], state.agents, broker, p, -1e9
        )

        @test sent == 2
        @test Set((o.from_id, o.to_id, o.channel) for o in ws.offers) ==
            Set([(1, 2, :broker), (2, 1, :broker)])
        @test ws.offer_pairs == [(1, 2)]
    end

    @testset "Broker offers do not use period strangers as access" begin
        p = default_params(N=20, T=5, T_burn=1, K=1, seed=19)
        state = initialize_model(p)
        broker = state.broker
        ws = state.workspace

        empty!(broker.roster)
        empty!(broker.current_clients)
        push!(broker.current_clients, 1)
        empty!(ws.period_strangers)
        push!(ws.period_strangers, 4)
        TransientBrokerage.reset_offer_book!(ws, p.N)

        sent = TransientBrokerage.append_broker_offers!(
            ws, [1], [:broker], [1], state.agents, broker, p, -1e9
        )

        @test sent == 0
        @test isempty(ws.offers)
    end
end
