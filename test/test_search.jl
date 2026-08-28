using Test
using BrokerageABM
using BrokerageABM: ActiveMatch, add_match_edge!, remove_agent_edges!
using StableRNGs: StableRNG

function set_constant_prediction!(nn, value::Float64)
    nn.W1 .= 0.0
    nn.b1 .= 1.0
    nn.w2 .= 0.0
    nn.b2 = value
    return nothing
end

@testset "Search" begin
    @testset "Self-search offers rank known neighbors by value" begin
        p = default_params(N=20, T=5, n_strangers=0, seed=7)
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

        BrokerageABM.rebuild_current_match_index!(ws, agents)
        BrokerageABM.reset_offer_book!(ws, p.N)
        sent = BrokerageABM.append_self_search_offers!(
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
        @test [(o.to_id, o.predicted_value) for o in ws.offer_book.offers] == [(2, 9.0), (3, 7.0)]
    end

    @testset "Self-search predicts for unobserved graph neighbors" begin
        p = default_params(N=20, T=5, n_strangers=0, seed=8)
        state = initialize_model(p)
        agents = state.agents
        G = state.G
        ws = state.workspace

        remove_agent_edges!(G, 1)
        add_match_edge!(G, 1, 2)
        agents[1].partner_sum[2] = 0.0
        agents[1].partner_count[2] = 0
        set_constant_prediction!(agents[1].nn, state.cal.r + 1.0)

        BrokerageABM.rebuild_current_match_index!(ws, agents)
        BrokerageABM.reset_offer_book!(ws, p.N)
        sent = BrokerageABM.append_self_search_offers!(
            ws,
            agents[1],
            1,
            agents,
            G,
            state.broker.node_id,
            Int[],
            state.cal.r;
            current_match_index_ready=true,
        )

        @test sent == 1
        @test [(o.to_id, o.predicted_value) for o in ws.offer_book.offers] ==
            [(2, state.cal.r + 1.0)]
    end

    @testset "Self-search no longer filters candidates by K capacity" begin
        p = default_params(N=20, T=5, K=1, n_strangers=0, seed=11)
        state = initialize_model(p)
        agents = state.agents
        G = state.G
        ws = state.workspace

        remove_agent_edges!(G, 1)
        add_match_edge!(G, 1, 2)
        agents[1].partner_sum[2] = 100.0
        agents[1].partner_count[2] = 1
        push!(agents[2].active_matches, ActiveMatch(8, :self))

        BrokerageABM.rebuild_current_match_index!(ws, agents)
        BrokerageABM.reset_offer_book!(ws, p.N)
        sent = BrokerageABM.append_self_search_offers!(
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
        @test ws.offer_book.offers[1].to_id == 2
    end

    @testset "Period stranger pool has fixed default size" begin
        p = default_params(N=80, T=5, seed=13)
        state = initialize_model(p)
        strangers = state.workspace.search.period_strangers

        BrokerageABM.sample_period_strangers!(
            strangers, p.N, p.n_strangers, StableRNG(701)
        )

        @test p.n_strangers == 10
        @test length(strangers) == 10
        @test length(unique(strangers)) == 10
        @test all(x -> 1 <= x <= p.N, strangers)
    end

    @testset "Self-search demanders draw independent stranger pools" begin
        p = default_params(N=20, T=5, K=1, n_strangers=3, seed=19)
        state = initialize_model(p)
        agents = state.agents
        G = state.G
        ws = state.workspace

        for i in 1:p.N
            remove_agent_edges!(G, i)
            set_constant_prediction!(agents[i].nn, state.cal.r + 10.0)
        end

        demand_agent_ids = [1, 2]
        demand_channels = [:self, :self]
        demand_counts = [1, 1]
        expected_rng = StableRNG(901)
        pools = [
            copy(
                BrokerageABM.sample_period_strangers!(
                    Int[], p.N, p.n_strangers, expected_rng
                ),
            ) for _ in 1:2
        ]
        expected_targets = [
            first(j for j in pools[idx] if j != demand_agent_ids[idx]) for idx in 1:2
        ]

        BrokerageABM.run_offer_market!(
            demand_agent_ids,
            demand_channels,
            demand_counts,
            agents,
            state.broker,
            state.env,
            G,
            p,
            state.cal,
            StableRNG(901);
            ws=ws,
            accepted_matches=BrokerageABM.AcceptedMatch[],
        )

        @test pools[1] != pools[2]
        @test [(o.from_id, o.to_id) for o in ws.offer_book.offers] == collect(zip(demand_agent_ids, expected_targets))
    end

    @testset "Offer book stores one unordered pair for reciprocal offers" begin
        ws = BrokerageABM.SimWorkspace()
        offer_book = ws.offer_book
        BrokerageABM.reset_offer_book!(offer_book, 5)

        @test BrokerageABM.add_offer!(offer_book, 1, 2, :self, 3.0)
        @test BrokerageABM.add_offer!(offer_book, 2, 1, :broker, 4.0)
        @test !BrokerageABM.add_offer!(offer_book, 1, 2, :self, 5.0)
        @test BrokerageABM.offer_ids(offer_book, 1, 2) == (1, 2)
        @test BrokerageABM.offer_at(offer_book, 1).predicted_value == 3.0
        @test length(offer_book.offers) == 2
        @test offer_book.offer_pairs == [(1, 2)]
    end

    @testset "Broker offers use per-client quota-bounded ranking" begin
        p = default_params(N=20, T=5, K=1, seed=17)
        state = initialize_model(p)
        broker = state.broker
        ws = state.workspace

        empty!(broker.roster)
        empty!(broker.current_clients)
        union!(broker.current_clients, [1, 2])
        BrokerageABM.reset_offer_book!(ws, p.N)

        sent = BrokerageABM.append_broker_offers!(
            ws, [1, 2], [:broker, :broker], [1, 1], state.agents, broker, p, -1e9
        )

        @test sent == 2
        @test Set((o.from_id, o.to_id, o.channel) for o in ws.offer_book.offers) ==
            Set([(1, 2, :broker), (2, 1, :broker)])
        @test ws.offer_book.offer_pairs == [(1, 2)]
    end

    @testset "Broker offers do not use period strangers as access" begin
        p = default_params(N=20, T=5, K=1, seed=19)
        state = initialize_model(p)
        broker = state.broker
        ws = state.workspace

        empty!(broker.roster)
        empty!(broker.current_clients)
        push!(broker.current_clients, 1)
        empty!(ws.search.period_strangers)
        push!(ws.search.period_strangers, 4)
        BrokerageABM.reset_offer_book!(ws, p.N)

        sent = BrokerageABM.append_broker_offers!(
            ws, [1], [:broker], [1], state.agents, broker, p, -1e9
        )

        @test sent == 0
        @test isempty(ws.offer_book.offers)
    end
end
