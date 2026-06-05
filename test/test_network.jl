using Test
using TransientBrokerage
using TransientBrokerage: Agent, NNGradBuffers, add_broker_edge!, add_entrant_edges!
using TransientBrokerage: add_match_edge!, build_network, init_neural_net
using TransientBrokerage: agent_hidden_width
using TransientBrokerage: remove_agent_edges!
using Graphs: nv, ne, neighbors, has_edge, degree
using StableRNGs: StableRNG
using LinearAlgebra: normalize

@testset "Network" begin
    N = 20

    @testset "build_network creates N+1 nodes" begin
        G = build_network(N, 6, 0.1, StableRNG(42))
        @test nv(G) == N + 1
        @test degree(G, N + 1) == 0  # broker initially isolated
        @test all(degree(G, i) >= 1 for i in 1:N)
    end

    @testset "add_match_edge! is idempotent" begin
        G = build_network(N, 6, 0.1, StableRNG(42))
        edges_before = ne(G)
        add_match_edge!(G, 1, 2)
        add_match_edge!(G, 1, 2)
        @test ne(G) - edges_before <= 1
        @test has_edge(G, 1, 2)
    end

    @testset "add_broker_edge!" begin
        G = build_network(N, 6, 0.1, StableRNG(42))
        add_broker_edge!(G, 5, N + 1)
        @test has_edge(G, 5, N + 1)
    end

    @testset "remove_agent_edges!" begin
        G = build_network(N, 6, 0.1, StableRNG(42))
        @test degree(G, 1) > 0
        remove_agent_edges!(G, 1)
        @test degree(G, 1) == 0
    end

    @testset "add_entrant_edges! by type proximity" begin
        rng = StableRNG(42)
        G = build_network(N, 6, 0.1, rng)
        types = [normalize(randn(rng, 8)) for _ in 1:N]
        # Build minimal Agent structs for the new add_entrant_edges! signature
        d = 8
        h_agent = agent_hidden_width(d)
        mock_agents = [
            Agent(
                id=i,
                type=types[i],
                history_X=Matrix{Float64}(undef, d, 0),
                history_q=Float64[],
                nn=init_neural_net(d, h_agent, rng),
                nn_grad=NNGradBuffers(init_neural_net(d, h_agent, rng)),
                predict_buf=zeros(h_agent),
                partner_sum=zeros(N),
                partner_count=zeros(Int, N),
            ) for i in 1:N
        ]
        remove_agent_edges!(G, 1)
        add_entrant_edges!(G, 1, types[1], mock_agents, rng; n_edges=3)
        @test degree(G, 1) == 3
        @test !has_edge(G, 1, 1)  # no self-edge
    end

    @testset "Network measures" begin
        G = build_network(5, 4, 0.0, StableRNG(42))
        for i in 1:5
            add_broker_edge!(G, i, 6)
        end
        bc = compute_betweenness(G, 6)
        @test isfinite(bc)
        @test 0.0 <= bc <= 1.0  # betweenness must be in [0, 1]
        @test isfinite(compute_burt_constraint(G, 6))
        @test isfinite(compute_effective_size(G, 6))
    end
end
