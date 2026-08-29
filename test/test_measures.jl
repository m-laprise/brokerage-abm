using Test
using BrokerageABM
using BrokerageABM: update_cached_network_measures!
using StableRNGs: StableRNG
using Statistics: mean

@testset "Measures" begin
    using Graphs: SimpleGraph, add_edge!, nv, path_graph, star_graph, cycle_graph

    # ─── Prediction quality ─────────────────────────────────────────────

    @testset "compute_prediction_quality: perfect prediction" begin
        pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        pq = compute_prediction_quality(pred, real, StableRNG(101))
        @test pq isa PredictionQuality
        @test pq.r_squared ≈ 1.0
        @test pq.bias ≈ 0.0
        @test pq.rank_corr ≈ 1.0
    end

    @testset "compute_prediction_quality: constant prediction" begin
        pred = fill(3.0, 10)
        real = collect(1.0:10.0)
        pq = compute_prediction_quality(pred, real, StableRNG(102))
        pq_repeat = compute_prediction_quality(pred, real, StableRNG(102))
        random_ranks = [
            compute_prediction_quality(pred, real, StableRNG(seed)).rank_corr for
            seed in 1:200
        ]
        @test pq.r_squared < 0.0  # worse than mean
        @test isapprox(pq.bias, 3.0 - 5.5; atol=0.01)
        @test isfinite(pq.rank_corr)
        @test pq.rank_corr == pq_repeat.rank_corr
        @test abs(mean(random_ranks)) < 0.08
        @test length(unique(random_ranks)) > 1
    end

    @testset "random strict ranks preserve prediction ordering" begin
        predicted = [1.0, 2.0, 2.0, 4.0]
        order = zeros(Int, 4)
        ranks = zeros(4)
        BrokerageABM.strict_random_ranks_prefix!(
            ranks, order, predicted, length(predicted), StableRNG(202)
        )

        @test ranks[1] == 1.0
        @test Set(ranks[2:3]) == Set([2.0, 3.0])
        @test ranks[4] == 4.0
    end

    @testset "compute_prediction_quality: uses standard R² denominator" begin
        pred = fill(3.0, 5)
        real = collect(1.0:5.0)
        pq = compute_prediction_quality(pred, real, StableRNG(103))
        @test pq.r_squared ≈ 0.0
        @test pq.bias ≈ 0.0
        @test isfinite(pq.rank_corr)
        @test -1.0 <= pq.rank_corr <= 1.0
    end

    @testset "compute_prediction_quality: accepts views without copying" begin
        pred = collect(0.0:6.0)
        real = collect(1.0:7.0)
        pq = compute_prediction_quality(@view(pred[2:6]), @view(real[2:6]), StableRNG(104))
        @test pq.r_squared ≈ 0.5
        @test pq.bias ≈ -1.0
        @test pq.rank_corr ≈ 1.0
    end

    @testset "compute_prediction_quality: perfect rank, wrong scale" begin
        pred = [10.0, 20.0, 30.0, 40.0, 50.0]
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        pq = compute_prediction_quality(pred, real, StableRNG(105))
        @test pq.rank_corr ≈ 1.0
        @test pq.r_squared < 1.0  # scale is off, MSE > 0
    end

    @testset "compute_prediction_quality: too few observations -> NaN" begin
        pq = compute_prediction_quality([1.0, 2.0], [1.0, 2.0], StableRNG(106))
        @test isnan(pq.r_squared)
        @test isnan(pq.bias)
        @test isnan(pq.rank_corr)
    end

    @testset "compute_prediction_quality: low variance -> NaN" begin
        pred = [1.0, 1.0, 1.0, 1.0, 1.0]
        real = [1.0, 1.001, 1.0, 0.999, 1.0]  # variance < threshold
        pq = compute_prediction_quality(pred, real, StableRNG(107))
        @test isnan(pq.r_squared)
    end

    @testset "compute_prediction_quality_with_true_ranks! matches public API" begin
        pred = [1.0, 2.5, 2.5, 4.0, 5.0, 6.5]
        real = [1.5, 2.0, 2.0, 4.5, 5.5, 6.0]
        pred_order = zeros(Int, length(pred))
        true_order = zeros(Int, length(real))
        pred_ranks = zeros(length(pred))
        true_ranks = zeros(length(real))

        BrokerageABM.prepare_true_ranks!(real, length(real), true_order, true_ranks)
        pq_prefix = BrokerageABM.compute_prediction_quality_with_true_ranks!(
            pred,
            real,
            length(pred);
            rng=StableRNG(108),
            sigma_eps=0.10,
            pred_order=pred_order,
            pred_ranks=pred_ranks,
            true_ranks=true_ranks,
        )
        pq_public = compute_prediction_quality(pred, real, StableRNG(108); sigma_eps=0.10)

        @test pq_prefix.r_squared ≈ pq_public.r_squared atol=1e-12
        @test pq_prefix.bias ≈ pq_public.bias atol=1e-12
        @test pq_prefix.rank_corr ≈ pq_public.rank_corr atol=1e-12
    end

    # ─── Betweenness centrality ──────────────────────────────────────────

    @testset "betweenness on star graph: center node" begin
        # Star with 5 leaves: center has betweenness 1.0
        G = star_graph(6)  # node 1 is center
        bc = compute_betweenness(G, 1)
        @test isapprox(bc, 1.0; atol=0.01)
    end

    @testset "betweenness on star graph: leaf node" begin
        G = star_graph(6)
        bc = compute_betweenness(G, 2)  # leaf
        @test isapprox(bc, 0.0; atol=0.01)
    end

    @testset "betweenness on path graph: middle node" begin
        # Path 1-2-3-4-5: node 3 has highest betweenness
        G = path_graph(5)
        bc3 = compute_betweenness(G, 3)
        bc1 = compute_betweenness(G, 1)
        @test bc3 > bc1
        # Node 3 lies on 4 of the 6 shortest paths between non-adjacent pairs
        # (1-4, 1-5, 2-4, 2-5) plus is endpoint for 4-5 and 1-2 (doesn't count).
        # Raw = 8 (counted from both BFS directions), norm = (n-1)*(n-2) = 12
        @test isapprox(bc3, 2.0/3; atol=0.05)
    end

    @testset "betweenness on complete graph: all equal and low" begin
        G = SimpleGraph(4)
        for i in 1:4, j in (i + 1):4
            add_edge!(G, i, j)
        end
        bc_vals = [compute_betweenness(G, i) for i in 1:4]
        # Complete graph: no node is "between" others (direct edges exist)
        @test all(isapprox(bc, 0.0; atol=0.01) for bc in bc_vals)
    end

    @testset "betweenness: two-node graph" begin
        G = SimpleGraph(2)
        add_edge!(G, 1, 2)
        @test compute_betweenness(G, 1) == 0.0
    end

    @testset "betweenness: isolated node" begin
        G = SimpleGraph(3)
        add_edge!(G, 1, 2)
        @test compute_betweenness(G, 3) == 0.0
    end

    @testset "betweenness on path: endpoints are zero" begin
        G = path_graph(5)
        @test isapprox(compute_betweenness(G, 1), 0.0; atol=0.01)
        @test isapprox(compute_betweenness(G, 5), 0.0; atol=0.01)
    end

    @testset "betweenness on path: symmetry" begin
        G = path_graph(5)
        @test isapprox(compute_betweenness(G, 2), compute_betweenness(G, 4); atol=1e-10)
    end

    @testset "betweenness on path: exact values (n=5)" begin
        # Path 1-2-3-4-5, n=5, norm = (n-1)(n-2) = 12
        # Node 2: on paths (1,3),(1,4),(1,5) = 3 pairs, raw from both directions = 6
        # Normalized: 6/12 = 0.5
        G = path_graph(5)
        @test isapprox(compute_betweenness(G, 2), 0.5; atol=1e-10)
        # Node 3: on paths (1,4),(1,5),(2,4),(2,5) = 4 pairs, raw = 8
        # Normalized: 8/12 = 2/3
        @test isapprox(compute_betweenness(G, 3), 2.0/3; atol=1e-10)
    end

    @testset "betweenness: bridge node in barbell graph" begin
        # Two K3 cliques (1,2,3) and (5,6,7) connected by bridge node 4
        # 4 connects to 3 and 5. All paths between cliques go through 4.
        G = SimpleGraph(7)
        # Clique 1: 1-2, 1-3, 2-3
        add_edge!(G, 1, 2);
        add_edge!(G, 1, 3);
        add_edge!(G, 2, 3)
        # Bridge: 3-4, 4-5
        add_edge!(G, 3, 4);
        add_edge!(G, 4, 5)
        # Clique 2: 5-6, 5-7, 6-7
        add_edge!(G, 5, 6);
        add_edge!(G, 5, 7);
        add_edge!(G, 6, 7)
        bc4 = compute_betweenness(G, 4)
        # Node 4 is on ALL 9 cross-clique shortest paths (3×3 pairs)
        # plus paths within each clique that route through 4 (none, since cliques are complete)
        # and paths between 3 and {5,6,7} and between {1,2,3} and 5
        # Exact: node 4 is on all s-t paths where s∈{1,2,3} and t∈{5,6,7} (9 pairs)
        # plus s=3,t∈{5,6,7} routes through 4 (3 of which overlap with above)
        # Raw from both directions: 18. Norm = (7-1)(7-2) = 30.
        @test bc4 > 0.5  # bridge node has high betweenness
        # Non-bridge nodes in a clique should have lower betweenness
        @test compute_betweenness(G, 1) < bc4
    end

    @testset "betweenness: disconnected graph" begin
        # Two isolated components: 1-2-3 and 4-5-6
        G = SimpleGraph(6)
        add_edge!(G, 1, 2);
        add_edge!(G, 2, 3)
        add_edge!(G, 4, 5);
        add_edge!(G, 5, 6)
        # Node 2: on path 1-3 only (can't reach 4,5,6)
        # Raw = 2 (from both directions), norm = (6-1)(6-2) = 20
        @test isapprox(compute_betweenness(G, 2), 2.0/20; atol=1e-10)
        # Node 5: same by symmetry
        @test isapprox(compute_betweenness(G, 5), 2.0/20; atol=1e-10)
    end

    @testset "betweenness: single node graph" begin
        G = SimpleGraph(1)
        @test compute_betweenness(G, 1) == 0.0
    end

    @testset "betweenness: allocation-free after warmup (N=100)" begin
        G = SimpleGraph(100)
        for i in 1:99
            ;
            add_edge!(G, i, i+1);
        end
        compute_betweenness(G, 50)  # warmup (allocates workspaces)
        a = @allocated compute_betweenness(G, 50)
        # CSR build + partial_bc zeros are O(n+m), small and bounded.
        # No per-source allocations.
        @test a < 100_000  # < 100 KB for a 100-node graph
    end

    # ─── Burt's constraint ───────────────────────────────────────────────

    @testset "constraint on star graph: center" begin
        G = star_graph(6)
        c = compute_burt_constraint(G, 1)
        # Center connected to 5 leaves, no edges among leaves
        # p_ij = 1/5, c_ij = p_ij (no indirect paths), C = 5 * (1/5)^2 = 1/5
        @test isapprox(c, 0.2; atol=0.01)
    end

    @testset "constraint on star graph: leaf" begin
        G = star_graph(6)
        c = compute_burt_constraint(G, 2)
        # Leaf has degree 1, p = 1, c_ij = 1, C = 1
        @test isapprox(c, 1.0; atol=0.01)
    end

    @testset "constraint on isolated node" begin
        G = SimpleGraph(3)
        add_edge!(G, 1, 2)
        @test compute_burt_constraint(G, 3) == 1.0  # isolated returns 1.0
    end

    @testset "constraint on complete graph K4" begin
        G = SimpleGraph(4)
        for i in 1:4, j in (i + 1):4
            ;
            add_edge!(G, i, j);
        end
        c = compute_burt_constraint(G, 1)
        # K4: each node has 3 neighbors, all interconnected
        # p = 1/3; c_ij = 1/3 + 2*(1/3)*(1/3) = 1/3 + 2/9 = 5/9
        # C = 3 * (5/9)^2 = 3 * 25/81 = 75/81 ≈ 0.926
        @test isapprox(c, 75.0/81; atol=0.01)
    end

    # ─── Effective size ──────────────────────────────────────────────────

    @testset "effective size on star graph: center" begin
        G = star_graph(6)
        es = compute_effective_size(G, 1)
        # Center: 5 neighbors, no edges among them -> redundancy = 0 -> ES = 5
        @test isapprox(es, 5.0; atol=0.01)
    end

    @testset "effective size on star graph: leaf" begin
        G = star_graph(6)
        es = compute_effective_size(G, 2)
        # Leaf: 1 neighbor -> ES = 1 (minimal)
        @test isapprox(es, 1.0; atol=0.01)
    end

    @testset "effective size on isolated node" begin
        G = SimpleGraph(3)
        add_edge!(G, 1, 2)
        @test compute_effective_size(G, 3) == 0.0
    end

    @testset "effective size on complete graph K4" begin
        G = SimpleGraph(4)
        for i in 1:4, j in (i + 1):4
            ;
            add_edge!(G, i, j);
        end
        es = compute_effective_size(G, 1)
        # K4: d=3, t=3 (all 3 neighbor pairs connected)
        # ES = 3 - 2*3/3 = 1.0 (Borgatti 1997)
        @test isapprox(es, 1.0; atol=0.01)
    end

    @testset "effective size and constraint: Muscillo (2021) Fig. 1" begin
        # 7-node graph from Muscillo (2021): A=1,B=2,C=3,D=4,E=5,F=6,G=7
        G = SimpleGraph(7)
        add_edge!(G, 1, 2);
        add_edge!(G, 1, 5);
        add_edge!(G, 1, 6);
        add_edge!(G, 1, 7)
        add_edge!(G, 2, 4);
        add_edge!(G, 2, 7)
        add_edge!(G, 3, 7);
        add_edge!(G, 4, 7);
        add_edge!(G, 5, 7);
        add_edge!(G, 6, 7)

        # Effective size from the paper (Table on p.4)
        @test isapprox(compute_effective_size(G, 1), 2.5; atol=0.001)   # A
        @test isapprox(compute_effective_size(G, 2), 5.0/3; atol=0.001)  # B
        @test isapprox(compute_effective_size(G, 3), 1.0; atol=0.001)   # C
        @test isapprox(compute_effective_size(G, 4), 1.0; atol=0.001)   # D
        @test isapprox(compute_effective_size(G, 5), 1.0; atol=0.001)   # E
        @test isapprox(compute_effective_size(G, 6), 1.0; atol=0.001)   # F
        @test isapprox(compute_effective_size(G, 7), 14.0/3; atol=0.001)  # G

        # Constraint: nodes with high effective size should have low constraint
        @test compute_burt_constraint(G, 7) < compute_burt_constraint(G, 3)
        @test compute_burt_constraint(G, 1) < compute_burt_constraint(G, 4)
    end

    @testset "constraint: hand-calculated example with unequal degrees" begin
        # Graph: 1-2, 1-4, 2-3, 2-4. Node 1 has d=2, node 2 has d=3, node 4 has d=2.
        # From hand calculation (Ruqin Ren):
        # c_12 = (1/2 + (1/2)*(1/2))^2 = (3/4)^2 = 9/16 (indirect via 4: p14*p42 = 0.5*0.5)
        # c_14 = (1/2 + (1/2)*(1/3))^2 = (2/3)^2 = 4/9 (indirect via 2: p12*p24 = 0.5*1/3)
        # C_1 = 9/16 + 4/9 = 1.007
        G = SimpleGraph(4)
        add_edge!(G, 1, 2);
        add_edge!(G, 1, 4);
        add_edge!(G, 2, 3);
        add_edge!(G, 2, 4)
        @test isapprox(compute_burt_constraint(G, 1), 9.0/16 + 4.0/9; atol=0.001)
    end

    # ─── update_cached_network_measures! ─────────────────────────────────

    @testset "update_cached_network_measures! populates all fields" begin
        p2 = default_params(N=50, seed=42)
        state = initialize_model(p2)
        update_cached_network_measures!(state)
        @test isfinite(state.cached_network.betweenness)
        @test isfinite(state.cached_network.constraint)
        @test isfinite(state.cached_network.effective_size)
        @test 0.0 <= state.cached_network.betweenness <= 1.0
    end
end
