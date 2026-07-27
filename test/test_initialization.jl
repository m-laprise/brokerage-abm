using Test
using BrokerageABM
using BrokerageABM: CalibrationConstants, MatchingEnv
using BrokerageABM: agent_hidden_width, broker_hidden_width, broker_pair_feature_dim
using StableRNGs: StableRNG

@testset "Initialization" begin
    using Graphs: nv, ne, degree, has_edge, neighbors
    using LinearAlgebra: norm, issymmetric, isposdef, tr, eigvals
    using Statistics: mean

    p = default_params(N=100, seed=42)
    state = initialize_model(p)
    N = p.N;
    d = p.d

    @testset "initialize_model returns valid ModelState" begin
        @test state isa ModelState
        @test state.env isa MatchingEnv
        @test state.cal isa CalibrationConstants
        @test state.period == 0
        @test length(state.agents) == N
        @test state.broker.node_id == N + 1
    end

    @testset "calibration constants are positive and consistent" begin
        @test state.cal.q_cal > 0
        @test state.cal.r > 0
        @test state.cal.phi > 0
        @test state.cal.c_s >= 0
        @test state.cal.r < state.cal.q_cal
        @test state.cal.r ≈ state.params.reservation_frac * state.cal.q_cal
    end

    @testset "parameter validation rejects invalid Watts-Strogatz degree" begin
        @test_throws AssertionError default_params(N=10, k=10)
    end

    @testset "agent types are unit vectors in R^d" begin
        @test all(length(a.type) == d for a in state.agents)
        @test all(isapprox(norm(a.type), 1.0; atol=1e-10) for a in state.agents)
        @test all(all(isfinite, a.type) for a in state.agents)
    end

    @testset "agent IDs match indices" begin
        @test all(state.agents[i].id == i for i in 1:N)
    end

    @testset "matching environment matrices have the intended geometry" begin
        env = state.env
        @test size(env.A) == (d, d)
        @test size(env.B) == (d, d)
        @test issymmetric(env.A)
        @test issymmetric(env.B)
        @test isposdef(env.A)
        # Trace normalization: tr(A) ≈ d (so E[x'Ax] ≈ 1 for unit vectors)
        @test isapprox(tr(env.A), Float64(d); atol=1e-10)
        @test minimum(eigvals(env.B)) < 0.0
        @test maximum(eigvals(env.B)) > 0.0
        types = [agent.type for agent in state.agents]
        @test BrokerageABM.weighted_regime_overlap(env.A, env.B, types) ≈ 0.0 atol=1e-12
        @test length(env.c) == d
        @test all(isfinite, env.c)
    end

    @testset "network has correct structure" begin
        G = state.G
        @test nv(G) == N + 1  # N agents + 1 broker
        # No self-edges
        @test !any(has_edge(G, i, i) for i in 1:N)
        # All agents have at least 1 edge (from WS initialization)
        @test all(degree(G, i) >= 1 for i in 1:N)
    end

    @testset "broker roster seeded correctly" begin
        roster = state.broker.roster
        expected_size = BrokerageABM.roster_target_size(N)
        @test length(roster) == expected_size
        @test all(1 <= rid <= N for rid in roster)
        # Roster members have broker edge
        @test all(has_edge(state.G, rid, state.broker.node_id) for rid in roster)
    end

    @testset "broker history seeded" begin
        roster_list = sort(collect(state.broker.roster))
        expected_broker_seed = count(
            has_edge(state.G, roster_list[i], roster_list[j]) for
            i in 1:length(roster_list) for j in (i + 1):length(roster_list)
        )
        @test state.broker.history_count == min(100, expected_broker_seed)
        @test state.broker.n_new_obs == 0  # training consumed them
    end

    @testset "agent histories seeded from all initial neighbors" begin
        expected_history_counts = [
            count(j -> 1 <= j <= N, neighbors(state.G, i)) for i in 1:N
        ]
        @test [a.history_count for a in state.agents] == expected_history_counts
        @test all(
            state.agents[i].partner_count[j] == 1 for i in 1:N for
            j in neighbors(state.G, i) if 1 <= j <= N
        )
        # n_new_obs was reset after initial training
        @test all(a.n_new_obs == 0 for a in state.agents)
    end

    @testset "neural networks initialized and trained" begin
        h_agent = agent_hidden_width(p)
        h_broker = broker_hidden_width(p)
        d_broker = broker_pair_feature_dim(d)
        @test all(size(a.nn.W1) == (h_agent, d) for a in state.agents)
        @test all(length(a.nn.b1) == h_agent for a in state.agents)
        @test all(length(a.nn.w2) == h_agent for a in state.agents)
        @test all(isfinite(a.nn.b2) && all(isfinite, a.nn.W1) for a in state.agents)
        @test all(
            size(a.train_X, 1) == d && length(a.train_q) == size(a.train_X, 2) for
            a in state.agents
        )
        @test size(state.broker.nn.W1) == (h_broker, d_broker)
        @test isfinite(state.broker.nn.b2)
    end

    @testset "satisfaction initialized from seed data" begin
        # Self-satisfaction set from mean of seed match outcomes
        @test all(a.satisfaction_self != 0.0 for a in state.agents if a.history_count > 0)
        @test all(isfinite(a.satisfaction_self) for a in state.agents)
        # Broker satisfaction set to broker reputation (market prior)
        @test all(
            a.satisfaction_broker == state.broker.last_reputation for a in state.agents
        )
        @test all(!a.tried_broker for a in state.agents)
        @test all(a.periods_alive == 0 for a in state.agents)
        # Broker reputation set from seed data when roster-roster ties exist
        if state.broker.history_count > 0
            @test state.broker.has_had_clients == true
            @test state.broker.last_reputation > 0.0
        else
            @test state.broker.has_had_clients == false
            @test state.broker.last_reputation == 0.0
        end
    end

    @testset "curve_point returns unit vectors" begin
        geo = state.curve_geo
        points = [
            BrokerageABM.curve_point(t, geo) for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        @test all(length(cp) == d for cp in points)
        @test all(isapprox(norm(cp), 1.0; atol=1e-10) for cp in points)
    end

    @testset "generate_agent_types produces N unit vectors" begin
        rng = StableRNG(123)
        geo = BrokerageABM.generate_curve_geometry(d, p.s, rng)
        types, inv = BrokerageABM.generate_agent_types(N, geo, p.sigma_x, rng)
        @test length(types) == N
        @test all(length(t) == d for t in types)
        @test all(isapprox(norm(t), 1.0; atol=1e-10) for t in types)
        # Default sort_by_pc1=false: inv_order is identity
        @test inv == collect(1:N)
    end

    @testset "sort_by_pc1 option produces sorted types" begin
        rng1 = StableRNG(7)
        geo = BrokerageABM.generate_curve_geometry(d, p.s, rng1)
        types_unsorted, _ = BrokerageABM.generate_agent_types(N, geo, p.sigma_x, rng1)

        rng2 = StableRNG(7)
        geo2 = BrokerageABM.generate_curve_geometry(d, p.s, rng2)
        types_sorted, inv = BrokerageABM.generate_agent_types(
            N, geo2, p.sigma_x, rng2; sort_by_pc1=true
        )

        @test length(types_sorted) == N
        # The sorted and unsorted should contain the same types (in different order)
        @test Set(types_sorted) == Set(types_unsorted)
        # inv_order should be a valid permutation
        @test sort(inv) == collect(1:N)
    end

    @testset "deterministic initialization: same seed -> identical state" begin
        s1 = initialize_model(default_params(N=50, seed=42))
        s2 = initialize_model(default_params(N=50, seed=42))
        @test all(s1.agents[i].type == s2.agents[i].type for i in 1:50)
        @test s1.env.A == s2.env.A
        @test s1.env.B == s2.env.B
        @test s1.env.c == s2.env.c
        @test s1.cal.q_cal == s2.cal.q_cal
        @test s1.broker.history_count == s2.broker.history_count
        @test s1.agents[1].nn.W1 == s2.agents[1].nn.W1
    end

    @testset "different seeds produce different states" begin
        s1 = initialize_model(default_params(N=50, seed=42))
        s2 = initialize_model(default_params(N=50, seed=99))
        @test s1.agents[1].type != s2.agents[1].type
        @test s1.env.A != s2.env.A
    end
end
