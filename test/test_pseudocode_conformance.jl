using Test
using BrokerageABM
using BrokerageABM: ActiveMatch, AcceptedMatch, CalibrationConstants
using BrokerageABM: MatchingEnv, NeuralNet, OfferCredit
using BrokerageABM: add_match_edge!, broker_access_size
using BrokerageABM: has_current_match, partner_mean
using BrokerageABM: record_agent_history!, record_broker_history!
using BrokerageABM: remove_agent_edges!, update_broker_reputation!
using BrokerageABM: update_partner_mean!
using BrokerageABM: update_satisfaction!
using Graphs: degree, has_edge, neighbors, nv
using LinearAlgebra: I, norm
using StableRNGs: StableRNG
using Statistics: mean

# Mechanism-level conformance tests for `paper/appendices/simulation_pseudocode.tex`. Small,
# deterministic fixtures make initialization, matching, updating, and metrics
# auditable independently of regression and distributional tests.

function pseudocode_constant_prediction!(nn::NeuralNet, value::Float64)
    fill!(nn.W1, 0.0)
    fill!(nn.b1, 0.0)
    fill!(nn.w2, 0.0)
    nn.b2 = value
    return nothing
end

function linear_prediction!(nn::NeuralNet, weights::AbstractVector{<:Real})
    fill!(nn.W1, 0.0)
    fill!(nn.b1, 0.0)
    fill!(nn.w2, 0.0)
    @assert length(weights) == size(nn.W1, 2)
    nn.W1[1, :] .= weights
    nn.w2[1] = 1.0
    nn.b2 = 0.0
    return nothing
end

function deterministic_types()
    s2 = inv(sqrt(2.0))
    return [
        [1.0, 0.0],
        [0.8, 0.6],
        [0.6, 0.8],
        [0.0, 1.0],
        [s2, s2],
        [-1.0, 0.0],
        [0.0, -1.0],
        [-0.8, 0.6],
        [0.6, -0.8],
        [-s2, s2],
    ]
end

function configure_micro_state!(state)
    p = state.params
    types = deterministic_types()
    @inbounds for i in 1:p.N
        state.agents[i].type .= types[i]
    end

    state.env = MatchingEnv(
        2,
        0.0,
        zeros(2),
        Matrix{Float64}(I, 2, 2),
        zeros(2, 2),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    state.cal = CalibrationConstants(2.0, 1.2, 0.2, 0.2)

    for i in 1:p.N
        remove_agent_edges!(state.G, i)
    end
    empty!(state.broker.roster)
    empty!(state.broker.current_clients)

    for agent in state.agents
        empty!(agent.active_matches)
        agent.history_count = 0
        agent.n_new_obs = 0
        empty!(agent.obs_period_marks)
        push!(agent.obs_period_marks, 0)
        fill!(agent.partner_sum, 0.0)
        fill!(agent.partner_count, 0)
        agent.satisfaction_self = 2.0
        agent.satisfaction_broker = 1.0
        agent.tried_broker = false
        agent.periods_alive = 0
        agent.n_matches_any = 0
        pseudocode_constant_prediction!(agent.nn, 0.0)
    end

    broker = state.broker
    broker.history_count = 0
    broker.n_new_obs = 0
    empty!(broker.obs_period_marks)
    push!(broker.obs_period_marks, 0)
    broker.last_reputation = 1.0
    broker.has_had_clients = false
    pseudocode_constant_prediction!(broker.nn, 0.0)
    BrokerageABM.reset_accumulators!(state.accum)
    BrokerageABM.reset_offer_book!(state.workspace, p.N)
    BrokerageABM.reset_current_match_index!(state.workspace, p.N)
    return state
end

function micro_state(; seed::Int=314)
    p = default_params(;
        N=10,
        d=2,
        s=2,
        K=2,
        p_demand=1.0,
        k=2,
        sigma_eps=0.0,
        eta=0.0,
        roster_churn=0.0,
        n_strangers=0,
        E_init=1,
        network_measure_interval=1,
        T=3,
        seed=seed,
    )
    return configure_micro_state!(initialize_model(p))
end

function set_partner_mean!(state, i::Int, j::Int, q::Float64; count::Int=1)
    state.agents[i].partner_sum[j] = q * count
    state.agents[i].partner_count[j] = count
    return nothing
end

function reset_offer_workspace!(state)
    BrokerageABM.reset_offer_book!(state.workspace, state.params.N)
    BrokerageABM.rebuild_current_match_index!(state.workspace, state.agents)
    return nothing
end

function match_buffers(state)
    ws = state.workspace
    if length(ws.match_output.Ax_buf) != state.params.d
        ws.match_output.Ax_buf = Vector{Float64}(undef, state.params.d)
        ws.match_output.Bx_buf = Vector{Float64}(undef, state.params.d)
    end
    return (ws.match_output.Ax_buf, ws.match_output.Bx_buf)
end

function assert_symmetric_active_match(agents, i, j; channel::Symbol)
    mij = [am for am in agents[i].active_matches if am.partner_id == j]
    mji = [am for am in agents[j].active_matches if am.partner_id == i]
    @test length(mij) == 1
    @test length(mji) == 1
    @test mij[1].channel == channel
    @test mji[1].channel == channel
    return nothing
end

function run_accept_pair!(state, i::Int, j::Int, accepted=AcceptedMatch[])
    Ax_buf, Bx_buf = match_buffers(state)
    BrokerageABM.accept_offer_pair!(
        accepted,
        i,
        j,
        state.agents,
        state.broker,
        state.env,
        state.G,
        state.cal,
        StableRNG(901);
        Ax_buf=Ax_buf,
        Bx_buf=Bx_buf,
        ws=state.workspace,
    )
    return accepted
end

@testset "Pseudocode conformance" begin
    @testset "Initialization follows documented state construction" begin
        p = default_params(;
            N=20,
            d=2,
            s=2,
            k=2,
            sigma_eps=0.0,
            eta=0.0,
            roster_churn=0.0,
            n_strangers=0,
            E_init=1,
            T=3,
            seed=501,
        )
        state = initialize_model(p)
        broker = state.broker

        @test length(state.agents) == p.N
        @test broker.node_id == p.N + 1
        @test nv(state.G) == p.N + 1
        @test all(state.agents[i].id == i for i in 1:p.N)
        @test all(isapprox(norm(a.type), 1.0; atol=1e-10) for a in state.agents)
        @test length(broker.roster) == BrokerageABM.roster_target_size(p)
        @test all(has_edge(state.G, rid, broker.node_id) for rid in broker.roster)

        expected_history_counts = [
            count(j -> 1 <= j <= p.N, neighbors(state.G, i)) for i in 1:p.N
        ]
        seeded_agents = [a for a in state.agents if a.history_count > 0]
        roster_list = sort(collect(broker.roster))
        expected_broker_seed = count(
            has_edge(state.G, roster_list[i], roster_list[j]) for i in
                                                                  1:length(roster_list) for
            j in (i + 1):length(roster_list)
        )
        @test [a.history_count for a in state.agents] == expected_history_counts
        @test broker.history_count == min(100, expected_broker_seed)
        @test !isempty(seeded_agents)
        if broker.history_count > 0
            @test broker.last_reputation ≈ mean(broker.history_q[1:broker.history_count])
        else
            @test broker.last_reputation == 0.0
        end
        @test all(
            a.satisfaction_self ≈ mean(a.history_q[1:a.history_count]) for
            a in seeded_agents
        )
        @test all(a.satisfaction_broker ≈ broker.last_reputation for a in state.agents)
        @test !any(a.tried_broker for a in state.agents)
    end

    @testset "Training schedule and channel choice follow period rules" begin
        state = micro_state()
        agents = state.agents

        agents[1].satisfaction_self = 3.0
        agents[1].satisfaction_broker = 1.0
        agents[1].tried_broker = true
        @test BrokerageABM.outsourcing_decision(agents[1], 0.0, StableRNG(1)) == :self

        agents[2].satisfaction_self = 1.0
        agents[2].satisfaction_broker = 3.0
        agents[2].tried_broker = true
        @test BrokerageABM.outsourcing_decision(agents[2], 0.0, StableRNG(2)) ==
            :broker

        agents[3].satisfaction_self = 2.0
        agents[3].satisfaction_broker = 100.0
        agents[3].tried_broker = false
        @test BrokerageABM.outsourcing_decision(agents[3], 1.0, StableRNG(3)) == :self
        @test BrokerageABM.outsourcing_decision(agents[3], 3.0, StableRNG(4)) ==
            :broker

        agents[4].satisfaction_self = 1.0
        agents[4].satisfaction_broker = 1.0
        agents[4].tried_broker = true
        tie_seed = 44
        expected_tie = rand(StableRNG(tie_seed)) < 0.5 ? :self : :broker
        @test BrokerageABM.outsourcing_decision(
            agents[4], 0.0, StableRNG(tie_seed)
        ) == expected_tie

        @test all(
            BrokerageABM.agent_retrains_this_period(i, t) == (isodd(i) == isodd(t))
            for i in 1:4, t in 1:4
        )

        state = micro_state()
        for i in 1:state.params.N
            state.agents[i].satisfaction_self = 10.0
            state.agents[i].satisfaction_broker = 0.0
            state.agents[i].tried_broker = true
        end
        record_agent_history!(state.agents[1], state.agents[3].type, 2.0)
        record_agent_history!(state.agents[2], state.agents[3].type, 2.0)
        record_broker_history!(
            state.broker, state.agents[1].type, state.agents[2].type, 2.0
        )
        @test state.agents[1].n_new_obs == 1
        @test state.agents[2].n_new_obs == 1
        @test state.broker.n_new_obs == 1

        step_period!(state)

        @test state.agents[1].n_new_obs == 0
        @test state.agents[2].n_new_obs == 1
        @test state.broker.n_new_obs == 0
    end

    @testset "Self-search offer construction follows candidate and quota rules" begin
        state = micro_state()
        agents = state.agents
        G = state.G
        ws = state.workspace

        add_match_edge!(G, 1, 2)
        add_match_edge!(G, 1, 3)
        add_match_edge!(G, 1, 4)
        BrokerageABM.add_broker_edge!(G, 1, state.broker.node_id)
        set_partner_mean!(state, 1, 2, 3.0)
        set_partner_mean!(state, 1, 3, 2.2)
        set_partner_mean!(state, 1, 4, 1.0)
        push!(agents[1].active_matches, ActiveMatch(3, :self))
        pseudocode_constant_prediction!(agents[1].nn, 2.5)

        reset_offer_workspace!(state)

        sent = BrokerageABM.append_self_search_offers!(
            ws,
            agents[1],
            2,
            agents,
            G,
            state.broker.node_id,
            [1, 2, 5, 6, 7],
            state.cal.r;
            rng=StableRNG(8801),
            current_match_index_ready=true,
        )

        @test sent == 2
        @test (ws.offer_book.offers[1].to_id, ws.offer_book.offers[1].predicted_value) ==
            (2, 3.0)
        @test ws.offer_book.offers[2].to_id in (5, 6, 7)
        @test ws.offer_book.offers[2].predicted_value == 2.5
        excluded = (1, 3, 4, state.broker.node_id)
        @test all(o -> !(o.to_id in excluded), ws.offer_book.offers)

        BrokerageABM.reset_offer_book!(ws, state.params.N)
        @test BrokerageABM.add_offer!(ws, 1, 2, :self, 3.0)
        @test BrokerageABM.add_offer!(ws, 2, 1, :broker, 4.0)
        @test !BrokerageABM.add_offer!(ws, 1, 2, :self, 5.0)
        @test BrokerageABM.offer_ids(ws.offer_book, 1, 2) == (1, 2)
        @test ws.offer_book.offer_pairs == [(1, 2)]
    end

    @testset "Broker offer construction follows access, pair, and quota rules" begin
        state = micro_state()
        broker = state.broker
        ws = state.workspace
        broker.roster = Set([2, 3])
        broker.current_clients = Set([1, 4])
        linear_prediction!(broker.nn, [1.0, 0.0, 0.0, 0.0, 0.0])

        broker_access = ws.broker_pairs.period_broker_access_ids
        BrokerageABM.collect_broker_access_ids!(
            broker_access, broker, state.agents, ws
        )
        @test Set(broker_access) == Set([1, 2, 3, 4])

        ws.search.period_strangers = [5]
        remaining = zeros(Int, state.params.N)
        remaining[1] = 3
        remaining[4] = 1
        sent = BrokerageABM.append_broker_offers!(
            ws,
            [1, 4],
            [:broker, :broker],
            [3, 1],
            state.agents,
            broker,
            state.params,
            0.5;
            rng=StableRNG(8802),
            remaining_demand=remaining,
        )

        @test sent == 4
        @test [(o.from_id, o.to_id) for o in ws.offer_book.offers] == [(1, 2), (1, 3), (1, 4), (4, 1)]
        @test all(o -> o.to_id != 5, ws.offer_book.offers)
        @test remaining[1] == 0
        @test remaining[4] == 0
        @test all(i != j for (_, i, j) in ws.broker_pairs.broker_pair_scores)
    end

    @testset "Acceptance and finalization follow offer-market rules" begin
        state = micro_state()
        ws = state.workspace
        agents = state.agents

        reset_offer_workspace!(state)
        @test BrokerageABM.add_offer!(ws, 1, 2, :self, -10.0)
        @test BrokerageABM.add_offer!(ws, 2, 1, :broker, -10.0)
        hb_before = state.broker.history_count
        accepted = run_accept_pair!(state, 1, 2)

        @test length(accepted) == 1
        @test accepted[1].q_realized ≈ BrokerageABM.Q_OFFSET + 0.8
        @test state.broker.history_count == hb_before + 1
        assert_symmetric_active_match(agents, 1, 2; channel=:broker)

        state = micro_state()
        ws = state.workspace
        agents = state.agents
        pseudocode_constant_prediction!(agents[4].nn, 1.0)
        reset_offer_workspace!(state)
        @test BrokerageABM.add_offer!(ws, 1, 4, :self, 2.0)
        h1 = agents[1].history_count
        h4 = agents[4].history_count
        accepted = run_accept_pair!(state, 1, 4)

        @test isempty(accepted)
        @test agents[1].history_count == h1
        @test agents[4].history_count == h4
        @test isempty(agents[1].active_matches)
        @test !has_edge(state.G, 1, 4)

        pseudocode_constant_prediction!(agents[3].nn, 2.5)
        BrokerageABM.reset_offer_book!(ws, state.params.N)
        @test BrokerageABM.add_offer!(ws, 1, 3, :self, 2.0)
        h1 = agents[1].history_count
        h3 = agents[3].history_count
        accepted = run_accept_pair!(state, 1, 3)

        @test length(accepted) == 1
        @test agents[1].history_count == h1 + 1
        @test agents[3].history_count == h3 + 1
        @test partner_mean(agents[1], 3) ≈ accepted[1].q_realized
        @test partner_mean(agents[3], 1) ≈ accepted[1].q_realized
        @test has_edge(state.G, 1, 3)
        assert_symmetric_active_match(agents, 1, 3; channel=:self)

        BrokerageABM.reset_offer_book!(ws, state.params.N)
        @test BrokerageABM.add_offer!(ws, 3, 1, :self, 2.0)
        h1 = agents[1].history_count
        h3 = agents[3].history_count
        accepted = run_accept_pair!(state, 1, 3, accepted)

        @test length(accepted) == 1
        @test agents[1].history_count == h1
        @test agents[3].history_count == h3
    end

    @testset "Satisfaction and reputation follow update formulas" begin
        state = micro_state()
        p = state.params
        cal = state.cal
        agents = state.agents
        agents[1].satisfaction_self = 1.0
        agents[2].satisfaction_broker = 2.0
        agents[3].satisfaction_broker = 3.0
        agents[5].satisfaction_self = 9.0

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
            agents, accepted, [1, 2, 3], [:self, :broker, :broker], [1, 1, 1], cal, p
        )

        @test agents[1].satisfaction_self ≈ (1 - p.omega) * 1.0 + p.omega * (4.0 - cal.c_s)
        @test agents[2].satisfaction_broker ≈
            (1 - p.omega) * 2.0 + p.omega * (4.0 - cal.phi)
        @test agents[3].satisfaction_broker ≈ (1 - p.omega) * 3.0
        @test agents[5].satisfaction_self == 9.0
        @test agents[2].tried_broker
        @test agents[3].tried_broker

        agents[2].satisfaction_broker = 4.0
        agents[3].satisfaction_broker = 6.0
        state.broker.last_reputation = 1.0
        update_broker_reputation!(state.broker, agents, [2, 3])
        @test state.broker.last_reputation == 5.0
        @test state.broker.has_had_clients
        update_broker_reputation!(state.broker, agents, Int[])
        @test state.broker.last_reputation == 5.0
    end

    @testset "Period update and metrics preserve ordering-sensitive invariants" begin
        state = micro_state()
        target_roster = BrokerageABM.roster_target_size(state.params)
        for i in 1:state.params.N
            state.agents[i].satisfaction_self = 10.0
            state.agents[i].satisfaction_broker = 0.0
            state.agents[i].tried_broker = true
        end
        state.agents[1].satisfaction_self = 0.0
        state.agents[1].tried_broker = false
        state.broker.last_reputation = 5.0
        state.broker.current_clients = Set([5])
        push!(state.agents[1].active_matches, ActiveMatch(2, :self))
        push!(state.agents[2].active_matches, ActiveMatch(1, :self))
        types_before = [copy(a.type) for a in state.agents]

        metrics = step_period!(state)

        @test state.period == 1
        @test 1 in state.broker.current_clients
        @test 5 ∉ state.broker.current_clients
        @test !has_current_match(state.agents[1], 2)
        @test !has_current_match(state.agents[2], 1)
        @test all(state.agents[i].type == types_before[i] for i in 1:state.params.N)
        @test metrics.period == 1
        @test metrics.n_total_matches == metrics.n_self_matches + metrics.n_broker_matches
        @test metrics.total_demand == state.params.N * state.params.K
        @test metrics.outsourcing_rate == metrics.outsourced_slots / metrics.total_demand
        @test metrics.roster_size == target_roster
        @test metrics.broker_access_size == broker_access_size(state.broker)
        @test metrics.broker_access_size >= metrics.roster_size
        @test metrics.mean_degree ≈ mean(degree(state.G)[1:state.params.N])
        @test isfinite(metrics.betweenness)
        @test isfinite(metrics.constraint)
        @test isfinite(metrics.effective_size)
        @test verify_invariants(state) === nothing
    end
end
