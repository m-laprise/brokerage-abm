using Test
using TransientBrokerage
using TransientBrokerage: ActiveMatch, AcceptedMatch, CalibrationConstants
using TransientBrokerage: MatchingEnv, NeuralNet, OfferCredit
using TransientBrokerage: add_match_edge!, broker_access_size, counterparty_ask
using TransientBrokerage: has_current_match, partner_mean
using TransientBrokerage: record_agent_history!, record_broker_history!
using TransientBrokerage: remove_agent_edges!, update_broker_reputation!
using TransientBrokerage: update_capture_confidence_mae!, update_partner_mean!
using TransientBrokerage: update_satisfaction!
using Graphs: degree, has_edge, nv
using LinearAlgebra: I, norm
using StableRNGs: StableRNG
using Statistics: mean

# Pseudocode conformance tests.
#
# This file checks that important implementation paths follow the algorithms in
# simulation_pseudocode.tex using small deterministic fixtures. These tests are
# not a replacement for regression baselines or distributional validation. They
# are mechanism-level checks designed to make each pseudocode step auditable in a
# hand-checkable setting.
#
# Coverage overview:
# - Initialization: graph and broker-node shape, roster size and broker edges,
#   seeded broker reputation, seed-history self satisfaction, broker-satisfaction
#   priors, unit types, and stable agent IDs.
# - Training and channel choice: self vs broker utility comparison, untried
#   broker-reputation fallback, deterministic tie handling, parity-based agent
#   retraining, and broker/agent new-observation reset behavior.
# - Self-search offers: known-neighbor partner means, stranger NN predictions,
#   reservation thresholding, exclusion of self/current matches/broker/captured
#   origins, descending offer ranking, demand quotas, and reciprocal offer-book
#   deduplication.
# - Broker offers: hybrid access set Roster union current_clients, exclusion of
#   period strangers and captured origins, unordered feasible pair construction,
#   sorted broker predictions, two-direction offers for broker demanders, and
#   quota decrement only when an offer is actually added.
# - Standard acceptance/finalization: reciprocal acceptance, one-sided receiver
#   thresholding, rejected-offer no-op behavior, bilateral history and partner
#   mean updates, graph-edge creation, active-match symmetry, single broker
#   history recording, and duplicate current-period match blocking.
# - Client-origin capture: readiness gates, ask fallback and reservation checks,
#   whole-lot feasibility, expected-gain positivity, captured-origin masking,
#   principal payments, broker-only learning for principal positions, rejected
#   exposure accounting, and principal active-match symmetry.
# - Satisfaction, reputation, and confidence: channel-specific EWMA formulas,
#   broker standard fees, principal-payment credit, demand-only updates,
#   tried-broker state, reputation update/hold behavior, and capture-confidence
#   MAE initialization, EWMA, and no-op periods.
# - Period update and metrics: start-of-period clearing, current-client rebuild,
#   no-entry behavior when eta=0, invariant verification, coherent match totals,
#   outsourcing rates, access/roster sizes, capture counters, and network metrics.
#
# The fixtures overwrite random initialization where exact expected values are
# needed, while leaving model logic unchanged.

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
    broker.last_reputation = 1.0
    broker.has_had_clients = false
    broker.capture_confidence_mae = 0.0
    broker.capture_confidence_ready = false
    broker.capture_error_count = 0
    pseudocode_constant_prediction!(broker.nn, 0.0)
    TransientBrokerage.reset_accumulators!(state.accum)
    TransientBrokerage.reset_offer_book!(state.workspace, p.N)
    TransientBrokerage.reset_current_match_index!(state.workspace, p.N)
    return state
end

function micro_state(; enable_principal::Bool=false, seed::Int=314)
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
        h_a=2,
        h_b=2,
        network_measure_interval=1,
        T=3,
        T_burn=1,
        seed=seed,
        enable_principal=enable_principal,
        capture_min_error_obs=1,
        capture_error_threshold=999.0,
    )
    return configure_micro_state!(initialize_model(p))
end

function set_partner_mean!(state, i::Int, j::Int, q::Float64; count::Int=1)
    state.agents[i].partner_sum[j] = q * count
    state.agents[i].partner_count[j] = count
    return nothing
end

function reset_offer_workspace!(state)
    TransientBrokerage.reset_offer_book!(state.workspace, state.params.N)
    TransientBrokerage.rebuild_current_match_index!(state.workspace, state.agents)
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

function assert_symmetric_active_match(agents, i, j; is_principal::Bool, channel::Symbol)
    mij = [am for am in agents[i].active_matches if am.partner_id == j]
    mji = [am for am in agents[j].active_matches if am.partner_id == i]
    @test length(mij) == 1
    @test length(mji) == 1
    @test mij[1].is_principal == is_principal
    @test mji[1].is_principal == is_principal
    @test mij[1].channel == channel
    @test mji[1].channel == channel
    return nothing
end

function run_accept_pair!(state, i::Int, j::Int, accepted=AcceptedMatch[])
    Ax_buf, Bx_buf = match_buffers(state)
    TransientBrokerage.accept_offer_pair!(
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

function capture_state(;
    receiver_value::Float64=10.0,
    broker_value::Float64=11.2,
    ask_value::Float64=2.2,
    n_recipients::Int=1,
)
    state = micro_state(; enable_principal=true)
    state.broker.capture_confidence_ready = true
    state.broker.capture_confidence_mae = 0.0
    state.broker.capture_error_count = 1
    pseudocode_constant_prediction!(state.broker.nn, broker_value)
    for id in 1:(n_recipients + 1)
        pseudocode_constant_prediction!(state.agents[id].nn, receiver_value)
    end
    state.broker.roster = Set(2:(n_recipients + 1))
    state.broker.current_clients = Set([1])
    state.agents[1].history_count = 1
    state.agents[1].history_q[1] = ask_value
    return state
end

function run_pseudocode_capture_round!(state, counts=[1])
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
        accepted_matches=state.workspace.ledger.accepted_matches,
        accum=state.accum,
    )
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
            T_burn=1,
            seed=501,
        )
        state = initialize_model(p)
        broker = state.broker

        @test length(state.agents) == p.N
        @test broker.node_id == p.N + 1
        @test nv(state.G) == p.N + 1
        @test all(state.agents[i].id == i for i in 1:p.N)
        @test all(isapprox(norm(a.type), 1.0; atol=1e-10) for a in state.agents)
        @test length(broker.roster) == TransientBrokerage.roster_target_size(p.N)
        @test all(has_edge(state.G, rid, broker.node_id) for rid in broker.roster)

        seeded_agents = [a for a in state.agents if a.history_count > 0]
        @test broker.history_count > 0
        @test !isempty(seeded_agents)
        @test broker.last_reputation ≈ mean(broker.history_q[1:broker.history_count])
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
        @test TransientBrokerage.outsourcing_decision(agents[1], 0.0, StableRNG(1)) ==
            :self

        agents[2].satisfaction_self = 1.0
        agents[2].satisfaction_broker = 3.0
        agents[2].tried_broker = true
        @test TransientBrokerage.outsourcing_decision(agents[2], 0.0, StableRNG(2)) ==
            :broker

        agents[3].satisfaction_self = 2.0
        agents[3].satisfaction_broker = 100.0
        agents[3].tried_broker = false
        @test TransientBrokerage.outsourcing_decision(agents[3], 1.0, StableRNG(3)) ==
            :self
        @test TransientBrokerage.outsourcing_decision(agents[3], 3.0, StableRNG(4)) ==
            :broker

        agents[4].satisfaction_self = 1.0
        agents[4].satisfaction_broker = 1.0
        agents[4].tried_broker = true
        tie_seed = 44
        expected_tie = rand(StableRNG(tie_seed)) < 0.5 ? :self : :broker
        @test TransientBrokerage.outsourcing_decision(
            agents[4], 0.0, StableRNG(tie_seed)
        ) == expected_tie

        @test all(
            TransientBrokerage.agent_retrains_this_period(i, t) ==
            (isodd(i) == isodd(t)) for i in 1:4, t in 1:4
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
        TransientBrokerage.add_broker_edge!(G, 1, state.broker.node_id)
        set_partner_mean!(state, 1, 2, 3.0)
        set_partner_mean!(state, 1, 3, 2.2)
        set_partner_mean!(state, 1, 4, 1.0)
        push!(agents[1].active_matches, ActiveMatch(3, false, :self))
        pseudocode_constant_prediction!(agents[1].nn, 2.5)

        captured_origin_mask = fill(false, state.params.N)
        captured_origin_mask[5] = true
        reset_offer_workspace!(state)

        sent = TransientBrokerage.append_self_search_offers!(
            ws,
            agents[1],
            2,
            agents,
            G,
            state.broker.node_id,
            [1, 2, 5, 6, 7],
            state.cal.r;
            current_match_index_ready=true,
            captured_origin_mask=captured_origin_mask,
        )

        @test sent == 2
        @test [(o.to_id, o.predicted_value) for o in ws.offer_book.offers] ==
            [(2, 3.0), (6, 2.5)]
        excluded = (1, 3, 4, 5, state.broker.node_id)
        @test all(o -> !(o.to_id in excluded), ws.offer_book.offers)

        TransientBrokerage.reset_offer_book!(ws, state.params.N)
        @test TransientBrokerage.add_offer!(ws, 1, 2, :self, 3.0)
        @test TransientBrokerage.add_offer!(ws, 2, 1, :broker, 4.0)
        @test !TransientBrokerage.add_offer!(ws, 1, 2, :self, 5.0)
        @test TransientBrokerage.offer_ids(ws.offer_book, 1, 2) == (1, 2)
        @test ws.offer_book.offer_pairs == [(1, 2)]
    end

    @testset "Broker offer construction follows access, pair, and quota rules" begin
        state = micro_state()
        broker = state.broker
        ws = state.workspace
        broker.roster = Set([2, 3])
        broker.current_clients = Set([1, 4])
        linear_prediction!(broker.nn, [1.0, 0.0, 1.0, 0.0])

        broker_access = ws.broker_pairs.period_broker_access_ids
        TransientBrokerage.collect_broker_access_ids!(
            broker_access, broker, state.agents, ws
        )
        @test Set(broker_access) == Set([1, 2, 3, 4])

        ws.search.period_strangers = [5]
        remaining = zeros(Int, state.params.N)
        remaining[1] = 3
        remaining[4] = 1
        sent = TransientBrokerage.append_broker_offers!(
            ws,
            [1, 4],
            [:broker, :broker],
            [3, 1],
            state.agents,
            broker,
            state.params,
            0.5;
            remaining_demand=remaining,
        )

        @test sent == 4
        @test [(o.from_id, o.to_id) for o in ws.offer_book.offers] ==
            [(1, 2), (1, 3), (1, 4), (4, 1)]
        @test all(o -> o.to_id != 5, ws.offer_book.offers)
        @test remaining[1] == 0
        @test remaining[4] == 0
        @test all(i != j for (_, i, j) in ws.broker_pairs.broker_pair_scores)

        captured_origin_mask = fill(false, state.params.N)
        captured_origin_mask[1] = true
        TransientBrokerage.reset_offer_book!(ws, state.params.N)
        fill!(remaining, 0)
        remaining[1] = 1
        remaining[4] = 1
        sent = TransientBrokerage.append_broker_offers!(
            ws,
            [1, 4],
            [:broker, :broker],
            [1, 1],
            state.agents,
            broker,
            state.params,
            0.5;
            remaining_demand=remaining,
            captured_origin_mask=captured_origin_mask,
        )

        @test sent == 1
        @test all(o -> o.from_id != 1 && o.to_id != 1, ws.offer_book.offers)
    end

    @testset "Standard acceptance and finalization follow offer-market rules" begin
        state = micro_state()
        ws = state.workspace
        agents = state.agents

        reset_offer_workspace!(state)
        @test TransientBrokerage.add_offer!(ws, 1, 2, :self, -10.0)
        @test TransientBrokerage.add_offer!(ws, 2, 1, :broker, -10.0)
        hb_before = state.broker.history_count
        accepted = run_accept_pair!(state, 1, 2)

        @test length(accepted) == 1
        @test accepted[1].q_realized ≈ 1.8
        @test state.broker.history_count == hb_before + 1
        assert_symmetric_active_match(agents, 1, 2; is_principal=false, channel=:broker)

        state = micro_state()
        ws = state.workspace
        agents = state.agents
        pseudocode_constant_prediction!(agents[4].nn, 1.0)
        reset_offer_workspace!(state)
        @test TransientBrokerage.add_offer!(ws, 1, 4, :self, 2.0)
        h1 = agents[1].history_count
        h4 = agents[4].history_count
        accepted = run_accept_pair!(state, 1, 4)

        @test isempty(accepted)
        @test agents[1].history_count == h1
        @test agents[4].history_count == h4
        @test isempty(agents[1].active_matches)
        @test !has_edge(state.G, 1, 4)

        pseudocode_constant_prediction!(agents[3].nn, 2.5)
        TransientBrokerage.reset_offer_book!(ws, state.params.N)
        @test TransientBrokerage.add_offer!(ws, 1, 3, :self, 2.0)
        h1 = agents[1].history_count
        h3 = agents[3].history_count
        accepted = run_accept_pair!(state, 1, 3)

        @test length(accepted) == 1
        @test agents[1].history_count == h1 + 1
        @test agents[3].history_count == h3 + 1
        @test partner_mean(agents[1], 3) ≈ accepted[1].q_realized
        @test partner_mean(agents[3], 1) ≈ accepted[1].q_realized
        @test has_edge(state.G, 1, 3)
        assert_symmetric_active_match(agents, 1, 3; is_principal=false, channel=:self)

        TransientBrokerage.reset_offer_book!(ws, state.params.N)
        @test TransientBrokerage.add_offer!(ws, 3, 1, :self, 2.0)
        h1 = agents[1].history_count
        h3 = agents[3].history_count
        accepted = run_accept_pair!(state, 1, 3, accepted)

        @test length(accepted) == 1
        @test agents[1].history_count == h1
        @test agents[3].history_count == h3
    end

    @testset "Client-origin resource capture follows whole-lot rules" begin
        state = capture_state()
        state.broker.capture_confidence_ready = false
        accepted = run_pseudocode_capture_round!(state)
        @test all(!m.is_principal for m in accepted)
        @test state.accum.captured_position_count == 0

        state = capture_state()
        state.broker.capture_confidence_mae = 1.0e9
        accepted = run_pseudocode_capture_round!(state)
        @test all(!m.is_principal for m in accepted)
        @test state.accum.captured_position_count == 0

        state = capture_state()
        agent = state.agents[1]
        agent.history_count = 0
        @test counterparty_ask(agent, state.cal.q_cal) == state.cal.q_cal
        agent.history_count = 3
        agent.history_q[1:3] .= [1.0, 2.0, 4.0]
        @test counterparty_ask(agent, state.cal.q_cal) ≈ 7.0 / 3

        state = capture_state(; ask_value=1.1)
        accepted = run_pseudocode_capture_round!(state)
        @test all(!m.is_principal for m in accepted)
        @test state.accum.captured_position_count == 0

        state = capture_state(; n_recipients=1)
        accepted = run_pseudocode_capture_round!(state, [2])
        @test all(!m.is_principal for m in accepted)
        @test state.accum.captured_position_count == 0

        state = capture_state(; broker_value=1.3, ask_value=1.25)
        expected_gain = 1.3 - 1.25 - state.cal.phi
        @test expected_gain < 0.0
        accepted = run_pseudocode_capture_round!(state)
        @test all(!m.is_principal for m in accepted)
        @test state.accum.captured_origin_count == 0

        state = capture_state(; n_recipients=2, broker_value=11.2, ask_value=2.2)
        h1 = state.agents[1].history_count
        h2 = state.agents[2].history_count
        hb = state.broker.history_count
        expected_gain = 2 * 11.2 - 2 * 2.2 - 2 * state.cal.phi
        accepted = run_pseudocode_capture_round!(state, [2])

        @test expected_gain > 0.0
        @test length(accepted) == 2
        @test all(m -> m.is_principal, accepted)
        @test state.accum.captured_origin_count == 1
        @test state.accum.captured_position_count == 2
        @test state.workspace.capture.captured_origin_mask[1]
        @test state.workspace.ledger.offer_remaining[1] == 0
        @test state.workspace.ledger.principal_payment[1] == 2 * 2.2
        @test state.agents[1].history_count == h1
        @test state.agents[2].history_count == h2
        @test state.agents[1].partner_count[2] == 0
        @test state.broker.history_count == hb + 2
        @test !has_edge(state.G, 1, 2)
        assert_symmetric_active_match(
            state.agents, 1, 2; is_principal=true, channel=:broker
        )

        state = capture_state(; receiver_value=0.0)
        accepted = run_pseudocode_capture_round!(state)
        @test isempty(accepted)
        @test state.accum.captured_origin_count == 1
        @test state.accum.captured_position_count == 1
        @test state.accum.principal_rejected == 1
        @test state.accum.capture_realized == [0.0]
        @test state.accum.broker_error_count == 1

        state = capture_state(; receiver_value=10.0)
        pseudocode_constant_prediction!(state.agents[2].nn, state.cal.r + 1.0)
        add_match_edge!(state.G, 3, 1)
        add_match_edge!(state.G, 3, 2)
        set_partner_mean!(state, 3, 1, state.cal.r + 10.0)
        set_partner_mean!(state, 3, 2, state.cal.r + 5.0)

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
            accepted_matches=state.workspace.ledger.accepted_matches,
            accum=state.accum,
        )

        self_offers = [o for o in state.workspace.offer_book.offers if o.from_id == 3]
        @test length(self_offers) == 1
        @test self_offers[1].to_id == 2
        @test all(o -> o.to_id != 1, state.workspace.offer_book.offers)
    end

    @testset "Satisfaction, reputation, and confidence follow update formulas" begin
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
                false,
                4.0,
                4.0,
                NaN,
                NaN,
                OfferCredit(1, 2, :self, 4.0, false),
                OfferCredit(2, 1, :broker, 4.0, false),
            ),
        ]
        principal_payment = zeros(Float64, p.N)
        principal_payment[3] = 5.0

        update_satisfaction!(
            agents,
            accepted,
            [1, 2, 3],
            [:self, :broker, :broker],
            [1, 1, 1],
            cal,
            p;
            principal_payment=principal_payment,
        )

        @test agents[1].satisfaction_self ≈ (1 - p.omega) * 1.0 +
            p.omega * (4.0 - cal.c_s)
        @test agents[2].satisfaction_broker ≈ (1 - p.omega) * 2.0 +
            p.omega * (4.0 - cal.phi)
        @test agents[3].satisfaction_broker ≈ (1 - p.omega) * 3.0 +
            p.omega * 5.0
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

        broker = state.broker
        broker.capture_confidence_ready = false
        broker.capture_confidence_mae = 0.0
        broker.capture_error_count = 0
        update_capture_confidence_mae!(broker, 6.0, 3, 0.2)
        @test broker.capture_confidence_ready
        @test broker.capture_confidence_mae == 2.0
        @test broker.capture_error_count == 3
        update_capture_confidence_mae!(broker, 3.0, 3, 0.2)
        @test broker.capture_confidence_mae ≈ 1.8
        @test broker.capture_error_count == 6
        update_capture_confidence_mae!(broker, 100.0, 0, 0.2)
        @test broker.capture_confidence_mae ≈ 1.8
        @test broker.capture_error_count == 6
    end

    @testset "Period update and metrics preserve ordering-sensitive invariants" begin
        state = micro_state()
        target_roster = TransientBrokerage.roster_target_size(state.params.N)
        for i in 1:state.params.N
            state.agents[i].satisfaction_self = 10.0
            state.agents[i].satisfaction_broker = 0.0
            state.agents[i].tried_broker = true
        end
        state.agents[1].satisfaction_self = 0.0
        state.agents[1].tried_broker = false
        state.broker.last_reputation = 5.0
        state.broker.current_clients = Set([5])
        push!(state.agents[1].active_matches, ActiveMatch(2, false, :self))
        push!(state.agents[2].active_matches, ActiveMatch(1, false, :self))
        types_before = [copy(a.type) for a in state.agents]

        step_period!(state)
        metrics = collect_period_metrics(state)

        @test state.period == 1
        @test 1 in state.broker.current_clients
        @test 5 ∉ state.broker.current_clients
        @test !has_current_match(state.agents[1], 2)
        @test !has_current_match(state.agents[2], 1)
        @test all(state.agents[i].type == types_before[i] for i in 1:state.params.N)
        @test metrics.period == 1
        @test metrics.n_total_matches ==
            metrics.n_self_matches +
            metrics.n_broker_standard +
            metrics.n_broker_principal
        @test metrics.total_demand == state.params.N * state.params.K
        @test metrics.outsourcing_rate ==
            metrics.outsourced_slots / metrics.total_demand
        @test metrics.roster_size == target_roster
        @test metrics.broker_access_size == broker_access_size(state.broker)
        @test metrics.broker_access_size >= metrics.roster_size
        @test metrics.captured_origin_count == 0
        @test metrics.captured_position_count == 0
        @test metrics.mean_degree ≈ mean(degree(state.G)[1:state.params.N])
        @test isfinite(metrics.betweenness)
        @test isfinite(metrics.constraint)
        @test isfinite(metrics.effective_size)
        @test verify_invariants(state) === nothing
    end
end
