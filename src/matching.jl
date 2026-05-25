"""
    matching.jl

Within-period offer formation, acceptance, satisfaction tracking, and
outsourcing decisions.
"""

using Random: AbstractRNG
using Graphs: has_edge, SimpleGraph

function receiver_offer_value(
    receiver_id::Int, sender_id::Int, agents::Vector{Agent}, G::SimpleGraph
)::Float64
    receiver = agents[receiver_id]
    if has_edge(G, receiver_id, sender_id) && receiver.partner_count[sender_id] > 0
        return partner_mean(receiver, sender_id)
    end
    return predict_nn!(receiver.nn, receiver.predict_buf, agents[sender_id].type)
end

@inline function has_broker_offer(
    offer1::DirectedOffer, offer2::Union{DirectedOffer,Nothing}
)
    offer1.channel == :broker && return true
    isnothing(offer2) && return false
    return offer2.channel == :broker
end

function push_accepted_relationship!(
    accepted::Vector{AcceptedMatch},
    offer1::DirectedOffer,
    offer2::Union{DirectedOffer,Nothing},
    agents::Vector{Agent},
    broker::Broker,
    env::MatchingEnv,
    G::SimpleGraph,
    rng::AbstractRNG;
    Ax_buf::Vector{Float64},
    Bx_buf::Vector{Float64},
    ws::SimWorkspace,
)
    i = min(offer1.from_id, offer1.to_id)
    j = max(offer1.from_id, offer1.to_id)
    has_current_match(ws, i, j) && return nothing

    q_realized = match_output!(Ax_buf, Bx_buf, agents[i].type, agents[j].type, env, rng)
    broker_involved = has_broker_offer(offer1, offer2)
    rel_channel = broker_involved ? :broker : :self
    offer1_connected = has_edge(G, offer1.from_id, offer1.to_id)
    offer2_connected = isnothing(offer2) ? false : has_edge(G, offer2.from_id, offer2.to_id)

    record_agent_history!(agents[i], agents[j].type, q_realized)
    record_agent_history!(agents[j], agents[i].type, q_realized)
    update_partner_mean!(agents[i], j, q_realized)
    update_partner_mean!(agents[j], i, q_realized)
    broker_involved &&
        record_broker_history!(broker, agents[i].type, agents[j].type, q_realized)
    add_match_edge!(G, i, j)
    push!(agents[i].active_matches, ActiveMatch(j, false, rel_channel))
    push!(agents[j].active_matches, ActiveMatch(i, false, rel_channel))
    mark_current_match!(ws, i, j)

    agents[i].n_matches_any += 1
    agents[j].n_matches_any += 1

    offer1_credit = OfferCredit(
        offer1.from_id,
        offer1.to_id,
        offer1.channel,
        offer1.predicted_value,
        offer1_connected,
    )
    offer2_credit = if isnothing(offer2)
        nothing
    else
        OfferCredit(
        offer2.from_id,
        offer2.to_id,
        offer2.channel,
        offer2.predicted_value,
        offer2_connected,
    )
    end

    push!(
        accepted,
        AcceptedMatch(
            offer1.from_id,
            offer1.to_id,
            rel_channel,
            false,
            q_realized,
            offer1.predicted_value,
            NaN,
            NaN,
            offer1_credit,
            offer2_credit,
        ),
    )
    return nothing
end

function accept_offer_pair!(
    accepted::Vector{AcceptedMatch},
    pair_i::Int,
    pair_j::Int,
    agents::Vector{Agent},
    broker::Broker,
    env::MatchingEnv,
    G::SimpleGraph,
    cal::CalibrationConstants,
    rng::AbstractRNG;
    Ax_buf::Vector{Float64},
    Bx_buf::Vector{Float64},
    ws::SimWorkspace,
)
    has_current_match(ws, pair_i, pair_j) && return nothing
    idx_ij = @inbounds ws.offer_index[pair_i, pair_j]
    idx_ji = @inbounds ws.offer_index[pair_j, pair_i]
    idx_ij == 0 && idx_ji == 0 && return nothing

    if idx_ij != 0 && idx_ji != 0
        offer1 = @inbounds ws.offers[idx_ij]
        offer2 = @inbounds ws.offers[idx_ji]
        return push_accepted_relationship!(
            accepted,
            offer1,
            offer2,
            agents,
            broker,
            env,
            G,
            rng;
            Ax_buf=Ax_buf,
            Bx_buf=Bx_buf,
            ws=ws,
        )
    end

    offer = @inbounds ws.offers[idx_ij != 0 ? idx_ij : idx_ji]
    receiver_offer_value(offer.to_id, offer.from_id, agents, G) > cal.r || return nothing
    return push_accepted_relationship!(
        accepted,
        offer,
        nothing,
        agents,
        broker,
        env,
        G,
        rng;
        Ax_buf=Ax_buf,
        Bx_buf=Bx_buf,
        ws=ws,
    )
end

"""
    run_offer_market!(...)

Run the shared binding-offer market for current active demand. When principal
mode is enabled, client-origin capture is executed before standard offers.
"""
function run_offer_market!(
    demand_agent_ids::Vector{Int},
    demand_channels::Vector{Symbol},
    demand_counts::Vector{Int},
    agents::Vector{Agent},
    broker::Broker,
    env::MatchingEnv,
    G::SimpleGraph,
    params::ModelParams,
    cal::CalibrationConstants,
    rng::AbstractRNG;
    ws::SimWorkspace,
    accepted_matches::Union{Vector{AcceptedMatch},Nothing}=nothing,
    accum::Union{PeriodAccumulators,Nothing}=nothing,
)
    accepted = isnothing(accepted_matches) ? AcceptedMatch[] : empty!(accepted_matches)
    isempty(demand_agent_ids) && return accepted

    N = length(agents)
    d = params.d
    if length(ws.Ax_buf) != d
        ws.Ax_buf = Vector{Float64}(undef, d)
        ws.Bx_buf = Vector{Float64}(undef, d)
    end
    Ax_buf = ws.Ax_buf
    Bx_buf = ws.Bx_buf

    rebuild_current_match_index!(ws, agents)
    reset_offer_book!(ws, N)
    reset_principal_payments!(ws, N)
    sample_period_strangers!(ws.period_strangers, N, params.n_strangers, rng)

    remaining = ws.offer_remaining
    if length(remaining) < N
        resize!(remaining, N)
    end
    fill!(remaining, 0)
    @inbounds for idx in eachindex(demand_agent_ids)
        remaining[demand_agent_ids[idx]] = demand_counts[idx]
    end

    captured_origin_mask = nothing
    if params.enable_principal
        isnothing(accum) && error("accum is required when enable_principal=true")
        reset_capture_buffers!(ws, N)

        broker_demanders = ws.period_broker_demanders
        empty!(broker_demanders)
        @inbounds for idx in eachindex(demand_agent_ids)
            demand_channels[idx] == :broker || continue
            did = demand_agent_ids[idx]
            remaining[did] > 0 || continue
            push!(broker_demanders, did)
        end

        broker_access = ws.period_broker_access_ids
        collect_broker_access_ids!(broker_access, broker, agents, ws)
        execute_client_origin_capture!(
            accepted,
            ws,
            broker_demanders,
            broker_access,
            remaining,
            agents,
            broker,
            env,
            G,
            params,
            cal,
            rng,
            accum;
            Ax_buf=Ax_buf,
            Bx_buf=Bx_buf,
        )
        captured_origin_mask = ws.captured_origin_mask
    elseif !isnothing(accum)
        accum.capture_ready = false
        accum.capture_scaled_mae = capture_scaled_mae(broker, cal)
    end

    @inbounds for idx in eachindex(demand_agent_ids)
        demand_channels[idx] == :self || continue
        did = demand_agent_ids[idx]
        remaining[did] > 0 || continue
        append_self_search_offers!(
            ws,
            agents[did],
            remaining[did],
            agents,
            G,
            broker.node_id,
            ws.period_strangers,
            cal.r;
            current_match_index_ready=true,
            captured_origin_mask=captured_origin_mask,
        )
    end

    append_broker_offers!(
        ws,
        demand_agent_ids,
        demand_channels,
        demand_counts,
        agents,
        broker,
        params,
        cal.r;
        remaining_demand=remaining,
        captured_origin_mask=captured_origin_mask,
    )

    @inbounds for (i, j) in ws.offer_pairs
        accept_offer_pair!(
            accepted,
            i,
            j,
            agents,
            broker,
            env,
            G,
            cal,
            rng;
            Ax_buf=Ax_buf,
            Bx_buf=Bx_buf,
            ws=ws,
        )
    end

    return accepted
end

# ─────────────────────────────────────────────────────────────────────────────
# Satisfaction tracking (§6a)
# ─────────────────────────────────────────────────────────────────────────────

function credit_offer!(
    demander_sum::Vector{Float64},
    broker_standard_count::Vector{Int},
    offer_from::Int,
    offer_channel::Symbol,
    q_realized::Float64,
    is_principal::Bool,
)
    offer_from <= 0 && return nothing
    is_principal && return nothing
    demander_sum[offer_from] += q_realized
    if offer_channel == :broker
        broker_standard_count[offer_from] += 1
    end
    return nothing
end

"""
    update_satisfaction!(agents, accepted_matches, demand_agent_ids, demand_channels,
                         demand_counts, cal, params; demander_sum=nothing,
                         broker_standard_count=nothing, principal_payment=nothing)

Update only agents with active demand. Successful outcomes are credited to the
directed active-search offer and the channel used to make that offer. Captured
broker clients receive their principal payment without paying a standard broker
fee.
"""
function update_satisfaction!(
    agents::Vector{Agent},
    accepted_matches::Vector{AcceptedMatch},
    demand_agent_ids::Vector{Int},
    demand_channels::Vector{Symbol},
    demand_counts::Vector{Int},
    cal::CalibrationConstants,
    params::ModelParams;
    demander_sum::Union{Vector{Float64},Nothing}=nothing,
    broker_standard_count::Union{Vector{Int},Nothing}=nothing,
    principal_payment::Union{Vector{Float64},Nothing}=nothing,
)
    omega = params.omega
    n_agents = length(agents)

    if isnothing(demander_sum)
        demander_sum = zeros(Float64, n_agents)
    elseif length(demander_sum) != n_agents
        resize!(demander_sum, n_agents)
    end
    if isnothing(broker_standard_count)
        broker_standard_count = zeros(Int, n_agents)
    elseif length(broker_standard_count) != n_agents
        resize!(broker_standard_count, n_agents)
    end
    if isnothing(principal_payment)
        principal_payment = zeros(Float64, n_agents)
    elseif length(principal_payment) != n_agents
        resize!(principal_payment, n_agents)
    end

    @inbounds for agent_id in demand_agent_ids
        demander_sum[agent_id] = 0.0
        broker_standard_count[agent_id] = 0
    end

    for m in accepted_matches
        for offer in (m.offer1, m.offer2)
            isnothing(offer) && continue
            credit_offer!(
                demander_sum,
                broker_standard_count,
                offer.from_id,
                offer.channel,
                m.q_realized,
                m.is_principal,
            )
        end
    end

    @inbounds for idx in eachindex(demand_agent_ids)
        agent_id = demand_agent_ids[idx]
        channel = demand_channels[idx]
        d_i = demand_counts[idx]
        agent = agents[agent_id]
        total_q = demander_sum[agent_id] + principal_payment[agent_id]

        if channel == :self
            tilde_q = total_q / d_i - cal.c_s
            agent.satisfaction_self =
                (1.0 - omega) * agent.satisfaction_self + omega * tilde_q
        else
            broker_fee = cal.phi * broker_standard_count[agent_id]
            tilde_q = (total_q - broker_fee) / d_i
            agent.satisfaction_broker =
                (1.0 - omega) * agent.satisfaction_broker + omega * tilde_q
            agent.tried_broker = true
        end
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Outsourcing decision (§6b)
# ─────────────────────────────────────────────────────────────────────────────

"""
    outsourcing_decision(agent, broker_rep, rng) -> Symbol

Choose :self or :broker by comparing channel satisfaction, using broker
reputation for agents that have not yet tried the broker.
"""
function outsourcing_decision(agent::Agent, broker_rep::Float64, rng::AbstractRNG)::Symbol
    score_self = agent.satisfaction_self
    score_broker = agent.tried_broker ? agent.satisfaction_broker : broker_rep

    if score_self > score_broker
        return :self
    elseif score_broker > score_self
        return :broker
    else
        return rand(rng) < 0.5 ? :self : :broker
    end
end

"""Return the broker's current reputation for untried agents."""
broker_reputation(broker::Broker)::Float64 = broker.last_reputation

"""
    update_broker_reputation!(broker, agents, client_ids)

Update broker reputation to mean broker satisfaction of current clients. If no
clients used the broker this period, hold the previous value.
"""
function update_broker_reputation!(
    broker::Broker, agents::Vector{Agent}, client_ids::AbstractVector{Int}
)
    isempty(client_ids) && return nothing
    total = 0.0
    @inbounds for cid in client_ids
        total += agents[cid].satisfaction_broker
    end
    broker.last_reputation = total / length(client_ids)
    broker.has_had_clients = true
    return nothing
end
