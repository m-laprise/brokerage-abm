"""
    matching.jl

Within-period match formation, satisfaction tracking, and outsourcing decision.
No wages in v0.2: match economics use fixed fees (phi, c_s) and the outside option r.
Under principal mode (Model 1), the broker's compensation is the spread q_ij - ask_j.
"""

using Random: AbstractRNG
using Graphs: has_edge, SimpleGraph

# ─────────────────────────────────────────────────────────────────────────────
# Match formation helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    counterparty_offer_score(pm, agents, G, r; ws) -> (admissible, score)

Return whether proposal `pm` clears the counterparty-side participation rule and
the score used for within-round counterparty ranking. Proposals to an already
active current-period counterparty are inadmissible. Standard proposals use the
counterparty's own evaluation. Principal-mode proposals are admissible without
counterparty evaluation and are ranked by the broker's own predicted value.
"""
function counterparty_offer_score(
    pm::ProposedMatch,
    agents::Vector{Agent},
    G::SimpleGraph,
    r::Float64;
    ws::SimWorkspace,
)
    i = pm.demander_id
    j = pm.counterparty_id
    has_current_match(ws, i, j) && return false, -Inf
    if pm.is_principal
        return true, pm.evaluation
    end

    counterparty_eval = if has_edge(G, j, i) && agents[j].partner_count[i] > 0
        partner_mean(agents[j], i)
    else
        predict_nn!(agents[j].nn, agents[j].predict_buf, agents[i].type)
    end

    return counterparty_eval > r, counterparty_eval
end

"""
    finalize_accepted_proposal!(accepted, pm, agents, broker, env, G, rng; Ax_buf, Bx_buf, ws)

Realize one accepted proposal, update histories/network/state, and append the
accepted-match record.
"""
function finalize_accepted_proposal!(
    accepted::Vector{AcceptedMatch},
    pm::ProposedMatch,
    agents::Vector{Agent},
    broker::Broker,
    env::MatchingEnv,
    G::SimpleGraph,
    rng::AbstractRNG;
    Ax_buf::Vector{Float64},
    Bx_buf::Vector{Float64},
    reserved_capacity::Union{Vector{Int},Nothing}=nothing,
    ws::SimWorkspace,
)
    i = pm.demander_id
    j = pm.counterparty_id
    @assert !has_current_match(ws, i, j) "Duplicate current-period relationship ($i, $j)"

    q_realized = match_output!(Ax_buf, Bx_buf, agents[i].type, agents[j].type, env, rng)

    if pm.is_principal
        record_broker_history!(broker, agents[i].type, agents[j].type, q_realized)
        if !isnothing(reserved_capacity) && reserved_capacity[j] > 0
            reserved_capacity[j] -= 1
        end
        push!(agents[i].active_matches, ActiveMatch(j, true, :broker))
        push!(agents[j].active_matches, ActiveMatch(i, true, :broker))
        agents[j].n_principal_acquired += 1
    else
        record_agent_history!(agents[i], agents[j].type, q_realized)
        record_agent_history!(agents[j], agents[i].type, q_realized)
        update_partner_mean!(agents[i], j, q_realized)
        update_partner_mean!(agents[j], i, q_realized)
        if pm.channel == :broker
            record_broker_history!(broker, agents[i].type, agents[j].type, q_realized)
        end
        add_match_edge!(G, i, j)
        push!(agents[i].active_matches, ActiveMatch(j, false, pm.channel))
        push!(agents[j].active_matches, ActiveMatch(i, false, pm.channel))
    end
    mark_current_match!(ws, i, j)

    agents[i].n_matches_any += 1
    agents[j].n_matches_any += 1

    push!(
        accepted,
        (
            demander_id=i,
            counterparty_id=j,
            channel=pm.channel,
            is_principal=pm.is_principal,
            q_realized=q_realized,
            q_predicted=pm.evaluation,
            ask_j=pm.ask_j,
            capture_qhat=pm.capture_qhat,
        ),
    )
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Round-based concurrent match formation (§9 Step 3)
# ─────────────────────────────────────────────────────────────────────────────

"""
    round_match_formation!(demand_agent_ids, demand_channels, demand_counts,
                           agents, broker, env, G, params, cal, rng;
                           ws, accepted_matches=nothing)

Run the approved within-period round protocol. Each round lets every still-live
demander attempt to fill one relationship through its chosen channel. Demanders can
fall back to next-best candidates within the round via deferred acceptance with
capacity. A demander that exhausts its feasible list without being held exits
for the rest of the period.
"""
function round_match_formation!(
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
)
    accepted = if isnothing(accepted_matches)
        AcceptedMatch[]
    else
        empty!(accepted_matches)
    end
    isempty(demand_agent_ids) && return accepted

    n_demanders = length(demand_agent_ids)
    N = length(agents)
    K = params.K
    d = params.d
    max_rounds = maximum(demand_counts)
    reserved_capacity = ws.principal_reserved_capacity
    rebuild_current_match_index!(ws, agents)

    remaining_demand = ws.demand_remaining;
    resize!(remaining_demand, n_demanders)
    remaining_demand .= demand_counts
    demand_failed = ws.demand_failed;
    resize!(demand_failed, n_demanders);
    fill!(demand_failed, false)
    active_positions = ws.round_active_positions;
    empty!(active_positions)
    open_agents = ws.round_open_agents;
    empty!(open_agents)
    broker_indices = ws.round_broker_indices;
    empty!(broker_indices)
    broker_demanders = ws.round_broker_demanders;
    empty!(broker_demanders)
    pref_offsets = ws.round_pref_offsets;
    empty!(pref_offsets)
    next_pref = ws.round_next_pref;
    empty!(next_pref)
    pref_owner = ws.round_pref_owner;
    empty!(pref_owner)
    queue = ws.round_queue;
    empty!(queue)
    if length(ws.round_agent_to_active) < N
        old = length(ws.round_agent_to_active)
        resize!(ws.round_agent_to_active, N)
        @inbounds for i in (old + 1):N
            ws.round_agent_to_active[i] = 0
        end
    end
    agent_to_active = ws.round_agent_to_active
    agent_to_active_touched = ws.round_agent_touched;
    empty!(agent_to_active_touched)
    outgoing_prop_idx = ws.round_outgoing_prop_idx;
    empty!(outgoing_prop_idx)
    hold_counts = ws.round_hold_counts
    resize!(hold_counts, N)
    if size(ws.round_held_prop_idx, 1) != N || size(ws.round_held_prop_idx, 2) != K
        ws.round_held_prop_idx = Matrix{Int}(undef, N, max(K, 1))
        ws.round_held_scores = Matrix{Float64}(undef, N, max(K, 1))
    end
    held_prop_idx = ws.round_held_prop_idx
    held_scores = ws.round_held_scores
    broker_slot_caps = ws.demander_remaining;
    empty!(broker_slot_caps)
    broker_pref_matches = ws.round_broker_pref_matches;
    empty!(broker_pref_matches)
    broker_pref_counts = ws.round_broker_pref_counts;
    empty!(broker_pref_counts)
    pref_matches = ws.all_proposals;
    empty!(pref_matches)
    round_capacity = ws.remaining_cap
    resize!(round_capacity, N)
    wc_i = ws.was_connected_i;
    empty!(wc_i)
    wc_j = ws.was_connected_j;
    empty!(wc_j)
    if length(ws.Ax_buf) != d
        ws.Ax_buf = Vector{Float64}(undef, d)
        ws.Bx_buf = Vector{Float64}(undef, d)
    end
    Ax_buf = ws.Ax_buf
    Bx_buf = ws.Bx_buf

    prepare_period_broker_round_cache!(
        broker,
        demand_agent_ids,
        demand_channels,
        agents,
        params;
        ws=ws,
        reserved_capacity=reserved_capacity,
    )

    @inline function requeue_or_fail!(pos::Int)
        idx = active_positions[pos]
        if next_pref[pos] <= pref_offsets[pos + 1] - 1
            push!(queue, pos)
        else
            demand_failed[idx] = true
        end
        return nothing
    end

    @inline function reject_held_offer!(agent_id::Int, slot::Int)
        rejected_prop_idx = held_prop_idx[agent_id, slot]
        rejected_pos = pref_owner[rejected_prop_idx]
        outgoing_prop_idx[rejected_pos] = 0
        last_slot = hold_counts[agent_id]
        held_prop_idx[agent_id, slot] = held_prop_idx[agent_id, last_slot]
        held_scores[agent_id, slot] = held_scores[agent_id, last_slot]
        hold_counts[agent_id] -= 1
        requeue_or_fail!(rejected_pos)
        return nothing
    end

    for _ in 1:max_rounds
        empty!(active_positions)
        empty!(open_agents)
        empty!(broker_indices)
        empty!(broker_demanders)
        @inbounds for i in 1:N
            cap_i = current_open_capacity(agents, i, K, reserved_capacity)
            round_capacity[i] = cap_i
            cap_i > 0 && push!(open_agents, i)
        end
        @inbounds for idx in eachindex(demand_agent_ids)
            if remaining_demand[idx] <= 0 || demand_failed[idx]
                continue
            end
            if round_capacity[demand_agent_ids[idx]] <= 0
                demand_failed[idx] = true
                continue
            end
            push!(active_positions, idx)
            if demand_channels[idx] == :broker
                push!(broker_indices, idx)
                push!(broker_demanders, demand_agent_ids[idx])
            end
        end
        isempty(active_positions) && break

        empty!(pref_matches)
        empty!(pref_owner)
        resize!(pref_offsets, length(active_positions) + 1)
        pref_offsets[1] = 1

        principal_round_accepts = 0
        broker_cursor = 1
        broker_pref_cursor = 1
        if !isempty(broker_demanders)
            if params.enable_principal
                principal_round_accepts = execute_inventory_round!(
                    accepted,
                    remaining_demand,
                    broker_indices,
                    broker_demanders,
                    agents,
                    broker,
                    env,
                    G,
                    params,
                    cal,
                    rng,
                    wc_i,
                    wc_j;
                    ws=ws,
                    Ax_buf=Ax_buf,
                    Bx_buf=Bx_buf,
                    current_match_index_ready=true,
                )
                if principal_round_accepts > 0
                    empty!(open_agents)
                    @inbounds for i in 1:N
                        cap_i = current_open_capacity(agents, i, K, reserved_capacity)
                        round_capacity[i] = cap_i
                        cap_i > 0 && push!(open_agents, i)
                    end
                end
            end
            resize!(broker_slot_caps, length(broker_demanders))
            @inbounds for di in eachindex(broker_demanders)
                did = broker_demanders[di]
                demand_idx = broker_indices[di]
                broker_slot_caps[di] = min(
                    remaining_demand[demand_idx], round_capacity[did]
                )
            end
            append_broker_round_preferences_from_cache!(
                broker_pref_matches,
                broker_pref_counts,
                broker_demanders,
                agents,
                params,
                cal.r;
                ws=ws,
                demander_slots=broker_slot_caps,
                reserved_capacity=reserved_capacity,
                round_capacity=round_capacity,
                current_match_index_ready=true,
            )
        else
            empty!(broker_pref_matches)
            resize!(broker_pref_counts, 0)
        end

        for pos in eachindex(active_positions)
            idx = active_positions[pos]
            agent_id = demand_agent_ids[idx]
            if remaining_demand[idx] <= 0 || round_capacity[agent_id] <= 0
                remaining_demand[idx] > 0 && (demand_failed[idx] = true)
                pref_offsets[pos + 1] = length(pref_matches) + 1
                continue
            end
            if demand_channels[idx] == :self
                n_added = append_self_round_preferences!(
                    pref_matches,
                    agents[agent_id],
                    agents,
                    G,
                    broker.node_id,
                    params,
                    rng,
                    cal.r;
                    ws=ws,
                    reserved_capacity=reserved_capacity,
                    open_agents=open_agents,
                    current_match_index_ready=true,
                )
                for _ in 1:n_added
                    push!(pref_owner, pos)
                end
            else
                n_added = broker_pref_counts[broker_cursor]
                for local_idx in 1:n_added
                    pm = broker_pref_matches[broker_pref_cursor]
                    broker_pref_cursor += 1
                    if has_edge(G, pm.demander_id, pm.counterparty_id)
                        push!(wc_i, pm.demander_id)
                        push!(wc_j, pm.counterparty_id)
                    end
                    push!(pref_matches, pm)
                    push!(pref_owner, pos)
                end
                broker_cursor += 1
            end
            pref_offsets[pos + 1] = length(pref_matches) + 1
        end

        resize!(next_pref, length(active_positions))
        resize!(outgoing_prop_idx, length(active_positions))
        fill!(outgoing_prop_idx, 0)
        empty!(queue)
        fill!(hold_counts, 0)

        @inbounds for pos in eachindex(active_positions)
            next_pref[pos] = pref_offsets[pos]
            agent_id = demand_agent_ids[active_positions[pos]]
            if remaining_demand[active_positions[pos]] <= 0 || round_capacity[agent_id] <= 0
                remaining_demand[active_positions[pos]] > 0 &&
                    (demand_failed[active_positions[pos]] = true)
                continue
            end
            push!(queue, pos)
            agent_to_active[agent_id] = pos
            push!(agent_to_active_touched, agent_id)
        end

        head = 1
        while head <= length(queue)
            pos = queue[head]
            head += 1
            outgoing_prop_idx[pos] == 0 || continue

            prop_start = next_pref[pos]
            prop_stop = pref_offsets[pos + 1] - 1
            if prop_start > prop_stop
                demand_failed[active_positions[pos]] = true
                continue
            end

            prop_idx = prop_start
            next_pref[pos] = prop_start + 1
            pm = pref_matches[prop_idx]
            admissible, hold_score = counterparty_offer_score(pm, agents, G, cal.r; ws=ws)
            admissible || (requeue_or_fail!(pos); continue)

            i = pm.demander_id
            j = pm.counterparty_id
            active_j_pos = agent_to_active[j]
            if active_j_pos > 0
                reverse_prop_idx = outgoing_prop_idx[active_j_pos]
                if reverse_prop_idx != 0 &&
                    pref_matches[reverse_prop_idx].counterparty_id == i
                    requeue_or_fail!(pos)
                    continue
                end
            end
            effective_cap =
                round_capacity[j] -
                ((active_j_pos > 0 && outgoing_prop_idx[active_j_pos] != 0) ? 1 : 0)
            if effective_cap <= 0
                requeue_or_fail!(pos)
                continue
            end

            accepted_here = false
            if hold_counts[j] < effective_cap
                hold_counts[j] += 1
                held_prop_idx[j, hold_counts[j]] = prop_idx
                held_scores[j, hold_counts[j]] = hold_score
                accepted_here = true
            else
                worst_slot = 1
                worst_score = held_scores[j, 1]
                @inbounds for slot in 2:hold_counts[j]
                    if held_scores[j, slot] < worst_score
                        worst_score = held_scores[j, slot]
                        worst_slot = slot
                    end
                end
                if hold_score > worst_score
                    reject_held_offer!(j, worst_slot)
                    hold_counts[j] += 1
                    held_prop_idx[j, hold_counts[j]] = prop_idx
                    held_scores[j, hold_counts[j]] = hold_score
                    accepted_here = true
                end
            end

            if !accepted_here
                requeue_or_fail!(pos)
                continue
            end

            outgoing_prop_idx[pos] = prop_idx
            while hold_counts[i] + 1 > round_capacity[i]
                worst_slot = 1
                worst_score = held_scores[i, 1]
                @inbounds for slot in 2:hold_counts[i]
                    if held_scores[i, slot] < worst_score
                        worst_score = held_scores[i, slot]
                        worst_slot = slot
                    end
                end
                reject_held_offer!(i, worst_slot)
            end
        end

        round_accepts = principal_round_accepts
        @inbounds for pos in eachindex(active_positions)
            prop_idx = outgoing_prop_idx[pos]
            prop_idx == 0 && continue
            pm = pref_matches[prop_idx]
            if has_current_match(ws, pm.demander_id, pm.counterparty_id)
                continue
            end
            finalize_accepted_proposal!(
                accepted,
                pm,
                agents,
                broker,
                env,
                G,
                rng;
                Ax_buf=Ax_buf,
                Bx_buf=Bx_buf,
                ws=ws,
            )
            remaining_demand[active_positions[pos]] -= 1
            round_accepts += 1
        end

        @inbounds for aid in agent_to_active_touched
            agent_to_active[aid] = 0
        end

        round_accepts == 0 && break
    end

    return accepted
end

# ─────────────────────────────────────────────────────────────────────────────
# Satisfaction tracking (§6a)
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_satisfaction!(agents, accepted_matches, demand_agent_ids, demand_channels,
                         demand_counts, cal, params; demander_sum=nothing,
                         broker_standard_count=nothing)

Update satisfaction indices for all agents that had demand this period.
Requested relationship positions are the averaging unit: realized outcomes are
summed over accepted matches, unfilled positions contribute zero output, and the
total is divided by requested demand d_i. Self-search pays c_s per requested
position, regardless of whether that position is filled; standard broker fees
are charged
only on successful standard brokered matches; principal-mode matches carry no
demander fee.
"""
function update_satisfaction!(
    agents::Vector{Agent},
    accepted_matches::Vector{<:NamedTuple},
    demand_agent_ids::Vector{Int},
    demand_channels::Vector{Symbol},
    demand_counts::Vector{Int},
    cal::CalibrationConstants,
    params::ModelParams;
    demander_sum::Union{Vector{Float64},Nothing}=nothing,
    broker_standard_count::Union{Vector{Int},Nothing}=nothing,
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

    # Reset only demanders that may be read below.
    @inbounds for agent_id in demand_agent_ids
        demander_sum[agent_id] = 0.0
        broker_standard_count[agent_id] = 0
    end

    # Group accepted matches by demander: accumulate realized output and the
    # number of successful standard brokered placements.
    for m in accepted_matches
        i = m.demander_id
        demander_sum[i] += m.q_realized
        if m.channel == :broker && !m.is_principal
            broker_standard_count[i] += 1
        end
    end

    # Update each agent that had demand
    for idx in eachindex(demand_agent_ids)
        agent_id = demand_agent_ids[idx]
        channel = demand_channels[idx]
        d_i = demand_counts[idx]
        agent = agents[agent_id]

        total_q = demander_sum[agent_id]
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

Agent chooses :self or :broker. Compares two reduced-form channel scores:
- score_self: EWMA satisfaction from past self-search outcomes. This stands in
  for the full internal-search option, including reuse of known partners under
  the self-search channel.
- score_broker: EWMA broker satisfaction, or broker reputation if untried

The agent outsources only if the broker beats the self-search channel's current
reduced-form value.
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

# ─────────────────────────────────────────────────────────────────────────────
# Broker reputation (§6c)
# ─────────────────────────────────────────────────────────────────────────────

"""
    broker_reputation(broker) -> Float64

Return the broker's current reputation for untried agents.
Returns 0 if the broker has never had clients (should not occur after initialization).
"""
function broker_reputation(broker::Broker)::Float64
    return broker.last_reputation  # 0.0 if never had clients, else mean client satisfaction
end

"""
    update_broker_reputation!(broker, agents, client_ids)

Update broker reputation to mean broker satisfaction of current clients.
If no clients this period, hold previous value.
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
