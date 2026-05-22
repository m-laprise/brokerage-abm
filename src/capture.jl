"""
    capture.jl

Resource capture (Model 1): literal same-period principal acquisition.
The broker evaluates whole counterparty blocks once, before within-period
rounds. If a block is worth taking, the broker acquires the full currently
available block immediately, removes it from the open market for the rest of
the period, and then tries to deploy that owned inventory round by round to
outsourced demanders. Unsold relationship positions expire at period end and
count as realized zero-value exposures for κ_b^t.
"""

using Graphs: has_edge, SimpleGraph
using Random: AbstractRNG

# ─────────────────────────────────────────────────────────────────────────────
# Counterparty ask price
# ─────────────────────────────────────────────────────────────────────────────

"""
    counterparty_ask(agent, q_cal) -> Float64

The counterparty's ask price: its average realized match quality from history,
or q_cal if it has no history. This is the minimum price the counterparty
accepts for selling its currently available block to the broker.
"""
function counterparty_ask(agent::Agent, q_cal::Float64)::Float64
    n = agent.history_count
    n <= 0 && return q_cal
    total = 0.0
    @inbounds for i in 1:n
        total += agent.history_q[i]
    end
    return total / n
end

"""Expected per-position advantage of capture relative to standard placement."""
capture_slot_margin(q_hat_b::Float64, ask_j::Float64, phi::Float64)::Float64 =
    q_hat_b - ask_j - phi

# ─────────────────────────────────────────────────────────────────────────────
# Confidence update
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_capture_confidence_mae!(broker, abs_error_sum, n_errors, omega) -> Nothing

Update the broker's live confidence scale from realized broker-controlled
exposure errors. If confidence has not yet been initialized, the first period
with any broker-controlled exposures sets κ directly to that period MAE. No-op
when there are no such exposures in the period.
"""
function update_capture_confidence_mae!(
    broker::Broker, abs_error_sum::Float64, n_errors::Int, omega::Float64
)
    n_errors <= 0 && return nothing
    current_mae = abs_error_sum / n_errors
    if !broker.capture_confidence_ready
        broker.capture_confidence_mae = current_mae
        broker.capture_confidence_ready = true
    else
        broker.capture_confidence_mae =
            (1.0 - omega) * broker.capture_confidence_mae + omega * current_mae
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Whole-block capture planning
# ─────────────────────────────────────────────────────────────────────────────

@inline principal_inventory_remaining_slots(ws::SimWorkspace, block_idx::Int)::Int =
    length(ws.principal_inventory_slot_qhats[block_idx]) -
    ws.principal_inventory_next_slot[block_idx] + 1

function reset_principal_inventory!(ws::SimWorkspace, N::Int)
    if length(ws.principal_reserved_capacity) < N
        old = length(ws.principal_reserved_capacity)
        resize!(ws.principal_reserved_capacity, N)
        @inbounds for idx in (old + 1):N
            ws.principal_reserved_capacity[idx] = 0
        end
    end
    @inbounds for agent_id in ws.principal_reserved_touched
        ws.principal_reserved_capacity[agent_id] = 0
    end
    empty!(ws.principal_reserved_touched)
    empty!(ws.principal_inventory_ids)
    empty!(ws.principal_inventory_asks)
    empty!(ws.principal_inventory_next_slot)
    empty!(ws.principal_inventory_round_ids)
    empty!(ws.principal_inventory_round_blocks)
    empty!(ws.principal_inventory_round_remaining)
    empty!(ws.capture_plan_remaining)
    empty!(ws.principal_round_taken)
    return nothing
end

function reserve_principal_capacity!(ws::SimWorkspace, counterparty_id::Int, n_slots::Int)
    if ws.principal_reserved_capacity[counterparty_id] == 0
        push!(ws.principal_reserved_touched, counterparty_id)
    end
    ws.principal_reserved_capacity[counterparty_id] += n_slots
    return nothing
end

"""
    score_capture_block!(sort_pairs, Q, roster_members, roster_idx, broker_demanders,
                         broker_slot_caps, agents, cal, K; ws, reserved_capacity=nothing)

Evaluate one candidate counterparty block under the whole-block rule. The
counterparty contributes all currently available relationship positions, and
the broker values the block by assigning those positions to the best distinct
current outsourced demanders.
Returns a NamedTuple with the block margin net of standard fees, whether the
whole block can be placed immediately, and the number of distinct demanders
among the selected positive-margin uses.
"""
function score_capture_block!(
    sort_pairs::Vector{Tuple{Float64,Int}},
    Q::AbstractMatrix{Float64},
    roster_members::Vector{Int},
    roster_idx::Int,
    broker_demanders::Vector{Int},
    broker_slot_caps::Vector{Int},
    agents::Vector{Agent},
    cal::CalibrationConstants,
    K::Int;
    reserved_capacity::Union{Vector{Int},Nothing}=nothing,
    ws::SimWorkspace,
)
    counterparty_id = roster_members[roster_idx]
    blocked_j = isnothing(reserved_capacity) ? 0 : reserved_capacity[counterparty_id]
    cap_j = available_capacity(agents[counterparty_id], K, blocked_j)
    ask_j = counterparty_ask(agents[counterparty_id], cal.q_cal)
    n_demanders = length(broker_demanders)

    if cap_j <= 0 || n_demanders == 0
        return (
            counterparty_id=counterparty_id,
            ask_j=ask_j,
            cap_j=cap_j,
            block_margin=(-Inf),
            profitable_depth=0,
            filled=false,
        )
    end

    length(sort_pairs) < n_demanders && resize!(sort_pairs, n_demanders)

    @inbounds for di in 1:n_demanders
        did = broker_demanders[di]
        slot_cap = broker_slot_caps[di]
        q_hat = Q[di, roster_idx]
        margin = if slot_cap <= 0 ||
                    did == counterparty_id ||
                    q_hat == -Inf ||
                    has_current_match(ws, did, counterparty_id)
            -Inf
        else
            capture_slot_margin(q_hat, ask_j, cal.phi)
        end
        sort_pairs[di] = (-margin, di)
    end
    sort!(view(sort_pairs, 1:n_demanders); alg=QuickSort)

    used = 0
    block_margin = 0.0
    profitable_depth = 0
    @inbounds for k in 1:n_demanders
        neg_margin, di = sort_pairs[k]
        margin = -neg_margin
        isfinite(margin) || break
        slot_cap = broker_slot_caps[di]
        slot_cap <= 0 && continue
        cap_j - used <= 0 && break
        block_margin += margin
        margin > 0.0 && (profitable_depth += 1)
        used += 1
        used == cap_j && break
    end

    return (
        counterparty_id=counterparty_id,
        ask_j=ask_j,
        cap_j=cap_j,
        block_margin=block_margin,
        profitable_depth=profitable_depth,
        filled=used == cap_j,
    )
end

"""
    best_capture_block!(sort_pairs, Q, roster_members, broker_demanders,
                        broker_slot_caps, agents, broker, cal, K; ws, reserved_capacity=nothing)

Pick the current best whole block to capture, if any. Blocks are ranked by
their net expected advantage after subtracting the block-size confidence hurdle
`cap_j * κ_b^t`. Returns `nothing` when no block is worth taking.
"""
function best_capture_block!(
    sort_pairs::Vector{Tuple{Float64,Int}},
    Q::AbstractMatrix{Float64},
    roster_members::Vector{Int},
    broker_demanders::Vector{Int},
    broker_slot_caps::Vector{Int},
    agents::Vector{Agent},
    broker::Broker,
    cal::CalibrationConstants,
    K::Int;
    reserved_capacity::Union{Vector{Int},Nothing}=nothing,
    ws::SimWorkspace,
)
    best_ri = 0
    best_excess = 0.0
    best_score = nothing

    @inbounds for roster_idx in eachindex(roster_members)
        score = score_capture_block!(
            sort_pairs,
            Q,
            roster_members,
            roster_idx,
            broker_demanders,
            broker_slot_caps,
            agents,
            cal,
            K;
            reserved_capacity=reserved_capacity,
            ws=ws,
        )
        score.filled || continue
        score.profitable_depth >= 2 || continue
        excess = score.block_margin - score.cap_j * broker.capture_confidence_mae
        if excess > best_excess
            best_excess = excess
            best_ri = roster_idx
            best_score = score
        end
    end

    best_ri == 0 && return nothing
    return (; roster_idx=best_ri, excess=best_excess, best_score...)
end

"""
    capture_block_slots!(slot_qhats, plan_remaining, sort_pairs, Q, roster_idx,
                         broker_demanders, broker_slot_caps, ask_j, counterparty_id, cal, cap_j)

Consume the best current distinct uses of a captured block, write one
acquisition-time q̂ per position into `slot_qhats`, and decrement the
corresponding planned demand.
"""
function capture_block_slots!(
    slot_qhats::Vector{Float64},
    plan_remaining::Vector{Int},
    sort_pairs::Vector{Tuple{Float64,Int}},
    Q::AbstractMatrix{Float64},
    roster_idx::Int,
    broker_demanders::Vector{Int},
    broker_slot_caps::Vector{Int},
    ask_j::Float64,
    counterparty_id::Int,
    cal::CalibrationConstants,
    cap_j::Int,
)::Int
    empty!(slot_qhats)
    length(sort_pairs) < length(broker_demanders) &&
        resize!(sort_pairs, length(broker_demanders))

    @inbounds for di in eachindex(broker_demanders)
        did = broker_demanders[di]
        slot_cap = broker_slot_caps[di]
        q_hat = Q[di, roster_idx]
        margin = if slot_cap <= 0 || did == counterparty_id || q_hat == -Inf
            -Inf
        else
            capture_slot_margin(q_hat, ask_j, cal.phi)
        end
        sort_pairs[di] = (-margin, di)
    end
    sort!(view(sort_pairs, 1:length(broker_demanders)); alg=QuickSort)

    used = 0
    @inbounds for k in 1:length(broker_demanders)
        neg_margin, di = sort_pairs[k]
        margin = -neg_margin
        isfinite(margin) || break
        slot_cap = broker_slot_caps[di]
        slot_cap <= 0 && continue
        cap_j - used <= 0 && break

        q_hat = Q[di, roster_idx]
        push!(slot_qhats, q_hat)
        plan_remaining[di] -= 1
        used += 1
        used == cap_j && break
    end

    return used
end

"""
    plan_period_capture!(demand_agent_ids, demand_channels, demand_counts,
                         agents, broker, params, cal; ws)

Before within-period rounds begin, greedily acquire whole counterparty blocks
that clear the Model 1 rule on the current residual state. Acquired positions
are stored as broker-owned same-period inventory in `ws` and are removed from
open market capacity immediately. Returns the number of acquired blocks.
"""
function plan_period_capture!(
    demand_agent_ids::Vector{Int},
    demand_channels::Vector{Symbol},
    demand_counts::Vector{Int},
    agents::Vector{Agent},
    broker::Broker,
    params::ModelParams,
    cal::CalibrationConstants;
    ws::SimWorkspace,
)::Int
    (!params.enable_principal || !broker.capture_confidence_ready) && return 0

    N = length(agents)
    K = params.K
    reset_principal_inventory!(ws, N)

    broker_demanders = ws.round_broker_demanders
    empty!(broker_demanders)
    plan_remaining = ws.capture_plan_remaining
    empty!(plan_remaining)
    broker_slot_caps = ws.demander_remaining
    empty!(broker_slot_caps)

    @inbounds for idx in eachindex(demand_agent_ids)
        demand_channels[idx] == :broker || continue
        push!(broker_demanders, demand_agent_ids[idx])
        push!(plan_remaining, demand_counts[idx])
    end
    isempty(broker_demanders) && return 0
    rebuild_current_match_index!(ws, agents)

    broker_matrix = prepare_broker_round_matrix!(
        broker, broker_demanders, agents, params; ws=ws
    )
    broker_matrix.n_roster == 0 && return 0

    resize!(broker_slot_caps, length(broker_demanders))
    n_blocks = 0

    while true
        @inbounds for di in eachindex(broker_demanders)
            did = broker_demanders[di]
            broker_slot_caps[di] = min(
                plan_remaining[di],
                available_capacity(agents[did], K, ws.principal_reserved_capacity[did]),
            )
        end

        best = best_capture_block!(
            broker_matrix.sort_pairs,
            broker_matrix.Q,
            broker_matrix.roster_members,
            broker_demanders,
            broker_slot_caps,
            agents,
            broker,
            cal,
            K;
            reserved_capacity=ws.principal_reserved_capacity,
            ws=ws,
        )
        isnothing(best) && break

        block_idx = length(ws.principal_inventory_ids) + 1
        if length(ws.principal_inventory_slot_qhats) < block_idx
            push!(ws.principal_inventory_slot_qhats, Float64[])
        end
        slot_qhats = ws.principal_inventory_slot_qhats[block_idx]
        used = capture_block_slots!(
            slot_qhats,
            plan_remaining,
            broker_matrix.sort_pairs,
            broker_matrix.Q,
            best.roster_idx,
            broker_demanders,
            broker_slot_caps,
            best.ask_j,
            best.counterparty_id,
            cal,
            best.cap_j,
        )
        used == best.cap_j || break

        push!(ws.principal_inventory_ids, best.counterparty_id)
        push!(ws.principal_inventory_asks, best.ask_j)
        push!(ws.principal_inventory_next_slot, 1)
        reserve_principal_capacity!(ws, best.counterparty_id, best.cap_j)
        n_blocks += 1
    end

    return n_blocks
end

"""
    execute_inventory_round!(accepted, remaining_demand, broker_indices, broker_demanders,
                             agents, broker, env, G, params, cal, rng,
                             was_connected_i, was_connected_j; ws, Ax_buf, Bx_buf)

Execute one round of principal placements from already acquired broker-owned
inventory. Each active outsourced demander can receive at most one owned
position in the round. The current-pair index is rebuilt by default for direct
calls and reused by the round matcher in the hot path. Returns the number of
realized principal-mode matches.
"""
function execute_inventory_round!(
    accepted::Vector{AcceptedMatch},
    remaining_demand::Vector{Int},
    broker_indices::Vector{Int},
    broker_demanders::Vector{Int},
    agents::Vector{Agent},
    broker::Broker,
    env::MatchingEnv,
    G::SimpleGraph,
    params::ModelParams,
    cal::CalibrationConstants,
    rng::AbstractRNG,
    was_connected_i::Vector{Int},
    was_connected_j::Vector{Int};
    ws::SimWorkspace,
    Ax_buf::Vector{Float64},
    Bx_buf::Vector{Float64},
    current_match_index_ready::Bool=false,
)::Int
    isempty(broker_demanders) && return 0
    isempty(ws.principal_inventory_ids) && return 0
    current_match_index_ready || rebuild_current_match_index!(ws, agents)

    round_ids = ws.principal_inventory_round_ids
    round_blocks = ws.principal_inventory_round_blocks
    round_remaining = ws.principal_inventory_round_remaining
    empty!(round_ids)
    empty!(round_blocks)
    empty!(round_remaining)

    @inbounds for block_idx in eachindex(ws.principal_inventory_ids)
        remaining_slots = principal_inventory_remaining_slots(ws, block_idx)
        remaining_slots > 0 || continue
        push!(round_ids, ws.principal_inventory_ids[block_idx])
        push!(round_blocks, block_idx)
        push!(round_remaining, remaining_slots)
    end
    isempty(round_ids) && return 0

    broker_matrix = prepare_broker_quality_matrix!(
        broker, broker_demanders, round_ids, agents, params; ws=ws
    )
    n_demanders = length(broker_demanders)
    n_inventory = length(round_ids)
    n_entries = n_demanders * n_inventory
    sort_pairs = broker_matrix.sort_pairs
    length(sort_pairs) < n_entries && resize!(sort_pairs, n_entries)

    idx = 0
    @inbounds for ri in 1:n_inventory
        for di in 1:n_demanders
            idx += 1
            did = broker_demanders[di]
            counterparty_id = round_ids[ri]
            q_hat =
                if has_current_match(ws, did, counterparty_id) ||
                    current_open_capacity(
                        agents, did, params.K, ws.principal_reserved_capacity
                    ) <= 0
                    -Inf
                else
                    broker_matrix.Q[di, ri]
                end
            sort_pairs[idx] = (-q_hat, idx)
        end
    end
    sort!(view(sort_pairs, 1:n_entries); alg=QuickSort)

    resize!(ws.principal_round_taken, n_demanders)
    fill!(ws.principal_round_taken, false)
    n_accepts = 0

    @inbounds for k in 1:n_entries
        neg_val, flat = sort_pairs[k]
        val = -neg_val
        val == -Inf && break
        val <= cal.r && break

        di = (flat - 1) % n_demanders + 1
        ri = (flat - 1) ÷ n_demanders + 1
        ws.principal_round_taken[di] && continue
        round_remaining[ri] <= 0 && continue

        did = broker_demanders[di]
        counterparty_id = round_ids[ri]
        has_current_match(ws, did, counterparty_id) && continue
        current_open_capacity(agents, did, params.K, ws.principal_reserved_capacity) > 0 ||
            continue
        block_idx = round_blocks[ri]
        slot_idx = ws.principal_inventory_next_slot[block_idx]
        capture_qhat = ws.principal_inventory_slot_qhats[block_idx][slot_idx]
        ask_j = ws.principal_inventory_asks[block_idx]

        has_edge(G, did, counterparty_id) && begin
            push!(was_connected_i, did)
            push!(was_connected_j, counterparty_id)
        end

        pm = ProposedMatch(did, counterparty_id, :broker, val, true, ask_j, capture_qhat)
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
            reserved_capacity=ws.principal_reserved_capacity,
            ws=ws,
        )
        remaining_demand[broker_indices[di]] -= 1
        ws.principal_inventory_next_slot[block_idx] += 1
        round_remaining[ri] -= 1
        ws.principal_round_taken[di] = true
        n_accepts += 1
    end

    return n_accepts
end

# ─────────────────────────────────────────────────────────────────────────────
# Capture surplus
# ─────────────────────────────────────────────────────────────────────────────

"""
    capture_surplus(q_realized, ask_j) -> Float64

Capture surplus Δq_{ij} = q_{ij} - q̄_j: the realized match output net of the
counterparty's acquisition reservation. Positive values indicate profitable
capture; negative values are realized inventory risk.
"""
function capture_surplus(q_realized::Float64, ask_j::Float64)::Float64
    return q_realized - ask_j
end
