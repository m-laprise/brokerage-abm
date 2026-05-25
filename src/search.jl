"""
    search.jl

Offer construction helpers for the shared active-demand market.
"""

using Graphs: neighbors
using Random: AbstractRNG
using StatsBase: sample

@inline function ensure_nbr_mask!(ws::SimWorkspace, N::Int)::Vector{Bool}
    search = ws.search
    if length(search.nbr_mask) < N + 1
        old_len = length(search.nbr_mask)
        resize!(search.nbr_mask, N + 1)
        @inbounds for i in (old_len + 1):(N + 1)
            search.nbr_mask[i] = false
        end
    end
    return search.nbr_mask
end

@inline function ensure_access_seen!(ws::SimWorkspace, N::Int)::Vector{Bool}
    broker_pairs = ws.broker_pairs
    if length(broker_pairs.access_seen) < N
        old_len = length(broker_pairs.access_seen)
        resize!(broker_pairs.access_seen, N)
        @inbounds for i in (old_len + 1):N
            broker_pairs.access_seen[i] = false
        end
    end
    return broker_pairs.access_seen
end

@inline function ensure_mask!(
    mask::Vector{Bool}, touched::Vector{Int}, N::Int
)::Vector{Bool}
    if length(mask) < N
        old_len = length(mask)
        resize!(mask, N)
        @inbounds for i in (old_len + 1):N
            mask[i] = false
        end
    end
    @inbounds for i in touched
        mask[i] = false
    end
    empty!(touched)
    return mask
end

function reset_offer_book!(offer_book::OfferBook, N::Int)
    if size(offer_book.offer_index, 1) != N || size(offer_book.offer_index, 2) != N
        offer_book.offer_index = zeros(Int, N, N)
        empty!(offer_book.offer_index_touched)
    else
        @inbounds for idx in offer_book.offer_index_touched
            offer_book.offer_index[idx] = 0
        end
        empty!(offer_book.offer_index_touched)
    end
    empty!(offer_book.offers)
    empty!(offer_book.offer_pairs)
    return nothing
end

reset_offer_book!(ws::SimWorkspace, N::Int) = reset_offer_book!(ws.offer_book, N)

@inline function add_offer!(
    offer_book::OfferBook,
    from_id::Int,
    to_id::Int,
    channel::Symbol,
    predicted_value::Float64,
)::Bool
    @assert from_id != to_id "Self-offer attempted for agent $from_id"
    offer_index = offer_book.offer_index
    @inbounds offer_index[from_id, to_id] != 0 && return false

    push!(offer_book.offers, DirectedOffer(from_id, to_id, channel, predicted_value))
    offer_id = length(offer_book.offers)
    @inbounds offer_index[from_id, to_id] = offer_id
    push!(offer_book.offer_index_touched, from_id + (to_id - 1) * size(offer_index, 1))

    @inbounds if offer_index[to_id, from_id] == 0
        lo = min(from_id, to_id)
        hi = max(from_id, to_id)
        push!(offer_book.offer_pairs, (lo, hi))
    end
    return true
end

@inline function add_offer!(
    ws::SimWorkspace, from_id::Int, to_id::Int, channel::Symbol, predicted_value::Float64
)::Bool
    return add_offer!(ws.offer_book, from_id, to_id, channel, predicted_value)
end

@inline function offer_ids(offer_book::OfferBook, i::Int, j::Int)::Tuple{Int,Int}
    return @inbounds (offer_book.offer_index[i, j], offer_book.offer_index[j, i])
end

@inline offer_at(offer_book::OfferBook, idx::Int)::DirectedOffer =
    @inbounds offer_book.offers[idx]

function sample_period_strangers!(
    out::Vector{Int}, N::Int, n_strangers::Int, rng::AbstractRNG
)
    empty!(out)
    n = min(n_strangers, N)
    n <= 0 && return out
    if n == N
        append!(out, 1:N)
    else
        append!(out, sample(rng, 1:N, n; replace=false))
    end
    return out
end

function append_self_search_offers!(
    ws::SimWorkspace,
    agent::Agent,
    demand_count::Int,
    agents::Vector{Agent},
    G::SimpleGraph,
    broker_node::Int,
    period_strangers::Vector{Int},
    r::Float64;
    current_match_index_ready::Bool=false,
    captured_origin_mask::Union{Vector{Bool},Nothing}=nothing,
)::Int
    demand_count <= 0 && return 0
    agent_id = agent.id
    N = length(agents)
    current_match_index_ready || rebuild_current_match_index!(ws, agents)

    search = ws.search
    candidate_ids = search.neighbor_ids
    candidate_vals = search.neighbor_evals
    empty!(candidate_ids)
    empty!(candidate_vals)
    nbr_mask = ensure_nbr_mask!(ws, N)
    nbr_marked = search.nbr_marked
    empty!(nbr_marked)

    @inbounds for nbr in neighbors(G, agent_id)
        nbr == broker_node && continue
        (nbr < 1 || nbr > N) && continue
        !isnothing(captured_origin_mask) && captured_origin_mask[nbr] && continue
        nbr_mask[nbr] = true
        push!(nbr_marked, nbr)
        has_current_match(ws, agent_id, nbr) && continue
        mean_q = partner_mean(agent, nbr)
        if !isnan(mean_q) && mean_q > r
            push!(candidate_ids, nbr)
            push!(candidate_vals, mean_q)
        end
    end

    @inbounds for j in period_strangers
        j == agent_id && continue
        !isnothing(captured_origin_mask) && captured_origin_mask[j] && continue
        nbr_mask[j] && continue
        has_current_match(ws, agent_id, j) && continue
        q_hat = predict_nn!(agent.nn, agent.predict_buf, agents[j].type)
        q_hat > r || continue
        push!(candidate_ids, j)
        push!(candidate_vals, q_hat)
    end

    @inbounds for nbr in nbr_marked
        nbr_mask[nbr] = false
    end

    n_candidates = length(candidate_ids)
    n_candidates == 0 && return 0
    sort_pairs = search.sort_pairs
    length(sort_pairs) < n_candidates && resize!(sort_pairs, n_candidates)
    @inbounds for idx in 1:n_candidates
        sort_pairs[idx] = (-candidate_vals[idx], idx)
    end
    sort!(view(sort_pairs, 1:n_candidates); alg=QuickSort)

    sent = 0
    offer_book = ws.offer_book
    @inbounds for rank_idx in 1:n_candidates
        sent == demand_count && break
        candidate_idx = sort_pairs[rank_idx][2]
        j = candidate_ids[candidate_idx]
        if add_offer!(offer_book, agent_id, j, :self, candidate_vals[candidate_idx])
            sent += 1
        end
    end
    return sent
end

function collect_broker_access_ids!(
    out::Vector{Int},
    broker::Broker,
    agents::Vector{Agent},
    ws::SimWorkspace;
    captured_origin_mask::Union{Vector{Bool},Nothing}=nothing,
)::Int
    empty!(out)
    N = length(agents)
    access_seen = ensure_access_seen!(ws, N)
    access_touched = ws.broker_pairs.access_touched
    empty!(access_touched)

    for ids in (broker.roster, broker.current_clients)
        @inbounds for rid in ids
            (rid < 1 || rid > N) && continue
            !isnothing(captured_origin_mask) && captured_origin_mask[rid] && continue
            access_seen[rid] && continue
            access_seen[rid] = true
            push!(access_touched, rid)
            push!(out, rid)
        end
    end

    @inbounds for rid in access_touched
        access_seen[rid] = false
    end
    return length(out)
end

function prepare_broker_pair_scores!(
    ws::SimWorkspace,
    broker::Broker,
    broker_demanders::Vector{Int},
    broker_access::Vector{Int},
    agents::Vector{Agent},
    params::ModelParams,
)
    broker_pairs = ws.broker_pairs
    pair_scores = broker_pairs.broker_pair_scores
    empty!(pair_scores)
    isempty(broker_demanders) && return pair_scores
    isempty(broker_access) && return pair_scores

    N = length(agents)
    demander_mask = ensure_mask!(
        broker_pairs.broker_demander_mask, broker_pairs.broker_demander_touched, N
    )
    access_mask = ensure_mask!(
        broker_pairs.broker_access_mask, broker_pairs.broker_access_touched, N
    )

    @inbounds for did in broker_demanders
        demander_mask[did] = true
        push!(broker_pairs.broker_demander_touched, did)
    end
    @inbounds for rid in broker_access
        access_mask[rid] = true
        push!(broker_pairs.broker_access_touched, rid)
    end

    @inbounds for did in broker_demanders
        for rid in broker_access
            did == rid && continue
            if did > rid && demander_mask[rid] && access_mask[did]
                continue
            end
            lo = min(did, rid)
            hi = max(did, rid)
            push!(pair_scores, (0.0, lo, hi))
        end
    end

    n_pairs = length(pair_scores)
    n_pairs == 0 && return pair_scores

    d = params.d
    d2 = 2 * d
    n_pred = 2 * n_pairs
    if size(broker_pairs.Z_batch, 1) != d2 || size(broker_pairs.Z_batch, 2) < n_pred
        cap = max(n_pred, 2 * size(broker_pairs.Z_batch, 2), 256)
        broker_pairs.Z_batch = Matrix{Float64}(undef, d2, cap)
        broker_pairs.H_batch = Matrix{Float64}(undef, params.h_b, cap)
        resize!(broker_pairs.Y_batch, cap)
    end
    Z_batch = broker_pairs.Z_batch
    H_batch = broker_pairs.H_batch
    Y_batch = broker_pairs.Y_batch

    @inbounds for pair_idx in 1:n_pairs
        _, i, j = pair_scores[pair_idx]
        xi = agents[i].type
        xj = agents[j].type
        col_a = 2 * pair_idx - 1
        col_b = col_a + 1
        for k in 1:d
            Z_batch[k, col_a] = xi[k]
            Z_batch[d + k, col_a] = xj[k]
            Z_batch[k, col_b] = xj[k]
            Z_batch[d + k, col_b] = xi[k]
        end
    end

    predict_nn_batch!(broker.nn, H_batch, Y_batch, Z_batch, n_pred)

    @inbounds for pair_idx in 1:n_pairs
        _, i, j = pair_scores[pair_idx]
        score = 0.5 * (Y_batch[2 * pair_idx - 1] + Y_batch[2 * pair_idx])
        pair_scores[pair_idx] = (-score, i, j)
    end
    sort!(pair_scores; alg=QuickSort)
    return pair_scores
end

function append_broker_offers!(
    ws::SimWorkspace,
    demand_agent_ids::Vector{Int},
    demand_channels::Vector{Symbol},
    demand_counts::Vector{Int},
    agents::Vector{Agent},
    broker::Broker,
    params::ModelParams,
    r::Float64;
    remaining_demand::Union{Vector{Int},Nothing}=nothing,
    captured_origin_mask::Union{Vector{Bool},Nothing}=nothing,
)::Int
    broker_pairs = ws.broker_pairs
    broker_demanders = broker_pairs.period_broker_demanders
    empty!(broker_demanders)
    remaining = isnothing(remaining_demand) ? ws.ledger.offer_remaining : remaining_demand
    N = length(agents)
    if length(remaining) < N
        resize!(remaining, N)
    end
    isnothing(remaining_demand) && fill!(remaining, 0)

    @inbounds for idx in eachindex(demand_agent_ids)
        demand_channels[idx] == :broker || continue
        did = demand_agent_ids[idx]
        !isnothing(captured_origin_mask) && captured_origin_mask[did] && continue
        isnothing(remaining_demand) || remaining[did] > 0 || continue
        push!(broker_demanders, did)
        isnothing(remaining_demand) && (remaining[did] = demand_counts[idx])
    end
    isempty(broker_demanders) && return 0

    broker_access = broker_pairs.period_broker_access_ids
    collect_broker_access_ids!(
        broker_access, broker, agents, ws; captured_origin_mask=captured_origin_mask
    )
    isempty(broker_access) && return 0

    pair_scores = prepare_broker_pair_scores!(
        ws, broker, broker_demanders, broker_access, agents, params
    )
    isempty(pair_scores) && return 0

    demander_mask = broker_pairs.broker_demander_mask
    access_mask = broker_pairs.broker_access_mask
    offer_book = ws.offer_book
    sent = 0
    @inbounds for item in pair_scores
        neg_score, i, j = item
        score = -neg_score
        score <= r && break

        if demander_mask[i] && access_mask[j] && remaining[i] > 0
            if add_offer!(offer_book, i, j, :broker, score)
                remaining[i] -= 1
                sent += 1
            end
        end
        if demander_mask[j] && access_mask[i] && remaining[j] > 0
            if add_offer!(offer_book, j, i, :broker, score)
                remaining[j] -= 1
                sent += 1
            end
        end
    end
    return sent
end
