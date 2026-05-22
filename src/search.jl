"""
    search.jl

Round-search preference construction and broker quality-cache helpers.
"""

using Graphs: neighbors
using Random: AbstractRNG
using StatsBase: sample

@inline function current_open_capacity(
    agents::Vector{Agent},
    agent_id::Int,
    K::Int,
    reserved_capacity::Union{Vector{Int},Nothing},
)
    if isnothing(reserved_capacity)
        return available_capacity(agents[agent_id], K)
    end
    return available_capacity(agents[agent_id], K, reserved_capacity[agent_id])
end

@inline function ensure_nbr_mask!(ws::SimWorkspace, N::Int)::Vector{Bool}
    if length(ws.nbr_mask) < N + 1
        old_len = length(ws.nbr_mask)
        resize!(ws.nbr_mask, N + 1)
        @inbounds for i in (old_len + 1):(N + 1)
            ws.nbr_mask[i] = false
        end
    end
    return ws.nbr_mask
end

@inline function ensure_access_seen!(ws::SimWorkspace, N::Int)::Vector{Bool}
    if length(ws.access_seen) < N
        old_len = length(ws.access_seen)
        resize!(ws.access_seen, N)
        @inbounds for i in (old_len + 1):N
            ws.access_seen[i] = false
        end
    end
    return ws.access_seen
end

@inline function ensure_period_open_mask!(ws::SimWorkspace, n_cols::Int)::Vector{Bool}
    if length(ws.period_broker_open_mask) < n_cols
        old_len = length(ws.period_broker_open_mask)
        resize!(ws.period_broker_open_mask, n_cols)
        @inbounds for col in (old_len + 1):n_cols
            ws.period_broker_open_mask[col] = false
        end
    end
    return ws.period_broker_open_mask
end

@inline function ensure_demander_idx!(ws::SimWorkspace, N::Int)::Vector{Int}
    if length(ws.demander_idx) < N
        old_len = length(ws.demander_idx)
        resize!(ws.demander_idx, N)
        @inbounds for i in (old_len + 1):N
            ws.demander_idx[i] = 0
        end
    end
    return ws.demander_idx
end

"""
    append_self_round_preferences!(out, agent, agents, G, broker_node, params, rng, r; ws) -> Int

Append demander-side ranked self-search options for one within-period round.
Each feasible candidate appears at most once, ordered by the demander's current
evaluation (known-partner mean for neighbors, NN prediction for strangers).
Returns the number of appended preferences.
"""
function append_self_round_preferences!(
    out::Vector{ProposedMatch},
    agent::Agent,
    agents::Vector{Agent},
    G::SimpleGraph,
    broker_node::Int,
    params::ModelParams,
    rng::AbstractRNG,
    r::Float64;
    ws::SimWorkspace,
    reserved_capacity::Union{Vector{Int},Nothing}=nothing,
    open_agents::Union{Vector{Int},Nothing}=nothing,
    current_match_index_ready::Bool=false,
)::Int
    K = params.K
    agent_id = agent.id
    N = length(agents)
    current_match_index_ready || rebuild_current_match_index!(ws, agents)

    neighbor_ids = ws.neighbor_ids;
    empty!(neighbor_ids)
    neighbor_evals = ws.neighbor_evals;
    empty!(neighbor_evals)
    stranger_ids = ws.stranger_ids;
    empty!(stranger_ids)
    stranger_evals = ws.stranger_evals;
    empty!(stranger_evals)
    eligible = ws.eligible;
    empty!(eligible)
    nbr_mask = ensure_nbr_mask!(ws, N)
    local_marked = ws.nbr_marked;
    empty!(local_marked)

    @inbounds for nbr in neighbors(G, agent_id)
        nbr == broker_node && continue
        (nbr < 1 || nbr > N) && continue
        nbr_mask[nbr] = true
        push!(local_marked, nbr)
        has_current_match(ws, agent_id, nbr) && continue
        current_open_capacity(agents, nbr, K, reserved_capacity) > 0 || continue
        mean_q = partner_mean(agent, nbr)
        if !isnan(mean_q) && mean_q > r
            push!(neighbor_ids, nbr)
            push!(neighbor_evals, mean_q)
        end
    end

    if params.n_strangers > 0
        if !isnothing(open_agents)
            @inbounds for j in open_agents
                j == agent_id && continue
                nbr_mask[j] && continue
                has_current_match(ws, agent_id, j) && continue
                current_open_capacity(agents, j, K, reserved_capacity) > 0 || continue
                push!(eligible, j)
            end
        else
            @inbounds for j in 1:N
                j == agent_id && continue
                nbr_mask[j] && continue
                has_current_match(ws, agent_id, j) && continue
                current_open_capacity(agents, j, K, reserved_capacity) > 0 || continue
                push!(eligible, j)
            end
        end

        n_sample = min(params.n_strangers, length(eligible))
        if n_sample > 0
            sampled = sample(rng, eligible, n_sample; replace=false)
            @inbounds for j in sampled
                q_hat = predict_nn!(agent.nn, agent.predict_buf, agents[j].type)
                q_hat > r || continue
                push!(stranger_ids, j)
                push!(stranger_evals, q_hat)
            end
        end
    end

    @inbounds for nbr in local_marked
        nbr_mask[nbr] = false
    end

    start_idx = length(out) + 1
    @inbounds for idx in eachindex(neighbor_ids)
        push!(
            out,
            ProposedMatch(
                agent_id, neighbor_ids[idx], :self, neighbor_evals[idx], false, NaN, NaN
            ),
        )
    end
    @inbounds for idx in eachindex(stranger_ids)
        push!(
            out,
            ProposedMatch(
                agent_id, stranger_ids[idx], :self, stranger_evals[idx], false, NaN, NaN
            ),
        )
    end

    n_added = length(out) - start_idx + 1
    n_added <= 0 && return 0

    sort!(view(out, start_idx:length(out)); by=pm -> pm.evaluation, rev=true)
    return n_added
end

function collect_broker_access_ids!(
    out::Vector{Int},
    broker::Broker,
    agents::Vector{Agent},
    K::Int,
    reserved_capacity::Union{Vector{Int},Nothing},
    ws::SimWorkspace,
)::Int
    access_seen = ensure_access_seen!(ws, length(agents))
    access_touched = ws.access_touched;
    empty!(access_touched)

    @inbounds for rid in broker.roster
        (rid < 1 || rid > length(agents)) && continue
        current_open_capacity(agents, rid, K, reserved_capacity) > 0 || continue
        access_seen[rid] && continue
        access_seen[rid] = true
        push!(access_touched, rid)
        push!(out, rid)
    end
    @inbounds for rid in broker.current_clients
        (rid < 1 || rid > length(agents)) && continue
        current_open_capacity(agents, rid, K, reserved_capacity) > 0 || continue
        access_seen[rid] && continue
        access_seen[rid] = true
        push!(access_touched, rid)
        push!(out, rid)
    end

    @inbounds for rid in access_touched
        access_seen[rid] = false
    end
    return length(out)
end

function prepare_broker_quality_matrix!(
    broker::Broker,
    demander_ids::Vector{Int},
    access_ids::Vector{Int},
    agents::Vector{Agent},
    params::ModelParams;
    ws::SimWorkspace,
)
    isempty(demander_ids) &&
        return (Q=ws.Q, roster_members=access_ids, sort_pairs=ws.sort_pairs, n_roster=0)
    isempty(access_ids) &&
        return (Q=ws.Q, roster_members=access_ids, sort_pairs=ws.sort_pairs, n_roster=0)

    d = params.d
    d2 = 2 * d
    h_b = params.h_b
    n_unique = length(demander_ids)
    n_roster = length(access_ids)

    if size(ws.Q, 1) < n_unique || size(ws.Q, 2) < n_roster
        ws.Q = Matrix{Float64}(
            undef, max(n_unique, size(ws.Q, 1)), max(n_roster, size(ws.Q, 2))
        )
    end
    Q = ws.Q

    n_pairs = n_unique * n_roster
    n_self = 0
    @inbounds for ri in 1:n_roster
        rid = access_ids[ri]
        for di in 1:n_unique
            demander_ids[di] == rid && (n_self += 1)
        end
    end
    n_pairs -= n_self

    if size(ws.Z_batch, 1) != d2 || size(ws.Z_batch, 2) < n_pairs
        cap = max(n_pairs, 2 * size(ws.Z_batch, 2), 256)
        ws.Z_batch = Matrix{Float64}(undef, d2, cap)
        ws.H_batch = Matrix{Float64}(undef, h_b, cap)
        resize!(ws.Y_batch, cap)
    end
    Z_batch = ws.Z_batch
    H_batch = ws.H_batch
    Y_batch = ws.Y_batch

    col = 0
    @inbounds for ri in 1:n_roster
        rid = access_ids[ri]
        xj = agents[rid].type
        for di in 1:n_unique
            did = demander_ids[di]
            if did == rid
                Q[di, ri] = -Inf
            else
                col += 1
                xi = agents[did].type
                for k in 1:d
                    Z_batch[k, col] = xi[k]
                    Z_batch[d + k, col] = xj[k]
                end
            end
        end
    end

    n_pairs > 0 && predict_nn_batch!(broker.nn, H_batch, Y_batch, Z_batch, n_pairs)

    col = 0
    @inbounds for ri in 1:n_roster
        rid = access_ids[ri]
        for di in 1:n_unique
            if demander_ids[di] != rid
                col += 1
                Q[di, ri] = Y_batch[col]
            end
        end
    end

    sort_pairs = ws.sort_pairs
    length(sort_pairs) < n_roster && resize!(sort_pairs, n_roster)

    return (Q=Q, roster_members=access_ids, sort_pairs=sort_pairs, n_roster=n_roster)
end

"""
    prepare_broker_round_matrix!(broker, demander_ids, agents, params; ws)

Build the broker's current-round quality matrix over `demander_ids` and the
current broker access set `A^t`. Returns a NamedTuple with the precomputed
matrix, the access-set agent IDs, reusable sort scratch, and the realized
access-set size.
"""
function prepare_broker_round_matrix!(
    broker::Broker,
    demander_ids::Vector{Int},
    agents::Vector{Agent},
    params::ModelParams;
    ws::SimWorkspace,
    reserved_capacity::Union{Vector{Int},Nothing}=nothing,
)
    roster_members = ws.roster_members;
    empty!(roster_members)
    isempty(demander_ids) &&
        return (Q=ws.Q, roster_members=roster_members, sort_pairs=ws.sort_pairs, n_roster=0)

    n_roster = collect_broker_access_ids!(
        roster_members, broker, agents, params.K, reserved_capacity, ws
    )
    if n_roster == 0
        return (Q=ws.Q, roster_members=roster_members, sort_pairs=ws.sort_pairs, n_roster=0)
    end

    return prepare_broker_quality_matrix!(
        broker, demander_ids, roster_members, agents, params; ws=ws
    )
end

"""
    prepare_period_broker_round_cache!(broker, demand_agent_ids, demand_channels,
                                       agents, params; ws, reserved_capacity=nothing)

Build the broker's within-period quality cache once, over all demanders that
chose the broker channel and the post-planning access set that is open at the
start of round matching. Later rounds reuse this matrix and only re-filter the
currently open counterparties.
"""
function prepare_period_broker_round_cache!(
    broker::Broker,
    demand_agent_ids::Vector{Int},
    demand_channels::Vector{Symbol},
    agents::Vector{Agent},
    params::ModelParams;
    ws::SimWorkspace,
    reserved_capacity::Union{Vector{Int},Nothing}=nothing,
)
    period_demanders = ws.period_broker_demanders;
    empty!(period_demanders)
    demander_idx = ensure_demander_idx!(ws, length(agents))
    @inbounds for aid in ws.demander_touched
        demander_idx[aid] = 0
    end
    empty!(ws.demander_touched)

    @inbounds for idx in eachindex(demand_agent_ids)
        demand_channels[idx] == :broker || continue
        push!(period_demanders, demand_agent_ids[idx])
    end

    period_access_ids = ws.period_broker_access_ids;
    empty!(period_access_ids)
    isempty(period_demanders) && return nothing

    collect_broker_access_ids!(
        period_access_ids, broker, agents, params.K, reserved_capacity, ws
    ) == 0 && return nothing

    broker_matrix = prepare_broker_quality_matrix!(
        broker, period_demanders, period_access_ids, agents, params; ws=ws
    )
    n_rows = length(period_demanders)
    n_cols = length(period_access_ids)
    if size(ws.period_broker_ranked_cols, 1) < n_rows ||
        size(ws.period_broker_ranked_cols, 2) < n_cols
        ws.period_broker_ranked_cols = Matrix{Int}(
            undef,
            max(n_rows, size(ws.period_broker_ranked_cols, 1)),
            max(n_cols, size(ws.period_broker_ranked_cols, 2)),
        )
    end

    ranked_cols = ws.period_broker_ranked_cols
    sort_pairs = ws.sort_pairs
    length(sort_pairs) < n_cols && resize!(sort_pairs, n_cols)
    Q = broker_matrix.Q

    @inbounds for row in 1:n_rows
        did = period_demanders[row]
        demander_idx[did] = row
        push!(ws.demander_touched, did)
        for col in 1:n_cols
            sort_pairs[col] = (-Q[row, col], col)
        end
        sort!(view(sort_pairs, 1:n_cols), alg=QuickSort)
        for rank_pos in 1:n_cols
            ranked_cols[row, rank_pos] = sort_pairs[rank_pos][2]
        end
    end
    return nothing
end

"""
    append_broker_round_preferences_from_cache!(out, counts, demander_ids, agents,
                                                params, r; ws, demander_slots=nothing)

Append demander-side ranked broker options using the period-level broker
quality cache prepared once before round matching. Only currently open
counterparties are considered in each round, preserving the live-capacity
logic while avoiding repeated broker NN passes.
"""
function append_broker_round_preferences_from_cache!(
    out::Vector{ProposedMatch},
    counts::Vector{Int},
    demander_ids::Vector{Int},
    agents::Vector{Agent},
    params::ModelParams,
    r::Float64;
    ws::SimWorkspace,
    demander_slots::Union{Vector{Int},Nothing}=nothing,
    reserved_capacity::Union{Vector{Int},Nothing}=nothing,
    round_capacity::Union{Vector{Int},Nothing}=nothing,
    current_match_index_ready::Bool=false,
)
    empty!(out)
    resize!(counts, length(demander_ids))
    fill!(counts, 0)
    isempty(demander_ids) && return out
    current_match_index_ready || rebuild_current_match_index!(ws, agents)

    period_demanders = ws.period_broker_demanders
    period_access_ids = ws.period_broker_access_ids
    isempty(period_demanders) && return out
    isempty(period_access_ids) && return out

    K = params.K
    Q = ws.Q
    ranked_cols = ws.period_broker_ranked_cols
    open_mask = ensure_period_open_mask!(ws, length(period_access_ids))
    open_touched = ws.period_broker_open_touched;
    empty!(open_touched)

    @inbounds for col in eachindex(period_access_ids)
        rid = period_access_ids[col]
        if isnothing(round_capacity)
            current_open_capacity(agents, rid, K, reserved_capacity) > 0 || continue
        else
            round_capacity[rid] > 0 || continue
        end
        open_mask[col] = true
        push!(open_touched, col)
    end
    isempty(open_touched) && return out

    demander_idx = ws.demander_idx
    @inbounds for di in eachindex(demander_ids)
        !isnothing(demander_slots) && demander_slots[di] <= 0 && continue
        did = demander_ids[di]
        row = demander_idx[did]
        row == 0 && error("Broker round cache desynchronized from current demander order")

        for rank_pos in 1:length(period_access_ids)
            col = ranked_cols[row, rank_pos]
            open_mask[col] || continue
            val = Q[row, col]
            val <= r && break
            rid = period_access_ids[col]
            did == rid && continue
            has_current_match(ws, did, rid) && continue
            push!(out, ProposedMatch(did, rid, :broker, val, false, NaN, NaN))
            counts[di] += 1
        end
    end

    @inbounds for col in open_touched
        open_mask[col] = false
    end

    return out
end
