"""
    measures.jl

Network position measures (betweenness, Burt's constraint, effective size)
and prediction quality metrics (R², bias, rank correlation).
All measures are computed on the single graph G which includes the broker node.

Betweenness centrality follows Brandes (2001), adapted for single-node
computation on undirected unweighted graphs, with a CSR adjacency structure
for allocation-free neighbor iteration inside threaded BFS.
"""

using Graphs: SimpleGraph, neighbors, nv, ne, has_edge
using Statistics: var, mean
using Base.Threads: @threads, nthreads

# ─────────────────────────────────────────────────────────────────────────────
# Prediction quality
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_prediction_quality(predicted, realized, rng; sigma_eps=0.10) -> PredictionQuality

Compute standard R², bias, and Spearman rank correlation between predicted and
realized values. Predicted values are first converted to a strict ranking, with
ties ordered uniformly at random using `rng`; realized ties retain average ranks.
R² uses the usual `1 - SSE / SST` definition, equivalently `1 - MSE /
var(realized; corrected=false)`.

Returns NaN for all metrics if fewer than 5 observations or if the population
variance of realized values is below `sigma_eps^2 / 6` (too little signal to
meaningfully evaluate R²).
"""
function compute_prediction_quality(
    predicted::AbstractVector{<:Real},
    realized::AbstractVector{<:Real},
    rng::AbstractRNG;
    sigma_eps::Float64=0.10,
)::PredictionQuality
    n = length(predicted)
    n < 5 && return PredictionQuality(NaN, NaN, NaN)
    @assert n == length(realized)

    var_realized = var(realized; corrected=false)
    var_realized < sigma_eps^2 / 6 && return PredictionQuality(NaN, NaN, NaN)

    err_sum = 0.0
    sq_err_sum = 0.0
    @inbounds for idx in eachindex(predicted, realized)
        err = predicted[idx] - realized[idx]
        err_sum += err
        sq_err_sum += err * err
    end
    mse = sq_err_sum / n
    r_squared = 1.0 - mse / var_realized
    bias = err_sum / n
    pred_order = Vector{Int}(undef, n)
    true_order = Vector{Int}(undef, n)
    pred_ranks = Vector{Float64}(undef, n)
    true_ranks = Vector{Float64}(undef, n)
    prepare_true_ranks!(realized, n, true_order, true_ranks)
    rank_corr = prefix_spearman_rank_corr!(
        predicted, n, pred_order, pred_ranks, true_ranks, rng
    )

    return PredictionQuality(r_squared, bias, rank_corr)
end

@inline function prefix_corr(x::Vector{Float64}, y::Vector{Float64}, n::Int)::Float64
    mean_x = 0.0
    mean_y = 0.0
    @inbounds for idx in 1:n
        mean_x += x[idx]
        mean_y += y[idx]
    end
    mean_x /= n
    mean_y /= n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    @inbounds for idx in 1:n
        dx = x[idx] - mean_x
        dy = y[idx] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    end

    denom = den_x * den_y
    return denom > 0.0 ? num / sqrt(denom) : NaN
end

@inline function sort_index_prefix!(
    order::Vector{Int}, values::AbstractVector{<:Real}, n::Int
)
    @inbounds for idx in 1:n
        order[idx] = idx
    end
    @inbounds for idx in 2:n
        current = order[idx]
        current_value = values[current]
        pos = idx - 1
        while pos >= 1 && values[order[pos]] > current_value
            order[pos + 1] = order[pos]
            pos -= 1
        end
        order[pos + 1] = current
    end
    return nothing
end

@inline function tied_ranks_prefix!(
    ranks::Vector{Float64}, order::Vector{Int}, values::AbstractVector{<:Real}, n::Int
)
    sort_index_prefix!(order, values, n)

    idx = 1
    @inbounds while idx <= n
        stop = idx
        value = values[order[idx]]
        while stop < n && values[order[stop + 1]] == value
            stop += 1
        end
        rank = 0.5 * (idx + stop)
        for pos in idx:stop
            ranks[order[pos]] = rank
        end
        idx = stop + 1
    end
    return nothing
end

@inline function prepare_true_ranks!(
    realized::AbstractVector{<:Real}, n::Int, order::Vector{Int}, ranks::Vector{Float64}
)
    tied_ranks_prefix!(ranks, order, realized, n)
    return nothing
end

@inline function strict_random_ranks_prefix!(
    ranks::Vector{Float64},
    order::Vector{Int},
    values::AbstractVector{<:Real},
    n::Int,
    rng::AbstractRNG,
)
    sort_index_prefix!(order, values, n)

    first_tie = 1
    @inbounds while first_tie <= n
        last_tie = first_tie
        value = values[order[first_tie]]
        while last_tie < n && values[order[last_tie + 1]] == value
            last_tie += 1
        end
        for pos in last_tie:-1:(first_tie + 1)
            swap_pos = rand(rng, first_tie:pos)
            order[pos], order[swap_pos] = order[swap_pos], order[pos]
        end
        first_tie = last_tie + 1
    end

    @inbounds for rank in 1:n
        ranks[order[rank]] = rank
    end
    return nothing
end

@inline function prefix_spearman_rank_corr!(
    predicted::AbstractVector{<:Real},
    n::Int,
    pred_order::Vector{Int},
    pred_ranks::Vector{Float64},
    true_ranks::Vector{Float64},
    rng::AbstractRNG,
)::Float64
    strict_random_ranks_prefix!(pred_ranks, pred_order, predicted, n, rng)
    return prefix_corr(pred_ranks, true_ranks, n)
end

"""
    compute_prediction_quality_with_true_ranks!(predicted, realized, n;
                                                rng,
                                                sigma_eps, pred_order,
                                                pred_ranks, true_ranks)

Low-allocation prefix variant of `compute_prediction_quality` for repeated use on
small fixed-length buffers where the realized ranks were already prepared.
"""
function compute_prediction_quality_with_true_ranks!(
    predicted::Vector{Float64},
    realized::Vector{Float64},
    n::Int;
    rng::AbstractRNG,
    sigma_eps::Float64=0.10,
    pred_order::Vector{Int},
    pred_ranks::Vector{Float64},
    true_ranks::Vector{Float64},
)::PredictionQuality
    n < 5 && return PredictionQuality(NaN, NaN, NaN)

    mean_realized = 0.0
    @inbounds for idx in 1:n
        mean_realized += realized[idx]
    end
    mean_realized /= n

    err_sum = 0.0
    sq_err_sum = 0.0
    var_sum = 0.0
    @inbounds for idx in 1:n
        real = realized[idx]
        err = predicted[idx] - real
        err_sum += err
        sq_err_sum += err * err
        centered = real - mean_realized
        var_sum += centered * centered
    end

    var_realized = var_sum / n
    var_realized < sigma_eps^2 / 6 && return PredictionQuality(NaN, NaN, NaN)

    mse = sq_err_sum / n
    r_squared = 1.0 - mse / var_realized
    bias = err_sum / n
    rank_corr = prefix_spearman_rank_corr!(
        predicted, n, pred_order, pred_ranks, true_ranks, rng
    )

    return PredictionQuality(r_squared, bias, rank_corr)
end

# ─────────────────────────────────────────────────────────────────────────────
# CSR adjacency structure (allocation-free neighbor iteration)
# ─────────────────────────────────────────────────────────────────────────────

"""Compressed Sparse Row adjacency for allocation-free neighbor iteration.
Built once from a SimpleGraph before entering threaded loops."""
struct CSRAdjacency
    offsets::Vector{Int}   # length n+1; neighbors of v are adj[offsets[v]:offsets[v+1]-1]
    adj::Vector{Int}       # flat neighbor list
end

"""Build CSR adjacency from a SimpleGraph. O(n + m) time and space."""
function build_csr(G::SimpleGraph)::CSRAdjacency
    n = nv(G)
    offsets = Vector{Int}(undef, n + 1)
    # Count degrees
    offsets[1] = 1
    @inbounds for v in 1:n
        offsets[v + 1] = offsets[v] + length(neighbors(G, v))
    end
    total = offsets[n + 1] - 1
    adj = Vector{Int}(undef, total)
    # Fill adjacency
    @inbounds for v in 1:n
        idx = offsets[v]
        for w in neighbors(G, v)
            adj[idx] = w
            idx += 1
        end
    end
    return CSRAdjacency(offsets, adj)
end

"""Iterate neighbors of v in CSR. Returns a range into csr.adj."""
@inline function csr_neighbors(csr::CSRAdjacency, v::Int)
    @inbounds return csr.offsets[v]:(csr.offsets[v + 1] - 1)
end

# ─────────────────────────────────────────────────────────────────────────────
# Betweenness centrality (Brandes 2001, single-node, threaded)
# ─────────────────────────────────────────────────────────────────────────────

"""Per-thread scratch buffers for Brandes BFS, reused across source vertices."""
mutable struct BrandesWorkspace
    n::Int
    sigma::Vector{Int}
    dist::Vector{Int}
    delta::Vector{Float64}
    queue::Vector{Int}
    stack::Vector{Int}
    # Flat predecessor storage: pred_data[pred_off[w]:pred_off[w]+pred_count[w]-1]
    pred_off::Vector{Int}       # length n+1 (prefix offsets, rebuilt per source)
    pred_data::Vector{Int}      # flat predecessor list, pre-allocated
    pred_count::Vector{Int}     # per-node predecessor count (reset each source)
end

function BrandesWorkspace(n::Int, max_preds::Int)
    BrandesWorkspace(n,
        Vector{Int}(undef, n), Vector{Int}(undef, n), Vector{Float64}(undef, n),
        Vector{Int}(undef, n), Vector{Int}(undef, n),
        Vector{Int}(undef, n + 1), Vector{Int}(undef, max(max_preds, 1)),
        Vector{Int}(undef, n))
end

"""Ensure workspace is sized for n vertices and max_preds predecessors."""
function ensure_workspace!(ws::BrandesWorkspace, n::Int, max_preds::Int)
    if ws.n < n
        resize!(ws.sigma, n); resize!(ws.dist, n); resize!(ws.delta, n)
        resize!(ws.queue, n); resize!(ws.stack, n)
        resize!(ws.pred_off, n + 1); resize!(ws.pred_count, n)
        ws.n = n
    end
    if length(ws.pred_data) < max_preds
        resize!(ws.pred_data, max_preds)
    end
    return ws
end

# Module-level thread-local workspaces
const _BRANDES_WS = Ref{Vector{BrandesWorkspace}}(BrandesWorkspace[])

"""
    compute_betweenness(G, node) -> Float64

Freeman betweenness centrality for a single node using the Brandes (2001)
algorithm adapted for undirected unweighted graphs.

The raw accumulation Σ_s δ_s•(node) counts each unordered (s,t) pair from
both BFS directions. For undirected graphs the standard normalization is
(n-1)(n-2)/2 unordered pairs, and the double-counting cancels with a factor
of 2, giving a combined divisor of (n-1)(n-2) (Brandes, 2001, p. 9).

Uses a CSR adjacency structure for allocation-free neighbor iteration and
pre-allocated per-thread workspaces. The BFS + back-propagation is O(m) per
source, O(nm) total, matching the Brandes complexity bound.
"""
function compute_betweenness(G::SimpleGraph, node::Int)::Float64
    n = nv(G)
    n <= 2 && return 0.0

    # Build CSR once (O(n+m), outside threaded loop)
    csr = build_csr(G)
    m = ne(G)

    # Upper bound on total predecessors per source: each edge contributes at
    # most one predecessor entry per direction, so total_preds <= 2m.
    max_preds = 2 * m + 1

    # Initialize/resize thread-local workspaces before @threads
    n_threads = nthreads()
    ws_vec = _BRANDES_WS[]
    if length(ws_vec) != n_threads
        ws_vec = [BrandesWorkspace(n, max_preds) for _ in 1:n_threads]
        _BRANDES_WS[] = ws_vec
    end
    for t in 1:n_threads
        ensure_workspace!(ws_vec[t], n, max_preds)
    end

    partial_bc = zeros(n_threads)

    # Inner function: run BFS + back-propagation for a range of sources.
    # Extracted to avoid closure boxing and to enable manual chunking.
    function _brandes_chunk!(ws::BrandesWorkspace, csr::CSRAdjacency,
                             n::Int, node::Int, s_range, tid::Int)
        sigma = ws.sigma; dist = ws.dist; delta = ws.delta
        queue = ws.queue; stack = ws.stack
        pred_off = ws.pred_off; pred_data = ws.pred_data; pred_count = ws.pred_count
        local_bc = 0.0

        for s in s_range
            s == node && continue

            # ── Reset per-source state ──
            @inbounds for i in 1:n
                sigma[i] = 0; dist[i] = -1; delta[i] = 0.0; pred_count[i] = 0
            end
            sigma[s] = 1; dist[s] = 0
            queue[1] = s; q_head = 1; q_tail = 1; stack_len = 0

            # ── BFS ──
            @inbounds while q_head <= q_tail
                v = queue[q_head]; q_head += 1
                stack_len += 1; stack[stack_len] = v
                for idx in csr_neighbors(csr, v)
                    w = csr.adj[idx]
                    if dist[w] < 0
                        dist[w] = dist[v] + 1
                        q_tail += 1; queue[q_tail] = w
                    end
                    if dist[w] == dist[v] + 1
                        sigma[w] += sigma[v]
                        pred_count[w] += 1
                    end
                end
            end

            # ── Build predecessor offset table ──
            @inbounds begin
                pred_off[1] = 1
                for i in 1:n
                    pred_off[i + 1] = pred_off[i] + pred_count[i]
                end
            end

            # ── Fill predecessor data ──
            @inbounds for i in 1:n; pred_count[i] = 0; end
            @inbounds for k in 1:stack_len
                v = stack[k]
                for idx in csr_neighbors(csr, v)
                    w = csr.adj[idx]
                    if dist[w] == dist[v] + 1
                        slot = pred_off[w] + pred_count[w]
                        pred_data[slot] = v
                        pred_count[w] += 1
                    end
                end
            end

            # ── Brandes back-propagation (Theorem 6) ──
            @inbounds while stack_len > 0
                w = stack[stack_len]; stack_len -= 1
                sw = sigma[w]
                sw == 0 && continue
                coeff = (1.0 + delta[w]) / sw
                p_begin = pred_off[w]
                p_end = p_begin + pred_count[w] - 1
                for p in p_begin:p_end
                    v = pred_data[p]
                    delta[v] += sigma[v] * coeff
                end
                if w == node
                    local_bc += delta[w]
                end
            end
        end  # for s

        partial_bc[tid] += local_bc
    end

    # Chunk sources across threads (one task per thread, not per source).
    # This reduces scheduling overhead from O(n) task creations to O(nthreads).
    if n_threads > 1
        chunk_size = cld(n, n_threads)
        @threads for tid in 1:n_threads
            s_lo = (tid - 1) * chunk_size + 1
            s_hi = min(tid * chunk_size, n)
            _brandes_chunk!(ws_vec[tid], csr, n, node, s_lo:s_hi, tid)
        end
    else
        _brandes_chunk!(ws_vec[1], csr, n, node, 1:n, 1)
    end

    bc = sum(partial_bc)

    # Normalization for undirected graphs (Brandes, 2001):
    # Raw sum counts each unordered pair (s,t) from both directions.
    # Standard normalized betweenness divides by (n-1)(n-2)/2 pairs and
    # corrects for double-counting by dividing by 2, yielding (n-1)(n-2).
    norm_factor = Float64((n - 1) * (n - 2))
    return norm_factor > 0.0 ? bc / norm_factor : 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Burt's constraint and effective size
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_burt_constraint(G, node) -> Float64

Burt's aggregate network constraint (Burt, 1992; Everett & Borgatti, 2020):

    C_i = Σ_{j∈N(i)} (p_ij + Σ_{k∈N(i)} p_ik · p_kj)²

where p_ij = 1/d_i (proportion of i's ties invested in j) for unweighted
networks, and p_kj = 1/d_k (proportion of k's ties invested in j).
Note: the indirect term uses k's degree, not i's degree.
Returns 1.0 if isolated.
"""
function compute_burt_constraint(G::SimpleGraph, node::Int)::Float64
    nbrs = neighbors(G, node)
    deg = length(nbrs)
    deg == 0 && return 1.0
    p_i = 1.0 / deg

    constraint = 0.0
    for j in nbrs
        c_ij = p_i  # direct: p_ij = 1/d_i
        for k in nbrs
            k == j && continue
            if has_edge(G, k, j)
                # indirect: p_ik * p_kj = (1/d_i) * (1/d_k)
                c_ij += p_i / length(neighbors(G, k))
            end
        end
        constraint += c_ij * c_ij
    end
    return constraint
end

"""
    compute_effective_size(G, node) -> Float64

Burt's effective size (Burt, 1992; Borgatti, 1997):

    s_i = d_i - 2t_i / d_i

where d_i is i's degree and t_i is the number of ties among i's neighbors
(not counting ties to i). Equivalently:

    s_i = Σ_{j∈N(i)} [1 - Σ_{q∈N(i), q≠j} (1/d_i) · m_jq]

where m_jq = 1 if j and q are connected, 0 otherwise.
Returns 0.0 if isolated.
"""
function compute_effective_size(G::SimpleGraph, node::Int)::Float64
    nbrs = neighbors(G, node)
    deg = length(nbrs)
    deg == 0 && return 0.0
    p = 1.0 / deg

    # Count ties among neighbors (each undirected edge counted once per
    # ordered pair in the double loop, so the sum = 2t_i * p = 2t_i/d_i)
    redundancy = 0.0
    for j in nbrs
        for h in nbrs
            h == j && continue
            if has_edge(G, j, h)
                redundancy += p  # p_iq * m_jq = (1/d_i) * 1
            end
        end
    end
    return deg - redundancy
end

"""
    update_cached_network_measures!(state)

Recompute betweenness, constraint, and effective size for the broker node.
"""
function update_cached_network_measures!(state::ModelState)
    broker_node = state.broker.node_id
    state.cached_network.betweenness = compute_betweenness(state.G, broker_node)
    state.cached_network.constraint = compute_burt_constraint(state.G, broker_node)
    state.cached_network.effective_size = compute_effective_size(state.G, broker_node)
    return nothing
end
