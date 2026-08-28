"""
    invariants.jl

Debug-time state consistency checks. Called via `run_simulation(...; verify=true)`.
"""

using Graphs: nv, has_edge

"""
    verify_invariants(state)

Assert that the simulation state is internally consistent. Intended for test
and debug runs; disable in production for performance.

Checks: population count, graph structure, match distinctness, match symmetry,
match-edge consistency, history buffer validity, broker roster consistency,
finite satisfaction/NN weights, and partner tracking.
"""
function verify_invariants(state::ModelState)
    p = state.params
    N = p.N
    agents = state.agents
    broker = state.broker
    G = state.G

    # ── Population and graph ──
    @assert length(agents) == N "Expected $N agents, got $(length(agents))"
    @assert nv(G) == N + 1 "Graph should have $(N+1) nodes, got $(nv(G))"
    @assert broker.node_id == N + 1 "Broker node_id should be $(N+1), got $(broker.node_id)"

    for i in 1:N
        @assert agents[i].id == i "Agent at index $i has id=$(agents[i].id)"
        @assert !has_edge(G, i, i) "Self-edge on agent $i"
    end

    # ── Distinct current-period counterparties ──
    for i in 1:N
        matches = agents[i].active_matches
        seen = Set{Int}()
        for am in matches
            @assert !(am.partner_id in seen) "Agent $i has duplicate active match with $(am.partner_id)"
            push!(seen, am.partner_id)
        end
    end

    # ── Active match symmetry ──
    for i in 1:N
        for am in agents[i].active_matches
            j = am.partner_id
            @assert 1 <= j <= N "Agent $i has match with invalid partner $j"
            partner_has_i = any(m -> m.partner_id == i, agents[j].active_matches)
            @assert partner_has_i "Agent $i lists partner $j but $j does not list $i"
        end
    end

    # ── Active match edges ──
    for i in 1:N
        for am in agents[i].active_matches
            @assert has_edge(G, i, am.partner_id) "Active match ($i, $(am.partner_id)) but no edge in G"
        end
    end

    # ── History buffer validity ──
    for i in 1:N
        a = agents[i]
        @assert 0 <= a.history_count <= size(a.history_X, 2) "Agent $i: history_count=$(a.history_count) > capacity=$(size(a.history_X, 2))"
        @assert a.history_count <= length(a.history_q) "Agent $i: history_count exceeds history_q length"
        @assert a.n_new_obs >= 0 "Agent $i: negative n_new_obs=$(a.n_new_obs)"
        @assert length(a.obs_period_marks) == a.periods_alive + 1 "Agent $i: period marks do not include exactly one period-0 boundary plus one boundary per period alive"
        @assert issorted(a.obs_period_marks) "Agent $i: period marks are not cumulative"
        @assert last(a.obs_period_marks) <= a.history_count "Agent $i: period mark exceeds history count"
    end
    @assert 0 <= broker.history_count <= size(broker.history_party1_types, 2) "Broker: history_count=$(broker.history_count) > capacity"
    @assert length(broker.obs_period_marks) == state.period + 1 "Broker period marks do not include period 0 through the current period"
    @assert issorted(broker.obs_period_marks) "Broker period marks are not cumulative"
    @assert last(broker.obs_period_marks) == broker.history_count "Broker's latest period mark must equal history count"

    # ── Broker roster and client-overlay consistency ──
    target_size = roster_target_size(p)
    @assert length(broker.roster) <= target_size "Broker roster size $(length(broker.roster)) exceeds target $target_size"
    for rid in broker.roster
        @assert 1 <= rid <= N "Broker roster contains invalid id $rid"
        @assert has_edge(G, rid, broker.node_id) "Agent $rid on roster but no broker edge"
    end
    for cid in broker.current_clients
        @assert 1 <= cid <= N "Broker current_clients contains invalid id $cid"
        @assert has_edge(G, cid, broker.node_id) "Agent $cid is a current broker client but has no broker edge"
    end
    for i in 1:N
        should_have_broker_edge =
            (i in broker.roster) ||
            (i in broker.current_clients) ||
            has_active_broker_match(agents[i])
        @assert has_edge(G, i, broker.node_id) == should_have_broker_edge "Broker edge mismatch at agent $i"
    end

    # ── Finite satisfaction and NN weights ──
    for i in 1:N
        a = agents[i]
        @assert isfinite(a.satisfaction_self) "NaN/Inf satisfaction_self at agent $i"
        @assert isfinite(a.satisfaction_broker) "NaN/Inf satisfaction_broker at agent $i"
        @assert all(isfinite, a.nn.W1) "Non-finite W1 at agent $i"
        @assert all(isfinite, a.nn.b1) "Non-finite b1 at agent $i"
        @assert all(isfinite, a.nn.w2) "Non-finite w2 at agent $i"
        @assert isfinite(a.nn.b2) "Non-finite b2 at agent $i"
    end
    @assert all(isfinite, broker.nn.W1) "Non-finite W1 at broker"
    @assert all(isfinite, broker.nn.b1) "Non-finite b1 at broker"
    @assert all(isfinite, broker.nn.w2) "Non-finite w2 at broker"
    @assert isfinite(broker.nn.b2) "Non-finite b2 at broker"

    # ── Partner tracking ──
    for i in 1:N
        a = agents[i]
        @assert length(a.partner_count) == N "Agent $i partner_count length $(length(a.partner_count)) != $N"
        @assert all(>=(0), a.partner_count) "Agent $i has negative partner_count"
        @assert all(isfinite, a.partner_sum) "Agent $i has non-finite partner_sum"
    end

    # ── Periods alive ──
    for i in 1:N
        @assert agents[i].periods_alive >= 0 "Agent $i has negative periods_alive"
    end

    return nothing
end
