"""
    step.jl

Main simulation loop: one period of the model, matching
`simulation_pseudocode.tex` (`PeriodUpdate`).

Flow: reset current matches and roster overlay; draw demand and channel choices;
train predictors; run the shared offer market; update satisfaction, reputation,
and diagnostics; process entry/exit; refresh cached network measures when due.
"""

using Random: shuffle!
using Distributions: Binomial
using Graphs: neighbors, has_edge
using LinearAlgebra: BLAS
using Base.Threads: @threads

# ─────────────────────────────────────────────────────────────────────────────

agent_retrains_this_period(agent_id::Int, period::Int)::Bool =
    isodd(agent_id) == isodd(period)

function refresh_broker_roster!(state::ModelState)
    p = state.params
    broker = state.broker
    rng = state.rng
    N = p.N
    target_size = roster_target_size(p)

    if p.roster_churn > 0.0 && !isempty(broker.roster)
        for rid in collect(broker.roster)
            rand(rng) < p.roster_churn && delete!(broker.roster, rid)
        end
    end

    n_missing = target_size - length(broker.roster)
    if n_missing > 0
        candidates = Int[]
        sizehint!(candidates, max(N - length(broker.roster), 0))
        for i in 1:N
            (i in broker.roster) && continue
            push!(candidates, i)
        end
        shuffle!(rng, candidates)
        for idx in 1:min(n_missing, length(candidates))
            push!(broker.roster, candidates[idx])
        end
    end

    sync_broker_edges!(state.G, state.agents, broker)
    return nothing
end

function prefix_rmse(
    predicted::AbstractVector{<:Real}, realized::AbstractVector{<:Real}, n::Int
)::Float64
    sq_err_sum = 0.0
    @inbounds for idx in 1:n
        err = predicted[idx] - realized[idx]
        sq_err_sum += err * err
    end
    return sqrt(sq_err_sum / n)
end

@inline ensure_length!(v::Vector, n::Int) = (length(v) == n || resize!(v, n); v)

function diagnostics_rng(seed::Int, period::Int)::StableRNG
    seed_bits = UInt64(reinterpret(UInt, seed))
    mixed = seed_bits ⊻ (UInt64(period) * 0x9e3779b97f4a7c15) ⊻ 0xbf58476d1ce4e5b9
    diag_seed = Int(mod(mixed, UInt64(typemax(Int32))) + 1)
    return StableRNG(diag_seed)
end

function holdout_quality_components!(
    predicted::Vector{Float64},
    realized::Vector{Float64},
    n::Int,
    sigma_eps::Float64,
    pred_order::Vector{Int},
    pred_ranks::Vector{Float64},
    true_ranks::Vector{Float64},
)::Tuple{Bool,Float64,Float64,Float64,Float64}
    pq = compute_prediction_quality_with_true_ranks!(
        predicted,
        realized,
        n;
        sigma_eps=sigma_eps,
        pred_order=pred_order,
        pred_ranks=pred_ranks,
        true_ranks=true_ranks,
    )
    isnan(pq.r_squared) && return (false, 0.0, 0.0, 0.0, 0.0)
    return (true, pq.r_squared, pq.bias, pq.rank_corr, prefix_rmse(predicted, realized, n))
end

@inline mean_or_nan(total::Float64, count::Int)::Float64 = count > 0 ? total / count : NaN

function update_holdout_metrics!(state::ModelState)
    p = state.params
    N = p.N
    d = p.d
    d_broker = broker_pair_feature_dim(d)
    agents = state.agents
    broker = state.broker
    env = state.env
    ws = state.workspace

    n_sample_agents = min(100, N)
    n_partners = min(40, N - 1)
    if n_partners < 5
        state.accum.agent_holdout_r2 = NaN
        state.accum.agent_holdout_bias = NaN
        state.accum.agent_holdout_rank = NaN
        state.accum.agent_holdout_rmse = NaN
        state.accum.broker_holdout_r2 = NaN
        state.accum.broker_holdout_bias = NaN
        state.accum.broker_holdout_rank = NaN
        state.accum.broker_holdout_rmse = NaN
        return nothing
    end

    diag_rng = diagnostics_rng(p.seed, state.period)
    match_output = ws.match_output
    holdout = ws.holdout
    if length(match_output.Ax_buf) != d ||
        length(match_output.Bx_buf) != d ||
        length(holdout.z_buf) != d_broker
        match_output.Ax_buf = Vector{Float64}(undef, d)
        match_output.Bx_buf = Vector{Float64}(undef, d)
        holdout.z_buf = Vector{Float64}(undef, d_broker)
    end
    for v in (
        holdout.agent_preds,
        holdout.agent_trues,
        holdout.broker_preds,
        holdout.pred_order,
        holdout.true_order,
        holdout.pred_ranks,
        holdout.true_ranks,
    )
        ensure_length!(v, n_partners)
    end

    eligible_agents = holdout.agent_ids
    empty!(eligible_agents)
    @inbounds for i in 1:N
        agents[i].history_count > 0 && push!(eligible_agents, i)
    end
    isempty(eligible_agents) && return nothing

    shuffle!(diag_rng, eligible_agents)
    n_sample_agents = min(n_sample_agents, length(eligible_agents))

    partner_ids = holdout.partner_ids
    ensure_length!(partner_ids, N - 1)

    Ax_buf = match_output.Ax_buf
    Bx_buf = match_output.Bx_buf
    z_buf = holdout.z_buf
    agent_preds = holdout.agent_preds
    agent_trues = holdout.agent_trues
    broker_preds = holdout.broker_preds
    pred_order = holdout.pred_order
    true_order = holdout.true_order
    pred_ranks = holdout.pred_ranks
    true_ranks = holdout.true_ranks

    agent_r2_sum = 0.0
    agent_bias_sum = 0.0
    agent_rank_sum = 0.0
    agent_rmse_sum = 0.0
    broker_r2_sum = 0.0
    broker_bias_sum = 0.0
    broker_rank_sum = 0.0
    broker_rmse_sum = 0.0
    n_agents_evaluated = 0
    n_broker_evaluated = 0
    se = env.sigma_eps

    @inbounds for sample_idx in 1:n_sample_agents
        i = eligible_agents[sample_idx]
        pos = 1
        for j in 1:N
            j == i && continue
            partner_ids[pos] = j
            pos += 1
        end
        shuffle!(diag_rng, partner_ids)

        for partner_idx in 1:n_partners
            j = partner_ids[partner_idx]
            q_true =
                Q_OFFSET +
                match_signal!(Ax_buf, Bx_buf, agents[i].type, agents[j].type, env)
            agent_preds[partner_idx] = predict_nn!(
                agents[i].nn, agents[i].predict_buf, agents[j].type
            )
            agent_trues[partner_idx] = q_true
            fill_broker_pair_features!(z_buf, agents[i].type, agents[j].type)
            broker_preds[partner_idx] = predict_nn!(broker.nn, broker.predict_buf, z_buf)
        end

        prepare_true_ranks!(agent_trues, n_partners, true_order, true_ranks)

        valid, r2, bias, rank, rmse = holdout_quality_components!(
            agent_preds, agent_trues, n_partners, se, pred_order, pred_ranks, true_ranks
        )
        if valid
            agent_r2_sum += r2
            agent_bias_sum += bias
            agent_rank_sum += rank
            agent_rmse_sum += rmse
            n_agents_evaluated += 1
        end

        valid, r2, bias, rank, rmse = holdout_quality_components!(
            broker_preds, agent_trues, n_partners, se, pred_order, pred_ranks, true_ranks
        )
        if valid
            broker_r2_sum += r2
            broker_bias_sum += bias
            broker_rank_sum += rank
            broker_rmse_sum += rmse
            n_broker_evaluated += 1
        end
    end

    state.accum.agent_holdout_r2 = mean_or_nan(agent_r2_sum, n_agents_evaluated)
    state.accum.agent_holdout_bias = mean_or_nan(agent_bias_sum, n_agents_evaluated)
    state.accum.agent_holdout_rank = mean_or_nan(agent_rank_sum, n_agents_evaluated)
    state.accum.agent_holdout_rmse = mean_or_nan(agent_rmse_sum, n_agents_evaluated)
    state.accum.broker_holdout_r2 = mean_or_nan(broker_r2_sum, n_broker_evaluated)
    state.accum.broker_holdout_bias = mean_or_nan(broker_bias_sum, n_broker_evaluated)
    state.accum.broker_holdout_rank = mean_or_nan(broker_rank_sum, n_broker_evaluated)
    state.accum.broker_holdout_rmse = mean_or_nan(broker_rmse_sum, n_broker_evaluated)

    return nothing
end

"""
    step_period!(state) -> NamedTuple

Execute one complete period of the simulation and return the pre-turnover period
metrics. Entry/exit turnover is processed after the returned metrics are
recorded.
"""
function step_period!(state::ModelState)
    p = state.params

    state.period += 1
    rng = state.rng
    N = p.N
    agents = state.agents
    broker = state.broker
    G = state.G
    env = state.env
    cal = state.cal
    ws = state.workspace
    accum = state.accum

    reset_accumulators!(accum)

    # ══════════════════════════════════════════════════════════════════════
    # Step 0: Current-period match reset
    # ══════════════════════════════════════════════════════════════════════
    for agent in agents
        empty!(agent.active_matches)
    end

    # Clear the current-client overlay from the prior period, then refresh the
    # standing roster after prior-period turnover and before current-period
    # demand realization.
    empty!(broker.current_clients)
    refresh_broker_roster!(state)

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Demand generation and outsourcing decisions
    # ══════════════════════════════════════════════════════════════════════
    ledger = ws.ledger
    demand_agent_ids = ledger.demand_agent_ids
    empty!(demand_agent_ids)
    demand_channels = ledger.demand_channels
    empty!(demand_channels)
    demand_counts = ledger.demand_counts
    empty!(demand_counts)
    broker_clients = ledger.broker_clients_ws
    empty!(broker_clients)

    broker_rep = broker_reputation(broker)

    for i in 1:N
        agents[i].periods_alive += 1
        d_i = rand(rng, Binomial(p.K, p.p_demand))
        d_i <= 0 && continue

        channel = outsourcing_decision(agents[i], broker_rep, rng)

        push!(demand_agent_ids, i)
        push!(demand_channels, channel)
        push!(demand_counts, d_i)
        accum.n_demanders += 1
        accum.total_demand += d_i

        if channel == :broker
            push!(broker_clients, i)
            push!(broker.current_clients, i)
            accum.n_outsourced += 1
            accum.outsourced_slots += d_i
        end
    end
    sync_broker_edges!(G, agents, broker)

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Candidate evaluation
    # ══════════════════════════════════════════════════════════════════════

    # 2.1: Train neural networks (adaptive steps).
    # Agents retrain on an alternating parity schedule so each agent updates
    # every other period while still accumulating all new observations.
    # Agent training copies the active window into contiguous scratch before
    # gradient steps, avoiding SubArray BLAS overhead.
    prev_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @threads for i in 1:N
        a = agents[i]
        a.history_count > 0 &&
            a.n_new_obs > 0 &&
            agent_retrains_this_period(i, state.period) &&
            train_agent_nn!(a, p)
    end
    BLAS.set_num_threads(prev_blas)
    if broker.history_count > 0 && broker.n_new_obs > 0
        train_broker_nn!(broker, p)
    end

    # 2.2: Shared active-demand offer market
    accepted = run_offer_market!(
        demand_agent_ids,
        demand_channels,
        demand_counts,
        agents,
        broker,
        env,
        G,
        p,
        cal,
        rng;
        ws=ws,
        accepted_matches=ledger.accepted_matches,
    )
    sync_broker_edges!(G, agents, broker)

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Learning and state updates
    # ══════════════════════════════════════════════════════════════════════

    # 3.1: Histories already recorded during run_offer_market!

    # 3.2: Satisfaction update
    update_satisfaction!(
        agents,
        accepted,
        demand_agent_ids,
        demand_channels,
        demand_counts,
        cal,
        p;
        demander_sum=ledger.demander_q_sum,
        broker_match_count=ledger.broker_match_count,
    )

    # 3.3: Broker reputation
    update_broker_reputation!(broker, agents, broker_clients)

    # Record accumulators
    for m in accepted
        # Selected-sample prediction quality is keyed to successful directed
        # offers; relationship counts remain bilateral.
        for offer in (m.offer1, m.offer2)
            isnothing(offer) && continue
            if offer.channel == :self
                push!(accum.agent_predicted, offer.predicted_value)
                push!(accum.agent_realized, m.q_realized)
            elseif offer.channel == :broker
                push!(accum.broker_predicted, offer.predicted_value)
                push!(accum.broker_realized, m.q_realized)
                if offer.was_connected
                    accum.assessment_count += 1
                else
                    accum.access_count += 1
                end
            end
        end

        if m.channel == :self
            accum.n_self_matches += 1
            push!(accum.q_self, m.q_realized)
        else
            accum.n_broker_matches += 1
            push!(accum.q_broker, m.q_realized)
        end
    end

    counterparty_counts = ws.holdout.pred_order
    length(counterparty_counts) == N || resize!(counterparty_counts, N)
    @inbounds for i in 1:N
        counterparty_counts[i] = length(agents[i].active_matches)
    end
    sort!(counterparty_counts)
    mid = N ÷ 2
    accum.median_counterparties = if isodd(N)
        Float64(counterparty_counts[mid + 1])
    else
        (counterparty_counts[mid] + counterparty_counts[mid + 1]) / 2
    end
    accum.max_counterparties = counterparty_counts[end]

    update_holdout_metrics!(state)

    accum.roster_size = length(broker.roster)
    accum.broker_access_size = broker_access_size(broker)

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Recording and measurement
    # ══════════════════════════════════════════════════════════════════════
    if state.period % p.network_measure_interval == 0
        update_cached_network_measures!(state)
    end
    record_agent_degree_summary!(state)
    metrics = collect_period_metrics(state)

    # Period mark for the period-based training window: every learner, every
    # period (independent of training parity), before entry/exit. Entrants reset
    # their marks in enter_agent!.
    @inbounds for i in 1:N
        push!(agents[i].obs_period_marks, agents[i].history_count)
    end
    push!(broker.obs_period_marks, broker.history_count)

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: Entry/exit
    # ══════════════════════════════════════════════════════════════════════
    process_entry_exit!(state, rng)
    sync_broker_edges!(G, agents, broker)

    return metrics
end
