"""
    capture.jl

Client-origin resource capture. When enabled, the broker can acquire a current
broker client's whole active lot before standard offers are created, then place
that captured origin through principal offers.
"""

"""
    counterparty_ask(agent, q_cal) -> Float64

Return an agent's average realized match quality from history, or `q_cal` if it
has no history. Under client-origin capture this is the origin client's ask.
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

"""Scaled live broker error used by the principal-mode readiness gate."""
function capture_scaled_mae(broker::Broker, cal::CalibrationConstants)::Float64
    broker.capture_confidence_ready || return NaN
    surplus_scale = cal.q_cal - cal.r
    surplus_scale <= 0.0 && return Inf
    return broker.capture_confidence_mae / surplus_scale
end

"""True when the broker has enough live accuracy to consider principal mode."""
function principal_mode_ready(
    broker::Broker, params::ModelParams, cal::CalibrationConstants
)::Bool
    params.enable_principal || return false
    broker.capture_confidence_ready || return false
    broker.capture_error_count >= params.capture_min_error_obs || return false
    return capture_scaled_mae(broker, cal) <= params.capture_error_threshold
end

"""
    update_capture_confidence_mae!(broker, abs_error_sum, n_errors, omega) -> Nothing

Update the broker's live broker-controlled exposure error scale. This is a no-op
when there are no broker prediction errors in the period.
"""
function update_capture_confidence_mae!(
    broker::Broker, abs_error_sum::Float64, n_errors::Int, omega::Float64
)
    n_errors <= 0 && return nothing
    broker.capture_error_count += n_errors
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

function reset_principal_payments!(ws::SimWorkspace, N::Int)
    principal_payment = ws.ledger.principal_payment
    if length(principal_payment) < N
        resize!(principal_payment, N)
    end
    fill!(principal_payment, 0.0)
    return nothing
end

function reset_capture_buffers!(ws::SimWorkspace, N::Int)
    capture = ws.capture
    if length(capture.captured_origin_mask) < N
        old = length(capture.captured_origin_mask)
        resize!(capture.captured_origin_mask, N)
        @inbounds for i in (old + 1):N
            capture.captured_origin_mask[i] = false
        end
    end
    @inbounds for i in capture.captured_origin_touched
        capture.captured_origin_mask[i] = false
    end
    empty!(capture.captured_origin_touched)

    return nothing
end

function ensure_capture_score_buffers!(ws::SimWorkspace, N::Int)
    capture = ws.capture
    if length(capture.capture_candidate_counts) < N
        resize!(capture.capture_candidate_counts, N)
        resize!(capture.capture_candidate_sums, N)
        resize!(capture.capture_asks, N)
    end
    return nothing
end

function initialize_capture_lot_candidates!(
    ws::SimWorkspace,
    pair_scores::Vector{Tuple{Float64,Int,Int}},
    broker_demanders::Vector{Int},
    remaining::Vector{Int},
    agents::Vector{Agent},
    cal::CalibrationConstants,
)::Vector{Tuple{Float64,Int}}
    N = length(agents)
    ensure_capture_score_buffers!(ws, N)
    capture = ws.capture
    lots = capture.capture_candidate_lots
    empty!(lots)

    counts = capture.capture_candidate_counts
    sums = capture.capture_candidate_sums
    asks = capture.capture_asks
    @inbounds for did in broker_demanders
        counts[did] = 0
        sums[did] = 0.0
        asks[did] = counterparty_ask(agents[did], cal.q_cal)
    end

    broker_pairs = ws.broker_pairs
    demander_mask = broker_pairs.broker_demander_mask
    access_mask = broker_pairs.broker_access_mask
    @inbounds for (neg_score, i, j) in pair_scores
        score = -neg_score
        score <= cal.r && break
        if demander_mask[i] && access_mask[j] && asks[i] >= cal.r
            if counts[i] < remaining[i]
                counts[i] += 1
                sums[i] += score
            end
        end
        if demander_mask[j] && access_mask[i] && asks[j] >= cal.r
            if counts[j] < remaining[j]
                counts[j] += 1
                sums[j] += score
            end
        end
    end

    @inbounds for did in broker_demanders
        d_i = remaining[did]
        d_i > 0 || continue
        counts[did] == d_i || continue
        expected_gain = sums[did] - d_i * asks[did] - d_i * cal.phi
        expected_gain > 0.0 || continue
        push!(lots, (-expected_gain, did))
    end
    sort!(lots; alg=QuickSort)
    return lots
end

function replan_capture_lot!(
    recipients::Vector{Int},
    qhats::Vector{Float64},
    pair_scores::Vector{Tuple{Float64,Int,Int}},
    origin_id::Int,
    demand_count::Int,
    ws::SimWorkspace,
    cal::CalibrationConstants,
)::Bool
    empty!(recipients)
    empty!(qhats)
    access_mask = ws.broker_pairs.broker_access_mask
    captured_mask = ws.capture.captured_origin_mask

    @inbounds for (neg_score, i, j) in pair_scores
        score = -neg_score
        score <= cal.r && break
        other = if i == origin_id
            j
        elseif j == origin_id
            i
        else
            continue
        end
        access_mask[other] || continue
        captured_mask[other] && continue
        has_current_match(ws, origin_id, other) && continue
        push!(recipients, other)
        push!(qhats, score)
        length(recipients) == demand_count && return true
    end
    return false
end

function push_principal_relationship!(
    accepted::Vector{AcceptedMatch},
    origin_id::Int,
    recipient_id::Int,
    qhat::Float64,
    ask_i::Float64,
    agents::Vector{Agent},
    broker::Broker,
    env::MatchingEnv,
    rng::AbstractRNG;
    Ax_buf::Vector{Float64},
    Bx_buf::Vector{Float64},
    ws::SimWorkspace,
)
    q_realized = match_output!(
        Ax_buf, Bx_buf, agents[origin_id].type, agents[recipient_id].type, env, rng
    )
    record_broker_history!(
        broker, agents[origin_id].type, agents[recipient_id].type, q_realized
    )
    push!(agents[origin_id].active_matches, ActiveMatch(recipient_id, true, :broker))
    push!(agents[recipient_id].active_matches, ActiveMatch(origin_id, true, :broker))
    agents[origin_id].n_matches_any += 1
    agents[recipient_id].n_matches_any += 1

    push!(
        accepted,
        AcceptedMatch(
            origin_id,
            recipient_id,
            :broker,
            true,
            q_realized,
            qhat,
            ask_i,
            qhat,
            nothing,
            nothing,
        ),
    )
    return q_realized
end

function execute_client_origin_capture!(
    accepted::Vector{AcceptedMatch},
    ws::SimWorkspace,
    broker_demanders::Vector{Int},
    broker_access::Vector{Int},
    remaining::Vector{Int},
    agents::Vector{Agent},
    broker::Broker,
    env::MatchingEnv,
    G::SimpleGraph,
    params::ModelParams,
    cal::CalibrationConstants,
    rng::AbstractRNG,
    accum::PeriodAccumulators;
    Ax_buf::Vector{Float64},
    Bx_buf::Vector{Float64},
)::Int
    ready = principal_mode_ready(broker, params, cal)
    accum.capture_ready = ready
    accum.capture_scaled_mae = capture_scaled_mae(broker, cal)
    ready || return 0
    isempty(broker_demanders) && return 0
    isempty(broker_access) && return 0

    pair_scores = prepare_broker_pair_scores!(
        ws, broker, broker_demanders, broker_access, agents, params
    )
    isempty(pair_scores) && return 0

    lots = initialize_capture_lot_candidates!(
        ws, pair_scores, broker_demanders, remaining, agents, cal
    )
    isempty(lots) && return 0

    capture = ws.capture
    recipients = capture.capture_plan_recipients
    qhats = capture.capture_plan_qhats
    captured = 0
    captured_mask = capture.captured_origin_mask

    @inbounds for (_, origin_id) in lots
        captured_mask[origin_id] && continue
        d_i = remaining[origin_id]
        d_i > 0 || continue
        ask_i = capture.capture_asks[origin_id]
        ask_i >= cal.r || continue
        replan_capture_lot!(recipients, qhats, pair_scores, origin_id, d_i, ws, cal) ||
            continue

        captured_mask[origin_id] = true
        push!(capture.captured_origin_touched, origin_id)
        remaining[origin_id] = 0
        ws.ledger.principal_payment[origin_id] += d_i * ask_i
        accum.captured_origin_count += 1
        captured += 1

        for idx in eachindex(recipients)
            recipient_id = recipients[idx]
            qhat = qhats[idx]
            mark_current_match!(ws, origin_id, recipient_id)
            accum.captured_position_count += 1

            if receiver_offer_value(recipient_id, origin_id, agents, G) > cal.r
                q_realized = push_principal_relationship!(
                    accepted,
                    origin_id,
                    recipient_id,
                    qhat,
                    ask_i,
                    agents,
                    broker,
                    env,
                    rng;
                    Ax_buf=Ax_buf,
                    Bx_buf=Bx_buf,
                    ws=ws,
                )
                push!(accum.capture_realized, q_realized)
                push!(accum.capture_ask, ask_i)
                push!(accum.capture_qhat, qhat)
                accum.broker_error_abs_sum += abs(q_realized - qhat)
                accum.broker_error_count += 1
            else
                push!(accum.capture_realized, 0.0)
                push!(accum.capture_ask, ask_i)
                push!(accum.capture_qhat, qhat)
                accum.principal_rejected += 1
                accum.broker_error_abs_sum += abs(qhat)
                accum.broker_error_count += 1
            end
        end
    end
    return captured
end

"""
    capture_surplus(q_realized, ask_i) -> Float64

Capture surplus for a principal position, `q_realized - ask_i`.
"""
function capture_surplus(q_realized::Float64, ask_i::Float64)::Float64
    return q_realized - ask_i
end
