"""
    entry_exit.jl

Agent entry and exit. Exiting agents are replaced by entrants with fresh types,
empty histories, and floor(k/2) edges to type-similar neighbors.
"""

using Random: AbstractRNG
using LinearAlgebra: norm
using Graphs: neighbors

"""
    exit_agent!(state, agent_id)

Remove agent from the simulation: clear all edges in G, terminate active matches,
remove from the broker's standing roster and
current client overlay, and clear any references to the exiting slot held
elsewhere in state (other agents' per-partner tracking). The agent's node index
is reused for the entrant.
"""
function exit_agent!(state::ModelState, agent_id::Int)
    agent = state.agents[agent_id]

    # Terminate all active matches: remove from partner's active list
    for am in agent.active_matches
        partner = state.agents[am.partner_id]
        idx = findfirst(m -> m.partner_id == agent_id, partner.active_matches)
        if idx !== nothing
            deleteat!(partner.active_matches, idx)
        end
    end
    empty!(agent.active_matches)

    # Remove all edges from G
    remove_agent_edges!(state.G, agent_id)

    # Remove from broker standing and current-period access sets
    delete!(state.broker.roster, agent_id)
    delete!(state.broker.current_clients, agent_id)

    # Clear other agents' per-partner tracking keyed on this slot.
    # Without this, a fresh entrant reusing the slot would inherit the prior
    # occupant's match statistics as soon as any edge to it re-forms.
    N = length(state.agents)
    @inbounds for j in 1:N
        j == agent_id && continue
        state.agents[j].partner_sum[agent_id] = 0.0
        state.agents[j].partner_count[agent_id] = 0
    end

    return nothing
end

"""
    enter_agent!(state, agent_id, rng)

Replace an exited agent with a fresh entrant at the same node index.
New type from curve + noise, empty history, `floor(k/2)` edges to
type-similar neighbors, self-satisfaction initialized from new neighbors'
mean self-satisfaction, and broker-satisfaction initialized from the current
broker reputation.
"""
function enter_agent!(state::ModelState, agent_id::Int, rng::AbstractRNG)
    p = state.params
    d = p.d
    geo = state.curve_geo

    # Generate fresh type on curve + noise
    cp = curve_point(rand(rng), geo)
    sigma_per_dim = p.sigma_x / sqrt(d)
    new_type = cp .+ sigma_per_dim .* randn(rng, d)
    new_type_norm = norm(new_type)
    if new_type_norm > 1e-12
        new_type ./= new_type_norm
    end

    agent = state.agents[agent_id]

    # Reset all fields
    agent.type .= new_type
    empty!(agent.active_matches)
    agent.history_count = 0
    agent.n_new_obs = 0
    empty!(agent.obs_period_marks)
    # The entrant's empty initialization history is its period-0 boundary.
    push!(agent.obs_period_marks, 0)

    # Re-initialize neural network
    agent.nn = init_neural_net(d, agent_hidden_width(p), rng)
    agent.nn_grad = NNGradBuffers(agent.nn)
    fill!(agent.predict_buf, 0.0)

    # Reset partner tracking
    fill!(agent.partner_sum, 0.0)
    fill!(agent.partner_count, 0)

    # Reset satisfaction
    agent.tried_broker = false
    agent.periods_alive = 0

    # Reset cumulative match counters for the new entrant.
    agent.n_matches_any = 0

    # Add edges to type-similar neighbors
    n_edges = p.k ÷ 2
    add_entrant_edges!(state.G, agent_id, new_type, state.agents, rng; n_edges=n_edges)

    # Self-satisfaction: mean of new neighbors' self-satisfaction (word-of-mouth)
    nbrs = neighbors(state.G, agent_id)
    n_nbrs = 0
    sat_sum = 0.0
    for nbr in nbrs
        nbr == state.broker.node_id && continue
        (nbr < 1 || nbr > p.N) && continue
        sat_sum += state.agents[nbr].satisfaction_self
        n_nbrs += 1
    end
    agent.satisfaction_self = n_nbrs > 0 ? sat_sum / n_nbrs : 0.0
    # Broker satisfaction: current broker reputation (market prior)
    agent.satisfaction_broker = broker_reputation(state.broker)

    return nothing
end

"""
    process_entry_exit!(state, rng)

Process agent turnover: each agent exits with probability η, immediately replaced.
"""
function process_entry_exit!(state::ModelState, rng::AbstractRNG)
    eta = state.params.eta
    eta <= 0.0 && return nothing

    for i in 1:state.params.N
        if rand(rng) < eta
            exit_agent!(state, i)
            enter_agent!(state, i, rng)
        end
    end

    return nothing
end
