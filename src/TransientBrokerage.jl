module TransientBrokerage

using Graphs:
    SimpleGraph,
    watts_strogatz,
    neighbors,
    add_edge!,
    add_vertex!,
    has_edge,
    rem_edge!,
    nv,
    ne,
    vertices
using StableRNGs: StableRNG
using LinearAlgebra: dot, norm, mul!, normalize
using Random: AbstractRNG, shuffle!
using Statistics: var, mean
using MultivariateStats: fit, predict, PCA
using DataFrames: DataFrame
using Distributions: Binomial

include("types.jl")
include("parameters.jl")
include("matching_function.jl")
include("network.jl")
include("learning.jl")
include("measures.jl")
include("search.jl")
include("matching.jl")
include("capture.jl")
include("initialization.jl")
include("entry_exit.jl")
include("step.jl")
include("invariants.jl")
include("diagnostics.jl")
include("simulation.jl")

# Types
export Agent, ActiveMatch, Broker, DirectedOffer, OfferCredit, AcceptedMatch
export NeuralNet, NNGradBuffers
export ModelParams, MatchingEnv, CalibrationConstants, CurveGeometry
export PredictionQuality, PeriodAccumulators, CachedNetworkMeasures, ModelState

# Type utilities
export effective_history_size, has_current_match, partner_mean
export record_agent_history!, update_partner_mean!, record_broker_history!
export reset_accumulators!

# Parameters
export default_params, validate_params, Q_OFFSET, R_BASE_FRAC

# Matching function
export generate_matching_env, match_signal, match_signal!, match_output, match_output!
export regime_gain, regime_gain!, calibrate

# Network
export build_network,
    add_match_edge!, add_broker_edge!, remove_agent_edges!, add_entrant_edges!

# Learning
export init_neural_net, predict_nn!, predict_nn_batch!, nn_loss, train_step!, train_nn!
export compute_adaptive_steps, train_agent_nn!, train_broker_nn!

# Measures
export compute_prediction_quality, compute_betweenness
export compute_burt_constraint, compute_effective_size, update_cached_network_measures!

# Matching
export update_satisfaction!
export outsourcing_decision, broker_reputation, update_broker_reputation!

# Capture
export capture_surplus, counterparty_ask, principal_mode_ready, capture_scaled_mae

# Entry/exit
export exit_agent!, enter_agent!, process_entry_exit!

# Invariants and diagnostics
export verify_invariants, diagnostic_summary

# Simulation
export initialize_model, step_period!, collect_period_metrics, run_simulation

end # module
