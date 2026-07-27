module BrokerageABM

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
include("initialization.jl")
include("entry_exit.jl")
include("step.jl")
include("invariants.jl")
include("diagnostics.jl")
include("simulation.jl")

# Public API. Lower-level model components remain accessible as
# BrokerageABM.<name> for tests, scripts, and advanced inspection, but are
# intentionally not imported by `using BrokerageABM`.
export ModelParams, ModelState, PredictionQuality

export default_params, validate_params

export initialize_model, step_period!, collect_period_metrics, run_simulation

export verify_invariants, diagnostic_summary

export compute_prediction_quality, compute_betweenness
export compute_burt_constraint, compute_effective_size

end # module
