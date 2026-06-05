"""
    parameters.jl

Default parameter construction and validation for the Transient Brokerage ABM (v0.2).
"""

"""Constant offset added to match output so calibrated quality is positive."""
const Q_OFFSET = 1.0

"""Reservation-output fraction used to set `r = R_BASE_FRAC * q_cal`."""
const R_BASE_FRAC = 0.60

# Shared search-cost rate, expressed as a share of the calibration surplus
# scale (q_cal - r). Both channels use the same level, but the self-search cost
# remains per demanded relationship position while the broker fee is contingent
# on realized standard placements.
const SEARCH_COST_RATE_BASE = 0.15

# Fixed roster target share: broker maintains this fraction of the population
# on its standing roster.
const ROSTER_TARGET_FRAC = 0.20

"""
    roster_target_size(N::Int) -> Int

Fixed target roster size implied by the standing broker roster share.
"""
roster_target_size(N::Int) = min(N, ceil(Int, ROSTER_TARGET_FRAC * N))

"""Broker hidden width implied by type dimensionality."""
broker_hidden_width(d::Int)::Int = max(1, 8 * d)

"""Agent hidden width implied by type dimensionality."""
agent_hidden_width(d::Int)::Int = max(1, 2 * d)

broker_hidden_width(p::ModelParams)::Int = broker_hidden_width(p.d)
agent_hidden_width(p::ModelParams)::Int = agent_hidden_width(p.d)

"""
    default_params(; seed=42, kwargs...)::ModelParams

Construct a `ModelParams` with baseline defaults, overriding any field via keyword arguments.
"""
function default_params(; seed::Int=42, kwargs...)::ModelParams
    defaults = Dict{Symbol,Any}(
        # Population and types
        :N => 1000,
        :d => 8,
        :s => 8,
        # Matching function
        :rho => 0.50,
        :delta => 0.5,
        :sigma_x => 0.5,
        :sigma_eps => 0.10,
        # Match accounting
        :K => 5,
        :p_demand => 0.50,
        # Network
        :k => 6,
        :p_rewire => 0.1,
        # Economics
        :omega => 0.2,
        :search_cost_rate => SEARCH_COST_RATE_BASE,
        # Neural network
        :eta_lr => 0.03,
        :E_init => 200,
        # Search
        :n_strangers => 10,
        :eta => 0.02,
        :roster_churn => 0.02,
        # Model 1
        :enable_principal => false,
        :capture_min_error_obs => 100,
        :capture_error_threshold => 0.65,
        # Simulation
        :network_measure_interval => 20,
        :T => 200,
        :T_burn => 30,
        :seed => seed,
    )
    for (kw, v) in kwargs
        haskey(defaults, kw) || error("Unknown parameter: $kw")
        defaults[kw] = v
    end
    p = ModelParams(
        defaults[:N],
        defaults[:d],
        defaults[:s],
        defaults[:rho],
        defaults[:delta],
        defaults[:sigma_x],
        defaults[:sigma_eps],
        defaults[:K],
        defaults[:p_demand],
        defaults[:k],
        defaults[:p_rewire],
        defaults[:omega],
        defaults[:search_cost_rate],
        defaults[:eta_lr],
        defaults[:E_init],
        defaults[:n_strangers],
        defaults[:eta],
        defaults[:roster_churn],
        defaults[:enable_principal],
        defaults[:capture_min_error_obs],
        defaults[:capture_error_threshold],
        defaults[:network_measure_interval],
        defaults[:T],
        defaults[:T_burn],
        defaults[:seed],
    )
    validate_params(p)
    return p
end

"""
    validate_params(p::ModelParams)

Assert that all parameter values satisfy model constraints. Throws on violation.
"""
function validate_params(p::ModelParams)
    # Population and types
    @assert p.N >= 10 "N must be >= 10, got $(p.N)"
    @assert p.d >= 2 "d must be >= 2, got $(p.d)"
    @assert 1 <= p.s <= p.d "s must be in [1, d], got s=$(p.s), d=$(p.d)"

    # Matching function
    @assert 0.0 <= p.rho <= 1.0 "rho must be in [0, 1], got $(p.rho)"
    @assert 0.0 <= p.delta <= 1.0 "delta must be in [0, 1], got $(p.delta)"
    @assert p.sigma_x > 0.0 "sigma_x must be > 0, got $(p.sigma_x)"
    @assert p.sigma_eps >= 0.0 "sigma_eps must be >= 0, got $(p.sigma_eps)"

    # Match accounting
    @assert p.K >= 1 "K must be >= 1, got $(p.K)"
    @assert 0.0 < p.p_demand <= 1.0 "p_demand must be in (0, 1], got $(p.p_demand)"

    # Network
    @assert p.k >= 2 "k must be >= 2, got $(p.k)"
    @assert iseven(p.k) "k must be even for Watts-Strogatz, got $(p.k)"
    @assert p.k < p.N "k must be < N for Watts-Strogatz, got k=$(p.k), N=$(p.N)"
    @assert 0.0 <= p.p_rewire <= 1.0 "p_rewire must be in [0, 1], got $(p.p_rewire)"

    # Economics
    @assert 0.0 < p.omega < 1.0 "omega must be in (0, 1), got $(p.omega)"
    @assert 0.0 <= p.search_cost_rate <= 1.0 "search_cost_rate must be in [0, 1], got $(p.search_cost_rate)"

    # Neural network
    @assert p.eta_lr > 0.0 "eta_lr must be > 0, got $(p.eta_lr)"
    @assert p.E_init >= 1 "E_init must be >= 1, got $(p.E_init)"
    @assert agent_hidden_width(p) >= 1 "agent hidden width must be >= 1"
    @assert broker_hidden_width(p) >= 1 "broker hidden width must be >= 1"

    # Search
    @assert p.n_strangers >= 0 "n_strangers must be >= 0, got $(p.n_strangers)"
    @assert 0.0 <= p.eta < 1.0 "eta must be in [0, 1), got $(p.eta)"
    @assert 0.0 <= p.roster_churn <= 1.0 "roster_churn must be in [0, 1], got $(p.roster_churn)"

    # Model 1
    @assert p.capture_min_error_obs >= 0 "capture_min_error_obs must be >= 0"
    @assert p.capture_error_threshold >= 0.0 "capture_error_threshold must be >= 0"

    # Simulation
    @assert p.network_measure_interval >= 1 "network_measure_interval must be >= 1"
    @assert p.T >= 1 "T must be >= 1, got $(p.T)"
    @assert p.T_burn >= 0 "T_burn must be >= 0, got $(p.T_burn)"
    @assert p.T_burn < p.T "T_burn must be < T, got T_burn=$(p.T_burn), T=$(p.T)"

    return nothing
end
