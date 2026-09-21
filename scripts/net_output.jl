"""
    net_output_accounting(gross_output, self_requests, broker_placements, N, c_s, phi)

Count each completed relationship's output once, subtract search costs on all
self-search requests and fees on successful broker placements, and divide by
the number of principals. Return the aggregate components and per-principal net
output. This accounting does not change channel satisfaction or model behavior.
"""
function net_output_accounting(
    gross_output::Real,
    self_requests::Integer,
    broker_placements::Integer,
    N::Integer,
    c_s::Real,
    phi::Real,
)
    N > 0 || throw(ArgumentError("the number of principals must be positive"))
    self_requests >= 0 || throw(ArgumentError("self-search requests must be nonnegative"))
    broker_placements >= 0 || throw(ArgumentError("broker placements must be nonnegative"))
    isfinite(gross_output) || throw(ArgumentError("match output must be finite"))
    isfinite(c_s) && c_s >= 0 || throw(ArgumentError("invalid self-search cost"))
    isfinite(phi) && phi >= 0 || throw(ArgumentError("invalid broker fee"))
    gross_match_output = Float64(gross_output)
    total_search_cost = Float64(c_s) * self_requests
    total_broker_fees = Float64(phi) * broker_placements
    net_output_per_principal =
        (gross_match_output - total_search_cost - total_broker_fees) / N
    return (;
        gross_match_output, total_search_cost, total_broker_fees, net_output_per_principal
    )
end
