"""
    scripts/monte_carlo.jl

Shared Monte Carlo uncertainty helpers for reporting scripts. Each input value
must represent one independent simulation seed. Intervals are two-sided
Student-t intervals for the seed mean.
"""

using Distributions: TDist, quantile
using Statistics: mean, std

"""Return the finite values in `values` as a `Vector{Float64}`."""
finite_seed_values(values) = filter(isfinite, Float64.(collect(values)))

"""
    monte_carlo_interval(values; level=0.95)

Return the mean, standard error, two-sided Student-t interval, and number of
finite seed values. The interval bounds and standard error are `NaN` when fewer
than two finite seeds are available.
"""
function monte_carlo_interval(values; level::Float64=0.95)
    0.0 < level < 1.0 || throw(ArgumentError("level must lie strictly between 0 and 1"))
    kept = finite_seed_values(values)
    n = length(kept)
    n == 0 && return (; mean=NaN, se=NaN, lower=NaN, upper=NaN, n)
    estimate = mean(kept)
    n == 1 && return (; mean=estimate, se=NaN, lower=NaN, upper=NaN, n)
    se = std(kept) / sqrt(n)
    critical = quantile(TDist(n - 1), 1.0 - (1.0 - level) / 2.0)
    half_width = critical * se
    return (;
        mean=estimate, se, lower=estimate - half_width, upper=estimate + half_width, n
    )
end

"""
    paired_monte_carlo_interval(reference, comparison; level=0.95)

Return a Student-t interval for the mean seed-level difference `comparison -
reference`. Nonfinite pairs are removed jointly, preserving pairing.
"""
function paired_monte_carlo_interval(reference, comparison; level::Float64=0.95)
    length(reference) == length(comparison) ||
        throw(DimensionMismatch("paired vectors must have the same length"))
    differences = Float64[
        Float64(comparison[i]) - Float64(reference[i]) for
        i in eachindex(reference) if isfinite(reference[i]) && isfinite(comparison[i])
    ]
    return merge(monte_carlo_interval(differences; level), (; differences))
end
