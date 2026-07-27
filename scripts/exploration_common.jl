"""
    exploration_common.jl

Shared helpers for exploration scripts so they do not drift by copy-paste.
"""

using DataFrames: DataFrame
using MultivariateStats: fit, predict, PCA
using BrokerageABM:
    Q_OFFSET, default_params, match_signal, regime_gain, run_simulation

"""Run `n_seeds` simulations and return one metrics DataFrame per seed."""
function run_ensemble(;
    base_kwargs, T::Int, N::Int, n_seeds::Int, return_final_agent_degrees::Bool=false
)
    mdfs = Vector{DataFrame}(undef, n_seeds)
    final_agent_degrees =
        return_final_agent_degrees ? Vector{Vector{Int}}(undef, n_seeds) : nothing
    for s in 1:n_seeds
        p = default_params(; N=N, T=T, seed=s, base_kwargs...)
        state, mdf = run_simulation(p)
        mdf[!, :seed] .= s
        mdfs[s] = mdf
        if return_final_agent_degrees
            final_agent_degrees[s] = copy(state.accum.agent_degrees)
        end
    end
    return_final_agent_degrees &&
        return (mdfs=mdfs, final_agent_degrees=final_agent_degrees)
    return mdfs
end

"""
    build_ordered_output_matrix(types, env; include_regime=false)

Build the noiseless output matrix after sorting agents by the first principal
component of their types. Optionally also return the regime-gain matrix.
"""
function build_ordered_output_matrix(types, env; include_regime::Bool=false)
    N = length(types)
    type_matrix = reduce(hcat, types)
    pca = fit(PCA, type_matrix; maxoutdim=1)
    pc1 = vec(predict(pca, type_matrix))
    order = sortperm(pc1)
    sorted = types[order]

    F = Matrix{Float64}(undef, N, N)
    G_regime = include_regime ? Matrix{Float64}(undef, N, N) : nothing
    @inbounds for j in 1:N
        xj = sorted[j]
        for i in 1:N
            xi = sorted[i]
            F[i, j] = Q_OFFSET + match_signal(xi, xj, env)
            if include_regime
                G_regime[i, j] = regime_gain(xi, xj, env)
            end
        end
    end

    return (F=F, G_regime=G_regime, pc1_sorted=pc1[order], sorted_types=sorted)
end
