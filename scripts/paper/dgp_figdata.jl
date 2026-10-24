"""Generate retained DGP diagnostics using the declared diagnostic design.
Uses model initialization, without sweep input or simulation periods."""

module DGPFigureData

using BrokerageABM
using BrokerageABM:
    MatchingEnv, Q_OFFSET, curve_point, generate_matching_dgp, match_signal_coefficients
using JLD2: jldsave
using LinearAlgebra: Symmetric, eigvals, svd
using StableRNGs: StableRNG
using Statistics: mean

isdefined(@__MODULE__, :AnalysisSettings) || include(joinpath(@__DIR__, "..", "analysis_settings.jl"))

include(joinpath(@__DIR__, "..", "sweep", "sweep_config.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUTFILE = joinpath(REPO, "output", "supplement", "dgp_figure_data.jld2")
const DGP_N = SWEEP_BASELINE.N
const DGP_SEEDS = collect(1:50)
const HEATMAP_N = 100
const HEATMAP_SEED = 1
const TYPE_CURVE_POINTS = 500
const SPECTRUM_RHOS = copy(RHO_OAT)
const BASELINE_DELTA = Float64(SWEEP_BASELINE.delta)
const HEATMAP_CONDITIONS = [
    (; rho=0.0, delta=0.0),
    (; rho=0.5, delta=0.0),
    (; rho=1.0, delta=0.0),
    (; rho=0.0, delta=1.0),
    (; rho=0.5, delta=1.0),
]

"""Return the scientifically distinct conditions in the current rho-by-delta grid."""
function effective_dgp_conditions(
    rhos=RHO_OAT, deltas=DELTA_VALS; baseline_delta::Float64=BASELINE_DELTA
)
    conditions = NamedTuple{(:rho, :delta),Tuple{Float64,Float64}}[]
    for delta in Float64.(deltas), rho in Float64.(rhos)
        rho == 1.0 && delta != baseline_delta && continue
        push!(conditions, (; rho, delta))
    end
    return conditions
end

"""Stable string identifier for one effective DGP condition."""
dgp_condition_id(rho::Real, delta::Real) = "rho=$(Float64(rho))|delta=$(Float64(delta))"

"""
    dgp_components(agent_types, env) -> NamedTuple

Return individual-quality scores and pairwise quality, complementarity, and
regime-sign matrices.
"""
function dgp_components(agent_types::Vector{Vector{Float64}}, env::MatchingEnv)
    X = reduce(hcat, agent_types)
    quality_scores = vec(X' * env.c)
    quality_surface = 0.5 .* (quality_scores .+ quality_scores')
    complementarity = X' * env.A * X
    regime_sign = sign.(X' * env.B * X)
    return (; quality_scores, quality_surface, complementarity, regime_sign)
end

"""Construct the matrix with entries `Q + f(x_i, x_j)` for fixed types and matching environment."""
function conditional_match_matrix(components, rho::Real, delta::Real)
    r = Float64(rho)
    d = Float64(delta)
    coefficients = match_signal_coefficients(r, d)
    interaction = (1.0 .+ d .* components.regime_sign) .* components.complementarity
    return Q_OFFSET .+
           coefficients.signal_shift .+
           coefficients.quality_weight .* components.quality_surface .+
           coefficients.interaction_weight .* interaction
end

"""Subtract the grand mean from a matrix of underlying match values."""
center_match_matrix(matrix::AbstractMatrix) = Matrix{Float64}(matrix .- mean(matrix))

"""Return descending singular values of a real symmetric matrix."""
function symmetric_singular_values(matrix::AbstractMatrix)
    values = abs.(eigvals(Symmetric(Matrix{Float64}(matrix))))
    sort!(values; rev=true)
    return values
end

"""Number of singular components reaching the specified spectral-energy fraction."""
function effective_rank(singular_values::AbstractVector; fraction=AnalysisSettings.ENERGY_FRACTION)
    0 < fraction <= 1 || throw(ArgumentError("energy fraction must lie in (0,1]"))
    energy = sum(abs2, singular_values)
    energy > 0.0 || throw(ArgumentError("singular spectrum has zero energy"))
    index = findfirst(>=(fraction), cumsum(abs2.(singular_values)) ./ energy)
    isnothing(index) && error("cumulative spectral energy did not reach the specified fraction")
    return index
end

"""
    type_geometry_projection(agent_types, curve_geo; n_curve=TYPE_CURVE_POINTS)

Project realized types and their noiseless curve onto its leading principal components.
"""
function type_geometry_projection(
    agent_types::Vector{Vector{Float64}}, curve_geo; n_curve::Int=TYPE_CURVE_POINTS
)
    n_curve >= AnalysisSettings.PROJECTION_COMPONENTS || throw(ArgumentError("insufficient curve points for the projection"))
    curve_parameter = collect(range(0.0, 1.0; length=n_curve))
    curve_points = reduce(hcat, curve_point.(curve_parameter, Ref(curve_geo)))
    center = vec(mean(curve_points; dims=2))
    centered_curve = curve_points .- center
    decomposition = svd(centered_curve; full=false)
    size(decomposition.U, 2) >= AnalysisSettings.PROJECTION_COMPONENTS ||
        throw(ArgumentError("type geometry has too few dimensions for the projection"))
    basis = copy(decomposition.U[:, 1:AnalysisSettings.PROJECTION_COMPONENTS])
    for component in axes(basis, 2)
        pivot = argmax(abs.(view(basis, :, component)))
        basis[pivot, component] < 0.0 && (basis[:, component] .*= -1.0)
    end
    type_points = reduce(hcat, agent_types)
    return Dict{String,Any}(
        "seed" => HEATMAP_SEED,
        "curve_parameter" => Float32.(curve_parameter),
        "curve_projection" => Float32.(basis' * centered_curve),
        "type_projection" => Float32.(basis' * (type_points .- center)),
        "projection_method" => "first $(AnalysisSettings.PROJECTION_COMPONENTS) principal components of the noiseless curve",
    )
end

"""Generate all seed-level inputs required by the three supplementary DGP figures."""
function build_dgp_figure_data(; seeds=DGP_SEEDS, parameter_overrides=(;))
    seed_values = Int.(collect(seeds))
    isempty(seed_values) && throw(ArgumentError("at least one DGP seed is required"))
    conditions = effective_dgp_conditions()
    rank_values = Dict(
        dgp_condition_id(condition.rho, condition.delta) => Int[] for
        condition in conditions
    )
    spectra = Dict(string(rho) => Vector{Vector{Float64}}() for rho in SPECTRUM_RHOS)
    heatmaps = Dict{String,Matrix{Float32}}()
    heatmap_quality_scores = Float64[]
    type_geometry = nothing
    overrides = merge((; N=DGP_N), parameter_overrides)

    for seed in seed_values
        params = default_params(; seed, overrides...)
        dgp = generate_matching_dgp(params, StableRNG(seed))
        if seed == HEATMAP_SEED
            type_geometry = type_geometry_projection(dgp.agent_types, dgp.curve_geo)
        end
        components = dgp_components(dgp.agent_types, dgp.env)
        quality_order = sortperm(components.quality_scores)
        heatmap_rank_indices = round.(
            Int,
            range(1, length(quality_order); length=min(HEATMAP_N, length(quality_order))),
        )
        heatmap_order = quality_order[heatmap_rank_indices]

        for condition in conditions
            matrix = conditional_match_matrix(components, condition.rho, condition.delta)
            centered = center_match_matrix(matrix)
            singular_values = symmetric_singular_values(centered)
            condition_id = dgp_condition_id(condition.rho, condition.delta)
            push!(rank_values[condition_id], effective_rank(singular_values))

            if condition.delta == BASELINE_DELTA && condition.rho in SPECTRUM_RHOS
                push!(
                    spectra[string(condition.rho)],
                    singular_values ./ first(singular_values),
                )
            end
        end

        if seed == HEATMAP_SEED
            for condition in HEATMAP_CONDITIONS
                matrix = conditional_match_matrix(components, condition.rho, condition.delta)
                centered = center_match_matrix(matrix)
                condition_id = dgp_condition_id(condition.rho, condition.delta)
                heatmaps[condition_id] = Float32.(centered[heatmap_order, heatmap_order])
            end
            append!(heatmap_quality_scores, components.quality_scores[heatmap_order])
        end
    end

    condition_rows = [
        Dict(
            "rho" => condition.rho,
            "delta" => condition.delta,
            "rank90_seed_values" =>
                rank_values[dgp_condition_id(condition.rho, condition.delta)],
        ) for condition in conditions
    ]
    spectrum_matrices = Dict(
        rho => reduce(hcat, seed_spectra) for (rho, seed_spectra) in spectra
    )
    baseline = default_params(; seed=first(seed_values), overrides...)
    isnothing(type_geometry) && error("type-geometry seed was not generated")
    return Dict{String,Any}(
        "seeds" => seed_values,
        "rho_values" => Float64.(RHO_OAT),
        "delta_values" => Float64.(DELTA_VALS),
        "baseline_delta" => BASELINE_DELTA,
        "conditions" => condition_rows,
        "spectra" => spectrum_matrices,
        "heatmap_seed" => HEATMAP_SEED,
        "heatmap_conditions" => copy(HEATMAP_CONDITIONS),
        "heatmaps" => heatmaps,
        "heatmap_quality_scores" => heatmap_quality_scores,
        "type_geometry" => type_geometry,
        "design" => Dict(
            "N" => baseline.N,
            "d" => baseline.d,
            "s" => baseline.s,
            "sigma_x" => baseline.sigma_x,
            "sigma_eps" => baseline.sigma_eps,
            "conditional_mean_without_match_noise" => true,
            "matrix_centering" => "grand mean",
            "ordering" => "x_i'c",
            "heatmap_display_N" => min(HEATMAP_N, baseline.N),
            "heatmap_sampling" => "evenly spaced ranks of x_i'c",
            "type_projection_components" => size(type_geometry["curve_projection"], 1),
            "type_curve_points" => TYPE_CURVE_POINTS,
            "effective_rank_energy_fraction" => AnalysisSettings.ENERGY_FRACTION,
        ),
    )
end

function main()
    exploratory = "--exploratory" in ARGS
    provenance = reporting_git_provenance(
        REPO;
        sources=("scripts/analysis_settings.jl",
            @__FILE__,
            "src",
            "scripts/sweep/sweep_config.jl",
        ),
        require_clean=!exploratory,
    )
    data = build_dgp_figure_data()
    data["meta"] = Dict(
        "source" => "scripts/paper/dgp_figdata.jl",
        "analysis_git_commit" => provenance.commit,
        "analysis_source_clean" => provenance.source_clean,
        "exploratory" => exploratory,
    )
    mkpath(dirname(OUTFILE))
    jldsave(OUTFILE; figdata=data)
    println(
        "wrote $OUTFILE ($(round(filesize(OUTFILE) / 1024^2; digits=1)) MB; ",
        length(data["conditions"]),
        " effective conditions, ",
        length(data["seeds"]),
        " seeds)",
    )
    return nothing
end

end # module DGPFigureData

if abspath(PROGRAM_FILE) == @__FILE__
    DGPFigureData.main()
end
