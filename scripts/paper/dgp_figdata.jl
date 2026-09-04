"""
    scripts/paper/dgp_figdata.jl

Generate the seed-level data for Supplementary Figures S1--S3 directly from the
model's initialization-stage data-generating process with 1,000 principals. For
each of 50 seeds, the realized types and matching-function objects are held fixed
while `rho` and `delta` vary over the effective current grid. No simulation periods
or match-noise draws are used. S1 projects the realized types and their latent
curve into three dimensions. S2 displays five distinct matching conditions for
100 principals sampled at evenly spaced general-quality ranks; S3 uses all 1,000
principals.

The output, `output/supplement/dgp_figdata.jld2`, contains the centered conditional
type-geometry projection used in S1, conditional match-value matrices used in S2,
normalized singular spectra and seed-level 90%-energy effective ranks used in S3,
and complete generation provenance.

Usage: julia --project --threads=auto scripts/paper/dgp_figdata.jl
"""

module DGPFigureData

using BrokerageABM
using BrokerageABM: MatchingEnv, Q_OFFSET, curve_point, generate_matching_dgp
using JLD2: jldsave
using LinearAlgebra: Symmetric, eigvals, svd
using StableRNGs: StableRNG
using Statistics: mean

include(joinpath(@__DIR__, "..", "sweep", "sweep_config.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUTFILE = joinpath(REPO, "output", "supplement", "dgp_figdata.jld2")
const DGP_N = 1000
const DGP_SEEDS = collect(1:50)
const HEATMAP_N = 100
const HEATMAP_SEED = 1
const TYPE_CURVE_POINTS = 500
const SPECTRUM_RHOS = copy(RHO_OAT)
const BASELINE_DELTA = Float64(SWEEP_BASELINE.delta)
const HEATMAP_CONDITIONS = [
    (; rho=0.0, delta=BASELINE_DELTA),
    (; rho=0.5, delta=BASELINE_DELTA),
    (; rho=1.0, delta=BASELINE_DELTA),
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

Construct the three fixed matrices that determine conditional pairwise match value:
the additive general-quality surface, base complementarity, and regime sign.
"""
function dgp_components(agent_types::Vector{Vector{Float64}}, env::MatchingEnv)
    X = reduce(hcat, agent_types)
    quality_scores = vec(X' * env.c)
    quality_surface = 0.5 .* (quality_scores .+ quality_scores')
    complementarity = X' * env.A * X
    regime_sign = sign.(X' * env.B * X)
    return (; quality_scores, quality_surface, complementarity, regime_sign)
end

"""Construct `E[q_ij | x_i, x_j]` without drawing match noise."""
function conditional_match_matrix(components, rho::Real, delta::Real)
    r = Float64(rho)
    d = Float64(delta)
    return Q_OFFSET .+ r .* components.quality_surface .+
           (1.0 - r) .* ((1.0 .+ d .* components.regime_sign) .* components.complementarity)
end

"""Subtract the grand mean from a conditional match-value matrix."""
center_match_matrix(matrix::AbstractMatrix) = Matrix{Float64}(matrix .- mean(matrix))

"""Return descending singular values of a real symmetric matrix."""
function symmetric_singular_values(matrix::AbstractMatrix)
    values = abs.(eigvals(Symmetric(Matrix{Float64}(matrix))))
    sort!(values; rev=true)
    return values
end

"""Number of singular components carrying at least 90% of spectral energy."""
function effective_rank_90(singular_values::AbstractVector)
    energy = sum(abs2, singular_values)
    energy > 0.0 || throw(ArgumentError("singular spectrum has zero energy"))
    index = findfirst(>=(0.90), cumsum(abs2.(singular_values)) ./ energy)
    isnothing(index) && error("cumulative spectral energy did not reach 90%")
    return index
end

"""
    type_geometry_projection(agent_types, curve_geo; n_curve=TYPE_CURVE_POINTS)

Project a realized type distribution and its noiseless curve onto the curve's
first three principal components for visualization only.
"""
function type_geometry_projection(
    agent_types::Vector{Vector{Float64}}, curve_geo; n_curve::Int=TYPE_CURVE_POINTS
)
    n_curve >= 3 || throw(ArgumentError("at least three curve points are required"))
    curve_parameter = collect(range(0.0, 1.0; length=n_curve))
    curve_points = reduce(hcat, curve_point.(curve_parameter, Ref(curve_geo)))
    center = vec(mean(curve_points; dims=2))
    centered_curve = curve_points .- center
    decomposition = svd(centered_curve; full=false)
    size(decomposition.U, 2) >= 3 ||
        throw(ArgumentError("type geometry must have at least three dimensions"))
    basis = copy(decomposition.U[:, 1:3])
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
        "projection_method" => "first three principal components of the noiseless curve",
    )
end

"""Generate all seed-level inputs required by the two supplementary DGP figures."""
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
            push!(rank_values[condition_id], effective_rank_90(singular_values))

            if condition.delta == BASELINE_DELTA && condition.rho in SPECTRUM_RHOS
                push!(
                    spectra[string(condition.rho)],
                    singular_values ./ first(singular_values),
                )
            end
            if seed == HEATMAP_SEED && condition in HEATMAP_CONDITIONS
                heatmaps[condition_id] = Float32.(centered[heatmap_order, heatmap_order])
                isempty(heatmap_quality_scores) && append!(
                    heatmap_quality_scores, components.quality_scores[heatmap_order]
                )
            end
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
            "ordering" => "general quality x_i'c",
            "heatmap_display_N" => min(HEATMAP_N, baseline.N),
            "heatmap_sampling" => "evenly spaced general-quality ranks",
            "type_projection_components" => 3,
            "type_curve_points" => TYPE_CURVE_POINTS,
            "effective_rank_energy_fraction" => 0.90,
        ),
    )
end

function main()
    provenance = reporting_git_provenance(REPO)
    data = build_dgp_figure_data()
    data["meta"] = Dict(
        "source" => "scripts/paper/dgp_figdata.jl",
        "analysis_git_commit" => provenance.commit,
        "analysis_source_clean" => provenance.source_clean,
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
