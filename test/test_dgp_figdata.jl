using Test
using BrokerageABM
using BrokerageABM: Q_OFFSET, generate_matching_dgp, match_signal
using LinearAlgebra: rank
using StableRNGs: StableRNG

include(joinpath(@__DIR__, "..", "scripts", "paper", "dgp_figdata.jl"))
const DGP = DGPFigureData

@testset "Supplementary DGP figure data" begin
    @testset "effective grid has 31 scientific conditions" begin
        @test DGP.DGP_N == 1000
        @test length(DGP.DGP_SEEDS) == 50
        @test DGP.HEATMAP_N == 100
        conditions = DGP.effective_dgp_conditions()
        @test length(conditions) == 31
        @test 0.15 in DGP.RHO_OAT
        @test count(condition -> condition.rho == 1.0, conditions) == 1
        @test only(condition.delta for condition in conditions if condition.rho == 1.0) ==
            0.5
    end

    @testset "matrix construction matches the production matching function" begin
        params = default_params(N=20, seed=116, rho=0.5, delta=0.5)
        dgp = generate_matching_dgp(params, StableRNG(params.seed))
        components = DGP.dgp_components(dgp.agent_types, dgp.env)
        matrix = DGP.conditional_match_matrix(components, params.rho, params.delta)
        expected = [
            Q_OFFSET + match_signal(dgp.agent_types[i], dgp.agent_types[j], dgp.env) for
            i in 1:params.N, j in 1:params.N
        ]
        @test matrix ≈ expected atol = 1e-12 rtol = 1e-12
        @test matrix ≈ matrix' atol = 1e-12 rtol = 1e-12
    end

    @testset "pure-quality boundary is delta-invariant and rank two" begin
        params = default_params(N=24, seed=117)
        dgp = generate_matching_dgp(params, StableRNG(params.seed))
        components = DGP.dgp_components(dgp.agent_types, dgp.env)
        low_delta = DGP.conditional_match_matrix(components, 1.0, 0.0)
        high_delta = DGP.conditional_match_matrix(components, 1.0, 1.0)
        centered = DGP.center_match_matrix(low_delta)
        @test low_delta == high_delta
        @test rank(centered; atol=1e-10) <= 2
        @test DGP.effective_rank_90(DGP.symmetric_singular_values(centered)) <= 2
    end

    @testset "quality ordering and failure path are explicit" begin
        params = default_params(N=20, seed=118)
        dgp = generate_matching_dgp(params, StableRNG(params.seed))
        components = DGP.dgp_components(dgp.agent_types, dgp.env)
        order = sortperm(components.quality_scores)
        @test issorted(components.quality_scores[order])
        @test_throws ArgumentError DGP.effective_rank_90(zeros(4))
    end

    @testset "figure-data assembly preserves seed-level inputs" begin
        data = DGP.build_dgp_figure_data(; seeds=1:2, parameter_overrides=(; N=20))
        @test data["seeds"] == [1, 2]
        @test length(data["conditions"]) == 31
        @test all(length(row["rank90_seed_values"]) == 2 for row in data["conditions"])
        @test all(size(spectrum) == (20, 2) for spectrum in values(data["spectra"]))
        @test length(data["spectra"]) == 7
        expected_heatmap_keys = Set([
            "rho=0.0|delta=0.0",
            "rho=0.5|delta=0.0",
            "rho=1.0|delta=0.0",
            "rho=0.0|delta=1.0",
            "rho=0.5|delta=1.0",
        ])
        @test Set(keys(data["heatmaps"])) == expected_heatmap_keys
        @test length(data["heatmap_conditions"]) == 5
        @test all(size(heatmap) == (20, 20) for heatmap in values(data["heatmaps"]))
        @test all(eltype(heatmap) == Float32 for heatmap in values(data["heatmaps"]))
        @test length(data["heatmap_quality_scores"]) == 20
        @test data["design"]["N"] == 20
        @test data["design"]["heatmap_display_N"] == 20
        @test issorted(data["heatmap_quality_scores"])
        @test data["design"]["conditional_mean_without_match_noise"] == true
        geometry = data["type_geometry"]
        @test geometry["seed"] == DGP.HEATMAP_SEED
        @test size(geometry["curve_projection"]) == (3, DGP.TYPE_CURVE_POINTS)
        @test size(geometry["type_projection"]) == (3, 20)
        @test length(geometry["curve_parameter"]) == DGP.TYPE_CURVE_POINTS
        @test first(geometry["curve_parameter"]) == 0.0f0
        @test last(geometry["curve_parameter"]) == 1.0f0
    end

    @testset "type projection rejects invalid display dimensions" begin
        params = default_params(N=10, d=2, s=2, seed=119)
        dgp = generate_matching_dgp(params, StableRNG(params.seed))
        @test_throws ArgumentError DGP.type_geometry_projection(
            dgp.agent_types, dgp.curve_geo
        )
        @test_throws ArgumentError DGP.type_geometry_projection(
            dgp.agent_types, dgp.curve_geo; n_curve=2
        )
    end
end
