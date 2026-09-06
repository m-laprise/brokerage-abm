using DataFrames: DataFrame, nrow
using Test

include(
    joinpath(
        @__DIR__, "..", "scripts", "ridge", "calibration", "summarize.jl"
    ),
)

module RidgeCalTestSweepDesign
include(joinpath(@__DIR__, "..", "scripts", "sweep", "sweep_config.jl"))
end

@testset "Ridge calibration design" begin
    @test RIDGECAL_SCHEMA_VERSION == 1
    @test RIDGECAL_BASELINE == RidgeCalTestSweepDesign.SWEEP_BASELINE
    @test RIDGECAL_N == RIDGECAL_BASELINE.N == 1000
    @test RIDGECAL_T == 500
    @test RIDGECAL_EARLY_PERIODS == 301:400
    @test RIDGECAL_LATE_PERIODS == 401:500
    @test RIDGECAL_LAMBDAS == [0.0003, 0.001, 0.003, 0.01, 0.03]
    @test isempty(intersect(RIDGECAL_SCREEN_SEEDS, RIDGECAL_CONFIRM_SEEDS))
    @test isempty(intersect(RIDGECAL_SCREEN_SEEDS, 1:50))
    @test isempty(intersect(RIDGECAL_CONFIRM_SEEDS, 1:50))

    configs = ridgecal_screen_configs()
    @test length(configs) == 25
    @test length(unique(ridgecal_config_key(config) for config in configs)) == 25
    @test sort(unique(config[:lambda_agent] for config in configs)) == RIDGECAL_LAMBDAS
    @test sort(unique(config[:lambda_broker] for config in configs)) == RIDGECAL_LAMBDAS
    @test length(ridgecal_build_entries(configs, RIDGECAL_SCREEN_SEEDS)) == 250

    confirmation = ridgecal_confirmation_config(0.003, 0.001)
    @test confirmation[:lambda_agent] == 0.003
    @test confirmation[:lambda_broker] == 0.001
    @test length(ridgecal_build_entries([confirmation], RIDGECAL_CONFIRM_SEEDS)) == 5
    @test_throws ErrorException ridgecal_confirmation_config(0.0001, 0.001)
    @test_throws ErrorException ridgecal_confirmation_config(0.003, 0.1)
    @test_throws ErrorException ridgecal_stage_seeds(:joint)
end

@testset "Ridge calibration provenance" begin
    expected = Dict{Symbol,Any}(
        :git_commit => "commit-a",
        :julia_version => "1.11.3",
        :pkg_manifest_hash => "manifest-a",
    )
    @test isempty(ridgecal_provenance_mismatches(expected, copy(expected)))

    current = copy(expected)
    current[:git_commit] = "commit-b"
    current[:pkg_manifest_hash] = "manifest-b"
    @test ridgecal_provenance_mismatches(expected, current) == [
        :git_commit, :pkg_manifest_hash
    ]
end

@testset "Ridge calibration summaries" begin
    @test ridgecal_finite_mean([1.0, 2.0], "fixture") == 1.5
    @test_throws ErrorException ridgecal_finite_mean(Float64[], "fixture")
    @test_throws ErrorException ridgecal_finite_mean([1.0, NaN], "fixture")

    summary = DataFrame(
        lambda_agent=repeat([0.001, 0.003], inner=2),
        lambda_broker=repeat([0.001, 0.003], outer=2),
        agent_rank_median=[0.4, 0.6, 0.8, 1.0],
        broker_rank_median=[0.9, 0.7, 0.5, 0.3],
        agent_rank_change_median=[0.01, 0.02, 0.03, 0.04],
        broker_rank_change_median=[-0.01, -0.02, -0.03, -0.04],
    )
    agent = ridgecal_marginal_summary(summary, :agent)
    broker = ridgecal_marginal_summary(summary, :broker)
    @test nrow(agent) == nrow(broker) == 2
    @test agent.lambda == [0.003, 0.001]
    @test broker.lambda == [0.001, 0.003]
    @test agent.rank_median_across_cells == [0.9, 0.5]
    @test broker.rank_median_across_cells == [0.7, 0.5]
    @test_throws ErrorException ridgecal_marginal_summary(summary, :combined)

    complete = DataFrame(config_id=[0, 0, 1, 1], seed=[1, 2, 1, 2])
    @test isnothing(ridgecal_validate_stage_seed_sets(complete, 1:2))
    incomplete = DataFrame(config_id=[0, 0, 1], seed=[1, 2, 1])
    @test_throws ErrorException ridgecal_validate_stage_seed_sets(incomplete, 1:2)
end
