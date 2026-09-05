using Test
using DataFrames: DataFrame

include(joinpath(@__DIR__, "..", "scripts", "nn_calibration", "summarize.jl"))

@testset "NN calibration design" begin
    settings = nncal_candidate_settings()
    @test length(settings) == 9
    @test all(setting.initial_steps == 2 * setting.recurrent_steps for setting in settings)

    configs = nncal_screen_configs()
    @test length(configs) == 17
    @test length(unique(nncal_config_key(config) for config in configs)) == 17
    @test count(config -> config[:agent_scan], configs) == 9
    @test count(config -> config[:broker_scan], configs) == 9
    @test length(nncal_build_entries(configs, NNCAL_SCREEN_SEEDS)) == 51
end

@testset "NN calibration shortlist eligibility" begin
    configs = nncal_screen_configs()
    shortlist = nncal_shortlist_configs(configs, [1, 2], [0, 9])
    by_id = Dict(config[:config_id] => config for config in shortlist)
    @test Set(keys(by_id)) == Set([0, 1, 2, 9])
    @test !by_id[0][:agent_scan]
    @test by_id[0][:broker_scan]
    @test by_id[1][:agent_scan]
    @test !by_id[1][:broker_scan]
end

@testset "NN calibration selection rule" begin
    summary = DataFrame(
        config_id=0:3,
        agent_scan=fill(true, 4),
        broker_scan=fill(false, 4),
        agent_rank_median=[0.60, 0.609, 0.611, 0.50],
        broker_rank_median=zeros(4),
        agent_recurrent_steps=[50, 50, 100, 200],
        broker_recurrent_steps=fill(100, 4),
        agent_eta_lr=[0.003, 0.01, 0.03, 0.01],
        broker_eta_lr=fill(0.01, 4),
    )
    selected = nncal_select_efficient(summary, :agent)
    @test selected.config_id == 1

    summary.agent_rank_median = [0.60, 0.6005, 0.62, 0.50]
    selected = nncal_select_efficient(summary, :agent)
    @test selected.config_id == 2
end
