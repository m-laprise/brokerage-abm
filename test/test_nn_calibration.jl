using Test

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

    confirmation = only(nncal_confirmation_config())
    @test confirmation[:agent_eta_lr] == 0.003
    @test confirmation[:broker_eta_lr] == 0.03
    @test confirmation[:agent_initial_steps] == 100
    @test confirmation[:broker_initial_steps] == 100
    @test confirmation[:agent_recurrent_steps] == 50
    @test confirmation[:broker_recurrent_steps] == 50
    @test !confirmation[:agent_scan]
    @test !confirmation[:broker_scan]
    @test length(nncal_build_entries([confirmation], NNCAL_CONFIRM_SEEDS)) == 5
    @test_throws ErrorException nncal_stage_seeds(:combined)
    @test_throws ErrorException nncal_stage_configs(:combined)
end
