"""
Test reporting-only net-output accounting from deterministic saved-metric fixtures.
Run separately from the simulation suite:
    julia --project --threads=auto test/test_net_output.jl
"""

using Test
using DataFrames: DataFrame

include(joinpath(@__DIR__, "..", "scripts", "net_output_data.jl"))

@testset "Net output per principal" begin
    account = NetOutputData.net_output_accounting
    result = account(18.0, 6, 3, 20, 0.2, 0.3)
    @test result.gross_match_output == 18.0
    @test result.total_search_cost ≈ 1.2
    @test result.total_broker_fees ≈ 0.9
    @test result.net_output_per_principal ≈ 0.795
    @test account(0.0, 6, 0, 20, 0.2, 0.3).net_output_per_principal ≈ -0.06
    @test account(0.0, 0, 0, 20, 0.2, 0.3).net_output_per_principal == 0.0
    @test account(-2.0, 0, 0, 20, 0.2, 0.3).net_output_per_principal == -0.1
    @test account(18.0, 6, 3, 20, 0.0, 0.0).net_output_per_principal == 0.9
    @test_throws ArgumentError account(1.0, 0, 0, 0, 0.2, 0.3)
    @test_throws ArgumentError account(1.0, -1, 0, 20, 0.2, 0.3)
    @test_throws ArgumentError account(1.0, 0, -1, 20, 0.2, 0.3)
    @test_throws ArgumentError account(NaN, 0, 0, 20, 0.2, 0.3)
    @test_throws ArgumentError account(1.0, 0, 0, 20, -0.2, 0.3)
    @test_throws ArgumentError account(1.0, 0, 0, 20, 0.2, Inf)

    # Reciprocity changes fees when it creates another broker placement, not
    # the output of the relationship. Mixed-channel offers also count it once.
    unilateral = account(4.0, 0, 1, 20, 0.2, 0.3)
    reciprocal = account(4.0, 0, 2, 20, 0.2, 0.3)
    mixed = account(4.0, 1, 1, 20, 0.2, 0.3)
    @test unilateral.gross_match_output ==
        reciprocal.gross_match_output ==
        mixed.gross_match_output
    @test unilateral.net_output_per_principal - reciprocal.net_output_per_principal ≈
        0.3 / 20
    @test unilateral.net_output_per_principal - mixed.net_output_per_principal ≈ 0.2 / 20

    raw = DataFrame(
        total_demand=[10, 0], outsourced_slots=[4, 0],
        access_count=[1, 0], assessment_count=[2, 0],
        n_self_matches=[2, 0], n_broker_matches=[2, 0],
        q_self_mean=[4.0, NaN], q_broker_mean=[5.0, NaN],
        self_net_output_per_requested_position=[99.0, NaN],
        broker_net_output_per_requested_position=[99.0, NaN],
        net_output_per_requested_position=[99.0, NaN],
    )
    reconstructed = deepcopy(raw)
    NetOutputData.reconstruct_frame!(reconstructed, 20, 0.2, 0.3)
    @test reconstructed.gross_match_output == [18.0, 0.0]
    @test reconstructed.total_search_cost ≈ [1.2, 0.0]
    @test reconstructed.total_broker_fees ≈ [0.9, 0.0]
    @test reconstructed.net_output_per_principal ≈ [0.795, 0.0]
    @test all(name -> name ∉ propertynames(reconstructed), NetOutputData.OLD_METRICS)
    @test raw.net_output_per_requested_position[1] == 99.0
    repeated = deepcopy(reconstructed)
    NetOutputData.reconstruct_frame!(repeated, 20, 0.2, 0.3)
    @test isequal(repeated, reconstructed)

    empty_channel = DataFrame(
        total_demand=[2, 0],
        outsourced_slots=[0, 0],
        access_count=[0, 0],
        assessment_count=[0, 0],
        n_self_matches=[0, 0],
        n_broker_matches=[0, 0],
        q_self_mean=[NaN, NaN],
        q_broker_mean=[NaN, NaN],
        net_output_per_requested_position=[-0.2, NaN],
    )
    NetOutputData.reconstruct_frame!(empty_channel, 20, 0.2, 0.3)
    @test empty_channel.net_output_per_principal ≈ [-0.02, 0.0]
    @test :net_output_per_requested_position ∉ propertynames(empty_channel)
    @test_throws ArgumentError NetOutputData.relationship_output(1, NaN)
    @test_throws ArgumentError NetOutputData.relationship_output(-1, 4.0)
    invalid = deepcopy(reconstructed)
    invalid.net_output_per_principal[1] += 1
    @test_throws ErrorException NetOutputData.reconstruct_frame!(
        invalid, 20, 0.2, 0.3
    )
    invalid = deepcopy(reconstructed)
    invalid.outsourced_slots[1] = invalid.total_demand[1] + 1
    @test_throws ErrorException NetOutputData.reconstruct_frame!(
        invalid, 20, 0.2, 0.3
    )
end
