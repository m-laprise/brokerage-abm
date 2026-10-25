using Test
using BrokerageABM
using Statistics: mean, std
using Distributions: TDist, quantile

include(joinpath(@__DIR__, "..", "scripts", "paper", "audit_convergence.jl"))

@testset "Convergence reporting" begin
    record(values; seeds=collect(eachindex(values))) = (;
        dataset="fixture",
        kind="level",
        outcome="outcome",
        window="late",
        rel="baseline",
        reference="",
        comparison="",
        seeds,
        values,
    )
    values = [5.0, 3.0, 6.0, 4.0]
    r = record(values; seeds=[3, 1, 4, 2])
    result = ConvergenceAudit.audit_estimate(r, 2)
    half_width = quantile(TDist(3), 0.975) * std(values) / sqrt(4)
    @test result.estimate == mean(values)
    @test result.nested_estimate == 3.5
    @test result.ci_half_width ≈ half_width
    @test result.relative_ci_half_width ≈ half_width / mean(values)
    @test result.relative_nested_change ≈ 1 / mean(values)
    @test result.interval_direction == 1

    zero = ConvergenceAudit.audit_estimate(record(zeros(4)), 2)
    @test (zero.estimate, zero.ci_half_width, zero.nested_change) == (0, 0, 0)
    @test !zero.relative_precision_defined && isnan(zero.relative_ci_half_width)
    crossing = ConvergenceAudit.audit_estimate(record([-2.0, -1.0, 1.0, 3.0]), 2)
    @test !crossing.relative_precision_defined && crossing.mean_sign_changed
    @test crossing.interval_direction == 0
    @test_throws ErrorException ConvergenceAudit.audit_estimate(
        record([1.0, NaN, 2.0, 3.0]), 2
    )
    @test_throws ErrorException ConvergenceAudit.audit_estimate(
        record(ones(4); seeds=[1, 1, 2, 3]), 2
    )
    @test_throws ErrorException ConvergenceAudit.audit_estimate(r, 4)

    a = record([30.0, 10.0, 20.0]; seeds=[3, 1, 2])
    b = record([22.0, 35.0, 50.0]; seeds=[2, 3, 4])
    seeds, differences = ConvergenceAudit.paired_values(a, b)
    @test seeds == [2, 3]
    @test differences == [2.0, 5.0]
    same_seeds = record([25.0, 31.0, 14.0]; seeds=[2, 3, 1])
    _, differences = ConvergenceAudit.paired_values(a, same_seeds)
    @test mean(differences) ≈ mean(same_seeds.values) - mean(a.values)
    @test_throws ErrorException ConvergenceAudit.paired_values(
        a, record([1.0, 2.0]; seeds=[3, 4])
    )
    @test_throws ErrorException ConvergenceAudit.paired_values(
        a, record([1.0, 2.0]; seeds=[2, 2])
    )
    @test_throws ErrorException ConvergenceAudit.paired_values(
        a, record([1.0]; seeds=[1, 2])
    )

    numerator = record([11.0, 3.0, 16.0, 6.0]; seeds=r.seeds)
    ratio = ConvergenceAudit.audit_ratio(r, numerator, 2)
    @test ratio.estimate == mean(numerator.values) / mean(r.values)
    @test ratio.nested_estimate == 9 / 7
    @test ratio.estimate != mean(numerator.values ./ r.values)
    @test_throws ErrorException ConvergenceAudit.audit_ratio(record(zeros(4)), numerator, 2)

    stored = Dict{NTuple{4,String},NamedTuple}()
    ConvergenceAudit.add_level!(
        stored, "fixture", "baseline", "outcome", [1, 2], [1.0, 2.0]
    )
    @test length(stored) == 1
    @test_throws ErrorException ConvergenceAudit.add_level!(
        stored, "fixture", "baseline", "outcome", [1, 2], [1.0, 3.0]
    )
    summary = only(ConvergenceAudit.summarize([zero]))
    @test summary.n_relative_precision == 0 && isnan(summary.p50_relative_ci_half_width)
end
