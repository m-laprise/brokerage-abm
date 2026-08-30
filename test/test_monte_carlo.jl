using Test

include(joinpath(@__DIR__, "..", "scripts", "monte_carlo.jl"))

@testset "Monte Carlo intervals" begin
    summary = monte_carlo_interval(1:20)
    @test summary.n == 20
    @test summary.mean == 10.5
    @test summary.se ≈ 1.3228756555322954
    @test summary.lower ≈ 7.731189431979745
    @test summary.upper ≈ 13.268810568020255

    filtered = monte_carlo_interval([1.0, 2.0, NaN, Inf])
    @test filtered.n == 2
    @test filtered.mean == 1.5

    constant_summary = monte_carlo_interval(fill(3.0, 20))
    @test constant_summary.se == 0.0
    @test constant_summary.lower == constant_summary.mean == constant_summary.upper

    singleton = monte_carlo_interval([4.0])
    @test singleton.mean == 4.0
    @test all(isnan, (singleton.se, singleton.lower, singleton.upper))

    paired = paired_monte_carlo_interval([1.0, 2.0, NaN, 4.0], [2.0, 4.0, 100.0, Inf])
    @test paired.n == 2
    @test paired.differences == [1.0, 2.0]
    @test paired.mean == 1.5

    @test_throws DimensionMismatch paired_monte_carlo_interval([1.0], [1.0, 2.0])
    @test_throws ArgumentError monte_carlo_interval([1.0, 2.0]; level=1.0)
end
