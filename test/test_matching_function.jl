using Test
using TransientBrokerage
using TransientBrokerage: Q_OFFSET, R_BASE_FRAC, calibrate, generate_matching_env
using TransientBrokerage: match_output, match_output!, match_signal, match_signal!
using TransientBrokerage: regime_gain, regime_gain!
using StableRNGs: StableRNG
using LinearAlgebra: dot, norm, normalize, eigvals, issymmetric

@testset "Matching Function" begin
    d = 8
    rho = 0.50

    function test_agent_types(d, n, rng)
        [normalize(randn(rng, d)) for _ in 1:n]
    end

    function test_env(
        types;
        rho::Float64=rho,
        delta::Float64=0.5,
        sigma_eps::Float64=0.25,
        env_seed::Int=42,
        geo_seed::Int=11,
    )
        geo = TransientBrokerage.generate_curve_geometry(d, d, StableRNG(geo_seed))
        return generate_matching_env(
            d, rho, delta, sigma_eps, types, StableRNG(env_seed); curve_geo=geo
        )
    end

    @testset "MatchingEnv construction" begin
        types = test_agent_types(d, 50, StableRNG(10))
        env = test_env(types)
        @test env.d == d
        @test env.rho == rho
        @test env.delta == 0.5
        @test length(env.c) == d
        @test size(env.A) == (d, d)
        @test size(env.B) == (d, d)
    end

    @testset "curve_geo path decouples c and A, while B depends on realized types" begin
        rng_geo = StableRNG(12)
        geo = TransientBrokerage.generate_curve_geometry(d, d, rng_geo)
        rng_types = StableRNG(13)
        types_1, _ = TransientBrokerage.generate_agent_types(50, geo, 0.5, rng_types)
        types_2 = test_agent_types(d, 50, StableRNG(14))

        env_1 = generate_matching_env(
            d, rho, 0.5, 0.25, types_1, StableRNG(99); curve_geo=geo
        )
        env_2 = generate_matching_env(
            d, rho, 0.5, 0.25, types_2, StableRNG(99); curve_geo=geo
        )

        @test env_1.c == env_2.c
        @test env_1.A == env_2.A
        @test env_1.B != env_2.B
        @test TransientBrokerage.weighted_regime_overlap(env_1.A, env_1.B, types_1) ≈ 0.0 atol=1e-12
        @test TransientBrokerage.weighted_regime_overlap(env_2.A, env_2.B, types_2) ≈ 0.0 atol=1e-12
    end

    @testset "A is SPD and B is weighted-orthogonalized" begin
        types = test_agent_types(d, 50, StableRNG(10))
        env = test_env(types)
        @test all(eigvals(env.A) .> 0)
        @test issymmetric(env.A)
        @test issymmetric(env.B)
        @test minimum(eigvals(env.B)) < 0.0
        @test maximum(eigvals(env.B)) > 0.0
        @test TransientBrokerage.weighted_regime_overlap(env.A, env.B, types) ≈ 0.0 atol=1e-12
    end

    @testset "Matching function symmetry: f(x_i, x_j) == f(x_j, x_i)" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env = test_env(types)
        rng = StableRNG(99)
        @test all(1:100) do _
            i, j = rand(rng, 1:20), rand(rng, 1:20)
            match_signal(types[i], types[j], env) ≈ match_signal(types[j], types[i], env)
        end
    end

    @testset "Regime gain values" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env = test_env(types)
        rng = StableRNG(77)
        gains = [
            regime_gain(types[rand(rng, 1:20)], types[rand(rng, 1:20)], env) for _ in 1:100
        ]
        # All gains should be 1+delta or 1-delta
        @test all(g -> g ≈ 0.5 || g ≈ 1.5, gains)
    end

    @testset "Regime gain symmetry: g(x_i, x_j) == g(x_j, x_i)" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env = test_env(types)
        rng = StableRNG(55)
        @test all(1:50) do _
            i, j = rand(rng, 1:20), rand(rng, 1:20)
            regime_gain(types[i], types[j], env) ≈ regime_gain(types[j], types[i], env)
        end
    end

    @testset "In-place regime_gain! matches allocating version" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env = test_env(types)
        Bx = zeros(d)
        rng = StableRNG(556)
        @test all(1:30) do _
            i, j = rand(rng, 1:20), rand(rng, 1:20)
            regime_gain(types[i], types[j], env) ≈ regime_gain!(Bx, types[i], types[j], env)
        end
    end

    @testset "At delta=0, gain is always 1.0" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env0 = test_env(types; delta=0.0)
        rng = StableRNG(77)
        @test all(1:50) do _
            i, j = rand(rng, 1:20), rand(rng, 1:20)
            regime_gain(types[i], types[j], env0) ≈ 1.0
        end
    end

    @testset "Mixing weight rho" begin
        types = test_agent_types(d, 20, StableRNG(10))
        # At rho=1: pure quality, no interaction
        env1 = test_env(types; rho=1.0)
        xi, xj = types[1], types[2]
        expected_q = 0.5 * (dot(xi, env1.c) + dot(xj, env1.c))
        @test match_signal(xi, xj, env1) ≈ expected_q

        # At rho=0: pure interaction, no quality
        env0 = test_env(types; rho=0.0)
        g = regime_gain(xi, xj, env0)
        expected_int = g * dot(xi, env0.A * xj)
        @test match_signal(xi, xj, env0) ≈ expected_int
    end

    @testset "In-place match_signal! matches allocating version" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env = test_env(types)
        Ax = zeros(d)
        Bx = zeros(d)
        rng = StableRNG(33)
        @test all(1:20) do _
            i, j = rand(rng, 1:20), rand(rng, 1:20)
            match_signal(types[i], types[j], env) ≈
            match_signal!(Ax, Bx, types[i], types[j], env)
        end
    end

    @testset "match_output includes Q_OFFSET and noise" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env = test_env(types)
        xi, xj = types[1], types[2]
        rng1 = StableRNG(99)
        q1 = match_output(xi, xj, env, rng1)
        rng2 = StableRNG(99)
        q2 = Q_OFFSET + match_signal(xi, xj, env) + env.sigma_eps * randn(rng2)
        @test q1 ≈ q2
    end

    @testset "In-place match_output! matches allocating version" begin
        types = test_agent_types(d, 20, StableRNG(10))
        env = test_env(types)
        xi, xj = types[3], types[9]
        Ax = zeros(d)
        Bx = zeros(d)
        rng1 = StableRNG(909)
        q1 = match_output(xi, xj, env, rng1)
        rng2 = StableRNG(909)
        q2 = match_output!(Ax, Bx, xi, xj, env, rng2)
        @test q1 ≈ q2
    end

    @testset "Calibration produces valid constants" begin
        types = test_agent_types(d, 100, StableRNG(10))
        env = test_env(types)
        p = default_params()
        cal = calibrate(env, types, p, StableRNG(55))
        surplus_scale = cal.q_cal - cal.r
        @test cal.q_cal > 0.0
        @test cal.r > 0.0
        @test cal.r ≈ R_BASE_FRAC * cal.q_cal
        @test cal.phi > 0.0
        @test cal.phi ≈ p.search_cost_rate * surplus_scale
        @test cal.c_s ≈ p.search_cost_rate * surplus_scale
        @test cal.c_s ≈ cal.phi
    end

    @testset "Shared-cost calibration responds to the common rate" begin
        types = test_agent_types(d, 100, StableRNG(10))
        env = test_env(types)

        cal_zero = calibrate(
            env, types, default_params(search_cost_rate=0.0), StableRNG(55)
        )
        @test cal_zero.phi ≈ 0.0 atol=1e-12
        @test cal_zero.c_s ≈ 0.0 atol=1e-12

        cal_high = calibrate(
            env, types, default_params(search_cost_rate=0.30), StableRNG(55)
        )
        @test cal_high.phi ≈ cal_high.c_s
        @test cal_high.phi > cal_zero.phi
    end

    @testset "Both regime-gain classes are populated" begin
        types = test_agent_types(d, 50, StableRNG(10))
        env = test_env(types; rho=0.0, sigma_eps=0.0)  # pure interaction, no noise
        # Find pairs in each regime
        high_gain = Float64[]
        low_gain = Float64[]
        for i in 1:50, j in (i + 1):50
            g = regime_gain(types[i], types[j], env)
            q = match_signal(types[i], types[j], env)
            if g > 1.0
                push!(high_gain, q)
            else
                push!(low_gain, q)
            end
        end
        # Both regimes should be populated
        @test length(high_gain) > 10
        @test length(low_gain) > 10
    end
end
