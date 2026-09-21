"""
Validate retained structural measurements and the manuscript figure's structural
panels against saved data and deterministic fixtures. Run separately from the model
suite. No simulations are run, and no production artifacts are written.

Usage: julia --project --threads=auto test/test_assessment_access_reporting.jl
"""

using Test
using BrokerageABM
using DataFrames: DataFrame
using Distributions: TDist, quantile
using Statistics: mean, std

include(joinpath(@__DIR__, "..", "scripts", "assessment_access", "figure_2.jl"))
include(joinpath(@__DIR__, "..", "scripts", "assessment_access", "centrality_data.jl"))

@testset "Centrality extraction and measurement cadence" begin
    recorded = [t < 20 ? 0.0 : (t ÷ 20) / 25 for t in 1:500]
    df = DataFrame(; period=1:500, betweenness=recorded)
    late = mean(recorded[481:500])
    @test measured_centrality(df, late) == collect(1:25) ./ 25
    @test_throws ErrorException measured_centrality(df[2:end, :], late)
    @test_throws ErrorException measured_centrality(df, late + 0.01)
    changed = deepcopy(df)
    changed.betweenness[21] = 0.1
    @test_throws ErrorException measured_centrality(changed, late)
    changed.betweenness[21] = NaN
    @test_throws ErrorException measured_centrality(changed, late)
end

@testset "Recorded degree extraction" begin
    df = DataFrame(; period=1:4, mean_degree=[0.0, 1.0, 2.0, 3.0])
    config = (; horizon=4, late_width=2, population=3)
    @test recorded_degree(df, 2.5; config...) == df.mean_degree
    @test_throws ErrorException recorded_degree(df[2:end, :], 2.5; config...)
    @test_throws ErrorException recorded_degree(df, 2.0; config...)
    invalid = deepcopy(df)
    invalid.mean_degree[1] = NaN
    @test_throws ErrorException recorded_degree(invalid, 2.5; config...)
    invalid.mean_degree[1] = 4.0
    @test_throws ErrorException recorded_degree(invalid, 2.5; config...)
end

@testset "Centrality figure integrity" begin
    data = read_centrality()
    @test validate_centrality(data) === data
    @test size(data["values"]["full"]) == (25, 50)
    for (key, value) in (
        ("rho", 1.0),
        ("periods", collect(0:20:480)),
        ("seeds", collect(1:49)),
        ("sweep_source_clean", false),
        ("network_measure_interval", 10),
    )
        invalid = copy(data)
        invalid[key] = value
        @test_throws ErrorException validate_centrality(invalid)
    end
    invalid = deepcopy(data)
    invalid["values"]["full"][1, 1] = NaN
    @test_throws ErrorException validate_centrality(invalid)
    invalid["values"]["full"][1, 1] = 1.1
    @test_throws ErrorException validate_centrality(invalid)
    invalid = deepcopy(data)
    delete!(invalid["values"], "full")
    @test_throws ErrorException validate_centrality(invalid)

    # Independent calculation for every plotted time-specific estimate and bound.
    matches = Bool[]
    for mode in ("full", "assessment_only", "access_only")
        series = centrality_series(data, mode)
        for t in 1:25
            values = data["values"][mode][t, :]
            estimate = mean(values)
            se = std(values) / sqrt(50)
            half = quantile(TDist(49), 0.975) * se
            push!(
                matches,
                all(
                    isapprox(a, b; atol=1e-12) for (a, b) in zip(
                        (
                            series.estimate[t],
                            series.se[t],
                            series.lower[t],
                            series.upper[t],
                            series.n[t],
                        ),
                        (estimate, se, estimate - half, estimate + half, 50),
                    )
                ),
            )
        end
    end
    @test length(matches) == 75
    @test all(matches)
    constant = deepcopy(data)
    constant["values"]["full"] .= 0.5
    series = centrality_series(constant, "full")
    @test all(iszero, series.se) &&
        all(==(0.5), vcat(series.estimate, series.lower, series.upper))

    retained = JLD2.load(DATA_PATH)
    baseline_rows = filter(retained["seed_rows"]) do row
        (row[2], row[3], String(row[5])) == (0.5, 0.02, "betweenness")
    end
    @test length(baseline_rows) == 150
    @test all(baseline_rows) do row
        values = data["values"][String(row[1])]
        seed = Int(row[4])
        isapprox(
            (19 * values[24, seed] + values[25, seed]) / 20, Float64(row[6]); atol=1e-12
        )
    end

    # Structural and outsourcing estimates do not depend on output accounting.
    estimates = (;
        summaries=interval_index(retained["summary_rows"]),
        contrasts=interval_index(retained["contrast_rows"]),
    )
    index = estimates.contrasts
    cases = [
        (s, eta, "outsourcing_rate", 1.0) for s in SERVICES for
        eta in TURNOVER_RATES
    ]
    append!(cases, [(s, 0.02, "betweenness", 1.0) for s in SERVICES])
    late_matches = Bool[]
    for (service, eta, metric, scale) in cases
        per_mode = Dict(
            mode => Dict(
                Int(r[4]) => Float64(r[6]) for r in retained["seed_rows"] if
                (String(r[1]), r[2], r[3], String(r[5])) == (mode, 0.5, eta, metric)
            ) for mode in ("full", service.mode)
        )
        seeds = sort!(collect(keys(per_mode["full"])))
        differences = [
            scale * (per_mode[service.mode][s] - per_mode["full"][s]) for s in seeds
        ]
        expected = monte_carlo_interval(differences)
        shown = service_effect(index, service, 0.5, eta, metric; scale)
        push!(
            late_matches,
            all(
                isapprox(a, b; atol=1e-12) for (a, b) in zip(
                    (shown.estimate, shown.se, shown.lower, shown.upper, shown.n),
                    (
                        expected.mean,
                        expected.se,
                        expected.lower,
                        expected.upper,
                        expected.n,
                    ),
                )
            ),
        )
    end
    @test length(cases) == 12
    @test all(late_matches)

    # Check every degree mean and interval independently from the cached seed records.
    degree_matches = Bool[]
    n_baseline = length(data["seeds"])
    for service in (FULL_SERVICE, SERVICES...)
        series = degree_series(data, service.mode)
        for t in eachindex(data["degree_periods"])
            values = data["degree_values"][service.mode][t, :]
            estimate = mean(values)
            se = std(values) / sqrt(n_baseline)
            half = quantile(TDist(n_baseline - 1), 0.975) * se
            push!(
                degree_matches,
                all(
                    isapprox(a, b; atol=1e-12) for (a, b) in zip(
                        (
                            series.estimate[t],
                            series.se[t],
                            series.lower[t],
                            series.upper[t],
                            series.n[t],
                        ),
                        (estimate, se, estimate - half, estimate + half, n_baseline),
                    )
                ),
            )
        end
    end
    @test length(degree_matches) == length(data["degree_periods"]) * 3 &&
        all(degree_matches)
    degree_rows = filter(retained["seed_rows"]) do row
        (row[2], row[3], String(row[5])) == (0.5, 0.02, "mean_degree")
    end
    @test length(degree_rows) == n_baseline * 3
    @test all(degree_rows) do row
        values = data["degree_values"][String(row[1])]
        seed_index = only(findall(==(Int(row[4])), data["seeds"]))
        isapprox(
            mean(values[(end - retained["late_width"] + 1):end, seed_index]),
            Float64(row[6]);
            atol=1e-12,
        )
    end
    constant_degree = deepcopy(data)
    constant_degree["degree_values"]["full"] .= 0.0
    degree = degree_series(constant_degree, "full")
    @test all(iszero, vcat(degree.estimate, degree.se, degree.lower, degree.upper))
    for key in ("degree_periods", "degree_values")
        invalid = deepcopy(data)
        if key == "degree_periods"
            pop!(invalid[key])
        else
            delete!(invalid[key], "full")
        end
        @test_throws ErrorException validate_centrality(invalid)
    end
    invalid = deepcopy(data)
    invalid["degree_values"]["full"][1, 1] = NaN
    @test_throws ErrorException validate_centrality(invalid)

    fixture = ("full", 0.5, 0.02, "outsourcing_rate", 1.0, 0.0, 1.0, 1.0, n_baseline)
    @test_throws ErrorException interval_index([fixture, fixture])
    for invalid in (
        Base.setindex(fixture, NaN, 5),
        Base.setindex(fixture, -1.0, 6),
        Base.setindex(fixture, 2.0, 7),
        Base.setindex(fixture, 1, 9),
    )
        @test_throws ErrorException interval_index([invalid])
    end
    # Test the retained structural panels, not retired layouts.
    fig = Figure(; size=(1000, 500))
    structure = argument_structure_panel!(fig[1, 1], estimates)
    trajectory = argument_trajectory_panel!(fig[1, 2], data)
    @test structure.xlabel[] == "Change in mean principal degree"
    @test trajectory.ylabel[] == "Broker betweenness centrality"
    @test trajectory.yticks[] == 0:0.2:1
    @test all(SERVICES) do service
        late = service_effect(index, service, 0.5, 0.02, "betweenness")
        final_only = mean(
            data["values"][service.mode][end, :] .- data["values"]["full"][end, :]
        )
        !isapprox(late.estimate, final_only; atol=1e-6)
    end
    @test_throws ArgumentError service_effect(
        index, SERVICES[1], 0.5, 0.02, "outsourcing_rate"; scale=-1
    )
    @test_throws ErrorException service_effect(
        index, SERVICES[1], 0.5, 0.025, "outsourcing_rate"
    )
    value = service_effect(index, SERVICES[1], 0.5, 0.02, "outsourcing_rate")
    @test_throws ErrorException effect_point!(nothing, 1, value, SERVICES[1], (0.0, 0.001))
    mktempdir() do temp
        path = joinpath(temp, "invalid.jld2")
        invalid = copy(data)
        invalid["manifest_hash"] = "wrong"
        JLD2.save(path, invalid)
        @test_throws ErrorException read_centrality(path)
        invalid = copy(data)
        invalid["retained_analysis_git_commit"] = "wrong"
        JLD2.save(path, invalid)
        @test_throws ErrorException read_centrality(path)
        invalid_estimates = copy(retained)
        invalid_estimates["analysis_source_clean"] = false
        JLD2.save(path, invalid_estimates)
        @test_throws ErrorException read_estimates(path)
    end
end
