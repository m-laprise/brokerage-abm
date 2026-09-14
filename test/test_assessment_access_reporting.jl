"""
Validate both main-figure versions, supplementary contrasts, and manuscript values
against retained data and deterministic fixtures. Run separately from the model
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

    # The twelve difference points preserve the original paired estimator.
    estimates = read_estimates()
    index = estimates.contrasts
    cases = [
        (s, eta, "net_output_per_requested_position", 1.0) for s in SERVICES for
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

    # Verify outsourcing against the manuscript and its generated values.
    outsourcing_matches = Bool[]
    manuscript = read(joinpath(ROOT, "paper", "section_source.tex"), String)
    definitions = read(
        joinpath(ROOT, "output", "assessment_access", "paper_values.tex"), String
    )
    for (label, service) in
        zip(("Full", "Assessment", "Access"), (FULL_SERVICE, SERVICES...))
        rows = filter(retained["seed_rows"]) do row
            (String(row[1]), row[2], row[3], String(row[5])) ==
            (service.mode, 0.5, 0.02, "outsourcing_rate")
        end
        estimate = mean(Float64(row[6]) for row in rows)
        summary = estimates.summaries[(service.mode, 0.5, 0.02, "outsourcing_rate")]
        key = "aa$(label)OutPercent"
        definition = "\\pvDefine{$key}{" * @sprintf("%.1f", 100 * estimate) * "}"
        push!(
            outsourcing_matches,
            length(rows) == n_baseline &&
                isapprox(summary.estimate, estimate; atol=1e-12) &&
                occursin(definition, definitions) &&
                occursin("\\pv{$key}\\%", manuscript),
        )
    end
    @test length(outsourcing_matches) == 3 && all(outsourcing_matches)
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
    fig = make_figure(estimates, data)
    axes = filter(block -> block isa Axis, fig.content)
    @test [axis.title[] for axis in axes] == [
        "A. Net output to principals",
        "",
        "B. Principal connectivity over time",
        "C. Late-mean centrality difference",
        "D. Broker centrality over time",
    ]
    @test axes[1].xticks[] == ([0.0], ["0"])
    @test axes[2].xticks[] ==
        (collect(TURNOVER_RATES[2:end]), string.(collect(TURNOVER_RATES[2:end])))
    @test axes[1].limits[][2] == axes[2].limits[][2]
    @test !axes[2].yticklabelsvisible[]
    @test axes[5].limits[][2] == (0.0, 1.0)
    @test axes[5].yticks[] == 0:0.2:1
    @test axes[3].limits[][2][1] == 0.0
    boundary_data = deepcopy(data)
    boundary_data["values"]["full"] .= 0.0
    boundary_data["values"]["assessment_only"] .= 1.0
    @test centrality_panel!(Figure()[1, 1], boundary_data) isa Axis
    @test axes[4].ylabel[] == "Betweenness difference\nfrom full service"
    @test axes[3].ylabel[] == "Mean principal degree"
    @test axes[1].ylabel[] == "Difference from full service\nper requested position"
    @test axes[2].xlabel[] == "Turnover rate (η)"
    @test !any(block -> block isa Label, fig.content)
    @test SERVICES[2].detail == "Principals' assessments"
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

# Keep the alternative renderer's names separate from the current figure's API.
module OverviewFigures
include(joinpath(@__DIR__, "..", "scripts", "assessment_access", "main_figure.jl"))
end

@testset "Alternative figure and supplementary contrasts" begin
    retained = JLD2.load(DATA_PATH)
    design = OverviewFigures.overview_design(retained)
    @test (design.rho, design.delta, design.eta, design.horizon) == (
        FIGURE_DESIGN.rho, FIGURE_DESIGN.delta, FIGURE_DESIGN.eta, FIGURE_DESIGN.horizon
    )
    @test design.late_start == FIGURE_DESIGN.horizon - FIGURE_DESIGN.late_width + 1
    @test design.interval_percent == @sprintf("%.0f", 100 * FIGURE_DESIGN.level)
    @test design.baseline_seeds == FIGURE_DESIGN.seed_counts[BASELINE_ETA]
    @test all(
        n == design.other_seeds for (eta, n) in FIGURE_DESIGN.seed_counts if
        eta != BASELINE_ETA
    )

    estimates = read_estimates()
    matches = Bool[]
    for panel in OverviewFigures.PANEL_SPECS, mode in OverviewFigures.MODES,
        eta in OverviewFigures.ETAS
        panel.relative && mode == "full" && continue
        per_mode = Dict(
            service => Dict(
                Int(row[4]) => Float64(row[6]) for row in retained["seed_rows"] if
                (String(row[1]), row[2], row[3], String(row[5])) ==
                (service, BASELINE_RHO, eta, panel.metric)
            ) for service in unique(("full", mode))
        )
        seeds = sort!(collect(keys(per_mode[mode])))
        values = [
            panel.relative ? per_mode[mode][s] - per_mode["full"][s] : per_mode[mode][s]
            for s in seeds
        ]
        expected = monte_carlo_interval(values; level=FIGURE_DESIGN.level)
        shown = OverviewFigures.displayed_interval(
            estimates.summaries, estimates.contrasts, mode, eta, panel
        )
        push!(
            matches,
            Set(seeds) == Set(keys(per_mode["full"])) && all(
                isapprox(a, b; atol=1e-12) for (a, b) in zip(
                    (shown.estimate, shown.se, shown.lower, shown.upper, shown.n),
                    (expected.mean, expected.se, expected.lower, expected.upper, expected.n),
                )
            ),
        )
    end
    @test length(matches) == sum(
        (panel.relative ? length(SERVICES) : length(SERVICES) + 1) * length(TURNOVER_RATES)
        for panel in OverviewFigures.PANEL_SPECS
    )
    @test all(matches)

    # This selector independently checks every supplementary contrast against seeds.
    selected = OverviewFigures.complementarity_estimates(retained)
    @test length(selected.shown) ==
        length(selected.panels) * length(selected.rhos) * length(selected.etas)
    for key in ("interval_level", "late_width")
        invalid = copy(retained)
        invalid[key] = 0
        @test_throws ErrorException OverviewFigures.overview_design(invalid)
    end
    invalid = deepcopy(retained)
    index = only(findall(invalid["contrast_rows"]) do row
        (String(row[1]), row[2], row[3], String(row[4])) ==
        ("assessment", first(selected.rhos), first(selected.etas),
         "net_output_per_requested_position")
    end)
    invalid["contrast_rows"][index][5] += 1
    @test_throws ErrorException OverviewFigures.complementarity_estimates(invalid)
end
