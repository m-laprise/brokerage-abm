using Test
using BrokerageABM
using JLD2: load
using TOML
using Statistics: mean

isdefined(@__MODULE__, :AnalysisSettings) ||
    include(joinpath(@__DIR__, "..", "scripts", "analysis_settings.jl"))
isdefined(@__MODULE__, :ReportingSettings) ||
    include(joinpath(@__DIR__, "..", "scripts", "reporting_settings.jl"))

@testset "Reporting settings" begin
    @testset "windows retain explicit analysis choices" begin
        @test AnalysisSettings.windows(500) == (early=(51, 70), late=(481, 500))
        @test AnalysisSettings.windows(13; early=(2, 4), late_width=3).late == (11, 13)
        @test_throws ErrorException AnalysisSettings.windows(60)
        @test_throws ErrorException AnalysisSettings.windows(
            13; early=(2, 12), late_width=3
        )
    end

    @testset "network snapshots exclude held measurements" begin
        periods = collect(1:500)
        held = Float64.(fld.(periods, 20))
        early, late = AnalysisSettings.windows(last(periods))
        select(metric, window) = AnalysisSettings.reporting_indices(
            periods, metric, window; network_interval=20
        )
        @test all(select(metric, early) == [60] && select(metric, late) == [500]
            for metric in AnalysisSettings.NETWORK_MEASURES)
        @test mean(held[select(:betweenness, early)]) == 3.0
        @test mean(held[select(:betweenness, late)]) == 25.0
        @test mean(held[481:500]) != mean(held[select(:betweenness, late)])
        @test select(:mean_degree, late) == collect(481:500)
        @test select(:outsourcing_rate, early) == collect(51:70)
        @test AnalysisSettings.network_snapshot_period((10, 13), 3) == 12
        @test_throws ErrorException AnalysisSettings.network_snapshot_period((61, 70), 20)
        @test_throws ErrorException AnalysisSettings.network_snapshot_period(early, 0)
        @test_throws ErrorException AnalysisSettings.reporting_indices(
            1:59, :constraint, early; network_interval=20
        )
        @test_throws ErrorException AnalysisSettings.reporting_indices(
            [60, 60], :effective_size, early; network_interval=20
        )
        root = normpath(joinpath(@__DIR__, ".."))
        settings = ReportingSettings.retained(joinpath(root, "output/main/figure_data.jld2"),
            load(joinpath(root, "output/main/figure_data.jld2"), "figdata"))
        sweep = joinpath(root, "runs/sweeps", "2026-09-05_adeaa06_nn")
        @test ReportingSettings.network_interval(sweep) == settings["network_measure_interval"]
    end

    @testset "baseline resolution does not depend on paths" begin
        baseline = Dict(:rho => 0.3, :eta => 0.07, :T => 31)
        first_condition = Dict(
            :resolved_params => Dict(:rho => 0.3, :eta => 0.07),
            :result_reldir => "arbitrary/baseline",
        )
        data = Dict(
            "spec" => Dict(:baseline => baseline),
            "meta" => Dict(:T => 31),
            "conditions" => [
                first_condition,
                Dict(
                    :resolved_params => Dict(:rho => 0.5, :eta => 0.07),
                    :result_reldir => "oat/rho=0.5",
                ),
            ],
        )
        @test ReportingSettings.baseline_condition(data) === first_condition
        @test ReportingSettings.baseline_condition(data; overrides=Dict(:rho => 0.5))[:result_reldir] ==
            "oat/rho=0.5"
        push!(data["conditions"], deepcopy(first_condition))
        @test_throws ErrorException ReportingSettings.baseline_condition(data)
        empty!(data["conditions"])
        @test_throws ErrorException ReportingSettings.baseline_condition(data)
    end

    @testset "retained inputs remain complete and independently sourced" begin
        root = normpath(joinpath(@__DIR__, ".."))
        paths = (
            "output/main/figure_data.jld2",
            "output/ridge/paired/figure_data.jld2",
            "output/supplement/structural_figure_data.jld2",
        )
        settings = map(paths) do path
            data = load(joinpath(root, path), "figdata")
            record = ReportingSettings.retained(joinpath(root, path), data)
            @test ReportingSettings.validate_figure(data, record) === record
            wrong = deepcopy(data)
            pop!(first(wrong["regime_cells"])["seeds"])
            @test_throws ErrorException ReportingSettings.validate_figure(wrong, record)
            embedded = deepcopy(data)
            embedded["meta"]["reporting_settings"] = record
            @test ReportingSettings.retained("unused", embedded) === record
            @test isempty(ReportingSettings.retained_dependencies(embedded))
            record
        end
        @test settings[1]["simulation_commit"] != settings[2]["simulation_commit"]
        @test settings[1]["baseline"]["eta_lr_agent"] == 0.003
        @test settings[2]["baseline"]["ridge_lambda_agent"] == 0.001
        mktempdir() do dir
            records = TOML.parsefile(ReportingSettings.RETAINED_PATH)
            baseline = settings[1]["baseline_reldir"]
            plan = records["seed_plan"]
            plan["sets"][plan["conditions"][baseline]][1] = -1
            metadata = joinpath(dir, "settings.toml")
            open(io -> TOML.print(io, records), metadata, "w")
            path = joinpath(root, first(paths))
            data = load(path, "figdata")
            wrong = ReportingSettings.retained(path, data; metadata_path=metadata)
            @test_throws ErrorException ReportingSettings.validate_figure(data, wrong)
        end
        mktempdir() do dir
            path = joinpath(dir, "changed.jld2")
            write(path, "changed data")
            meta = joinpath(dir, "settings.toml")
            entry = Dict("sha256" => "incorrect", "manifest_hash" => "unused")
            open(meta, "w") do io
                TOML.print(io, Dict("artifacts" => Dict(relpath(path, root) => entry)))
            end
            @test_throws ErrorException ReportingSettings.retained(
                path, Dict(); metadata_path=meta
            )
        end
    end

    @testset "fixed settings record the adopted model" begin
        settings = BrokerageABM.fixed_model_settings()
        @test settings.Q == BrokerageABM.Q_OFFSET
        @test settings.calibration_pairs == 10_000
        @test settings.curve_frequencies == collect(1:5)
        @test settings.match_calibration_seeds == collect(10_001:10_050)
        @test (settings.adam_beta1, settings.adam_beta2, settings.adam_epsilon) ==
            (0.9, 0.999, 1e-8)
        settings.curve_frequencies[1] = -1
        @test first(BrokerageABM.fixed_model_settings().curve_frequencies) == 1
    end
end
