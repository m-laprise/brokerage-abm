"""
Check reconstructed review data and output panels without publication writes.

Usage: julia --project --threads=auto test/test_net_output_reporting.jl
"""

using Test
using DataFrames: DataFrame
using Statistics: mean, std
using Distributions: TDist, quantile
using Printf: @sprintf

module NetOutputReview
include(joinpath(@__DIR__, "..", "scripts", "paper", "net_output_review.jl"))
include(joinpath(@__DIR__, "..", "scripts", "paper", "net_output_review_figures.jl"))
end

@testset "Information-source manuscript and figure" begin
    root = normpath(joinpath(@__DIR__, ".."))
    path = joinpath(root, "output", "main", "net_output_review.jld2")
    data = NetOutputReview.checked_net_output_review(path)
    channels = NetOutputReview.checked_channel_review(
        joinpath(root, "output", "main", "net_output_channels.jld2"), path, data
    )
    fd = NetOutputReview.channel_review_figure_data(data, channels)
    reporting = NetOutputReview.InformationSources
    defs = Dict(reporting.manuscript_values(fd))
    expected = Dict(
        "infoRegimeN" => "31", "infoComplementarityN" => "30",
        "infoPairBrokerRank" => "0.84", "infoPairPrincipalRank" => "0.47",
        "infoPairGap" => "0.37", "infoSizeBrokerRank" => "0.76",
        "infoSizePrincipalRank" => "0.72", "infoSizeGap" => "0.04",
        "infoSizeBrokerLossLower" => "0.08", "infoSizeBrokerLossUpper" => "0.10",
        "infoSizeGapLossLower" => "0.32", "infoSizeGapLossUpper" => "0.35",
        "infoSingleBrokerRank" => "0.44", "infoSingleGap" => "-0.25",
        "infoSingleGapLower" => "-0.30", "infoSingleGapUpper" => "-0.20",
        "infoSingleNegativeN" => "21", "infoAdditiveBrokerRank" => "0.47",
        "infoAdditiveGap" => "-0.18", "infoAdditiveGapLoss" => "0.55",
        "infoAdditiveGapLossLower" => "0.51", "infoAdditiveGapLossUpper" => "0.60",
        "infoAdditiveGapDecreaseCIN" => "29", "infoPairBrokerSharePercent" => "96",
        "infoSizeBrokerSharePercent" => "81", "infoSingleBrokerSharePercent" => "55",
        "infoAdditiveBrokerSharePercent" => "61", "infoSizeOutputPercent" => "97",
        "infoSingleOutputPercent" => "90", "infoAdditiveOutputPercent" => "90",
    )
    @test all(defs[key] == value for (key, value) in expected)
    @test fd["meta"]["review_only"] && !fd["meta"]["analysis_source_clean"]
    source = read(joinpath(root, "paper", "section_source.tex"), String)
    captions = read(joinpath(root, "paper", "captions.tex"), String)
    refs = Set(m[1] for m in eachmatch(r"\\pv\{(info\w+)\}", source * captions))
    @test issubset(refs, keys(defs))
    baseline = reporting.baseline_condition(fd)
    level = fd["meta"]["interval_level"]
    checks = [
        begin
            differences = baseline[field]["pair"] .- baseline[field][model]
            half = quantile(TDist(length(differences) - 1), (1 + level) / 2) *
                std(differences) / sqrt(length(differences))
            lower, upper = mean(differences) - half, mean(differences) + half
            defs["info$(label)$(suffix)Lower"] == @sprintf("%.2f", lower) &&
                defs["info$(label)$(suffix)Upper"] == @sprintf("%.2f", upper)
        end for (model, label) in zip(reporting.MODEL_KEYS[2:end], reporting.MODEL_LABELS[2:end])
        for (field, suffix) in (("broker_ranks", "BrokerLoss"), ("rank_gaps", "GapLoss"))
    ]
    @test all(checks)
    reordered = deepcopy(fd)
    for c in reordered["conditions"]
        reverse!(c["seeds"])
        for field in ("broker_ranks", "principal_ranks", "rank_gaps", "channel_values", "net_output")
            foreach(reverse!, values(c[field]))
        end
    end
    @test Dict(reporting.manuscript_values(reordered)) == defs
    corrupt = deepcopy(fd)
    first(corrupt["conditions"])["rank_gaps"]["pair"][1] += 0.1
    @test_throws ErrorException reporting.validate_data(corrupt)
    corrupt = deepcopy(fd)
    pop!(first(corrupt["conditions"])["broker_ranks"]["pair"])
    @test_throws ErrorException reporting.validate_data(corrupt)
    corrupt = deepcopy(fd)
    first(corrupt["conditions"])["net_output"]["pair"][1] += 1
    @test_throws ErrorException reporting.validate_data(corrupt)
    corrupt = deepcopy(fd)
    corrupt["meta"]["n_conditions"] += 1
    @test_throws ErrorException reporting.validate_data(corrupt)
    @test_throws ArgumentError reporting.render(fd; difficulties=(9.0,))
    @test_throws ArgumentError reporting.late_channel_values(DataFrame(), 0, 1)
    @test_throws ArgumentError reporting.late_channel_values(DataFrame(), 1, 1)
    # Pure rendering from the same retained observations must preserve the
    # reviewed asset exactly. No publication analysis is overwritten.
    mktempdir() do temp
        rendered = reporting.render(fd; output_path=joinpath(temp, "information.png"))
        approved = joinpath(root, "output", "main", "figures", "information_sources_net_output_channels.png")
        @test read(rendered) == read(approved)
    end
end

module ServicePanels
include(joinpath(@__DIR__, "..", "scripts", "assessment_access", "paper_values.jl"))
end

module SupplementPanels
include(joinpath(@__DIR__, "..", "scripts", "assessment_access", "supplement_figure.jl"))
end

@testset "Independent trajectory analysis provenance" begin
    retained = ServicePanels.JLD2.load(ServicePanels.DATA_PATH)
    trajectory = ServicePanels.read_centrality()
    commits = split(readchomp(`git -C $(ServicePanels.ROOT) log -5 --format=%H`), '\n')
    other_commit = first(filter(!=(trajectory["retained_analysis_git_commit"]), commits))
    retained["analysis_git_commit"] = other_commit
    @test other_commit != trajectory["retained_analysis_git_commit"]
    @test ServicePanels.figure_design(retained, trajectory).rho == ServicePanels.BASELINE_RHO
    mktempdir() do temp
        path = joinpath(temp, "retained.jld2")
        ServicePanels.JLD2.save(path, retained)
        @test ServicePanels.read_centrality(; retained_path=path)["seeds"] == trajectory["seeds"]
        corrupt = deepcopy(retained)
        row = first(filter(corrupt["seed_rows"]) do row
            String(row[1]) == "full" && row[2] == trajectory["rho"] &&
                row[3] == trajectory["eta"] && string(row[5]) == "betweenness"
        end)
        row[6] += 0.1
        ServicePanels.JLD2.save(path, corrupt)
        @test_throws ErrorException ServicePanels.read_centrality(; retained_path=path)
        corrupt = deepcopy(retained)
        corrupt["manifest_hash"] = "different manifest"
        ServicePanels.JLD2.save(path, corrupt)
        @test_throws ErrorException ServicePanels.read_centrality(; retained_path=path)
        corrupt = deepcopy(retained)
        corrupt["analysis_git_commit"] = repeat("f", 40)
        ServicePanels.JLD2.save(path, corrupt)
        @test_throws ErrorException ServicePanels.read_centrality(; retained_path=path)
    end
end

@testset "Channel output decomposition" begin
    frame = DataFrame(
        total_demand=[8, 8, 4],
        outsourced_slots=[4, 4, 0],
        access_count=[1, 1, 0],
        assessment_count=[1, 1, 0],
        n_self_matches=[2, 2, 0],
        q_self_mean=[30.0, 3.0, NaN],
        n_broker_matches=[1, 1, 0],
        q_broker_mean=[100.0, 10.0, NaN],
    )
    original = copy(frame)
    calibration = (; N=10, c_s=0.5, phi=1.0)
    values = NetOutputReview.channel_late_values(frame, calibration, 2)
    @test all(isapprox.(collect(values), [0.1, 0.4, 0.5]))
    @test values.self + values.broker ≈ values.total
    @test isequal(frame, original)
    @test_throws ArgumentError NetOutputReview.channel_late_values(frame, calibration, 4)
    invalid = copy(frame)
    invalid.outsourced_slots[end] = 5
    @test_throws ErrorException NetOutputReview.channel_late_values(invalid, calibration, 2)
    invalid = copy(frame)
    invalid.q_self_mean[2] = NaN
    @test_throws ArgumentError NetOutputReview.channel_late_values(invalid, calibration, 2)
end


@testset "Ensemble means of seed-level net-output shares" begin
    values = [(; self=1.0, broker=3.0, total=4.0), (; self=9.0, broker=1.0, total=10.0)]
    shares = NetOutputReview.channel_output_shares(values, 10)
    # Give each seed's percentage equal weight, independently of its total output.
    @test all(isapprox.(
        [shares.self.mean, shares.broker.mean, shares.total], [0.575, 0.425, 70.0]
    ))
    @test shares.self.mean + shares.broker.mean ≈ 1
    @test !isapprox(
        shares.broker.mean, mean(v.broker for v in values) / mean(v.total for v in values)
    )
    seed_shares = [0.75, 0.1]
    se = std(seed_shares) / sqrt(length(seed_shares))
    half = quantile(TDist(length(seed_shares) - 1), 0.975) * se
    @test shares.broker.n == shares.self.n == 2
    @test shares.broker.se ≈ se
    @test shares.broker.lower ≈ mean(seed_shares) - half
    @test shares.broker.upper ≈ mean(seed_shares) + half
    @test shares.self.lower ≈ 1 - shares.broker.upper
    @test shares.self.upper ≈ 1 - shares.broker.lower
    larger_population = NetOutputReview.channel_output_shares(values, 20)
    @test larger_population.broker == shares.broker && larger_population.self == shares.self
    @test larger_population.total ≈ 2 * shares.total
    rescaled = [map(x -> 2 * x, v) for v in values]
    @test NetOutputReview.channel_output_shares(rescaled, 5) == shares
    singleton = NetOutputReview.channel_output_shares(
        [(; self=0.0, broker=2.0, total=2.0)], 10
    )
    @test singleton.broker.mean == 1 && isnan(singleton.broker.lower)
    endpoints = NetOutputReview.channel_output_shares(
        [(; self=0.0, broker=1.0, total=1.0), (; self=0.0, broker=3.0, total=3.0)], 10
    )
    @test endpoints.self.lower == endpoints.self.upper == 0 &&
        endpoints.broker.lower == endpoints.broker.upper == 1
    @test_throws ArgumentError NetOutputReview.channel_output_shares(values, 0)
    @test_throws ArgumentError NetOutputReview.channel_output_shares(values, 10; level=1.0)
    @test_throws ArgumentError NetOutputReview.channel_output_shares(NamedTuple[], 10)
    @test_throws ArgumentError NetOutputReview.channel_output_shares(
        [(; self=0.0, broker=0.0, total=0.0); values], 10
    )
    @test_throws ArgumentError NetOutputReview.channel_output_shares(
        [(; self=-1.0, broker=-1.0, total=-2.0); values], 10
    )
    @test_throws ArgumentError NetOutputReview.channel_output_shares(
        [(; self=1.0, broker=2.0, total=4.0)], 10
    )
    @test_throws ArgumentError NetOutputReview.channel_output_shares(
        [(; self=-1.0, broker=2.0, total=1.0); values], 10
    )
    @test_throws ArgumentError NetOutputReview.channel_output_shares(
        [(; self=NaN, broker=2.0, total=2.0)], 10
    )
end

@testset "Reconstructed net-output reporting" begin
    path = joinpath(@__DIR__, "..", "output", "main", "net_output_review.jld2")
    data = NetOutputReview.checked_net_output_review(path)
    channel_path = joinpath(@__DIR__, "..", "output", "main", "net_output_channels.jld2")
    channels = NetOutputReview.checked_channel_review(channel_path, path, data)
    @test Set(keys(channels["ridge"])) == Set(keys(data["ridge"]))
    channel_checks = [
        begin
            retained = data["ridge"][model][rel]
            level = data["meta"]["interval_level"]
            share = NetOutputReview.channel_output_shares(
                condition.values, retained["N"]; level
            )
            ratios = [v.broker / v.total for v in condition.values]
            half = quantile(TDist(length(ratios) - 1), (1 + level) / 2) *
                std(ratios) / sqrt(length(ratios))
            late = retained["late"]
            total = mean(
                late["gross_match_output"] .- late["total_search_cost"] .-
                late["total_broker_fees"]
            )
            share.self.mean + share.broker.mean ≈ 1 &&
                share.broker.mean ≈ mean(ratios) &&
                share.broker.lower ≈ mean(ratios) - half &&
                share.broker.upper ≈ mean(ratios) + half &&
                share.self.lower ≈ 1 - share.broker.upper &&
                share.self.upper ≈ 1 - share.broker.lower &&
                share.total ≈ total &&
                length(condition.values) ==
                length(condition.seeds) ==
                length(condition.broker_rank) ==
                length(condition.principal_rank) == share.broker.n == share.self.n
        end for (model, conditions) in channels["ridge"] for (rel, condition) in conditions
    ]
    @test all(channel_checks)
    models = [(; key=model) for model in sort!(collect(keys(channels["ridge"])))]
    ranking = NetOutputReview.channel_review_ranking_data(data, channels, models)
    selected = filter(c -> c["rho"] == 1 || c["delta"] == 0.5, ranking.conditions)
    @test length(selected) == length(unique(c["rho"] for c in ranking.conditions))
    @test Set(c["rho"] for c in selected) == Set(c["rho"] for c in ranking.conditions)
    @test all(c -> c["delta"] == 0.5, filter(c -> c["rho"] != 1, selected))
    rank_interval_checks = [
        begin
            retained = channels["ridge"][model.key][condition["result_reldir"]]
            differences = retained.broker_rank .- retained.principal_rank
            level = data["meta"]["interval_level"]
            interval = NetOutputReview.monte_carlo_interval(
                condition["rank_gaps"][model.key]; level
            )
            se = std(differences) / sqrt(length(differences))
            half = quantile(TDist(length(differences) - 1), (1 + level) / 2) * se
            interval.n == length(retained.seeds) &&
                interval.mean ≈ mean(differences) && interval.se ≈ se &&
                interval.lower ≈ mean(differences) - half &&
                interval.upper ≈ mean(differences) + half &&
                -1 < interval.lower <= interval.upper < 1.25
        end for model in models for condition in selected
    ]
    @test all(rank_interval_checks)
    @test channels["meta"]["review_only"] && !channels["meta"]["analysis_source_clean"]
    @test data["meta"]["review_only"]
    @test !data["meta"]["analysis_source_clean"]
    @test all(
        hashes -> length(hashes) == 4, values(data["meta"]["calibration_source_hashes"])
    )
    conditions = [
        collect(values(data["access"]));
        [c for model in values(data["ridge"]) for c in values(model)]
    ]
    late_width = data["meta"]["late_width"]
    matches = map(conditions) do c
        v = c["late"]
        reconstructed =
            (v["gross_match_output"] .- v["total_search_cost"] .- v["total_broker_fees"]) ./ c["N"]
        series_means = vec(mean(c["series"][(end - late_width + 1):end, :]; dims=1))
        length(c["seeds"]) == length(unique(c["seeds"])) == length(reconstructed) &&
            all(isfinite, reconstructed) &&
            isapprox(reconstructed, v["net_output_per_principal"]) &&
            isapprox(series_means, v["net_output_per_principal"])
    end
    @test all(matches)
    contrasts = Dict{Tuple{String,Float64,Float64,String},NamedTuple}()
    summaries = empty(contrasts)
    level = data["meta"]["interval_level"]
    metric = "net_output_per_principal"
    summary(v) = (; estimate=v.mean, se=v.se, lower=v.lower, upper=v.upper, n=v.n)
    for ((mode, rho, eta), c) in data["access"]
        v = NetOutputReview.monte_carlo_interval(NetOutputReview.net_values(c); level)
        summaries[(mode, rho, eta, metric)] = summary(v)
    end
    interval_checks = Bool[]
    for (rho, eta) in unique((key[2], key[3]) for key in keys(data["access"]))
        for (label, reference, comparison) in (
            ("assessment", "access_only", "full"),
            ("access", "assessment_only", "full"),
            ("assessment_vs_access", "access_only", "assessment_only"),
        )
            a, b = data["access"][(reference, rho, eta)],
            data["access"][(comparison, rho, eta)]
            diff = NetOutputReview.net_values(b) .- NetOutputReview.net_values(a)
            expected_mean, expected_se = mean(diff), std(diff) / sqrt(length(diff))
            half = quantile(TDist(length(diff) - 1), (1 + level) / 2) * expected_se
            v = NetOutputReview.review_service_effect(data, reference, comparison, rho, eta)
            push!(
                interval_checks,
                a["seeds"] == b["seeds"] &&
                    isapprox(v.mean, expected_mean) &&
                    isapprox(v.se, expected_se) &&
                    isapprox(v.lower, expected_mean - half) &&
                    isapprox(v.upper, expected_mean + half),
            )
            contrasts[(label, rho, eta, metric)] = summary(v)
        end
    end
    @test all(interval_checks)
    # Validate manuscript values in memory. Review data are never promoted to
    # clean publication inputs or written over the retained analysis.
    retained = ServicePanels.JLD2.load(ServicePanels.DATA_PATH)
    replaced_metrics = Set([string.(NetOutputReview.NetOutputData.OLD_METRICS); metric])
    filter!(row -> !(String(row[5]) in replaced_metrics), retained["seed_rows"])
    filter!(row -> !(String(row[4]) in replaced_metrics), retained["summary_rows"])
    filter!(row -> !(String(row[4]) in replaced_metrics), retained["contrast_rows"])
    for ((mode, rho, eta), c) in data["access"]
        append!(retained["seed_rows"], [
            Any[mode, rho, eta, seed, metric, value] for
            (seed, value) in zip(c["seeds"], NetOutputReview.net_values(c))
        ])
    end
    for (rows, index) in (
        (retained["summary_rows"], summaries), (retained["contrast_rows"], contrasts)
    )
        append!(rows, [
            Any[key..., v.estimate, v.se, v.lower, v.upper, v.n] for (key, v) in index
        ])
    end
    manuscript_estimates = (;
        summaries=ServicePanels.interval_index(retained["summary_rows"]),
        contrasts=ServicePanels.interval_index(retained["contrast_rows"]),
    )
    centrality = ServicePanels.read_centrality()
    defs = Dict(ServicePanels.manuscript_values(retained, manuscript_estimates, centrality))
    expected_values = Dict(
        "aaAccessOutputLoss" => "1.02",
        "aaAccessOutputLower" => "0.96",
        "aaAccessOutputUpper" => "1.08",
        "aaAssessmentOutputLoss" => "1.05",
        "aaAssessmentOutputLower" => "1.00",
        "aaAssessmentOutputUpper" => "1.10",
        "aaStableAssessmentGain" => "0.26",
        "aaStableAssessmentLower" => "0.22",
        "aaStableAssessmentUpper" => "0.30",
        "aaHighTurnoverPercent" => "3",
        "aaHighTurnoverAccessGain" => "0.22",
        "aaHighTurnoverAccessLower" => "0.13",
        "aaHighTurnoverAccessUpper" => "0.32",
    )
    @test all(defs[key] == value for (key, value) in expected_values)
    reverse!(retained["seed_rows"])
    @test Dict(ServicePanels.manuscript_values(retained, manuscript_estimates, centrality)) == defs
    @test !any(startswith(key, "aaAssessmentAccessOutput") for key in keys(defs))
    source = read(joinpath(@__DIR__, "..", "paper", "section_source.tex"), String)
    captions = read(joinpath(@__DIR__, "..", "paper", "captions.tex"), String)
    references = Set(m[1] for m in eachmatch(r"\\pv\{(aa\w+)\}", source * captions))
    @test issubset(references, keys(defs))
    corrupted = deepcopy(retained)
    row = only(filter(
        r -> (r[1], r[2], r[3], r[4]) == ("assessment_vs_access", 0.0, 0.0, metric),
        corrupted["contrast_rows"],
    ))
    row[7] -= 0.1
    @test_throws ErrorException ServicePanels.manuscript_values(
        corrupted,
        (; summaries=manuscript_estimates.summaries,
            contrasts=ServicePanels.interval_index(corrupted["contrast_rows"])),
        centrality,
    )
    incomplete = deepcopy(retained)
    deleteat!(incomplete["seed_rows"], findfirst(row -> row[5] == metric, incomplete["seed_rows"]))
    @test_throws ErrorException ServicePanels.manuscript_values(
        incomplete, manuscript_estimates, centrality
    )
    for bounds in ((-0.01, 0.02), (-2.1, 1.8), (0.0, 0.0))
        display = NetOutputReview.interval_axis(bounds)
        @test display.limits[1] <=
            min(0, first(bounds)) <=
            max(0, last(bounds)) <=
            display.limits[2]
    end
    @test_throws ErrorException NetOutputReview.interval_axis((NaN, 1.0))
    # Render the approved figures from reconstructed output and unchanged structure.
    ServicePanels.validate_argument_estimates(retained, manuscript_estimates, centrality)
    fig = ServicePanels.make_figure(manuscript_estimates, centrality)
    axes = filter(block -> block isa ServicePanels.Axis, fig.content)
    @test length(axes) == 6
    @test count(ax -> occursin("per principal", ax.ylabel[]), axes) == 1
    @test count(ax -> ax.ylabel[] == "Broker betweenness centrality", axes) == 1
    supplementary = SupplementPanels.complementarity_estimates(retained)
    @test length(supplementary.shown) ==
        length(supplementary.panels) * length(supplementary.rhos) * length(supplementary.etas)
    supplement_figure = SupplementPanels.make_complementarity_figure(retained)
    supplement_axes = filter(block -> block isa SupplementPanels.Axis, supplement_figure.content)
    @test length(supplement_axes) == 2
    @test supplement_axes[1].ylabel[] == "Change in net output\nper principal"
    @test supplement_axes[1].limits[][2] == supplement_axes[2].limits[][2]
    @test all(values(supplementary.shown)) do value
        lower, upper = supplement_axes[1].limits[][2]
        lower <= value.lower <= value.upper <= upper
    end
    invalid = deepcopy(retained)
    row = first(filter(row -> String(row[4]) == metric, invalid["contrast_rows"]))
    row[5] += 1
    @test_throws ErrorException SupplementPanels.complementarity_estimates(invalid)
    # Inspection copies are temporary, never publication outputs.
    mktempdir() do temp
        path = joinpath(temp, "assessment_access.png")
        ServicePanels.save(path, fig)
        @test filesize(path) > 0
        supplement_path = joinpath(temp, "complementarity_contributions.png")
        SupplementPanels.save(supplement_path, supplement_figure)
        @test filesize(supplement_path) > 0
    end
end
