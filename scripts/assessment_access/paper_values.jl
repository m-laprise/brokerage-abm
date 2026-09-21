"""
Generate manuscript values from the retained assessment-access experiment.

Validate quoted means and paired contrasts against saved seed-level observations,
and validate trajectory late means against the same retained observations. Design
settings and seed counts come from the retained metadata. Original analysis and
simulation provenance remain separate from this manuscript-formatting step.

Usage: julia --project --threads=auto scripts/assessment_access/paper_values.jl
Output: output/assessment_access/paper_values.tex
"""

using Dates
using SHA: sha256
using Statistics: mean

include(joinpath(@__DIR__, "figure_2.jl"))

const VALUES_PATH = joinpath(ROOT, "output", "assessment_access", "paper_values.tex")

"""Recover one condition's seed-level late means, rejecting duplicates or omissions."""
function retained_seed_values(retained, mode, metric; rho=BASELINE_RHO, eta=BASELINE_ETA)
    rows = filter(retained["seed_rows"]) do row
        (String(row[1]), row[2], row[3], String(row[5])) == (mode, rho, eta, metric)
    end
    expected_n = only(
        Int(row[9]) for row in retained["summary_rows"] if
        (String(row[1]), row[2], row[3], String(row[4])) == (mode, rho, eta, metric)
    )
    values = Dict(Int(row[4]) => Float64(row[6]) for row in rows)
    length(values) == length(rows) == expected_n ||
        error("incomplete or duplicate seeds: $mode, $rho, $eta, $metric")
    all(isfinite, Base.values(values)) || error("nonfinite seed values")
    return values
end

"""Check a saved interval against its seed-level calculation."""
function check_interval(saved, expected)
    all(
        isapprox(a, b; atol=1e-12) for (a, b) in zip(
            (saved.estimate, saved.se, saved.lower, saved.upper, saved.n),
            (expected.mean, expected.se, expected.lower, expected.upper, expected.n),
        )
    ) || error("saved estimate or interval disagrees with seed-level data")
    return saved
end

"""Validate a condition mean and interval before formatting it for the manuscript."""
function checked_summary(retained, estimates, mode, metric; eta=BASELINE_ETA)
    seeds = retained_seed_values(retained, mode, metric; eta)
    values = [seeds[s] for s in sort!(collect(keys(seeds)))]
    expected = monte_carlo_interval(values; level=FIGURE_DESIGN.level)
    return check_interval(estimates.summaries[(mode, BASELINE_RHO, eta, metric)], expected)
end

"""Validate a comparison-minus-reference contrast using matched seed sets."""
function checked_pair(
    retained, saved, reference, comparison, metric; rho=BASELINE_RHO, eta=BASELINE_ETA
)
    reference_values = retained_seed_values(retained, reference, metric; rho, eta)
    comparison_values = retained_seed_values(retained, comparison, metric; rho, eta)
    Set(keys(reference_values)) == Set(keys(comparison_values)) ||
        error("paired seed sets differ")
    differences = [
        comparison_values[s] - reference_values[s] for
        s in sort!(collect(keys(reference_values)))
    ]
    expected = monte_carlo_interval(differences; level=FIGURE_DESIGN.level)
    return check_interval(saved, expected)
end

"""Validate a restricted-minus-full contrast with explicitly matched seed sets."""
function checked_effect(retained, estimates, service, metric; eta=BASELINE_ETA)
    return checked_pair(
        retained,
        service_effect(estimates.contrasts, service, BASELINE_RHO, eta, metric),
        "full",
        service.mode,
        metric;
        eta,
    )
end

"""Check quoted output, outsourcing, and structural comparisons across regimes."""
function check_service_comparisons(retained, estimates)
    cells = sort!(unique((key[2], key[3]) for key in keys(estimates.summaries)))
    rhos = unique(first.(cells))
    common_positive_rates = sort!(filter(unique(last.(cells))) do eta
        eta > 0 && all((rho, eta) in cells for rho in rhos)
    end)
    isempty(common_positive_rates) && error("no positive turnover rates shared across compositions")
    output_metric = "net_output_per_requested_position"
    output_differences = Dict{Tuple{Float64,Float64},NamedTuple}()
    access_outsourcing = Float64[]
    for (rho, eta) in cells
        output = checked_pair(
            retained,
            estimates.contrasts[("assessment_vs_access", rho, eta, output_metric)],
            "access_only",
            "assessment_only",
            output_metric;
            rho,
            eta,
        )
        output.lower > 0 || error("assessment-only output advantage changed: $rho, $eta")
        output_differences[(rho, eta)] = output
        assessment, access =
            map(("assessment", "access"), ("access_only", "assessment_only")) do label, mode
                checked_pair(
                    retained,
                    estimates.contrasts[(label, rho, eta, "outsourcing_rate")],
                    mode,
                    "full",
                    "outsourcing_rate";
                    rho,
                    eta,
                )
            end
        assessment.lower > 0 || error("assessment outsourcing gain changed: $rho, $eta")
        abs(access.estimate) < assessment.estimate ||
            error("access outsourcing effect is no longer smaller: $rho, $eta")
        push!(access_outsourcing, access.estimate)

        access_output = checked_pair(
            retained, estimates.contrasts[("access", rho, eta, output_metric)],
            "assessment_only", "full", output_metric; rho, eta,
        )
        if iszero(eta)
            access_output.upper < 0 || error("zero-turnover access cost changed: $rho")
        elseif eta in common_positive_rates
            access_output.lower > 0 || error("positive-turnover access benefit changed: $rho, $eta")
        end
        for metric in ("mean_degree", "betweenness")
            effect = checked_pair(
                retained, estimates.contrasts[("access", rho, eta, metric)],
                "assessment_only", "full", metric; rho, eta,
            )
            supported = metric == "mean_degree" ? effect.lower > 0 : effect.upper < 0
            supported || error("access structural effect changed: $rho, $eta, $metric")
        end
        if rho == BASELINE_RHO
            effect = checked_pair(
                retained, estimates.contrasts[("assessment", rho, eta, "betweenness")],
                "access_only", "full", "betweenness"; rho, eta,
            )
            effect.lower > 0 || error("assessment centrality gain changed: $rho, $eta")
        end
    end
    any(>(0), access_outsourcing) && any(<(0), access_outsourcing) ||
        error("access outsourcing effects are no longer mixed")
    return (;
        output=output_differences[(BASELINE_RHO, BASELINE_ETA)],
        regime_count=length(cells), common_positive_rates,
    )
end

"""Build only the definitions used by the experiment prose and caption."""
function manuscript_values(retained, estimates, data)
    check_trajectory_late_means(retained, data)
    comparisons = check_service_comparisons(retained, estimates)
    assessment_access_output = comparisons.output
    design = FIGURE_DESIGN
    rate_string(x) = iszero(x) ? "0" : string(x)
    fmt(x) = @sprintf("%.3f", x)
    assessment, access = SERVICES
    output_metric = "net_output_per_requested_position"
    output_effects = Dict(
        (service.mode, eta) =>
            checked_effect(retained, estimates, service, output_metric; eta) for
        service in SERVICES for eta in TURNOVER_RATES
    )
    loss_rates = filter(
        eta -> output_effects[(assessment.mode, eta)].upper < 0, TURNOVER_RATES
    )
    output_effects[(assessment.mode, first(TURNOVER_RATES))].lower > 0 ||
        error("zero-turnover assessment-only advantage changed")
    all(output_effects[(access.mode, eta)].upper < 0 for eta in TURNOVER_RATES) ||
        error("access-only net-output loss is not supported at every turnover rate")
    baseline_loss = -output_effects[(assessment.mode, design.eta)].estimate
    baseline_assessment_loss = -output_effects[(access.mode, design.eta)].estimate
    all(>(0), (baseline_loss, baseline_assessment_loss)) ||
        error("baseline output-loss sign changed")

    defs = Pair{String,String}[
        "aaRho" => rate_string(design.rho),
        "aaDelta" => rate_string(design.delta),
        "aaEta" => rate_string(design.eta),
        "aaEtaCount" => string(length(TURNOVER_RATES)),
        "aaRegimeN" => string(comparisons.regime_count),
        "aaCommonPositiveTurnoverPercents" => join(
            [@sprintf("%.0f", 100eta) * "\\%" for eta in comparisons.common_positive_rates],
            " or ",
        ),
        "aaAccessLossEtas" => join(rate_string.(loss_rates), ","),
        "aaLateWidth" => string(design.late_width),
        "aaLateStart" => string(design.horizon - design.late_width + 1),
        "aaHorizon" => string(design.horizon),
        "aaDegreeStart" => string(design.degree_start),
        "aaCentralityFirst" => string(first(data["periods"])),
        "aaCentralitySecond" => string(data["periods"][2]),
        "aaIntervalPercent" => @sprintf("%.0f", 100 * design.level),
        "aaBaselineSeeds" => string(length(data["seeds"])),
        "aaOtherSeeds" => string(
            only(unique(n for (eta, n) in design.seed_counts if eta != design.eta))
        ),
        "aaAccessOutputLoss" => fmt(baseline_loss),
        "aaAccessOutputLower" => fmt(-output_effects[(assessment.mode, design.eta)].upper),
        "aaAccessOutputUpper" => fmt(-output_effects[(assessment.mode, design.eta)].lower),
        "aaAssessmentOutputLoss" => fmt(baseline_assessment_loss),
        "aaAssessmentAccessOutputDifference" => fmt(assessment_access_output.estimate),
        "aaAssessmentAccessOutputLower" => fmt(assessment_access_output.lower),
        "aaAssessmentAccessOutputUpper" => fmt(assessment_access_output.upper),
    ]
    for (label, service) in
        (("Full", FULL_SERVICE), ("Assessment", assessment), ("Access", access))
        outsourcing = checked_summary(retained, estimates, service.mode, "outsourcing_rate")
        push!(
            defs,
            "aa$(label)OutPercent" => @sprintf("%.1f", 100 * outsourcing.estimate),
        )
        degree = checked_summary(retained, estimates, service.mode, "mean_degree")
        push!(defs, "aa$(label)Degree" => fmt(degree.estimate))
        if service.mode != access.mode
            centrality = checked_summary(retained, estimates, service.mode, "betweenness")
            push!(defs, "aa$(label)Centrality" => fmt(centrality.estimate))
        end
        if service.mode == access.mode
            centrality = checked_effect(retained, estimates, service, "betweenness")
            append!(
                defs,
                [
                    "aa$(label)CentralityDifference" =>
                        @sprintf("%+.3f", centrality.estimate),
                    "aa$(label)CentralityLower" => fmt(centrality.lower),
                    "aa$(label)CentralityUpper" => fmt(centrality.upper),
                ],
            )
        end
    end
    for (label, metric) in
        (("Broker", "broker_holdout_rank"), ("Principal", "agent_holdout_rank"))
        value = checked_summary(retained, estimates, access.mode, metric)
        push!(defs, "aaAccess$(label)Rank" => fmt(value.estimate))
    end
    length(unique(first.(defs))) == length(defs) || error("duplicate manuscript key")
    return defs
end

"""Write validated values with original-data and current-formatting provenance."""
function write_manuscript_values(; output_path=VALUES_PATH)
    provenance = manuscript_git_provenance(
        ROOT;
        sources=(
            @__FILE__,
            "scripts/assessment_access/figure_2.jl",
            "scripts/monte_carlo.jl",
            "scripts/figure_style.jl",
        ),
    )
    retained = JLD2.load(DATA_PATH)
    data = read_centrality()
    analysis_commit = validate_analysis_commit(
        provenance,
        retained["analysis_git_commit"];
        artifact="assessment-access retained analysis",
    )
    sweep_commit = validate_analysis_commit(
        provenance, data["sweep_git_commit"]; artifact="assessment-access simulation"
    )
    defs = manuscript_values(retained, read_estimates(), data)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(
            io, "% paper_values.tex: generated by scripts/assessment_access/paper_values.jl"
        )
        println(io, "% Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM"))
        println(io, "% Data analysis commit: ", analysis_commit)
        println(io, "% Retained analysis source clean: ", retained["analysis_source_clean"])
        println(io, "% Sweep commit: ", sweep_commit)
        println(io, "% Manifest: ", retained["manifest_hash"])
        println(io, "% Supplement manifest: ", retained["supplement_manifest_hash"])
        println(io, "% Manuscript formatting commit: ", provenance.commit)
        println(io, "% Manuscript formatting source clean: ", provenance.source_clean)
        write_source_provenance(io, provenance)
        println(io, "% Formatter SHA256: ", bytes2hex(sha256(read(@__FILE__))))
        println(io, "% Figure data SHA256: ", bytes2hex(sha256(read(DATA_PATH))))
        println(io, "% Trajectory data SHA256: ", bytes2hex(sha256(read(CENTRALITY_PATH))))
        println(
            io,
            "% Do not edit: all scientific values derive from retained data or metadata.",
        )
        for (key, value) in defs
            println(io, "\\pvDefine{", key, "}{", value, "}")
        end
    end
    println("Wrote $output_path ($(length(defs)) validated values)")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && write_manuscript_values()
