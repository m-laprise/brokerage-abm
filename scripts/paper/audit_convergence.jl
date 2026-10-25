"""Audit reported window summaries, network snapshots, and paired comparisons.

Run without arguments from retained inputs. One-time preparation of the main
NN/Ridge seed summaries: --prepare <saved-sweep-summary.jld2>. No simulations run.
"""
module ConvergenceAudit

using JLD2, Statistics, Printf, SHA
include(joinpath(@__DIR__, "..", "reporting_settings.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output/main/convergence")
const SEEDS = joinpath(OUT, "seed_values.jld2")
const MAIN = "output/main/figure_data.jld2"
const RIDGE = "output/ridge/paired/figure_data.jld2"
const SNAPSHOTS = "output/main/network_snapshots.jld2"
const MAIN_METRICS = (
    "outsourcing_rate",
    "access_fraction",
    "agent_holdout_rank",
    "broker_holdout_rank",
    "rank_gap",
)
const INPUTS = (
    SEEDS,
    MAIN,
    RIDGE,
    SNAPSHOTS,
    "output/main/access_windows.jld2",
    "output/ridge/paired/access_windows.jld2",
    "output/main/centrality_data.jld2",
    "output/ridge/ablations/figure_data.jld2",
    "output/assessment_access/figure_data.jld2",
    "output/assessment_access/centrality_trajectories.jld2",
    "output/supplement/structural_comparisons.jld2",
    "output/supplement/structural_figure_data.jld2",
    "output/reporting/settings.toml",
)
path(p) = joinpath(ROOT, p)
hashfile(p) = bytes2hex(sha256(read(p)))
function provenance()
    manuscript_git_provenance(
        ROOT;
        sources=(
            @__FILE__,
            "scripts/analysis_settings.jl",
            "scripts/reporting_settings.jl",
            "scripts/monte_carlo.jl",
        ),
    )
end

"""Retain only the main outcomes missing complete seed vectors in figure inputs."""
function prepare(summary_path)
    raw, snapshots = load(summary_path), load(path(SNAPSHOTS))
    hashfile(summary_path) == snapshots["metadata"]["validation_summary_sha256"] ||
        error("summary differs from the saved network extraction")
    index = Dict((r["dataset"], r["rel"]) => r for r in snapshots["conditions"])
    sources = Dict(
        name => snapshots["metadata"]["sources"][name] for name in ("nn", "ridge_pair")
    )
    rows = Dict{String,Any}[]
    for c in raw["conditions"]
        c["dataset"] in keys(sources) || continue
        r = index[(c["dataset"], c["rel"])]
        c["seeds"] == r["seeds"] && c["source_sha256"] == r["source_sha256"] ||
            error("saved runs differ from the snapshot extraction")
        push!(
            rows,
            Dict(
                "dataset" => c["dataset"],
                "rel" => c["rel"],
                "seeds" => c["seeds"],
                "source_sha256" => c["source_sha256"],
                "values" => Dict(m => c["metrics"][m] for m in MAIN_METRICS),
            ),
        )
    end
    for (name, file) in (("nn", MAIN), ("ridge_pair", RIDGE))
        fd = load(path(file), "figdata")
        sources[name]["manifest_hash"] == fd["meta"]["manifest_hash"] ||
            error("manifest differs")
        retained = Dict(r["rel"] => r for r in rows if r["dataset"] == name)
        Set(keys(retained)) == Set(c["rel"] for c in fd["regime_cells"]) ||
            error("regimes differ")
        for c in fd["regime_cells"]
            r = retained[c["rel"]]
            r["seeds"] == c["seeds"] || error("seeds differ")
            for (metric, key) in (("access_fraction", "access"), ("rank_gap", "rankgap"))
                isapprox(r["values"][metric], c["seed_values"][key]; atol=1e-12) ||
                    error("retained $metric differs")
            end
        end
    end
    metadata = Dict(
        "sources" => sources,
        "summary_sha256" => hashfile(summary_path),
        "analysis_commit" => raw["meta"]["analysis_commit"],
        "extractor_sha256" => raw["meta"]["script_sha256"],
        "late_window" => [
            first(first(raw["conditions"])["late_periods"]),
            last(first(raw["conditions"])["late_periods"]),
        ],
    )
    mkpath(OUT)
    jldsave(SEEDS; metadata, conditions=rows)
    p = provenance()
    record_publication_stage(
        p;
        stage="scripts/paper/audit_convergence.jl --prepare",
        inputs=(SNAPSHOTS, MAIN, RIDGE),
        outputs=(SEEDS,),
        analysis_commits=Dict("data analysis commit" => metadata["analysis_commit"]),
        settings=Dict("source_summary_sha256" => metadata["summary_sha256"]),
    )
    println("Retained main outcome seeds: ", SEEDS)
end

"""Align independent seeds before differencing, including unequal seed sets."""
function paired_values(reference, comparison)
    length(reference.seeds) == length(reference.values) ||
        error("incomplete reference values")
    length(comparison.seeds) == length(comparison.values) ||
        error("incomplete comparison values")
    length(unique(reference.seeds)) == length(reference.seeds) ||
        error("duplicate reference seed")
    length(unique(comparison.seeds)) == length(comparison.seeds) ||
        error("duplicate comparison seed")
    seeds = sort(intersect(reference.seeds, comparison.seeds))
    length(seeds) > 1 || error("insufficient common seeds")
    a = Dict(zip(reference.seeds, reference.values))
    b = Dict(zip(comparison.seeds, comparison.values))
    return seeds, [b[s] - a[s] for s in seeds]
end

"""Check the reported ratio of ensemble means without changing it to a mean of ratios."""
function audit_ratio(reference, comparison, nested_n)
    seeds, _ = paired_values(reference, comparison)
    a, b = Dict(zip(reference.seeds, reference.values)),
    Dict(zip(comparison.seeds, comparison.values))
    1 < nested_n < length(seeds) || error("invalid nested ratio count")
    ratio(ss) = mean(b[s] for s in ss) / mean(a[s] for s in ss)
    all(a[s] > 0 && isfinite(b[s]) for s in seeds) || error("invalid output ratio")
    full, nested = ratio(seeds), ratio(seeds[1:nested_n])
    return (;
        dataset=comparison.dataset,
        outcome="net_output_ratio_to_reference",
        result_reldir=comparison.rel,
        n_seeds=length(seeds),
        nested_n,
        estimate=full,
        nested_estimate=nested,
        nested_change=abs(full - nested),
    )
end

"""Compare full and nested seed means in original units and, when defined, relative units."""
function audit_estimate(record, nested_n; level=AnalysisSettings.INTERVAL_LEVEL)
    seeds, values = record.seeds, record.values
    length(seeds) == length(values) == length(unique(seeds)) ||
        error("invalid seed coverage")
    all(isfinite, values) || error("nonfinite seed estimate")
    1 < nested_n < length(seeds) || error("invalid nested seed count")
    order = sortperm(seeds)
    full = monte_carlo_interval(values; level)
    nested = monte_carlo_interval(values[order[1:nested_n]]; level)
    half_width = (full.upper - full.lower) / 2
    change = abs(full.mean - nested.mean)
    relative = (full.lower > 0 || full.upper < 0) && !iszero(full.mean)
    direction(ci) = if ci.lower > 0
        1
    elseif ci.upper < 0
        -1
    else
        0
    end
    return (;
        dataset=record.dataset,
        kind=record.kind,
        outcome=record.outcome,
        window=record.window,
        result_reldir=record.rel,
        reference=record.reference,
        comparison=record.comparison,
        n_seeds=full.n,
        nested_n,
        estimate=full.mean,
        ci_lower=full.lower,
        ci_upper=full.upper,
        ci_half_width=half_width,
        nested_estimate=nested.mean,
        nested_ci_lower=nested.lower,
        nested_ci_upper=nested.upper,
        nested_change=change,
        relative_precision_defined=relative,
        relative_ci_half_width=relative ? half_width / abs(full.mean) : NaN,
        relative_nested_change=relative ? change / abs(full.mean) : NaN,
        mean_sign_changed=full.mean * nested.mean < 0,
        interval_direction=direction(full),
        nested_interval_direction=direction(nested),
    )
end

function add_level!(records, dataset, rel, outcome, seeds, values; window="late")
    key = (dataset, rel, outcome, window)
    r = (;
        dataset,
        rel,
        outcome,
        window,
        kind="level",
        reference="",
        comparison="",
        seeds=Int.(seeds),
        values=Float64.(values),
    )
    if haskey(records, key)
        old = records[key]
        old.seeds == r.seeds && isapprox(old.values, r.values; atol=1e-12) ||
            error("inconsistent retained values: $key")
    else
        records[key] = r
    end
    return records[key]
end

function add_contrast!(contrasts, label, a, b; rel=b.rel, window=b.window)
    a.outcome == b.outcome || error("outcome mismatch")
    seeds, values = paired_values(a, b)
    push!(
        contrasts,
        (;
            dataset=label,
            rel,
            outcome=b.outcome,
            window,
            kind="difference",
            reference="$(a.dataset):$(a.rel):$(a.window)",
            comparison="$(b.dataset):$(b.rel):$(b.window)",
            seeds,
            values,
        ),
    )
end

"""Collect exactly the retained summaries used in the simulation results."""
function reported_estimates()
    records, contrasts = Dict{NTuple{4,String},NamedTuple}(), NamedTuple[]
    main = load(path(MAIN), "figdata")
    settings = ReportingSettings.validate_figure(
        main, ReportingSettings.retained(path(MAIN), main)
    )
    early, late = Tuple(settings["early_window"]), Tuple(settings["late_window"])
    input = load(SEEDS)
    input["metadata"]["late_window"] == settings["late_window"] ||
        error("late window differs")
    for r in input["conditions"], (metric, values) in r["values"]
        add_level!(records, r["dataset"], r["rel"], metric, r["seeds"], values)
    end
    getlevel(dataset, rel, metric; window="late") = records[(dataset, rel, metric, window)]
    baseline = settings["baseline_reldir"]
    for (name, file, accessfile) in (
        ("nn", MAIN, "output/main/access_windows.jld2"),
        ("ridge_pair", RIDGE, "output/ridge/paired/access_windows.jld2"),
    )
        fd = load(path(file), "figdata")
        rs = ReportingSettings.validate_figure(
            fd, ReportingSettings.retained(path(file), fd)
        )
        for key in ("early_window", "late_window", "interval_level")
            rs[key] == settings[key] || error("NN/Ridge reporting settings differ")
        end
        input["metadata"]["sources"][name]["manifest_hash"] ==
        fd["meta"]["manifest_hash"] || error("main seed-summary manifest differs")
        access = load(path(accessfile))
        access["metadata"]["manifest_hash"] == fd["meta"]["manifest_hash"] ||
            error("access manifest differs")
        for c in access["regimes"]
            fractions = c["access_count"] ./ (c["access_count"] .+ c["assessment_count"])
            for (window, bounds) in (("early", early), ("late", late))
                ix = findall(t -> bounds[1] <= t <= bounds[2], access["periods"])
                values = [
                    mean(finite_seed_values(fractions[ix, j])) for j in axes(fractions, 2)
                ]
                add_level!(
                    records, name, c["rel"], "access_fraction", c["seeds"], values; window
                )
                add_level!(
                    records,
                    name,
                    c["rel"],
                    "assessment_fraction",
                    c["seeds"],
                    1 .- values;
                    window,
                )
            end
            for m in ("access_fraction", "assessment_fraction")
                add_contrast!(
                    contrasts,
                    name * "_time",
                    getlevel(name, c["rel"], m; window="early"),
                    getlevel(name, c["rel"], m);
                    window="late_minus_early",
                )
            end
        end
        for (window, bounds) in (("early", early), ("late", late)),
            (outcome, key) in
            (("outsourcing_rate", "outsourcing"), ("betweenness", "betweenness"))

            ix = AnalysisSettings.reporting_indices(
                fd["period"],
                outcome,
                bounds;
                network_interval=rs["network_measure_interval"],
            )
            values = vec(mean(fd["series_seed_values"][key][ix, :]; dims=1))
            add_level!(
                records, name, baseline, outcome, fd["baseline_seeds"], values; window
            )
        end
        for m in ("outsourcing_rate", "betweenness")
            add_contrast!(
                contrasts,
                name * "_time",
                getlevel(name, baseline, m; window="early"),
                getlevel(name, baseline, m);
                window="late_minus_early",
            )
        end
        for c in fd["regime_cells"],
            (metric, key) in (("access_fraction", "access"), ("rank_gap", "rankgap"))

            add_level!(records, name, c["rel"], metric, c["seeds"], c["seed_values"][key])
        end
    end
    for c in main["regime_cells"], metric in MAIN_METRICS
        add_contrast!(
            contrasts,
            "ridge_minus_nn",
            getlevel("nn", c["rel"], metric),
            getlevel("ridge_pair", c["rel"], metric),
        )
    end
    central = load(path("output/main/centrality_data.jld2"), "data")
    structural = load(path("output/supplement/structural_figure_data.jld2"), "figdata")
    for c in central.nn
        for metric in (
            "betweenness",
            "access_fraction",
            "outsourcing_rate",
            "mean_degree",
            "max_degree",
        )
            add_level!(records, "nn", c.rel, metric, c.seeds, c.values[metric])
        end
        s = only(r for r in structural["regime_cells"] if r["rel"] == c.rel)
        for (metric, key) in (("constraint", "constraint"), ("effective_size", "effsize"))
            add_level!(records, "nn", c.rel, metric, s["seeds"], s["seed_values"][key])
        end
    end
    # The composition and turnover comparisons stated in IV.E.
    rhos, etas = sort(unique(c.rho for c in central.nn)),
    sort(unique(c.eta for c in central.nn))
    mixed = maximum(filter(<(1), rhos))
    point(rho, eta) = only(c for c in central.nn if c.rho == rho && c.eta == eta)
    for eta in etas, m in ("betweenness", "access_fraction", "constraint", "effective_size")
        a, b = point(mixed, eta), point(first(rhos), eta)
        add_contrast!(
            contrasts, "nn_composition", getlevel("nn", a.rel, m), getlevel("nn", b.rel, m)
        )
    end
    base = first(central.ridge).cfg
    for m in
        ("betweenness", "mean_degree", "access_fraction", "constraint", "effective_size")
        a, b = point(base["rho"], first(etas)), point(base["rho"], last(etas))
        add_contrast!(
            contrasts, "nn_turnover", getlevel("nn", a.rel, m), getlevel("nn", b.rel, m)
        )
    end
    a, b = point(first(rhos), base["eta"]), point(last(rhos), base["eta"])
    for m in ("mean_degree", "max_degree")
        add_contrast!(
            contrasts, "nn_composition", getlevel("nn", a.rel, m), getlevel("nn", b.rel, m)
        )
    end
    variants = Dict(
        "pair" => "ridge_pair",
        "size_matched" => "ridge_size",
        "single_principal" => "ridge_one",
        "additive" => "ridge_additive",
    )
    ablations = load(path("output/ridge/ablations/figure_data.jld2"), "figdata")
    ablations["meta"]["late_width"] == length(late[1]:late[2]) ||
        error("ablation window differs")
    ablations["meta"]["interval_level"] == settings["interval_level"] ||
        error("ablation interval differs")
    ridge_grid = load(path(RIDGE), "figdata")["grid_cells"]
    reference_rel = Dict(
        c["result_reldir"] => only(
            unique(
                r["rel"] for r in ridge_grid if
                r["rho"] == c["rho"] && (c["rho"] == 1 || r["delta"] == c["delta"])
            ),
        ) for c in ablations["conditions"]
    )
    for c in ablations["conditions"], (model, name) in variants
        rel = name == "ridge_pair" ? reference_rel[c["result_reldir"]] : c["result_reldir"]
        for (field, metric) in (
            ("principal_ranks", "agent_holdout_rank"),
            ("broker_ranks", "broker_holdout_rank"),
            ("rank_gaps", "rank_gap"),
        )
            add_level!(records, name, rel, metric, c["seeds"], c[field][model])
        end
        channels = c["channel_values"][model]
        all(v -> v.total > 0 && isapprox(v.self + v.broker, v.total), channels) ||
            error("invalid channel totals")
        for channel in (:self, :broker)
            values = [getproperty(v, channel) / v.total for v in channels]
            add_level!(
                records, name, rel, "$(channel)_net_output_share", c["seeds"], values
            )
        end
        if c["result_reldir"] == ablations["meta"]["baseline_reldir"]
            for (field, metric) in (
                (:self, "self_net_output_per_principal"),
                (:broker, "broker_net_output_per_principal"),
                (:total, "net_output_per_principal"),
            )
                add_level!(
                    records, name, rel, metric, c["seeds"], getproperty.(channels, field)
                )
            end
        end
    end
    learning_metrics = (
        "agent_holdout_rank",
        "broker_holdout_rank",
        "rank_gap",
        "self_net_output_share",
        "broker_net_output_share",
    )
    for c in ablations["conditions"],
        name in filter(!=("ridge_pair"), collect(Base.values(variants))),
        m in learning_metrics

        rel = c["result_reldir"]
        add_contrast!(
            contrasts,
            name * "_minus_pair",
            getlevel("ridge_pair", reference_rel[rel], m),
            getlevel(name, rel, m),
        )
    end
    services = load(path("output/assessment_access/figure_data.jld2"))
    services["late_width"] == length(late[1]:late[2]) &&
    services["interval_level"] == settings["interval_level"] ||
        error("service reporting settings differ")
    service_metrics = (
        "net_output_per_principal", "outsourcing_rate", "mean_degree", "betweenness"
    )
    grouped = Dict{Tuple{String,String,String},Vector{Tuple{Int,Float64}}}()
    service_rel(rho, eta) = "rho=$(rho)/eta=$(eta)"
    for r in services["seed_rows"]
        String(r[5]) in service_metrics || continue
        key = ("service_" * String(r[1]), service_rel(r[2], r[3]), String(r[5]))
        push!(get!(grouped, key, Tuple{Int,Float64}[]), (Int(r[4]), Float64(r[6])))
    end
    for ((name, rel, metric), observations) in grouped
        sort!(observations; by=first)
        add_level!(records, name, rel, metric, first.(observations), last.(observations))
    end
    supplemental = load(path("output/supplement/structural_comparisons.jld2"), "data")
    for r in supplemental.services,
        m in ("constraint", "effective_size", "mean_degree", "betweenness")

        add_level!(
            records, "service_" * r.mode, service_rel(r.rho, r.eta), m, r.seeds, r.values[m]
        )
    end
    for r in supplemental.ridge,
        m in (
            "constraint",
            "effective_size",
            "betweenness",
            "mean_degree",
            "outsourcing_rate",
            "access_fraction",
        )

        add_level!(records, variants[r.key], baseline, m, r.seeds, r.values[m])
    end
    for name in filter(!=("ridge_pair"), collect(Base.values(variants))),
        m in (
            "constraint",
            "effective_size",
            "betweenness",
            "mean_degree",
            "outsourcing_rate",
            "access_fraction",
            "self_net_output_per_principal",
            "broker_net_output_per_principal",
            "net_output_per_principal",
        )

        add_contrast!(
            contrasts,
            name * "_minus_pair",
            getlevel("ridge_pair", baseline, m),
            getlevel(name, baseline, m),
        )
    end
    for rel in unique(r.rel for r in Base.values(records) if r.dataset == "service_full"),
        m in (service_metrics..., "constraint", "effective_size")

        full = getlevel("service_full", rel, m)
        access = getlevel("service_access_only", rel, m)
        assessment = getlevel("service_assessment_only", rel, m)
        add_contrast!(contrasts, "adding_access", assessment, full)
        add_contrast!(contrasts, "adding_assessment", access, full)
        m == "net_output_per_principal" &&
            add_contrast!(contrasts, "assessment_minus_access", access, assessment)
    end
    trajectory = load(path("output/assessment_access/centrality_trajectories.jld2"))
    trajectory["manifest_hash"] == services["manifest_hash"] ||
        error("service trajectory manifest differs")
    rel = service_rel(trajectory["rho"], trajectory["eta"])
    for (mode, values) in trajectory["values"]
        name = "service_" * mode
        for (window, bounds) in (("early", early), ("late", late))
            ix = AnalysisSettings.reporting_indices(
                trajectory["periods"],
                "betweenness",
                bounds;
                network_interval=trajectory["network_measure_interval"],
            )
            add_level!(
                records,
                name,
                rel,
                "betweenness",
                trajectory["seeds"],
                vec(values[ix, :]);
                window,
            )
        end
        add_contrast!(
            contrasts,
            name * "_time",
            getlevel(name, rel, "betweenness"; window="early"),
            getlevel(name, rel, "betweenness");
            window="late_minus_early",
        )
    end
    return (;
        records=[collect(Base.values(records)); contrasts],
        settings,
        input_metadata=input["metadata"],
        ratios=[
            (
                getlevel("ridge_pair", baseline, "net_output_per_principal"),
                getlevel(name, baseline, "net_output_per_principal"),
            ) for name in sort(filter(!=("ridge_pair"), collect(Base.values(variants))))
        ],
    )
end

function write_table(file, rows)
    isempty(rows) && error("empty audit table")
    content =
        join(
            [
                join(string.(propertynames(first(rows))), '\t');
                [join(string.(Tuple(r)), '\t') for r in rows]
            ],
            '\n',
        ) * "\n"
    write_if_changed(joinpath(OUT, file), content)
end

function summarize(rows)
    groups = sort(unique((r.dataset, r.kind, r.window, r.outcome) for r in rows))
    finite_summary(values, f) =
        isempty(filter(isfinite, values)) ? NaN : f(filter(isfinite, values))
    return map(groups) do (dataset, kind, window, outcome)
        group = filter(
            r ->
                (r.dataset, r.kind, r.window, r.outcome) ==
                (dataset, kind, window, outcome),
            rows,
        )
        ci = getproperty.(group, :relative_ci_half_width)
        change = getproperty.(group, :relative_nested_change)
        (;
            dataset,
            kind,
            window,
            outcome,
            n_conditions=length(group),
            n_relative_precision=count(isfinite, ci),
            p50_relative_ci_half_width=finite_summary(ci, median),
            p90_relative_ci_half_width=finite_summary(ci, x -> quantile(x, 0.9)),
            max_relative_ci_half_width=finite_summary(ci, maximum),
            p50_relative_nested_change=finite_summary(change, median),
            max_relative_nested_change=finite_summary(change, maximum),
            mean_sign_changes=count(r -> r.mean_sign_changed, group),
            interval_direction_changes=count(
                r -> r.interval_direction != r.nested_interval_direction, group
            ),
        )
    end
end

function main()
    data = reported_estimates()
    counts = sort(unique(length(r.seeds) for r in data.records))
    nested = Dict(
        counts[1] => data.settings["nested_general"],
        counts[end] => data.settings["nested_baseline"],
    )
    length(counts) == 2 || error("check nested seed design for changed seed counts")
    rows = [
        audit_estimate(
            r, nested[length(r.seeds)]; level=Float64(data.settings["interval_level"])
        ) for r in data.records
    ]
    sort!(
        rows; by=r -> (r.dataset, r.kind, r.window, r.outcome, r.result_reldir, r.reference)
    )
    length(
        unique(
            (r.dataset, r.kind, r.window, r.outcome, r.result_reldir, r.reference) for
            r in rows
        ),
    ) == length(rows) || error("duplicate audit estimate")
    summaries = summarize(rows)
    mkpath(OUT)
    write_table("condition_audit.tsv", rows)
    write_table("outcome_summary.tsv", summaries)
    ratios = [audit_ratio(a, b, nested[length(a.seeds)]) for (a, b) in data.ratios]
    write_table("derived_audit.tsv", ratios)
    definitions = Dict("mcOutcomeCount" => string(length(unique(r.outcome for r in rows))))
    for (kind, prefix) in (("level", "mcLevel"), ("difference", "mcDifference"))
        subset = filter(r -> r.kind == kind, summaries)
        definitions[prefix * "Count"] = string(count(r -> r.kind == kind, rows))
        for (field, label) in (
            (:p50_relative_ci_half_width, "CiMedian"),
            (:p50_relative_nested_change, "NestedMedian"),
        )
            values = filter(isfinite, getproperty.(subset, field))
            definitions[prefix * label * "MinPercent"] = @sprintf(
                "%.1f", 100minimum(values)
            )
            definitions[prefix * label * "MaxPercent"] = @sprintf(
                "%.1f", 100maximum(values)
            )
        end
    end
    write_if_changed(
        joinpath(OUT, "values.tex"),
        join(
            [
                "% Generated by scripts/paper/audit_convergence.jl.";
                [
                    "\\pvDefine{$key}{$(definitions[key])}" for
                    key in sort(collect(keys(definitions)))
                ]
            ],
            '\n',
        ) * "\n",
    )
    write_if_changed(
        joinpath(OUT, "summary.txt"),
        join(
            [
                "scope=reported simulation window summaries, network snapshots, and paired comparisons",
                "excluded=pointwise trajectory stability and initialization-only DGP diagnostics",
                "experiments=NN, Ridge, learning restrictions, service provision",
                "normalization=absolute full-sample estimate; omitted when its interval contains zero",
                "summary_groups=experiment, outcome, window, and level or difference",
                "nested_comparisons=" *
                join(["first $(nested[n]) of $n seeds" for n in counts], "; "),
                "interval_level=$(data.settings["interval_level"])",
                "early_window=$(join(data.settings["early_window"], ':'))",
                "late_window=$(join(data.settings["late_window"], ':'))",
                ["$key=$(definitions[key])" for key in sort(collect(keys(definitions)))]...,
            ],
            '\n',
        ) * "\n",
    )
    p = provenance()
    commits = Dict(
        "main seed-summary analysis commit" => data.input_metadata["analysis_commit"]
    )
    for file in INPUTS
        record = publication_record(ROOT, file)
        record === nothing && continue
        validate_publication_record(ROOT, record; sources=false)
        for (label, commit) in record["analysis_commits"]
            commits["$(relpath(path(file), ROOT)): $label"] = validate_analysis_commit(
                p, commit
            )
        end
    end
    for (file, field) in (
        (MAIN, "figdata"),
        (RIDGE, "figdata"),
        ("output/ridge/ablations/figure_data.jld2", "figdata"),
    )
        fd = load(path(file), field)
        commits[file * ": data analysis commit"] = validate_analysis_commit(
            p, fd["meta"]["analysis_git_commit"]
        )
    end
    record_publication_stage(
        p;
        stage="scripts/paper/audit_convergence.jl",
        inputs=INPUTS,
        outputs=Tuple(
            joinpath(OUT, f) for f in (
                "condition_audit.tsv",
                "outcome_summary.tsv",
                "derived_audit.tsv",
                "values.tex",
                "summary.txt",
            )
        ),
        analysis_commits=commits,
        settings=Dict("nested_seed_counts" => [nested[n] for n in counts]),
    )
    println("Audited $(length(rows)) estimates across $(length(summaries)) summaries.")
    println(read(joinpath(OUT, "summary.txt"), String))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    if isempty(ARGS)
        main()
    elseif length(ARGS) == 2 && first(ARGS) == "--prepare"
        prepare(ARGS[2])
    else
        error("usage: audit_convergence.jl [--prepare SUMMARY]")
    end
end

end
