"""
Render the structural-results figure and manuscript values from retained seed values.

Prepare its compact input from the saved full-sweep summary:
    julia --project --threads=auto scripts/paper/centrality_and_assessment.jl \
        --prepare <spectrum-results.jld2>
Render again without sweep access:
    julia --project --threads=auto scripts/paper/centrality_and_assessment.jl

Outputs: output/main/centrality_data.jld2 (preparation only),
output/main/centrality_values.tex, and output/main/figures/centrality_and_assessment.png.
"""
module CentralityAndAssessment

using JLD2
using SHA: sha256
using Printf: @printf, @sprintf
include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = joinpath(ROOT, "output", "main", "centrality_data.jld2")
const OUTPUT = joinpath(ROOT, "output", "main", "figures", "centrality_and_assessment.png")
const VALUES_OUTPUT = joinpath(ROOT, "output", "main", "centrality_values.tex")
const MAIN_INPUT = joinpath(ROOT, "output", "main", "figure_data.jld2")
const RIDGE_INPUT = joinpath(ROOT, "output", "ridge", "ablations", "figure_data.jld2")
const VARIANTS = (
    (dataset="ridge_pair", key="pair", label="Reference\nbroker"),
    (dataset="ridge_size", key="size_matched", label="Fewer\nobservations"),
    (dataset="ridge_additive", key="additive", label="No interaction\nterm"),
    (dataset="ridge_one", key="single_principal", label="One party\nrecorded"),
)
const RIDGE_METRICS = (
    (key="outsourcing_rate", label="Outsourcing rate", color=PUB_BROKER, marker=:circle),
    (key="access_fraction", label="Access fraction", color=PUB_ACCESS, marker=:utriangle),
    (
        key="betweenness",
        label="Broker betweenness centrality",
        color=PUB_CENTRALITY,
        marker=:rect,
    ),
)
const DEGREE_METRICS = (
    (key="max_degree", label="Maximum"), (key="mean_degree", label="Mean")
)

file_hash(path) = bytes2hex(sha256(read(path)))

"""Retain the displayed estimates with their seeds, source hashes, and extractor."""
function prepare(path)
    raw = load(path)
    main = load(MAIN_INPUT)["figdata"]
    ridge = load(RIDGE_INPUT)["figdata"]
    provenance = manuscript_git_provenance(ROOT; sources=(@__FILE__,))
    validate_analysis_commit(provenance, raw["meta"]["analysis_commit"])
    for source in values(raw["meta"]["sources"])
        validate_analysis_commit(provenance, source["simulation_commit"])
    end
    main["meta"]["analysis_source_clean"] === true || error("Unidentified main analysis")
    ridge["meta"]["analysis_source_clean"] === true || error("Unidentified Ridge analysis")
    raw["meta"]["sources"]["nn"]["manifest_hash"] == main["meta"]["manifest_hash"] ||
        error("Main sweep manifest mismatch")
    by_source = Dict((c["dataset"], c["rel"]) => c for c in raw["conditions"])
    length(by_source) == length(raw["conditions"]) || error("Duplicate input conditions")
    baseline = only(
        filter(
            c -> c["result_reldir"] == ridge["meta"]["baseline_reldir"], ridge["conditions"]
        ),
    )
    ridge_rows = map(VARIANTS) do variant
        c = by_source[(variant.dataset, baseline["result_reldir"])]
        c["seeds"] == baseline["seeds"] || error("Ridge seeds differ")
        c["cfg"]["rho"] == baseline["rho"] && c["cfg"]["delta"] == baseline["delta"] ||
            error("Ridge baseline differs")
        c["metrics"]["broker_holdout_rank"] ≈ baseline["broker_ranks"][variant.key] ||
            error("Ridge broker rankings differ from publication input")
        c["metrics"]["agent_holdout_rank"] ≈ baseline["principal_ranks"][variant.key] ||
            error("Ridge principal rankings differ from publication input")
        (;
            dataset=variant.dataset,
            key=variant.key,
            seeds=c["seeds"],
            values=Dict(metric.key => c["metrics"][metric.key] for metric in RIDGE_METRICS),
            cfg=c["cfg"],
            periods=c["late_periods"],
            source_sha256=c["source_sha256"],
        )
    end
    nn_rows = map(main["rho_eta_cells"]) do cell
        c = by_source[("nn", cell["rel"])]
        c["seeds"] == cell["seeds"] || error("NN seeds differ")
        for (metric, retained) in (("betweenness", "betw"), ("access_fraction", "access"))
            c["metrics"][metric] ≈ cell["seed_values"][retained] ||
                error("NN $metric differs from publication input")
        end
        c["cfg"]["rho"] == cell["rho"] && c["cfg"]["eta"] == cell["eta"] ||
            error("NN grid coordinates differ")
        metrics = Dict(
            k => c["metrics"][k] for k in (
                "betweenness",
                "access_fraction",
                "outsourcing_rate",
                "mean_degree",
                "median_degree",
                "max_degree",
            )
        )
        (;
            rel=c["rel"],
            rho=cell["rho"],
            eta=cell["eta"],
            seeds=c["seeds"],
            values=metrics,
            cfg=c["cfg"],
            periods=c["late_periods"],
            source_sha256=c["source_sha256"],
        )
    end
    sources = Dict(
        name => Dict(
            k => source[k] for k in
            ("root", "simulation_commit", "manifest_hash", "manifest_sha256", "meta")
        ) for (name, source) in raw["meta"]["sources"] if
        name in ["nn"; [v.dataset for v in VARIANTS]]
    )
    metadata = Dict(
        "extraction" => Dict(k => v for (k, v) in raw["meta"] if k != "sources"),
        "sources" => sources,
        "summary_sha256" => file_hash(path),
        "validation_inputs" =>
            Dict(relpath(p, ROOT) => file_hash(p) for p in (MAIN_INPUT, RIDGE_INPUT)),
        "preparation_commit" => provenance.commit,
        "preparation_source_clean" => provenance.source_clean,
        "preparation_source" => read(@__FILE__, String),
        "interval_level" => ridge["meta"]["interval_level"],
        "late_width" => ridge["meta"]["late_width"],
        "degree_definition" => "Principal-node degree, including the broker edge when present",
    )
    data = (; metadata, ridge=ridge_rows, nn=nn_rows)
    validate(data)
    jldsave(INPUT; data)
    println(
        "Retained $(length(nn_rows)) NN regimes and $(length(ridge_rows)) Ridge versions in $INPUT",
    )
    return data
end

"""Check common windows, parameter settings, seed coverage, and measure bounds."""
function validate(data)
    for (relative, expected) in data.metadata["validation_inputs"]
        file_hash(joinpath(ROOT, relative)) == expected || error("Input changed: $relative")
    end
    0 < data.metadata["interval_level"] < 1 || error("Invalid interval level")
    length(data.ridge) == length(VARIANTS) || error("Incomplete Ridge versions")
    rhos = sort(unique(c.rho for c in data.nn))
    etas = sort(unique(c.eta for c in data.nn))
    Set((c.rho, c.eta) for c in data.nn) == Set((r, e) for r in rhos for e in etas) ||
        error("Incomplete composition-by-turnover grid")
    length(data.nn) == length(rhos) * length(etas) || error("Duplicate NN coordinates")
    reference = first(data.ridge)
    for c in (data.nn..., data.ridge...)
        length(c.seeds) == length(unique(c.seeds)) > 1 || error("Invalid seed coverage")
        c.periods == reference.periods || error("Reporting windows differ")
        length(c.periods) == data.metadata["late_width"] || error("Incomplete late window")
        for key in ("N", "delta", "roster_frac", "k", "n_strangers", "reservation_frac")
            c.cfg[key] == reference.cfg[key] || error("Unexpected variation in $key")
        end
    end
    for c in data.ridge
        c.seeds == reference.seeds || error("Ridge seeds are not aligned")
        c.cfg["rho"] == reference.cfg["rho"] && c.cfg["eta"] == reference.cfg["eta"] ||
            error("Ridge baseline settings differ")
        Set(keys(c.values)) == Set(metric.key for metric in RIDGE_METRICS) ||
            error("Incomplete Ridge metrics")
        for (metric, values) in c.values
            length(values) == length(c.seeds) || error("Incomplete Ridge observations")
            all(v -> isfinite(v) && 0 <= v <= 1, values) || error("Invalid Ridge $metric")
        end
    end
    for c in data.nn
        for (metric, values) in c.values
            length(values) == length(c.seeds) || error("Incomplete NN observations")
            upper = if metric in ("betweenness", "access_fraction", "outsourcing_rate")
                1
            else
                c.cfg["N"]
            end
            all(v -> isfinite(v) && 0 <= v <= upper, values) || error("Invalid NN $metric")
        end
        all(c.values["max_degree"] .>= c.values["mean_degree"]) &&
        all(c.values["max_degree"] .>= c.values["median_degree"]) ||
            error("Invalid degree summaries")
    end
    return data
end

"""Calculate quoted estimates and verify the paired comparisons used in the prose."""
function manuscript_values(data)
    validate(data)
    reference = only(c for c in data.ridge if c.key == "pair")
    additive = only(c for c in data.ridge if c.key == "additive")
    baseline_rho, baseline_eta = reference.cfg["rho"], reference.cfg["eta"]
    rhos = sort(unique(c.rho for c in data.nn))
    etas = sort(unique(c.eta for c in data.nn))
    low_rho, mixed_rho, quality_rho = first(rhos), maximum(filter(<(1), rhos)), last(rhos)
    point(rho, eta) = only(c for c in data.nn if c.rho == rho && c.eta == eta)
    complementarity = point(low_rho, baseline_eta)
    mixed = point(mixed_rho, baseline_eta)
    quality = point(quality_rho, baseline_eta)
    stable = point(baseline_rho, first(etas))
    high_turnover = point(baseline_rho, last(etas))
    level = data.metadata["interval_level"]
    for eta in etas
        lower, higher = point(mixed_rho, eta), point(low_rho, eta)
        lower.seeds == higher.seeds || error("Composition contrast seeds are not aligned")
        for metric in ("betweenness", "access_fraction")
            ci = paired_monte_carlo_interval(
                lower.values[metric], higher.values[metric]; level
            )
            supported = metric == "betweenness" ? ci.lower > 0 : ci.upper < 0
            supported || error("Composition contrast changed: $eta, $metric")
        end
    end
    for metric in ("outsourcing_rate", "betweenness")
        mean(additive.values[metric]) < mean(reference.values[metric]) ||
            error("Ridge restriction no longer reduces $metric")
    end
    defs = Dict{String,String}()
    for (name, row, metric) in (
        ("centralMixedBetweenness", mixed, "betweenness"),
        ("centralComplementarityBetweenness", complementarity, "betweenness"),
        ("centralQualityBetweenness", quality, "betweenness"),
        ("centralQualityMaxDegree", quality, "max_degree"),
        ("centralComplementarityMaxDegree", complementarity, "max_degree"),
        ("centralQualityMeanDegree", quality, "mean_degree"),
        ("centralComplementarityMeanDegree", complementarity, "mean_degree"),
        ("centralStableMeanDegree", stable, "mean_degree"),
        ("centralHighTurnoverMeanDegree", high_turnover, "mean_degree"),
        ("centralStableBetweenness", stable, "betweenness"),
        ("centralHighTurnoverBetweenness", high_turnover, "betweenness"),
        ("centralRidgeReferenceBetweenness", reference, "betweenness"),
        ("centralRidgeAdditiveBetweenness", additive, "betweenness"),
    )
        # Round decimal ties before formatting to avoid binary representation artifacts.
        estimate = round(BigFloat(string(mean(row.values[metric]))); digits=2)
        defs[name] = @sprintf("%.2f", estimate)
    end
    for (name, row) in (
        ("centralMixedAssessmentPercent", mixed),
        ("centralComplementarityAssessmentPercent", complementarity),
        ("centralQualityAssessmentPercent", quality),
    )
        defs[name] = @sprintf("%.0f", 100 * mean(1 .- row.values["access_fraction"]))
    end
    for (name, row, metric) in (
        ("centralQualityOutPercent", quality, "outsourcing_rate"),
        ("centralStableAccessPercent", stable, "access_fraction"),
        ("centralHighTurnoverAccessPercent", high_turnover, "access_fraction"),
        ("centralRidgeReferenceOutPercent", reference, "outsourcing_rate"),
        ("centralRidgeAdditiveOutPercent", additive, "outsourcing_rate"),
        ("centralRidgeReferenceAccessPercent", reference, "access_fraction"),
        ("centralRidgeAdditiveAccessPercent", additive, "access_fraction"),
    )
        defs[name] = @sprintf("%.0f", 100 * mean(row.values[metric]))
    end
    seed_counts = sort(unique(length(c.seeds) for c in (data.nn..., data.ridge...)))
    defs["centralEtaCount"] = string(length(etas))
    defs["centralGridCount"] = string(length(data.nn))
    defs["centralBaselineSeeds"] = string(length(reference.seeds))
    defs["centralOtherSeeds"] = string(first(seed_counts))
    defs["centralIntervalPercent"] = @sprintf("%.0f", 100 * level)
    defs["centralHighTurnoverPercent"] = @sprintf("%.0f", 100 * last(etas))
    defs["centralBaselineEta"] = string(baseline_eta)
    defs["centralBaselineDelta"] = string(reference.cfg["delta"])
    defs["centralMixedRho"] = @sprintf("%g", mixed_rho)
    defs["centralComplementarityRho"] = @sprintf("%g", low_rho)
    return defs
end

"""Write data-derived manuscript values with separate data and reporting provenance."""
function write_manuscript_values(data)
    defs = manuscript_values(data)
    provenance = manuscript_git_provenance(
        ROOT; sources=(@__FILE__, "scripts/monte_carlo.jl", "scripts/figure_style.jl")
    )
    analysis_commit = validate_analysis_commit(
        provenance, data.metadata["extraction"]["analysis_commit"]
    )
    open(VALUES_OUTPUT, "w") do io
        println(
            io, "% Generated by scripts/paper/centrality_and_assessment.jl; do not edit."
        )
        println(io, "% data analysis commit: ", analysis_commit)
        println(io, "% manuscript commit: ", provenance.commit)
        println(io, "% Input: ", relpath(INPUT, ROOT), "; SHA256: ", file_hash(INPUT))
        for (name, source) in sort(collect(data.metadata["sources"]); by=first)
            println(io, "% Simulation commit (", name, "): ", source["simulation_commit"])
            println(io, "% Manifest hash (", name, "): ", source["manifest_hash"])
        end
        write_source_provenance(io, provenance)
        for key in sort(collect(keys(defs)))
            println(io, "\\pvDefine{", key, "}{", defs[key], "}")
        end
    end
    println(
        "Wrote $VALUES_OUTPUT; paired composition intervals verified at every turnover rate"
    )
    return VALUES_OUTPUT
end

"""Draw pointwise Monte Carlo intervals without treating periods as replications."""
function whiskers!(axis, x, y, intervals; color, direction=:y)
    estimates = direction == :x ? x : y
    errorbars!(
        axis,
        x,
        y,
        estimates .- getproperty.(intervals, :lower),
        getproperty.(intervals, :upper) .- estimates;
        direction,
        color=(color, 0.42),
        whiskerwidth=5,
        linewidth=1.3,
    )
end

"""Render the publication figure from seed-level late means and Monte Carlo intervals."""
function render(data)
    validate(data)
    publication_theme!()
    level = data.metadata["interval_level"]
    interval(x) = monte_carlo_interval(x; level)
    summary = [
        merge(c, (; intervals=Dict(k => interval(v) for (k, v) in c.values))) for
        c in data.nn
    ]
    rhos = sort(unique(c.rho for c in summary))
    etas = sort(unique(c.eta for c in summary))
    baseline_eta = first(data.ridge).cfg["eta"]
    baseline_rows = sort(filter(row -> row.eta == baseline_eta, summary); by=row -> row.rho)
    length(baseline_rows) == length(rhos) || error("Incomplete baseline-turnover sweep")
    population = first(data.ridge).cfg["N"]
    fig = Figure(; size=(1540, 1160), figure_padding=(24, 26, 20, 40))

    ridge_rows = [
        only(c for c in data.ridge if c.key == variant.key && c.dataset == variant.dataset)
        for variant in VARIANTS
    ]
    y = Float64.(collect(4:-1:1))
    a = Axis(
        fig[2, 2];
        title="D. Consequences of broker learning",
        subtitle="Ridge regression · baseline",
        subtitlesize=TICK_FS,
        subtitlecolor=:gray40,
        xlabel="Late mean",
        yticks=(y, [v.label for v in VARIANTS]),
        xticks=0:0.2:1,
        limits=((0, 1), (0.5, 4.5)),
        xgridvisible=true,
        xgridcolor=(:black, 0.065),
        xminorticks=IntervalsBetween(2),
        xminorgridvisible=true,
        xminorgridcolor=(:black, 0.035),
        xminorgridwidth=0.6,
        ygridvisible=false,
        leftspinevisible=false,
        yticksize=0,
    )
    hspan!(a, 3.6, 4.4; color=(:black, 0.025))
    ridge_intervals = Dict(
        metric.key => [interval(c.values[metric.key]) for c in ridge_rows] for
        metric in RIDGE_METRICS
    )
    for metric in RIDGE_METRICS
        intervals = ridge_intervals[metric.key]
        x = getproperty.(intervals, :mean)
        lines!(a, x, y; color=metric.color, linewidth=1.5)
        whiskers!(a, x, y, intervals; color=metric.color, direction=:x)
        scatter!(
            a,
            x,
            y;
            color=metric.color,
            marker=metric.marker,
            label=metric.label,
            markersize=13,
            strokecolor=:white,
            strokewidth=0.8,
        )
    end
    Legend(
        fig[3, 2],
        a;
        PUB_LEGEND...,
        labelsize=FOOTER_FS,
        nbanks=2,
        patchsize=(18, 14),
        colgap=18,
        rowgap=4,
    )

    access_bounds = [
        100 * v for c in summary for
        v in (c.intervals["access_fraction"].lower, c.intervals["access_fraction"].upper)
    ]
    access_axis = interval_axis(access_bounds; target_intervals=4)
    b = Axis(
        fig[1, 1];
        title="A. Bridging behavior",
        xlabel="Broker betweenness centrality",
        ylabel="Access fraction (%)",
        xticks=0:0.2:1,
        yticks=access_axis.ticks,
        limits=((0, 1), access_axis.limits),
    )
    all(
        row.intervals[metric.key].lower >= 1 for row in baseline_rows for
        metric in DEGREE_METRICS
    ) || error("Degree intervals fall below the selected log-axis lower limit")
    degree_upper =
        1.3 *
        max(population, maximum(row.intervals["max_degree"].upper for row in baseline_rows))
    degree_ticks = 10 .^ (0:floor(Int, log10(degree_upper)))
    c = Axis(
        fig[1, 2];
        title="B. Principal degree distribution",
        subtitle="Baseline turnover (η = $baseline_eta)",
        subtitlesize=TICK_FS,
        subtitlecolor=:gray40,
        xlabel=PUB_RHO_LABEL,
        ylabel="Principal degree\n(log₁₀ scale)",
        xticks=(rhos, string.(rhos)),
        yscale=log10,
        yticks=(degree_ticks, string.(degree_ticks)),
        limits=((-0.025, 1.035), (1, degree_upper)),
    )
    degree_bounds = [
        v for row in summary for
        v in (row.intervals["mean_degree"].lower, row.intervals["mean_degree"].upper)
    ]
    degree_axis = interval_axis(degree_bounds; target_intervals=5)
    d = Axis(
        fig[2, 1];
        title="C. Principal connectivity",
        xlabel="Broker betweenness centrality",
        ylabel="Mean principal degree",
        xticks=0:0.2:1,
        yticks=degree_axis.ticks,
        limits=((0, 1), degree_axis.limits),
    )

    for eta in etas
        rows = sort(filter(row -> row.eta == eta, summary); by=row -> row.rho)
        color = PUB_ETA_COLORS[eta]
        interior = findall(row -> row.rho < 1, rows)
        rho_values = [row.rho for row in rows]
        beta = [row.intervals["betweenness"] for row in rows]
        access = [
            monte_carlo_interval(100 .* row.values["access_fraction"]; level) for
            row in rows
        ]
        mean_degree = [row.intervals["mean_degree"] for row in rows]
        for (axis, xs, ys, xintervals, yintervals) in (
            (b, getproperty.(beta, :mean), getproperty.(access, :mean), beta, access),
            (
                d,
                getproperty.(beta, :mean),
                getproperty.(mean_degree, :mean),
                beta,
                mean_degree,
            ),
        )
            whiskers!(axis, xs, ys, xintervals; color, direction=:x)
            whiskers!(axis, xs, ys, yintervals; color)
            lines!(axis, xs[interior], ys[interior]; color, linewidth=1.5)
            for (rho, px, py) in zip(rho_values, xs, ys)
                scatter!(
                    axis,
                    [px],
                    [py];
                    marker=PUB_RHO_MARKERS[rho],
                    color,
                    markersize=12,
                    strokecolor=:gray15,
                    strokewidth=0.6,
                )
            end
        end
    end

    degree_colors = (PUB_ETA_COLORS[baseline_eta], PUB_PRINCIPAL)
    composition = [row.rho for row in baseline_rows]
    interior = findall(rho -> rho < 1, composition)
    for (metric, color) in zip(DEGREE_METRICS, degree_colors)
        intervals = [row.intervals[metric.key] for row in baseline_rows]
        values = getproperty.(intervals, :mean)
        whiskers!(c, composition, values, intervals; color)
        lines!(c, composition[interior], values[interior]; color, linewidth=1.5)
        for (rho, value) in zip(composition, values)
            scatter!(
                c,
                [rho],
                [value];
                marker=PUB_RHO_MARKERS[rho],
                color,
                markersize=12,
                strokecolor=:gray15,
                strokewidth=0.6,
            )
        end
    end
    axislegend(
        c,
        [LineElement(; color, linewidth=1.5) for color in degree_colors],
        [metric.label for metric in DEGREE_METRICS];
        position=:lt,
        labelsize=FOOTER_FS,
        framevisible=false,
        margin=(12, 12, 12, 12),
        patchsize=(24, 14),
        rowgap=5,
        tellwidth=false,
        tellheight=false,
    )

    eta_elements = [
        LineElement(; color=PUB_ETA_COLORS[value], linewidth=1.5) for value in etas
    ]
    rho_elements = [
        MarkerElement(;
            color=:gray60,
            marker=PUB_RHO_MARKERS[value],
            markersize=12,
            strokecolor=:gray20,
            strokewidth=0.5,
        ) for value in rhos
    ]
    Legend(
        fig[4, 1:2],
        [eta_elements, rho_elements],
        [string.(etas), string.(rhos)],
        ["Turnover (η)", PUB_RHO_LABEL];
        PUB_LEGEND...,
        nbanks=2,
        groupgap=65,
    )
    colgap!(fig.layout, 55)
    rowgap!(fig.layout, 1, 38)
    rowgap!(fig.layout, 2, 8)
    rowgap!(fig.layout, 3, 20)
    rowsize!(fig.layout, 1, Auto(1))
    rowsize!(fig.layout, 2, Auto(1))
    mkpath(dirname(OUTPUT))
    save(OUTPUT, fig; px_per_unit=2)
    println("Rendered $OUTPUT")
    for metric in RIDGE_METRICS, (variant, ci) in zip(VARIANTS, ridge_intervals[metric.key])
        @printf(
            "D %s %s: %.6f [%.6f, %.6f], n=%d\n",
            metric.key,
            variant.key,
            ci.mean,
            ci.lower,
            ci.upper,
            ci.n
        )
    end
    return OUTPUT
end

"""Prepare on request, then render from the validated compact dataset."""
function main(args=ARGS)
    data = if isempty(args)
        load(INPUT)["data"]
    elseif length(args) == 2 && first(args) == "--prepare"
        prepare(last(args))
    else
        error("Usage: centrality_and_assessment.jl [--prepare <spectrum-results.jld2>]")
    end
    render(data)
    write_manuscript_values(data)
    return nothing
end

end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && CentralityAndAssessment.main()
