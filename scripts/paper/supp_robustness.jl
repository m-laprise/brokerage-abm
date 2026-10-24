"""
Alternative network measures for the main structural comparisons.

Retain service and learning results from the saved full-sweep summary:
    julia --project --threads=auto scripts/paper/supp_robustness.jl --prepare <summary.jld2>
Use the same summary as centrality_and_assessment.jl; subsequent publication
builds use the compact retained input.
"""
module SupplementRobustness

using JLD2
include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = joinpath(ROOT, "output", "supplement", "structural_comparisons.jld2")
const CENTRALITY = joinpath(ROOT, "output", "main", "centrality_data.jld2")
const SERVICES = joinpath(ROOT, "output", "assessment_access", "figure_data.jld2")
const MEASURES = (
    (key="constraint", label="Broker constraint"),
    (key="effective_size", label="Broker effective size"),
)
const LEARNERS = (
    (key="pair", label="Reference\nbroker"),
    (key="size_matched", label="Fewer\nobservations"),
    (key="additive", label="No interaction\nterm"),
    (key="single_principal", label="One party\nrecorded"),
)
const METRICS = (
    "constraint",
    "effective_size",
    "betweenness",
    "mean_degree",
    "access_fraction",
    "outsourcing_rate",
)
file_hash(path) = bytes2hex(sha256(read(path)))

"""Copy recorded seed means, checking them against the main figures' inputs."""
function prepare(path)
    raw = load(path)
    centrality = load(CENTRALITY)["data"]
    services = load(SERVICES)
    file_hash(path) == centrality.metadata["summary_sha256"] ||
        error("summary differs from the main structural analysis")
    provenance = manuscript_git_provenance(ROOT; sources=(@__FILE__,))
    commit = validate_analysis_commit(provenance, raw["meta"]["analysis_commit"])
    baseline = first(centrality.ridge).cfg
    periods = first(centrality.ridge).periods
    length(periods) == services["late_width"] || error("reporting windows differ")
    level = centrality.metadata["interval_level"]
    level == services["interval_level"] || error("interval levels differ")
    for (name, hash) in (
        ("services", services["manifest_hash"]),
        ("services_extra", services["supplement_manifest_hash"]),
    )
        raw["meta"]["sources"][name]["manifest_hash"] == hash ||
            error("service manifest differs")
    end
    selected = filter(raw["conditions"]) do row
        row["dataset"] in ("services", "services_extra") || any(
            r -> r.dataset == row["dataset"] && r.source_sha256 == row["source_sha256"],
            centrality.ridge,
        )
    end
    service_index = Dict(
        (String(r[1]), Float64(r[2]), Float64(r[3]), Int(r[4]), String(r[5])) =>
            Float64(r[6]) for r in services["seed_rows"]
    )
    service_rows, ridge_rows = NamedTuple[], NamedTuple[]
    for row in selected
        cfg, seeds, metrics = row["cfg"], row["seeds"], row["metrics"]
        row["late_periods"] == periods || error("late periods differ")
        all(
            cfg[k] == baseline[k] for
            k in ("N", "delta", "roster_frac", "k", "n_strangers", "reservation_frac")
        ) || error("nonbaseline control parameter")
        values = Dict(k => metrics[k] for k in METRICS)
        all(length(v) == length(seeds) && all(isfinite, v) for v in Base.values(values)) ||
            error("incomplete or nonfinite structural measures")
        common = (; seeds, values, source_sha256=row["source_sha256"])
        if row["dataset"] in ("services", "services_extra")
            mode, rho, eta = String(cfg["broker_service"]), cfg["rho"], cfg["eta"]
            for metric in
                ("betweenness", "mean_degree", "access_fraction", "outsourcing_rate")
                expected = [service_index[(mode, rho, eta, seed, metric)] for seed in seeds]
                isapprox(values[metric], expected; atol=1e-12) || error(
                    "service data differ from main results: $mode, $rho, $eta, $metric"
                )
            end
            push!(service_rows, (; common..., mode, rho, eta))
        else
            reference = only(r for r in centrality.ridge if r.dataset == row["dataset"])
            seeds == reference.seeds || error("Ridge seeds differ")
            for metric in keys(reference.values)
                isapprox(values[metric], reference.values[metric]; atol=1e-12) ||
                    error("Ridge data differ from main results: $metric")
            end
            push!(ridge_rows, (; common..., key=reference.key))
        end
    end
    expected = Set(
        (String(r[1]), Float64(r[2]), Float64(r[3])) for r in services["seed_rows"]
    )
    Set((r.mode, r.rho, r.eta) for r in service_rows) == expected ||
        error("missing service regimes")
    length(service_rows) == length(expected) || error("duplicate service regimes")
    Set(r.key for r in ridge_rows) == Set(r.key for r in LEARNERS) ||
        error("missing Ridge versions")
    sources = Dict(
        name => Dict(
            k => source[k] for
            k in ("root", "simulation_commit", "manifest_hash", "manifest_sha256")
        ) for (name, source) in raw["meta"]["sources"] if name != "nn"
    )
    all(
        raw["meta"]["sources"][name]["meta"][:git_dirty] === false for name in keys(sources)
    ) || error("simulation source was dirty")
    metadata = Dict(
        "analysis_git_commit" => commit,
        "summary_sha256" => file_hash(path),
        "sources" => sources,
        "late_periods" => periods,
        "interval_level" => level,
        "baseline" => Dict(k => baseline[k] for k in ("N", "rho", "delta", "eta")),
        "validation_inputs" =>
            Dict(relpath(p, ROOT) => file_hash(p) for p in (CENTRALITY, SERVICES)),
    )
    data = (; metadata, services=service_rows, ridge=ridge_rows)
    jldsave(INPUT; data)
    record_publication_stage(
        provenance;
        stage="scripts/paper/supp_robustness.jl --prepare",
        inputs=(CENTRALITY, SERVICES),
        outputs=(INPUT,),
        analysis_commits=Dict("data analysis commit" => commit),
        settings=Dict("source_summary_sha256" => file_hash(path)),
    )
    println(
        "Retained $(length(service_rows)) service conditions and $(length(ridge_rows)) Ridge versions",
    )
    return data
end

"""Join the same conditions and seeds used in the main composition-turnover figure."""
function condition_data(structural, centrality)
    structural["meta"]["manifest_hash"] ==
    centrality.metadata["sources"]["nn"]["manifest_hash"] || error("NN manifest differs")
    by_rel = Dict(c["rel"] => c for c in structural["regime_cells"])
    return map(centrality.nn) do row
        saved = by_rel[row.rel]
        saved["seeds"] == row.seeds || error("NN seeds differ")
        values = merge(
            row.values,
            Dict(
                "constraint" => saved["seed_values"]["constraint"],
                "effective_size" => saved["seed_values"]["effsize"],
            ),
        )
        (; row..., values)
    end
end

"""Draw uncertainty on both coordinates of a regime or paired contrast."""
function point!(axis, x, y; color, marker=:circle, markersize=11)
    rangebars!(
        axis,
        [y.mean],
        [x.lower],
        [x.upper];
        direction=:x,
        color=(color, 0.45),
        whiskerwidth=5,
        linewidth=1.2,
    )
    rangebars!(
        axis,
        [x.mean],
        [y.lower],
        [y.upper];
        color=(color, 0.45),
        whiskerwidth=5,
        linewidth=1.2,
    )
    scatter!(
        axis,
        [x.mean],
        [y.mean];
        color,
        marker,
        markersize,
        strokecolor=:gray20,
        strokewidth=0.5,
    )
end

"""Repeat the main access and principal-degree comparisons with both measures."""
function render_conditions(structural, centrality, output)
    rows = condition_data(structural, centrality)
    level = centrality.metadata["interval_level"]
    interval(v) = monte_carlo_interval(v; level)
    etas = sort(unique(r.eta for r in rows))
    rhos = sort(unique(r.rho for r in rows))
    fig = Figure(; size=(1400, 1100))
    for (column, measure) in enumerate(MEASURES),
        (row, (metric, label, scale)) in enumerate((
            ("access_fraction", "Access fraction (%)", 100),
            ("mean_degree", "Mean principal degree", 1),
        ))

        letter = Char('A' + (row - 1) * length(MEASURES) + column - 1)
        axis = Axis(
            fig[row, column];
            title="$letter. $(row == 1 ? "Access fraction" : "Principal ties")",
            xlabel=measure.label,
            ylabel=label,
        )
        for eta in etas
            group = sort(filter(r -> r.eta == eta, rows); by=r -> r.rho)
            xs = [interval(r.values[measure.key]) for r in group]
            ys = [interval(scale .* r.values[metric]) for r in group]
            interior = findall(r -> r.rho < 1, group)
            color = PUB_ETA_COLORS[eta]
            lines!(
                axis,
                getproperty.(xs[interior], :mean),
                getproperty.(ys[interior], :mean);
                color,
                linewidth=1.5,
            )
            for (r, x, y) in zip(group, xs, ys)
                point!(axis, x, y; color, marker=PUB_RHO_MARKERS[r.rho])
            end
        end
    end
    Legend(
        fig[3, 1:2],
        [
            [LineElement(; color=PUB_ETA_COLORS[v], linewidth=1.5) for v in etas],
            [
                MarkerElement(; color=:gray60, marker=PUB_RHO_MARKERS[v], markersize=11) for
                v in rhos
            ],
        ],
        [string.(etas), string.(rhos)],
        ["Turnover (η)", PUB_RHO_LABEL];
        PUB_LEGEND...,
        nbanks=2,
    )
    colgap!(fig.layout, 45)
    rowgap!(fig.layout, 35)
    save(output, fig; px_per_unit=2)
end

"""Repeat the service contrasts and baseline learning comparisons in the main text."""
function render_experiments(data, output)
    for (path, expected) in data.metadata["validation_inputs"]
        file_hash(joinpath(ROOT, path)) == expected || error("main input changed: $path")
    end
    level = data.metadata["interval_level"]
    interval(v) = monte_carlo_interval(v; level)
    rows = data.services
    regimes = sort(unique((r.rho, r.eta) for r in rows))
    rhos = sort(unique(r.rho for r in rows))
    services = (
        (mode="access_only", label="Adding assessment", marker=:circle),
        (mode="assessment_only", label="Adding access", marker=:utriangle),
    )
    fig = Figure(; size=(1400, 1080))
    for (column, measure) in enumerate(MEASURES)
        axis = Axis(
            fig[1, column];
            title="$(Char('A' + column - 1)). Service comparisons",
            subtitle="Neural-network learning",
            subtitlesize=FOOTER_FS,
            subtitlecolor=:gray40,
            xlabel="Change in mean principal degree",
            ylabel="Change in $(lowercase(measure.label))",
        )
        hlines!(axis, [0.0]; color=:gray65, linestyle=:dash, linewidth=1)
        vlines!(axis, [0.0]; color=:gray65, linestyle=:dash, linewidth=1)
        shown = [(; axis, markersize=11)]
        if column == 1
            # Separate the smaller effects from the much larger pure-quality contrasts.
            interior = filter(r -> r.rho < 1, rows)
            effects = [
                interval(
                    only(
                        r for
                        r in interior if r.rho == rho && r.eta == eta && r.mode == "full"
                    ).values[measure.key] .- only(
                        r for r in interior if
                        r.rho == rho && r.eta == eta && r.mode == service.mode
                    ).values[measure.key],
                ) for (rho, eta) in regimes if rho < 1 for service in services
            ]
            bounds = interval_axis([v for e in effects for v in (e.lower, e.upper)])
            inset = Axis(
                fig[1, column];
                width=Relative(0.53),
                height=Relative(0.48),
                halign=0.98,
                valign=0.96,
                tellwidth=false,
                tellheight=false,
                title="Near zero",
                titlesize=16,
                titlegap=5,
                xticklabelsize=14,
                yticklabelsize=14,
                yticks=bounds.ticks,
                limits=(nothing, bounds.limits),
                backgroundcolor=:white,
            )
            hlines!(inset, [0.0]; color=:gray65, linestyle=:dash, linewidth=1)
            vlines!(inset, [0.0]; color=:gray65, linestyle=:dash, linewidth=1)
            push!(shown, (; axis=inset, markersize=7))
        end
        for (rho, eta) in regimes
            full = only(r for r in rows if r.rho == rho && r.eta == eta && r.mode == "full")
            points = map(services) do service
                other = only(
                    r for
                    r in rows if r.rho == rho && r.eta == eta && r.mode == service.mode
                )
                full.seeds == other.seeds || error("unpaired service seeds")
                x = interval(full.values["mean_degree"] .- other.values["mean_degree"])
                y = interval(full.values[measure.key] .- other.values[measure.key])
                (; x, y, service)
            end
            for display in shown
                lines!(
                    display.axis,
                    [p.x.mean for p in points],
                    [p.y.mean for p in points];
                    color=:gray60,
                    linestyle=:dash,
                    linewidth=1.2,
                )
                for p in points
                    point!(
                        display.axis,
                        p.x,
                        p.y;
                        color=PUB_RHO_COLORS[rho],
                        marker=p.service.marker,
                        markersize=display.markersize,
                    )
                end
            end
        end
        learning = Axis(
            fig[3, column];
            title="$(Char('C' + column - 1)). Learning restrictions",
            subtitle="Ridge regression · baseline",
            subtitlesize=FOOTER_FS,
            subtitlecolor=:gray40,
            xlabel=measure.label,
            yticks=(collect(length(LEARNERS):-1:1), [r.label for r in LEARNERS]),
            limits=(nothing, (0.5, length(LEARNERS) + 0.5)),
            xgridvisible=true,
            ygridvisible=false,
            leftspinevisible=false,
            yticksize=0,
        )
        estimates = [
            interval(
                only(r for r in data.ridge if r.key == learner.key).values[measure.key]
            ) for learner in LEARNERS
        ]
        ys = collect(length(LEARNERS):-1:1)
        hspan!(
            learning, length(LEARNERS) - 0.4, length(LEARNERS) + 0.4; color=(:black, 0.025)
        )
        lines!(
            learning,
            getproperty.(estimates, :mean),
            ys;
            color=PUB_CENTRALITY,
            linewidth=1.5,
        )
        rangebars!(
            learning,
            ys,
            getproperty.(estimates, :lower),
            getproperty.(estimates, :upper);
            direction=:x,
            color=PUB_CENTRALITY,
            linewidth=1.5,
            whiskerwidth=7,
        )
        scatter!(
            learning,
            getproperty.(estimates, :mean),
            ys;
            color=PUB_CENTRALITY,
            marker=:rect,
            markersize=11,
        )
    end
    Legend(
        fig[2, 1:2],
        [
            [
                MarkerElement(; color=:gray45, marker=s.marker, markersize=11) for
                s in services
            ],
            [
                MarkerElement(; color=PUB_RHO_COLORS[rho], marker=:circle, markersize=11)
                for rho in rhos
            ],
        ],
        [[s.label for s in services], string.(rhos)],
        ["Service contrast", PUB_RHO_LABEL];
        PUB_LEGEND...,
    )
    colgap!(fig.layout, 50)
    rowgap!(fig.layout, 30)
    save(output, fig; px_per_unit=2)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 2 && first(ARGS) == "--prepare" ||
        error("expected --prepare <summary.jld2>")
    prepare(last(ARGS))
end
end
