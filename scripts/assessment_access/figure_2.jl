"""
Render the manuscript assessment-access figure from retained seed-level data.

Panels show output and outsourcing contributions, paired structural changes,
and baseline broker betweenness centrality. The caption is in paper/captions.tex.

Usage: julia --project --threads=auto scripts/assessment_access/figure_2.jl
Output: output/main/figures/assessment_access_3.png
"""


using CairoMakie
using JLD2
using Printf: @sprintf

include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_PATH = joinpath(ROOT, "output", "assessment_access", "figure_data.jld2")
const OUTPUT_PATH = joinpath(ROOT, "output", "main", "figures", "assessment_access_3.png")
const CENTRALITY_PATH = joinpath(
    ROOT, "output", "assessment_access", "centrality_trajectories.jld2"
)

"""Recover figure settings and seed counts from the retained experiment metadata."""
function figure_design(retained, centrality)
    retained["manifest_hash"] == centrality["manifest_hash"] || error("manifest mismatch")
    config = centrality["configs"]["full"]
    rho, delta, eta = Float64.((config["rho"], config["delta"], config["eta"]))
    turnover = sort!(unique(Float64(r[3]) for r in retained["summary_rows"] if r[2] == rho))
    seed_counts = Dict(
        rate => only(
            unique(
                Int(r[9]) for r in retained["summary_rows"] if (r[2], r[3]) == (rho, rate)
            ),
        ) for rate in turnover
    )
    horizon = Int(config["T"])
    late_width = Int(retained["late_width"])
    level = Float64(retained["interval_level"])
    0 < late_width <= horizon || error("invalid late window")
    0 < level < 1 || error("invalid interval level")
    first(turnover) == 0 || error("expected a zero-turnover comparison")
    return (;
        rho,
        delta,
        eta,
        turnover,
        seed_counts,
        horizon,
        late_width,
        level,
        population=Int(config["N"]),
        interval=Int(centrality["network_measure_interval"]),
        seeds=Int.(centrality["seeds"]),
        degree_start=first(centrality["degree_periods"]),
    )
end

const FIGURE_DESIGN = figure_design(JLD2.load(DATA_PATH), JLD2.load(CENTRALITY_PATH))
const BASELINE_RHO = FIGURE_DESIGN.rho
const BASELINE_ETA = FIGURE_DESIGN.eta
const TURNOVER_RATES = Tuple(FIGURE_DESIGN.turnover)
const FULL_SERVICE = (
    mode="full",
    label="Full service",
    detail="Assessment and access",
    color=Makie.to_color("#333A40"),
    marker=:rect,
    linestyle=:dot,
)
const SERVICES = (
    (
        mode="assessment_only",
        contrast="access",
        label="Assessment only",
        detail="No extra broker access",
        color=Makie.to_color("#237A93"),
        marker=:circle,
        linestyle=:solid,
    ),
    (
        mode="access_only",
        contrast="assessment",
        label="Access only",
        detail="Principals' assessments",
        color=Makie.to_color("#BA5A3A"),
        marker=:utriangle,
        linestyle=:dash,
    ),
)

"""Index saved means or paired differences, validating each estimate and interval."""
function interval_index(rows)
    index = Dict{Tuple{String,Float64,Float64,String},NamedTuple}()
    for row in rows
        key = (String(row[1]), Float64(row[2]), Float64(row[3]), String(row[4]))
        haskey(index, key) && error("duplicate retained estimate: $key")
        value = (;
            estimate=Float64(row[5]),
            se=Float64(row[6]),
            lower=Float64(row[7]),
            upper=Float64(row[8]),
            n=Int(row[9]),
        )
        all(isfinite, (value.estimate, value.se, value.lower, value.upper)) ||
            error("nonfinite retained estimate: $key")
        value.se >= 0 && value.n > 1 || error("invalid uncertainty metadata: $key")
        value.lower <= value.estimate <= value.upper || error("invalid interval: $key")
        index[key] = value
    end
    return index
end

"""Read retained late-window means and seed-paired differences."""
function read_estimates(path=DATA_PATH)
    data = JLD2.load(path)
    data["analysis_source_clean"] === true || error("analysis sources were dirty")
    get(data, "net_output_definition", nothing) == "net_output_per_principal" ||
        error("regenerate assessment-access analysis with net output per principal")
    data["interval_level"] == FIGURE_DESIGN.level || error("interval level mismatch")
    data["late_width"] == FIGURE_DESIGN.late_width || error("late window mismatch")
    return (;
        contrasts=interval_index(data["contrast_rows"]),
        summaries=interval_index(data["summary_rows"]),
    )
end

"""Validate seed coverage and the distinct centrality and degree recording grids."""
function validate_centrality(data)
    design = FIGURE_DESIGN
    (data["rho"], data["delta"], data["eta"]) == (design.rho, design.delta, design.eta) ||
        error("centrality is not at baseline")
    data["network_measure_interval"] == design.interval ||
        error("unexpected measurement cadence")
    data["periods"] == collect(design.interval:design.interval:design.horizon) ||
        error("incomplete centrality measurement grid")
    data["seeds"] == design.seeds || error("centrality seed set mismatch")
    length(data["seeds"]) == design.seed_counts[design.eta] ||
        error("baseline seed count mismatch")
    data["sweep_source_clean"] === true || error("centrality simulation source was dirty")
    modes = (FULL_SERVICE, SERVICES...)
    Set(keys(data["values"])) == Set(s.mode for s in modes) ||
        error("centrality service modes mismatch")
    Set(keys(data["degree_values"])) == Set(s.mode for s in modes) ||
        error("degree service modes mismatch")
    for service in modes
        values = data["values"][service.mode]
        size(values) == (length(data["periods"]), length(data["seeds"])) ||
            error("centrality matrix shape mismatch")
        all(v -> isfinite(v) && 0 <= v <= 1, values) || error("invalid measured centrality")
        config = data["configs"][service.mode]
        data["degree_periods"] == collect(1:config["T"]) || error("incomplete degree grid")
        degree = data["degree_values"][service.mode]
        size(degree) == (length(data["degree_periods"]), length(data["seeds"])) ||
            error("degree matrix shape mismatch")
        all(v -> isfinite(v) && 0 <= v <= config["N"], degree) ||
            error("invalid mean degree")
    end
    return data
end

"""Read the compact trajectories and check their manifest against retained results."""
function read_centrality(path=CENTRALITY_PATH; retained_path=DATA_PATH)
    data = validate_centrality(JLD2.load(path))
    retained = JLD2.load(retained_path)
    data["manifest_hash"] == retained["manifest_hash"] ||
        error("centrality manifest mismatch")
    for commit in (data["retained_analysis_git_commit"], retained["analysis_git_commit"])
        validate_analysis_commit((; root=ROOT), commit; artifact="assessment-access analysis")
    end
    check_trajectory_late_means(retained, data)
    return data
end

"""Validate cached trajectories against retained seed values, independently of analysis commit."""
function check_trajectory_late_means(retained, data)
    for service in (FULL_SERVICE, SERVICES...)
        config = data["configs"][service.mode]
        late = (config["T"] - retained["late_width"] + 1):config["T"]
        for (metric, field, indices) in (
            ("mean_degree", "degree_values", late),
            ("betweenness", "values", searchsortedlast.(Ref(data["periods"]), late)),
        )
            all(>(0), indices) || error("late window precedes trajectory measurements")
            rows = filter(retained["seed_rows"]) do row
                (String(row[1]), row[2], row[3], String(row[5])) ==
                    (service.mode, config["rho"], config["eta"], metric)
            end
            expected = Dict(Int(row[4]) => Float64(row[6]) for row in rows)
            length(expected) == length(rows) == length(data["seeds"]) ||
                error("trajectory seed coverage differs")
            Set(keys(expected)) == Set(data["seeds"]) || error("trajectory seeds differ")
            all(isapprox(mean(data[field][service.mode][indices, column]), expected[seed]; atol=1e-12)
                for (column, seed) in enumerate(data["seeds"])) ||
                error("trajectory late means differ from retained $metric")
        end
    end
    return nothing
end

"""Summarize each recorded period with a pointwise seed-mean interval."""
function trajectory_series(periods, values)
    intervals = [
        monte_carlo_interval(view(values, t, :); level=FIGURE_DESIGN.level) for
        t in axes(values, 1)
    ]
    all(v -> v.n == size(values, 2), intervals) || error("incomplete trajectory interval")
    return (;
        periods,
        estimate=[v.mean for v in intervals],
        lower=[v.lower for v in intervals],
        upper=[v.upper for v in intervals],
        se=[v.se for v in intervals],
        n=[v.n for v in intervals],
    )
end

"""Summarize centrality only at its original measurement periods."""
centrality_series(data, mode) = trajectory_series(data["periods"], data["values"][mode])

"""Summarize degree at every recorded period, retaining broker edges."""
function degree_series(data, mode)
    trajectory_series(data["degree_periods"], data["degree_values"][mode])
end

"""Return a broker model minus full service, optionally in percentage points."""
function service_effect(index, service, rho, eta, metric; scale=1.0)
    scale > 0 || throw(ArgumentError("display scale must be positive"))
    key = (service.contrast, rho, eta, metric)
    haskey(index, key) || error("missing paired estimate: $key")
    value = index[key]
    expected_n = FIGURE_DESIGN.seed_counts[eta]
    value.n == expected_n || error("unexpected seed count: $key")
    return (;
        estimate=-scale * value.estimate,
        se=scale * value.se,
        lower=-scale * value.upper,
        upper=-scale * value.lower,
        n=value.n,
    )
end

"""Draw a saved point and interval, rejecting bounds that would be clipped."""
function effect_point!(axis, x, value, service, ylimits; markersize=10, linewidth=2)
    ylimits[1] <= value.lower <= value.upper <= ylimits[2] ||
        error("axis limits would clip a retained interval")
    rangebars!(
        axis,
        [x],
        [value.lower],
        [value.upper];
        color=service.color,
        linewidth,
        whiskerwidth=9,
    )
    scatter!(
        axis, [x], [value.estimate]; color=service.color, marker=service.marker, markersize
    )
    return nothing
end

"""Reconstruct all displayed late means and paired intervals from retained seeds."""
function validate_argument_estimates(retained, estimates, centrality)
    metrics = (
        "net_output_per_principal", "outsourcing_rate", "mean_degree", "betweenness",
    )
    groups = Dict{Tuple{String,Float64,Float64,String},Dict{Int,Float64}}()
    for row in retained["seed_rows"]
        key = (String(row[1]), Float64(row[2]), Float64(row[3]), String(row[5]))
        key[4] in metrics || continue
        values = get!(groups, key, Dict{Int,Float64}())
        seed = Int(row[4])
        haskey(values, seed) && error("duplicate seed in $key")
        isfinite(row[6]) || error("nonfinite seed value in $key")
        values[seed] = Float64(row[6])
    end
    agrees(saved, calculated) = all(isapprox(a, b; atol=1e-12, rtol=1e-10) for (a, b) in zip(
        (saved.estimate, saved.se, saved.lower, saved.upper, saved.n),
        (calculated.mean, calculated.se, calculated.lower, calculated.upper, calculated.n),
    ))
    references = Dict(
        "access" => ("assessment_only", "full"),
        "assessment" => ("access_only", "full"),
        "assessment_vs_access" => ("access_only", "assessment_only"),
    )
    summary_count, contrast_count = 0, 0
    for (key, saved) in estimates.summaries
        key[4] in metrics || continue
        values = groups[key]
        expected = monte_carlo_interval(collect(Base.values(values)); level=FIGURE_DESIGN.level)
        agrees(saved, expected) || error("summary differs from seed-level data: $key")
        summary_count += 1
    end
    for ((contribution, rho, eta, metric), saved) in estimates.contrasts
        metric in metrics || continue
        reference, comparison = references[contribution]
        x, y = groups[(reference, rho, eta, metric)], groups[(comparison, rho, eta, metric)]
        Set(keys(x)) == Set(keys(y)) || error("paired seed coverage differs")
        seeds = sort!(collect(keys(x)))
        expected = monte_carlo_interval([y[s] - x[s] for s in seeds]; level=FIGURE_DESIGN.level)
        agrees(saved, expected) || error("paired contrast differs from seed-level data")
        contrast_count += 1
    end
    late = (FIGURE_DESIGN.horizon - FIGURE_DESIGN.late_width + 1):FIGURE_DESIGN.horizon
    indices = searchsortedlast.(Ref(centrality["periods"]), late)
    for service in (FULL_SERVICE, SERVICES...)
        saved = groups[(service.mode, BASELINE_RHO, BASELINE_ETA, "betweenness")]
        sort!(collect(keys(saved))) == centrality["seeds"] || error("trajectory seed coverage differs")
        all(isapprox(mean(centrality["values"][service.mode][indices, j]), saved[seed];
            atol=1e-12, rtol=1e-10) for (j, seed) in enumerate(centrality["seeds"])) ||
            error("trajectory late means differ from retained summaries")
    end
    return (; summary_count, contrast_count)
end

"""Add a panel heading and a concise description of its comparison or scope."""
function argument_panel_layout!(slot, title, scope)
    grid = GridLayout(slot)
    Label(grid[1, 1], title; font=:bold, fontsize=TITLE_FS, halign=:left, tellwidth=false)
    Label(grid[2, 1], scope; fontsize=FOOTER_FS, color=:gray35, halign=:left, tellwidth=false)
    rowgap!(grid, 13)
    rowgap!(grid, 1, 5)
    return grid
end

"""Draw paired output or outsourcing contributions with the same turnover layout."""
function argument_turnover_panel!(slot, estimates, kind)
    kind in (:output, :outsourcing) || throw(ArgumentError("unknown turnover panel"))
    output = kind == :output
    title = output ? "A. Contributions to principals’ net output" :
        "C. Contributions to principals’ outsourcing"
    grid = argument_panel_layout!(slot, title,
        "Late-mean differences · matching composition ρ = $BASELINE_RHO")
    plot_grid = GridLayout(grid[3, 1])
    series = (
        (; SERVICES[1]..., label="Adding assessment\nFull service − access-only", contrast="assessment"),
        (; SERVICES[2]..., label="Adding access\nFull service − assessment-only", contrast="access"),
    )
    metric = output ? "net_output_per_principal" : "outsourcing_rate"
    scale = output ? 1.0 : 100.0
    interval(service, eta) = begin
        value = estimates.contrasts[(service.contrast, BASELINE_RHO, eta, metric)]
        (; estimate=scale * value.estimate, se=scale * value.se,
            lower=scale * value.lower, upper=scale * value.upper, n=value.n)
    end
    shown = [interval(service, eta) for service in series for eta in TURNOVER_RATES]
    display = interval_axis([v for p in shown for v in (p.lower, p.upper)];
        upper_padding=output ? 0 : 1)
    ylimits, ticks = display.limits, display.ticks
    axes = Axis[]
    for (column, zero) in enumerate((true, false))
        rates = zero ? TURNOVER_RATES[1:1] : TURNOVER_RATES[2:end]
        spacing = minimum(diff(collect(TURNOVER_RATES)))
        ax = Axis(plot_grid[1, column];
            xlabel=zero ? "" : "Turnover rate (η)",
            ylabel=zero ? (output ? "Change in net output to principals\nper principal" :
                "Change in requested positions\noutsourced (percentage points)") : "",
            limits=(zero ? (-0.7, 0.7) : (-spacing, last(TURNOVER_RATES) + 2spacing), ylimits),
            xticks=(collect(rates), zero ? ["0"] : string.(collect(rates))), yticks=ticks,
            yticklabelsvisible=zero, yticksvisible=zero, leftspinevisible=zero,
            backgroundcolor=zero ? (:gray40, 0.045) : :white,
        )
        push!(axes, ax)
        if !zero
            vspan!(ax, [BASELINE_ETA - spacing], [BASELINE_ETA + spacing]; color=(:gray40, 0.06))
            text!(ax, BASELINE_ETA, ylimits[1] + 0.035 * (ylimits[2] - ylimits[1]);
                text="Baseline", color=:gray40, fontsize=FOOTER_FS - 1,
                align=(:center, :bottom))
        end
        hlines!(ax, [0.0]; color=:gray45, linewidth=1.1)
        for (i, service) in enumerate(series)
            values = [interval(service, eta) for eta in rates]
            offset = (i - (length(series) + 1) / 2) * (zero ? 0.35 : 0.35spacing)
            positions = collect(rates) .+ offset
            if !zero
                lines!(ax, positions, [v.estimate for v in values];
                    color=service.color, linestyle=service.linestyle, linewidth=2.5)
            end
            for (eta, x, value) in zip(rates, positions, values)
                value.n == FIGURE_DESIGN.seed_counts[eta] || error("unexpected turnover seed count")
                effect_point!(ax, x, value, service, ylimits;
                    markersize=eta == BASELINE_ETA ? 11 : 9, linewidth=1.7)
            end
        end
        if !zero
            elements = [[
                LineElement(; color=s.color, linestyle=s.linestyle, linewidth=2.4),
                MarkerElement(; color=s.color, marker=s.marker, markersize=9),
            ] for s in series]
            axislegend(ax, elements, [s.label for s in series];
                position=output ? :lt : :rt, framevisible=false,
                labelsize=FOOTER_FS - 1,
                rowgap=10, patchsize=(28, 14),
                backgroundcolor=(:white, 0.9), padding=(5, 5, 5, 5))
        end
    end
    linkyaxes!(axes...)
    colsize!(plot_grid, 1, Fixed(60))
    colgap!(plot_grid, 16)
    return axes
end

"""Show paired structural effects across turnover at baseline matching composition."""
function argument_structure_panel!(slot, estimates)
    regimes = sort!(unique((key[2], key[3]) for key in keys(estimates.summaries)
        if key[2] == BASELINE_RHO))
    last.(regimes) == FIGURE_DESIGN.turnover || error("structural panel turnover coverage differs")
    series = (
        (; SERVICES[1]..., label="Adding assessment\nFull service − access-only", contrast="assessment"),
        (; SERVICES[2]..., label="Adding access\nFull service − assessment-only", contrast="access"),
    )
    value(service, rho, eta, metric) = estimates.contrasts[(service.contrast, rho, eta, metric)]
    points = [
        (; service, rho, eta, x=value(service, rho, eta, "mean_degree"),
            y=value(service, rho, eta, "betweenness"))
        for service in series for (rho, eta) in regimes
    ]
    grid = argument_panel_layout!(slot, "B. Structural consequences of both services",
        "Late-mean differences · ρ = $BASELINE_RHO · $(length(regimes)) regimes")
    xmin = min(0, floor(minimum(p.x.lower for p in points) / 5) * 5)
    xmax = max(0, ceil(maximum(p.x.upper for p in points) / 5) * 5)
    ymin = min(0, floor(minimum(p.y.lower for p in points) / 0.1) * 0.1)
    ymax = max(0, ceil(maximum(p.y.upper for p in points) / 0.1) * 0.1)
    xpad, ypad = 0.025 * (xmax - xmin), 0.025 * (ymax - ymin)
    ax = Axis(grid[3, 1]; xlabel="Change in mean principal degree",
        ylabel="Change in broker\nbetweenness centrality",
        limits=((xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)),
        xticks=(ceil(xmin / 10) * 10):10:xmax, yticks=ymin:0.1:ymax,
    )
    hlines!(ax, [0.0]; color=:gray70, linestyle=:dash, linewidth=1.0)
    vlines!(ax, [0.0]; color=:gray70, linestyle=:dash, linewidth=1.0)
    for (rho, eta) in regimes
        pair = filter(p -> (p.rho, p.eta) == (rho, eta), points)
        length(pair) == 2 || error("expected two structural contrasts per regime")
        lines!(ax, [Point2d(p.x.estimate, p.y.estimate) for p in pair];
            color=:gray60, linestyle=:dash, linewidth=1.2)
    end
    for p in points
        p.x.n == p.y.n || error("structural outcomes have different seed counts")
        color = p.service.color
        rangebars!(ax, [p.y.estimate], [p.x.lower], [p.x.upper];
            direction=:x, color=(color, 0.65), linewidth=1.6, whiskerwidth=6)
        rangebars!(ax, [p.x.estimate], [p.y.lower], [p.y.upper];
            color=(color, 0.65), linewidth=1.6, whiskerwidth=6)
        scatter!(ax, [p.x.estimate], [p.y.estimate]; color=(color, 0.85),
            marker=p.service.marker, markersize=9, strokecolor=:white, strokewidth=0.6)
    end
    elements = Any[MarkerElement(; color=s.color, marker=s.marker, markersize=9) for s in series]
    push!(elements, LineElement(; color=:gray60, linestyle=:dash, linewidth=1.2))
    axislegend(ax, elements, vcat([s.label for s in series], ["Same regime"]);
        position=:rt, framevisible=false, labelsize=FOOTER_FS - 1,
        rowgap=9, patchsize=(20, 14), backgroundcolor=(:white, 0.9), padding=(5, 5, 5, 5))
    return ax
end

"""Show absolute broker betweenness centrality trajectories and the late window."""
function argument_trajectory_panel!(slot, centrality)
    grid = argument_panel_layout!(slot, "D. Broker betweenness centrality over time",
        "Baseline · turnover η = $BASELINE_ETA")
    horizon = FIGURE_DESIGN.horizon
    ax = Axis(grid[3, 1]; xlabel="Period", ylabel="Broker betweenness centrality",
        limits=((0, 1.025horizon), (0, 1.025)),
        xticks=range(0, horizon; length=6), yticks=0:0.2:1)
    late_start = horizon - FIGURE_DESIGN.late_width + 1
    vspan!(ax, [late_start], [horizon]; color=(:gray40, 0.06))
    text!(ax, horizon, 0.035; text="Late", color=:gray40, fontsize=FOOTER_FS - 1,
        align=(:right, :bottom))
    for service in (FULL_SERVICE, SERVICES...)
        values = centrality_series(centrality, service.mode)
        all(0 .<= values.lower .<= values.upper .<= 1) || error("centrality interval outside bounds")
        band!(ax, values.periods, values.lower, values.upper; color=(service.color, 0.12))
        lines!(ax, values.periods, values.estimate; color=service.color,
            linestyle=service.linestyle, linewidth=2.7, label=service.label)
        scatter!(ax, values.periods, values.estimate; color=service.color,
            marker=service.marker, markersize=5)
    end
    axislegend(ax; position=:lb, framevisible=false, labelsize=FOOTER_FS,
        rowgap=4, patchsize=(28, 14), padding=(5, 5, 5, 5), backgroundcolor=(:white, 0.9))
    return ax
end

"""Build the manuscript's four panels without writing an output file."""
function make_figure(estimates, centrality)
    publication_theme!()
    fig = Figure(; size=(1420, 1010), figure_padding=(28, 32, 26, 24))
    argument_turnover_panel!(fig[1, 1], estimates, :output)
    argument_structure_panel!(fig[1, 2], estimates)
    argument_turnover_panel!(fig[2, 1], estimates, :outsourcing)
    argument_trajectory_panel!(fig[2, 2], centrality)
    colgap!(fig.layout, 52)
    rowgap!(fig.layout, 34)
    return fig
end

"""Render the manuscript assessment-access figure from validated retained results."""
function main(; output_path=OUTPUT_PATH)
    provenance = manuscript_git_provenance(
        ROOT;
        sources=(
            @__FILE__,
            "scripts/monte_carlo.jl",
            "scripts/figure_style.jl",
        ),
    )
    retained = JLD2.load(DATA_PATH)
    estimates, centrality = read_estimates(), read_centrality()
    analysis_commit = validate_analysis_commit(provenance, retained["analysis_git_commit"];
        artifact="assessment-access analysis")
    validate_analysis_commit(provenance, centrality["sweep_git_commit"];
        artifact="assessment-access simulation")
    validation = validate_argument_estimates(retained, estimates, centrality)
    fig = make_figure(estimates, centrality)
    mkpath(dirname(output_path))
    save(output_path, fig; px_per_unit=2)
    println("Wrote $output_path")
    println("Verified $(validation.summary_count) means and $(validation.contrast_count) paired intervals against retained seeds.")
    println("Analysis commit: $analysis_commit")
    println("Rendering commit: $(provenance.commit); source clean: $(provenance.source_clean)")
    return fig
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    isempty(ARGS) || error("usage: figure_2.jl")
    main()
end
