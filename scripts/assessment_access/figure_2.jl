"""
Render the current main-text assessment vs access figure.

Panels show net-output differences across turnover, baseline principal degree,
seed-paired late-mean centrality differences, and baseline broker centrality.
The manuscript caption supplies design and interval details. Degree retains all
recorded periods; centrality trajectories use measured periods, while its late
means preserve the retained cached records.
Intervals use the existing seed-level Student-t method. No simulation is changed.
The original figure is not read or overwritten.

Usage: julia --project --threads=auto scripts/assessment_access/figure_2.jl
Output: output/main/figures/assessment_access_2.png
"""

using CairoMakie
using JLD2
using Printf: @sprintf

include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_PATH = joinpath(ROOT, "output", "assessment_access", "figure_data.jld2")
const OUTPUT_PATH = joinpath(ROOT, "output", "main", "figures", "assessment_access_2.png")
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
    data["retained_analysis_git_commit"] == retained["analysis_git_commit"] ||
        error("centrality analysis commit mismatch")
    return data
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

"""Draw measured centrality levels on the full zero-to-one scale."""
function centrality_panel!(slot, data)
    ylimits = (0.0, 1.0)
    axis = Axis(
        slot;
        title="D. Broker centrality over time",
        titlealign=:left,
        titlesize=25,
        xlabel="Period",
        ylabel="Broker betweenness",
        xlabelsize=21,
        ylabelsize=21,
        limits=((0, 1.03 * FIGURE_DESIGN.horizon), ylimits),
        xticks=range(0, FIGURE_DESIGN.horizon; length=6),
        yticks=0:0.2:1,
        xticklabelsize=19,
        yticklabelsize=19,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridcolor=(:black, 0.07),
    )
    for service in (FULL_SERVICE, SERVICES...)
        series = centrality_series(data, service.mode)
        all(ylimits[1] .<= series.lower .<= series.upper .<= ylimits[2]) ||
            error("centrality interval would be clipped")
        band!(axis, series.periods, series.lower, series.upper; color=(service.color, 0.14))
        lines!(
            axis,
            series.periods,
            series.estimate;
            color=service.color,
            linestyle=service.linestyle,
            linewidth=2.7,
        )
        scatter!(
            axis,
            series.periods,
            series.estimate;
            color=service.color,
            marker=service.marker,
            markersize=6,
        )
    end
    return axis
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

"""Draw the seed-paired late-mean centrality differences at baseline."""
function centrality_difference_panel!(slot, index)
    ylimits = (-0.23, 0.12)
    axis = Axis(
        slot;
        title="C. Late-mean centrality difference",
        titlealign=:left,
        titlesize=25,
        ylabel="Betweenness difference\nfrom full service",
        ylabelsize=21,
        limits=((0.45, 2.55), ylimits),
        xticks=([1, 2], [s.label for s in SERVICES]),
        yticks=-0.2:0.1:0.1,
        xticklabelsize=20,
        yticklabelsize=19,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridcolor=(:black, 0.07),
        xticksize=0,
    )
    hlines!(axis, [0.0]; color=:gray45, linewidth=1.2)
    for (x, service) in enumerate(SERVICES)
        value = service_effect(index, service, BASELINE_RHO, BASELINE_ETA, "betweenness")
        effect_point!(axis, x, value, service, ylimits; markersize=14, linewidth=2.5)
        text!(
            axis,
            x,
            value.upper + 0.035 * (ylimits[2] - ylimits[1]);
            text=@sprintf("%+.3f", value.estimate),
            align=(:center, :bottom),
            color=service.color,
            fontsize=24,
            font=:bold,
        )
    end
    return axis
end

"""Draw baseline principal-degree trajectories from the retained period records."""
function degree_panel!(slot, data)
    ylimits = (0.0, 30.0)
    axis = Axis(
        slot;
        title="B. Principal connectivity over time",
        titlealign=:left,
        titlesize=25,
        xlabel="Period",
        ylabel="Mean principal degree",
        xlabelsize=21,
        ylabelsize=21,
        limits=((0, 1.03 * FIGURE_DESIGN.horizon), ylimits),
        xticks=range(0, FIGURE_DESIGN.horizon; length=6),
        yticks=0:5:30,
        xticklabelsize=19,
        yticklabelsize=19,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridcolor=(:black, 0.07),
    )
    for service in (FULL_SERVICE, SERVICES...)
        series = degree_series(data, service.mode)
        all(ylimits[1] .<= series.lower .<= series.upper .<= ylimits[2]) ||
            error("degree interval would be clipped")
        band!(axis, series.periods, series.lower, series.upper; color=(service.color, 0.14))
        lines!(
            axis,
            series.periods,
            series.estimate;
            color=service.color,
            linestyle=service.linestyle,
            linewidth=2.7,
        )
        scatter!(
            axis,
            data["periods"],
            series.estimate[data["periods"]];
            color=service.color,
            marker=service.marker,
            markersize=6,
        )
    end
    return axis
end

"""Draw net-output differences with zero turnover in a separate shaded strip."""
function output_panel!(slot, index)
    ylimits = (-0.43, 0.13)
    grid = GridLayout(slot)
    panel_axes = Axis[]
    for (column, zero) in enumerate((true, false))
        rates = zero ? TURNOVER_RATES[1:1] : TURNOVER_RATES[2:end]
        axis = Axis(
            grid[1, column];
            title=zero ? "A. Net output to principals" : "",
            titlealign=:left,
            titlesize=25,
            xlabel=zero ? "" : "Turnover rate (η)",
            xlabelsize=21,
            ylabel=zero ? "Difference from full service\nper requested position" : "",
            ylabelsize=21,
            limits=(
                if zero
                    (-0.7, 0.7)
                else
                    (-TURNOVER_RATES[2], last(TURNOVER_RATES) + 2 * TURNOVER_RATES[2])
                end,
                ylimits,
            ),
            xticks=(collect(rates), zero ? ["0"] : string.(collect(rates))),
            yticks=-0.4:0.1:0.1,
            xticklabelsize=19,
            yticklabelsize=19,
            yticklabelsvisible=zero,
            yticksvisible=zero,
            leftspinevisible=zero,
            topspinevisible=false,
            rightspinevisible=false,
            xgridvisible=false,
            ygridcolor=(:black, 0.065),
            backgroundcolor=zero ? (:black, 0.025) : :white,
        )
        push!(panel_axes, axis)
        if !zero
            band_width = minimum(diff(collect(TURNOVER_RATES)))
            vspan!(
                axis,
                BASELINE_ETA - band_width,
                BASELINE_ETA + band_width;
                color=(:black, 0.045),
            )
            text!(
                axis,
                BASELINE_ETA,
                0.105;
                text="baseline",
                align=(:center, :center),
                color=:gray40,
                fontsize=16,
            )
        end
        hlines!(axis, [0.0]; color=:gray45, linewidth=1.2)
        for (service_index, service) in enumerate(SERVICES)
            values = [
                service_effect(
                    index, service, BASELINE_RHO, eta, "net_output_per_requested_position"
                ) for eta in rates
            ]
            # Match the original figure's horizontal offsets without changing estimates.
            positions = collect(rates) .+ (service_index - 1) * (zero ? 0.27 : 0.0005)
            if !zero
                lines!(
                    axis,
                    positions,
                    [v.estimate for v in values];
                    color=service.color,
                    linestyle=service.linestyle,
                    linewidth=2.4,
                )
            end
            for (eta, position, value) in zip(rates, positions, values)
                baseline = eta == BASELINE_ETA
                effect_point!(
                    axis,
                    position,
                    value,
                    service,
                    ylimits;
                    markersize=baseline ? 11 : 8,
                    linewidth=1.5,
                )
            end
        end
    end
    linkyaxes!(panel_axes...)
    colsize!(grid, 1, Fixed(60))
    colgap!(grid, 17)
    return panel_axes
end

"""Build four publication panels; the manuscript supplies the title and caption."""
function make_figure(estimates, centrality)
    fig = Figure(; size=(1160, 840), fontsize=19, figure_padding=(28, 28, 24, 20))
    Legend(
        fig[1, 1],
        [
            [
                LineElement(;
                    color=service.color, linestyle=service.linestyle, linewidth=2.4
                ),
                MarkerElement(; color=service.color, marker=service.marker, markersize=11),
            ] for service in (FULL_SERVICE, SERVICES...)
        ],
        [
            "$(service.label)\n($(service.detail))" for
            service in (FULL_SERVICE, SERVICES...)
        ];
        orientation=:horizontal,
        framevisible=false,
        labelsize=18,
        colgap=30,
        halign=:center,
        padding=(0, 0, 0, 0),
    )
    panels = GridLayout(fig[2, 1])
    output_panel!(panels[1, 1], estimates.contrasts)
    degree_panel!(panels[1, 2], centrality)
    centrality_difference_panel!(panels[2, 1], estimates.contrasts)
    centrality_panel!(panels[2, 2], centrality)
    rowsize!(panels, 1, Auto(1))
    rowsize!(panels, 2, Auto(1))
    rowgap!(fig.layout, 25)
    rowgap!(panels, 30)
    colgap!(panels, 45)
    return fig
end

"""Write the alternative PNG, leaving the existing figure untouched."""
function main(; output_path=OUTPUT_PATH)
    provenance = manuscript_git_provenance(ROOT)
    data = read_centrality()
    validate_analysis_commit(
        provenance, data["retained_analysis_git_commit"]; artifact="assessment-access analysis"
    )
    validate_analysis_commit(
        provenance, data["sweep_git_commit"]; artifact="assessment-access simulation"
    )
    fig = make_figure(read_estimates(), data)
    mkpath(dirname(output_path))
    save(output_path, fig; px_per_unit=2)
    println("Wrote $output_path")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
