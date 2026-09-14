"""
    scripts/paper/figures.jl

Figure assets for the paper's results section. TeX figure numbers follow their
placement in `paper/section_source.tex`; the asset filenames remain stable.
Reads the retained datasets `output/main/figure_data.jld2`,
`output/ridge/ablations/figure_data.jld2`, and the condition audit under
`output/main/convergence/`, so figures render locally without the sweeps. No
simulation; no hard-coded results
(literal constants are display conventions only). Outputs print-resolution PNGs
to output/main/figures/ and the display-convention keys to output/main/figmeta.tex.

  baseline_dynamics   baseline dynamics
  information_sources
                      sources of the broker's ranking advantage
  matching_grid       six outcomes across the rho x delta grid, lines per delta
                      with 95% Monte Carlo interval whiskers
  centrality_and_access
                      centrality against principal degree and access across regimes
  structural_advantage
                      structural measures vs informational/output gaps

Usage: julia --project --threads=auto scripts/paper/figures.jl
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
using JLD2
using Statistics: mean, median

publication_theme!()

const OUT = normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figures"))
const INFORMATION_FIGDATA = normpath(
    joinpath(@__DIR__, "..", "..", "output", "ridge", "ablations", "figure_data.jld2")
)
mkpath(OUT)
const PXU = 2.0                       # px_per_unit: ~330+ dpi at printed full-page width
# Display conventions, quoted in the captions (paper/captions.tex) via the keys
# emitted to output/main/figmeta.tex. BETWINT matches the generation-time
# network_measure_interval; TSTART is the end of burn-in (display trim only).
const ROLLW = 5                       # rolling-mean window, in observations
const BETWINT = 20                    # betweenness measurement interval, periods
const TSTART = 30                     # displayed axes start here; data never cut
const RHO_COLORS = PUB_RHO_COLORS
const DELTA_COLORS = PUB_DELTA_COLORS

const FD = JLD2.load(
    normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figure_data.jld2"))
)["figdata"]
const REPORTING_PROVENANCE = reporting_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."));
    allowed_dirty_paths=(MANUSCRIPT_ITERATION_PATHS..., "scripts/paper/figures.jl"),
)
const DATA_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE, FD["meta"]["analysis_git_commit"]; artifact="main figure data"
)
FD["meta"]["analysis_source_clean"] == true ||
    error("main figure data were extracted from dirty analysis sources")
isfile(INFORMATION_FIGDATA) ||
    error("missing Ridge ablation figure data: $INFORMATION_FIGDATA")
const INFORMATION_FD = JLD2.load(INFORMATION_FIGDATA)["figdata"]
const INFORMATION_META = INFORMATION_FD["meta"]
const INFORMATION_ANALYSIS_COMMIT = validate_analysis_commit(
    REPORTING_PROVENANCE,
    INFORMATION_META["analysis_git_commit"];
    artifact="Ridge ablation figure data",
)
INFORMATION_ANALYSIS_COMMIT == DATA_ANALYSIS_COMMIT ||
    error("main and Ridge ablation figure data use different analysis commits")
INFORMATION_META["analysis_source_clean"] == true ||
    error("Ridge ablation figure data were extracted from dirty analysis sources")
const PER = FD["period"]
const SER = FD["series"]
const SER_SEEDS = FD["series_seed_values"]
const TEND = maximum(PER)
const TIME_TICK_STEP = TEND <= 250 ? 50 : 100
const TIME_TICKS = TIME_TICK_STEP:TIME_TICK_STEP:TEND
const ADV_MARKER_SIZE = 10
function savefig(fname, fig)
    (save(joinpath(OUT, fname), fig; px_per_unit=PXU); println("  $fname done"))
end

function summarized_series(key; measured=false)
    indices = if measured
        [index for index in eachindex(PER) if PER[index] % BETWINT == 0]
    else
        collect(eachindex(PER))
    end
    raw = SER_SEEDS[key][indices, :]
    smoothed = reduce(
        hcat, (rolling_mean(view(raw, :, seed_index), ROLLW) for seed_index in axes(raw, 2))
    )
    summaries = [
        monte_carlo_interval(view(smoothed, period_index, :)) for
        period_index in axes(smoothed, 1)
    ]
    return (
        PER[indices],
        [summary.mean for summary in summaries],
        [summary.lower for summary in summaries],
        [summary.upper for summary in summaries],
    )
end

function draw_interval_series!(
    axis, series, color; label=nothing, points=false, linestyle=:solid, linewidth=2.2
)
    x, estimate, lower, upper = series
    band!(axis, x, lower, upper; color=(color, 0.16))
    keywords = isnothing(label) ? (;) : (; label)
    if points
        scatterlines!(
            axis, x, estimate; color, linewidth, markersize=6, linestyle, keywords...
        )
    else
        lines!(axis, x, estimate; color, linewidth, linestyle, keywords...)
    end
    return nothing
end

function information_sources()
    models = (
        (key="pair", label="Full pair\nmodel", color="#333A40"),
        (key="size_matched", label="Fewer\nobservations", color="#BC9331"),
        (key="single_principal", label="One party\nonly", color="#399775"),
        (key="additive", label="No pair\ninteractions", color="#377DA5"),
    )
    variants = models[2:end]
    conditions = INFORMATION_FD["conditions"]
    length(conditions) == INFORMATION_META["n_conditions"] ||
        error("Ridge ablation condition count does not match its metadata")
    baseline = only(
        filter(
            condition -> condition["result_reldir"] == INFORMATION_META["baseline_reldir"],
            conditions,
        ),
    )
    length(baseline["seeds"]) == 50 || error("expected 50 baseline ablation seeds")
    all(
        length(condition["seeds"]) ==
        (condition["result_reldir"] == INFORMATION_META["baseline_reldir"] ? 50 : 20) for
        condition in conditions
    ) || error("unexpected Ridge ablation seed plan")

    pair_baseline = baseline["rank_gaps"]["pair"]
    baseline_intervals = [
        paired_monte_carlo_interval(pair_baseline, baseline["rank_gaps"][model.key]) for
        model in variants
    ]
    condition_values = Dict(
        model.key => [
            monte_carlo_interval(condition["rank_gaps"][model.key]).mean for
            condition in conditions
        ] for model in models
    )

    fig = Figure(; size=(1200, 520))
    axis_style = (;
        titlesize=TITLE_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
    )
    ax_baseline = Axis(
        fig[1, 1];
        title="A. Effects at baseline",
        ylabel="Change in ranking advantage\nfrom the full pair model",
        xticks=(1:length(variants), [model.label for model in variants]),
        limits=((0.5, length(variants) + 0.5), nothing),
        axis_style...,
    )
    hlines!(ax_baseline, [0.0]; color=:gray55, linestyle=:dash, linewidth=1.4)
    for (index, (model, interval)) in enumerate(zip(variants, baseline_intervals))
        rangebars!(
            ax_baseline,
            [index],
            [interval.lower],
            [interval.upper];
            color=model.color,
            linewidth=1.6,
            whiskerwidth=12,
        )
        scatter!(
            ax_baseline,
            [index],
            [interval.mean];
            color=model.color,
            marker=:diamond,
            markersize=15,
            strokecolor=:gray20,
            strokewidth=0.5,
        )
    end

    ax_conditions = Axis(
        fig[1, 2];
        title="B. Ranking advantage across regimes",
        ylabel="Ranking advantage\n(broker minus principal)",
        xticks=(1:length(models), [model.label for model in models]),
        limits=((0.5, length(models) + 0.5), nothing),
        axis_style...,
    )
    hlines!(ax_conditions, [0.0]; color=:gray55, linestyle=:dash, linewidth=1.4)
    for condition_index in eachindex(conditions)
        lines!(
            ax_conditions,
            1:length(models),
            [condition_values[model.key][condition_index] for model in models];
            color=(:gray45, 0.20),
            linewidth=0.8,
        )
    end
    for (index, model) in enumerate(models)
        values = condition_values[model.key]
        scatter!(
            ax_conditions,
            fill(index, length(values)),
            values;
            color=(model.color, 0.58),
            markersize=6,
        )
        scatter!(
            ax_conditions,
            [index],
            [median(values)];
            color=model.color,
            marker=:diamond,
            markersize=16,
            strokecolor=:gray20,
            strokewidth=0.6,
        )
    end
    colgap!(fig.layout, 38)
    colsize!(fig.layout, 1, Relative(0.45))
    savefig("information_sources.png", fig)
    return nothing
end

"""Read degree intervals from the matching retained audit and verify shared outcomes."""
function principal_degree_intervals(cells)
    audit_dir = normpath(joinpath(OUT, "..", "convergence"))
    metadata = Dict(
        Tuple(split(line, '='; limit=2)) for
        line in readlines(joinpath(audit_dir, "summary.txt"))
    )
    audit_commit = validate_analysis_commit(
        REPORTING_PROVENANCE, metadata["analysis_commit"]; artifact="condition audit"
    )
    audit_commit == DATA_ANALYSIS_COMMIT || error("condition audit analysis commit mismatch")
    metadata["analysis_source_clean"] == "true" || error("condition audit used dirty sources")
    metadata["manifest"] == FD["meta"]["manifest_hash"] ||
        error("condition audit manifest mismatch")

    lines = readlines(joinpath(audit_dir, "condition_audit.tsv"))
    header = split(first(lines), '\t')
    rows = [Dict(zip(header, split(line, '\t'))) for line in lines[2:end]]
    audit = Dict((row["outcome"], row["result_reldir"]) => row for row in rows)
    length(audit) == length(rows) || error("duplicate condition audit rows")
    interval(row) = (
        mean=parse(Float64, row["estimate"]),
        lower=parse(Float64, row["ci_lower"]),
        upper=parse(Float64, row["ci_upper"]),
    )
    return map(cells) do cell
        n_seeds = length(cell["seed_values"]["betw"])
        for (outcome, key) in (("betweenness", "betw"), ("access_fraction", "access"))
            row = audit[(outcome, cell["rel"])]
            parse(Int, row["n_seeds"]) == n_seeds ||
                error("condition audit seed-count mismatch")
            saved = interval(row)
            expected = monte_carlo_interval(cell["seed_values"][key])
            all(
                isapprox(
                    getproperty(saved, field), getproperty(expected, field);
                    atol=1e-12, rtol=1e-10,
                ) for field in (:mean, :lower, :upper)
            ) || error("condition audit does not reproduce $key for $(cell["rel"])")
        end
        row = audit[("mean_degree", cell["rel"])]
        parse(Int, row["n_seeds"]) == n_seeds || error("degree seed-count mismatch")
        degree = interval(row)
        all(isfinite, degree) && 0 <= degree.lower <= degree.mean <= degree.upper ||
            error("invalid principal-degree interval for $(cell["rel"])")
        degree
    end
end

# ── Position: centrality against principal degree and access across rho x eta ──
function centrality_and_access()
    cells = FD["rho_eta_cells"]
    degree_intervals = principal_degree_intervals(cells)
    betweenness = [c["betw"] for c in cells]
    access = [c["access"] for c in cells]
    rho = Float64[c["rho"] for c in cells]
    eta = Float64[c["eta"] for c in cells]
    eta_values = sort(unique(eta))
    rho_values = sort(unique(rho))
    eta_colors = PUB_ETA_COLORS
    rho_markers = PUB_RHO_MARKERS
    access_intervals = [monte_carlo_interval(c["seed_values"]["access"]) for c in cells]
    centrality_intervals = [monte_carlo_interval(c["seed_values"]["betw"]) for c in cells]
    fig = Figure(; size=(1200, 610))
    panels = (
        (
            "A. Principal connectivity", "Mean principal degree",
            [interval.mean for interval in degree_intervals], degree_intervals,
        ),
        ("B. Bridging behavior", "Access fraction", access, access_intervals),
    )
    for (column, (title, xlabel, estimates, intervals)) in enumerate(panels)
        axis = Axis(
            fig[1, column]; title, xlabel, ylabel="Broker betweenness",
            limits=(nothing, (0, 1)),
        )
        for value in eta_values
            indices = sort(findall(eta .== value); by=index -> rho[index])
            xs = estimates[indices]
            ys = betweenness[indices]
            rhos = rho[indices]
            xintervals = intervals[indices]
            yintervals = centrality_intervals[indices]
            errorbars!(
                axis,
                xs,
                ys,
                xs .- [interval.lower for interval in xintervals],
                [interval.upper for interval in xintervals] .- xs;
                direction=:x,
                color=(eta_colors[value], 0.45),
                whiskerwidth=5,
            )
            errorbars!(
                axis,
                xs,
                ys,
                ys .- [interval.lower for interval in yintervals],
                [interval.upper for interval in yintervals] .- ys;
                color=(eta_colors[value], 0.45),
                whiskerwidth=5,
            )
            interior = findall(rhos .< 1.0)
            lines!(axis, xs[interior], ys[interior]; color=eta_colors[value], linewidth=2.2)
            for (r, x, y) in zip(rhos, xs, ys)
                scatter!(
                    axis,
                    [x],
                    [y];
                    marker=rho_markers[r],
                    color=eta_colors[value],
                    markersize=13,
                    strokecolor=:black,
                    strokewidth=0.5,
                )
            end
        end
    end
    eta_elements = [
        LineElement(; color=eta_colors[value], linewidth=3) for value in eta_values
    ]
    rho_elements = [
        MarkerElement(;
            marker=rho_markers[value],
            color=:gray60,
            strokecolor=:black,
            strokewidth=0.5,
            markersize=11,
        ) for value in rho_values
    ]
    Legend(
        fig[2, 1:2], [eta_elements, rho_elements],
        [string.(eta_values), string.(rho_values)],
        ["Turnover (η)", "General-quality share (ρ)"];
        PUB_LEGEND..., nbanks=2, groupgap=60,
    )
    colgap!(fig.layout, 42)
    rowgap!(fig.layout, 18)
    savefig("centrality_and_access.png", fig)
end

# ── Matching grid: six outcomes vs rho, with rho = 1 as a separate boundary ──
function matching_grid()
    gcells = FD["grid_cells"]
    dls = sort(unique([c["delta"] for c in gcells]))
    cells = Dict((c["rho"], c["delta"]) => c for c in gcells)
    boundary_cells = [c for c in gcells if c["rho"] == 1.0]
    length(unique(c["rel"] for c in boundary_cells)) == 1 ||
        error("rho = 1 grid coordinates do not share one effective realization")
    boundary_cell = first(boundary_cells)
    # Structural panels start at zero; access uses the requested 0-0.5 scale.
    # Prediction and output panels retain their data-driven limits.
    keys = [
        "Betweenness centrality" "Broker rank correlation" "Rank correlation gap";
        "Access fraction" "Principal rank correlation" "Output gap q"
    ]
    labels = [
        "A. Broker centrality" "B. Broker ranking accuracy" "C. Ranking advantage";
        "D. Access fraction" "E. Principal ranking accuracy" "F. Match-output difference"
    ]
    ylabels = [
        "Betweenness" "Rank correlation" "Broker minus principal";
        "Share of brokered placements" "Rank correlation" "Broker minus self-search"
    ]
    fig = Figure(; size=(1280, 780))
    difficulty_legend!(fig[0, 1:3], dls)
    for rr in 1:2, cc in 1:3
        key = keys[rr, cc]
        ttl = labels[rr, cc]
        ax = Axis(
            fig[rr, cc];
            title=ttl,
            xlabel=rr == 2 ? PUB_RHO_LABEL : "",
            ylabel=ylabels[rr, cc],
            xticks=([0, 0.5, 1], ["0", "0.5", "1"]),
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
            limits=cc == 1 ? (nothing, (0, rr == 1 ? 1.02 : 0.5)) : (nothing, nothing),
        )
        for d in dls
            pts = sort(
                [
                    let interval = monte_carlo_interval(c["outcome_seed_values"][key])
                        (
                            rho=r,
                            mean=interval.mean,
                            lower=interval.lower,
                            upper=interval.upper,
                        )
                    end for ((r, dd), c) in cells if dd == d && r < 1.0
                ];
                by=point -> point.rho,
            )
            rangebars!(
                ax,
                [point.rho for point in pts],
                [point.lower for point in pts],
                [point.upper for point in pts];
                color=(DELTA_COLORS[d], 0.72),
                linewidth=1.2,
                whiskerwidth=8,
            )
            scatterlines!(
                ax,
                [point.rho for point in pts],
                [point.mean for point in pts];
                color=DELTA_COLORS[d],
                marker=PUB_DELTA_MARKERS[d],
                linewidth=2.0,
                markersize=10,
                strokewidth=0.4,
                strokecolor=:gray30,
                label="δ = $d",
            )
        end
        boundary_interval = monte_carlo_interval(boundary_cell["outcome_seed_values"][key])
        rangebars!(
            ax,
            [1.0],
            [boundary_interval.lower],
            [boundary_interval.upper];
            color=(:gray25, 0.8),
            linewidth=1.2,
            whiskerwidth=8,
        )
        scatter!(
            ax,
            [1.0],
            [boundary_interval.mean];
            color=:gray25,
            marker=:diamond,
            markersize=11,
            strokewidth=0.4,
            strokecolor=:gray20,
            label="ρ = 1 boundary",
        )
    end
    colgap!(fig.layout, 26)
    rowgap!(fig.layout, 26)
    savefig("matching_grid.png", fig)
end

# ── Advantage: structural measures vs informational/output differences (4 panels) ──
function structural_advantage()
    bc = FD["regime_cells"]
    rho = [c["rho"] for c in bc]
    bw = [c["betw"] for c in bc];
    ac = [c["access"] for c in bc]
    rg = [c["rankgap"] for c in bc];
    qg = [c["qgap"] for c in bc]
    xs = [("Broker betweenness", bw), ("Access fraction", ac)]
    ys = [("Rank correlation:\nbroker minus principal", rg),
          ("Match output:\nbroker minus self-search", qg)]
    titles = ["A. Ranking advantage" "B. Ranking advantage";
              "C. Match-output difference" "D. Match-output difference"]
    fig = Figure(; size=(1200, 860))
    composition_legend!(fig[0, 1:2], sort(unique(rho)))
    for (ri, (ylab, yv)) in enumerate(ys), (ci, (xlab, xv)) in enumerate(xs)
        ax = Axis(
            fig[ri, ci];
            xlabel=xlab,
            ylabel=ci == 1 ? ylab : "",
            title=titles[ri, ci],
            titlesize=TITLE_FS,
            xlabelsize=LABEL_FS,
            ylabelsize=LABEL_FS,
            xticklabelsize=TICK_FS,
            yticklabelsize=TICK_FS,
        )
        for rv in sort(unique(rho))
            mm = rho .== rv
            scatter!(
                ax,
                xv[mm],
                yv[mm];
                color=(RHO_COLORS[rv], 0.8),
                marker=PUB_RHO_MARKERS[rv],
                markersize=ADV_MARKER_SIZE,
                strokewidth=0.3,
                strokecolor=:gray30,
            )
        end
    end
    colgap!(fig.layout, 34)
    rowgap!(fig.layout, 24)
    savefig("structural_advantage.png", fig)
end

# ── Baseline dynamics, placed first in the results section ──
function baseline_dynamics()
    # Each seed is smoothed first. The line and pointwise interval are then
    # computed across seeds; display trimming remains axis-only.
    ot(key) = summarized_series(key)
    otb() = summarized_series("betweenness"; measured=true)
    yrange(ss...) = (
        v=filter(
            !isnan,
            vcat((vcat(s[3][s[1] .>= TSTART], s[4][s[1] .>= TSTART]) for s in ss)...),
        );   # displayed window only
        hi=maximum(v);
        (0, hi + 0.06 * (hi + eps()))
    )   # all y-axes start at zero
    mpa = ot("mpa")
    dmean, dmedian = ot("mean_degree"), ot("median_degree")
    betweenness = otb()
    access = ot("access")
    outsourcing = ot("outsourcing")
    yl = (
        yrange(mpa), yrange(dmean, dmedian), yrange(betweenness), yrange(access), (0, 1.02)
    )

    fig = Figure(; size=(1200, 660))
    T4, L4, K4 = TITLE_FS, LABEL_FS, TICK_FS
    mk(row, column, ttl, ylim) = Axis(
        fig[row, column];
        title=ttl,
        xlabel="Period",
        xticks=TIME_TICKS,
        titlesize=T4,
        xlabelsize=L4,
        xticklabelsize=K4,
        yticklabelsize=K4,
        limits=((TSTART, TEND + 1), ylim),
    )
    # a `label` keyword is only passed when a label is requested: a plot with
    # label="" would still register a (blank) legend entry
    drw!(ax, s, col; lbl=nothing, ls=:solid, pts=false, lw=2.2) = draw_interval_series!(
        ax, s, col; label=lbl, points=pts, linestyle=ls, linewidth=lw
    )
    let a = mk(1, 1, "A. Outsourcing rate", yl[5])
        drw!(a, outsourcing, PUB_BROKER)
    end
    let a = mk(1, 2, "B. Matches per principal", yl[1])
        drw!(a, mpa, PUB_PRINCIPAL)
    end
    let a = mk(1, 3, "C. Principal degree", yl[2])
        drw!(a, dmean, PUB_PRINCIPAL; lbl="Mean")
        drw!(a, dmedian, :gray55; lbl="Median", ls=:dash)
        axislegend(a; position=:cb, orientation=:horizontal, labelsize=18,
            framevisible=true, framecolor=:gray65, framewidth=0.8,
            backgroundcolor=:white, padding=(8, 8, 5, 5))
    end
    let a = mk(2, 1, "D. Access fraction", yl[4])
        drw!(a, access, PUB_ACCESS)
    end
    let a = mk(2, 2, "E. Broker betweenness", yl[3])
        drw!(a, betweenness, PUB_CENTRALITY; pts=true)
    end
    for column in 1:3
        colsize!(fig.layout, column, Auto(1))
    end
    colgap!(fig.layout, 24)
    rowgap!(fig.layout, 26)
    savefig("baseline_dynamics.png", fig)
end

foreach(
    function_name -> function_name(),
    (
        baseline_dynamics,
        information_sources,
        matching_grid,
        centrality_and_access,
        structural_advantage,
    ),
)
# emit the display-convention keys quoted by the captions (paper/captions.tex)
open(normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figmeta.tex")), "w") do io
    println(
        io, "% figmeta.tex: generated by scripts/paper/figures.jl. Do not edit by hand."
    )
    println(io, "% Data analysis commit: $DATA_ANALYSIS_COMMIT")
    println(io, "% Rendering commit: $(REPORTING_PROVENANCE.commit)")
    println(
        io,
        "% Display conventions used to render output/main/figures/, quoted in captions via \\pv keys.",
    )
    println(io, "\\pvDefine{rollWin}{$ROLLW}")
    println(io, "\\pvDefine{betwInterval}{$BETWINT}")
    println(io, "\\pvDefine{axisStart}{$TSTART}")
end
println("main figures done (+ figmeta.tex)")
