"""
    scripts/paper/figures.jl

Figure assets for the paper's results section. TeX figure numbers follow their
placement in `paper/section_source.tex`; the asset filenames remain stable.
Reads the retained datasets `output/main/figure_data.jld2`,
`output/ridge/{paired,ablations}/figure_data.jld2`, the NN and Ridge access-window
archives, the paired Ridge condition comparison, and the condition audit under
`output/main/convergence/`, so figures render locally without the sweeps. No
simulation; no hard-coded results
(literal constants are display conventions only). Outputs print-resolution PNGs
to output/main/figures/ and the display-convention keys to output/main/figmeta.tex.

  assessment_not_access
                      broker use, access, and assessment accuracy
  information_sources
                      sources of the broker's ranking advantage
  matching_grid       six outcomes across the rho x delta grid, lines per delta
                      with 95% Monte Carlo interval whiskers
  centrality_and_access
                      centrality against principal degree and access across regimes
  structural_advantage
                      structural measures vs informational/output gaps

Usage: julia --project --threads=auto scripts/paper/figures.jl
Subsection-1 figure only:
    julia --project --threads=auto scripts/paper/figures.jl --assessment-not-access
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))   # CairoMakie, COL_*, FS, LEG_KW, rolling_mean
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
using JLD2
using SHA: sha256
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
const REPORTING_PROVENANCE = manuscript_git_provenance(
    normpath(joinpath(@__DIR__, "..", "..")),
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
        yminorticks=IntervalsBetween(2),
        yminorgridvisible=true,
        yminorgridcolor=(:black, 0.045),
        yminorgridwidth=0.6,
    )
    ax_baseline = Axis(
        fig[1, 1];
        title="A. Effects at baseline",
        ylabel="Change in ranking advantage\nfrom the full pair model",
        yticks=-1.0:0.1:0.0,
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
        yticks=-1.0:0.2:1.0,
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

"""Read the retained condition audit after checking analysis and sweep provenance."""
function condition_audit()
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
    return audit
end

"""Read degree intervals from the matching retained audit and verify shared outcomes."""
function principal_degree_intervals(cells)
    audit = condition_audit()
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
    # Structural panels start at zero; access uses the requested 0-0.25 scale.
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
            limits=cc == 1 ? (nothing, (0, rr == 1 ? 1.02 : 0.25)) : (nothing, nothing),
        )
        rr == 2 && cc == 1 && (ax.yticks = 0.0:0.05:0.25)
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

# ── Retained baseline-only diagnostic ──
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

"""Summarize smoothed baseline trajectories without treating periods as replications."""
function assessment_trajectory(data, key)
    raw = data["series_seed_values"][key]
    size(raw, 2) == length(data["baseline_seeds"]) || error("baseline seed mismatch")
    smoothed = reduce(
        hcat, (rolling_mean(view(raw, :, seed), ROLLW) for seed in axes(raw, 2))
    )
    summaries = [
        monte_carlo_interval(view(smoothed, period, :)) for period in axes(raw, 1)
    ]
    all(
        summary.n == size(raw, 2) for (period, summary) in
        zip(data["period"], summaries) if period >= TSTART
    ) || error("missing baseline observations in the displayed window")
    return (
        data["period"],
        [s.mean for s in summaries],
        [s.lower for s in summaries],
        [s.upper for s in summaries],
    )
end

"""Compute early/late access means from raw counts and verify the retained late values."""
function access_window_summaries(figure_data, path)
    cells = figure_data["regime_cells"]
    data = JLD2.load(path)
    meta = data["metadata"]
    meta["manifest_hash"] == figure_data["meta"]["manifest_hash"] || error("access-window sweep mismatch")
    meta["schema_version"] == figure_data["meta"]["schema_version"] || error("access-window schema mismatch")
    validate_analysis_commit(
        REPORTING_PROVENANCE, meta["repository_commit"]; artifact="access-window reader"
    )
    bytes2hex(sha256(meta["exporter_source"])) == meta["exporter_sha256"] ||
        error("access-window exporter source hash mismatch")
    bytes2hex(sha256(read(joinpath(@__DIR__, "access_windows.jl")))) == meta["exporter_sha256"] ||
        error("access-window exporter differs from its retained source")
    bytes2hex(sha256(read(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl")))) ==
        meta["reader_sha256"] || error("access-window sweep reader differs")
    early_window, late_window = meta["early_window"], meta["late_window"]
    early_window == (51, 70) && late_window == (TEND - 19, TEND) ||
        error("access windows differ from the manuscript")
    early_mask = early_window[1] .<= data["periods"] .<= early_window[2]
    late_mask = late_window[1] .<= data["periods"] .<= late_window[2]
    count(early_mask) == count(late_mask) == 20 || error("incomplete access windows")
    regimes = Dict(regime["rel"] => regime for regime in data["regimes"])
    Set(keys(regimes)) == Set(cell["rel"] for cell in cells) || error("access regime coverage differs")
    summaries = map(cells) do cell
        regime = regimes[cell["rel"]]
        regime["seeds"] == cell["seeds"] || error("access-window seeds differ")
        expected_size = (length(data["periods"]), length(cell["seeds"]))
        access, assessment = regime["access_count"], regime["assessment_count"]
        size(access) == size(assessment) == expected_size || error("access-window dimensions differ")
        all(>=(0), access) && all(>=(0), assessment) || error("negative placement counts")
        fractions = access ./ (access .+ assessment)
        window_values(mask) = [
            mean(finite_seed_values(view(fractions, mask, seed))) for seed in axes(fractions, 2)
        ]
        early, late = window_values(early_mask), window_values(late_mask)
        all(isfinite, [early; late]) || error("missing seed-level access mean")
        all(isapprox.(late, cell["seed_values"]["access"]; atol=1e-12, rtol=1e-10)) ||
            error("raw access counts do not reproduce retained late means")
        (; early=mean(early), late=mean(late), change=monte_carlo_interval(late .- early))
    end
    baseline = only(findall(cell -> cell["rel"] == "oat/rho=0.5", cells))
    baseline_mask = early_window[1] .<= figure_data["period"] .<= early_window[2]
    expected_early = mean(figure_data["series_seed_values"]["access"][baseline_mask, :])
    isapprox(summaries[baseline].early, expected_early; atol=1e-12, rtol=1e-10) ||
        error("early access does not reproduce the retained baseline")
    return (; summaries, early_window, late_window)
end

"""Label an equality line using its displayed angle and a small perpendicular gap."""
function equality_label!(axis, position, label)
    angle = lift(axis.scene.viewport, axis.finallimits) do area, limits
        atan(area.widths[2] / limits.widths[2], area.widths[1] / limits.widths[1])
    end
    offset = lift(angle) do rotation
        Vec2f(-5sin(rotation), 5cos(rotation))
    end
    text!(axis, position, position; text=label, rotation=angle, offset,
        color=:gray45, fontsize=FOOTER_FS, align=(:center, :bottom))
    return nothing
end

"""Read Ridge late means, checking provenance and shared retained outcomes."""
function ridge_condition_means(ridge, nn_audit)
    root = normpath(joinpath(OUT, "..", "..", "ridge", "paired", "analysis"))
    meta = Dict(Tuple(split(line, '='; limit=2)) for
        line in readlines(joinpath(root, "provenance.txt")))
    commit = validate_analysis_commit(
        REPORTING_PROVENANCE, meta["analysis_commit"]; artifact="Ridge condition comparison"
    )
    meta["analysis_source_clean"] == "true" || error("Ridge comparison used dirty sources")
    meta["late_periods"] == "$(TEND - 19):$TEND" || error("Ridge comparison window differs")
    for (prefix, data) in (("nn", FD), ("ridge", ridge))
        meta["$(prefix)_manifest"] == data["meta"]["manifest_hash"] ||
            error("$prefix comparison manifest differs")
        parse(Int, meta["$(prefix)_schema"]) == data["meta"]["schema_version"] ||
            error("$prefix comparison schema differs")
        for cell in data["regime_cells"]
            key = cell["rel"] == "oat/rho=0.5" ? "baseline_comparison_seeds" : "other_comparison_seeds"
            parse.(Int, split(meta[key], ',')) == cell["seeds"] || error("comparison seeds differ")
        end
    end
    lines = readlines(joinpath(root, "condition_comparison.tsv"))
    header = split(first(lines), '\t')
    rows = [Dict(zip(header, split(line, '\t'))) for line in lines[2:end]]
    by_rel = Dict(row["result_reldir"] => row for row in rows)
    cells = ridge["regime_cells"]
    length(by_rel) == length(rows) || error("duplicate Ridge comparison rows")
    Set(keys(by_rel)) == Set(cell["rel"] for cell in cells) || error("Ridge comparison coverage differs")
    value(cell, key) = parse(Float64, by_rel[cell["rel"]][key])
    for cell in cells
        for metric in ("outsourcing_rate", "access_fraction", "agent_holdout_rank",
            "broker_holdout_rank", "rank_gap")
            parse(Int, by_rel[cell["rel"]]["delta_n_$metric"]) == length(cell["seeds"]) ||
                error("Ridge comparison has missing observations")
            isapprox(value(cell, "nn_$metric"),
                parse(Float64, nn_audit[(metric, cell["rel"])]["estimate"]);
                atol=1e-12, rtol=1e-10) || error("NN comparison differs from its retained audit")
        end
        for (metric, key) in (("access_fraction", "access"), ("rank_gap", "rankgap"))
            isapprox(value(cell, "ridge_$metric"), mean(cell["seed_values"][key]);
                atol=1e-12, rtol=1e-10) || error("Ridge comparison differs from retained $metric")
        end
        isapprox(value(cell, "ridge_broker_holdout_rank") - value(cell, "ridge_agent_holdout_rank"),
            value(cell, "ridge_rank_gap"); atol=1e-12, rtol=1e-10) ||
            error("Ridge ranking means do not reproduce the paired gap")
    end
    baseline = only(filter(cell -> cell["rel"] == "oat/rho=0.5", cells))
    late_mask = ridge["period"] .>= TEND - 19
    isapprox(value(baseline, "ridge_outsourcing_rate"),
        mean(ridge["series_seed_values"]["outsourcing"][late_mask, :]);
        atol=1e-12, rtol=1e-10) || error("Ridge baseline outsourcing differs")
    return (;
        outsourcing=[value(cell, "ridge_outsourcing_rate") for cell in cells],
        principal_rank=[value(cell, "ridge_agent_holdout_rank") for cell in cells],
        broker_rank=[value(cell, "ridge_broker_holdout_rank") for cell in cells],
        commit,
    )
end

"""Plot regime means and identify the baseline with a color-matched diamond."""
function regime_points!(axis, series)
    for (x, y, _, color) in series
        scatter!(axis, x, y; color=(color, 0.65), markersize=9,
            strokecolor=:white, strokewidth=0.5)
    end
    for (x, y, baseline, color) in series
        scatter!(axis, [x[baseline]], [y[baseline]];
            color, marker=:diamond, markersize=15, strokecolor=:gray15, strokewidth=1.0)
    end
    return nothing
end

"""
Smooth equally weighted regime means with Gaussian kernels reflected at the
measure's bounds. Both learners share a pooled Silverman bandwidth and density scale.
"""
function regime_densities(groups, bounds)
    lower, upper = bounds
    pooled = reduce(vcat, groups)
    all(x -> isfinite(x) && lower <= x <= upper, pooled) ||
        error("marginal density values fall outside the measure's bounds")
    bandwidth = Makie.KernelDensity.default_bandwidth(pooled)
    bandwidth > 0 || error("invalid marginal density bandwidth")
    grid = range(lower, upper; length=1025)
    kernel(z) = exp(-0.5 * (z / bandwidth)^2)
    curves = map(groups) do values
        density = [
            sum(kernel(x - value) + kernel(x - (2lower - value)) +
                kernel(x - (2upper - value)) for value in values) /
            (length(values) * bandwidth * sqrt(2pi)) for x in grid
        ]
        area = step(grid) * (sum(density) - (first(density) + last(density)) / 2)
        isapprox(area, 1; atol=1e-4) || error("marginal density does not integrate to one")
        density
    end
    return (; grid, curves)
end

"""Add a scatterplot with translucent, axis-aligned top and right marginal densities."""
function regime_density_panel!(slot, title, series; xbounds=(0.0, 1.0),
    ybounds=(0.0, 1.0), kwargs...)
    layout = GridLayout(slot)
    Label(layout[1, 1:2], title; font=:bold, fontsize=TITLE_FS,
        halign=:left, tellwidth=false)
    top = Axis(layout[2, 1])
    axis = Axis(layout[3, 1]; kwargs...)
    right = Axis(layout[3, 2])
    for marginal in (top, right)
        hidedecorations!(marginal)
        hidespines!(marginal)
    end
    xdensity = regime_densities([entry[1] for entry in series], xbounds)
    ydensity = regime_densities([entry[2] for entry in series], ybounds)
    # Draw fills before outlines so neither learner's curve is obscured.
    for (index, (_, _, _, color)) in enumerate(series)
        band!(top, xdensity.grid, zeros(length(xdensity.grid)), xdensity.curves[index];
            color=(color, 0.25))
        band!(right, ydensity.grid, zeros(length(ydensity.grid)), ydensity.curves[index];
            direction=:y, color=(color, 0.25))
    end
    for (index, (_, _, _, color)) in enumerate(series)
        lines!(top, xdensity.grid, xdensity.curves[index]; color=(color, 0.85), linewidth=1.4)
        lines!(right, ydensity.curves[index], ydensity.grid; color=(color, 0.85), linewidth=1.4)
    end
    linkxaxes!(axis, top)
    linkyaxes!(axis, right)
    xlimits, ylimits = kwargs[:limits]
    limits!(axis, xlimits, ylimits)
    xlims!(top, xlimits)
    ylims!(right, ylimits)
    ylims!(top, 0, 1.08 * maximum(maximum, xdensity.curves))
    xlims!(right, 0, 1.08 * maximum(maximum, ydensity.curves))
    rowsize!(layout, 2, 46)
    colsize!(layout, 2, 46)
    rowgap!(layout, 1, 14)
    rowgap!(layout, 2, 6)
    colgap!(layout, 6)
    regime_points!(axis, series)
    return axis
end

"""Render Figure 1 using retained NN/Ridge series and audited late means."""
function assessment_not_access()
    ridge_path = normpath(joinpath(OUT, "..", "..", "ridge", "paired", "figure_data.jld2"))
    ridge = JLD2.load(ridge_path)["figdata"]
    ridge_meta = ridge["meta"]
    ridge_meta["analysis_source_clean"] == true || error("Ridge analysis sources were dirty")
    ridge_meta["learning_model"] == "ridge" || error("expected Ridge learning")
    ridge_commit = validate_analysis_commit(
        REPORTING_PROVENANCE, ridge_meta["analysis_git_commit"]; artifact="Ridge figure data"
    )
    ridge["period"] == PER || error("baseline time axes differ")
    ridge["baseline_seeds"] == FD["baseline_seeds"] || error("baseline seed plans differ")
    Set(keys(ridge_meta["condition_seed_counts"])) ==
        Set(keys(FD["meta"]["condition_seed_counts"])) || error("regime coverage differs")

    audit = condition_audit()
    cells = FD["regime_cells"]
    length(cells) == length(FD["meta"]["condition_seed_counts"]) ||
        error("missing effective regimes")
    row(cell, outcome) = audit[(outcome, cell["rel"])]
    value(cell, outcome) = parse(Float64, row(cell, outcome)["estimate"])
    for cell in cells
        for outcome in ("outsourcing_rate", "access_fraction", "broker_holdout_rank",
            "agent_holdout_rank", "rank_gap")
            parse(Int, row(cell, outcome)["n_seeds"]) == length(cell["seeds"]) ||
                error("audit seed count differs for $(cell["rel"]), $outcome")
        end
        for (outcome, key) in (("access_fraction", "access"), ("rank_gap", "rankgap"))
            interval = monte_carlo_interval(cell["seed_values"][key])
            for (column, field) in (("estimate", :mean), ("ci_lower", :lower), ("ci_upper", :upper))
                isapprox(
                    parse(Float64, row(cell, outcome)[column]), getproperty(interval, field);
                    atol=1e-12, rtol=1e-10,
                ) || error("audit does not reproduce $outcome for $(cell["rel"])")
            end
        end
        isapprox(
            value(cell, "broker_holdout_rank") - value(cell, "agent_holdout_rank"),
            value(cell, "rank_gap"); atol=1e-12, rtol=1e-10,
        ) || error("ranking means do not reproduce the paired gap")
    end
    baseline = only(findall(cell -> cell["rel"] == "oat/rho=0.5", cells))
    outsourcing = [value(cell, "outsourcing_rate") for cell in cells]
    access = [value(cell, "access_fraction") for cell in cells]
    principal_rank = [value(cell, "agent_holdout_rank") for cell in cells]
    broker_rank = [value(cell, "broker_holdout_rank") for cell in cells]
    all(isfinite, [outsourcing; access; principal_rank; broker_rank]) || error("nonfinite means")
    all(x -> 0 <= x <= 1, [outsourcing; principal_rank; broker_rank]) ||
        error("a regime mean falls outside the plotted unit interval")
    all(x -> 0 <= x <= 0.25, access) || error("an access mean falls outside the plotted range")
    windows = access_window_summaries(
        FD, normpath(joinpath(OUT, "..", "access_windows.jld2"))
    )
    ridge_windows = access_window_summaries(
        ridge, joinpath(dirname(ridge_path), "access_windows.jld2")
    )
    (windows.early_window, windows.late_window) ==
        (ridge_windows.early_window, ridge_windows.late_window) || error("access windows differ")
    access_early = [summary.early for summary in windows.summaries]
    ridge_early = [summary.early for summary in ridge_windows.summaries]
    ridge_late = [summary.late for summary in ridge_windows.summaries]
    ridge_baseline = only(findall(cell -> cell["rel"] == "oat/rho=0.5", ridge["regime_cells"]))
    ridge_means = ridge_condition_means(ridge, audit)
    all(x -> isfinite(x) && 0 <= x <= 1,
        [ridge_means.outsourcing; ridge_means.principal_rank; ridge_means.broker_rank]) ||
        error("a Ridge mean falls outside the plotted unit interval")
    all(x -> isfinite(x) && 0 <= x <= 0.25, ridge_late) ||
        error("a Ridge access mean falls outside the plotted range")

    fig = Figure(; size=(1350, 950))
    percent_ticks = (0:0.25:1, string.(0:25:100))
    baseline_layout = GridLayout(fig[1, 1])
    Label(baseline_layout[1, 1], "A. Broker use and access at baseline";
        font=:bold, fontsize=TITLE_FS, halign=:left, tellwidth=false)
    a = Axis(
        baseline_layout[2, 1];
        xlabel="Period", ylabel="Share (%)", xticks=TIME_TICKS, yticks=percent_ticks,
        limits=((TSTART, TEND), (0, 1.025)),
    )
    rowgap!(baseline_layout, 14)
    early_window, late_window = windows.early_window, windows.late_window
    for (window, label) in ((early_window, "Early"), (late_window, "Late"))
        vspan!(a, [window[1]], [window[2]]; color=(:gray40, 0.08))
        text!(a, mean(window), 0.028; text=label, color=:gray35,
            fontsize=FOOTER_FS, align=label == "Late" ? (:right, :bottom) : (:center, :bottom))
    end
    trajectories = [
        (series=assessment_trajectory(data, key), color=color, is_ridge=is_ridge)
        for (data, is_ridge) in ((FD, false), (ridge, true))
        for (key, color) in (("outsourcing", PUB_BROKER), ("access", PUB_ACCESS))
    ]
    for trajectory in trajectories
        x, _, lower, upper = trajectory.series
        band!(a, x, lower, upper;
            color=(trajectory.color, trajectory.is_ridge ? 0.07 : 0.12))
    end
    for trajectory in trajectories
        x, estimate = trajectory.series
        lines!(a, x, estimate; color=trajectory.color,
            linewidth=trajectory.is_ridge ? 1.7 : 3.0,
            linestyle=trajectory.is_ridge ? :dash : :solid)
    end
    text!(a, 235, 0.83; text="Outsourcing rate", color=PUB_BROKER,
        fontsize=LABEL_FS, align=(:left, :center))
    text!(a, 235, 0.24; text="Access fraction", color=PUB_ACCESS,
        fontsize=LABEL_FS, align=(:left, :center))
    axislegend(
        a,
        [LineElement(; color=:gray25, linewidth=3.0),
            LineElement(; color=:gray25, linewidth=1.7, linestyle=:dash)],
        ["Neural network", "Ridge"];
        position=:rc, labelsize=TICK_FS, patchsize=(36, 14),
        framevisible=false, backgroundcolor=(:white, 0.85), margin=(18, 18, 8, 8),
    )

    access_limit = ceil(maximum([access_early; access; ridge_early; ridge_late]) / 0.05) * 0.05
    ridge_color = PUB_DELTA_COLORS[0.25]
    b = regime_density_panel!(
        fig[1, 2], "B. Access declines across regimes", (
            (access_early, access, baseline, PUB_PRINCIPAL),
            (ridge_early, ridge_late, ridge_baseline, ridge_color),
        );
        xlabel="Early-mean access fraction (%)", ylabel="Late-mean access fraction (%)",
        limits=((0, access_limit + 0.005), (0, access_limit + 0.005)),
    )
    access_ticks = 0:0.05:access_limit
    tick_spec = (access_ticks, string.(round.(Int, 100 .* access_ticks)))
    b.xticks = tick_spec
    b.yticks = tick_spec
    limits!(b, 0, access_limit + 0.005, 0, access_limit + 0.005)
    lines!(b, [0, access_limit], [0, access_limit];
        color=:gray60, linestyle=:dash, linewidth=1.2)
    axislegend(
        b, [MarkerElement(; color, marker=:circle, markersize=10)
            for color in (PUB_PRINCIPAL, ridge_color)], ["Neural network", "Ridge"];
        position=:lt, framevisible=false, labelsize=TICK_FS,
        backgroundcolor=(:white, 0.85), padding=(5, 5, 5, 5),
    )
    equality_label!(b, 0.68 * access_limit, "No change")

    c = regime_density_panel!(
        fig[2, 1], "C. Broker use and access across regimes", (
            (outsourcing, access, baseline, PUB_PRINCIPAL),
            (ridge_means.outsourcing, ridge_late, ridge_baseline, ridge_color),
        );
        xlabel="Outsourcing rate (%)", ylabel="Access fraction (%)",
        xticks=percent_ticks, yticks=(0:0.05:0.25, string.(0:5:25)),
        limits=((0, 1.025), (0, 0.255)),
    )
    axislegend(
        c, [MarkerElement(; color=:gray15, marker=:diamond, markersize=12)], ["Baseline"];
        position=:lt, framevisible=false, labelsize=TICK_FS,
        backgroundcolor=(:white, 0.85), padding=(5, 5, 5, 5),
    )

    d = regime_density_panel!(
        fig[2, 2], "D. Assessment accuracy across regimes", (
            (principal_rank, broker_rank, baseline, PUB_PRINCIPAL),
            (ridge_means.principal_rank, ridge_means.broker_rank, ridge_baseline, ridge_color),
        );
        xbounds=(-1.0, 1.0), ybounds=(-1.0, 1.0),
        xlabel="Principal rank correlation", ylabel="Broker rank correlation",
        xticks=0:0.2:1, yticks=0:0.2:1, limits=((0, 1.025), (0, 1.025)),
    )
    lines!(d, [0, 1], [0, 1]; color=:gray60, linestyle=:dash, linewidth=1.2)
    equality_label!(d, 0.62, "Equal accuracy")

    colgap!(fig.layout, 44)
    rowgap!(fig.layout, 30)
    savefig("assessment_not_access.png", fig)
    println("Figure inputs: NN analysis $DATA_ANALYSIS_COMMIT; Ridge analysis $ridge_commit")
    println("Ridge condition comparison: $(ridge_means.commit)")
    println("Rendering commit: $(REPORTING_PROVENANCE.commit); source clean: $(REPORTING_PROVENANCE.source_clean)")
    println("Regimes: $(length(cells)); above half outsourcing: $(count(>(0.5), outsourcing)); below one-fifth access: $(count(<(0.2), access))")
    println("Baseline late means: outsourcing=$(outsourcing[baseline]), access=$(access[baseline]), broker rank=$(broker_rank[baseline]), principal rank=$(principal_rank[baseline])")
    println("Access declines: $(count(s -> s.change.mean < 0, windows.summaries)); paired intervals below zero: $(count(s -> s.change.upper < 0, windows.summaries))")
    println("Ridge access declines: $(count(s -> s.change.mean < 0, ridge_windows.summaries)); paired intervals below zero: $(count(s -> s.change.upper < 0, ridge_windows.summaries))")
    return nothing
end

if ARGS == ["--assessment-not-access"]
    assessment_not_access()
elseif isempty(ARGS)
    foreach(
        function_name -> function_name(),
        (
            assessment_not_access,
            information_sources,
            matching_grid,
            centrality_and_access,
            structural_advantage,
        ),
    )
    # Emit the display conventions quoted by the publication captions.
    open(normpath(joinpath(@__DIR__, "..", "..", "output", "main", "figmeta.tex")), "w") do io
        println(io, "% figmeta.tex: generated by scripts/paper/figures.jl. Do not edit by hand.")
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
else
    error("usage: figures.jl [--assessment-not-access]")
end
