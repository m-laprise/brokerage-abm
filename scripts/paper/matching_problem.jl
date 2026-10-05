"""
Render the matching-problem review figure from retained seed-level estimates.

Prepare once with `--prepare ASSESSMENT_SHARES NN_SHARES NN_LATE_VALUES`.
The Ridge input is output/ridge/ablations/figure_data.jld2. Assessment and NN
inputs are saved-data extracts from the subsection4 reporting directories on
Della. Input hashes and provenance are retained in the compact figure dataset.

Run: julia --project --threads=auto scripts/paper/matching_problem.jl
Output: output/main/figures/matching_problem_2.png

Use --split for the separate main and supplementary drafts. This retains the
original PNG and saves matching_problem_3.png and matching_problem_contributions.png.
Use --nn-vs-ridge-difficulty to save Panel D as NNvsRidge_by_difficulty.png.
Use --nn-grid for the complete NN matching grid in matching_problem_4.png.
Use --nn-difficulty-grid for difficulty on the x-axis in matching_problem_5.png,
omitting ρ = 0.3 and 0.7 from that display.
Use --turnover-grid [EXTRACT] for matching_problem_4t.png. EXTRACT is the output
of matching_turnover_data.jl and is needed only to initialize the retained data.
Use --composition-differences for paired ρ = 0 minus ρ = 0.85 contrasts in
matching_composition_differences.png.
Use --ablation-composition-differences for changes in ablation contributions
from ρ = 1 to ρ = 0 in matching_composition_ablation_differences.png.
"""
module MatchingProblemFigure

using JLD2
using SHA
using Printf: @sprintf
include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_PATH = joinpath(ROOT, "output", "main", "matching_problem_data.jld2")
const FIGURE_PATH = joinpath(ROOT, "output", "main", "figures", "matching_problem_2.png")
const RIDGE_PATH = joinpath(ROOT, "output", "ridge", "ablations", "figure_data.jld2")
const SPLIT_MAIN_PATH = joinpath(ROOT, "output", "main", "figures", "matching_problem_3.png")
const SPLIT_SUPP_PATH = joinpath(ROOT, "output", "supplement", "figures", "matching_problem_contributions.png")
const DIFFICULTY_PATH = joinpath(ROOT, "output", "main", "figures", "NNvsRidge_by_difficulty.png")
const NN_GRID_PATH = joinpath(ROOT, "output", "main", "figures", "matching_problem_4.png")
const NN_DIFFICULTY_GRID_PATH = joinpath(ROOT, "output", "main", "figures", "matching_problem_5.png")
const TURNOVER_GRID_PATH = joinpath(ROOT, "output", "main", "figures", "matching_problem_4t.png")
const COMPOSITION_DIFFERENCES_PATH = joinpath(ROOT, "output", "main", "figures", "matching_composition_differences.png")
const ABLATION_COMPOSITION_PATH = joinpath(ROOT, "output", "main", "figures", "matching_composition_ablation_differences.png")

# Ordered colors for the continuous turnover parameter in the review figure.
const TURNOVER_REVIEW_COLORS = Dict(zip(
    (0.0, 0.001, 0.01, 0.02, 0.03),
    ("#83B5D1", "#589CC3", "#327FAF", "#175D8D", "#083C62"),
))
const TURNOVER_REVIEW_MARKERS = Dict(zip(
    (0.0, 0.001, 0.01, 0.02, 0.03), (:circle, :rect, :diamond, :utriangle, :dtriangle),
))

# Review palette for difficulty, distinct from the established turnover colors.
const DIFFICULTY_COLORS = Dict(zip(
    (0.0, 0.25, 0.5, 0.75, 1.0),
    ("#BD8B13", "#D96A48", "#C63E83", "#914494", "#52256F"),
))

file_hash(path) = bytes2hex(sha256(read(path)))

"""Calculate percentage shares within seeds, checking channel accounting."""
function shares(values)
    all(v -> isfinite(v.total) && v.total > 0 &&
        isapprox(v.self + v.broker, v.total), values) || error("invalid channel totals")
    result = [100 * v.broker / v.total for v in values]
    all(x -> isfinite(x) && 0 <= x <= 100, result) || error("invalid output shares")
    return result
end

"""Subtract paired seed values after explicitly aligning seed identifiers."""
function paired_values(a_seeds, a, b_seeds, b)
    length(a_seeds) == length(a) == length(unique(a_seeds)) || error("invalid first seed set")
    length(b_seeds) == length(b) == length(unique(b_seeds)) || error("invalid second seed set")
    Set(a_seeds) == Set(b_seeds) || error("unmatched seed sets")
    left, right = Dict(zip(a_seeds, a)), Dict(zip(b_seeds, b))
    seeds = sort(collect(keys(left)))
    return seeds, [left[s] - right[s] for s in seeds]
end

"""Retain only displayed seed vectors, together with each input's provenance."""
function prepare(assessment_path, nn_shares_path, nn_values_path)
    aa, ns, nv = load.((assessment_path, nn_shares_path, nn_values_path))
    ridge = load(RIDGE_PATH)["figdata"]
    am, nm, np, rm = aa["metadata"], ns["metadata"], nv["provenance"], ridge["meta"]
    am["definition"] == nm["definition"] == "mean_of_seed_level_late_net_output_shares" ||
        error("wrong share definition")
    np["definition"] == rm["net_output_definition"] == "net_output_per_principal" ||
        error("wrong source output definition")
    nm["parent_sha256"] == file_hash(nn_values_path) || error("NN parent changed")
    nm["late_periods"] == np["late_periods"] || error("NN reporting periods differ")
    am["late_width"] == length(nm["late_periods"]) == rm["late_width"] ||
        error("reporting windows differ")
    level = Float64(am["interval_level"])
    level == nm["interval_level"] == rm["interval_level"] || error("interval levels differ")

    rows = NamedTuple[]
    function add(panel, series, x, seeds, values)
        length(seeds) == length(values) == length(unique(seeds)) || error("invalid seed values")
        all(isfinite, values) || error("nonfinite estimate")
        push!(rows, (; panel, series, x=Float64(x), seeds=Int.(seeds), values=Float64.(values)))
    end
    ac, rc = aa["conditions"], ridge["conditions"]
    rhos = sort(unique(c["rho"] for c in ac))
    rates = filter(sort(unique(c["eta"] for c in ac))) do eta
        all(rho -> count(c -> c["eta"] == eta && c["rho"] == rho, ac) == 3, rhos)
    end
    for eta in rates, rho in rhos
        full = only(filter(c -> c["service"] == "full" && c["rho"] == rho && c["eta"] == eta, ac))
        access = only(filter(c -> c["service"] == "access_only" && c["rho"] == rho && c["eta"] == eta, ac))
        seeds, values = paired_values(full["seeds"], shares(full["channel_values"]),
                                      access["seeds"], shares(access["channel_values"]))
        add("A", string(eta), rho, seeds, values)
    end

    all_deltas = sort(unique(c["delta"] for c in rc if c["rho"] < maximum(rhos)))
    deltas = unique([first(all_deltas), rm["baseline_difficulty"], last(all_deltas)])
    for c in rc
        boundary = c["rho"] == maximum(rhos)
        (boundary || c["delta"] in deltas) || continue
        seeds, values = paired_values(c["seeds"], shares(c["channel_values"]["pair"]),
                                      c["seeds"], shares(c["channel_values"]["additive"]))
        add("B", boundary ? "boundary" : string(c["delta"]), c["rho"], seeds, values)
    end

    difficulty_rho = minimum(rhos)
    for c in ns["conditions"]
        c["rho"] == difficulty_rho || continue
        n = only(filter(n -> n["rel"] == c["rel"], nv["conditions"]))
        c["seeds"] == n["seeds"] || error("NN seed order differs")
        c["N"] == n["N"] || error("NN principal counts differ")
        isapprox(getproperty.(c["channel_values"], :total), n["late"]["net_output_per_principal"]) ||
            error("NN output totals differ")
        isapprox(shares(c["channel_values"]), 100 .* c["broker_shares"]) ||
            error("NN share reconstruction differs")
        for (actor, metric) in (("broker", "broker_holdout_rank"), ("principal", "agent_holdout_rank"))
            add("C", "nn_" * actor, c["delta"], c["seeds"], n["late"][metric])
        end
        add("D", "nn", c["delta"], c["seeds"], shares(c["channel_values"]))
    end
    for c in rc
        c["rho"] == difficulty_rho || continue
        add("C", "ridge_broker", c["delta"], c["seeds"], c["broker_ranks"]["pair"])
        add("C", "ridge_principal", c["delta"], c["seeds"], c["principal_ranks"]["pair"])
        add("D", "ridge", c["delta"], c["seeds"], shares(c["channel_values"]["pair"]))
        add("D", "no_interaction", c["delta"], c["seeds"], shares(c["channel_values"]["additive"]))
    end
    sources = Dict(
        name => Dict("input" => abspath(path), "sha256" => file_hash(path), "provenance" => meta)
        for (name, path, meta) in (("assessment", assessment_path, am),
            ("nn_shares", nn_shares_path, nm), ("nn_ranks", nn_values_path, np),
            ("ridge", RIDGE_PATH, rm))
    )
    metadata = Dict("sources" => sources, "interval_level" => level,
        "late_periods" => nm["late_periods"], "turnover" => rates, "difficulty" => deltas,
        "difficulty_rho" => difficulty_rho, "baseline_difficulty" => rm["baseline_difficulty"],
        "baseline_turnover" => only(unique(c["eta"] for c in ns["conditions"])),
        "definition" => "mean_of_seed_level_late_net_output_shares",
        "script_sha256" => file_hash(@__FILE__),
        "monte_carlo_sha256" => file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")),
        "git_commit" => strip(read(`git -C $ROOT rev-parse HEAD`, String)))
    jldsave(DATA_PATH; metadata, rows)
    return (; metadata, rows)
end

"""Get pointwise Monte Carlo intervals for a displayed series."""
function estimates(data, panel, series)
    selected = sort(filter(r -> r.panel == panel && r.series == series, data.rows); by=r -> r.x)
    isempty(selected) && error("missing series: $panel $series")
    return [(; x=r.x, monte_carlo_interval(r.values; level=data.metadata["interval_level"])...)
            for r in selected]
end

"""Draw means and intervals, optionally connecting the displayed conditions."""
function draw_series!(ax, points; color, marker=:circle, linestyle=:solid, label="",
    connect=true, hollow=false, linewidth=2.3,
    intervalwidth=1.3, markersize=10, whiskerwidth=8)
    x, y = getproperty.(points, :x), getproperty.(points, :mean)
    rangebars!(ax, x, getproperty.(points, :lower), getproperty.(points, :upper);
        color=(color, 0.8), linewidth=intervalwidth, whiskerwidth)
    connect && lines!(ax, x, y; color, linewidth, linestyle)
    markercolor = hollow ? :white : color
    strokecolor, strokewidth = hollow ? (color, 1.7) : (:white, 0.6)
    scatter!(ax, x, y; color=markercolor, marker, markersize, strokecolor, strokewidth)
    return [LineElement(; color, linewidth, linestyle),
            MarkerElement(; color=markercolor, marker, markersize, strokecolor, strokewidth)]
end

"""Compact panel-specific legend, positioned outside the data area."""
function legend!(slot, elements, labels; title=nothing, nbanks=1)
    return Legend(slot, elements, labels, title;
        PUB_LEGEND..., nbanks, labelsize=17, titlesize=17, colgap=18,
        patchsize=(25, 12), halign=:left)
end

"""Annotate the paired contribution of retaining the Ridge interaction term."""
function contribution_label!(ax, data, x; side)
    a = only(filter(r -> r.panel == "D" && r.series == "ridge" && r.x == x, data.rows))
    b = only(filter(r -> r.panel == "D" && r.series == "no_interaction" && r.x == x, data.rows))
    _, values = paired_values(a.seeds, a.values, b.seeds, b.values)
    interval = monte_carlo_interval(values; level=data.metadata["interval_level"])
    low, high = mean(b.values), mean(a.values)
    bracket_x = x + (side == :left ? 0.025 : -0.025)
    lines!(ax, [bracket_x, bracket_x], [low + 1, high - 1]; color=:gray60, linewidth=1)
    text!(ax, bracket_x, (low + high) / 2;
        text=@sprintf("+%.0f pp\n[%.0f, %.0f]", interval.mean, interval.lower, interval.upper),
        align=(side == :left ? :left : :right, :center),
        offset=(side == :left ? 10 : -10, 0), fontsize=16, color=:gray35)
    return nothing
end

"""Render composition contrasts above aligned difficulty comparisons."""
function render(data)
    publication_theme!()
    fig = Figure(; size=(1320, 1040), figure_padding=(25, 30, 22, 22))
    top, bottom = GridLayout(fig[1, 1]), GridLayout(fig[2, 1])
    grids = [GridLayout(top[1, 1]), GridLayout(top[1, 2]),
             GridLayout(bottom[1, 1]), GridLayout(bottom[1, 2])]
    titles = ("A. Contribution of assessment", "B. Contribution of learning interactions",
              "C. Ranking accuracy as difficulty increases", "D. Sources of output as difficulty increases")
    for (g, title) in zip(grids, titles)
        Label(g[1, 1], title; fontsize=22, font=:bold, halign=:left, tellwidth=false)
    end
    for g in grids[3:4]
        Label(g[2, 1], "ρ = $(Int(data.metadata["difficulty_rho"])) (match output depends entirely on complementarity)";
            fontsize=17, color=:gray35, halign=:left, tellwidth=false)
    end
    rho_ticks = ([0, 0.5, 1], ["0\nComplementarity", "0.5", "1\nGeneral quality"])
    axes = [Axis(grids[i][i <= 2 ? 3 : 4, 1];
        xlabel=i <= 2 ? PUB_RHO_LABEL : "Difficulty (δ)",
        ylabel=i <= 2 ? "Increase in share from\nbrokered matches (percentage points)" :
            i == 3 ? "Late-mean rank correlation" : "Principals’ net output\nfrom brokered matches (%)",
        xticks=i <= 2 ? rho_ticks : (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]),
        xlabelsize=21, ylabelsize=20, xticklabelsize=17, yticklabelsize=18,
        limits=(i <= 2 ? (-0.10, 1.10) : (-0.045, 1.045), nothing)) for i in 1:4]
    a, b, c, d = axes

    elements, labels = [], String[]
    for (eta, marker, style) in zip(data.metadata["turnover"], (:circle, :rect, :utriangle), (:dot, :dash, :solid))
        pts = estimates(data, "A", string(eta))
        push!(elements, draw_series!(a, pts; color=PUB_ETA_COLORS[eta], marker, linestyle=style))
        push!(labels, iszero(eta) ? "None" : @sprintf("%.0f%%", 100eta))
    end
    legend!(grids[1][2, 1], elements, labels; title="Principals replaced per period")
    ylims!(a, 0, 35)
    a.yticks = 0:10:30

    elements, labels = [], String[]
    for delta in data.metadata["difficulty"]
        push!(elements, draw_series!(b, estimates(data, "B", string(delta));
            color=DIFFICULTY_COLORS[delta], marker=PUB_DELTA_MARKERS[delta]))
        push!(labels, string(delta))
    end
    draw_series!(b, estimates(data, "B", "boundary"); color=:gray30, marker=:diamond, connect=false)
    push!(elements, [MarkerElement(; color=:gray30, marker=:diamond, markersize=10)])
    push!(labels, "N/A")
    legend!(grids[2][2, 1], elements, labels; title="Difficulty (δ)")
    hlines!(b, [0]; color=(:gray40, 0.7), linestyle=:dash, linewidth=1)
    ylims!(b, -5, 80)
    b.yticks = 0:20:80

    elements, labels = [], String[]
    for (series, label, color, marker, style) in (
        ("nn_broker", "Broker, NN", PUB_BROKER, :circle, :solid),
        ("nn_principal", "Principals, NN", PUB_PRINCIPAL, :rect, :solid),
        ("ridge_broker", "Broker, Ridge", PUB_BROKER, :circle, :dash),
        ("ridge_principal", "Principals, Ridge", PUB_PRINCIPAL, :rect, :dash))
        push!(elements, draw_series!(c, estimates(data, "C", series); color, marker, linestyle=style))
        push!(labels, label)
    end
    legend!(grids[3][3, 1], elements, labels; nbanks=2)
    ylims!(c, 0, 1.03)
    c.yticks = 0:0.2:1

    elements, labels = [], String[]
    for (series, label, color, marker, style) in (
        ("nn", "NN", PUB_BROKER, :circle, :solid),
        ("ridge", "Ridge", PUB_BROKER, :circle, :dash),
        ("no_interaction", "Ridge, no interaction term", "#575C61", :diamond, :dot))
        push!(elements, draw_series!(d, estimates(data, "D", series); color, marker, linestyle=style))
        push!(labels, label)
    end
    legend!(grids[4][3, 1], elements, labels; nbanks=2)
    ylims!(d, 0, 103)
    d.yticks = 0:20:100
    endpoints = extrema(p.x for p in estimates(data, "D", "ridge"))
    contribution_label!(d, data, endpoints[1]; side=:left)
    contribution_label!(d, data, endpoints[2]; side=:right)

    for (i, g) in enumerate(grids)
        rowgap!(g, 8)
        rowsize!(g, i <= 2 ? 2 : 3, Fixed(58))
    end
    colgap!(top, 40)
    colgap!(bottom, 40)
    rowgap!(bottom, 10)
    rowgap!(fig.layout, 28)
    save(FIGURE_PATH, fig; px_per_unit=2)
    println("Saved ", FIGURE_PATH)
    return fig
end

"""Retain a validated seed vector for one panel and condition."""
function add_row!(rows, panel, series, x, seeds, values)
    length(seeds) == length(values) == length(unique(seeds)) || error("invalid seed vector")
    all(isfinite, values) || error("nonfinite seed value")
    any(r -> (r.panel, r.series, r.x) == (panel, series, x), rows) &&
        error("duplicate panel condition")
    push!(rows, (; panel, series, x=Float64(x), seeds=Int.(seeds), values=Float64.(values)))
    return nothing
end

"""Prepare reference-model levels and experimental contrasts from identified inputs."""
function prepare_split!(saved)
    meta = saved["metadata"]
    sources = meta["sources"]
    for source in values(sources)
        file_hash(source["input"]) == source["sha256"] || error("source input changed")
    end
    aa = load(sources["assessment"]["input"])
    ns = load(sources["nn_shares"]["input"])
    nv = load(sources["nn_ranks"]["input"])
    ridge = load(sources["ridge"]["input"])["figdata"]
    aa_path = joinpath(ROOT, "output", "assessment_access", "figure_data.jld2")
    aa_data = load(aa_path)
    file_hash(aa_path) == aa["metadata"]["source_input_sha256"] ||
        error("assessment ranks and channel estimates have different parents")
    aa_data["late_width"] == ridge["meta"]["late_width"] == length(meta["late_periods"]) ||
        error("late windows differ")
    aa_data["interval_level"] == ridge["meta"]["interval_level"] == meta["interval_level"] ||
        error("interval levels differ")

    main_rows, supplement_rows = NamedTuple[], NamedTuple[]
    delta, rho = meta["baseline_difficulty"], meta["difficulty_rho"]
    boundary_rho = maximum(c["rho"] for c in ns["conditions"])
    for c in ns["conditions"]
        composition = c["delta"] == delta || c["rho"] == boundary_rho
        difficulty = c["rho"] == rho
        (composition || difficulty) || continue
        n = only(filter(n -> n["rel"] == c["rel"], nv["conditions"]))
        c["seeds"] == n["seeds"] || error("NN seed order differs")
        isapprox(getproperty.(c["channel_values"], :total), n["late"]["net_output_per_principal"]) ||
            error("NN channel totals differ")
        for (keep, rank_panel, share_panel, x) in (
            (composition, "A", "B", c["rho"]), (difficulty, "C", "D", c["delta"]))
            keep || continue
            for (actor, metric) in (("broker", "broker_holdout_rank"), ("principal", "agent_holdout_rank"))
                add_row!(main_rows, rank_panel, "nn_" * actor, x, c["seeds"], n["late"][metric])
            end
            add_row!(main_rows, share_panel, "nn", x, c["seeds"], shares(c["channel_values"]))
        end
    end
    for c in ridge["conditions"]
        composition = c["delta"] == delta || c["rho"] == boundary_rho
        difficulty = c["rho"] == rho
        for (keep, rank_panel, share_panel, x) in (
            (composition, "A", "B", c["rho"]), (difficulty, "C", "D", c["delta"]))
            keep || continue
            add_row!(main_rows, rank_panel, "ridge_broker", x, c["seeds"], c["broker_ranks"]["pair"])
            add_row!(main_rows, rank_panel, "ridge_principal", x, c["seeds"], c["principal_ranks"]["pair"])
            add_row!(main_rows, share_panel, "ridge", x, c["seeds"], shares(c["channel_values"]["pair"]))
        end
        composition || continue
        for restricted in ("size_matched", "single_principal", "additive")
            for (panel, full, comparison) in (
                ("C", c["broker_ranks"]["pair"], c["broker_ranks"][restricted]),
                ("D", shares(c["channel_values"]["pair"]), shares(c["channel_values"][restricted])))
                seeds, values = paired_values(c["seeds"], full, c["seeds"], comparison)
                add_row!(supplement_rows, panel, restricted, c["rho"], seeds, values)
            end
        end
    end

    aa_rhos = sort(unique(c["rho"] for c in aa["conditions"]))
    function broker_ranks(service, rho, eta)
        rows = filter(aa_data["seed_rows"]) do row
            string(row[1]) == service && row[2] == rho && row[3] == eta &&
                string(row[5]) == "broker_holdout_rank"
        end
        return Int[r[4] for r in rows], Float64[r[6] for r in rows]
    end
    for eta in meta["turnover"], rho in aa_rhos
        ac = aa["conditions"]
        full = only(filter(c -> c["service"] == "full" && c["rho"] == rho && c["eta"] == eta, ac))
        access = only(filter(c -> c["service"] == "access_only" && c["rho"] == rho && c["eta"] == eta, ac))
        fs, fv = broker_ranks("full", rho, eta)
        as, av = broker_ranks("access_only", rho, eta)
        Set(fs) == Set(full["seeds"]) && Set(as) == Set(access["seeds"]) || error("rank/share seeds differ")
        seeds, values = paired_values(fs, fv, as, av)
        add_row!(supplement_rows, "A", string(eta), rho, seeds, values)
        seeds, values = paired_values(full["seeds"], shares(full["channel_values"]),
                                      access["seeds"], shares(access["channel_values"]))
        add_row!(supplement_rows, "B", string(eta), rho, seeds, values)
    end
    split_metadata = deepcopy(meta)
    split_metadata["sources"]["assessment_ranks"] = Dict("input" => aa_path,
        "sha256" => file_hash(aa_path), "analysis_git_commit" => aa_data["analysis_git_commit"],
        "manifest_hashes" => [aa_data["manifest_hash"], aa_data["supplement_manifest_hash"]])
    split_metadata["script_sha256"] = file_hash(@__FILE__)
    saved["split_metadata"] = split_metadata
    saved["main_rows"] = main_rows
    saved["supplement_rows"] = supplement_rows
    jldsave(DATA_PATH; (Symbol(key) => value for (key, value) in saved)...)
    return saved
end

"""Use the Figure 2 layout, separating pure general quality in a shaded strip."""
function split_layout(titles, notes; supplement=false)
    publication_theme!()
    fig = Figure(; size=(1420, 1010), figure_padding=(28, 32, 26, 24))
    grids = [GridLayout(fig[r, c]) for r in 1:2 for c in 1:2]
    panels = NamedTuple[]
    for (i, g) in enumerate(grids)
        Label(g[1, 1], titles[i]; fontsize=TITLE_FS, font=:bold, halign=:left,
            justification=:left, tellwidth=false)
        Label(g[2, 1], notes[i]; fontsize=FOOTER_FS, color=:gray35,
            halign=:left, tellwidth=false)
        composition = supplement || i <= 2
        rank = isodd(i)
        ylabel = if supplement
            rank ? "Change in broker\nrank correlation" :
                "Change in share from\nbrokered matches\n(percentage points)"
        else
            rank ? "Late-mean rank correlation" : "Principals’ net output\nfrom brokered matches (%)"
        end
        ticks = composition ? ([0, 0.5], ["0\nComplementarity", "0.5"]) :
            (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"])
        plot_grid = GridLayout(g[3, 1])
        axis = Axis(plot_grid[1, 1]; xlabel=composition ? PUB_RHO_LABEL : "Difficulty (δ)",
            ylabel, xticks=ticks,
            limits=(composition ? (-0.04, 0.90) : (-0.045, 1.045), nothing))
        boundary = if composition
            ax = Axis(plot_grid[1, 2]; xticks=([1], ["1\nGeneral quality"]),
                limits=((0.9, 1.1), nothing), yticklabelsvisible=false,
                yticksvisible=false, leftspinevisible=false,
                backgroundcolor=(:gray40, 0.045))
            linkyaxes!(axis, ax)
            colsize!(plot_grid, 2, Fixed(100))
            colgap!(plot_grid, 16)
            ax
        else
            nothing
        end
        push!(panels, (; axis, boundary))
        rowgap!(g, 13)
        rowgap!(g, 1, 5)
    end
    colgap!(fig.layout, 52)
    rowgap!(fig.layout, 34)
    return fig, panels
end

"""Draw a series with the Figure 2 line, marker, and interval sizing."""
function draw_split_series!(panel, points; color, marker, linestyle, hollow=false)
    kwargs = (; color, marker, linestyle, hollow, linewidth=2.5,
        intervalwidth=1.7, markersize=9, whiskerwidth=9)
    interior = isnothing(panel.boundary) ? points : filter(p -> p.x < 1, points)
    elements = draw_series!(panel.axis, interior; kwargs...)
    if !isnothing(panel.boundary)
        boundary = filter(p -> p.x == 1, points)
        length(boundary) == 1 || error("expected one pure-general-quality estimate")
        draw_series!(panel.boundary, boundary; kwargs..., connect=false)
    end
    return elements
end

"""Place a compact legend inside the panel, matching Figure 2."""
function split_legend!(ax, elements, labels; position=:lb, nbanks=1, title=nothing)
    return axislegend(ax, elements, labels, title; position, nbanks,
        framevisible=false, labelsize=FOOTER_FS - 1, titlesize=FOOTER_FS - 1,
        titlehalign=:left, gridshalign=:left, rowgap=6, colgap=18, patchsize=(28, 14),
        backgroundcolor=(:white, 0.9), padding=(5, 5, 5, 5))
end

"""Keep the main axis and detached boundary on identical vertical scales."""
function split_ylimits!(panel, limits, ticks)
    for ax in (panel.axis, panel.boundary)
        isnothing(ax) && continue
        ylims!(ax, limits...)
        ax.yticks = ticks
    end
    return nothing
end

"""Derive the common output-share scale used by the reference-model panels."""
function reference_share_scale(data)
    bounds = Float64[]
    for panel in ("B", "D"), series in ("nn", "ridge")
        pts = estimates(data, panel, series)
        append!(bounds, getproperty.(pts, :lower))
        append!(bounds, getproperty.(pts, :upper))
    end
    bottom = 5floor((minimum(bounds) - 1) / 5)
    top = max(100.5, maximum(bounds) + 0.5)
    return (; limits=(bottom, top), ticks=bottom:5:100)
end

"""Draw reference-model outcomes as composition or difficulty varies."""
function render_split_main(data)
    titles = ("A. Ranking accuracy\nas matching composition varies",
        "B. Sources of output\nas matching composition varies",
        "C. Ranking accuracy\nas difficulty increases",
        "D. Sources of output\nas difficulty increases")
    delta_note = "Late means · difficulty δ = $(data.metadata["baseline_difficulty"])"
    rho_note = "Late means · matching composition ρ = $(Int(data.metadata["difficulty_rho"]))"
    fig, panels = split_layout(titles, (delta_note, delta_note, rho_note, rho_note))
    for (i, panel) in enumerate(("A", "B", "C", "D"))
        elements, labels = [], String[]
        specs = isodd(i) ? (
            ("nn_broker", "Broker, NN", PUB_BROKER, :circle, :solid),
            ("nn_principal", "Principals, NN", PUB_PRINCIPAL, :rect, :solid),
            ("ridge_broker", "Broker, Ridge", PUB_BROKER, :circle, :dash),
            ("ridge_principal", "Principals, Ridge", PUB_PRINCIPAL, :rect, :dash)) : (
            ("nn", "NN", PUB_BROKER, :circle, :solid),
            ("ridge", "Ridge", PUB_BROKER, :circle, :dash))
        for (series, label, color, marker, style) in specs
            pts = estimates(data, panel, series)
            push!(elements, draw_split_series!(panels[i], pts; color, marker,
                linestyle=style, hollow=startswith(series, "ridge")))
            push!(labels, label)
        end
        split_legend!(panels[i].axis, elements, labels; nbanks=isodd(i) ? 2 : 1)
        if isodd(i)
            split_ylimits!(panels[i], (0, 1.03), 0:0.2:1)
        end
    end
    share_scale = reference_share_scale(data)
    for i in (2, 4)
        split_ylimits!(panels[i], share_scale.limits, share_scale.ticks)
    end
    save(SPLIT_MAIN_PATH, fig; px_per_unit=2)
    println("Saved ", SPLIT_MAIN_PATH)
    return fig
end

"""Save the main figure's difficulty/output panel as a standalone figure."""
function render_difficulty()
    saved = load(DATA_PATH)
    meta = saved["split_metadata"]
    meta["monte_carlo_sha256"] == file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")) ||
        error("Monte Carlo definition changed")
    data = (; metadata=meta, rows=saved["main_rows"])
    publication_theme!()
    fig = Figure(; size=(720, 520), figure_padding=(28, 32, 26, 24))
    grid = GridLayout(fig[1, 1])
    Label(grid[1, 1], "Sources of output\nas difficulty increases";
        fontsize=TITLE_FS, font=:bold, halign=:left, justification=:left, tellwidth=false)
    Label(grid[2, 1], "Late means · matching composition ρ = $(Int(meta["difficulty_rho"]))";
        fontsize=FOOTER_FS, color=:gray35, halign=:left, tellwidth=false)
    ax = Axis(grid[3, 1]; xlabel="Difficulty (δ)",
        ylabel="Principals’ net output\nfrom brokered matches (%)",
        xticks=(0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]),
        limits=((-0.045, 1.045), nothing))
    panel = (; axis=ax, boundary=nothing)
    elements, labels = [], String[]
    for (series, label, style) in (("nn", "NN", :solid), ("ridge", "Ridge", :dash))
        push!(elements, draw_split_series!(panel, estimates(data, "D", series);
            color=PUB_BROKER, marker=:circle, linestyle=style, hollow=series == "ridge"))
        push!(labels, label)
    end
    split_legend!(ax, elements, labels)
    scale = reference_share_scale(data)
    split_ylimits!(panel, scale.limits, scale.ticks)
    rowgap!(grid, 13)
    rowgap!(grid, 1, 5)
    save(DIFFICULTY_PATH, fig; px_per_unit=2)
    println("Saved ", DIFFICULTY_PATH)
    return fig
end

"""Draw paired rank-correlation and output-share changes for each experiment."""
function render_split_supplement(data)
    titles = ("A. Providing assessment:\nchange in broker rank correlation",
        "B. Providing assessment:\nchange in output share",
        "C. Retaining information:\nchange in broker rank correlation",
        "D. Retaining information:\nchange in output share")
    delta = data.metadata["baseline_difficulty"]
    aa_note = "Full service minus access-only · δ = $delta"
    ridge_note = "Ridge reference minus restricted · δ = $delta"
    fig, panels = split_layout(titles, (aa_note, aa_note, ridge_note, ridge_note); supplement=true)
    restrictions = (
        ("size_matched", "Fewer observations", "#A07820", :circle, :solid),
        ("single_principal", "One party recorded", "#914494", :rect, :dash),
        ("additive", "No interaction term", "#575C61", :diamond, :dot))
    for (i, panel) in enumerate(("A", "B", "C", "D"))
        elements, labels, bounds = [], String[], Float64[]
        specs = if i <= 2
            [(string(eta), iszero(eta) ? "None" : @sprintf("%.0f%%", 100eta),
                PUB_ETA_COLORS[eta], marker, style) for (eta, marker, style) in
                zip(data.metadata["turnover"], (:circle, :rect, :utriangle), (:dot, :dash, :solid))]
        else
            restrictions
        end
        for (series, label, color, marker, style) in specs
            pts = estimates(data, panel, series)
            push!(elements, draw_split_series!(panels[i], pts; color, marker, linestyle=style))
            push!(labels, label)
            append!(bounds, getproperty.(pts, :lower))
            append!(bounds, getproperty.(pts, :upper))
        end
        split_legend!(panels[i].axis, elements, labels; position=i <= 2 ? :lt : :rt,
            title=i <= 2 ? "Principals replaced per period" : nothing)
        scale = interval_axis(bounds)
        padding = 0.15scale.step
        split_ylimits!(panels[i], (scale.limits[1] - padding, scale.limits[2] + padding), scale.ticks)
        for ax in (panels[i].axis, panels[i].boundary)
            hlines!(ax, [0]; color=:gray70, linestyle=:dash, linewidth=1)
        end
    end
    save(SPLIT_SUPP_PATH, fig; px_per_unit=2)
    println("Saved ", SPLIT_SUPP_PATH)
    return fig
end

"""Retain the full NN grid, with within-seed ranking gaps and output shares."""
function prepare_nn_grid!(saved)
    meta = saved["metadata"]
    sources = Dict(key => deepcopy(meta["sources"][key]) for key in ("nn_shares", "nn_ranks"))
    for source in values(sources)
        file_hash(source["input"]) == source["sha256"] || error("NN source input changed")
    end
    ns, nv = load(sources["nn_shares"]["input"]), load(sources["nn_ranks"]["input"])
    sm, vm = ns["metadata"], nv["provenance"]
    sm["parent_sha256"] == sources["nn_ranks"]["sha256"] || error("NN parent mismatch")
    sm["definition"] == "mean_of_seed_level_late_net_output_shares" || error("wrong share definition")
    vm["definition"] == "net_output_per_principal" || error("wrong net-output definition")
    sm["late_periods"] == vm["late_periods"] == meta["late_periods"] || error("NN windows differ")
    sm["interval_level"] == vm["interval_level"] == meta["interval_level"] || error("NN intervals differ")
    conditions = ns["conditions"]
    keys_for(cs) = Set((c["rho"], c["delta"], c["rel"]) for c in cs)
    keys_for(conditions) == keys_for(nv["conditions"]) || error("NN condition coverage differs")
    interior = filter(c -> c["rho"] < 1, conditions)
    rhos = sort(unique(c["rho"] for c in interior))
    deltas = sort(unique(c["delta"] for c in interior))
    Set((c["rho"], c["delta"]) for c in interior) == Set(Iterators.product(rhos, deltas)) ||
        error("incomplete NN composition/difficulty grid")
    count(c -> c["rho"] == 1, conditions) == 1 || error("expected one general-quality boundary")
    length(conditions) == length(rhos) * length(deltas) + 1 || error("duplicated NN regime")
    turnover = only(unique(c["eta"] for c in conditions))
    rows = NamedTuple[]
    for c in conditions
        n = only(filter(n -> n["rel"] == c["rel"], nv["conditions"]))
        (c["rho"], c["delta"], c["eta"], c["N"], c["seeds"]) ==
            (n["rho"], n["delta"], n["eta"], n["N"], n["seeds"]) || error("NN regime or seed mismatch")
        all(isapprox.(getproperty.(c["channel_values"], :total),
            n["late"]["net_output_per_principal"]; atol=1e-12)) || error("NN channel totals differ")
        output_share = shares(c["channel_values"])
        all(isapprox.(output_share, 100 .* c["broker_shares"]; atol=1e-12)) || error("NN shares differ")
        all(isapprox.(c["broker_shares"] .+ c["self_shares"], 1; atol=1e-12)) ||
            error("NN source shares do not sum to one")
        broker, principal = n["late"]["broker_holdout_rank"], n["late"]["agent_holdout_rank"]
        seeds, gap = paired_values(n["seeds"], broker, n["seeds"], principal)
        series = c["rho"] == 1 ? "boundary" : string(c["delta"])
        for (panel, ids, values) in (("A", n["seeds"], broker), ("B", n["seeds"], principal),
            ("C", seeds, gap), ("D", c["seeds"], output_share))
            add_row!(rows, panel, series, c["rho"], ids, values)
        end
    end
    grid_meta = Dict("sources" => sources, "interval_level" => sm["interval_level"],
        "late_periods" => sm["late_periods"], "difficulty" => deltas, "rho" => rhos,
        "turnover" => turnover, "regime_count" => length(conditions),
        "share_definition" => sm["definition"], "ranking_gap_definition" => "broker_minus_principal_within_seed",
        "script_sha256" => file_hash(@__FILE__),
        "monte_carlo_sha256" => file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")),
        "git_commit" => strip(read(`git -C $ROOT rev-parse HEAD`, String)))
    saved["nn_grid_metadata"], saved["nn_grid_rows"] = grid_meta, rows
    jldsave(DATA_PATH; (Symbol(key) => value for (key, value) in saved)...)
    return saved
end

"""Transpose the retained NN grid without changing any seed values."""
function nn_grid_by_difficulty(data)
    rows = NamedTuple[]
    for row in data.rows
        boundary = row.series == "boundary"
        series = boundary ? "boundary" : string(row.x)
        x = boundary ? row.x : parse(Float64, row.series)
        add_row!(rows, row.panel, series, x, row.seeds, row.values)
    end
    return (; metadata=data.metadata, rows)
end

"""Show the ordered composition colors, with no line for pure general quality."""
function nn_composition_legend!(slot, rhos)
    values = [rhos; 1.0]
    elements = [
        rho == 1 ? [MarkerElement(; color=PUB_RHO_COLORS[rho],
            marker=PUB_RHO_MARKERS[rho], markersize=10)] :
        [LineElement(; color=PUB_RHO_COLORS[rho], linewidth=2),
            MarkerElement(; color=PUB_RHO_COLORS[rho], marker=PUB_RHO_MARKERS[rho], markersize=9)]
        for rho in values
    ]
    return Legend(slot, elements, string.(values), PUB_RHO_LABEL; PUB_LEGEND..., nbanks=2)
end

"""Render NN regimes against composition or difficulty, keeping ρ = 1 separate."""
function render_nn_grid(data; by_difficulty=false, by_turnover=false)
    by_difficulty && by_turnover && error("difficulty-by-turnover results are not available")
    shown = by_difficulty ? nn_grid_by_difficulty(data) : data
    levels = if by_turnover
        data.metadata["turnover"]
    elseif by_difficulty
        filter(rho -> !(rho in (0.3, 0.7)), data.metadata["rho"])
    else
        data.metadata["difficulty"]
    end
    publication_theme!()
    fig = Figure(; size=(1420, 1050), figure_padding=(28, 32, 26, 24))
    header = GridLayout(fig[1, 1:2])
    scope = by_turnover ? "difficulty δ = $(data.metadata["difficulty"])" :
        "turnover η = $(data.metadata["turnover"])"
    Label(header[1, 1], "NN learning · late means · $scope";
        fontsize=FOOTER_FS, color=:gray35, halign=:left, tellwidth=false)
    if by_turnover
        elements = [[LineElement(; color=TURNOVER_REVIEW_COLORS[eta], linewidth=2),
            MarkerElement(; color=TURNOVER_REVIEW_COLORS[eta], marker=TURNOVER_REVIEW_MARKERS[eta],
                markersize=9)] for eta in levels]
        Legend(header[1, 2], elements, string.(levels), "Turnover rate (η)"; PUB_LEGEND..., nbanks=2)
    elseif by_difficulty
        nn_composition_legend!(header[1, 2], levels)
    else
        difficulty_legend!(header[1, 2], data.metadata["difficulty"]; nbanks=2)
    end
    titles = ("A. Broker ranking accuracy", "B. Principal ranking accuracy",
        "C. Broker ranking advantage", "D. Sources of output")
    ylabels = ("Rank correlation", "Rank correlation", "Rank correlation\nbroker minus principal",
        "Principals’ net output\nfrom brokered matches (%)")
    for (i, panel) in enumerate(("A", "B", "C", "D"))
        row, column = i <= 2 ? (2, i) : (3, i - 2)
        grid = GridLayout(fig[row, column])
        Label(grid[1, 1], titles[i]; fontsize=TITLE_FS, font=:bold, halign=:left, tellwidth=false)
        plot_grid = GridLayout(grid[2, 1])
        ticks = by_difficulty ? data.metadata["difficulty"] : filter(<(1.0), data.metadata["rho"])
        ticklabels = [!by_difficulty && iszero(x) ? "0\nComplementarity" : @sprintf("%.2g", x) for x in ticks]
        main_column, boundary_column = by_difficulty ? (2, 1) : (1, 2)
        ax = Axis(plot_grid[1, main_column]; xlabel=by_difficulty ? "Difficulty (δ)" : PUB_RHO_LABEL,
            ylabel=by_difficulty ? "" : ylabels[i], xticks=(ticks, ticklabels),
            yticklabelsvisible=!by_difficulty, yticksvisible=!by_difficulty,
            leftspinevisible=!by_difficulty,
            limits=(by_difficulty ? (-0.045, 1.045) : (-0.04, 0.90), nothing))
        boundary = Axis(plot_grid[1, boundary_column];
            xticks=([1], [by_difficulty ? "N/A\nρ = 1" : "1\nGeneral quality"]),
            ylabel=by_difficulty ? ylabels[i] : "", limits=((0.9, 1.1), nothing),
            yticklabelsvisible=by_difficulty, yticksvisible=by_difficulty,
            leftspinevisible=by_difficulty, backgroundcolor=(:gray40, 0.045))
        linkyaxes!(ax, boundary)
        colsize!(plot_grid, boundary_column, Fixed(100))
        colgap!(plot_grid, 16)
        rowgap!(grid, 13)
        bounds = Float64[]
        colors = by_turnover ? TURNOVER_REVIEW_COLORS : by_difficulty ? PUB_RHO_COLORS : PUB_DELTA_COLORS
        markers = by_turnover ? TURNOVER_REVIEW_MARKERS : by_difficulty ? PUB_RHO_MARKERS : PUB_DELTA_MARKERS
        for (index, level) in enumerate(levels)
            pts = estimates(shown, panel, string(level))
            interior = by_turnover ? filter(p -> p.x < 1, pts) : pts
            draw_series!(ax, interior; color=colors[level], marker=markers[level],
                linewidth=2.5, intervalwidth=1.7, markersize=9, whiskerwidth=9)
            if by_turnover
                point = only(filter(p -> p.x == 1, pts))
                offset = 0.03 * (index - (length(levels) + 1) / 2)
                draw_series!(boundary, [merge(point, (; x=point.x + offset))];
                    color=colors[level], marker=markers[level], connect=false,
                    intervalwidth=1.7, markersize=9, whiskerwidth=7)
            end
            append!(bounds, getproperty.(pts, :lower))
            append!(bounds, getproperty.(pts, :upper))
        end
        if !by_turnover
            pts = estimates(shown, panel, "boundary")
            draw_series!(boundary, pts; color=by_difficulty ? PUB_RHO_COLORS[1.0] : :gray25,
                marker=by_difficulty ? PUB_RHO_MARKERS[1.0] : :diamond, connect=false,
                intervalwidth=1.7, markersize=10, whiskerwidth=9)
            append!(bounds, getproperty.(pts, :lower))
            append!(bounds, getproperty.(pts, :upper))
        end
        scale = if i <= 2
            (; limits=(0.0, 1.03), ticks=0:0.2:1)
        elseif i == 3
            interval_axis(bounds; upper_padding=0.15)
        else
            bottom = 5floor((minimum(bounds) - 1) / 5)
            tickstep = by_turnover && 100 - bottom > 35 ? 10 : 5
            bottom = tickstep * floor(bottom / tickstep)
            (; limits=(bottom, max(100.5, maximum(bounds) + 0.5)), ticks=bottom:tickstep:100)
        end
        minimum(bounds) >= scale.limits[1] && maximum(bounds) <= scale.limits[2] ||
            error("NN grid axis would clip an interval")
        split_ylimits!((; axis=ax, boundary), scale.limits, scale.ticks)
        if i == 3
            for a in (ax, boundary)
                hlines!(a, [0]; color=:gray70, linestyle=:dash, linewidth=1)
            end
        end
    end
    colgap!(fig.layout, 52)
    rowgap!(fig.layout, 34)
    path = by_turnover ? TURNOVER_GRID_PATH : by_difficulty ? NN_DIFFICULTY_GRID_PATH : NN_GRID_PATH
    save(path, fig; px_per_unit=2)
    println("Saved ", path)
    return fig
end

"""Retain verified turnover-grid seed values and their extraction provenance."""
function prepare_turnover_grid!(saved, input)
    extracted = load(input)
    meta, conditions = extracted["metadata"], extracted["conditions"]
    meta["share_definition"] == "mean_of_seed_level_late_net_output_shares" || error("wrong share definition")
    meta["ranking_gap_definition"] == "broker_minus_principal_within_seed" || error("wrong ranking-gap definition")
    meta["late_periods"] == saved["metadata"]["late_periods"] || error("late windows differ")
    meta["interval_level"] == saved["metadata"]["interval_level"] || error("interval levels differ")
    reference = saved["metadata"]["sources"]["nn_ranks"]["provenance"]
    meta["manifest_hash"] == reference["manifest_hash"] || error("NN sweep manifests differ")
    meta["simulation_commit"] == reference["simulation_commit"] || error("NN simulation sources differ")
    coordinates = Set((c["rho"], c["eta"]) for c in conditions)
    coordinates == Set(Iterators.product(meta["rho"], meta["turnover"])) || error("incomplete turnover grid")
    length(coordinates) == length(conditions) == meta["regime_count"] || error("duplicated turnover condition")
    rows = NamedTuple[]
    for c in conditions
        c["delta"] == meta["difficulty"] || error("difficulty varies across turnover grid")
        isapprox(c["broker_ranks"] .- c["principal_ranks"], c["rank_gaps"]; atol=1e-12) || error("ranking gap differs")
        output_share = shares(c["channel_values"])
        isapprox(output_share, 100 .* c["broker_shares"]; atol=1e-12) || error("output shares differ")
        for (panel, values) in (("A", c["broker_ranks"]), ("B", c["principal_ranks"]),
            ("C", c["rank_gaps"]), ("D", output_share))
            add_row!(rows, panel, string(c["eta"]), c["rho"], c["seeds"], values)
        end
    end
    meta["input"] = abspath(input)
    meta["input_sha256"] = file_hash(input)
    meta["condition_sources"] = [Dict(key => c[key] for key in
        ("rel", "rho", "eta", "delta", "seeds", "config", "source_path", "source_sha256")) for c in conditions]
    saved["turnover_grid_metadata"], saved["turnover_grid_rows"] = meta, rows
    jldsave(DATA_PATH; (Symbol(key) => value for (key, value) in saved)...)
    return nothing
end

"""Render the saved NN composition-by-turnover grid without rerunning simulations."""
function turnover_grid_main(input=nothing)
    saved = load(DATA_PATH)
    isnothing(input) || prepare_turnover_grid!(saved, input)
    haskey(saved, "turnover_grid_rows") || error("supply the matching_turnover_data.jl extract")
    meta = saved["turnover_grid_metadata"]
    meta["monte_carlo_sha256"] == file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")) ||
        error("Monte Carlo definition changed")
    return render_nn_grid((; metadata=meta, rows=saved["turnover_grid_rows"]); by_turnover=true)
end

"""Read the full NN grid and render the requested view without changing other figures."""
function nn_grid_main(; by_difficulty=false)
    saved = load(DATA_PATH)
    haskey(saved, "nn_grid_rows") || prepare_nn_grid!(saved)
    meta = saved["nn_grid_metadata"]
    meta["monte_carlo_sha256"] == file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")) ||
        error("Monte Carlo definition changed")
    rows = saved["nn_grid_rows"]
    all(count(r -> r.panel == panel, rows) == meta["regime_count"] for panel in ("A", "B", "C", "D")) ||
        error("NN panel coverage differs")
    println("Retained NN grid: $(meta["regime_count"]) regimes per panel; seed counts ",
        sort(unique(length(r.seeds) for r in rows)))
    return render_nn_grid((; metadata=meta, rows); by_difficulty)
end

"""Render only the two split drafts, preserving the original review PNG."""
function split_main()
    saved = load(DATA_PATH)
    haskey(saved, "split_metadata") || prepare_split!(saved)
    meta = saved["split_metadata"]
    meta["monte_carlo_sha256"] == file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")) ||
        error("Monte Carlo definition changed")
    render_split_main((; metadata=meta, rows=saved["main_rows"]))
    render_split_supplement((; metadata=meta, rows=saved["supplement_rows"]))
    return nothing
end

"""
Compare two compositions within seeds at baseline and the saved parameter extremes.
For ranking advantage, form broker minus principal within each seed and composition
before differencing compositions and calculating the Monte Carlo interval.
"""
function composition_differences(saved; rhos=(0.0, 0.85))
    baseline = saved["metadata"]
    difficulty = saved["nn_grid_metadata"]
    turnover = saved["turnover_grid_metadata"]
    eta, delta = baseline["baseline_turnover"], baseline["baseline_difficulty"]
    difficulty["turnover"] == eta || error("difficulty grid is not at baseline turnover")
    turnover["difficulty"] == delta || error("turnover grid is not at baseline difficulty")
    for meta in (difficulty, turnover)
        meta["late_periods"] == baseline["late_periods"] || error("late windows differ")
        meta["interval_level"] == baseline["interval_level"] || error("interval levels differ")
        meta["share_definition"] == "mean_of_seed_level_late_net_output_shares" ||
            error("wrong output-share definition")
        meta["monte_carlo_sha256"] == file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")) ||
            error("Monte Carlo definition changed")
        all(rho -> rho in meta["rho"], rhos) || error("requested compositions are not saved")
    end
    low_eta, high_eta = extrema(turnover["turnover"])
    low_delta, high_delta = extrema(difficulty["difficulty"])
    conditions = [
        (; label="Baseline", eta, delta, grid="nn_grid_rows", series=string(delta)),
        (; label="Low turnover", eta=low_eta, delta, grid="turnover_grid_rows", series=string(low_eta)),
        (; label="High turnover", eta=high_eta, delta, grid="turnover_grid_rows", series=string(high_eta)),
        (; label="Low difficulty", eta, delta=low_delta, grid="nn_grid_rows", series=string(low_delta)),
        (; label="High difficulty", eta, delta=high_delta, grid="nn_grid_rows", series=string(high_delta)),
    ]
    select(rows, panel, series, rho) = only(filter(
        r -> r.panel == panel && r.series == series && r.x == rho, rows))
    function seed_outcome(rows, panel, series, rho)
        retained = select(rows, panel, series, rho)
        panel == "C" || return retained
        broker = select(rows, "A", series, rho)
        principal = select(rows, "B", series, rho)
        seeds, values = paired_values(broker.seeds, broker.values, principal.seeds, principal.values)
        _, residual = paired_values(seeds, values, retained.seeds, retained.values)
        all(x -> isapprox(x, 0; atol=1e-12), residual) || error("saved ranking advantage differs")
        return (; seeds, values)
    end
    # The two grids overlap at baseline. Confirm the same saved seed values.
    for panel in ("B", "A", "C", "D"), rho in rhos
        a = seed_outcome(saved["nn_grid_rows"], panel, string(delta), rho)
        b = seed_outcome(saved["turnover_grid_rows"], panel, string(eta), rho)
        _, residual = paired_values(a.seeds, a.values, b.seeds, b.values)
        all(x -> isapprox(x, 0; atol=1e-12), residual) || error("baseline grid estimates differ")
    end
    level = baseline["interval_level"]
    rows = NamedTuple[]
    for (index, condition) in enumerate(conditions), panel in ("B", "A", "C", "D")
        source = saved[condition.grid]
        a = seed_outcome(source, panel, condition.series, rhos[1])
        b = seed_outcome(source, panel, condition.series, rhos[2])
        seeds, values = paired_values(a.seeds, a.values, b.seeds, b.values)
        length(seeds) >= 2 && all(isfinite, values) || error("invalid paired values")
        interval = monte_carlo_interval(values; level)
        push!(rows, (; condition=index, panel, seeds, values, interval...))
    end
    return (; conditions, rows, rhos, level, late_periods=baseline["late_periods"])
end

"""Render paired contrasts, combining both accuracy series on a common ranking scale."""
function render_composition_differences(data)
    publication_theme!()
    fig = Figure(; size=(1500, 650), figure_padding=(28, 32, 28, 24))
    rho_label(rho) = @sprintf("%.2g", rho)
    header = GridLayout(fig[1, 1:3])
    Label(header[1, 1], "Moving from quality to complementarity";
        fontsize=TITLE_FS, font=:bold, halign=:left)
    Label(header[1, 2], "ρ = $(rho_label(data.rhos[2])) → ρ = $(rho_label(data.rhos[1]))";
        fontsize=FOOTER_FS, color=:gray35, halign=:right)
    labels = ["$(c.label)\nη = $(rho_label(c.eta)), δ = $(rho_label(c.delta))" for c in data.conditions]
    positions = collect(eachindex(labels))
    rank_bounds = [x for r in data.rows if r.panel != "D" for x in (r.lower, r.upper)]
    rank_scale = interval_axis(rank_bounds; target_intervals=5)
    panels = (("accuracy", "A. Impact on ranking\naccuracy"),
        ("C", "B. Impact on broker\nranking advantage"),
        ("D", "C. Impact on the brokered\nshare of net output"))
    for (column, (panel, title)) in enumerate(panels)
        series = panel == "accuracy" ?
            ((; key="B", color=PUB_PRINCIPAL, offset=-0.14, marker=:circle),
             (; key="A", color=PUB_BROKER, offset=0.14, marker=:diamond)) :
            ((; key=panel, color=PUB_BROKER, offset=0.0, marker=:circle),)
        grid = GridLayout(fig[2, column])
        Label(grid[1, 1], title; fontsize=22, font=:bold, halign=:left, tellwidth=false)
        xlabel = panel == "D" ? "Difference in brokered share of\nprincipals’ net output (percentage points)" :
            panel == "C" ? "Difference in broker-minus-principal\nrank correlation" :
            "Difference in\nrank correlation"
        ax = Axis(grid[2, 1]; xlabel, yticks=(positions, labels), yreversed=true,
            yticklabelsvisible=column == 1, yticksvisible=false, leftspinevisible=false,
            xgridvisible=true, xgridcolor=(:black, 0.065), xgridwidth=0.8,
            yticklabelpad=18)
        hspan!(ax, [0.55], [1.45]; color=(:gray40, 0.055))
        hspan!(ax, [1.55], [3.45]; color=("#4285B4", 0.035))
        hspan!(ax, [3.55], [5.45]; color=("#C99836", 0.04))
        vlines!(ax, [0]; color=:gray55, linestyle=:dash, linewidth=1.2)
        bounds = Float64[]
        for s in series
            rows = filter(r -> r.panel == s.key, data.rows)
            getproperty.(rows, :condition) == positions || error("condition order differs")
            y = positions .+ s.offset
            colors = [r.lower <= 0 <= r.upper ? "#B5B5B5" : s.color for r in rows]
            rangebars!(ax, y, getproperty.(rows, :lower), getproperty.(rows, :upper);
                direction=:x, color=colors, linewidth=2, whiskerwidth=10)
            scatter!(ax, getproperty.(rows, :mean), y;
                color=colors, marker=s.marker, markersize=12, strokecolor=:white, strokewidth=0.8)
            append!(bounds, [x for r in rows for x in (r.lower, r.upper)])
        end
        if panel == "accuracy"
            elements = [MarkerElement(; color=s.color, marker=s.marker, markersize=11) for s in series]
            split_legend!(ax, elements, ["Principal", "Broker"]; position=:lt)
        end
        scale = panel == "D" ? interval_axis(bounds; target_intervals=5) : rank_scale
        xlims!(ax, scale.limits[1] - 0.15scale.step, scale.limits[2] + 0.15scale.step)
        ax.xticks = scale.ticks
        ylims!(ax, length(positions) + 0.6, 0.4)
        rowgap!(grid, 20)
    end
    colgap!(fig.layout, 44)
    rowgap!(fig.layout, 32)
    save(COMPOSITION_DIFFERENCES_PATH, fig; px_per_unit=2)
    println("Saved ", COMPOSITION_DIFFERENCES_PATH)
    for row in data.rows
        println(data.conditions[row.condition].label, " / ", row.panel, ": ",
            @sprintf("%.4f [%.4f, %.4f]", row.mean, row.lower, row.upper), " (n = $(row.n))")
    end
    return fig
end

"""Keep the four outcomes aligned to one explicitly checked seed vector."""
function ablation_observation(seeds, broker, principal, share; gap=nothing)
    length(seeds) == length(unique(seeds)) > 1 || error("invalid ablation seeds")
    all(v -> length(v) == length(seeds) && all(isfinite, v), (broker, principal, share)) ||
        error("invalid ablation outcomes")
    all(x -> -1 <= x <= 1, [broker; principal]) || error("invalid rank correlation")
    all(x -> 0 <= x <= 100, share) || error("invalid brokered share")
    advantage = broker .- principal
    isnothing(gap) || isapprox(advantage, gap; atol=1e-12) || error("ranking advantage differs")
    values = Dict("A" => Float64.(broker), "B" => Float64.(principal),
        "C" => Float64.(advantage), "D" => Float64.(share))
    return (; seeds=Int.(seeds), values)
end

"""Retain the observed four-arm comparisons and their input-specific provenance."""
function prepare_ablation_composition!(saved)
    base = saved["metadata"]
    sources = Dict(key => deepcopy(base["sources"][key]) for key in ("ridge", "nn_shares", "nn_ranks"))
    file_hash(sources["ridge"]["input"]) == sources["ridge"]["sha256"] || error("Ridge input changed")
    ridge = load(sources["ridge"]["input"])["figdata"]
    ridge["meta"]["late_width"] == length(base["late_periods"]) || error("late windows differ")
    ridge["meta"]["interval_level"] == base["interval_level"] || error("interval levels differ")
    saved["nn_grid_metadata"]["share_definition"] ==
        "mean_of_seed_level_late_net_output_shares" || error("share definitions differ")
    ridge["meta"]["net_output_definition"] == "net_output_per_principal" || error("stale Ridge output")
    base["baseline_difficulty"] == ridge["meta"]["baseline_difficulty"] || error("baseline difficulty differs")
    rhos = (0.0, 1.0)
    deltas = sort(unique(c["delta"] for c in ridge["conditions"] if c["rho"] == rhos[1]))
    shown_deltas = [base["baseline_difficulty"], first(deltas), last(deltas)]
    length(unique(shown_deltas)) == 3 || error("baseline difficulty is not interior")
    eta = base["baseline_turnover"]
    eta == saved["nn_grid_metadata"]["turnover"] || error("NN turnover differs")
    difficulty_conditions = [
        (; label, delta, eta, group=delta == base["baseline_difficulty"] ? "baseline" : "difficulty")
        for (label, delta) in zip(("Baseline difficulty", "Low difficulty", "High difficulty"), shown_deltas)
    ]
    features = [
        (; key="nn", title="How does the impact of broker and principal neural-network learning\nchange when complementarity dominates?", contrast="NN − Ridge, both learners",
            reference="nn", comparison="pair", conditions=difficulty_conditions),
        (; key="pair_data", title="How does the impact of broker learning from data about both parties\nchange when complementarity dominates?", contrast="Reference Ridge − one party recorded",
            reference="pair", comparison="single_principal", conditions=difficulty_conditions),
        (; key="interaction", title="How does the impact of broker learning with an interaction term\nchange when complementarity dominates?", contrast="Reference Ridge − no interaction term",
            reference="pair", comparison="additive", conditions=difficulty_conditions),
    ]
    function learning_observation(model, rho, difficulty)
        if model == "nn"
            series = rho == rhos[2] ? "boundary" : string(difficulty)
            row(panel) = only(filter(r -> r.panel == panel && r.series == series && r.x == rho,
                saved["nn_grid_rows"]))
            b, p, g, s = row.(("A", "B", "C", "D"))
            b.seeds == p.seeds == g.seeds == s.seeds || error("NN seed order differs")
            return ablation_observation(b.seeds, b.values, p.values, s.values; gap=g.values)
        end
        # Difficulty has no effect at pure general quality, represented once.
        c = only(filter(c -> c["rho"] == rho && (rho == rhos[2] || c["delta"] == difficulty),
            ridge["conditions"]))
        return ablation_observation(c["seeds"], c["broker_ranks"][model], c["principal_ranks"][model],
            shares(c["channel_values"][model]); gap=c["rank_gaps"][model])
    end
    cells = NamedTuple[]
    for feature in features, (index, condition) in enumerate(feature.conditions)
        observe(model, rho) = learning_observation(model, rho, condition.delta)
        components = (observe(feature.reference, rhos[1]), observe(feature.comparison, rhos[1]),
            observe(feature.reference, rhos[2]), observe(feature.comparison, rhos[2]))
        push!(cells, (; feature=feature.key, condition=index, components))
    end
    metadata = Dict("features" => features, "rhos" => rhos,
        "definition" => "feature_effect_rho0_minus_feature_effect_rho1",
        "component_weights" => [1, -1, -1, 1],
        "sources" => sources, "interval_level" => base["interval_level"],
        "late_periods" => base["late_periods"], "script_sha256" => file_hash(@__FILE__),
        "monte_carlo_sha256" => file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")),
        "git_commit" => strip(read(`git -C $ROOT rev-parse HEAD`, String)))
    saved["ablation_composition_metadata"], saved["ablation_composition_cells"] = metadata, cells
    jldsave(DATA_PATH; (Symbol(key) => value for (key, value) in saved)...)
    return nothing
end

"""Calculate intervals from paired differences of differences, including ranking gaps."""
function ablation_composition_differences(saved)
    meta, cells = saved["ablation_composition_metadata"], saved["ablation_composition_cells"]
    meta["monte_carlo_sha256"] == file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")) ||
        error("Monte Carlo definition changed")
    rows = NamedTuple[]
    for cell in cells, panel in ("B", "A", "C", "D")
        a, b, c, d = cell.components
        cs, cv = paired_values(a.seeds, a.values[panel], b.seeds, b.values[panel])
        qs, qv = paired_values(c.seeds, c.values[panel], d.seeds, d.values[panel])
        seeds, values = paired_values(cs, cv, qs, qv)
        all(isfinite, values) || error("nonfinite ablation contrast")
        interval = monte_carlo_interval(values; level=meta["interval_level"])
        interval.n == length(seeds) > 1 || error("incomplete ablation interval")
        push!(rows, (; feature=cell.feature, condition=cell.condition, panel, seeds, values, interval...))
    end
    return (; metadata=meta, rows)
end

"""Show learning comparisons on continuous difficulty axes in aligned outcome columns."""
function render_ablation_composition(data)
    publication_theme!()
    features = data.metadata["features"]
    turnover = only(unique(c.eta for feature in features for c in feature.conditions))
    fig = Figure(; size=(1550, 220 + 294length(features)), figure_padding=(28, 32, 30, 24))
    header = GridLayout(fig[1, 1:3])
    Label(header[1, 1], "Moving from quality to complementarity";
        fontsize=TITLE_FS, font=:bold, halign=:left)
    Label(header[1, 2], "ρ = 1 → ρ = 0 · turnover η = $turnover";
        fontsize=FOOTER_FS, color=:gray35, halign=:right)
    Label(fig[2, 1:3], "Change in each feature’s contribution";
        fontsize=LABEL_FS, halign=:left)
    titles = ("Ranking accuracy", "Broker ranking advantage", "Brokered share of net output")
    for (column, title) in enumerate(titles)
        Label(fig[3, column], title; fontsize=21, font=:bold, halign=:left, tellwidth=false)
    end
    elements = [MarkerElement(; color=PUB_PRINCIPAL, marker=:circle, markersize=10),
        MarkerElement(; color=PUB_BROKER, marker=:diamond, markersize=10)]
    Legend(fig[4, 1], elements, ["Principal", "Broker"]; PUB_LEGEND..., labelsize=17, halign=:left)
    Label(fig[4, 2:3], "Difficulty δ applies at ρ = 0; it is not applicable at ρ = 1.";
        fontsize=17, color=:gray35, halign=:left, tellwidth=false)
    rank_bounds = [x for r in data.rows if r.panel != "D" for x in (r.lower, r.upper)]
    rank_limits = interval_axis(rank_bounds; target_intervals=12)
    rank_ticks = interval_axis(rank_bounds; target_intervals=6)
    rank_scale = (; limits=rank_limits.limits, step=rank_limits.step,
        ticks=filter(x -> rank_limits.limits[1] <= x <= rank_limits.limits[2], rank_ticks.ticks))
    share_scale = interval_axis([x for r in data.rows if r.panel == "D" for x in (r.lower, r.upper)];
        target_intervals=4)
    for (index, feature) in enumerate(features)
        title_row, plot_row = 2index + 3, 2index + 4
        Label(fig[title_row, 1:3], "$(Char('A' + index - 1)). $(feature.title)";
            fontsize=21, font=:bold, halign=:left, justification=:left, tellwidth=false)
        # Keep the experimental contrast explicit without using a caption in the image.
        Label(fig[title_row, 2:3], feature.contrast;
            fontsize=17, color=:gray35, halign=:right, tellwidth=false)
        order = sortperm(feature.conditions; by=c -> c.delta)
        conditions = feature.conditions[order]
        positions = getproperty.(conditions, :delta)
        delta_min, delta_max = extrema(positions)
        delta_span = delta_max - delta_min
        delta_span > 0 || error("difficulty axis needs distinct levels")
        ticks = positions
        for (column, panel) in enumerate(("accuracy", "C", "D"))
            bottom = index == length(features)
            xlabel = !bottom ? "" : panel == "D" ? "Change in contribution to principals’\nnet output share (percentage points)" :
                panel == "C" ? "Change in contribution to\nbroker ranking advantage" :
                "Change in contribution to\nrank correlation"
            scale = panel == "D" ? share_scale : rank_scale
            ax = Axis(fig[plot_row, column]; xlabel, ylabel=column == 1 ? "Difficulty (δ)" : "",
                yticks=(ticks, [@sprintf("%.2g", x) for x in ticks]),
                yticklabelsvisible=column == 1, yticklabelsize=17, yticksvisible=column == 1,
                leftspinevisible=column == 1, bottomspinevisible=bottom, xticklabelsvisible=bottom,
                xticksvisible=bottom, xticks=scale.ticks, xgridvisible=true,
                xgridcolor=(:black, 0.065), xgridwidth=0.8, yticklabelpad=18)
            for condition in conditions
                condition.group == "baseline" || continue
                hspan!(ax, [condition.delta - 0.06delta_span], [condition.delta + 0.06delta_span];
                    color=(:gray40, 0.055))
            end
            vlines!(ax, [0]; color=:gray55, linestyle=:dash, linewidth=1.1)
            series = panel == "accuracy" ?
                ((; key="B", color=PUB_PRINCIPAL, marker=:circle),
                 (; key="A", color=PUB_BROKER, marker=:diamond)) :
                ((; key=panel, color=PUB_BROKER, marker=:circle),)
            for s in series
                rows = filter(r -> r.feature == feature.key && r.panel == s.key, data.rows)
                getproperty.(rows, :condition) == collect(eachindex(feature.conditions)) ||
                    error("ablation condition order differs")
                rows = rows[order]
                all(r -> scale.limits[1] <= r.lower <= r.upper <= scale.limits[2], rows) ||
                    error("ablation axis would clip an interval")
                colors = [r.lower <= 0 <= r.upper ? "#B5B5B5" : s.color for r in rows]
                rangebars!(ax, positions, getproperty.(rows, :lower), getproperty.(rows, :upper);
                    direction=:x, color=colors, linewidth=2, whiskerwidth=10)
                scatter!(ax, getproperty.(rows, :mean), positions;
                    color=colors, marker=s.marker, markersize=12, strokecolor=:white, strokewidth=0.7)
            end
            xlims!(ax, scale.limits[1] - 0.12scale.step, scale.limits[2] + 0.12scale.step)
            ylims!(ax, delta_min - 0.12delta_span, delta_max + 0.12delta_span)
        end
    end
    colgap!(fig.layout, 44)
    rowgap!(fig.layout, 16)
    save(ABLATION_COMPOSITION_PATH, fig; px_per_unit=2)
    println("Saved ", ABLATION_COMPOSITION_PATH)
    for r in data.rows
        println(r.feature, " / ", r.condition, " / ", r.panel, ": ",
            @sprintf("%.4f [%.4f, %.4f]", r.mean, r.lower, r.upper), " (n = $(r.n))")
    end
    return fig
end

"""Render the companion ablation figure, retaining original figures and source cells."""
function ablation_composition_main()
    saved = load(DATA_PATH)
    haskey(saved, "ablation_composition_cells") || prepare_ablation_composition!(saved)
    return render_ablation_composition(ablation_composition_differences(saved))
end

"""Prepare from source inputs when requested, otherwise use the retained dataset."""
function main(args=ARGS)
    args == ["--split"] && return split_main()
    args == ["--nn-vs-ridge-difficulty"] && return render_difficulty()
    args == ["--nn-grid"] && return nn_grid_main()
    args == ["--nn-difficulty-grid"] && return nn_grid_main(; by_difficulty=true)
    args == ["--composition-differences"] &&
        return render_composition_differences(composition_differences(load(DATA_PATH)))
    args == ["--ablation-composition-differences"] && return ablation_composition_main()
    if !isempty(args) && first(args) == "--turnover-grid"
        length(args) in (1, 2) || error("expected --turnover-grid [EXTRACT]")
        return turnover_grid_main(length(args) == 2 ? args[2] : nothing)
    end
    if !isempty(args)
        length(args) == 4 && first(args) == "--prepare" || error("expected --prepare AA_SHARES NN_SHARES NN_VALUES")
        data = prepare(args[2:4]...)
    else
        saved = load(DATA_PATH)
        data = (; metadata=saved["metadata"], rows=saved["rows"])
    end
    data.metadata["monte_carlo_sha256"] == file_hash(joinpath(@__DIR__, "..", "monte_carlo.jl")) ||
        error("Monte Carlo definition changed; rebuild the retained estimates")
    for panel in ("A", "B", "C", "D")
        any(r -> r.panel == panel, data.rows) || error("missing panel $panel")
    end
    return render(data)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    MatchingProblemFigure.main()
end
