"""
    scripts/paper/ridge_supplement.jl

Render the focused paired-Ridge robustness figure used in the Supplementary
Material and emit the small set of Ridge values quoted in the paper. The input
is `output/ridge/paired/figdata.jld2`, extracted from the complete paired-Ridge
sweep by `scripts/paper/figdata.jl`.

Outputs:

  * `output/supplement/figs/supp_S5_ridge_robustness.png`
  * `output/ridge/paired/results/paper_values.tex`

No raw sweep access is required. Every displayed value and interval is computed
from the retained seed-level figure data. The input and two output paths can be
overridden with `BROKERAGE_ABM_RIDGE_FIGDATA_PATH`,
`BROKERAGE_ABM_RIDGE_SUPPLEMENT_FIGURE_PATH`, and
`BROKERAGE_ABM_RIDGE_PAPER_VALUES_PATH`.
"""

include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

using JLD2
using Printf: @sprintf
using Statistics: median

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = get(
    ENV,
    "BROKERAGE_ABM_RIDGE_FIGDATA_PATH",
    joinpath(REPO, "output", "ridge", "paired", "figdata.jld2"),
)
const FIGURE = get(
    ENV,
    "BROKERAGE_ABM_RIDGE_SUPPLEMENT_FIGURE_PATH",
    joinpath(REPO, "output", "supplement", "figs", "supp_S5_ridge_robustness.png"),
)
const VALUES = get(
    ENV,
    "BROKERAGE_ABM_RIDGE_PAPER_VALUES_PATH",
    joinpath(REPO, "output", "ridge", "paired", "results", "paper_values.tex"),
)
const BASELINE_REL = "oat/rho=0.5"
const DELTA_COLORS = Dict(
    0.0 => :steelblue,
    0.25 => :cadetblue,
    0.5 => :goldenrod,
    0.75 => :darkorange,
    1.0 => :firebrick,
)

const PROVENANCE = reporting_git_provenance(REPO)
isfile(INPUT) || error("missing paired-Ridge figure data: $INPUT")
const FD = JLD2.load(INPUT)["figdata"]
const META = FD["meta"]
META["analysis_git_commit"] == PROVENANCE.commit ||
    error("paired-Ridge figure data were extracted by a different analysis commit")
META["analysis_source_clean"] == true ||
    error("paired-Ridge figure data were extracted from dirty analysis sources")
META["learning_model"] == "ridge" || error("expected paired-Ridge figure data")

const REGIMES = FD["regime_cells"]
const GRID = FD["grid_cells"]
length(REGIMES) == 98 || error("expected 98 effective paired-Ridge realizations")
all(
    length(cell["seeds"]) == (cell["rel"] == BASELINE_REL ? 50 : 20) for
    cell in REGIMES
) || error("unexpected paired-Ridge seed plan")

rank_interval(cell) = monte_carlo_interval(cell["seed_values"]["rankgap"])
grid_rank_interval(cell) =
    monte_carlo_interval(cell["outcome_seed_values"]["Rank correlation gap"])

const REGIME_POINTS = sort(
    [
        let interval = rank_interval(cell)
            (
                rel=cell["rel"],
                mean=interval.mean,
                lower=interval.lower,
                upper=interval.upper,
            )
        end for cell in REGIMES
    ];
    by=point -> point.mean,
)
const BASELINE = only(point for point in REGIME_POINTS if point.rel == BASELINE_REL)
const RIDGE_MEDIAN = median(point.mean for point in REGIME_POINTS)
const POSITIVE_N = count(point -> point.mean > 0.0, REGIME_POINTS)

function ridge_robustness_figure()
    fig = Figure(; size=(1280, 580), fontsize=16)

    ax_all = Axis(
        fig[1, 1];
        title="A. Across effective realizations",
        xlabel="Effective realization, ordered by gap",
        ylabel="Broker - principal rank-correlation gap",
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
    )
    indices = collect(eachindex(REGIME_POINTS))
    hlines!(ax_all, [0.0]; color=:gray55, linestyle=:dash, linewidth=1.3)
    rangebars!(
        ax_all,
        indices,
        [point.lower for point in REGIME_POINTS],
        [point.upper for point in REGIME_POINTS];
        color=(:steelblue, 0.42),
        linewidth=0.8,
        whiskerwidth=2,
    )
    scatter!(
        ax_all,
        indices,
        [point.mean for point in REGIME_POINTS];
        color=(:steelblue, 0.82),
        markersize=6,
    )

    ax_grid = Axis(
        fig[1, 2];
        title="B. Matching-complexity grid",
        xlabel="ρ (complementarity vs quality)",
        ylabel="Broker - principal rank-correlation gap",
        xticks=[0, 0.3, 0.5, 0.7, 0.85, 1],
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
    )
    hlines!(ax_grid, [0.0]; color=:gray55, linestyle=:dash, linewidth=1.3)
    for delta in sort(unique(cell["delta"] for cell in GRID))
        points = sort(
            [
                let interval = grid_rank_interval(cell)
                    (
                        rho=cell["rho"],
                        mean=interval.mean,
                        lower=interval.lower,
                        upper=interval.upper,
                    )
                end for cell in GRID if cell["delta"] == delta
            ];
            by=point -> point.rho,
        )
        rangebars!(
            ax_grid,
            [point.rho for point in points],
            [point.lower for point in points],
            [point.upper for point in points];
            color=(DELTA_COLORS[delta], 0.65),
            linewidth=1.0,
            whiskerwidth=6,
        )
        scatterlines!(
            ax_grid,
            [point.rho for point in points],
            [point.mean for point in points];
            color=DELTA_COLORS[delta],
            linewidth=1.8,
            markersize=8,
            label="δ = $delta",
        )
    end
    Legend(
        fig[0, 1:2],
        ax_grid,
        "Regime gain";
        orientation=:horizontal,
        framevisible=false,
        tellwidth=false,
    )
    colgap!(fig.layout, 18)

    mkpath(dirname(FIGURE))
    save(FIGURE, fig; px_per_unit=2.0)
    println("wrote $FIGURE")
    return nothing
end

function write_values()
    mkpath(dirname(VALUES))
    open(VALUES, "w") do io
        println(io, "% Generated by scripts/paper/ridge_supplement.jl.")
        println(io, "% Analysis commit: $(PROVENANCE.commit)")
        println(io, "% Analysis source clean: $(PROVENANCE.source_clean)")
        println(io, "% Ridge sweep: ", META["sweep"])
        println(io, "\\pvDefine{ridgeNConditions}{$(length(REGIME_POINTS))}")
        println(io, "\\pvDefine{ridgePositiveN}{$POSITIVE_N}")
        println(
            io, "\\pvDefine{ridgeBaselineGap}{", @sprintf("%.3f", BASELINE.mean), "}"
        )
        println(io, "\\pvDefine{ridgeMedianGap}{", @sprintf("%.3f", RIDGE_MEDIAN), "}")
    end
    println("wrote $VALUES")
    return nothing
end

ridge_robustness_figure()
write_values()
