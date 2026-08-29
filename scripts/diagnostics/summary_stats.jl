"""
    summary_stats.jl

Print descriptive statistics from a completed single-model sweep. No simulation
is run. All summaries use the same early and late windows as the paper pipeline.

Usage: BROKERAGE_ABM_SWEEP_DIR=<sweep root> julia --project scripts/diagnostics/summary_stats.jl
"""

using Statistics

include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end

nm(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
r2(x) = round(x; digits=2)
win(m, c, lo, hi) = nm([nm(d[(d.period .>= lo) .& (d.period .<= hi), c]) for d in m])
early(m, c) = win(m, c, 50, 70)
late(m, c) = nm([nm(d[d.period .>= maximum(d.period) - 19, c]) for d in m])
function accfrac(s)
    nm([
        (t=s.access_count[i] + s.assessment_count[i]; t > 0 ? s.access_count[i] / t : NaN)
        for i in 1:size(s, 1)
    ])
end
accwin(m, lo, hi) = nm([accfrac(d[(d.period .>= lo) .& (d.period .<= hi), :]) for d in m])
acctail(m) = nm([accfrac(d[d.period .>= maximum(d.period) - 19, :]) for d in m])

const SWEEP = load_sweep_dataset(ROOT)
const cells = SWEEP.results
println("effective realizations: $(length(cells))")
cellat(rel) = grid_result(SWEEP, rel)
BL = cellat("oat/rho=0.5")

agg(f) = nm([f(c.mdfs) for c in cells])
rng(f) = (v=filter(!isnan, [f(c.mdfs) for c in cells]); (minimum(v), maximum(v)))
sec(t) = println("\n========== $t ==========")
out(m) = late(m, :outsourcing_rate)
qgap(m) = late(m, :q_broker_mean) - late(m, :q_self_mean)
qedge(m) = 100 * qgap(m) / late(m, :q_self_mean)

sec("OVERVIEW / OUTSOURCING")
println("baseline outsourcing (late mean): ", r2(out(BL.mdfs)))
println("range across effective realizations: ", r2.(rng(out)))
println(
    "baseline early->late: ",
    r2(early(BL.mdfs, :outsourcing_rate)),
    " -> ",
    r2(late(BL.mdfs, :outsourcing_rate)),
)
println(
    "effective realizations early->late (means): ",
    r2(agg(m -> early(m, :outsourcing_rate))),
    " -> ",
    r2(agg(m -> late(m, :outsourcing_rate))),
)
println(
    "outsourcing vs eta: ",
    [(e, r2(out(cellat("oat/eta=$e").mdfs))) for e in ETA_VALS],
)
println(
    "outsourcing vs reservation: ",
    [(r, r2(out(cellat("oat/reservation_frac=$r").mdfs))) for r in (0.4, 0.6, 0.9, 1.2)],
)
println(
    "outsourcing vs rho: ",
    [(r, r2(out(cellat("oat/rho=$r").mdfs))) for r in (0.0, 0.3, 0.5, 0.7, 0.85, 1.0)],
)
println(
    "access fraction vs rho: ",
    [(r, r2(acctail(cellat("oat/rho=$r").mdfs))) for r in (0.0, 0.3, 0.5, 0.7, 0.85, 1.0)],
)
println(
    "output gap mean over effective realizations: ",
    r2(agg(qgap)),
    "  range: ",
    r2.(rng(qgap)),
)
println(
    "output edge % vs eta: ",
    [(e, round(Int, qedge(cellat("oat/eta=$e").mdfs))) for e in ETA_VALS],
)

sec("RHO x DELTA GRID")
metrics = [
    ("betweenness", m -> late(m, :betweenness)),
    ("access", acctail),
    ("brkR2", m -> late(m, :broker_holdout_r2)),
    ("R2gap", m -> late(m, :broker_holdout_r2) - late(m, :agent_holdout_r2)),
    ("brkRank", m -> late(m, :broker_holdout_rank)),
    ("rankGap", m -> late(m, :broker_holdout_rank) - late(m, :agent_holdout_rank)),
    ("qBrk", m -> late(m, :q_broker_mean)),
    ("qGap", qgap),
    ("outsrc", out),
]
rho_delta_cells = sort(
    [cell for cell in SWEEP.grid_cells if get(cell, :pair, nothing) == "rho_delta"];
    by=cell -> (cell[:xi], cell[:yi]),
)
for grid in rho_delta_cells
    rho = grid[:resolved_params][:rho]
    delta = grid[:resolved_params][:delta]
    m = grid_result(SWEEP, grid[:reldir]).mdfs
    print("rho=$rho delta=$delta: ")
    for (name, f) in metrics
        print("$name=", r2(f(m)), " ")
    end
    println()
end

sec("NETWORK TOPOLOGY TRENDS")
println(
    "baseline mean degree early->late: ",
    r2(early(BL.mdfs, :mean_degree)),
    " -> ",
    r2(late(BL.mdfs, :mean_degree)),
    "   betweenness: ",
    r2(early(BL.mdfs, :betweenness)),
    " -> ",
    r2(late(BL.mdfs, :betweenness)),
)
for (name, col) in (
    ("mean degree", :mean_degree),
    ("median degree", :median_degree),
    ("betweenness", :betweenness),
)
    println(
        "$name, effective realizations early->late: ",
        r2(agg(m -> early(m, col))),
        " -> ",
        r2(agg(m -> late(m, col))),
    )
end

sec("ACCESS FRACTION")
println(
    "late mean across effective realizations: ",
    r2(agg(acctail)),
    "  range: ",
    r2.(rng(acctail)),
)
println("baseline early->late: ", r2(accwin(BL.mdfs, 50, 70)), " -> ", r2(acctail(BL.mdfs)))
increasing = [c.rel for c in cells if acctail(c.mdfs) > accwin(c.mdfs, 50, 70)]
println(
    "effective realizations where access increases: " *
    "$(length(increasing))/$(length(cells))",
)

function pearson(x, y)
    (
        k=findall(i -> !isnan(x[i]) && !isnan(y[i]), eachindex(x));
        length(k) < 3 ? NaN : cor(x[k], y[k])
    )
end

sec("ADVANTAGE CORRELATIONS")
bw = [late(c.mdfs, :betweenness) for c in cells]
ac = [acctail(c.mdfs) for c in cells]
rg = [late(c.mdfs, :broker_holdout_rank) - late(c.mdfs, :agent_holdout_rank) for c in cells]
qg = [qgap(c.mdfs) for c in cells]
br = [late(c.mdfs, :broker_holdout_rank) for c in cells]
println(
    "corr(betweenness, rank gap) = ",
    r2(pearson(bw, rg)),
    "   corr(betweenness, output gap) = ",
    r2(pearson(bw, qg)),
)
println(
    "corr(access, rank gap) = ",
    r2(pearson(ac, rg)),
    "   corr(access, output gap) = ",
    r2(pearson(ac, qg)),
)
println(
    "corr(broker rank, output gap) = ",
    r2(pearson(br, qg)),
    "   corr(rank gap, output gap) = ",
    r2(pearson(rg, qg)),
)

println("\nDONE")
