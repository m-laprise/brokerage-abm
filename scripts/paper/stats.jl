"""
    scripts/paper/stats.jl

Compute every statistic quoted in the paper's results section and emit it to
output/main/values.tex as \\pvDefine{key}{value} pairs, from the saved sweep data ONLY.
No simulation. No hard-coded results: every emitted value is derived from data.jld2
files and their config metadata at run time. Literal constants below are selection
conventions only (window bounds, baseline parameter values, display rounding).

Conventions:
  late mean  = time average over the final 20 periods (headline statistic)
  early mean = time average over t in [50, 70]
  "across regimes" = unweighted mean over effective model realizations
  (each realization first averaged over its planned seeds)

Usage: julia --project scripts/paper/stats.jl
"""

using Statistics, Printf, Dates

include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUTTEX =
    normpath(joinpath(@__DIR__, "..", "..", "output", "main", "values.tex"))
const LATE_WIDTH = 20
const EARLY = (50, 70)
const BASELINE_REL = "oat/rho=0.5"   # baseline regime cell (defaults everywhere else)

nm(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
winm(d, v, (lo, hi)) = nm(v[(d.period .>= lo) .& (d.period .<= hi)])
cellw(m, f, w) = nm([winm(d, f(d), w) for d in m])     # seed mean of window means
late_window(m) = (maximum(m[1].period) - LATE_WIDTH + 1, maximum(m[1].period))
late(m, f) = cellw(m, f, late_window(m));
early(m, f) = cellw(m, f, EARLY)
col(c) = d -> d[!, c]
# display formats (rounding conventions, not results)
f1(x) = @sprintf("%.1f", x)
f2(x) = @sprintf("%.2f", x)
fs2(x) = (x >= 0 ? "+" : "") * f2(x)                     # explicit sign for gaps/corrs
fint(x) = string(round(Int, x))

# derived per-period series
function accessf(d)
    (
        t=(d.access_count .+ d.assessment_count);
        [t[i] > 0 ? d.access_count[i] / t[i] : NaN for i in eachindex(t)]
    )
end
ogap(d) = d.q_broker_mean .- d.q_self_mean
rankgap(d) = d.broker_holdout_rank .- d.agent_holdout_rank

# ── load the complete set of effective model realizations ──
const SWEEP = load_sweep_dataset(ROOT)
const B = SWEEP.results
cellat(rel) = grid_result(SWEEP, rel)
BL = cellat(BASELINE_REL)
const BASELINE_N_SEEDS = length(SWEEP.result_by_rel[BASELINE_REL].seeds)
const GENERAL_SEED_COUNTS = [
    length(result.seeds) for result in B if result.rel != BASELINE_REL
]
all(==(20), GENERAL_SEED_COUNTS) || error("expected 20 seeds outside the baseline")
BASELINE_N_SEEDS == 50 || error("expected 50 baseline seeds")
oatb(ax, v) = cellat("oat/$ax=$v")
function rdcell(rho, delta)
    first(
        grid_result(SWEEP, cell[:reldir]) for
        cell in SWEEP.grid_cells if get(cell, :pair, nothing) == "rho_delta" &&
            cell[:resolved_params][:rho] == rho &&
            cell[:resolved_params][:delta] == delta
    )
end

# ── emitted values, in order ──
VALS = Pair{String,String}[]
pv(k, v) = (push!(VALS, k => v); println(rpad(k, 28), v); v)

# counts
pv("nRegimes", fint(length(B)))
pv("nRuns", fint(sum(length(c.seeds) for c in B)))
pv("nGeneralSeeds", fint(only(unique(GENERAL_SEED_COUNTS))))
pv("nBaselineSeeds", fint(BASELINE_N_SEEDS))

# ── 5.1 ──
pv("outBaselineLate", f2(late(BL.mdfs, col(:outsourcing_rate))))
pv("outBaselineEarly", f2(early(BL.mdfs, col(:outsourcing_rate))))
ob = [late(c.mdfs, col(:outsourcing_rate)) for c in B]
pv("outBaseMin", f2(minimum(ob)));
pv("outBaseMax", f2(maximum(ob)))
og = [late(c.mdfs, ogap) for c in B]
pv("ogapBaseMean", fs2(nm(og)))
pv("ogapBasePosN", fint(count(>(0), filter(!isnan, og))))
ac_l = [late(c.mdfs, accessf) for c in B];
ac_e = [early(c.mdfs, accessf) for c in B]
pv("accessAcrossBaseLate", f2(nm(ac_l)))
pv("accessAcrossBaseEarly", f2(nm(ac_e)))
pv("brokerRankBaseline", f2(late(BL.mdfs, col(:broker_holdout_rank))))
d0 = [
    late(c.mdfs, col(:broker_holdout_rank)) for c in unique_effective_results([
        grid_result(SWEEP, cell[:reldir]) for cell in SWEEP.grid_cells if
        get(cell, :pair, nothing) == "rho_delta" && cell[:resolved_params][:delta] == 0.0
    ],)
]
pv("brokerRankD0Min", f2(minimum(d0)));
pv("brokerRankD0Max", f2(maximum(d0)))
pv("brokerR2Baseline", f2(late(BL.mdfs, col(:broker_holdout_r2))))
pv("agentRankBaseline", f2(late(BL.mdfs, col(:agent_holdout_rank))))
pv("agentR2BaselineBase", f2(late(BL.mdfs, col(:agent_holdout_r2))))

# ── Table 1 (section 1): baseline early/late + across-regime median ──
# Across-regime summary is the median of the late mean over the regimes;
# the assessment-quality rows are late-only, so their early cells are blank.
med(v) = median(filter(!isnan, v))
pv("ogapBaselineEarly", fs2(early(BL.mdfs, ogap)))
pv("ogapBaselineLate", fs2(late(BL.mdfs, ogap)))
pv("accessBaselineEarly", f2(early(BL.mdfs, accessf)))
pv("accessBaselineLate", f2(late(BL.mdfs, accessf)))
pv("brokerRankBaselineEarly", f2(early(BL.mdfs, col(:broker_holdout_rank))))
pv("brokerR2BaselineEarly", f2(early(BL.mdfs, col(:broker_holdout_r2))))
pv("agentRankBaselineEarly", f2(early(BL.mdfs, col(:agent_holdout_rank))))
pv("agentR2BaselineEarly", f2(early(BL.mdfs, col(:agent_holdout_r2))))
pv("outBaseMed", f2(med(ob)))
pv("ogapBaseMed", fs2(med(og)))
pv("accessBaseMed", f2(med(ac_l)))
# table summary = median over regimes; prose parentheticals = mean
brk_rank = [late(c.mdfs, col(:broker_holdout_rank)) for c in B]
brk_r2 = [late(c.mdfs, col(:broker_holdout_r2)) for c in B]
agt_rank = [late(c.mdfs, col(:agent_holdout_rank)) for c in B]
agt_r2 = [late(c.mdfs, col(:agent_holdout_r2)) for c in B]
pv("brokerRankBaseMed", f2(med(brk_rank)));
pv("brokerRankBaseMean", f2(nm(brk_rank)))
pv("brokerR2BaseMed", f2(med(brk_r2)));
pv("brokerR2BaseMean", f2(nm(brk_r2)))
pv("agentRankBaseMed", f2(med(agt_rank)));
pv("agentRankBaseMean", f2(nm(agt_rank)))
pv("agentR2BaseMed", f2(med(agt_r2)));
pv("agentR2BaseMean", f2(nm(agt_r2)))

# ── 5.2 ──
pv("betwRho0", f2(late(oatb("rho", "0.0").mdfs, col(:betweenness))))
pv("betwRho1", f2(late(oatb("rho", "1.0").mdfs, col(:betweenness))))
pv("accessRho0", f2(late(oatb("rho", "0.0").mdfs, accessf)))
pv("accessRho1", f2(late(oatb("rho", "1.0").mdfs, accessf)))
pv("betwDelta0", f2(late(oatb("delta", "0.0").mdfs, col(:betweenness))))
pv("betwDelta05", f2(late(BL.mdfs, col(:betweenness))))
pv("betwDelta075", f2(late(oatb("delta", "0.75").mdfs, col(:betweenness))))
pv("brokerRankHardCorner", f2(late(rdcell(0.0, 0.75).mdfs, col(:broker_holdout_rank))))
pv("brokerR2Rho0D0", fs2(late(rdcell(0.0, 0.0).mdfs, col(:broker_holdout_r2))))
pv("brokerR2Rho0D05", f2(late(rdcell(0.0, 0.5).mdfs, col(:broker_holdout_r2))))
pv("brokerR2Rho0D075", f2(late(rdcell(0.0, 0.75).mdfs, col(:broker_holdout_r2))))
pv("r2GapHardCorner", f2(late(rdcell(0.0, 0.75).mdfs, col(:r2_gap))))
pv("ogapRho0OAT", fs2(late(oatb("rho", "0.0").mdfs, ogap)))
pv("ogapRho1OAT", fs2(late(oatb("rho", "1.0").mdfs, ogap)))
pv("rankGapRho0D0", f2(late(rdcell(0.0, 0.0).mdfs, rankgap)))
pv("rankGapRho0D075", f2(late(rdcell(0.0, 0.75).mdfs, rankgap)))

# ── 5.3 ──
pv("degBaselineEarly", f1(early(BL.mdfs, col(:mean_degree))))
pv("degBaselineLate", f1(late(BL.mdfs, col(:mean_degree))))
pv("betwBaselineEarly", f2(early(BL.mdfs, col(:betweenness))))
pv("betwBaselineLate", f2(late(BL.mdfs, col(:betweenness))))
dd = [late(c.mdfs, col(:mean_degree)) - early(c.mdfs, col(:mean_degree)) for c in B]
db = [late(c.mdfs, col(:betweenness)) - early(c.mdfs, col(:betweenness)) for c in B]
pv("comoveN", fint(count((dd .< 0) .& (db .> 0))))
pv("degFallsN", fint(count(dd .< 0)))
e1 = cellat("oat/eta=0.01")
pv("betwEta001Early", f2(early(e1.mdfs, col(:betweenness))))
pv("betwEta001Late", f2(late(e1.mdfs, col(:betweenness))))
# Rho groups use common support across rho levels. The rho x delta grid is
# excluded because delta is not an effective dimension at rho=1.
function rhofam(c)
    startswith(c.rel, "oat/rho=") || any(
        startswith(c.rel, "phase/$pair/cells/") for pair in ("rho_eta", "rho_N", "rho_r")
    )
end
function rhocells(rv)
    unique_effective_results([
        result for cell in SWEEP.grid_cells for
        result in (grid_result(SWEEP, cell[:reldir]),) if
        result.cfg["rho"] == rv && rhofam(result)
    ],)
end
grp(rv, f) = nm([f(c) for c in rhocells(rv)])
let ns = [length(rhocells(v)) for v in (0.0, 0.5, 1.0)]
    allequal(ns) || error("rho groups have unequal sizes: $ns")
    pv("rhoGroupN", fint(ns[1]))
end
pv("outRho0Group", f2(grp(0.0, c -> late(c.mdfs, col(:outsourcing_rate)))))
pv("outRho05Group", f2(grp(0.5, c -> late(c.mdfs, col(:outsourcing_rate)))))
pv("outRho1Group", f2(grp(1.0, c -> late(c.mdfs, col(:outsourcing_rate)))))
pv("accessRho0Group", f2(grp(0.0, c -> late(c.mdfs, accessf))))
pv("accessRho05Group", f2(grp(0.5, c -> late(c.mdfs, accessf))))
pv("accessRho1Group", f2(grp(1.0, c -> late(c.mdfs, accessf))))
pv("betwRho0Group", f2(grp(0.0, c -> late(c.mdfs, col(:betweenness)))))
pv("betwRho05Group", f2(grp(0.5, c -> late(c.mdfs, col(:betweenness)))))
pv("betwRho1Group", f2(grp(1.0, c -> late(c.mdfs, col(:betweenness)))))
pv("degRho0Group", f1(grp(0.0, c -> late(c.mdfs, col(:mean_degree)))))
pv("degRho05Group", f1(grp(0.5, c -> late(c.mdfs, col(:mean_degree)))))
pv("degRho1Group", f1(grp(1.0, c -> late(c.mdfs, col(:mean_degree)))))
pv("ogapRho0Group", fs2(grp(0.0, c -> late(c.mdfs, ogap))))
pv("ogapRho05Group", fs2(grp(0.5, c -> late(c.mdfs, ogap))))
pv("ogapRho1Group", fs2(grp(1.0, c -> late(c.mdfs, ogap))))
rho1 = [c for c in B if c.cfg["rho"] == 1.0]
pv("rhoOneN", fint(length(rho1)))
pv(
    "rhoOneThinN",
    fint(
        count(c -> late(c.mdfs, col(:mean_degree)) < early(c.mdfs, col(:mean_degree)), rho1)
    ),
)

# ── emit values.tex ──
mkpath(dirname(OUTTEX))
open(OUTTEX, "w") do io
    println(
        io,
        "% values.tex: generated by scripts/paper/stats.jl on ",
        Dates.format(now(), "yyyy-mm-dd HH:MM"),
    )
    println(
        io,
        "% sweep: ",
        basename(ROOT),
        "   effective realizations: ",
        length(B),
        "   grid coordinates: ",
        length(SWEEP.grid_cells),
    )
    println(io, "% manifest: ", SWEEP.manifest_hash)
    println(io, "% Do not edit by hand; every value is computed from the saved sweep data.")
    println(
        io,
        raw"\newcommand{\pvDefine}[2]{\expandafter\newcommand\csname pv@#1\endcsname{#2}}",
    )
    println(io, raw"\newcommand{\pv}[1]{\csname pv@#1\endcsname}")
    for (k, v) in VALS
        println(io, "\\pvDefine{$k}{$v}")
    end
end
println("\nwrote $(length(VALS)) values to $OUTTEX")
