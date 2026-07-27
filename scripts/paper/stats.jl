"""
    scripts/paper/stats.jl

Compute every statistic quoted in the paper's results section and emit it to
paper/values.tex as \\pvDefine{key}{value} pairs, from the saved sweep data ONLY.
No simulation. No hard-coded results: every emitted value is derived from data.jld2
files and their config metadata at run time. Literal constants below are selection
conventions only (window bounds, baseline parameter values, display rounding).

Conventions:
  late mean  = time average over t in [181, 200] (headline statistic)
  early mean = time average over t in [50, 70]
  "across regimes" = unweighted mean over all saved cells
  (each cell first averaged over its 5 seeds)

Usage: julia --project scripts/paper/stats.jl
"""

using JLD2, DataFrames, Statistics, Printf, Dates

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUTTEX = normpath(joinpath(@__DIR__, "..", "..", "paper", "values.tex"))
const LATE = (181, 200)
const EARLY = (50, 70)
const BASELINE_REL = "oat/rho=0.5"   # baseline regime cell (defaults everywhere else)

nm(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
winm(d, v, (lo, hi)) = nm(v[(d.period .>= lo) .& (d.period .<= hi)])
cellw(m, f, w) = nm([winm(d, f(d), w) for d in m])     # seed mean of window means
late(m, f) = cellw(m, f, LATE);
early(m, f) = cellw(m, f, EARLY)
col(c) = d -> d[!, c]
function ncor(x, y)
    k = .!isnan.(x) .& .!isnan.(y)
    sum(k) < 3 ? NaN : cor(x[k], y[k])
end

# display formats (rounding conventions, not results)
f1(x) = @sprintf("%.1f", x)
f2(x) = @sprintf("%.2f", x)
fs2(x) = (x >= 0 ? "+" : "") * f2(x)                     # explicit sign for gaps/corrs
fint(x) = string(round(Int, x))
fcomma(x) = replace(fint(x), r"(?<=\d)(?=(\d{3})+$)" => ",")

# derived per-period series
function accessf(d)
    (
        t=(d.access_count .+ d.assessment_count);
        [t[i] > 0 ? d.access_count[i] / t[i] : NaN for i in eachindex(t)]
    )
end
ogap(d) = d.q_broker_mean .- d.q_self_mean
rankgap(d) = d.broker_holdout_rank .- d.agent_holdout_rank

# ── load all cells ──
struct Cell
    ;
    rel::String;
    mdfs::Vector{DataFrame};
    cfg::Dict;
    seeds::Vector{Int};
end
cells = Cell[]
for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
    "data.jld2" in files || continue
    m, c, s = jldopen(joinpath(root, "data.jld2"), "r") do f
        ;
        (f["mdfs"], f["config"], f["seeds"])
    end
    push!(cells, Cell(replace(root, ROOT * "/" => ""), m, c, s))
end
B = cells
cellat(rel) = first(c for c in cells if c.rel == rel)
BL = cellat(BASELINE_REL)
oatb(ax, v) = cellat("oat/$ax=$v")
pcell(i, j) = cellat("phase/rho_delta/cells/$(i)_$(j)")  # x=rho{0,0.5,1}, y=delta{0,0.5,0.75}

# ── emitted values, in order ──
VALS = Pair{String,String}[]
pv(k, v) = (push!(VALS, k => v); println(rpad(k, 28), v); v)

# counts
pv("nRegimes", fint(length(B)))
pv("nRuns", fint(sum(length(c.seeds) for c in cells)))

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
d0 = [late(c.mdfs, col(:broker_holdout_rank)) for c in B if c.cfg["delta"] == 0.0]
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
pv("brokerRankHardCorner", f2(late(pcell(0, 2).mdfs, col(:broker_holdout_rank))))
pv("brokerR2Rho0D0", fs2(late(pcell(0, 0).mdfs, col(:broker_holdout_r2))))
pv("brokerR2Rho0D05", f2(late(pcell(0, 1).mdfs, col(:broker_holdout_r2))))
pv("brokerR2Rho0D075", f2(late(pcell(0, 2).mdfs, col(:broker_holdout_r2))))
pv("r2GapHardCorner", f2(late(pcell(0, 2).mdfs, col(:r2_gap))))
pv("ogapRho0OAT", fs2(late(oatb("rho", "0.0").mdfs, ogap)))
pv("ogapRho1OAT", fs2(late(oatb("rho", "1.0").mdfs, ogap)))
pv("rankGapRho0D0", f2(late(pcell(0, 0).mdfs, rankgap)))
pv("rankGapRho0D075", f2(late(pcell(0, 2).mdfs, rankgap)))

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
# rho groups: symmetric composition across levels = the rho-family cells only
# (the OAT rho cell plus the rho-paired phase-grid cells at that level)
rhofam(c) = startswith(c.rel, "oat/rho=") || startswith(c.rel, "phase/rho_")
rhocells(rv) = [c for c in B if c.cfg["rho"] == rv && rhofam(c)]
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
open(OUTTEX, "w") do io
    println(
        io,
        "% values.tex: generated by scripts/paper/stats.jl on ",
        Dates.format(now(), "yyyy-mm-dd HH:MM"),
    )
    println(io, "% sweep: ", basename(ROOT), "   cells: ", length(B))
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
