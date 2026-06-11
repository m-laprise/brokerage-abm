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
  "across regimes" = unweighted mean over all saved cells of that model
  (each cell first averaged over its 5 seeds)

Usage: julia --project scripts/paper/stats.jl
"""

using JLD2, DataFrames, Statistics, Printf, Dates

const ROOT = get(ENV, "TB_SWEEP_DIR") do
    error("set TB_SWEEP_DIR to the sweep root directory")
end
const OUTTEX = normpath(joinpath(@__DIR__, "..", "..", "paper", "values.tex"))
const LATE = (181, 200)
const EARLY = (50, 70)
const BASELINE_REL = "oat/rho=0.5"   # baseline regime cell (defaults everywhere else)

nm(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
winm(d, v, (lo, hi)) = nm(v[(d.period .>= lo) .& (d.period .<= hi)])
cellw(m, f, w) = nm([winm(d, f(d), w) for d in m])     # seed mean of window means
late(m, f) = cellw(m, f, LATE); early(m, f) = cellw(m, f, EARLY)
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
accessf(d) = (t = d.access_count .+ d.assessment_count;
              [t[i] > 0 ? d.access_count[i] / t[i] : NaN for i in eachindex(t)])
ogap(d) = d.q_broker_standard_mean .- d.q_self_mean
rankgap(d) = d.broker_holdout_rank .- d.agent_holdout_rank
agentobs(d) = 2.0 .* (d.n_self_matches .+ d.n_broker_standard)   # matching.jl:52-53; capture.jl records none
prinshare(d) = d.n_broker_principal ./ max.(d.n_total_matches, 1)

# ── load all cells ──
struct Cell; rel::String; model::String; mdfs::Vector{DataFrame}; cfg::Dict; seeds::Vector{Int}; end
cells = Cell[]
for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
    "data.jld2" in files || continue
    model = basename(root); model in ("base", "capture") || continue
    m, c, s = jldopen(joinpath(root, "data.jld2"), "r") do f; (f["mdfs"], f["config"], f["seeds"]) end
    push!(cells, Cell(replace(root, ROOT * "/" => ""), model, m, c, s))
end
B = [c for c in cells if c.model == "base"]
C = [c for c in cells if c.model == "capture"]
cellat(rel) = first(c for c in cells if c.rel == rel)
BL = cellat("$BASELINE_REL/base"); BLC = cellat("$BASELINE_REL/capture")
pairs = [(b, cellat(replace(b.rel, "/base" => "/capture"))) for b in B
         if any(c -> c.rel == replace(b.rel, "/base" => "/capture"), C)]
oatb(ax, v) = cellat("oat/$ax=$v/base"); oatc(ax, v) = cellat("oat/$ax=$v/capture")
pcell(i, j, model) = cellat("phase/rho_delta/cells/$(i)_$(j)/$model")  # x=rho{0,0.5,1}, y=delta{0,0.5,0.75}

# ── emitted values, in order ──
VALS = Pair{String,String}[]
pv(k, v) = (push!(VALS, k => v); println(rpad(k, 28), v); v)

# counts
pv("nBase", fint(length(B)))
pv("nCapture", fint(length(C)))
pv("nPairs", fint(length(pairs)))
pv("nRuns", fint(sum(length(c.seeds) for c in cells)))

# ── 5.1 ──
pv("outBaselineLate", f2(late(BL.mdfs, col(:outsourcing_rate))))
pv("outBaselineEarly", f2(early(BL.mdfs, col(:outsourcing_rate))))
ob = [late(c.mdfs, col(:outsourcing_rate)) for c in B]
pv("outBaseMin", f2(minimum(ob))); pv("outBaseMax", f2(maximum(ob)))
og = [late(c.mdfs, ogap) for c in B]
pv("ogapBaseMean", fs2(nm(og)))
pv("ogapBasePosN", fint(count(>(0), filter(!isnan, og))))
ac_l = [late(c.mdfs, accessf) for c in B]; ac_e = [early(c.mdfs, accessf) for c in B]
pv("accessAcrossBaseLate", f2(nm(ac_l)))
pv("accessAcrossBaseEarly", f2(nm(ac_e)))
pv("accessDecliningBaseN", fint(count(ac_l .< ac_e)))
pv("brokerRankBaseline", f2(late(BL.mdfs, col(:broker_holdout_rank))))
d0 = [late(c.mdfs, col(:broker_holdout_rank)) for c in B if c.cfg["delta"] == 0.0]
pv("brokerRankD0Min", f2(minimum(d0))); pv("brokerRankD0Max", f2(maximum(d0)))
pv("brokerR2Baseline", f2(late(BL.mdfs, col(:broker_holdout_r2))))
pv("agentRankBaseline", f2(late(BL.mdfs, col(:agent_holdout_rank))))
pv("agentR2BaselineBase", f2(late(BL.mdfs, col(:agent_holdout_r2))))

# ── 5.2 ──
pv("betwRho0", f2(late(oatb("rho", "0.0").mdfs, col(:betweenness))))
pv("betwRho1", f2(late(oatb("rho", "1.0").mdfs, col(:betweenness))))
pv("accessRho0", f2(late(oatb("rho", "0.0").mdfs, accessf)))
pv("accessRho1", f2(late(oatb("rho", "1.0").mdfs, accessf)))
pv("betwDelta0", f2(late(oatb("delta", "0.0").mdfs, col(:betweenness))))
pv("betwDelta05", f2(late(BL.mdfs, col(:betweenness))))
pv("betwDelta075", f2(late(oatb("delta", "0.75").mdfs, col(:betweenness))))
pv("brokerRankHardCorner", f2(late(pcell(0, 2, "base").mdfs, col(:broker_holdout_rank))))
pv("brokerR2Rho0D0", fs2(late(pcell(0, 0, "base").mdfs, col(:broker_holdout_r2))))
pv("brokerR2Rho0D05", f2(late(pcell(0, 1, "base").mdfs, col(:broker_holdout_r2))))
pv("brokerR2Rho0D075", f2(late(pcell(0, 2, "base").mdfs, col(:broker_holdout_r2))))
pv("r2GapHardCorner", f2(late(pcell(0, 2, "base").mdfs, col(:r2_gap))))
pv("ogapRho0OAT", fs2(late(oatb("rho", "0.0").mdfs, ogap)))
pv("ogapRho1OAT", fs2(late(oatb("rho", "1.0").mdfs, ogap)))
pv("rankGapRho0D0", f2(late(pcell(0, 0, "base").mdfs, rankgap)))
pv("rankGapRho0D075", f2(late(pcell(0, 2, "base").mdfs, rankgap)))

# ── 5.3 ──
pv("degBaselineEarly", f1(early(BL.mdfs, col(:mean_degree))))
pv("degBaselineLate", f1(late(BL.mdfs, col(:mean_degree))))
pv("betwBaselineEarly", f2(early(BL.mdfs, col(:betweenness))))
pv("betwBaselineLate", f2(late(BL.mdfs, col(:betweenness))))
dd = [late(c.mdfs, col(:mean_degree)) - early(c.mdfs, col(:mean_degree)) for c in B]
db = [late(c.mdfs, col(:betweenness)) - early(c.mdfs, col(:betweenness)) for c in B]
pv("comoveN", fint(count((dd .< 0) .& (db .> 0))))
pv("degFallsN", fint(count(dd .< 0)))
e1 = cellat("oat/eta=0.01/base")
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
pv("rhoOneThinN", fint(count(c -> late(c.mdfs, col(:mean_degree)) < early(c.mdfs, col(:mean_degree)), rho1)))
bt = [late(c.mdfs, col(:betweenness)) for c in B]
brk = [late(c.mdfs, col(:broker_holdout_rank)) for c in B]
pv("corrBetwAccess", f2(ncor(bt, ac_l)))
fam = Dict{String,Vector{Int}}()
for (i, c) in enumerate(B)
    startswith(c.rel, "oat/") || continue
    push!(get!(fam, split(split(c.rel, "/")[2], "=")[1], Int[]), i)
end
pv("corrBetwAccessEta", fs2(ncor(bt[fam["eta"]], ac_l[fam["eta"]])))
pv("corrBetwAccessN", fs2(ncor(bt[fam["N"]], ac_l[fam["N"]])))
pv("corrBetwAccessFr", f2(ncor(bt[fam["reservation_frac"]], ac_l[fam["reservation_frac"]])))
pv("corrBetwAccessRho", f2(ncor(bt[fam["rho"]], ac_l[fam["rho"]])))
pv("corrOgapBetw", fs2(ncor(og, bt)))
pv("corrOgapAccess", f2(ncor(og, ac_l)))
pv("corrOgapBrokerRank", f2(ncor(og, brk)))

# ── 5.4 ──
ps(m) = late(m, col(:principal_mode_share))
pv("capRho0", f2(ps(oatc("rho", "0.0").mdfs)));  pv("capRho03", f2(ps(oatc("rho", "0.3").mdfs)))
pv("capRho05", f2(ps(oatc("rho", "0.5").mdfs))); pv("capRho07", f2(ps(oatc("rho", "0.7").mdfs)))
pv("capRho1", f2(ps(oatc("rho", "1.0").mdfs)))
pv("capFr04", f2(ps(oatc("reservation_frac", "0.4").mdfs)))
pv("capFr06", f2(ps(oatc("reservation_frac", "0.6").mdfs)))
pv("capFr09", f2(ps(oatc("reservation_frac", "0.9").mdfs)))
pv("capFr12", f2(ps(oatc("reservation_frac", "1.2").mdfs)))
pv("capD0", f2(ps(oatc("delta", "0.0").mdfs))); pv("capD075", f2(ps(oatc("delta", "0.75").mdfs)))
pv("capEta001", f2(ps(oatc("eta", "0.01").mdfs))); pv("capEta002", f2(ps(oatc("eta", "0.02").mdfs)))
pv("capEta003", f2(ps(oatc("eta", "0.03").mdfs)))
pv("capN500", f2(ps(oatc("N", "500").mdfs))); pv("capN1000", f2(ps(oatc("N", "1000").mdfs)))
pv("capN1500", f2(ps(oatc("N", "1500").mdfs)))
lr(v) = late(oatc("reservation_frac", v).mdfs, col(:capture_loss_rate))
pv("lossFr04", f2(lr("0.4"))); pv("lossFr06", f2(lr("0.6")))
pv("lossFr09", f2(lr("0.9"))); pv("lossFr12", f2(lr("1.2")))
pv("degCaptureLate", f1(late(BLC.mdfs, col(:mean_degree))))
mpaB = c -> d -> 2.0 .* d.n_total_matches ./ c.cfg["N"]   # matches per agent (both sides)
pv("mpaCaptureLate", f1(late(BLC.mdfs, mpaB(BLC))))
pv("mpaBaseLate", f1(late(BL.mdfs, mpaB(BL))))
pv("betwCaptureLate", f2(late(BLC.mdfs, col(:betweenness))))
pv("accessCaptureLate", f2(late(BLC.mdfs, accessf)))
pv("accessCaptureEarly", f2(early(BLC.mdfs, accessf)))
pv("accessBaselineLate", f2(late(BL.mdfs, accessf)))
acC_l = [late(c.mdfs, accessf) for c in C]; acC_e = [early(c.mdfs, accessf) for c in C]
pv("accessRisingCaptureN", fint(count(acC_l .> acC_e)))
pv("outAcrossCaptureLate", f2(nm([late(c.mdfs, col(:outsourcing_rate)) for c in C])))
pv("outAcrossBaseLate", f2(nm(ob)))
pv("outCaptureBaselineLate", f2(late(BLC.mdfs, col(:outsourcing_rate))))
pv("satCaptureEarly", f2(early(BLC.mdfs, col(:mean_satisfaction_broker))))
pv("satCaptureLate", f2(late(BLC.mdfs, col(:mean_satisfaction_broker))))
pv("satBaseEarly", f2(early(BL.mdfs, col(:mean_satisfaction_broker))))
pv("satBaseLate", f2(late(BL.mdfs, col(:mean_satisfaction_broker))))
qprin = late(BLC.mdfs, col(:q_broker_principal_mean))
surp = late(BLC.mdfs, col(:capture_surplus_mean))
pv("qPrincipalCapBaseline", f2(qprin))
pv("askPaidCapBaseline", f2(qprin - surp))                 # paid ask = realized - surplus
pv("qStandardCapBaseline", f2(late(BLC.mdfs, col(:q_broker_standard_mean))))
pv("agentObsBase", fcomma(nm([late(b.mdfs, agentobs) for (b, _) in pairs])))
pv("agentObsCapture", fcomma(nm([late(c.mdfs, agentobs) for (_, c) in pairs])))
pv("agentObsBaselineBase", fcomma(late(BL.mdfs, agentobs)))
pv("agentObsBaselineCapture", fcomma(late(BLC.mdfs, agentobs)))
pv("prinShareCapture", fint(100 * nm([late(c.mdfs, prinshare) for c in C])))
pv("agentRankCapBaseline", f2(late(BLC.mdfs, col(:agent_holdout_rank))))
pv("agentR2CapBaseline", f2(late(BLC.mdfs, col(:agent_holdout_r2))))
pv("agentRankAcrossBase", f2(nm([late(b.mdfs, col(:agent_holdout_rank)) for (b, _) in pairs])))
pv("agentRankAcrossCapture", f2(nm([late(c.mdfs, col(:agent_holdout_rank)) for (_, c) in pairs])))
pv("agentR2AcrossBase", f2(nm([late(b.mdfs, col(:agent_holdout_r2)) for (b, _) in pairs])))
pv("agentR2AcrossCapture", f2(nm([late(c.mdfs, col(:agent_holdout_r2)) for (_, c) in pairs])))
pv("agentRankEarlyAcrossBase", f2(nm([early(b.mdfs, col(:agent_holdout_rank)) for (b, _) in pairs])))
pv("agentR2EarlyAcrossBase", f2(nm([early(b.mdfs, col(:agent_holdout_r2)) for (b, _) in pairs])))

# ── emit values.tex ──
open(OUTTEX, "w") do io
    println(io, "% values.tex: generated by scripts/paper/stats.jl on ", Dates.format(now(), "yyyy-mm-dd HH:MM"))
    println(io, "% sweep: ", basename(ROOT), "   cells: ", length(B), " base + ", length(C), " capture")
    println(io, "% Do not edit by hand; every value is computed from the saved sweep data.")
    println(io, raw"\newcommand{\pvDefine}[2]{\expandafter\newcommand\csname pv@#1\endcsname{#2}}")
    println(io, raw"\newcommand{\pv}[1]{\csname pv@#1\endcsname}")
    for (k, v) in VALS
        println(io, "\\pvDefine{$k}{$v}")
    end
end
println("\nwrote $(length(VALS)) values to $OUTTEX")
