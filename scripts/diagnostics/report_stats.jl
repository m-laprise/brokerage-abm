"""
    report_stats.jl

Recompute every derived statistic quoted in questions_report.tex from committed data,
so the report's numbers are reproducible from code rather than ad-hoc shell sessions.
Reads the parameter-sweep data (Exp. A) and the data-binding re-run (Exp. D). Prints by
report section. No simulation is run here.

Usage: julia --project scripts/diagnostics/report_stats.jl
"""

using JLD2, DataFrames, Statistics
const ROOT = "/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438"

nm(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
load(rel) = JLD2.load(joinpath(ROOT, rel, "data.jld2"), "mdfs")
r3(x) = round(x; digits=3); r2(x) = round(x; digits=2)
tail(m, c; t0=30) = mean([nm(d[d.period .> t0, c]) for d in m])
win(m, c, lo, hi) = mean([nm(d[(d.period .>= lo) .& (d.period .<= hi), c]) for d in m])
accfrac(s) = nm([(t = s.access_count[i] + s.assessment_count[i]; t > 0 ? s.access_count[i] / t : NaN) for i in 1:size(s, 1)])
acc(m; t0=30) = mean([accfrac(d[d.period .> t0, :]) for d in m])
function overallq(m; t0=30)   # count-weighted realized output over both channels
    vals = Float64[]
    for d in m
        s = d[d.period .> t0, :]; tot = 0.0; cnt = 0.0
        for i in 1:size(s, 1)
            ns = Float64(s.n_self_matches[i]); nb = Float64(s.n_broker_standard[i])
            ns > 0 && !isnan(s.q_self_mean[i]) && (tot += ns * s.q_self_mean[i])
            nb > 0 && !isnan(s.q_broker_standard_mean[i]) && (tot += nb * s.q_broker_standard_mean[i])
            cnt += ns + nb
        end
        cnt > 0 && push!(vals, tot / cnt)
    end
    mean(vals)
end
gap(m, b, a) = tail(m, b) - tail(m, a)
RHO5 = [0.0, 0.3, 0.5, 0.7, 1.0]; DELTAS = [0.0, 0.5, 0.75]

sec(s) = (println("\n", "="^72); println(s); println("-"^72))

sec("§1 Market overview  [A]")
println("outsourcing rate, tail, across rho: ", [r2(tail(load("oat/rho=$(r)/base"), :outsourcing_rate)) for r in RHO5])
println("market-wide realized q (count-weighted), across rho: ", [r2(overallq(load("oat/rho=$(r)/base"))) for r in RHO5])

sec("§2 DGP: rank-direction disentanglement on the rho x delta grid  [A]; rank lever from [D]")
cell(xi, yi) = load("phase/rho_delta/cells/$(xi)_$(yi)/base")
for (nm_, f) in (("betweenness", m -> tail(m, :betweenness)),
                 ("broker holdout R2", m -> tail(m, :broker_holdout_r2)),
                 ("holdout rank gap", m -> gap(m, :broker_holdout_rank, :agent_holdout_rank)),
                 ("realized gap q", m -> gap(m, :q_broker_standard_mean, :q_self_mean)),
                 ("access fraction", m -> acc(m)))
    println(nm_, "  (rows delta=0/0.5/0.75, cols rho=0/0.5/1):")
    for yi in 0:2
        println("   delta=$(DELTAS[yi+1])  ", [r2(f(cell(xi, yi))) for xi in 0:2])
    end
end

sec("§3 Structural advantage  [A]")
for r in (0.0, 0.5, 1.0)
    m = load("oat/rho=$(r)/base")
    println("rho=$r  betweenness=", r2(tail(m, :betweenness)), " constraint=", r3(tail(m, :constraint)),
            " eff_size=", round(Int, tail(m, :effective_size)), " rankgap=", r2(gap(m, :broker_holdout_rank, :agent_holdout_rank)),
            " degree=", r2(tail(m, :mean_degree)))
end
for r in (0.0, 0.5)
    m = load("oat/rho=$(r)/base")
    println("rho=$r betweenness windows [50,70]/[181,200]: ", r2(win(m, :betweenness, 50, 70)), " / ", r2(win(m, :betweenness, 181, 200)))
end

sec("§4 Access vs assessment  [A]")
println("access fraction across rho: ", [r2(acc(load("oat/rho=$(r)/base"))) for r in RHO5])
for r in (0.0, 0.5)
    m = load("oat/rho=$(r)/base")
    af(lo, hi) = mean([accfrac(d[(d.period .>= lo) .& (d.period .<= hi), :]) for d in m])
    println("rho=$r access windows [1,10]/[50,70]/[181,200]: ", r2(af(1, 10)), " / ", r2(af(50, 70)), " / ", r2(af(181, 200)))
end
let bs = Float64[], as_ = Float64[]
    for ax in ("rho=0.0","rho=0.3","rho=0.5","rho=0.7","rho=1.0","eta=0.01","eta=0.02","eta=0.03","N=500","N=1000","N=1500",
               "reservation_frac=0.4","reservation_frac=0.6","reservation_frac=0.9","reservation_frac=1.2","delta=0.0","delta=0.75","k=4","k=12")
        isfile(joinpath(ROOT, "oat/$ax/base", "data.jld2")) || continue
        m = load("oat/$ax/base"); push!(bs, tail(m, :betweenness)); push!(as_, acc(m))
    end
    println("cross-cell cor(betweenness, access) over $(length(bs)) base OAT cells = ", r2(cor(bs, as_)))
end

sec("§5 Informational: holdout broker/agent across rho  [A]")
for r in RHO5
    m = load("oat/rho=$(r)/base")
    println("rho=$r  rank b/a=", r2(tail(m, :broker_holdout_rank)), "/", r2(tail(m, :agent_holdout_rank)),
            "  R2 b/a=", r2(tail(m, :broker_holdout_r2)), "/", r2(tail(m, :agent_holdout_r2)),
            "  RMSE b/a=", r2(tail(m, :broker_holdout_rmse)), "/", r2(tail(m, :agent_holdout_rmse)),
            "  bias b/a=", r2(tail(m, :broker_holdout_bias)), "/", r2(tail(m, :agent_holdout_bias)))
end
println("selected rank b/a (baseline rho=0.5): ", r2(tail(load("oat/rho=0.5/base"), :broker_selected_rank)), "/", r2(tail(load("oat/rho=0.5/base"), :agent_selected_rank)))
println("\nholdout learning curves over t (broker rank / R2):")
for (lab, rel) in (("rho=0.5", "oat/rho=0.5/base"), ("rho=0", "oat/rho=0.0/base"), ("rho=1", "oat/rho=1.0/base"))
    m = load(rel); at(t, c) = mean([d[d.period .== t, c][1] for d in m])
    println("  $lab : ", [(t, r2(at(t, :broker_holdout_rank)), r2(at(t, :broker_holdout_r2))) for t in (5, 30, 100, 200)])
end
println("agent learning curve (rank/R2) rho=0.5: ",
        let m = load("oat/rho=0.5/base"); at(t, c) = mean([d[d.period .== t, c][1] for d in m]);
            [(t, r2(at(t, :agent_holdout_rank)), r2(at(t, :agent_holdout_r2))) for t in (5, 30, 100, 200)] end)
println("holdout vs N (broker rank/R2, agent rank/R2):")
for n in (500, 1000, 1500)
    m = load("oat/N=$(n)/base")
    println("  N=$n  brk ", r2(tail(m, :broker_holdout_rank)), "/", r2(tail(m, :broker_holdout_r2)),
            "  agt ", r2(tail(m, :agent_holdout_rank)), "/", r2(tail(m, :agent_holdout_r2)),
            "  broker_history=", round(Int, tail(m, :broker_history_size)))
end

sec("§5 Generalization / data-binding  [D]")
if isfile(joinpath(@__DIR__, "_results", "agent_data_binding.jld2"))
    f = JLD2.load(joinpath(@__DIR__, "_results", "agent_data_binding.jld2"))
    d = Float64.(f["data"]); rk = f["rank"]; rr = f["r2"]; o = sortperm(d); n = length(d)
    for i in 1:5
        idx = o[(round(Int, (i - 1) * n / 5) + 1):round(Int, i * n / 5)]
        println("  agent Q$i data=", round(Int, mean(d[idx])), " rank=", r2(mean(rk[idx])), " R2=", r2(mean(rr[idx])))
    end
    println("  broker ref: rank=", r2(f["brk_rank"]), " R2=", r2(f["brk_r2"]), " (agent width $(f["agent_width"]), broker width $(f["broker_width"]))")
else
    println("  (run agent_data_binding.jl first)")
end

sec("§6 Realized output by channel  [A]")
for r in (0.0, 0.5, 1.0)
    m = load("oat/rho=$(r)/base")
    qs = tail(m, :q_self_mean); qb = tail(m, :q_broker_standard_mean)
    println("rho=$r  q_self=", r2(qs), " q_broker=", r2(qb), " edge=", round(Int, 100 * (qb - qs) / qs), "%")
end

sec("§7 Performance gaps across regimes  [A]")
for (axn, cells) in (("rho", ["oat/rho=$(r)/base" for r in (0.0,0.5,1.0)]),
                     ("N", ["oat/N=$(n)/base" for n in (500,1000,1500)]),
                     ("eta", ["oat/eta=$(e)/base" for e in (0.01,0.02,0.03)]),
                     ("k", ["oat/k=4/base","oat/rho=0.5/base","oat/k=12/base"]))
    rg = [r2(gap(load(c), :broker_holdout_rank, :agent_holdout_rank)) for c in cells]
    qg = [r2(gap(load(c), :q_broker_standard_mean, :q_self_mean)) for c in cells]
    println("$axn  rank gap=", rg, "  realized gap=", qg)
end

sec("§8-9 Capture  [A]")
let m = load("oat/reservation_frac=0.4/capture"), m2 = load("oat/reservation_frac=1.2/capture")
    println("captured-demand share, lambda_r=0.4 / 1.2: ", r2(tail(m, :principal_mode_share)), " / ", r2(tail(m2, :principal_mode_share)))
end
let b = load("oat/rho=0.5/base"), c = load("oat/rho=0.5/capture")
    println("mean degree base/capture (rho=0.5): ", r2(tail(b, :mean_degree)), " / ", r2(tail(c, :mean_degree)))
    println("selected rank base/capture (rho=0.5): ", r2(tail(b, :broker_selected_rank)), " / ", r2(tail(c, :broker_selected_rank)))
end
let b = load("oat/rho=1.0/base"), c = load("oat/rho=1.0/capture")
    println("betweenness base/capture (rho=1): ", r2(tail(b, :betweenness)), " / ", r2(tail(c, :betweenness)))
end
println("\nDONE")
