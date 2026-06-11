"""
    summary_stats.jl

Compute every statistic needed by the results-summary report (results_report.tex) from
the saved sweep data only — no simulation. Organized by report section; each block
prints the numbers that fill a [MISSING]/[VERIFY]/[Q] placeholder in results_summary.md.

Conventions (match the report):
  late mean (the headline statistic) = mean over t in [181, 200]; no whole-run averages
  early        = mean over t in [50, 70]
  late         = mean over t in [181, 200]
  "across regimes" = unweighted mean over all saved cells of that model
  (each cell first averaged over its seeds)

Usage: julia --project scripts/diagnostics/summary_stats.jl
"""

using JLD2, DataFrames, Statistics

const ROOT = get(ENV, "TB_SWEEP_DIR") do
    error("set TB_SWEEP_DIR to the sweep root directory")
end

nm(v) = (w = filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
r2(x) = round(x; digits=2); r3(x) = round(x; digits=3)
tail(m, c) = win(m, c, 181, 200)   # headline statistic: late-window mean [181,200]
win(m, c, lo, hi) = nm([nm(d[(d.period .>= lo) .& (d.period .<= hi), c]) for d in m])
early(m, c) = win(m, c, 50, 70); late(m, c) = win(m, c, 181, 200)
accfrac(s) = nm([(t = s.access_count[i] + s.assessment_count[i]; t > 0 ? s.access_count[i] / t : NaN) for i in 1:size(s, 1)])
acctail(m) = accwin(m, 181, 200)
accwin(m, lo, hi) = nm([accfrac(d[(d.period .>= lo) .& (d.period .<= hi), :]) for d in m])

# ── enumerate all cells ──
struct Cell; rel::String; model::String; mdfs::Vector{DataFrame}; cfg::Dict; end
cells = Cell[]
for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
    "data.jld2" in files || continue
    model = basename(root); model in ("base", "capture") || continue
    m, c = jldopen(joinpath(root, "data.jld2"), "r") do f; (f["mdfs"], f["config"]) end
    push!(cells, Cell(replace(root, ROOT * "/" => ""), model, m, c))
end
B = [c for c in cells if c.model == "base"]; C = [c for c in cells if c.model == "capture"]
println("cells: $(length(B)) base, $(length(C)) capture")
basecell(rel) = first(c for c in cells if c.rel == rel)
BL  = basecell("oat/rho=0.5/base")      # the baseline regime (no capture, all defaults)
BLC = basecell("oat/rho=0.5/capture")   # capture counterpart

agg(cs, f) = nm([f(c.mdfs) for c in cs])
rng(cs, f) = (v = filter(!isnan, [f(c.mdfs) for c in cs]); (minimum(v), maximum(v)))

sec(t) = println("\n========== $t ==========")

# ───────────────────────── Overview and outsourcing ─────────────────────────
sec("OVERVIEW / OUTSOURCING")
out(m) = tail(m, :outsourcing_rate)
println("baseline outsourcing (late mean): ", r2(out(BL.mdfs)))
println("range across base cells: ", r2.(rng(B, out)), "   capture cells: ", r2.(rng(C, out)))
println("baseline early->late: ", r2(early(BL.mdfs, :outsourcing_rate)), " -> ", r2(late(BL.mdfs, :outsourcing_rate)))
println("base cells early->late (means): ", r2(agg(B, m -> early(m, :outsourcing_rate))), " -> ", r2(agg(B, m -> late(m, :outsourcing_rate))))
println("capture cells early->late (means): ", r2(agg(C, m -> early(m, :outsourcing_rate))), " -> ", r2(agg(C, m -> late(m, :outsourcing_rate))))
println("outsourcing vs eta (base): ", [(e, r2(out(basecell("oat/eta=$e/base").mdfs))) for e in (0.01, 0.02, 0.03)])
println("outsourcing vs reservation (base): ", [(r, r2(out(basecell("oat/reservation_frac=$r/base").mdfs))) for r in (0.4, 0.6, 0.9, 1.2)])
println("outsourcing vs rho (base): ", [(r, r2(out(basecell("oat/rho=$r/base").mdfs))) for r in (0.0, 0.3, 0.5, 0.7, 1.0)])
println("access fraction vs rho (base): ", [(r, r2(acctail(basecell("oat/rho=$r/base").mdfs))) for r in (0.0, 0.3, 0.5, 0.7, 1.0)])
qgap(m) = tail(m, :q_broker_standard_mean) - tail(m, :q_self_mean)
qedge(m) = 100 * qgap(m) / tail(m, :q_self_mean)
println("output gap (q_broker - q_self) mean over base cells: ", r2(agg(B, qgap)), "  range: ", r2.(rng(B, qgap)))
println("output edge % vs eta (base): ", [(e, round(Int, qedge(basecell("oat/eta=$e/base").mdfs))) for e in (0.01, 0.02, 0.03)])
println("output gap by rho (base): ", [(r, r2(qgap(basecell("oat/rho=$r/base").mdfs))) for r in (0.0, 0.3, 0.5, 0.7, 1.0)])

# ───────────────────────── Matching function (rho x delta grid) ─────────────────────────
sec("RHO x DELTA GRID (Fig 1 numbers, base)")
function rd_cells()
    s = jldopen(joinpath(ROOT, "phase/rho_delta/base/summary.jld2"), "r") do f; (xv=f["xvals"], yv=f["yvals"]) end
    out = Tuple{Float64,Float64,Vector{DataFrame}}[]
    for (xi, rho) in enumerate(s.xv), (yi, dl) in enumerate(s.yv)
        rel = "phase/rho_delta/cells/$(xi-1)_$(yi-1)/base"
        any(c -> c.rel == rel, cells) && push!(out, (rho, dl, basecell(rel).mdfs))
    end
    for r in (0.0, 0.3, 0.5, 0.7, 1.0)   # OAT rho cells sit at delta=0.5
        push!(out, (r, 0.5, basecell("oat/rho=$r/base").mdfs))
    end
    out
end
RD = rd_cells()
metrics = [("betweenness", m -> tail(m, :betweenness)), ("access", acctail),
           ("brkR2", m -> tail(m, :broker_holdout_r2)), ("R2gap", m -> tail(m, :broker_holdout_r2) - tail(m, :agent_holdout_r2)),
           ("brkRank", m -> tail(m, :broker_holdout_rank)), ("rankGap", m -> tail(m, :broker_holdout_rank) - tail(m, :agent_holdout_rank)),
           ("qBrk", m -> tail(m, :q_broker_standard_mean)), ("qGap", qgap), ("outsrc", out)]
for (rho, dl, m) in sort(RD; by=x -> (x[2], x[1]))
    print("rho=$(rho) delta=$(dl): "); for (n, f) in metrics; print("$n=", r2(f(m)), " "); end; println()
end

# ───────────────────────── Structural advantage ─────────────────────────
sec("NETWORK TOPOLOGY TRENDS")
println("baseline mean degree early->late: ", r2(early(BL.mdfs, :mean_degree)), " -> ", r2(late(BL.mdfs, :mean_degree)),
        "   betweenness: ", r2(early(BL.mdfs, :betweenness)), " -> ", r2(late(BL.mdfs, :betweenness)))
for (lbl, cs) in (("base", B), ("capture", C))
    println("$lbl cells, early->late means:")
    println("  mean degree:  ", r2(agg(cs, m -> early(m, :mean_degree))), " -> ", r2(agg(cs, m -> late(m, :mean_degree))))
    println("  median degree:", r2(agg(cs, m -> early(m, :median_degree))), " -> ", r2(agg(cs, m -> late(m, :median_degree))))
    println("  betweenness:  ", r2(agg(cs, m -> early(m, :betweenness))), " -> ", r2(agg(cs, m -> late(m, :betweenness))))
end
println("-- co-movement classification per cell (late - early sign): deg+/betw+ etc. --")
for (lbl, cs) in (("base", B), ("capture", C))
    classes = Dict{String,Vector{String}}()
    for c in cs
        dd = late(c.mdfs, :mean_degree) - early(c.mdfs, :mean_degree)
        db = late(c.mdfs, :betweenness) - early(c.mdfs, :betweenness)
        k = (dd >= 0 ? "deg+" : "deg-") * (db >= 0 ? " betw+" : " betw-")
        push!(get!(classes, k, String[]), c.rel)
    end
    println("$lbl: ", Dict(k => length(v) for (k, v) in classes))
    for (k, v) in classes; k != "deg+ betw+" && println("  $k: ", join(v, ", ")); end
end

sec("ACCESS FRACTION")
println("late mean across base cells: ", r2(agg(B, acctail)), "  range: ", r2.(rng(B, acctail)))
println("  (capture cells: ", r2(agg(C, acctail)), ", range ", r2.(rng(C, acctail)), ")")
println("baseline early->late: ", r2(accwin(BL.mdfs, 50, 70)), " -> ", r2(accwin(BL.mdfs, 181, 200)))
println("base cells early->late means: ", r2(agg(B, m -> accwin(m, 50, 70))), " -> ", r2(agg(B, m -> accwin(m, 181, 200))))
println("capture cells early->late means: ", r2(agg(C, m -> accwin(m, 50, 70))), " -> ", r2(agg(C, m -> accwin(m, 181, 200))))
for (lbl, cs) in (("base", B), ("capture", C))
    inc = [c.rel for c in cs if accwin(c.mdfs, 181, 200) > accwin(c.mdfs, 50, 70)]
    println("$lbl cells where access INCREASES over time: $(length(inc))/$(length(cs))")
    foreach(x -> println("   ", x), inc)
end

sec("BETWEENNESS vs ACCESS COUPLING (per sweep family, base)")
pearson(x, y) = (k = findall(i -> !isnan(x[i]) && !isnan(y[i]), eachindex(x)); length(k) < 3 ? NaN : cor(x[k], y[k]))
fams = Dict("rho" => r"(rho=|rho_)", "eta" => r"(eta=|eta_)", "N" => r"(N=|_N)", "r" => r"(reservation|rho_r|eta_r|r_N)",
            "delta" => r"(delta)", "k" => r"k=")
for (fam, pat) in sort(collect(fams); by=first)
    cs = [c for c in B if occursin(pat, c.rel)]
    bw = [tail(c.mdfs, :betweenness) for c in cs]; ac = [acctail(c.mdfs) for c in cs]
    println("$fam cells (n=$(length(cs))): corr(betweenness, access) = ", r2(pearson(bw, ac)))
end
let bw = [tail(c.mdfs, :betweenness) for c in B], ac = [acctail(c.mdfs) for c in B]
    println("ALL base cells: corr = ", r2(pearson(bw, ac)))
end

# ───────────────────────── Structure vs information ─────────────────────────
sec("ADVANTAGE CORRELATIONS (all base cells; same set as Figure 3)")
let
    bw = Float64[]; ac = Float64[]; rg = Float64[]; qg = Float64[]; br = Float64[]
    for c in B
        push!(bw, tail(c.mdfs, :betweenness)); push!(ac, acctail(c.mdfs))
        push!(rg, tail(c.mdfs, :broker_holdout_rank) - tail(c.mdfs, :agent_holdout_rank))
        push!(br, tail(c.mdfs, :broker_holdout_rank)); push!(qg, qgap(c.mdfs))
    end
    println("corr(betweenness, rank gap) = ", r2(pearson(bw, rg)), "   corr(betweenness, output gap) = ", r2(pearson(bw, qg)))
    println("corr(access, rank gap)      = ", r2(pearson(ac, rg)), "   corr(access, output gap)      = ", r2(pearson(ac, qg)))
    println("corr(broker rank, output gap) = ", r2(pearson(br, qg)), "   corr(rank gap, output gap)  = ", r2(pearson(rg, qg)))
end

# ───────────────────────── Capture ─────────────────────────
sec("CAPTURE")
println("captured share (of outsourced) vs rho: ", [(r, r2(tail(basecell("oat/rho=$r/capture").mdfs, :principal_mode_share))) for r in (0.0, 0.3, 0.5, 0.7, 1.0)])
println("captured share vs reservation: ", [(r, r2(tail(basecell("oat/reservation_frac=$r/capture").mdfs, :principal_mode_share))) for r in (0.4, 0.6, 0.9, 1.2)])
println("captured share vs eta: ", [(e, r2(tail(basecell("oat/eta=$e/capture").mdfs, :principal_mode_share))) for e in (0.01, 0.02, 0.03)])
println("captured share vs delta: ", [(d, r2(tail(basecell("oat/delta=$d/capture").mdfs, :principal_mode_share))) for d in (0.0, 0.75)])
println("-- baseline base vs capture --")
for (lbl, c, f) in (("matches/agent", :n_total_matches, m -> 2 * tail(m, :n_total_matches) / 1000),
                    ("mean degree", :mean_degree, m -> tail(m, :mean_degree)),
                    ("median degree", :median_degree, m -> tail(m, :median_degree)),
                    ("betweenness", :betweenness, m -> tail(m, :betweenness)),
                    ("outsourcing", :outsourcing_rate, m -> tail(m, :outsourcing_rate)),
                    ("agent holdout rank", :agent_holdout_rank, m -> tail(m, :agent_holdout_rank)),
                    ("agent holdout R2", :agent_holdout_r2, m -> tail(m, :agent_holdout_r2)),
                    ("broker holdout rank", :broker_holdout_rank, m -> tail(m, :broker_holdout_rank)))
    println("  $lbl: base=", r2(f(BL.mdfs)), "  capture=", r2(f(BLC.mdfs)))
end
println("  access fraction: base=", r2(acctail(BL.mdfs)), "  capture=", r2(acctail(BLC.mdfs)))
println("  baseline outsourcing early->late, base: ", r2(early(BL.mdfs, :outsourcing_rate)), "->", r2(late(BL.mdfs, :outsourcing_rate)),
        "   capture: ", r2(early(BLC.mdfs, :outsourcing_rate)), "->", r2(late(BLC.mdfs, :outsourcing_rate)))
println("  baseline matches/agent early->late, base: ", r2(2 * early(BL.mdfs, :n_total_matches) / 1000), "->", r2(2 * late(BL.mdfs, :n_total_matches) / 1000),
        "   capture: ", r2(2 * early(BLC.mdfs, :n_total_matches) / 1000), "->", r2(2 * late(BLC.mdfs, :n_total_matches) / 1000))
println("  baseline mean degree early->late, base: ", r2(early(BL.mdfs, :mean_degree)), "->", r2(late(BL.mdfs, :mean_degree)),
        "   capture: ", r2(early(BLC.mdfs, :mean_degree)), "->", r2(late(BLC.mdfs, :mean_degree)))
println("-- agent learning under capture (paired cells) --")
let pairs = [(c, basecell(replace(c.rel, "/capture" => "/base"))) for c in C if any(b -> b.rel == replace(c.rel, "/capture" => "/base"), cells)]
    dr = [tail(cc.mdfs, :agent_holdout_rank) - tail(bb.mdfs, :agent_holdout_rank) for (cc, bb) in pairs]
    d2 = [tail(cc.mdfs, :agent_holdout_r2) - tail(bb.mdfs, :agent_holdout_r2) for (cc, bb) in pairs]
    db = [tail(cc.mdfs, :broker_holdout_rank) - tail(bb.mdfs, :broker_holdout_rank) for (cc, bb) in pairs]
    println("  n pairs=", length(pairs), "  mean Δ agent rank (capture - base) = ", r3(nm(dr)),
            "  mean Δ agent R2 = ", r2(nm(d2)), "  mean Δ broker rank = ", r3(nm(db)))
end
println("\nDONE")
