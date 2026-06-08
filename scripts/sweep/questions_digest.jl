"""
    questions_digest.jl

Extract structural and informational metrics for the question-driven report:
broker structural position (betweenness, Burt constraint, effective size), and the
full informational decomposition for broker AND agent --- holdout {rank, rmse,
bias} and selected {rank, rmse, bias} --- so that divergences between measures can
be read off. Windows: steady-state tail (t>30), early (t in [50,70]), late (t in
[181,200]). Reads saved cell aggregates only (no re-simulation).
"""

using JLD2, DataFrames, Statistics, Printf

const ROOT = "/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438"

win(df, col, lo, hi) = begin
    m = (df.period .>= lo) .& (df.period .<= hi)
    v = filter(!isnan, Float64.(collect(skipmissing(df[m, col]))))
    isempty(v) ? NaN : mean(v)
end
tail(df, col; t0=30) = begin
    v = filter(!isnan, Float64.(collect(skipmissing(df[df.period .> t0, col]))))
    isempty(v) ? NaN : mean(v)
end
# seed-average of a per-seed scalar function
sa(mdfs, f) = (vs = filter(!isnan, [f(d) for d in mdfs]); isempty(vs) ? NaN : mean(vs))

g(x) = isnan(x) ? "  NaN " : @sprintf("%.4g", x)

cell_dirs() = begin
    out = String[]
    for ax in sort(readdir(joinpath(ROOT, "oat"))), m in ("base", "capture")
        d = joinpath(ROOT, "oat", ax, m)
        isdir(d) && isfile(joinpath(d, "data.jld2")) && push!(out, d)
    end
    out
end

# only the cells the report cites
want(c) = any(occursin(p, c) for p in (
    "rho=0.0/", "rho=0.5/", "rho=1.0/",
    "N=500/", "N=1500/",
    "k=4/", "k=12/",
    "delta=0.0/", "delta=0.75/",
    "eta=0.01/", "eta=0.03/",
))

function dump_cell(d)
    mdfs = JLD2.load(joinpath(d, "data.jld2"), "mdfs")
    lab = replace(relpath(d, ROOT), "oat/" => "")
    tm(c) = sa(mdfs, x -> tail(x, c))
    println("### ", lab)
    @printf("  STRUCT  betw=%s  constr=%s  effsz=%s  deg=%s\n",
        g(tm(:betweenness)), g(tm(:constraint)), g(tm(:effective_size)), g(tm(:mean_degree)))
    @printf("  HOLDOUT rank b/a=%s/%s  rmse b/a=%s/%s  bias b/a=%s/%s\n",
        g(tm(:broker_holdout_rank)), g(tm(:agent_holdout_rank)),
        g(tm(:broker_holdout_rmse)), g(tm(:agent_holdout_rmse)),
        g(tm(:broker_holdout_bias)), g(tm(:agent_holdout_bias)))
    @printf("  SELECT  rank b/a=%s/%s  rmse b/a=%s/%s  bias b/a=%s/%s\n",
        g(tm(:broker_selected_rank)), g(tm(:agent_selected_rank)),
        g(tm(:broker_selected_rmse)), g(tm(:agent_selected_rmse)),
        g(tm(:broker_selected_bias)), g(tm(:agent_selected_bias)))
    @printf("  R2      hold b/a=%s/%s  sel b/a=%s/%s  outsrc=%s  pshare=%s\n",
        g(tm(:broker_holdout_r2)), g(tm(:agent_holdout_r2)),
        g(tm(:broker_selected_r2)), g(tm(:agent_selected_r2)),
        g(tm(:outsourcing_rate)), g(tm(:principal_mode_share)))
    ew(c) = (sa(mdfs, x -> win(x, c, 50, 70)), sa(mdfs, x -> win(x, c, 181, 200)))
    for c in (:betweenness, :constraint, :effective_size, :broker_holdout_rank,
              :agent_holdout_rank, :broker_holdout_rmse, :agent_holdout_rmse,
              :broker_selected_rank, :mean_degree)
        e, l = ew(c)
        @printf("    e50->late %-22s %s -> %s\n", c, g(e), g(l))
    end
end

for d in cell_dirs()
    want(d) && dump_cell(d)
end
println("DONE")
