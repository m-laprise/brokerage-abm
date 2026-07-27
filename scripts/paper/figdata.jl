"""
    scripts/paper/figdata.jl

Cluster-side extract for the figure pipeline. Reads the full sweep (BROKERAGE_ABM_SWEEP_DIR)
plus the initialization-only DGP rank grid and writes paper/figdata.jld2: the
small derived dataset from which scripts/paper/figures.jl renders every figure,
locally, with no access to the sweep. No hard-coded results: every stored value
is computed from the saved sweep data at run time.

Rerun only when the underlying numbers change (a new sweep, or a figure needing a
metric not yet extracted); styling iteration needs only figures.jl.

Contents of figdata.jld2 (single key "figdata", a Dict):
  period         per-period time axis of the baseline runs
  series         per-period ensemble means at the baseline:
                 betweenness, access, mean_degree, median_degree, mpa
                 (matches per agent, both sides), outsourcing
  oat_cells      late means per one-at-a-time regime: betw, access, qgap
  grid_cells     rho, delta, and the nine named outcome late means per rho x delta
                 grid cell (plus the OAT rho = 0.3, 0.7 refiners)
  regime_cells   rho, delta, betw, access, rankgap, qgap per regime
  r90            effective-rank grid (rho, delta) => r90
  meta           sweep id, generation time, generating script

Usage: BROKERAGE_ABM_SWEEP_DIR=<sweep root> julia --project scripts/paper/figdata.jl
"""

using JLD2, DataFrames, Statistics, Dates

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUTFILE = normpath(joinpath(@__DIR__, "..", "..", "paper", "figdata.jld2"))
const LATE = (181, 200)   # late-window mean, the headline statistic

nanmean(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
tailmean(df, col) = nanmean(df[(df.period .>= LATE[1]) .& (df.period .<= LATE[2]), col])
function seedstat(mdfs, col)
    (
        vs=filter(!isnan, [tailmean(d, col) for d in mdfs]);
        isempty(vs) ? (NaN, NaN) : (mean(vs), std(vs))
    )
end
function accessf(df)
    (
        t=(df.access_count .+ df.assessment_count);
        [t[i] > 0 ? df.access_count[i] / t[i] : NaN for i in eachindex(t)]
    )
end
access_tail(df) = nanmean(accessf(df)[(df.period .>= LATE[1]) .& (df.period .<= LATE[2])])
cell_access(mdfs) = nanmean([access_tail(d) for d in mdfs])
load_mdfs(rel) =
    jldopen(joinpath(ROOT, rel, "data.jld2"), "r") do f
        ;
        f["mdfs"]
    end
load_cfg(rel) =
    jldopen(joinpath(ROOT, rel, "data.jld2"), "r") do f
        ;
        f["config"]
    end
qgap(m) = seedstat(m, :q_broker_mean)[1] - seedstat(m, :q_self_mean)[1]
rankgap(m) = seedstat(m, :broker_holdout_rank)[1] - seedstat(m, :agent_holdout_rank)[1]
function ens(mdfs, f)
    (per=mdfs[1].period; [nanmean(Float64[f(d)[t] for d in mdfs]) for t in eachindex(per)])
end

# the nine outcomes shared by figures 2 and 3; names must match figures.jl panels
function outcomes(m)
    Dict{String,Float64}(
        "Betweenness centrality" => seedstat(m, :betweenness)[1],
        "Access fraction" => cell_access(m),
        "Broker prediction R²" => seedstat(m, :broker_holdout_r2)[1],
        "Prediction R² gap" =>
            seedstat(m, :broker_holdout_r2)[1] - seedstat(m, :agent_holdout_r2)[1],
        "Broker rank correlation" => seedstat(m, :broker_holdout_rank)[1],
        "Rank correlation gap" => rankgap(m),
        "Broker output q" => seedstat(m, :q_broker_mean)[1],
        "Output gap q" => qgap(m),
        "Outsourcing rate" => seedstat(m, :outsourcing_rate)[1],
    )
end

fd = Dict{String,Any}()

# ── baseline per-period ensemble series ──
baseline_rel = "oat/rho=0.5"
baseline = load_mdfs(baseline_rel)
N = load_cfg(baseline_rel)["N"]
fd["period"] = collect(baseline[1].period)
function series(m, N)
    Dict{String,Vector{Float64}}(
        "betweenness" => ens(m, d -> d.betweenness),
        "access" => ens(m, accessf),
        "mean_degree" => ens(m, d -> d.mean_degree),
        "median_degree" => ens(m, d -> d.median_degree),
        "mpa" => ens(m, d -> 2 .* d.n_total_matches ./ N),
        "outsourcing" => ens(m, d -> d.outsourcing_rate),
    )
end
fd["series"] = series(baseline, N)

# ── one-at-a-time cells (figure 1 scatter) ──
cellspec = vcat(
    ["oat/rho=$(r)" for r in (0.0, 0.3, 0.5, 0.7, 1.0)],
    [
        "oat/$a" for a in (
            "eta=0.01",
            "eta=0.02",
            "eta=0.03",
            "N=500",
            "N=1000",
            "N=1500",
            "reservation_frac=0.4",
            "reservation_frac=0.6",
            "reservation_frac=0.9",
            "reservation_frac=1.2",
            "delta=0.0",
            "delta=0.75",
            "k=4",
            "k=12",
        )
    ],
)
fd["oat_cells"] = let out = Dict{String,Float64}[]
    for rel in cellspec
        isfile(joinpath(ROOT, rel, "data.jld2")) || continue
        m = load_mdfs(rel)
        b = seedstat(m, :betweenness)[1];
        a = cell_access(m)
        (isnan(b) || isnan(a)) && continue
        push!(out, Dict("betw" => b, "access" => a, "qgap" => qgap(m)))
    end
    out
end

# ── rho x delta grid cells + OAT rho refiners (figures 2 and 3) ──
fd["grid_cells"] = let out = Dict{String,Any}[]
    cellsdir = joinpath(ROOT, "phase", "rho_delta", "cells")
    for d in sort(readdir(cellsdir))
        p = joinpath(cellsdir, d, "data.jld2")
        isfile(p) || continue
        m, cfg = jldopen(p, "r") do f
            ;
            (f["mdfs"], f["config"])
        end
        push!(
            out,
            Dict(
                "rho" => Float64(cfg["rho"]),
                "delta" => Float64(cfg["delta"]),
                "outcomes" => outcomes(m),
            ),
        )
    end
    for r in (0.3, 0.7)   # OAT rho cells refine the delta = 0.5 line
        push!(
            out,
            Dict(
                "rho" => r,
                "delta" => 0.5,
                "outcomes" => outcomes(load_mdfs("oat/rho=$r")),
            ),
        )
    end
    out
end

# ── every regime (figure 3) ──
fd["regime_cells"] = let out = Dict{String,Float64}[]
    for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
        "data.jld2" in files || continue
        m, cfg = jldopen(joinpath(root, "data.jld2"), "r") do f
            ;
            (f["mdfs"], f["config"])
        end
        b = seedstat(m, :betweenness)[1];
        isnan(b) && continue
        push!(
            out,
            Dict(
                "rho" => Float64(cfg["rho"]),
                "delta" => Float64(cfg["delta"]),
                "betw" => b,
                "access" => cell_access(m),
                "rankgap" => rankgap(m),
                "qgap" => qgap(m),
            ),
        )
    end
    out
end

# ── effective-rank grid (axis/size/color scales) ──
fd["r90"] = JLD2.load(
    joinpath(@__DIR__, "..", "diagnostics", "_results", "dgp_rank_grid.jld2")
)["r90"]

fd["meta"] = Dict(
    "sweep" => basename(ROOT),
    "generated" => string(now()),
    "source" => "scripts/paper/figdata.jl",
)

jldsave(OUTFILE; figdata=fd)
println(
    "wrote $OUTFILE ($(round(filesize(OUTFILE) / 1024; digits=1)) KB; ",
    length(fd["regime_cells"]),
    " regime cells)",
)
