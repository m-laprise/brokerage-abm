"""
    scripts/paper/supp_figdata.jl

Cluster-side extract for the SUPPLEMENTARY figure pipeline. Standalone twin of
scripts/paper/figdata.jl: it reads the full sweep (BROKERAGE_ABM_SWEEP_DIR) and writes
paper/supp_figdata.jld2, the small derived dataset from which
scripts/paper/supp_figures.jl renders the supplement figures (S1-S4) locally,
with no access to the sweep. This script shares no state with the results-section
pipeline, so the two can be regenerated independently of each other.

The main results use broker BETWEENNESS centrality as the structural-advantage
measure. The supplement reproduces the same analyses with the broker's two other
saved ego-network measures: Burt's aggregate CONSTRAINT and Burt's EFFECTIVE
SIZE (src/measures.jl). Both are recomputed for the broker node every
network_measure_interval (20) periods, exactly like betweenness, so the
per-period series carry a fresh value only on multiples of that interval (held
constant in between); the renderer selects those measurement periods.

No hard-coded results: every stored value is computed from the saved sweep data
at run time. Literal constants are selection/window conventions only (the late
window, the OAT regime list), kept identical to figdata.jl.

Contents of supp_figdata.jld2 (single key "figdata", a Dict):
  period      per-period time axis of the baseline runs
  series      per-period ensemble means at the baseline: constraint and
              effective_size (full per-period vectors; the renderer keeps the
              measurement periods)
  oat_cells   late means per one-at-a-time regime: access,
              constraint, effsize (for the cross-regime scatter, S2 right)
  grid_cells  rho, delta, and the constraint/effsize late means per rho x delta
              grid cell (plus the OAT rho refiners), for S1
  regime_cells rho, constraint, effsize, rankgap, qgap per regime, for S3
  meta        sweep id, generation time, generating script

Usage: BROKERAGE_ABM_SWEEP_DIR=<sweep root> julia --project scripts/paper/supp_figdata.jl
"""

using JLD2, DataFrames, Statistics, Dates

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUTFILE = normpath(joinpath(@__DIR__, "..", "..", "paper", "supp_figdata.jld2"))
const LATE = (181, 200)   # late-window mean, the headline statistic (matches figdata.jl)

# ── shared reducers (kept local so this extract does not depend on figdata.jl) ──
nanmean(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
tailmean(df, col) = nanmean(df[(df.period .>= LATE[1]) .& (df.period .<= LATE[2]), col])
seedmean(mdfs, col) = nanmean([tailmean(d, col) for d in mdfs])
function accessf(df)
    (
        t=(df.access_count .+ df.assessment_count);
        [t[i] > 0 ? df.access_count[i] / t[i] : NaN for i in eachindex(t)]
    )
end
access_tail(df) = nanmean(accessf(df)[(df.period .>= LATE[1]) .& (df.period .<= LATE[2])])
cell_access(mdfs) = nanmean([access_tail(d) for d in mdfs])
# broker-minus-self gaps, the y-axes of S3 (identical definitions to figdata.jl)
qgap(m) = seedmean(m, :q_broker_mean) - seedmean(m, :q_self_mean)
rankgap(m) = seedmean(m, :broker_holdout_rank) - seedmean(m, :agent_holdout_rank)
# per-period ensemble mean of f(df) over seeds
function ens(mdfs, f)
    (per=mdfs[1].period; [nanmean(Float64[f(d)[t] for d in mdfs]) for t in eachindex(per)])
end
load_mdfs(rel) =
    jldopen(joinpath(ROOT, rel, "data.jld2"), "r") do f
        ;
        f["mdfs"]
    end

fd = Dict{String,Any}()

# ── baseline per-period ensemble series (S2 left and S4) ──
baseline = load_mdfs("oat/rho=0.5")
fd["period"] = collect(baseline[1].period)
function series(m)
    Dict{String,Vector{Float64}}(
        "constraint" => ens(m, d -> d.constraint),
        "effective_size" => ens(m, d -> d.effective_size),
    )
end
fd["series"] = series(baseline)

# ── one-at-a-time cells (S2 right scatter) ──
# same regime list as figdata.jl's oat_cells, so the supplement scatter spans the
# same regimes as the main-text Figure 2.
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
        a = cell_access(m);
        c = seedmean(m, :constraint);
        e = seedmean(m, :effective_size)
        (isnan(a) || isnan(c) || isnan(e)) && continue
        push!(out, Dict("access" => a, "constraint" => c, "effsize" => e))
    end
    out
end

# ── rho x delta grid cells + OAT rho refiners (S1) ──
# identical cell set to figdata.jl's grid_cells, so the supplement S1 lines span
# the same grid as the main-text Figure 1.
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
                "constraint" => seedmean(m, :constraint),
                "effsize" => seedmean(m, :effective_size),
            ),
        )
    end
    for r in (0.3, 0.7)   # OAT rho cells refine the delta = 0.5 line
        m = load_mdfs("oat/rho=$r")
        push!(
            out,
            Dict(
                "rho" => r,
                "delta" => 0.5,
                "constraint" => seedmean(m, :constraint),
                "effsize" => seedmean(m, :effective_size),
            ),
        )
    end
    out
end

# ── every regime (S3) ──
fd["regime_cells"] = let out = Dict{String,Float64}[]
    for sub in ("oat", "phase"), (root, _, files) in walkdir(joinpath(ROOT, sub))
        "data.jld2" in files || continue
        m, cfg = jldopen(joinpath(root, "data.jld2"), "r") do f
            ;
            (f["mdfs"], f["config"])
        end
        c = seedmean(m, :constraint);
        isnan(c) && continue
        push!(
            out,
            Dict(
                "rho" => Float64(cfg["rho"]),
                "delta" => Float64(cfg["delta"]),
                "constraint" => c,
                "effsize" => seedmean(m, :effective_size),
                "rankgap" => rankgap(m),
                "qgap" => qgap(m),
            ),
        )
    end
    out
end

fd["meta"] = Dict(
    "sweep" => basename(ROOT),
    "generated" => string(now()),
    "source" => "scripts/paper/supp_figdata.jl",
)

jldsave(OUTFILE; figdata=fd)
println(
    "wrote $OUTFILE ($(round(filesize(OUTFILE) / 1024; digits=1)) KB; ",
    length(fd["regime_cells"]),
    " regime cells, ",
    length(fd["oat_cells"]),
    " OAT cells, ",
    length(fd["grid_cells"]),
    " grid cells)",
)
