"""
    scripts/paper/figdata.jl

Cluster-side extract for the figure pipeline. Reads the full sweep (BROKERAGE_ABM_SWEEP_DIR)
    and writes paper/figdata.jld2: the
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
                 grid coordinate
  regime_cells   rho, delta, betw, access, rankgap, qgap per effective realization
  meta           sweep id, generation time, generating script

Usage: BROKERAGE_ABM_SWEEP_DIR=<sweep root> julia --project scripts/paper/figdata.jl
"""

using JLD2, DataFrames, Statistics, Dates

include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))

const ROOT = get(ENV, "BROKERAGE_ABM_SWEEP_DIR") do
    error("set BROKERAGE_ABM_SWEEP_DIR to the sweep root directory")
end
const OUTFILE = normpath(joinpath(@__DIR__, "..", "..", "paper", "figdata.jld2"))
const LATE_WIDTH = 20   # final-period window, the headline statistic
const SWEEP = load_sweep_dataset(ROOT)

nanmean(v) = (w=filter(!isnan, Float64.(collect(v))); isempty(w) ? NaN : mean(w))
late_mask(df) = df.period .>= maximum(df.period) - LATE_WIDTH + 1
tailmean(df, col) = nanmean(df[late_mask(df), col])
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
access_tail(df) = nanmean(accessf(df)[late_mask(df)])
cell_access(mdfs) = nanmean([access_tail(d) for d in mdfs])
load_mdfs(rel) = grid_result(SWEEP, rel).mdfs
load_cfg(rel) = grid_result(SWEEP, rel).cfg
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
fd["oat_cells"] = let out = Dict{String,Float64}[]
    seen = Set{String}()
    for grid in SWEEP.grid_cells
        grid[:kind] == "oat" || continue
        result = SWEEP.result_by_rel[grid[:result_reldir]]
        result.rel in seen && continue
        push!(seen, result.rel)
        m = result.mdfs
        b = seedstat(m, :betweenness)[1];
        a = cell_access(m)
        (isnan(b) || isnan(a)) && continue
        push!(out, Dict("betw" => b, "access" => a, "qgap" => qgap(m)))
    end
    out
end

# ── rho x delta grid cells + OAT rho refiners (figures 2 and 3) ──
fd["grid_cells"] = let out = Dict{String,Any}[]
    for grid in SWEEP.grid_cells
        get(grid, :pair, nothing) == "rho_delta" || continue
        m = SWEEP.result_by_rel[grid[:result_reldir]].mdfs
        cfg = grid[:resolved_params]
        push!(
            out,
            Dict(
                "rho" => Float64(cfg[:rho]),
                "delta" => Float64(cfg[:delta]),
                "outcomes" => outcomes(m),
            ),
        )
    end
    out
end

# ── every effective realization (figure 3) ──
fd["regime_cells"] = let out = Dict{String,Float64}[]
    for result in SWEEP.results
        m, cfg = result.mdfs, result.cfg
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

fd["meta"] = Dict(
    "sweep" => basename(ROOT),
    "manifest_hash" => SWEEP.manifest_hash,
    "schema_version" => SWEEP.schema_version,
    "generated" => string(now()),
    "source" => "scripts/paper/figdata.jl",
)

jldsave(OUTFILE; figdata=fd)
println(
    "wrote $OUTFILE ($(round(filesize(OUTFILE) / 1024; digits=1)) KB; ",
    length(fd["regime_cells"]),
    " effective realizations)",
)
