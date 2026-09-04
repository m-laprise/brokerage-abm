"""
    explore_phase_diagram.jl

Two-parameter sweep producing heatmaps of late-period metrics.
Sweeps one of: rho x s (default), rho x eta, rho x delta, rho x snr.

Data cached as JLD2; pass --rerun to force re-simulation.

Usage:
  julia --project --threads=auto scripts/explore_phase_diagram.jl              # rho x s
  julia --project --threads=auto scripts/explore_phase_diagram.jl rho_eta      # rho x eta
  julia --project --threads=auto scripts/explore_phase_diagram.jl rho_delta    # rho x delta
  julia --project --threads=auto scripts/explore_phase_diagram.jl rho_snr      # rho x 1/sigma_eps
  julia --project --threads=auto scripts/explore_phase_diagram.jl rho_s --rerun
"""

using BrokerageABM
using CairoMakie
using Statistics: mean
using DataFrames: DataFrame
using JLD2

const OUTDIR = joinpath(@__DIR__, "..", "data", "figures", "phase_diagram")
const DATADIR = joinpath(@__DIR__, "..", "data", "sims", "phase_diagram")
const PHASE_DIAGRAM_SCHEMA_VERSION = 3
mkpath(OUTDIR)
mkpath(DATADIR)

# ─────────────────────────────────────────────────────────────────────────────
# Parse CLI
# ─────────────────────────────────────────────────────────────────────────────

RERUN = "--rerun" in ARGS

sweep_type = :rho_s  # default
for a in ARGS
    a == "--rerun" && continue
    if a in ("rho_s", "rho_eta", "rho_delta", "rho_snr")
        global sweep_type = Symbol(a)
    else
        @warn "Unknown argument: $a"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Sweep grids
# ─────────────────────────────────────────────────────────────────────────────

rho_vals = [0.0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0]

if sweep_type == :rho_s
    y_vals = [2, 4, 6, 8]
    y_label = "s (active dimensions)"
    y_key = :s
    y_strings = string.(y_vals)
elseif sweep_type == :rho_eta
    y_vals = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    y_label = "eta (turnover)"
    y_key = :eta
    y_strings = string.(y_vals)
elseif sweep_type == :rho_delta
    y_vals = [0.0, 0.10, 0.25, 0.50, 0.75]
    y_label = "delta (regime gain)"
    y_key = :delta
    y_strings = string.(y_vals)
elseif sweep_type == :rho_snr
    sigma_vals = [0.50, 0.30, 0.20, 0.10, 0.05, 0.025, 0.01]
    y_vals = sigma_vals
    y_label = "sigma_eps (noise)"
    y_key = :sigma_eps
    y_strings = string.(sigma_vals)
else
    error("Unknown sweep type: $sweep_type")
end

# ─────────────────────────────────────────────────────────────────────────────
# Simulation settings
# ─────────────────────────────────────────────────────────────────────────────

T_run = 200
T_burn = 30
N_run = 200
N_SEEDS = 5

n_rho = length(rho_vals)
n_y = length(y_vals)
n_total = n_rho * n_y

datafile = joinpath(DATADIR, "$(sweep_type).jld2")

println("Phase diagram: $sweep_type ($n_total cells, $N_SEEDS seeds, N=$N_run, T=$T_run)")
RERUN && println("  --rerun: forcing re-simulation")

const STEADY_STATE_FIELDS = (
    :r2_gap,
    :broker_r2,
    :agent_r2,
    :outsourcing,
    :betweenness,
    :broker_rank,
    :agent_rank,
    :q_self,
    :q_broker,
)

function nanmean_or_nan(v)
    total = 0.0
    n = 0
    for x in v
        isnan(x) && continue
        total += x
        n += 1
    end
    return n == 0 ? NaN : total / n
end

function maxabs_or_one(M)
    max_abs = 0.0
    found = false
    for x in M
        isnan(x) && continue
        max_abs = max(max_abs, abs(x))
        found = true
    end
    return found ? max_abs : 1.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Run or load
# ─────────────────────────────────────────────────────────────────────────────

function steady_state_metrics(dfs::Vector, T_burn)
    tails = [df[df.period .> T_burn, :] for df in dfs]
    combined = vcat(tails...)
    return (
        r2_gap=nanmean_or_nan(combined.r2_gap),
        broker_r2=nanmean_or_nan(combined.broker_holdout_r2),
        agent_r2=nanmean_or_nan(combined.agent_holdout_r2),
        outsourcing=mean(combined.outsourcing_rate),
        betweenness=mean(combined.betweenness),
        broker_rank=nanmean_or_nan(combined.broker_holdout_rank),
        agent_rank=nanmean_or_nan(combined.agent_holdout_rank),
        q_self=nanmean_or_nan(combined.q_self_mean),
        q_broker=nanmean_or_nan(combined.q_broker_mean),
    )
end

function metric_results_have_fields(results, fields)::Bool
    results isa AbstractMatrix || return false
    size(results) == (n_rho, n_y) || return false
    for r in results
        r === nothing && return false
        all(field -> field in propertynames(r), fields) || return false
    end
    return true
end

function phase_cache_current(saved, results)::Bool
    get(saved, "phase_diagram_schema_version", 0) == PHASE_DIAGRAM_SCHEMA_VERSION ||
        return false
    get(saved, "sweep_type", "") == string(sweep_type) || return false
    return metric_results_have_fields(results, STEADY_STATE_FIELDS)
end

function run_phase_sweep()
    results = Matrix{Any}(nothing, n_rho, n_y)

    cell = 0
    for (j, yv) in enumerate(y_vals), (i, rho) in enumerate(rho_vals)
        cell += 1
        print("  [$cell/$n_total] rho=$rho, $y_key=$yv ... ")

        kw = Dict{Symbol,Any}(:N => N_run, :T => T_run, :rho => rho, y_key => yv)

        dfs = [
            begin
                p = default_params(; seed=s, kw...)
                _, df = run_simulation(p)
                df
            end for s in 1:N_SEEDS
        ]
        results[i, j] = steady_state_metrics(dfs, T_burn)

        r = results[i, j]
        println("gap=$(round(r.r2_gap, digits=3)), out=$(round(r.outsourcing, digits=3))")
    end

    jldsave(
        datafile;
        results=results,
        rho_vals=rho_vals,
        y_vals=y_vals,
        sweep_type=string(sweep_type),
        y_key=string(y_key),
        phase_diagram_schema_version=PHASE_DIAGRAM_SCHEMA_VERSION,
    )
    println("Saved data: $datafile")
    return results
end

if !RERUN && isfile(datafile)
    println("Loading cached data: $datafile")
    saved = JLD2.load(datafile)
    results = get(saved, "results", nothing)
    if !phase_cache_current(saved, results)
        println("Cached data are stale for requested outputs, re-simulating")
        results = run_phase_sweep()
    end
else
    results = run_phase_sweep()
end

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

function extract_metric(results, field::Symbol)
    M = Matrix{Float64}(undef, size(results)...)
    for j in axes(results, 2), i in axes(results, 1)
        r = results[i, j]
        M[i, j] = if r === nothing || !(field in propertynames(r))
            NaN
        else
            getfield(r, field)
        end
    end
    return M
end

function plot_heatmap(M, title_str, filename; colormap=:RdBu, colorrange=nothing, label="")
    fig = Figure(; size=(700, 500))
    cr = colorrange === nothing ? let m = maxabs_or_one(M);
        (-m, m)
    end : colorrange
    ax = Axis(
        fig[1, 1];
        title=title_str,
        xlabel="rho",
        ylabel=y_label,
        xticks=(1:n_rho, string.(rho_vals)),
        yticks=(1:n_y, y_strings),
    )
    hm = heatmap!(ax, 1:n_rho, 1:n_y, M; colormap=colormap, colorrange=cr)
    Colorbar(fig[1, 2], hm; label=label)
    save(joinpath(OUTDIR, filename), fig)
    println("  Saved: $filename")
end

prefix = string(sweep_type)

for (field, title_str, cmap, cr, label) in [
    (:r2_gap, "R2 Gap (Broker - Agent)", :RdBu, nothing, "R2 gap"),
    (:broker_r2, "Broker Holdout R2", :viridis, nothing, "R2"),
    (:agent_r2, "Agent Holdout R2", :viridis, nothing, "R2"),
    (:outsourcing, "Outsourcing Rate", :YlOrRd, (0, 1), "Rate"),
    (:betweenness, "Broker Betweenness", :viridis, nothing, "C_B"),
    (:broker_rank, "Broker Rank Correlation", :viridis, (0, 1), "rho"),
]
    M = extract_metric(results, field)
    plot_heatmap(
        M, title_str, "$(prefix)_$(field).png"; colormap=cmap, colorrange=cr, label=label
    )
end

println("\nFigures: $OUTDIR")
println("Data: $DATADIR")
println("Done.")
