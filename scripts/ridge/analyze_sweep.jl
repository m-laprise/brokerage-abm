"""
    scripts/ridge/analyze_sweep.jl

Compare the paired-Ridge reporting sweep with the canonical NN reporting sweep.
The analysis gives each effective model realization one observation. Comparisons
use the planned seeds common to both sweeps: 50 at baseline and 20 elsewhere.

Required environment:

  BROKERAGE_ABM_NN_SWEEP_DIR
  BROKERAGE_ABM_RIDGE_SWEEP_DIR

Outputs are written to `output/ridge/paired/results/`.
"""

using CairoMakie
using DataFrames: DataFrame
using Dates: now
using JLD2: jldopen
using Printf: @sprintf
using Statistics: cor, mean, median, quantile, std

const NN_ROOT = get(ENV, "BROKERAGE_ABM_NN_SWEEP_DIR") do
    error("BROKERAGE_ABM_NN_SWEEP_DIR is required")
end
const RIDGE_ROOT = get(ENV, "BROKERAGE_ABM_RIDGE_SWEEP_DIR") do
    error("BROKERAGE_ABM_RIDGE_SWEEP_DIR is required")
end
const OUT_DIR =
    normpath(joinpath(@__DIR__, "..", "..", "output", "ridge", "paired", "results"))
const BASELINE_REL = "oat/rho=0.5"
const LATE_WIDTH = 20
const DESIGN_KEYS = (
    "N",
    "T",
    "T_burn",
    "rho",
    "eta",
    "delta",
    "k",
    "roster_frac",
    "n_strangers",
    "s",
    "reservation_frac",
)

struct ComparisonResult
    rel::String
    mdfs::Vector{DataFrame}
    config::Dict{String,Any}
    seeds::Vector{Int}
    condition_index::Int
end

struct ComparisonSweep
    results::Dict{String,ComparisonResult}
    grid_cells::Vector{Dict{Symbol,Any}}
    meta::Dict{Symbol,Any}
    manifest_hash::String
    schema_version::Int
    git_commit::String
end

string_dict(values) = Dict{String,Any}(string(key) => value for (key, value) in values)

function load_comparison_sweep(root::AbstractString)::ComparisonSweep
    manifest_path = joinpath(root, "manifest.jld2")
    isfile(manifest_path) || error("missing manifest: $manifest_path")
    grid_cells, conditions, meta, manifest_hash, schema_version =
        jldopen(manifest_path, "r") do file
            (
                file["cells"],
                file["conditions"],
                file["meta"],
                String(file["manifest_hash"]),
                Int(file["schema_version"]),
            )
        end
    length(conditions) == meta[:n_conditions] || error("condition count mismatch")
    length(grid_cells) == meta[:n_grid_cells] || error("grid-cell count mismatch")

    expected_periods = collect(1:Int(meta[:T]))
    results = Dict{String,ComparisonResult}()
    commits = Set{String}()
    for condition in conditions
        rel = String(condition[:result_reldir])
        path = joinpath(root, rel, "data.jld2")
        isfile(path) || error("missing aggregate: $path")
        result, provenance = jldopen(path, "r") do file
            provenance = file["provenance"]
            get(provenance, "manifest_hash", nothing) == manifest_hash ||
                error("manifest provenance mismatch: $path")
            get(provenance, "schema_version", nothing) == schema_version ||
                error("schema provenance mismatch: $path")
            file["result_reldir"] == rel || error("result path mismatch: $path")
            (
                ComparisonResult(
                    rel,
                    file["mdfs"],
                    string_dict(file["realized_config"]),
                    Int.(file["seeds"]),
                    Int(file["condition_index"]),
                ),
                provenance,
            )
        end

        # Schema 5 used one common seed set stored in manifest metadata. Schema 7
        # stores a seed set on each condition so the baseline may have more seeds.
        expected_seeds = if haskey(condition, :seeds)
            Int.(condition[:seeds])
        else
            Int.(meta[:seeds])
        end
        result.seeds == expected_seeds || error("seed mismatch: $rel")
        length(result.mdfs) == length(result.seeds) || error("metrics/seed mismatch: $rel")
        all(df -> collect(df.period) == expected_periods, result.mdfs) ||
            error("period coverage mismatch: $rel")
        for (key, value) in condition[:resolved_params]
            result.config[string(key)] == value || error("resolved config mismatch: $rel")
        end
        result.condition_index == condition[:condition_index] ||
            error("condition-index mismatch: $rel")
        haskey(results, rel) && error("duplicate effective realization: $rel")
        results[rel] = result
        push!(commits, String(provenance["git_commit"]))
    end
    length(commits) == 1 || error("mixed source commits in $root: $commits")
    return ComparisonSweep(
        results, grid_cells, meta, manifest_hash, schema_version, only(commits)
    )
end

nanmean(values) = begin
    kept = filter(!isnan, Float64.(collect(values)))
    isempty(kept) ? NaN : mean(kept)
end

function metric_series(df::DataFrame, metric::Symbol)
    if metric == :rank_gap
        return df.broker_holdout_rank .- df.agent_holdout_rank
    elseif metric == :r2_gap
        return df.broker_holdout_r2 .- df.agent_holdout_r2
    elseif metric == :output_gap
        return df.q_broker_mean .- df.q_self_mean
    elseif metric == :access_fraction
        total = df.access_count .+ df.assessment_count
        return [
            total[i] > 0 ? df.access_count[i] / total[i] : NaN for i in eachindex(total)
        ]
    elseif metric == :matches_per_agent
        error("matches_per_agent requires the condition configuration")
    end
    return df[!, metric]
end

function seed_late(df::DataFrame, metric::Symbol)::Float64
    first_period = maximum(df.period) - LATE_WIDTH + 1
    mask = df.period .>= first_period
    return nanmean(metric_series(df, metric)[mask])
end

function seed_values(result::ComparisonResult, metric::Symbol; seeds=result.seeds)
    index = Dict(seed => idx for (idx, seed) in enumerate(result.seeds))
    all(haskey(index, seed) for seed in seeds) ||
        error("requested seeds absent from $(result.rel)")
    return [seed_late(result.mdfs[index[seed]], metric) for seed in seeds]
end

function condition_mean(result::ComparisonResult, metric::Symbol; seeds=result.seeds)
    nanmean(seed_values(result, metric; seeds=seeds))
end

function validate_comparison(nn::ComparisonSweep, ridge::ComparisonSweep)
    length(nn.results) == 98 || error("expected 98 NN effective realizations")
    length(ridge.results) == 98 || error("expected 98 Ridge effective realizations")
    length(nn.grid_cells) == 161 || error("expected 161 NN grid coordinates")
    length(ridge.grid_cells) == 161 || error("expected 161 Ridge grid coordinates")
    Set(keys(nn.results)) == Set(keys(ridge.results)) || error("effective designs differ")
    for rel in keys(nn.results)
        nr = nn.results[rel]
        rr = ridge.results[rel]
        for key in DESIGN_KEYS
            nr.config[key] == rr.config[key] || error("design mismatch for $rel: $key")
        end
        expected_seeds = rel == BASELINE_REL ? collect(1:50) : collect(1:20)
        nr.seeds == expected_seeds || error("unexpected NN seed set: $rel")
        rr.seeds == expected_seeds || error("unexpected Ridge seed set: $rel")
    end
    return nothing
end

function monte_carlo_se(values)::Float64
    kept = filter(!isnan, Float64.(collect(values)))
    length(kept) > 1 || return NaN
    return std(kept) / sqrt(length(kept))
end

f3(value) = @sprintf("%.3f", value)
signed3(value) = @sprintf("%+.3f", value)
fint(value) = string(Int(value))

function write_tsv(path::AbstractString, header, rows)
    open(path, "w") do io
        println(io, join(header, '\t'))
        for row in rows
            println(io, join(row, '\t'))
        end
    end
    return nothing
end

function ensemble_series(result::ComparisonResult, metric::Symbol; seeds=result.seeds)
    index = Dict(seed => idx for (idx, seed) in enumerate(result.seeds))
    periods = result.mdfs[index[first(seeds)]].period
    values = Vector{Float64}(undef, length(periods))
    for period_idx in eachindex(periods)
        values[period_idx] = nanmean(
            metric_series(result.mdfs[index[seed]], metric)[period_idx] for seed in seeds
        )
    end
    return periods, values
end

function comparison_figure(nn, ridge, rows)
    common = nn.results[BASELINE_REL].seeds
    period, nn_broker = ensemble_series(nn.results[BASELINE_REL], :broker_holdout_rank)
    _, nn_agent = ensemble_series(nn.results[BASELINE_REL], :agent_holdout_rank)
    _, ridge_broker = ensemble_series(
        ridge.results[BASELINE_REL], :broker_holdout_rank; seeds=common
    )
    _, ridge_agent = ensemble_series(
        ridge.results[BASELINE_REL], :agent_holdout_rank; seeds=common
    )

    fig = Figure(; size=(1500, 540), fontsize=18)
    ax1 = Axis(
        fig[1, 1];
        title="A. Baseline ranking dynamics",
        xlabel="Period",
        ylabel="Holdout rank correlation",
    )
    lines!(ax1, period, nn_broker; color=:black, linewidth=2.5, label="NN broker")
    lines!(ax1, period, nn_agent; color=:gray45, linewidth=2.5, label="NN agents")
    lines!(ax1, period, ridge_broker; color=:steelblue, linewidth=2.5, label="Ridge broker")
    lines!(ax1, period, ridge_agent; color=:darkorange, linewidth=2.5, label="Ridge agents")
    Legend(
        fig[0, 1:3],
        ax1;
        orientation=:horizontal,
        framevisible=false,
        tellwidth=false,
    )

    function identity_scatter(position, x, y, title, label)
        lo = min(minimum(x), minimum(y))
        hi = max(maximum(x), maximum(y))
        pad = max(0.02, 0.05 * (hi - lo))
        ax = Axis(
            position;
            title=title,
            xlabel="NN",
            ylabel="Paired Ridge",
            limits=(lo - pad, hi + pad, lo - pad, hi + pad),
        )
        lines!(
            ax, [lo - pad, hi + pad], [lo - pad, hi + pad]; color=:gray55, linestyle=:dash
        )
        scatter!(ax, x, y; color=(:steelblue, 0.75), markersize=11)
        text!(ax, lo, hi; text=label, align=(:left, :top), fontsize=15)
        return ax
    end

    nn_rank_gap = Float64[row.nn_rank_gap for row in rows]
    ridge_rank_gap = Float64[row.ridge_rank_gap for row in rows]
    identity_scatter(
        fig[1, 2],
        nn_rank_gap,
        ridge_rank_gap,
        "B. Broker-agent rank gap",
        "98 effective realizations",
    )
    nn_output_gap = Float64[row.nn_output_gap for row in rows]
    ridge_output_gap = Float64[row.ridge_output_gap for row in rows]
    identity_scatter(
        fig[1, 3],
        nn_output_gap,
        ridge_output_gap,
        "C. Broker-agent output gap",
        "98 effective realizations",
    )
    save(joinpath(OUT_DIR, "ridge_comparison.png"), fig; px_per_unit=2)
    return nothing
end

function main()
    mkpath(OUT_DIR)
    nn = load_comparison_sweep(NN_ROOT)
    ridge = load_comparison_sweep(RIDGE_ROOT)
    validate_comparison(nn, ridge)

    metrics = (
        :agent_holdout_rank,
        :broker_holdout_rank,
        :rank_gap,
        :agent_holdout_r2,
        :broker_holdout_r2,
        :r2_gap,
        :output_gap,
        :outsourcing_rate,
        :access_fraction,
        :betweenness,
        :mean_degree,
    )

    rows = NamedTuple[]
    for rel in sort!(collect(keys(nn.results)))
        nr = nn.results[rel]
        rr = ridge.results[rel]
        common_seeds = nr.seeds
        values = Dict{Symbol,Float64}()
        for metric in metrics
            values[Symbol("nn_", metric)] = condition_mean(nr, metric)
            values[Symbol("ridge_", metric)] = condition_mean(
                rr, metric; seeds=common_seeds
            )
        end
        push!(
            rows,
            (;
                rel,
                rho=Float64(nr.config["rho"]),
                eta=Float64(nr.config["eta"]),
                delta=Float64(nr.config["delta"]),
                N=Int(nr.config["N"]),
                values...,
            ),
        )
    end

    condition_header = [
        "result_reldir",
        "rho",
        "eta",
        "delta",
        "N",
        [string(prefix, metric) for metric in metrics for prefix in ("nn_", "ridge_")]...,
        ["delta_$(metric)" for metric in metrics]...,
    ]
    condition_rows = Vector{Any}[]
    for row in rows
        values = Any[row.rel, row.rho, row.eta, row.delta, row.N]
        for metric in metrics
            push!(values, getproperty(row, Symbol("nn_", metric)))
            push!(values, getproperty(row, Symbol("ridge_", metric)))
        end
        for metric in metrics
            push!(
                values,
                getproperty(row, Symbol("ridge_", metric)) -
                getproperty(row, Symbol("nn_", metric)),
            )
        end
        push!(condition_rows, values)
    end
    write_tsv(
        joinpath(OUT_DIR, "condition_comparison.tsv"), condition_header, condition_rows
    )

    nn_baseline = nn.results[BASELINE_REL]
    ridge_baseline = ridge.results[BASELINE_REL]
    common_seeds = nn_baseline.seeds
    baseline_rows = Vector{Any}[]
    for seed in common_seeds
        row = Any[seed]
        for metric in metrics
            nn_value = only(seed_values(nn_baseline, metric; seeds=[seed]))
            ridge_value = only(seed_values(ridge_baseline, metric; seeds=[seed]))
            append!(row, (nn_value, ridge_value, ridge_value - nn_value))
        end
        push!(baseline_rows, row)
    end
    baseline_header = [
        "seed",
        [
            string(prefix, metric) for metric in metrics for
            prefix in ("nn_", "ridge_", "delta_")
        ]...,
    ]
    write_tsv(
        joinpath(OUT_DIR, "baseline_common_seeds.tsv"), baseline_header, baseline_rows
    )

    values = Pair{String,String}[]
    add(key, value; formatter=f3) = push!(values, key => formatter(value))
    add("nConditions", length(rows); formatter=fint)
    add("nCommonSeeds", length(common_seeds); formatter=fint)

    for metric in metrics
        nn_value = condition_mean(nn_baseline, metric)
        ridge_common = condition_mean(ridge_baseline, metric; seeds=common_seeds)
        paired =
            seed_values(ridge_baseline, metric; seeds=common_seeds) .-
            seed_values(nn_baseline, metric; seeds=common_seeds)
        label = replace(string(metric), "_" => " ")
        add("nnBaseline_$(metric)", nn_value)
        add("ridgeBaseline_$(metric)", ridge_common)
        add("baselineDelta_$(metric)", nanmean(paired); formatter=signed3)
        add("baselineDeltaSE_$(metric)", monte_carlo_se(paired))
        println(
            rpad(label, 24),
            " NN=",
            f3(nn_value),
            " Ridge=",
            f3(ridge_common),
            " paired delta=",
            signed3(nanmean(paired)),
            " (MC SE ",
            f3(monte_carlo_se(paired)),
            ")",
        )
    end

    for metric in metrics
        nn_values = Float64[getproperty(row, Symbol("nn_", metric)) for row in rows]
        ridge_values = Float64[getproperty(row, Symbol("ridge_", metric)) for row in rows]
        differences = ridge_values .- nn_values
        add("nnMedian_$(metric)", median(nn_values))
        add("ridgeMedian_$(metric)", median(ridge_values))
        add("deltaMean_$(metric)", mean(differences); formatter=signed3)
        add("deltaMedian_$(metric)", median(differences); formatter=signed3)
        add("deltaQ10_$(metric)", quantile(differences, 0.10); formatter=signed3)
        add("deltaQ90_$(metric)", quantile(differences, 0.90); formatter=signed3)
        add("ridgeGreaterN_$(metric)", count(>(0.0), differences); formatter=fint)
        add("nnPositiveN_$(metric)", count(>(0.0), nn_values); formatter=fint)
        add("ridgePositiveN_$(metric)", count(>(0.0), ridge_values); formatter=fint)
        add("conditionCorrelation_$(metric)", cor(nn_values, ridge_values))
    end

    values_path = joinpath(OUT_DIR, "values.tex")
    open(values_path, "w") do io
        println(io, "% Generated by scripts/ridge/analyze_sweep.jl on $(now()).")
        println(io, "% NN sweep: $(basename(NN_ROOT)); commit: $(nn.git_commit)")
        println(io, "% Ridge sweep: $(basename(RIDGE_ROOT)); commit: $(ridge.git_commit)")
        println(io, "% Effective realizations receive equal weight.")
        println(
            io,
            raw"\newcommand{\rvDefine}[2]{\expandafter\newcommand\csname rv@#1\endcsname{#2}}",
        )
        println(io, raw"\newcommand{\rv}[1]{\csname rv@#1\endcsname}")
        for (key, value) in values
            println(io, "\\rvDefine{$key}{$value}")
        end
    end

    open(joinpath(OUT_DIR, "provenance.txt"), "w") do io
        println(io, "generated=$(now())")
        println(io, "nn_sweep=$(basename(normpath(NN_ROOT)))")
        println(io, "nn_manifest=$(nn.manifest_hash)")
        println(io, "nn_schema=$(nn.schema_version)")
        println(io, "nn_commit=$(nn.git_commit)")
        println(io, "ridge_sweep=$(basename(normpath(RIDGE_ROOT)))")
        println(io, "ridge_manifest=$(ridge.manifest_hash)")
        println(io, "ridge_schema=$(ridge.schema_version)")
        println(io, "ridge_commit=$(ridge.git_commit)")
        println(io, "late_periods=$(Int(nn.meta[:T]) - LATE_WIDTH + 1):$(nn.meta[:T])")
        println(io, "weighting=one observation per effective realization")
        println(io, "baseline_comparison_seeds=$(join(common_seeds, ','))")
        println(io, "other_comparison_seeds=$(join(1:20, ','))")
    end

    comparison_figure(nn, ridge, rows)
    println("wrote Ridge comparison outputs to $OUT_DIR")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
