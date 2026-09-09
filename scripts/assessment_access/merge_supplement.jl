"""
Merge a completed supplemental sweep into the retained assessment-versus-access
figure data.

Usage:
  julia --project --threads=auto scripts/assessment_access/merge_supplement.jl \
      <base figure_data.jld2> <supplement sweep root> <output directory>
"""

using DataFrames: DataFrame
using JLD2: jldsave, load
using SHA: sha256
using Statistics: mean

const REPO_ROOT = normpath(
    get(ENV, "BROKERAGE_ABM_REPO_ROOT", joinpath(@__DIR__, "..", ".."))
)
include(joinpath(REPO_ROOT, "scripts", "sweep", "sweep_results.jl"))
include(joinpath(REPO_ROOT, "scripts", "monte_carlo.jl"))
include(joinpath(REPO_ROOT, "scripts", "reporting_provenance.jl"))

const LATE_WIDTH = 20
const LEVEL = 0.95
const MODES = (:full, :assessment_only, :access_only)
const METRICS = (
    :agent_holdout_rank,
    :broker_holdout_rank,
    :rank_gap,
    :broker_selected_rank,
    :outsourcing_rate,
    :self_fill_rate,
    :broker_fill_rate,
    :self_net_output_per_requested_position,
    :broker_net_output_per_requested_position,
    :net_output_per_requested_position,
    :q_gap,
    :access_fraction,
    :betweenness,
    :mean_degree,
)

finite_mean(values) = begin
    kept = filter(isfinite, Float64.(collect(values)))
    isempty(kept) ? NaN : mean(kept)
end

function metric_series(df::DataFrame, metric::Symbol)
    if metric == :rank_gap
        return df.broker_holdout_rank .- df.agent_holdout_rank
    elseif metric == :q_gap
        return df.q_broker_mean .- df.q_self_mean
    elseif metric == :access_fraction
        total = df.access_count .+ df.assessment_count
        return [
            total[i] > 0 ? df.access_count[i] / total[i] : NaN for i in eachindex(total)
        ]
    end
    return df[!, metric]
end

function seed_values(result::SweepResult, metric::Symbol)
    first_period = maximum(result.mdfs[1].period) - LATE_WIDTH + 1
    return Dict(
        seed => finite_mean(
            metric_series(result.mdfs[i], metric)[result.mdfs[i].period .>= first_period],
        ) for (i, seed) in enumerate(result.seeds)
    )
end

function paired_interval(reference::SweepResult, comparison::SweepResult, metric::Symbol)
    reference_values = seed_values(reference, metric)
    comparison_values = seed_values(comparison, metric)
    seeds = sort!(collect(intersect(keys(reference_values), keys(comparison_values))))
    return paired_monte_carlo_interval(
        [reference_values[seed] for seed in seeds],
        [comparison_values[seed] for seed in seeds];
        level=LEVEL,
    )
end

function replace_rows(base_rows, added_rows, key_columns)
    added_keys = Set(Tuple(row[column] for column in key_columns) for row in added_rows)
    retained = filter(base_rows) do row
        Tuple(row[column] for column in key_columns) ∉ added_keys
    end
    append!(retained, added_rows)
    return retained
end

function write_tsv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, '\t'))
        foreach(row -> println(io, join(row, '\t')), rows)
    end
end

function provenance_value(path, key)
    prefix = "$key="
    for line in eachline(path)
        startswith(line, prefix) && return line[(length(prefix) + 1):end]
    end
    error("missing $key in $path")
end

function main()
    length(ARGS) == 3 ||
        error("expected base data, supplemental sweep, and output directory")
    base_path, supplement_root, output_dir = ARGS
    base = load(base_path)
    base["analysis_source_clean"] == true || error("base analysis source was not clean")
    base["late_width"] == LATE_WIDTH || error("late-window mismatch")
    base["interval_level"] == LEVEL || error("interval-level mismatch")

    analysis_provenance = reporting_git_provenance(REPO_ROOT)
    validate_analysis_commit(
        analysis_provenance,
        base["analysis_git_commit"];
        artifact="base assessment vs access analysis",
    )

    dataset = load_sweep_dataset(supplement_root)
    dataset.schema_version == base["schema_version"] || error("schema mismatch")
    base_provenance = joinpath(dirname(base_path), "provenance.txt")
    simulation_commit = provenance_value(base_provenance, "sweep_git_commit")
    dataset.meta[:git_commit] == simulation_commit ||
        error("base and supplemental simulation commits differ")
    index = Dict(
        (
            Symbol(result.cfg["broker_service"]),
            Float64(result.cfg["rho"]),
            Float64(result.cfg["eta"]),
        ) => result for result in dataset.results
    )
    expected = Set((mode, 0.5, 0.001) for mode in MODES)
    Set(keys(index)) == expected || error("unexpected supplemental conditions")
    all(result -> result.seeds == collect(1:20), values(index)) ||
        error("supplemental seed set mismatch")

    seed_rows = Vector{Vector{Any}}()
    summary_rows = Vector{Vector{Any}}()
    for key in sort!(collect(keys(index)); by=string)
        mode, rho, eta = key
        result = index[key]
        for metric in METRICS
            per_seed = seed_values(result, metric)
            for seed in sort!(collect(keys(per_seed)))
                push!(seed_rows, Any[mode, rho, eta, seed, metric, per_seed[seed]])
            end
            interval = monte_carlo_interval(collect(values(per_seed)); level=LEVEL)
            push!(
                summary_rows,
                Any[
                    mode,
                    rho,
                    eta,
                    metric,
                    interval.mean,
                    interval.se,
                    interval.lower,
                    interval.upper,
                    interval.n,
                ],
            )
        end
    end

    contrast_rows = Vector{Vector{Any}}()
    for metric in METRICS
        for (label, reference, comparison) in (
            ("assessment", :access_only, :full),
            ("access", :assessment_only, :full),
            ("assessment_vs_access", :access_only, :assessment_only),
        )
            interval = paired_interval(
                index[(reference, 0.5, 0.001)], index[(comparison, 0.5, 0.001)], metric
            )
            push!(
                contrast_rows,
                Any[
                    label,
                    0.5,
                    0.001,
                    metric,
                    interval.mean,
                    interval.se,
                    interval.lower,
                    interval.upper,
                    interval.n,
                ],
            )
        end
    end

    combined_seed_rows = replace_rows(base["seed_rows"], seed_rows, 1:5)
    combined_summary_rows = replace_rows(base["summary_rows"], summary_rows, 1:4)
    combined_contrast_rows = replace_rows(base["contrast_rows"], contrast_rows, 1:4)
    sort!(combined_seed_rows; by=row -> string(row[1:5]))
    sort!(combined_summary_rows; by=row -> string(row[1:4]))
    sort!(combined_contrast_rows; by=row -> string(row[1:4]))

    mkpath(output_dir)
    write_tsv(
        joinpath(output_dir, "seed_level.tsv"),
        ["broker_service", "rho", "eta", "seed", "metric", "late_value"],
        combined_seed_rows,
    )
    write_tsv(
        joinpath(output_dir, "condition_summary.tsv"),
        ["broker_service", "rho", "eta", "metric", "estimate", "se", "lower", "upper", "n"],
        combined_summary_rows,
    )
    write_tsv(
        joinpath(output_dir, "paired_contrasts.tsv"),
        ["contribution", "rho", "eta", "metric", "estimate", "se", "lower", "upper", "n"],
        combined_contrast_rows,
    )
    script_hash = bytes2hex(sha256(read(@__FILE__)))
    base_manifest_hash = base["manifest_hash"]
    base_source_root = base["source_root"]
    jldsave(
        joinpath(output_dir, "figure_data.jld2");
        seed_rows=combined_seed_rows,
        summary_rows=combined_summary_rows,
        contrast_rows=combined_contrast_rows,
        manifest_hash=base_manifest_hash,
        supplement_manifest_hash=dataset.manifest_hash,
        schema_version=dataset.schema_version,
        analysis_git_commit=analysis_provenance.commit,
        analysis_source_clean=analysis_provenance.source_clean,
        source_root=base_source_root,
        supplement_source_root=abspath(supplement_root),
        merge_script_sha256=script_hash,
        late_width=LATE_WIDTH,
        interval_level=LEVEL,
    )
    open(joinpath(output_dir, "provenance.txt"), "w") do io
        println(io, "analysis_git_commit=$(analysis_provenance.commit)")
        println(io, "analysis_source_clean=$(analysis_provenance.source_clean)")
        println(io, "sweep_git_commit=$simulation_commit")
        println(io, "manifest_hash=$base_manifest_hash")
        println(io, "supplement_manifest_hash=$(dataset.manifest_hash)")
        println(io, "schema_version=$(dataset.schema_version)")
        println(io, "sweep_root=$base_source_root")
        println(io, "supplement_sweep_root=$(abspath(supplement_root))")
        println(io, "merge_script_sha256=$script_hash")
    end
    println("Merged supplemental assessment-versus-access data into $output_dir")
end

main()
