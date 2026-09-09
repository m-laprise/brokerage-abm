"""
Summarize the five-seed broker assessment vs access pilot from current raw
shards.

Set `BROKERAGE_ABM_ASSESSMENT_ACCESS_SWEEP_DIR` to the sweep root and optionally
set `BROKERAGE_ABM_ASSESSMENT_ACCESS_OUTPUT_DIR` for the output directory.

Usage: julia --project --threads=auto scripts/assessment_access/analyze_pilot.jl
"""

include(joinpath(@__DIR__, "analyze.jl"))
include(normpath(joinpath(@__DIR__, "..", "sweep", "shard_validation.jl")))

using JLD2: jldopen

const PILOT_SEEDS = collect(1:5)

function load_pilot_results()
    manifest_path = joinpath(SWEEP_ROOT, "manifest.jld2")
    conditions, meta, provenance, manifest_hash, schema_version =
        jldopen(manifest_path, "r") do file
            (
                file["conditions"],
                file["meta"],
                file["prov"],
                file["manifest_hash"],
                file["schema_version"],
            )
        end
    meta[:scope] in (:assessment_access, :broker_services) ||
        error("unexpected sweep scope")
    meta[:git_commit] == ANALYSIS_PROVENANCE.commit ||
        error("sweep and analysis commits differ")
    baseline_conditions = filter(conditions) do condition
        params = condition[:resolved_params]
        params[:rho] == 0.5 && params[:eta] == 0.02
    end
    length(baseline_conditions) == 3 || error("expected three service-mode baselines")

    results = SweepResult[]
    for condition in baseline_conditions
        frames = DataFrame[]
        config = Dict(string(key) => value for (key, value) in condition[:resolved_params])
        for seed in PILOT_SEEDS
            shard = joinpath(SWEEP_ROOT, condition[:result_reldir], "seed_$(seed).jld2")
            shard_is_current(shard, provenance) ||
                error("missing current pilot shard: $shard")
            frame = jldopen(shard, "r") do file
                file["df"]
            end
            collect(frame.period) == collect(1:Int(meta[:T])) ||
                error("incomplete period coverage: $shard")
            push!(frames, frame)
        end
        push!(
            results,
            SweepResult(
                condition[:result_reldir],
                condition[:result_reldir],
                frames,
                config,
                PILOT_SEEDS,
                condition[:condition_index],
            ),
        )
    end
    return (; results, meta, manifest_hash=String(manifest_hash), schema_version)
end

function main()
    pilot = load_pilot_results()
    index = Dict(condition_key(result) => result for result in pilot.results)
    mkpath(FIGURE_DIR)

    rows = Vector{Vector{Any}}()
    seed_rows = Vector{Vector{Any}}()
    for mode in MODES, metric in METRICS
        result = index[(mode, 0.5, 0.02)]
        seed_metric_values = seed_values(result, metric)
        for seed in PILOT_SEEDS
            push!(seed_rows, Any[mode, seed, metric, seed_metric_values[seed]])
        end
        interval = monte_carlo_interval(collect(values(seed_metric_values)); level=LEVEL)
        push!(
            rows,
            Any[
                mode,
                metric,
                interval.mean,
                interval.se,
                interval.lower,
                interval.upper,
                interval.n,
            ],
        )
    end
    write_tsv(
        joinpath(OUT_DIR, "pilot_seed_level.tsv"),
        ["broker_service", "seed", "metric", "late_value"],
        seed_rows,
    )
    write_tsv(
        joinpath(OUT_DIR, "pilot_summary.tsv"),
        ["broker_service", "metric", "estimate", "se", "lower", "upper", "n"],
        rows,
    )
    baseline_figure(index)
    jldsave(
        joinpath(OUT_DIR, "pilot_figure_data.jld2");
        seed_rows,
        summary_rows=rows,
        manifest_hash=pilot.manifest_hash,
        schema_version=pilot.schema_version,
        analysis_git_commit=ANALYSIS_PROVENANCE.commit,
        analysis_source_clean=ANALYSIS_PROVENANCE.source_clean,
        late_width=LATE_WIDTH,
        interval_level=LEVEL,
    )
    open(joinpath(OUT_DIR, "provenance.txt"), "w") do io
        println(io, "analysis_git_commit=$(ANALYSIS_PROVENANCE.commit)")
        println(io, "analysis_source_clean=$(ANALYSIS_PROVENANCE.source_clean)")
        println(io, "sweep_git_commit=$(pilot.meta[:git_commit])")
        println(io, "manifest_hash=$(pilot.manifest_hash)")
        println(io, "schema_version=$(pilot.schema_version)")
        println(io, "sweep_root=$(abspath(SWEEP_ROOT))")
    end
    println("Broker assessment vs access pilot analysis written to $OUT_DIR")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
