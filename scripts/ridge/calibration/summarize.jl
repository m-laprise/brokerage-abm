"""
    summarize.jl <screen|confirm>

Validate and summarize one Ridge calibration stage. Screening preserves
separate principal and broker rankings and does not select penalties.
"""

using DataFrames: DataFrame, eachrow, groupby, names, nrow
using JLD2: jldopen, jldsave, load
using Statistics: mean, median, quantile

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "provenance.jl"))

function ridgecal_finite_mean(values, label)
    all(isfinite, values) || error("nonfinite values in $label")
    isempty(values) && error("empty values in $label")
    return mean(values)
end

function ridgecal_write_tsv(path, table::DataFrame)
    columns = names(table)
    open(path, "w") do io
        println(io, join(columns, '\t'))
        for row in eachrow(table)
            println(io, join((row[column] for column in columns), '\t'))
        end
    end
    return nothing
end

function ridgecal_load_manifest(stage::Symbol)
    path = joinpath(ridgecal_stage_dir(stage), "manifest.jld2")
    isfile(path) || error("manifest not found: $path")
    return load(path)
end

function ridgecal_run_row(stage::Symbol, entry, provenance)
    path = joinpath(
        ridgecal_stage_dir(stage), "runs", entry[:reldir], "seed_$(entry[:seed]).jld2"
    )
    isfile(path) || error("missing Ridge calibration shard: $path")
    artifact = load(path)
    for key in RIDGECAL_PROVENANCE_KEYS
        name = string(key)
        haskey(artifact, name) || error("missing $name provenance: $path")
        artifact[name] == provenance[key] || error("$name mismatch: $path")
    end

    artifact["task_id"] == entry[:index] || error("task id mismatch: $path")
    artifact["seed"] == entry[:seed] || error("seed mismatch: $path")
    config = entry[:config]
    artifact_config = artifact["config"]
    ridgecal_config_key(artifact_config) == ridgecal_config_key(config) ||
        error("configuration mismatch: $path")
    resolved = artifact["resolved_params"]
    resolved["learning_model"] == :ridge || error("non-Ridge shard: $path")
    resolved["ridge_broker_variant"] == :pair || error("non-pair Ridge shard: $path")
    resolved["ridge_lambda_agent"] == config[:lambda_agent] ||
        error("resolved agent penalty mismatch: $path")
    resolved["ridge_lambda_broker"] == config[:lambda_broker] ||
        error("resolved broker penalty mismatch: $path")

    df = artifact["df"]
    nrow(df) == RIDGECAL_T || error("incomplete period table: $path")
    df.period == collect(1:RIDGECAL_T) || error("invalid period sequence: $path")
    early = df[in.(df.period, Ref(RIDGECAL_EARLY_PERIODS)), :]
    late = df[in.(df.period, Ref(RIDGECAL_LATE_PERIODS)), :]
    nrow(early) == length(RIDGECAL_EARLY_PERIODS) ||
        error("incomplete early window: $path")
    nrow(late) == length(RIDGECAL_LATE_PERIODS) ||
        error("incomplete late window: $path")

    agent_early = ridgecal_finite_mean(
        early.agent_holdout_rank, "$path principal early rank"
    )
    agent_late = ridgecal_finite_mean(
        late.agent_holdout_rank, "$path principal late rank"
    )
    broker_early = ridgecal_finite_mean(
        early.broker_holdout_rank, "$path broker early rank"
    )
    broker_late = ridgecal_finite_mean(
        late.broker_holdout_rank, "$path broker late rank"
    )

    return (
        stage=string(stage),
        config_id=Int(config[:config_id]),
        seed=Int(entry[:seed]),
        lambda_agent=Float64(config[:lambda_agent]),
        lambda_broker=Float64(config[:lambda_broker]),
        agent_rank_early=agent_early,
        agent_rank_late=agent_late,
        agent_rank_change=agent_late - agent_early,
        broker_rank_early=broker_early,
        broker_rank_late=broker_late,
        broker_rank_change=broker_late - broker_early,
        agent_rmse_late=ridgecal_finite_mean(
            late.agent_holdout_rmse, "$path principal RMSE"
        ),
        broker_rmse_late=ridgecal_finite_mean(
            late.broker_holdout_rmse, "$path broker RMSE"
        ),
        agent_bias_late=ridgecal_finite_mean(
            late.agent_holdout_bias, "$path principal bias"
        ),
        broker_bias_late=ridgecal_finite_mean(
            late.broker_holdout_bias, "$path broker bias"
        ),
        matches_late=ridgecal_finite_mean(late.n_total_matches, "$path matches"),
        outsourcing_late=ridgecal_finite_mean(
            late.outsourcing_rate, "$path outsourcing"
        ),
        elapsed_s=Float64(artifact["elapsed_s"]),
    )
end

function ridgecal_stage_rows(stage::Symbol)
    manifest = ridgecal_load_manifest(stage)
    provenance = manifest["provenance"]
    ridgecal_verify_runtime_provenance(
        provenance;
        require_clean=get(ENV, "BROKERAGE_ABM_ALLOW_DIRTY", "0") != "1",
    )
    return DataFrame(
        ridgecal_run_row(stage, entry, provenance) for entry in manifest["entries"]
    )
end

function ridgecal_config_summary(rows::DataFrame)
    output = NamedTuple[]
    for config_id in sort(unique(rows.config_id))
        group = rows[rows.config_id .== config_id, :]
        length(unique(group.seed)) == nrow(group) ||
            error("duplicate seed for config $config_id")
        push!(
            output,
            (
                config_id=config_id,
                n_seeds=nrow(group),
                lambda_agent=only(unique(group.lambda_agent)),
                lambda_broker=only(unique(group.lambda_broker)),
                agent_rank_median=median(group.agent_rank_late),
                agent_rank_q25=quantile(group.agent_rank_late, 0.25),
                agent_rank_q75=quantile(group.agent_rank_late, 0.75),
                broker_rank_median=median(group.broker_rank_late),
                broker_rank_q25=quantile(group.broker_rank_late, 0.25),
                broker_rank_q75=quantile(group.broker_rank_late, 0.75),
                agent_rank_change_median=median(group.agent_rank_change),
                broker_rank_change_median=median(group.broker_rank_change),
                agent_rmse_median=median(group.agent_rmse_late),
                broker_rmse_median=median(group.broker_rmse_late),
                agent_bias_median=median(group.agent_bias_late),
                broker_bias_median=median(group.broker_bias_late),
                matches_median=median(group.matches_late),
                outsourcing_median=median(group.outsourcing_late),
                runtime_median_s=median(group.elapsed_s),
            ),
        )
    end
    return DataFrame(output)
end

function ridgecal_marginal_summary(summary::DataFrame, learner::Symbol)
    learner in (:agent, :broker) || error("unknown learner: $learner")
    lambda_column = learner == :agent ? :lambda_agent : :lambda_broker
    rank_column = learner == :agent ? :agent_rank_median : :broker_rank_median
    change_column =
        learner == :agent ? :agent_rank_change_median : :broker_rank_change_median
    output = NamedTuple[]
    for lambda in sort(unique(summary[!, lambda_column]))
        group = summary[summary[!, lambda_column] .== lambda, :]
        ranks = group[!, rank_column]
        changes = group[!, change_column]
        push!(
            output,
            (
                lambda=lambda,
                n_other_penalties=nrow(group),
                rank_median_across_cells=median(ranks),
                rank_minimum_across_cells=minimum(ranks),
                rank_maximum_across_cells=maximum(ranks),
                rank_change_median_across_cells=median(changes),
            ),
        )
    end
    result = DataFrame(output)
    sort!(result, :rank_median_across_cells; rev=true)
    return result
end

function ridgecal_validate_stage_seed_sets(rows, expected_seeds)
    expected = Set(expected_seeds)
    all(Set(group.seed) == expected for group in groupby(rows, :config_id)) ||
        error("calibration seed set is incomplete")
    return nothing
end

function ridgecal_summarize_screen()
    rows = ridgecal_stage_rows(:screen)
    ridgecal_validate_stage_seed_sets(rows, RIDGECAL_SCREEN_SEEDS)
    summary = ridgecal_config_summary(rows)
    nrow(summary) == length(RIDGECAL_LAMBDAS)^2 ||
        error("screen configuration grid is incomplete")
    agent_ranking = ridgecal_marginal_summary(summary, :agent)
    broker_ranking = ridgecal_marginal_summary(summary, :broker)

    outdir = ridgecal_summary_dir()
    mkpath(outdir)
    ridgecal_write_tsv(joinpath(outdir, "screen_runs.tsv"), rows)
    ridgecal_write_tsv(joinpath(outdir, "screen_by_config.tsv"), summary)
    ridgecal_write_tsv(joinpath(outdir, "screen_agent_ranking.tsv"), agent_ranking)
    ridgecal_write_tsv(joinpath(outdir, "screen_broker_ranking.tsv"), broker_ranking)
    jldsave(
        joinpath(outdir, "screen_summary.jld2");
        rows=rows,
        summary=summary,
        agent_ranking=agent_ranking,
        broker_ranking=broker_ranking,
        selection_rule="researcher review of each learner's own ranking, stability, and level error; no automatic selection",
    )
    println("principal marginal ranking:")
    show(stdout, "text/plain", agent_ranking)
    println("\nbroker marginal ranking:")
    show(stdout, "text/plain", broker_ranking)
    println("\nNo penalties were selected automatically.")
    return nothing
end

function ridgecal_summarize_confirm()
    manifest = ridgecal_load_manifest(:confirm)
    rows = ridgecal_stage_rows(:confirm)
    ridgecal_validate_stage_seed_sets(rows, RIDGECAL_CONFIRM_SEEDS)
    summary = ridgecal_config_summary(rows)
    nrow(summary) == 1 || error("confirmation must contain one configuration")
    result = summary[1, :]
    stable =
        abs(result.agent_rank_change_median) <= RIDGECAL_PRACTICAL_TOLERANCE &&
        abs(result.broker_rank_change_median) <= RIDGECAL_PRACTICAL_TOLERANCE
    selection_note = manifest["meta"][:selection_note]

    outdir = ridgecal_summary_dir()
    mkpath(outdir)
    ridgecal_write_tsv(joinpath(outdir, "confirm_runs.tsv"), rows)
    ridgecal_write_tsv(joinpath(outdir, "confirm_summary.tsv"), summary)
    jldsave(
        joinpath(outdir, "confirm_summary.jld2");
        rows=rows,
        summary=summary,
        late_window_stable=stable,
        practical_tolerance=RIDGECAL_PRACTICAL_TOLERANCE,
        selection_note=selection_note,
        screen_manifest_hash=manifest["meta"][:screen_manifest_hash],
    )
    println("late_window_stable=$stable")
    println("selection_note=$selection_note")
    show(stdout, "text/plain", summary)
    println()
    return nothing
end

function ridgecal_summary_main()
    length(ARGS) == 1 || error("usage: summarize.jl <screen|confirm>")
    stage = Symbol(only(ARGS))
    stage == :screen && return ridgecal_summarize_screen()
    stage == :confirm && return ridgecal_summarize_confirm()
    error("invalid stage: $stage")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && ridgecal_summary_main()
