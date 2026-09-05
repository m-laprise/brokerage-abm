"""
    summarize.jl <screen|confirm>

Validate and summarize one neural-network calibration stage. Screening ranks
each learner's candidate settings. Confirmation checks the selected joint
configuration over five seeds.
"""

using DataFrames: DataFrame, eachrow, groupby, names, nrow
using JLD2
using Statistics: mean, median, quantile

include(joinpath(@__DIR__, "calibration_config.jl"))
include(joinpath(@__DIR__, "provenance.jl"))

function nncal_finite_mean(values, label)
    all(isfinite, values) || error("nonfinite values in $label")
    return mean(values)
end

function nncal_write_tsv(path, table::DataFrame)
    columns = names(table)
    open(path, "w") do io
        println(io, join(columns, '\t'))
        for row in eachrow(table)
            println(io, join((row[column] for column in columns), '\t'))
        end
    end
    return nothing
end

function nncal_load_manifest(stage::Symbol)
    path = joinpath(nncal_stage_dir(stage), "manifest.jld2")
    isfile(path) || error("manifest not found: $path")
    return load(path)
end

function nncal_run_row(stage::Symbol, entry, provenance)
    path = joinpath(
        nncal_stage_dir(stage), "runs", entry[:reldir], "seed_$(entry[:seed]).jld2"
    )
    isfile(path) || error("missing calibration shard: $path")
    artifact = load(path)
    for key in NNCAL_PROVENANCE_KEYS
        name = string(key)
        haskey(artifact, name) || error("missing $name provenance: $path")
        artifact[name] == provenance[key] || error("$name mismatch: $path")
    end

    df = artifact["df"]
    early = df[in.(df.period, Ref(NNCAL_EARLY_PERIODS)), :]
    late = df[in.(df.period, Ref(NNCAL_LATE_PERIODS)), :]
    nrow(early) == length(NNCAL_EARLY_PERIODS) || error("incomplete early window: $path")
    nrow(late) == length(NNCAL_LATE_PERIODS) || error("incomplete late window: $path")
    config = entry[:config]
    agent_early = nncal_finite_mean(early.agent_holdout_rank, "$path agent early rank")
    broker_early = nncal_finite_mean(early.broker_holdout_rank, "$path broker early rank")
    agent_late = nncal_finite_mean(late.agent_holdout_rank, "$path agent late rank")
    broker_late = nncal_finite_mean(late.broker_holdout_rank, "$path broker late rank")

    return (
        stage=string(stage),
        config_id=Int(config[:config_id]),
        seed=Int(entry[:seed]),
        agent_eta_lr=Float64(config[:agent_eta_lr]),
        agent_initial_steps=Int(config[:agent_initial_steps]),
        agent_recurrent_steps=Int(config[:agent_recurrent_steps]),
        broker_eta_lr=Float64(config[:broker_eta_lr]),
        broker_initial_steps=Int(config[:broker_initial_steps]),
        broker_recurrent_steps=Int(config[:broker_recurrent_steps]),
        agent_scan=Bool(config[:agent_scan]),
        broker_scan=Bool(config[:broker_scan]),
        agent_rank_early=agent_early,
        agent_rank_late=agent_late,
        agent_rank_change=agent_late - agent_early,
        broker_rank_early=broker_early,
        broker_rank_late=broker_late,
        broker_rank_change=broker_late - broker_early,
        agent_rmse_late=nncal_finite_mean(late.agent_holdout_rmse, "$path agent RMSE"),
        broker_rmse_late=nncal_finite_mean(late.broker_holdout_rmse, "$path broker RMSE"),
        agent_bias_late=nncal_finite_mean(late.agent_holdout_bias, "$path agent bias"),
        broker_bias_late=nncal_finite_mean(late.broker_holdout_bias, "$path broker bias"),
        elapsed_s=Float64(artifact["elapsed_s"]),
    )
end

function nncal_stage_rows(stage::Symbol)
    manifest = nncal_load_manifest(stage)
    entries = manifest["entries"]
    provenance = manifest["provenance"]
    nncal_verify_runtime_provenance(
        provenance;
        require_clean=get(ENV, "BROKERAGE_ABM_ALLOW_DIRTY", "0") != "1",
    )
    return DataFrame(nncal_run_row(stage, entry, provenance) for entry in entries)
end

function nncal_config_summary(rows::DataFrame)
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
                agent_eta_lr=only(unique(group.agent_eta_lr)),
                agent_initial_steps=only(unique(group.agent_initial_steps)),
                agent_recurrent_steps=only(unique(group.agent_recurrent_steps)),
                broker_eta_lr=only(unique(group.broker_eta_lr)),
                broker_initial_steps=only(unique(group.broker_initial_steps)),
                broker_recurrent_steps=only(unique(group.broker_recurrent_steps)),
                agent_scan=only(unique(group.agent_scan)),
                broker_scan=only(unique(group.broker_scan)),
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
                runtime_median_s=median(group.elapsed_s),
            ),
        )
    end
    return DataFrame(output)
end

function nncal_scan_order(summary::DataFrame, learner::Symbol)
    scan_column = learner == :agent ? :agent_scan : :broker_scan
    metric_column = learner == :agent ? :agent_rank_median : :broker_rank_median
    steps_column = learner == :agent ? :agent_recurrent_steps : :broker_recurrent_steps
    rate_column = learner == :agent ? :agent_eta_lr : :broker_eta_lr
    candidates = summary[summary[!, scan_column], :]
    order = sortperm(
        1:nrow(candidates);
        by=index -> (
            -candidates[index, metric_column],
            candidates[index, steps_column],
            abs(log(candidates[index, rate_column] / NNCAL_REFERENCE_LEARNING_RATE)),
            candidates[index, :config_id],
        ),
    )
    return candidates[order, :]
end

function nncal_summarize_screen()
    rows = nncal_stage_rows(:screen)
    expected = Set(NNCAL_SCREEN_SEEDS)
    all(Set(group.seed) == expected for group in groupby(rows, :config_id)) ||
        error("screen seed set is incomplete")
    summary = nncal_config_summary(rows)
    agent_ranking = nncal_scan_order(summary, :agent)
    broker_ranking = nncal_scan_order(summary, :broker)

    outdir = nncal_summary_dir()
    mkpath(outdir)
    nncal_write_tsv(joinpath(outdir, "screen_runs.tsv"), rows)
    nncal_write_tsv(joinpath(outdir, "screen_by_config.tsv"), summary)
    jldsave(
        joinpath(outdir, "screen_summary.jld2");
        rows=rows,
        summary=summary,
        agent_ranking=agent_ranking,
        broker_ranking=broker_ranking,
    )
    println("agent screen ranking:")
    show(stdout, "text/plain", agent_ranking)
    println("\nbroker screen ranking:")
    show(stdout, "text/plain", broker_ranking)
    println()
    return nothing
end

function nncal_summarize_confirm()
    rows = nncal_stage_rows(:confirm)
    Set(rows.seed) == Set(NNCAL_CONFIRM_SEEDS) || error("confirmation seed set is incomplete")
    summary = nncal_config_summary(rows)
    nrow(summary) == 1 || error("confirmation stage must contain one configuration")
    result = summary[1, :]
    stable =
        abs(result.agent_rank_change_median) <= NNCAL_PRACTICAL_TOLERANCE &&
        abs(result.broker_rank_change_median) <= NNCAL_PRACTICAL_TOLERANCE

    outdir = nncal_summary_dir()
    mkpath(outdir)
    nncal_write_tsv(joinpath(outdir, "confirm_runs.tsv"), rows)
    nncal_write_tsv(joinpath(outdir, "confirm_summary.tsv"), summary)
    jldsave(
        joinpath(outdir, "confirm_summary.jld2");
        rows=rows,
        summary=summary,
        late_window_stable=stable,
        practical_tolerance=NNCAL_PRACTICAL_TOLERANCE,
        selection_rationale="ranking performance, late-window stability, runtime, and a common 50-step update budget",
    )
    println("late_window_stable=$stable")
    show(stdout, "text/plain", summary)
    println()
    return nothing
end

function main()
    length(ARGS) == 1 || error("usage: summarize.jl <screen|confirm>")
    stage = Symbol(only(ARGS))
    stage == :screen && return nncal_summarize_screen()
    stage == :confirm && return nncal_summarize_confirm()
    error("invalid stage: $stage")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
