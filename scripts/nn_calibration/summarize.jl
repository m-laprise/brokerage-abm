"""
    summarize.jl <screen|confirm|combined>

Validate and summarize one neural-network calibration stage. Screening selects
two configurations per learner for confirmation. Confirmation applies the
prespecified practical-equivalence rule. The combined summary checks whether
the separately selected settings remain adequate when used together.
"""

using DataFrames: DataFrame, eachrow, groupby, names, nrow
using JLD2
using Statistics: mean, median, quantile

include(joinpath(@__DIR__, "calibration_config.jl"))

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
    artifact["git_commit"] == provenance[:git_commit] || error("commit mismatch: $path")
    artifact["pkg_manifest_hash"] == provenance[:pkg_manifest_hash] ||
        error("package manifest mismatch: $path")
    artifact["manifest_hash"] == provenance[:manifest_hash] ||
        error("calibration manifest mismatch: $path")
    artifact["schema_version"] == provenance[:schema_version] ||
        error("schema mismatch: $path")

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

function nncal_select_efficient(summary::DataFrame, learner::Symbol)
    ranking = nncal_scan_order(summary, learner)
    metric_column = learner == :agent ? :agent_rank_median : :broker_rank_median
    steps_column = learner == :agent ? :agent_recurrent_steps : :broker_recurrent_steps
    rate_column = learner == :agent ? :agent_eta_lr : :broker_eta_lr
    best = maximum(ranking[!, metric_column])
    eligible = ranking[ranking[!, metric_column] .>= best - NNCAL_PRACTICAL_TOLERANCE, :]
    minimum_steps = minimum(eligible[!, steps_column])
    eligible = eligible[eligible[!, steps_column] .== minimum_steps, :]
    preferred = eligible[eligible[!, rate_column] .== NNCAL_REFERENCE_LEARNING_RATE, :]
    !isempty(preferred) && (eligible = preferred)
    order = sortperm(
        1:nrow(eligible);
        by=index -> (-eligible[index, metric_column], eligible[index, rate_column]),
    )
    return eligible[order[1], :]
end

function nncal_config_by_id(configs, config_id::Integer)
    matches = filter(config -> config[:config_id] == config_id, configs)
    length(matches) == 1 || error("expected one config with id $config_id")
    return only(matches)
end

function nncal_shortlist_configs(configs, agent_ids, broker_ids)
    shortlisted_ids = unique(vcat(agent_ids, broker_ids))
    shortlisted = Dict{Symbol,Any}[]
    for config_id in shortlisted_ids
        config = deepcopy(nncal_config_by_id(configs, config_id))
        config[:agent_scan] = config_id in agent_ids
        config[:broker_scan] = config_id in broker_ids
        push!(shortlisted, config)
    end
    return shortlisted
end

function nncal_agent_setting(config)
    return nncal_setting(config[:agent_eta_lr], config[:agent_recurrent_steps])
end

function nncal_broker_setting(config)
    return nncal_setting(config[:broker_eta_lr], config[:broker_recurrent_steps])
end

function nncal_boundary_flag(setting)
    return setting.eta_lr in (first(NNCAL_LEARNING_RATES), last(NNCAL_LEARNING_RATES)) ||
           setting.recurrent_steps in
           (first(NNCAL_RECURRENT_STEPS), last(NNCAL_RECURRENT_STEPS))
end

function nncal_summarize_screen()
    rows = nncal_stage_rows(:screen)
    expected = Set(NNCAL_SCREEN_SEEDS)
    all(Set(group.seed) == expected for group in groupby(rows, :config_id)) ||
        error("screen seed set is incomplete")
    summary = nncal_config_summary(rows)
    agent_ranking = nncal_scan_order(summary, :agent)
    broker_ranking = nncal_scan_order(summary, :broker)
    agent_shortlist_ids = agent_ranking.config_id[1:NNCAL_SHORTLIST_SIZE]
    broker_shortlist_ids = broker_ranking.config_id[1:NNCAL_SHORTLIST_SIZE]
    configs = nncal_load_manifest(:screen)["configs"]
    shortlist_configs = nncal_shortlist_configs(
        configs, agent_shortlist_ids, broker_shortlist_ids
    )
    shortlist_ids = [config[:config_id] for config in shortlist_configs]

    outdir = nncal_summary_dir()
    mkpath(outdir)
    nncal_write_tsv(joinpath(outdir, "screen_runs.tsv"), rows)
    nncal_write_tsv(joinpath(outdir, "screen_by_config.tsv"), summary)
    jldsave(
        joinpath(outdir, "screen_selection.jld2");
        rows=rows,
        summary=summary,
        agent_ranking=agent_ranking,
        broker_ranking=broker_ranking,
        agent_shortlist_config_ids=agent_shortlist_ids,
        broker_shortlist_config_ids=broker_shortlist_ids,
        shortlist_config_ids=shortlist_ids,
        shortlist_configs=shortlist_configs,
        shortlist_rule="two highest three-seed median rank correlations per learner",
    )
    println("agent screen ranking:")
    show(stdout, "text/plain", agent_ranking)
    println("\nbroker screen ranking:")
    show(stdout, "text/plain", broker_ranking)
    println("\nshortlist_config_ids=$(join(shortlist_ids, ','))")
    return nothing
end

function nncal_summarize_confirm()
    screen = nncal_stage_rows(:screen)
    confirm = nncal_stage_rows(:confirm)
    finalist_ids = Set(unique(confirm.config_id))
    rows = vcat(screen[in.(screen.config_id, Ref(finalist_ids)), :], confirm)
    expected = Set(NNCAL_ALL_SEEDS)
    all(Set(group.seed) == expected for group in groupby(rows, :config_id)) ||
        error("confirmation seed set is incomplete")
    summary = nncal_config_summary(rows)
    selected_agent = nncal_select_efficient(summary, :agent)
    selected_broker = nncal_select_efficient(summary, :broker)
    configs = nncal_load_manifest(:confirm)["configs"]
    agent_config = nncal_config_by_id(configs, selected_agent.config_id)
    broker_config = nncal_config_by_id(configs, selected_broker.config_id)
    agent_setting = nncal_agent_setting(agent_config)
    broker_setting = nncal_broker_setting(broker_config)

    outdir = nncal_summary_dir()
    nncal_write_tsv(joinpath(outdir, "confirm_runs.tsv"), rows)
    nncal_write_tsv(joinpath(outdir, "confirm_by_config.tsv"), summary)
    jldsave(
        joinpath(outdir, "confirmed_selection.jld2");
        rows=rows,
        summary=summary,
        selected_agent_config_id=selected_agent.config_id,
        selected_broker_config_id=selected_broker.config_id,
        selected_agent_setting=agent_setting,
        selected_broker_setting=broker_setting,
        agent_boundary_extension_required=nncal_boundary_flag(agent_setting),
        broker_boundary_extension_required=nncal_boundary_flag(broker_setting),
        selection_rule="smallest recurrent budget within 0.01 of the best five-seed median rank; prefer learning rate 0.01 at equal cost",
    )
    println("selected_agent_setting=$agent_setting")
    println("selected_broker_setting=$broker_setting")
    println("agent_boundary_extension_required=$(nncal_boundary_flag(agent_setting))")
    println("broker_boundary_extension_required=$(nncal_boundary_flag(broker_setting))")
    return nothing
end

function nncal_summarize_combined()
    rows = nncal_stage_rows(:combined)
    Set(rows.seed) == Set(NNCAL_ALL_SEEDS) || error("combined seed set is incomplete")
    summary = nncal_config_summary(rows)
    nrow(summary) == 1 || error("combined stage must contain one configuration")
    selection = load(joinpath(nncal_summary_dir(), "confirmed_selection.jld2"))
    confirm_summary = selection["summary"]
    agent_reference = only(
        confirm_summary[
            confirm_summary.config_id .== selection["selected_agent_config_id"],
            :agent_rank_median,
        ],
    )
    broker_reference = only(
        confirm_summary[
            confirm_summary.config_id .== selection["selected_broker_config_id"],
            :broker_rank_median,
        ],
    )
    combined = summary[1, :]
    agent_interaction_change = combined.agent_rank_median - agent_reference
    broker_interaction_change = combined.broker_rank_median - broker_reference
    interaction_ok =
        agent_interaction_change >= -NNCAL_PRACTICAL_TOLERANCE &&
        broker_interaction_change >= -NNCAL_PRACTICAL_TOLERANCE
    stable =
        abs(combined.agent_rank_change_median) <= NNCAL_PRACTICAL_TOLERANCE &&
        abs(combined.broker_rank_change_median) <= NNCAL_PRACTICAL_TOLERANCE

    outdir = nncal_summary_dir()
    nncal_write_tsv(joinpath(outdir, "combined_runs.tsv"), rows)
    nncal_write_tsv(joinpath(outdir, "combined_summary.tsv"), summary)
    jldsave(
        joinpath(outdir, "combined_summary.jld2");
        rows=rows,
        summary=summary,
        agent_interaction_change=agent_interaction_change,
        broker_interaction_change=broker_interaction_change,
        interaction_ok=interaction_ok,
        late_window_stable=stable,
        practical_tolerance=NNCAL_PRACTICAL_TOLERANCE,
    )
    println("agent_interaction_change=$agent_interaction_change")
    println("broker_interaction_change=$broker_interaction_change")
    println("interaction_ok=$interaction_ok")
    println("late_window_stable=$stable")
    return nothing
end

function main()
    length(ARGS) == 1 || error("usage: summarize.jl <screen|confirm|combined>")
    stage = Symbol(only(ARGS))
    stage == :screen && return nncal_summarize_screen()
    stage == :confirm && return nncal_summarize_confirm()
    stage == :combined && return nncal_summarize_combined()
    error("invalid stage: $stage")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
