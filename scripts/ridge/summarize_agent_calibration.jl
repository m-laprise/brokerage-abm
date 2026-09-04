"""
    summarize_agent_calibration.jl

Summarize the baseline calibration of the agent Ridge penalty with the broker
penalty fixed at 0.001. The selected agent penalty maximizes the median across
calibration seeds of the late-period mean agent holdout rank correlation.

Required environment:
  BROKERAGE_ABM_RIDGE_AGENT_CALIBRATION_DIR  directory containing pilot JLD2 files

Usage: julia --project --threads=auto scripts/ridge/summarize_agent_calibration.jl
"""

using DataFrames: DataFrame, eachrow, groupby, names
using JLD2: load, jldsave
using Statistics: mean, median, quantile

const CALIBRATION_DIR = get(ENV, "BROKERAGE_ABM_RIDGE_AGENT_CALIBRATION_DIR") do
    error("BROKERAGE_ABM_RIDGE_AGENT_CALIBRATION_DIR is required")
end
const EXPECTED_AGENT_LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
const EXPECTED_BROKER_LAMBDA = 0.001
const EXPECTED_SEEDS = collect(9001:9010)
const LATE_PERIODS = 401:500

finite_mean(values) = mean(filter(isfinite, values))

function run_summary(path)
    artifact = load(path)
    df = artifact["df"]
    config = artifact["config"]
    rows = df[in.(df.period, Ref(LATE_PERIODS)), :]
    return (
        lambda_agent=Float64(config["ridge_lambda_agent"]),
        lambda_broker=Float64(config["ridge_lambda_broker"]),
        seed=Int(config["seed"]),
        agent_rank=finite_mean(rows.agent_holdout_rank),
        broker_rank=finite_mean(rows.broker_holdout_rank),
        agent_rmse=finite_mean(rows.agent_holdout_rmse),
        broker_rmse=finite_mean(rows.broker_holdout_rmse),
        agent_r2=finite_mean(rows.agent_holdout_r2),
        broker_r2=finite_mean(rows.broker_holdout_r2),
        matches=finite_mean(rows.n_total_matches),
        outsourcing=finite_mean(rows.outsourcing_rate),
    )
end

function summarize_lambda(rows)
    return (
        lambda_agent=only(unique(rows.lambda_agent)),
        lambda_broker=only(unique(rows.lambda_broker)),
        n_seeds=length(rows.lambda_agent),
        agent_rank_median=median(rows.agent_rank),
        agent_rank_q25=quantile(rows.agent_rank, 0.25),
        agent_rank_q75=quantile(rows.agent_rank, 0.75),
        broker_rank_median=median(rows.broker_rank),
        agent_rmse_median=median(rows.agent_rmse),
        broker_rmse_median=median(rows.broker_rmse),
        agent_r2_median=median(rows.agent_r2),
        broker_r2_median=median(rows.broker_r2),
        matches_median=median(rows.matches),
        outsourcing_median=median(rows.outsourcing),
    )
end

function write_tsv(path, table::DataFrame)
    columns = names(table)
    open(path, "w") do io
        println(io, join(columns, '\t'))
        for row in eachrow(table)
            println(io, join((row[column] for column in columns), '\t'))
        end
    end
    return nothing
end

function main()
    paths = sort(
        filter(
            path ->
                startswith(basename(path), "paired_ridge_lambda_a=") &&
                endswith(path, ".jld2"),
            readdir(CALIBRATION_DIR; join=true),
        ),
    )
    expected_count = length(EXPECTED_AGENT_LAMBDAS) * length(EXPECTED_SEEDS)
    length(paths) == expected_count ||
        error("expected $expected_count artifacts, found $(length(paths))")

    runs = DataFrame([run_summary(path) for path in paths])
    sort!(runs, [:lambda_agent, :seed])
    sort(unique(runs.lambda_agent)) == EXPECTED_AGENT_LAMBDAS ||
        error("unexpected agent penalty grid")
    unique(runs.lambda_broker) == [EXPECTED_BROKER_LAMBDA] ||
        error("broker penalty was not held fixed")
    for lambda_agent in EXPECTED_AGENT_LAMBDAS
        sort(runs[runs.lambda_agent .== lambda_agent, :seed]) == EXPECTED_SEEDS ||
            error("unexpected seed set for lambda_agent=$lambda_agent")
    end

    by_lambda = DataFrame(
        summarize_lambda(group) for
        group in groupby(runs, :lambda_agent; sort=true)
    )
    selected_lambda_agent =
        by_lambda[argmax(by_lambda.agent_rank_median), :lambda_agent]
    selection_rule = "maximize the median across seeds of the late-period mean agent holdout rank correlation"
    source_commit = readchomp(`git rev-parse HEAD`)

    write_tsv(joinpath(CALIBRATION_DIR, "agent_calibration_runs.tsv"), runs)
    write_tsv(joinpath(CALIBRATION_DIR, "agent_calibration_by_lambda.tsv"), by_lambda)
    jldsave(
        joinpath(CALIBRATION_DIR, "agent_calibration_summary.jld2");
        runs=runs,
        by_lambda=by_lambda,
        selected_lambda_agent=selected_lambda_agent,
        fixed_lambda_broker=EXPECTED_BROKER_LAMBDA,
        selection_rule=selection_rule,
        candidate_agent_lambdas=EXPECTED_AGENT_LAMBDAS,
        calibration_seeds=EXPECTED_SEEDS,
        late_periods=collect(LATE_PERIODS),
        source_commit=source_commit,
    )

    show(stdout, "text/plain", by_lambda)
    println("\nselected_lambda_agent=$selected_lambda_agent")
    println("fixed_lambda_broker=$EXPECTED_BROKER_LAMBDA")
    println("selection_rule=$selection_rule")
    println("source_commit=$source_commit")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
