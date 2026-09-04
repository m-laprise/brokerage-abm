"""
    summarize_calibration.jl

Summarize the baseline Ridge-penalty calibration. The selected penalty maximizes
the median, across calibration seeds, of the late-period mean of agent and
broker holdout rank correlations. RMSE and trajectory metrics are retained as
secondary checks.

Required environment:
  BROKERAGE_ABM_RIDGE_CALIBRATION_DIR  directory containing pilot JLD2 files

Usage: julia --project --threads=auto scripts/ridge/summarize_calibration.jl
"""

using DataFrames: DataFrame, eachrow, groupby, names
using JLD2: load, jldsave
using Statistics: mean, median, quantile

const CALIBRATION_DIR = get(ENV, "BROKERAGE_ABM_RIDGE_CALIBRATION_DIR") do
    error("BROKERAGE_ABM_RIDGE_CALIBRATION_DIR is required")
end
const EXPECTED_LAMBDAS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
const EXPECTED_SEEDS = collect(9001:9010)
const LATE_PERIODS = 401:500

finite_mean(values) = mean(filter(isfinite, values))

function run_summary(path)
    artifact = load(path)
    df = artifact["df"]
    config = artifact["config"]
    rows = df[in.(df.period, Ref(LATE_PERIODS)), :]
    agent_rank = finite_mean(rows.agent_holdout_rank)
    broker_rank = finite_mean(rows.broker_holdout_rank)
    lambda_agent = Float64(config["ridge_lambda_agent"])
    lambda_broker = Float64(config["ridge_lambda_broker"])
    lambda_agent == lambda_broker || error("joint calibration requires equal penalties: $path")
    return (
        lambda=lambda_agent,
        seed=Int(config["seed"]),
        agent_rank=agent_rank,
        broker_rank=broker_rank,
        combined_rank=(agent_rank + broker_rank) / 2,
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
        lambda=only(unique(rows.lambda)),
        n_seeds=length(rows.lambda),
        combined_rank_median=median(rows.combined_rank),
        combined_rank_q25=quantile(rows.combined_rank, 0.25),
        combined_rank_q75=quantile(rows.combined_rank, 0.75),
        agent_rank_median=median(rows.agent_rank),
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
            path -> begin
                filename = basename(path)
                (startswith(filename, "paired_ridge_lambda=") ||
                 startswith(filename, "paired_ridge_lambda_a=")) &&
                    endswith(path, ".jld2")
            end,
            readdir(CALIBRATION_DIR; join=true),
        ),
    )
    length(paths) == length(EXPECTED_LAMBDAS) * length(EXPECTED_SEEDS) || error(
        "expected $(length(EXPECTED_LAMBDAS) * length(EXPECTED_SEEDS)) artifacts, found $(length(paths))",
    )

    runs = DataFrame([run_summary(path) for path in paths])
    sort!(runs, [:lambda, :seed])
    sort(unique(runs.lambda)) == EXPECTED_LAMBDAS || error("unexpected penalty grid")
    for lambda in EXPECTED_LAMBDAS
        sort(runs[runs.lambda .== lambda, :seed]) == EXPECTED_SEEDS ||
            error("unexpected seed set for lambda=$lambda")
    end

    by_lambda = DataFrame(
        summarize_lambda(group) for group in groupby(runs, :lambda; sort=true)
    )
    selected_row = by_lambda[argmax(by_lambda.combined_rank_median), :]
    selected_lambda = selected_row.lambda
    selection_rule = "maximize the median across seeds of the late-period mean of agent and broker holdout rank correlations"
    source_commit = readchomp(`git rev-parse HEAD`)

    write_tsv(joinpath(CALIBRATION_DIR, "calibration_runs.tsv"), runs)
    write_tsv(joinpath(CALIBRATION_DIR, "calibration_by_lambda.tsv"), by_lambda)
    jldsave(
        joinpath(CALIBRATION_DIR, "calibration_summary.jld2");
        runs=runs,
        by_lambda=by_lambda,
        selected_lambda=selected_lambda,
        selection_rule=selection_rule,
        candidate_lambdas=EXPECTED_LAMBDAS,
        calibration_seeds=EXPECTED_SEEDS,
        late_periods=collect(LATE_PERIODS),
        source_commit=source_commit,
    )

    show(stdout, "text/plain", by_lambda)
    println("\nselected_lambda=$selected_lambda")
    println("selection_rule=$selection_rule")
    println("source_commit=$source_commit")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
