"""
Extract NN ranking diagnostics and output sources from saved ρ-by-turnover runs.
Run on a compute node with the simulation's Julia version; no periods are rerun.

Usage: julia --project --threads=auto matching_turnover_data.jl \
    REPO SWEEP FIGURE_DATA REPORTING_DATA OUTPUT
REPORTING_DATA supplies the approved late window and interval level.
"""

using JLD2
using DataFrames
using Statistics: mean
using SHA: sha256
using Dates: now, UTC

length(ARGS) == 5 || error("expected REPO SWEEP FIGURE_DATA REPORTING_DATA OUTPUT")
repo, root, figure_path, reporting_path, output = ARGS
include(joinpath(repo, "scripts/net_output_data.jl"))
include(joinpath(repo, "scripts/sweep/sweep_results.jl"))
include(joinpath(repo, "scripts/paper/information_sources.jl"))

hashfile(path) = bytes2hex(sha256(read(path)))
dependencies = ["scripts/net_output_data.jl", "scripts/net_output.jl",
    "scripts/sweep/sweep_results.jl", "scripts/paper/information_sources.jl",
    "scripts/monte_carlo.jl", "scripts/figure_style.jl"]
isempty(read(`git -C $repo diff HEAD -- $dependencies`, String)) ||
    error("analysis dependencies have uncommitted changes")
manifest = load(joinpath(root, "manifest.jld2"))
figdata = load(figure_path)["figdata"]
reporting = load(reporting_path)["metadata"]
manifest["manifest_hash"] == figdata["meta"]["manifest_hash"] || error("manifest mismatch")
meta = manifest["meta"]
commit = String(meta[:git_commit])
periods = collect(1:Int(meta[:T]))
late = Int.(reporting["late_periods"])
late == collect((last(periods) - length(late) + 1):last(periods)) || error("invalid late window")
level = Float64(reporting["interval_level"])
cells = figdata["rho_eta_cells"]
rhos = sort(unique(c["rho"] for c in cells))
turnover = sort(unique(c["eta"] for c in cells))
length(cells) == length(rhos) * length(turnover) || error("incomplete turnover grid")
Set((c["rho"], c["eta"]) for c in cells) == Set(Iterators.product(rhos, turnover)) ||
    error("duplicated turnover coordinate")
regimes = Dict(c["rel"] => c for c in figdata["regime_cells"])
conditions = Dict{String,Any}[]
calibrations = NamedTuple[]

for cell in sort(cells; by=c -> (c["eta"], c["rho"]))
    rel = cell["rel"]
    condition = only(c for c in manifest["conditions"] if c[:result_reldir] == rel)
    path = joinpath(root, rel, "data.jld2")
    raw = load(path)
    raw["provenance"]["manifest_hash"] == manifest["manifest_hash"] || error("aggregate manifest differs")
    raw["provenance"]["git_commit"] == commit || error("simulation commit differs")
    raw["schema_version"] == manifest["schema_version"] || error("aggregate schema differs")
    raw["result_reldir"] == rel || error("aggregate path differs")
    result = SweepResult(rel, rel, raw["mdfs"], string_dict(raw["realized_config"]),
        Int.(raw["seeds"]), Int(raw["condition_index"]))
    seeds = Int.(get(condition, :seeds, meta[:seeds]))
    validate_result(result, condition, seeds, periods)
    seeds == cell["seeds"] == regimes[rel]["seeds"] || error("publication seed order differs")
    (result.cfg["rho"], result.cfg["eta"]) == (cell["rho"], cell["eta"]) || error("grid coordinate differs")
    string(result.cfg["learning_model"]) == "nn" || error("expected NN learning")
    append!(calibrations, NetOutputData.reconstruct_result!(
        result, root, repo, commit, manifest["manifest_hash"]))
    broker = [mean(df[late, :broker_holdout_rank]) for df in result.mdfs]
    principal = [mean(df[late, :agent_holdout_rank]) for df in result.mdfs]
    gap = broker .- principal
    all(isfinite, [broker; principal; gap]) || error("nonfinite ranking diagnostic")
    isapprox(gap, regimes[rel]["seed_values"]["rankgap"]; atol=1e-12, rtol=1e-10) ||
        error("ranking gaps disagree with publication input")
    channels = [InformationSources.late_channel_values(df, Int(result.cfg["N"]), length(late))
        for df in result.mdfs]
    summary = InformationSources.channel_output_shares(channels, Int(result.cfg["N"]); level)
    push!(conditions, Dict("rel" => rel, "rho" => cell["rho"], "eta" => cell["eta"],
        "delta" => result.cfg["delta"], "N" => result.cfg["N"], "seeds" => seeds,
        "broker_ranks" => broker, "principal_ranks" => principal, "rank_gaps" => gap,
        "channel_values" => channels, "broker_shares" => [v.broker / v.total for v in channels],
        "broker_summary" => summary.broker, "config" => result.cfg,
        "source_path" => path, "source_sha256" => hashfile(path)))
    println("Verified ", rel, "; rho=", cell["rho"], ", eta=", cell["eta"], ", seeds=", length(seeds))
    flush(stdout)
end

delta = only(unique(c["delta"] for c in conditions))
metadata = Dict("sweep_root" => root, "manifest_hash" => manifest["manifest_hash"],
    "simulation_commit" => commit, "analysis_dependencies_commit" => strip(read(`git -C $repo rev-parse HEAD`, String)),
    "analysis_dependency_hashes" => Dict(p => hashfile(joinpath(repo, p)) for p in dependencies),
    "analysis_script_sha256" => hashfile(@__FILE__), "analysis_script" => read(@__FILE__, String),
    "figure_input_sha256" => hashfile(figure_path), "reporting_input_sha256" => hashfile(reporting_path),
    "figure_analysis_commit" => figdata["meta"]["analysis_git_commit"],
    "created_utc" => string(now(UTC)), "julia_version" => string(VERSION),
    "late_periods" => late, "interval_level" => level, "rho" => rhos, "turnover" => turnover,
    "difficulty" => delta, "regime_count" => length(conditions),
    "share_definition" => "mean_of_seed_level_late_net_output_shares",
    "ranking_gap_definition" => "broker_minus_principal_within_seed",
    "calibration_source_hashes" => NetOutputData.CALIBRATION_HASHES, "calibrations" => calibrations,
    "monte_carlo_sha256" => hashfile(joinpath(repo, "scripts/monte_carlo.jl")))
isfile(output) && error("refusing to overwrite existing extraction")
mkpath(dirname(output))
jldsave(output; metadata, conditions)
println("Saved ", output)
