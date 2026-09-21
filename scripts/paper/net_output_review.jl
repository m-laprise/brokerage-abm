"""
Review net output per principal from saved Ridge and assessment-access sweeps.

Extract on a compute node with the original Julia version:
    julia --project --threads=auto scripts/paper/net_output_review.jl extract \
        <input directory> <source repository> <review.jld2>
The input directory contains ridge.jld2, ridge_provenance.txt, access.jld2,
and access_provenance.txt. No simulation periods are run. Review artifacts retain
source hashes and are not publication-analysis inputs.

Render locally:
    julia --project --threads=auto scripts/paper/net_output_review.jl figures <review.jld2>
"""

using JLD2
using DataFrames: DataFrame
using SHA: sha256
using Statistics: mean

include(joinpath(@__DIR__, "..", "net_output_data.jl"))
include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))

read_provenance(path) = Dict(Tuple(split(line, '='; limit=2)) for line in readlines(path))
file_hash(path) = bytes2hex(sha256(read(path)))

"""Validate one saved condition and retain its reconstructed seed-level outcomes."""
function review_condition(root, condition, meta, manifest_hash, repo, late_width)
    rel = condition[:result_reldir]
    path = joinpath(root, rel, "data.jld2")
    raw = JLD2.load(path)
    raw["provenance"]["manifest_hash"] == manifest_hash || error("manifest differs: $path")
    raw["provenance"]["git_commit"] == meta[:git_commit] ||
        error("simulation source differs")
    raw["schema_version"] == raw["provenance"]["schema_version"] || error("schema differs")
    seeds = Int.(get(condition, :seeds, meta[:seeds]))
    result = SweepResult(
        rel,
        rel,
        raw["mdfs"],
        raw["realized_config"],
        Int.(raw["seeds"]),
        Int(raw["condition_index"]),
    )
    validate_result(result, condition, seeds, collect(1:Int(meta[:T])))
    calibration = NetOutputData.reconstruct_result!(
        result, root, repo, String(meta[:git_commit]), manifest_hash
    )
    late = (Int(meta[:T]) - late_width + 1):Int(meta[:T])
    outcomes = Dict(
        string(metric) => [mean(df[late, metric]) for df in result.mdfs] for
        metric in NetOutputData.OUTPUT_COLUMNS
    )
    return Dict(
        "rho" => result.cfg["rho"],
        "eta" => result.cfg["eta"],
        "delta" => result.cfg["delta"],
        "N" => result.cfg["N"],
        "seeds" => seeds,
        "late" => outcomes,
        "calibrations" => calibration,
        "series" => reduce(hcat, (df.net_output_per_principal for df in result.mdfs)),
        "source_path" => path,
        "source_sha256" => file_hash(path),
        "manifest_hash" => manifest_hash,
        "simulation_commit" => meta[:git_commit],
    )
end

"""Extract a clearly marked review archive without modifying retained publication inputs."""
function extract_review(input, repo, output)
    ridge = JLD2.load(joinpath(input, "ridge.jld2"))["figdata"]
    ridge_meta = read_provenance(joinpath(input, "ridge_provenance.txt"))
    access = JLD2.load(joinpath(input, "access.jld2"))
    access_meta = read_provenance(joinpath(input, "access_provenance.txt"))
    ridge_meta["analysis_source_clean"] == access_meta["analysis_source_clean"] == "true" ||
        error("retained source analysis is not clean")
    late_width = Int(ridge["meta"]["late_width"])
    late_width == access["late_width"] || error("late windows differ")
    parent = dirname(access_meta["sweep_root"])
    ridge_values = Dict{String,Any}()
    for model in ("pair", "size_matched", "single_principal", "additive")
        root = joinpath(parent, ridge_meta[model * "_sweep"])
        manifest = JLD2.load(joinpath(root, "manifest.jld2"))
        manifest["manifest_hash"] == ridge_meta[model * "_manifest"] ||
            error("Ridge manifest differs")
        values = Dict{String,Any}()
        for c in ridge["conditions"]
            candidates = filter(
                x -> x[:result_reldir] == c["result_reldir"], manifest["conditions"]
            )
            if isempty(candidates)
                grid = only(
                    filter(x -> x[:reldir] == c["result_reldir"], manifest["cells"])
                )
                candidates = filter(
                    x -> x[:result_reldir] == grid[:result_reldir], manifest["conditions"]
                )
            end
            condition = only(candidates)
            value = review_condition(
                root,
                condition,
                manifest["meta"],
                manifest["manifest_hash"],
                repo,
                late_width,
            )
            value["seeds"] == c["seeds"] || error("Ridge seed coverage differs")
            value["rho"] == c["rho"] || error("Ridge composition differs")
            value["rho"] == 1 ||
                value["delta"] == c["delta"] ||
                error("Ridge difficulty differs")
            values[c["result_reldir"]] = value
        end
        ridge_values[model] = values
        println("Reconstructed $model: $(length(values)) regimes")
        flush(stdout)
    end
    access_values = Dict{Tuple{String,Float64,Float64},Any}()
    for (root_key, hash_key) in (
        ("sweep_root", "manifest_hash"),
        ("supplement_sweep_root", "supplement_manifest_hash"),
    )
        root = access_meta[root_key]
        manifest = JLD2.load(joinpath(root, "manifest.jld2"))
        manifest["manifest_hash"] == access_meta[hash_key] ||
            error("access manifest differs")
        for condition in manifest["conditions"]
            cfg = condition[:resolved_params]
            key = (String(cfg[:broker_service]), Float64(cfg[:rho]), Float64(cfg[:eta]))
            haskey(access_values, key) && error("duplicate service condition")
            value = review_condition(
                root,
                condition,
                manifest["meta"],
                manifest["manifest_hash"],
                repo,
                late_width,
            )
            expected = sort([
                Int(r[4]) for r in access["seed_rows"] if
                (String(r[1]), Float64(r[2]), Float64(r[3])) == key &&
                String(r[5]) == "outsourcing_rate"
            ])
            value["seeds"] == expected || error("access seed coverage differs")
            access_values[key] = value
        end
        println("Reconstructed service conditions: $(length(access_values))")
        flush(stdout)
    end
    expected = Set(
        (String(r[1]), Float64(r[2]), Float64(r[3])) for r in access["seed_rows"]
    )
    Set(keys(access_values)) == expected || error("access regime coverage differs")
    meta = Dict(
        "review_only" => true,
        "analysis_source_clean" => false,
        "definition" => "net_output_per_principal",
        "late_width" => late_width,
        "interval_level" => access["interval_level"],
        "julia_version" => string(VERSION),
        "baseline_reldir" => ridge["meta"]["baseline_reldir"],
        "extractor_sha256" => file_hash(@__FILE__),
        "reconstruction_sha256" =>
            file_hash(joinpath(@__DIR__, "..", "net_output_data.jl")),
        "accounting_sha256" =>
            file_hash(joinpath(@__DIR__, "..", "net_output.jl")),
        "calibration_source_hashes" => NetOutputData.CALIBRATION_HASHES,
        "input_hashes" => Dict(
            name => file_hash(joinpath(input, name)) for name in (
                "ridge.jld2",
                "ridge_provenance.txt",
                "access.jld2",
                "access_provenance.txt",
            )
        ),
    )
    mkpath(dirname(abspath(output)))
    jldsave(output; meta, ridge=ridge_values, access=access_values)
    println("Saved review archive: $output")
end

if !isempty(ARGS) && first(ARGS) == "extract"
    length(ARGS) == 4 ||
        error("extract requires input directory, source repository, output")
    extract_review(ARGS[2:end]...)
elseif !isempty(ARGS) && first(ARGS) == "figures"
    length(ARGS) == 2 || error("figures requires review archive")
    include(joinpath(@__DIR__, "net_output_review_figures.jl"))
    render_net_output_review(ARGS[2])
elseif abspath(PROGRAM_FILE) == @__FILE__
    error("expected extract or figures")
end
