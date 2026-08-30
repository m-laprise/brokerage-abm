"""
Validated access to a completed sweep.

The manifest distinguishes analysis-grid references from effective model
realizations. Paper and diagnostic summaries must operate on `results`, which
contains each effective realization exactly once. Grid-specific analyses use
`grid_result` to resolve a grid coordinate to its canonical result.
"""

using DataFrames: DataFrame
using JLD2: jldopen

struct SweepResult
    rel::String
    result_rel::String
    mdfs::Vector{DataFrame}
    cfg::Dict{String,Any}
    seeds::Vector{Int}
    condition_index::Int
end

struct SweepDataset
    grid_cells::Vector{Dict{Symbol,Any}}
    results::Vector{SweepResult}
    result_by_rel::Dict{String,SweepResult}
    grid_by_rel::Dict{String,Dict{Symbol,Any}}
    meta::Dict{Symbol,Any}
    manifest_hash::String
    schema_version::Int
end

string_dict(values) = Dict{String,Any}(string(key) => value for (key, value) in values)

function validate_result(result::SweepResult, condition, expected_seeds, expected_periods)
    result.rel == condition[:result_reldir] ||
        error("result path mismatch for condition $(condition[:condition_index])")
    result.condition_index == condition[:condition_index] ||
        error("condition index mismatch for $(result.rel)")
    result.seeds == expected_seeds || error(
        "incomplete seed set for $(result.rel): got $(result.seeds), expected $expected_seeds",
    )
    length(result.mdfs) == length(expected_seeds) ||
        error("metrics/seed count mismatch for $(result.rel)")
    all(df -> collect(df.period) == expected_periods, result.mdfs) ||
        error("period coverage mismatch for $(result.rel)")
    for (key, value) in condition[:resolved_params]
        name = string(key)
        haskey(result.cfg, name) || error("realized config for $(result.rel) lacks $name")
        result.cfg[name] == value || error(
            "realized config mismatch for $(result.rel): $name=$(result.cfg[name]), expected $value",
        )
    end
    return result
end

"""Load all effective results and fail if the sweep is incomplete or internally mixed."""
function load_sweep_dataset(root::AbstractString)
    manifest_path = joinpath(root, "manifest.jld2")
    isfile(manifest_path) || error("sweep manifest not found: $manifest_path")
    grid_cells, conditions, meta, manifest_hash, schema_version =
        jldopen(manifest_path, "r") do file
            (
                file["cells"],
                file["conditions"],
                file["meta"],
                file["manifest_hash"],
                file["schema_version"],
            )
        end

    expected_periods = collect(1:Int(meta[:T]))
    length(conditions) == meta[:n_conditions] || error(
        "manifest condition count mismatch: $(length(conditions)) != $(meta[:n_conditions])",
    )
    length(grid_cells) == meta[:n_grid_cells] || error(
        "manifest grid-coordinate count mismatch: $(length(grid_cells)) != $(meta[:n_grid_cells])",
    )

    results = SweepResult[]
    for condition in sort(conditions; by=c -> c[:condition_index])
        # Schema 5 stored one common seed set in manifest metadata. Schema 7
        # stores seeds on each condition so the baseline may use a larger set.
        expected_seeds = Int.(get(condition, :seeds, meta[:seeds]))
        rel = condition[:result_reldir]
        path = joinpath(root, rel, "data.jld2")
        isfile(path) || error("missing aggregate for effective realization: $rel")
        result = jldopen(path, "r") do file
            haskey(file, "realized_config") ||
                error("aggregate lacks realized_config: $path")
            haskey(file, "provenance") || error("aggregate lacks provenance: $path")
            provenance = file["provenance"]
            get(provenance, "manifest_hash", nothing) == manifest_hash ||
                error("manifest provenance mismatch: $path")
            get(provenance, "schema_version", nothing) == schema_version ||
                error("schema provenance mismatch: $path")
            file["schema_version"] == schema_version ||
                error("aggregate schema mismatch: $path")
            file["result_reldir"] == rel ||
                error("aggregate result path mismatch: $path")
            SweepResult(
                rel,
                rel,
                file["mdfs"],
                string_dict(file["realized_config"]),
                Int.(file["seeds"]),
                Int(file["condition_index"]),
            )
        end
        push!(results, validate_result(result, condition, expected_seeds, expected_periods))
    end

    result_by_rel = Dict(result.rel => result for result in results)
    length(result_by_rel) == length(results) || error("duplicate effective result paths")
    grid_by_rel = Dict(cell[:reldir] => cell for cell in grid_cells)
    length(grid_by_rel) == length(grid_cells) || error("duplicate grid-coordinate paths")
    all(haskey(result_by_rel, cell[:result_reldir]) for cell in grid_cells) ||
        error("grid coordinate references an unknown effective result")

    return SweepDataset(
        grid_cells,
        results,
        result_by_rel,
        grid_by_rel,
        meta,
        String(manifest_hash),
        Int(schema_version),
    )
end

"""Resolve one analysis-grid coordinate to its effective result and grid metadata."""
function grid_result(dataset::SweepDataset, rel::AbstractString)
    haskey(dataset.grid_by_rel, rel) || error("unknown grid coordinate: $rel")
    grid = dataset.grid_by_rel[rel]
    result = dataset.result_by_rel[grid[:result_reldir]]
    cfg = copy(result.cfg)
    for (key, value) in grid[:resolved_params]
        cfg[string(key)] = value
    end
    return SweepResult(
        String(rel), result.rel, result.mdfs, cfg, result.seeds, result.condition_index
    )
end

"""Deduplicate grid-resolved results by effective realization identity."""
function unique_effective_results(results)
    seen = Set{String}()
    unique_results = SweepResult[]
    for result in results
        result.result_rel in seen && continue
        push!(seen, result.result_rel)
        push!(unique_results, result)
    end
    return unique_results
end
