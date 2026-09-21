"""
Extract baseline centrality and principal degree from the original service sweep.

No simulations are run. Periods before the first network measurement and cached
copies between centrality measurements are excluded. Degree keeps every recorded
period, including broker edges. Both measures are checked against retained
seed-level late means and stored together in the existing trajectory input.

Usage: julia --project --threads=auto scripts/assessment_access/centrality_data.jl \
    <local sweep root>
"""

using JLD2
using SHA: sha256
using Statistics: mean

include(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const CENTRALITY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const CENTRALITY_MODES = ("full", "assessment_only", "access_only")
const CENTRALITY_OUTPUT = joinpath(
    CENTRALITY_ROOT, "output", "assessment_access", "centrality_trajectories.jld2"
)

"""Validate measurement cadence and recover one seed's observed centrality."""
function measured_centrality(df, retained_late_value; interval=20, horizon=500)
    collect(df.period) == collect(1:horizon) || error("incomplete period coverage")
    values = Float64.(df.betweenness)
    all(v -> isfinite(v) && 0 <= v <= 1, values) || error("invalid betweenness")
    periods = collect(interval:interval:horizon)
    all(t -> values[t] == values[interval * (t ÷ interval)], interval:horizon) ||
        error("centrality changes between recorded measurement periods")
    isapprox(mean(values[(horizon - 19):horizon]), retained_late_value; atol=1e-12) ||
        error("raw centrality does not match the retained seed-level late value")
    return values[periods]
end

"""Recover recorded degree without removing broker edges or changing its timing."""
function recorded_degree(df, retained_late_value; horizon, late_width, population)
    collect(df.period) == collect(1:horizon) || error("incomplete degree period coverage")
    values = Float64.(df.mean_degree)
    all(v -> isfinite(v) && 0 <= v <= population, values) || error("invalid mean degree")
    isapprox(
        mean(values[(horizon - late_width + 1):horizon]), retained_late_value; atol=1e-12
    ) || error("raw degree does not match the retained seed-level late value")
    return values
end

"""Validate three original aggregates and save a compact seed-level input."""
function extract_centrality(sweep_root; output_path=CENTRALITY_OUTPUT)
    retained = JLD2.load(
        joinpath(CENTRALITY_ROOT, "output", "assessment_access", "figure_data.jld2")
    )
    retained["analysis_source_clean"] === true || error("retained analysis was dirty")
    retained["late_width"] == 20 || error("unexpected late window")
    manifest_path = joinpath(sweep_root, "manifest.jld2")
    manifest = JLD2.load(manifest_path)
    manifest["manifest_hash"] == retained["manifest_hash"] || error("manifest mismatch")
    manifest["schema_version"] == retained["schema_version"] || error("schema mismatch")
    meta = manifest["meta"]
    meta[:git_dirty] === false || error("simulation source was dirty")
    meta[:T] == 500 || error("unexpected simulation horizon")
    # The aggregate omits default-valued measurement cadence. Verify that default
    # in the recorded simulation commit, not in the current working tree.
    parameter_ref = "$(meta[:git_commit]):src/parameters.jl"
    parameter_source = read(`git -C $CENTRALITY_ROOT show $parameter_ref`, String)
    occursin(r":network_measure_interval\s*=>\s*20\b", parameter_source) ||
        error("unexpected measurement default in the simulation source")
    seeds = Int.(meta[:baseline_seeds])
    periods = collect(20:20:500)
    degree_periods = collect(1:Int(meta[:T]))
    values = Dict{String,Matrix{Float64}}()
    degree_values = Dict{String,Matrix{Float64}}()
    source_hashes = Dict("manifest.jld2" => bytes2hex(sha256(read(manifest_path))))
    configs = Dict{String,Dict{String,Any}}()
    for mode in CENTRALITY_MODES
        rel = "broker_services/$mode/baseline"
        condition = only(filter(c -> c[:result_reldir] == rel, manifest["conditions"]))
        path = joinpath(sweep_root, rel, "data.jld2")
        data = JLD2.load(path)
        provenance = data["provenance"]
        provenance["manifest_hash"] == manifest["manifest_hash"] ||
            error("aggregate manifest mismatch")
        provenance["git_commit"] == meta[:git_commit] || error("aggregate commit mismatch")
        provenance["schema_version"] ==
        data["schema_version"] ==
        manifest["schema_version"] || error("aggregate schema mismatch")
        result = SweepResult(
            data["result_reldir"],
            data["result_reldir"],
            data["mdfs"],
            string_dict(data["realized_config"]),
            Int.(data["seeds"]),
            Int(data["condition_index"]),
        )
        validate_result(result, condition, seeds, collect(1:500))
        get(result.cfg, "network_measure_interval", 20) == 20 ||
            error("unexpected measurement cadence")
        (result.cfg["rho"], result.cfg["delta"], result.cfg["eta"], result.cfg["N"]) ==
        (0.5, 0.5, 0.02, 1000) || error("not the reporting baseline")
        late = Dict{Int,Float64}()
        degree_late = Dict{Int,Float64}()
        for row in retained["seed_rows"]
            if (String(row[1]), row[2], row[3], String(row[5])) ==
                (mode, 0.5, 0.02, "betweenness")
                haskey(late, Int(row[4])) && error("duplicate retained seed")
                late[Int(row[4])] = Float64(row[6])
            end
            if (String(row[1]), row[2], row[3], String(row[5])) ==
                (mode, result.cfg["rho"], result.cfg["eta"], "mean_degree")
                haskey(degree_late, Int(row[4])) && error("duplicate retained degree seed")
                degree_late[Int(row[4])] = Float64(row[6])
            end
        end
        sort!(collect(keys(late))) == seeds || error("retained seed set mismatch")
        sort!(collect(keys(degree_late))) == seeds ||
            error("retained degree seed set mismatch")
        values[mode] = reduce(
            hcat,
            [measured_centrality(df, late[seed]) for (df, seed) in zip(result.mdfs, seeds)],
        )
        degree_values[mode] = reduce(
            hcat,
            [
                recorded_degree(
                    df,
                    degree_late[seed];
                    horizon=meta[:T],
                    late_width=retained["late_width"],
                    population=result.cfg["N"],
                ) for (df, seed) in zip(result.mdfs, seeds)
            ],
        )
        source_hashes["$rel/data.jld2"] = bytes2hex(sha256(read(path)))
        configs[mode] = result.cfg
    end
    extraction = reporting_git_provenance(
        CENTRALITY_ROOT;
        sources=(
            @__FILE__,
            "scripts/sweep/sweep_results.jl",
        ),
        require_clean=false,
    )
    mkpath(dirname(output_path))
    jldsave(
        output_path;
        values,
        periods,
        degree_values,
        degree_periods,
        seeds,
        configs,
        source_hashes,
        rho=0.5,
        delta=0.5,
        eta=0.02,
        network_measure_interval=20,
        parameter_source_sha256=bytes2hex(sha256(parameter_source)),
        manifest_hash=manifest["manifest_hash"],
        sweep_git_commit=meta[:git_commit],
        sweep_source_clean=(!meta[:git_dirty]),
        source_root=retained["source_root"],
        retained_analysis_git_commit=retained["analysis_git_commit"],
        extraction_git_commit=extraction.commit,
        extraction_source_clean=extraction.source_clean,
        extraction_source_status=extraction.source_status,
        extraction_script_sha256=bytes2hex(sha256(read(@__FILE__))),
        sweep_reader_sha256=bytes2hex(
            sha256(read(joinpath(@__DIR__, "..", "sweep", "sweep_results.jl")))
        ),
    )
    println(
        "Wrote $output_path (centrality and degree for $(length(seeds)) seeds per broker)"
    )
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 1 || error("expected the local sweep root")
    extract_centrality(only(ARGS))
end
