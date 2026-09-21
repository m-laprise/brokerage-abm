"""Reconstruct net output from saved relationships and original calibrated costs."""
module NetOutputData

using DataFrames: DataFrame, eachrow, nrow, select!, Not
using JLD2
using SHA: sha256

include(joinpath(@__DIR__, "net_output.jl"))

const CALIBRATION_FILES = (
    "src/types.jl", "src/parameters.jl", "src/matching_function.jl", "src/initialization.jl"
)
const CALIBRATION_MODULES = Dict{String,Module}()
const CALIBRATION_HASHES = Dict{String,Dict{String,String}}()
const CALIBRATIONS = Dict{Tuple,NamedTuple}()
const OLD_METRICS = (
    :self_net_output_per_requested_position,
    :broker_net_output_per_requested_position,
    :net_output_per_requested_position,
)
const OUTPUT_COLUMNS = (
    :gross_match_output, :total_search_cost, :total_broker_fees, :net_output_per_principal
)

"""Load only the original types, defaults, matching environment, and initialization."""
function calibration_module(repo, commit)
    return get!(CALIBRATION_MODULES, commit) do
        mod = Module(gensym(:RecordedCalibration))
        Core.eval(mod, :(using Random, StableRNGs, Graphs, LinearAlgebra, Statistics))
        hashes = Dict{String,String}()
        for path in CALIBRATION_FILES
            source = read(`git -C $repo show $commit:$path`, String)
            hashes[path] = bytes2hex(sha256(source))
            Base.include_string(mod, source, "$commit:$path")
        end
        CALIBRATION_HASHES[commit] = hashes
        mod
    end
end

"""Recover original seed-specific fees without fitting learners or advancing periods."""
function recover_calibration(repo, commit, cfg, seed)
    mod = calibration_module(repo, commit)
    supplied = Dict(
        Symbol(k) => v for (k, v) in cfg if Symbol(k) in fieldnames(mod.ModelParams)
    )
    for key in (:learning_model, :ridge_broker_variant, :broker_service)
        haskey(supplied, key) && (supplied[key] = Symbol(supplied[key]))
    end
    supplied[:seed] = seed
    p = Base.invokelatest(mod.default_params; supplied...)
    key = (
        commit,
        seed,
        p.N,
        p.d,
        p.s,
        p.sigma_x,
        p.sigma_eps,
        p.rho,
        p.delta,
        p.search_cost_rate,
    )
    calibration = get!(CALIBRATIONS, key) do
        rng = Base.invokelatest(mod.StableRNG, seed)
        dgp = Base.invokelatest(mod.generate_matching_dgp, p, rng)
        cal = Base.invokelatest(mod.calibrate, dgp.env, dgp.agent_types, p, rng)
        (;
            q_cal=cal.q_cal,
            c_s=cal.c_s,
            phi=cal.phi,
            quality_weight=dgp.env.quality_weight,
            interaction_weight=dgp.env.interaction_weight,
            signal_shift=dgp.env.signal_shift,
        )
    end
    for name in (:quality_weight, :interaction_weight, :signal_shift)
        haskey(cfg, string(name)) || error("missing saved calibration check: $name")
        isapprox(
            getproperty(calibration, name), cfg[string(name)]; atol=1e-12, rtol=1e-10
        ) || error("original initialization differs: seed=$seed, $name")
    end
    for (name, value) in (
        ("q_cal", calibration.q_cal),
        ("search_cost", calibration.c_s),
        ("broker_fee", calibration.phi),
    )
        if haskey(cfg, name)
            isapprox(cfg[name], value; atol=1e-12, rtol=1e-10) ||
                error("saved $name differs")
        end
    end
    return calibration
end

"""A zero-match channel contributes zero, even when its saved mean is NaN."""
function relationship_output(count, average)
    count >= 0 || throw(ArgumentError("negative relationship count"))
    iszero(count) && return 0.0
    isfinite(average) || throw(ArgumentError("nonfinite mean for completed relationships"))
    return count * average
end

"""Reconstruct each period before averaging; reject invalid counts or inconsistent new data."""
function reconstruct_frame!(df::DataFrame, N, c_s, phi)
    values = [Float64[] for _ in OUTPUT_COLUMNS]
    for row in eachrow(df)
        0 <= row.outsourced_slots <= row.total_demand || error("invalid requested counts")
        placements = row.access_count + row.assessment_count
        0 <= placements <= row.outsourced_slots || error("invalid broker placement count")
        gross =
            relationship_output(row.n_self_matches, row.q_self_mean) +
            relationship_output(row.n_broker_matches, row.q_broker_mean)
        accounting = net_output_accounting(
            gross, row.total_demand - row.outsourced_slots, placements, N, c_s, phi
        )
        for (column, name) in zip(values, OUTPUT_COLUMNS)
            push!(column, getproperty(accounting, name))
        end
    end
    for (name, column) in zip(OUTPUT_COLUMNS, values)
        if name in propertynames(df)
            all(isapprox.(df[!, name], column; atol=1e-12, rtol=1e-10)) ||
                error("recorded $name disagrees with reconstructed accounting")
        end
        df[!, name] = column
    end
    select!(df, Not(intersect(collect(OLD_METRICS), propertynames(df))))
    return df
end

"""Reconstruct a validated result using each seed's own saved configuration."""
function reconstruct_result!(result, root, repo, commit, manifest_hash)
    metadata = NamedTuple[]
    for (i, seed) in enumerate(result.seeds)
        path = joinpath(root, result.rel, "seed_$seed.jld2")
        cfg = JLD2.jldopen(path, "r") do file
            file["manifest_hash"] == manifest_hash || error("seed manifest mismatch: $path")
            file["git_commit"] == commit || error("seed source mismatch: $path")
            file["seed"] == seed || error("seed mismatch: $path")
            string(file["julia_version"]) == string(VERSION) || error(
                "calibration replay requires recorded Julia $(file["julia_version"])",
            )
            file["config"]
        end
        cal = recover_calibration(repo, commit, cfg, seed)
        reconstruct_frame!(result.mdfs[i], Int(cfg["N"]), cal.c_s, cal.phi)
        push!(
            metadata,
            (;
                result=result.rel,
                seed,
                N=Int(cfg["N"]),
                q_cal=cal.q_cal,
                c_s=cal.c_s,
                phi=cal.phi,
            ),
        )
    end
    return metadata
end

"""Add the approved measure to an in-memory sweep; never overwrite simulation files."""
function reconstruct_dataset!(dataset, root, repo; results=dataset.results)
    commit = String(dataset.meta[:git_commit])
    metadata = reduce(
        vcat,
        (
            reconstruct_result!(result, root, repo, commit, dataset.manifest_hash) for
            result in results
        ),
    )
    return Dict(
        "simulation_commit" => commit,
        "calibration_source_hashes" => CALIBRATION_HASHES[commit],
        "calibrations" => metadata,
        "accounting_sha256" =>
            bytes2hex(sha256(read(joinpath(@__DIR__, "net_output.jl")))),
        "reconstruction_sha256" => bytes2hex(sha256(read(@__FILE__))),
    )
end

end
