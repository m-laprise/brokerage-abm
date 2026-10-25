"""Resolve reporting settings from manifests, recorded defaults, and retained metadata."""
module ReportingSettings

using JLD2: load
using SHA: sha256
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RETAINED_PATH = joinpath(ROOT, "output", "reporting", "settings.toml")
const DEFAULTS_CACHE = Dict{Tuple{String,String},Module}()
const JSON_CODECS = Dict{String,Module}()

textkeys(values) = Dict{String,Any}(string(k) => plain(v) for (k, v) in pairs(values))
plain(x::Symbol) = string(x)
plain(x::AbstractDict) = textkeys(x)
plain(x::NamedTuple) = textkeys(x)
plain(x::Tuple) = plain.(collect(x))
plain(x::AbstractArray) = plain.(x)
plain(x) = x
filehash(path) = bytes2hex(sha256(read(path)))

"""Read a manifest and verify its stored content hash."""
function manifest(root)
    path = isdir(root) ? joinpath(root, "manifest.jld2") : root
    data = load(path)
    json = joinpath(dirname(path), "manifest.json")
    signature = joinpath(dirname(path), "manifest.sha256")
    filehash(json) == strip(read(signature, String)) == data["manifest_hash"] ||
        error("manifest hash mismatch: $path")
    commit = String(data["meta"][:git_commit])
    codec = get!(JSON_CODECS, commit) do
        source = read(`git -C $ROOT show $commit:scripts/sweep/sweep_config.jl`, String)
        mod = Module(gensym(:ManifestJSON))
        overrides = [
            key => nothing for key in keys(ENV) if startswith(key, "BROKERAGE_ABM_")
        ]
        withenv(overrides...) do
            Base.include_string(mod, source, "scripts/sweep/sweep_config.jl")
        end
        mod
    end
    document = Dict(
        Symbol(key) => data[key] for
        key in ("meta", "spec", "cells", "conditions", "entries", "plot_jobs")
    )
    encoded = Base.invokelatest(codec.to_json_string, document)
    check = "import json,sys; assert json.load(open(sys.argv[1])) == json.load(sys.stdin)"
    run(pipeline(Cmd(["python3", "-c", check, json]); stdin=IOBuffer(encoded)))
    length(data["conditions"]) == data["meta"][:n_conditions] ||
        error("manifest condition count mismatch")
    expected = Set(
        (c[:result_reldir], seed) for c in data["conditions"] for seed in c[:seeds]
    )
    entries = [(e[:reldir], e[:seed]) for e in data["entries"]]
    length(entries) == length(unique(entries)) == data["meta"][:n_runs] ||
        error("manifest run count mismatch")
    Set(entries) == expected || error("manifest seed allocation mismatch")
    return data
end

"""Resolve the canonical baseline from declared parameters."""
function baseline_condition(data; overrides=Dict{Symbol,Any}())
    baseline = merge(Dict(data["spec"][:baseline]), overrides)
    matches = filter(data["conditions"]) do condition
        params = merge(data["meta"], condition[:resolved_params])
        all(
            haskey(params, key) && plain(params[key]) == plain(value) for
            (key, value) in baseline
        )
    end
    length(matches) == 1 || error("manifest does not identify one baseline")
    return only(matches)
end

baseline_reldir(root) = String(baseline_condition(manifest(root))[:result_reldir])

"""Recover the common network measurement interval from the recorded run settings."""
function network_interval(root)
    data = manifest(root)
    commit = String(data["meta"][:git_commit])
    intervals = [
        resolved_parameters(ROOT, commit, c[:resolved_params])["network_measure_interval"]
        for c in data["conditions"]
    ]
    return only(unique(intervals))
end

"""Load the analyst's selected defaults at the recorded simulation commit."""
function default_module(repo, commit)
    occursin(r"^[0-9a-f]{40}$", commit) || error("expected an immutable simulation commit")
    return get!(DEFAULTS_CACHE, (abspath(repo), commit)) do
        mod = Module(gensym(:RecordedDefaults))
        Core.eval(mod, :(using Random: AbstractRNG))
        Core.eval(mod, :(using StableRNGs: StableRNG))
        Core.eval(mod, :(using Graphs: SimpleGraph))
        for path in ("src/types.jl", "src/parameters.jl")
            source = read(`git -C $repo show $commit:$path`, String)
            Base.include_string(mod, source, "$commit:$path")
        end
        mod
    end
end

"""Complete a recorded configuration without initializing or running the model."""
function resolved_parameters(repo, commit, config)
    mod = default_module(repo, commit)
    fields = fieldnames(mod.ModelParams)
    supplied = Dict(Symbol(k) => v for (k, v) in pairs(config) if Symbol(k) in fields)
    for key in (:learning_model, :ridge_broker_variant, :broker_service)
        haskey(supplied, key) && (supplied[key] = Symbol(supplied[key]))
    end
    params = Base.invokelatest(mod.default_params; supplied...)
    return Dict(string(key) => plain(getfield(params, key)) for key in fields)
end

"""Retain the full baseline and exact seed plan for a verified sweep."""
function sweep_metadata(repo, data; baseline_overrides=Dict{Symbol,Any}())
    meta = data["meta"]
    baseline = baseline_condition(data; overrides=baseline_overrides)
    config = Dict{Symbol,Any}(baseline[:resolved_params])
    config[:T] = meta[:T]
    for (recorded, field) in (
        :learning_model => :learning_model,
        :nn_eta_lr_agent => :eta_lr_agent,
        :nn_eta_lr_broker => :eta_lr_broker,
        :nn_E_init_agent => :E_init_agent,
        :nn_E_init_broker => :E_init_broker,
        :nn_train_steps_agent => :train_steps_agent,
        :nn_train_steps_broker => :train_steps_broker,
        :ridge_lambda_agent => :ridge_lambda_agent,
        :ridge_lambda_broker => :ridge_lambda_broker,
        :ridge_broker_variant => :ridge_broker_variant,
    )
        haskey(meta, recorded) && (config[field] = meta[recorded])
    end
    commit = String(meta[:git_commit])
    params = resolved_parameters(repo, commit, config)
    sources = Dict(
        path => bytes2hex(sha256(read(`git -C $repo show $commit:$path`))) for
        path in ("src/types.jl", "src/parameters.jl", "scripts/sweep/sweep_config.jl")
    )
    return Dict{String,Any}(
        "manifest_hash" => data["manifest_hash"],
        "simulation_commit" => commit,
        "baseline_reldir" => String(baseline[:result_reldir]),
        "baseline" => params,
        "horizon" => Int(meta[:T]),
        "network_measure_interval" => params["network_measure_interval"],
        "condition_seeds" =>
            Dict(String(c[:result_reldir]) => Int.(c[:seeds]) for c in data["conditions"]),
        "default_sources" => sources,
    )
end

"""Read settings embedded in new data or verified alongside an older artifact."""
function retained(path, data; metadata_path=RETAINED_PATH)
    meta = get(data, "meta", data)
    if haskey(meta, "reporting_settings")
        return meta["reporting_settings"]
    end
    records = TOML.parsefile(metadata_path)
    relative = relpath(abspath(path), ROOT)
    entry = records["artifacts"][relative]
    filehash(path) == entry["sha256"] || error("retained settings are stale for $relative")
    sweep = records["sweeps"][entry["manifest_hash"]]
    plan = records["seed_plan"]
    seeds = Dict(rel => plan["sets"][index] for (rel, index) in plan["conditions"])
    return merge(sweep, records["analysis"], Dict("condition_seeds" => seeds))
end

"""Additional file dependencies for older retained inputs."""
function retained_dependencies(data)
    meta = get(data, "meta", data)
    return haskey(meta, "reporting_settings") ? String[] : [RETAINED_PATH]
end

"""Check retained figure conditions and seed identities against their manifest."""
function validate_figure(data, settings)
    expected = settings["condition_seeds"]
    cells = data["regime_cells"]
    length(cells) == length(expected) ||
        error("figure condition count differs from manifest")
    Set(c["rel"] for c in cells) == Set(keys(expected)) ||
        error("figure conditions differ from manifest")
    all(c["seeds"] == expected[c["rel"]] for c in cells) ||
        error("figure seeds differ from manifest")
    data["baseline_seeds"] == expected[settings["baseline_reldir"]] ||
        error("baseline seed mismatch")
    data["meta"]["baseline_n_seeds"] == length(data["baseline_seeds"]) ||
        error("baseline count metadata mismatch")
    data["period"] == collect(1:settings["horizon"]) ||
        error("figure period coverage mismatch")
    data["meta"]["manifest_hash"] == settings["manifest_hash"] ||
        error("figure manifest mismatch")
    data["meta"]["n_runs"] == sum(length, values(expected)) ||
        error("figure run count mismatch")
    counts = Dict(rel => length(seeds) for (rel, seeds) in expected)
    data["meta"]["condition_seed_counts"] == counts ||
        error("figure seed-count metadata mismatch")
    return settings
end

end
