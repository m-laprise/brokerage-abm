"""
    manifest.jl <screen|confirm|combined>

Create the immutable task manifest for one neural-network calibration stage.
The reporting workflow requires a clean committed worktree. Set
`BROKERAGE_ABM_ALLOW_DIRTY=1` only for local workflow tests.
"""

using Dates: Dates
using JLD2
using SHA: sha256

include(joinpath(@__DIR__, "calibration_config.jl"))

const NNCAL_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

nncal_git_commit() = strip(read(`git -C $NNCAL_REPO_ROOT rev-parse HEAD`, String))
function nncal_git_dirty()
    !isempty(strip(read(`git -C $NNCAL_REPO_ROOT status --porcelain`, String)))
end
nncal_file_hash(path) = isfile(path) ? bytes2hex(sha256(read(path))) : "absent"

function nncal_write_manifest_tsv(path, entries)
    open(path, "w") do io
        println(
            io,
            join(
                (
                    "task_id",
                    "config_id",
                    "seed",
                    "agent_eta_lr",
                    "agent_initial_steps",
                    "agent_recurrent_steps",
                    "broker_eta_lr",
                    "broker_initial_steps",
                    "broker_recurrent_steps",
                    "agent_scan",
                    "broker_scan",
                ),
                '\t',
            ),
        )
        for entry in entries
            config = entry[:config]
            println(
                io,
                join(
                    (
                        entry[:index],
                        entry[:config_id],
                        entry[:seed],
                        config[:agent_eta_lr],
                        config[:agent_initial_steps],
                        config[:agent_recurrent_steps],
                        config[:broker_eta_lr],
                        config[:broker_initial_steps],
                        config[:broker_recurrent_steps],
                        config[:agent_scan],
                        config[:broker_scan],
                    ),
                    '\t',
                ),
            )
        end
    end
    return nothing
end

function main()
    length(ARGS) == 1 || error("usage: manifest.jl <screen|confirm|combined>")
    stage = Symbol(only(ARGS))
    stage in (:screen, :confirm, :combined) || error("invalid stage: $stage")

    dirty = nncal_git_dirty()
    dirty &&
        get(ENV, "BROKERAGE_ABM_ALLOW_DIRTY", "0") != "1" &&
        error(
            "refusing to create a calibration manifest from a dirty worktree; " *
            "commit the intended code first",
        )

    configs = nncal_stage_configs(stage)
    seeds = nncal_stage_seeds(stage)
    entries = nncal_build_entries(configs, seeds)
    stage_dir = nncal_stage_dir(stage)
    isfile(joinpath(stage_dir, "manifest.jld2")) &&
        error("calibration manifest already exists: $stage_dir")
    mkpath(stage_dir)

    commit = nncal_git_commit()
    package_hash = nncal_file_hash(joinpath(NNCAL_REPO_ROOT, "Manifest.toml"))
    meta = Dict{Symbol,Any}(
        :stage => string(stage),
        :date => string(Dates.today()),
        :git_commit => commit,
        :git_dirty => dirty,
        :julia_version => string(VERSION),
        :pkg_manifest_hash => package_hash,
        :schema_version => NNCAL_SCHEMA_VERSION,
        :N => NNCAL_N,
        :T => NNCAL_T,
        :early_periods => collect(NNCAL_EARLY_PERIODS),
        :late_periods => collect(NNCAL_LATE_PERIODS),
        :seeds => seeds,
        :learning_rates => NNCAL_LEARNING_RATES,
        :recurrent_steps => NNCAL_RECURRENT_STEPS,
        :initial_to_recurrent_ratio => NNCAL_INITIAL_TO_RECURRENT_RATIO,
        :practical_tolerance => NNCAL_PRACTICAL_TOLERANCE,
        :n_configs => length(configs),
        :n_runs => length(entries),
    )

    tsv_path = joinpath(stage_dir, "manifest.tsv")
    nncal_write_manifest_tsv(tsv_path, entries)
    manifest_hash = nncal_file_hash(tsv_path)
    write(joinpath(stage_dir, "manifest.sha256"), manifest_hash * "\n")
    write(
        joinpath(stage_dir, "counts.env"),
        "NCONFIGS=$(length(configs))\nNRUNS=$(length(entries))\n",
    )

    provenance = Dict{Symbol,Any}(
        :git_commit => commit,
        :julia_version => string(VERSION),
        :pkg_manifest_hash => package_hash,
        :manifest_hash => manifest_hash,
        :schema_version => NNCAL_SCHEMA_VERSION,
    )
    jldsave(
        joinpath(stage_dir, "manifest.jld2");
        configs=configs,
        entries=entries,
        meta=meta,
        provenance=provenance,
        manifest_hash=manifest_hash,
    )

    println("calibration stage: $stage")
    println("directory: $stage_dir")
    println("git commit: $commit$(dirty ? " (dirty workflow test)" : "")")
    println("configs: $(length(configs))")
    println("NRUNS=$(length(entries))")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
