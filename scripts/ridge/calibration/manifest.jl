"""
    manifest.jl <screen|confirm>

Create an immutable manifest for one Ridge calibration stage. Confirmation
requires explicitly selected penalties and a concise selection rationale.
"""

using Dates: Dates
using JLD2: jldopen, jldsave

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "provenance.jl"))

module RIDGECALSweepDesign
include(joinpath(@__DIR__, "..", "..", "sweep", "sweep_config.jl"))
end

function ridgecal_write_manifest_tsv(path, entries)
    open(path, "w") do io
        println(io, "task_id\tconfig_id\tseed\tlambda_agent\tlambda_broker")
        for entry in entries
            config = entry[:config]
            println(
                io,
                join(
                    (
                        entry[:index],
                        entry[:config_id],
                        entry[:seed],
                        config[:lambda_agent],
                        config[:lambda_broker],
                    ),
                    '\t',
                ),
            )
        end
    end
    return nothing
end

function ridgecal_confirmation_inputs()
    lambda_agent = parse(
        Float64,
        get(ENV, "BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_AGENT") do
            error("BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_AGENT is required")
        end,
    )
    lambda_broker = parse(
        Float64,
        get(ENV, "BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_BROKER") do
            error("BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_BROKER is required")
        end,
    )
    selection_note = strip(
        get(ENV, "BROKERAGE_ABM_RIDGE_SELECTION_NOTE") do
            error("BROKERAGE_ABM_RIDGE_SELECTION_NOTE is required")
        end,
    )
    isempty(selection_note) && error("Ridge selection note cannot be empty")
    return lambda_agent, lambda_broker, selection_note
end

function ridgecal_screen_link(source)
    screen_dir = ridgecal_stage_dir(:screen)
    manifest_path = joinpath(screen_dir, "manifest.jld2")
    summary_path = joinpath(ridgecal_summary_dir(), "screen_summary.jld2")
    isfile(manifest_path) || error("screen manifest not found: $manifest_path")
    isfile(summary_path) || error("screen summary not found: $summary_path")
    provenance, manifest_hash = jldopen(manifest_path, "r") do file
        file["provenance"], file["manifest_hash"]
    end
    mismatches = ridgecal_provenance_mismatches(provenance, source)
    isempty(mismatches) || error(
        "confirmation source differs from screen: " * join(string.(mismatches), ", "),
    )
    return manifest_hash
end

function ridgecal_manifest_main()
    length(ARGS) == 1 || error("usage: manifest.jl <screen|confirm>")
    stage = Symbol(only(ARGS))
    stage in (:screen, :confirm) || error("invalid stage: $stage")
    RIDGECAL_BASELINE == RIDGECALSweepDesign.SWEEP_BASELINE || error(
        "Ridge calibration baseline does not match the reporting sweep baseline",
    )

    source = ridgecal_current_source_provenance()
    dirty = source[:git_dirty]
    dirty && get(ENV, "BROKERAGE_ABM_ALLOW_DIRTY", "0") != "1" && error(
        "refusing to create a Ridge calibration manifest from a dirty worktree",
    )

    selection_note = ""
    screen_manifest_hash = ""
    configs = if stage == :screen
        ridgecal_screen_configs()
    else
        lambda_agent, lambda_broker, selection_note = ridgecal_confirmation_inputs()
        screen_manifest_hash = ridgecal_screen_link(source)
        [ridgecal_confirmation_config(lambda_agent, lambda_broker)]
    end
    seeds = ridgecal_stage_seeds(stage)
    entries = ridgecal_build_entries(configs, seeds)
    stage_dir = ridgecal_stage_dir(stage)
    isfile(joinpath(stage_dir, "manifest.jld2")) &&
        error("Ridge calibration manifest already exists: $stage_dir")
    mkpath(stage_dir)

    tsv_path = joinpath(stage_dir, "manifest.tsv")
    ridgecal_write_manifest_tsv(tsv_path, entries)
    manifest_hash = ridgecal_file_hash(tsv_path)
    write(joinpath(stage_dir, "manifest.sha256"), manifest_hash * "\n")
    write(
        joinpath(stage_dir, "counts.env"),
        "NCONFIGS=$(length(configs))\nNRUNS=$(length(entries))\n",
    )

    provenance = Dict{Symbol,Any}(
        :git_commit => source[:git_commit],
        :julia_version => source[:julia_version],
        :pkg_manifest_hash => source[:pkg_manifest_hash],
        :manifest_hash => manifest_hash,
        :schema_version => RIDGECAL_SCHEMA_VERSION,
    )
    meta = Dict{Symbol,Any}(
        :stage => string(stage),
        :date => string(Dates.today()),
        :git_commit => source[:git_commit],
        :git_dirty => dirty,
        :julia_version => source[:julia_version],
        :pkg_manifest_hash => source[:pkg_manifest_hash],
        :schema_version => RIDGECAL_SCHEMA_VERSION,
        :baseline => Dict(pairs(RIDGECAL_BASELINE)),
        :N => RIDGECAL_N,
        :T => RIDGECAL_T,
        :early_periods => collect(RIDGECAL_EARLY_PERIODS),
        :late_periods => collect(RIDGECAL_LATE_PERIODS),
        :seeds => seeds,
        :candidate_lambdas => RIDGECAL_LAMBDAS,
        :practical_tolerance => RIDGECAL_PRACTICAL_TOLERANCE,
        :selection_note => selection_note,
        :screen_manifest_hash => screen_manifest_hash,
        :n_configs => length(configs),
        :n_runs => length(entries),
    )
    jldsave(
        joinpath(stage_dir, "manifest.jld2");
        configs=configs,
        entries=entries,
        meta=meta,
        provenance=provenance,
        manifest_hash=manifest_hash,
    )

    println("Ridge calibration stage: $stage")
    println("directory: $stage_dir")
    println("git commit: $(source[:git_commit])$(dirty ? " (dirty workflow test)" : "")")
    println("configs: $(length(configs))")
    println("NRUNS=$(length(entries))")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && ridgecal_manifest_main()
