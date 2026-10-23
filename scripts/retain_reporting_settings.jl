"""Recover settings for retained analyses without reading or rerunning simulations."""

using JLD2: load
using TOML
using SHA: sha256

include(joinpath(@__DIR__, "reporting_settings.jl"))
include(joinpath(@__DIR__, "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RS = ReportingSettings

"""Recover a numeric constant from its recorded source and retain the source hash."""
function recorded_number(records, commit, path, pattern)
    source = read(`git -C $ROOT show $commit:$path`, String)
    records["$commit:$path"] = bytes2hex(sha256(source))
    found = match(pattern, source)
    isnothing(found) && error("missing recorded setting in $commit:$path")
    return parse(Float64, found[1])
end

function main()
    isempty(ARGS) || error("usage: retain_reporting_settings.jl")
    provenance = manuscript_git_provenance(
        ROOT; sources=(@__FILE__, "scripts/reporting_settings.jl")
    )
    sources = Dict{String,Any}()
    inputs = String[]
    sweeps = Dict{String,Any}()
    artifacts = Dict{String,Any}()
    readinput(path) = (push!(inputs, path); load(joinpath(ROOT, path)))
    paired_prov = Dict(
        Tuple(split(line, '='; limit=2)) for
        line in eachline(joinpath(ROOT, "output/ridge/paired/analysis/provenance.txt"))
    )
    push!(inputs, "output/ridge/paired/analysis/provenance.txt")
    services = readinput("output/assessment_access/figure_data.jld2")
    for tag in (paired_prov["nn_sweep"], paired_prov["ridge_sweep"])
        directory = joinpath("runs", "sweeps", tag)
        data = RS.manifest(joinpath(ROOT, directory))
        append!(
            inputs,
            [
                joinpath(directory, file) for
                file in ("manifest.json", "manifest.jld2", "manifest.sha256")
            ],
        )
        sweeps[data["manifest_hash"]] = RS.sweep_metadata(ROOT, data)
    end

    windows = readinput("output/main/access_windows.jld2")["metadata"]
    paired_windows = readinput("output/ridge/paired/access_windows.jld2")["metadata"]
    windows["early_window"] == paired_windows["early_window"] ||
        error("early windows differ")
    windows["late_window"] == paired_windows["late_window"] || error("late windows differ")
    audit_path = "output/main/convergence/condition_audit.tsv"
    push!(inputs, audit_path)
    rows = readlines(joinpath(ROOT, audit_path))
    header = split(first(rows), '\t')
    audit = [Dict(zip(header, split(row, '\t'))) for row in rows[2:end]]
    main_sweep = sweeps[paired_prov["nn_manifest"]]
    baseline = main_sweep["baseline_reldir"]
    nested_baseline = only(
        unique(parse(Int, r["nested_n"]) for r in audit if r["result_reldir"] == baseline)
    )
    nested_general = only(
        unique(parse(Int, r["nested_n"]) for r in audit if r["result_reldir"] != baseline)
    )
    all(
        parse(Int, r["n_seeds"]) ==
        length(main_sweep["condition_seeds"][r["result_reldir"]]) for r in audit
    ) || error("convergence seed counts differ from the manifest")

    for path in (
        "output/main/figure_data.jld2",
        "output/ridge/paired/figure_data.jld2",
        "output/supplement/structural_figure_data.jld2",
    )
        data = readinput(path)["figdata"]
        sweep = sweeps[data["meta"]["manifest_hash"]]
        RS.validate_figure(data, sweep)
        commit = data["meta"]["analysis_git_commit"]
        level = recorded_number(
            sources, commit, "scripts/monte_carlo.jl", r"level::Float64\s*=\s*([0-9.]+)"
        )
        extractor = if path == "output/supplement/structural_figure_data.jld2"
            "scripts/paper/supp_figdata.jl"
        else
            "scripts/paper/figdata.jl"
        end
        width = Int(
            recorded_number(sources, commit, extractor, r"const LATE_WIDTH\s*=\s*(\d+)")
        )
        bounds = collect(windows["late_window"])
        bounds == [sweep["horizon"] - width + 1, sweep["horizon"]] ||
            error("late windows differ")
        artifacts[path] = Dict(
            "sha256" => RS.filehash(joinpath(ROOT, path)),
            "manifest_hash" => sweep["manifest_hash"],
            "analysis" => Dict(
                "early_window" => collect(windows["early_window"]),
                "late_window" => bounds,
                "interval_level" => level,
                "nested_baseline" => nested_baseline,
                "nested_general" => nested_general,
            ),
        )
    end
    ablations = readinput("output/ridge/ablations/figure_data.jld2")["figdata"]
    services["interval_level"] == ablations["meta"]["interval_level"] ||
        error("interval levels differ")
    services["late_width"] == ablations["meta"]["late_width"] || error("late widths differ")
    for entry in values(artifacts)
        entry["analysis"]["interval_level"] == services["interval_level"] ||
            error("interval levels differ")
        diff(entry["analysis"]["late_window"])[1] + 1 == services["late_width"] ||
            error("late widths differ")
    end

    analysis = only(unique(pop!(entry, "analysis") for entry in values(artifacts)))
    plan = only(unique(pop!(sweep, "condition_seeds") for sweep in values(sweeps)))
    seed_sets = sort(unique(collect(values(plan))); by=Tuple)
    seed_plan = Dict(
        "sets" => seed_sets,
        "conditions" =>
            Dict(rel => findfirst(==(seeds), seed_sets) for (rel, seeds) in plan),
    )
    content = sprint(
        io -> TOML.print(
            io,
            Dict(
                "sweeps" => sweeps,
                "artifacts" => artifacts,
                "analysis" => analysis,
                "seed_plan" => seed_plan,
            );
            sorted=true,
        ),
    )
    mkpath(dirname(RS.RETAINED_PATH))
    write_if_changed(RS.RETAINED_PATH, content)
    record_publication_stage(
        provenance;
        stage="scripts/retain_reporting_settings.jl",
        inputs=sort(unique(inputs)),
        outputs=(RS.RETAINED_PATH,),
        settings=Dict("historical_sources" => sources),
    )
    println("Retained verified reporting settings; simulation results were not rewritten.")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
