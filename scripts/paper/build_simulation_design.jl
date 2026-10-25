"""
Generate the supplementary simulation design from retained reporting data and
verified sweep manifests. Calibration designs use the recorded simulation commits.
Record each input's provenance in `output/publication.toml`.

Edit prose in `paper/simulation_design_source.tex`. No simulations are run.
Requires the seven reporting manifests under `runs/sweeps/<tag>/`.

Usage: julia --project --threads=auto scripts/paper/build_simulation_design.jl
"""

using JLD2
using SHA: sha256
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
include(joinpath(@__DIR__, "..", "reporting_settings.jl"))
include(joinpath(@__DIR__, "..", "analysis_settings.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const TEMPLATE = joinpath(ROOT, "paper", "simulation_design_source.tex")
const OUTPUT = joinpath(ROOT, "output", "supplement", "simulation_design.tex")
const PROVENANCE = manuscript_git_provenance(ROOT;
    sources=(@__FILE__, TEMPLATE, "scripts/reporting_settings.jl", "scripts/analysis_settings.jl"))
isempty(ARGS) || error("usage: build_simulation_design.jl")
const RECORDS = Dict{String,Any}()

"""Record a retained input's content hash and return its absolute path."""
function input(path)
    full = joinpath(ROOT, path)
    RECORDS[path] = Dict("sha256" => bytes2hex(sha256(read(full))))
    return full
end

"""Read the analysis provenance recorded with retained results."""
function provenance(path)
    Dict(split(line, "="; limit=2) |> Tuple for line in eachline(input(path)) if occursin("=", line))
end

"""Read calibration or manifest utilities at their recorded simulation commit."""
function historical_module(name, commit, path)
    commit = validate_analysis_commit(PROVENANCE, commit; artifact=path)
    source = read(Cmd(["git", "-C", ROOT, "show", "$commit:$path"]), String)
    RECORDS["$commit:$path"] = Dict("commit" => commit, "sha256" => bytes2hex(sha256(source)))
    mod = Module(name)
    overrides = Pair{String,Any}[key => nothing for key in keys(ENV) if startswith(key, "BROKERAGE_ABM_")]
    withenv(overrides...) do
        Base.include_string(mod, source, path)
    end
    return mod
end

"""Read named values from a generated TeX fragment."""
function definitions(path, prefix)
    source = read(input(path), String)
    pattern = Regex("\\\\" * prefix * raw"Define\{([^}]+)\}\{([^}]+)\}")
    Dict(m[1] => m[2] for m in eachmatch(pattern, source))
end

fmt(x::Real) = isinteger(x) ? string(Int(x)) : string(x)
fmt(x) = string(x)
values_text(xs) = join(fmt.(xs), ", ")
range_text(xs) = first(xs) == last(xs) ? fmt(first(xs)) : "$(fmt(first(xs)))--$(fmt(last(xs)))"
number_text(x) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1{,}")

main = load(input("output/main/figure_data.jld2"), "figdata")
ridge = load(input("output/ridge/paired/figure_data.jld2"), "figdata")
ablations = load(input("output/ridge/ablations/figure_data.jld2"), "figdata")
services = load(input("output/assessment_access/figure_data.jld2"))
windows = load(input("output/main/access_windows.jld2"), "metadata")
paired_prov = provenance("output/ridge/paired/analysis/provenance.txt")
ablation_prov = provenance("output/ridge/ablations/analysis/provenance.txt")
ridge_values = definitions("output/ridge/paired/analysis/values.tex", "rv")
main_values = definitions("output/main/values.tex", "pv")
ablation_values = definitions("output/ridge/ablations/analysis/values.tex", "rav")

for (data, prefix) in ((main, "nn"), (ridge, "ridge"))
    @assert data["meta"]["sweep"] == paired_prov[prefix * "_sweep"]
    @assert data["meta"]["manifest_hash"] == paired_prov[prefix * "_manifest"]
end
@assert windows["manifest_hash"] == main["meta"]["manifest_hash"]
@assert ablation_prov["pair_manifest"] == ridge["meta"]["manifest_hash"]
seedsets = unique(row["seeds"] for row in main["regime_cells"])
general_seeds = only(filter(s -> s != main["baseline_seeds"], seedsets))
"""Read a verified manifest, retaining its own simulation provenance."""
function manifest(tag, expected_hash)
    directory = joinpath("runs", "sweeps", tag)
    for name in ("manifest.json", "manifest.jld2", "manifest.sha256")
        input(joinpath(directory, name))
    end
    data = ReportingSettings.manifest(joinpath(ROOT, directory))
    @assert data["manifest_hash"] == expected_hash
    return data
end

main_manifest = manifest(paired_prov["nn_sweep"], paired_prov["nn_manifest"])
ridge_manifest = manifest(paired_prov["ridge_sweep"], paired_prov["ridge_manifest"])
for key in (:oat_axes, :phase_pairs, :baseline, :seeds, :baseline_seeds)
    @assert main_manifest["spec"][key] == ridge_manifest["spec"][key]
end
@assert main_manifest["cells"] == ridge_manifest["cells"]
@assert main_manifest["conditions"] == ridge_manifest["conditions"]
spec = main_manifest["spec"]
design = (;
    SWEEP_BASELINE = (; spec[:baseline]...),
    SWEEP_T = main_manifest["meta"][:T],
    OAT_AXES = [(key=Symbol(a[:key]), vals=a[:vals]) for a in spec[:oat_axes]],
    PHASE_PAIRS = [(xkey=Symbol(g[:xkey]), ykey=Symbol(g[:ykey]), xvals=g[:xvals], yvals=g[:yvals]) for g in spec[:phase_pairs]],
    BASELINE_RELDIR = only(c[:result_reldir] for c in main_manifest["conditions"] if all(spec[:baseline][k] == v for (k,v) in c[:resolved_params])),
)
cells = main_manifest["cells"]
conditions = main_manifest["conditions"]
@assert general_seeds == spec[:seeds]
@assert main["baseline_seeds"] == ridge["baseline_seeds"] == spec[:baseline_seeds]
for data in (main, ridge)
    byrel = Dict(row["rel"] => row for row in data["regime_cells"])
    @assert Set(keys(byrel)) == Set(cell[:result_reldir] for cell in conditions)
    @assert all(byrel[cell[:result_reldir]]["seeds"] == cell[:seeds] for cell in conditions)
    @assert all(byrel[cell[:result_reldir]][key] == cell[:resolved_params][Symbol(key)] for cell in conditions for key in ("rho", "delta"))
    @assert sum(length(row["seeds"]) for row in values(byrel)) == data["meta"]["n_runs"]
    @assert data["period"] == collect(1:design.SWEEP_T)
    @assert Set((row["rho"], row["delta"]) for row in data["grid_cells"]) ==
        Set((cell[:xval], cell[:yval]) for cell in cells if get(cell, :pair, "") == "rho_delta")
end
@assert length(conditions) == parse(Int, main_values["nRegimes"])
@assert main["meta"]["n_runs"] == parse(Int, main_values["nRuns"])
baseline = design.SWEEP_BASELINE
names = Dict(
    :rho => ("Matching composition", raw"\rho"),
    :eta => ("Turnover probability", raw"\eta"),
    :N => ("Number of principals", "N"),
    :reservation_frac => ("Reservation coefficient", "f_r"),
    :delta => ("Matching difficulty", raw"\delta"),
    :k => ("Initial ring degree", "k_G"),
    :roster_frac => ("Broker roster share", raw"\alpha_R"),
    :n_strangers => ("Self-search stranger pool", raw"n_{\mathrm{strangers}}"),
)
vals = Dict{String,String}()
put(key, value) = (vals[key] = fmt(value))
oat_rows = String[]
for axis in design.OAT_AXES
    label, symbol = names[axis.key]
    # Include baseline when it is a separate canonical run (notably initial degree).
    levels = sort(unique(vcat(axis.vals, baseline[axis.key])))
    push!(oat_rows, "$label, \\($symbol\\) & $(fmt(baseline[axis.key])) & $(values_text(levels)) " * "\\\\")
end
put("OAT_ROWS", join(oat_rows, "\n"))
grid_rows = String[]
for grid in design.PHASE_PAIRS
    xsymbol, ysymbol = names[grid.xkey][2], names[grid.ykey][2]
    push!(grid_rows, "\\($xsymbol\\times $ysymbol\\) & $(values_text(grid.xvals)) & $(values_text(grid.yvals)) " * "\\\\")
end
put("GRID_ROWS", join(grid_rows, "\n"))
put("HORIZON", design.SWEEP_T)
put("REGIMES", length(conditions))
put("RUNS", number_text(main["meta"]["n_runs"]))
put("GENERAL_SEEDS", range_text(general_seeds))
put("BASELINE_SEEDS", range_text(main["baseline_seeds"]))
boundary = parse(Float64, ablation_values["rhoBoundary"])
put("RHO_BOUNDARY", boundary)
grid_conditions = filter(c -> get(c, :pair, "") == "rho_delta", cells)
@assert Set((c[:resolved_params][:rho], c[:resolved_params][:rho] == boundary ? baseline.delta : c[:resolved_params][:delta]) for c in grid_conditions) == Set((c["rho"], c["rho"] == boundary ? baseline.delta : c["delta"]) for c in ablations["conditions"])
@assert length(ablations["conditions"]) == ablations["meta"]["n_conditions"]
@assert all(row["N"] == baseline.N for row in ablations["conditions"])
@assert all(Set(keys(row["rank_gaps"])) == Set(["pair", "size_matched", "single_principal", "additive"]) for row in ablations["conditions"])
@assert all(row["seeds"] == (row["result_reldir"] == design.BASELINE_RELDIR ? main["baseline_seeds"] : general_seeds) for row in ablations["conditions"])
put("ABLATION_REGIMES", length(ablations["conditions"]))
put("ABLATION_RUNS", number_text(sum(length(row["seeds"]) for row in ablations["conditions"])))

ablation_manifests = Dict(variant => manifest(ablation_prov[variant * "_sweep"], ablation_prov[variant * "_manifest"]) for variant in ("size_matched", "single_principal", "additive"))
market_key(rho, delta) = (rho, rho == boundary ? baseline.delta : delta)
retained_restrictions = Dict(market_key(c["rho"], c["delta"]) => c for c in ablations["conditions"])
for (variant, data) in ablation_manifests
    @assert string(data["meta"][:ridge_broker_variant]) == variant
    @assert data["meta"][:T] == design.SWEEP_T
    @assert data["spec"][:baseline] == spec[:baseline]
    @assert length(data["conditions"]) == length(retained_restrictions)
    for c in data["conditions"]
        retained = retained_restrictions[market_key(c[:resolved_params][:rho], c[:resolved_params][:delta])]
        @assert retained["seeds"] == c[:seeds]
        @assert retained["N"] == c[:resolved_params][:N]
    end
end

service_prov = provenance("output/assessment_access/provenance.txt")
service_manifest = manifest(basename(services["source_root"]), service_prov["manifest_hash"])
service_supplement = manifest(basename(services["supplement_source_root"]), service_prov["supplement_manifest_hash"])
@assert service_manifest["manifest_hash"] == services["manifest_hash"]
@assert service_supplement["manifest_hash"] == services["supplement_manifest_hash"]
@assert service_supplement["spec"][:baseline] == service_manifest["spec"][:baseline]
@assert service_supplement["meta"][:T] == service_manifest["meta"][:T]
@assert service_manifest["meta"][:T] == design.SWEEP_T
@assert all(service_manifest["spec"][:baseline][key] == baseline[key] for key in keys(baseline))
@assert all(row.N == baseline.N for part in values(services["output_accounting"]) for row in part["calibrations"])
service_seeds = Dict{Tuple{String,Float64,Float64},Set{Int}}()
for row in services["seed_rows"]
    key = (string(row[1]), Float64(row[2]), Float64(row[3]))
    push!(get!(service_seeds, key, Set{Int}()), Int(row[4]))
end
manifest_service_seeds = Dict((string(c[:resolved_params][:broker_service]), Float64(c[:resolved_params][:rho]), Float64(c[:resolved_params][:eta])) => Set(c[:seeds]) for data in (service_manifest, service_supplement) for c in data["conditions"])
@assert manifest_service_seeds == service_seeds
modes = sort(unique(first(key) for key in keys(service_seeds)))
market_cells = sort(unique((key[2], key[3]) for key in keys(service_seeds)))
@assert all(haskey(service_seeds, (mode, rho, eta)) for mode in modes for (rho, eta) in market_cells)
@assert length(modes) == 3
rhos = sort(unique(first.(market_cells)))
etas = sort(unique(last.(market_cells)))
crossed_etas = [eta for eta in etas if all((rho, eta) in market_cells for rho in rhos)]
extra = filter(c -> !(c[2] in crossed_etas), market_cells)
extra_text = join(["\\((\\rho,\\eta)=($(fmt(rho)),$(fmt(eta)))\\)" for (rho, eta) in extra], " and ")
put("SERVICE_RHO", values_text(rhos))
put("SERVICE_ETA", values_text(crossed_etas))
put("SERVICE_EXTRA", extra_text)
put("SERVICE_REGIMES", length(market_cells))
put("SERVICE_RUNS", number_text(sum(length(seeds) for seeds in values(service_seeds))))
put("SERVICE_BASELINE_SEEDS", only(unique(length(seeds) for ((mode, rho, eta), seeds) in service_seeds if (rho, eta) == (baseline.rho, baseline.eta))))
put("SERVICE_OTHER_SEEDS", only(unique(length(seeds) for ((mode, rho, eta), seeds) in service_seeds if (rho, eta) != (baseline.rho, baseline.eta))))

nncal = historical_module(:NNCalibration, paired_prov["nn_commit"], "scripts/nn_calibration/calibration_config.jl")
ridgecal = historical_module(:RidgeCalibration, paired_prov["ridge_commit"], "scripts/ridge/calibration/config.jl")
nnconfigs = Base.invokelatest(nncal.nncal_screen_configs)
# Report the adopted settings, not a recalculated calibration selection.
nnselected = ReportingSettings.sweep_metadata(ROOT, main_manifest)["baseline"]
put("NN_RATES", values_text(nncal.NNCAL_LEARNING_RATES))
put("NN_STEPS", values_text(nncal.NNCAL_RECURRENT_STEPS))
put("NN_RATIO", nncal.NNCAL_INITIAL_TO_RECURRENT_RATIO == 2 ? "twice" : "$(fmt(nncal.NNCAL_INITIAL_TO_RECURRENT_RATIO)) times")
put("NN_CONFIGS", length(nnconfigs))
put("NN_SCREEN_SEEDS", length(nncal.NNCAL_SCREEN_SEEDS))
put("NN_CONFIRM_SEEDS", length(nncal.NNCAL_CONFIRM_SEEDS))
@assert issubset(nncal.NNCAL_SCREEN_SEEDS, nncal.NNCAL_CONFIRM_SEEDS)
put("NN_HORIZON", nncal.NNCAL_T)
put("NN_AGENT_RATE", nnselected["eta_lr_agent"])
put("NN_BROKER_RATE", nnselected["eta_lr_broker"])
put("NN_INITIAL", nnselected["E_init_agent"])
put("NN_RECURRENT", nnselected["train_steps_agent"])
put("RIDGE_CANDIDATES", values_text(ridgecal.RIDGECAL_LAMBDAS))
put("RIDGE_SCREEN_SEEDS", length(ridgecal.RIDGECAL_SCREEN_SEEDS))
put("RIDGE_CONFIRM_SEEDS", length(ridgecal.RIDGECAL_CONFIRM_SEEDS))
@assert isempty(intersect(ridgecal.RIDGECAL_SCREEN_SEEDS, ridgecal.RIDGECAL_CONFIRM_SEEDS))
put("RIDGE_HORIZON", ridgecal.RIDGECAL_T)
for (suffix, macro_suffix) in (("AGENT", "Agent"), ("BROKER", "Broker"))
    key = "ridgeLambda" * macro_suffix
    @assert ridge_values[key] == ablation_values[key]
    for data in (ridge_manifest, values(ablation_manifests)...)
        @assert parse(Float64, ridge_values[key]) == data["meta"][Symbol("ridge_lambda_" * lowercase(suffix))]
    end
    put("RIDGE_" * suffix, ridge_values[key])
end
put("RIDGE_PENALTIES", vals["RIDGE_AGENT"] == vals["RIDGE_BROKER"] ? vals["RIDGE_AGENT"] * " for both learners" : vals["RIDGE_AGENT"] * " for principals and " * vals["RIDGE_BROKER"] * " for the broker")
delete!(vals, "RIDGE_AGENT")
delete!(vals, "RIDGE_BROKER")
@assert isempty(intersect(main["baseline_seeds"], vcat(nncal.NNCAL_SCREEN_SEEDS, nncal.NNCAL_CONFIRM_SEEDS, ridgecal.RIDGECAL_SCREEN_SEEDS, ridgecal.RIDGECAL_CONFIRM_SEEDS)))
put("EARLY_WINDOW", range_text(windows["early_window"]))
put("LATE_WINDOW", range_text(windows["late_window"]))
for (key, field) in (("EARLY_NETWORK_SNAPSHOT", "early_window"), ("LATE_NETWORK_SNAPSHOT", "late_window"))
    put(key, AnalysisSettings.network_snapshot_period(windows[field], ReportingSettings.sweep_metadata(ROOT, main_manifest)["network_measure_interval"]))
end
@assert windows["late_window"] == (design.SWEEP_T - services["late_width"] + 1, design.SWEEP_T)
@assert ablations["meta"]["late_width"] == services["late_width"]
@assert ablations["meta"]["interval_level"] == services["interval_level"]
put("INTERVAL_LEVEL", 100 * services["interval_level"])

template = read(TEMPLATE, String)
used = Set(m[1] for m in eachmatch(r"\\pv\{(\w+)\}", template))
isempty(setdiff(used, keys(vals))) || error("undefined simulation-design value")
document = replace(template, r"\\pv\{(\w+)\}" => token -> vals[token[5:(end - 1)]])
input_provenance = analysis_input_provenance(PROVENANCE, [
    joinpath(ROOT, path) for path in (
        "output/main/values.tex",
        "output/ridge/paired/analysis/provenance.txt",
        "output/ridge/ablations/analysis/provenance.txt",
        "output/assessment_access/provenance.txt",
    )
])
write_if_changed(OUTPUT, document)
record_publication_stage(PROVENANCE;
    stage="scripts/paper/build_simulation_design.jl",
    inputs=sort([path for (path, record) in RECORDS if !haskey(record, "commit")]),
    outputs=(OUTPUT,),
    analysis_commits=Dict("$(input.path): $label" => commit
        for input in input_provenance for (label, commit) in input.commits),
    settings=Dict(
        "simulation_commits" => Dict(data["meta"][:tag] => data["meta"][:git_commit]
            for data in (main_manifest, ridge_manifest, values(ablation_manifests)...,
                service_manifest, service_supplement)),
        "historical_sources" => Dict(
            path => record for (path, record) in RECORDS if haskey(record, "commit"))),
)
println("Generated supplementary simulation design from seven verified manifests.")
