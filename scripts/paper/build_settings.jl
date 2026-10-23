"""Generate manuscript settings from retained runs and their recorded model sources."""

using JLD2: load
using Printf: @sprintf
using SHA: sha256
include(joinpath(@__DIR__, "..", "reporting_settings.jl"))
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTPUT = joinpath(ROOT, "output", "manuscript", "settings.tex")
const PROVENANCE = manuscript_git_provenance(
    ROOT; sources=(@__FILE__, "scripts/reporting_settings.jl")
)
const INPUTS = String[]

function retained_settings(relative)
    path = joinpath(ROOT, relative)
    data = load(path, "figdata")
    append!(INPUTS, [path; ReportingSettings.retained_dependencies(data)])
    return ReportingSettings.retained(path, data)
end

main = retained_settings("output/main/figure_data.jld2")
ridge = retained_settings("output/ridge/paired/figure_data.jld2")
structural = retained_settings("output/supplement/structural_figure_data.jld2")
baseline = main["baseline"]
commit = main["simulation_commit"]
defaults = ReportingSettings.default_module(ROOT, commit)
sources = Dict{String,String}()

function recorded_source(path)
    get!(sources, path) do
        read(`git -C $ROOT show $commit:$path`, String)
    end
end

"""Read a setting that older runs recorded only in source code."""
function recorded_setting(name, path, pattern)
    isdefined(defaults, name) && return getfield(defaults, name)
    found = match(pattern, recorded_source(path))
    isnothing(found) && error("setting $name not found in $commit:$path")
    return Core.eval(defaults, Meta.parse(found[1]))
end

for path in ("src/types.jl", "src/parameters.jl")
    recorded_source(path)
end
calibration_n = recorded_setting(
    :MATCH_CALIBRATION_N,
    "scripts/calibrate_match_component_moments.jl",
    r"const CALIBRATION_N = (\d+)",
)
calibration_seeds = recorded_setting(
    :MATCH_CALIBRATION_SEEDS,
    "scripts/calibrate_match_component_moments.jl",
    r"const CALIBRATION_SEEDS = ([\d_:]+)",
)
frequencies = recorded_setting(
    :CURVE_FREQUENCIES, "src/initialization.jl", r"freqs = \[rand\(rng, ([\d:]+)\)"
)
broker_seed_cap = recorded_setting(
    :BROKER_HISTORY_SEED_CAP, "src/initialization.jl", r"n_broker_seed = min\((\d+),"
)
holdout_principals = recorded_setting(
    :HOLDOUT_PRINCIPALS, "src/step.jl", r"n_sample_agents = min\((\d+),"
)
holdout_partners = recorded_setting(
    :HOLDOUT_PARTNERS, "src/step.jl", r"n_partners = min\((\d+),"
)
calibration_pairs = recorded_setting(
    :CALIBRATION_PAIRS, "src/matching_function.jl", r"n_samples::Int\s*=\s*([\d_]+)"
)
adam = [
    recorded_setting(name, "src/learning.jl", Regex("const $name = ([^\\n]+)")) for
    name in (:ADAM_BETA1, :ADAM_BETA2, :ADAM_EPS)
]
pair_dimension = match(
    r"(?m)^@inline broker_pair_feature_dim[^\n]+", recorded_source("src/learning.jl")
)
isnothing(pair_dimension) && error("broker feature dimension not found in recorded source")
Base.include_string(defaults, pair_dimension.match)

plain(x::Real) = isinteger(x) ? string(Int(x)) : string(x)
plain(x) = string(x)
grouped(x) = replace(string(x), r"(\d)(?=(\d{3})+$)" => s"\1{,}")
function word(x)
    if 0 <= x <= 12
        split("zero one two three four five six seven eight nine ten eleven twelve")[x + 1]
    else
        string(x)
    end
end
vals = Dict{String,String}()
put(key, value) = (vals["setting" * key] = plain(value))
for key in split("""
    N T d s sigma_x K rho delta eta eta_lr_agent eta_lr_broker E_init_agent E_init_broker
    train_steps_agent train_steps_broker train_window_periods train_max_obs reservation_frac
    search_cost_rate k p_rewire p_demand n_strangers roster_frac roster_churn omega
    network_measure_interval
    """)
    put(key, baseline[key])
end
for key in ("reservation_frac", "rho", "p_demand", "roster_frac")
    put(key * "Fixed", @sprintf("%.2f", baseline[key]))
end
put("NGrouped", grouped(baseline["N"]))
put("dWord", word(baseline["d"]))
put("Q", @sprintf("%.2f", defaults.Q_OFFSET))
put("sigma_eps", @sprintf("%.2f", baseline["sigma_eps"]))
for key in ("ridge_lambda_agent", "ridge_lambda_broker")
    put(key, ridge["baseline"][key])
end
put("CalibrationN", grouped(calibration_n))
put("CalibrationMarkets", length(calibration_seeds))
put("CalibrationPairsGrouped", grouped(calibration_pairs))
put("Frequencies", join(frequencies, ","))
put("FrequencyMin", first(frequencies))
put("FrequencyMax", last(frequencies))
put("BrokerSeedCap", broker_seed_cap)
put("HoldoutPrincipals", holdout_principals)
put("HoldoutPartners", holdout_partners)
put("MinCandidatePairsWord", word(recorded_setting(:MIN_CANDIDATE_PAIRS,
    "src/measures.jl", r"n < (\d+) && return PredictionQuality")))
put("VarianceDivisor", recorded_setting(:VARIANCE_DIVISOR,
    "src/measures.jl", r"var_realized < sigma_eps\^2 / (\d+)"))
put("AgentWidth", Base.invokelatest(defaults.agent_hidden_width, baseline["d"]))
put("BrokerWidth", Base.invokelatest(defaults.broker_hidden_width, baseline["d"]))
put("BrokerDimension", Base.invokelatest(defaults.broker_pair_feature_dim, baseline["d"]))
put(
    "RosterSize",
    Base.invokelatest(defaults.roster_target_size, baseline["N"], baseline["roster_frac"]),
)
put("AdamBetaOne", adam[1])
put("AdamBetaTwo", adam[2])
epsilon = replace(@sprintf("%.0e", adam[3]), r"e(-?)0*(\d+)" => s"\\times10^{\1\2}")
put("AdamEpsilon", replace(epsilon, r"^1\\times" => ""))
for (key, field) in (("EarlyWindow", "early_window"), ("LateWindow", "late_window"))
    put(key, join(main[field], "--"))
end
put("IntervalPercent", 100 * main["interval_level"])
put("SuppIntervalPercent", 100 * structural["interval_level"])
put("NestedGeneral", main["nested_general"])
put("NestedBaseline", main["nested_baseline"])
dgp_path = joinpath(ROOT, "output", "supplement", "dgp_figure_data.jld2")
push!(INPUTS, dgp_path)
dgp = load(dgp_path, "figdata")
put("DgpDimensionsWord", word(dgp["design"]["d"]))
put("DgpProjectionWord", word(dgp["design"]["type_projection_components"]))

content = IOBuffer()
println(content, "% Generated by scripts/paper/build_settings.jl; do not edit.")
println(
    content, raw"\providecommand{\pvDefine}[2]{\expandafter\def\csname pv@#1\endcsname{#2}}"
)
println(content, raw"\providecommand{\pv}[1]{\csname pv@#1\endcsname}")
for key in sort(collect(keys(vals)))
    println(content, "\\pvDefine{$key}{$(vals[key])}")
end
write_if_changed(OUTPUT, String(take!(content)))
record_publication_stage(
    PROVENANCE;
    stage="scripts/paper/build_settings.jl",
    inputs=unique(INPUTS),
    outputs=(OUTPUT,),
    analysis_commits=Dict(
        "main simulation" => commit,
        "Ridge simulation" => ridge["simulation_commit"],
        "structural simulation" => structural["simulation_commit"],
        "DGP analysis" => dgp["meta"]["analysis_git_commit"],
    ),
    settings=Dict(
        "historical_sources" => Dict(
            "$commit:$path" => bytes2hex(sha256(source)) for (path, source) in sources
        ),
    ),
)
println("Generated $(length(vals)) manuscript settings from retained inputs.")
