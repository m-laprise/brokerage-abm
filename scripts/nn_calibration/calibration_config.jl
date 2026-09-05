"""
    calibration_config.jl

Fixed design and helpers for neural-network optimizer calibration.

The screen varies one learner at a time around the reference setting. The
confirmation stage adds seeds only for the two best screen configurations per
learner. The combined stage evaluates the selected agent and broker settings
together. All scientific decisions use seed-level summaries.
"""

const NNCAL_SCHEMA_VERSION = 1
const NNCAL_N = 1000
const NNCAL_T = 200
const NNCAL_EARLY_PERIODS = 101:150
const NNCAL_LATE_PERIODS = 151:200
const NNCAL_SCREEN_SEEDS = collect(9_000_001:9_000_003)
const NNCAL_CONFIRM_SEEDS = collect(9_000_004:9_000_005)
const NNCAL_ALL_SEEDS = vcat(NNCAL_SCREEN_SEEDS, NNCAL_CONFIRM_SEEDS)
const NNCAL_LEARNING_RATES = [0.003, 0.01, 0.03]
const NNCAL_RECURRENT_STEPS = [50, 100, 200]
const NNCAL_REFERENCE_LEARNING_RATE = 0.01
const NNCAL_REFERENCE_RECURRENT_STEPS = 100
const NNCAL_INITIAL_TO_RECURRENT_RATIO = 2
const NNCAL_PRACTICAL_TOLERANCE = 0.01
const NNCAL_SHORTLIST_SIZE = 2

function nncal_setting(eta_lr::Real, recurrent_steps::Integer)
    steps = Int(recurrent_steps)
    return (
        eta_lr=Float64(eta_lr),
        recurrent_steps=steps,
        initial_steps=NNCAL_INITIAL_TO_RECURRENT_RATIO * steps,
    )
end

function nncal_reference_setting()
    nncal_setting(NNCAL_REFERENCE_LEARNING_RATE, NNCAL_REFERENCE_RECURRENT_STEPS)
end

function nncal_candidate_settings()
    return [
        nncal_setting(eta_lr, steps) for steps in NNCAL_RECURRENT_STEPS for
        eta_lr in NNCAL_LEARNING_RATES
    ]
end

function nncal_same_setting(left, right)
    return left.eta_lr == right.eta_lr &&
           left.recurrent_steps == right.recurrent_steps &&
           left.initial_steps == right.initial_steps
end

function nncal_config(agent, broker; agent_scan::Bool, broker_scan::Bool)
    return Dict{Symbol,Any}(
        :agent_eta_lr => agent.eta_lr,
        :agent_initial_steps => agent.initial_steps,
        :agent_recurrent_steps => agent.recurrent_steps,
        :broker_eta_lr => broker.eta_lr,
        :broker_initial_steps => broker.initial_steps,
        :broker_recurrent_steps => broker.recurrent_steps,
        :agent_scan => agent_scan,
        :broker_scan => broker_scan,
    )
end

function nncal_config_key(config)
    return (
        config[:agent_eta_lr],
        config[:agent_initial_steps],
        config[:agent_recurrent_steps],
        config[:broker_eta_lr],
        config[:broker_initial_steps],
        config[:broker_recurrent_steps],
    )
end

function nncal_assign_ids!(configs)
    for (index, config) in enumerate(configs)
        config[:config_id] = index - 1
        config[:reldir] = "config_$(lpad(index - 1, 3, '0'))"
    end
    return configs
end

function nncal_screen_configs()
    reference = nncal_reference_setting()
    configs = [nncal_config(reference, reference; agent_scan=true, broker_scan=true)]
    for setting in nncal_candidate_settings()
        nncal_same_setting(setting, reference) && continue
        push!(configs, nncal_config(setting, reference; agent_scan=true, broker_scan=false))
    end
    for setting in nncal_candidate_settings()
        nncal_same_setting(setting, reference) && continue
        push!(configs, nncal_config(reference, setting; agent_scan=false, broker_scan=true))
    end
    length(configs) == 17 || error("expected 17 screen configurations")
    length(unique(nncal_config_key(config) for config in configs)) == length(configs) ||
        error("duplicate screen configuration")
    return nncal_assign_ids!(configs)
end

function nncal_combined_config(agent_setting, broker_setting)
    return nncal_assign_ids!([
        nncal_config(agent_setting, broker_setting; agent_scan=false, broker_scan=false)
    ])
end

function nncal_build_entries(configs, seeds)
    entries = Dict{Symbol,Any}[]
    index = 0
    for config in configs, seed in seeds
        push!(
            entries,
            Dict{Symbol,Any}(
                :index => index,
                :config_id => config[:config_id],
                :reldir => config[:reldir],
                :seed => seed,
                :config => config,
            ),
        )
        index += 1
    end
    return entries
end

function nncal_calibration_root()
    return get(ENV, "BROKERAGE_ABM_NN_CALIBRATION_DIR") do
        error("BROKERAGE_ABM_NN_CALIBRATION_DIR is required")
    end
end

nncal_stage_dir(stage::Symbol) = joinpath(nncal_calibration_root(), "stages", string(stage))
nncal_summary_dir() = joinpath(nncal_calibration_root(), "summaries")

function nncal_stage_seeds(stage::Symbol)
    stage == :screen && return NNCAL_SCREEN_SEEDS
    stage == :confirm && return NNCAL_CONFIRM_SEEDS
    stage == :combined && return NNCAL_ALL_SEEDS
    error("unknown calibration stage: $stage")
end

function nncal_stage_configs(stage::Symbol)
    stage == :screen && return nncal_screen_configs()

    if stage == :confirm
        path = joinpath(nncal_summary_dir(), "screen_selection.jld2")
        isfile(path) || error("screen selection not found: $path")
        return JLD2.load(path, "shortlist_configs")
    end

    if stage == :combined
        path = joinpath(nncal_summary_dir(), "confirmed_selection.jld2")
        isfile(path) || error("confirmed selection not found: $path")
        selection = JLD2.load(path)
        return nncal_combined_config(
            selection["selected_agent_setting"], selection["selected_broker_setting"]
        )
    end

    error("unknown calibration stage: $stage")
end
