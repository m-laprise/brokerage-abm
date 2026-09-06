"""Fixed design and paths for the Ridge-penalty calibration."""

const RIDGECAL_SCHEMA_VERSION = 1
const RIDGECAL_BASELINE = (
    rho=0.5,
    eta=0.02,
    N=1000,
    reservation_frac=0.60,
    delta=0.50,
    k=6,
    roster_frac=0.20,
    n_strangers=10,
)
const RIDGECAL_N = RIDGECAL_BASELINE.N
const RIDGECAL_T = 500
const RIDGECAL_EARLY_PERIODS = 301:400
const RIDGECAL_LATE_PERIODS = 401:500
const RIDGECAL_SCREEN_SEEDS = collect(9_001:9_010)
const RIDGECAL_CONFIRM_SEEDS = collect(9_011:9_015)
const RIDGECAL_LAMBDAS = [0.0003, 0.001, 0.003, 0.01, 0.03]
const RIDGECAL_PRACTICAL_TOLERANCE = 0.01

function ridgecal_config(lambda_agent::Real, lambda_broker::Real)
    return Dict{Symbol,Any}(
        :lambda_agent => Float64(lambda_agent),
        :lambda_broker => Float64(lambda_broker),
    )
end

ridgecal_config_key(config) = (config[:lambda_agent], config[:lambda_broker])

function ridgecal_assign_ids!(configs)
    for (index, config) in enumerate(configs)
        config[:config_id] = index - 1
        config[:reldir] = "config_$(lpad(index - 1, 3, '0'))"
    end
    return configs
end

function ridgecal_screen_configs()
    configs = [
        ridgecal_config(lambda_agent, lambda_broker) for
        lambda_agent in RIDGECAL_LAMBDAS for lambda_broker in RIDGECAL_LAMBDAS
    ]
    return ridgecal_assign_ids!(configs)
end

function ridgecal_confirmation_config(lambda_agent::Real, lambda_broker::Real)
    Float64(lambda_agent) in RIDGECAL_LAMBDAS ||
        error("confirmation agent penalty is outside the screened grid")
    Float64(lambda_broker) in RIDGECAL_LAMBDAS ||
        error("confirmation broker penalty is outside the screened grid")
    return only(ridgecal_assign_ids!([ridgecal_config(lambda_agent, lambda_broker)]))
end

function ridgecal_build_entries(configs, seeds)
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

function ridgecal_root()
    return get(ENV, "BROKERAGE_ABM_RIDGE_CALIBRATION_DIR") do
        error("BROKERAGE_ABM_RIDGE_CALIBRATION_DIR is required")
    end
end

ridgecal_stage_dir(stage::Symbol) = joinpath(ridgecal_root(), "stages", string(stage))
ridgecal_summary_dir() = joinpath(ridgecal_root(), "summaries")

function ridgecal_stage_seeds(stage::Symbol)
    stage == :screen && return RIDGECAL_SCREEN_SEEDS
    stage == :confirm && return RIDGECAL_CONFIRM_SEEDS
    error("unknown Ridge calibration stage: $stage")
end

