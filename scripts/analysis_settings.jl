"""Declared analysis and display choices, independent of simulation defaults."""
module AnalysisSettings

const EARLY_WINDOW = (51, 70)
const LATE_WIDTH = 20
const INTERVAL_LEVEL = 0.95
const NESTED_SEEDS = (general=10, baseline=20)
const SUMMARY_QUANTILES = (0.10, 0.90)
const ROLLING_WIDTH = 5
const AXIS_START = 30
const SPECTRUM_COMPONENTS = 25
const ENERGY_FRACTION = 0.90
const PROJECTION_COMPONENTS = 3

"""Resolve and validate the declared windows for a simulation horizon."""
function windows(horizon; early=EARLY_WINDOW, late_width=LATE_WIDTH)
    late = (horizon - late_width + 1, horizon)
    1 <= early[1] <= early[2] < late[1] <= late[2] ||
        error("invalid or overlapping reporting windows")
    return (; early, late)
end

"""Settings to retain with a newly computed analysis."""
function metadata(horizon)
    periods = windows(horizon)
    return Dict{String,Any}(
        "early_window" => collect(periods.early),
        "late_window" => collect(periods.late),
        "interval_level" => INTERVAL_LEVEL,
        "nested_general" => NESTED_SEEDS.general,
        "nested_baseline" => NESTED_SEEDS.baseline,
    )
end

end
