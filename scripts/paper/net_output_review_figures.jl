"""Render the information-source figure from validated, saved seed-level values."""

include(joinpath(@__DIR__, "..", "figure_style.jl"))
using Printf: @sprintf

include(joinpath(@__DIR__, "information_sources.jl"))
using .InformationSources: channel_output_shares

"""Check the review's scientific source hashes without relaxing publication provenance."""
function checked_net_output_review(path)
    data = JLD2.load(path)
    meta = data["meta"]
    meta["review_only"] === true || error("expected review data")
    meta["definition"] == "net_output_per_principal" || error("wrong outcome definition")
    for (key, source) in (
        ("extractor_sha256", joinpath(@__DIR__, "net_output_review.jl")),
        ("reconstruction_sha256", joinpath(@__DIR__, "..", "net_output_data.jl")),
        ("accounting_sha256", joinpath(@__DIR__, "..", "net_output.jl")),
    )
        meta[key] == file_hash(source) || error("review data are stale: $key")
    end
    return data
end

net_values(condition) = condition["late"]["net_output_per_principal"]

"""Calculate late channel contributions, each divided by the full principal population."""
function channel_late_values(df, calibration, late_width)
    1 <= late_width <= nrow(df) || throw(ArgumentError("invalid late window"))
    frame = copy(df)
    N, c_s, phi = calibration.N, calibration.c_s, calibration.phi
    NetOutputData.reconstruct_frame!(frame, N, c_s, phi)
    return InformationSources.late_channel_values(frame, N, late_width)
end

"""Validate a saved aggregate and recover seed-level channel contributions and ranks."""
function condition_channel_values(data, retained)
    path = retained["source_path"]
    file_hash(path) == retained["source_sha256"] || error("channel source differs: $path")
    raw = JLD2.load(path)
    raw["seeds"] == retained["seeds"] || error("channel seeds differ: $path")
    length(raw["mdfs"]) == length(raw["seeds"]) || error("missing channel frames")
    calibrations = Dict(c.seed => c for c in retained["calibrations"])
    Set(keys(calibrations)) == Set(raw["seeds"]) || error("calibration seeds differ")
    width = data["meta"]["late_width"]
    values = map(raw["seeds"], raw["mdfs"]) do seed, frame
        calibration = calibrations[seed]
        calibration.N == retained["N"] || error("principal count differs")
        nrow(frame) == size(retained["series"], 1) || error("period coverage differs")
        channel_late_values(frame, calibration, width)
    end
    isapprox(getproperty.(values, :total), net_values(retained)) ||
        error("channel totals differ from review archive: $path")
    late(column) = [mean(df[(end - width + 1):end, column]) for df in raw["mdfs"]]
    return (;
        seeds=raw["seeds"],
        values,
        source_sha256=retained["source_sha256"],
        broker_rank=late(:broker_holdout_rank),
        principal_rank=late(:agent_holdout_rank),
    )
end

"""
Extract all Ridge channel contributions from saved aggregates on a compute node.

No simulations or calibration replay are run. The compact review artifact retains
the parent archive hash and source hashes. Rendering does not require raw sweeps.
"""
function extract_net_output_channels(review_path, output)
    data = checked_net_output_review(review_path)
    ridge = Dict{String,Any}()
    for model in sort!(collect(keys(data["ridge"])))
        conditions = Dict{String,Any}()
        for rel in sort!(collect(keys(data["ridge"][model])))
            conditions[rel] = condition_channel_values(data, data["ridge"][model][rel])
        end
        ridge[model] = conditions
        println("Extracted $model: $(length(conditions)) regimes")
        flush(stdout)
    end
    meta = Dict(
        "review_only" => true,
        "analysis_source_clean" => false,
        "definition" => "net_output_per_principal",
        "review_sha256" => file_hash(review_path),
        "extractor_sha256" => file_hash(@__FILE__),
        "julia_version" => string(VERSION),
        "late_width" => data["meta"]["late_width"],
    )
    mkpath(dirname(abspath(output)))
    JLD2.jldsave(output; meta, ridge)
    println("Saved $output")
    return output
end

"""Validate the channel archive's parent, source files, seed identities, and totals."""
function checked_channel_review(path, review_path, data)
    channels = JLD2.load(path)
    meta = channels["meta"]
    meta["review_only"] === true || error("expected review channel data")
    meta["definition"] == data["meta"]["definition"] || error("output definitions differ")
    meta["review_sha256"] == file_hash(review_path) ||
        error("channel parent archive differs")
    meta["late_width"] == data["meta"]["late_width"] || error("channel late window differs")
    Set(keys(channels["ridge"])) == Set(keys(data["ridge"])) ||
        error("Ridge variants differ")
    for (model, conditions) in channels["ridge"]
        Set(keys(conditions)) == Set(keys(data["ridge"][model])) ||
            error("channel regimes differ")
        for (rel, condition) in conditions
            retained = data["ridge"][model][rel]
            condition.seeds == retained["seeds"] || error("channel seeds differ")
            condition.source_sha256 == retained["source_sha256"] ||
                error("channel source differs")
            isapprox(getproperty.(condition.values, :total), net_values(retained)) ||
                error("channel totals differ")
            channel_output_shares(
                condition.values, retained["N"]; level=data["meta"]["interval_level"]
            )
        end
    end
    return channels
end

"""Recover the retained ranking evidence used alongside the channel-output panel."""
function channel_review_ranking_data(data, channels, models)
    root = joinpath(@__DIR__, "..", "..", "output")
    ridge_path = joinpath(root, "ridge", "ablations", "figure_data.jld2")
    file_hash(ridge_path) == data["meta"]["input_hashes"]["ridge.jld2"] ||
        error("Ridge ranking and output inputs differ")
    ridge = JLD2.load(ridge_path)["figdata"]
    ridge["meta"]["analysis_source_clean"] === true ||
        error("Ridge ranking input was not generated from a clean source")
    width = data["meta"]["late_width"]
    width == ridge["meta"]["late_width"] ||
        error("ranking and output late windows differ")
    level = data["meta"]["interval_level"]
    baseline_rel = data["meta"]["baseline_reldir"]
    baseline_rel == ridge["meta"]["baseline_reldir"] || error("baselines differ")
    baseline_condition = only(
        filter(c -> c["result_reldir"] == baseline_rel, ridge["conditions"])
    )
    baseline = Dict{String,NamedTuple}()
    for model in models
        retained = data["ridge"][model.key][baseline_rel]
        saved_channels = channels["ridge"][model.key][baseline_rel]
        saved_channels.seeds == retained["seeds"] == baseline_condition["seeds"] ||
            error("baseline ranking seeds differ")
        broker, principal = saved_channels.broker_rank, saved_channels.principal_rank
        all(v -> isfinite(v) && -1 <= v <= 1, [broker; principal]) || error("invalid ranks")
        isapprox(broker .- principal, baseline_condition["rank_gaps"][model.key]) ||
            error("baseline ranking advantage differs")
        baseline[model.key] = (;
            broker=monte_carlo_interval(broker; level),
            principal=monte_carlo_interval(principal; level),
        )
        Set(c["result_reldir"] for c in ridge["conditions"]) ==
        Set(keys(data["ridge"][model.key])) || error("ranking and output regimes differ")
        for condition in ridge["conditions"]
            condition["seeds"] ==
            data["ridge"][model.key][condition["result_reldir"]]["seeds"] ||
                error("ranking and output regime seeds differ")
            saved = channels["ridge"][model.key][condition["result_reldir"]]
            isapprox(
                saved.broker_rank .- saved.principal_rank, condition["rank_gaps"][model.key]
            ) || error("ranking and channel archives disagree")
        end
    end
    return (; baseline, conditions=ridge["conditions"])
end

"""Combine validated review rankings and channels without publication writes."""
function channel_review_figure_data(data, channels)
    models = [(; key) for key in InformationSources.MODEL_KEYS]
    ranking = channel_review_ranking_data(data, channels, models)
    conditions = map(ranking.conditions) do source
        c = copy(source)
        rel = c["result_reldir"]
        c["N"] = data["ridge"]["pair"][rel]["N"]
        all(data["ridge"][m.key][rel]["N"] == c["N"] for m in models) ||
            error("principal counts differ across versions")
        c["broker_ranks"] = Dict(m.key => channels["ridge"][m.key][rel].broker_rank for m in models)
        c["principal_ranks"] = Dict(m.key => channels["ridge"][m.key][rel].principal_rank for m in models)
        c["channel_values"] = Dict(m.key => channels["ridge"][m.key][rel].values for m in models)
        c["net_output"] = Dict(m.key => net_values(data["ridge"][m.key][rel]) for m in models)
        c
    end
    baseline = only(filter(c -> c["result_reldir"] == data["meta"]["baseline_reldir"], conditions))
    return Dict(
        "meta" => Dict(
            "review_only" => true,
            "analysis_source_clean" => false,
            "net_output_definition" => data["meta"]["definition"],
            "n_conditions" => length(conditions),
            "baseline_reldir" => data["meta"]["baseline_reldir"],
            "baseline_difficulty" => baseline["delta"],
            "interval_level" => data["meta"]["interval_level"],
            "late_width" => data["meta"]["late_width"],
        ),
        "conditions" => conditions,
    )
end

"""Render the information-source figure from validated review archives."""
function render_net_output_channels(path, channel_path; difficulties=nothing, output_path=nothing)
    data = checked_net_output_review(path)
    channels = checked_channel_review(channel_path, path, data)
    retained = channel_review_figure_data(data, channels)
    selected = isnothing(difficulties) ? (retained["meta"]["baseline_difficulty"],) : difficulties
    return InformationSources.render(retained; difficulties=selected, output_path)
end

"""Calculate a paired service contrast, checking seed identity explicitly."""
function review_service_effect(data, reference, comparison, rho, eta)
    a, b = data["access"][(reference, rho, eta)], data["access"][(comparison, rho, eta)]
    a["seeds"] == b["seeds"] || error("unpaired service seeds")
    return paired_monte_carlo_interval(
        net_values(a), net_values(b); level=data["meta"]["interval_level"]
    )
end

"""Render the approved information-source figure from the current review archives."""
function render_net_output_review(path)
    channels = joinpath(dirname(path), "net_output_channels.jld2")
    return render_net_output_channels(path, channels)
end
