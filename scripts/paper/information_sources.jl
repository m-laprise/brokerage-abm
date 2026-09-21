"""Shared statistics and rendering for the Ridge information-source figure."""
module InformationSources

using Statistics: mean
using Printf: @sprintf
include(joinpath(@__DIR__, "..", "figure_style.jl"))
include(joinpath(@__DIR__, "..", "monte_carlo.jl"))

"""
Average seed-level late-mean channel shares with Student-t Monte Carlo intervals.

Saved contributions are per principal. Restore aggregate units for the total
using the retained principal count `N`; this scaling cancels in each seed's share.
"""
function channel_output_shares(values, N::Integer; level=0.95)
    isempty(values) && throw(ArgumentError("no channel observations"))
    N > 0 || throw(ArgumentError("principal count must be positive"))
    all(
        v ->
            all(isfinite, (v.self, v.broker, v.total)) &&
            isapprox(v.self + v.broker, v.total),
        values,
    ) || throw(ArgumentError("invalid channel decomposition"))
    all(v -> v.total > 0, values) ||
        throw(ArgumentError("each seed's net output must be positive for percentage shares"))
    self_shares = [v.self / v.total for v in values]
    broker_shares = [v.broker / v.total for v in values]
    all(v -> 0 <= v <= 1, [self_shares; broker_shares]) ||
        throw(ArgumentError("channel shares fall outside 0–100%"))
    self = monte_carlo_interval(self_shares; level)
    broker = monte_carlo_interval(broker_shares; level)
    isapprox(self.mean + broker.mean, 1) || error("channel shares do not sum to 100%")
    total = N * mean(v.total for v in values)
    return (; self, broker, total)
end
const MODEL_KEYS = ("pair", "size_matched", "single_principal", "additive")
const MODEL_LABELS = ("Pair", "Size", "Single", "Additive")

"""Compute channel late means from a frame with reconstructed costs and fees."""
function late_channel_values(frame, N::Integer, late_width::Integer)
    N > 0 || throw(ArgumentError("principal count must be positive"))
    1 <= late_width <= size(frame, 1) || throw(ArgumentError("invalid late window"))
    rows = (size(frame, 1) - late_width + 1):size(frame, 1)
    output(count, q) = iszero(count) ? 0.0 : count * q
    self = mean(
        output.(frame.n_self_matches[rows], frame.q_self_mean[rows]) .-
        frame.total_search_cost[rows]
    ) / N
    broker = mean(
        output.(frame.n_broker_matches[rows], frame.q_broker_mean[rows]) .-
        frame.total_broker_fees[rows]
    ) / N
    total = mean(frame.net_output_per_principal[rows])
    all(isfinite, (self, broker, total)) || error("nonfinite channel output")
    isapprox(self + broker, total) || error("channel output does not sum to total")
    return (; self, broker, total)
end

"""Find the unique baseline condition identified by the retained metadata."""
function baseline_condition(data)
    return only(filter(
        c -> c["result_reldir"] == data["meta"]["baseline_reldir"], data["conditions"]
    ))
end

"""Validate common seed coverage and the accounting identities used by Figure 3."""
function validate_data(data)
    meta, conditions = data["meta"], data["conditions"]
    meta["net_output_definition"] == "net_output_per_principal" ||
        error("regenerate Ridge analysis with net output per principal")
    0 < meta["interval_level"] < 1 || error("invalid interval level")
    length(conditions) == meta["n_conditions"] || error("incomplete Ridge regimes")
    length(unique(c["result_reldir"] for c in conditions)) == length(conditions) ||
        error("duplicate Ridge regime")
    baseline = baseline_condition(data)
    baseline["delta"] == meta["baseline_difficulty"] || error("baseline difficulty differs")
    for c in conditions
        seeds = c["seeds"]
        length(seeds) == length(unique(seeds)) > 1 || error("invalid seed coverage")
        for field in ("broker_ranks", "principal_ranks", "rank_gaps", "channel_values", "net_output")
            Set(keys(c[field])) == Set(MODEL_KEYS) || error("Ridge versions differ")
            all(length(c[field][model]) == length(seeds) for model in MODEL_KEYS) ||
                error("incomplete seed values: $field")
        end
        for model in MODEL_KEYS
            broker, principal = c["broker_ranks"][model], c["principal_ranks"][model]
            all(v -> isfinite(v) && -1 <= v <= 1, [broker; principal]) ||
                error("invalid rank correlation")
            isapprox(broker .- principal, c["rank_gaps"][model]) ||
                error("ranking advantage disagrees with component rankings")
            isapprox(getproperty.(c["channel_values"][model], :total), c["net_output"][model]) ||
                error("channel totals disagree with net output")
            channel_output_shares(c["channel_values"][model], c["N"]; level=meta["interval_level"])
        end
    end
    return data
end

"""Summarize baseline rankings without replacing the seed-level regime data."""
function ranking_data(data, models)
    condition = baseline_condition(data)
    level = data["meta"]["interval_level"]
    baseline = Dict(
        model.key => (;
            broker=monte_carlo_interval(condition["broker_ranks"][model.key]; level),
            principal=monte_carlo_interval(condition["principal_ranks"][model.key]; level),
        ) for model in models
    )
    return (; baseline, conditions=data["conditions"])
end

"""Generate manuscript estimates and validate the accompanying comparative claims."""
function manuscript_values(data)
    validate_data(data)
    meta, conditions = data["meta"], data["conditions"]
    baseline = baseline_condition(data)
    level = meta["interval_level"]
    interval(v) = monte_carlo_interval(v; level)
    loss(c, model, field) = interval(c[field]["pair"] .- c[field][model])
    fmt(v) = @sprintf("%.2f", v)
    percent(v) = @sprintf("%.0f", 100 * v)
    definitions = Pair{String,String}[]
    add(key, value) = push!(definitions, key => fmt(value))
    complementary = filter(c -> c["rho"] < 1, conditions)
    quality = only(filter(c -> c["rho"] == 1, conditions))
    append!(definitions, [
        "infoRegimeN" => string(length(conditions)),
        "infoComplementarityN" => string(length(complementary)),
        "infoIntervalPercent" => percent(level),
        "infoBaselineDifficulty" => string(meta["baseline_difficulty"]),
    ])
    for (model, label) in zip(MODEL_KEYS, MODEL_LABELS)
        for (field, suffix) in (
            ("broker_ranks", "BrokerRank"), ("principal_ranks", "PrincipalRank"), ("rank_gaps", "Gap")
        )
            summary = interval(baseline[field][model])
            add("info$(label)$(suffix)", summary.mean)
            add("info$(label)$(suffix)Lower", summary.lower)
            add("info$(label)$(suffix)Upper", summary.upper)
        end
        shares = channel_output_shares(baseline["channel_values"][model], baseline["N"]; level)
        reference = channel_output_shares(baseline["channel_values"]["pair"], baseline["N"]; level)
        push!(definitions, "info$(label)BrokerSharePercent" => percent(shares.broker.mean))
        push!(definitions, "info$(label)OutputPercent" => percent(shares.total / reference.total))
        if model != "pair"
            for (field, suffix) in (("broker_ranks", "BrokerLoss"), ("rank_gaps", "GapLoss"))
                difference = loss(baseline, model, field)
                add("info$(label)$(suffix)", difference.mean)
                add("info$(label)$(suffix)Lower", difference.lower)
                add("info$(label)$(suffix)Upper", difference.upper)
            end
        end
    end
    for model in ("size_matched", "single_principal"), field in ("broker_ranks", "rank_gaps")
        all(loss(c, model, field).lower > 0 for c in conditions) ||
            error("paired ranking decreases changed: $model, $field")
    end
    count(c -> mean(c["rank_gaps"]["size_matched"]) > 0, conditions) > length(conditions) / 2 ||
        error("size-matched ranking advantage no longer persists in most regimes")
    all(loss(c, "additive", "broker_ranks").lower > 0 for c in complementary) ||
        error("additive broker ranking decreases changed")
    all(loss(c, "additive", "rank_gaps").mean > 0 for c in complementary) ||
        error("additive ranking advantage decreases changed")
    all(loss(quality, "additive", field).mean < 0 for field in ("broker_ranks", "rank_gaps")) ||
        error("pure-quality additive ranking gains changed")
    push!(definitions, "infoSingleNegativeN" => string(count(
        c -> interval(c["rank_gaps"]["single_principal"]).upper < 0, conditions
    )))
    push!(definitions, "infoAdditiveGapDecreaseCIN" => string(count(
        c -> loss(c, "additive", "rank_gaps").lower > 0, complementary
    )))
    # Check the full difficulty design, not just the subset illustrated in C--D.
    for delta in unique(c["delta"] for c in complementary)
        group = sort(filter(c -> c["delta"] == delta, complementary); by=c -> c["rho"])
        first(group)["rho"] == 0 || error("missing pure-complementarity regime")
        for model in ("single_principal", "additive")
            rank_losses = [loss(c, model, "rank_gaps").mean for c in [group; quality]]
            first(rank_losses) >= maximum(rank_losses) ||
                error("largest ranking loss is no longer under pure complementarity")
            share_losses = [
                channel_output_shares(c["channel_values"]["pair"], c["N"]; level).broker.mean -
                channel_output_shares(c["channel_values"][model], c["N"]; level).broker.mean
                for c in [group; quality]
            ]
            all(diff(share_losses) .< 0) ||
                error("channel shift is no longer stronger with complementarity")
        end
    end
    for model in MODEL_KEYS[2:end]
        restricted = baseline["channel_values"][model]
        reference = baseline["channel_values"]["pair"]
        mean(v.self for v in restricted) > mean(v.self for v in reference) &&
            mean(v.broker for v in restricted) < mean(v.broker for v in reference) &&
            mean(v.total for v in restricted) < mean(v.total for v in reference) ||
            error("baseline channel offset changed: $model")
    end
    length(unique(first.(definitions))) == length(definitions) || error("duplicate value key")
    return definitions
end

"""Connect the same regime across groups, using each marker's displayed position."""
function connect_channel_regimes!(axis, positions, values)
    length(positions) == length(values) || error("regime trace groups differ")
    n = length(first(values))
    all(v -> length(v) == n, [positions; values]) || error("regime trace lengths differ")
    for regime in 1:n
        lines!(
            axis,
            [x[regime] for x in positions],
            [y[regime] for y in values];
            color=(:gray45, 0.20),
            linewidth=0.8,
        )
    end
    return nothing
end

"""Label regime colors in two rows, retaining the pure-quality diamond."""
function channel_regime_legend!(axis, rhos)
    elements = [
        MarkerElement(;
            color=rho == 1 ? :white : PUB_RHO_COLORS[rho],
            marker=rho == 1 ? :diamond : :circle,
            strokecolor=PUB_RHO_COLORS[rho],
            strokewidth=rho == 1 ? 1.8 : 0,
            markersize=rho == 1 ? 12 : 10,
        ) for rho in rhos
    ]
    return axislegend(
        axis, elements, string.(rhos), PUB_RHO_LABEL;
        position=:lt,
        orientation=:horizontal,
        nbanks=2,
        titleposition=:top,
        titlesize=TICK_FS,
        titlegap=3,
        framevisible=false,
        labelsize=FOOTER_FS,
        patchsize=(14, 12),
        rowgap=2,
        colgap=12,
        padding=(2, 2, 2, 2),
        margin=(2, 2, 2, 2),
    )
end

"""Render the four-panel figure from validated seed-level Ridge results."""
function render(data; difficulties=(data["meta"]["baseline_difficulty"],), output_path=nothing)
    validate_data(data)
    reference = baseline_condition(data)
    publication_theme!()
    models = (
        (key="pair", label="Reference\nbroker"),
        (key="size_matched", label="Fewer\nobservations"),
        (key="single_principal", label="One party\nrecorded"),
        (key="additive", label="No interaction\nterm"),
    )
    baseline = [reference["channel_values"][m.key] for m in models]
    shares = [
        channel_output_shares(
            values,
            reference["N"];
            level=data["meta"]["interval_level"],
        ) for values in baseline
    ]
    reference_index = only(findall(model -> model.key == "pair", models))
    relative_output = [100 * share.total / shares[reference_index].total for share in shares]
    ranking = ranking_data(data, models)
    conditions = ranking.conditions
    if difficulties !== nothing
        requested = Set(Float64.(difficulties))
        available = Set(c["delta"] for c in conditions if c["rho"] != 1)
        !isempty(requested) && issubset(requested, available) ||
            throw(ArgumentError("select at least one retained difficulty level"))
        conditions = filter(c -> c["rho"] == 1 || c["delta"] in requested, conditions)
    end
    conditions = sort(conditions; by=c -> (c["rho"], c["delta"]))
    regime_offsets = length(conditions) == 1 ? [0.0] :
        collect(range(-0.24, 0.24; length=length(conditions)))
    regime_positions = [index .+ regime_offsets for index in eachindex(models)]
    channels = (
        (key=:broker, label="Brokered matches", color=PUB_BROKER, marker=:circle),
        (key=:self, label="Self-search matches", color=PUB_PRINCIPAL, marker=:rect),
    )
    fig = Figure(; size=(1540, 1120))
    ys = collect(length(models):-1:1)
    labels = [m.label for m in models]
    common = (;
        subtitlesize=TICK_FS,
        subtitlecolor=:gray40,
        subtitlegap=7,
        xgridvisible=true,
        xgridcolor=(:black, 0.065),
        ygridvisible=false,
        yticksize=0,
        leftspinevisible=false,
    )
    a = Axis(
        fig[1, 1];
        title="A. Rank correlation at baseline",
        xlabel="Late-mean rank correlation",
        xticks=0:0.2:1,
        yticks=(reverse(ys), reverse(labels)),
        limits=((0, 1), (0.45, length(models) + 0.8)),
        common...,
    )
    for (model, y) in zip(models, ys)
        values = ranking.baseline[model.key]
        lines!(
            a,
            [values.principal.mean, values.broker.mean],
            [y, y];
            color=:gray65,
            linewidth=2,
        )
        for (value, color, marker) in
            ((values.broker, PUB_BROKER, :circle), (values.principal, PUB_PRINCIPAL, :rect))
            rangebars!(
                a,
                [y],
                [value.lower],
                [value.upper];
                direction=:x,
                color,
                linewidth=2,
                whiskerwidth=10,
            )
            scatter!(
                a,
                [value.mean],
                [y];
                color,
                marker,
                markersize=12,
                strokecolor=:white,
                strokewidth=0.8,
            )
            text!(
                a,
                value.mean,
                y + 0.15;
                text=@sprintf("%.2f", value.mean),
                color,
                fontsize=TICK_FS,
                font=:bold,
                align=(
                    if abs(values.broker.mean - values.principal.mean) < 0.12
                        value.mean == min(values.broker.mean, values.principal.mean) ?
                            :right : :left
                    else
                        :center
                    end,
                    :bottom,
                ),
            )
        end
    end
    axislegend(
        a,
        [
            MarkerElement(; color=PUB_BROKER, marker=:circle, markersize=12),
            MarkerElement(; color=PUB_PRINCIPAL, marker=:rect, markersize=12),
        ],
        ["Broker", "Principals"];
        position=:lt,
        orientation=:horizontal,
        framevisible=false,
        labelsize=TICK_FS,
        patchsize=(18, 14),
        padding=(2, 2, 2, 2),
        margin=(2, 2, 2, 2),
    )
    b = Axis(
        fig[1, 2];
        title="B. Sources of principals’ net output\nat baseline",
        xlabel="Share of net output (%)",
        xticks=0:25:100,
        yticks=(reverse(ys), reverse(labels)),
        limits=((0, 100), (0.45, length(models) + 0.8)),
        common...,
    )
    text!(
        b,
        100,
        ys[reference_index] - 0.18;
        text="% of reference output",
        color=:gray40,
        fontsize=FOOTER_FS,
        align=(:right, :top),
    )
    for (index, (share, y)) in enumerate(zip(shares, ys))
        lines!(
            b,
            100 .* [share.self.mean, share.broker.mean],
            [y, y];
            color=:gray75,
            linewidth=1.5,
        )
        for channel in channels
            value = getproperty(share, channel.key)
            percentage = 100 * value.mean
            0 <= value.lower <= value.upper <= 1 ||
                error("baseline share interval would be clipped")
            rangebars!(
                b,
                [y],
                [100 * value.lower],
                [100 * value.upper];
                direction=:x,
                color=channel.color,
                linewidth=2,
                whiskerwidth=10,
            )
            scatter!(
                b,
                [percentage],
                [y];
                color=channel.color,
                marker=channel.marker,
                markersize=12,
                strokecolor=:white,
                strokewidth=0.8,
            )
            text!(
                b,
                percentage,
                y + 0.15;
                text="$(round(Int, percentage))%",
                color=channel.color,
                fontsize=TICK_FS,
                font=:bold,
                align=(
                    if percentage < 10
                        :left
                    elseif percentage > 90
                        :right
                    elseif abs(share.broker.mean - share.self.mean) < 0.25
                        (channel.key == :self ? :right : :left)
                    else
                        :center
                    end,
                    :bottom,
                ),
            )
        end
        if index != reference_index
            text!(
                b,
                100,
                y - 0.18;
                text="$(round(Int, relative_output[index]))%",
                color=:gray40,
                fontsize=FOOTER_FS,
                align=(:right, :top),
            )
        end
    end
    axislegend(
        b,
        [MarkerElement(; color=c.color, marker=c.marker, markersize=12) for c in channels],
        [c.label for c in channels];
        position=:lt,
        orientation=:horizontal,
        framevisible=false,
        labelsize=TICK_FS,
        patchsize=(18, 14),
        padding=(2, 2, 2, 2),
        margin=(2, 2, 2, 2),
    )
    c = Axis(
        fig[2, 1];
        title="C. Ranking advantage across regimes",
        ylabel="Late-mean ranking advantage\n(broker minus principal)",
        yticks=-1:0.25:1,
        xticks=(1:length(models), labels),
        limits=((0.5, length(models) + 0.5), (-1, 1.25)),
        subtitlesize=TICK_FS,
        subtitlecolor=:gray40,
        subtitlegap=7,
        ygridvisible=true,
    )
    hlines!(c, [0]; color=:gray50, linestyle=:dash, linewidth=1.2)
    quality = [condition["rho"] == 1 for condition in conditions]
    rhos = sort!(unique([condition["rho"] for condition in conditions]))
    regime_colors = [PUB_RHO_COLORS[condition["rho"]] for condition in conditions]
    rank_intervals = [
        [
            monte_carlo_interval(
                condition["rank_gaps"][model.key]; level=data["meta"]["interval_level"]
            ) for condition in conditions
        ] for model in models
    ]
    rank_means = [getproperty.(intervals, :mean) for intervals in rank_intervals]
    connect_channel_regimes!(c, regime_positions, rank_means)
    for (values, intervals, positions) in zip(rank_means, rank_intervals, regime_positions)
        for (position, interval, color) in zip(positions, intervals, regime_colors)
            -1 < interval.lower <= interval.upper < 1.25 ||
                error("ranking interval would be clipped")
            rangebars!(
                c,
                [position],
                [interval.lower],
                [interval.upper];
                color,
                linewidth=1.5,
                whiskerwidth=7,
            )
        end
        scatter!(
            c,
            positions[.!quality],
            values[.!quality];
            color=regime_colors[.!quality],
            markersize=9,
            strokecolor=:white,
            strokewidth=0.4,
        )
        scatter!(
            c,
            positions[quality],
            values[quality];
            marker=:diamond,
            color=:white,
            strokecolor=regime_colors[quality],
            strokewidth=1.8,
            markersize=14,
        )
    end
    channel_regime_legend!(c, rhos)

    d = Axis(
        fig[2, 2];
        title="D. Sources of principals’ net output\nacross regimes",
        ylabel="Share from brokered matches (%)",
        yticks=0:25:100,
        xticks=(1:length(models), labels),
        limits=((0.5, length(models) + 0.5), (0, 125)),
        subtitlesize=TICK_FS,
        subtitlecolor=:gray40,
        subtitlegap=7,
        ygridvisible=true,
    )
    output_intervals = [
        [
            channel_output_shares(
                condition["channel_values"][model.key], condition["N"];
                level=data["meta"]["interval_level"],
            ).broker for condition in conditions
        ] for model in models
    ]
    output_shares = [getproperty.(intervals, :mean) for intervals in output_intervals]
    connect_channel_regimes!(
        d, regime_positions, [100 .* values for values in output_shares]
    )
    for (values, intervals, positions) in
        zip(output_shares, output_intervals, regime_positions)
        for (position, interval, color) in zip(positions, intervals, regime_colors)
            0 <= 100 * interval.lower <= 100 * interval.upper < 125 ||
                error("output-share interval would be clipped")
            rangebars!(
                d,
                [position],
                [100 * interval.lower],
                [100 * interval.upper];
                color,
                linewidth=1.5,
                whiskerwidth=7,
            )
        end
        scatter!(
            d,
            positions[.!quality],
            100 .* values[.!quality];
            color=regime_colors[.!quality],
            markersize=9,
            strokecolor=:white,
            strokewidth=0.4,
        )
        scatter!(
            d,
            positions[quality],
            100 .* values[quality];
            marker=:diamond,
            color=:white,
            strokecolor=regime_colors[quality],
            strokewidth=1.8,
            markersize=14,
        )
    end
    channel_regime_legend!(d, rhos)

    colsize!(fig.layout, 1, Relative(0.5))
    rowsize!(fig.layout, 1, Relative(0.48))
    rowgap!(fig.layout, 52)
    colgap!(fig.layout, 74)
    out = joinpath(@__DIR__, "..", "..", "output", "main", "figures")
    output = isnothing(output_path) ?
        joinpath(out, "information_sources_net_output_channels.png") : output_path
    mkpath(dirname(output))
    save(output, fig; px_per_unit=2)
    println("Saved $output")
    println("C–D: $(length(conditions)) regimes per condition")
    for (model, share) in zip(models, shares)
        println(model.key, ": ", share)
    end
    return output
end


end
