"""
    sweep_plot.jl  (one task = one OAT cell, or one phase pair)

Aggregate per-seed shards from `sweep_run.jl` and render sweep figures without
rerunning simulations.

Plot-job kinds (from `manifest.jld2` `plot_jobs`):
  * `oat_cell`: aggregate one OAT cell and render its dynamics and network figures.
  * `phase_pair`: aggregate one two-parameter grid and render its metric heatmaps.

Usage:
  BROKERAGE_ABM_SWEEP_DIR=... SLURM_ARRAY_TASK_ID=<i> julia --project --threads=auto scripts/sweep/sweep_plot.jl
  BROKERAGE_ABM_SWEEP_DIR=... julia --project --threads=auto scripts/sweep/sweep_plot.jl <i>
"""

Threads.nthreads() == 1 && @warn "Running single-threaded; start Julia with --threads=auto"

include(joinpath(@__DIR__, "sweep_config.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

# Plotting uses saved DataFrames and does not load BrokerageABM or Enzyme.
using DataFrames: DataFrame
using Statistics: mean
using JLD2: jldsave, jldopen

include(joinpath(@__DIR__, "shard_validation.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Shard loading
# ─────────────────────────────────────────────────────────────────────────────

"""Load all planned current seed shards in `celldir`."""
function load_cell(celldir, expected_provenance, expected_seeds)
    mdfs = DataFrame[]
    final_degs = Vector{Int}[]
    seeds = Int[]
    config = nothing
    prov = nothing
    for s in expected_seeds
        path = joinpath(celldir, "seed_$(s).jld2")
        isfile(path) || continue
        if !shard_is_current(path, expected_provenance)
            @warn "ignoring shard with stale or mixed provenance" path
            continue
        end
        try
            jldopen(path, "r") do f
                push!(mdfs, f["df"])
                push!(final_degs, f["final_agent_degrees"])
                push!(seeds, s)
                if config === nothing
                    config = f["config"]
                    prov = Dict(
                        "git_commit" => f["git_commit"],
                        "julia_version" => f["julia_version"],
                        "pkg_manifest_hash" => f["pkg_manifest_hash"],
                        "manifest_hash" => f["manifest_hash"],
                        "schema_version" => f["schema_version"],
                    )
                end
            end
        catch err
            @warn "failed to read shard" path err
        end
    end
    return (mdfs=mdfs, final_degs=final_degs, config=config, prov=prov, seeds=seeds)
end

"""Return whether every planned seed shard exists with current provenance."""
function shards_complete(celldir, expected_provenance, expected_seeds)
    all(
        seed ->
            shard_is_current(joinpath(celldir, "seed_$(seed).jld2"), expected_provenance),
        expected_seeds,
    )
end

const GRID_METADATA_KEYS = (
    :axis, :key, :value, :pair, :xkey, :xval, :xi, :ykey, :yval, :yi
)

"""Seed-independent configuration of the canonical realized result."""
function realized_config(config)
    cfg = Dict{String,Any}(string(k) => v for (k, v) in config)
    pop!(cfg, "seed", nothing)
    for key in GRID_METADATA_KEYS
        pop!(cfg, string(key), nothing)
    end
    return cfg
end

"""Grid-coordinate metadata paired with the canonical realized-result config."""
function grid_config(config, grid)
    cfg = realized_config(config)
    cfg["kind"] = haskey(grid, :pair) ? "phase" : "oat"
    cfg["reldir"] = grid[:reldir]
    cfg["result_reldir"] = grid[:result_reldir]
    cfg["condition_index"] = grid[:condition_index]
    for (key, value) in grid[:resolved_params]
        cfg[string(key)] = value
    end
    for key in GRID_METADATA_KEYS
        haskey(grid, key) && (cfg[string(key)] = grid[key])
    end
    return cfg
end

"""Write a grid-coordinate aggregate that references one canonical result set."""
function write_cell_data(celldir, cell, grid)
    mkpath(celldir)
    realized = realized_config(cell.config)
    jldsave(
        joinpath(celldir, "data.jld2");
        mdfs=cell.mdfs,
        final_agent_degrees=cell.final_degs,
        seeds=cell.seeds,
        config=grid_config(cell.config, grid),
        realized_config=realized,
        requested_params=Dict(
            string(key) => value for (key, value) in grid[:resolved_params]
        ),
        provenance=cell.prov,
        grid_reldir=grid[:reldir],
        result_reldir=grid[:result_reldir],
        condition_index=grid[:condition_index],
        schema_version=SWEEP_SCHEMA_VERSION,
    )
end

maxabs_or_one(M) = (
    m=0.0;
    found=false;
    for x in M
        ;
        isnan(x) && continue;
        m = max(m, abs(x));
        found = true;
    end;
    found ? m : 1.0
)

# ─────────────────────────────────────────────────────────────────────────────
# Dynamics panel (5x4)
# ─────────────────────────────────────────────────────────────────────────────

function plot_dynamics(mdfs, suptitle, outpath; T_burn=SWEEP_T_BURN, window=20)
    isempty(mdfs) && return nothing
    n_seeds = length(mdfs)
    periods = mdfs[1].period
    T = last(periods);
    xlims = (first(periods), T);
    akw = ax_kw(T)
    pm!(ax, fn; kw...) = plot_metric!(ax, periods, mdfs, fn; window=window, kw...)

    fig = Figure(; size=(1500, 1100), figure_padding=(5, 15, 5, 5))
    all_axes = Axis[]
    newax(pos; kw...) = (a=Axis(pos; kw...); push!(all_axes, a); a)
    Label(
        fig[0, 1:4],
        suptitle;
        fontsize=SUPTITLE_FS,
        font=:bold,
        halign=:center,
        tellwidth=false,
    )
    AGT="Agents (mean)";
    BRK="Broker (mean)"

    Label(
        fig[1, 0],
        "Market";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )
    ax = newax(
        fig[1, 1];
        title="Outsourcing rate (positions)",
        ylabel="Rate",
        limits=(xlims, (-0.02, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.outsourcing_rate; color=COL_BROKER)
    ax = newax(
        fig[1, 2];
        title="Matches by channel",
        ylabel="Count",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> Float64.(mdf.n_self_matches); label="Self", color=COL_AGENT)
    pm!(ax, mdf -> Float64.(mdf.n_broker_matches); label="Broker", color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)
    ax = newax(
        fig[1, 3];
        title="Total demand & matches",
        ylabel="Count",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> Float64.(mdf.total_demand); label="Demand", color=COL_DIAG)
    pm!(ax, mdf -> Float64.(mdf.n_total_matches); label="Matches", color=COL_AGENT)
    axislegend(ax; position=:rt, LEG_KW...)
    ax = newax(
        fig[1, 4];
        title="Broker access and roster",
        ylabel="Count",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> Float64.(mdf.broker_access_size); label="Access set", color=COL_BROKER)
    pm!(
        ax,
        mdf -> Float64.(mdf.roster_size);
        label="Roster",
        color=COL_REFERENCE,
        linestyle=:dash,
    )
    axislegend(ax; position=:rb, LEG_KW...)

    Label(
        fig[2, 0],
        "Selected";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )
    ax = newax(
        fig[2, 1];
        title="Selected rank corr.",
        ylabel="Spearman ρ",
        limits=(xlims, (0, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.agent_selected_rank; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_rank; label=BRK, color=COL_BROKER)
    axislegend(ax; position=:rb, LEG_KW...)
    ax = newax(
        fig[2, 2]; title="Selected R²", ylabel="R²", limits=(xlims, (nothing, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_selected_r2; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_r2; label=BRK, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8);
    axislegend(ax; position=:rb, LEG_KW...)
    ax = newax(
        fig[2, 3]; title="Selected RMSE", ylabel="RMSE", limits=(xlims, (0, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_selected_rmse; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_rmse; label=BRK, color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)
    ax = newax(
        fig[2, 4]; title="Selected bias", ylabel="Bias", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.agent_selected_bias; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_selected_bias; label=BRK, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8);
    axislegend(ax; position=:rt, LEG_KW...)

    Label(
        fig[3, 0],
        "Holdout";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )
    ax = newax(
        fig[3, 1];
        title="Holdout rank corr.",
        ylabel="Spearman ρ",
        limits=(xlims, (0, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> mdf.agent_holdout_rank; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_rank; label=BRK, color=COL_BROKER)
    axislegend(ax; position=:rb, LEG_KW...)
    ax = newax(
        fig[3, 2]; title="Holdout R²", ylabel="R²", limits=(xlims, (nothing, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_holdout_r2; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_r2; label=BRK, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8);
    axislegend(ax; position=:rb, LEG_KW...)
    ax = newax(
        fig[3, 3]; title="Holdout RMSE", ylabel="RMSE", limits=(xlims, (0, 1.02)), akw...
    )
    pm!(ax, mdf -> mdf.agent_holdout_rmse; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_rmse; label=BRK, color=COL_BROKER)
    axislegend(ax; position=:rt, LEG_KW...)
    ax = newax(
        fig[3, 4]; title="Holdout bias", ylabel="Bias", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.agent_holdout_bias; label=AGT, color=COL_AGENT)
    pm!(ax, mdf -> mdf.broker_holdout_bias; label=BRK, color=COL_BROKER)
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8);
    axislegend(ax; position=:rt, LEG_KW...)

    Label(
        fig[4, 0],
        "Advantage";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )
    ax = newax(
        fig[4, 1]; title="Holdout rank gap", ylabel="Δ ρ", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.rank_gap; color=COL_GAP);
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    ax = newax(
        fig[4, 2]; title="Holdout R² gap", ylabel="Δ R²", limits=(xlims, nothing), akw...
    )
    pm!(ax, mdf -> mdf.r2_gap; color=COL_GAP);
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    ax = newax(
        fig[4, 3];
        title="Holdout RMSE gap",
        ylabel="Δ RMSE",
        limits=(xlims, nothing),
        akw...,
    )
    pm!(ax, mdf -> mdf.rmse_gap; color=COL_GAP);
    hlines!(ax, [0.0]; color=:gray50, linewidth=0.8)
    ax = newax(
        fig[4, 4];
        title="Access fraction",
        ylabel="Fraction",
        limits=(xlims, (-0.02, 1.02)),
        akw...,
    )
    pm!(ax, mdf -> access_fraction(mdf); color=COL_ACCESS)

    Label(
        fig[5, 0],
        "Dynamics";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )
    ax = newax(
        fig[5, 1];
        title="Broker betweenness",
        xlabel="Period",
        ylabel="Betweenness",
        limits=(xlims, (0, 1.02)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.betweenness; color=COL_BROKER)
    ax = newax(
        fig[5, 2];
        title="Mean output by channel",
        xlabel="Period",
        ylabel="Output",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.q_self_mean; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.q_broker_mean; label="Broker", color=COL_BROKER)
    axislegend(ax; position=:rb, LEG_KW...)
    ax = newax(
        fig[5, 3];
        title="Satisfaction + reputation",
        xlabel="Period",
        ylabel="Satisfaction",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.mean_satisfaction_self; label="Self", color=COL_AGENT)
    pm!(ax, mdf -> mdf.mean_satisfaction_broker; label="Broker", color=COL_BROKER)
    pm!(ax, mdf -> mdf.broker_reputation; label="Reputation", color=COL_REPUTATION)
    axislegend(ax; position=:rb, LEG_KW...)

    for a in all_axes
        ;
        add_burnin!(a, T_burn);
    end
    add_footer!(fig, 6, 1:4; n_seeds=n_seeds, window=window, T_burn=T_burn)
    apply_layout!(fig; n_panel_rows=5, n_panel_cols=4, suptitle_row=0, footer_row=6)
    save(outpath, fig)
    println("  saved $(basename(outpath))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Network statistics figure
# ─────────────────────────────────────────────────────────────────────────────

function plot_network_stats(
    mdfs, final_degs, suptitle, outpath; T_burn=SWEEP_T_BURN, window=20
)
    (isempty(mdfs) || isempty(final_degs)) && return nothing
    n_seeds = length(mdfs)
    periods = mdfs[1].period
    T = last(periods);
    xlims = (first(periods), T);
    akw = ax_kw(T)
    pm!(ax, fn; kw...) = plot_metric!(ax, periods, mdfs, fn; window=window, kw...)

    fig = Figure(; size=(1500, 720), figure_padding=(5, 15, 5, 5))
    time_axes = Axis[]
    newax(pos; kw...) = (a=Axis(pos; kw...); push!(time_axes, a); a)
    Label(
        fig[0, 1:4],
        "Network statistics: $suptitle";
        fontsize=SUPTITLE_FS,
        font=:bold,
        halign=:center,
        tellwidth=false,
    )

    # Row 1: agent degree statistics
    Label(
        fig[1, 0],
        "Agents";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )
    ax = newax(
        fig[1, 1];
        title="Mean and median degree",
        ylabel="Degree",
        limits=(xlims, (0, nothing)),
        akw...,
    )
    pm!(ax, mdf -> mdf.mean_degree; label="Mean", color=COL_DIAG)
    pm!(ax, mdf -> mdf.median_degree; label="Median", color=COL_REFERENCE, linestyle=:dash)
    axislegend(ax; position=:lt, LEG_KW...)
    ax = newax(
        fig[1, 2]; title="Min degree", ylabel="Degree", limits=(xlims, (0, nothing)), akw...
    )
    pm!(ax, mdf -> Float64.(mdf.min_degree); color=COL_DIAG)
    ax = newax(
        fig[1, 3]; title="Max degree", ylabel="Degree", limits=(xlims, (0, nothing)), akw...
    )
    pm!(ax, mdf -> Float64.(mdf.max_degree); color=COL_DIAG)
    pooled = reduce(vcat, final_degs);
    maxd = maximum(pooled)
    hist_ax = Axis(
        fig[1, 4];
        title="Final degree distribution",
        xlabel="Degree",
        ylabel="Agent count",
        limits=((0, maxd + 1), (0, nothing)),
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
    )
    hist!(
        hist_ax,
        pooled;
        bins=0:(maxd + 1),
        color=(COL_DIAG, 0.65),
        strokewidth=0.5,
        strokecolor=:gray35,
    )

    # Row 2: broker betweenness, Burt constraint, and effective size.
    Label(
        fig[2, 0],
        "Broker";
        fontsize=ROW_LABEL_FS,
        font=:bold,
        rotation=π/2,
        tellheight=false,
    )
    ax = newax(
        fig[2, 1];
        title="Broker betweenness",
        xlabel="Period",
        ylabel="Betweenness",
        limits=(xlims, (0, 1.02)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.betweenness; color=COL_BROKER)
    ax = newax(
        fig[2, 2];
        title="Broker Burt constraint",
        xlabel="Period",
        ylabel="Constraint",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.constraint; color=COL_BROKER)
    ax = newax(
        fig[2, 3];
        title="Broker effective size",
        xlabel="Period",
        ylabel="Effective size",
        limits=(xlims, (0, nothing)),
        akw...,
        xlabelsize=LABEL_FS,
    )
    pm!(ax, mdf -> mdf.effective_size; color=COL_BROKER)

    for a in time_axes
        ;
        add_burnin!(a, T_burn);
    end
    add_footer!(fig, 3, 1:4; n_seeds=n_seeds, window=window, T_burn=T_burn)
    apply_layout!(fig; n_panel_rows=2, n_panel_cols=4, suptitle_row=0, footer_row=3)
    save(outpath, fig)
    println("  saved $(basename(outpath))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Late-period metrics and two-parameter grid heatmaps
# ─────────────────────────────────────────────────────────────────────────────

"""Tail-averaged (period > T_burn) steady-state metrics for a grid cell's seeds."""
function steady_state_metrics(mdfs, T_burn)
    isempty(mdfs) && return nothing
    tails = [df[df.period .> T_burn, :] for df in mdfs]
    c = vcat(tails...)
    return (
        r2_gap=nanmean_or_nan(c.r2_gap),
        rank_gap=nanmean_or_nan(c.rank_gap),
        broker_r2=nanmean_or_nan(c.broker_holdout_r2),
        agent_r2=nanmean_or_nan(c.agent_holdout_r2),
        broker_rank=nanmean_or_nan(c.broker_holdout_rank),
        agent_rank=nanmean_or_nan(c.agent_holdout_rank),
        outsourcing=mean(c.outsourcing_rate),
        betweenness=mean(c.betweenness),
        constraint=mean(c.constraint),
        effective_size=mean(c.effective_size),
        q_self=nanmean_or_nan(c.q_self_mean),
        q_broker=nanmean_or_nan(c.q_broker_mean),
    )
end

function extract_metric(results, field)
    [
        (r=results[i, j]; r === nothing ? NaN : getfield(r, field)) for
        i in axes(results, 1), j in axes(results, 2)
    ]
end

function plot_heatmap(
    M,
    title_str,
    outpath;
    xlabel,
    ylabel,
    xvals,
    yvals,
    colormap=:RdBu,
    colorrange=nothing,
    label="",
)
    nx, ny = size(M)
    cr = colorrange === nothing ? (m=maxabs_or_one(M); (-m, m)) : colorrange
    fig = Figure(; size=(720, 520))
    ax = Axis(
        fig[1, 1];
        title=title_str,
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=(1:nx, string.(xvals)),
        yticks=(1:ny, string.(yvals)),
        titlesize=TITLE_FS,
        xlabelsize=LABEL_FS,
        ylabelsize=LABEL_FS,
        xticklabelsize=TICK_FS,
        yticklabelsize=TICK_FS,
    )
    hm = heatmap!(ax, 1:nx, 1:ny, M; colormap=colormap, colorrange=cr)
    Colorbar(fig[1, 2], hm; label=label, labelsize=LABEL_FS, ticklabelsize=TICK_FS)
    save(outpath, fig)
    println("  saved $(basename(outpath))")
end

const HEATMAPS = [
    (:r2_gap, "R² gap (broker − agent)", :RdBu, nothing, "Δ R²"),
    (:rank_gap, "Rank gap (broker − agent)", :RdBu, nothing, "Δ ρ"),
    (:broker_r2, "Broker holdout R²", :viridis, nothing, "R²"),
    (:agent_r2, "Agent holdout R²", :viridis, nothing, "R²"),
    (:broker_rank, "Broker rank corr.", :viridis, (0, 1), "ρ"),
    (:agent_rank, "Agent rank corr.", :viridis, (0, 1), "ρ"),
    (:outsourcing, "Outsourcing rate", :YlOrRd, (0, 1), "Rate"),
    (:betweenness, "Broker betweenness", :viridis, nothing, "C_B"),
    (:constraint, "Broker Burt constraint", :viridis, nothing, "Constraint"),
    (:effective_size, "Broker effective size", :viridis, nothing, "Eff. size"),
    (:q_self, "Mean self output", :viridis, nothing, "q"),
    (:q_broker, "Mean broker output", :viridis, nothing, "q"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Job dispatch
# ─────────────────────────────────────────────────────────────────────────────

function do_oat_cell(sweepdir, job, expected_provenance)
    celldir = joinpath(sweepdir, job[:reldir])
    resultdir = joinpath(sweepdir, job[:result_reldir])
    # Idempotent: skip cells whose figures already exist (BROKERAGE_ABM_REPLOT=1 forces a rebuild).
    primary = joinpath(celldir, "dynamics.png")
    aggregate = joinpath(celldir, "data.jld2")
    if get(ENV, "BROKERAGE_ABM_REPLOT", "0") == "0" &&
        shards_complete(resultdir, expected_provenance, job[:seeds]) &&
        aggregate_is_current(aggregate, expected_provenance) &&
        isfile(primary) &&
        isfile(joinpath(celldir, "network_stats.png"))
        println("SKIP plot (figures exist): $(job[:reldir])")
        return nothing
    end
    cell = load_cell(resultdir, expected_provenance, job[:seeds])
    if isempty(cell.mdfs)
        @warn "no referenced shards found for OAT cell; skipping" celldir resultdir
        return nothing
    end
    println(
        "OAT cell $(job[:reldir]) -> $(job[:result_reldir])  " *
        "($(length(cell.seeds)) seeds: $(cell.seeds))",
    )
    write_cell_data(celldir, cell, job)

    label = "$(job[:axis])=$(job[:value])"
    plot_network_stats(
        cell.mdfs, cell.final_degs, label, joinpath(celldir, "network_stats.png")
    )

    plot_dynamics(cell.mdfs, label, joinpath(celldir, "dynamics.png"))
end

function do_phase_pair(sweepdir, job, expected_provenance)
    pairdir = joinpath(sweepdir, job[:reldir])
    # Idempotent: skip pairs whose summary already exists (BROKERAGE_ABM_REPLOT=1 forces a rebuild).
    summary_path = joinpath(pairdir, "summary.jld2")
    referenced_results = Dict(ref[:result_reldir] => ref[:seeds] for ref in job[:cell_refs])
    all_shards_complete = all(referenced_results) do (rel, seeds)
        shards_complete(joinpath(sweepdir, rel), expected_provenance, seeds)
    end
    if get(ENV, "BROKERAGE_ABM_REPLOT", "0") == "0" &&
        all_shards_complete &&
        aggregate_is_current(summary_path, expected_provenance)
        println("SKIP plot (summary exists): $(job[:pair])")
        return nothing
    end
    xvals = job[:xvals];
    yvals = job[:yvals]
    nx = length(xvals);
    ny = length(yvals)
    results = Matrix{Any}(nothing, nx, ny)
    nfound = 0

    for xi in 1:nx, yi in 1:ny
        ref = only(r for r in job[:cell_refs] if r[:xi] == xi && r[:yi] == yi)
        celldir = joinpath(sweepdir, ref[:reldir])
        resultdir = joinpath(sweepdir, ref[:result_reldir])
        cell = load_cell(resultdir, expected_provenance, ref[:seeds])
        isempty(cell.mdfs) && continue
        nfound += 1
        write_cell_data(celldir, cell, ref)         # Grid view of canonical shards.
        results[xi, yi] = steady_state_metrics(cell.mdfs, SWEEP_T_BURN)
    end
    println("phase $(job[:pair]): $nfound/$(nx * ny) grid points with data")
    nfound == 0 && (@warn "no grid-point data; skipping pair"; return nothing)

    outdir = pairdir
    mkpath(outdir)

    tensors = Dict{Symbol,Matrix{Float64}}()
    for (field, _, _, _, _) in HEATMAPS
        tensors[field] = extract_metric(results, field)
    end
    jldsave(
        joinpath(outdir, "summary.jld2");
        tensors=tensors,
        results=results,
        xkey=job[:xkey],
        xvals=xvals,
        ykey=job[:ykey],
        yvals=yvals,
        cell_refs=job[:cell_refs],
        provenance=Dict(string(key) => value for (key, value) in expected_provenance),
        T_burn=SWEEP_T_BURN,
        schema_version=SWEEP_SCHEMA_VERSION,
    )

    for (field, title_str, cmap, cr, lab) in HEATMAPS
        M = tensors[field]
        plot_heatmap(
            M,
            title_str,
            joinpath(outdir, "$(field).png");
            xlabel=job[:xkey],
            ylabel=job[:ykey],
            xvals=xvals,
            yvals=yvals,
            colormap=cmap,
            colorrange=cr,
            label=lab,
        )
    end
end

function main()
    sweepdir = sweep_dir()
    manifest = joinpath(sweepdir, "manifest.jld2")
    isfile(manifest) || error("manifest not found: $manifest")
    plot_jobs, provenance = jldopen(manifest, "r") do f
        ;
        (f["plot_jobs"], f["prov"])
    end

    id = if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    else
        parse(Int, first(filter(a -> a != "--rerun", ARGS)))
    end
    (0 <= id < length(plot_jobs)) ||
        error("plot id $id out of range 0..$(length(plot_jobs) - 1)")
    job = plot_jobs[id + 1]

    if job[:kind] == "oat_cell"
        do_oat_cell(sweepdir, job, provenance)
    elseif job[:kind] == "phase_pair"
        do_phase_pair(sweepdir, job, provenance)
    else
        error("unknown plot job kind: $(job[:kind])")
    end
    println("plot job [$id] done.")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
