"""
    quick_diagnostic.jl

Quick single-seed run at default params, producing a multi-panel diagnostic figure.
Use after any code change to eyeball the dynamics.

Usage: julia --project --threads=auto scripts/quick_diagnostic.jl
"""

using BrokerageABM
using CairoMakie
using Statistics: mean

const OUTDIR = joinpath(@__DIR__, "..", "data", "figures")
mkpath(OUTDIR)

function nanmean_or_nan(v)
    total = 0.0
    n = 0
    for x in v
        isnan(x) && continue
        total += x
        n += 1
    end
    return n == 0 ? NaN : total / n
end

println("Running quick diagnostic (T=200, default params)...")
@time state, df = run_simulation(default_params())

periods = df.period
brokered_with_origin = df.access_count .+ df.assessment_count
access_frac = [
    t > 0 ? df.access_count[i] / t : NaN for (i, t) in enumerate(brokered_with_origin)
]

fig = Figure(; size=(1400, 1600))

ax1 = Axis(fig[1, 1]; title="Outsourcing Rate", ylabel="Rate")
lines!(ax1, periods, df.outsourcing_rate; color=:steelblue)

ax2 = Axis(fig[1, 2]; title="Holdout Prediction Quality (R²)", ylabel="R²")
lines!(ax2, periods, df.broker_holdout_r2; color=:crimson, label="Broker")
lines!(ax2, periods, df.agent_holdout_r2; color=:steelblue, label="Agent")
axislegend(ax2; position=:rb)

ax3 = Axis(fig[2, 1]; title="Betweenness Centrality (Broker)", ylabel="C_B(broker)")
lines!(ax3, periods, df.betweenness; color=:steelblue)

ax4 = Axis(fig[2, 2]; title="Mean Output by Channel", ylabel="Output")
lines!(ax4, periods, df.q_self_mean; color=:steelblue, label="Self-search")
lines!(ax4, periods, df.q_broker_mean; color=:crimson, label="Broker")
axislegend(ax4; position=:rb)

ax5 = Axis(fig[3, 1]; title="Access Fraction of Brokered Matches", ylabel="Fraction")
lines!(ax5, periods, access_frac; color=:darkorange)

ax6 = Axis(fig[3, 2]; title="Broker Access, Roster & History", ylabel="Count")
lines!(ax6, periods, Float64.(df.broker_access_size); color=:crimson, label="Access set")
lines!(
    ax6,
    periods,
    Float64.(df.roster_size);
    color=:steelblue,
    linestyle=:dash,
    label="Standing roster",
)
lines!(ax6, periods, Float64.(df.broker_history_size); color=:darkorange, label="History")
axislegend(ax6; position=:rb)

ax7 = Axis(fig[4, 1]; title="Matches per Period", xlabel="Period", ylabel="Count")
lines!(ax7, periods, Float64.(df.n_self_matches); color=:steelblue, label="Self")
lines!(ax7, periods, Float64.(df.n_broker_matches); color=:crimson, label="Broker")
axislegend(ax7; position=:rt)

ax8 = Axis(fig[4, 2]; title="R² Gap (Broker - Agent)", xlabel="Period", ylabel="ΔR²")
lines!(ax8, periods, df.r2_gap; color=:purple)
hlines!(ax8, [0.0]; color=:gray, linestyle=:dash)

outpath = joinpath(OUTDIR, "quick_diagnostic.png")
save(outpath, fig)
println("Saved: $outpath")

# Summary stats
println("\n=== Summary (last 50 periods) ===")
tail = df[max(1, end - 49):end, :]
println("  Outsourcing rate: $(round(mean(tail.outsourcing_rate), digits=3))")
println("  Broker holdout R²: $(round(nanmean_or_nan(tail.broker_holdout_r2), digits=3))")
println("  Agent holdout R²: $(round(nanmean_or_nan(tail.agent_holdout_r2), digits=3))")
println("  R² gap: $(round(nanmean_or_nan(tail.r2_gap), digits=3))")
println("  Betweenness: $(round(mean(tail.betweenness), digits=4))")
println("  Mean self output: $(round(nanmean_or_nan(tail.q_self_mean), digits=3))")
println("  Mean broker output: $(round(nanmean_or_nan(tail.q_broker_mean), digits=3))")
println("  Broker access set: $(round(mean(tail.broker_access_size), digits=0))")
println("  Roster: $(round(mean(tail.roster_size), digits=0))")
println("  Broker history: $(round(mean(tail.broker_history_size), digits=0))")
println("  Total matches/period: $(round(mean(tail.n_total_matches), digits=0))")
