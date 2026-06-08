"""
    agent_data_binding.jl

Test whether agents are data-bound by exploiting heterogeneity in agent data volume.
Every agent shares the same NN width (2d) and the same training budget, so the variation
in how many matches an agent accumulates is a natural experiment that holds capacity and
compute fixed: if holdout generalization rises with an agent's own data, agents are
data-bound. Runs one baseline simulation, then for each agent measures its holdout
skill (its NN vs the true noiseless signal over random partners) against its training
data volume (history_count). The broker (width 8d, trained on all matches) is the
reference point.
"""

using TransientBrokerage
using TransientBrokerage: match_signal, Q_OFFSET, predict_nn!, agent_hidden_width, broker_hidden_width
using Statistics: mean, var
using StatsBase: corspearman
using StableRNGs: StableRNG
using JLD2

p = default_params(; rho=0.5, N=1000, T=200, seed=1)
state, df = run_simulation(p)
env = state.env; agents = state.agents
rng = StableRNG(12345)
N = length(agents); K = 40

data = Int[]; rnk = Float64[]; r2 = Float64[]
for a in agents
    a.history_count >= 5 || continue
    preds = Float64[]; truth = Float64[]
    for _ in 1:K
        j = rand(rng, 1:N)
        push!(preds, predict_nn!(a.nn, a.predict_buf, agents[j].type))
        push!(truth, Q_OFFSET + match_signal(a.type, agents[j].type, env))
    end
    v = var(truth); v > 1e-8 || continue
    push!(data, a.history_count)
    push!(rnk, corspearman(preds, truth))
    push!(r2, 1 - mean((preds .- truth) .^ 2) / v)
end

btail(c) = mean(df[df.period .> 30, c])
n = length(data); order = sortperm(data)

println("agent NN width = ", agent_hidden_width(p), " ; broker NN width = ", broker_hidden_width(p))
println("n agents (history>=5) = ", n)
println("cor(data, holdout rank) = ", round(corspearman(data, rnk); digits=3),
        " ; cor(data, holdout R2) = ", round(corspearman(data, r2); digits=3))
println("\n=== agent holdout by data-volume quintile (capacity + compute held fixed) ===")
for i in 1:5
    idx = order[(round(Int, (i - 1) * n / 5) + 1):round(Int, i * n / 5)]
    println("  Q$i  mean data=", round(Int, mean(data[idx])),
            "  holdout rank=", round(mean(rnk[idx]); digits=2),
            "  R2=", round(mean(r2[idx]); digits=2), "  (n=", length(idx), ")")
end
println("\nBROKER (width $(broker_hidden_width(p)), ~all matches): rank=",
        round(btail(:broker_holdout_rank); digits=2), "  R2=", round(btail(:broker_holdout_r2); digits=2))
println("pop-mean agent (df):                       rank=",
        round(btail(:agent_holdout_rank); digits=2), "  R2=", round(btail(:agent_holdout_r2); digits=2))

mkpath(joinpath(@__DIR__, "_results"))
jldsave(joinpath(@__DIR__, "_results", "agent_data_binding.jld2");
    data=data, rank=rnk, r2=r2,
    brk_rank=btail(:broker_holdout_rank), brk_r2=btail(:broker_holdout_r2),
    agent_width=agent_hidden_width(p), broker_width=broker_hidden_width(p))
println("DONE")
