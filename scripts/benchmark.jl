"""
    benchmark.jl

Performance benchmarking and profiling for BrokerageABM v0.3.
Each benchmark sample uses fresh setup so mutating workloads do not decay into
no-op timings across repetitions.

Usage: julia --project --threads=auto scripts/benchmark.jl
"""

using BrokerageABM
using BrokerageABM: step_period!, train_agent_nn!, train_broker_nn!,
                          process_entry_exit!, update_cached_network_measures!,
                          predict_nn!, init_neural_net, NNGradBuffers,
                          train_step!, agent_hidden_width, broker_hidden_width,
                          broker_pair_feature_dim
using BenchmarkTools
using StableRNGs: StableRNG
using Profile
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3.0

function summary_line(label::String, b::BenchmarkTools.Trial)
    m = median(b)
    @printf("  %-42s  median=%8.3f ms   allocs=%8d   bytes=%10d\n",
            label, m.time / 1e6, m.allocs, m.memory)
    flush(stdout)
end

function warmed_state(params::ModelParams; burnin::Int = 10)
    state = initialize_model(params)
    for _ in 1:burnin
        step_period!(state)
    end
    return state
end

function state_with_pending_broker_training(params::ModelParams; burnin::Int = 10,
                                            max_extra_steps::Int = 25)
    state = warmed_state(params; burnin=burnin)
    extra_steps = 0
    while state.broker.n_new_obs == 0 && extra_steps < max_extra_steps
        step_period!(state)
        extra_steps += 1
    end
    state.broker.n_new_obs > 0 || error("Could not construct benchmark state with pending broker updates")
    return state
end

println("=" ^ 80)
println("BENCHMARK: BrokerageABM v0.3 (post-calibration)")
println("=" ^ 80)

# ─────────────────────────────────────────────────────────────────────────────
# Warm-up and baseline full simulation
# ─────────────────────────────────────────────────────────────────────────────

params = default_params(; N=500, T=50, seed=42)
println("\nWarming up (N=500, T=50)..."); flush(stdout)
run_simulation(params)

println("\n── Full simulation (N=500, T=50) ──"); flush(stdout)
b = @benchmark run_simulation($params) samples=3 evals=1
summary_line("run_simulation", b)

# ─────────────────────────────────────────────────────────────────────────────
# Per-step
# ─────────────────────────────────────────────────────────────────────────────

state = initialize_model(params)
for _ in 1:10; step_period!(state); end  # burn-in to steady state

println("\n── step_period! (after 10-period burn-in) ──")
b_step = @benchmark step_period!(state) setup=(state = warmed_state($params; burnin=10)) samples=10 evals=1
summary_line("step_period!", b_step)

# ─────────────────────────────────────────────────────────────────────────────
# Per-step subsystems
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Subsystem timings (single call, median over repeats) ──")

# Agent NN training (all agents)
b_agent_train = @benchmark begin
    for a in state.agents
        a.history_count > 0 && a.n_new_obs > 0 && train_agent_nn!(a, state.params)
    end
end setup=(state = warmed_state($params; burnin=10)) samples=10 evals=1
summary_line("train all agents (N=500)", b_agent_train)

# Broker NN training
b_broker_train = @benchmark train_broker_nn!(state.broker, state.params) setup=(state = state_with_pending_broker_training($params; burnin=10)) samples=10 evals=1
summary_line("train broker", b_broker_train)

# Entry/exit
using Random: MersenneTwister
rng_b = MersenneTwister(99)
b_ee = @benchmark process_entry_exit!(state, $rng_b) setup=(state = warmed_state($params; burnin=10)) samples=20 evals=1
summary_line("process_entry_exit! (N=500)", b_ee)

# Network measures
b_net = @benchmark update_cached_network_measures!(state) setup=(state = warmed_state($params; burnin=10)) samples=5 evals=1
summary_line("update_cached_network_measures!", b_net)

# ─────────────────────────────────────────────────────────────────────────────
# Micro-benchmarks: NN primitives
# ─────────────────────────────────────────────────────────────────────────────

println("\n── NN primitives ──")

# Agent prediction
rng = StableRNG(99)
d = 8
h_agent = agent_hidden_width(d)
h_broker = broker_hidden_width(d)
d_broker = broker_pair_feature_dim(d)
nn_a = init_neural_net(d, h_agent, rng)
buf_a = zeros(h_agent)
z_a = randn(rng, d)
b_pred_a = @benchmark predict_nn!($nn_a, $buf_a, $z_a) samples=1000 evals=1000
summary_line("predict_nn! (agent, d=8, h=$(h_agent))", b_pred_a)

# Broker prediction
nn_b_nn = init_neural_net(d_broker, h_broker, rng)
buf_b = zeros(h_broker)
z_b = randn(rng, d_broker)
b_pred_b = @benchmark predict_nn!($nn_b_nn, $buf_b, $z_b) samples=1000 evals=1000
summary_line("predict_nn! (broker, features=$(d_broker), h=$(h_broker))", b_pred_b)

# Agent training step (20 obs)
grad_a = NNGradBuffers(nn_a)
X_a = randn(rng, d, 20); q_a = randn(rng, 20)
b_step_a = @benchmark train_step!($nn_a, $grad_a, $X_a, $q_a, 0.03) samples=30 evals=1
summary_line("train_step! (agent, n=20)", b_step_a)

# Agent training step (200 obs, near typical window)
X_a200 = randn(rng, d, 200); q_a200 = randn(rng, 200)
b_step_a200 = @benchmark train_step!($nn_a, $grad_a, $X_a200, $q_a200, 0.03) samples=30 evals=1
summary_line("train_step! (agent, n=200)", b_step_a200)

# Broker training step (400 observed pairs)
grad_b = NNGradBuffers(nn_b_nn)
X_b400 = randn(rng, d_broker, 400); q_b400 = randn(rng, 400)
b_step_b400 = @benchmark train_step!($nn_b_nn, $grad_b, $X_b400, $q_b400, 0.03) samples=20 evals=1
summary_line("train_step! (broker, n=400)", b_step_b400)

# Broker training step (2000 obs, full window)
X_b2000 = randn(rng, d_broker, 2000); q_b2000 = randn(rng, 2000)
b_step_b2000 = @benchmark train_step!($nn_b_nn, $grad_b, $X_b2000, $q_b2000, 0.03) samples=10 evals=1
summary_line("train_step! (broker, n=2000)", b_step_b2000)

# ─────────────────────────────────────────────────────────────────────────────
# Allocation report: 3 consecutive full runs
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Full simulation with @time (allocations shown) ──")
@time run_simulation(params)
@time run_simulation(params)
@time run_simulation(params)

# ─────────────────────────────────────────────────────────────────────────────
# Profiling: 5 full simulations, combined profile
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Profile (5 full simulations) ──")
Profile.clear()
Profile.init(; n=10^7, delay=0.001)
@profile begin
    for _ in 1:5
        run_simulation(params)
    end
end

# Print top entries
println("\nTop hot spots (by self-time, min 2% of samples):")
Profile.print(IOContext(stdout, :displaysize => (30, 120));
              format=:flat, sortedby=:count, mincount=30, noisefloor=2.0)
