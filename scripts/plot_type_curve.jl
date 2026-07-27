"""
    plot_type_curve.jl

Visualize the agent type curve on the unit sphere. Samples smooth curve
positions and noisy agent types, then plots three orthogonal projections
with depth color-coding.

Adapted from v0.1 plot_firm_curve_3d.jl for the unimodal agent model.

Usage: julia --project scripts/plot_type_curve.jl
"""

using BrokerageABM
using BrokerageABM: generate_curve_geometry, curve_point
using CairoMakie
using LinearAlgebra: norm
using StableRNGs: StableRNG

const OUTDIR = joinpath(@__DIR__, "..", "data", "figures")
mkpath(OUTDIR)

# Use d=3 for visualization (projections are trivial)
d = 3; s = 3; sigma_x = 0.5
rng = StableRNG(42)
geo = generate_curve_geometry(d, s, rng)

# Sample smooth curve points
n_curve = 500
t_vals = range(0, 1; length=n_curve)
curve_pts = [curve_point(t, geo) for t in t_vals]
cx = [p[1] for p in curve_pts]
cy = [p[2] for p in curve_pts]
cz = [p[3] for p in curve_pts]

# Sample noisy agent types
n_agents = 100
sigma_per_dim = sigma_x / sqrt(d)
agent_pts = Vector{Vector{Float64}}(undef, n_agents)
for i in 1:n_agents
    t_i = rand(rng)
    cp = curve_point(t_i, geo)
    noisy = cp .+ sigma_per_dim .* randn(rng, d)
    n = norm(noisy)
    agent_pts[i] = n > 1e-12 ? noisy ./ n : noisy
end
ax_pts = [p[1] for p in agent_pts]
ay_pts = [p[2] for p in agent_pts]
az_pts = [p[3] for p in agent_pts]

# ── Three orthogonal projections with depth color ─────────────────────────

fig = Figure(size=(1200, 400))

projections = [
    ("XY", cx, cy, ax_pts, ay_pts, cz, az_pts),
    ("XZ", cx, cz, ax_pts, az_pts, cy, ay_pts),
    ("YZ", cy, cz, ay_pts, az_pts, cx, ax_pts),
]

for (col, (name, c1, c2, a1, a2, c_depth, a_depth)) in enumerate(projections)
    ax = Axis(fig[1, col]; title="$name projection", xlabel=string(name[1]), ylabel=string(name[2]),
              aspect=1)

    # Curve (colored by depth)
    lines!(ax, c1, c2; color=c_depth, colormap=:viridis, linewidth=1.5)

    # Agent types (colored by depth)
    scatter!(ax, a1, a2; color=a_depth, colormap=:viridis, markersize=6, strokewidth=0.5,
             strokecolor=:gray40)
end

Colorbar(fig[1, 4]; colormap=:viridis, label="Depth (orthogonal axis)", width=15)

save(joinpath(OUTDIR, "type_curve_3d.png"), fig)
println("Saved: $(joinpath(OUTDIR, "type_curve_3d.png"))")
