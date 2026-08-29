using Test
using BrokerageABM

include(joinpath(@__DIR__, "..", "scripts", "sweep", "sweep_config.jl"))

@testset "deduplicated sweep design" begin
    cells = build_cells()
    conditions = result_cells(cells)
    entries = build_entries(cells)
    plot_jobs = build_plot_jobs(cells)

    @test SWEEP_SCHEMA_VERSION == 6
    @test SWEEP_T == 500
    @test SWEEP_SEEDS == collect(1:20)
    @test SWEEP_BASELINE_SEEDS == collect(1:20)
    @test SWEEP_LEARNING_MODEL == :nn
    @test SWEEP_SCOPE == :full
    @test ETA_VALS == [0.0, 0.001, 0.01, 0.02, 0.03]
    @test length(cells) == 161
    @test count(c -> c[:kind] == "oat", cells) == 31
    @test count(c -> c[:kind] == "phase", cells) == 130

    @test length(conditions) == 98
    @test length(
        unique(Tuple(c[:resolved_params][key] for key in SWEEP_KEYS) for c in cells)
    ) == 102
    @test length(unique(condition_key(c[:params]) for c in cells)) == 98
    @test count(c -> !c[:is_canonical], cells) == 63
    @test sort(unique(c[:condition_index] for c in cells)) == collect(0:97)
    @test all(conditions) do cell
        p = default_params(; cell[:resolved_params]...)
        expected_delta = p.rho == 1.0 ? SWEEP_BASELINE.delta : p.delta
        condition_key(cell[:params]) == (
            p.rho,
            p.eta,
            p.N,
            p.reservation_frac,
            expected_delta,
            p.k,
            p.roster_frac,
            p.n_strangers,
        )
    end

    canonical_by_dir = Dict(c[:result_reldir] => c for c in conditions)
    @test all(haskey(canonical_by_dir, c[:result_reldir]) for c in cells)
    @test all(
        condition_key(c[:params]) ==
        condition_key(canonical_by_dir[c[:result_reldir]][:params]) for c in cells
    )
    @test all(
        canonical_by_dir[c[:result_reldir]][:grid_index] <= c[:grid_index] for c in cells
    )

    baseline_cells = filter(c -> condition_key(c[:params]) == Tuple(SWEEP_BASELINE), cells)
    @test length(baseline_cells) > 1
    @test all(c[:result_reldir] == BASELINE_RELDIR for c in baseline_cells)

    @test length(entries) == 1960
    @test length(unique((e[:condition_index], e[:seed]) for e in entries)) == 1960
    @test all(e[:seed] in SWEEP_SEEDS for e in entries)
    @test all(c[:seeds] == SWEEP_SEEDS for c in conditions)
    @test all(e[:reldir] in keys(canonical_by_dir) for e in entries)

    @test length(plot_jobs) == 39
    oat_jobs = filter(j -> j[:kind] == "oat_cell", plot_jobs)
    @test length(oat_jobs) == 31
    @test all(haskey(j, :key) for j in oat_jobs)
    phase_jobs = filter(j -> j[:kind] == "phase_pair", plot_jobs)
    @test length(phase_jobs) == 8
    @test sum(length(j[:cell_refs]) for j in phase_jobs) == 130
    @test all(
        ref[:result_reldir] in keys(canonical_by_dir) for job in phase_jobs for
        ref in job[:cell_refs]
    )

    rho_delta = only(j for j in phase_jobs if j[:pair] == "rho_delta")
    @test rho_delta[:xvals] == RHO_OAT
    @test 0.85 in rho_delta[:xvals]
    rho_one_refs = filter(r -> r[:resolved_params][:rho] == 1.0, rho_delta[:cell_refs])
    @test length(rho_one_refs) == 5
    @test all(r[:result_reldir] == "oat/rho=1.0" for r in rho_one_refs)
    @test sort([r[:resolved_params][:delta] for r in rho_one_refs]) == DELTA_VALS

    rho_eta = only(j for j in phase_jobs if j[:pair] == "rho_eta")
    eta_r = only(j for j in phase_jobs if j[:pair] == "eta_r")
    eta_N = only(j for j in phase_jobs if j[:pair] == "eta_N")
    rho_N = only(j for j in phase_jobs if j[:pair] == "rho_N")
    rho_r = only(j for j in phase_jobs if j[:pair] == "rho_r")
    @test rho_eta[:xvals] == RHO_EXTENDED_VALS
    @test rho_eta[:yvals] == ETA_VALS
    @test eta_r[:xvals] == ETA_VALS
    @test eta_N[:xvals] == ETA_VALS
    @test rho_N[:xvals] == RHO_EXTENDED_VALS
    @test rho_r[:xvals] == RHO_CORE_VALS
    @test 0.85 in [j[:value] for j in oat_jobs if j[:key] == "rho"]

    # Rho-group summaries use the same effective support at rho = 0, 0.5, and 1.
    # The rho x delta grid is separate because delta is inactive at rho = 1.
    common_pairs = Set(("rho_eta", "rho_N", "rho_r"))
    for rho in (0.0, 0.5, 1.0)
        refs = filter(cells) do cell
            cell[:resolved_params][:rho] == rho && (
                (cell[:kind] == "oat" && cell[:axis] == "rho") ||
                (cell[:kind] == "phase" && get(cell, :pair, "") in common_pairs)
            )
        end
        @test length(unique(c[:condition_index] for c in refs)) == 10
    end
end

@testset "rho=1 effective realization is delta-invariant" begin
    common = (N=30, k=4, T=3, E_init=1, train_steps=1, eta=0.0, rho=1.0, seed=90210)
    state_zero, metrics_zero = run_simulation(default_params(; common..., delta=0.0))
    state_one, metrics_one = run_simulation(default_params(; common..., delta=1.0))
    @test isequal(metrics_zero, metrics_one)
    @test state_zero.accum.agent_degrees == state_one.accum.agent_degrees
end
