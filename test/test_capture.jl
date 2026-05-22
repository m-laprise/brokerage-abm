using Test
using TransientBrokerage

@testset "Resource Capture (Principal Mode)" begin
    @testset "counterparty_ask uses history mean" begin
        p = default_params(N=20, T=5, T_burn=1, seed=42)
        state = initialize_model(p)
        agent = state.agents[1]
        agent.history_count = 3
        agent.history_q[1] = 2.0
        agent.history_q[2] = 4.0
        agent.history_q[3] = 3.0
        @test counterparty_ask(agent, state.cal.q_cal) ≈ 3.0
    end

    @testset "counterparty_ask falls back to q_cal with no history" begin
        p = default_params(N=20, T=5, T_burn=1, seed=42)
        state = initialize_model(p)
        agent = state.agents[1]
        agent.history_count = 0
        @test counterparty_ask(agent, state.cal.q_cal) == state.cal.q_cal
    end

    @testset "principal capture requires live confidence" begin
        p = default_params(N=20, K=5, seed=42, enable_principal=true)
        state = initialize_model(p)
        state.broker.capture_confidence_mae = 0.1
        state.agents[4].history_count = 1
        state.agents[4].history_q[1] = 1.0

        n_blocks = TransientBrokerage.plan_period_capture!(
            [1, 2],
            [:broker, :broker],
            [2, 1],
            state.agents,
            state.broker,
            p,
            state.cal;
            ws=state.workspace,
        )

        @test n_blocks == 0
        @test isempty(state.workspace.principal_inventory_ids)
    end

    @testset "score_capture_block uses whole current block and distinct current depth" begin
        p = default_params(N=10, K=3, seed=42, enable_principal=true)
        state = initialize_model(p)
        state.agents[4].history_count = 1
        state.agents[4].history_q[1] = 1.0
        TransientBrokerage.rebuild_current_match_index!(state.workspace, state.agents)

        score = TransientBrokerage.score_capture_block!(
            Tuple{Float64,Int}[],
            reshape([4.5, 2.7, 0.8], 3, 1),
            [4],
            1,
            [1, 2, 3],
            [2, 1, 1],
            state.agents,
            state.cal,
            p.K;
            ws=state.workspace,
        )

        @test score.counterparty_id == 4
        @test score.cap_j == 3
        @test score.filled
        @test score.profitable_depth == 2
        expected_margin =
            (4.5 - 1.0 - state.cal.phi) +
            (2.7 - 1.0 - state.cal.phi) +
            (0.8 - 1.0 - state.cal.phi)
        @test score.block_margin ≈ expected_margin
    end

    @testset "score_capture_block excludes already active relationships" begin
        p = default_params(N=10, K=3, seed=43, enable_principal=true)
        state = initialize_model(p)
        state.agents[4].history_count = 1
        state.agents[4].history_q[1] = 1.0
        push!(state.agents[1].active_matches, ActiveMatch(4, false, :self))
        push!(state.agents[4].active_matches, ActiveMatch(1, false, :self))
        TransientBrokerage.rebuild_current_match_index!(state.workspace, state.agents)

        score = TransientBrokerage.score_capture_block!(
            Tuple{Float64,Int}[],
            reshape([9.0, 4.0, 3.0], 3, 1),
            [4],
            1,
            [1, 2, 3],
            [1, 1, 1],
            state.agents,
            state.cal,
            p.K;
            ws=state.workspace,
        )

        @test score.cap_j == 2
        @test score.filled
        expected_margin = (4.0 - 1.0 - state.cal.phi) + (3.0 - 1.0 - state.cal.phi)
        @test score.block_margin ≈ expected_margin
    end

    @testset "best_capture_block rejects one-demander dominance and unfilled blocks" begin
        p = default_params(N=10, K=3, seed=42, enable_principal=true)
        state = initialize_model(p)
        state.broker.capture_confidence_mae = 0.1
        state.agents[4].history_count = 1
        state.agents[4].history_q[1] = 1.0
        TransientBrokerage.rebuild_current_match_index!(state.workspace, state.agents)

        one_demander = TransientBrokerage.best_capture_block!(
            Tuple{Float64,Int}[],
            reshape([6.0, 1.0, 1.0], 3, 1),
            [4],
            [1, 2, 3],
            [1, 1, 1],
            state.agents,
            state.broker,
            state.cal,
            p.K;
            ws=state.workspace,
        )
        @test isnothing(one_demander)

        unfilled = TransientBrokerage.best_capture_block!(
            Tuple{Float64,Int}[],
            reshape([4.5, 4.2], 2, 1),
            [4],
            [1, 2],
            [1, 1],
            state.agents,
            state.broker,
            state.cal,
            p.K;
            ws=state.workspace,
        )
        @test isnothing(unfilled)
    end

    @testset "literal acquisition plans whole block with distinct demanders" begin
        p = default_params(N=10, K=3, T=5, T_burn=1, seed=42, enable_principal=true)
        state = initialize_model(p)
        state.broker.capture_confidence_ready = true
        state.broker.capture_confidence_mae = 0.1
        state.broker.roster = Set([4])
        empty!(state.broker.current_clients)
        fill!(state.broker.nn.W1, 0.0)
        fill!(state.broker.nn.b1, 0.0)
        fill!(state.broker.nn.w2, 0.0)
        state.broker.nn.b2 = 4.0
        state.agents[4].history_count = 1
        state.agents[4].history_q[1] = 1.0

        n_blocks = TransientBrokerage.plan_period_capture!(
            [1, 2, 3],
            [:broker, :broker, :broker],
            [2, 1, 1],
            state.agents,
            state.broker,
            p,
            state.cal;
            ws=state.workspace,
        )

        @test n_blocks == 1
        @test state.workspace.principal_inventory_ids == [4]
        @test state.workspace.principal_inventory_asks == [1.0]
        @test state.workspace.principal_reserved_capacity[4] == 3
        @test available_capacity(
            state.agents[4], p.K, state.workspace.principal_reserved_capacity[4]
        ) == 0

        accepted = TransientBrokerage.AcceptedMatch[]
        remaining_demand = [2, 1, 1]
        broker_indices = [1, 2, 3]
        broker_demanders = [1, 2, 3]
        wc_i = Int[]
        wc_j = Int[]

        n_captured = TransientBrokerage.execute_inventory_round!(
            accepted,
            remaining_demand,
            broker_indices,
            broker_demanders,
            state.agents,
            state.broker,
            state.env,
            state.G,
            p,
            state.cal,
            state.rng,
            wc_i,
            wc_j;
            ws=state.workspace,
            Ax_buf=Vector{Float64}(undef, p.d),
            Bx_buf=Vector{Float64}(undef, p.d),
        )

        @test n_captured == 3
        @test remaining_demand == [1, 0, 0]
        @test length(accepted) == 3
        @test all(m -> m.is_principal, accepted)
        @test count(==(1), getfield.(accepted, :demander_id)) == 1
        @test count(==(2), getfield.(accepted, :demander_id)) == 1
        @test count(==(3), getfield.(accepted, :demander_id)) == 1
        @test all(==(4), getfield.(accepted, :counterparty_id))
        @test state.workspace.principal_reserved_capacity[4] == 0
        @test length(state.agents[4].active_matches) == 3
        @test length(unique(getfield.(state.agents[4].active_matches, :partner_id))) == 3

        wc_i2 = Int[]
        wc_j2 = Int[]
        n_captured2 = TransientBrokerage.execute_inventory_round!(
            accepted,
            remaining_demand,
            [1],
            [1],
            state.agents,
            state.broker,
            state.env,
            state.G,
            p,
            state.cal,
            state.rng,
            wc_i2,
            wc_j2;
            ws=state.workspace,
            Ax_buf=Vector{Float64}(undef, p.d),
            Bx_buf=Vector{Float64}(undef, p.d),
        )

        @test n_captured2 == 0
        @test remaining_demand == [1, 0, 0]
        @test length(accepted) == 3
        @test state.workspace.principal_reserved_capacity[4] == 0
    end

    @testset "capture confidence starts unavailable before live broker matches" begin
        state = initialize_model(default_params(N=30, seed=42, enable_principal=true))
        @test state.broker.capture_confidence_ready == false
        @test state.broker.capture_confidence_mae == 0.0
    end

    @testset "capture confidence initializes from first live broker period and holds with no data" begin
        state = initialize_model(default_params(N=20, seed=42, enable_principal=true))
        @test state.broker.capture_confidence_ready == false

        TransientBrokerage.update_capture_confidence_mae!(state.broker, 3.0, 2, 0.25)
        @test state.broker.capture_confidence_ready == true
        @test state.broker.capture_confidence_mae ≈ 1.5

        TransientBrokerage.update_capture_confidence_mae!(state.broker, 1.0, 1, 0.25)
        @test state.broker.capture_confidence_mae ≈ (0.75 * 1.5 + 0.25 * 1.0)

        prev = state.broker.capture_confidence_mae
        TransientBrokerage.update_capture_confidence_mae!(state.broker, 0.0, 0, 0.25)
        @test state.broker.capture_confidence_mae == prev
    end

    @testset "first period is standard-only and initializes confidence from live broker MAE" begin
        p = default_params(N=80, T=3, T_burn=0, seed=42, enable_principal=true)
        state = initialize_model(p)

        step_period!(state)
        metrics = collect_period_metrics(state)

        @test metrics.period == 1
        @test metrics.n_broker_principal == 0
        @test isnan(metrics.broker_confidence_mae)
        @test !isnan(metrics.broker_selected_mae)
        @test state.broker.capture_confidence_ready == true
        @test state.broker.capture_confidence_mae ≈ metrics.broker_selected_mae
    end

    @testset "capture_surplus" begin
        @test capture_surplus(3.0, 1.0) ≈ 2.0
        @test capture_surplus(0.5, 1.0) ≈ -0.5
        @test capture_surplus(1.0, 1.0) ≈ 0.0
    end

    @testset "Full simulation with principal mode" begin
        p = default_params(N=50, T=10, T_burn=2, seed=42, enable_principal=true)
        _, df = run_simulation(p)
        @test length(df.n_broker_principal) == p.T
        @test all(0.0 .<= df.principal_mode_share .<= 1.0)
        @test df.n_broker_principal[1] == 0
        @test isnan(df.broker_confidence_mae[1])
        @test :broker_selected_mae in propertynames(df)
        @test :broker_confidence_mae in propertynames(df)
        sel_mae = filter(!isnan, df.broker_selected_mae)
        conf_mae = filter(!isnan, df.broker_confidence_mae)
        @test all(>=(0.0), sel_mae)
        @test all(>=(0.0), conf_mae)
    end

    @testset "Cumulative match counters increment as expected" begin
        p = default_params(N=60, T=10, T_burn=2, seed=321, eta=0.0)
        state, df = run_simulation(p)

        total_counter = sum(ag.n_matches_any for ag in state.agents)
        @test total_counter == 2 * sum(df.n_total_matches)
        @test all(ag -> ag.n_principal_acquired == 0, state.agents)

        p2 = default_params(N=60, T=10, T_burn=2, seed=321, eta=0.0, enable_principal=true)
        state2, df2 = run_simulation(p2)
        total_principal_acq = sum(ag.n_principal_acquired for ag in state2.agents)
        @test total_principal_acq == sum(df2.n_broker_principal)
    end

    @testset "Counters reset on agent exit/entry" begin
        p = default_params(N=40, T=5, T_burn=1, seed=202, eta=0.0, enable_principal=true)
        state = initialize_model(p)
        agent = state.agents[1]
        agent.n_matches_any = 7
        agent.n_principal_acquired = 3
        rng = state.rng
        exit_agent!(state, 1)
        enter_agent!(state, 1, rng)
        @test state.agents[1].n_matches_any == 0
        @test state.agents[1].n_principal_acquired == 0
    end

    @testset "Capture outcome metrics match hand computation" begin
        using Statistics: mean
        p = default_params(N=60, T=12, T_burn=2, seed=777, enable_principal=true)
        state, df = run_simulation(p)

        a = state.accum
        if !isempty(a.capture_realized)
            expected_delta = a.capture_realized .- a.capture_ask
            expected_mean = mean(expected_delta)
            @test df.capture_surplus_mean[end] ≈ expected_mean
            expected_loss_rate = count(<(0.0), expected_delta) / length(expected_delta)
            @test df.capture_loss_rate[end] ≈ expected_loss_rate
        else
            @test isnan(df.capture_surplus_mean[end])
            @test isnan(df.capture_loss_rate[end])
        end

        p3 = default_params(N=60, T=10, T_burn=2, seed=111, enable_principal=false)
        _, df3 = run_simulation(p3)
        @test all(isnan, df3.capture_surplus_mean)
        @test all(isnan, df3.capture_loss_rate)
        @test all(isnan, df3.capture_decision_rank)
        @test all(isnan, df3.capture_decision_rmse)
    end

    @testset "Broker dependency columns are well-formed" begin
        p = default_params(N=60, T=12, T_burn=2, seed=444, enable_principal=true)
        _, df = run_simulation(p)
        deps = filter(!isnan, df.broker_dependency_mean)
        @test all(0.0 .<= deps .<= 1.0)
        frac = filter(!isnan, df.broker_dependency_frac_above_half)
        @test all(0.0 .<= frac .<= 1.0)
        g = filter(!isnan, df.broker_dependency_gini)
        @test all(0.0 .<= g .<= 1.0)
    end
end
