using Test
using DataFrames: DataFrame
using JLD2: jldsave

include(joinpath(@__DIR__, "..", "scripts", "sweep", "shard_validation.jl"))
include(joinpath(@__DIR__, "..", "scripts", "sweep", "sweep_results.jl"))

@testset "sweep artifact validation" begin
    provenance = Dict{Symbol,Any}(
        :schema_version => 5,
        :git_commit => "abc",
        :julia_version => "1.11.3",
        :pkg_manifest_hash => "packages",
        :manifest_hash => "manifest",
    )

    mktempdir() do dir
        shard = joinpath(dir, "seed_1.jld2")
        jldsave(
            shard;
            schema_version=5,
            git_commit="abc",
            julia_version="1.11.3",
            pkg_manifest_hash="packages",
            manifest_hash="manifest",
        )
        @test shard_is_current(shard, provenance)

        stale = copy(provenance)
        stale[:manifest_hash] = "different"
        @test !shard_is_current(shard, stale)

        aggregate = joinpath(dir, "data.jld2")
        jldsave(
            aggregate; provenance=Dict(string(key) => value for (key, value) in provenance)
        )
        @test aggregate_is_current(aggregate, provenance)
        @test !aggregate_is_current(aggregate, stale)
    end
end

@testset "completed sweep loading" begin
    mktempdir() do root
        result_rel = "oat/rho=1.0"
        alias_rel = "phase/rho_delta/cells/5_0"
        resolved = Dict{Symbol,Any}(:rho => 1.0, :delta => 0.5)
        grid_cells = [
            Dict{Symbol,Any}(
                :kind => "oat",
                :reldir => result_rel,
                :result_reldir => result_rel,
                :condition_index => 0,
                :resolved_params => copy(resolved),
            ),
            Dict{Symbol,Any}(
                :kind => "phase",
                :pair => "rho_delta",
                :reldir => alias_rel,
                :result_reldir => result_rel,
                :condition_index => 0,
                :resolved_params => Dict{Symbol,Any}(:rho => 1.0, :delta => 0.0),
            ),
        ]
        conditions = [
            Dict{Symbol,Any}(
                :reldir => result_rel,
                :result_reldir => result_rel,
                :condition_index => 0,
                :resolved_params => resolved,
                :seeds => [1, 2],
            ),
        ]
        meta = Dict{Symbol,Any}(
            :seeds => [1, 2], :T => 3, :n_conditions => 1, :n_grid_cells => 2
        )
        manifest_hash = "manifest"
        schema_version = 5
        jldsave(
            joinpath(root, "manifest.jld2");
            cells=grid_cells,
            conditions=conditions,
            meta=meta,
            manifest_hash=manifest_hash,
            schema_version=schema_version,
        )

        result_dir = joinpath(root, result_rel)
        mkpath(result_dir)
        frames = [DataFrame(period=1:3, value=fill(seed, 3)) for seed in 1:2]
        provenance = Dict("manifest_hash" => manifest_hash, "schema_version" => 5)
        jldsave(
            joinpath(result_dir, "data.jld2");
            mdfs=frames,
            seeds=[1, 2],
            realized_config=Dict("rho" => 1.0, "delta" => 0.5),
            provenance=provenance,
            result_reldir=result_rel,
            condition_index=0,
            schema_version=5,
        )

        sweep = load_sweep_dataset(root)
        @test length(sweep.results) == 1
        @test length(sweep.grid_cells) == 2
        @test sweep.manifest_hash == manifest_hash
        @test sweep.schema_version == schema_version
        alias = grid_result(sweep, alias_rel)
        @test alias.result_rel == result_rel
        @test alias.cfg["delta"] == 0.0
        @test sweep.results[1].cfg["delta"] == 0.5
        @test length(unique_effective_results([sweep.results[1], alias])) == 1

        jldsave(
            joinpath(result_dir, "data.jld2");
            mdfs=frames[1:1],
            seeds=[1],
            realized_config=Dict("rho" => 1.0, "delta" => 0.5),
            provenance=provenance,
            result_reldir=result_rel,
            condition_index=0,
            schema_version=5,
        )
        @test_throws ErrorException load_sweep_dataset(root)
    end
end
