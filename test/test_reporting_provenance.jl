using Test

include(joinpath(@__DIR__, "..", "scripts", "reporting_provenance.jl"))

@testset "Reporting Git provenance" begin
    mktempdir() do repository
        run(`git -C $repository init -q`)
        run(`git -C $repository config user.email test@example.com`)
        run(`git -C $repository config user.name "Test User"`)
        write(joinpath(repository, "source.txt"), "committed\n")
        run(`git -C $repository add source.txt`)
        run(`git -C $repository commit -q -m initial`)

        clean = reporting_git_provenance(repository)
        @test clean.source_clean
        @test length(clean.commit) == 40
        @test clean.short_commit == first(clean.commit, 7)

        mkpath(joinpath(repository, "output"))
        write(joinpath(repository, "output", "generated.txt"), "generated\n")
        @test reporting_git_provenance(repository).source_clean

        mkpath(joinpath(repository, "notes"))
        write(joinpath(repository, "notes", "planning.md"), "draft note\n")
        @test reporting_git_provenance(repository).source_clean

        write(joinpath(repository, "source.txt"), "modified\n")
        dirty = reporting_git_provenance(repository; require_clean=false)
        @test !dirty.source_clean
        @test_throws ErrorException reporting_git_provenance(repository)

        permitted = reporting_git_provenance(
            repository;
            allowed_dirty_paths=("source.txt",),
        )
        @test !permitted.source_clean
        @test occursin("source.txt", permitted.source_status)

        write(joinpath(repository, "analysis.jl"), "uncommitted\n")
        @test_throws ErrorException reporting_git_provenance(
            repository;
            allowed_dirty_paths=("source.txt",),
        )

        mkpath(joinpath(repository, "paper", "appendices"))
        write(joinpath(repository, "paper", "manuscript.tex"), "manuscript\n")
        write(
            joinpath(repository, "paper", "appendices", "model_specifications.tex"),
            "specifications\n",
        )
        write(
            joinpath(repository, "paper", "appendices", "simulation_pseudocode.tex"),
            "pseudocode\n",
        )
        run(`git -C $repository add paper`)
        run(`git -C $repository commit -q -m paper`)
        write(joinpath(repository, "source.txt"), "committed\n")
        write(joinpath(repository, "analysis.jl"), "")
        run(`git -C $repository add source.txt analysis.jl`)
        run(`git -C $repository commit -q -m cleanup`)

        write(joinpath(repository, "paper", "manuscript.tex"), "revised manuscript\n")
        @test !manuscript_git_provenance(repository).source_clean
        write(
            joinpath(repository, "paper", "appendices", "model_specifications.tex"),
            "revised specifications\n",
        )
        write(
            joinpath(repository, "paper", "appendices", "simulation_pseudocode.tex"),
            "revised pseudocode\n",
        )
        revised = manuscript_git_provenance(repository)
        @test !revised.source_clean
        @test occursin("model_specifications.tex", revised.source_status)
        @test occursin("simulation_pseudocode.tex", revised.source_status)

        write(
            joinpath(repository, ".gitignore"),
            "*.jld2\n!output/assessment_access/centrality_trajectories.jld2\n",
        )
        artifact_inclusion = manuscript_git_provenance(repository)
        @test !artifact_inclusion.source_clean
        @test occursin(".gitignore", artifact_inclusion.source_status)
        @test_throws ErrorException reporting_git_provenance(repository)

        reporting_paths = filter(
            path -> startswith(path, "scripts/assessment_access/") ||
                path in (
                    "test/test_assessment_access_reporting.jl",
                    "scripts/paper/pdf_output.jl",
                    "scripts/figure_style.jl",
                    "test/test_pdf_output.jl",
                ),
            MANUSCRIPT_ITERATION_PATHS,
        )
        for path in reporting_paths
            mkpath(dirname(joinpath(repository, path)))
            write(joinpath(repository, path), "uncommitted presentation change\n")
        end
        presentation = manuscript_git_provenance(repository)
        @test !isempty(reporting_paths) && !presentation.source_clean
        @test all(path -> occursin(path, presentation.source_status), reporting_paths)
        @test_throws ErrorException reporting_git_provenance(repository)
        write(
            joinpath(repository, "scripts", "assessment_access", "analyze.jl"),
            "uncommitted scientific analysis\n",
        )
        @test_throws ErrorException manuscript_git_provenance(repository)

        write(joinpath(repository, "analysis.jl"), "revised analysis\n")
        @test_throws ErrorException manuscript_git_provenance(repository)
    end
end
