using Test
using SHA: sha256
using Base64: base64decode

include(joinpath(@__DIR__, "..", "scripts", "reporting_provenance.jl"))

@testset "Scoped reporting provenance" begin
    mktempdir() do repository
        run(`git -C $repository init -q`)
        run(`git -C $repository config user.email test@example.com`)
        run(`git -C $repository config user.name "Test User"`)
        mkpath(joinpath(repository, "paper"))
        mkpath(joinpath(repository, "src"))
        write(joinpath(repository, "analysis.jl"), "# committed analysis\n")
        write(joinpath(repository, "src", "accounting.jl"), "# accounting\n")
        write(joinpath(repository, "paper", "manuscript.tex"), "Manuscript.\n")
        write(joinpath(repository, "Project.toml"), "[deps]\n")
        write(joinpath(repository, ".gitignore"), "ignored.jl\n")
        run(`git -C $repository add .`)
        run(`git -C $repository commit -q -m initial`)
        sources = ("analysis.jl", "src")
        clean = reporting_git_provenance(repository; sources)
        @test clean.source_clean
        @test length(clean.commit) == 40 && clean.short_commit == first(clean.commit, 7)
        @test Set(clean.source_files) == Set(["analysis.jl", "src/accounting.jl", "Project.toml"])
        @test all(length(hash) == 64 for hash in values(clean.source_hashes))
        @test clean.source_hashes["analysis.jl"] == bytes2hex(sha256("# committed analysis\n"))
        @test_throws ErrorException reporting_git_provenance(repository; sources=())
        @test_throws ErrorException reporting_git_provenance(repository; sources=("missing.jl",))
        @test_throws ErrorException reporting_git_provenance(repository; sources=("../outside",))
        @test_throws ErrorException reporting_git_provenance(repository; sources=(".",))

        write(joinpath(repository, "paper", "manuscript.tex"), "Revised manuscript.\n")
        write(joinpath(repository, "other_plot.jl"), "# unrelated plot\n")
        unrelated = reporting_git_provenance(repository; sources)
        @test unrelated.source_clean && unrelated.source_fingerprint == clean.source_fingerprint

        write(joinpath(repository, "analysis.jl"), "# revised analysis\n")
        @test_throws ErrorException reporting_git_provenance(repository; sources)
        presentation = manuscript_git_provenance(repository; sources=("paper/manuscript.tex",))
        @test !presentation.source_clean
        @test !occursin("analysis.jl", presentation.source_status)
        snapshot = sprint(io -> write_source_provenance(io, presentation))
        @test occursin("Source SHA256: paper/manuscript.tex", snapshot)
        patch = match(r"(?m)^% Source patch \(base64\): (.+)$", snapshot)[1]
        @test occursin("+Revised manuscript.", String(base64decode(patch)))
        artifact = joinpath(repository, "artifact.tex")
        write(artifact, snapshot)
        @test isnothing(validate_source_hashes(presentation, artifact, ("paper/manuscript.tex",)))
        write(joinpath(repository, "paper", "manuscript.tex"), "Another revision.\n")
        @test_throws ErrorException validate_source_hashes(presentation, artifact, ("paper/manuscript.tex",))
        @test_throws ErrorException validate_source_hashes(presentation, artifact, ("analysis.jl",))
        dirty = reporting_git_provenance(repository; sources, require_clean=false)
        @test !dirty.source_clean && dirty.source_fingerprint != clean.source_fingerprint

        write(joinpath(repository, "analysis.jl"), "# committed analysis\n")
        write(joinpath(repository, "src", "accounting.jl"), "# revised accounting\n")
        @test_throws ErrorException reporting_git_provenance(repository; sources)
        write(joinpath(repository, "src", "accounting.jl"), "# accounting\n")
        write(joinpath(repository, "ignored.jl"), "# ignored dependency\n")
        @test_throws ErrorException reporting_git_provenance(repository; sources=("ignored.jl",))
        untracked = manuscript_git_provenance(repository; sources=("ignored.jl",))
        @test !untracked.source_clean
        snapshot = sprint(io -> write_source_provenance(io, untracked))
        contents = match(r"(?m)^% Untracked source \(base64\): ignored.jl (.+)$", snapshot)[1]
        @test String(base64decode(contents)) == "# ignored dependency\n"
        write(joinpath(repository, "Project.toml"), "[deps]\n# changed environment\n")
        @test_throws ErrorException reporting_git_provenance(repository; sources)
        write(joinpath(repository, "Project.toml"), "[deps]\n")

        # A separately maintained analysis commit does not need to be an ancestor.
        first_commit = clean.commit
        write(joinpath(repository, "analysis.jl"), "# second analysis\n")
        run(`git -C $repository add analysis.jl`)
        run(`git -C $repository commit -q -m second`)
        second_commit = readchomp(`git -C $repository rev-parse HEAD`)
        run(`git -C $repository switch -q --detach $first_commit`)
        current = reporting_git_provenance(repository; sources)
        @test validate_analysis_commit(current, first_commit) == first_commit
        @test validate_analysis_commit(current, second_commit) == second_commit
        @test validate_analysis_commit(current, first(second_commit, 12)) == second_commit
        @test_throws ErrorException validate_analysis_commit(current, "")
        @test_throws ErrorException validate_analysis_commit(current, "HEAD")
        @test_throws ErrorException validate_analysis_commit(current, repeat("f", 40))

        a, b = joinpath.(repository, ("a.tex", "b.txt"))
        write(a, "% Data analysis commit: $first_commit\n% NN analysis commit: $second_commit\n")
        write(b, "analysis_commit=$second_commit\n")
        @test recorded_analysis_commit(a) == first_commit
        @test recorded_analysis_commit(b) == second_commit
        @test length(recorded_analysis_commits(a)) == 2
        inputs = analysis_input_provenance(current, (a, b))
        @test length(inputs) == 2 && length(inputs[1].commits) == 2
        @test inputs[1].sha256 == bytes2hex(sha256(read(a)))
        write(b, "ridge_analysis_commit=$second_commit\nnn_analysis_commit=$first_commit\n")
        @test length(analysis_input_provenance(current, (b,))[1].commits) == 2
        @test_throws ErrorException recorded_analysis_commit(b)
        write(b, "analysis_commit=$second_commit\nanalysis_commit=$first_commit\n")
        @test_throws ErrorException recorded_analysis_commits(b)
        write(b, "analysis_commit=$(repeat("f", 40))\n")
        @test_throws ErrorException analysis_input_provenance(current, (b,))
        write(b, "no provenance\n")
        @test_throws ErrorException recorded_analysis_commits(b)
    end
end

@testset "Section assembly with independent analysis commits" begin
    project = normpath(joinpath(@__DIR__, ".."))
    mktempdir() do repository
        run(`git -C $repository init -q`)
        run(`git -C $repository config user.email test@example.com`)
        run(`git -C $repository config user.name "Test User"`)
        write(joinpath(repository, "source.txt"), "first analysis\n")
        run(`git -C $repository add source.txt`)
        run(`git -C $repository commit -q -m first`)
        first_commit = readchomp(`git -C $repository rev-parse HEAD`)
        write(joinpath(repository, "source.txt"), "second analysis\n")
        run(`git -C $repository add source.txt`)
        run(`git -C $repository commit -q -m second`)
        second_commit = readchomp(`git -C $repository rev-parse HEAD`)
        files = Dict(
            "paper/section_source.tex" => "\\pv{example}\n\\begin{figure}\n" *
                "\\includegraphics{figures/test.png}\n" *
                "\\caption{\\pvtitle{example}. \\pvcaption{example}}\n\\end{figure}\n",
            "paper/captions.tex" => "\\begin{ptitle}{example}Title\\end{ptitle}\n" *
                "\\begin{pcaption}{example}Caption\\end{pcaption}\n",
            "paper/supplement.tex" => "\\begin{figure}\\label{fig:supp-test}\\end{figure}\n",
            "output/main/values.tex" => "% Analysis commit: $first_commit\n\\pvDefine{example}{1.25}\n",
            "output/main/convergence/values.tex" => "% Analysis commit: $second_commit\n",
            "output/ridge/paired/analysis/paper_values.tex" => "% Data analysis commit: $first_commit\n" *
                "% NN-Ridge comparison analysis commit: $second_commit\n",
            "output/ridge/ablations/analysis/paper_values.tex" => "% Analysis commit: $second_commit\n",
            "output/assessment_access/paper_values.tex" => "% Data analysis commit: $second_commit\n",
            "output/main/figmeta.tex" => "% Data analysis commit: $first_commit\n" *
                "% Ridge ablation analysis commit: $second_commit\n",
            # Assembly checks file identity, not image decoding or TeX layout.
            "output/main/figures/test.png" => "image fixture\n",
        )
        for (path, contents) in files
            target = joinpath(repository, path)
            mkpath(dirname(target))
            write(target, contents)
        end
        mkpath(joinpath(repository, "scripts", "paper"))
        cp(joinpath(project, "scripts", "reporting_provenance.jl"),
            joinpath(repository, "scripts", "reporting_provenance.jl"))
        builder = joinpath(repository, "scripts", "paper", "build_section.jl")
        cp(joinpath(project, "scripts", "paper", "build_section.jl"), builder)
        # Exercise the real assembly through the generated TeX write, without a PDF build.
        assembly = first(split(read(builder, String), "# Compile smoke test"))
        scope = Module(gensym(:SectionAssembly))
        Core.eval(scope, :(include(path) = Base.include(@__MODULE__, path)))
        Base.include_string(scope, assembly, builder)
        output = joinpath(repository, "output", "main", "results_section.tex")
        result = read(output, String)
        @test occursin("1.25", result) && !occursin("\\pv{example}", result)
        @test occursin("analysis commit: $first_commit", result)
        @test occursin("analysis commit: $second_commit", result)
        @test length(collect(eachmatch(r"(?m)^% Input: ", result))) == 6
        @test occursin("% Figure SHA256: figures/test.png ", result)
        @test isnothing(validate_source_hashes((; root=repository), output,
            ("paper/section_source.tex", "paper/captions.tex", "paper/supplement.tex")))
    end
end
