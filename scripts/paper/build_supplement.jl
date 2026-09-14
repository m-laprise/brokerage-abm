"""
    scripts/paper/build_supplement.jl

Compile the standalone Supplementary Material to
`output/supplement/supplement.pdf`. Inputs are the hand-edited
`paper/supplement.tex`, generated `output/supplement/figmeta.tex` and
`output/assessment_access/paper_values.tex`, and the supplementary figures.

The build fails on a missing input, an undefined `\\pv` value, an unused display
convention, a missing figure, or a LaTeX error.
Auxiliary files are created in a temporary directory and discarded.
Existing PDFs are preserved when only build dates or document IDs differ.

With retained data available, render the required figures and generate the
assessment-access manuscript values first. See scripts/paper/README.md.

Usage: julia --project --threads=auto scripts/paper/build_supplement.jl
"""

include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
include(joinpath(@__DIR__, "pdf_output.jl"))

const PAPER = normpath(joinpath(@__DIR__, "..", "..", "paper"))
const GENERATED = normpath(joinpath(@__DIR__, "..", "..", "output", "supplement"))
const SRC = joinpath(PAPER, "supplement.tex")
const FIGMETA = joinpath(GENERATED, "figmeta.tex")
const ASSESSMENT_VALUES = normpath(
    joinpath(GENERATED, "..", "assessment_access", "paper_values.tex")
)
const FIGDIR = joinpath(GENERATED, "figures")
const PDF = joinpath(GENERATED, "supplement.pdf")
const REPORTING_PROVENANCE = manuscript_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."))
)

fail(msg) = (println("BUILD FAILED: ", msg); exit(1))

isfile(SRC) || fail("missing $SRC")
isfile(FIGMETA) || fail("missing $FIGMETA (run scripts/paper/supp_figures.jl first)")
isfile(ASSESSMENT_VALUES) ||
    fail("missing $ASSESSMENT_VALUES (run scripts/assessment_access/paper_values.jl first)")
figmeta_source = read(FIGMETA, String)
analysis_commits = Dict{String,String}()
for label in ("DGP", "Structural")
    match_result = match(
        Regex("(?im)^%\\s*$label data analysis commit:\\s*([0-9a-f]{7,40})\\s*\$"),
        figmeta_source,
    )
    isnothing(match_result) && fail("figmeta.tex records no $label analysis commit")
    analysis_commits[lowercase(label)] = validate_analysis_commit(
        REPORTING_PROVENANCE, match_result[1]; artifact="$label supplement figure data"
    )
end
analysis_commits["assessment_access"] = validate_analysis_commit(
    REPORTING_PROVENANCE,
    recorded_analysis_commit(ASSESSMENT_VALUES);
    artifact="assessment-access manuscript values",
)

defs = Set{String}()
figmeta_defs = Set{String}()
for values_file in (FIGMETA, ASSESSMENT_VALUES), line in eachline(values_file)
    match_result = match(r"^\\pvDefine\{([^}]+)\}\{.*\}\s*$", line)
    isnothing(match_result) || begin
        match_result[1] in defs && fail("duplicate \\pv definition: $(match_result[1])")
        push!(defs, match_result[1])
        values_file == FIGMETA && push!(figmeta_defs, match_result[1])
    end
end
isempty(defs) && fail("no \\pvDefine entries in $(basename(FIGMETA))")

source = read(SRC, String)
body = split(source, "\\begin{document}"; limit=2)[end]
body = join([line for line in split(body, '\n') if !startswith(lstrip(line), "%")], "\n")
refs = Set(match_result[1] for match_result in eachmatch(r"\\pv\{([^}]+)\}", body))
undefined = sort(collect(setdiff(refs, defs)))
unused = sort(collect(setdiff(figmeta_defs, refs)))
isempty(undefined) ||
    fail("\\pv references with no generated definition: " * join(undefined, ", "))
isempty(unused) || fail("figmeta.tex definitions never referenced: " * join(unused, ", "))

figures = [
    match_result[1] for
    match_result in eachmatch(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
]
for figure in figures
    isfile(joinpath(FIGDIR, figure)) || fail("missing supplement figure file: $figure")
end

mkpath(GENERATED)
mktempdir() do build
    log_path = joinpath(build, "pdflatex.out")
    command = Cmd(
        Cmd([
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory=$build",
            "supplement.tex",
        ]);
        dir=PAPER,
    )
    ok = true
    for _ in 1:2
        process = run(
            pipeline(ignorestatus(command); stdout=log_path, stderr=log_path); wait=true
        )
        ok &= success(process)
    end
    log = read(log_path, String)
    errors = count("\n!", log)
    ok || fail("pdflatex failed while building the supplement")
    errors == 0 || fail("$errors LaTeX errors while building the supplement")
    occursin("There were undefined references", log) &&
        fail("undefined supplement references")
    update_pdf(joinpath(build, "supplement.pdf"), PDF)
    open(joinpath(GENERATED, "provenance.txt"), "w") do io
        println(io, "dgp_analysis_commit=$(analysis_commits["dgp"])")
        println(io, "structural_analysis_commit=$(analysis_commits["structural"])")
        println(
            io, "assessment_access_analysis_commit=$(analysis_commits["assessment_access"])"
        )
        println(io, "manuscript_commit=$(REPORTING_PROVENANCE.commit)")
        println(io, "manuscript_source_clean=$(REPORTING_PROVENANCE.source_clean)")
    end
    println("supplement validated ($(length(figures)) figures, 0 errors)")
end
