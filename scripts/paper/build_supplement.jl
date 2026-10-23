"""
Compile the supplement from its prose, generated settings, simulation design,
and figures. Check required inputs and references; allow unused values.

Usage: julia --project --threads=auto scripts/paper/build_supplement.jl
"""

include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
include(joinpath(@__DIR__, "pdf_output.jl"))

const PAPER = normpath(joinpath(@__DIR__, "..", "..", "paper"))
const GENERATED = normpath(joinpath(@__DIR__, "..", "..", "output", "supplement"))
const SRC = joinpath(PAPER, "supplement.tex")
const FIGMETA = joinpath(GENERATED, "figmeta.tex")
const SETTINGS = normpath(joinpath(GENERATED, "..", "manuscript", "settings.tex"))
const DESIGN = joinpath(GENERATED, "simulation_design.tex")
const ASSESSMENT_VALUES = normpath(
    joinpath(GENERATED, "..", "assessment_access", "paper_values.tex")
)
const FIGDIR = joinpath(GENERATED, "figures")
const PDF = joinpath(GENERATED, "supplement.pdf")
const REPORTING_PROVENANCE = manuscript_git_provenance(
    normpath(joinpath(@__DIR__, "..", ".."));
    sources=(
        @__FILE__,
        "scripts/paper/pdf_output.jl",
        "paper/supplement.tex",
    ),
)

fail(msg) = (println("BUILD FAILED: ", msg); exit(1))

isfile(SRC) || fail("missing $SRC")
isfile(FIGMETA) || fail("missing $FIGMETA (run scripts/paper/supp_figures.jl first)")
isfile(DESIGN) || fail("missing $DESIGN (run scripts/paper/build_simulation_design.jl first)")
validate_publication_output(REPORTING_PROVENANCE.root, DESIGN)
isfile(ASSESSMENT_VALUES) ||
    fail("missing $ASSESSMENT_VALUES (run scripts/assessment_access/paper_values.jl first)")
input_provenance = analysis_input_provenance(REPORTING_PROVENANCE, (FIGMETA, ASSESSMENT_VALUES, DESIGN, SETTINGS))

defs = Set{String}()
for values_file in (FIGMETA, ASSESSMENT_VALUES, SETTINGS), line in eachline(values_file)
    match_result = match(r"^\\pvDefine\{([^}]+)\}\{.*\}\s*$", line)
    isnothing(match_result) || begin
        match_result[1] in defs && fail("duplicate \\pv definition: $(match_result[1])")
        push!(defs, match_result[1])
    end
end
isempty(defs) && fail("no \\pvDefine entries in $(basename(FIGMETA))")

source = read(SRC, String)
source = replace(source, raw"\input{../output/supplement/simulation_design.tex}" => read(DESIGN, String))
body = split(source, "\\begin{document}"; limit=2)[end]
body = join([line for line in split(body, '\n') if !startswith(lstrip(line), "%")], "\n")
refs = Set(match_result[1] for match_result in eachmatch(r"\\pv\{([^}]+)\}", body))
undefined = sort(collect(setdiff(refs, defs)))
isempty(undefined) ||
    fail("\\pv references with no generated definition: " * join(undefined, ", "))

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
    record_publication_stage(REPORTING_PROVENANCE;
        stage="scripts/paper/build_supplement.jl",
        inputs=(FIGMETA, ASSESSMENT_VALUES, DESIGN, SETTINGS, joinpath.(FIGDIR, figures)...),
        outputs=(PDF,),
        analysis_commits=Dict("$(input.path): $label" => commit
            for input in input_provenance for (label, commit) in input.commits),
    )
    println("supplement validated ($(length(figures)) figures, 0 errors)")
end
