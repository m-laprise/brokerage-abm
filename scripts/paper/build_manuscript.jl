"""
    scripts/paper/build_manuscript.jl

Create a submission-ready manuscript bundle from the editable TeX root and the
generated results section. The output is a single manuscript TeX file with the
results section inlined, plus its bibliography, referenced figures, Biber output,
and a compile-checked PDF under `output/manuscript/`. The builder also writes a
complete working-paper PDF containing the manuscript, Appendix A, Appendix B,
and the Supplementary Material. The generated TeX omits the editable source's
file-specific header.

Run `scripts/paper/build_section.jl` first whenever the results prose, values,
captions, figures, or analysis provenance changes. This script never edits or
regenerates scientific results.

Usage: julia --project --threads=auto scripts/paper/build_manuscript.jl
"""

using Dates: now

include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PAPER = joinpath(ROOT, "paper")
const MAIN_OUTPUT = joinpath(ROOT, "output", "main")
const MAIN_STEM = "brokers_who_do_not_bridge_without_appendices"
const COMPLETE_STEM = "brokers_who_do_not_bridge_with_appendices"
const SOURCE = joinpath(PAPER, MAIN_STEM * ".tex")
const RESULTS = joinpath(MAIN_OUTPUT, "results_section.tex")
const BIBLIOGRAPHY = joinpath(PAPER, "references.bib")
const OUTPUT = joinpath(ROOT, "output", "manuscript")
const OUTPUT_TEX = joinpath(OUTPUT, MAIN_STEM * ".tex")
const MAIN_PDF = joinpath(OUTPUT, MAIN_STEM * ".pdf")
const PSEUDOCODE = joinpath(ROOT, "simulation_pseudocode.pdf")
const SPECIFICATIONS = joinpath(ROOT, "model_specifications.pdf")
const SUPPLEMENT = joinpath(ROOT, "output", "supplement", "supplement.pdf")
const COMPLETE_PDF = joinpath(OUTPUT, COMPLETE_STEM * ".pdf")
const EDITABLE_ROOT_NOTICE =
    "% Editable root for the complete manuscript. The results section is generated\n" *
    "% separately and included below. Use scripts/paper/build_manuscript.jl to create\n" *
    "% the flattened, submission-ready TeX bundle under output/manuscript/.\n\n"
const INPUT_MARKER = "\\input{output/main/results_section.tex}"
const SOURCE_BIBLIOGRAPHY = "\\addbibresource{paper/references.bib}"
const BUNDLE_BIBLIOGRAPHY = "\\addbibresource{references.bib}"
const SOURCE_GRAPHICSPATH = "\\graphicspath{{output/main/}}"
const BUNDLE_GRAPHICSPATH = "\\graphicspath{{./}}"
const FORMAL_ILLUSTRATION_SECTION =
    "\\section{How Brokers Accumulate Informational and Structural Advantage: " *
    "A Formal Illustration}"
const MANUSCRIPT_PROVENANCE = manuscript_git_provenance(ROOT)

fail(message) = error("manuscript build failed: $message")

for path in (
    SOURCE,
    RESULTS,
    BIBLIOGRAPHY,
    PSEUDOCODE,
    SPECIFICATIONS,
    SUPPLEMENT,
)
    isfile(path) || fail("missing $path")
end

source = read(SOURCE, String)
results = read(RESULTS, String)

startswith(source, EDITABLE_ROOT_NOTICE) ||
    fail("editable root must begin with its source-file notice")
length(findall(INPUT_MARKER, source)) == 1 ||
    fail("editable root must contain exactly one $INPUT_MARKER")
length(findall(SOURCE_BIBLIOGRAPHY, source)) == 1 ||
    fail("editable root must contain exactly one repo-local bibliography declaration")
length(findall(SOURCE_GRAPHICSPATH, source)) == 1 ||
    fail("editable root must contain exactly one repo-local graphics path")
occursin(FORMAL_ILLUSTRATION_SECTION, results) ||
    fail("generated fragment does not contain the formal-illustration section")
occursin("% manuscript commit $(MANUSCRIPT_PROVENANCE.commit)", results) ||
    fail("generated results section does not match the current manuscript commit")

figure_paths = unique([
    match[1] for
    match in eachmatch(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", results)
])
isempty(figure_paths) && fail("generated results section references no figures")
for relative_path in figure_paths
    isfile(joinpath(MAIN_OUTPUT, relative_path)) ||
        fail("missing generated figure $relative_path")
end

flattened = replace(source, EDITABLE_ROOT_NOTICE => ""; count=1)
flattened = replace(flattened, INPUT_MARKER => results)
flattened = replace(flattened, SOURCE_BIBLIOGRAPHY => BUNDLE_BIBLIOGRAPHY)
flattened = replace(flattened, SOURCE_GRAPHICSPATH => BUNDLE_GRAPHICSPATH)
occursin(INPUT_MARKER, flattened) && fail("results input was not flattened")

source_state = MANUSCRIPT_PROVENANCE.source_clean ?
               "clean manuscript sources" :
               "uncommitted manuscript/build changes included"
header = """
% $(basename(OUTPUT_TEX)): GENERATED, do not edit.
% Built by scripts/paper/build_manuscript.jl on $(now()).
% Manuscript commit: $(MANUSCRIPT_PROVENANCE.commit); $source_state.
% The editable root is paper/$(basename(SOURCE)); the results provenance is
% recorded in the inlined output/main/results_section.tex header below.

"""

mkpath(OUTPUT)
write(OUTPUT_TEX, header * flattened)
cp(BIBLIOGRAPHY, joinpath(OUTPUT, "references.bib"); force=true)
for relative_path in figure_paths
    destination = joinpath(OUTPUT, relative_path)
    mkpath(dirname(destination))
    cp(joinpath(MAIN_OUTPUT, relative_path), destination; force=true)
end

mktempdir() do build
    tex_name = basename(OUTPUT_TEX)
    cp(OUTPUT_TEX, joinpath(build, tex_name))
    cp(joinpath(OUTPUT, "references.bib"), joinpath(build, "references.bib"))
    for relative_path in figure_paths
        destination = joinpath(build, relative_path)
        mkpath(dirname(destination))
        cp(joinpath(OUTPUT, relative_path), destination)
    end

    build_log = joinpath(build, "latexmk.out")
    process = run(
        pipeline(
            ignorestatus(
                Cmd(
                    `latexmk -pdf -interaction=nonstopmode -halt-on-error $tex_name`;
                    dir=build,
                ),
            );
            stdout=build_log,
            stderr=build_log,
        );
        wait=true,
    )
    final_log_path = joinpath(build, MAIN_STEM * ".log")
    if !success(process)
        cp(build_log, joinpath(OUTPUT, "latexmk-failure.log"); force=true)
        fail("latexmk compile check failed; see output/manuscript/latexmk-failure.log")
    end
    isfile(final_log_path) || fail("latexmk did not create a final LaTeX log")
    log = read(final_log_path, String)
    for pattern in (
        r"LaTeX Warning: Citation .* undefined",
        r"LaTeX Warning: There were undefined references",
        r"LaTeX Warning: Empty bibliography",
        r"Package biblatex Warning: Please \(re\)run Biber",
    )
        occursin(pattern, log) && fail("compile check reported unresolved references")
    end

    for extension in ("pdf", "bbl")
        artifact = joinpath(build, MAIN_STEM * "." * extension)
        isfile(artifact) || fail("latexmk did not create $artifact")
        cp(artifact, joinpath(OUTPUT, basename(artifact)); force=true)
    end
end

pdfunite = Sys.which("pdfunite")
isnothing(pdfunite) && fail("pdfunite is required to assemble the complete PDF")

components = [
    MAIN_PDF,
    PSEUDOCODE,
    SPECIFICATIONS,
    SUPPLEMENT,
]
mktempdir() do build
    combined = joinpath(build, basename(COMPLETE_PDF))
    process = run(ignorestatus(Cmd([pdfunite, components..., combined])); wait=true)
    success(process) || fail("pdfunite failed while assembling the complete PDF")
    isfile(combined) || fail("pdfunite did not create the complete PDF")
    cp(combined, COMPLETE_PDF; force=true)
end

println("wrote complete manuscript bundle to $OUTPUT")
println("  $(basename(OUTPUT_TEX)): flattened main manuscript without appendices")
println("  references.bib: $(count(line -> startswith(line, '@'), eachline(BIBLIOGRAPHY))) entries")
println("  figures: $(length(figure_paths))")
println("  compile check: OK")
println("  $(basename(COMPLETE_PDF)): main manuscript + Appendices A--B + Supplementary Material")
