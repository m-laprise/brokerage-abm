"""
    scripts/paper/build_manuscript.jl

Create a submission-ready manuscript bundle from the editable TeX root and the
generated results section. The output is a single manuscript TeX file with the
results section inlined, plus its bibliography, referenced figures, Biber output,
and a compile-checked PDF under `output/manuscript/`. The builder also writes a
complete working-paper PDF containing the manuscript, Appendix A, Appendix B,
and the Supplementary Material.

Run `scripts/paper/build_section.jl` first whenever the results prose, values,
captions, figures, or analysis provenance changes. This script never edits or
regenerates scientific results.

Usage: julia --project --threads=auto scripts/paper/build_manuscript.jl
"""

using Dates: now

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PAPER = joinpath(ROOT, "paper")
const MAIN_OUTPUT = joinpath(ROOT, "output", "main")
const SOURCE = joinpath(PAPER, "transientbrokerage.tex")
const RESULTS = joinpath(MAIN_OUTPUT, "results_section.tex")
const BIBLIOGRAPHY = joinpath(PAPER, "references.bib")
const OUTPUT = joinpath(ROOT, "output", "manuscript")
const OUTPUT_TEX = joinpath(OUTPUT, "transientbrokerage.tex")
const PSEUDOCODE = joinpath(ROOT, "simulation_pseudocode.pdf")
const SPECIFICATIONS = joinpath(ROOT, "model_specifications.pdf")
const SUPPLEMENT = joinpath(ROOT, "output", "supplement", "supplement.pdf")
const COMPLETE_PDF = joinpath(OUTPUT, "transientbrokerage_complete.pdf")
const INPUT_MARKER = "\\input{output/main/results_section.tex}"
const SOURCE_BIBLIOGRAPHY = "\\addbibresource{paper/references.bib}"
const BUNDLE_BIBLIOGRAPHY = "\\addbibresource{references.bib}"
const SOURCE_GRAPHICSPATH = "\\graphicspath{{output/main/}}"
const BUNDLE_GRAPHICSPATH = "\\graphicspath{{./}}"

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

length(findall(INPUT_MARKER, source)) == 1 ||
    fail("editable root must contain exactly one $INPUT_MARKER")
length(findall(SOURCE_BIBLIOGRAPHY, source)) == 1 ||
    fail("editable root must contain exactly one repo-local bibliography declaration")
length(findall(SOURCE_GRAPHICSPATH, source)) == 1 ||
    fail("editable root must contain exactly one repo-local graphics path")
occursin(r"\\section\{ABM Results:", results) ||
    fail("generated fragment does not contain the ABM results section")

figure_paths = unique([
    match[1] for
    match in eachmatch(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", results)
])
isempty(figure_paths) && fail("generated results section references no figures")
for relative_path in figure_paths
    isfile(joinpath(MAIN_OUTPUT, relative_path)) ||
        fail("missing generated figure $relative_path")
end

flattened = replace(source, INPUT_MARKER => results)
flattened = replace(flattened, SOURCE_BIBLIOGRAPHY => BUNDLE_BIBLIOGRAPHY)
flattened = replace(flattened, SOURCE_GRAPHICSPATH => BUNDLE_GRAPHICSPATH)
occursin(INPUT_MARKER, flattened) && fail("results input was not flattened")

header = """
% transientbrokerage.tex: GENERATED, do not edit.
% Built by scripts/paper/build_manuscript.jl on $(now()).
% The editable root is paper/transientbrokerage.tex; the results provenance is
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
    cp(OUTPUT_TEX, joinpath(build, "transientbrokerage.tex"))
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
                    `latexmk -pdf -interaction=nonstopmode -halt-on-error transientbrokerage.tex`;
                    dir=build,
                ),
            );
            stdout=build_log,
            stderr=build_log,
        );
        wait=true,
    )
    final_log_path = joinpath(build, "transientbrokerage.log")
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
        artifact = joinpath(build, "transientbrokerage.$extension")
        isfile(artifact) || fail("latexmk did not create $artifact")
        cp(artifact, joinpath(OUTPUT, basename(artifact)); force=true)
    end
end

pdfunite = Sys.which("pdfunite")
isnothing(pdfunite) && fail("pdfunite is required to assemble the complete PDF")

components = [
    joinpath(OUTPUT, "transientbrokerage.pdf"),
    PSEUDOCODE,
    SPECIFICATIONS,
    SUPPLEMENT,
]
mktempdir() do build
    combined = joinpath(build, "transientbrokerage_complete.pdf")
    process = run(ignorestatus(Cmd([pdfunite, components..., combined])); wait=true)
    success(process) || fail("pdfunite failed while assembling the complete PDF")
    isfile(combined) || fail("pdfunite did not create the complete PDF")
    cp(combined, COMPLETE_PDF; force=true)
end

println("wrote complete manuscript bundle to $OUTPUT")
println("  transientbrokerage.tex: flattened full manuscript")
println("  references.bib: $(count(line -> startswith(line, '@'), eachline(BIBLIOGRAPHY))) entries")
println("  figures: $(length(figure_paths))")
println("  compile check: OK")
println("  transientbrokerage_complete.pdf: manuscript + Appendices A--B + Supplementary Material")
