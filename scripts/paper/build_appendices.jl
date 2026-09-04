"""
    scripts/paper/build_appendices.jl

Compile the standalone simulation pseudocode and model specifications from
`paper/appendices/` into `output/appendices/`. Auxiliary files are created in a
temporary directory and discarded.

Usage: julia --project --threads=auto scripts/paper/build_appendices.jl
"""

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SOURCE_DIR = joinpath(ROOT, "paper", "appendices")
const OUTPUT_DIR = joinpath(ROOT, "output", "appendices")
const APPENDICES = ("simulation_pseudocode.tex", "model_specifications.tex")

mkpath(OUTPUT_DIR)
for source_name in APPENDICES
    source_path = joinpath(SOURCE_DIR, source_name)
    isfile(source_path) || error("missing appendix source: $source_path")
    mktempdir() do build
        log_path = joinpath(build, "pdflatex.out")
        command = Cmd(
            Cmd([
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory=$build",
                source_name,
            ]);
            dir=SOURCE_DIR,
        )
        ok = true
        for _ in 1:2
            process = run(
                pipeline(ignorestatus(command); stdout=log_path, stderr=log_path); wait=true
            )
            ok &= success(process)
        end
        ok || error("failed to compile $source_name")
        pdf_name = replace(source_name, r"\.tex$" => ".pdf")
        pdf_path = joinpath(build, pdf_name)
        isfile(pdf_path) || error("pdflatex did not create $pdf_path")
        cp(pdf_path, joinpath(OUTPUT_DIR, pdf_name); force=true)
        println("wrote $(joinpath(OUTPUT_DIR, pdf_name))")
    end
end
