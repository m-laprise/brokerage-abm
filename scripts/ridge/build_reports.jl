"""
    scripts/ridge/build_reports.jl

Compile the base Ridge and Ridge-ablation TeX sources under `paper/ridge/` and
write the report PDFs under `output/ridge/`. LaTeX auxiliary files are created
in temporary directories and discarded.

Run the corresponding analysis scripts first.

Usage: julia --project --threads=auto scripts/ridge/build_reports.jl
"""

include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PAPER = joinpath(REPO_ROOT, "paper", "ridge")
const REPORTING_PROVENANCE = reporting_git_provenance(REPO_ROOT)
const REPORTS = (
    (
        source="ridge_experiment.tex",
        output=joinpath(REPO_ROOT, "output", "ridge", "paired", "ridge_experiment.pdf"),
        provenance=(
            joinpath(REPO_ROOT, "output", "ridge", "paired", "analysis", "provenance.txt"),
            joinpath(REPO_ROOT, "output", "ridge", "paired", "figure_provenance.txt"),
        ),
    ),
    (
        source="ridge_ablation_experiment.tex",
        output=joinpath(
            REPO_ROOT, "output", "ridge", "ablations", "ridge_ablation_experiment.pdf"
        ),
        provenance=(
            joinpath(
                REPO_ROOT, "output", "ridge", "ablations", "analysis", "provenance.txt"
            ),
        ),
    ),
)

function build_report(report)
    source_path = joinpath(PAPER, report.source)
    isfile(source_path) || error("missing report source: $source_path")
    analysis_commits = String[]
    for provenance_path in report.provenance
        isfile(provenance_path) || error("missing report provenance: $provenance_path")
        push!(
            analysis_commits,
            validate_analysis_commit(
                REPORTING_PROVENANCE,
                recorded_analysis_commit(provenance_path);
                artifact=provenance_path,
            ),
        )
    end
    length(unique(analysis_commits)) == 1 ||
        error("report inputs record different analysis commits for $(report.source)")
    mkpath(dirname(report.output))
    mktempdir() do build
        log_path = joinpath(build, "lualatex.out")
        texmf_cache = joinpath(build, "texmf-cache")
        mkpath(texmf_cache)
        command = addenv(
            Cmd(
                Cmd([
                    "lualatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory=$build",
                    report.source,
                ]);
                dir=PAPER,
            ),
            "TEXMFVAR" => texmf_cache,
            "TEXMFCACHE" => texmf_cache,
        )
        ok = true
        for _ in 1:2
            process = run(
                pipeline(ignorestatus(command); stdout=log_path, stderr=log_path); wait=true
            )
            ok &= success(process)
        end
        ok || error("lualatex failed for $(report.source)")
        pdf_name = replace(report.source, r"\.tex$" => ".pdf")
        built_pdf = joinpath(build, pdf_name)
        isfile(built_pdf) || error("lualatex did not create $built_pdf")
        cp(built_pdf, report.output; force=true)
        println("wrote $(report.output)")
    end
    return nothing
end

foreach(build_report, REPORTS)
