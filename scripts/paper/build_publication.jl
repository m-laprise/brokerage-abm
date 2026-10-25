"""
    scripts/paper/build_publication.jl

Run the publication build steps from retained analyses and reporting manifests.
Run this script after refreshing those inputs. Each input retains its own
analysis commit; presentation changes are recorded in output/publication.toml.
Unchanged steps are skipped. Overleaf is a separate editing export.

Usage: julia --project --threads=auto scripts/paper/build_publication.jl [--force]
"""

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const JULIA = Base.julia_cmd()
include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
ARGS in (String[], ["--force"]) || error("usage: build_publication.jl [--force]")
const BUILD_STEPS = (
    ("scripts/paper/build_settings.jl",),
    ("scripts/paper/build_appendices.jl",),
    ("scripts/paper/audit_convergence.jl",),
    ("scripts/paper/figures.jl", "--assessment-not-access"),
    ("scripts/paper/figures.jl", "--information-sources"),
    ("scripts/paper/figures.jl", "--metadata"),
    ("scripts/paper/centrality_and_assessment.jl",),
    ("scripts/paper/ridge_supplement.jl",),
    ("scripts/ridge/paired_figures.jl",),
    ("scripts/paper/supp_figures.jl",),
    ("scripts/assessment_access/figure_2.jl",),
    ("scripts/assessment_access/supplement_figure.jl",),
    ("scripts/assessment_access/paper_values.jl",),
    ("scripts/paper/build_section.jl",),
    ("scripts/paper/build_simulation_design.jl",),
    ("scripts/paper/build_supplement.jl",),
    ("scripts/paper/build_manuscript.jl",),
)

for (relative_path, arguments...) in BUILD_STEPS
    script = joinpath(ROOT, relative_path)
    isfile(script) || error("missing publication build step: $script")
    stage = join((relative_path, arguments...), " ")
    if isempty(ARGS) && publication_stage_current(ROOT, stage)
        println("Up to date: $stage")
        continue
    end
    println("\n==> $relative_path")
    run(`$JULIA --project=$ROOT --threads=auto $script $arguments`)
end

println("\npublication build complete")
