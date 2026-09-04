"""
    scripts/paper/build_publication.jl

Run the publication build steps that do not require raw sweep access. Run this
script from a clean committed worktree so all generated artifacts receive valid
provenance.

Usage: julia --project --threads=auto scripts/paper/build_publication.jl
"""

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const JULIA = Base.julia_cmd()
const BUILD_STEPS = (
    "scripts/paper/build_appendices.jl",
    "scripts/paper/figures.jl",
    "scripts/paper/ridge_supplement.jl",
    "scripts/ridge/paired_figures.jl",
    "scripts/paper/supp_figures.jl",
    "scripts/paper/build_section.jl",
    "scripts/paper/build_supplement.jl",
    "scripts/ridge/build_reports.jl",
    "scripts/paper/build_manuscript.jl",
)

for relative_path in BUILD_STEPS
    script = joinpath(ROOT, relative_path)
    isfile(script) || error("missing publication build step: $script")
    println("\n==> $relative_path")
    run(`$JULIA --project=$ROOT --threads=auto $script`)
end

println("\npublication build complete")
