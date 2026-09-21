"""
Retain raw access and assessment counts for the manuscript's early/late windows.

Reads a completed sweep without running simulations. The renderer computes the
window means and paired intervals from these counts. The export records its full
source and the sweep-reader hash separately from the repository commit.

Required: BROKERAGE_ABM_SWEEP_DIR.
Optional: BROKERAGE_ABM_REPO, BROKERAGE_ABM_ACCESS_WINDOWS_PATH.
Usage: julia --project --threads=auto scripts/paper/access_windows.jl
Output: output/main/access_windows.jld2
"""

using JLD2
using SHA: sha256

const REPO = get(ENV, "BROKERAGE_ABM_REPO", normpath(joinpath(@__DIR__, "..", "..")))
const READER = joinpath(REPO, "scripts", "sweep", "sweep_results.jl")
include(READER)
include(joinpath(REPO, "scripts", "reporting_provenance.jl"))

"""Export the two reporting windows, preserving regime and seed identities."""
function main()
    root = ENV["BROKERAGE_ABM_SWEEP_DIR"]
    output = get(
        ENV, "BROKERAGE_ABM_ACCESS_WINDOWS_PATH",
        joinpath(REPO, "output", "main", "access_windows.jld2"),
    )
    provenance = reporting_git_provenance(
        REPO; sources=(@__FILE__, "scripts/sweep/sweep_results.jl",),
    )
    sweep = load_sweep_dataset(root)
    horizon = Int(sweep.meta[:T])
    early = (51, 70)
    late = (horizon - 19, horizon)
    early[2] < late[1] || error("reporting windows overlap")
    periods = [collect(early[1]:early[2]); collect(late[1]:late[2])]
    regimes = map(sweep.results) do result
        indices = findall(period -> period in periods, result.mdfs[1].period)
        all(df -> collect(df.period[indices]) == periods, result.mdfs) ||
            error("window coverage differs for $(result.rel)")
        access = reduce(hcat, (Int.(df.access_count[indices]) for df in result.mdfs))
        assessment = reduce(hcat, (Int.(df.assessment_count[indices]) for df in result.mdfs))
        all(>=(0), access) && all(>=(0), assessment) || error("negative placement counts")
        Dict(
            "rel" => result.rel,
            "seeds" => copy(result.seeds),
            "access_count" => access,
            "assessment_count" => assessment,
        )
    end
    source = read(@__FILE__, String)
    metadata = Dict(
        "sweep" => basename(root),
        "manifest_hash" => sweep.manifest_hash,
        "schema_version" => sweep.schema_version,
        "repository_commit" => provenance.commit,
        "repository_source_clean" => provenance.source_clean,
        "exporter_source" => source,
        "exporter_sha256" => bytes2hex(sha256(source)),
        "reader_sha256" => bytes2hex(sha256(read(READER))),
        "early_window" => early,
        "late_window" => late,
    )
    mkpath(dirname(output))
    jldsave(output; metadata, periods, regimes)
    println("Exported $(length(regimes)) regimes, $(sum(length(r["seeds"]) for r in regimes)) seeds.")
    println("Windows: $early and $late. Sweep manifest: $(sweep.manifest_hash)")
    println("Raw counts saved to $output")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
