"""
    scripts/reporting_provenance.jl

Commit-provenance gate shared by scientific reporting scripts. Generated files
under `output/` are excluded from the cleanliness check because each reporting
stage updates them. Source, specification, test, and paper files must match the
recorded commit exactly.
"""

"""
    reporting_git_provenance(path; require_clean=true)

Return the repository root and current Git commit. When `require_clean` is
true, fail if any path outside the top-level `output/` directory differs from
the commit.
"""
function reporting_git_provenance(path; require_clean::Bool=true)
    root = readchomp(`git -C $path rev-parse --show-toplevel`)
    commit = readchomp(`git -C $root rev-parse HEAD`)
    status_command = Cmd([
        "git",
        "-C",
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)output",
    ])
    source_status = strip(read(status_command, String))
    source_clean = isempty(source_status)
    if require_clean && !source_clean
        error(
            "reporting source does not match commit $commit; commit or remove " *
            "these non-output changes before generating scientific artifacts:\n" *
            source_status,
        )
    end
    return (; root, commit, short_commit=first(commit, 7), source_clean)
end
