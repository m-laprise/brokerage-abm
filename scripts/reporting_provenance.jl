"""
    scripts/reporting_provenance.jl

Commit-provenance helpers shared by scientific reporting scripts. Generated files
under `output/` are excluded from the cleanliness check because each reporting
stage updates them. Scientific analysis inputs must match a recorded commit.
Manuscript builders may include explicitly identified, uncommitted presentation
changes, which remain marked as dirty in downstream provenance. Retained analysis
inputs may record an earlier ancestor commit.
"""

"""
    reporting_git_provenance(path; require_clean=true, allowed_dirty_paths=())

Return the repository root and current Git commit. When `require_clean` is
true, fail if any path outside the top-level `output/` directory and
`allowed_dirty_paths` differs from the commit. `source_clean` still reports
whether all non-output files match the commit.
"""
function reporting_git_provenance(
    path;
    require_clean::Bool=true,
    allowed_dirty_paths=(),
)
    root = readchomp(`git -C $path rev-parse --show-toplevel`)
    commit = readchomp(`git -C $root rev-parse HEAD`)
    status_args = [
        "git",
        "-C",
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)output",
    ]
    source_status = strip(read(Cmd(status_args), String))
    source_clean = isempty(source_status)
    permitted_args = copy(status_args)
    append!(permitted_args, [":(exclude)$dirty_path" for dirty_path in allowed_dirty_paths])
    unpermitted_status = strip(read(Cmd(permitted_args), String))
    if require_clean && !isempty(unpermitted_status)
        error(
            "reporting source does not match commit $commit; commit or remove " *
            "these non-output, non-presentation changes before generating " *
            "scientific artifacts:\n" * unpermitted_status,
        )
    end
    return (;
        root,
        commit,
        short_commit=first(commit, 7),
        source_clean,
        source_status,
    )
end

const MANUSCRIPT_ITERATION_PATHS = (
    "paper",
    "scripts/paper/build_section.jl",
    "scripts/paper/build_supplement.jl",
    "scripts/paper/build_manuscript.jl",
    "scripts/reporting_provenance.jl",
    "test/test_reporting_provenance.jl",
)

"""
    manuscript_git_provenance(path)

Return provenance for a manuscript build while allowing uncommitted paper and
manuscript-builder edits. Changes to analysis code, specifications, tests, or
other source files still stop the build, except for this helper's focused test.
"""
function manuscript_git_provenance(path)
    return reporting_git_provenance(
        path;
        allowed_dirty_paths=MANUSCRIPT_ITERATION_PATHS,
    )
end

"""
    validate_analysis_commit(provenance, recorded; artifact="analysis input")

Resolve the commit recorded by a retained analysis input and require it to be an
ancestor of the current clean manuscript or rendering commit. Return the full
recorded commit. This permits presentation-only iteration without relabeling old
analysis artifacts as if they had been regenerated.
"""
function validate_analysis_commit(provenance, recorded; artifact="analysis input")
    candidate = strip(String(recorded))
    isempty(candidate) && error("$artifact has an empty analysis commit")
    resolve = Cmd([
        "git",
        "-C",
        provenance.root,
        "rev-parse",
        "--verify",
        "$(candidate)^{commit}",
    ])
    resolved = try
        readchomp(resolve)
    catch
        error("$artifact records an unknown analysis commit: $candidate")
    end
    ancestor = run(
        ignorestatus(
            Cmd([
                "git",
                "-C",
                provenance.root,
                "merge-base",
                "--is-ancestor",
                resolved,
                provenance.commit,
            ]),
        ),
    )
    success(ancestor) || error(
        "$artifact records analysis commit $resolved, which is not an ancestor " *
        "of manuscript commit $(provenance.commit)",
    )
    return resolved
end

"""
    recorded_analysis_commit(path)

Read the analysis commit recorded in a generated text artifact. New two-layer
headers use `Data analysis commit`; legacy headers use `Analysis commit` or
`analysis_commit=`.
"""
function recorded_analysis_commit(path)
    source = read(path, String)
    patterns = (
        r"(?im)^%\s*data analysis commit:\s*([0-9a-f]{7,40})\s*$",
        r"(?im)^%\s*analysis commit:\s*([0-9a-f]{7,40})\s*$",
        r"(?im)^data_analysis_commit=([0-9a-f]{7,40})\s*$",
        r"(?im)^analysis_commit=([0-9a-f]{7,40})\s*$",
    )
    for pattern in patterns
        match_result = match(pattern, source)
        isnothing(match_result) || return String(match_result[1])
    end
    error("generated artifact records no analysis commit: $path")
end
