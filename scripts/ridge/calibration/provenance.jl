"""Commit and package-environment checks for Ridge calibration artifacts."""

using SHA: sha256

const RIDGECAL_REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RIDGECAL_PROVENANCE_KEYS = (
    :git_commit, :julia_version, :pkg_manifest_hash, :manifest_hash, :schema_version
)

ridgecal_git_commit() =
    strip(read(`git -C $RIDGECAL_REPO_ROOT rev-parse HEAD`, String))
ridgecal_git_dirty() =
    !isempty(strip(read(`git -C $RIDGECAL_REPO_ROOT status --porcelain`, String)))
ridgecal_file_hash(path) = isfile(path) ? bytes2hex(sha256(read(path))) : "absent"

function ridgecal_current_source_provenance()
    return Dict{Symbol,Any}(
        :git_commit => ridgecal_git_commit(),
        :git_dirty => ridgecal_git_dirty(),
        :julia_version => string(VERSION),
        :pkg_manifest_hash => ridgecal_file_hash(
            joinpath(RIDGECAL_REPO_ROOT, "Manifest.toml")
        ),
    )
end

function ridgecal_provenance_mismatches(expected, current)
    return [
        key for key in (:git_commit, :julia_version, :pkg_manifest_hash) if
        current[key] != expected[key]
    ]
end

function ridgecal_verify_runtime_provenance(expected; require_clean::Bool=true)
    current = ridgecal_current_source_provenance()
    mismatches = ridgecal_provenance_mismatches(expected, current)
    isempty(mismatches) || error(
        "Ridge calibration runtime does not match its manifest: " *
        join(string.(mismatches), ", "),
    )
    require_clean && current[:git_dirty] &&
        error("Ridge calibration requires a clean worktree")
    return current
end
