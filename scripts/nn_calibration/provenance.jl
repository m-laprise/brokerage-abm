"""Commit and package-environment checks for NN calibration artifacts."""

using SHA: sha256

const NNCAL_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const NNCAL_PROVENANCE_KEYS = (
    :git_commit, :julia_version, :pkg_manifest_hash, :manifest_hash, :schema_version
)

nncal_git_commit() = strip(read(`git -C $NNCAL_REPO_ROOT rev-parse HEAD`, String))
function nncal_git_dirty()
    return !isempty(strip(read(`git -C $NNCAL_REPO_ROOT status --porcelain`, String)))
end
nncal_file_hash(path) = isfile(path) ? bytes2hex(sha256(read(path))) : "absent"

function nncal_current_source_provenance()
    return Dict{Symbol,Any}(
        :git_commit => nncal_git_commit(),
        :git_dirty => nncal_git_dirty(),
        :julia_version => string(VERSION),
        :pkg_manifest_hash => nncal_file_hash(joinpath(NNCAL_REPO_ROOT, "Manifest.toml")),
    )
end

function nncal_provenance_mismatches(expected, current)
    return [
        key for key in (:git_commit, :julia_version, :pkg_manifest_hash) if
        current[key] != expected[key]
    ]
end

function nncal_verify_runtime_provenance(expected; require_clean::Bool=true)
    current = nncal_current_source_provenance()
    mismatches = nncal_provenance_mismatches(expected, current)
    isempty(mismatches) || error(
        "calibration runtime does not match its manifest: " * join(string.(mismatches), ", "),
    )
    require_clean && current[:git_dirty] &&
        error("calibration runtime requires a clean worktree")
    return current
end
