"""Manifest provenance fields that identify a simulation shard."""
const SHARD_PROVENANCE_KEYS = (
    :schema_version, :git_commit, :julia_version, :pkg_manifest_hash, :manifest_hash
)

"""Return whether `path` contains every provenance value required by `expected`."""
function shard_is_current(path, expected)
    isfile(path) || return false
    try
        return jldopen(path, "r") do file
            all(SHARD_PROVENANCE_KEYS) do key
                name = string(key)
                haskey(file, name) && file[name] == expected[key]
            end
        end
    catch
        return false
    end
end

"""Return whether an aggregate artifact records the expected provenance."""
function aggregate_is_current(path, expected)
    isfile(path) || return false
    try
        return jldopen(path, "r") do file
            haskey(file, "provenance") || return false
            provenance = file["provenance"]
            all(SHARD_PROVENANCE_KEYS) do key
                name = string(key)
                haskey(provenance, name) && provenance[name] == expected[key]
            end
        end
    catch
        return false
    end
end
