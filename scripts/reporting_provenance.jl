"""Track reporting dependencies and validate each artifact's commits independently."""

using SHA: sha256
using Base64: base64encode

"""Resolve explicit source dependencies, including the Julia environment and this helper."""
function reporting_source_files(root, sources)
    isempty(sources) && error("reporting requires explicit source dependencies")
    root = realpath(root)
    paths = String[]
    for source in sources
        path = normpath(joinpath(root, source))
        ispath(path) && (path = realpath(path))
        relative = relpath(path, root)
        (relative == ".." || startswith(relative, "../") || relative == ".") &&
            error("reporting dependency must be inside the repository: $source")
        if isdir(path)
            for (directory, _, files) in walkdir(path), file in files
                push!(paths, relpath(joinpath(directory, file), root))
            end
        elseif isfile(path)
            push!(paths, relative)
        else
            error("missing reporting dependency: $source")
        end
    end
    for file in ("Project.toml", "Manifest.toml", "scripts/reporting_provenance.jl")
        isfile(joinpath(root, file)) && push!(paths, file)
    end
    isempty(paths) && error("no reporting source files were selected")
    return sort!(unique(paths))
end

"""
    reporting_git_provenance(path; sources, require_clean=true)

Check only declared source dependencies against HEAD. Scientific analyses require
committed dependencies. Presentation steps may record uncommitted revisions.
"""
function reporting_git_provenance(path; sources, require_clean::Bool=true)
    root = readchomp(`git -C $path rev-parse --show-toplevel`)
    commit = readchomp(`git -C $root rev-parse HEAD`)
    files = reporting_source_files(root, sources)
    pathspecs = [":(literal)$file" for file in files]
    source_status = strip(read(
        `git -C $root status --porcelain=v1 --untracked-files=all -- $pathspecs`, String
    ))
    tracked = Set(split(read(`git -C $root ls-files -z -- $pathspecs`, String), '\0'))
    # Ignored, untracked sources must not be mistaken for committed dependencies.
    untracked = setdiff(files, tracked)
    source_clean = isempty(source_status) && isempty(untracked)
    if !isempty(untracked)
        source_status *= "\nUncommitted dependencies: " * join(untracked, ", ")
    end
    require_clean && !source_clean && error(
        "reporting dependencies do not match commit $commit; commit these sources " *
        "before generating scientific analysis:\n$source_status",
    )
    source_hashes = Dict(file => bytes2hex(sha256(read(joinpath(root, file)))) for file in files)
    fingerprint = bytes2hex(sha256(join(["$file=$(source_hashes[file])" for file in files], "\n")))
    return (;
        root, commit, short_commit=first(commit, 7), source_clean, source_status,
        source_files=files, source_hashes, source_fingerprint=fingerprint,
    )
end

"""Record actual presentation dependencies without blocking uncommitted revisions."""
function manuscript_git_provenance(path; sources)
    return reporting_git_provenance(path; sources, require_clean=false)
end

"""
Resolve an immutable recorded commit. Independent artifacts may use different
commits or branches; compatibility is established from their data, not ancestry.
"""
function validate_analysis_commit(provenance, recorded; artifact="analysis input")
    candidate = strip(String(recorded))
    occursin(r"^[0-9a-fA-F]{7,40}$", candidate) ||
        error("$artifact must record a commit hash, not a mutable revision: $candidate")
    return try
        readchomp(pipeline(
            `git -C $(provenance.root) rev-parse --verify $(candidate * "^{commit}")`;
            stderr=devnull,
        ))
    catch
        error("$artifact records an unknown analysis commit: $candidate")
    end
end

"""Read every labeled analysis commit from a generated text artifact."""
function recorded_analysis_commits(path)
    pattern = r"(?im)^(?:%\s*)?([^\n:=]*analysis(?:[ _]git)?[ _]commit)\s*[:=]\s*([0-9a-f]{7,40})\s*$"
    records = Pair{String,String}[]
    for found in eachmatch(pattern, read(path, String))
        label = lowercase(strip(found[1]))
        any(record -> first(record) == label, records) &&
            error("duplicate analysis provenance label in $path: $label")
        push!(records, label => String(found[2]))
    end
    isempty(records) && error("generated artifact records no analysis commit: $path")
    return records
end

"""Read the primary analysis commit of a single-analysis artifact."""
function recorded_analysis_commit(path)
    records = Dict(recorded_analysis_commits(path))
    for label in ("data analysis commit", "analysis commit", "data_analysis_commit", "analysis_commit", "analysis_git_commit")
        haskey(records, label) && return records[label]
    end
    length(records) == 1 && return only(values(records))
    error("artifact has multiple analysis commits but no primary label: $path")
end

"""Validate independent analysis records and identify the exact consumed files."""
function analysis_input_provenance(provenance, paths)
    return map(collect(paths)) do path
        commits = [
            label => validate_analysis_commit(provenance, commit; artifact=path)
            for (label, commit) in recorded_analysis_commits(path)
        ]
        (; path=relpath(path, provenance.root), sha256=bytes2hex(sha256(read(path))), commits)
    end
end

"""Check recorded source content instead of requiring the current repository commit."""
function validate_source_hashes(provenance, artifact, paths)
    records = Dict(
        found[1] => found[2] for found in eachmatch(
            r"(?m)^% Source SHA256: (.+) ([0-9a-f]{64})$", read(artifact, String)
        )
    )
    for path in paths
        relative = relpath(joinpath(provenance.root, path), provenance.root)
        haskey(records, relative) || error("$artifact lacks source hash for $relative; rebuild it")
        records[relative] == bytes2hex(sha256(read(joinpath(provenance.root, path)))) ||
            error("$artifact is stale: $relative changed; rebuild it")
    end
    return nothing
end

"""Write source hashes and archive dirty presentation revisions as commented text."""
function write_source_provenance(io, provenance; prefix="% ")
    println(io, prefix, "Source fingerprint: ", provenance.source_fingerprint)
    for file in provenance.source_files
        println(io, prefix, "Source SHA256: ", file, " ", provenance.source_hashes[file])
    end
    if !provenance.source_clean
        pathspecs = [":(literal)$file" for file in provenance.source_files]
        patch = read(`git -C $(provenance.root) diff --binary HEAD -- $pathspecs`, String)
        # Encoding keeps archived TeX out of downstream command/reference scans.
        println(io, prefix, "Source patch (base64): ", base64encode(patch))
        tracked = Set(split(read(`git -C $(provenance.root) ls-files -z -- $pathspecs`, String), '\0'))
        for file in setdiff(provenance.source_files, tracked)
            println(io, prefix, "Untracked source (base64): ", file, " ",
                base64encode(read(joinpath(provenance.root, file))))
        end
    end
    return nothing
end
