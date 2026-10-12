"""Track reporting dependencies and validate each artifact's commits independently."""

using SHA: sha256
using TOML

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
    root = readchomp(`git -C $(dirname(abspath(path))) rev-parse --show-toplevel`)
    record = publication_record(root, path)
    if record !== nothing
        validate_publication_record(root, record; sources=false)
        isempty(record["analysis_commits"]) && error("no analysis provenance for $path")
        return sort!(collect(Pair{String,String}, record["analysis_commits"]); by=first)
    end
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

"""Read the single publication record; scientific analysis manifests remain separate."""
function publication_manifest(root)
    path = joinpath(root, "output", "publication.toml")
    return isfile(path) ? TOML.parsefile(path) : Dict{String,Any}("stages" => Dict{String,Any}())
end

"""Hash the exact files consumed or produced by a reporting step."""
function publication_hashes(root, paths)
    root = realpath(root)
    return Dict(relpath(realpath(joinpath(root, path)), root) =>
        bytes2hex(sha256(read(joinpath(root, path)))) for path in paths)
end

"""Write generated text only when its content changed."""
function write_if_changed(path, content)
    isfile(path) && read(path, String) == content && return false
    mkpath(dirname(path))
    write(path, content)
    return true
end

"""Record a completed reporting step without copying source contents or Git patches."""
function record_publication_stage(provenance; stage, inputs=(), outputs,
        analysis_commits=Dict{String,String}(), settings=Dict{String,Any}())
    root = provenance.root
    manifest = publication_manifest(root)
    record = Dict{String,Any}(
        "source_commit" => provenance.commit,
        "source_clean" => provenance.source_clean,
        "sources" => provenance.source_hashes,
        "inputs" => publication_hashes(root, inputs),
        "outputs" => publication_hashes(root, outputs),
        "analysis_commits" => Dict(analysis_commits),
        "settings" => settings,
        "julia_version" => string(VERSION),
    )
    # A commit or build date alone must not rewrite an otherwise identical record.
    previous = get(manifest["stages"], stage, nothing)
    if previous !== nothing && all(previous[key] == record[key] for key in
            ("source_clean", "sources", "inputs", "outputs", "analysis_commits", "settings", "julia_version"))
        return nothing
    end
    manifest["stages"][stage] = record
    write_if_changed(joinpath(root, "output", "publication.toml"),
        sprint(io -> TOML.print(io, manifest; sorted=true)))
    return nothing
end

"""Find the reporting step that produced an artifact."""
function publication_record(root, artifact)
    root = realpath(root)
    path = joinpath(root, artifact)
    relative = relpath(isfile(path) ? realpath(path) : abspath(path), root)
    records = [record for record in values(publication_manifest(root)["stages"])
        if haskey(record["outputs"], relative)]
    isempty(records) && return nothing
    length(records) == 1 || error("multiple publication records for $relative")
    return only(records)
end

"""Reject changed or missing dependencies and outputs, without comparing Git HEADs."""
function validate_publication_record(root, record; sources=true)
    groups = sources ? ("sources", "inputs", "outputs") : ("inputs", "outputs")
    for group in groups, (relative, expected) in record[group]
        path = joinpath(root, relative)
        isfile(path) && bytes2hex(sha256(read(path))) == expected ||
            error("stale publication output: $relative changed or is missing; rerun its reporting step")
    end
    return nothing
end

"""Validate an artifact against its separate publication record."""
function validate_publication_output(root, artifact; sources=true)
    record = publication_record(root, artifact)
    record === nothing && error("missing publication record for $artifact; rerun its reporting step")
    validate_publication_record(root, record; sources)
    return record
end

"""Return whether a recorded reporting step still matches its files and Julia version."""
function publication_stage_current(root, stage)
    record = get(publication_manifest(root)["stages"], stage, nothing)
    record === nothing && return false
    get(record, "julia_version", "") == string(VERSION) || return false
    try
        validate_publication_record(root, record)
    catch
        return false
    end
    return true
end
