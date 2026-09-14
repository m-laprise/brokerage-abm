"""Preserve existing PDFs when a build changes only dates or document identifiers."""

"""Return comparison bytes with volatile fields removed from PDF metadata only."""
function pdf_comparison_bytes(bytes::Vector{UInt8})
    source = String(copy(bytes))
    ending = match(r"startxref\s+(\d+)\s+%%EOF\s*\z"a, source)
    isnothing(ending) && return bytes
    offset = tryparse(Int, ending[1])
    (isnothing(offset) || !(0 <= offset < ncodeunits(source))) && return bytes
    trailer = SubString(source, offset + 1)
    stop = match(r"(?:\r?\nstream\r?\n|startxref)"a, trailer)
    isnothing(stop) && return bytes
    metadata = String(codeunits(trailer)[1:(stop.offset - 1)])

    # Dates belong to the referenced Info object, never to page text or streams.
    info = match(r"/Info\s+(\d+)\s+(\d+)\s+R\b"a, metadata)
    if !isnothing(info)
        pattern = Regex(
            "^$(info[1]) $(info[2]) obj\\s*\\n(.*?)\\nendobj\\b", "msa"
        )
        objects = collect(eachmatch(pattern, source))
        if length(objects) == 1 && !occursin(r"\bstream\b"a, only(objects)[1])
            object = only(objects).match
            normalized = replace(
                object,
                r"^/(?:CreationDate|ModDate)[ \t]*\(D:\d{14}(?:Z|[+-]\d{2}'\d{2}')?\)[ \t]*\r?$"ma => "",
            )
            source = replace(source, object => normalized; count=1)
        end
    end

    # pdfTeX uses hex IDs; pdfunite can use escaped literal strings instead.
    normalized_metadata = replace(
        metadata,
        r"/ID\s*\[\s*(?:<[0-9A-Fa-f]+>|\((?:\\.|[^\\()])*\))\s*(?:<[0-9A-Fa-f]+>|\((?:\\.|[^\\()])*\))\s*\]"sa => "",
    )
    source = replace(source, metadata => normalized_metadata; count=1)
    return collect(codeunits(source))
end

"""Copy a built PDF only when more than volatile build metadata has changed."""
function update_pdf(source_path, destination)
    candidate = read(source_path)
    startswith(String(copy(candidate)), "%PDF-") || error("not a PDF: $source_path")
    if isfile(destination) &&
        pdf_comparison_bytes(candidate) == pdf_comparison_bytes(read(destination))
        println("unchanged $destination")
        return false
    end
    mkpath(dirname(destination))
    cp(source_path, destination; force=true)
    println("wrote $destination")
    return true
end
