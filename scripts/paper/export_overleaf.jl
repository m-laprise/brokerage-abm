"""
Create the separate, editable Overleaf bundle from the current manuscript.
This development export is not part of the publication build.

Usage: julia --project --threads=auto scripts/paper/export_overleaf.jl
"""

include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTPUT = joinpath(ROOT, "output", "overleaf")
const NAME = "brokers_who_do_not_bridge_with_appendices.tex"
const ZIP = "brokerage_abm_overleaf_review.zip"

"""Extract the body of a standalone TeX document."""
document_body(text) = strip(split(split(text, raw"\begin{document}"; limit=2)[2],
    raw"\end{document}"; limit=2)[1])

"""Flatten the current sources, retaining the established appendix layout."""
function overleaf_source()
    results_path = joinpath(ROOT, "output", "main", "results_section.tex")
    validate_publication_output(ROOT, results_path)
    main = read(joinpath(ROOT, "paper", "manuscript.tex"), String)
    pseudo = read(joinpath(ROOT, "paper", "appendices", "simulation_pseudocode.tex"), String)
    specs = read(joinpath(ROOT, "paper", "appendices", "model_specifications.tex"), String)
    settings = Dict(m[1] => m[2] for m in eachmatch(
        r"(?m)^\\pvDefine\{([^}]+)\}\{(.*)\}\s*$",
        read(joinpath(ROOT, "output", "manuscript", "settings.tex"), String),
    ))
    main = replace(main,
        r"\A% Editable manuscript root\..*?\n\n"s =>
            "% Editable review copy. Compile with pdfLaTeX and Biber.\n",
        raw"\usepackage{graphicx,booktabs}" =>
            raw"\usepackage{graphicx,booktabs,array,calc,longtable,tabularx}" * "\n" *
            raw"\usepackage{algorithm}" * "\n" * raw"\usepackage[noend]{algpseudocode}",
        raw"\graphicspath{{output/main/}}" * "\n" => "",
        raw"\addbibresource{paper/references.bib}" => raw"\addbibresource{references.bib}",
        raw"\input{output/main/results_section.tex}" => read(results_path, String),
    )
    main = replace(main, r"(\\includegraphics(?:\[[^\]]*\])?\{)figures/" => s"\1figures/main/")
    macros = match(r"(\\definecolor\{diagblue\}.*?)(?=\\title)"s, pseudo)[1]
    unicode = match(r"\\ifdefined\\DeclareUnicodeCharacter.*?\\fi"s, specs).match
    main = replace(main, raw"\title{" => macros * "\n" * unicode * "\n\n" * raw"\title{"; count=1)
    appendices = String[]
    for (letter, source) in (("A", pseudo), ("B", specs))
        title = match(r"(?m)^\\title\{(.*)\}$", source)[1]
        depth_setting = match(r"\\setcounter\{secnumdepth\}\{([^}]+)\}", source)
        depth = depth_setting === nothing ? "3" : depth_setting[1]
        # Preserve the approved new-page start for each appendix.
        push!(appendices, """
        \\clearpage
        \\setcounter{section}{0}
        \\setcounter{secnumdepth}{$depth}
        \\renewcommand{\\thesection}{\\arabic{section}}
        \\renewcommand{\\thesubsection}{\\thesection.\\arabic{subsection}}
        \\makeatletter
        \\renewcommand{\\p@subsection}{}
        \\makeatother
        \\renewcommand{\\theHsection}{appendix$letter.\\arabic{section}}
        \\renewcommand{\\theHsubsection}{appendix$letter.\\arabic{section}.\\arabic{subsection}}
        \\section*{$title}
        """ * strip(replace(document_body(source), raw"\maketitle" => "",
            r"\\pv\{([^}]+)\}" => s -> settings[s[5:(end - 1)]])))
    end
    definitions = copy(settings)
    for path in ("output/supplement/figmeta.tex", "output/assessment_access/paper_values.tex")
        validate_publication_output(ROOT, joinpath(ROOT, path))
        for m in eachmatch(r"(?m)^\\pvDefine\{([^}]+)\}\{(.*)\}\s*$", read(joinpath(ROOT, path), String))
            definitions[m[1]] = m[2]
        end
    end
    supplement = document_body(read(joinpath(ROOT, "paper", "supplement.tex"), String))
    design = joinpath(ROOT, "output", "supplement", "simulation_design.tex")
    validate_publication_output(ROOT, design)
    supplement = replace(supplement,
        raw"\input{../output/supplement/simulation_design.tex}" => read(design, String))
    supplement = replace(supplement, r"\\pv\{([^}]+)\}" => s -> definitions[s[5:(end - 1)]])
    supplement = replace(supplement,
        r"(\\includegraphics(?:\[[^\]]*\])?\{)" => s"\1figures/supplement/")
    tail = join(appendices, "\n\n") * "\n\n" * """
    \\clearpage
    \\setcounter{figure}{0}
    \\renewcommand{\\thefigure}{S\\arabic{figure}}
    \\renewcommand{\\theHfigure}{supp.\\arabic{figure}}
    \\setcounter{table}{0}
    \\renewcommand{\\thetable}{S\\arabic{table}}
    \\renewcommand{\\theHtable}{supp.\\arabic{table}}
    \\setlength{\\parskip}{0.6\\baselineskip}
    \\setlength{\\parindent}{0pt}
    """ * supplement * "\n\n" * raw"\end{document}"
    return replace(main, raw"\end{document}" => tail)
end

source = overleaf_source()
figures = unique(m[1] for m in eachmatch(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source))
mkpath(OUTPUT)
mktempdir() do build
    write(joinpath(build, NAME), source)
    cp(joinpath(ROOT, "paper", "references.bib"), joinpath(build, "references.bib"))
    for figure in figures
        original = replace(figure, "figures/main/" => "output/main/figures/",
            "figures/supplement/" => "output/supplement/figures/")
        mkpath(dirname(joinpath(build, figure)))
        cp(joinpath(ROOT, original), joinpath(build, figure))
    end
    members = [NAME, "references.bib", sort(collect(figures))...]
    # Stable archive timestamps keep identical exports byte-for-byte unchanged.
    run(Cmd(`touch -t 198001010000 $members`; dir=build))
    run(Cmd(`zip -q -X $ZIP $members`; dir=build))
    run(`unzip -tq $(joinpath(build, ZIP))`)
    destination = joinpath(OUTPUT, ZIP)
    if !isfile(destination) || read(destination) != read(joinpath(build, ZIP))
        cp(joinpath(build, ZIP), destination; force=true)
    end
    write_if_changed(joinpath(OUTPUT, NAME), source)
end
println("Wrote the separate editing bundle to $OUTPUT")
