"""Extract network snapshots from saved runs, then refresh retained reporting inputs.

Extraction on a compute node:
    julia --project --threads=auto network_snapshots.jl --extract SUMMARY SETTINGS OUTPUT
SETTINGS records each sweep's measurement interval and approved reporting windows.
Refresh locally with --refresh SNAPSHOTS. No simulations are run.
"""
module NetworkSnapshots

using JLD2, DataFrames, Statistics, SHA, TOML, Printf
include(joinpath(@__DIR__, "..", "analysis_settings.jl"))
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const MEASURES = string.(AnalysisSettings.NETWORK_MEASURES)
hashfile(path) = bytes2hex(sha256(read(path)))

"""Retain measured values and verify their correspondence to the previous summaries."""
function extract(summary_path, settings_path, output)
    summary, settings = load(summary_path), TOML.parsefile(settings_path)
    rows = Dict{String,Any}[]
    for condition in summary["conditions"]
        name, rel = condition["dataset"], condition["rel"]
        source, config = summary["meta"]["sources"][name], settings[name]
        source["manifest_hash"] == config["manifest_hash"] || error("manifest differs")
        source["simulation_commit"] == config["simulation_commit"] ||
            error("commit differs")
        path = joinpath(source["root"], rel, "data.jld2")
        hashfile(path) == condition["source_sha256"] ||
            error("saved aggregate changed: $path")
        raw = load(path)
        raw["provenance"]["manifest_hash"] == config["manifest_hash"] ||
            error("aggregate manifest differs")
        raw["provenance"]["git_commit"] == config["simulation_commit"] ||
            error("aggregate commit differs")
        raw["seeds"] == condition["seeds"] || error("seeds differ")
        raw["result_reldir"] == rel || error("condition differs")
        early, late = Tuple(config["early_window"]), Tuple(config["late_window"])
        interval = config["network_measure_interval"]
        periods = Dict(
            k => AnalysisSettings.network_snapshot_period(w, interval) for
            (k, w) in (("early", early), ("late", late))
        )
        all(collect(df.period) == collect(1:late[2]) for df in raw["mdfs"]) ||
            error("period coverage differs")
        old = Dict(
            m => [mean(df[late[1]:late[2], Symbol(m)]) for df in raw["mdfs"]] for
            m in MEASURES
        )
        all(isapprox(old[m], condition["metrics"][m]; atol=1e-12) for m in MEASURES) ||
            error("previous network summaries differ: $name $rel")
        values = Dict(
            k => Dict(
                m => [Float64(df[t, Symbol(m)]) for df in raw["mdfs"]] for m in MEASURES
            ) for (k, t) in periods
        )
        all(all(isfinite, v) for x in Base.values(values) for v in Base.values(x)) ||
            error("nonfinite snapshot")
        push!(
            rows,
            Dict(
                "dataset" => name,
                "rel" => rel,
                "seeds" => condition["seeds"],
                "cfg" => condition["cfg"],
                "source_sha256" => condition["source_sha256"],
                "periods" => periods,
                "values" => values,
                "previous_late" => old,
                "early_degree" =>
                    [mean(df[early[1]:early[2], :mean_degree]) for df in raw["mdfs"]],
                "late_degree" =>
                    [mean(df[late[1]:late[2], :mean_degree]) for df in raw["mdfs"]],
            ),
        )
        println(name, " ", rel)
        flush(stdout)
    end
    sources = Dict(
        name => Dict(
            k => source[k] for
            k in ("root", "simulation_commit", "manifest_hash", "manifest_sha256")
        ) for (name, source) in summary["meta"]["sources"]
    )
    metadata = Dict(
        "sources" => sources,
        "settings" => settings,
        "validation_summary_sha256" => hashfile(summary_path),
        "extractor_sha256" => hashfile(@__FILE__),
        "julia_version" => string(VERSION),
        "selection_sha256" => hashfile(joinpath(@__DIR__, "..", "analysis_settings.jl")),
    )
    jldsave(output; metadata, conditions=rows)
    println("Saved ", length(rows), " conditions to ", output)
end

"""Replace a cached-window estimate only after checking the saved seeds and values."""
function replace_network!(values, key, row, metric)
    previous, current = row["previous_late"][metric], row["values"]["late"][metric]
    (
        isapprox(values[key], previous; atol=1e-12) ||
        isapprox(values[key], current; atol=1e-12)
    ) || error("retained $metric differs: $(row["dataset"]) $(row["rel"])")
    values[key] = copy(current)
end

function read_table(path)
    lines = readlines(path)
    header = split(first(lines), '\t')
    return header, [Dict(zip(header, split(line, '\t'))) for line in lines[2:end]]
end

function write_table(path, header, rows)
    write_if_changed(
        path,
        join(
            [join(header, '\t'); [join((r[k] for k in header), '\t') for r in rows]], '\n'
        ) * "\n",
    )
end

function replace_definitions(path, definitions)
    source = read(path, String)
    for (key, value) in definitions
        pattern = Regex("(\\\\(?:pv|rv|rav)Define\\{" * key * "\\}\\{)[^}]+(\\})")
        occursin(pattern, source) || error("missing generated value $key")
        source = replace(source, pattern => s -> replace(s, r"\{[^{}]*\}$" => "{$value}"))
    end
    write_if_changed(path, source)
end

"""Refresh only network summaries; preserve other outcomes and their original provenance."""
function refresh(snapshot_path)
    snapshots = load(snapshot_path)
    rows, metadata = snapshots["conditions"], snapshots["metadata"]
    index = Dict((r["dataset"], r["rel"]) => r for r in rows)
    aliases = Dict{String,Dict{String,String}}()
    for (name, source) in metadata["sources"]
        manifest = ReportingSettings.manifest(
            joinpath(ROOT, "runs/sweeps", basename(source["root"]))
        )
        manifest["manifest_hash"] == source["manifest_hash"] || error("manifest mismatch")
        aliases[name] = Dict(
            String(c[:reldir]) => String(c[:result_reldir]) for c in manifest["cells"]
        )
    end
    row(name, rel) = index[(name, get(aliases[name], rel, rel))]
    values(name, rel) = row(name, rel)["values"]["late"]["betweenness"]
    outputs = String[]
    path(p) = joinpath(ROOT, p)
    save_data(p, d) = (jldsave(path(p); figdata=d); push!(outputs, p))
    snapshot_record = Dict(
        "path" => relpath(snapshot_path, ROOT),
        "sha256" => hashfile(snapshot_path),
        "measures" => collect(MEASURES),
        "periods" => only(unique(r["periods"] for r in rows)),
    )
    settings_path = path("output/reporting/settings.toml")
    settings = TOML.parsefile(settings_path)
    main = nothing
    for (name, p) in (
        ("nn", "output/main/figure_data.jld2"),
        ("ridge_pair", "output/ridge/paired/figure_data.jld2"),
        ("nn", "output/supplement/structural_figure_data.jld2"),
    )
        data = load(path(p), "figdata")
        ReportingSettings.retained(path(p), data)
        for group in ("oat_cells", "grid_cells", "rho_eta_cells", "regime_cells")
            for cell in get(data, group, [])
                r = row(name, cell["rel"])
                cell["seeds"] == r["seeds"] || error("retained seeds differ")
                for (stored, metric) in (
                    ("betw", "betweenness"),
                    ("constraint", "constraint"),
                    ("effsize", "effective_size"),
                )
                    haskey(get(cell, "seed_values", Dict()), stored) || continue
                    replace_network!(cell["seed_values"], stored, r, metric)
                    cell[stored] = mean(cell["seed_values"][stored])
                end
                if haskey(cell, "outcome_seed_values")
                    replace_network!(
                        cell["outcome_seed_values"],
                        "Betweenness centrality",
                        r,
                        "betweenness",
                    )
                    cell["outcomes"]["Betweenness centrality"] = mean(
                        cell["outcome_seed_values"]["Betweenness centrality"]
                    )
                end
            end
        end
        data["meta"]["network_snapshots"] = snapshot_record
        save_data(p, data)
        settings["artifacts"][p]["sha256"] = hashfile(path(p))
        p == "output/main/figure_data.jld2" && (main = data)
    end
    write_if_changed(settings_path, sprint(io -> TOML.print(io, settings; sorted=true)))
    push!(outputs, "output/reporting/settings.toml")
    baseline = ReportingSettings.retained(path("output/main/figure_data.jld2"), main)["baseline_reldir"]

    p = "output/assessment_access/figure_data.jld2"
    services = load(path(p))
    service_index = Dict(
        (String(r["cfg"]["broker_service"]), r["cfg"]["rho"], r["cfg"]["eta"]) => r for
        r in rows if startswith(r["dataset"], "services")
    )
    for saved in services["seed_rows"]
        String(saved[5]) == "betweenness" || continue
        r = service_index[(String(saved[1]), saved[2], saved[3])]
        j = only(findall(==(Int(saved[4])), r["seeds"]))
        old, new = r["previous_late"]["betweenness"][j],
        r["values"]["late"]["betweenness"][j]
        isapprox(saved[6], old; atol=1e-12) ||
            isapprox(saved[6], new; atol=1e-12) ||
            error("service seed differs")
        saved[6] = new
    end
    service_values(mode, rho, eta) = service_index[(String(mode), rho, eta)]["values"]["late"]["betweenness"]
    contrasts = Dict(
        "access" => ("assessment_only", "full"),
        "assessment" => ("access_only", "full"),
        "assessment_vs_access" => ("access_only", "assessment_only"),
    )
    for saved in services["summary_rows"]
        String(saved[4]) == "betweenness" || continue
        ci = monte_carlo_interval(
            service_values(saved[1:3]...); level=services["interval_level"]
        )
        saved[5:9] .= Any[ci.mean, ci.se, ci.lower, ci.upper, ci.n]
    end
    for saved in services["contrast_rows"]
        String(saved[4]) == "betweenness" || continue
        a, b = contrasts[String(saved[1])]
        service_index[(a, saved[2], saved[3])]["seeds"] ==
        service_index[(b, saved[2], saved[3])]["seeds"] || error("unpaired services")
        ci = monte_carlo_interval(
            service_values(b, saved[2], saved[3]) .- service_values(a, saved[2], saved[3]);
            level=services["interval_level"],
        )
        saved[5:9] .= Any[ci.mean, ci.se, ci.lower, ci.upper, ci.n]
    end
    services["network_snapshots"] = snapshot_record
    jldsave(path(p); (Symbol(k) => v for (k, v) in services)...)
    push!(outputs, p)
    for (name, key) in (
        ("seed_level", "seed_rows"),
        ("condition_summary", "summary_rows"),
        ("paired_contrasts", "contrast_rows"),
    )
        p = "output/assessment_access/$name.tsv"
        header = first(readlines(path(p)))
        write_if_changed(
            path(p),
            header * "\n" * join([join(r, '\t') for r in services[key]], '\n') * "\n",
        )
        push!(outputs, p)
    end

    p = "output/main/centrality_data.jld2"
    centrality = load(path(p), "data")
    for c in (centrality.nn..., centrality.ridge...)
        r = hasproperty(c, :rel) ? row("nn", c.rel) : row(c.dataset, baseline)
        c.seeds == r["seeds"] || error("centrality seeds differ")
        replace_network!(c.values, "betweenness", r, "betweenness")
    end
    centrality.metadata["network_snapshots"] = snapshot_record
    for relative in keys(centrality.metadata["validation_inputs"])
        centrality.metadata["validation_inputs"][relative] = hashfile(path(relative))
    end
    jldsave(path(p); data=centrality)
    push!(outputs, p)
    p = "output/supplement/structural_comparisons.jld2"
    structural = load(path(p), "data")
    variants = Dict(
        "pair" => "ridge_pair",
        "size_matched" => "ridge_size",
        "single_principal" => "ridge_one",
        "additive" => "ridge_additive",
    )
    for c in (structural.services..., structural.ridge...)
        r = if hasproperty(c, :mode)
            service_index[(c.mode, c.rho, c.eta)]
        else
            row(variants[c.key], baseline)
        end
        c.seeds == r["seeds"] || error("structural seeds differ")
        for metric in MEASURES
            replace_network!(c.values, metric, r, metric)
        end
    end
    structural.metadata["network_snapshots"] = snapshot_record
    for relative in keys(structural.metadata["validation_inputs"])
        structural.metadata["validation_inputs"][relative] = hashfile(path(relative))
    end
    jldsave(path(p); data=structural)
    push!(outputs, p)

    defs = Dict{String,String}()
    for (key, rel, window) in (
        ("betwRho0", "oat/rho=0.0", "late"),
        ("betwRho085", "oat/rho=0.85", "late"),
        ("betwRho1", "oat/rho=1.0", "late"),
        ("betwBaselineEarly", baseline, "early"),
        ("betwBaselineLate", baseline, "late"),
        ("betwEta0Late", "oat/eta=0.0", "late"),
        ("betwEta003Late", "oat/eta=0.03", "late"),
    )
        defs[key] = @sprintf("%.2f", mean(row("nn", rel)["values"][window]["betweenness"]))
    end
    nn = filter(r -> r["dataset"] == "nn", rows)
    defs["comoveN"] = string(
        count(
            r ->
                mean(r["late_degree"]) < mean(r["early_degree"]) &&
                mean(r["values"]["late"]["betweenness"]) >
                mean(r["values"]["early"]["betweenness"]),
            nn,
        ),
    )
    p = "output/main/values.tex"
    replace_definitions(path(p), defs)
    push!(outputs, p)
    refresh_ridge_tables!(outputs, row, baseline, variants)
    provenance = manuscript_git_provenance(
        ROOT;
        sources=(
            @__FILE__,
            "scripts/analysis_settings.jl",
            "scripts/reporting_settings.jl",
            "scripts/monte_carlo.jl",
        ),
    )
    # These preparation records describe files superseded by this refresh.
    manifest = publication_manifest(ROOT)
    for stage in (
        "scripts/paper/supp_robustness.jl --prepare", "scripts/retain_reporting_settings.jl"
    )
        pop!(manifest["stages"], stage, nothing)
    end
    write_if_changed(
        joinpath(ROOT, "output/publication.toml"),
        sprint(io -> TOML.print(io, manifest; sorted=true)),
    )
    record_publication_stage(
        provenance;
        stage="scripts/paper/network_snapshots.jl --refresh",
        inputs=(snapshot_path,),
        outputs,
        analysis_commits=Dict("snapshot revision base commit" => provenance.commit),
        settings=Dict("network_snapshots" => snapshot_record),
    )
    println("Refreshed ", length(outputs), " retained reporting files")
end

"""Refresh the retained Ridge comparisons without changing nonnetwork columns."""
function refresh_ridge_tables!(outputs, row, baseline, variants)
    path(p) = joinpath(ROOT, p)
    vector(name, rel, stage) =
        if stage == "late"
            row(name, rel)["values"][stage]["betweenness"]
        else
            row(name, rel)["previous_late"]["betweenness"]
        end
    function paired_columns(a, b, prefix; ablation=false)
        ci = monte_carlo_interval(b .- a)
        fields = if ablation
            (""=>:mean, "_se"=>:se, "_ci_lower"=>:lower, "_ci_upper"=>:upper, "_n"=>:n)
        else
            (
                "delta_"=>:mean,
                "delta_se_"=>:se,
                "delta_ci_lower_"=>:lower,
                "delta_ci_upper_"=>:upper,
                "delta_n_"=>:n,
            )
        end
        return Dict(
            (ablation ? prefix * suffix : suffix * "betweenness") => getproperty(ci, field)
            for (suffix, field) in fields
        )
    end
    function update!(saved, old, new)
        for (key, value) in new
            haskey(saved, key) || continue
            original = parse(Float64, saved[key])
            isapprox(original, old[key]; atol=1e-12) ||
                isapprox(original, value; atol=1e-12) ||
                error("Ridge column differs: $key")
            saved[key] = string(value)
        end
    end
    paired_means, ablation_means = nothing, nothing
    for kind in ("paired", "ablations")
        p = "output/ridge/$kind/analysis/condition_comparison.tsv"
        header, table = read_table(path(p))
        for saved in table
            summaries = map(("previous", "late")) do stage
                if kind == "paired"
                    rel = saved["result_reldir"]
                    a, b = vector("nn", rel, stage), vector("ridge_pair", rel, stage)
                    row("nn", rel)["seeds"] == row("ridge_pair", rel)["seeds"] ||
                        error("NN-Ridge seeds differ")
                    merge(
                        Dict("nn_betweenness" => mean(a), "ridge_betweenness" => mean(b)),
                        paired_columns(a, b, ""),
                    )
                else
                    pairrel, rel = saved["pair_result_reldir"], saved["ablation_result_reldir"]
                    a = vector("ridge_pair", pairrel, stage)
                    result = Dict{String,Real}("pair_betweenness" => mean(a))
                    for (key, dataset) in variants
                        key == "pair" && continue
                        b = vector(dataset, rel, stage)
                        row("ridge_pair", pairrel)["seeds"] == row(dataset, rel)["seeds"] || error("Ridge seeds differ")
                        result[key * "_betweenness"] = mean(b)
                        merge!(
                            result,
                            paired_columns(
                                a, b, key * "_minus_pair_betweenness"; ablation=true
                            ),
                        )
                    end
                    result
                end
            end
            update!(saved, summaries...)
        end
        write_table(path(p), header, table)
        push!(outputs, p)
        kind == "paired" ? (paired_means=table) : (ablation_means=table)
        p = "output/ridge/$kind/analysis/baseline_common_seeds.tsv"
        header, table = read_table(path(p))
        for saved in table
            summaries = map(("previous", "late")) do stage
                if kind == "paired"
                    names = Dict("nn" => "nn", "ridge" => "ridge_pair")
                else
                    names = variants
                end
                result = Dict{String,Float64}()
                for (key, dataset) in names
                    j = only(
                        findall(
                            ==(parse(Int, saved["seed"])),
                            row(dataset, baseline)["seeds"],
                        ),
                    )
                    result[key * "_betweenness"] = vector(dataset, baseline, stage)[j]
                end
                if kind == "paired"
                    result["delta_betweenness"] =
                        result["ridge_betweenness"] - result["nn_betweenness"]
                else
                    for key in keys(variants)
                        key == "pair" && continue
                        result[key * "_minus_pair_betweenness"] =
                            result[key * "_betweenness"] - result["pair_betweenness"]
                    end
                end
                result
            end
            update!(saved, summaries...)
        end
        write_table(path(p), header, table)
        push!(outputs, p)
    end
    f3(x) = @sprintf("%.3f", x)
    signed(x) = @sprintf("%+.3f", x)
    definitions = Dict{String,String}()
    add(key, x; fmt=f3) = (definitions[key * "_betweenness"] = fmt(x))
    a, b = vector("nn", baseline, "late"), vector("ridge_pair", baseline, "late")
    ci = monte_carlo_interval(b .- a)
    add("nnBaseline", mean(a));
    add("ridgeBaseline", mean(b))
    for (key, field) in (
        ("baselineDelta", :mean),
        ("baselineDeltaSE", :se),
        ("baselineDeltaLower", :lower),
        ("baselineDeltaUpper", :upper),
    )
        add(key, getproperty(ci, field); fmt=field==:se ? f3 : signed)
    end
    a = [parse(Float64, r["nn_betweenness"]) for r in paired_means]
    b = [parse(Float64, r["ridge_betweenness"]) for r in paired_means]
    d = b .- a
    add("nnMedian", median(a));
    add("ridgeMedian", median(b))
    for (key, value) in (
        ("deltaMean", mean(d)),
        ("deltaMedian", median(d)),
        ("deltaQ10", quantile(d, first(AnalysisSettings.SUMMARY_QUANTILES))),
        ("deltaQ90", quantile(d, last(AnalysisSettings.SUMMARY_QUANTILES))),
    )
        add(key, value; fmt=signed)
    end
    add("ridgeGreaterN", count(>(0), d); fmt=string)
    add("nnPositiveN", count(>(0), a); fmt=string);
    add("ridgePositiveN", count(>(0), b); fmt=string)
    add("conditionCorrelation", cor(a, b))
    p="output/ridge/paired/analysis/values.tex"
    replace_definitions(path(p), definitions);
    push!(outputs, p)
    empty!(definitions)
    a = [parse(Float64, r["pair_betweenness"]) for r in ablation_means]
    for (key, dataset) in variants
        b = [parse(Float64, r[key * "_betweenness"]) for r in ablation_means]
        add(key * "Baseline", mean(vector(dataset, baseline, "late")))
        add(key * "Median", median(b));
        add(key * "PositiveN", count(>(0), b); fmt=string)
        key == "pair" && continue
        ci = monte_carlo_interval(
            vector(dataset, baseline, "late") .- vector("ridge_pair", baseline, "late")
        )
        for (suffix, field) in (
            ("BaselineDelta", :mean),
            ("BaselineDeltaSE", :se),
            ("BaselineDeltaLower", :lower),
            ("BaselineDeltaUpper", :upper),
        )
            add(key * suffix, getproperty(ci, field); fmt=field==:se ? f3 : signed)
        end
        d = b .- a
        for (suffix, value) in (
            ("DeltaMean", mean(d)),
            ("DeltaMedian", median(d)),
            ("DeltaQ10", quantile(d, first(AnalysisSettings.SUMMARY_QUANTILES))),
            ("DeltaQ90", quantile(d, last(AnalysisSettings.SUMMARY_QUANTILES))),
        )
            add(key * suffix, value; fmt=signed)
        end
        add(key * "GreaterN", count(>(0), d); fmt=string);
        add(key * "LessN", count(<(0), d); fmt=string)
        add(key * "ConditionCorrelation", cor(a, b))
    end
    p="output/ridge/ablations/analysis/values.tex"
    replace_definitions(path(p), definitions);
    push!(outputs, p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 4 && first(ARGS) == "--extract"
        extract(ARGS[2:4]...)
    elseif length(ARGS) == 2 && first(ARGS) == "--refresh"
        include(joinpath(@__DIR__, "..", "monte_carlo.jl"))
        include(joinpath(@__DIR__, "..", "reporting_settings.jl"))
        include(joinpath(@__DIR__, "..", "reporting_provenance.jl"))
        refresh(abspath(ARGS[2]))
    else
        error("expected --extract SUMMARY SETTINGS OUTPUT or --refresh SNAPSHOTS")
    end
end
end
