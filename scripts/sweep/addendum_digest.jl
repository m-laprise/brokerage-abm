# addendum_digest.jl — Agent B follow-up extraction (delta, k_G, rho x delta).
# Mirrors scripts/sweep/report_digest.jl conventions: tail-mean steady state
# (period>30), early (31-50) vs late (181-200) windows, mean +/- seed SD over 5
# seeds, per-seed tail Pearson correlation diagnostics for H1.3.
# Reads ONLY the NEW cells added for the addendum; does NOT touch the original
# digest.{txt,jld2}. Writes report/digest_addendum.{jld2,txt-via-stdout}.
# Run via srun. Uses ONLY JLD2, DataFrames, Statistics (no TransientBrokerage).

using JLD2, DataFrames, Statistics

const ROOT  = "/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438"
const TBURN = 30
const EARLY = (31, 50)
const LATE  = (181, 200)

# ---- helpers (identical semantics to report_digest.jl) ---------------------

function winmean(df::DataFrame, col::Symbol, lo::Int, hi::Int)
    hasproperty(df, col) || return NaN
    v = Float64[]
    for r in eachrow(df)
        if r.period >= lo && r.period <= hi
            x = getproperty(r, col)
            if x !== missing && x !== nothing && !(x isa Number && isnan(x))
                push!(v, Float64(x))
            end
        end
    end
    isempty(v) ? NaN : mean(v)
end

tailmean(df, col) = winmean(df, col, TBURN+1, 1_000_000)

function seedstat(mdfs, col::Symbol, statfun)
    vals = Float64[]
    for df in mdfs
        x = statfun(df, col)
        (x isa Number && !isnan(x)) && push!(vals, Float64(x))
    end
    isempty(vals) && return (mean=NaN, sd=NaN, n=0)
    (mean=mean(vals), sd=(length(vals)>1 ? std(vals) : 0.0), n=length(vals))
end

nt2d(t) = Dict("mean"=>t.mean, "sd"=>t.sd, "n"=>t.n)

pm(d) = begin
    d === nothing && return "NA"
    haskey(d,"mean") || return "NA"
    m=d["mean"]; s=d["sd"]
    (m isa Number && isnan(m)) && return "NaN"
    string(round(m,digits=4), "±", round(s,digits=4))
end

# per-seed tail Pearson corr between two columns (decoupling probe).
function seedcorr(mdfs, a::Symbol, b::Symbol)
    corrs = Float64[]
    for df in mdfs
        sub = df[df.period .> TBURN, :]
        if hasproperty(sub,a) && hasproperty(sub,b)
            x = Float64.(coalesce.(getproperty(sub,a), NaN))
            y = Float64.(coalesce.(getproperty(sub,b), NaN))
            ok = .!isnan.(x) .& .!isnan.(y)
            if sum(ok) > 3 && std(x[ok])>0 && std(y[ok])>0
                push!(corrs, cor(x[ok], y[ok]))
            end
        end
    end
    isempty(corrs) ? Dict("mean"=>NaN,"sd"=>NaN,"n"=>0) :
        Dict("mean"=>mean(corrs),"sd"=>(length(corrs)>1 ? std(corrs) : 0.0),"n"=>length(corrs))
end

const STEADY_COLS = [
    :broker_holdout_r2, :agent_holdout_r2, :r2_gap,
    :broker_holdout_rank, :agent_holdout_rank, :rank_gap,
    :outsourcing_rate, :betweenness, :constraint, :effective_size,
    :q_self_mean, :q_broker_standard_mean,
    :principal_mode_share, :captured_position_count,
    :principal_acceptance_rate, :capture_surplus_mean, :capture_loss_rate,
    :capture_scaled_mae, :capture_ready,
    :mean_degree, :median_degree, :min_degree, :max_degree,
]

const TRAJ_COLS = [
    :betweenness, :constraint, :effective_size,
    :broker_holdout_r2, :agent_holdout_r2, :r2_gap,
    :rank_gap, :broker_holdout_rank, :agent_holdout_rank,
    :outsourcing_rate, :mean_degree,
    :principal_mode_share, :captured_position_count, :capture_scaled_mae,
    :capture_ready, :capture_surplus_mean,
]

function digest_oat_cell(path)
    out = Dict{String,Any}()
    isfile(path) || (out["__missing__"]=true; return out)
    local mdfs, cfg
    try
        jldopen(path,"r") do f
            mdfs = f["mdfs"]
            cfg  = haskey(f,"config") ? f["config"] : nothing
        end
    catch e
        out["__corrupt__"] = string(e); return out
    end
    out["n_seeds"] = length(mdfs)
    out["config_delta"] = (cfg !== nothing && haskey(cfg,"delta")) ? cfg["delta"] : nothing
    out["config_rho"]   = (cfg !== nothing && haskey(cfg,"rho"))   ? cfg["rho"]   : nothing
    out["config_value"] = (cfg !== nothing && haskey(cfg,"value")) ? cfg["value"] : nothing
    out["config_key"]   = (cfg !== nothing && haskey(cfg,"key"))   ? String(cfg["key"]) : nothing
    steady = Dict{String,Any}()
    for c in STEADY_COLS
        all(df->hasproperty(df,c), mdfs) && (steady[String(c)] = nt2d(seedstat(mdfs, c, tailmean)))
    end
    out["steady"] = steady
    early = Dict{String,Any}(); late = Dict{String,Any}()
    for c in TRAJ_COLS
        if all(df->hasproperty(df,c), mdfs)
            early[String(c)] = nt2d(seedstat(mdfs, c, (d,col)->winmean(d,col,EARLY[1],EARLY[2])))
            late[String(c)]  = nt2d(seedstat(mdfs, c, (d,col)->winmean(d,col,LATE[1],LATE[2])))
        end
    end
    out["early"] = early; out["late"] = late
    # Structural<->informational coupling diagnostics (1.3): same betw vs r2_gap
    # as the original report, plus a robust betw vs rank_gap variant.
    out["corr_betw_r2gap"]   = seedcorr(mdfs, :betweenness, :r2_gap)
    out["corr_betw_rankgap"] = seedcorr(mdfs, :betweenness, :rank_gap)
    return out
end

function digest_phase(path)
    out = Dict{String,Any}()
    isfile(path) || (out["__missing__"]=true; return out)
    try
        jldopen(path,"r") do f
            T = f["tensors"]
            out["xkey"]=String(f["xkey"]); out["ykey"]=String(f["ykey"]); out["model"]=String(f["model"])
            out["xvals"]=collect(Float64.(f["xvals"])); out["yvals"]=collect(Float64.(f["yvals"]))
            tens = Dict{String,Any}()
            for (k,M) in T
                tens[String(k)] = [ [ (isnan(M[i,j]) ? nothing : M[i,j]) for j in 1:size(M,2)] for i in 1:size(M,1) ]
            end
            out["tensors"] = tens
        end
    catch e
        out["__corrupt__"] = string(e)
    end
    return out
end

# Per grid-cell raw-data diagnostics for the rho x delta phase grid: early/late
# rank_gap & r2_gap dynamics and structural<->informational corr, plus a
# degeneracy check (are seed outputs identical across delta?).
function digest_rho_delta_cells(model)
    cells = Dict{String,Any}()
    for xi in 0:2, yi in 0:2
        p = joinpath(ROOT,"phase/rho_delta/cells/$(xi)_$(yi)/$(model)/data.jld2")
        c = Dict{String,Any}()
        if !isfile(p); c["__missing__"]=true; cells["$(xi)_$(yi)"]=c; continue; end
        local mdfs, cfg
        try
            jldopen(p,"r") do f; mdfs=f["mdfs"]; cfg=haskey(f,"config") ? f["config"] : nothing; end
        catch e; c["__corrupt__"]=string(e); cells["$(xi)_$(yi)"]=c; continue; end
        c["rho"]   = (cfg!==nothing && haskey(cfg,"rho"))   ? cfg["rho"]   : nothing
        c["delta"] = (cfg!==nothing && haskey(cfg,"delta")) ? cfg["delta"] : nothing
        for col in [:rank_gap,:r2_gap,:betweenness,:broker_holdout_rank,:agent_holdout_rank]
            if all(df->hasproperty(df,col), mdfs)
                c["early_"*String(col)] = nt2d(seedstat(mdfs,col,(d,k)->winmean(d,k,EARLY[1],EARLY[2])))
                c["late_"*String(col)]  = nt2d(seedstat(mdfs,col,(d,k)->winmean(d,k,LATE[1],LATE[2])))
                c["tail_"*String(col)]  = nt2d(seedstat(mdfs,col,tailmean))
            end
        end
        if model=="capture"
            for col in [:principal_mode_share,:capture_ready,:capture_surplus_mean]
                all(df->hasproperty(df,col), mdfs) && (c["tail_"*String(col)] = nt2d(seedstat(mdfs,col,tailmean)))
            end
        end
        c["corr_betw_r2gap"]   = seedcorr(mdfs,:betweenness,:r2_gap)
        c["corr_betw_rankgap"] = seedcorr(mdfs,:betweenness,:rank_gap)
        # fingerprint of seed-1 tail rank_gap for the delta-degeneracy check
        c["seed1_tail_rankgap"] = length(mdfs)>=1 ? tailmean(mdfs[1],:rank_gap) : NaN
        cells["$(xi)_$(yi)"] = c
    end
    return cells
end

# ---- collect ---------------------------------------------------------------

oat = Dict{String,Any}()
# delta axis (baseline delta=0.5 is rho=0.5 main baseline)
oat["delta"] = Dict{String,Any}()
for v in ["0.0","0.75"], m in ["base","capture"]
    oat["delta"][v] = get(oat["delta"], v, Dict{String,Any}())
    oat["delta"][v][m] = digest_oat_cell(joinpath(ROOT,"oat","delta=$(v)",m,"data.jld2"))
end
# baseline delta=0.5 = oat/rho=0.5 (re-read so the addendum is self-contained)
oat["delta"]["0.5"] = Dict{String,Any}()
for m in ["base","capture"]
    oat["delta"]["0.5"][m] = digest_oat_cell(joinpath(ROOT,"oat","rho=0.5",m,"data.jld2"))
end
# k_G axis (baseline k=6 = main rho=0.5 baseline)
oat["k"] = Dict{String,Any}()
for v in ["4","12"], m in ["base","capture"]
    oat["k"][v] = get(oat["k"], v, Dict{String,Any}())
    oat["k"][v][m] = digest_oat_cell(joinpath(ROOT,"oat","k=$(v)",m,"data.jld2"))
end
oat["k"]["6"] = Dict{String,Any}()
for m in ["base","capture"]
    oat["k"]["6"][m] = digest_oat_cell(joinpath(ROOT,"oat","rho=0.5",m,"data.jld2"))
end

phase = Dict{String,Any}()
phase["rho_delta"] = Dict{String,Any}()
for m in ["base","capture"]
    phase["rho_delta"][m] = digest_phase(joinpath(ROOT,"phase","rho_delta",m,"summary.jld2"))
    phase["rho_delta"][m*"_cells"] = digest_rho_delta_cells(m)
end

digest = Dict("root"=>ROOT,"tburn"=>TBURN,"early"=>EARLY,"late"=>LATE,"oat"=>oat,"phase"=>phase)
mkpath(joinpath(ROOT,"report"))
jldsave(joinpath(ROOT,"report","digest_addendum.jld2"); digest=digest)
println("WROTE ", joinpath(ROOT,"report","digest_addendum.jld2"))

# ---- print human-readable ---------------------------------------------------

println("\n############ ADDENDUM OAT STEADY-STATE (tail mean period>30, mean±seedSD) ############")
for (axis, vals) in [("delta",["0.0","0.5","0.75"]), ("k",["4","6","12"])]
    println("\n==== AXIS $axis ====")
    for v in vals
        for m in ["base","capture"]
            c = oat[axis][v][m]
            haskey(c,"__missing__") && (println("  $axis=$v/$m  MISSING"); continue)
            haskey(c,"__corrupt__") && (println("  $axis=$v/$m  CORRUPT: ",c["__corrupt__"]); continue)
            s=c["steady"]; g(k)= haskey(s,k) ? pm(s[k]) : "-"
            println("  $axis=$v/$m: [cfg delta=",c["config_delta"]," rho=",c["config_rho"]," key=",c["config_key"]," value=",c["config_value"],"]")
            println("      bR2=",g("broker_holdout_r2")," aR2=",g("agent_holdout_r2")," r2gap=",g("r2_gap"),
                    " brank=",g("broker_holdout_rank")," arank=",g("agent_holdout_rank")," rankgap=",g("rank_gap"))
            println("      outsrc=",g("outsourcing_rate")," betw=",g("betweenness")," constr=",g("constraint"),
                    " effsz=",g("effective_size")," meandeg=",g("mean_degree"))
            if m=="capture"
                println("      CAP: pshare=",g("principal_mode_share")," capt_pos=",g("captured_position_count"),
                        " pacc=",g("principal_acceptance_rate")," surplus=",g("capture_surplus_mean"),
                        " loss=",g("capture_loss_rate")," scMAE=",g("capture_scaled_mae")," ready=",g("capture_ready"))
            end
            println("      corr(betw,r2gap)=",pm(c["corr_betw_r2gap"]),"  corr(betw,rankgap)=",pm(c["corr_betw_rankgap"]))
        end
    end
end

println("\n############ ADDENDUM OAT EARLY(31-50) vs LATE(181-200) ############")
for (axis, vals) in [("delta",["0.0","0.5","0.75"]), ("k",["4","6","12"])]
    for v in vals, m in ["base","capture"]
        c = oat[axis][v][m]
        (haskey(c,"__missing__")||haskey(c,"__corrupt__")) && continue
        e=c["early"]; l=c["late"]
        ge(k)=haskey(e,k) ? pm(e[k]) : "-"; gl(k)=haskey(l,k) ? pm(l[k]) : "-"
        println("  $axis=$v/$m:")
        for k in ["betweenness","constraint","effective_size","r2_gap","rank_gap","broker_holdout_rank","agent_holdout_rank","mean_degree","principal_mode_share","capture_ready","capture_surplus_mean"]
            (haskey(e,k)||haskey(l,k)) || continue
            println("      $k: early=",ge(k)," -> late=",gl(k))
        end
    end
end

println("\n############ PHASE rho x delta TENSORS (rows=rho [0,0.5,1.0], cols=delta [0,0.5,0.75]) ############")
for m in ["base","capture"]
    ph = phase["rho_delta"][m]
    (haskey(ph,"__missing__")||haskey(ph,"__corrupt__")) && (println("rho_delta/$m MISSING/CORRUPT"); continue)
    println("\n---- rho_delta / $m  ($(ph["xkey"]) x $(ph["ykey"]))  xvals=$(ph["xvals"]) yvals=$(ph["yvals"]) ----")
    for met in sort(collect(keys(ph["tensors"])))
        M = ph["tensors"][met]
        print("   $met:")
        for row in M
            print("  [", join([ (x===nothing ? "NaN" : string(round(Float64(x),digits=4))) for x in row], ", "), "]")
        end
        println()
    end
end

println("\n############ rho x delta PER-CELL DIAGNOSTICS (early->late, corr, delta-degeneracy) ############")
for m in ["base","capture"]
    println("\n==== model=$m ====")
    cells = phase["rho_delta"][m*"_cells"]
    for xi in 0:2, yi in 0:2
        key="$(xi)_$(yi)"; c=cells[key]
        (haskey(c,"__missing__")||haskey(c,"__corrupt__")) && (println("  cell $key MISSING/CORRUPT"); continue)
        f(k)=haskey(c,k) ? pm(c[k]) : "-"
        println("  cell $key (rho=",c["rho"],", delta=",c["delta"],")  seed1_tail_rankgap=",round(c["seed1_tail_rankgap"],digits=4))
        println("      rank_gap: early=",f("early_rank_gap")," -> late=",f("late_rank_gap")," (tail=",f("tail_rank_gap"),")")
        println("      r2_gap:   early=",f("early_r2_gap")," -> late=",f("late_r2_gap")," (tail=",f("tail_r2_gap"),")")
        println("      brank:    early=",f("early_broker_holdout_rank")," -> late=",f("late_broker_holdout_rank"))
        println("      corr(betw,r2gap)=",f("corr_betw_r2gap")," corr(betw,rankgap)=",f("corr_betw_rankgap"))
        if m=="capture"
            println("      CAP tail: pshare=",f("tail_principal_mode_share")," ready=",f("tail_capture_ready")," surplus=",f("tail_capture_surplus_mean"))
        end
    end
end
println("\nDONE")
