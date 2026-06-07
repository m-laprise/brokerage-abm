# report_digest.jl — Agent B extraction script.
# Loads every OAT data.jld2 and phase summary.jld2, computes tail-mean steady
# state (period>30), early (31-50) vs late (181-200) windows for OAT, averages
# across the 5 seeds (mean +/- seed sd), and dumps report/digest.json.
# Run via srun. Uses ONLY JLD2, DataFrames, Statistics (no TransientBrokerage).
# Digest is written as JLD2 (digest.jld2) and the full numeric tables are printed
# to stdout (captured to report/digest.txt) as the human-readable source of cited numbers.

using JLD2, DataFrames, Statistics

const ROOT = "/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438"
const TBURN = 30
const EARLY = (31, 50)
const LATE  = (181, 200)

# ---- helpers ---------------------------------------------------------------

# Mean over a period window, ignoring NaN. Returns NaN if all NaN/empty.
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

# mean +/- sd across seeds of a per-seed window-statistic
function seedstat(mdfs, col::Symbol, statfun)
    vals = Float64[]
    for df in mdfs
        x = statfun(df, col)
        (x isa Number && !isnan(x)) && push!(vals, Float64(x))
    end
    if isempty(vals)
        return (mean=NaN, sd=NaN, n=0)
    end
    (mean=mean(vals), sd=(length(vals)>1 ? std(vals) : 0.0), n=length(vals))
end

# tuple -> small dict for JSON
nt2d(t) = Dict("mean"=>t.mean, "sd"=>t.sd, "n"=>t.n)

# All columns we want tail-mean steady state on (OAT). Capture cols only present in capture cells.
const STEADY_COLS = [
    :broker_holdout_r2, :agent_holdout_r2, :r2_gap,
    :broker_holdout_rank, :agent_holdout_rank, :rank_gap,
    :outsourcing_rate, :betweenness, :constraint, :effective_size,
    :q_self_mean, :q_broker_standard_mean, :q_broker_principal_mean,
    :principal_mode_share, :captured_position_count, :captured_origin_count,
    :principal_acceptance_rate, :capture_surplus_mean, :capture_loss_rate,
    :capture_scaled_mae, :capture_ready, :broker_confidence_mae,
    :mean_satisfaction_self, :mean_satisfaction_broker, :broker_reputation,
    :mean_degree, :median_degree, :min_degree, :max_degree,
    :access_count, :assessment_count, :broker_holdout_rmse, :agent_holdout_rmse,
    :n_broker_standard, :n_broker_principal, :n_self_matches,
]

# For dynamics (early vs late) we want these trajectory cols.
const TRAJ_COLS = [
    :betweenness, :constraint, :effective_size,
    :broker_holdout_r2, :agent_holdout_r2, :r2_gap,
    :rank_gap, :outsourcing_rate, :mean_degree,
    :principal_mode_share, :captured_position_count, :capture_scaled_mae,
    :capture_ready, :broker_holdout_rank, :agent_holdout_rank,
]

function digest_oat_cell(path)
    out = Dict{String,Any}()
    isfile(path) || (out["__missing__"]=true; return out)
    local mdfs, cfg
    try
        jldopen(path,"r") do f
            mdfs = f["mdfs"]
            cfg  = haskey(f, "config") ? f["config"] : nothing
        end
    catch e
        out["__corrupt__"] = string(e); return out
    end
    out["n_seeds"] = length(mdfs)
    # steady state
    steady = Dict{String,Any}()
    for c in STEADY_COLS
        if all(df->hasproperty(df,c), mdfs)
            steady[String(c)] = nt2d(seedstat(mdfs, c, tailmean))
        end
    end
    out["steady"] = steady
    # early/late trajectories
    early = Dict{String,Any}(); late = Dict{String,Any}()
    for c in TRAJ_COLS
        if all(df->hasproperty(df,c), mdfs)
            early[String(c)] = nt2d(seedstat(mdfs, c, (d,col)->winmean(d,col,EARLY[1],EARLY[2])))
            late[String(c)]  = nt2d(seedstat(mdfs, c, (d,col)->winmean(d,col,LATE[1],LATE[2])))
        end
    end
    out["early"] = early
    out["late"]  = late
    # correlation diagnostic for H1.3: per-seed tail Pearson corr between
    # betweenness and r2_gap across the post-burn tail (decoupling probe).
    corrs = Float64[]
    for df in mdfs
        sub = df[df.period .> TBURN, :]
        if hasproperty(sub,:betweenness) && hasproperty(sub,:r2_gap)
            b = Float64.(coalesce.(sub.betweenness, NaN))
            g = Float64.(coalesce.(sub.r2_gap, NaN))
            ok = .!isnan.(b) .& .!isnan.(g)
            if sum(ok) > 3 && std(b[ok])>0 && std(g[ok])>0
                push!(corrs, cor(b[ok], g[ok]))
            end
        end
    end
    out["corr_betw_r2gap"] = isempty(corrs) ? Dict("mean"=>NaN,"sd"=>NaN,"n"=>0) :
        Dict("mean"=>mean(corrs),"sd"=>(length(corrs)>1 ? std(corrs) : 0.0),"n"=>length(corrs))
    return out
end

function digest_phase(path)
    out = Dict{String,Any}()
    isfile(path) || (out["__missing__"]=true; return out)
    try
        jldopen(path,"r") do f
            T = f["tensors"]
            out["xkey"] = String(f["xkey"]); out["ykey"] = String(f["ykey"])
            out["model"] = String(f["model"])
            out["xvals"] = collect(Float64.(f["xvals"]))
            out["yvals"] = collect(Float64.(f["yvals"]))
            tens = Dict{String,Any}()
            for (k,M) in T
                # M is X x Y matrix; store as nested array rows=x
                tens[String(k)] = [ [ (isnan(M[i,j]) ? nothing : M[i,j]) for j in 1:size(M,2)] for i in 1:size(M,1) ]
            end
            out["tensors"] = tens
        end
    catch e
        out["__corrupt__"] = string(e)
    end
    return out
end

# ---- OAT axes --------------------------------------------------------------

oat = Dict{String,Any}()
oat_axes = [
    ("rho", ["0.0","0.5","1.0"], ["base","capture"]),
    ("eta", ["0.01","0.02","0.03"], ["base","capture"]),
    ("N",   ["500","1000","1500"], ["base","capture"]),
    ("reservation_frac", ["0.4","0.6","0.9","1.2"], ["base","capture"]),
    ("kappa_max", ["0.4","0.5","0.65"], ["capture"]),
]
for (axis, vals, models) in oat_axes
    axd = Dict{String,Any}()
    for v in vals
        cd = Dict{String,Any}()
        for m in models
            p = joinpath(ROOT, "oat", "$(axis)=$(v)", m, "data.jld2")
            cd[m] = digest_oat_cell(p)
        end
        axd[v] = cd
    end
    oat[axis] = axd
end

# ---- phase pairs -----------------------------------------------------------

phase = Dict{String,Any}()
for pair in ["rho_eta","rho_N","rho_r","eta_r","eta_N","r_N"]
    pd = Dict{String,Any}()
    for m in ["base","capture"]
        p = joinpath(ROOT, "phase", pair, m, "summary.jld2")
        pd[m] = digest_phase(p)
    end
    phase[pair] = pd
end

digest = Dict("root"=>ROOT, "tburn"=>TBURN, "early"=>EARLY, "late"=>LATE,
              "oat"=>oat, "phase"=>phase)

mkpath(joinpath(ROOT,"report"))
jldsave(joinpath(ROOT,"report","digest.jld2"); digest=digest)
println("WROTE ", joinpath(ROOT,"report","digest.jld2"))

# ---- Print a compact human-readable summary for sanity / citation ----------

function pm(d)
    d === nothing && return "NA"
    haskey(d,"mean") || return "NA"
    m=d["mean"]; s=d["sd"]
    (m isa Number && isnan(m)) && return "NaN"
    string(round(m,digits=4), "±", round(s,digits=4))
end

println("\n############ OAT STEADY-STATE DIGEST (tail mean period>30, mean±seedSD) ############")
for (axis, vals, models) in oat_axes
    println("\n==== AXIS $axis ====")
    for v in vals
        for m in models
            c = oat[axis][v][m]
            haskey(c,"__missing__") && (println("  $axis=$v/$m  MISSING"); continue)
            haskey(c,"__corrupt__") && (println("  $axis=$v/$m  CORRUPT: ", c["__corrupt__"]); continue)
            s = c["steady"]
            g(k)= haskey(s,k) ? pm(s[k]) : "-"
            println("  $axis=$v/$m: bR2=",g("broker_holdout_r2")," aR2=",g("agent_holdout_r2"),
                    " r2gap=",g("r2_gap")," rankgap=",g("rank_gap"),
                    " outsrc=",g("outsourcing_rate")," betw=",g("betweenness"),
                    " constr=",g("constraint")," effsz=",g("effective_size"),
                    " qself=",g("q_self_mean")," qbrk=",g("q_broker_standard_mean"))
            if m=="capture"
                println("        CAP: pshare=",g("principal_mode_share")," capt_pos=",g("captured_position_count"),
                        " capt_orig=",g("captured_origin_count")," pacc=",g("principal_acceptance_rate"),
                        " surplus=",g("capture_surplus_mean")," loss=",g("capture_loss_rate"),
                        " scMAE=",g("capture_scaled_mae")," ready=",g("capture_ready"))
            end
            println("        corr(betw,r2gap)=", pm(c["corr_betw_r2gap"]))
        end
    end
end

println("\n############ OAT EARLY(31-50) vs LATE(181-200) TRAJECTORIES ############")
for (axis, vals, models) in oat_axes
    for v in vals
        for m in models
            c = oat[axis][v][m]
            (haskey(c,"__missing__")||haskey(c,"__corrupt__")) && continue
            e=c["early"]; l=c["late"]
            ge(k)=haskey(e,k) ? pm(e[k]) : "-"; gl(k)=haskey(l,k) ? pm(l[k]) : "-"
            println("  $axis=$v/$m:")
            for k in ["betweenness","constraint","effective_size","r2_gap","outsourcing_rate","mean_degree","principal_mode_share","capture_scaled_mae","capture_ready"]
                (haskey(e,k)||haskey(l,k)) || continue
                println("      $k: early=",ge(k)," -> late=",gl(k))
            end
        end
    end
end

println("\n############ PHASE TENSORS (X rows = xkey, Y cols = ykey) ############")
for pair in ["rho_eta","rho_N","rho_r","eta_r","eta_N","r_N"]
    for m in ["base","capture"]
        ph = phase[pair][m]
        (haskey(ph,"__missing__")||haskey(ph,"__corrupt__")) && (println("$pair/$m MISSING/CORRUPT"); continue)
        println("\n---- $pair / $m  ($(ph["xkey"]) x $(ph["ykey"]))  xvals=$(ph["xvals"]) yvals=$(ph["yvals"]) ----")
        for met in sort(collect(keys(ph["tensors"])))
            M = ph["tensors"][met]
            print("   $met:")
            for row in M
                print("  [", join([ (x===nothing ? "NaN" : string(round(Float64(x),digits=3))) for x in row], ", "), "]")
            end
            println()
        end
    end
end
println("\nDONE")
