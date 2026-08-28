#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Sweep orchestrator (staged).
#
# Canonical flow (smoke-test then submit):
#   ./submit.sh resolve          # LOGIN node: registry update + resolve + download
#   ./submit.sh setup            # sbatch compute: precompile only; wait for it
#   ./submit.sh manifest         # srun compute: write manifest.{json,jld2}+counts.env
#   ./submit.sh smoke [idx]      # run ONE array task (default 0); inspect output
#   ./submit.sh compute          # submit the full compute array  (prints JOBID)
#   ./submit.sh plot             # submit the dependent plot array (afterany)
#   ./submit.sh status           # squeue for this user's sweep jobs
#
# Cluster compute nodes are assumed to have NO internet: every network/IO step (registry update,
# resolve, package download) happens in `resolve` on the login node; only the
# compute-heavy precompile/simulation run under sbatch/srun.
#
# Cluster settings come from the environment (BROKERAGE_ABM_ACCOUNT,
# BROKERAGE_ABM_DATA_ROOT). Override via env: BROKERAGE_ABM_DATA_ROOT,
# BROKERAGE_ABM_TAG, BROKERAGE_ABM_THROTTLE (array %K, default 200),
# BROKERAGE_ABM_CPUS (CPUs and Julia threads per simulation, default 2),
# BROKERAGE_ABM_PLOT_THROTTLE (default 24), and BROKERAGE_ABM_TIME (compute
# walltime, default 6 hours for the 500-period design).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${BROKERAGE_ABM_DATA_ROOT:?set BROKERAGE_ABM_DATA_ROOT to the directory that will hold the sweep data}"
ACCOUNT="${BROKERAGE_ABM_ACCOUNT:?set BROKERAGE_ABM_ACCOUNT to your SLURM account}"
THROTTLE="${BROKERAGE_ABM_THROTTLE:-200}"
COMPUTE_CPUS="${BROKERAGE_ABM_CPUS:-2}"
PLOT_THROTTLE="${BROKERAGE_ABM_PLOT_THROTTLE:-24}"
COMPUTE_TIME="${BROKERAGE_ABM_TIME:-06:00:00}"
JULIA_MODULE="${BROKERAGE_ABM_JULIA_MODULE:-julia/1.11.3}"

SHA="$(git -C "$REPO" rev-parse --short HEAD)"
TODAY="$(date +%Y-%m-%d)"
TAG="${BROKERAGE_ABM_TAG:-${TODAY}_${SHA}}"
SWEEP_DIR="$DATA_ROOT/sweep/$TAG"
LOGDIR="$SWEEP_DIR/logs"
ENVFILE="$SWEEP_DIR/sweep.env"

load_julia() {
    # make `module` available in a non-interactive shell, then load Julia
    if ! command -v module >/dev/null 2>&1; then
        source /usr/share/Modules/init/bash 2>/dev/null || true
    fi
    module purge 2>/dev/null || true
    module load "$JULIA_MODULE"
}

stage="${1:-help}"
case "$stage" in
  resolve)
    # LOGIN-NODE network step (compute nodes have no internet): update the
    # General registry, resolve a fresh Manifest (none is committed), and
    # download every package into the depot. Network/IO only — the compute-heavy
    # precompile is the `setup` stage.
    load_julia
    cd "$REPO"
    # JULIA_PKG_PRECOMPILE_AUTO=0: keep the compute-heavy precompile OUT of the
    # login node — instantiate here only downloads; precompile is the `setup` job.
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -e 'using Pkg; Pkg.Registry.update(); Pkg.resolve(); Pkg.instantiate(); @info "resolved + downloaded" julia=VERSION'
    echo "resolve + download complete; Manifest.toml written"
    ;;

  setup)
    mkdir -p "$LOGDIR"
    jid=$(sbatch --parsable --account="$ACCOUNT" \
        --output="$LOGDIR/setup_%j.out" --error="$LOGDIR/setup_%j.err" \
        "$SCRIPT_DIR/slurm_setup.sh" "$REPO")
    echo "setup job submitted: $jid"
    echo "  watch: tail -f $LOGDIR/setup_${jid}.out   (wait for SETUP_OK)"
    ;;

  manifest)
    # Run on a compute node (never the login node), even though it is light.
    mkdir -p "$LOGDIR"
    srun --account="$ACCOUNT" --partition=cpu --time=00:10:00 \
         --cpus-per-task=1 --mem=4G --job-name=brokerage_abm_manifest \
         bash -c "command -v module >/dev/null 2>&1 || source /usr/share/Modules/init/bash 2>/dev/null || true; \
                  module purge 2>/dev/null || true; module load $JULIA_MODULE; cd '$REPO'; \
                  BROKERAGE_ABM_SWEEP_DIR='$SWEEP_DIR' julia --project --threads=auto scripts/sweep/sweep_manifest.jl"
    echo "BROKERAGE_ABM_SWEEP_DIR=$SWEEP_DIR" > "$ENVFILE"
    echo "manifest + counts.env written under $SWEEP_DIR"
    ;;

  smoke)
    idx="${2:-0}"
    mkdir -p "$LOGDIR"
    jid=$(sbatch --parsable --account="$ACCOUNT" --time="$COMPUTE_TIME" \
        --cpus-per-task="$COMPUTE_CPUS" --array="${idx}-${idx}" \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        "$SCRIPT_DIR/slurm_sweep.sh" "$REPO" "$SWEEP_DIR")
    echo "smoke job submitted: $jid (task $idx, ${COMPUTE_CPUS} CPUs)"
    echo "  watch: tail -f $LOGDIR/${jid}_${idx}.out"
    ;;

  compute)
    [ -f "$SWEEP_DIR/counts.env" ] || { echo "run ./submit.sh manifest first"; exit 1; }
    source "$SWEEP_DIR/counts.env"
    jid=$(sbatch --parsable --account="$ACCOUNT" --time="$COMPUTE_TIME" \
        --cpus-per-task="$COMPUTE_CPUS" \
        --array="0-$((NRUNS - 1))%${THROTTLE}" \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        "$SCRIPT_DIR/slurm_sweep.sh" "$REPO" "$SWEEP_DIR")
    echo "COMPUTE_JOBID=$jid" >> "$ENVFILE"
    echo "compute array submitted: $jid  (0-$((NRUNS - 1))%${THROTTLE}, ${NRUNS} tasks, ${COMPUTE_CPUS} CPUs/task)"
    ;;

  plot)
    [ -f "$SWEEP_DIR/counts.env" ] || { echo "run ./submit.sh manifest first"; exit 1; }
    source "$SWEEP_DIR/counts.env"
    dep=""
    if [ -f "$ENVFILE" ]; then
        # shellcheck disable=SC1090
        source "$ENVFILE"
        [ -n "${COMPUTE_JOBID:-}" ] && dep="--dependency=afterany:${COMPUTE_JOBID}"
    fi
    [ -z "$dep" ] && echo "WARNING: no COMPUTE_JOBID found; submitting plot array with no dependency"
    jid=$(sbatch --parsable --account="$ACCOUNT" $dep \
        --array="0-$((NPLOT - 1))%${PLOT_THROTTLE}" \
        --output="$LOGDIR/plot_%A_%a.out" --error="$LOGDIR/plot_%A_%a.err" \
        "$SCRIPT_DIR/slurm_plot.sh" "$REPO" "$SWEEP_DIR")
    echo "plot array submitted: $jid  (0-$((NPLOT - 1))%${PLOT_THROTTLE}) ${dep:-(no dependency)}"
    ;;

  status)
    squeue -u "$USER" -o "%.18i %.12j %.8T %.10M %.6D %R" || true
    ;;

  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    echo "sweep dir: $SWEEP_DIR"
    ;;
esac
