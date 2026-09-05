#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Staged sweep submission driver.
#
# Canonical flow (smoke-test then submit):
#   ./submit.sh resolve          # login node: registry update, resolve, and download
#   ./submit.sh setup            # submit precompilation and wait for completion
#   ./submit.sh manifest         # srun compute: write manifest.{json,jld2}+counts.env
#   ./submit.sh smoke [idx]      # run one array task (default 0), then inspect it
#   ./submit.sh compute          # submit the full compute array and print its job ID
#   ./submit.sh plot             # submit the dependent plot array (afterany)
#   ./submit.sh status           # squeue for this user's sweep jobs
#
# The `resolve` stage performs network operations on the login node. Precompilation
# and simulation run on compute nodes through `sbatch` or `srun`.
#
# Cluster settings come from the environment (BROKERAGE_ABM_ACCOUNT,
# BROKERAGE_ABM_DATA_ROOT). Override via env: BROKERAGE_ABM_DATA_ROOT,
# BROKERAGE_ABM_TAG, BROKERAGE_ABM_THROTTLE (array %K, default 200),
# BROKERAGE_ABM_CPUS (CPUs and Julia threads per simulation, default 2),
# BROKERAGE_ABM_PLOT_THROTTLE (default 24), BROKERAGE_ABM_TIME (compute
# walltime, default 6 hours for the 500-period design), learning-model settings
# documented in sweep_config.jl, JULIA_DEPOT_PATH, and JULIA_CPU_TARGET.
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
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"
export BROKERAGE_ABM_LEARNING_MODEL="${BROKERAGE_ABM_LEARNING_MODEL:-nn}"
export BROKERAGE_ABM_NN_ETA_LR_AGENT="${BROKERAGE_ABM_NN_ETA_LR_AGENT:-0.01}"
export BROKERAGE_ABM_NN_ETA_LR_BROKER="${BROKERAGE_ABM_NN_ETA_LR_BROKER:-0.01}"
export BROKERAGE_ABM_NN_E_INIT_AGENT="${BROKERAGE_ABM_NN_E_INIT_AGENT:-200}"
export BROKERAGE_ABM_NN_E_INIT_BROKER="${BROKERAGE_ABM_NN_E_INIT_BROKER:-200}"
export BROKERAGE_ABM_NN_TRAIN_STEPS_AGENT="${BROKERAGE_ABM_NN_TRAIN_STEPS_AGENT:-100}"
export BROKERAGE_ABM_NN_TRAIN_STEPS_BROKER="${BROKERAGE_ABM_NN_TRAIN_STEPS_BROKER:-100}"
export BROKERAGE_ABM_RIDGE_LAMBDA_AGENT="${BROKERAGE_ABM_RIDGE_LAMBDA_AGENT:-0.001}"
export BROKERAGE_ABM_RIDGE_LAMBDA_BROKER="${BROKERAGE_ABM_RIDGE_LAMBDA_BROKER:-0.001}"
export BROKERAGE_ABM_RIDGE_BROKER_VARIANT="${BROKERAGE_ABM_RIDGE_BROKER_VARIANT:-pair}"
export BROKERAGE_ABM_SWEEP_SCOPE="${BROKERAGE_ABM_SWEEP_SCOPE:-full}"
export BROKERAGE_ABM_N_SEEDS="${BROKERAGE_ABM_N_SEEDS:-20}"
export BROKERAGE_ABM_BASELINE_N_SEEDS="${BROKERAGE_ABM_BASELINE_N_SEEDS:-$BROKERAGE_ABM_N_SEEDS}"

SHA="$(git -C "$REPO" rev-parse --short HEAD)"
TODAY="$(date +%Y-%m-%d)"
DEFAULT_TAG="${TODAY}_${SHA}"
if [ "$BROKERAGE_ABM_LEARNING_MODEL" != "nn" ]; then
    DEFAULT_TAG="${DEFAULT_TAG}_${BROKERAGE_ABM_LEARNING_MODEL}_${BROKERAGE_ABM_RIDGE_BROKER_VARIANT}_lambda_a${BROKERAGE_ABM_RIDGE_LAMBDA_AGENT}_lambda_b${BROKERAGE_ABM_RIDGE_LAMBDA_BROKER}"
fi
if [ "$BROKERAGE_ABM_SWEEP_SCOPE" != "full" ]; then
    DEFAULT_TAG="${DEFAULT_TAG}_${BROKERAGE_ABM_SWEEP_SCOPE}"
fi
TAG="${BROKERAGE_ABM_TAG:-$DEFAULT_TAG}"
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
    # Login-node network step: update the
    # General registry, check Project/Manifest consistency, and download every
    # pinned package into the shared depot. Manifest.toml is committed, so any
    # resolver change is an error requiring local review. Network/IO only; the
    # compute-heavy precompile is the `setup` stage.
    load_julia
    cd "$REPO"
    # JULIA_PKG_PRECOMPILE_AUTO=0 keeps compute-heavy precompilation out of the
    # login node. A fresh shared depot has no registry, so add General once.
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --project --threads=auto -e '
        using Pkg
        isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General")
        Pkg.Registry.update()
        Pkg.resolve()
        Pkg.instantiate()
        @info "resolved + downloaded" julia=VERSION depot=DEPOT_PATH[1]
    '
    git diff --exit-code -- Project.toml Manifest.toml || {
        echo "ERROR: resolve changed Project.toml or the committed Manifest.toml"
        exit 1
    }
    echo "resolve + download complete; committed Manifest.toml unchanged"
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
                  BROKERAGE_ABM_SWEEP_DIR='$SWEEP_DIR' julia --compiled-modules=strict \
                  --pkgimages=existing --project --threads=auto scripts/sweep/sweep_manifest.jl"
    {
        echo "BROKERAGE_ABM_SWEEP_DIR=$SWEEP_DIR"
        echo "BROKERAGE_ABM_LEARNING_MODEL=$BROKERAGE_ABM_LEARNING_MODEL"
        echo "BROKERAGE_ABM_NN_ETA_LR_AGENT=$BROKERAGE_ABM_NN_ETA_LR_AGENT"
        echo "BROKERAGE_ABM_NN_ETA_LR_BROKER=$BROKERAGE_ABM_NN_ETA_LR_BROKER"
        echo "BROKERAGE_ABM_NN_E_INIT_AGENT=$BROKERAGE_ABM_NN_E_INIT_AGENT"
        echo "BROKERAGE_ABM_NN_E_INIT_BROKER=$BROKERAGE_ABM_NN_E_INIT_BROKER"
        echo "BROKERAGE_ABM_NN_TRAIN_STEPS_AGENT=$BROKERAGE_ABM_NN_TRAIN_STEPS_AGENT"
        echo "BROKERAGE_ABM_NN_TRAIN_STEPS_BROKER=$BROKERAGE_ABM_NN_TRAIN_STEPS_BROKER"
        echo "BROKERAGE_ABM_RIDGE_LAMBDA_AGENT=$BROKERAGE_ABM_RIDGE_LAMBDA_AGENT"
        echo "BROKERAGE_ABM_RIDGE_LAMBDA_BROKER=$BROKERAGE_ABM_RIDGE_LAMBDA_BROKER"
        echo "BROKERAGE_ABM_RIDGE_BROKER_VARIANT=$BROKERAGE_ABM_RIDGE_BROKER_VARIANT"
        echo "BROKERAGE_ABM_SWEEP_SCOPE=$BROKERAGE_ABM_SWEEP_SCOPE"
        echo "BROKERAGE_ABM_N_SEEDS=$BROKERAGE_ABM_N_SEEDS"
        echo "BROKERAGE_ABM_BASELINE_N_SEEDS=$BROKERAGE_ABM_BASELINE_N_SEEDS"
    } > "$ENVFILE"
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
