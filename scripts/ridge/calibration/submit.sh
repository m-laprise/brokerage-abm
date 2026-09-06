#!/bin/bash
# Staged Della workflow for Ridge-penalty calibration.
#
#   ./scripts/ridge/calibration/submit.sh resolve
#   ./scripts/ridge/calibration/submit.sh setup
#   ./scripts/ridge/calibration/submit.sh manifest screen
#   ./scripts/ridge/calibration/submit.sh smoke screen
#   ./scripts/ridge/calibration/submit.sh compute screen
#   ./scripts/ridge/calibration/submit.sh summarize screen
#   ./scripts/ridge/calibration/submit.sh manifest confirm
#   ./scripts/ridge/calibration/submit.sh smoke confirm
#   ./scripts/ridge/calibration/submit.sh compute confirm
#   ./scripts/ridge/calibration/submit.sh summarize confirm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_ROOT="${BROKERAGE_ABM_DATA_ROOT:?set BROKERAGE_ABM_DATA_ROOT}"
ACCOUNT="${BROKERAGE_ABM_ACCOUNT:?set BROKERAGE_ABM_ACCOUNT}"
THROTTLE="${BROKERAGE_ABM_RIDGE_CALIBRATION_THROTTLE:-100}"
CPUS="${BROKERAGE_ABM_RIDGE_CALIBRATION_CPUS:-2}"
TIME="${BROKERAGE_ABM_RIDGE_CALIBRATION_TIME:-01:00:00}"
JULIA_MODULE="${BROKERAGE_ABM_JULIA_MODULE:-julia/1.11.3}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

SHA="$(git -C "$REPO" rev-parse --short HEAD)"
TAG="${BROKERAGE_ABM_RIDGE_CALIBRATION_TAG:?set a stable tag, for example ridge_${SHA}}"
CALIBRATION_DIR="$DATA_ROOT/ridge_calibration/$TAG"
LOGDIR="$CALIBRATION_DIR/logs"
export BROKERAGE_ABM_RIDGE_CALIBRATION_DIR="$CALIBRATION_DIR"

load_julia() {
    if ! command -v module >/dev/null 2>&1; then
        source /usr/share/Modules/init/bash 2>/dev/null || true
    fi
    module purge 2>/dev/null || true
    module load "$JULIA_MODULE"
}

action="${1:-help}"
stage="${2:-}"
case "$action" in
  resolve)
    load_julia
    cd "$REPO"
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --project --threads=auto -e '
        using Pkg
        isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General")
        Pkg.Registry.update()
        Pkg.resolve()
        Pkg.instantiate()
    '
    git diff --exit-code -- Project.toml Manifest.toml || {
        echo "ERROR: resolve changed Project.toml or Manifest.toml"
        exit 1
    }
    ;;

  setup)
    mkdir -p "$LOGDIR"
    jid=$(sbatch --parsable --account="$ACCOUNT" \
        --output="$LOGDIR/setup_%j.out" --error="$LOGDIR/setup_%j.err" \
        "$REPO/scripts/sweep/slurm_setup.sh" "$REPO")
    echo "setup submitted: $jid"
    echo "wait for SETUP_OK in $LOGDIR/setup_${jid}.out"
    ;;

  manifest)
    [[ "$stage" =~ ^(screen|confirm)$ ]] || { echo "invalid stage: $stage"; exit 2; }
    if [[ "$stage" == "confirm" ]]; then
        : "${BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_AGENT:?set selected agent penalty}"
        : "${BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_BROKER:?set selected broker penalty}"
        : "${BROKERAGE_ABM_RIDGE_SELECTION_NOTE:?set concise selection rationale}"
        export BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_AGENT
        export BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_BROKER
        export BROKERAGE_ABM_RIDGE_SELECTION_NOTE
    fi
    mkdir -p "$LOGDIR"
    srun --account="$ACCOUNT" --partition=cpu --time=00:10:00 \
        --cpus-per-task=1 --mem=4G --job-name="ridgecal_manifest_${stage}" \
        bash -c "command -v module >/dev/null 2>&1 || source /usr/share/Modules/init/bash 2>/dev/null || true; \
                 module purge 2>/dev/null || true; module load '$JULIA_MODULE'; cd '$REPO'; \
                 BROKERAGE_ABM_RIDGE_CALIBRATION_DIR='$CALIBRATION_DIR' \
                 julia --compiled-modules=strict --pkgimages=existing --project --threads=auto \
                 scripts/ridge/calibration/manifest.jl '$stage'"
    ;;

  smoke)
    [[ "$stage" =~ ^(screen|confirm)$ ]] || { echo "invalid stage: $stage"; exit 2; }
    stage_dir="$CALIBRATION_DIR/stages/$stage"
    mkdir -p "$LOGDIR"
    jid=$(sbatch --parsable --account="$ACCOUNT" --time=00:10:00 \
        --cpus-per-task="$CPUS" --array=0-0 \
        --output="$LOGDIR/smoke_${stage}_%A_%a.out" \
        --error="$LOGDIR/smoke_${stage}_%A_%a.err" \
        "$SCRIPT_DIR/slurm_run.sh" "$REPO" "$stage_dir" --smoke)
    echo "smoke submitted: $jid"
    ;;

  compute)
    [[ "$stage" =~ ^(screen|confirm)$ ]] || { echo "invalid stage: $stage"; exit 2; }
    stage_dir="$CALIBRATION_DIR/stages/$stage"
    source "$stage_dir/counts.env"
    mkdir -p "$LOGDIR"
    jid=$(sbatch --parsable --account="$ACCOUNT" --time="$TIME" \
        --cpus-per-task="$CPUS" --array="0-$((NRUNS - 1))%${THROTTLE}" \
        --output="$LOGDIR/${stage}_%A_%a.out" --error="$LOGDIR/${stage}_%A_%a.err" \
        "$SCRIPT_DIR/slurm_run.sh" "$REPO" "$stage_dir")
    echo "compute submitted: $jid ($NRUNS tasks)"
    ;;

  summarize)
    [[ "$stage" =~ ^(screen|confirm)$ ]] || { echo "invalid stage: $stage"; exit 2; }
    mkdir -p "$LOGDIR"
    srun --account="$ACCOUNT" --partition=cpu --time=00:20:00 \
        --cpus-per-task=1 --mem=8G --job-name="ridgecal_summary_${stage}" \
        bash -c "command -v module >/dev/null 2>&1 || source /usr/share/Modules/init/bash 2>/dev/null || true; \
                 module purge 2>/dev/null || true; module load '$JULIA_MODULE'; cd '$REPO'; \
                 BROKERAGE_ABM_RIDGE_CALIBRATION_DIR='$CALIBRATION_DIR' \
                 julia --compiled-modules=strict --pkgimages=existing --project --threads=auto \
                 scripts/ridge/calibration/summarize.jl '$stage'"
    ;;

  status)
    squeue -u "$USER" -o "%.18i %.24j %.8T %.10M %.6D %R" || true
    ;;

  *)
    sed -n '2,15p' "${BASH_SOURCE[0]}"
    echo "calibration directory: $CALIBRATION_DIR"
    ;;
esac
