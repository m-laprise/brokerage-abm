#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Staged sweep submission driver.
#
# Canonical flow (smoke-test then submit):
#   ./submit.sh resolve          # login node: registry update, resolve, and download
#   ./submit.sh setup            # submit precompilation and wait for completion
#   ./submit.sh manifest         # srun compute: write manifest.{json,jld2}+counts.env
#   ./submit.sh smoke [idx]      # run one array task (default 0), then inspect it
#   ./submit.sh pilot            # assessment vs access experiment: baseline pilot
#   ./submit.sh pilot-analyze    # assessment vs access experiment: pilot summary
#   ./submit.sh compute          # submit the full compute array and print its job ID
#   ./submit.sh plot             # submit the dependent plot array (afterany)
#   ./submit.sh analyze          # assessment vs access experiment: final analysis
#   ./submit.sh status           # squeue for this user's sweep jobs
#
# The `resolve` stage performs network operations on the login node. Precompilation
# and simulation run on compute nodes through `sbatch` or `srun`.
#
# Scientific sweep settings are defined in sweep_config.jl.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${BROKERAGE_ABM_DATA_ROOT:?set BROKERAGE_ABM_DATA_ROOT to the directory that will hold the sweep data}"
ACCOUNT="${BROKERAGE_ABM_ACCOUNT:?set BROKERAGE_ABM_ACCOUNT to your SLURM account}"
THROTTLE="${BROKERAGE_ABM_THROTTLE:-200}"
COMPUTE_CPUS="${BROKERAGE_ABM_CPUS:-2}"
PLOT_THROTTLE="${BROKERAGE_ABM_PLOT_THROTTLE:-24}"
JULIA_MODULE="${BROKERAGE_ABM_JULIA_MODULE:-julia/1.11.3}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"
export BROKERAGE_ABM_LEARNING_MODEL="${BROKERAGE_ABM_LEARNING_MODEL:-nn}"
DEFAULT_COMPUTE_TIME="01:01:00"
COMPUTE_TIME="${BROKERAGE_ABM_TIME:-$DEFAULT_COMPUTE_TIME}"
COMPUTE_TIME_MIN="${BROKERAGE_ABM_TIME_MIN:-}"
COMPUTE_QOS="${BROKERAGE_ABM_QOS:-short}"
export BROKERAGE_ABM_NN_ETA_LR_AGENT="${BROKERAGE_ABM_NN_ETA_LR_AGENT:-0.003}"
export BROKERAGE_ABM_NN_ETA_LR_BROKER="${BROKERAGE_ABM_NN_ETA_LR_BROKER:-0.03}"
export BROKERAGE_ABM_NN_E_INIT_AGENT="${BROKERAGE_ABM_NN_E_INIT_AGENT:-100}"
export BROKERAGE_ABM_NN_E_INIT_BROKER="${BROKERAGE_ABM_NN_E_INIT_BROKER:-100}"
export BROKERAGE_ABM_NN_TRAIN_STEPS_AGENT="${BROKERAGE_ABM_NN_TRAIN_STEPS_AGENT:-50}"
export BROKERAGE_ABM_NN_TRAIN_STEPS_BROKER="${BROKERAGE_ABM_NN_TRAIN_STEPS_BROKER:-50}"
export BROKERAGE_ABM_RIDGE_LAMBDA_AGENT="${BROKERAGE_ABM_RIDGE_LAMBDA_AGENT:-0.001}"
export BROKERAGE_ABM_RIDGE_LAMBDA_BROKER="${BROKERAGE_ABM_RIDGE_LAMBDA_BROKER:-0.001}"
export BROKERAGE_ABM_RIDGE_BROKER_VARIANT="${BROKERAGE_ABM_RIDGE_BROKER_VARIANT:-pair}"
export BROKERAGE_ABM_SWEEP_SCOPE="${BROKERAGE_ABM_SWEEP_SCOPE:-full}"
export BROKERAGE_ABM_N_SEEDS="${BROKERAGE_ABM_N_SEEDS:-20}"
export BROKERAGE_ABM_BASELINE_N_SEEDS="${BROKERAGE_ABM_BASELINE_N_SEEDS:-$BROKERAGE_ABM_N_SEEDS}"
export BROKERAGE_ABM_SERVICE_BASELINE_N_SEEDS="${BROKERAGE_ABM_SERVICE_BASELINE_N_SEEDS:-50}"

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
        echo "BROKERAGE_ABM_SERVICE_BASELINE_N_SEEDS=$BROKERAGE_ABM_SERVICE_BASELINE_N_SEEDS"
    } > "$ENVFILE"
    echo "manifest + counts.env written under $SWEEP_DIR"
    ;;

  smoke)
    idx="${2:-0}"
    mkdir -p "$LOGDIR"
    jid=$(sbatch --parsable --account="$ACCOUNT" --time=00:15:00 \
        --cpus-per-task="$COMPUTE_CPUS" --array="${idx}-${idx}" \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        "$SCRIPT_DIR/slurm_sweep.sh" "$REPO" "$SWEEP_DIR" --smoke)
    echo "smoke job submitted: $jid (task $idx, ${COMPUTE_CPUS} CPUs)"
    echo "  watch: tail -f $LOGDIR/${jid}_${idx}.out"
    ;;

  compute)
    [ -f "$SWEEP_DIR/counts.env" ] || { echo "run ./submit.sh manifest first"; exit 1; }
    source "$SWEEP_DIR/counts.env"
    time_min_args=()
    if [ -n "$COMPUTE_TIME_MIN" ]; then
        time_min_args+=(--time-min="$COMPUTE_TIME_MIN")
    fi
    jid=$(sbatch --parsable --account="$ACCOUNT" --qos="$COMPUTE_QOS" \
        --time="$COMPUTE_TIME" "${time_min_args[@]}" \
        --cpus-per-task="$COMPUTE_CPUS" \
        --array="0-$((NRUNS - 1))%${THROTTLE}" \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        "$SCRIPT_DIR/slurm_sweep.sh" "$REPO" "$SWEEP_DIR")
    echo "COMPUTE_JOBID=$jid" >> "$ENVFILE"
    echo "compute array submitted: $jid  (0-$((NRUNS - 1))%${THROTTLE}, ${NRUNS} tasks, ${COMPUTE_CPUS} CPUs/task)"
    ;;

  pilot)
    [ -f "$SWEEP_DIR/counts.env" ] || { echo "run ./submit.sh manifest first"; exit 1; }
    source "$SWEEP_DIR/counts.env"
    [ "${NPILOT:-0}" -gt 0 ] || { echo "this sweep has no pilot task set"; exit 1; }
    jid=$(sbatch --parsable --account="$ACCOUNT" --qos="$COMPUTE_QOS" \
        --time="$COMPUTE_TIME" --cpus-per-task="$COMPUTE_CPUS" \
        --array="${PILOT_ARRAY}%${NPILOT}" \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        "$SCRIPT_DIR/slurm_sweep.sh" "$REPO" "$SWEEP_DIR")
    echo "PILOT_JOBID=$jid" >> "$ENVFILE"
    echo "pilot array submitted: $jid  (${NPILOT} tasks, ${COMPUTE_CPUS} CPUs/task)"
    ;;

  pilot-analyze)
    [ -f "$ENVFILE" ] || { echo "run the manifest and pilot stages first"; exit 1; }
    source "$ENVFILE"
    [ -n "${PILOT_JOBID:-}" ] || { echo "run ./submit.sh pilot first"; exit 1; }
    jid=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:${PILOT_JOBID}" \
        --output="$LOGDIR/pilot_analyze_%j.out" --error="$LOGDIR/pilot_analyze_%j.err" \
        "$REPO/scripts/broker_services/slurm_analyze.sh" "$REPO" "$SWEEP_DIR" --pilot)
    echo "PILOT_ANALYSIS_JOBID=$jid" >> "$ENVFILE"
    echo "pilot analysis submitted: $jid  (afterok:${PILOT_JOBID})"
    ;;

  plot)
    [ -f "$SWEEP_DIR/counts.env" ] || { echo "run ./submit.sh manifest first"; exit 1; }
    source "$SWEEP_DIR/counts.env"
    if [ "$NPLOT" -eq 0 ]; then
        echo "no plot jobs for this sweep scope"
        exit 0
    fi
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
    echo "PLOT_JOBID=$jid" >> "$ENVFILE"
    echo "plot array submitted: $jid  (0-$((NPLOT - 1))%${PLOT_THROTTLE}) ${dep:-(no dependency)}"
    ;;

  analyze)
    [ -f "$ENVFILE" ] || { echo "run the manifest and plot stages first"; exit 1; }
    source "$ENVFILE"
    [ "${BROKERAGE_ABM_SWEEP_SCOPE:-}" = "broker_services" ] || {
        echo "analyze stage is defined only for the broker-services scope"
        exit 1
    }
    [ -n "${PLOT_JOBID:-}" ] || { echo "run ./submit.sh plot first"; exit 1; }
    jid=$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:${PLOT_JOBID}" \
        --output="$LOGDIR/analyze_%j.out" --error="$LOGDIR/analyze_%j.err" \
        "$REPO/scripts/broker_services/slurm_analyze.sh" "$REPO" "$SWEEP_DIR")
    echo "ANALYSIS_JOBID=$jid" >> "$ENVFILE"
    echo "analysis job submitted: $jid  (afterok:${PLOT_JOBID})"
    ;;

  status)
    squeue -u "$USER" -o "%.18i %.12j %.8T %.10M %.6D %R" || true
    ;;

  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    echo "sweep dir: $SWEEP_DIR"
    ;;
esac
