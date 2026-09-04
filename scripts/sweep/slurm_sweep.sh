#!/bin/bash
# Compute array with one independent task per unique condition and seed. Submit
# with the array range, throttle, and log paths on the command line:
#   sbatch --array=0-$((NRUNS - 1))%200 \
#          --output=$LOGDIR/%A_%a.out --error=$LOGDIR/%A_%a.err \
#          slurm_sweep.sh <repo_root> <sweep_dir>
# Args: $1 = repo root, $2 = sweep dir (exported as BROKERAGE_ABM_SWEEP_DIR).
#SBATCH --job-name=brokerage_abm_run
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
set -euo pipefail

REPO="${1:?usage: slurm_sweep.sh <repo_root> <sweep_dir>}"
export BROKERAGE_ABM_SWEEP_DIR="${2:?usage: slurm_sweep.sh <repo_root> <sweep_dir>}"

module purge
module load julia/1.11.3
export OMP_NUM_THREADS=1   # Julia threads handle the only intra-task parallelism
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
# Must match slurm_setup.sh exactly so every task loads the one multiversioned
# precompile cache (no per-node recompiles / lock races across the mixed CPUs).
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

cd "$REPO"
echo "task=${SLURM_ARRAY_TASK_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} sweep=$BROKERAGE_ABM_SWEEP_DIR"

run_args=()
if [[ "${BROKERAGE_ABM_RERUN:-0}" == "1" ]]; then
    run_args+=(--rerun)
fi

julia --compiled-modules=strict --pkgimages=existing --project \
    --threads="${SLURM_CPUS_PER_TASK:-2}" scripts/sweep/sweep_run.jl "${run_args[@]}"
