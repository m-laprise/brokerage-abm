#!/bin/bash
# Run one neural-network calibration task from a stage manifest.
# Args: repository root, stage directory, and optional runner flags.
#SBATCH --job-name=brokerage_nn_calibration
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
set -euo pipefail

REPO="${1:?usage: slurm_run.sh <repo_root> <stage_dir> [runner flags]}"
export BROKERAGE_ABM_NN_CALIBRATION_STAGE_DIR="${2:?missing stage directory}"
shift 2

module purge
module load julia/1.11.3
export OMP_NUM_THREADS=1
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

cd "$REPO"
echo "task=${SLURM_ARRAY_TASK_ID:-manual} host=$(hostname) stage=$BROKERAGE_ABM_NN_CALIBRATION_STAGE_DIR"
julia --compiled-modules=strict --pkgimages=existing --project \
    --threads="${SLURM_CPUS_PER_TASK:-2}" scripts/nn_calibration/run.jl "$@"
