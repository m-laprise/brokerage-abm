#!/bin/bash
#SBATCH --job-name=brokerage_services_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail

REPO="${1:?usage: slurm_analyze.sh <repo_root> <sweep_dir>}"
SWEEP_DIR="${2:?usage: slurm_analyze.sh <repo_root> <sweep_dir>}"

module purge
module load julia/1.11.3
export OMP_NUM_THREADS=1
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"
export BROKERAGE_ABM_BROKER_SERVICE_SWEEP_DIR="$SWEEP_DIR"

ANALYSIS="analyze.jl"
OUTPUT_SUBDIR="analysis"
if [ "${3:-}" = "--pilot" ]; then
    ANALYSIS="analyze_pilot.jl"
    OUTPUT_SUBDIR="pilot_analysis"
fi
export BROKERAGE_ABM_BROKER_SERVICE_OUTPUT_DIR="$SWEEP_DIR/$OUTPUT_SUBDIR"

cd "$REPO"
julia --compiled-modules=strict --pkgimages=existing --project \
    --threads="${SLURM_CPUS_PER_TASK:-2}" "scripts/broker_services/$ANALYSIS"
