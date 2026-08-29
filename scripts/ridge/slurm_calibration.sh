#!/bin/bash
# Baseline Ridge-penalty calibration. Submit as a 70-task array, with one task
# per penalty and calibration seed. The final sweep uses a separate seed set.
#
# Required environment:
#   BROKERAGE_ABM_RIDGE_CALIBRATION_DIR  durable output directory
# Optional environment:
#   BROKERAGE_ABM_REPO                   repository root
#   JULIA_DEPOT_PATH                     shared Julia depot
#
# Candidate penalties: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1
# Calibration seeds:   9001:9010
#SBATCH --job-name=brokerage_ridge_calibration
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail

REPO="${BROKERAGE_ABM_REPO:-/projects/BSTEWART/mlaprise/brokerage-abm}"
OUTPUT_DIR="${BROKERAGE_ABM_RIDGE_CALIBRATION_DIR:?set calibration output directory}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

LAMBDAS=(0.0001 0.0003 0.001 0.003 0.01 0.1 1.0)
N_SEEDS=10
N_TASKS=$((${#LAMBDAS[@]} * N_SEEDS))
((TASK_ID >= 0 && TASK_ID < N_TASKS)) || {
    echo "invalid task index: $TASK_ID (expected 0-$((N_TASKS - 1)))" >&2
    exit 2
}

LAMBDA_INDEX=$((TASK_ID / N_SEEDS))
SEED_INDEX=$((TASK_ID % N_SEEDS))
export BROKERAGE_ABM_RIDGE_LAMBDA_AGENT="${LAMBDAS[$LAMBDA_INDEX]}"
export BROKERAGE_ABM_RIDGE_LAMBDA_BROKER="${LAMBDAS[$LAMBDA_INDEX]}"
export BROKERAGE_ABM_RIDGE_PILOT_SEED="$((9001 + SEED_INDEX))"
export BROKERAGE_ABM_RIDGE_PILOT_T=500
export BROKERAGE_ABM_RIDGE_PILOT_DIR="$OUTPUT_DIR"

module purge
module load julia/1.11.3
export OMP_NUM_THREADS=1
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

cd "$REPO"
echo "task=$TASK_ID lambda=$BROKERAGE_ABM_RIDGE_LAMBDA_AGENT seed=$BROKERAGE_ABM_RIDGE_PILOT_SEED host=$(hostname)"
julia --compiled-modules=strict --pkgimages=existing --project \
    --threads="${SLURM_CPUS_PER_TASK:-2}" scripts/ridge/pilot.jl
