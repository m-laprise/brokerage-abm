#!/bin/bash
# Baseline calibration of the agent Ridge penalty while holding the broker
# penalty fixed at 0.001. Submit as a 30-task array, one task per penalty-seed
# combination. Reporting sweeps use a separate seed set.
#
# Required environment:
#   BROKERAGE_ABM_RIDGE_AGENT_CALIBRATION_DIR  durable output directory
# Optional environment:
#   BROKERAGE_ABM_REPO                        repository root
#   JULIA_DEPOT_PATH                          shared Julia depot
#
# Agent penalties:   0.1, 0.3, 0.5
# Broker penalty:    0.001
# Calibration seeds: 9001:9010
#SBATCH --job-name=brokerage_ridge_agent_calibration
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail

REPO="${BROKERAGE_ABM_REPO:-/projects/BSTEWART/mlaprise/brokerage-abm}"
OUTPUT_DIR="${BROKERAGE_ABM_RIDGE_AGENT_CALIBRATION_DIR:?set agent calibration output directory}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

AGENT_LAMBDAS=(0.1 0.3 0.5)
BROKER_LAMBDA=0.001
N_SEEDS=10
N_TASKS=$((${#AGENT_LAMBDAS[@]} * N_SEEDS))
((TASK_ID >= 0 && TASK_ID < N_TASKS)) || {
    echo "invalid task index: $TASK_ID (expected 0-$((N_TASKS - 1)))" >&2
    exit 2
}

LAMBDA_INDEX=$((TASK_ID / N_SEEDS))
SEED_INDEX=$((TASK_ID % N_SEEDS))
export BROKERAGE_ABM_RIDGE_LAMBDA_AGENT="${AGENT_LAMBDAS[$LAMBDA_INDEX]}"
export BROKERAGE_ABM_RIDGE_LAMBDA_BROKER="$BROKER_LAMBDA"
export BROKERAGE_ABM_RIDGE_PILOT_SEED="$((9001 + SEED_INDEX))"
export BROKERAGE_ABM_RIDGE_PILOT_T=500
export BROKERAGE_ABM_RIDGE_PILOT_DIR="$OUTPUT_DIR"

module purge
module load julia/1.11.3
export OMP_NUM_THREADS=1
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

cd "$REPO"
echo "task=$TASK_ID lambda_a=$BROKERAGE_ABM_RIDGE_LAMBDA_AGENT lambda_b=$BROKERAGE_ABM_RIDGE_LAMBDA_BROKER seed=$BROKERAGE_ABM_RIDGE_PILOT_SEED host=$(hostname)"
julia --compiled-modules=strict --pkgimages=existing --project \
    --threads="${SLURM_CPUS_PER_TASK:-2}" scripts/ridge/pilot.jl
