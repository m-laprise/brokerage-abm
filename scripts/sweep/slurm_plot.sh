#!/bin/bash
# Dependent plotting array: one task per OAT cell or phase pair.
# Submit after the compute array with a dependency + array range + log paths:
#   sbatch --dependency=afterany:<compute_jobid> --array=0-25%24 \
#          --output=$LOGDIR/plot_%A_%a.out --error=$LOGDIR/plot_%A_%a.err \
#          slurm_plot.sh <repo_root> <sweep_dir>
# afterany (not afterok) so plotting proceeds even if a few compute tasks fail;
# each plot task tolerates missing shards and the report agent lists gaps.
# Args: $1 = repo root, $2 = sweep dir (exported as BROKERAGE_ABM_SWEEP_DIR).
#SBATCH --job-name=brokerage_abm_plot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00   # >1h so QOS maps to `short` (400 concurrent), not `test` (2)
set -euo pipefail

REPO="${1:?usage: slurm_plot.sh <repo_root> <sweep_dir>}"
export BROKERAGE_ABM_SWEEP_DIR="${2:?usage: slurm_plot.sh <repo_root> <sweep_dir>}"

module purge
module load julia/1.11.3
export OMP_NUM_THREADS=1
# Match slurm_setup.sh / slurm_sweep.sh so plot jobs share the multiversioned cache.
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

cd "$REPO"
echo "plot task=${SLURM_ARRAY_TASK_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} sweep=$BROKERAGE_ABM_SWEEP_DIR"

julia --project --threads="${SLURM_CPUS_PER_TASK:-2}" scripts/sweep/sweep_plot.jl
