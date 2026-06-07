#!/bin/bash
# Setup job: resolve + instantiate + precompile the project ONCE under the
# pinned Julia, so the shared depot cache is warm and per-array-task startup is
# amortized (Enzyme/CairoMakie are expensive to precompile). Run before the
# compute array. Args: $1 = repo root.
#SBATCH --job-name=tb_setup
#SBATCH --account=bstewart
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -euo pipefail

REPO="${1:?usage: slurm_setup.sh <repo_root>}"

module purge
module load julia/1.11.3
export OMP_NUM_THREADS=1   # avoid OpenMP oversubscription in linked libs
# Multiversioned precompile: one portable cache with optimized code paths for the
# cpu partition's mixed CPUs (Intel Cascade/Ice Lake = skylake-avx512; AMD Genoa
# = znver3-compatible), plus a generic baseline so it loads on ANY node. Must be
# IDENTICAL in slurm_sweep.sh / slurm_plot.sh so all jobs share this cache slug.
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

cd "$REPO"
echo "host=$(hostname) julia=$(julia --version) cpus=${SLURM_CPUS_PER_TASK:-?}"

# Registry update + resolve + download are done first on a network node (see
# submit.sh `resolve` stage / orchestrator), so here we only instantiate (a fast
# no-op if already downloaded, no internet needed) and run the compute-heavy
# precompile.
julia --project --threads="${SLURM_CPUS_PER_TASK:-8}" -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    @info "precompiling package code"
    using TransientBrokerage, CairoMakie, JLD2, DataFrames
    @info "setup complete" julia=VERSION nthreads=Threads.nthreads()
'
echo "SETUP_OK"
