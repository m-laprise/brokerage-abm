#!/bin/bash
# Instantiate and precompile the project once under the pinned Julia version.
# Run before the compute array so its tasks share the compiled depot cache.
# Argument: $1 = repository root.
#SBATCH --job-name=brokerage_abm_setup
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
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/BSTEWART/${USER}/julia_depot_brokerage}"
# Multiversioned precompile: one portable cache with optimized code paths for the
# cpu partition's mixed CPUs (Intel Cascade/Ice Lake = skylake-avx512; AMD Genoa
# = znver3-compatible), plus a generic baseline that loads on every node. Keep
# this value identical in slurm_sweep.sh and slurm_plot.sh so jobs share the cache.
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;skylake-avx512,clone_all;znver3,clone_all}"

cd "$REPO"
echo "host=$(hostname) julia=$(julia --version) cpus=${SLURM_CPUS_PER_TASK:-?} depot=$JULIA_DEPOT_PATH"

# The `submit.sh resolve` stage updates the registry, resolves dependencies, and
# downloads packages on a networked node. This stage instantiates the resolved
# environment and precompiles it for compute jobs.
julia --project --threads="${SLURM_CPUS_PER_TASK:-8}" -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    @info "precompiling package code"
    using BrokerageABM, CairoMakie, JLD2, DataFrames
    @info "setup complete" julia=VERSION nthreads=Threads.nthreads()
'

# Later stages use strict existing-cache mode. Verify that their package set can
# load without creating or refreshing compiled modules or native package images.
julia --compiled-modules=strict --pkgimages=existing --project \
    --threads="${SLURM_CPUS_PER_TASK:-8}" -e '
    using BrokerageABM, CairoMakie, JLD2, DataFrames
    @info "strict cache verification complete" depot=DEPOT_PATH[1]
'
echo "SETUP_OK"
