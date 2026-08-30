# ABM of Brokerage in Matching Markets

Simulation and replication code for an agent-based model of a brokered matching
market. Agents can search through their own networks or outsource search to a
broker that has broader access and learns from mediated matches.

This repository contains everything needed to build the study's results from
scratch: the model, the simulation pipeline, and the reporting pipeline. The
project follows two reproducibility requirements:

1. **Simulation data can be regenerated exactly** from the recorded code,
   parameters, seeds, and software environment.
2. **No result is hard-coded.** Every number, figure, and caption value in the
   results section is computed from the data each time the section is built.

## Data status

The current NN reporting outputs use 98 effective model realizations, with 20
seeds generally and 50 seeds at the baseline, for 1,990 runs. The generated
artifacts record the source sweep and manifest hash. Ridge outputs identify
their source sweeps in their provenance files.

## What is in the repository

| Path | Contents |
|---|---|
| `src/` | The model: matching, learning, networks, and the simulation loop |
| `test/` | Automated tests of the model (`julia --project --threads=auto -e 'using Pkg; Pkg.test()'`) |
| `scripts/sweep/` | Data generation: runs the model across all parameter settings on a compute cluster (staged via `submit.sh`) |
| `scripts/paper/` | Results pipeline: computes the statistics, renders figures, assembles results section |
| `scripts/diagnostics/` | Exploratory analyses, including the matching-function rank grid used by the figures |
| `paper/` | Hand-edited TeX sources for the results section, supplement, and Ridge reports |
| `output/` | Generated figures, values, tables, provenance files, and reports |
| `model_specifications.tex` | Model specifications |
| `simulation_pseudocode.tex` | Simulation-loop pseudocode |

```
src/ (model)  ->  scripts/sweep/ (simulation runs)  ->  saved dataset
                                                              |
output/  <-  scripts/paper/ and scripts/ridge/ (statistics, figures, TeX)
```

## Software environment

The code is written in [Julia](https://julialang.org), version 1.11.3.
`Project.toml` lists the packages used; `Manifest.toml` records the exact
version of every package and of all their dependencies, so the software stack
can be rebuilt identically:

```bash
julia --project --threads=auto -e 'using Pkg; Pkg.instantiate()'
```

Every generated run records the Git commit, Julia version, parameter values,
random seed, run-manifest hash, and a SHA-256 fingerprint of `Manifest.toml`.
This connects each result to the exact code and software environment that
produced it.

## Generating simulation data

Results will derive from a parameter sweep that runs the model repeatedly while
varying its parameters in a systematic design. Every run covers 500 periods. The
design has 161 OAT and phase-grid coordinates, which reference 98 effective
model realizations. Each realization is simulated under 20 independent seeds,
for 1,960 runs. Grid coordinates that represent the same realization reference
the same canonical result shards, including the $\rho=1$ coordinates where
$\delta$ is inactive by construction. Grid-reference multiplicity has no
scientific weight. No realization-seed combination is simulated twice. Randomness is
controlled with StableRNGs, and every run saves complete period-by-period
metrics rather than only summaries.

Each grid coordinate's saved file (`data.jld2`) holds the per-period metrics
tables (one per seed), its requested grid parameters, the exact realized
parameters, the seed list, the canonical result path it references, and its
provenance record: the git commit of the code that ran, the Julia version, the
`Manifest.toml` fingerprint, and the hash of the run manifest. The sweep root
holds `manifest.json` (the full grid design, effective-realization mapping, and
unique realization-seed jobs),
environment files, and per-task logs under `logs/`.

The sweep runs on a SLURM compute cluster. The flow is staged so each step can
be checked before the next:

```bash
./scripts/sweep/submit.sh resolve    # login node: download packages
./scripts/sweep/submit.sh setup      # compute: precompile the project once
./scripts/sweep/submit.sh manifest   # compute: write the run manifest + counts
./scripts/sweep/submit.sh smoke      # run one simulation task and inspect it
./scripts/sweep/submit.sh compute    # run the full simulation array
./scripts/sweep/submit.sh plot       # aggregation jobs that write each regime's data.jld2
```

The default manifest uses NN learning, 20 seeds per effective condition, and
the full grid. The paper reporting ensemble adds 30 baseline seeds by setting
`BROKERAGE_ABM_BASELINE_N_SEEDS=50`, for 1,990 runs in total. The
paired-Ridge experiment uses the same 98 effective
conditions, with 20 seeds generally and 50 at baseline:

```bash
export BROKERAGE_ABM_LEARNING_MODEL=ridge
export BROKERAGE_ABM_RIDGE_BROKER_VARIANT=pair
export BROKERAGE_ABM_RIDGE_LAMBDA_AGENT=0.003
export BROKERAGE_ABM_RIDGE_LAMBDA_BROKER=0.001
export BROKERAGE_ABM_N_SEEDS=20
export BROKERAGE_ABM_BASELINE_N_SEEDS=50
```

Each Ridge broker ablation is a separate sweep root. Set its broker variant to
`size_matched`, `single_principal`, or `additive`, and set
`BROKERAGE_ABM_SWEEP_SCOPE=rho_delta` to run only the baseline and the
`rho` by `delta` grid. The manifest records the estimator, penalty, variant,
scope, and both seed sets.

An initial joint baseline calibration used `scripts/ridge/slurm_calibration.sh`
to apply the same seven candidate penalties to agents and the broker. Ten fixed
calibration seeds, separate from the reporting seeds, selected a common penalty
of 0.001, which is retained for the broker. Holding the broker penalty at 0.001,
`scripts/ridge/slurm_agent_calibration.sh` compares agent penalties 0.001,
0.003, 0.01, 0.03, 0.1, 0.3, and 0.5 on those same seeds.
`scripts/ridge/summarize_agent_calibration.jl` selects 0.003 by the highest
median late-period agent holdout rank correlation. Each summarizer saves
run-level results, penalty-level summaries, the selected penalty, and the
source commit.

The manifest stage refuses a dirty Git worktree because a commit hash cannot
reconstruct uncommitted code. `BROKERAGE_ABM_ALLOW_DIRTY=1` is available only
for non-reporting smoke tests. Existing shards are reused only when their Git,
Julia, package-manifest, sweep-manifest, and schema provenance all match.

The cluster account and the data destination are taken from the environment
(`BROKERAGE_ABM_ACCOUNT`, `BROKERAGE_ABM_DATA_ROOT`). Simulation tasks request
2 CPUs by default, and Julia uses exactly that many threads. Set
`BROKERAGE_ABM_CPUS` to override the allocation for smoke tests or the full
array. Per-task wall times are in the sweep's `logs/`.

## Reproducing the results section

The pipeline is two-tiered (see `scripts/paper/README.md`): the data-dependent
steps run where the sweep is available and write small outputs (`values.tex`,
`figdata.jld2`), after which the figures and the assembled TeX rebuild locally
from a clone, with no access to the dataset. Editing prose, captions, or figure
styling therefore never requires the cluster:

```bash
# on the cluster, with BROKERAGE_ABM_SWEEP_DIR set:
julia --project --threads=auto scripts/paper/stats.jl
julia --project --threads=auto scripts/paper/figdata.jl
# locally, from a clone, no data access needed:
julia --project --threads=auto scripts/paper/figures.jl
julia --project --threads=auto scripts/paper/build_section.jl
```

The final step fails if any value referenced in the prose was not computed, if
any computed value goes unused, or if a required figure file is missing. It
then compiles the results section.

The paired-Ridge report includes comparative counterparts to all four main
figures. Each asset places NN and paired Ridge on shared axes. After extracting
the Ridge figure data from its sweep, render and compile the supplement with:

```bash
BROKERAGE_ABM_SWEEP_DIR=<paired-ridge-sweep> \
BROKERAGE_ABM_FIGDATA_PATH=output/ridge/paired/figdata.jld2 \
  julia --project --threads=auto scripts/paper/figdata.jl
julia --project --threads=auto scripts/ridge/paired_figures.jl
julia --project --threads=auto scripts/ridge/build_reports.jl
```

## Reviewable outputs

All generators write directly to the top-level `output/` directory. Nothing is
copied there from `paper/`. The output index is `output/README.md`.

The **Supplementary Material** (`output/supplement/supplement.pdf`) reproduces the structural-
advantage analyses with the broker's two other ego-network measures, Burt's
constraint and effective size. It is a separate, self-contained pipeline
(`supp_figdata.jl` -> `supp_figures.jl` -> `build_supplement.jl`), so it and the
results section regenerate independently; see `scripts/paper/README.md`.
