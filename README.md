# Transient Brokerage

Replication code for a dissertation chapter on transient brokerage that
contains an agent-based model of a brokered matching market, with and without
client capture by the broker.

This repository contains everything needed to build the chapter's results from
scratch: the model itself, the pipeline that generated the simulation data, and
the pipeline that turns the data into the results section. By construction:

1. **The data can be regenerated exactly** from this repository alone.
2. **No result is hard-coded.** Every number, figure, and caption value in the
   results section is computed from the data each time the section is built.

## What is in the repository

| Path | Contents |
|---|---|
| `src/` | The model: matching, learning, networks, capture, simulation loop |
| `test/` | Automated tests of the model (`julia --project -e 'using Pkg; Pkg.test()'`) |
| `scripts/sweep/` | Data generation: runs the model across all parameter settings on a compute cluster (staged via `submit.sh`) |
| `scripts/paper/` | Results pipeline: computes the statistics, renders figures, assembles results section |
| `scripts/diagnostics/` | Exploratory analyses, incl. the matching-function rank grid used by the figures |
| `paper/` | Results-section sources (prose, captions) and generated outputs (values, figures, TeX) |
| `model_specifications.tex` | Model specifications |
| `simulation_pseudocode.tex` | Simulation-loop pseudocode |

```
src/ (model)  ->  scripts/sweep/ (925 simulation runs)  ->  saved dataset
                                                                 |
paper/results_section.tex  <-  scripts/paper/ (numbers, figures, TeX)
```

## Software environment

The code is written in [Julia](https://julialang.org), version 1.11.3.
`Project.toml` lists the packages used; `Manifest.toml` records the exact
version of every package and of all their dependencies, so the software stack
can be rebuilt identically:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The link between this environment and the data is verifiable: every simulation 
run stored a fingerprint (SHA-256 hash) of the `Manifest.toml` it ran under, and 
the committed `Manifest.toml` matches that fingerprint byte for byte
(`ac0a668fd39a07cfd89bf7f88f2f9caf516a2addc81b06212a3b3591b9dfcab0`).

## The data

Results derive from one **parameter sweep**: the model run repeatedly while
its parameters are varied in a systematic design. The sweep
(`2026-06-07_f424438`) covers 185 parameter settings, called *regimes* (91
without capture, 94 with capture; each varies one parameter at a time or a pair
of parameters on a grid). Every regime is simulated five times with different
random seeds, for 925 runs of 200 simulated periods each, and every run saves
its complete period-by-period metrics, not just summaries, so later analyses
never need to re-simulate.

The data are exactly regenerable from this repository alone. The sweep design is 
fully specified in code (`scripts/sweep/sweep_config.jl`), randomness is 
controlled (StableRNGs, seeds 1 to 5, so the same seed always yields the same 
run), the software versions are pinned by the committed `Manifest.toml`, and the 
model code at the current commit is identical to the code that generated the 
data. Running the pipeline below reproduces the per-period metrics of every 
regime and seed exactly.

Each regime's saved file (`data.jld2`) holds the per-period metrics tables (one
per seed), the exact parameter values used, the seed list, and its own
provenance record: the git commit of the code that ran, the Julia version, the
`Manifest.toml` fingerprint, and the hash of the run manifest. The sweep root
holds `manifest.json` (the full design: every regime, parameter set, and seed),
environment files, and per-task logs under `logs/`.

## Regenerating the data

Rebuilding the dataset means rerunning all 925 simulations on a compute cluster 
running SLURM. The flow is staged so each step can be checked before the next:

```bash
./scripts/sweep/submit.sh resolve    # login node: download packages 
./scripts/sweep/submit.sh setup      # compute: precompile the project once
./scripts/sweep/submit.sh manifest   # compute: write the run manifest + counts
./scripts/sweep/submit.sh smoke      # run ONE simulation task, inspect its output
./scripts/sweep/submit.sh compute    # the full 925-task simulation array
./scripts/sweep/submit.sh plot       # aggregation jobs that write each regime's data.jld2
```

The cluster account and the data destination are taken from the environment
(`TB_ACCOUNT`, `TB_DATA_ROOT`). Per-task wall times are in the sweep's `logs/`.

## Reproducing the results section

The pipeline is two-tiered (see `scripts/paper/README.md`): the data-dependent
steps run on the cluster and commit their small outputs (`values.tex`,
`figdata.jld2`), after which the figures and the assembled TeX rebuild locally
from a clone, with no access to the dataset. Editing prose, captions, or figure
styling therefore never requires the cluster:

```bash
# on the cluster, with TB_SWEEP_DIR set:
julia --project scripts/diagnostics/dgp_rank_grid.jl   # once: effective-rank grid
julia --project scripts/paper/stats.jl                 # -> paper/values.tex (every quoted number)
julia --project scripts/paper/figdata.jl               # -> paper/figdata.jld2 (figure inputs)
# locally, from a clone, no data access needed:
julia --project scripts/paper/figures.jl               # -> paper/figs/*.png (print resolution)
julia scripts/paper/build_section.jl                   # -> paper/results_section.tex
```

The last step fails if any value referenced in the prose was not computed, if 
any computed value goes unused, or if a figure file is missing, and it finishes 
by compiling the section.

The **Supplementary Material** (`paper/supplement.pdf`) reproduces the structural-
advantage analyses with the broker's two other ego-network measures, Burt's
constraint and effective size, in four figures. It is a separate, self-contained
pipeline (`supp_figdata.jl` -> `supp_figures.jl` -> `build_supplement.jl`), so it
and the results section regenerate independently; see `scripts/paper/README.md`.
