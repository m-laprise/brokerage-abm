# Simulation sweep pipeline

This directory contains the reproducible SLURM pipeline for generating complete
model sweeps. The canonical design is defined in `sweep_config.jl`; `submit.sh`
orchestrates package setup, manifest creation, simulation, and aggregation.

## Requirements

- A SLURM cluster with a `cpu` partition and a module system
- Julia 1.11.3
- A durable data directory outside the repository
- A shared Julia depot accessible to every compute node

Set the required environment before running a stage:

```bash
export BROKERAGE_ABM_ACCOUNT=<slurm-account>
export BROKERAGE_ABM_DATA_ROOT=<durable-data-directory>
export JULIA_DEPOT_PATH=<shared-julia-depot>
```

Sweep outputs are stored under
`$BROKERAGE_ABM_DATA_ROOT/sweep/<tag>/`. The default tag contains the date and
source commit. A nondefault learning model, broker variant, or scope is appended.

## Staged workflow

Run each stage from the repository root and inspect it before continuing:

```bash
./scripts/sweep/submit.sh resolve
./scripts/sweep/submit.sh setup
./scripts/sweep/submit.sh manifest
./scripts/sweep/submit.sh smoke
./scripts/sweep/submit.sh compute
./scripts/sweep/submit.sh plot
```

The stages perform the following tasks:

1. `resolve` downloads the pinned packages on a networked login node. It does
   not precompile and fails if package resolution changes `Project.toml` or
   `Manifest.toml`.
2. `setup` submits one compute job that precompiles the project into the shared
   depot. Wait for `SETUP_OK` in its output.
3. `manifest` creates the run manifest and `counts.env` on a compute node.
   Inspect the effective-regime and run counts before proceeding.
4. `smoke` submits one array task. Inspect its output, error log, runtime, saved
   shard, and provenance.
5. `compute` submits the full simulation array.
6. `plot` submits aggregation jobs that produce each regime's `data.jld2` and
   diagnostic figures.

`./scripts/sweep/submit.sh status` reports the relevant queue state. The helper
scripts `status.sh` and `monitor.sh` provide more detailed progress checks.

## Sweep controls

The default learning model is the neural network. The main controls are:

| Variable | Purpose | Default |
|---|---|---|
| `BROKERAGE_ABM_LEARNING_MODEL` | `nn` or `ridge` | `nn` |
| `BROKERAGE_ABM_N_SEEDS` | Seeds for each regime | `20` |
| `BROKERAGE_ABM_BASELINE_N_SEEDS` | Seeds at the baseline | Same as other regimes |
| `BROKERAGE_ABM_SWEEP_SCOPE` | `full` or `rho_delta` | `full` |
| `BROKERAGE_ABM_RIDGE_LAMBDA_AGENT` | Principal Ridge penalty | `0.001` |
| `BROKERAGE_ABM_RIDGE_LAMBDA_BROKER` | Broker Ridge penalty | `0.001` |
| `BROKERAGE_ABM_RIDGE_BROKER_VARIANT` | Ridge broker specification | `pair` |
| `BROKERAGE_ABM_TAG` | Output tag | Date and commit based |
| `BROKERAGE_ABM_CPUS` | CPUs and Julia threads per simulation | `2` |
| `BROKERAGE_ABM_THROTTLE` | Maximum concurrent simulation tasks | `200` |
| `BROKERAGE_ABM_TIME` | Simulation-task wall time | `06:00:00` |

The reporting NN sweep uses 20 seeds for every regime and 50 at the baseline:

```bash
export BROKERAGE_ABM_N_SEEDS=20
export BROKERAGE_ABM_BASELINE_N_SEEDS=50
```

## Ridge experiments

The reported base Ridge sweep uses the same full design and seed plan as the
NN sweep:

```bash
export BROKERAGE_ABM_LEARNING_MODEL=ridge
export BROKERAGE_ABM_RIDGE_BROKER_VARIANT=pair
export BROKERAGE_ABM_RIDGE_LAMBDA_AGENT=0.003
export BROKERAGE_ABM_RIDGE_LAMBDA_BROKER=0.001
export BROKERAGE_ABM_N_SEEDS=20
export BROKERAGE_ABM_BASELINE_N_SEEDS=50
```

The three broker ablations use `size_matched`, `single_principal`, or
`additive`. Each is stored as a separate sweep and runs only the baseline and
the `rho` by `delta` design:

```bash
export BROKERAGE_ABM_SWEEP_SCOPE=rho_delta
export BROKERAGE_ABM_RIDGE_BROKER_VARIANT=<variant>
```

The Ridge penalties were selected with calibration seeds excluded from the
reporting ensembles. A joint baseline calibration selected 0.001 for the
broker. Holding that value fixed, the principal calibration selected 0.003 by
median late-period holdout rank correlation. The calibration jobs and
summaries are implemented in `scripts/ridge/slurm_calibration.sh`,
`slurm_agent_calibration.sh`, `summarize_calibration.jl`, and
`summarize_agent_calibration.jl`.

## Provenance and reuse

The manifest stage requires a clean committed worktree for reporting sweeps.
`BROKERAGE_ABM_ALLOW_DIRTY=1` is limited to nonreporting smoke tests.

Each sweep root contains:

- `manifest.json` and `manifest.jld2`, which record the requested grid,
  resolved regimes, seeds, and jobs;
- `manifest.sha256` and package-environment fingerprints;
- `counts.env`, which records the array sizes;
- per-task logs under `logs/`;
- simulation shards and aggregated regime outputs.

Existing shards are reused only when their code, Julia, package manifest,
sweep manifest, schema, parameters, and seeds match. Grid coordinates that
resolve to the same scientific regime reference the same canonical shards, so
they are neither simulated nor weighted more than once.

After a sweep completes, use the
[`scripts/paper/` reporting pipeline](../paper/README.md) to generate the paper
statistics, figure data, figures, and TeX outputs.
