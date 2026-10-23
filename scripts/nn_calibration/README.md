# Neural-network training calibration

This workflow selects separate Adam learning rates and update budgets for
principals and the broker at the baseline model condition. Calibration seeds
are disjoint from reporting seeds. Selection uses each learner's own holdout
rank correlation, not the difference between learners.

The candidate settings, windows, seed sets, and confirmation design are defined in
[`calibration_config.jl`](calibration_config.jl). Manifest creation checks the
calibration baseline against the reporting sweep.

The screen ranks candidates without selecting the scientific specification.
The principal setting was chosen for ranking performance. The broker setting
balances ranking performance, stability across seeds and periods, and runtime;
the larger update budget had an inconsistent gain across paired seeds. The
adopted learner settings are codified in
[`src/parameters.jl`](../../src/parameters.jl), and the confirmation design records
the settings evaluated together. These choices are not recomputed from rankings
when building the paper.

Each stage writes a human-readable task manifest, a native JLD2 manifest, its
SHA-256 hash, seed-level period tables, and aggregate summaries. Shards record
the code commit, Julia version, package-manifest hash, and calibration-manifest
hash. Simulation and summary stages verify the live checkout, Julia version,
package manifest, and clean-worktree state against the calibration manifest
before using or labeling any result.

## Della workflow

From a clean committed checkout, set:

```bash
export BROKERAGE_ABM_ACCOUNT=bstewart
export BROKERAGE_ABM_DATA_ROOT=/projects/BSTEWART/mlaprise/tb_sweeps
export JULIA_DEPOT_PATH=/scratch/gpfs/BSTEWART/mlaprise/julia_depot_brokerage
export BROKERAGE_ABM_NN_CALIBRATION_TAG=nn_$(git rev-parse --short HEAD)
```

Keep the same explicit tag for every stage. Manifests are immutable once
created.

Run the stages in order:

```bash
./scripts/nn_calibration/submit.sh resolve
./scripts/nn_calibration/submit.sh setup
./scripts/nn_calibration/submit.sh manifest screen
./scripts/nn_calibration/submit.sh smoke screen
./scripts/nn_calibration/submit.sh compute screen
./scripts/nn_calibration/submit.sh summarize screen
./scripts/nn_calibration/submit.sh manifest confirm
./scripts/nn_calibration/submit.sh smoke confirm
./scripts/nn_calibration/submit.sh compute confirm
./scripts/nn_calibration/submit.sh summarize confirm
```

Inspect the smoke log and artifact before each compute submission. The
confirmation summary reports whether the median rank correlation for each
learner changes by more than 0.01 between the two late-period windows.
