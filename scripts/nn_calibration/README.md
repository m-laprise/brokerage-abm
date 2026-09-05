# Neural-network training calibration

This workflow selects separate Adam learning rates and update budgets for
principals and the broker at the baseline model condition. Calibration seeds
are disjoint from reporting seeds. Selection uses each learner's own holdout
rank correlation, not the difference between learners.

The fixed candidate learning rates are 0.003, 0.01, and 0.03. Recurrent update
budgets are 50, 100, and 200 steps. Initial budgets are twice the corresponding
recurrent budget. Every run uses `N=1000`, `T=200`, the production Adam
implementation, and otherwise baseline parameters.

The screen evaluates 17 unique one-learner-at-a-time configurations on seeds
9,000,001 through 9,000,003. Agents attain their highest median period-151--200
rank correlation with learning rate 0.003 and 50 recurrent steps. For the
broker, learning rate 0.03 with 200 recurrent steps has the highest median, but
its gain over 50 steps is inconsistent across paired seeds, its late-period
rank correlation is still rising, and it takes about twice as long. The
selected settings therefore use learning rates 0.003 for agents and 0.03 for
the broker, with 100 initial and 50 recurrent steps for both. The confirmation
stage evaluates these settings together on seeds 9,000,001 through 9,000,005
and compares periods 101--150 with 151--200.

The joint confirmation passed. Median period-151--200 rank correlation was
0.647 for agents and 0.913 for the broker. Median changes from periods
101--150 were -0.001 and 0.001, respectively, and median runtime was 351
seconds.

Each stage writes a human-readable task manifest, a native JLD2 manifest, its
SHA-256 hash, seed-level period tables, and aggregate summaries. Shards record
the code commit, Julia version, package-manifest hash, and calibration-manifest
hash.

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
