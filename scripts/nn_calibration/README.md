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
9,000,001 through 9,000,003. The two configurations with the highest median
period-151--200 rank correlation for each learner advance to confirmation on
seeds 9,000,004 and 9,000,005. The confirmed choice is the smallest recurrent
budget within 0.01 of the best five-seed median. At equal cost, learning rate
0.01 is preferred when eligible. The combined stage evaluates the selected
agent and broker settings together on all five seeds and compares periods
101--150 with 151--200.

Each stage writes a human-readable task manifest, a native JLD2 manifest, its
SHA-256 hash, seed-level period tables, aggregate TSV files, and the selection
record. Shards record the code commit, Julia version, package-manifest hash, and
calibration-manifest hash.

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
./scripts/nn_calibration/submit.sh manifest combined
./scripts/nn_calibration/submit.sh smoke combined
./scripts/nn_calibration/submit.sh compute combined
./scripts/nn_calibration/submit.sh summarize combined
```

Inspect the smoke log and artifact before each compute submission. If a
confirmed choice lies on a tested boundary or the combined summary reports a
decline greater than 0.01, stop and extend only the affected comparison.
