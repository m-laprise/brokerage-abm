# Neural-network training calibration

This workflow selects separate Adam learning rates and update budgets for
principals and the broker at the baseline model condition. Calibration seeds
are disjoint from reporting seeds. Selection uses each learner's own holdout
rank correlation, not the difference between learners.

The fixed candidate learning rates are 0.003, 0.01, and 0.03. Recurrent update
budgets are 50, 100, and 200 steps. Initial budgets are twice the corresponding
recurrent budget. Every run uses `N=1000`, `T=200`, and the production Adam
implementation. The baseline is `rho=0.5`, `eta=0.02`,
`reservation_frac=0.60`, `delta=0.50`, `k=6`, `roster_frac=0.20`, and
`n_strangers=10`. Manifest creation verifies that these values match the
reporting sweep baseline.

The screen evaluates 17 unique one-learner-at-a-time configurations on seeds
9,000,001 through 9,000,003. The summary ranks the candidates but does not make
the scientific choice automatically. I reviewed ranking performance over
periods 151--200, change from periods 101--150, paired seed results, and
runtime. The principal attained its highest median late-period rank correlation
with learning rate 0.003 and 50 recurrent steps. For the broker, learning rate
0.03 with 200 recurrent steps had the highest median, but the gain over 50
steps was inconsistent across paired seeds, performance was still changing
between the two late windows, and runtime was about twice as long. I therefore
selected learning rates 0.003 for principals and 0.03 for the broker, with 100
initial and 50 recurrent steps for both. The confirmation stage evaluates
these settings together on seeds 9,000,001 through 9,000,005.

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
