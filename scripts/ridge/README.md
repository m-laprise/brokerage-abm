# Ridge experiments

## Penalty calibration

The calibration evaluates separate penalties for principals and the broker at
the reporting baseline. The screen crosses
`{0.0003, 0.001, 0.003, 0.01, 0.03}` for the two learners, giving 25
configurations. Each configuration runs for 500 periods with 10 common seeds
(9001 through 9010), all excluded from reported simulations.

The screen reports each learner's late-period holdout rank correlation, change
from periods 301--400 to periods 401--500, RMSE, and bias. It preserves the
complete two-penalty surface and separate marginal summaries for principals and
the broker. It does not combine the two rank correlations or select penalties
automatically. Select each penalty by reviewing its learner's own ranking
performance, late-period stability, and level error across the other learner's
penalty. If the preferred value lies at an edge of the grid, extend that edge
before confirming it.

The selected pair is then confirmed together on five new seeds (9011 through
9015). The confirmation manifest requires the selected values and a concise
selection rationale. It records the screen-manifest hash so the decision and
confirmation remain linked to the screened surface.

All manifests and result shards record the code commit, Julia version, package
manifest hash, calibration-manifest hash, and schema version. Summarization
fails on missing runs, incomplete period windows, unexpected parameters,
provenance mismatches, or nonfinite diagnostics.

On Della, use a clean checkout at the intended commit:

```bash
export BROKERAGE_ABM_ACCOUNT=bstewart
export BROKERAGE_ABM_DATA_ROOT=/projects/BSTEWART/mlaprise/tb_sweeps
export JULIA_DEPOT_PATH=/scratch/gpfs/BSTEWART/mlaprise/julia_depot_brokerage
export BROKERAGE_ABM_RIDGE_CALIBRATION_TAG=ridge_$(git rev-parse --short HEAD)

./scripts/ridge/calibration/submit.sh resolve
./scripts/ridge/calibration/submit.sh setup
# Wait for SETUP_OK before continuing.
./scripts/ridge/calibration/submit.sh manifest screen
./scripts/ridge/calibration/submit.sh smoke screen
# Inspect the smoke log and shard before submitting the screen.
./scripts/ridge/calibration/submit.sh compute screen
./scripts/ridge/calibration/submit.sh summarize screen
```

After reviewing the screen summaries, create the confirmation manifest without
changing the checkout:

```bash
export BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_AGENT=<selected principal penalty>
export BROKERAGE_ABM_RIDGE_CONFIRM_LAMBDA_BROKER=<selected broker penalty>
export BROKERAGE_ABM_RIDGE_SELECTION_NOTE='<concise rationale>'

./scripts/ridge/calibration/submit.sh manifest confirm
./scripts/ridge/calibration/submit.sh smoke confirm
# Inspect the smoke log and shard before submitting confirmation.
./scripts/ridge/calibration/submit.sh compute confirm
./scripts/ridge/calibration/submit.sh summarize confirm
```

Outputs are stored under
`$BROKERAGE_ABM_DATA_ROOT/ridge_calibration/$BROKERAGE_ABM_RIDGE_CALIBRATION_TAG`.
The `summaries/` directory contains seed-level TSV files, configuration-level
TSV files, separate principal and broker rankings, and JLD2 copies with the
selection metadata.

## Reporting sweeps

After confirmation, pass the selected values explicitly as
`BROKERAGE_ABM_RIDGE_LAMBDA_AGENT` and
`BROKERAGE_ABM_RIDGE_LAMBDA_BROKER`. The base experiment uses the `pair` broker
variant over the complete reporting design. Each broker ablation uses the same
penalties and runs the baseline and rho by delta design with one of
`size_matched`, `single_principal`, or `additive`.
