# Ridge experiments

## Penalty calibration

The screen crosses separate principal and broker penalties at the reporting
baseline. Candidate values, windows, horizons, and seed sets are defined in
[`calibration/config.jl`](calibration/config.jl).

Review each learner's ranking performance, late-period stability, RMSE, and bias
across the other learner's penalty. The summaries preserve the full surface and
marginal comparisons; they do not select penalties automatically. If the preferred
value is at a grid boundary, extend that boundary before confirmation.

Confirmation evaluates the analyst-selected pair on separate seeds. Its manifest
records the selected values, rationale, and screen-manifest hash. The adopted
model defaults are codified in [`src/parameters.jl`](../../src/parameters.jl).

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
