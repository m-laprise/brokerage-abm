# Ridge penalty calibration

The Ridge experiment uses separate penalties for principals and the broker. The
calibration uses baseline simulations and seeds 9001 through 9010, which are
excluded from reported results.

First, run the joint screen across the candidate penalties in
`slurm_calibration.sh`, then summarize it with `summarize_calibration.jl`. The
summary selects the common penalty that maximizes the median across seeds of
the late-period mean of principal and broker holdout rank correlations.

Second, hold the selected broker penalty fixed and run the principal screen in
`slurm_agent_calibration.sh`, then summarize it with
`summarize_agent_calibration.jl`. The summary selects the principal penalty
that maximizes the median across seeds of late-period principal holdout rank
correlation. Both summary scripts verify the candidate grid and seed set and
save seed-level results, aggregated results, the selection rule, and the source
commit.

On Della, from a clean checkout at the intended commit:

```bash
export BROKERAGE_ABM_REPO=/projects/BSTEWART/mlaprise/brokerage-abm
export JULIA_DEPOT_PATH=/scratch/gpfs/BSTEWART/mlaprise/julia_depot_brokerage

export BROKERAGE_ABM_RIDGE_CALIBRATION_DIR=/projects/BSTEWART/mlaprise/tb_sweeps/ridge_calibration/joint
sbatch --array=0-69 scripts/ridge/slurm_calibration.sh
# After the array completes:
julia --project --threads=auto scripts/ridge/summarize_calibration.jl

export BROKERAGE_ABM_RIDGE_AGENT_CALIBRATION_DIR=/projects/BSTEWART/mlaprise/tb_sweeps/ridge_calibration/agent
sbatch --array=0-69 scripts/ridge/slurm_agent_calibration.sh
# After the array completes:
julia --project --threads=auto scripts/ridge/summarize_agent_calibration.jl
```

The calibrated reporting values are `ridge_lambda_agent = 0.003` and
`ridge_lambda_broker = 0.001`.
