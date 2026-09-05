# Neural-network training calibration

## Problem

The current neural-network settings are documented but not reproducibly calibrated for the current model. Agents and the broker both use learning rate 0.01, an initial budget of 200 Adam steps, and a minimum recurrent budget of 100 steps. The supporting diagnostics predate important changes to initialization, histories, ranking, and the experimental design. They also do not preserve a calibration manifest or results that reproduce the current values.

Agents and the broker should not be required to share optimization settings. Their feature spaces and training histories differ. The calibration must select settings using each learner's own out-of-sample ranking performance, not the broker-minus-agent rank difference.

## Minimal calibration design

- Calibrate only the baseline condition at the reporting population size, `N=1000`, and hold the selected settings constant in every experimental condition. This defines a neutral reference and avoids an arbitrary rule for weighting regimes. It does not claim that the settings are optimal in every regime.
- Reserve seeds 9,000,001 onward for calibration. Do not use them in reported ensembles.
- Allow separate agent and broker learning rates and recurrent step budgets.
- Compare learning rates 0.003, 0.01, and 0.03 and recurrent budgets 50, 100, and 200. Preserve the current two-to-one initialization-to-recurrent budget ratio, giving initialization budgets 100, 200, and 400.
- Hold network width, training window, observation cap, fitting cadence, Adam coefficients, and observation selection fixed during calibration.
- Screen the 17 unique one-learner-at-a-time combinations with three seeds through period 200. Select agents by their own holdout rank correlation and the broker by its own holdout rank correlation, averaged over periods 151--200. Record prediction error as a diagnostic and reject nonfinite or divergent fits.
- Prefer the least costly setting whose median rank correlation is within 0.01 of the best median. Retain the current setting if it satisfies this rule.
- Combine the selected agent and broker settings and add two seeds, for five seeds total. Confirm that the selection is unchanged and that median agent and broker rank correlations each change by less than 0.01 between periods 101 to 150 and 151 to 200.
- Do not run to period 500 if these stability checks pass. If they fail, extend only the current and selected configurations in 100-period increments until stable, with period 500 as a maximum rather than a required horizon.
- Save the full configuration grid, seed list, code commit, package environment, seed-level time series, aggregate selection table, runtimes, and the selection decision.

## Targeted robustness checks

After calibration, use the selected optimization settings at the baseline with five new robustness seeds, 9,100,001 through 9,100,005, through period 200. Compare the reference with:

1. Less training history: window 20 and cap 1000.
2. More training history: window 80 and cap 4000.
3. Lower network capacity: agent width `d` and broker width `4d`, compared with `2d` and `8d`.
4. Higher network capacity: agent width `4d` and broker width `16d`.
5. More frequent agent fitting: agents eligible every period rather than every other period. This is a conservative check because it gives agents at least as many fitting opportunities as the broker.

These are robustness checks, not a second calibration. The paired seed-level comparisons should report agent rank, broker rank, and their difference. If a combined window-and-cap check materially changes a conclusion, only then separate window from cap in a follow-up. Do not sweep Adam's standard moment coefficients unless training is unstable.

## Obsolete diagnostics

Replace and then delete `scripts/diagnostics/param_sweep.jl`. It uses `T=80`, seeds 42 to 44, a one-at-a-time schedule grid, and the broker-minus-agent rank difference. It prints summaries without preserving results or provenance and does not calibrate agent and broker learning rates or budgets separately.

The following standalone debugging scripts do not calibrate the live current model and should be removed after confirming that no retained analysis depends on them:

- `scripts/diagnostics/stage1_representation.jl`
- `scripts/diagnostics/stage1b_representation_ceiling.jl`
- `scripts/diagnostics/stage2_training.jl`
- `scripts/diagnostics/stage3_endogenous_data.jl`
- `scripts/diagnostics/broker_learning_common.jl`, which is used only by the preceding scripts and `param_sweep.jl`

In particular, the learning-rate comparison in `stage2_training.jl` defaults to vanilla gradient descent rather than production Adam. The fixed-data and representation scripts remain useful as historical debugging records, but they cannot justify the production hyperparameters. `scripts/diagnostics/broker_learning_investigation.md` should remain unchanged as a historical memo. The Ridge calibration scripts are separate, current experiments and are not deletion targets.
