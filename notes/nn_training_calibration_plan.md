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
- Advance the two configurations with the highest three-seed median rank correlation for each learner. Add two seeds for these finalists, giving five seeds per finalist, then select the least costly setting whose five-seed median is within 0.01 of the best median. The current setting receives no special preference and is retained only if it qualifies under the same rule.
- Combine the selected agent and broker settings on all five seeds. Confirm that median agent and broker rank correlations each change by less than 0.01 between periods 101 to 150 and 151 to 200, and that using the selected settings together does not lower either learner's median rank correlation by more than 0.01 relative to its one-learner-at-a-time result.
- End calibration at period 200. If a selected setting lies on a tested boundary or the stability check fails, diagnose and extend only the affected comparison before accepting the calibration.
- Save the full configuration grid, seed list, code commit, package environment, seed-level time series, aggregate selection table, runtimes, and the selection decision.

## Deferred robustness checks

These checks are useful but do not block the time-constrained calibration. If they are later run, use the selected optimization settings at the baseline with five new robustness seeds, 9,100,001 through 9,100,005, through period 200. Compare the reference with:

1. Less training history: window 20 and cap 1000.
2. More training history: window 80 and cap 4000.
3. Lower network capacity: agent width `d` and broker width `4d`, compared with `2d` and `8d`.
4. Higher network capacity: agent width `4d` and broker width `16d`.
5. More frequent agent fitting: agents eligible every period rather than every other period. This is a conservative check because it gives agents at least as many fitting opportunities as the broker.

These are robustness checks, not a second calibration. The paired seed-level comparisons should report agent rank, broker rank, and their difference. If a combined window-and-cap check materially changes a conclusion, only then separate window from cap in a follow-up. Do not sweep Adam's standard moment coefficients unless training is unstable.

## Reporting

The manuscript should state in one or two sentences that principal and broker training settings were calibrated separately at the baseline using out-of-sample ranking performance and seeds excluded from the reported simulations. The replication code should preserve the candidate grid, manifests, selection rule, seed-level results, and final decision.
