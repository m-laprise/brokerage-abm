# Broker interaction-learning: investigation summary

## Goal
Determine whether the broker can learn the **interaction term** of the matching
function — the regime-gated *gain*, `(1−ρ)·δ·sign(xᵢᵀBxⱼ)·xᵢᵀAxⱼ` — and whether the
prior failure was a *wrong* learning function or merely a *suboptimal* one.

## Method
Staged diagnostics (not the full ABM), separating three questions: (1) can the
function class represent the gain? (2) can training recover it from clean data?
(3) does the live broker get enough useful data? Headline metric: gain recovery
`βg` (1=learned, 0=ignored) plus the model's own `broker_holdout_rank` and
`rank_gap = broker − agent` (decision-relevant, seed-stable).

## What we found
- **Representation is fine.** A 1-hidden-layer ReLU net on the symmetric pair
  features can represent the gain (βg→0.93 on clean data). Not "wrong."
- **The optimizer was the cause.** Vanilla full-batch GD (the old spec) can't find
  the gain: the symmetric features are ill-conditioned, so a single global step
  size learns high-variance quality/core and starves the low-curvature gain.
  **Adam** recovers it.
- **Data coverage is a real but scale-dependent secondary effect.** Matched data
  is ~96% high-gain; this starves the gain at small N but **not** at the default
  N=1000 (enough absolute low-gain examples in the window).

## Mistakes made (and corrected)
1. **Weight decay** as a fix — hacked R² while destroying βg. Reverted.
2. **Validated at N=300** for speed and concluded the fix was net-negative / the
   data "fundamentally" starves the gain. Both were **small-N artifacts**; at the
   default N=1000 the fix clearly works. Redone at N=1000.
3. **Bogus "imbalance overfitting" story** for the cap sweep — the real confound
   was the adaptive step count being tied to the cap (higher cap → fewer steps →
   undertrained). Decoupled steps from cap and re-swept.
4. **Invoked "staleness"** to bound the window — the DGP is stationary, so older
   data is never stale; longer windows are strictly more informative.
5. **Assumed the broker's extra training steps unfairly advantaged it.** A small
   experiment showed agents *don't* benefit from more steps (they overfit their
   sparse data), so the broker's edge is genuinely **data-driven**, not a steps
   artifact — and the parity schedule was kept.

(Each was caught by user push-back; the lesson: validate at the real scale, watch
for confounds in one-at-a-time sweeps, and verify a premise before acting on it.)

## What we converged to
- **Adam** (lr 0.01) for agents and broker, replacing vanilla GD.
- **Period-based joint training window** (fair across learner throughput),
  subsampled uniformly across its span to a cap, full-budget even spacing.
- **Per-period step count decoupled** from window/cap (computed over full
  history) and made a tunable parameter.
- Three tunable params with defaults set to their cost/benefit knees:
  `train_window_periods=40`, `train_max_obs=2000`, `train_steps=100`.
- Knees (N=1000, 3 seeds): **cap** — more data helps monotonically, knee ≈2000,
  cost ∝ cap; **steps** — knee ≈100, cost ∝ steps; **wp** — compute-free (broker
  capped), more is better/stabler up to all-history, no staleness ceiling. At
  equal compute, **data beats steps**.

## Results (N=1000, T=200, seed 42)
| metric | OLD (GD, 500-obs) | NEW (Adam, 40/2000/100) |
|---|---|---|
| βg (gain) | 0.25 | **1.07** |
| broker_rank | 0.81 | **0.87** |
| rank_gap | 0.37 | **0.40** |
| broker_r2 | 0.37 | **0.67** |
| offline R² | 0.62 | **0.85** |
| low-gain bias | −0.19 | **+0.05** |

The broker now learns the full interaction (gain + core), generalizes better, and
its advantage over agents is larger and confirmed data-driven (supports H1.2).
Full test suite 698/698; regression baseline refreshed; spec §2d updated.

## Reproducing
All runs are deterministic (fixed `StableRNG` seeds; Julia 1.11). Use
`--threads=auto`; results are thread-robust within the regression test tolerances.
- **Staged diagnostics:** `stage1`/`stage1b`/`stage2`/`stage3_*.jl` (Q1–Q3 above).
- **Schedule sweep (cap/wp/steps):** `param_sweep.jl` (multi-seed; `SWEEP_QUICK=1`
  for a fast check). This regenerates the knee tables.
- **Regression-baseline pins:** `refresh_regression_baseline.jl`.
- **Parity verification (agents every period vs every other):** flip
  `agent_retrains_this_period` (src/step.jl) to `true` and compare
  `agent_holdout_rank` across seeds at N=500,T=60 — every-period was ≤ parity
  (data-poor agents overfit), so the parity schedule was kept.

## Caveats / future work
Single-DGP, one-at-a-time sweeps (cap×wp interaction not fully mapped). The
regime-imbalance effect is benign at default scale but would re-emerge in small
markets or very interaction-dominant regimes (low ρ); uncertainty-aware
exploration (the spec's Bayesian-last-layer extension) is the lever there.
