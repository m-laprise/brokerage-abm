# Della parameter sweep + automated report — agent instructions

Instructions for two agents: **(A)** run a SLURM parameter sweep on the Della
cluster, saving each run's data and plots in separate directories; **(B)** a
second agent that reads the whole sweep's saved data and compiles a LaTeX summary
report. Build on the existing exploration scripts — do **not** reinvent them.

> Items marked **[CONFIRM]** are proposals to confirm/adjust with the user before
> launching. Items marked **[DELLA]** are cluster specifics to fill in (unknown at
> authoring time).
>
> **Both agents: ask for clarification.** If anything in these instructions is
> ambiguous, underspecified, or conflicts with what you find (a missing `[DELLA]`
> value, an unclear metric or cell definition, an unexpected data shape), stop and
> ask the user rather than guessing or proceeding on an assumption.

---

## 0. Objective

Sweep the four main parameters **r (reservation), η (turnover), ρ (quality–
interaction mix), N (population)** and produce:
1. **Dynamics over time** (base-model + capture diagrams) for one-at-a-time (OAT)
   sweeps of each parameter, and
2. **Stable (steady-state) dynamics as a function of parameter *pairs*** (2D phase
   diagrams).

Then evaluate the **propositions and hypotheses in `model_specifications.md`**,
illustrate the theoretical framework, and surface any other interesting/surprising
trends. **5 seeds per cell. ≤5 values per swept parameter, others at baseline. No
full 4-D grid.**

---

## 1. Prerequisites

- **Julia 1.11.9** — the version under which `Manifest.toml` is resolved (project
  compat is `julia = "1.11"`). Use this exact version so results match the pinned
  Manifest and the project's regression baseline; do **not** use 1.12.x. On Della:
  **[DELLA]** how 1.11.9 is provided (`module load`, juliaup, or conda). Run
  everything with `--threads=auto`.
- One **setup job** (not per-array-task, to avoid races): `julia --project -e 'using Pkg; Pkg.instantiate()'`.
- **`r` is sweepable** via the `reservation_frac` parameter (default 0.60):
  `r = reservation_frac · q_cal`, validated to `≥ 0` (may exceed 1; the OAT sweep
  goes to 1.20). Set it through `default_params(reservation_frac=…)`.

---

## 2. Sweep specification

**Baseline** (`default_params`): ρ=0.50, η=0.02, N=1000, reservation_frac=0.60,
T=200, T_burn=30; all other parameters at their defaults. **Seeds = 1…5.**

### 2a. OAT time-series (build on `explore_base_model.jl` + `explore_capture.jl`)
For each parameter, hold the other three at baseline and run **≤5 values × 5 seeds**,
for **both** the base model and Model 1 (capture). Values (confirmed):

| param | values | note |
|---|---|---|
| ρ | 0, 0.3, 0.5, 0.7, 1.0 | full mix range incl. channel extremes (ρ=0 pure interaction, ρ=1 pure quality) |
| η | 0.01, 0.02, 0.03, 0.04 | baseline 0.02 |
| N | 500, 1000, 1500 | baseline 1000 |
| λ_r (`reservation_frac`) | 0.40, 0.60, 0.80, 1.00, 1.20 | baseline 0.60; sets r = λ_r·q_cal. Range chosen (N=1000 scan) to span the base broker-advantage sign flip (~0.8) and the capture on→off decline (capshare 0.99→0.17); 1.2 stays non-degenerate. Report should map realized r against each cell's match-value spread σ_f |

Output per config: seed-averaged dynamics panels with seed bands (reuse the
`explore_base_model.jl` 5×4 dynamics panel and the `explore_capture.jl` base-vs-
Model-1 panel), **plus the per-config network-statistics figure**
(`explore_base_model.jl::plot_network_stats`: broker betweenness over time, agent
degree mean/median/min/max over time, final agent-degree distribution).
**Extend that figure to also plot the broker's Burt `constraint` and
`effective_size`** (both are in the per-period df but absent from the current
figure) — they are the structural-advantage diagnostics for H1.3. These are the
"results over time."

### 2b. Phase diagrams (build on `explore_phase_diagram.jl`)
Grids over parameter **pairs** (axis levels per §2a: 5 for ρ/r, 4 for η, **3 for N**),
others at baseline, **5 seeds/cell**, steady-state (tail-averaged over the
post-burn-in window) → per-metric heatmaps.
Pairs (confirmed): all six pairwise combinations of {ρ, η, N, r} —
`ρ×η`, `ρ×N`, `ρ×r`, `η×r`, `η×N`, `r×N` (this is full *pairwise* coverage, still
not a 4-D grid). The existing `ρ×δ`, `ρ×s`, `ρ×snr` axes may be kept. **Run both the
base model and Model 1 (capture) for every pair**, so each pair yields a base and a
capture heatmap set; this doubles the phase cell count (factored into `NRUNS`).

**Rationale for the pairs.** ρ is the DGP's master knob — it sets the balance
between the *attributional* (quality) and *relational* (interaction) channels, and
H1.2 states the broker's relational advantage and its dominance *depend on DGP
structure*. So ρ paired with each other knob maps how that knob's effect changes
across the quality↔relational regime; the three non-ρ pairs map the structural /
market-environment side (H1.3, self-liquidation) at baseline ρ.
- **ρ×η** — turnover resets agent information and networks each period; tests
  whether the broker's *accumulated* relational advantage (H1.1b, self-reinforcing
  with history) survives churn, and whether churn bites hardest where the
  relational channel dominates (low ρ).
- **ρ×N** — N drives broker throughput (data volume) and network density; tests
  H1.1b's "advantage grows with volume and diversity" and the small-market limit
  where the interaction is hard to learn, conditional on the channel mix.
- **ρ×r** — r is the outside option, governing the extensive margin (which matches
  clear); maps the participation/tightness regime against the value-composition regime.
- **η×r** — pure market-environment pair: churn × participation threshold; isolates
  the economic liquidity/stability regime (thickness, outsourcing, capture).
- **η×N** — whether a large, high-throughput market lets the broker "outrun"
  turnover; turnover re-opens structural holes (regeneration of brokerage), so this
  maps sustained brokerage demand and broker network position vs (churn, size) (H1.3).
- **r×N** — market thickness × outside option: locates the frozen↔liquid region of
  the extensive margin and where broker *access* value concentrates — thin markets
  with a high reservation should need the broker most, thick low-r markets least (H1.3a).

Steady-state metrics per cell (tail mean ± seed band): `broker_holdout_rank`,
`agent_holdout_rank`, `rank_gap` (broker−agent), `broker_holdout_r2`,
`outsourcing_rate`, match quality (`q_self_mean`, `q_broker_standard_mean`),
network position (betweenness, Burt constraint, effective size), and capture
metrics under Model 1. Reuse whatever the existing scripts already compute.

### 2c. Capture-threshold (κ_max) sweep — Model 1 only
`capture_error_threshold` (κ_max) affects only Model 1, so it is swept on its own:
**capture model only, no pairs**, all other parameters at baseline, **3 values × 5
seeds**. Values (confirmed):

| param | values | note |
|---|---|---|
| κ_max (`capture_error_threshold`) | 0.40, 0.50, 0.65 | baseline 0.50; spans the contested→near-complete capture regimes |

**Rationale.** κ_max selects the capture *regime*, not just a rate
(`model_specifications.md` §12a). 0.40 = stringent (contested / low capture: the
broker captures only its highest-confidence subset and the readiness gate may be
intermittent), 0.50 = baseline (substantial but partial capture, near the regime
boundary), 0.65 = lenient (near-complete capture, standard brokerage residual).
Output per value: the `explore_capture.jl` base-vs-Model-1 dynamics panel with the
capture metrics (principal-mode share, captured positions, principal acceptance,
capture surplus, scaled MAE, readiness) plus rank gap and outsourcing over time, so
the report can show how the capture regime and its downstream lock-in respond to
the confidence bar. These are capture-only cells (`model = capture`).

---

## 3. Execution on Della (SLURM)

- **Step 0 — build the manifest (run once, before the array).** A generator
  (`scripts/sweep/sweep_manifest.jl`) expands the §2 value sets and pairs into the
  full ordered list of `(cell, seed)` jobs, assigns each a 0-based index, and writes
  `manifest.json`. `NRUNS` = the number of entries, used directly as
  `--array=0-(NRUNS-1)`. Every array task and the plot step look up their config by
  index in this file, so it is the single source of truth for the sweep's contents.
- **Parallelism model.** The sweep is embarrassingly parallel across runs: use a
  **SLURM job array**, not MPI or cross-node `Distributed`. The array *is* the
  multi-node parallelism — each task is an independent single-node process and
  SLURM packs tasks onto free cores across the allocated nodes. Request per-task
  resources and an array size, never nodes directly.
- **Granularity: one array task per (cell, seed)** — a single `run_simulation`,
  not 5 seeds per task. A *cell* is `(param, value, model∈{base,capture})` for OAT
  or `(pair, vi, vj)` for phase diagrams; the array index maps to `(cell, seed)`.
  Per-run cost is dominated by serial NN training (betweenness is not the
  bottleneck), so spread runs *wide* (one per task) rather than threads-deep; this
  gives the scheduler perfect load balancing and avoids intra-task parallelism.
- **Cores per task: `--cpus-per-task=2`** (`julia --threads=$SLURM_CPUS_PER_TASK`):
  a couple of threads for the parallel-Brandes betweenness; the rest of the run is
  serial, so more cores per task buy little. Raise to 4 only if betweenness at
  N=1500 proves slow.
- **No write contention:** each task writes its own `seed_<s>.jld2` shard in the
  cell directory (§4). Figures and the per-cell `data.jld2` are produced by a
  **dependent aggregation+plot job** (`--dependency=afterok:<arrayjobid>`), one
  task per cell, that loads the cell's seed shards and runs the sweep's plotting
  script (§5).
- **Sweep scripts (keep lean; reuse the simulation machinery, §5):**
  - `scripts/sweep/sweep_manifest.jl` — builds `manifest.json` (Step 0).
  - `scripts/sweep/sweep_run.jl` — maps `$SLURM_ARRAY_TASK_ID` → `(cell, seed)` from
    `manifest.json` → `default_params(...)` → `run_simulation` → writes the shard.
  - `scripts/sweep/sweep_plot.jl` — loads a cell's shards → its figures + aggregate
    `data.jld2`.
  - `scripts/sweep/slurm_sweep.sh` (compute array) and `slurm_plot.sh` (dependent
    per-cell plotting).
- **Concurrency throttle.** `--array=0-(NRUNS-1)%K`, with `K` set to your QOS /
  fair-share simultaneous-task allowance. `NRUNS = Σ_cells × 5 seeds`.
- **Caching / idempotency.** A task skips if its `seed_<s>.jld2` already exists
  (same config + schema version + git commit); `--rerun` forces recompute.
  Re-submitting fills only missing/failed indices (`--array=<failed ids>`), and
  cells are keyed by their own config, so a run shared across sweeps (the baseline
  cell across OAT axes, or a grid point shared between overlapping pairs) is
  computed once and reused.
- **[DELLA] SBATCH placeholders** to fill: `--account`, `--partition`, `--qos`,
  `--time`, `--mem`, `--cpus-per-task`, `--array=0-(NRUNS-1)%K`, and the Julia
  module/activation lines. Sizing measured on the dev machine (one `run_simulation`,
  `cpus-per-task=2`, betweenness on, T=200): worst case at N=1500 ≈ **8 min** wall
  and **~2.1 GB** peak RSS, so request e.g. `--time=00:30:00` and `--mem=4G` (margin
  over the measured single-task cost). Storage root: **[DELLA]** scratch vs
  `/projects` (data can be large; keep off the repo).
- Run the **setup/instantiate job once** (precompile the depot / build a sysimage
  so per-task Julia startup is amortized), then submit the compute array followed
  by the dependent plot job.
- **Determinism:** seeds 1–5; record git commit + `julia --version` + resolved
  Manifest hash in `manifest.json` and in each shard.

---

## 4. Storage layout (separate directory per run/cell)

```
<DATA_ROOT>/sweep/<tag>/                # <tag> = date + git short-sha
  manifest.json                         # full spec: params, values, seeds, pairs,
                                        #   git commit, Julia version, Manifest hash, date
  oat/<param>=<value>/
    base/    { seed_<s>.jld2, data.jld2, *.png }   # per-seed shards (array) + aggregate + panels
    capture/ { seed_<s>.jld2, data.jld2, *.png }   # Model 1 (vs base reference)
  phase/<pair>/
    cells/<vi>_<vj>/{base,capture}/ { seed_<s>.jld2, data.jld2 }  # shards + aggregate per model per grid point
    {base,capture}/summary.jld2         # derived steady-state tensors per model (for fast heatmaps)
    {base,capture}/<metric>.png         # one heatmap per metric, per model
  logs/<array_task_id>.{out,err}
```

The κ_max sweep (§2c) reuses the `oat/` pattern as `oat/kappa_max=<value>/capture/`;
it is capture-only, so it has no `base/` sibling.

**Save all raw data — raw is the source of truth.** Each `data.jld2` stores the
**complete per-seed, per-period metrics `DataFrame`s** — every column, every period
(the full `run_simulation`/`run_ensemble` output), *not* just summaries — plus the
cell config, seed list, git commit, Julia version, Manifest hash, and a schema
version. **Phase-diagram cells save the raw per-seed full `DataFrame` at every grid
point** (under `cells/<vi>_<vj>/`); the tail-averaged `summary.jld2` is a derived
convenience for heatmaps and can always be recomputed from the raw cells. This way
plotting and analysis (different metrics, summary windows, statistics) can be redone
later without re-simulating. Do not discard or down-sample the per-period data.
Plots: CairoMakie via `scripts/figure_style.jl` for consistent styling.

---

## 5. Reuse vs. adapt the existing infrastructure

**Reuse the simulation and metrics machinery directly** — do not reinvent it:
- `scripts/exploration_common.jl` → `default_params` / `run_simulation` / `run_ensemble`
  produce the per-seed, per-period metrics `DataFrame`s the sweep saves.
- `scripts/figure_style.jl` → shared CairoMakie styling.

**Treat the exploration/plotting scripts as blueprints, not drop-in code.** They are
written for interactive exploration — each *runs its own simulations and plots in one
pass* — so they cannot be called as-is on the sweep's saved shards. The sweep needs its
own plotting script (`scripts/sweep/sweep_plot.jl`) that loads the saved per-seed shards
and produces the sweep figures, modeled on these:
- `scripts/explore_base_model.jl` (its 5×4 dynamics panel and `plot_network_stats`) →
  blueprint for the OAT base panels and the network-stats figure; extend the latter to
  add the broker's Burt `constraint` and `effective_size` (§2a).
- `scripts/explore_capture.jl` → blueprint for the Model 1 base-vs-capture panels
  (§2a, §2c).
- `scripts/explore_phase_diagram.jl` → blueprint for the heatmaps. Its layout is fixed
  to ρ×Y and supports only `ρ×{s,η,δ,snr}`, so the sweep's phase plotting must
  **generalize to an arbitrary X×Y pair** to cover the non-ρ pairs (`η×r`, `η×N`,
  `r×N`), use the §2a axis values, and add `constraint`/`effective_size` heatmaps
  (it currently maps only `betweenness`; all three are in the per-period df).

---

## 6. Report agent (agent B)

**Inputs (read-only):** the entire `<DATA_ROOT>/sweep/<tag>/` tree (JLD2 + figures
+ manifest), `model_specifications.md` (the propositions/hypotheses and theoretical
framework), and `simulation_pseudocode.tex`.

**Task:** produce a LaTeX report that, *grounded only in the saved sweep data*:
1. **Evaluates every proposition/hypothesis** in `model_specifications.md` (e.g.
   §1 hypotheses 1.1a/1.1b, 1.2, 1.3a, the capture transitions 3a/3b — enumerate
   them from the spec). For each: state it, cite the relevant metric(s) and
   figure(s) from specific cells, give the numbers, and a **verdict**
   (supported / mixed / refuted / insufficient data).
2. **Illustrates the theoretical framework** — the attributional vs relational
   channels, self-liquidating structural advantage, the broker's data/diversity
   edge, and the capture transitions — using the saved OAT and phase figures.
3. **Surfaces other findings** — non-monotonicities, phase boundaries, surprising
   or counter-theoretical trends, and parameter regimes where behavior changes
   qualitatively.

**Evidence discipline (required).** Comment only on what the data shows. Do not
extrapolate beyond the swept cells, generalize to unswept regimes, or assert
mechanisms merely because they sound consistent with the theory. Every claim must
cite the specific cell(s), metric(s), and figure(s) it rests on. Any statement
that goes beyond the data (a conjectured mechanism, an extrapolated trend, a claim
about an unswept regime) must be explicitly labeled as a conjecture and paired with
the concrete probe that would confirm or refute it — the parameter values, seeds,
and diagnostics to run. Prefer "insufficient data" to a plausible-sounding guess.

**Output:** `<DATA_ROOT>/sweep/<tag>/report/sweep_report.tex` (compile to PDF if a
LaTeX engine is available; otherwise leave the `.tex` and note it). Self-contained
(preamble, `\includegraphics` of saved figures by path), with summary tables
(per-parameter trends; phase-boundary locations) and a hypotheses-verdict table.

**Autonomy / "done":** works only from saved data (no re-running simulations; if a
cell is missing/corrupt, list it and proceed). **Do not fabricate numbers** —
every quantitative claim must trace to a specific cell/metric in the saved data;
where data is insufficient for a hypothesis, say so explicitly. Done = a
compilable `.tex` covering all propositions + a "surprises" section + the tables
and figures, with traceable claims. Keep it concise; do not restate the spec.

---

## 7. Reproducibility & hygiene

- Fully deterministic: `StableRNG` seeds 1–5; Julia 1.11.9 + pinned Manifest;
  record git commit, Julia version, and Manifest hash in `manifest.json`.
- Always `--threads=auto`; results are thread-robust within test tolerances.
- Keep `<DATA_ROOT>` off the repo (large); the report references absolute/relative
  paths under the sweep tag. Verify `data/` is gitignored.

---

## Open items to confirm before launch
1. **[DELLA]** Confirm N=1500 is feasible within the cluster's per-task time/memory limits (see test #1); lower the top N if not.
2. **[DELLA]** Account/partition/qos/time/mem/cpus, Julia provisioning, and storage root.
