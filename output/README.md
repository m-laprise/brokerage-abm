# Scientific outputs

This directory is the canonical location for generated article outputs,
figures, result tables, and figure-input datasets. Hand-edited article sources
are under `paper/`; generator and build scripts are under `scripts/`; research
notes are under `notes/`.

## Main results

The source sweep, manifest hash, condition count, run count, and seed counts are
stored in `main/figure_data.jld2`. The generated header in `main/values.tex` also
records the source sweep and manifest hash. Figure-input datasets retain the
seed-level time series and late-window values used to compute Monte Carlo
intervals. Reporting artifacts also record the clean analysis-code commit, and
builders reject inputs produced by a different commit.

- [Results section](main/results_section.tex)
- [Figure 1: Baseline dynamics](main/figures/baseline_dynamics.png)
- [Figure 2: Sources of the broker's ranking advantage](main/figures/information_sources.png)
- [Figure 3: Matching grid](main/figures/matching_grid.png)
- [Figure 4: Position and work](main/figures/centrality_and_access.png)
- [Figure 5: Structural advantage](main/figures/structural_advantage.png)
- Monte Carlo convergence diagnostics: `main/convergence/condition_audit.tsv`
  and `main/convergence/outcome_summary.tsv`. These are reproducibility
  diagnostics and do not appear as paper figures or appendix analyses.

## Manuscript appendices and Supplementary Material

- [Main manuscript without appendices](manuscript/brokers_who_do_not_bridge_without_appendices.pdf)
- [Complete working paper with appendices](manuscript/brokers_who_do_not_bridge_with_appendices.pdf)
- [Appendix A: Simulation pseudocode](appendices/simulation_pseudocode.pdf)
- [Appendix B: Model specifications](appendices/model_specifications.pdf)
- [Supplementary Material: Model diagnostics and robustness analyses](supplement/supplement.pdf)
- [Figure S1: Principal types and their latent curve](supplement/figures/type_geometry.png)
- [Figure S2: Conditional match-value surfaces](supplement/figures/match_value_surfaces.png)
- [Figure S3: Effective dimensionality of match value](supplement/figures/effective_dimensionality.png)
- [Figure S4: Alternative measures across the matching grid](supplement/figures/alternative_measures_grid.png)
- [Figure S5: Alternative structural position measures](supplement/figures/alternative_measures_position.png)
- [Figure S6: Alternative measures and broker advantage](supplement/figures/alternative_measures_advantage.png)

## Ridge experiments

- [Figure R1: Baseline dynamics](ridge/paired/figures/baseline_dynamics.png)
- [Figure R2: Matching grid](ridge/paired/figures/matching_grid.png)
- [Figure R3: Position and work](ridge/paired/figures/centrality_and_access.png)
- [Figure R4: Structural advantage](ridge/paired/figures/structural_advantage.png)
- [Figure R5: Direct NN-Ridge comparison](ridge/paired/figures/ridge_comparison.png)
- [Figure RA1: Ridge ablation contrasts](ridge/ablations/figures/ridge_ablations.png)
- [Figure RA2: Ridge ablation grid](ridge/ablations/figures/ridge_ablation_grid.png)

## Research notes

- [High-value follow-up experiments](../notes/high_value_followup_experiments.md),
  a prioritized to-do memo covering principal learning by tenure, friction-free
  outsourcing, and a no-broker counterfactual
- [Broker exploitation and principal learning](../notes/explore_exploit_research_note.md),
  a hypothesis and proposed experimental design

## Canonical generators

- Main results: `scripts/paper/stats.jl`, `figdata.jl`, `figures.jl`,
  `audit_convergence.jl`, `ridge_supplement.jl`, and `build_section.jl`
- Supplement: `scripts/paper/dgp_figdata.jl`, `supp_figdata.jl`,
  `supp_figures.jl`, and `build_supplement.jl`
- Ridge analyses: `scripts/ridge/analyze_sweep.jl` and
  `analyze_ablations.jl`
- Base Ridge figures: `scripts/ridge/paired_figures.jl`

The model specification and simulation pseudocode sources are under
`paper/appendices/`; their standalone review PDFs are under `output/appendices/`.
