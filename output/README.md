# Scientific outputs and research notes

This directory is the single canonical location for generated reports,
figures, result tables, and figure-input datasets. Hand-edited TeX sources are
under `paper/`; generator and build scripts are under `scripts/`. Explicitly
labeled research notes may also be preserved here beside the outputs that
motivate them. They are not generated results or manuscript claims.

## Main results

The source sweep, manifest hash, condition count, run count, and seed counts are
stored in `main/figdata.jld2`. The generated header in `main/values.tex` also
records the source sweep and manifest hash. Figure-input datasets retain the
seed-level time series and late-window values used to compute Monte Carlo
intervals. Reporting artifacts also record the clean analysis-code commit, and
builders reject inputs produced by a different commit.

- [Results section](main/results_section.tex)
- [Figure 1: Baseline dynamics](main/figs/fig1_dynamics.png)
- [Figure 2: Sources of the broker's ranking advantage](main/figs/fig_information_sources.png)
- [Figure 3: Matching grid](main/figs/fig2_grid_lines.png)
- [Figure 4: Position and work](main/figs/fig3_position_work.png)
- [Figure 5: Structural advantage](main/figs/fig4_advantage.png)
- Monte Carlo convergence diagnostics: `main/convergence/condition_audit.tsv`
  and `main/convergence/outcome_summary.tsv`. These are reproducibility
  diagnostics and do not appear as paper figures or appendix analyses.

## Manuscript appendices and Supplementary Material

- [Main manuscript without appendices](manuscript/brokers_who_do_not_bridge_without_appendices.pdf)
- [Complete working paper with appendices](manuscript/brokers_who_do_not_bridge_with_appendices.pdf)
- [Appendix A: Simulation pseudocode](../simulation_pseudocode.pdf)
- [Appendix B: Model specifications](../model_specifications.pdf)
- [Supplementary Material: Model diagnostics and robustness analyses](supplement/supplement.pdf)
- [Figure S1: Principal types and their latent curve](supplement/figs/supp_S1_type_geometry.png)
- [Figure S2: Conditional match-value surfaces](supplement/figs/supp_S2_dgp_structure.png)
- [Figure S3: Effective dimensionality of match value](supplement/figs/supp_S3_dgp_dimension.png)
- [Figure S4: Alternative measures across the matching grid](supplement/figs/supp_S4_grid_lines.png)
- [Figure S5: Alternative structural position measures](supplement/figs/supp_S5_position.png)
- [Figure S6: Alternative measures and broker advantage](supplement/figs/supp_S6_advantage.png)

## Ridge experiments

- [Base Ridge report](ridge/paired/ridge_experiment.pdf)
- [Figure R1: Baseline dynamics](ridge/paired/figures/figR1_dynamics.png)
- [Figure R2: Matching grid](ridge/paired/figures/figR2_grid_lines.png)
- [Figure R3: Position and work](ridge/paired/figures/figR3_position_work.png)
- [Figure R4: Structural advantage](ridge/paired/figures/figR4_advantage.png)
- [Figure R5: Direct NN-Ridge comparison](ridge/paired/results/ridge_comparison.png)
- [Ridge ablation report](ridge/ablations/ridge_ablation_experiment.pdf)
- [Figure RA1: Ridge ablation contrasts](ridge/ablations/results/ridge_ablations.png)
- [Figure RA2: Ridge ablation grid](ridge/ablations/results/ridge_ablation_grid.png)
- [Research note: broker exploitation and principal learning](ridge/ablations/explore_exploit_research_note.md),
  a hand-authored hypothesis and proposed experimental design, not a generated
  result or manuscript claim

## Research notes

- [High-value follow-up experiments](research_notes/high_value_followup_experiments.md),
  a prioritized to-do memo covering principal learning by tenure, friction-free
  outsourcing, and a no-broker counterfactual

## Canonical generators

- Main results: `scripts/paper/stats.jl`, `figdata.jl`, `figures.jl`,
  `audit_convergence.jl`, `ridge_supplement.jl`, and `build_section.jl`
- Supplement: `scripts/paper/dgp_figdata.jl`, `supp_figdata.jl`,
  `supp_figures.jl`, and `build_supplement.jl`
- Ridge analyses: `scripts/ridge/analyze_sweep.jl` and
  `analyze_ablations.jl`
- Base Ridge figures: `scripts/ridge/paired_figures.jl`
- Ridge reports: `scripts/ridge/build_reports.jl`

The model specification and simulation pseudocode remain canonical at the
repository root. They are not copied here.
