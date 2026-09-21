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
intervals. Each reporting input records its own analysis commit. Builders check
input identity and compatibility; independent analyses may use different commits.

- [Results section source](../paper/section_source.tex)
- [Generated Results section](main/results_section.tex)
- [Figure 1: Assessment, not access](main/figures/assessment_not_access.png)
- [Figure 2: Brokerage with access, assessment, or both](main/figures/assessment_access_3.png)
- [Figure 3: Sources of the broker's advantage](main/figures/information_sources_net_output_channels.png)
- [Figure 4: Matching grid](main/figures/matching_grid.png)
- [Figure 5: Position and work](main/figures/centrality_and_access.png)
- [Figure 6: Structural advantage](main/figures/structural_advantage.png)
- Monte Carlo convergence diagnostics: `main/convergence/condition_audit.tsv`
  and `main/convergence/outcome_summary.tsv`. These are reproducibility
  diagnostics and do not appear as paper figures or appendix analyses.

## Manuscript appendices and Supplementary Material

- [Manuscript source](../paper/manuscript.tex)
- [Manuscript](manuscript/brokers_who_do_not_bridge_without_appendices.pdf)
- [Manuscript with appendices and supplement](manuscript/brokers_who_do_not_bridge_with_appendices.pdf)
- [Editable Overleaf bundle](overleaf/brokerage_abm_overleaf_review.zip)
- [Appendix A: Simulation pseudocode](appendices/simulation_pseudocode.pdf)
- [Appendix B: Model specifications](appendices/model_specifications.pdf)
- [Supplementary Material](supplement/supplement.pdf)
- [Figure S1: Principal types and their latent curve](supplement/figures/type_geometry.png)
- [Figure S2: Conditional match-value surfaces](supplement/figures/match_value_surfaces.png)
- [Figure S3: Effective dimensionality of match value](supplement/figures/effective_dimensionality.png)
- [Figure S4: Assessment and access across matching composition](supplement/figures/complementarity_contributions.png)
- [Figure S5: Alternative measures across the matching grid](supplement/figures/alternative_measures_grid.png)
- [Figure S6: Alternative structural position measures](supplement/figures/alternative_measures_position.png)
- [Figure S7: Alternative measures and broker advantage](supplement/figures/alternative_measures_advantage.png)

## Ridge experiments

- [Figure R1: Baseline dynamics](ridge/paired/figures/baseline_dynamics.png)
- [Figure R2: Matching grid](ridge/paired/figures/matching_grid.png)
- [Figure R3: Position and work](ridge/paired/figures/centrality_and_access.png)
- [Figure R4: Structural advantage](ridge/paired/figures/structural_advantage.png)
- [Figure R5: Direct NN-Ridge comparison](ridge/paired/figures/ridge_comparison.png)
- [Figure RA1: Ridge ablation contrasts](ridge/ablations/figures/ridge_ablations.png)
- [Figure RA2: Ridge ablation grid](ridge/ablations/figures/ridge_ablation_grid.png)

## Broker assessment vs access experiment

- [Baseline dynamics](assessment_access/figures/baseline_dynamics.png)
- [Net-output comparisons](assessment_access/figures/net_output_contributions.png)
- Condition summaries: `assessment_access/condition_summary.tsv`
- Paired comparisons: `assessment_access/paired_contrasts.tsv`
- Seed-level results: `assessment_access/seed_level.tsv` and
  `assessment_access/figure_data.jld2`

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
- Broker assessment vs access experiment: `scripts/assessment_access/analyze.jl`

The model specification and simulation pseudocode sources are under
`paper/appendices/`; their standalone review PDFs are under `output/appendices/`.
