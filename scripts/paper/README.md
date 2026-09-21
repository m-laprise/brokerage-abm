# Paper reproducibility pipeline

Backs the paper's results section.

Generate paper outputs from a complete reporting sweep.

**No script hard-codes or handwrites any number or result.** Every emitted
value (statistics, counts, figure data, display scales) is derived from the raw data
or its config metadata at run time. Literal constants are limited to selection and
display conventions like window bounds or baseline parameter values.

The pipeline has two tiers, so figures and prose iterate locally.

Net output per principal is derived from saved metrics by `scripts/net_output_data.jl`
using `scripts/net_output.jl`. Reporting discards the legacy per-request net-output
columns. Simulation code and saved simulation files are unchanged.

Cluster tier (needs the sweep; set `BROKERAGE_ABM_SWEEP_DIR` to its root; run on a compute
node, `srun --partition=cpu --mem=8G`):

The reporting root must contain 20 seeds for every effective realization and
50 seeds for the baseline, for 1,630 runs in total. The data extractors check
this seed plan before writing outputs. Scientific analyses require their declared
dependencies to match the recorded Git commit. Unrelated
manuscript, plotting, and test edits do not block analysis. Each input retains its
own analysis commit; different commits or branches are allowed. Combined inputs
must agree on the relevant manifests, seeds, windows, and outcome definitions.
Manuscript assembly records input hashes and per-input commits. Presentation
revisions are marked uncommitted and retain their source hashes and patches.

1. `julia --project --threads=auto scripts/paper/stats.jl`
   Computes every statistic quoted in the section and writes `output/main/values.tex`
   (one `\pvDefine{key}{value}` per quoted number).
2. `julia --project --threads=auto scripts/paper/figdata.jl`
   Extracts the figure-input dataset to `output/main/figure_data.jld2`: the
   seed-level baseline series and seed-level late-window values each figure
   consumes. Ensemble and condition means are retained for convenience, but
   uncertainty is reconstructed from the saved seed values.
   Run the same extractor against the base Ridge sweep with
   `BROKERAGE_ABM_FIGDATA_PATH=output/ridge/paired/figure_data.jld2`; the exact
   command is given below.
3. `julia --project --threads=auto scripts/paper/audit_convergence.jl`
   Writes the reproducible seed-convergence audit to
   `output/main/convergence/`. The condition table also supplies principal-degree
   estimates and intervals for the centrality figure. Relative precision follows
   the conventional interval-half-width-to-estimate definition; cells whose
   interval contains zero retain only absolute precision. Only the concise
   non-$R^2$ range in `values.tex` is consumed by the methods section.
4. `julia --project --threads=auto scripts/ridge/analyze_ablations.jl`
   Reads the four complete Ridge sweeps named by
   `BROKERAGE_ABM_RIDGE_PAIR_SWEEP_DIR`,
   `BROKERAGE_ABM_RIDGE_SIZE_MATCHED_SWEEP_DIR`,
   `BROKERAGE_ABM_RIDGE_SINGLE_PRINCIPAL_SWEEP_DIR`, and
   `BROKERAGE_ABM_RIDGE_ADDITIVE_SWEEP_DIR`. It writes the detailed ablation
   report inputs under `output/ridge/ablations/analysis/`, the small set of
   main-text values in `paper_values.tex`, and seed-level rankings and channel
   net output in `output/ridge/ablations/figure_data.jld2`.

Local tier (uses retained data; no access to the raw sweep is needed):

5. `julia --project --threads=auto scripts/paper/figures.jl`
   Renders five results assets at print resolution, including the four-panel
   assessment and information-source figures. It reads
   `output/main/figure_data.jld2`, `output/ridge/{paired,ablations}/figure_data.jld2`,
   and the condition audit under
   `output/main/convergence/`; it also writes
   `output/main/figmeta.tex` (the display conventions quoted in captions:
   rolling window, measurement interval, axis start).
   Add `--assessment-not-access` to render only `assessment_not_access.png`.
   Add `--information-sources` to render only `information_sources_net_output_channels.png`.
   Raw early/late counts are retained in `output/main/access_windows.jld2` and
   `output/ridge/paired/access_windows.jld2`. Extract them from each completed sweep with
   `julia --project --threads=auto scripts/paper/access_windows.jl`, using
   `BROKERAGE_ABM_SWEEP_DIR` and `BROKERAGE_ABM_ACCESS_WINDOWS_PATH` for the input and output.
   Ridge late means also use `output/ridge/paired/analysis/condition_comparison.tsv`.
   Marginal densities weight regimes equally, share a pooled Silverman bandwidth
   between learners for each measure, and use Gaussian kernels reflected at the
   measure's bounds (0–1 for shares; −1–1 for rank correlations).
6. `julia --project --threads=auto scripts/paper/ridge_supplement.jl`
   Computes the Ridge and NN comparison values quoted in the section from retained
   figure data and the paired condition comparison, checking their provenance.
7. `julia --project --threads=auto scripts/paper/build_section.jl`
   Flattens `paper/section_source.tex` (canonical prose; numbers appear only as
   `\pv{key}` references, titles and captions as `\pvtitle{name}` /
   `\pvcaption{name}` references resolved from `paper/captions.tex`) into
   `output/main/results_section.tex`, an `\input`-ready fragment with literal numbers
   and a provenance header. Fails on undefined or duplicate values, undefined or
   unused titles or captions, or missing figures. Unquoted retained values are
   reported without requiring reanalysis. It compile-checks the fragment in a
   temporary directory. The results section contains six figures, numbered by
   their order of first citation rather than by their asset filenames. Needs
   only stock Julia and `pdflatex`.
8. `julia --project --threads=auto scripts/paper/build_appendices.jl`
   Compiles the two sources under `paper/appendices/` into standalone review PDFs
   under `output/appendices/`, without leaving LaTeX auxiliary files in the repository.
9. `julia --project --threads=auto scripts/paper/build_manuscript.jl`
   Compiles `paper/manuscript.tex` against the generated results and figures,
   writing only the two manuscript PDFs under `output/manuscript/`. It uses `pdfunite` to
   create `brokers_who_do_not_bridge_with_appendices.pdf`, containing the main manuscript,
   Appendix A (simulation pseudocode), Appendix B (model specifications), and
   the Supplementary Material, in that order. Run step 7 after changing Results
   prose or captions. Rebuild appendices or the supplement only when their own
   inputs change.

After the retained datasets, analysis outputs, and generated values exist,
running `julia --project --threads=auto scripts/paper/build_publication.jl`
performs the raw-data-free rendering and publication-build steps in dependency
order.

PDF builders preserve existing files when only build dates or document IDs differ.

Iterating on figure styling (colors, legends, fonts, layout, smoothing) means
editing `figures.jl` and rerunning steps 5--7 locally. The cluster tier reruns
only when the underlying numbers change: a new sweep, or a figure needing a
metric not yet extracted, which `figdata.jl` must then be taught to include.

Hand-edited sources: `paper/section_source.tex` (prose) and `paper/captions.tex`
(figure titles and captions). Generated artifacts are under `output/main/`.

## Assessment and access reporting

Inputs are in `output/assessment_access/`: `figure_data.jld2` holds seed-level
results and contrasts; `centrality_trajectories.jld2` holds baseline network
trajectories. TSV tables, `provenance.txt`, and diagnostic `figures/` accompany them.
Rendering requires no raw sweeps or simulations. Captions are in TeX.

Run these scripts with `julia --project --threads=auto scripts/assessment_access/<script>`:

| Script | Output under `output/` |
| --- | --- |
| `figure_2.jl` | `main/figures/assessment_access_3.png` |
| `supplement_figure.jl` | `supplement/figures/complementarity_contributions.png` |
| `paper_values.jl` | `assessment_access/paper_values.tex` |

Check structural measurements with
`julia --project --threads=auto test/test_assessment_access_reporting.jl`.
Check reconstructed net output, manuscript values, and current figures with
`julia --project --threads=auto test/test_net_output_reporting.jl`.
`build_publication.jl` renders the manuscript figure, the supplementary figure,
and the generated values.

## Base Ridge figure supplement

The base Ridge analysis compares NN and Ridge outcomes in four research figures
with shared axes. Direct NN-Ridge and
ablation contrasts use intervals on common-seed differences. To create its compact
Ridge input dataset on the cluster, point the general extractor at the paired
Ridge sweep and a separate output file:

```bash
BROKERAGE_ABM_SWEEP_DIR=<base-ridge-sweep> \
BROKERAGE_ABM_FIGDATA_PATH=output/ridge/paired/figure_data.jld2 \
  julia --project --threads=auto scripts/paper/figdata.jl
```

Then render the four comparative figures locally:

```bash
julia --project --threads=auto scripts/ridge/paired_figures.jl
```

The figure renderer validates that both datasets contain the same 98 effective
realizations and the 20-seed general, 50-seed baseline reporting plan. Outputs
are under `output/ridge/paired/figures/`. The Ridge research notes can be rebuilt with
`julia --project --threads=auto scripts/ridge/build_reports.jl`.

## Supplementary Material

Supplementary Figures S1--S3 visualize the matching-function data-generating
process. Figure S4 compares the value of assessment and access across matching
composition. Figures S5--S7 reproduce the main structural analyses with Burt's
aggregate **constraint** and **effective size** (`src/measures.jl`). The pipeline
retains seed-level inputs for all figures.

Cluster tier (run on a compute node):

1. `julia --project --threads=auto scripts/paper/dgp_figdata.jl`
   Generates `output/supplement/dgp_figure_data.jld2` from the production
   initialization code. Within each of 50 DGP seeds, 1,000 realized types and the
   matching-function objects are fixed across the effective current
   `rho`-by-`delta` grid. The retained data include a three-component projection
   of the realized type distribution for S1, centered conditional-value
   matrices for the five distinct conditions and 100 principals displayed in S2,
   normalized full-population singular spectra, and seed-level 90%-energy
   effective dimensions for S3. This stage does not run simulation periods or draw
   match noise.
2. `julia --project --threads=auto scripts/paper/supp_figdata.jl`
   Extracts the supplement's figure-input dataset to `output/supplement/structural_figure_data.jld2`:
   the seed-level baseline constraint/effective-size series, the one-at-a-time
   and grid late values, and the per-realization late values S5--S7 consume. This
   step needs the sweep and `BROKERAGE_ABM_SWEEP_DIR`.

Local tier (uses retained data):

3. `julia --project --threads=auto scripts/paper/supp_figures.jl`
   Renders Supplementary Figures S1--S3 and S5--S7 from the two retained datasets and writes
   `output/supplement/figmeta.tex`, which records each dataset's analysis commit
   and the display conventions quoted in the captions.
4. `julia --project --threads=auto scripts/assessment_access/supplement_figure.jl`
   Renders Figure S4 from the retained assessment-access contrasts, checking them
   against the saved seed-level values. The caption is in `paper/supplement.tex`.
   Generate its caption values with
   `julia --project --threads=auto scripts/assessment_access/paper_values.jl`.
5. `julia --project --threads=auto scripts/paper/build_supplement.jl`
   Compiles the standalone `paper/supplement.tex`. Caption values are resolved
   from generated `\pv` definitions, so no result is hand-written. The builder
   validates every `\pv` reference and figure path first. Needs only stock Julia
   and `pdflatex`.

For the structural checks, **S5** covers the rho x delta grid, **S6** shows the
baseline time path and the relationship with access across regimes, and **S7**
shows the ranking and output differences against each measure.

Main-text supplementary figure references use the labels in `paper/supplement.tex`.
The results builder resolves their numbers from the figure order.

Hand-edited source: `paper/supplement.tex` (standalone document and captions).
Generated artifacts are under `output/supplement/`. LaTeX auxiliary files are
created in a temporary directory and discarded.
