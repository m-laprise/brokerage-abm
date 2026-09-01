# Paper reproducibility pipeline

Backs the paper's results section.

Generated paper outputs are intentionally absent until the pipeline is regenerated
from a complete reporting sweep. The former temporary single-model fixtures were
removed because they were not scientific results.

**No script hard-codes or handwrites any number or result.** Every emitted
value (statistics, counts, figure data, display scales) is derived from the raw data
or its config metadata at run time. Literal constants are limited to selection and
display conventions like window bounds or baseline parameter values.

The pipeline has two tiers, so figures and prose iterate locally.

Cluster tier (needs the sweep; set `BROKERAGE_ABM_SWEEP_DIR` to its root; run on a compute
node, `srun --partition=cpu --mem=8G`):

The reporting root must contain 20 seeds for every effective realization and
50 seeds for the baseline, for 1,990 runs in total. The data extractors check
this seed plan before writing outputs. Every reporting stage also requires all
source, paper, specification, and test files to match the current Git commit.
Generated files under `output/` may differ. Data-derived inputs retain the clean
analysis commit that produced them. Figure renderers and manuscript builders may
consume inputs from an earlier ancestor analysis commit, while recording the
current rendering or manuscript commit separately. This permits rapid iteration
on prose, captions, and presentation without relabeling or recomputing unchanged
scientific results.

1. `julia --project --threads=auto scripts/paper/stats.jl`
   Computes every statistic quoted in the section and writes `output/main/values.tex`
   (one `\pvDefine{key}{value}` per quoted number).
2. `julia --project --threads=auto scripts/paper/figdata.jl`
   Extracts the figure-input dataset to `output/main/figdata.jld2`: the
   seed-level baseline series and seed-level late-window values each figure
   consumes. Ensemble and condition means are retained for convenience, but
   uncertainty is reconstructed from the saved seed values.
   Run the same extractor against the paired-Ridge sweep with
   `BROKERAGE_ABM_FIGDATA_PATH=output/ridge/paired/figdata.jld2`; the exact
   command is given below.
3. `julia --project --threads=auto scripts/paper/audit_convergence.jl`
   Writes the reproducible seed-convergence audit to
   `output/main/convergence/`. The condition and outcome tables are retained as
   diagnostics and are not included in the paper. Relative precision follows
   the conventional interval-half-width-to-estimate definition; cells whose
   interval contains zero retain only absolute precision. Only the concise
   non-$R^2$ range in `values.tex` is consumed by the methods section.
4. `julia --project --threads=auto scripts/ridge/analyze_ablations.jl`
   Reads the four complete Ridge sweeps named by
   `BROKERAGE_ABM_RIDGE_PAIR_SWEEP_DIR`,
   `BROKERAGE_ABM_RIDGE_SIZE_MATCHED_SWEEP_DIR`,
   `BROKERAGE_ABM_RIDGE_SINGLE_PRINCIPAL_SWEEP_DIR`, and
   `BROKERAGE_ABM_RIDGE_ADDITIVE_SWEEP_DIR`. It writes the detailed ablation
   report inputs under `output/ridge/ablations/results/`, the small set of
   main-text values in `paper_values.tex`, and the seed-level main-figure data
   in `output/ridge/ablations/figdata.jld2`.

Local tier (no data access; works from a clone of this repository):

5. `julia --project --threads=auto scripts/paper/figures.jl`
   Renders the five results assets at print resolution, including the two-panel
   information-source figure. It reads only `output/main/figdata.jld2` and
   `output/ridge/ablations/figdata.jld2`; it also writes
   `output/main/figmeta.tex` (the display conventions quoted in captions:
   rolling window, measurement interval, axis start).
6. `julia --project --threads=auto scripts/paper/ridge_supplement.jl`
   Computes the four base Ridge values quoted in the section from the retained
   seed-level figure data.
7. `julia --project --threads=auto scripts/paper/build_section.jl`
   Flattens `paper/section_source.tex` (canonical prose; numbers appear only as
   `\pv{key}` references, titles and captions as `\pvtitle{name}` /
   `\pvcaption{name}` references resolved from `paper/captions.tex`) into
   `output/main/results_section.tex`, an `\input`-ready fragment with literal numbers
   and a provenance header. Fails on any undefined or unused value, title, or
   caption block, or missing figure, then compile-checks the fragment in a
   temporary directory. The results section contains five figures, numbered by
   their order of first citation rather than by their asset filenames. Needs
   only stock Julia and `pdflatex`.
8. `julia --project --threads=auto scripts/paper/build_manuscript.jl`
   Inlines the generated results section into the editable manuscript root and
   writes a submission-ready bundle under `output/manuscript/`. The bundle's
   `brokers_who_do_not_bridge_without_appendices.tex` contains the main
   manuscript with no results
   `\input`; `references.bib`, all referenced figures, a Biber `.bbl`, and a
   compile-checked main PDF are placed beside it. It also uses `pdfunite` to
   create `brokers_who_do_not_bridge_with_appendices.pdf`, containing the main manuscript,
   Appendix A (simulation pseudocode), Appendix B (model specifications), and
   the Supplementary Material, in that order. Run step 7 first whenever any
   results input or provenance changes.

Iterating on figure styling (colors, legends, fonts, layout, smoothing) means
editing `figures.jl` and rerunning steps 5--7 locally. The cluster tier reruns
only when the underlying numbers change: a new sweep, or a figure needing a
metric not yet extracted, which `figdata.jl` must then be taught to include.

Word export (optional). To produce an editable `.docx` of the section for
co-authors or track-changes, convert the generated fragment with pandoc:

```
pandoc output/main/results_section.tex -f latex -o results_section.docx \
  --resource-path=output/main
```

Pandoc reads the flattened fragment directly: math becomes native Word equations,
the `booktabs` tables become Word tables, and the figures in `output/main/figs/` are
embedded (hence `--resource-path=output/main`). Run steps 4--7 first. The `.docx`
reflects whatever is in `results_section.tex`. Needs only pandoc (>= 3),
no LibreOffice.
Document styling such as fonts can be set with a pandoc `--reference-doc`.

Hand-edited sources: `paper/section_source.tex` (prose) and `paper/captions.tex`
(figure titles and captions). Generated artifacts are under `output/main/`.

## Paired-Ridge figure supplement

The paired-Ridge report reproduces the full content of Main Figures 1--4 with
NN and paired Ridge in the same assets and shared axes. Direct NN-Ridge and
ablation contrasts use intervals on common-seed differences. To create its compact
Ridge input dataset on the cluster, point the general extractor at the paired
Ridge sweep and a separate output file:

```bash
BROKERAGE_ABM_SWEEP_DIR=<paired-ridge-sweep> \
BROKERAGE_ABM_FIGDATA_PATH=output/ridge/paired/figdata.jld2 \
  julia --project --threads=auto scripts/paper/figdata.jl
```

Then render the four comparative figures and rebuild the reports locally:

```bash
julia --project --threads=auto scripts/ridge/paired_figures.jl
julia --project --threads=auto scripts/ridge/build_reports.jl
```

The figure renderer validates that both datasets contain the same 98 effective
realizations and the 20-seed general, 50-seed baseline reporting plan. Outputs
are under `output/ridge/paired/figures/` and are embedded in the paired-Ridge
report.

## Supplementary Material

The results section measures the broker's structural advantage by betweenness
centrality. The Supplementary Material reproduces the same analyses with the
broker's two other saved ego-network measures, Burt's aggregate **constraint** and
**effective size** (`src/measures.jl`), in Figures S1-S3. The same two tiers apply.

Cluster tier (needs the sweep; set `BROKERAGE_ABM_SWEEP_DIR`; run on a compute node):

1. `julia --project --threads=auto scripts/paper/supp_figdata.jl`
   Extracts the supplement's figure-input dataset to `output/supplement/figdata.jld2`:
   the seed-level baseline constraint/effective-size series, the one-at-a-time
   and grid late values, and the per-realization late values S1-S3 consume. Standalone
   twin of `figdata.jl`; no hard-coded results.

Local tier (no data access; works from a clone):

2. `julia --project --threads=auto scripts/paper/supp_figures.jl`
   Renders Supplementary Figures S1--S3 from
   `output/supplement/figdata.jld2` only, and writes
   `output/supplement/figmeta.tex` (the
   display conventions quoted in the captions). Standalone twin of `figures.jl`.
3. `julia --project --threads=auto scripts/paper/build_supplement.jl`
   Compiles the standalone `paper/supplement.tex`. Caption values are resolved
   from generated `\pv` definitions, so no result is hand-written. The builder
   validates every `\pv` reference and figure path first. Needs only stock Julia
   and `pdflatex`.

The three figures each redo a main-text structural-advantage analysis for
constraint and effective size: **S1** covers the rho x delta grid; **S2** shows
the baseline time path and the relationship with access across regimes; **S3**
shows the ranking and output differences against each measure.

Hand-edited source: `paper/supplement.tex` (standalone document and captions).
Generated artifacts are under `output/supplement/`. LaTeX auxiliary files are
created in a temporary directory and discarded.
