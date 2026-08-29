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

1. `julia --project --threads=auto scripts/paper/stats.jl`
   Computes every statistic quoted in the section and writes `paper/values.tex`
   (one `\pvDefine{key}{value}` per quoted number).
2. `julia --project --threads=auto scripts/paper/figdata.jl`
   Extracts the small figure-input dataset to `paper/figdata.jld2` (~260 KB):
   the baseline ensemble series and the per-realization late means each figure
   consumes.

Local tier (no data access; works from a clone of this repository):

3. `julia --project --threads=auto scripts/paper/figures.jl`
   Renders the four results assets, `fig1_*.png` through `fig4_*.png`, at print
   resolution,
   numbered in order of first citation in the prose, reading only
   `paper/figdata.jld2`; also writes `paper/figmeta.tex` (the display
   conventions quoted in captions: rolling window, measurement interval, axis
   start).
4. `julia --project --threads=auto scripts/paper/build_section.jl`
   Flattens `paper/section_source.tex` (canonical prose; numbers appear only as
   `\pv{key}` references, titles and captions as `\pvtitle{name}` /
   `\pvcaption{name}` references resolved from `paper/captions.tex`) into
   `paper/results_section.tex`, an `\input`-ready fragment with literal numbers
   and a provenance header. Fails on any undefined or unused value, title, or
   caption block, or missing figure, then compile-checks the fragment
   (`paper/_build/wrapper.pdf`). Needs only stock Julia and `pdflatex`.

Iterating on figure styling (colors, legends, fonts, layout, smoothing) means
editing `figures.jl` and rerunning steps 3-4 locally. The cluster tier reruns
only when the underlying numbers change: a new sweep, or a figure needing a
metric not yet extracted, which `figdata.jl` must then be taught to include.

Word export (optional). To produce an editable `.docx` of the section — for
co-authors or track-changes — convert the generated fragment with pandoc:

```
pandoc paper/results_section.tex -f latex -o results_section.docx \
  --resource-path=paper
```

Pandoc reads the flattened fragment directly: math becomes native Word equations,
the `booktabs` tables become Word tables, and the figures in `paper/figs/` are
embedded (hence `--resource-path=paper`). Run step 4 first — the `.docx` reflects
whatever is in `results_section.tex`. Needs only pandoc (>= 3), no LibreOffice.
Document styling such as fonts can be set with a pandoc `--reference-doc`.

Hand-edited sources: `paper/section_source.tex` (prose) and `paper/captions.tex`
(figure titles and captions). `values.tex`, `figdata.jld2`, `figs/`,
`figmeta.tex`, and `results_section.tex` are generated and committed.

## Supplementary Material (alternative structural measures)

The results section measures the broker's structural advantage by betweenness
centrality. The Supplementary Material reproduces the same analyses with the
broker's two other saved ego-network measures, Burt's aggregate **constraint** and
**effective size** (`src/measures.jl`), in four figures (S1-S4). It is a **separate,
self-contained pipeline**: it shares no inputs or outputs with the steps above, so
the results section and the supplement are regenerated independently of each other.
The same two tiers apply.

Cluster tier (needs the sweep; set `BROKERAGE_ABM_SWEEP_DIR`; run on a compute node):

1. `julia --project --threads=auto scripts/paper/supp_figdata.jl`
   Extracts the supplement's figure-input dataset to `paper/supp_figdata.jld2`:
   the baseline per-period constraint/effective-size series, the one-at-a-time
   and grid late means, and the per-realization late means S1-S4 consume. Standalone
   twin of `figdata.jl`; no hard-coded results.

Local tier (no data access; works from a clone):

2. `julia --project --threads=auto scripts/paper/supp_figures.jl`
   Renders `paper/supp_figs/supp_S1_*.png` ... `supp_S4_*.png` from
   `paper/supp_figdata.jld2` only, and writes `paper/supp_figmeta.tex` (the
   display conventions quoted in the captions). Standalone twin of `figures.jl`.
3. `julia --project --threads=auto scripts/paper/build_supplement.jl`
   Compiles the standalone `paper/supplement.tex` (own preamble; captions quote
   display conventions only, via `\pv` keys resolved from `supp_figmeta.tex`, so
   no number is hand-written) to `paper/supplement.pdf`. Validates every `\pv`
   reference and figure path first. Needs only stock Julia and `pdflatex`.

The four figures each redo a main-text structural-advantage analysis for
constraint and effective size: **S1** covers the rho x delta grid; **S2** shows
the baseline time path and the relationship with access across regimes; **S3**
shows the prediction and output gaps against each measure; **S4** reproduces the
betweenness panel of the baseline-dynamics figure.

Hand-edited source: `paper/supplement.tex` (standalone document and captions).
`supp_figdata.jld2`, `supp_figs/`, `supp_figmeta.tex`, and `supplement.pdf` are
generated; the `pdflatex` aux/log artifacts are not committed.
