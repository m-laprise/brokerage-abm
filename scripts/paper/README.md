# Paper reproducibility pipeline

Backs the paper's results section.

**No script hard-codes or handwrites any number or result.** Every emitted
value (statistics, counts, figure data, display scales) is derived from the raw data
or its config metadata at run time. Literal constants are limited to selection and
display conventions like window bounds or baseline parameter values.

The pipeline has two tiers, so figures and prose iterate locally.

Cluster tier (needs the sweep; set `TB_SWEEP_DIR` to its root; run on a compute
node, `srun --partition=cpu --mem=8G`):

1. `julia --project scripts/paper/stats.jl`
   Computes every statistic quoted in the section and writes `paper/values.tex`
   (one `\pvDefine{key}{value}` per quoted number).
2. `julia --project scripts/paper/figdata.jl`
   Extracts the small figure-input dataset to `paper/figdata.jld2` (~260 KB):
   the baseline-pair ensemble series and the per-regime late means each figure
   consumes, plus the effective-rank grid (which itself comes from
   `scripts/diagnostics/dgp_rank_grid.jl`).

Local tier (no data access; works from a clone of this repository):

3. `julia --project scripts/paper/figures.jl`
   Renders `paper/figs/fig1_*.png` ... `fig6_*.png` at print resolution,
   numbered in order of first citation in the prose, reading only
   `paper/figdata.jld2`; also writes `paper/figmeta.tex` (the display
   conventions quoted in captions: rolling window, measurement interval, axis
   start).
4. `julia scripts/paper/build_section.jl`
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

Hand-edited sources: `paper/section_source.tex` (prose) and `paper/captions.tex`
(figure titles and captions). `values.tex`, `figdata.jld2`, `figs/`,
`figmeta.tex`, and `results_section.tex` are generated and committed.
