# Paper reproducibility pipeline

Backs the paper's results section. Reads only the saved sweep (root taken from
`TB_SWEEP_DIR`) and the
initialization-only DGP rank grid (`scripts/diagnostics/_results/dgp_rank_grid.jld2`,
produced by `scripts/diagnostics/dgp_rank_grid.jl`).

**No script hard-codes or handwrites any number or result.** Every emitted
value (statistics, counts, figure data, display scales) is derived from the raw data
or its config metadata at run time. Literal constants are limited to selection and
display conventions like window bounds or baseline parameter values.

Run order (1 and 2 are independent; both feed 3):

1. `julia --project scripts/paper/stats.jl`
   Computes every statistic quoted in the section and writes `paper/values.tex`
   (one `\pvDefine{key}{value}` per quoted number).
2. `julia --project scripts/paper/figures.jl`
   Renders `paper/figs/fig1_*.png` ... `fig6_*.png` at print resolution, numbered
   in order of first citation in the prose, and writes `paper/figmeta.tex` (the
   display conventions quoted in captions: rolling window, measurement interval,
   axis start).
3. `julia scripts/paper/build_section.jl`
   Flattens `paper/section_source.tex` (canonical prose; numbers appear only as
   `\pv{key}` references, captions as `\pvcaption{name}` references resolved from
   `paper/captions.tex`) into `paper/results_section.tex`, an `\input`-ready
   fragment with literal numbers and a provenance header. Fails on any undefined
   or unused value, undefined or unused caption block, or missing figure, then
   compile-checks the fragment (`paper/_build/wrapper.pdf`).

Hand-edited sources: `paper/section_source.tex` (prose) and `paper/captions.tex`
(figure titles and captions). `results_section.tex`, `values.tex`, and `figmeta.tex` are
generated. On the cluster, run steps 1-2 on a compute node
(`srun --partition=cpu --mem=8G` for stats, `--mem=24G` for
figures); step 3 is light.
