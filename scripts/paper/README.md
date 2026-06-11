# Paper reproducibility pipeline

Backs the paper's results section. Reads ONLY the saved sweep
(`/projects/BSTEWART/mlaprise/tb_sweeps/sweep/2026-06-07_f424438`) and the
initialization-only DGP rank grid (`scripts/diagnostics/_results/dgp_rank_grid.jld2`,
produced by `scripts/diagnostics/dgp_rank_grid.jl`). No script here simulates.

**Rule: no script may hard-code or handwrite any number or result.** Every emitted
value (statistics, counts, figure data, display scales) is derived from the raw data
or its config metadata at run time. Literal constants are limited to selection and
display conventions: window bounds (late = [181,200], early = [50,70]), baseline
parameter values, rounding formats.

Run order (1 and 2 are independent; both feed 3):

1. `julia --project scripts/paper/stats.jl`
   Computes every statistic quoted in the section and writes `paper/values.tex`
   (one `\pvDefine{key}{value}` per quoted number).
2. `julia --project scripts/paper/figures.jl`
   Renders `paper/figs/fig1_*.png` ... `fig6_*.png` at print resolution, numbered
   in order of first citation in the prose.
3. `julia scripts/paper/build_section.jl`
   Flattens `paper/section_source.tex` (canonical prose; numbers appear only as
   `\pv{key}` references) into `paper/results_section.tex`, an `\input`-ready
   fragment with literal numbers and a provenance header. Fails on any undefined
   reference, unused definition, or missing figure, then compile-checks the
   fragment (`paper/_build/wrapper.pdf`).

Edit prose in `paper/section_source.tex` only; `results_section.tex` and
`values.tex` are generated. On the cluster, run steps 1-2 on a compute node
(`srun --account=bstewart --partition=cpu --mem=8G` for stats, `--mem=24G` for
figures); step 3 is light.
