# Brokerage-ABM: Brokerage in Matching Markets

Replication code for an agent-based study of brokerage in matching markets.

## The model

A population of market participants, called principals, repeatedly forms
pairwise matches. Principals can search through their own networks or outsource
search to a broker. The broker
observes the matches it mediates and pools data across clients; each principal
learns only from its own matches. Match value can depend on general quality,
pair-specific complementarity, or both. Satisfaction affects later channel
choice, and turnover continually changes the network.

The default model uses neural networks for learning and prediction. Parallel
experiments replace them with Ridge regression to test whether the broker's
advantage depends on flexible prediction or on the amount and pair-level
content of its data. The simulations separately measure predictive accuracy,
realized match value, brokered access, outsourcing, and structural centrality.

## Results and documentation

- [Manuscript source](paper/manuscript.tex)
- [Results section source](paper/section_source.tex)
- [Scientific output index](output/README.md), covering the main figures,
  supplementary analyses, and Ridge experiments
- [Appendix A: Simulation pseudocode](output/appendices/simulation_pseudocode.pdf)
- [Appendix B: Model specifications](output/appendices/model_specifications.pdf)
- [Supplementary Material: Model diagnostics and structural robustness](output/supplement/supplement.pdf)

## Quick start

The project uses Julia 1.11.3. From the repository root:

```bash
julia --project --threads=auto -e 'using Pkg; Pkg.instantiate()'
julia --project --threads=auto -e 'using Pkg; Pkg.test()'
```

To run the baseline model and generate exploratory figures and saved simulation
data:

```bash
julia --project --threads=auto scripts/explore_model.jl --baseline --rerun
```

The outputs are written under `runs/exploration/`.

## Repository structure

| Path | Contents |
|---|---|
| `src/` | Model implementation: matching, learning, networks, and simulation loop |
| `test/` | Deterministic, invariant, regression, and performance tests |
| `scripts/explore_model.jl` | Local baseline and parameter exploration |
| `scripts/sweep/` | Reproducible SLURM sweep pipeline and [operating guide](scripts/sweep/README.md) |
| `scripts/paper/` | Statistical reporting, figures, and [publication builds](#reproducing-the-paper) |
| `scripts/ridge/` | Ridge experiments, ablations, calibration, and analysis |
| `paper/` | Hand-edited manuscript, supplement, and appendix sources |
| `notes/` | Research notes |
| `runs/` | Ignored raw simulation runs and local exploratory artifacts |
| `output/` | Generated results, figures, reports, and provenance records |

## Study design and reproducibility

The main reporting ensemble covers 80 scientifically distinct parameter
regimes. Each regime has 20 independent seeds, and the baseline has 30
additional seeds, for 1,630 runs over 500 periods. Some displayed grid
coordinates resolve to the same regime when a parameter is inactive. Such
duplicates reuse one simulation result and receive no additional analytical
weight.

Raw sweep shards are generated outside Git. The repository retains the
reviewable reports, figures, tables, provenance records, and selected seed-level
figure data needed for scientific inspection. Every simulation records its Git
commit, Julia version, parameter values, random seed, run-manifest hash, and
package-manifest fingerprint.

Reported numbers and figure values are computed from saved data rather than
entered in the manuscript by hand. Each analysis input retains its own source
commit; independent inputs may use different commits. The reporting pipeline
checks manifests, seeds, reporting windows, outcome definitions, and input
hashes. Undefined manuscript values fail the build; retained values not quoted
in the manuscript are allowed.

For a fresh full sweep, follow the
[`scripts/sweep/` guide](scripts/sweep/README.md).

## Reproducing the paper

Edit manuscript prose in `paper/manuscript.tex`, Results prose in
`paper/section_source.tex`, figure captions in `paper/captions.tex`, and the
supplement and appendices in `paper/supplement.tex` and `paper/appendices/`.
Results estimates use `\pv{key}` references resolved from generated values.
Generated files are written under `output/` and should not be edited manually.

PDF builds require a TeX installation with `pdflatex`, `latexmk`, and `biber`,
plus Poppler's `pdfunite`. Run commands from the repository root.

### Build from retained analysis inputs

After editing Results prose or captions, regenerate the Results fragment and
compile the manuscript:

```bash
julia --project --threads=auto scripts/paper/build_section.jl
julia --project --threads=auto scripts/paper/build_manuscript.jl
```

For edits confined to `paper/manuscript.tex`, only the second command is needed.
After changing appendix sources, run `scripts/paper/build_appendices.jl` first.
After changing the supplement, run `scripts/paper/build_section.jl` and
`scripts/paper/build_supplement.jl` before building the manuscript.
The manuscript builder writes PDFs with and without appendices under
`output/manuscript/`.

To refresh all publication figures, generated values, and PDFs from the retained
analysis inputs, without accessing raw sweeps or rerunning simulations:

```bash
julia --project --threads=auto scripts/paper/build_publication.jl
```

For individual figure changes, run the relevant renderer listed in
[`build_publication.jl`](scripts/paper/build_publication.jl), then rebuild the
affected documents. `scripts/paper/figures.jl` accepts `--assessment-not-access`
or `--information-sources` to render only that figure.
`scripts/paper/centrality_and_assessment.jl` renders Figure 4 and generates its
manuscript values from `output/main/centrality_data.jld2`.

### Regenerate analysis inputs from completed sweeps

Run these scripts with `julia --project --threads=auto`. On Della, use a compute
job and the environment described in the [sweep guide](scripts/sweep/README.md).
Analysis source dependencies must match their recorded commit; unrelated
manuscript and figure edits do not block analysis.

| Analysis inputs | Scripts | Required settings |
|---|---|---|
| Main estimates, figure data, access windows, and convergence summaries | `scripts/paper/stats.jl`, `figdata.jl`, `access_windows.jl`, `audit_convergence.jl` | `BROKERAGE_ABM_SWEEP_DIR` points to the main sweep |
| Ridge figure data and access windows | `scripts/paper/figdata.jl`, `access_windows.jl` | `BROKERAGE_ABM_SWEEP_DIR` points to the reference Ridge sweep; set `BROKERAGE_ABM_FIGDATA_PATH=output/ridge/paired/figure_data.jld2` and `BROKERAGE_ABM_ACCESS_WINDOWS_PATH=output/ridge/paired/access_windows.jld2` |
| NN-Ridge comparisons | `scripts/ridge/analyze_sweep.jl` | `BROKERAGE_ABM_NN_SWEEP_DIR` and `BROKERAGE_ABM_RIDGE_SWEEP_DIR` |
| Ridge learning experiments | `scripts/ridge/analyze_ablations.jl` | `BROKERAGE_ABM_RIDGE_PAIR_SWEEP_DIR`, `BROKERAGE_ABM_RIDGE_SIZE_MATCHED_SWEEP_DIR`, `BROKERAGE_ABM_RIDGE_SINGLE_PRINCIPAL_SWEEP_DIR`, and `BROKERAGE_ABM_RIDGE_ADDITIVE_SWEEP_DIR` |
| Assessment and access comparisons | `scripts/assessment_access/analyze.jl` | `BROKERAGE_ABM_ASSESSMENT_ACCESS_SWEEP_DIR` points to the base service sweep |
| Supplementary structural data | `scripts/paper/supp_figdata.jl` | `BROKERAGE_ABM_SWEEP_DIR` points to the main sweep |
| Matching-function diagnostics | `scripts/paper/dgp_figdata.jl` | No sweep input; generates data from model initialization without simulation periods |

After analyzing the base service sweep, extract its network trajectories and
merge the supplemental service sweep:

```bash
julia --project --threads=auto scripts/assessment_access/centrality_data.jl <base-sweep-root>
julia --project --threads=auto scripts/assessment_access/merge_supplement.jl \
    output/assessment_access/figure_data.jld2 <supplement-sweep-root> output/assessment_access
```

Once these inputs are available locally under `output/`, run
`scripts/paper/build_publication.jl` to render and assemble the paper.

## License

This project is distributed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE).
