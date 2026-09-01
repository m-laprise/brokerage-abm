# BrokerageABM: Brokerage in Matching Markets

Replication code for an agent-based study of brokerage in matching markets.
The model examines when brokers create value by finding new counterparties,
when they create value by assessing possible matches, and how their information
and services shape their network position over time.

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

- [Generated results section](output/main/results_section.tex)
- [Scientific output index](output/README.md), covering the main figures,
  supplementary analyses, and Ridge experiments
- [Supplementary material](output/supplement/supplement.pdf)
- [Model specifications](model_specifications.pdf)
- [Simulation pseudocode](simulation_pseudocode.pdf)

The canonical editable sources are
[`model_specifications.tex`](model_specifications.tex),
[`simulation_pseudocode.tex`](simulation_pseudocode.tex), and the TeX files in
[`paper/`](paper/).

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

The outputs are written under `data/figures/exploration/` and
`data/sims/exploration/`.

## Repository structure

| Path | Contents |
|---|---|
| `src/` | Model implementation: matching, learning, networks, and simulation loop |
| `test/` | Deterministic, invariant, regression, and performance tests |
| `scripts/explore_model.jl` | Local baseline and parameter exploration |
| `scripts/sweep/` | Reproducible SLURM sweep pipeline and [operating guide](scripts/sweep/README.md) |
| `scripts/paper/` | Statistical reporting and figure pipeline, with a [reproduction guide](scripts/paper/README.md) |
| `scripts/ridge/` | Ridge experiments, ablations, calibration, and reports |
| `paper/` | Hand-edited TeX sources |
| `output/` | Generated results, figures, reports, and provenance records |

## Study design and reproducibility

The main reporting ensemble covers 98 scientifically distinct parameter
regimes. Each regime has 20 independent seeds, and the baseline has 30
additional seeds, for 1,990 runs over 500 periods. Some displayed grid
coordinates resolve to the same regime when a parameter is inactive. Such
duplicates reuse one simulation result and receive no additional analytical
weight.

Raw sweep shards are generated outside Git. The repository retains the
reviewable reports, figures, tables, provenance records, and selected seed-level
figure data needed for scientific inspection. Every simulation records its Git
commit, Julia version, parameter values, random seed, run-manifest hash, and
package-manifest fingerprint.

Reported numbers and figure values are computed from saved data rather than
entered in the manuscript by hand. The reporting pipeline checks its inputs,
requires consistent provenance, and fails on missing or unused manuscript
values.

For a fresh full sweep, follow the
[`scripts/sweep/` guide](scripts/sweep/README.md). To regenerate statistics,
figures, and TeX outputs from completed sweeps, follow the
[`scripts/paper/` guide](scripts/paper/README.md).

## License

This project is distributed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE).
