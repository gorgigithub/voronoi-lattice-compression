# Voronoi–lattice data compression — companion code

This repository contains the code for the paper
*"Voronoi clustering on rank-1 lattices for data compression in machine
learning: theory and an experimental comparison"*.

Running `publication_experiments.ipynb` reproduces every table of the
experimental section. Each table is rendered as a PNG image into
`article_assets/`, and each cell prints the matching `\includegraphics`
snippet for the LaTeX source.

## Contents

- **`publication_experiments.ipynb`** — the main notebook. Cell 1 sets the
  protocol and the rendering helpers; Cell 2 is the benchmark engine; Cells
  3–7 produce, in order, the compression-error tables (rate benchmark, the
  seven functional forms, the robustness study, the two dimension studies)
  and the per-function timing table.
- **`src/`** — the compression methods and supporting routines:
  - `compressors.py` — the Voronoi–lattice / Voronoi–net compressor and the
    supercompress baseline (Joseph & Mak).
  - `lattices.py` — component-by-component construction of rank-1 lattice
    rules.
  - `digital_nets.py` — construction of the Sobol' digital nets.
  - `qmc_averaging.py` — the Dick–Feischl inclusion–exclusion weights.
  - `ghw_lattice.py`, `ghw_fast.py`, `index_sets.py` — the
    Gnewuch–Harsha–Wnuk lattice compressor (step-hyperbolic-cross weights via
    the Dirichlet-kernel factorisation) and its index sets.
  - `test_functions.py` — the test functions (the three of the reference
    study and the Genz family).
- **`article_assets/`** — the rendered table images used by the manuscript.

## Requirements

Python 3 with NumPy, SciPy, scikit-learn and Matplotlib. See
`requirements.txt`.

## Reproducing the results

1. Install the dependencies: `pip install -r requirements.txt`.
2. Open `publication_experiments.ipynb` and run all cells.

By default the notebook runs with `N_REPEATS = 10` for quick exploration; set
`N_REPEATS = 100` in Cell 1 for the final run reported in the paper. Each
computing cell also exposes a local `R_LOCAL` override so a single table can
be regenerated on its own. The rank-1 lattice and its Voronoi partition are
deterministic, so the lattice-based results are reproducible across runs and
machines.

## Author

Abbass Gorgi.
