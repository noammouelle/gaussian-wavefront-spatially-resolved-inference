# Kinematic / beta-recovery reproduction branch

This branch is a minimal, self-contained slice of the
`gaussian-wavefront-spatially-resolved-inference` repo, reduced to exactly
what's needed to reproduce the results in
`paper/kinematic_estimation_comparison.pdf`: per-shot kinematic parameter
(theta) recovery and signal parameter (beta = (A_s, A_c)) recovery, compared
across four methods -- **Null** (prior mean), **MAP-moments** (Kalman-gain
4-moment estimator), **Pixel-likelihood** ("best": phi-marginalized
pixel-resolved likelihood, bins=32, tight detector range), and **Oracle**
(true theta) -- on two datasets (1e6 atoms/shot and 1e8 atoms/shot).

It intentionally does **not** include the rest of the parent repo (other
experiments, the full notes history, unrelated scripts). The goal is a clean
base to fork for trying different hyperparameters (bin count, tight-range
half-width, prior, N_RUNS/N_SHOTS, ...) for a future paper, without carrying
the full repo's history and clutter.

## What's here

```
python-scripts/   core inference library (theta/beta estimators, likelihoods)
helpers/           shared utilities (dataset I/O, PSMAP-adjacent helpers)
analysis/          the 4 scripts that generate + plot the results
notebooks/          reproduce_results.ipynb -- numbers + figures in one place
results/            already-computed results (JSON), N_RUNS=10, N_SHOTS=50
figures/            already-generated PNGs (same figures as in the paper)
paper/              kinematic_estimation_comparison.tex/.pdf (the writeup)
download_data.sh    fetches the two datasets + PSMAP files from Drive backup
requirements.txt    pinned Python dependencies
```

`data/` and `output-files/` (the raw simulated images and PSMAP surrogate
files) are **not** committed to git -- they're ~17 GB total. Fetch them with
`download_data.sh` (see below). You only need them if you want to
regenerate `results/*.json` from scratch (e.g. with different
hyperparameters); if you just want the existing numbers/plots, `results/`
and `figures/` already have everything.

## 1. Environment setup

Python 3.13.9 was used to produce these results (should work with any
recent 3.11+). Install pinned dependencies:

```bash
pip install -r requirements.txt
```

This includes `cupy-cuda13x` for GPU acceleration. **A CUDA-capable NVIDIA
GPU is strongly recommended** -- `map_inference.py` and
`pixel_acs_grad*.py` use `cupy` when available and fall back to plain numpy
otherwise, but the pixel-likelihood ("best") method is orders of magnitude
slower on CPU. If your CUDA version isn't 13.x, install the matching
`cupy-cudaXXx` wheel instead (see https://docs.cupy.dev/en/stable/install.html).
Everything still runs on the numbers already in `results/` without a GPU --
you only need the GPU to regenerate results from raw data.

### aispy dependency (required)

The core library imports `aispy.psmap` (`load_psmap`, `PSMAPSurrogate`) to
read and interpolate the point-spread-map surrogate files. This is **not**
a pip package -- it's a sibling repo that must be installed separately:

```bash
git clone git@github.com:noammouelle/aispy.git ~/local/aispy
cd ~/local/aispy
git checkout c3a39e35d46f9a60984170aa4eacbb867029eae8   # branch: trajectory-plots
pip install -e .
```

(`pip install -e .` works because `aispy`'s `setup.py` declares the
`aispy` package at the repo root; after this, `import aispy.psmap` just
works from anywhere.) If you'd rather not `pip install`, the library code
also has a fallback that inserts `<repo-root>/../../local/aispy` onto
`sys.path` -- but the explicit `pip install -e` above is more robust to
where you clone this branch.

### aispp / ais++ dependency (NOT required to reproduce these results)

`aispp` (branch `confocal-mirror-type`, commit
`fdf924dabef3af5d5f0abc9576730d37b80d7450`) is the underlying wavefront/
photon-detection simulator that was used to generate the raw datasets in
`data/` and the PSMAP surrogate files in `output-files/`. **None of the
code in this branch imports aispp** -- the analysis scripts only consume
already-generated images (`data_IMG.h5`) and PSMAP files (`.h5`), both of
which you fetch via `download_data.sh` rather than regenerate. aispp is
documented here purely for provenance / in case you want to generate *new*
datasets (e.g. different atom counts, geometries) from scratch:

```bash
git clone git@github.com:noammouelle/aispp.git ~/local/aispp
cd ~/local/aispp
git checkout fdf924dabef3af5d5f0abc9576730d37b80d7450   # branch: confocal-mirror-type
# see aispp's own README for build instructions (it's a C++/Python simulator)
```

Regenerating datasets/PSMAPs with aispp from scratch is out of scope for
this branch's scripts -- they assume `data/` and `output-files/` are
already populated in the format `download_data.sh` produces.

## 2. Get the data

```bash
./download_data.sh
```

Requires `rclone` configured with a remote named `gdrive` that has access
to `gdrive:PhD/data/gaussian-wavefront-spatially-resolved-inference/`. This
fetches ~17.4 GB: the two datasets actually used
(`R80_N200_A1000000_..._f0.3000/` for 1e6 atoms/shot,
`R40_N50_A100000000_..._f0.3000/` for 1e8 atoms/shot) and the two fine-grid
PSMAP surrogates (`PSGRID4D_CONFOCAL_FINE_Z0.h5`, `..._Z100.h5`). The
remote directory has other datasets/files too; this script only pulls what
these scripts read.

Skip this step if you only want to reproduce the plots/numbers from the
already-committed `results/*.json` -- raw data is only needed to
regenerate results from scratch (Section 4).

## 3. Reproduce the numbers and figures (fast path, no data/GPU needed)

```bash
jupyter notebook notebooks/reproduce_results.ipynb
```

Run all cells. It loads `results/kinematic_estimates_*.json` and
`results/beta_fits_*.json` (already in this branch), prints the RMSE
tables (theta in um/um-s, beta in mrad -- matching the paper's tables), and
regenerates all figures (residual histograms, beta scatter, 2D joint
residual structure) inline. This exactly reproduces
`paper/kinematic_estimation_comparison.pdf`'s numbers and figures.

Equivalently, the same figures can be regenerated as standalone PNGs with:

```bash
cd analysis
python make_paper_figures.py        # -> figures/*.png, results/paper_tables.txt
python make_2d_kinematic_plots.py   # -> figures/residuals_2d_*.png
```

## 4. Regenerate results from scratch / with different parameters

This is the path for trying different hyperparameters for a future paper.
Requires `data/` and `output-files/` populated (Section 2) and a GPU is
strongly recommended.

```bash
cd analysis
python generate_kinematic_estimates.py 1e6 10 50   # <dataset> <n_runs> <n_shots>
python generate_kinematic_estimates.py 1e8 10 50
python beta_fits_from_kinematics.py 1e6 10 50
python beta_fits_from_kinematics.py 1e8 10 50
```

Each `generate_kinematic_estimates.py` run computes, per shot, theta
estimates under all 4 methods (fitting "best" via L-BFGS-B on the
phi-marginalized pixel likelihood is the expensive step) and saves
`results/kinematic_estimates_<dataset>_N<n>_shots<s>.json`.
`beta_fits_from_kinematics.py` consumes that JSON (no re-fitting of theta)
and fits beta = (A_s, A_c) per run per method, saving
`results/beta_fits_<dataset>_N<n>_shots<s>.json`.

To change hyperparameters, edit the module-level constants near the top of
each script:

- `generate_kinematic_estimates.py`: `BINS_BEST` (pixel-likelihood bin
  count, default 32), `TIGHT_HALF_RANGE` (tightened detector half-range in
  meters, default 1.54e-3 -- see paper appendix "Tight-range derivation"
  for why), `PIXEL_NGH` (Gauss-Hermite order for the cloud-averaging
  quadrature, default 8), `M_PHI` (phi-marginalization grid size, default
  2000), and `prior_mean`/`prior_std` (theta prior).
- `beta_fits_from_kinematics.py`: `GH_ORDER` (Gauss-Hermite order for beta
  fitting, default 12), `GH_CHUNK` (batching chunk size, default 20),
  `BINS_BETA` (bin count for the beta-fit likelihood, default 16, native
  detector range -- independent of the theta-fit's tight-range bins=32).

After regenerating, re-run `notebooks/reproduce_results.ipynb` (or the
`make_*` scripts) to get updated tables/figures -- just make sure `N_RUNS`/
`N_SHOTS` in the notebook's second cell match what you passed on the
command line.

## Notes on file layout / portability

All scripts resolve paths relative to the repo root
(`Path(__file__).resolve().parent.parent`), so this branch can be cloned
anywhere -- just keep `python-scripts/`, `helpers/`, `analysis/`, `data/`,
`output-files/`, `results/`, `figures/` as direct children of the same
root (which is how this branch is laid out).
