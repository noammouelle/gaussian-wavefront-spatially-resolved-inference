# Wavefront-inference methods: reproduction & extension branch

This branch is a minimal, self-contained slice of the
`gaussian-wavefront-spatially-resolved-inference` repo, reduced to exactly
what's needed to (1) generate synthetic interferometer datasets and (2)
reproduce both position-resolved inference pipelines built on top of them:

- **`non_phase_shear/`** -- per-shot kinematic parameter (theta) and signal
  parameter (beta = (A_s,A_c)) recovery, comparing **Null** (prior mean),
  **MAP-moments** (Kalman-gain estimator), **Pixel-likelihood** ("best":
  phi-marginalized pixel-resolved likelihood), and **Oracle** (true theta).
- **`phase_shear/`** -- recovery of a differential phase signal in the
  presence of a spatially-varying (linear-ramp) wavefront systematic,
  comparing **raw** (uncorrected), a **fitted-feature regression**
  correction (practically available: regress on cloud-shape parameters
  fitted from the image), and an **oracle regression** correction
  (idealised: regress on the true simulated kinematics instead).

Both pipelines consume the same underlying simulated datasets (images from
a Gaussian atom cloud imaged through the PSMAP point-spread surrogate) --
they differ only in whether a phase ramp (wavefront gradient) was injected
at generation time, and in which observable/model each pipeline fits. This
is intentional: the plan is to extend `generate_data.py`'s currently
linear-only phase-ramp injection to **arbitrary wavefronts** and rerun both
pipelines unchanged against the new data -- see "Extending to arbitrary
wavefronts" below.

It intentionally does **not** include the rest of the parent repo (other
experiments, the full notes history, unrelated scripts) -- goal is a clean,
easy-to-extend base.

## Layout

```
python-scripts/       shared core library + data generation
    generate_data.py        synthetic dataset generation (PSMAP -> images)
    phase_shear_fit.py       phase-shear MLE fit (Gaussian x fringe model on images)
    crb_signal.py, map_inference.py, joint_profile_inference.py,
    pixel_acs_grad.py, pixel_acs_grad_batch.py, profile_cloud_nuisances.py
                             non_phase_shear inference library (theta/beta estimators)
helpers/               shared utilities (dataset I/O, PSMAP-adjacent helpers)

non_phase_shear/
    analysis/           generate_kinematic_estimates.py, beta_fits_from_kinematics.py,
                         make_paper_figures.py, make_2d_kinematic_plots.py
    notebooks/          reproduce_results.ipynb -- numbers + figures in one place
    results/, figures/  already-computed results (N_RUNS=10, N_SHOTS=50)
    paper/              kinematic_estimation_comparison.tex/.pdf

phase_shear/
    notebooks/          shear_mle_distributions.ipynb -- numbers + figures in one place
    results/            already-computed results (phase_shear_1e6/, phase_shear_1e8/,
                         N_RUNS=20, N_SHOTS=20)
    paper/              phase_shear_results.tex/.pdf

download_data.sh       fetches data/ and output-files/ from the Drive backup
requirements.txt       pinned Python dependencies
```

`data/` and `output-files/` (raw simulated images and PSMAP surrogate
files, ~31 GB total) are **not** committed to git -- fetch them with
`download_data.sh`, or generate your own with `generate_data.py` (only
`output-files/` is a hard prerequisite for that). If you only want the
existing numbers/plots, `*/results/` and `*/figures/` already have
everything and no data download is needed.

## 1. Environment setup

Both pipelines were run under a single conda environment (referred to here
as `aispy_env`) that has `aispy` installed editable, plus everything in
`requirements.txt`. Recommended: reproduce that setup rather than mixing
environments.

```bash
conda create -n aispy_env python=3.14
conda activate aispy_env
pip install -r requirements.txt
```

**A CUDA-capable NVIDIA GPU is strongly recommended** for `non_phase_shear/`
(the pixel-likelihood method is orders of magnitude slower on CPU) and for
`generate_data.py`; `phase_shear/` fits directly on binned images with
`scipy.optimize` and does not use the GPU. Everything still runs on the
numbers already in `*/results/` without a GPU or without `aispy` at all --
you only need both for data generation / regenerating results from scratch.

### aispy dependency (required for generation and for both analysis pipelines)

The core library imports `aispy.psmap` (`load_psmap`, `PSMAPSurrogate`) to
read and interpolate the point-spread-map surrogate files, and
`generate_data.py` imports the same to simulate new datasets. This is
**not** a pip package -- it's a sibling repo that must be installed
separately:

```bash
git clone git@github.com:noammouelle/aispy.git ~/local/aispy
cd ~/local/aispy
git checkout c3a39e35d46f9a60984170aa4eacbb867029eae8   # branch: trajectory-plots
pip install -e .
```

### aispp / ais++ dependency (only needed to regenerate PSMAP files from scratch)

`aispp` is the underlying wavefront/photon-detection simulator used to
generate the PSMAP surrogate files in `output-files/` in the first place.
**Nothing in this branch imports aispp** -- `generate_data.py` and both
analysis pipelines only ever consume the already-built PSMAP `.h5` files
(fetched via `download_data.sh`). aispp is documented here purely for
provenance / in case you want to regenerate the PSMAP itself (e.g. to
change the physical interferometer geometry, not the injected wavefront --
see next section for that):

```bash
git clone git@github.com:noammouelle/aispp.git ~/local/aispp
cd ~/local/aispp
git checkout d89bbbc77f696069d78d540c7ed748a2a60fb57a   # branch: confocal-mirror-type
# see aispp's own README for build instructions (it's a C++/Python simulator)
```

Note: this aispp commit ("Wire GetDelPhi to the existing
gaussianGradientWavefront for Gaussian beams") already touches
wavefront-gradient handling and may be directly relevant groundwork for
the arbitrary-wavefront extension below -- worth checking before
duplicating that work in `generate_data.py`.

## 2. Get the data

```bash
./download_data.sh
```

See the script header for exactly what's fetched and why (~31 GB: PSMAP
surrogates + the two `non_phase_shear` datasets + the two `phase_shear`
datasets).

## 3. Reproduce existing numbers and figures (fast path, no data/GPU needed)

```bash
jupyter notebook non_phase_shear/notebooks/reproduce_results.ipynb
jupyter notebook phase_shear/notebooks/shear_mle_distributions.ipynb
```

Run all cells in either. Each loads its own `results/*.json` or
`results/phase_shear_*/*.pkl` (already in this branch), prints the summary
tables, and regenerates all figures inline.

## 4. Generate new data

```bash
python python-scripts/generate_data.py \
    --n_runs 20 --n_shots 200 --n_atoms 1000000 \
    --linear_phase_kappa 3.14e4 --linear_phase_site both   # omit these two flags for non_phase_shear-style data
```

Key flags (see `generate_data.py --help` for the full list): `--n_runs`,
`--n_shots`, `--n_atoms`, `--mu_x_std`/`--mu_vx_std` (kinematic scatter),
`--sigma_x_mean`/`--sigma_x_std` (cloud width scatter), `--signal_amp`/
`--signal_freq`/`--signal_phase` (injected (A_s,A_c) signal), and
`--linear_phase_kappa`/`--linear_phase_site` (the wavefront phase ramp --
0 = off, which is what `non_phase_shear/` datasets use; nonzero is what
`phase_shear/` datasets use). Output goes to `data/<auto-named-run>/`.

Reproducing the exact datasets used here (same seed=0 default):
```bash
# non_phase_shear, 1e6 atoms:
python python-scripts/generate_data.py --n_runs 80 --n_shots 200 --n_atoms 1000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5
# non_phase_shear, 1e8 atoms:
python python-scripts/generate_data.py --n_runs 40 --n_shots 50 --n_atoms 100000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5
# phase_shear, 1e6 / 1e8 atoms:
python python-scripts/generate_data.py --n_runs 20 --n_shots 200 --n_atoms 1000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5 \
    --linear_phase_kappa 3.14e4 --linear_phase_site both
python python-scripts/generate_data.py --n_runs 20 --n_shots 200 --n_atoms 100000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5 \
    --linear_phase_kappa 3.14e4 --linear_phase_site both
```

## 5. Regenerate results from new data / with different parameters

**non_phase_shear:**
```bash
cd non_phase_shear/analysis
python generate_kinematic_estimates.py <1e6|1e8|your_dataset_key> <n_runs> <n_shots>
python beta_fits_from_kinematics.py    <same args>
python make_paper_figures.py
python make_2d_kinematic_plots.py
```
`generate_kinematic_estimates.py` and `beta_fits_from_kinematics.py` hardcode
a `DATASETS` dict mapping a short key (`'1e6'`, `'1e8'`) to the dataset
directory name under `data/` -- add an entry for a new dataset there.
Hyperparameters (pixel-likelihood bins, tight-range half-width,
Gauss-Hermite order, prior, ...) are module-level constants near the top of
each script -- see comments in `README` of the previous single-pipeline
branch (or just read the scripts, they're short and documented inline).

**phase_shear:**
```bash
python python-scripts/phase_shear_fit.py \
    --data_root data/<your_dataset> \
    --out_dir phase_shear/results/<your_label> \
    --bins 128
```
Then point `phase_shear/notebooks/shear_mle_distributions.ipynb`'s
`DATASETS` dict at the new `phase_shear/results/<your_label>` directory
and re-run.

## Extending to arbitrary wavefronts

Currently `generate_data.py --linear_phase_kappa` only supports a linear
phase ramp `phi = kappa * xf` (see `_linear_phase_profile()` in that file).
To inject an arbitrary wavefront:

1. Generalise `_linear_phase_profile()` (or add a new function) to accept
   an arbitrary `phi(xf, yf, vxf, vyf)` callable instead of a fixed linear
   form -- the rest of `generate_data.py`'s pipeline (`apply_ramp`,
   `phase_profile` argument threading) already treats the phase profile as
   an opaque function, so this should be a localised change.
2. `phase_shear_fit.py`'s fringe model (`_phase()`, currently
   `kappa_x*xf + kappa_y*yf + gamma_x*xf**2 + gamma_y*yf**2`, i.e. linear +
   quadratic) would need matching terms added for higher-order wavefronts
   (e.g. Zernike terms) if you want the MLE fit itself (not just the
   injected truth) to track a more complex wavefront shape.
3. Both `non_phase_shear/` and `phase_shear/` analysis pipelines are
   dataset-key-driven (see Section 5) and don't otherwise assume anything
   about the wavefront -- pointing them at a new dataset directory (with a
   new `DATASETS` entry) should be enough to rerun both comparisons
   unchanged.

The current aispp HEAD (`d89bbbc7`, "Wire GetDelPhi to the existing
gaussianGradientWavefront for Gaussian beams") may already have relevant
groundwork on the simulator side for this -- check before duplicating.

## Notes on file layout / portability

Scripts under `non_phase_shear/analysis/` and `python-scripts/` resolve
paths relative to their own location (`Path(__file__).resolve()...`), not
hardcoded absolute paths, so this branch can be cloned anywhere as long as
the directory structure above is kept intact.
