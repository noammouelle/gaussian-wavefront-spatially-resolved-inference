# Wavefront-inference methods: pipeline & extension branch

This branch is a minimal, self-contained slice of the
`gaussian-wavefront-spatially-resolved-inference` repo containing two full
pipelines, from raw data generation through to final plots, for comparing
position-resolved inference methods:

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

**The goal of this README is to walk through the whole pipeline
end-to-end** -- generate → fit → plot -- so that running it again with new
parameters (different atom count, cloud scatter, wavefront shape, ...) is
a matter of changing a few CLI flags, not re-deriving the workflow. If you
just want the existing numbers/plots without running anything, jump to
Section 3.

It intentionally does **not** include the rest of the parent repo (other
experiments, the full notes history, unrelated scripts) -- goal is a clean,
easy-to-extend base.

## Layout

```
python-scripts/       shared core library + data generation
    generate_data.py        synthetic dataset generation (PSMAP -> images)
    phase_shear_fit.py       phase-shear MLE fit (Gaussian x fringe model on images)
    crb_signal.py, map_inference.py, pixel_acs_grad.py, profile_cloud_nuisances.py
                             non_phase_shear inference library (theta/beta estimators)
helpers/               shared utilities (dataset I/O, PSMAP-adjacent helpers, run tagging)
runs_manifest.jsonl    append-only log of every pipeline stage that's been run (see Section 6)

non_phase_shear/
    analysis/           generate_kinematic_estimates.py, beta_fits_from_kinematics.py,
                         make_paper_figures.py, make_2d_kinematic_plots.py
    notebooks/          reproduce_results.ipynb -- numbers + figures in one place
    results/, figures/  already-computed results (label=1e6/1e8, N_RUNS=10, N_SHOTS=50)
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
(the pixel-likelihood method is orders of magnitude slower on CPU, and even
on GPU the first call in a fresh process pays a one-off ~30-60s cupy/kernel
warmup cost -- expect that delay before you see any progress output) and
for `generate_data.py`. `phase_shear/` fits directly on binned images with
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
see "Extending to arbitrary wavefronts" for that):

```bash
git clone git@github.com:noammouelle/aispp.git ~/local/aispp
cd ~/local/aispp
git checkout d89bbbc77f696069d78d540c7ed748a2a60fb57a   # branch: confocal-mirror-type
# see aispp's own README for build instructions (it's a C++/Python simulator)
```

Note: this aispp commit ("Wire GetDelPhi to the existing
gaussianGradientWavefront for Gaussian beams") already touches
wavefront-gradient handling and may be directly relevant groundwork for
the arbitrary-wavefront extension -- worth checking before duplicating
that work in `generate_data.py`.

## 2. Get the PSMAP files (required prerequisite for everything)

```bash
./download_data.sh
```

This fetches the two PSMAP surrogate files (`output-files/`, ~3 GB,
required before you can generate *any* data) plus the four pre-generated
datasets used for the results already checked into `*/results/` (~28 GB
more -- skip these if you're only generating your own new datasets).

## 3. Reproduce existing numbers and figures (fast path, no data/GPU needed)

```bash
jupyter notebook non_phase_shear/notebooks/reproduce_results.ipynb
jupyter notebook phase_shear/notebooks/shear_mle_distributions.ipynb
```

Run all cells in either. Each loads its own `results/*.json` or
`results/phase_shear_*/*.pkl` (already in this branch), prints the summary
tables, and regenerates all figures inline. No data download needed.

## 4. The pipeline end-to-end

Both pipelines follow the same three stages -- **generate → fit → plot**
-- and share one convention throughout: every dataset and every downstream
result is identified by a **tag** (the dataset's directory name under
`data/`, optionally aliased to a short **label** for output filenames). See
Section 6 for the full tagging convention; the walkthrough below uses it
inline.

### 4a. Generate: `generate_data.py`

One script generates data for *both* pipelines -- the only thing that
differs is whether you pass `--linear_phase_kappa` (nonzero = inject a
wavefront ramp = phase_shear-style data).

```bash
python python-scripts/generate_data.py \
    --n_runs 20 --n_shots 200 --n_atoms 1000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5
    # add --linear_phase_kappa 3.14e4 --linear_phase_site both for phase_shear-style data
```

Key flags (`generate_data.py --help` for the full list):

| flag | meaning |
|---|---|
| `--n_runs`, `--n_shots`, `--n_atoms` | number of independent runs, shots/run, atoms/shot |
| `--mu_x_std`, `--mu_vx_std` | shot-to-shot cloud position/velocity scatter |
| `--sigma_x_mean`, `--sigma_x_std` | cloud width mean/scatter |
| `--signal_amp`, `--signal_freq`, `--signal_phase` | the injected (A_s,A_c) signal (0 = no signal) |
| `--linear_phase_kappa`, `--linear_phase_site` | wavefront phase ramp: 0 = off (non_phase_shear), nonzero = on (phase_shear) |
| `--run_name` | explicit tag; omit to auto-name from the params above |

Output goes to `data/<run_name>/run_000/{Z0,Z100}/data_IMG.h5` (one
`data_IMG.h5` pair per run). A manifest entry is logged automatically (see
Section 6). Reproducing the four datasets already used for the checked-in
results, all with the default `--seed 0`:

```bash
# non_phase_shear, 1e6 atoms  (label used downstream: 1e6)
python python-scripts/generate_data.py --n_runs 80 --n_shots 200 --n_atoms 1000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5
# non_phase_shear, 1e8 atoms  (label: 1e8)
python python-scripts/generate_data.py --n_runs 40 --n_shots 50 --n_atoms 100000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5
# phase_shear, 1e6 atoms
python python-scripts/generate_data.py --n_runs 20 --n_shots 200 --n_atoms 1000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5 \
    --linear_phase_kappa 3.14e4 --linear_phase_site both
# phase_shear, 1e8 atoms
python python-scripts/generate_data.py --n_runs 20 --n_shots 200 --n_atoms 100000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5 \
    --linear_phase_kappa 3.14e4 --linear_phase_site both
```

### 4b. Fit: theta/beta estimators (non_phase_shear) or phase-shear MLE (phase_shear)

**non_phase_shear** -- two scripts, run in sequence, both taking the
dataset directory name (the tag from 4a) as their first positional arg:

```bash
cd non_phase_shear/analysis
python generate_kinematic_estimates.py <dataset_dir> <n_runs> <n_shots> --label <short_tag>
python beta_fits_from_kinematics.py    <dataset_dir> <n_runs> <n_shots> --label <short_tag>
```

- `generate_kinematic_estimates.py` fits per-shot theta under all 4 methods
  (null/moments/best/oracle) for both Z0 and Z100, and also stashes the raw
  photon counts each shot needs later -- this is the expensive step (the
  pixel-likelihood/"best" fit is a per-shot L-BFGS-B optimisation).
  Hyperparameters for the "best" method (`BINS_BEST`, `TIGHT_HALF_RANGE`,
  `PIXEL_NGH`, `M_PHI`, the theta prior) are module-level constants near
  the top of the file -- edit them directly for a new configuration.
  Writes `non_phase_shear/results/kinematic_estimates_<label>_N<n_runs>_shots<n_shots>.json`.
- `beta_fits_from_kinematics.py` consumes that JSON (no theta re-fitting)
  and fits beta=(A_s,A_c) per run per method via Gauss-Hermite-batched
  pixel likelihood. `GH_ORDER`, `GH_CHUNK`, `BINS_BETA` are the equivalent
  module-level constants here. Writes
  `non_phase_shear/results/beta_fits_<label>_N<n_runs>_shots<n_shots>.json`.

**phase_shear** -- one script, sweeping all runs under a dataset directory:

```bash
python python-scripts/phase_shear_fit.py \
    --data_root data/<dataset_dir> \
    --out_dir phase_shear/results/<label> \
    --bins 128
```

Fits the Gaussian x fringe model (`model_image()` in the script -- Gaussian
envelope x (1 + fringe), fringe phase = linear + quadratic terms in x,y) to
every shot's image independently, writing one
`phase_shear/results/<label>/phase_shear_run_NNN.pkl` per run. `--bins`
controls the image downsampling before fitting (128 is the default used
for the checked-in results; lower is faster but less resolved).

### 4c. Plot: figures + tables

**non_phase_shear** (reads the JSON from 4b, writes to `non_phase_shear/figures/` + `results/paper_tables.txt`):
```bash
cd non_phase_shear/analysis
python make_paper_figures.py <label1> [<label2> ...] --n_runs <n> --n_shots <s>
python make_2d_kinematic_plots.py <label1> [<label2> ...] --n_runs <n> --n_shots <s>
```
Both default to `1e6 1e8 --n_runs 10 --n_shots 50` (the checked-in
comparison) if called with no arguments. Pass one or more labels to
compare a different set of datasets/params -- e.g. a single new label to
just plot one configuration, or several to compare side-by-side the way
`1e6`/`1e8` are compared now. All labels passed in one call must share the
same `n_runs`/`n_shots` (that's what the JSON filenames are keyed on).

**phase_shear**: open `phase_shear/notebooks/shear_mle_distributions.ipynb`
and edit the `DATASETS` dict in the first code cell to point at your new
`phase_shear/results/<label>/` directory (plus `N_RUNS`/`N_SHOTS` a couple
of lines below it), then run all cells -- this is a notebook rather than a
CLI script because the phase-shear pipeline's real output *is* the
figures/regression diagnostics shown inline (systematic-bias surface
plots, residual histograms, etc.), not just a couple of summary PNGs.

## 5. Applying this to new parameters -- worked example

Say you want to try 1e7 atoms/shot for the non_phase_shear pipeline:

```bash
# 4a. generate (auto-named tag, since we didn't pass --run_name)
python python-scripts/generate_data.py --n_runs 20 --n_shots 100 --n_atoms 10000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5
# -> prints "Run name : R20_N100_A10000000_..." -- that's your dataset tag

# 4b. fit (use that tag, pick a short label for filenames)
cd non_phase_shear/analysis
python generate_kinematic_estimates.py R20_N100_A10000000_..._phi0random_sig_A0.100_f0.3000 20 100 --label 1e7
python beta_fits_from_kinematics.py    R20_N100_A10000000_..._phi0random_sig_A0.100_f0.3000 20 100 --label 1e7

# 4c. plot -- compare the new 1e7 run against the existing 1e6/1e8
python make_paper_figures.py 1e6 1e7 1e8 --n_runs 20 --n_shots 100
```

(Note `--n_runs`/`--n_shots` must match across all labels passed to
`make_paper_figures.py` in one call -- if 1e6/1e8 were generated with
different n_runs/n_shots than your new 1e7 run, plot them separately or
regenerate 1e6/1e8 with matching values.)

For phase_shear the same idea applies with `phase_shear_fit.py` +
`--out_dir phase_shear/results/1e7_shear` in place of steps 4b, then point
the notebook's `DATASETS` dict at it.

## 6. Tagging convention & run manifest

Every dataset generated by `generate_data.py` gets a **tag** = its
directory name under `data/` -- either auto-generated from the physical
parameters (atom count, cloud scatter, signal, wavefront kappa, ...) or an
explicit `--run_name` you choose. That tag is the single source of truth
for "which dataset was this": every downstream script takes the dataset
tag as an explicit argument (`generate_kinematic_estimates.py`'s first
positional arg, `phase_shear_fit.py --data_root`) rather than a hardcoded
lookup table, so there's no risk of silently reusing the wrong dataset.

A **label** is a short alias for a tag, used only to keep output filenames
readable (`kinematic_estimates_1e6_N10_shots50.json` instead of
`kinematic_estimates_R80_N200_A1000000_..._N10_shots50.json`). Pass
`--label` explicitly wherever you want one; it defaults to the full tag
otherwise.

Every stage (`generate_data.py`, `generate_kinematic_estimates.py`,
`beta_fits_from_kinematics.py`, `phase_shear_fit.py`) appends one line to
`runs_manifest.jsonl` at the repo root recording the tag/label, the
params used, a timestamp, and the output path (see
`helpers/run_manifest.py`). This is the single place to check what's been
run so far and with what parameters:

```bash
cat runs_manifest.jsonl | python -m json.tool --json-lines   # or: jq . runs_manifest.jsonl
grep '"label": "1e7"' runs_manifest.jsonl                     # everything for one tag
```

It's a plain append-only log, not a database -- if you regenerate the same
tag twice you'll get two entries; the most recent one reflects the current
state of that tag's output files.

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
   tag/label-driven (Section 6) and don't otherwise assume anything about
   the wavefront -- pointing them at a new dataset tag should be enough to
   rerun both comparisons unchanged.

The current aispp HEAD (`d89bbbc7`, "Wire GetDelPhi to the existing
gaussianGradientWavefront for Gaussian beams") may already have relevant
groundwork on the simulator side for this -- check before duplicating.

## Notes on file layout / portability

Scripts under `non_phase_shear/analysis/` and `python-scripts/` resolve
paths relative to their own location (`Path(__file__).resolve()...`), not
hardcoded absolute paths, so this branch can be cloned anywhere as long as
the directory structure above is kept intact.
