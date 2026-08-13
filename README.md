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
they differ only in whether a wavefront systematic was injected at PSMAP
generation time, and in which observable/model each pipeline fits. This
branch now also includes the machinery to generate **arbitrary wavefronts**
(not just an analytic confocal/gaussian beam) via `aisoptics` +
`ais++`'s `wtype=interpolated` beam, and rerun both pipelines against a
PSMAP built from one -- see Section 2b and "Arbitrary wavefronts: status
and what's still open" below.

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
python-scripts/       shared core library + data/PSMAP generation
    phase_space_grids.py     builds the PSMAP itself (ais++ psgrid mode): confocal
                             (analytic, default) or interpolated (arbitrary wavefront)
    generate_data.py        synthetic shot-image dataset generation (PSMAP -> images)
    phase_shear_fit.py       phase-shear MLE fit (Gaussian x fringe model on images)
    one_atom_traj.py         generates the reference trajectory (needs a built ais++ binary)
    crb_signal.py, map_inference.py, pixel_acs_grad.py, profile_cloud_nuisances.py
                             non_phase_shear inference library (theta/beta estimators)
helpers/               shared utilities (dataset I/O, PSMAP-adjacent helpers, run tagging)
runs_manifest.jsonl    append-only log of every pipeline stage that's been run (see Section 6)

optics/                arbitrary-wavefront generation (aisoptics), independent of any
                        one dataset -- see Section 4a-pre
    generate_wavefront.py    builds down (perfect)/up (aberrated-on-reflection) beam
                             fields, exports HDF5 for phase_space_grids.py --wtype interpolated
    convergence_check.py     grid self-convergence check before trusting a resolution
    notebooks/wavefront_visualization.ipynb
                             phase/amplitude maps at the mirror plane, down vs. up
    fields/                 generated HDF5 fields (gitignored; regenerate as needed)

notebooks/             shared inspection notebooks (data_inspection.ipynb,
                        psmap_inspection.ipynb, trajectory_inspection.ipynb)
output-files/one_atom.h5, one_atom_TRAJ.h5
                        small reference trajectory files, committed directly (not
                        gitignored like the rest of output-files/ -- no download needed)

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

### aispy dependency (required for everything)

The core library imports `aispy.psmap` (`load_psmap`, `PSMAPSurrogate`) to
read and interpolate the point-spread-map surrogate files, and
`generate_data.py`/`phase_space_grids.py`/`optics/` all build `.aisi` input
files through `aispy.utils.AISFlow`. This is **not** a pip package -- it's
a sibling repo that must be installed separately, **as an editable
install** (a plain, non-editable `pip install` silently freezes a stale
copy, which caused real bugs during development -- see `git log` for
`_write_wavefront_params` in `aispy` if curious):

```bash
git clone git@github.com:noammouelle/aispy.git ~/local/aispy
cd ~/local/aispy
git checkout v0.0.2   # or master; commit 720c098d.. + local fixes as of this writing
pip install -e .
```

### aisoptics dependency (required only for Section 2b -- arbitrary wavefronts)

`aisoptics` builds and samples optical fields (analytic Gaussian beams,
Fourier-mode/sampled-map perturbations, composite fields) and exports them
in the HDF5 layout `aispp`'s `wtype=interpolated` beam reads. Not needed
for anything in Sections 1-6 (the confocal-PSMAP-based pipelines); only for
`optics/` in Section 2b.

```bash
git clone git@github.com:noammouelle/aisoptics.git ~/local/aisoptics
cd ~/local/aisoptics
git checkout v0.0.2
pip install -e ".[dev]"
```

### aispp / ais++ dependency (only needed to (re)generate PSMAP files, Section 2b)

`aispp` is the underlying atom-interferometer simulator (compiled C++
binary, `ais++`) used to generate the PSMAP surrogate files in
`output-files/` in the first place, via `phase_space_grids.py`. **Nothing
in the day-to-day pipeline (Sections 3-6) imports or runs aispp** --
`generate_data.py` and both analysis pipelines only ever consume the
already-built PSMAP `.h5` files (fetched via `download_data.sh`, or built
once via Section 2b). You need a built `ais++` binary only if you're
generating a new PSMAP (default confocal, or arbitrary wavefront via
`optics/`) or the reference trajectory (`one_atom_traj.py`).

```bash
git clone git@github.com:noammouelle/aispp.git ~/local/aispp
cd ~/local/aispp
git checkout v0.0.2   # commit 61be38d, includes wtype=interpolated support
mkdir build && cd build
cmake .. && cmake --build . --target ais++ -j4   # needs HDF5 (H5Cpp.h) + GSL + OpenMP dev packages
```
`KNOWN_ISSUES.md` in the aispp repo is worth reading before relying on
`wtype=interpolated` for anything beyond phase observables: as of v0.0.2,
`AISLaserBeam::GetDelPhi` returns zero for *every* beam type (confocal,
gaussian, and interpolated alike), so the wavefront gradient does not
perturb the atom's trajectory/recoil -- only the accumulated phase is
affected. Verified directly (see Section 2b): an interpolated beam with no
aberration reproduces the sampled-vs-analytic case closely, and adding a
Fourier-mode aberration changes the output phase shift by a real,
non-trivial amount while leaving position/velocity bit-identical.

## 2. Get the PSMAP files (required prerequisite for everything)

```bash
./download_data.sh
```

This fetches the two PSMAP surrogate files (`output-files/`, ~3 GB,
required before you can generate *any* data) plus the four pre-generated
datasets used for the results already checked into `*/results/` (~28 GB
more -- skip these if you're only generating your own new datasets).

## 2b. Optional: generate your own PSMAP (confocal or arbitrary wavefront)

Everything above assumes the two downloaded PSMAP files
(`PSGRID4D_CONFOCAL_FINE_Z{0,100}.h5`, analytic confocal-mirror beam). This
section is for when you want a *different* PSMAP -- either regenerating the
same confocal one from scratch, or building one under an **arbitrary
wavefront** (any field `aisoptics` can produce: Fourier modes, sampled
maps, ...) instead of the analytic beam. This is a much bigger step up from
`generate_data.py`'s own `--linear_phase_kappa` (a cheap post-hoc phase
ramp painted onto images that already exist) -- it reruns the actual
`ais++` atom-optics simulation with a different beam and produces a new
PSMAP that captures how that wavefront really affects `dphi`/amplitude
as a function of the atom's initial phase-space coordinates.

**Requires a built `ais++` binary** (see the aispp section above) -- this
is the one place in this repo (besides `one_atom_traj.py`) that needs more
than `aispy`+`aisoptics`.

### Physical model

A perfect, unaberrated beam travels down to the retroreflecting mirror at
z=0 and picks up a wavefront aberration on reflection (mirror surface
imperfections), so the returning (upward) beam is the aberrated one:

```
down beam = GaussianBeam(...)                              (perfect)
up beam   = GaussianBeam(...) + phase_perturbation(x, y)    (aberrated)
```

### Step 1: build and visualise the wavefront (`optics/`)

Two aberration bases, combinable and summed at the mirror plane:

```bash
cd optics
python generate_wavefront.py --tag my_wavefront \
    --zernike noll=4 amp=0.05 \
    --mode qx=1571 qy=0 amp=0.15 phase=0 \
    --nx 9 --ny 9 --nz 401 --zlim -5 25
```
- `--mode`: Fourier mode (`qx`, `qy` in rad/m, `amp` in rad, `phase` in
  rad). Repeat for a sum of several.
- `--zernike`: Zernike polynomial term (`noll=<Noll index>`, `amp=<rad>`,
  same Noll-index/normalisation convention as `ais++`'s native
  `zernikecoeff_N` -- see `aisoptics.ZernikeAberration`'s docstring).
  Repeat for a sum of several. `--beam_radius` sets the aperture
  (`rho = r/beam_radius`; must stay <= your `--xlim`/`--ylim`, see below).

Omit both entirely for a flat (unaberrated) mirror -- useful as a sanity
check that `wtype=interpolated` reproduces the analytic confocal/gaussian
result. Writes `optics/fields/my_wavefront_{down,up}.h5`.

**How the aberration actually gets applied -- this matters, read before
using**: the aberration (Zernike and/or Fourier) is imprinted **only at
the mirror plane** (`--mirror_z`, default = `--focus_z`), then the whole
field is **propagated** (Fresnel/paraxial, via `aisoptics.ParaxialPropagator`)
to every other z the grid covers. This is deliberately *not* how
`ais++`'s own native `wtype=confocal zernikecoeff_N` works -- that applies
the identical transverse pattern at whatever z the atom happens to be at,
with no z-dependence at all (`GetZernikePhase` only reads `pos[0]`,
`pos[1]`). A real aberrated wavefront diffracts away from where it was
imprinted; the rigid native version silently assumes it doesn't. Passing
the same coefficient to both paths should agree closely right at the
mirror and diverge with distance -- `optics/notebooks/wavefront_visualization.ipynb`'s
last section demonstrates this directly (and is how this was validated
during development: propagated field reproduces the imprinted pattern to
~1e-9 rad exactly at the mirror plane, and the RMS deviation from that
rigid pattern grows monotonically and measurably, though slowly, with
distance -- consistent with the beam's own Fresnel length, ~1300 m for the
default waist/wavelength, being much longer than the ~25 m tested).

**Two easy-to-hit failure modes, both checked automatically (the script
warns; it does not silently produce garbage without saying so)**:
- *Zernike aperture vs. grid extent*: `rho = r/beam_radius` is only
  physically meaningful for `rho <= 1`. If `--xlim`/`--ylim` extend past
  `--beam_radius`, points outside the aperture get large/unphysical raw
  polynomial values, not an error -- keep `--beam_radius >= half of your
  --xlim/--ylim span`.
- *FFT periodic-boundary aliasing*: both propagators are FFT-based, which
  implicitly assumes the field is periodic across the transverse grid. If
  the beam's amplitude hasn't decayed to ~0 by the `--xlim`/`--ylim` edge,
  the propagation wraps around and corrupts the result with no error --
  only a warning if the edge/peak amplitude ratio exceeds 1e-3. This is
  set by the **Gaussian beam's waist**, not by `--beam_radius` -- the
  defaults (`--xlim`/`--ylim` = 3x `--waist`) satisfy it (edge ratio
  ~1e-4); shrinking `--xlim`/`--ylim` to match a small `--beam_radius`
  (rather than the waist) is exactly how to reintroduce this bug.

**Before trusting a grid resolution**, check it's actually resolved the
wavefront (and its propagation) you asked for:
```bash
python convergence_check.py --zernike noll=4 amp=0.05 \
    --nxy 5 9 17 33 --nz 41 81 161 321
```
Compares increasingly fine grids against the finest one (self-consistency
under refinement -- not proof the finest grid is exact). Watch `phase_rms`
in the printed report and `convergence_report.png`; `amplitude_relative_rms`
is not a useful metric here (it blows up in the beam's low-amplitude wings
where dividing by ~0 amplitude dominates the "relative" error -- see the
script's own printed note). Higher spatial-frequency `--mode`/`--zernike`
terms need finer grids to resolve; there's no universally-correct default
resolution. This also runs the same edge-amplitude check as
`generate_wavefront.py` and warns if it fails at any tested resolution.

Then look at it:
```bash
jupyter notebook optics/notebooks/wavefront_visualization.ipynb
```
Loads the `_down.h5`/`_up.h5` files directly (exactly what `ais++` will
read, not a re-derivation) -- just set `DOWN_FILE`/`UP_FILE` in the config
cell to the pair you generated. Optionally set `REFERENCE_UP_FILE` to a
second, unaberrated up-beam file (same grid, no `--mode`/`--zernike`) to
isolate the aberration cleanly and get the propagation-vs-distance
residual plot; without it the notebook still runs, just showing raw phase
instead of an isolated aberration. Shows phase + amplitude at the mirror
plane (nearest loaded z to `MIRROR_Z`) for both beams, the isolated
aberration, a 3D surface, and the propagation-vs-distance demonstration
described above (all read straight off the loaded z-grid, since the file
already covers the full range it was generated with).

**z-range matters**: the grid must cover wherever the atom actually is when
a pulse fires. `phase_space_grids.py` launches atoms from `z0=0` (bottom
source) or `z0=100` (top source, MAGIS-100 baseline) and they fly up and
back down; the z=0 source stays within a few tens of metres of the mirror
(z0-relative apex height depends on `--T`/`lmt_order`, ~20m for the
defaults), but the z=100 source needs the field sampled out to ~120m+.
Building one field that covers both is expensive; it's usually more
practical to generate a wavefront per `z0` value if you need both.

#### Recipe: random Zernike aberration with a target total RMS

`--zernike_random n=<count> rms=<rad>` picks `n` distinct random Noll
indices and random amplitudes such that the combined phase RMS over the
aperture equals `rms` (radians). This relies on `aisoptics`'s Zernike terms
being individually RMS-normalised to 1 over the unit disk (Noll convention,
verified numerically) and mutually orthogonal there, so RMS combines in
quadrature: `rms = 2*pi*sqrt(sum(amplitude_i**2))`. It defaults to
excluding Noll 1-3 (piston is physically inert; tip/tilt is a beam-pointing
offset, not a wavefront distortion) -- override with `min_noll=1` if you
actually want those. `split=equal` (default) gives every term the same RMS
share; `split=dirichlet` gives a random, uneven split instead. Always pass
`seed=<int>` for a reproducible draw -- otherwise every run differs and
can't be regenerated later.

```bash
cd optics
python generate_wavefront.py --tag confocal_random5 \
    --zernike_random n=5 rms=0.1 seed=42 \
    --beam_radius 0.03 --nx 9 --ny 9 --nz 401 --zlim -5 25
```
This combines with explicit `--zernike`/`--mode` terms if you pass both.
The chosen `(noll, amplitude)` pairs are printed to stdout and saved in the
`runs_manifest.jsonl` entry (`zernike_terms`) -- that's the actual
reproducible spec; `zernike_random_spec` alone only reproduces it if
`aisoptics`'s RNG usage doesn't change between versions, so if you need to
rerun an old draw exactly, prefer copying the printed `--zernike noll=...
amp=...` pairs over relying on the same `--zernike_random` args.

Note this only controls what the aberration *is* -- it has nothing to do
with confocal vs. flat mirror geometry, which is a property of the
`ais++`/`phase_space_grids.py` beam config (`wtype=confocal` vs.
`wtype=gaussian`, both driven by the same `wtype=interpolated` field files
here) and is set entirely in Step 2 below.

Same workflow as any other `--zernike` run from here: convergence-check it
before trusting the resolution (random higher-Noll terms can need a finer
grid than a hand-picked low-order one), then look at it in the notebook:
```bash
python convergence_check.py --zernike_random n=5 rms=0.1 seed=42 \
    --nxy 5 9 17 33 --nz 41 81 161 321

jupyter notebook optics/notebooks/wavefront_visualization.ipynb
```
For the notebook, just point `DOWN_FILE`/`UP_FILE` at `confocal_random5_{down,up}.h5`
(and `REFERENCE_UP_FILE` at a flat-mirror run's `_up.h5` on the same grid,
if you want the isolated-aberration/propagation-residual plots rather than
raw phase) -- no need to copy the printed `noll=... amp=...` pairs in by
hand, since the notebook loads whatever was actually written to disk.

### Step 2: build the PSMAP (`phase_space_grids.py`)

```bash
cd ../python-scripts
python phase_space_grids.py --nx 25 --ny 25 --nvx 25 --nvy 25 \
    --wtype interpolated \
    --beam_file_down ../optics/fields/my_wavefront_down.h5 \
    --beam_file_up   ../optics/fields/my_wavefront_up.h5 \
    --tag MY_WAVEFRONT
```
(`--nx`/`--ny`/`--nvx`/`--nvy` must be odd, ≥5 -- `PSMAPSurrogate` uses
cubic interpolation, which needs at least 4 points per axis; the production
PSMAP uses 25.) Omit `--wtype`/`--beam_file_*`/`--tag` entirely to
regenerate the default analytic confocal PSMAP (`--tag` then defaults to
`CONFOCAL_FINE`, matching the already-downloaded files' names). Writes
`input-files/PSGRID4D_<tag>_Z{0,100}.aisi`; run each through `ais++` as the
script prints at the end:
```bash
ais++ -i ../input-files/PSGRID4D_MY_WAVEFRONT_Z0.aisi   -o ../output-files/PSGRID4D_MY_WAVEFRONT_Z0.h5
ais++ -i ../input-files/PSGRID4D_MY_WAVEFRONT_Z100.aisi -o ../output-files/PSGRID4D_MY_WAVEFRONT_Z100.h5
```
A 25×25×25×25 grid is 390,625 atoms per file and will take a while; test
with a small grid first (`--nx 5 --ny 5 --nvx 5 --nvy 5` runs in seconds)
to confirm the wiring works before committing to the full run.

Verify the result loads normally:
```python
from aispy.psmap import load_psmap, PSMAPSurrogate
psmap = load_psmap('output-files/PSGRID4D_MY_WAVEFRONT_Z0.h5')
surrogate = PSMAPSurrogate(psmap, t_det=3.8, use_gpu=True)   # same class map_inference.py uses
```

### Step 3: point the inference pipeline at it

Every script that loads a PSMAP takes `--psmap_tag` (default
`CONFOCAL_FINE`, matching the downloaded analytic files -- omit the flag
entirely for the default confocal pipeline). Pass your `phase_space_grids.py
--tag` value to run the exact same generate/fit/plot pipeline from Sections
4-6 against your arbitrary-wavefront PSMAP instead:

```bash
cd python-scripts
python generate_data.py --psmap_tag MY_WAVEFRONT \
    --n_runs 20 --n_shots 200 --n_atoms 1000000 \
    --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5
    # -> data/<auto-named-run>/, generated using the MY_WAVEFRONT PSMAP

cd ../non_phase_shear/analysis
python generate_kinematic_estimates.py <dataset_dir> 20 200 \
    --label my_wavefront --psmap_tag MY_WAVEFRONT
python beta_fits_from_kinematics.py <dataset_dir> 20 200 \
    --label my_wavefront --psmap_tag MY_WAVEFRONT
python make_paper_figures.py my_wavefront --n_runs 20 --n_shots 200
```

`--psmap_tag` on `generate_kinematic_estimates.py`/`beta_fits_from_kinematics.py`
should match whatever `--psmap_tag` the *dataset itself* was generated
under (`generate_data.py`'s), not necessarily a different one -- theta/beta
estimation reads the same PSMAP the images were rendered through. Also
supported on `map_inference.py` and `crb_signal.py` (their own standalone
CLIs). `profile_cloud_nuisances.py` already had a similar, more general
`--psmap-z0`/`--psmap-z100` (full path, not tag) from before this branch.

`phase_shear_fit.py` doesn't take `--psmap_tag` -- the phase_shear pipeline
fits a fringe model directly on images and never touches the PSMAP at all
(see Section 4b), so it's unaffected by which PSMAP a dataset was generated
under and needs no flag here.

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

### Visualise the raw data / PSMAP / trajectory (recommended before fitting)

`notebooks/` (repo root, shared by both pipelines -- these aren't specific
to phase_shear or non_phase_shear) has three inspection notebooks for
sanity-checking things *before* trusting any fit built on top of them:

- **`data_inspection.ipynb`** -- single-shot images, ground-state-fraction
  maps (Z0, Z100, and their differential), and the port-count "fringe
  ellipse" across shots, for any dataset under `data/`. Point it at a
  freshly-generated dataset to confirm the cloud/fringe look sane before
  running 4b.
- **`psmap_inspection.ipynb`** -- visualises `PSMAPSurrogate.eval()` (the
  differential phase and per-port amplitudes vs. pairs of initial
  kinematic coordinates) for a PSMAP file. Check for smooth surfaces (no
  interpolation artifacts) before trusting fits that depend on it.
- **`trajectory_inspection.ipynb`** -- spacetime diagram of a single-atom
  reference trajectory (`output-files/one_atom_TRAJ.h5`, committed
  directly to this branch since it's small -- no download needed). Useful
  for checking pulse-sequence timing looks right. Regenerating it needs a
  built `ais++` binary from `aispp` (see the notebook's header cell) --
  the one place in this branch that actually requires more than `aispy`.

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

## Arbitrary wavefronts: status and what's still open

Section 2b covers the now-working path: `optics/` (aisoptics) builds a
sampled wavefront, `phase_space_grids.py --wtype interpolated` bakes it
into a real PSMAP via `ais++`, and everything downstream (Sections 3-6)
consumes that PSMAP exactly like the confocal one, once you've done the
manual swap described there. This supersedes `generate_data.py
--linear_phase_kappa` (a cheap post-hoc phase ramp painted onto images
after generation, not run through the actual beam-optics simulation) for
anything where physical fidelity matters -- `--linear_phase_kappa` is
still there and still useful as a fast/cheap approximation when it's good
enough.

Genuinely still open:

- **`GetDelPhi` is zero for every beam type** (aispp `KNOWN_ISSUES.md`) --
  an arbitrary wavefront affects the PSMAP's phase but not the simulated
  trajectory/recoil. Fine for this pipeline's phase-based observables;
  would matter if a future observable depended on the wavefront kicking
  the atom's momentum.
- **`phase_shear_fit.py`'s fringe model** (`_phase()`, currently
  `kappa_x*xf + kappa_y*yf + gamma_x*xf**2 + gamma_y*yf**2`, i.e. linear +
  quadratic) only has enough terms to *fit* a linear/quadratic wavefront.
  A PSMAP built from a higher-order arbitrary wavefront (e.g. several
  Fourier modes) will still generate correctly and flow through
  `non_phase_shear/` (which doesn't fit a fringe model at all, just theta/
  beta from the pixel likelihood) but `phase_shear/`'s MLE fit itself won't
  track the extra structure unless matching terms are added here.
- **z-range cost for the z0=100 source** -- see Section 2b's z-range note;
  no shortcut implemented, just flagged.

## Notes on file layout / portability

Scripts under `non_phase_shear/analysis/` and `python-scripts/` resolve
paths relative to their own location (`Path(__file__).resolve()...`), not
hardcoded absolute paths, so this branch can be cloned anywhere as long as
the directory structure above is kept intact.
