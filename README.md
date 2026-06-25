The goal of this project is to demonstrate the efficacy of position-resolved measurements in atom interferometry.

## Reusable 2D PCA pipeline

The expensive atom-level HDF5 pass is separated from lightweight PCA and regression analysis.

Build compact shot artifacts with:

    python python-scripts/build_2d_shot_features.py DATA_ROOT --output results/2d-shot-features --bins 64 --workers 1

The output contains memory-mappable ground/excited count maps, scalar shot summaries and initial-condition metadata, plus a source manifest. Increase workers to 2 only after benchmarking: HDF5 decompression and disk throughput are normally the bottleneck.

Fit compact PCA artifacts with:

    python python-scripts/fit_2d_excitation_pca.py results/2d-shot-features --output results/2d-shot-features/pca_results.npz --components 10 --representation contrast

Population contrast, (ne - ng) / (ne + ng), is the default. After per-shot offset removal and per-pixel standardization it is affine-equivalent to using ne / (ne + ng). Raw ground and excited maps are retained, so PCA can be refitted without rereading the large HDF5 files.

The optional GPU backend uses CuPy for the compact covariance eigendecomposition. This does not accelerate the expensive extraction pass and is usually unnecessary; CPU PCA is already fast once compact artifacts exist.

Open notebooks/summary_statistics_test_2D.ipynb after creating the artifacts. It only loads compact files and performs visualization, analytical template comparisons, regression, and a grouped-by-run holdout check.

The 2D PCA extraction builds count maps in cloud-normalized final coordinates, `(x - mean_x) / std_x` and `(y - mean_y) / std_y`. The physical `mean_x`, `mean_y`, `std_x`, and `std_y` are still saved as scalar summary features for regression.

## End-to-end MLE experiment workflow

This section is the reproducible path from simulated data to feature-conditioned
MLE comparisons.  Replace `DATA_ROOT` with the generated dataset directory you
want to analyze, for example:

    DATA_ROOT=data/R80_N50_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx309um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000

### 1. Generate or choose a dataset

Datasets are expected to have repeated-run subdirectories of the form:

    DATA_ROOT/run_000/Z0/data_PROB.h5
    DATA_ROOT/run_000/Z100/data_PROB.h5
    DATA_ROOT/run_001/Z0/data_PROB.h5
    DATA_ROOT/run_001/Z100/data_PROB.h5
    ...

If the dataset already exists, skip generation.  To inspect generation options:

    python python-scripts/generate_data.py --help

### 2. Build compact shot features for both interferometer sites

Build the Z0 artifact:

    python python-scripts/build_2d_shot_features.py "$DATA_ROOT" \
      --port Z0 \
      --output results/2d-shot-features-z0 \
      --bins 64 \
      --workers 1

Build the Z100 artifact:

    python python-scripts/build_2d_shot_features.py "$DATA_ROOT" \
      --port Z100 \
      --output results/2d-shot-features-z100 \
      --bins 64 \
      --workers 1

Each artifact contains:

- `shot_metadata.npz`: per-shot scalar summaries and initial-condition metadata
- `ground_counts.npy`, `excited_counts.npy`: memory-mappable count images
- `manifest.json`: source runs and extraction settings

The MLE fitter uses these feature artifacts directly.  The diagnostics notebook
also uses their compact count maps for all-shot residual plots, avoiding slow raw
atom-level HDF5 scans.

### 3. Fit PCA artifacts for both sites, if using PC score features

Fit Z0 PCA scores:

    python python-scripts/fit_2d_excitation_pca.py \
      results/2d-shot-features-z0 \
      --output results/2d-shot-features-z0/2d-shot-pcas.npz \
      --components 10 \
      --representation contrast

Fit Z100 PCA scores:

    python python-scripts/fit_2d_excitation_pca.py \
      results/2d-shot-features-z100 \
      --output results/2d-shot-features-z100/2d-shot-pcas.npz \
      --components 10 \
      --representation contrast

Skip this step if you only use scalar summary features such as `mean_x`, `std_x`,
or `cov_x_state`.

The fitter also accepts the derived scalar features `var_x=std_x**2` and
`var_y=std_y**2`. They are squared before global standardization and reuse the
existing feature artifacts.

### 4. Fit the count-only baseline

    python python-scripts/fit_mle_distributions.py "$DATA_ROOT" \
      --frequency 0.3 \
      --output results/mle_count_only.pkl

This reproduces the original atom-count likelihood: global `A0`, `A100`, `C0`,
`C100`, marginalized common phase, and sinusoidal differential science signal.

### 5. Fit a feature-conditioned likelihood

The preferred feature-conditioned workflow uses local image summaries for local
nuisances:

    p0_i   = A0_i   + 0.5*C0_i*cos(theta_i)
    p100_i = A100_i + 0.5*C100_i*cos(theta_i + dphi_signal_i + dpsi_i)

    dpsi_i = beta_phi @ [s0_i, s100_i]
    A0_i   = A0   + beta_A0   @ s0_i
    A100_i = A100 + beta_A100 @ s100_i
    C0_i   = C0   * exp(beta_C0   @ s0_i)
    C100_i = C100 * exp(beta_C100 @ s100_i)

Here `s0_i` is the standardized selected feature vector from the Z0 artifact and
`s100_i` is the standardized selected feature vector from the Z100 artifact.
This means Z100 centroids/widths do not directly correct Z0 offset or contrast.
The differential phase nuisance uses the paired Z0 and Z100 feature vectors.
The Z0 absolute phase nuisance is still absorbed into the marginalized common
`theta_i`; the fitted term is the feature-predicted differential residual phase.

Phase-only first pass:

    python python-scripts/fit_mle_distributions.py "$DATA_ROOT" \
      --frequency 0.3 \
      --feature-dir-z0 results/2d-shot-features-z0 \
      --feature-dir-z100 results/2d-shot-features-z100 \
      --features-z0 mean_x var_x mean_y var_y \
      --features-z100 mean_x var_x mean_y var_y \
      --feature-nuisance phase \
      --output results/mle_feature_phase_variance.pkl

Full phase + offset + contrast model:

    python python-scripts/fit_mle_distributions.py "$DATA_ROOT" \
      --frequency 0.3 \
      --feature-dir-z0 results/2d-shot-features-z0 \
      --feature-dir-z100 results/2d-shot-features-z100 \
      --features-z0 mean_x std_x cov_x_state cov_y_state \
      --features-z100 mean_x std_x cov_x_state cov_y_state \
      --n-pcs-z0 2 \
      --n-pcs-z100 2 \
      --output results/mle_feature_full.pkl

Useful controls:

- `--feature-nuisance phase`: only fit `beta_phi @ [s0_i, s100_i]` in the differential phase
- `--feature-nuisance phase contrast`: fit phase and contrast, but no offset
- `--feature-nuisance phase offset contrast`: full model, also the default
- `--beta-phi-prior-std`: Gaussian prior std for phase coefficients in radians
- `--beta-A-prior-std`: Gaussian prior std for additive offset coefficients
- `--beta-C-prior-std`: Gaussian prior std for log-contrast coefficients
- `--fast`: useful for smoke tests; use full mode for final distributions
- `--resume`: continue an interrupted run from its output pickle

### 6. Compare MLE distributions and residuals

Open and run:

    notebooks/mle_distributions.ipynb

At the top of the notebook, add the result files you want to compare:

    RESULT_FILES = {
        'count-only': REPO / 'results' / 'mle_count_only.pkl',
        'feature phase': REPO / 'results' / 'mle_feature_phase.pkl',
        'feature full': REPO / 'results' / 'mle_feature_full.pkl',
    }

The notebook shows:

- histograms of repeated-experiment amplitude and phase MLEs
- amplitude-vs-phase scatter
- run-index traces of the estimators
- residuals versus global shot index with a `±1 sigma_shot` filled band
- normalized residual histograms compared to `N(0, 1)`
- residual standard deviation divided by the mean binomial shot-noise floor

The residual diagnostics use compact count artifacts from both sites when they
exist.  If only the Z0 feature artifact exists, only Z0 all-shot residuals are
shown.

