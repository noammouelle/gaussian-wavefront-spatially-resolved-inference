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
