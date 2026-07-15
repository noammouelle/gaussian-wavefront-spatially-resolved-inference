# Eta learnability from the simulated images

This study tests whether the response-only Regime-II structure is usable by
compact supervised image inference. Every cloud nuisance realization is unique
per shot, so the primary deterministic split is by shot. PCA and all scalers are
fit on the training split only. A run-grouped split remains available as an
ablation because the injected differential signal has run structure.

The image representation follows the existing `shot_feature_pipeline`: each
2048×2048 state image is recentered and rescaled by its measured final cloud
mean and width, then converted to a cloud-normalized state-contrast map. The
benchmarks are ballistic final-image summaries, training-only contrast PCA,
PCA conditioned on true total phase through sine/cosine interactions, paired
Z0/Z100 PCA, and supervised PLS on the full compact contrast profile. Total
phase is predicted separately from normalized shape, total count, global state
fraction, and their combination so brightness cannot be confused with shape.

## Current results

The completed snapshot contains 1,000 sampled pairs from the 10^6-atom data
and 600 completed pairs from the still-growing 10^8-atom directory.

Final-position moments learn `mu_vx0` with R2 about 0.93 and `sigma_vx` with
R2 above 0.99 because the 3.8 s expansion makes velocity dominate final mean
and width. This is ballistic information, not evidence of fibre breaking.
`mu_x0`, `sigma_x`, and the standardized fibre coordinate proportional to
`mu_vx0 - t_det mu_x0` remain at approximately prior-level RMSE. Leading PCA,
known-phase PCA interactions, paired-port PCA, and supervised PLS do not
produce a stable held-out improvement for that hidden coordinate.

Total phase is nevertheless highly learnable from normalized shape. Shape-only
PCA gives about 0.049–0.057 rad RMS in the 10^6 sample and 0.057–0.064 rad in
the current 10^8 sample. Total count alone gives 1.82–1.97 rad and therefore
does not predict phase. Global state fraction alone gives only 0.89–1.03 rad.
Combining normalized shape with brightness/global contrast gives about
0.046–0.047 rad in the 10^8 sample. These are typical random-phase errors, not
worst-orbit guarantees.

Thus Regime II should be read as "nonzero response-level nuisance information
exists after ballistic projection," not "all eta coordinates are easily
recoverable by leading PCs." The response Fisher calculation identified a
small local mode; the simulation experiment shows that compact global
regressors do not yet exploit it. The next justified escalation is a targeted
conditional likelihood/posterior model for the one-dimensional hidden mode,
using the known forward model and marginalizing phase, rather than a larger
generic image regressor.

Artifacts are in `results/eta_learnability`. Rerunning the command with the same
label reuses the compact cache; a new label takes a fresh snapshot as generation
continues.
