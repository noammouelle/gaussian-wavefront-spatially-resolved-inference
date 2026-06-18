"""
Compare three feature sets for predicting cloud initial conditions.

Baseline : original 5 summary features + 5 PCA scores
Approach1: add 6 contrast-moment features (cov_x_state, cov_y_state, etc.)
Approach2: add 2 phase-map gradient features (gx_fit, gy_fit from per-shot linear phase fit)

Prints R² and RMSE tables for all 8 targets.
"""

import sys
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "helpers"))
from helpers import ImageShotDataset  # noqa: E402
ARTIFACT_DIR = REPO / "results" / "2d-shot-features"
DATA_ROOT = REPO / "data" / (
    "R80_N50_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx309um"
    "_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"
)

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_contrast_moments(img_path):
    """Return (n_shots, 6) array of contrast-moment summary features from a data_IMG.h5 file."""
    ds = ImageShotDataset(img_path)
    pc = ds.pixel_centers

    with h5py.File(img_path, "r") as f:
        imgs_s0 = f["images_s0"][:].astype(np.float64)  # (n_shots, res, res)
        imgs_s1 = f["images_s1"][:].astype(np.float64)

    total = imgs_s0 + imgs_s1
    counts = total.sum(axis=(1, 2))
    safe_counts = np.where(counts > 0, counts, 1.0)

    pc2 = pc ** 2
    mean_x = (total * pc[None, :, None]).sum(axis=(1, 2)) / safe_counts
    mean_y = (total * pc[None, None, :]).sum(axis=(1, 2)) / safe_counts
    var_x  = (total * pc2[None, :, None]).sum(axis=(1, 2)) / safe_counts - mean_x ** 2
    var_y  = (total * pc2[None, None, :]).sum(axis=(1, 2)) / safe_counts - mean_y ** 2

    n1 = imgs_s1
    exc_total = n1.sum(axis=(1, 2))
    mean_s = exc_total / safe_counts
    safe_exc = np.where(exc_total > 0, exc_total, 1.0)

    cov_x_state  = (n1 * pc[None, :, None]).sum(axis=(1, 2)) / safe_counts - mean_x * mean_s
    cov_y_state  = (n1 * pc[None, None, :]).sum(axis=(1, 2)) / safe_counts - mean_y * mean_s
    cov_x2_state = ((n1 * pc2[None, :, None]).sum(axis=(1, 2)) / safe_counts
                    - (var_x + mean_x ** 2) * mean_s)
    cov_y2_state = ((n1 * pc2[None, None, :]).sum(axis=(1, 2)) / safe_counts
                    - (var_y + mean_y ** 2) * mean_s)
    mean_x_exc   = (n1 * pc[None, :, None]).sum(axis=(1, 2)) / safe_exc
    mean_y_exc   = (n1 * pc[None, None, :]).sum(axis=(1, 2)) / safe_exc

    return np.column_stack([
        cov_x_state, cov_y_state, cov_x2_state, cov_y2_state,
        mean_x_exc, mean_y_exc,
    ])


def fit_phase_gradients(ground, excited, valid_pixels, x_edges, y_edges):
    """
    Per-shot linear fit of the excitation fraction map.
    Returns (n_shots, 2) array of [gx_fit, gy_fit].

    Model: f(x,y) - 0.5 ≈ gx*x + gy*y  (weighted by atom count)
    This captures the dominant phase-gradient direction which breaks the
    mu_x0 / mu_vx0 degeneracy that mean_x alone cannot resolve.
    """
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    xg, yg = np.meshgrid(x_c, y_c, indexing="ij")

    xv = xg[valid_pixels]
    yv = yg[valid_pixels]
    # Design matrix: [x, y] (no intercept — we already subtracted 0.5)
    A = np.column_stack([xv, yv])  # (n_valid_px, 2)

    total    = (ground + excited).astype(np.float64)
    fraction = (excited + 0.5) / (total + 1.0)

    fv = fraction[:, valid_pixels] - 0.5  # (n_shots, n_valid_px)
    wv = total[:, valid_pixels]            # (n_shots, n_valid_px)

    gradients = np.zeros((fv.shape[0], 2))
    for i, (fi, wi) in enumerate(zip(fv, wv)):
        Aw = A * wi[:, None]
        gradients[i] = np.linalg.lstsq(Aw, fi * wi, rcond=None)[0]

    return gradients


def quadratic_ridge():
    return Pipeline([
        ("input_scaler",      StandardScaler()),
        ("quadratic_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("quadratic_scaler",  StandardScaler()),
        ("ridge",             Ridge(alpha=1e-2)),
    ])


def run_regression(X, targets, label):
    rows = []
    for name, y in targets.items():
        model = clone(quadratic_ridge()).fit(X, y)
        pred  = model.predict(X)
        rows.append({
            "target": name,
            "R2":     r2_score(y, pred),
            "RMSE":   float(np.sqrt(np.mean((y - pred)**2))),
        })
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'target':<14}  {'R²':>7}  {'RMSE':>12}")
    print(f"  {'-'*38}")
    for r in rows:
        print(f"  {r['target']:<14}  {r['R2']:>7.4f}  {r['RMSE']:>12.4e}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading existing artifacts …")
    pca_artifact = np.load(ARTIFACT_DIR / "2d-shot-pcas.npz")
    metadata     = np.load(ARTIFACT_DIR / "shot_metadata.npz")

    summary_features = metadata["summary_features"]   # (4000, 5) — original
    pca_scores       = pca_artifact["scores"][:, :5]  # (4000, 5)
    valid_pixels     = pca_artifact["valid_pixels"]
    x_edges          = pca_artifact["x_edges"]
    y_edges          = pca_artifact["y_edges"]

    targets = {
        "mu_x0":     metadata["mu_x0"],
        "mu_y0":     metadata["mu_y0"],
        "sigma_x0":  metadata["sigma_x"],
        "sigma_y0":  metadata["sigma_y"],
        "mu_vx0":    metadata["mu_vx0"],
        "mu_vy0":    metadata["mu_vy0"],
        "sigma_vx0": metadata["sigma_vx"],
        "sigma_vy0": metadata["sigma_vy"],
    }

    # ── Baseline ──────────────────────────────────────────────────────────────
    X_base = np.column_stack([summary_features, pca_scores])
    base_rows = run_regression(X_base, targets, "BASELINE  (5 summary + 5 PCA)")

    # ── Approach 1: contrast moments ──────────────────────────────────────────
    print("\nExtracting contrast-moment features from HDF5 files …")
    run_dirs = sorted(DATA_ROOT.glob("run_*/Z0/data_IMG.h5"))
    if not run_dirs:
        print(f"ERROR: no HDF5 files found under {DATA_ROOT}")
        sys.exit(1)

    contrast_chunks = []
    for i, p in enumerate(run_dirs):
        contrast_chunks.append(extract_contrast_moments(p))
        print(f"  [{i+1:3d}/{len(run_dirs)}] {p.parent.parent.name}", end="\r")
    print()
    contrast_features = np.concatenate(contrast_chunks, axis=0)  # (4000, 6)
    print(f"  Contrast features shape: {contrast_features.shape}")

    X_a1 = np.column_stack([summary_features, pca_scores, contrast_features])
    a1_rows = run_regression(X_a1, targets, "APPROACH 1  (+ contrast moments)")

    # ── Approach 2: phase-map gradient ────────────────────────────────────────
    print("\nFitting per-shot phase-map gradients …")
    ground  = np.load(ARTIFACT_DIR / "ground_counts.npy",  mmap_mode="r")
    excited = np.load(ARTIFACT_DIR / "excited_counts.npy", mmap_mode="r")
    phase_grads = fit_phase_gradients(ground, excited, valid_pixels, x_edges, y_edges)
    print(f"  Phase gradient features shape: {phase_grads.shape}")

    X_a2 = np.column_stack([summary_features, pca_scores, phase_grads])
    a2_rows = run_regression(X_a2, targets, "APPROACH 2  (+ phase-map gradient)")

    # ── Combined ──────────────────────────────────────────────────────────────
    X_both = np.column_stack([summary_features, pca_scores, contrast_features, phase_grads])
    both_rows = run_regression(X_both, targets, "COMBINED    (+ both)")

    # ── Delta table ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  R² DELTA vs BASELINE  (positive = improvement)")
    print(f"{'='*60}")
    print(f"  {'target':<14}  {'A1 delta':>10}  {'A2 delta':>10}  {'Both delta':>10}")
    print(f"  {'-'*48}")
    base_r2 = {r["target"]: r["R2"] for r in base_rows}
    a1_r2   = {r["target"]: r["R2"] for r in a1_rows}
    a2_r2   = {r["target"]: r["R2"] for r in a2_rows}
    both_r2 = {r["target"]: r["R2"] for r in both_rows}
    for name in targets:
        d1   = a1_r2[name]   - base_r2[name]
        d2   = a2_r2[name]   - base_r2[name]
        both = both_r2[name] - base_r2[name]
        print(f"  {name:<14}  {d1:>+10.4f}  {d2:>+10.4f}  {both:>+10.4f}")


if __name__ == "__main__":
    main()
