"""Fit Gaussian-envelope plus phase-gradient fringe features.

For each state-resolved image, this script computes physical Gaussian envelope
moments from total counts and fits a contrast fringe model

    contrast(u, v) = offset + a cos(kx u + ky v) + b sin(kx u + ky v)

where u and v are centered/scaled by the fitted final cloud envelope. The fitted
phase, contrast amplitude, and phase-gradient coefficients are then tested as
features for held-out-run initial-condition regression.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


TARGET_NAMES = (
    "mu_x0", "mu_y0", "mu_vx0", "mu_vy0",
    "sigma_x", "sigma_y", "sigma_vx", "sigma_vy",
)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


def downsample_sum(image: np.ndarray, bins: int) -> np.ndarray:
    block = image.shape[0] // bins
    return image.reshape(bins, block, bins, block).sum(axis=(1, 3)).astype(np.float64)


def weighted_lstsq(design: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sw = np.sqrt(np.maximum(weights, 0.0))
    return np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]


def fit_one_shot(ground: np.ndarray, excited: np.ndarray, centers: np.ndarray) -> np.ndarray:
    total = ground + excited
    counts = max(float(total.sum()), 1.0)
    x = centers[:, None]
    y = centers[None, :]
    x2 = x * x
    y2 = y * y
    mean_x = float((total * x).sum() / counts)
    mean_y = float((total * y).sum() / counts)
    var_x = float((total * x2).sum() / counts - mean_x**2)
    var_y = float((total * y2).sum() / counts - mean_y**2)
    std_x = np.sqrt(max(var_x, 1e-300))
    std_y = np.sqrt(max(var_y, 1e-300))
    cov_xy = float((total * x * y).sum() / counts - mean_x * mean_y)

    u = np.broadcast_to((x - mean_x) / std_x, total.shape)
    v = np.broadcast_to((y - mean_y) / std_y, total.shape)
    radius2 = u * u + v * v
    contrast = (excited - ground) / (total + 1.0)
    valid = (total > max(total.max() * 1e-5, 1.0)) & (radius2 < 12.25)
    if valid.sum() < 12:
        return np.full(len(FEATURE_NAMES), np.nan)

    uf = u[valid]
    vf = v[valid]
    cf = contrast[valid]
    wf = total[valid]
    # Compress huge count dynamic range so the central pixels do not entirely
    # silence the phase-gradient information in the shoulders.
    wf_fit = np.sqrt(wf)

    linear_design = np.column_stack([np.ones_like(uf), uf, vf, uf * uf, uf * vf, vf * vf])
    linear_coef = weighted_lstsq(linear_design, cf, wf_fit)
    offset0 = float(linear_coef[0])
    grad_u0 = float(linear_coef[1])
    grad_v0 = float(linear_coef[2])
    curvature = linear_coef[3:]
    centered = cf - np.average(cf, weights=wf_fit)
    amp0 = float(np.sqrt(np.average(centered * centered, weights=wf_fit)) * np.sqrt(2.0))
    amp0 = max(amp0, 1e-3)

    def residual(params: np.ndarray) -> np.ndarray:
        offset, a_cos, b_sin, kx, ky = params
        phase_arg = kx * uf + ky * vf
        model = offset + a_cos * np.cos(phase_arg) + b_sin * np.sin(phase_arg)
        return np.sqrt(wf_fit) * (model - cf)

    # A few cheap starts help avoid the k=0 degeneracy.
    starts = [
        np.array([offset0, amp0, 0.0, 0.0, 0.0]),
        np.array([offset0, amp0, amp0, grad_u0 / amp0, grad_v0 / amp0]),
        np.array([offset0, amp0, -amp0, -grad_u0 / amp0, -grad_v0 / amp0]),
        np.array([offset0, 0.0, amp0, 0.25, 0.0]),
        np.array([offset0, 0.0, amp0, 0.0, 0.25]),
    ]
    bounds = ([-2.0, -2.0, -2.0, -10.0, -10.0], [2.0, 2.0, 2.0, 10.0, 10.0])
    best = None
    for start in starts:
        start = np.clip(start, bounds[0], bounds[1])
        result = least_squares(residual, start, bounds=bounds, max_nfev=100, xtol=1e-5, ftol=1e-5, gtol=1e-5)
        if best is None or result.cost < best.cost:
            best = result
    offset, a_cos, b_sin, kx, ky = best.x
    fringe_amp = float(np.hypot(a_cos, b_sin))
    fringe_phase = float(np.arctan2(-b_sin, a_cos))
    grad_norm = float(np.hypot(kx, ky))
    # Convert normalized-coordinate gradient to physical rad/m.
    kx_phys = float(kx / std_x)
    ky_phys = float(ky / std_y)

    return np.array([
        mean_x, mean_y, std_x, std_y, cov_xy,
        offset0, grad_u0, grad_v0, *curvature,
        offset, a_cos, b_sin, fringe_amp, fringe_phase, np.sin(fringe_phase), np.cos(fringe_phase),
        kx, ky, grad_norm, kx_phys, ky_phys,
    ], dtype=float)


FEATURE_NAMES = (
    "mean_x", "mean_y", "std_x", "std_y", "cov_xy",
    "linear_offset", "linear_grad_u", "linear_grad_v",
    "linear_curv_uu", "linear_curv_uv", "linear_curv_vv",
    "fringe_offset", "fringe_a_cos", "fringe_b_sin", "fringe_amp",
    "fringe_phase", "fringe_sin_phase", "fringe_cos_phase",
    "fringe_k_u", "fringe_k_v", "fringe_k_norm", "fringe_kx_phys", "fringe_ky_phys",
)


def build_features(data_root: Path, port: str, bins: int, max_shots: int | None, output_path: Path) -> None:
    paths = sorted(data_root.glob(f"run_*/{port}/data_IMG.h5"))
    if not paths:
        raise FileNotFoundError(f"No run_*/{port}/data_IMG.h5 below {data_root}")
    features = []
    targets = []
    run_ids = []
    shot_ids = []
    phi0 = []
    started = time.perf_counter()
    for path in paths:
        run_id = path.parents[1].name
        with h5py.File(path, "r") as handle:
            image_res = int(handle["images_s0"].shape[1])
            if image_res % bins:
                raise ValueError(f"image_res={image_res} is not divisible by bins={bins}")
            edges = np.linspace(-float(handle.attrs["image_half_range"]), float(handle.attrs["image_half_range"]), bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for shot_id in range(handle["images_s0"].shape[0]):
                ground = downsample_sum(handle["images_s0"][shot_id], bins)
                excited = downsample_sum(handle["images_s1"][shot_id], bins)
                features.append(fit_one_shot(ground, excited, centers))
                targets.append([float(handle[name][shot_id]) for name in TARGET_NAMES])
                run_ids.append(run_id)
                shot_ids.append(shot_id)
                phi0.append(float(handle["phi0"][shot_id]))
                if len(features) % 100 == 0:
                    logging.info("Fitted %d shots", len(features))
                if max_shots is not None and len(features) >= max_shots:
                    break
        if max_shots is not None and len(features) >= max_shots:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(features),
        feature_names=np.asarray(FEATURE_NAMES),
        targets=np.asarray(targets),
        target_names=np.asarray(TARGET_NAMES),
        run_id=np.asarray(run_ids),
        shot_id=np.asarray(shot_ids, dtype=np.int32),
        phi0=np.asarray(phi0),
        bins=np.asarray(bins),
    )
    logging.info("Wrote %s with %d shots in %.1fs", output_path, len(features), time.perf_counter() - started)


def evaluate(cache_path: Path, degree: int, alpha: float) -> None:
    data = np.load(cache_path)
    X = np.asarray(data["features"], dtype=float)
    y = np.asarray(data["targets"], dtype=float)
    groups = np.asarray(data["run_id"])
    feature_names = [str(name) for name in data["feature_names"]]
    target_names = [str(name) for name in data["target_names"]]
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    y = y[finite]
    groups = groups[finite]

    feature_sets = {
        "envelope": ["mean_x", "mean_y", "std_x", "std_y", "cov_xy"],
        "linear fringe": ["mean_x", "mean_y", "std_x", "std_y", "cov_xy", "linear_offset", "linear_grad_u", "linear_grad_v", "linear_curv_uu", "linear_curv_uv", "linear_curv_vv"],
        "nonlinear fringe": list(FEATURE_NAMES),
    }
    logging.info("Evaluating %d finite shots from %s", len(X), cache_path)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for label, names in feature_sets.items():
        cols = [feature_names.index(name) for name in names]
        Xs = X[:, cols]
        print(f"\n{label} ({Xs.shape[1]} features)")
        print("target       R2_gkf   RMSE_um")
        for target_index, target_name in enumerate(target_names):
            pred = np.empty(len(y), dtype=float)
            for train, test in cv.split(Xs, y[:, target_index], groups):
                steps = [("scale", StandardScaler())]
                if degree > 1:
                    steps += [
                        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                        ("poly_scale", StandardScaler()),
                    ]
                steps.append(("ridge", Ridge(alpha=alpha)))
                model = Pipeline(steps).fit(Xs[train], y[train, target_index])
                pred[test] = model.predict(Xs[test])
            r2 = r2_score(y[:, target_index], pred)
            rmse = np.sqrt(np.mean((y[:, target_index] - pred) ** 2)) * 1e6
            print(f"{target_name:<10} {r2:8.4f} {rmse:9.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/R20_N200_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"),
    )
    parser.add_argument("--port", default="Z0")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/a1e8-gaussian-fringe-z0"))
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir / "gaussian_fringe_features.log")
    cache = args.output_dir / f"features_{args.port}_bins{args.bins}.npz"
    if args.rebuild or not cache.exists():
        build_features(args.data_root, args.port, args.bins, args.max_shots, cache)
    else:
        logging.info("Using existing cache %s", cache)
    evaluate(cache, args.degree, args.alpha)


if __name__ == "__main__":
    main()
