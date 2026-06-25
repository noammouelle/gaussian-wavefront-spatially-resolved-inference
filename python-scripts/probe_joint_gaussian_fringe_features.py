"""Fit joint Gaussian-envelope plus linear-fringe image features.

The fitted state-resolved model is

    s0(x, y) = 0.5 * A(x, y) * (1 - C cos(phi0 + kx x + ky y))
    s1(x, y) = 0.5 * A(x, y) * (1 + C cos(phi0 + kx x + ky y))

with A a separable 2D Gaussian envelope. The fitted envelope center, widths,
contrast, phase offset, and phase-gradient coefficients are then evaluated as
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

FEATURE_NAMES = (
    "moment_mu_x", "moment_mu_y", "moment_std_x", "moment_std_y",
    "fit_log_amp", "fit_mu_x", "fit_mu_y", "fit_std_x", "fit_std_y",
    "fit_contrast", "fit_phi", "fit_sin_phi", "fit_cos_phi",
    "fit_kx", "fit_ky", "fit_k_norm", "fit_cost", "fit_success",
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


def weighted_moments(total: np.ndarray, centers: np.ndarray) -> tuple[float, float, float, float]:
    counts = max(float(total.sum()), 1.0)
    x = centers[:, None]
    y = centers[None, :]
    mu_x = float((total * x).sum() / counts)
    mu_y = float((total * y).sum() / counts)
    var_x = float((total * x * x).sum() / counts - mu_x**2)
    var_y = float((total * y * y).sum() / counts - mu_y**2)
    return mu_x, mu_y, np.sqrt(max(var_x, 1e-300)), np.sqrt(max(var_y, 1e-300))


def fit_one_shot(ground: np.ndarray, excited: np.ndarray, centers: np.ndarray) -> np.ndarray:
    total = ground + excited
    mu_x0, mu_y0, std_x0, std_y0 = weighted_moments(total, centers)
    x = np.broadcast_to(centers[:, None], total.shape)
    y = np.broadcast_to(centers[None, :], total.shape)
    radius2 = ((x - mu_x0) / std_x0) ** 2 + ((y - mu_y0) / std_y0) ** 2
    valid = (total > max(total.max() * 1e-5, 1.0)) & (radius2 < 16.0)
    if valid.sum() < len(FEATURE_NAMES):
        return np.full(len(FEATURE_NAMES), np.nan)

    xf = x[valid]
    yf = y[valid]
    g = ground[valid]
    e = excited[valid]
    tf = total[valid]
    amp0 = max(float(total.max()), 1.0)
    contrast_img = (e - g) / (tf + 1.0)
    c0 = float(np.clip(np.percentile(np.abs(contrast_img), 90) * np.sqrt(2.0), 1e-3, 0.95))

    def unpack(params: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float]:
        log_amp, mu_x, mu_y, log_std_x, log_std_y, raw_c, phi, raw_kx, raw_ky = params
        amp = np.exp(log_amp)
        std_x = np.exp(log_std_x)
        std_y = np.exp(log_std_y)
        contrast = 0.999 * np.tanh(raw_c)
        kx = raw_kx / std_x0
        ky = raw_ky / std_y0
        return amp, mu_x, mu_y, std_x, std_y, contrast, phi, kx, ky

    def residual(params: np.ndarray) -> np.ndarray:
        amp, mu_x, mu_y, std_x, std_y, contrast, phi, kx, ky = unpack(params)
        envelope = amp * np.exp(-0.5 * (((xf - mu_x) / std_x) ** 2 + ((yf - mu_y) / std_y) ** 2))
        fringe = contrast * np.cos(phi + kx * xf + ky * yf)
        pred_g = 0.5 * envelope * (1.0 - fringe)
        pred_e = 0.5 * envelope * (1.0 + fringe)
        # Poisson-like Pearson residuals, softened so empty/noisy shoulder pixels
        # do not dominate the nonlinear fit.
        return np.r_[
            (pred_g - g) / np.sqrt(pred_g + 4.0),
            (pred_e - e) / np.sqrt(pred_e + 4.0),
        ]

    base = np.array([
        np.log(amp0), mu_x0, mu_y0, np.log(std_x0), np.log(std_y0),
        np.arctanh(np.clip(c0 / 0.999, -0.99, 0.99)), 0.0, 0.0, 0.0,
    ])
    starts = [
        base,
        base + np.array([0, 0, 0, 0, 0, 0, np.pi / 2, 0.25, 0.0]),
        base + np.array([0, 0, 0, 0, 0, 0, -np.pi / 2, -0.25, 0.0]),
        base + np.array([0, 0, 0, 0, 0, 0, np.pi, 0.0, 0.25]),
        base + np.array([0, 0, 0, 0, 0, 0, -np.pi, 0.0, -0.25]),
    ]
    lower = np.array([
        np.log(amp0 * 0.05), mu_x0 - 4 * std_x0, mu_y0 - 4 * std_y0,
        np.log(std_x0 * 0.25), np.log(std_y0 * 0.25),
        -4.0, -4 * np.pi, -10.0, -10.0,
    ])
    upper = np.array([
        np.log(amp0 * 20.0), mu_x0 + 4 * std_x0, mu_y0 + 4 * std_y0,
        np.log(std_x0 * 4.0), np.log(std_y0 * 4.0),
        4.0, 4 * np.pi, 10.0, 10.0,
    ])

    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=120,
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
        )
        if best is None or result.cost < best.cost:
            best = result

    amp, mu_x, mu_y, std_x, std_y, contrast, phi, kx, ky = unpack(best.x)
    return np.array([
        mu_x0, mu_y0, std_x0, std_y0,
        np.log(amp), mu_x, mu_y, std_x, std_y,
        contrast, phi, np.sin(phi), np.cos(phi),
        kx, ky, np.hypot(kx, ky), best.cost / max(valid.sum(), 1), float(best.success),
    ], dtype=float)


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
                features.append(fit_one_shot(
                    downsample_sum(handle["images_s0"][shot_id], bins),
                    downsample_sum(handle["images_s1"][shot_id], bins),
                    centers,
                ))
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
        "moments": ["moment_mu_x", "moment_mu_y", "moment_std_x", "moment_std_y"],
        "joint gaussian": [
            "fit_mu_x", "fit_mu_y", "fit_std_x", "fit_std_y",
            "fit_contrast", "fit_sin_phi", "fit_cos_phi",
            "fit_kx", "fit_ky", "fit_k_norm", "fit_cost",
        ],
        "moments + joint gaussian": list(FEATURE_NAMES),
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/a1e8-joint-gaussian-fringe-z0"))
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir / "joint_gaussian_fringe_features.log")
    cache = args.output_dir / f"features_{args.port}_bins{args.bins}.npz"
    if args.rebuild or not cache.exists():
        build_features(args.data_root, args.port, args.bins, args.max_shots, cache)
    else:
        logging.info("Using existing cache %s", cache)
    evaluate(cache, args.degree, args.alpha)


if __name__ == "__main__":
    main()
