"""Probe compact feature families for initial-condition regression.

This script is intentionally lightweight: it consumes existing 2D shot-feature
artifacts and reports held-out regression scores for candidate summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


TARGET_NAMES = (
    "mu_x0", "mu_y0", "mu_vx0", "mu_vy0",
    "sigma_x", "sigma_y", "sigma_vx", "sigma_vy",
)


def make_model(degree: int, alpha: float) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if degree > 1:
        steps += [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("poly_scale", StandardScaler()),
        ]
    steps.append(("ridge", Ridge(alpha=alpha)))
    return Pipeline(steps)


def load_artifact(feature_dir: Path, n_pcs: int) -> dict[str, np.ndarray]:
    meta = np.load(feature_dir / "shot_metadata.npz")
    pca = np.load(feature_dir / "2d-shot-pcas.npz")
    ground = np.load(feature_dir / "ground_counts.npy", mmap_mode="r")
    excited = np.load(feature_dir / "excited_counts.npy", mmap_mode="r")
    names = [str(name) for name in meta["summary_names"]]
    summaries = np.asarray(meta["summary_features"], dtype=float)
    scores = np.asarray(pca["scores"][:, :n_pcs], dtype=float)
    targets = {name: np.asarray(meta[name], dtype=float) for name in TARGET_NAMES}
    return {
        "summary_names": np.asarray(names),
        "summaries": summaries,
        "scores": scores,
        "ground": ground,
        "excited": excited,
        "valid_pixels": pca["valid_pixels"],
        "x_edges": pca["x_edges"],
        "y_edges": pca["y_edges"],
        "targets": targets,
    }


def summary_columns(artifact: dict[str, np.ndarray], wanted: list[str]) -> np.ndarray:
    names = list(artifact["summary_names"])
    indices = [names.index(name) for name in wanted if name in names]
    if not indices:
        return np.empty((artifact["summaries"].shape[0], 0))
    return artifact["summaries"][:, indices]


def weighted_design_terms(xv: np.ndarray, yv: np.ndarray, degree: int) -> tuple[np.ndarray, list[str]]:
    terms = [np.ones_like(xv)]
    names = ["c0"]
    for total_degree in range(1, degree + 1):
        for px in range(total_degree, -1, -1):
            py = total_degree - px
            terms.append((xv ** px) * (yv ** py))
            names.append(f"x{px}y{py}")
    return np.column_stack(terms), names


def fraction_poly_features(
    ground: np.ndarray,
    excited: np.ndarray,
    valid_pixels: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    degree: int = 3,
) -> tuple[np.ndarray, list[str]]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xg, yg = np.meshgrid(x_centers, y_centers, indexing="ij")
    xv = xg[valid_pixels]
    yv = yg[valid_pixels]
    design, names = weighted_design_terms(xv, yv, degree)

    total = (ground + excited).astype(float)
    fraction = (excited + 0.5) / (total + 1.0)
    fv = fraction[:, valid_pixels]
    wv = total[:, valid_pixels]
    coefs = np.empty((fv.shape[0], design.shape[1]), dtype=float)
    for i in range(fv.shape[0]):
        weights = np.sqrt(np.maximum(wv[i], 0.0))
        aw = design * weights[:, None]
        bw = fv[i] * weights
        coefs[i] = np.linalg.lstsq(aw, bw, rcond=None)[0]
    return coefs, [f"frac_poly_{name}" for name in names]


def contrast_moment_features(
    ground: np.ndarray,
    excited: np.ndarray,
    valid_pixels: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    max_order: int = 4,
) -> tuple[np.ndarray, list[str]]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xg, yg = np.meshgrid(x_centers, y_centers, indexing="ij")
    xv = xg[valid_pixels]
    yv = yg[valid_pixels]

    total = (ground + excited).astype(float)[:, valid_pixels]
    contrast = (excited.astype(float) - ground.astype(float))[:, valid_pixels] / (total + 1.0)
    weight_sum = np.maximum(total.sum(axis=1), 1.0)
    global_contrast = (total * contrast).sum(axis=1, keepdims=True) / weight_sum[:, None]
    centered_contrast = contrast - global_contrast

    columns = []
    names = []
    for order in range(1, max_order + 1):
        for px in range(order, -1, -1):
            py = order - px
            basis = (xv ** px) * (yv ** py)
            columns.append((total * centered_contrast * basis[None, :]).sum(axis=1) / weight_sum)
            names.append(f"contrast_moment_x{px}y{py}")
    return np.column_stack(columns), names


def cv_scores(X: np.ndarray, targets: dict[str, np.ndarray], degree: int, alpha: float, folds: int, seed: int):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, y in targets.items():
        pred = np.empty_like(y, dtype=float)
        for train, test in splitter.split(X):
            model = clone(make_model(degree, alpha))
            model.fit(X[train], y[train])
            pred[test] = model.predict(X[test])
        rows.append((name, r2_score(y, pred), float(np.sqrt(np.mean((y - pred) ** 2)))))
    return rows


def print_table(label: str, rows: list[tuple[str, float, float]]) -> None:
    print(f"\n{label}")
    print("target          R2_cv      RMSE_um")
    print("----------------------------------")
    for name, r2, rmse in rows:
        print(f"{name:<12} {r2:8.4f} {rmse * 1e6:11.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    art = load_artifact(args.feature_dir, args.n_pcs)
    base_names = ["mean_x", "mean_y", "std_x", "std_y", "cov_xy"]
    state_names = [
        "cov_x_state", "cov_y_state", "cov_x2_state", "cov_y2_state",
        "mean_x_exc", "mean_y_exc",
    ]

    base = summary_columns(art, base_names)
    state = summary_columns(art, state_names)
    pcs = art["scores"]
    poly, _ = fraction_poly_features(
        art["ground"], art["excited"], art["valid_pixels"], art["x_edges"], art["y_edges"], degree=3
    )
    moments, _ = contrast_moment_features(
        art["ground"], art["excited"], art["valid_pixels"], art["x_edges"], art["y_edges"], max_order=4
    )

    feature_sets = {
        "final moments": base,
        "final moments + PCA": np.column_stack([base, pcs]),
        "final moments + state moments": np.column_stack([base, state]),
        "final moments + fraction poly": np.column_stack([base, poly]),
        "final moments + contrast moments": np.column_stack([base, moments]),
        "all compact candidates": np.column_stack([base, state, pcs, poly, moments]),
    }
    print(f"feature_dir={args.feature_dir}")
    print(f"n_shots={base.shape[0]} n_pcs={pcs.shape[1]} model_degree={args.degree}")
    for label, X in feature_sets.items():
        print_table(label, cv_scores(X, art["targets"], args.degree, args.alpha, args.folds, args.seed))


if __name__ == "__main__":
    main()
