"""Train a small supervised image MLP for initial-condition features.

The available environment does not include a CNN framework, so this script uses
scikit-learn's MLPRegressor on downsampled physical-coordinate image channels.
It logs every epoch and saves held-out predictions plus bottleneck features.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
import warnings

import h5py
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


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
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def downsample_sum(image: np.ndarray, bins: int) -> np.ndarray:
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected square image, got {image.shape}")
    if image.shape[0] % bins:
        raise ValueError(f"Image resolution {image.shape[0]} is not divisible by bins={bins}")
    block = image.shape[0] // bins
    return image.reshape(bins, block, bins, block).sum(axis=(1, 3)).astype(np.float32)


def build_cache(data_root: Path, port: str, bins: int, max_shots: int | None, output_path: Path) -> None:
    paths = sorted(data_root.glob(f"run_*/{port}/data_IMG.h5"))
    if not paths:
        raise FileNotFoundError(f"No run_*/{port}/data_IMG.h5 files below {data_root}")

    logging.info("Building image cache from %d HDF5 files", len(paths))
    x_rows = []
    y_rows = []
    run_ids = []
    shot_ids = []
    phi0_rows = []
    started = time.perf_counter()

    for path in paths:
        run_id = path.parents[1].name
        with h5py.File(path, "r") as handle:
            n_shots = int(handle["images_s0"].shape[0])
            for shot_id in range(n_shots):
                ground = downsample_sum(handle["images_s0"][shot_id], bins)
                excited = downsample_sum(handle["images_s1"][shot_id], bins)
                total = ground + excited
                total_sum = max(float(total.sum()), 1.0)
                density = total / total_sum
                contrast = (excited - ground) / (total + 1.0)
                x_rows.append(np.concatenate([density.ravel(), contrast.ravel()]))
                y_rows.append([float(handle[name][shot_id]) for name in TARGET_NAMES])
                run_ids.append(run_id)
                shot_ids.append(shot_id)
                phi0_rows.append(float(handle["phi0"][shot_id]))
                if len(x_rows) % 200 == 0:
                    logging.info("Cached %d shots", len(x_rows))
                if max_shots is not None and len(x_rows) >= max_shots:
                    break
        if max_shots is not None and len(x_rows) >= max_shots:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.float64),
        run_id=np.asarray(run_ids),
        shot_id=np.asarray(shot_ids, dtype=np.int32),
        phi0=np.asarray(phi0_rows, dtype=np.float64),
        target_names=np.asarray(TARGET_NAMES),
        bins=np.asarray(bins),
        port=np.asarray(port),
        data_root=np.asarray(str(data_root)),
    )
    logging.info(
        "Wrote cache %s with %d shots in %.1fs",
        output_path,
        len(x_rows),
        time.perf_counter() - started,
    )


def split_by_run(groups: np.ndarray, seed: int, test_fraction: float, val_fraction: float):
    indices = np.arange(len(groups))
    train_val, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
            indices, groups=groups
        )
    )
    rel_val_fraction = val_fraction / (1.0 - test_fraction)
    train_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=rel_val_fraction, random_state=seed + 1).split(
            train_val, groups=groups[train_val]
        )
    )
    train = train_val[train_rel]
    val = train_val[val_rel]
    return train, val, test


def rmse_um(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)) * 1e6


def r2_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])


def hidden_activations(model: MLPRegressor, X_scaled: np.ndarray) -> np.ndarray:
    activations = X_scaled
    for layer_index, (weights, bias) in enumerate(zip(model.coefs_[:-1], model.intercepts_[:-1])):
        activations = activations @ weights + bias
        if model.activation == "relu":
            activations = np.maximum(activations, 0)
        elif model.activation == "tanh":
            activations = np.tanh(activations)
        elif model.activation == "logistic":
            activations = 1.0 / (1.0 + np.exp(-activations))
        elif model.activation != "identity":
            raise ValueError(f"Unsupported activation for feature extraction: {model.activation}")
        logging.debug("Hidden layer %d activations shape: %s", layer_index, activations.shape)
    return activations


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/R20_N200_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"),
    )
    parser.add_argument("--port", default="Z0")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/a1e8-image-mlp-z0"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 16])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.log"
    setup_logging(log_path)
    logging.info("Arguments: %s", vars(args))

    cache_path = args.cache or args.output_dir / f"cache_{args.port}_bins{args.bins}.npz"
    if not cache_path.exists():
        build_cache(args.data_root, args.port, args.bins, args.max_shots, cache_path)
    else:
        logging.info("Using existing cache: %s", cache_path)

    cache = np.load(cache_path)
    X = np.asarray(cache["X"], dtype=np.float32)
    y = np.asarray(cache["y"], dtype=np.float64)
    groups = np.asarray(cache["run_id"])
    target_names = [str(name) for name in cache["target_names"]]
    train, val, test = split_by_run(groups, args.seed, args.test_fraction, args.val_fraction)
    logging.info(
        "Loaded X=%s y=%s; train/val/test shots=%d/%d/%d; runs=%d/%d/%d",
        X.shape,
        y.shape,
        len(train),
        len(val),
        len(test),
        len(np.unique(groups[train])),
        len(np.unique(groups[val])),
        len(np.unique(groups[test])),
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X[train])
    X_val = x_scaler.transform(X[val])
    X_test = x_scaler.transform(X[test])
    y_train = y_scaler.fit_transform(y[train])
    y_val = y_scaler.transform(y[val])

    model = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden),
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.learning_rate_init,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=args.seed,
        early_stopping=False,
        verbose=False,
    )

    history = []
    best_val = np.inf
    best_state = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        train_loss = float(np.mean((pred_train - y_train) ** 2))
        val_loss = float(np.mean((pred_val - y_val) ** 2))
        val_pred_phys = y_scaler.inverse_transform(pred_val)
        val_r2 = r2_columns(y[val], val_pred_phys)
        val_rmse = rmse_um(y[val], val_pred_phys)
        history.append({
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "val_mse_scaled": val_loss,
            **{f"val_r2_{name}": float(val_r2[i]) for i, name in enumerate(target_names)},
            **{f"val_rmse_um_{name}": float(val_rmse[i]) for i, name in enumerate(target_names)},
        })
        logging.info(
            "epoch=%03d train_mse=%.6f val_mse=%.6f "
            "val_R2(mu_x0,mu_vx0,sigma_vx)=%.4f %.4f %.4f "
            "val_RMSE_um(mu_x0,mu_vx0,sigma_vx)=%.3f %.3f %.3f",
            epoch,
            train_loss,
            val_loss,
            val_r2[target_names.index("mu_x0")],
            val_r2[target_names.index("mu_vx0")],
            val_r2[target_names.index("sigma_vx")],
            val_rmse[target_names.index("mu_x0")],
            val_rmse[target_names.index("mu_vx0")],
            val_rmse[target_names.index("sigma_vx")],
        )
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            best_state = {
                "coefs": [coef.copy() for coef in model.coefs_],
                "intercepts": [intercept.copy() for intercept in model.intercepts_],
                "n_iter": model.n_iter_,
                "loss": model.loss_,
            }
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logging.info("Early stopping after %d stale epochs", stale_epochs)
                break

    if best_state is not None:
        model.coefs_ = best_state["coefs"]
        model.intercepts_ = best_state["intercepts"]
        model.n_iter_ = best_state["n_iter"]
        model.loss_ = best_state["loss"]

    X_all_scaled = x_scaler.transform(X)
    pred_all = y_scaler.inverse_transform(model.predict(X_all_scaled))
    latent_all = hidden_activations(model, X_all_scaled)
    metrics = {}
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        split_r2 = r2_columns(y[split], pred_all[split])
        split_rmse = rmse_um(y[split], pred_all[split])
        metrics[split_name] = {
            name: {"r2": float(split_r2[i]), "rmse_um": float(split_rmse[i])}
            for i, name in enumerate(target_names)
        }
        logging.info("%s metrics:", split_name)
        for i, name in enumerate(target_names):
            logging.info(
                "  %-8s R2=% .4f RMSE=%.3f um",
                name,
                split_r2[i],
                split_rmse[i],
            )

    np.savez_compressed(
        args.output_dir / "image_mlp_features.npz",
        features=latent_all,
        predictions=pred_all,
        targets=y,
        run_id=groups,
        shot_id=cache["shot_id"],
        phi0=cache["phi0"],
        target_names=np.asarray(target_names),
        train_index=train,
        val_index=val,
        test_index=test,
        input_mean=x_scaler.mean_,
        input_scale=x_scaler.scale_,
        target_mean=y_scaler.mean_,
        target_scale=y_scaler.scale_,
    )
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "history": history,
                "metrics": metrics,
                "elapsed_seconds": time.perf_counter() - started,
            },
            handle,
            indent=2,
        )
    logging.info("Saved features and predictions to %s", args.output_dir / "image_mlp_features.npz")
    logging.info("Saved metrics to %s", args.output_dir / "metrics.json")
    logging.info("Training log: %s", log_path)


if __name__ == "__main__":
    main()
