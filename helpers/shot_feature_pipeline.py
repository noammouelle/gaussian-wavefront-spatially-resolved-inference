"""Streaming 2D shot-feature extraction and compact PCA artifacts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import h5py
import numpy as np
from numpy.lib.format import open_memmap


METADATA_KEYS = (
    "mu_x0",
    "mu_y0",
    "mu_vx0",
    "mu_vy0",
    "sigma_x",
    "sigma_y",
    "sigma_vx",
    "sigma_vy",
)
SUMMARY_NAMES = ("mean_x", "mean_y", "std_x", "std_y", "cov_xy")


@dataclass(frozen=True)
class ExtractionConfig:
    bins: int = 64
    xy_min: float = -5e-3
    xy_max: float = 5e-3
    workers: int = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def discover_run_paths(data_root, port="Z0", max_runs=None):
    paths = sorted(Path(data_root).glob(f"run_*/{port}/data_PROB.h5"))
    if max_runs is not None:
        paths = paths[:max_runs]
    if not paths:
        raise FileNotFoundError(f"No run_*/{port}/data_PROB.h5 files below {data_root}")
    return paths


def inspect_runs(paths):
    records = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            missing = [key for key in METADATA_KEYS if key not in handle]
            if missing:
                raise KeyError(f"{path} is missing metadata datasets: {missing}")
            records.append(
                {
                    "path": str(path.resolve()),
                    "run_id": path.parents[1].name,
                    "n_shots": int(handle["phi0"].shape[0]),
                    "n_atoms": int(handle["states"].shape[0]),
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
    return records


def _bincount_by_shot(shot_index, values, n_shots):
    return np.bincount(shot_index, weights=values, minlength=n_shots)


def extract_run(path, bins, xy_min, xy_max):
    """Return compact arrays for one HDF5 run."""
    path = Path(path)
    started = time.perf_counter()
    edges = np.linspace(xy_min, xy_max, bins + 1)
    n_pixels = bins * bins

    with h5py.File(path, "r") as handle:
        shot_index = handle["shot_index"][:]
        positions = handle["positions"][:, :2]
        states = handle["states"][:].astype(bool, copy=False)
        n_shots = int(handle["phi0"].shape[0])

        x = positions[:, 0]
        y = positions[:, 1]
        ix = (np.searchsorted(edges, x, side="right") - 1).astype(np.int16)
        iy = (np.searchsorted(edges, y, side="right") - 1).astype(np.int16)
        inside = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
        flat_pixel = (ix[inside] * bins + iy[inside]).astype(np.int32)
        flat_shot_pixel = shot_index[inside].astype(np.int64) * n_pixels + flat_pixel
        inside_states = states[inside]

        ground = np.bincount(
            flat_shot_pixel[~inside_states], minlength=n_shots * n_pixels
        ).reshape(n_shots, bins, bins)
        excited = np.bincount(
            flat_shot_pixel[inside_states], minlength=n_shots * n_pixels
        ).reshape(n_shots, bins, bins)
        if max(ground.max(initial=0), excited.max(initial=0)) > np.iinfo(np.uint32).max:
            raise OverflowError(f"Pixel count exceeds uint32 range in {path}")

        counts = np.bincount(shot_index, minlength=n_shots)
        mean_x = _bincount_by_shot(shot_index, x, n_shots) / counts
        mean_y = _bincount_by_shot(shot_index, y, n_shots) / counts
        var_x = _bincount_by_shot(shot_index, x * x, n_shots) / counts - mean_x**2
        var_y = _bincount_by_shot(shot_index, y * y, n_shots) / counts - mean_y**2
        cov_xy = (
            _bincount_by_shot(shot_index, x * y, n_shots) / counts - mean_x * mean_y
        )
        summaries = np.column_stack(
            [mean_x, mean_y, np.sqrt(np.maximum(var_x, 0)), np.sqrt(np.maximum(var_y, 0)), cov_xy]
        )
        metadata = {key: handle[key][:] for key in METADATA_KEYS}

    return {
        "path": str(path.resolve()),
        "run_id": path.parents[1].name,
        "n_shots": n_shots,
        "n_atoms": len(shot_index),
        "ground": ground.astype(np.uint32),
        "excited": excited.astype(np.uint32),
        "summaries": summaries,
        "metadata": metadata,
        "seconds": time.perf_counter() - started,
    }


def _extract_run_task(args):
    return extract_run(*args)


def build_feature_artifacts(paths, output_dir, config, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = inspect_runs(paths)
    n_shots_total = sum(record["n_shots"] for record in records)
    shape = (n_shots_total, config.bins, config.bins)
    ground_out = open_memmap(output_dir / "ground_counts.npy", mode="w+", dtype=np.uint32, shape=shape)
    excited_out = open_memmap(output_dir / "excited_counts.npy", mode="w+", dtype=np.uint32, shape=shape)

    summary_chunks = []
    metadata_chunks = {key: [] for key in METADATA_KEYS}
    run_ids = []
    shot_ids = []
    started = time.perf_counter()
    offset = 0
    tasks = [(record["path"], config.bins, config.xy_min, config.xy_max) for record in records]

    def consume(result, index):
        nonlocal offset
        stop = offset + result["n_shots"]
        ground_out[offset:stop] = result["ground"]
        excited_out[offset:stop] = result["excited"]
        summary_chunks.append(result["summaries"])
        for key in METADATA_KEYS:
            metadata_chunks[key].append(result["metadata"][key])
        run_ids.extend([result["run_id"]] * result["n_shots"])
        shot_ids.extend(range(result["n_shots"]))
        offset = stop
        elapsed = time.perf_counter() - started
        eta = elapsed / (index + 1) * (len(records) - index - 1)
        logging.info(
            "[%02d/%02d] %s: %d shots, %s atoms, %.1fs; elapsed %.1f min, ETA %.1f min",
            index + 1,
            len(records),
            result["run_id"],
            result["n_shots"],
            f"{result['n_atoms']:,}",
            result["seconds"],
            elapsed / 60,
            eta / 60,
        )

    if config.workers == 1:
        for index, task in enumerate(tasks):
            consume(_extract_run_task(task), index)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            for index, result in enumerate(executor.map(_extract_run_task, tasks)):
                consume(result, index)

    ground_out.flush()
    excited_out.flush()
    metadata_output = {
        "run_id": np.asarray(run_ids),
        "shot_id": np.asarray(shot_ids, dtype=np.int32),
        "summary_names": np.asarray(SUMMARY_NAMES),
        "summary_features": np.concatenate(summary_chunks),
    }
    metadata_output.update({key: np.concatenate(chunks) for key, chunks in metadata_chunks.items()})
    np.savez_compressed(output_dir / "shot_metadata.npz", **metadata_output)

    manifest = {
        "artifact_type": "2d_shot_features",
        "created_utc": utc_now(),
        "config": {
            "bins": config.bins,
            "xy_min": config.xy_min,
            "xy_max": config.xy_max,
            "workers": config.workers,
        },
        "n_runs": len(records),
        "n_shots": n_shots_total,
        "count_shape": list(shape),
        "count_dtype": "uint32",
        "summary_names": list(SUMMARY_NAMES),
        "metadata_keys": list(METADATA_KEYS),
        "source_runs": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _fit_pca_cpu(matrix, n_components):
    from sklearn.decomposition import PCA

    model = PCA(n_components=n_components)
    scores = model.fit_transform(matrix)
    return model.components_, scores, model.explained_variance_, model.explained_variance_ratio_


def _fit_pca_gpu(matrix, n_components):
    import cupy as cp

    matrix_gpu = cp.asarray(matrix)
    covariance = matrix_gpu.T @ matrix_gpu / (matrix_gpu.shape[0] - 1)
    eigenvalues, eigenvectors = cp.linalg.eigh(covariance)
    order = cp.argsort(eigenvalues)[::-1][:n_components]
    explained_variance = eigenvalues[order]
    components = eigenvectors[:, order].T
    scores = matrix_gpu @ components.T
    total_variance = cp.trace(covariance)
    return tuple(
        cp.asnumpy(value)
        for value in (components, scores, explained_variance, explained_variance / total_variance)
    )


def fit_pca_artifact(
    feature_dir,
    output_path,
    n_components=10,
    min_atoms_per_pixel=20,
    min_shot_fraction=0.95,
    backend="cpu",
    representation="contrast",
):
    feature_dir = Path(feature_dir)
    manifest = json.loads((feature_dir / "manifest.json").read_text())
    ground = np.load(feature_dir / "ground_counts.npy", mmap_mode="r")
    excited = np.load(feature_dir / "excited_counts.npy", mmap_mode="r")
    total = ground + excited
    valid_pixels = (total >= min_atoms_per_pixel).mean(axis=0) >= min_shot_fraction
    if valid_pixels.sum() < n_components:
        raise ValueError("Too few valid pixels for requested PCA components")

    valid_ground = ground[:, valid_pixels].astype(np.float64)
    valid_excited = excited[:, valid_pixels].astype(np.float64)
    valid_total = valid_ground + valid_excited
    fractions = (valid_excited + 0.5) / (valid_total + 1.0)
    if representation == "fraction":
        profiles = fractions
    elif representation == "contrast":
        # Exactly 2 * fraction - 1 with the same Jeffreys smoothing.
        profiles = (valid_excited - valid_ground) / (valid_total + 1.0)
    else:
        raise ValueError("representation must be 'fraction' or 'contrast'")
    global_profile = np.sum(valid_total * profiles, axis=1, keepdims=True) / np.sum(
        valid_total, axis=1, keepdims=True
    )
    shape_only = profiles - global_profile
    scaler_mean = shape_only.mean(axis=0)
    scaler_scale = shape_only.std(axis=0)
    scaler_scale[scaler_scale == 0] = 1
    standardized = (shape_only - scaler_mean) / scaler_scale

    if backend == "gpu":
        components, scores, explained_variance, explained_variance_ratio = _fit_pca_gpu(
            standardized, n_components
        )
    elif backend == "cpu":
        components, scores, explained_variance, explained_variance_ratio = _fit_pca_cpu(
            standardized, n_components
        )
    else:
        raise ValueError("backend must be 'cpu' or 'gpu'")

    physical_pc_deltas = components * scaler_scale[None, :] * np.sqrt(explained_variance)[:, None]
    bins = manifest["config"]["bins"]
    edges = np.linspace(manifest["config"]["xy_min"], manifest["config"]["xy_max"], bins + 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        valid_pixels=valid_pixels,
        x_edges=edges,
        y_edges=edges,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        components=components,
        scores=scores,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        physical_pc_deltas=physical_pc_deltas,
        global_profile=global_profile.ravel(),
        mean_valid_counts=valid_total.mean(axis=0),
        representation=np.asarray(representation),
        backend=np.asarray(backend),
        min_atoms_per_pixel=np.asarray(min_atoms_per_pixel),
        min_shot_fraction=np.asarray(min_shot_fraction),
    )
    return {
        "n_shots": int(scores.shape[0]),
        "n_valid_pixels": int(valid_pixels.sum()),
        "n_components": int(n_components),
        "backend": backend,
        "representation": representation,
        "explained_variance_ratio": explained_variance_ratio.tolist(),
    }
