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
SUMMARY_NAMES = (
    "mean_x", "mean_y", "std_x", "std_y", "cov_xy",
    # Excitation-contrast moments: sensitive to phase-gradient across the cloud.
    # cov(x, state) ~ k_x * sigma_x^2 * sin(phi0), where k_x is the local phase
    # gradient in x. These are the features that break the mu_x0/mu_vx0 degeneracy.
    "cov_x_state", "cov_y_state",
    "cov_x2_state", "cov_y2_state",
    # Excited-atom centroid: mean position of atoms that fired, shifted relative
    # to the total-cloud centroid by the fringe pattern.
    "mean_x_exc", "mean_y_exc",
)


@dataclass(frozen=True)
class ExtractionConfig:
    bins: int = 64
    xy_min: float = -4.0
    xy_max: float = 4.0
    workers: int = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def discover_run_paths(data_root, port="Z0", max_runs=None):
    paths = sorted(Path(data_root).glob(f"run_*/{port}/data_IMG.h5"))
    if max_runs is not None:
        paths = paths[:max_runs]
    if not paths:
        raise FileNotFoundError(f"No run_*/{port}/data_IMG.h5 files below {data_root}")
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
                    "n_atoms": int(handle.attrs.get("n_atoms_launched", 0)),
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
    return records


def extract_run(path, bins, xy_min, xy_max):
    """Return compact arrays for one data_IMG.h5 run.

    Count maps are built in cloud-normalized final coordinates:
    u = (x - mean_x) / std_x and v = (y - mean_y) / std_y.
    Moments are computed from pixel-weighted sums over the 2D histograms.
    Physical final COM and spread are still returned as summary features.
    """
    path = Path(path)
    started = time.perf_counter()
    bin_width = (xy_max - xy_min) / bins
    n_pixels = bins * bins

    with h5py.File(path, "r") as handle:
        imgs_s0 = handle["images_s0"][:].astype(np.float64)   # (n_shots, res, res)
        imgs_s1 = handle["images_s1"][:].astype(np.float64)
        half_range = float(handle.attrs["image_half_range"])
        n_shots = imgs_s0.shape[0]
        metadata = {key: handle[key][:] for key in METADATA_KEYS}

    res = imgs_s0.shape[1]
    img_edges = np.linspace(-half_range, half_range, res + 1)
    pc = 0.5 * (img_edges[:-1] + img_edges[1:])   # (res,) physical pixel centers

    total = imgs_s0 + imgs_s1                          # (n_shots, res, res)
    counts = total.sum(axis=(1, 2))                    # (n_shots,)
    safe_counts = np.where(counts > 0, counts, 1.0)

    # Position moments via pixel-weighted sums (pc first axis = x, second = y)
    pc2 = pc ** 2
    mean_x  = (total * pc[None, :, None]).sum(axis=(1, 2)) / safe_counts
    mean_y  = (total * pc[None, None, :]).sum(axis=(1, 2)) / safe_counts
    mean_x2 = (total * pc2[None, :, None]).sum(axis=(1, 2)) / safe_counts
    mean_y2 = (total * pc2[None, None, :]).sum(axis=(1, 2)) / safe_counts
    var_x   = mean_x2 - mean_x ** 2
    var_y   = mean_y2 - mean_y ** 2
    std_x   = np.sqrt(np.maximum(var_x, 0))
    std_y   = np.sqrt(np.maximum(var_y, 0))
    cov_xy  = ((total * pc[None, :, None] * pc[None, None, :]).sum(axis=(1, 2))
               / safe_counts - mean_x * mean_y)

    # Contrast (state) moments
    n1 = imgs_s1
    exc_total = n1.sum(axis=(1, 2))
    mean_state = exc_total / safe_counts
    safe_exc   = np.where(exc_total > 0, exc_total, 1.0)

    cov_x_state  = (n1 * pc[None, :, None]).sum(axis=(1, 2)) / safe_counts - mean_x * mean_state
    cov_y_state  = (n1 * pc[None, None, :]).sum(axis=(1, 2)) / safe_counts - mean_y * mean_state
    cov_x2_state = ((n1 * pc2[None, :, None]).sum(axis=(1, 2)) / safe_counts
                    - (var_x + mean_x ** 2) * mean_state)
    cov_y2_state = ((n1 * pc2[None, None, :]).sum(axis=(1, 2)) / safe_counts
                    - (var_y + mean_y ** 2) * mean_state)
    mean_x_exc   = (n1 * pc[None, :, None]).sum(axis=(1, 2)) / safe_exc
    mean_y_exc   = (n1 * pc[None, None, :]).sum(axis=(1, 2)) / safe_exc

    summaries = np.column_stack([
        mean_x, mean_y, std_x, std_y, cov_xy,
        cov_x_state, cov_y_state, cov_x2_state, cov_y2_state,
        mean_x_exc, mean_y_exc,
    ])

    # Cloud-normalized count maps: scatter each image pixel into the target grid
    safe_std_x = np.where(std_x > 0, std_x, 1.0)
    safe_std_y = np.where(std_y > 0, std_y, 1.0)
    u  = (pc[None, :] - mean_x[:, None]) / safe_std_x[:, None]  # (n_shots, res)
    v  = (pc[None, :] - mean_y[:, None]) / safe_std_y[:, None]
    bu = np.floor((u - xy_min) / bin_width).astype(np.int32)     # (n_shots, res)
    bv = np.floor((v - xy_min) / bin_width).astype(np.int32)

    ground_out  = np.zeros((n_shots, bins, bins), dtype=np.uint32)
    excited_out = np.zeros((n_shots, bins, bins), dtype=np.uint32)

    for j in range(n_shots):
        ix_v = np.where((bu[j] >= 0) & (bu[j] < bins))[0]
        iy_v = np.where((bv[j] >= 0) & (bv[j] < bins))[0]
        if ix_v.size == 0 or iy_v.size == 0:
            continue
        flat_bins = (bu[j][ix_v, None] * bins + bv[j][None, iy_v]).ravel()
        w_g = imgs_s0[j][np.ix_(ix_v, iy_v)].ravel()
        w_e = imgs_s1[j][np.ix_(ix_v, iy_v)].ravel()
        ground_out[j]  = np.bincount(flat_bins, weights=w_g, minlength=n_pixels).reshape(bins, bins).astype(np.uint32)
        excited_out[j] = np.bincount(flat_bins, weights=w_e, minlength=n_pixels).reshape(bins, bins).astype(np.uint32)

    return {
        "path": str(path.resolve()),
        "run_id": path.parents[1].name,
        "n_shots": n_shots,
        "n_atoms": int(total.sum()),
        "ground": ground_out,
        "excited": excited_out,
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
            "coordinate_system": "cloud_normalized_final_position",
            "coordinate_definition": {
                "x": "(final_x - mean_x) / std_x",
                "y": "(final_y - mean_y) / std_y",
            },
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
