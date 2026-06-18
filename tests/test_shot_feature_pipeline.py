import json

import h5py
import numpy as np

from helpers.shot_feature_pipeline import (
    ExtractionConfig,
    build_feature_artifacts,
    extract_run,
    fit_pca_artifact,
)


def make_run(path, run_id, seed):
    rng = np.random.default_rng(seed)
    n_shots = 4
    n_per_shot = 200
    shot_index = np.repeat(np.arange(n_shots, dtype=np.int32), n_per_shot)
    x = rng.normal(np.repeat(np.linspace(-2e-4, 2e-4, n_shots), n_per_shot), 8e-4)
    y = rng.normal(0, 8e-4, len(shot_index))
    states = rng.binomial(1, 0.5 + 0.15 * np.cos(1500 * x + 2e6 * (x * x + y * y)))

    run_path = path / run_id / "Z0" / "data_PROB.h5"
    run_path.parent.mkdir(parents=True)
    with h5py.File(run_path, "w") as handle:
        handle.create_dataset("positions", data=np.column_stack([x, y, np.zeros_like(x)]))
        handle.create_dataset("states", data=states)
        handle.create_dataset("shot_index", data=shot_index)
        handle.create_dataset("phi0", data=np.linspace(0, 1, n_shots))
        for key in ("mu_x0", "mu_y0", "mu_vx0", "mu_vy0"):
            handle.create_dataset(key, data=rng.normal(size=n_shots))
        for key in ("sigma_x", "sigma_y", "sigma_vx", "sigma_vy"):
            handle.create_dataset(key, data=rng.uniform(0.5, 1.5, size=n_shots))
    return run_path


def test_extract_run_counts_and_summaries(tmp_path):
    run_path = make_run(tmp_path, "run_000", 1)
    result = extract_run(run_path, bins=8, xy_min=-4.0, xy_max=4.0)

    assert result["ground"].shape == (4, 8, 8)
    assert result["excited"].shape == (4, 8, 8)
    assert result["summaries"].shape == (4, 11)
    totals = result["ground"].sum(axis=(1, 2)) + result["excited"].sum(axis=(1, 2))
    assert np.all((totals > 190) & (totals <= 200))


def test_artifacts_and_contrast_fraction_equivalence(tmp_path):
    paths = [make_run(tmp_path / "data", f"run_{index:03d}", index) for index in range(3)]
    feature_dir = tmp_path / "features"
    manifest = build_feature_artifacts(
        paths,
        feature_dir,
        ExtractionConfig(bins=8, xy_min=-4.0, xy_max=4.0),
    )

    assert manifest["n_runs"] == 3
    assert manifest["n_shots"] == 12
    assert np.load(feature_dir / "ground_counts.npy", mmap_mode="r").shape == (12, 8, 8)
    metadata = np.load(feature_dir / "shot_metadata.npz")
    assert metadata["summary_features"].shape == (12, 11)
    assert len(np.unique(metadata["run_id"])) == 3
    manifest_json = json.loads((feature_dir / "manifest.json").read_text())
    assert manifest_json["n_shots"] == 12
    assert manifest_json["config"]["coordinate_system"] == "cloud_normalized_final_position"

    fraction_path = tmp_path / "fraction.npz"
    contrast_path = tmp_path / "contrast.npz"
    fit_pca_artifact(
        feature_dir,
        fraction_path,
        n_components=3,
        min_atoms_per_pixel=1,
        min_shot_fraction=0.25,
        representation="fraction",
    )
    fit_pca_artifact(
        feature_dir,
        contrast_path,
        n_components=3,
        min_atoms_per_pixel=1,
        min_shot_fraction=0.25,
        representation="contrast",
    )
    fraction = np.load(fraction_path)
    contrast = np.load(contrast_path)

    np.testing.assert_allclose(
        fraction["explained_variance_ratio"],
        contrast["explained_variance_ratio"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.abs(fraction["scores"]),
        np.abs(contrast["scores"]),
        atol=1e-9,
    )
