#!/usr/bin/env python
"""
Fit the MLE distribution for every repeated experiment in a generated dataset.

This is the terminal-friendly equivalent of
notebooks/mle_distributions_inspection.ipynb, without plotting. Results are
checkpointed after every completed fit, so an interrupted job can be resumed.

Feature-conditioned likelihood
------------------------------
Passing --feature-dir-z0 and --feature-dir-z100 switches from the count-only
likelihood to the nonlinear image-summary-conditioned likelihood.  Selected
summary features and PCA scores are loaded from compact artifacts produced by
build_2d_shot_features.py and fit_2d_excitation_pca.py.  Each site's selected
feature matrix is standardized once over all artifact rows, then the same
mean/std is reused for every run.

For standardized Z0 feature vector s0_i and Z100 feature vector s100_i, the full
default model is

    p0_i = A0_i + 0.5*C0_i*cos(theta_i)
    p1_i = A1_i + 0.5*C1_i*cos(theta_i + dphi_signal_i + dpsi_i)

    dpsi_i = beta_phi @ [s0_i, s100_i]
    A0_i   = A0 + beta_A0 @ s0_i
    A1_i   = A1 + beta_A1 @ s100_i
    C0_i   = C0 * exp(beta_C0 @ s0_i)
    C1_i   = C1 * exp(beta_C1 @ s100_i)

The common phase theta_i is still marginalized as before.  The feature phase
correction is differential and is placed only in the second site; this fixes the
gauge by absorbing the first site's phase nuisance into theta_i.  Offset and
contrast corrections are local to each site.  Use --feature-nuisance to restrict
the enabled blocks, e.g. "--feature-nuisance phase" for the first minimal model.
"""

import argparse
from dataclasses import asdict
import logging
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "helpers"))

from fitting import fit_feature_conditioned_from_datasets, fit_from_datasets  # noqa: E402
from helpers import ImageShotDataset  # noqa: E402


DEFAULT_RUN_NAME = (
    REPO_ROOT
    / "data"
    / "R20_N50_A1000000_muXStd10.0um_muVxStd10.0um_"
      "sigX100um_sigVx309um_sigXStd10.0um_sigVxStd10.0um_"
      "phi0random_sig_A0.100_f0.3000"
)
RESULT_COLUMNS = [
    "A1", "A2", "C1", "C2", "phi0", "As", "Ac", "amp", "phase",
    "logL", "ntheta", "f", "converged",
    "feature_names", "feature_names_z0", "feature_names_z100", "feature_names_phase", "feature_nuisance",
    "beta_phi", "beta_A1", "beta_A2", "beta_C1", "beta_C2",
    "beta_phi_prior_std", "beta_A_prior_std", "beta_C_prior_std",
    "beta_penalty", "log_posterior",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit all runs from mle_distributions_inspection.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_name",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_NAME,
        help="Dataset directory containing run_### subdirectories.",
    )
    parser.add_argument("--frequency", "-f", type=float, default=0.3)
    parser.add_argument(
        "--run-start",
        type=int,
        default=0,
        help="First run index to fit. Ignored when --resume has more rows.",
    )
    parser.add_argument(
        "--run-stop",
        type=int,
        help="Exclusive final run index. Defaults to the run count in the name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output pickle path. Defaults to results/<run_name>.pkl.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log path. Defaults to logs/fit_<run_name>.log.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from rows already saved in the output pickle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of refusing to start.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the fitting helper's faster, slightly less accurate mode.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force NumPy likelihood evaluation instead of using CuPy.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        help="Override the adaptive fine quadrature grid size.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        help=(
            "Legacy shorthand for using the same feature artifact for both sites. "
            "Prefer --feature-dir-z0 and --feature-dir-z100."
        ),
    )
    parser.add_argument("--feature-dir-z0", type=Path, help="Z0 feature artifact directory.")
    parser.add_argument("--feature-dir-z100", type=Path, help="Z100 feature artifact directory.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=[],
        help=(
            "Legacy shorthand for the same summary feature names at both sites. "
            "Prefer --features-z0 and --features-z100."
        ),
    )
    parser.add_argument("--features-z0", nargs="+", help="Summary feature names from the Z0 artifact.")
    parser.add_argument("--features-z100", nargs="+", help="Summary feature names from the Z100 artifact.")
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=0,
        help="Legacy shorthand for the same number of PCA score features at both sites.",
    )
    parser.add_argument("--n-pcs-z0", type=int, help="Number of Z0 PCA score features to append.")
    parser.add_argument("--n-pcs-z100", type=int, help="Number of Z100 PCA score features to append.")
    parser.add_argument(
        "--pca-path",
        type=Path,
        help="Legacy shorthand for the same PCA artifact at both sites.",
    )
    parser.add_argument("--pca-path-z0", type=Path, help="Z0 PCA artifact path. Defaults to <feature-dir-z0>/2d-shot-pcas.npz.")
    parser.add_argument("--pca-path-z100", type=Path, help="Z100 PCA artifact path. Defaults to <feature-dir-z100>/2d-shot-pcas.npz.")
    parser.add_argument(
        "--beta-phi-prior-std",
        type=float,
        default=0.3,
        help="Gaussian prior std [rad] for each standardized beta_phi coefficient.",
    )
    parser.add_argument(
        "--feature-nuisance",
        nargs="+",
        choices=["phase", "offset", "contrast"],
        default=["phase", "offset", "contrast"],
        help=(
            "Feature-conditioned nuisance blocks to fit. Defaults to the full "
            "model. 'phase' fits beta_phi @ [s0_i, s100_i] in the differential phase; "
            "'offset' fits beta_A0/beta_A100 additive probability offsets; "
            "'contrast' fits C_z*exp(beta_Cz @ s_i)."
        ),
    )
    parser.add_argument(
        "--beta-A-prior-std",
        type=float,
        default=0.03,
        help=(
            "Gaussian prior std [probability] for each standardized offset "
            "coefficient beta_A0 and beta_A100."
        ),
    )
    parser.add_argument(
        "--beta-C-prior-std",
        type=float,
        default=0.3,
        help=(
            "Gaussian prior std [log-contrast] for each standardized contrast "
            "coefficient beta_C0 and beta_C100."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include debug details in the log.",
    )
    return parser.parse_args()


def infer_run_count(run_name):
    match = re.match(r"R(\d+)(?:_|$)", run_name.name)
    if match:
        return int(match.group(1))

    run_dirs = sorted(run_name.glob("run_[0-9][0-9][0-9]"))
    if not run_dirs:
        raise ValueError(
            f"Could not infer run count from {run_name.name!r} or its contents"
        )
    return max(int(path.name.removeprefix("run_")) for path in run_dirs) + 1


def configure_logging(log_file, verbose):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )


def save_checkpoint(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    temporary = output.with_suffix(output.suffix + ".tmp")
    df.to_pickle(temporary)
    os.replace(temporary, output)


def load_feature_artifacts(feature_dir, selected_features, n_pcs, pca_path=None):
    """Load selectable summary and PC features from compact feature artifacts."""
    feature_dir = feature_dir.expanduser().resolve()
    metadata_path = feature_dir / "shot_metadata.npz"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")

    metadata = np.load(metadata_path)
    summary_names = [str(name) for name in metadata["summary_names"].tolist()]
    summary_features = np.asarray(metadata["summary_features"], dtype=float)

    requested = list(selected_features or [])
    if "all-summary" in requested:
        if len(requested) > 1:
            raise ValueError("Use either all-summary or explicit summary names, not both")
        requested = summary_names

    columns = []
    names = []
    for name in requested:
        if name not in summary_names:
            raise ValueError(
                f"Unknown summary feature {name!r}. Available: {', '.join(summary_names)}"
            )
        index = summary_names.index(name)
        columns.append(summary_features[:, index])
        names.append(name)

    if n_pcs:
        pca_path = (
            pca_path.expanduser().resolve()
            if pca_path
            else feature_dir / "2d-shot-pcas.npz"
        )
        if not pca_path.is_file():
            raise FileNotFoundError(f"Missing PCA artifact: {pca_path}")
        pca = np.load(pca_path)
        scores = np.asarray(pca["scores"], dtype=float)
        if n_pcs < 0 or n_pcs > scores.shape[1]:
            raise ValueError(f"Requested --n-pcs={n_pcs}, but PCA artifact has {scores.shape[1]}")
        for index in range(n_pcs):
            columns.append(scores[:, index])
            names.append(f"PC{index + 1}_score")

    if not columns:
        raise ValueError("Feature-conditioned likelihood requires --features and/or --n-pcs")
    features = np.column_stack(columns)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale = np.where(feature_scale > 0, feature_scale, 1.0)
    return {
        "run_id": metadata["run_id"].astype(str),
        "shot_id": np.asarray(metadata["shot_id"], dtype=int),
        "features": features,
        "feature_names": tuple(names),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
    }



def resolve_site_feature_args(args):
    """Resolve legacy/shared feature CLI arguments into explicit Z0/Z100 args."""
    feature_dir_z0 = args.feature_dir_z0 or args.feature_dir
    feature_dir_z100 = args.feature_dir_z100 or args.feature_dir
    if (feature_dir_z0 is None) != (feature_dir_z100 is None):
        raise ValueError("Use both --feature-dir-z0 and --feature-dir-z100, or neither")
    if feature_dir_z0 is None:
        return None

    features_z0 = args.features_z0 if args.features_z0 is not None else args.features
    features_z100 = args.features_z100 if args.features_z100 is not None else args.features
    n_pcs_z0 = args.n_pcs_z0 if args.n_pcs_z0 is not None else args.n_pcs
    n_pcs_z100 = args.n_pcs_z100 if args.n_pcs_z100 is not None else args.n_pcs
    pca_path_z0 = args.pca_path_z0 or args.pca_path
    pca_path_z100 = args.pca_path_z100 or args.pca_path
    return {
        "z0": (feature_dir_z0, features_z0, n_pcs_z0, pca_path_z0),
        "z100": (feature_dir_z100, features_z100, n_pcs_z100, pca_path_z100),
    }


def select_run_features(feature_artifacts, run_idx, n_shots):
    run_id = f"run_{run_idx:03d}"
    mask = feature_artifacts["run_id"] == run_id
    if mask.sum() != n_shots:
        raise ValueError(
            f"Feature artifact has {mask.sum()} rows for {run_id}, expected {n_shots}"
        )
    order = np.argsort(feature_artifacts["shot_id"][mask])
    return feature_artifacts["features"][mask][order]


def main():
    args = parse_args()
    run_name = args.run_name.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else REPO_ROOT / "results" / f"{run_name.name}.pkl"
    )
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file
        else REPO_ROOT / "logs" / f"fit_{run_name.name}.log"
    )
    configure_logging(log_file, args.verbose)

    if not run_name.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {run_name}")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Use --resume or --overwrite."
        )

    n_runs = infer_run_count(run_name)
    run_stop = n_runs if args.run_stop is None else args.run_stop
    if not 0 <= args.run_start <= run_stop <= n_runs:
        raise ValueError(
            f"Expected 0 <= run-start <= run-stop <= {n_runs}, got "
            f"{args.run_start} and {run_stop}"
        )

    feature_artifacts = None
    site_feature_args = resolve_site_feature_args(args)
    if site_feature_args is not None:
        feature_artifacts = {
            site: load_feature_artifacts(*site_args)
            for site, site_args in site_feature_args.items()
        }
        logging.info("Z0 feature-conditioned likelihood features: %s", ", ".join(feature_artifacts["z0"]["feature_names"]))
        logging.info("Z100 feature-conditioned likelihood features: %s", ", ".join(feature_artifacts["z100"]["feature_names"]))
        logging.info("Feature nuisance blocks: %s", ", ".join(args.feature_nuisance))
        logging.info(
            "beta prior stds: phi=%.6g rad, A=%.6g probability, C=%.6g log-contrast",
            args.beta_phi_prior_std, args.beta_A_prior_std, args.beta_C_prior_std,
        )

    rows = []
    start = args.run_start
    if args.resume and output.exists():
        existing = pd.read_pickle(output)
        missing = [column for column in RESULT_COLUMNS if column not in existing]
        for column in missing:
            if column in {
                "feature_names", "feature_names_z0", "feature_names_z100",
                "feature_names_phase", "feature_nuisance", "beta_phi", "beta_A1",
                "beta_A2", "beta_C1", "beta_C2",
            }:
                existing[column] = [tuple() for _ in range(len(existing))]
            elif column in {
                "beta_phi_prior_std", "beta_A_prior_std", "beta_C_prior_std",
                "log_posterior",
            }:
                existing[column] = np.nan
            elif column == "beta_penalty":
                existing[column] = 0.0
            else:
                raise ValueError(f"Cannot resume; output is missing column: {column}")
        rows = existing[RESULT_COLUMNS].to_dict("records")
        start = max(start, len(rows))

    logging.info("Dataset: %s", run_name)
    logging.info("Runs: %d through %d (exclusive)", start, run_stop)
    logging.info("Frequency: %.8g cycles/shot", args.frequency)
    logging.info("Mode: %s; backend: %s", "fast" if args.fast else "full",
                 "CPU" if args.cpu else "GPU when available")
    logging.info("Checkpoint: %s", output)
    logging.info("Log: %s", log_file)

    job_start = time.perf_counter()
    for run_idx in range(start, run_stop):
        run_start = time.perf_counter()
        run_dir = run_name / f"run_{run_idx:03d}"
        z0_path = run_dir / "Z0" / "data_IMG.h5"
        z100_path = run_dir / "Z100" / "data_IMG.h5"
        if not z0_path.is_file() or not z100_path.is_file():
            raise FileNotFoundError(
                f"Missing input for run {run_idx}: {z0_path} or {z100_path}"
            )

        logging.info("[%d/%d] Loading run_%03d", run_idx + 1, run_stop, run_idx)
        z0 = ImageShotDataset(z0_path)
        z100 = ImageShotDataset(z100_path)
        logging.debug(
            "run_%03d: Z0=%d shots/%d atoms launched; Z100=%d shots/%d atoms launched",
            run_idx, z0.n_shots, z0.n_atoms_launched,
            z100.n_shots, z100.n_atoms_launched,
        )

        if feature_artifacts is None:
            result = fit_from_datasets(
                z0,
                z100,
                f=args.frequency,
                use_gpu=not args.cpu,
                ntheta=args.ntheta,
                fast=args.fast,
            )
        else:
            features_z0 = select_run_features(feature_artifacts["z0"], run_idx, z0.n_shots)
            features_z100 = select_run_features(feature_artifacts["z100"], run_idx, z100.n_shots)
            result = fit_feature_conditioned_from_datasets(
                z0,
                z100,
                features_z0,
                features_z100,
                f=args.frequency,
                feature_names_z0=feature_artifacts["z0"]["feature_names"],
                feature_names_z100=feature_artifacts["z100"]["feature_names"],
                use_gpu=not args.cpu,
                ntheta=args.ntheta,
                feature_nuisance=args.feature_nuisance,
                beta_phi_prior_std=args.beta_phi_prior_std,
                beta_A_prior_std=args.beta_A_prior_std,
                beta_C_prior_std=args.beta_C_prior_std,
                feature_mean_z0=feature_artifacts["z0"]["feature_mean"],
                feature_scale_z0=feature_artifacts["z0"]["feature_scale"],
                feature_mean_z100=feature_artifacts["z100"]["feature_mean"],
                feature_scale_z100=feature_artifacts["z100"]["feature_scale"],
                fast=args.fast,
            )
        rows.append(asdict(result))
        save_checkpoint(rows, output)

        elapsed = time.perf_counter() - run_start
        logging.info(
            "[%d/%d] run_%03d complete in %.1fs | amp=%.8g phase=%.8g "
            "logL=%.8g ntheta=%d converged=%s",
            run_idx + 1, run_stop, run_idx, elapsed, result.amp, result.phase,
            result.logL, result.ntheta, result.converged,
        )

    elapsed = time.perf_counter() - job_start
    converged = sum(bool(row["converged"]) for row in rows)
    logging.info(
        "Finished %d fits in %.1fs; %d/%d converged. Saved %s",
        max(0, run_stop - start), elapsed, converged, len(rows), output,
    )


if __name__ == "__main__":
    main()
