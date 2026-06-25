#!/usr/bin/env python
"""Oracle-cloud run-level sinusoidal MLE using full 2D pixel likelihoods.

This is a debugging/oracle baseline, not the final nuisance-aware likelihood:
it fixes each shot's cloud parameters to simulation metadata and fits only the
run-level signal parameters.  The physically relevant production likelihood must
profile or marginalize those cloud parameters, assuming only beam/PSMAP parameters
are fixed.

For each run, this script precomputes state-resolved PSMAP image templates for
every shot and site, then maximizes

    log L = sum_i log int dtheta_i p(Z0_i | theta_i)
                              p(Z100_i | theta_i + dphi_i)

where

    dphi_i = phi0 + As sin(2*pi*f*i) + Ac cos(2*pi*f*i).

The output pickle uses the same result columns as the count-only fitter so it
can be added directly to ``notebooks/mle_distributions.ipynb``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
import os
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "helpers"))

from fitting import FitResult  # noqa: E402
from psmap_fisher import PARAMETER_NAMES, PSMAPConditionalImageModel  # noqa: E402

try:
    from aispy.psmap import load_psmap
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT.parents[2] / "local" / "aispy"))
    from aispy.psmap import load_psmap


DEFAULT_RUN_NAME = (
    REPO_ROOT
    / "data"
    / "R20_N200_A100000000_muXStd10.0um_muVxStd10.0um_"
      "sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_"
      "phi0random_sig_A0.100_f0.3000"
)
DEFAULT_PSMAP_Z0 = REPO_ROOT / "output-files" / "PSGRID4D_CONFOCAL_Z0.h5"
DEFAULT_PSMAP_Z100 = REPO_ROOT / "output-files" / "PSGRID4D_CONFOCAL_Z100.h5"
T_DET = 3.8

RESULT_COLUMNS = [
    "A1", "A2", "C1", "C2", "phi0", "As", "Ac", "amp", "phase",
    "logL", "ntheta", "f", "converged",
    "optimizer_status", "optimizer_message", "optimizer_nit", "optimizer_nfev",
    "optimizer_objective", "selected_fine_start",
    "fine_start_success", "fine_start_status", "fine_start_messages",
    "fine_start_nit", "fine_start_nfev", "fine_start_objectives",
    "feature_names", "feature_names_z0", "feature_names_z100", "feature_names_phase", "feature_nuisance",
    "beta_phi", "beta_A1", "beta_A2", "beta_C1", "beta_C2",
    "beta_phi_prior_std", "beta_A_prior_std", "beta_C_prior_std",
    "beta_penalty", "log_posterior",
]

H5_KEYS = {
    "mu_x0": "mu_x0",
    "mu_y0": "mu_y0",
    "mu_vx0": "mu_vx0",
    "mu_vy0": "mu_vy0",
    "sigma_x0": "sigma_x",
    "sigma_y0": "sigma_y",
    "sigma_vx0": "sigma_vx",
    "sigma_vy0": "sigma_vy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", nargs="?", type=Path, default=DEFAULT_RUN_NAME)
    parser.add_argument("--frequency", "-f", type=float, default=0.3)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--run-stop", type=int)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "r20_n200_a1e8_mle_pixel_2d_oracle_cloud.pkl")
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs" / "fit_pixel_mle_2d.log")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--psmap-z0", type=Path, default=DEFAULT_PSMAP_Z0)
    parser.add_argument("--psmap-z100", type=Path, default=DEFAULT_PSMAP_Z100)
    parser.add_argument("--bins", type=int, default=32, help="Downsampled image bins per axis for the pixel likelihood.")
    parser.add_argument("--hermite-order", type=int, default=8)
    parser.add_argument("--ntheta", type=int, default=128, help="Common-phase quadrature points.")
    parser.add_argument("--phase-grid", type=int, default=8, help="Starting grid for global phi0.")
    parser.add_argument("--amp-starts", nargs="+", type=float, default=[0.03, 0.07, 0.10])
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--fast", action="store_true", help="Use fewer starts and looser optimizer tolerances.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def configure_logging(log_file, verbose):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a")],
    )


def infer_run_count(run_name):
    match = re.match(r"R(\d+)(?:_|$)", run_name.name)
    if match:
        return int(match.group(1))
    run_dirs = sorted(run_name.glob("run_[0-9][0-9][0-9]"))
    if not run_dirs:
        raise ValueError(f"Could not infer run count from {run_name}")
    return max(int(path.name.removeprefix("run_")) for path in run_dirs) + 1


def save_checkpoint(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    temporary = output.with_suffix(output.suffix + ".tmp")
    df.to_pickle(temporary)
    os.replace(temporary, output)


def downsample_sum(image, bins):
    block = image.shape[0] // bins
    return image.reshape(bins, block, bins, block).sum(axis=(1, 3)).astype(np.float64)


def downsampled_edges(handle, bins):
    image_res = int(handle.attrs["image_res"])
    if image_res % bins:
        raise ValueError(f"image_res={image_res} is not divisible by bins={bins}")
    half_range = float(handle.attrs["image_half_range"])
    return np.linspace(-half_range, half_range, bins + 1)


def shot_theta(handle, shot):
    return np.array([float(handle[H5_KEYS[name]][shot]) for name in PARAMETER_NAMES], dtype=float)


def load_observed_images(path, bins):
    with h5py.File(path, "r") as handle:
        n_shots = int(handle["images_s0"].shape[0])
        edges = downsampled_edges(handle, bins)
        observed = np.empty((n_shots, 2, bins * bins), dtype=np.float64)
        theta = np.empty((n_shots, len(PARAMETER_NAMES)), dtype=np.float64)
        for shot in range(n_shots):
            observed[shot, 0] = downsample_sum(handle["images_s0"][shot], bins).ravel()
            observed[shot, 1] = downsample_sum(handle["images_s1"][shot], bins).ravel()
            theta[shot] = shot_theta(handle, shot)
    return observed, theta, edges


def conditional_detected(model, theta):
    probs = model.detected_probabilities(theta)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid detected probability from PSMAP model")
    return probs / total


def build_phase_basis(psmap_path, theta_by_shot, edges, hermite_order):
    """Return p_mean, p_cos, p_sin arrays with shape (n_shots, 2, n_pixels)."""
    psmap = load_psmap(str(psmap_path))
    models = [
        PSMAPConditionalImageModel.from_psmap(psmap, T_DET, phase, edges, edges, hermite_order=hermite_order)
        for phase in (0.0, 0.5 * np.pi, np.pi)
    ]
    n_shots = theta_by_shot.shape[0]
    n_pix = (len(edges) - 1) ** 2
    p0 = np.empty((n_shots, 2 * n_pix), dtype=np.float64)
    p90 = np.empty_like(p0)
    p180 = np.empty_like(p0)
    for shot, theta in enumerate(theta_by_shot):
        p0[shot] = conditional_detected(models[0], theta)
        p90[shot] = conditional_detected(models[1], theta)
        p180[shot] = conditional_detected(models[2], theta)
    mean = 0.5 * (p0 + p180)
    cos_coef = 0.5 * (p0 - p180)
    sin_coef = mean - p90
    return (
        mean.reshape(n_shots, 2, n_pix),
        cos_coef.reshape(n_shots, 2, n_pix),
        sin_coef.reshape(n_shots, 2, n_pix),
    )


class PixelTimeSeriesLikelihood:
    def __init__(self, observed_z0, basis_z0, observed_z100, basis_z100, ntheta):
        if observed_z0.shape != observed_z100.shape:
            raise ValueError("Z0 and Z100 observed arrays must have matching shape")
        self.obs0 = observed_z0.reshape(observed_z0.shape[0], -1)
        self.obs1 = observed_z100.reshape(observed_z100.shape[0], -1)
        self.basis0 = tuple(part.reshape(part.shape[0], -1) for part in basis_z0)
        self.basis1 = tuple(part.reshape(part.shape[0], -1) for part in basis_z100)
        self.n_shots = self.obs0.shape[0]
        self.ntheta = int(ntheta)
        self.theta_grid = np.linspace(0.0, 2.0 * np.pi, self.ntheta, endpoint=False)
        self.cos_theta = np.cos(self.theta_grid)
        self.sin_theta = np.sin(self.theta_grid)
        self.t = np.arange(self.n_shots, dtype=float)
        self.ll0_theta = self._log_image_prob(self.obs0, self.basis0, self.cos_theta, self.sin_theta)

    @staticmethod
    def _log_image_prob(observed, basis, cos_phase, sin_phase):
        mean, cos_coef, sin_coef = basis
        cos_phase = np.asarray(cos_phase)
        sin_phase = np.asarray(sin_phase)
        if cos_phase.ndim == 1:
            cos_phase = cos_phase[None, :, None]
            sin_phase = sin_phase[None, :, None]
        elif cos_phase.ndim == 2:
            cos_phase = cos_phase[:, :, None]
            sin_phase = sin_phase[:, :, None]
        else:
            raise ValueError("phase arrays must be 1D or 2D")
        probs = mean[:, None, :] + cos_phase * cos_coef[:, None, :] + sin_phase * sin_coef[:, None, :]
        probs = np.clip(probs, 1e-300, None)
        return np.sum(observed[:, None, :] * np.log(probs), axis=2)

    def signal_ll(self, phi0, As, Ac, f):
        dphi = phi0 + As * np.sin(2.0 * np.pi * f * self.t) + Ac * np.cos(2.0 * np.pi * f * self.t)
        phase1 = self.theta_grid[None, :] + dphi[:, None]
        ll1 = self._log_image_prob(self.obs1, self.basis1, np.cos(phase1), np.sin(phase1))
        return float(np.sum(logsumexp(self.ll0_theta + ll1, axis=1) - np.log(self.ntheta)))


def valid_signal_params(p, amp_bound=np.pi):
    phi0, As, Ac = p
    return 0.0 <= phi0 <= 2.0 * np.pi and 0.0 <= As <= amp_bound and -amp_bound <= Ac <= amp_bound


def fit_pixel_signal(ev, f, phase_grid, amp_starts, maxiter, fast):
    if fast:
        phase_grid = min(phase_grid, 4)
        amp_starts = [amp_starts[-1] if amp_starts else 0.07]
        maxiter = min(maxiter, 500)
        xatol, fatol = 1e-4, 1e-1
    else:
        xatol, fatol = 1e-6, 1e-2

    def objective(p):
        if not valid_signal_params(p):
            return 1e300
        value = ev.signal_ll(p[0], p[1], p[2], f)
        return -value if np.isfinite(value) else 1e300

    best = None
    phi_grid = np.linspace(0, 2 * np.pi, phase_grid, endpoint=False)
    for phi in phi_grid:
        for amp in amp_starts:
            amp = max(float(amp), 0.0)
            starts = [
                np.array([phi, amp, 0.0]),
                np.array([phi, amp, 0.5 * amp]),
                np.array([phi, amp, -0.5 * amp]),
                np.array([phi, 0.5 * amp, amp]),
                np.array([phi, 0.5 * amp, -amp]),
            ]
            for start in starts:
                result = minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=[(0.0, 2.0 * np.pi), (0.0, np.pi), (-np.pi, np.pi)],
                    options={"maxiter": maxiter, "ftol": 1e-7 if fast else 1e-9, "gtol": 1e-4 if fast else 1e-6},
                )
                if best is None or result.fun < best.fun:
                    best = result

    phi0, As, Ac = best.x
    As = abs(float(As))
    Ac = float(Ac)
    amp = float(np.hypot(As, Ac))
    phase = float(np.arctan2(Ac, As))
    return FitResult(
        A1=np.nan,
        A2=np.nan,
        C1=np.nan,
        C2=np.nan,
        phi0=float(phi0),
        As=As,
        Ac=Ac,
        amp=amp,
        phase=phase,
        logL=-float(best.fun),
        ntheta=ev.ntheta,
        f=float(f),
        converged=bool(best.success),
        optimizer_status=int(best.status),
        optimizer_message=str(best.message),
        optimizer_nit=int(best.nit),
        optimizer_nfev=int(best.nfev),
        optimizer_objective=float(best.fun),
    )


def main():
    args = parse_args()
    run_name = args.run_name.expanduser().resolve()
    output = args.output.expanduser().resolve()
    log_file = args.log_file.expanduser().resolve()
    configure_logging(log_file, args.verbose)

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --resume or --overwrite.")
    if not run_name.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {run_name}")

    n_runs = infer_run_count(run_name)
    run_stop = n_runs if args.run_stop is None else args.run_stop
    rows = []
    start = args.run_start
    if args.resume and output.exists():
        existing = pd.read_pickle(output)
        rows = existing.reindex(columns=RESULT_COLUMNS).to_dict("records")
        start = max(start, len(rows))

    logging.info("Dataset: %s", run_name)
    logging.info("Runs: %d through %d (exclusive)", start, run_stop)
    logging.info("Pixel bins: %d x %d; Hermite order: %d; ntheta: %d", args.bins, args.bins, args.hermite_order, args.ntheta)
    logging.info("Checkpoint: %s", output)

    job_start = time.perf_counter()
    for run_idx in range(start, run_stop):
        run_start = time.perf_counter()
        run_dir = run_name / f"run_{run_idx:03d}"
        z0_path = run_dir / "Z0" / "data_IMG.h5"
        z100_path = run_dir / "Z100" / "data_IMG.h5"
        logging.info("[%d/%d] Loading observed images for run_%03d", run_idx + 1, run_stop, run_idx)
        obs_z0, theta_z0, edges_z0 = load_observed_images(z0_path, args.bins)
        obs_z100, theta_z100, edges_z100 = load_observed_images(z100_path, args.bins)
        if not np.allclose(edges_z0, edges_z100):
            raise ValueError("Z0 and Z100 image edges do not match")

        logging.info("[%d/%d] Building Z0 pixel templates", run_idx + 1, run_stop)
        basis_z0 = build_phase_basis(args.psmap_z0, theta_z0, edges_z0, args.hermite_order)
        logging.info("[%d/%d] Building Z100 pixel templates", run_idx + 1, run_stop)
        basis_z100 = build_phase_basis(args.psmap_z100, theta_z100, edges_z0, args.hermite_order)
        ev = PixelTimeSeriesLikelihood(obs_z0, basis_z0, obs_z100, basis_z100, args.ntheta)

        result = fit_pixel_signal(
            ev,
            f=args.frequency,
            phase_grid=args.phase_grid,
            amp_starts=args.amp_starts,
            maxiter=args.maxiter,
            fast=args.fast,
        )
        rows.append(asdict(result))
        save_checkpoint(rows, output)
        elapsed = time.perf_counter() - run_start
        logging.info(
            "[%d/%d] run_%03d complete in %.1fs | amp=%.8g phase=%.8g logL=%.8g converged=%s nfev=%d",
            run_idx + 1,
            run_stop,
            run_idx,
            elapsed,
            result.amp,
            result.phase,
            result.logL,
            result.converged,
            result.optimizer_nfev,
        )

    elapsed = time.perf_counter() - job_start
    logging.info("Finished %d fits in %.1fs. Saved %s", max(0, run_stop - start), elapsed, output)


if __name__ == "__main__":
    main()
