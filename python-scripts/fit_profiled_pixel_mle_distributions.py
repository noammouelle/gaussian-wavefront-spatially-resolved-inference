#!/usr/bin/env python
"""Run-level 2D pixel MLE with per-shot cloud parameters profiled.

This is the nuisance-aware pixel likelihood intended for comparison with the
count-only MLE distribution.  Beam/PSMAP parameters are fixed.  For every run,
shot, site, and absolute laser phase on a grid, the script profiles the eight
Gaussian cloud parameters against the state-resolved 2D image.  The resulting
site phase-profile tables are combined into per-shot differential-phase
likelihood curves, which are then used to fit the run-level sinusoidal signal
parameters phi0, As, and Ac.

The expensive per-site profile tables are checkpointed as NPZ files so the
signal fit can be repeated without redoing the cloud profiling.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run_name", nargs="?", type=Path, default=DEFAULT_RUN_NAME)
    parser.add_argument("--frequency", "-f", type=float, default=0.3)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--run-stop", type=int)
    parser.add_argument("--max-shots", type=int, help="Debug limit per run.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "r20_n200_a1e8_mle_pixel_2d_profiled.pkl")
    parser.add_argument("--profile-dir", type=Path, default=REPO_ROOT / "results" / "pixel_2d_profile_tables")
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs" / "fit_profiled_pixel_mle_2d.log")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-profiles", action="store_true")
    parser.add_argument("--psmap-z0", type=Path, default=DEFAULT_PSMAP_Z0)
    parser.add_argument("--psmap-z100", type=Path, default=DEFAULT_PSMAP_Z100)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--hermite-order", type=int, default=4)
    parser.add_argument("--phase-grid", type=int, default=16, help="Absolute phase grid for per-site cloud profiling.")
    parser.add_argument("--delta-grid", type=int, default=128, help="Differential phase grid used after profile-table combination.")
    parser.add_argument("--profile-maxiter", type=int, default=80)
    parser.add_argument("--signal-maxiter", type=int, default=300)
    parser.add_argument("--fit-widths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mu-bound-um", type=float, default=80.0)
    parser.add_argument("--v-bound-um-s", type=float, default=80.0)
    parser.add_argument("--sigma-x-min-um", type=float, default=50.0)
    parser.add_argument("--sigma-x-max-um", type=float, default=180.0)
    parser.add_argument("--sigma-v-min-um-s", type=float, default=40.0)
    parser.add_argument("--sigma-v-max-um-s", type=float, default=180.0)
    parser.add_argument("--nominal-sigma-x-um", type=float, default=100.0)
    parser.add_argument("--nominal-sigma-v-um-s", type=float, default=100.0)
    parser.add_argument("--phase-starts", type=int, default=8)
    parser.add_argument("--amp-starts", nargs="+", type=float, default=[0.03, 0.07, 0.10])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
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
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_pickle(temporary)
    os.replace(temporary, output)


def downsample_sum(image, bins):
    block = image.shape[0] // bins
    return image.reshape(bins, block, bins, block).sum(axis=(1, 3)).astype(np.float64)


def load_images(path, bins, max_shots=None):
    with h5py.File(path, "r") as handle:
        image_res = int(handle.attrs["image_res"])
        if image_res % bins:
            raise ValueError(f"image_res={image_res} is not divisible by bins={bins}")
        n_shots = int(handle["images_s0"].shape[0])
        if max_shots is not None:
            n_shots = min(n_shots, max_shots)
        half_range = float(handle.attrs["image_half_range"])
        edges = np.linspace(-half_range, half_range, bins + 1)
        obs = np.empty((n_shots, 2, bins * bins), dtype=np.float64)
        for shot in range(n_shots):
            obs[shot, 0] = downsample_sum(handle["images_s0"][shot], bins).ravel()
            obs[shot, 1] = downsample_sum(handle["images_s1"][shot], bins).ravel()
    return obs, edges


def moment_start(observed, edges, args):
    total = observed[0].reshape(len(edges) - 1, len(edges) - 1) + observed[1].reshape(len(edges) - 1, len(edges) - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = max(float(total.sum()), 1.0)
    x = centers[:, None]
    y = centers[None, :]
    final_x = float((total * x).sum() / counts)
    final_y = float((total * y).sum() / counts)
    return np.array([
        final_x,
        final_y,
        0.0,
        0.0,
        args.nominal_sigma_x_um * 1e-6,
        args.nominal_sigma_x_um * 1e-6,
        args.nominal_sigma_v_um_s * 1e-6,
        args.nominal_sigma_v_um_s * 1e-6,
    ], dtype=float)


def pack_theta(theta, fit_widths):
    if fit_widths:
        return np.r_[theta[:4], np.log(theta[4:])]
    return theta[:4].copy()


def unpack_theta(params, fixed_widths, fit_widths):
    if fit_widths:
        return np.r_[params[:4], np.exp(params[4:8])]
    return np.r_[params[:4], fixed_widths]


def profile_bounds(args, fixed_widths, fit_widths):
    lower4 = np.array([-args.mu_bound_um, -args.mu_bound_um, -args.v_bound_um_s, -args.v_bound_um_s]) * 1e-6
    upper4 = -lower4
    if not fit_widths:
        return list(zip(lower4, upper4))
    lower_widths = np.array([
        args.sigma_x_min_um, args.sigma_x_min_um,
        args.sigma_v_min_um_s, args.sigma_v_min_um_s,
    ]) * 1e-6
    upper_widths = np.array([
        args.sigma_x_max_um, args.sigma_x_max_um,
        args.sigma_v_max_um_s, args.sigma_v_max_um_s,
    ]) * 1e-6
    return list(zip(np.r_[lower4, np.log(lower_widths)], np.r_[upper4, np.log(upper_widths)]))


def profile_one_shot_phase(model, observed, start_theta, fixed_widths, bounds, fit_widths, maxiter):
    obs = observed.reshape(-1)

    def objective(params):
        theta = unpack_theta(params, fixed_widths, fit_widths)
        try:
            probs = model.detected_probabilities(theta)
        except Exception:
            return 1e300
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            return 1e300
        probs = np.clip(probs / total, 1e-300, None)
        return -float(np.sum(obs * np.log(probs)))

    result = minimize(
        objective,
        pack_theta(start_theta, fit_widths),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-5, "maxls": 20},
    )
    theta_hat = unpack_theta(result.x, fixed_widths, fit_widths)
    return -float(result.fun), theta_hat, result


def profile_cache_path(args, run_idx, site):
    suffix = (
        f"run{run_idx:03d}_{site}_bins{args.bins}_h{args.hermite_order}_"
        f"p{args.phase_grid}_fitw{int(args.fit_widths)}"
    )
    if args.max_shots is not None:
        suffix += f"_n{args.max_shots}"
    return args.profile_dir / f"{suffix}.npz"


def build_or_load_profiles(args, run_idx, site, image_path, psmap_path):
    cache = profile_cache_path(args, run_idx, site)
    if cache.exists() and not args.rebuild_profiles:
        logging.info("Using cached profile table %s", cache)
        data = np.load(cache)
        return data["phase_grid"], data["log_profile"]

    observed, edges = load_images(image_path, args.bins, args.max_shots)
    psmap = load_psmap(str(psmap_path))
    phase_grid = np.linspace(0.0, 2.0 * np.pi, args.phase_grid, endpoint=False)
    models = [
        PSMAPConditionalImageModel.from_psmap(psmap, T_DET, phase, edges, edges, hermite_order=args.hermite_order)
        for phase in phase_grid
    ]
    n_shots = observed.shape[0]
    log_profile = np.empty((n_shots, len(phase_grid)), dtype=np.float64)
    theta_hats = np.empty((n_shots, len(phase_grid), 8), dtype=np.float64)
    success = np.empty((n_shots, len(phase_grid)), dtype=bool)
    nfev = np.empty((n_shots, len(phase_grid)), dtype=np.int32)

    fixed_widths = np.array([
        args.nominal_sigma_x_um, args.nominal_sigma_x_um,
        args.nominal_sigma_v_um_s, args.nominal_sigma_v_um_s,
    ]) * 1e-6
    bounds = profile_bounds(args, fixed_widths, args.fit_widths)
    started = time.perf_counter()
    for shot in range(n_shots):
        start_theta = moment_start(observed[shot], edges, args)
        for phase_index, model in enumerate(models):
            if phase_index:
                start_theta = theta_hats[shot, phase_index - 1]
            ll, theta_hat, result = profile_one_shot_phase(
                model, observed[shot], start_theta, fixed_widths, bounds,
                args.fit_widths, args.profile_maxiter,
            )
            log_profile[shot, phase_index] = ll
            theta_hats[shot, phase_index] = theta_hat
            success[shot, phase_index] = bool(result.success)
            nfev[shot, phase_index] = int(result.nfev)
        if (shot + 1) % max(1, min(10, n_shots)) == 0 or shot + 1 == n_shots:
            logging.info(
                "%s run_%03d profiled %d/%d shots in %.1fs",
                site, run_idx, shot + 1, n_shots, time.perf_counter() - started,
            )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        phase_grid=phase_grid,
        log_profile=log_profile,
        theta_hats=theta_hats,
        success=success,
        nfev=nfev,
        args=json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}),
    )
    logging.info("Wrote profile table %s", cache)
    return phase_grid, log_profile


def interp_periodic(grid, values, points):
    grid = np.asarray(grid)
    period = 2.0 * np.pi
    step = period / len(grid)
    extended_x = np.r_[grid, period]
    extended_y = np.r_[values, values[0]]
    return np.interp(np.mod(points, period), extended_x, extended_y, period=period)


def combine_delta_profiles(phase_grid, log_z0, log_z100, delta_grid):
    n_shots = log_z0.shape[0]
    ell = np.empty((n_shots, len(delta_grid)), dtype=np.float64)
    for shot in range(n_shots):
        for j, delta in enumerate(delta_grid):
            z100_shift = interp_periodic(phase_grid, log_z100[shot], phase_grid + delta)
            ell[shot, j] = logsumexp(log_z0[shot] + z100_shift) - np.log(len(phase_grid))
    return ell


def fit_signal(delta_grid, ell_delta, frequency, args):
    t = np.arange(ell_delta.shape[0], dtype=float)

    def shot_ell(delta_values):
        return np.array([interp_periodic(delta_grid, ell_delta[i], delta_values[i]) for i in range(len(t))])

    def objective(params):
        phi0, As, Ac = params
        if not (0.0 <= phi0 <= 2*np.pi and 0.0 <= As <= np.pi and -np.pi <= Ac <= np.pi):
            return 1e300
        dphi = phi0 + As * np.sin(2*np.pi*frequency*t) + Ac * np.cos(2*np.pi*frequency*t)
        value = float(np.sum(shot_ell(dphi)))
        return -value if np.isfinite(value) else 1e300

    phase_starts = min(args.phase_starts, 4) if args.fast else args.phase_starts
    amp_starts = [args.amp_starts[-1]] if args.fast else args.amp_starts
    best = None
    for phi in np.linspace(0, 2*np.pi, phase_starts, endpoint=False):
        for amp in amp_starts:
            for ac in (0.0, 0.5*amp, -0.5*amp):
                result = minimize(
                    objective,
                    np.array([phi, amp, ac], dtype=float),
                    method="L-BFGS-B",
                    bounds=[(0.0, 2*np.pi), (0.0, np.pi), (-np.pi, np.pi)],
                    options={"maxiter": args.signal_maxiter, "ftol": 1e-8, "gtol": 1e-5},
                )
                if best is None or result.fun < best.fun:
                    best = result
    phi0, As, Ac = best.x
    amp = float(np.hypot(As, Ac))
    phase = float(np.arctan2(Ac, As))
    return FitResult(
        A1=np.nan, A2=np.nan, C1=np.nan, C2=np.nan,
        phi0=float(phi0), As=float(As), Ac=float(Ac), amp=amp, phase=phase,
        logL=-float(best.fun), ntheta=len(delta_grid), f=float(frequency),
        converged=bool(best.success), optimizer_status=int(best.status),
        optimizer_message=str(best.message), optimizer_nit=int(best.nit),
        optimizer_nfev=int(best.nfev), optimizer_objective=float(best.fun),
    )


def main():
    args = parse_args()
    run_name = args.run_name.expanduser().resolve()
    output = args.output.expanduser().resolve()
    configure_logging(args.log_file.expanduser().resolve(), args.verbose)
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --resume or --overwrite")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")

    run_count = infer_run_count(run_name)
    run_stop = run_count if args.run_stop is None else args.run_stop
    rows = []
    start = args.run_start
    if args.resume and output.exists():
        existing = pd.read_pickle(output)
        rows = existing.reindex(columns=RESULT_COLUMNS).to_dict("records")
        start = max(start, len(rows))

    logging.info("Dataset: %s", run_name)
    logging.info("Runs: %d through %d (exclusive)", start, run_stop)
    logging.info(
        "Profiled pixel likelihood: bins=%d hermite=%d phase_grid=%d delta_grid=%d fit_widths=%s max_shots=%s",
        args.bins, args.hermite_order, args.phase_grid, args.delta_grid, args.fit_widths, args.max_shots,
    )
    job_start = time.perf_counter()
    for run_idx in range(start, run_stop):
        run_start = time.perf_counter()
        run_dir = run_name / f"run_{run_idx:03d}"
        z0_phase, z0_log = build_or_load_profiles(args, run_idx, "Z0", run_dir / "Z0" / "data_IMG.h5", args.psmap_z0)
        z100_phase, z100_log = build_or_load_profiles(args, run_idx, "Z100", run_dir / "Z100" / "data_IMG.h5", args.psmap_z100)
        if not np.allclose(z0_phase, z100_phase):
            raise ValueError("Z0 and Z100 profile phase grids differ")
        delta_grid = np.linspace(0.0, 2.0*np.pi, args.delta_grid, endpoint=False)
        ell_delta = combine_delta_profiles(z0_phase, z0_log, z100_log, delta_grid)
        result = fit_signal(delta_grid, ell_delta, args.frequency, args)
        rows.append(asdict(result))
        save_checkpoint(rows, output)
        logging.info(
            "[%d/%d] run_%03d complete in %.1fs | amp=%.8g phase=%.8g logL=%.8g converged=%s nfev=%d",
            run_idx + 1, run_stop, run_idx, time.perf_counter() - run_start,
            result.amp, result.phase, result.logL, result.converged, result.optimizer_nfev,
        )
    logging.info("Finished in %.1fs. Saved %s", time.perf_counter() - job_start, output)


if __name__ == "__main__":
    main()
