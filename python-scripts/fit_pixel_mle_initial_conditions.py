"""Pixelwise PSMAP MLE for Gaussian initial-condition parameters.

This is a shot-level baseline estimator.  For each state-resolved image it
maximizes the multinomial pixel likelihood under the PSMAP image model,
profiling the unknown shot phase phi0.  The four cloud widths can either be
fixed to known metadata or fitted as nuisance/target parameters.

The implementation keeps the PSMAP node arrays on CuPy when available and uses
GPU bincounts to aggregate expected probabilities into image pixels.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "helpers"))

from psmap_fisher import (  # noqa: E402
    PARAMETER_NAMES,
    _final_bin_indices,
    _normal_pdf,
    _port_probabilities,
    _psmap_nodes_and_state_probabilities,
    _trapezoid_weights,
)

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    from aispy.psmap import load_psmap
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO.parents[2] / "local" / "aispy"))
    from aispy.psmap import load_psmap


DEFAULT_DATA_ROOT = Path(
    "data/R20_N200_A100000000_muXStd10.0um_muVxStd10.0um_"
    "sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_"
    "phi0random_sig_A0.100_f0.3000"
)
DEFAULT_PSMAP = Path("output-files/PSGRID4D_CONFOCAL_Z0.h5")
T_DET = 3.8
TARGET_NAMES = PARAMETER_NAMES
H5_TARGET_KEYS = {
    "mu_x0": "mu_x0",
    "mu_y0": "mu_y0",
    "mu_vx0": "mu_vx0",
    "mu_vy0": "mu_vy0",
    "sigma_x0": "sigma_x",
    "sigma_y0": "sigma_y",
    "sigma_vx0": "sigma_vx",
    "sigma_vy0": "sigma_vy",
}


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


def to_float(value, xp) -> float:
    return float(value.get()) if xp is cp else float(value)


class NativePSMAPPixelLikelihood:
    """GPU-aware native-grid PSMAP multinomial pixel likelihood."""

    def __init__(self, psmap, t_det, x_edges, y_edges, use_gpu=True):
        self.xp = cp if (use_gpu and cp is not None) else np
        nodes, ground0, excited0 = _psmap_nodes_and_state_probabilities(psmap, phi0=0.0)
        _, ground90, excited90 = _psmap_nodes_and_state_probabilities(psmap, phi0=0.5 * np.pi)
        _, ground180, excited180 = _psmap_nodes_and_state_probabilities(psmap, phi0=np.pi)
        axes = [np.unique(nodes[:, index]) for index in range(4)]
        axis_weights = [_trapezoid_weights(axis) for axis in axes]
        indices = [np.searchsorted(axis, nodes[:, index]) for index, axis in enumerate(axes)]
        quad_weights = np.prod(
            np.column_stack([weight[index] for weight, index in zip(axis_weights, indices)]),
            axis=1,
        )
        bin_index, n_bins = _final_bin_indices(nodes, t_det, x_edges, y_edges)

        # p(phi) = p_mean + cos(phi)*p_cos + sin(phi)*p_sin.
        # Need three phase samples: p(pi/2) = p_mean - p_sin.
        prob_mean_g = 0.5 * (ground0 + ground180)
        prob_mean_e = 0.5 * (excited0 + excited180)
        prob_cos_g = 0.5 * (ground0 - ground180)
        prob_cos_e = 0.5 * (excited0 - excited180)
        prob_sin_g = prob_mean_g - ground90
        prob_sin_e = prob_mean_e - excited90

        xp = self.xp
        self.coordinates = xp.asarray(nodes, dtype=xp.float64)
        self.quad_weights = xp.asarray(quad_weights, dtype=xp.float64)
        self.bin_index = xp.asarray(bin_index, dtype=xp.int64)
        self.inside = self.bin_index >= 0
        self.n_bins = int(n_bins)
        self.prob_mean_g = xp.asarray(prob_mean_g, dtype=xp.float64)
        self.prob_mean_e = xp.asarray(prob_mean_e, dtype=xp.float64)
        self.prob_cos_g = xp.asarray(prob_cos_g, dtype=xp.float64)
        self.prob_cos_e = xp.asarray(prob_cos_e, dtype=xp.float64)
        self.prob_sin_g = xp.asarray(prob_sin_g, dtype=xp.float64)
        self.prob_sin_e = xp.asarray(prob_sin_e, dtype=xp.float64)
        self.backend = "cupy" if xp is cp else "numpy"

    def probabilities(self, theta, phi0):
        xp = self.xp
        theta = xp.asarray(theta, dtype=xp.float64)
        means = theta[:4]
        sigmas = theta[4:]
        if bool(to_float(xp.any(sigmas <= 0), xp)):
            raise ValueError("All widths must be positive")
        centered = self.coordinates - means[None, :]
        z = centered / sigmas[None, :]
        density = xp.exp(-0.5 * xp.sum(z * z, axis=1)) / (
            (2.0 * xp.pi) ** 2 * xp.prod(sigmas)
        )
        raw_weights = self.quad_weights * density
        grid_mass = xp.sum(raw_weights)
        weights = raw_weights / grid_mass

        c = np.cos(float(phi0))
        s = np.sin(float(phi0))
        pg_node = self.prob_mean_g + c * self.prob_cos_g + s * self.prob_sin_g
        pe_node = self.prob_mean_e + c * self.prob_cos_e + s * self.prob_sin_e
        inside = self.inside
        bins = self.bin_index[inside]
        ground = xp.bincount(
            bins,
            weights=(weights * xp.maximum(pg_node, 0.0))[inside],
            minlength=self.n_bins,
        )
        excited = xp.bincount(
            bins,
            weights=(weights * xp.maximum(pe_node, 0.0))[inside],
            minlength=self.n_bins,
        )
        detected = ground + excited
        outside = xp.maximum(1.0 - xp.sum(detected), 1e-300)
        return ground, excited, outside

    def nll(self, observed_ground, observed_excited, n_launched, theta, phi0, launched=True):
        xp = self.xp
        ground, excited, outside = self.probabilities(theta, phi0)
        eps = 1e-300
        if not launched:
            detected_probability = xp.maximum(xp.sum(ground) + xp.sum(excited), eps)
            ground = ground / detected_probability
            excited = excited / detected_probability
        ll = xp.sum(observed_ground * xp.log(xp.maximum(ground, eps)))
        ll += xp.sum(observed_excited * xp.log(xp.maximum(excited, eps)))
        if launched:
            n_outside = max(float(n_launched) - float(observed_ground.sum() + observed_excited.sum()), 0.0)
            ll += n_outside * xp.log(outside)
        return -to_float(ll, xp)


def image_edges(handle) -> np.ndarray:
    half_range = float(handle.attrs["image_half_range"])
    res = int(handle.attrs["image_res"])
    return np.linspace(-half_range, half_range, res + 1)


def image_moment_start(ground, excited, centers, metadata):
    total = ground + excited
    counts = max(float(total.sum()), 1.0)
    x = centers[:, None]
    y = centers[None, :]
    final_x = float((total * x).sum() / counts)
    final_y = float((total * y).sum() / counts)
    theta = np.array([metadata[name] for name in TARGET_NAMES], dtype=float)
    theta[0] = final_x - T_DET * theta[2]
    theta[1] = final_y - T_DET * theta[3]
    return theta


def pack(theta, phi0, fit_widths):
    if fit_widths:
        return np.r_[theta[:4], np.log(theta[4:]), phi0]
    return np.r_[theta[:4], phi0]


def unpack(params, width_values, fit_widths):
    if fit_widths:
        theta = np.r_[params[:4], np.exp(params[4:8])]
        phi0 = params[8]
    else:
        theta = np.r_[params[:4], width_values]
        phi0 = params[4]
    return theta, phi0


def fit_one_shot(evaluator, ground, excited, n_launched, metadata, centers, args):
    xp = evaluator.xp
    observed_ground = xp.asarray(ground.ravel().astype(np.float64))
    observed_excited = xp.asarray(excited.ravel().astype(np.float64))
    true_theta = np.array([metadata[name] for name in TARGET_NAMES], dtype=float)
    start_theta = true_theta.copy() if args.start == "true" else image_moment_start(ground, excited, centers, metadata)
    starts = [pack(start_theta, float(metadata["phi0"]), args.fit_widths)]
    if args.start != "true":
        for phase in np.linspace(0, 2 * np.pi, args.phase_starts, endpoint=False):
            starts.append(pack(start_theta, phase, args.fit_widths))

    mu_bound = args.mu_bound_um * 1e-6
    v_bound = args.v_bound_um_s * 1e-6
    width_values = true_theta[4:]
    if args.fit_widths:
        lower = np.r_[-mu_bound, -mu_bound, -v_bound, -v_bound, np.log(width_values * 0.25), -4 * np.pi]
        upper = np.r_[mu_bound, mu_bound, v_bound, v_bound, np.log(width_values * 4.0), 4 * np.pi]
    else:
        lower = np.r_[-mu_bound, -mu_bound, -v_bound, -v_bound, -4 * np.pi]
        upper = np.r_[mu_bound, mu_bound, v_bound, v_bound, 4 * np.pi]

    def objective(params):
        theta, phi0 = unpack(params, width_values, args.fit_widths)
        if np.any(theta[4:] <= 0):
            return 1e300
        value = evaluator.nll(
            observed_ground,
            observed_excited,
            n_launched,
            theta,
            phi0,
            launched=not args.detected_conditional,
        )
        return value if np.isfinite(value) else 1e300

    best = None
    for start in starts:
        result = minimize(
            objective,
            np.clip(start, lower, upper),
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": args.maxiter, "ftol": args.ftol, "gtol": args.gtol, "maxls": 20},
        )
        if best is None or result.fun < best.fun:
            best = result
    theta_hat, phi_hat = unpack(best.x, width_values, args.fit_widths)
    return theta_hat, phi_hat, best


def run(args) -> None:
    setup_logging(args.output_dir / "pixel_mle.log")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_root / f"run_{args.run:03d}" / args.port / "data_IMG.h5"
    logging.info("Loading image file %s", path)
    with h5py.File(path, "r") as handle:
        edges = image_edges(handle)
        centers = 0.5 * (edges[:-1] + edges[1:])
        n_launched = int(handle.attrs["n_atoms_launched"])
        psmap = load_psmap(str(args.psmap))
        evaluator = NativePSMAPPixelLikelihood(psmap, T_DET, edges, edges, use_gpu=not args.cpu)
        logging.info("Pixel likelihood backend: %s", evaluator.backend)
        rows = []
        n_shots = int(handle["images_s0"].shape[0])
        shot_ids = list(range(n_shots))[: args.max_shots]
        started = time.perf_counter()
        for count, shot_id in enumerate(shot_ids, start=1):
            metadata = {name: float(handle[H5_TARGET_KEYS[name]][shot_id]) for name in TARGET_NAMES}
            metadata["phi0"] = float(handle["phi0"][shot_id])
            theta_hat, phi_hat, opt = fit_one_shot(
                evaluator,
                np.asarray(handle["images_s0"][shot_id], dtype=np.float64),
                np.asarray(handle["images_s1"][shot_id], dtype=np.float64),
                n_launched,
                metadata,
                centers,
                args,
            )
            row = {
                "run": args.run,
                "port": args.port,
                "shot": shot_id,
                "phi0_hat": float(phi_hat),
                "phi0_true": metadata["phi0"],
                "nll": float(opt.fun),
                "success": bool(opt.success),
                "message": str(opt.message),
                "nfev": int(opt.nfev),
                "nit": int(opt.nit),
            }
            for name, value, truth in zip(TARGET_NAMES, theta_hat, [metadata[n] for n in TARGET_NAMES]):
                row[f"{name}_hat"] = float(value)
                row[f"{name}_true"] = float(truth)
                row[f"{name}_err"] = float(value - truth)
            rows.append(row)
            if count % args.log_every == 0 or count == len(shot_ids):
                elapsed = time.perf_counter() - started
                rmse_x = np.sqrt(np.mean([r["mu_x0_err"] ** 2 for r in rows])) * 1e6
                rmse_vx = np.sqrt(np.mean([r["mu_vx0_err"] ** 2 for r in rows])) * 1e6
                logging.info(
                    "Fitted %d/%d shots in %.1fs; RMSE mu_x0=%.3f um, mu_vx0=%.3f um/s",
                    count,
                    len(shot_ids),
                    elapsed,
                    rmse_x,
                    rmse_vx,
                )
    out_jsonl = args.output_dir / f"pixel_mle_run{args.run:03d}_{args.port}.jsonl"
    with out_jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    logging.info("Wrote %s", out_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--psmap", type=Path, default=DEFAULT_PSMAP)
    parser.add_argument("--port", default="Z0")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--max-shots", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/a1e8-pixel-mle-z0-known-widths"))
    parser.add_argument("--fit-widths", action="store_true", help="Fit sigma_x/y/vx/vy instead of fixing them to metadata")
    parser.add_argument("--detected-conditional", action="store_true", help="Drop launched/outside-count likelihood term")
    parser.add_argument("--start", choices=("true", "moments"), default="true")
    parser.add_argument("--phase-starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--ftol", type=float, default=1e-8)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--mu-bound-um", type=float, default=80.0)
    parser.add_argument("--v-bound-um-s", type=float, default=80.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
