#!/usr/bin/env python
"""
fit_surrogate_signal.py — End-to-end surrogate signal pipeline.

Given per-shot cloud nuisance estimates theta_hat (from profile_cloud_nuisances.py)
or the true theta, computes per-shot differential phase logL curves and fits the
gradiometer science signal.

Pipeline
--------
For each shot i:
  1. Load theta (estimated from JSONL, or true from H5).
  2. Evaluate pixel-level ACS = (A, Cc, Cs) per AI via SurrogatePixelACS.
  3. Compute logL_i(delta) = logsumexp_phi [ ll_Z0(phi) + ll_Z100(phi+delta) ] - log K
     using the cos/sin rotation trick — one ACS eval per shot, O(K*D) inner loop.
  4. Fit phi0 + As*sin(2pi*f*t) + Ac*cos(2pi*f*t) across shots.

Outputs
-------
  results/<stem>_<mode>_signal.jsonl   — per-run fit results (one row per run)
  results/<stem>_<mode>_delta_curves.npz  — per-shot logL(delta) arrays

Usage
-----
  # Oracle baseline: use true theta from H5
  python fit_surrogate_signal.py --use-true-theta

  # From estimated theta
  python fit_surrogate_signal.py \\
      --theta-file results/R80_N50_..._run000_cloud_mle.jsonl

  # Compare both in one shot
  python fit_surrogate_signal.py --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "helpers"))
sys.path.insert(0, str(REPO / "python-scripts"))

from helpers import ImageShotDataset  # noqa: E402
from profile_cloud_nuisances import (  # noqa: E402
    SurrogatePixelACS, H5_KEYS, PARAM_NAMES, T_DET,
    _downsample, _HAS_GPU,
)

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as _cp_lse
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

try:
    from aispy.psmap import load_psmap, PSMAPSurrogate
except ImportError:
    sys.path.insert(0, str(REPO.parents[1] / "local" / "aispy"))
    from aispy.psmap import load_psmap, PSMAPSurrogate

DEFAULT_RUN = (
    REPO / "data"
    / "R80_N50_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx309um_"
      "sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"
)
DEFAULT_PSMAP_Z0   = REPO / "output-files" / "PSGRID4D_CONFOCAL_Z0.h5"
DEFAULT_PSMAP_Z100 = REPO / "output-files" / "PSGRID4D_CONFOCAL_Z100.h5"


# ── per-shot logL(delta) ───────────────────────────────────────────────────────

def _logL_delta_grid(n_g0, n_e0, acs0, n_g1, n_e1, acs1,
                     delta_grid, n_theta, xp, lse_fn):
    """
    Compute logL(delta) for a single shot on a grid of delta values.

    Uses the rotation identity:
        cos(phi + delta) = cos(phi)*cos(delta) - sin(phi)*sin(delta)
        sin(phi + delta) = sin(phi)*cos(delta) + cos(phi)*sin(delta)

    So the effective Z100 ACS coefficients at phi+delta are:
        Cc_eff = Cc*cos(delta) + Cs*sin(delta)
        Cs_eff = -Cc*sin(delta) + Cs*cos(delta)

    Z0 logL is computed once; Z100 logL is computed with rotated coefficients
    for each delta. O(K * D * n_bins) total, no extra GPU memory allocation.

    Returns numpy array of shape (D,).
    """
    A_g0, Cc_g0, Cs_g0, A_e0, Cc_e0, Cs_e0 = acs0
    A_g1, Cc_g1, Cs_g1, A_e1, Cc_e1, Cs_e1 = acs1

    tot0 = float((A_g0 + A_e0).sum()); tot1 = float((A_g1 + A_e1).sum())
    L0 = float((n_g0 + n_e0).sum()) / max(tot0, 1e-300)
    L1 = float((n_g1 + n_e1).sum()) / max(tot1, 1e-300)

    phi   = xp.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)  # (K,)
    c_phi = xp.cos(phi)
    s_phi = xp.sin(phi)

    # Z0 log-likelihood at each phi: (K,)
    lam_g0 = L0 * xp.maximum(
        A_g0[None] + Cc_g0[None]*c_phi[:, None] + Cs_g0[None]*s_phi[:, None], 1e-300)
    lam_e0 = L0 * xp.maximum(
        A_e0[None] + Cc_e0[None]*c_phi[:, None] + Cs_e0[None]*s_phi[:, None], 1e-300)
    ll0 = (n_g0[None]*xp.log(lam_g0) - lam_g0
           + n_e0[None]*xp.log(lam_e0) - lam_e0).sum(1)  # (K,)

    results = np.empty(len(delta_grid))
    log_K   = np.log(n_theta)
    for j, delta in enumerate(delta_grid):
        cd = float(np.cos(delta)); sd = float(np.sin(delta))
        Cc_g_r = Cc_g1*cd + Cs_g1*sd
        Cs_g_r = -Cc_g1*sd + Cs_g1*cd
        Cc_e_r = Cc_e1*cd + Cs_e1*sd
        Cs_e_r = -Cc_e1*sd + Cs_e1*cd
        lam_g1 = L1 * xp.maximum(
            A_g1[None] + Cc_g_r[None]*c_phi[:, None] + Cs_g_r[None]*s_phi[:, None], 1e-300)
        lam_e1 = L1 * xp.maximum(
            A_e1[None] + Cc_e_r[None]*c_phi[:, None] + Cs_e_r[None]*s_phi[:, None], 1e-300)
        ll1 = (n_g1[None]*xp.log(lam_g1) - lam_g1
               + n_e1[None]*xp.log(lam_e1) - lam_e1).sum(1)
        results[j] = float(lse_fn(ll0 + ll1) - log_K)

    return results


# ── signal fit ─────────────────────────────────────────────────────────────────

def _interp_periodic(grid, values, points):
    period = 2.0 * np.pi
    x = np.r_[grid, period]
    y = np.r_[values, values[0]]
    return np.interp(np.mod(points, period), x, y)


def _fit_signal(delta_grid, ell_delta, t_shots, frequency,
                n_starts=8, amp_starts=(0.03, 0.07, 0.10), maxiter=500):
    """
    Maximise sum_i logL_i(delta_i(phi0, As, Ac)) over (phi0, As, Ac).

    delta_i = phi0 + As*sin(2pi*f*t_i) + Ac*cos(2pi*f*t_i)

    Returns dict with hat estimates and optimizer metadata.
    """
    def shot_ells(delta_vals):
        return np.array([
            _interp_periodic(delta_grid, ell_delta[i], delta_vals[i])
            for i in range(len(t_shots))
        ])

    def objective(params):
        phi0, As, Ac = params
        dphi = phi0 + As*np.sin(2*np.pi*frequency*t_shots) + Ac*np.cos(2*np.pi*frequency*t_shots)
        val = float(np.sum(shot_ells(dphi)))
        return -val if np.isfinite(val) else 1e300

    best = None
    for phi_start in np.linspace(0, 2*np.pi, n_starts, endpoint=False):
        for amp in amp_starts:
            for ac in (0.0, 0.5*amp, -0.5*amp):
                res = minimize(
                    objective, [phi_start, float(amp), ac], method="L-BFGS-B",
                    bounds=[(0.0, 2*np.pi), (0.0, np.pi), (-np.pi, np.pi)],
                    options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-5},
                )
                if best is None or res.fun < best.fun:
                    best = res

    phi0_h, As_h, Ac_h = best.x
    return {
        "phi0_hat":   float(phi0_h),
        "As_hat":     float(As_h),
        "Ac_hat":     float(Ac_h),
        "amp_hat":    float(np.hypot(As_h, Ac_h)),
        "phase_hat":  float(np.arctan2(Ac_h, As_h)),
        "logL":       -float(best.fun),
        "converged":  bool(best.success),
        "nit":        int(best.nit),
        "nfev":       int(best.nfev),
        "message":    str(best.message),
    }


# ── theta sources ──────────────────────────────────────────────────────────────

def _load_true_theta(ds_z0, ds_z100, shot_ids):
    """Load true per-shot theta from H5 metadata."""
    thetas_z0   = []
    thetas_z100 = []
    for sid in shot_ids:
        meta_z0   = ds_z0.meta(sid)
        meta_z100 = ds_z100.meta(sid)
        thetas_z0.append(np.array([meta_z0[H5_KEYS[n]]   for n in PARAM_NAMES]))
        thetas_z100.append(np.array([meta_z100[H5_KEYS[n]] for n in PARAM_NAMES]))
    return thetas_z0, thetas_z100


def _load_estimated_theta(jsonl_path, shot_ids):
    """Load per-shot theta_hat from a JSONL produced by profile_cloud_nuisances."""
    by_shot = {}
    with open(jsonl_path) as fh:
        for line in fh:
            row = json.loads(line)
            by_shot[int(row["shot"])] = row

    thetas_z0, thetas_z100 = [], []
    for sid in shot_ids:
        if sid not in by_shot:
            raise KeyError(f"Shot {sid} not found in {jsonl_path}")
        row = by_shot[sid]
        thetas_z0.append(np.array([row[f"{n}_z0_hat"]   for n in PARAM_NAMES]))
        thetas_z100.append(np.array([row[f"{n}_z100_hat"] for n in PARAM_NAMES]))
    return thetas_z0, thetas_z100


# ── per-run signal estimation ──────────────────────────────────────────────────

def _run_one(run_dir, eval_z0, eval_z100, bins, edges, n_theta, delta_grid,
             mode, theta_file, max_shots, xp, lse_fn, args):
    """
    Estimate signal parameters for a single run directory.

    mode: 'true' or 'estimated'
    Returns dict of fit result + metadata, and (delta_grid, ell_delta) for checkpointing.
    """
    ds_z0   = ImageShotDataset(str(run_dir / "Z0"   / "data_IMG.h5"))
    ds_z100 = ImageShotDataset(str(run_dir / "Z100" / "data_IMG.h5"))
    n_shots = ds_z0.n_shots
    if max_shots is not None:
        n_shots = min(n_shots, max_shots)
    shot_ids = list(range(n_shots))

    # Load true signal parameters from H5 attrs
    with h5py.File(run_dir / "Z0" / "data_IMG.h5") as f:
        sig_amp   = float(f.attrs.get("signal_amp",   np.nan))
        sig_freq  = float(f.attrs.get("signal_freq",  np.nan))
        sig_phase = float(f.attrs.get("signal_phase", np.nan))
        true_delta_phi = f["delta_phi"][:n_shots]  # (n_shots,)

    if mode == "true":
        thetas_z0, thetas_z100 = _load_true_theta(ds_z0, ds_z100, shot_ids)
    else:
        thetas_z0, thetas_z100 = _load_estimated_theta(theta_file, shot_ids)

    ell_delta = np.empty((n_shots, len(delta_grid)))
    t0 = time.perf_counter()
    for i, sid in enumerate(shot_ids):
        img_z0   = _downsample(ds_z0[sid],   bins)  # (2, bins, bins)
        img_z100 = _downsample(ds_z100[sid], bins)
        n_g0, n_e0   = xp.asarray(img_z0[0].ravel()),   xp.asarray(img_z0[1].ravel())
        n_g1, n_e1   = xp.asarray(img_z100[0].ravel()), xp.asarray(img_z100[1].ravel())

        acs0 = eval_z0.pixel_acs(thetas_z0[i])    # 6-tuple of (n_bins,) GPU arrays
        acs1 = eval_z100.pixel_acs(thetas_z100[i])

        ell_delta[i] = _logL_delta_grid(
            n_g0, n_e0, acs0, n_g1, n_e1, acs1,
            delta_grid, n_theta, xp, lse_fn,
        )

        if (i + 1) % 10 == 0 or i + 1 == n_shots:
            logging.info("  [%s] %d/%d shots  %.1fs",
                         mode, i + 1, n_shots, time.perf_counter() - t0)

    t_shots = np.arange(n_shots, dtype=float)
    fit = _fit_signal(delta_grid, ell_delta, t_shots, sig_freq,
                      n_starts=args.signal_starts,
                      amp_starts=args.amp_starts,
                      maxiter=args.signal_maxiter)

    # True signal decomposition: dphi_i = phi0 + As*sin + Ac*cos at t=i
    # We have true per-shot delta_phi; extract As, Ac by least-squares fit
    M = np.c_[
        np.ones(n_shots),
        np.sin(2*np.pi*sig_freq*t_shots),
        np.cos(2*np.pi*sig_freq*t_shots),
    ]
    coeff, *_ = np.linalg.lstsq(M, true_delta_phi, rcond=None)
    phi0_true, As_true, Ac_true = coeff
    amp_true   = float(np.hypot(As_true, Ac_true))
    phase_true = float(np.arctan2(Ac_true, As_true))

    result = {
        "mode": mode,
        "run_dir": str(run_dir),
        "n_shots": n_shots,
        # truth
        "signal_amp_true":   sig_amp,
        "signal_freq":       sig_freq,
        "signal_phase_true": sig_phase,
        "phi0_true":         float(phi0_true),
        "As_true":           float(As_true),
        "Ac_true":           float(Ac_true),
        "amp_true":          amp_true,
        "phase_true":        phase_true,
        # estimates
        **fit,
        # errors
        "As_err":    fit["As_hat"]    - float(As_true),
        "Ac_err":    fit["Ac_hat"]    - float(Ac_true),
        "amp_err":   fit["amp_hat"]   - amp_true,
        "phase_err": fit["phase_hat"] - phase_true,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    return result, ell_delta


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_name", nargs="?", type=Path, default=DEFAULT_RUN)
    p.add_argument("--run-idx",   type=int, default=0,
                   help="Run index within the dataset. (default: 0)")
    p.add_argument("--psmap-z0",   type=Path, default=DEFAULT_PSMAP_Z0)
    p.add_argument("--psmap-z100", type=Path, default=DEFAULT_PSMAP_Z100)
    # Theta source
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--use-true-theta", action="store_true",
                     help="Oracle mode: use true cloud params from H5.")
    grp.add_argument("--theta-file", type=Path, default=None,
                     help="JSONL of theta_hat from profile_cloud_nuisances.py.")
    grp.add_argument("--compare", action="store_true",
                     help="Run both true-theta and estimated-theta and compare.")
    # Grid / quadrature
    p.add_argument("--bins",        type=int,   default=32)
    p.add_argument("--ntheta",      type=int,   default=512,
                   help="phi grid size for logsumexp marginalisation.")
    p.add_argument("--delta-grid",  type=int,   default=256,
                   help="Number of delta points in logL(delta) curves.")
    p.add_argument("--n-quad",      type=int,   default=20_000,
                   help="QMC quadrature points for ACS evaluation.")
    p.add_argument("--max-shots",   type=int,   default=None)
    # Signal fit
    p.add_argument("--frequency",   type=float, default=0.3)
    p.add_argument("--signal-starts", type=int, default=8)
    p.add_argument("--amp-starts",  type=float, nargs="+", default=[0.03, 0.07, 0.10])
    p.add_argument("--signal-maxiter", type=int, default=500)
    # Output
    p.add_argument("--output-dir",  type=Path,  default=REPO / "results")
    p.add_argument("--no-gpu",      action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _setup_logging(verbose):
    (REPO / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


def main():
    args = parse_args()
    _setup_logging(args.verbose)

    use_gpu = _HAS_GPU and _HAS_CUPY and not args.no_gpu
    xp      = cp if use_gpu else np
    lse_fn  = _cp_lse if use_gpu else \
              __import__("scipy.special", fromlist=["logsumexp"]).logsumexp
    logging.info("GPU=%s  bins=%d  ntheta=%d  delta_grid=%d  n_quad=%d",
                 use_gpu, args.bins, args.ntheta, args.delta_grid, args.n_quad)

    run_path = Path(args.run_name).expanduser().resolve()
    run_dir  = run_path / f"run_{args.run_idx:03d}"
    run_stem = run_path.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build edges and evaluators
    # Read half_range from the Z0 dataset
    ds_tmp = ImageShotDataset(str(run_dir / "Z0" / "data_IMG.h5"))
    edges  = np.linspace(-ds_tmp.half_range, ds_tmp.half_range, args.bins + 1)
    del ds_tmp

    logging.info("Loading PSMAPs ...")
    sur_z0   = PSMAPSurrogate(load_psmap(str(args.psmap_z0)),   T_DET, use_gpu=use_gpu)
    sur_z100 = PSMAPSurrogate(load_psmap(str(args.psmap_z100)), T_DET, use_gpu=use_gpu)
    eval_z0   = SurrogatePixelACS(sur_z0,   T_DET, edges, edges, n_quad=args.n_quad)
    eval_z100 = SurrogatePixelACS(sur_z100, T_DET, edges, edges, n_quad=args.n_quad)
    logging.info("Evaluators ready  n_bins=%d", eval_z0.n_bins)

    delta_grid = np.linspace(0.0, 2.0 * np.pi, args.delta_grid, endpoint=False)

    # Determine which modes to run
    modes = []
    if args.compare:
        # auto-find theta file if not provided
        auto_theta = args.output_dir / f"{run_stem}_run{args.run_idx:03d}_cloud_mle.jsonl"
        if not auto_theta.exists():
            raise FileNotFoundError(
                f"--compare requires {auto_theta}; run profile_cloud_nuisances.py first.")
        modes = [("true", None), ("estimated", auto_theta)]
    elif args.use_true_theta:
        modes = [("true", None)]
    elif args.theta_file is not None:
        modes = [("estimated", args.theta_file)]
    else:
        # Default: try estimated, fall back to true
        auto_theta = args.output_dir / f"{run_stem}_run{args.run_idx:03d}_cloud_mle.jsonl"
        if auto_theta.exists():
            logging.info("Found theta file %s; using estimated theta.", auto_theta)
            modes = [("estimated", auto_theta)]
        else:
            logging.info("No theta file found; falling back to true theta (oracle).")
            modes = [("true", None)]

    results = []
    for mode, theta_file in modes:
        logging.info("=== Mode: %s ===", mode)
        result, ell_delta = _run_one(
            run_dir, eval_z0, eval_z100, args.bins, edges,
            args.ntheta, delta_grid, mode, theta_file,
            args.max_shots, xp, lse_fn, args,
        )
        results.append(result)

        # Save per-shot logL(delta) curves
        npz_path = args.output_dir / f"{run_stem}_run{args.run_idx:03d}_{mode}_delta_curves.npz"
        np.savez_compressed(npz_path, delta_grid=delta_grid, ell_delta=ell_delta)
        logging.info("Saved delta curves -> %s", npz_path)

        logging.info(
            "[%s] amp: true=%.4f hat=%.4f err=%+.4f | "
            "phase: true=%.4f hat=%.4f err=%+.4f | logL=%.1f converged=%s",
            mode,
            result["amp_true"], result["amp_hat"], result["amp_err"],
            result["phase_true"], result["phase_hat"], result["phase_err"],
            result["logL"], result["converged"],
        )

    # Write JSONL summary
    jsonl_path = args.output_dir / f"{run_stem}_run{args.run_idx:03d}_signal.jsonl"
    with jsonl_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    logging.info("Wrote signal results -> %s", jsonl_path)

    # Print comparison if both modes ran
    if len(results) == 2:
        r_true, r_est = results
        print("\n=== Signal recovery comparison ===")
        print(f"{'':15s}  {'true_theta':>12s}  {'est_theta':>12s}  {'delta':>10s}")
        for key in ("amp_hat", "amp_err", "phase_hat", "phase_err", "As_hat", "Ac_hat"):
            vt = r_true.get(key, float("nan"))
            ve = r_est.get(key, float("nan"))
            print(f"  {key:15s}  {vt:12.6f}  {ve:12.6f}  {ve-vt:+10.6f}")
        print(f"\n  True amp={r_true['amp_true']:.4f}  phase={r_true['phase_true']:.4f}")


if __name__ == "__main__":
    main()
