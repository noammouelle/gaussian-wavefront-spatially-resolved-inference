#!/usr/bin/env python3
"""
diagnose_eta_profiles.py — Per-shot likelihood profiles over cloud nuisance parameters.

For a single shot, sweeps each component of eta_iz (initial cloud parameters)
independently and plots the phase-marginalised log-likelihood as a function of
each parameter.  No science signal is assumed (As = Ac = 0).

Because the two AIs share the same common phase phi_i, the full per-shot
marginal is:

    log L_i(eta_z0, eta_z100)
        = logsumexp_k [ log L_z0(phi_k, eta_z0) + log L_z100(phi_k, eta_z100) ]
          - log(n_theta)

We sweep one parameter of eta_z0 (or eta_z100) at a time, fixing everything
else at the values stored in the generation metadata (the "true" values).

The resulting plots show whether the likelihood is peaked (informative about
the nuisance) or flat (non-identifiable from the image data).

Usage:
    python python-scripts/diagnose_eta_profiles.py \\
        [--run RUN]           run index (default 0)
        [--shot SHOT]         shot index within run (default 0)
        [--bins BINS]         image downsampling bins (default 32)
        [--hermite HERMITE]   GH quadrature order (default 5)
        [--ntheta NTHETA]     phase grid size (default 512)
        [--npoints NPOINTS]   sweep points per parameter (default 25)
        [--nsigma NSIGMA]     sweep half-width in units of param value (default 0.5)
        [--out OUT]           output figure path (default diagnose_eta_profiles.pdf)
        [--run-name RUN_NAME] dataset run name (default R80_N50_A1000000_...)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
AISPY_ROOT = REPO_ROOT.parents[1] / "local" / "aispy"
for p in [str(REPO_ROOT / "helpers"), str(REPO_ROOT), str(AISPY_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aispy.psmap import load_psmap         # noqa: E402
from helpers import ImageShotDataset       # noqa: E402
from psmap_fisher import PSMAPConditionalImageModel  # noqa: E402

T_DET = 3.8  # seconds

# Parameter names and display labels (8-vector order matches theta)
PARAM_NAMES  = ["mu_x0", "mu_y0", "mu_vx0", "mu_vy0",
                "sigma_x0", "sigma_y0", "sigma_vx0", "sigma_vy0"]
PARAM_LABELS = [r"$\mu_{x_0}$ [mm]", r"$\mu_{y_0}$ [mm]",
                r"$\mu_{v_x}$ [mm/s]", r"$\mu_{v_y}$ [mm/s]",
                r"$\sigma_x$ [mm]", r"$\sigma_y$ [mm]",
                r"$\sigma_{v_x}$ [mm/s]", r"$\sigma_{v_y}$ [mm/s]"]
SCALE        = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]   # m → mm, m/s → mm/s


# ── helpers ───────────────────────────────────────────────────────────────────

def downsample(image: np.ndarray, bins: int) -> np.ndarray:
    """Block-sum image from (res, res) to (bins, bins)."""
    res  = image.shape[0]
    step = res // bins
    out  = image.reshape(bins, step, bins, step).sum(axis=(1, 3))
    return out.astype(np.float64)


def build_models(psmap, t_det: float, x_edges, y_edges, hermite_order: int):
    """
    Build PSMAPConditionalImageModel at phi0 = 0, pi/2, pi.

    These three models are built once per AI and reused across all theta
    evaluations — only detected_probabilities(theta) is called per sweep point.
    """
    m0   = PSMAPConditionalImageModel.from_psmap(psmap, t_det, 0.0,        x_edges, y_edges, hermite_order)
    m90  = PSMAPConditionalImageModel.from_psmap(psmap, t_det, np.pi / 2,  x_edges, y_edges, hermite_order)
    m180 = PSMAPConditionalImageModel.from_psmap(psmap, t_det, np.pi,      x_edges, y_edges, hermite_order)
    return m0, m90, m180


def pixel_stats(m0, m90, m180, theta: np.ndarray):
    """
    3-point phase extraction.

    Returns A, Cc, Cs each of shape (2*n_pix,).  The expected per-atom
    detection probability at phase phi is A + Cc*cos(phi) + Cs*sin(phi).
    """
    p0   = m0.detected_probabilities(theta)
    p90  = m90.detected_probabilities(theta)
    p180 = m180.detected_probabilities(theta)
    A    = 0.5 * (p0 + p180)
    Cc   = 0.5 * (p0 - p180)
    Cs   = p90 - A
    return A, Cc, Cs


def per_shot_logL(
    n_z0:   np.ndarray,   # (2*n_pix,) observed counts, Z0
    n_z100: np.ndarray,   # (2*n_pix,) observed counts, Z100
    A_z0:   np.ndarray, Cc_z0: np.ndarray, Cs_z0: np.ndarray,
    A_z100: np.ndarray, Cc_z100: np.ndarray, Cs_z100: np.ndarray,
    n_theta: int = 512,
) -> float:
    """
    Phase-marginalised per-shot Poisson log-likelihood for both AIs jointly.

    The two AIs share the common phase phi_i, so we must sum their
    conditional log-likelihoods BEFORE the logsumexp over phi:

        log L_i = logsumexp_k [ LL_z0(phi_k) + LL_z100(phi_k) ] - log(n_theta)

    Atom-number normalisation: Lambda_z = N^+_z / sum(A_z), which matches
    the observed total count at every phi (to first order in the fringe).
    """
    N_z0   = n_z0.sum()
    N_z100 = n_z100.sum()

    # Normalise so that Lambda * sum(A) = N^+ (phi-averaged expected total = observed total)
    Lambda_z0   = N_z0   / np.maximum(A_z0.sum(),   1e-300)
    Lambda_z100 = N_z100 / np.maximum(A_z100.sum(), 1e-300)

    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ll_phi   = np.empty(n_theta)

    for k, phi in enumerate(phi_grid):
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        # Z0 conditional log-likelihood at phi_k
        p_z0  = np.maximum(A_z0  + Cc_z0  * cos_phi + Cs_z0  * sin_phi, 1e-300)
        lam_z0 = Lambda_z0  * p_z0
        ll_z0  = np.dot(n_z0,   np.log(np.maximum(lam_z0,  1e-300))) - lam_z0.sum()

        # Z100 conditional log-likelihood at phi_k
        p_z100 = np.maximum(A_z100 + Cc_z100 * cos_phi + Cs_z100 * sin_phi, 1e-300)
        lam_z100 = Lambda_z100 * p_z100
        ll_z100  = np.dot(n_z100, np.log(np.maximum(lam_z100, 1e-300))) - lam_z100.sum()

        ll_phi[k] = ll_z0 + ll_z100

    return float(logsumexp(ll_phi) - np.log(n_theta))


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep_parameter(
    param_idx: int,
    theta_true_z0: np.ndarray,
    theta_true_z100: np.ndarray,
    n_z0: np.ndarray,
    n_z100: np.ndarray,
    models_z0: tuple,
    models_z100: tuple,
    n_points: int,
    n_sigma: float,
    n_theta: int,
    sweep_z: int = 0,           # which AI to sweep (0 = Z0, 100 = Z100)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep one parameter of eta (for the chosen AI) over n_sigma * true_value
    on either side, returning (param_values, delta_logL).

    The other AI is held fixed at its true theta throughout.
    delta_logL is zero at the true value by construction of the grid centre.
    """
    theta_z0   = theta_true_z0.copy()
    theta_z100 = theta_true_z100.copy()

    # Centre and half-width for sweep
    true_val = (theta_true_z0 if sweep_z == 0 else theta_true_z100)[param_idx]
    # For positions/velocities: sweep ± n_sigma * |true_val| or ± 0.5 mm if near zero
    # For spreads: sweep ± n_sigma * true_val
    half_width = max(abs(true_val) * n_sigma, 5e-5)   # at least 50 µm / 50 µm/s

    vals   = np.linspace(true_val - half_width, true_val + half_width, n_points)
    ll_arr = np.empty(n_points)

    # Pre-compute pixel stats for the FIXED AI
    if sweep_z == 0:
        A_fix, Cc_fix, Cs_fix = pixel_stats(*models_z100, theta_true_z100)
    else:
        A_fix, Cc_fix, Cs_fix = pixel_stats(*models_z0, theta_true_z0)

    for j, v in enumerate(vals):
        if sweep_z == 0:
            theta_z0[param_idx] = v
            # Enforce positivity for spread parameters (indices 4–7)
            if param_idx >= 4 and theta_z0[param_idx] <= 0:
                ll_arr[j] = np.nan
                continue
            A_sw, Cc_sw, Cs_sw = pixel_stats(*models_z0, theta_z0)
            ll_arr[j] = per_shot_logL(n_z0, n_z100,
                                      A_sw, Cc_sw, Cs_sw,
                                      A_fix, Cc_fix, Cs_fix,
                                      n_theta)
        else:
            theta_z100[param_idx] = v
            if param_idx >= 4 and theta_z100[param_idx] <= 0:
                ll_arr[j] = np.nan
                continue
            A_sw, Cc_sw, Cs_sw = pixel_stats(*models_z100, theta_z100)
            ll_arr[j] = per_shot_logL(n_z0, n_z100,
                                      A_fix, Cc_fix, Cs_fix,
                                      A_sw, Cc_sw, Cs_sw,
                                      n_theta)

    # delta log L relative to true value
    ll_true  = per_shot_logL(n_z0, n_z100,
                             *pixel_stats(*models_z0,   theta_true_z0),
                             *pixel_stats(*models_z100, theta_true_z100),
                             n_theta)
    delta_ll = ll_arr - ll_true

    return vals, delta_ll


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_profiles(results_z0, results_z100, theta_true_z0, theta_true_z100,
                  shot_idx: int, run_idx: int, out_path: str) -> None:
    """
    Two 8-panel figures (one per AI), each showing delta log L vs each
    parameter.  True value is marked with a vertical dashed line.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f"Likelihood profiles — Z0   (run {run_idx:03d}, shot {shot_idx})",
                 fontsize=12)
    for i, ax in enumerate(axes.ravel()):
        vals, dll = results_z0[i]
        sc = SCALE[i]
        ax.plot(vals * sc, dll, "b-", lw=1.5)
        ax.axvline(theta_true_z0[i] * sc, color="r", ls="--", lw=1.2, label="true")
        ax.axhline(-0.5, color="gray", ls=":", lw=1.0, label=r"$-\frac{1}{2}$ (1σ)")
        ax.set_xlabel(PARAM_LABELS[i], fontsize=9)
        ax.set_ylabel(r"$\Delta\log L$" if i % 4 == 0 else "", fontsize=9)
        ax.set_title(PARAM_NAMES[i], fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    path_z0 = out_path.replace(".pdf", "_Z0.pdf")
    fig.savefig(path_z0, dpi=150)
    print(f"Saved {path_z0}")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f"Likelihood profiles — Z100  (run {run_idx:03d}, shot {shot_idx})",
                 fontsize=12)
    for i, ax in enumerate(axes.ravel()):
        vals, dll = results_z100[i]
        sc = SCALE[i]
        ax.plot(vals * sc, dll, "b-", lw=1.5)
        ax.axvline(theta_true_z100[i] * sc, color="r", ls="--", lw=1.2, label="true")
        ax.axhline(-0.5, color="gray", ls=":", lw=1.0, label=r"$-\frac{1}{2}$ (1σ)")
        ax.set_xlabel(PARAM_LABELS[i], fontsize=9)
        ax.set_ylabel(r"$\Delta\log L$" if i % 4 == 0 else "", fontsize=9)
        ax.set_title(PARAM_NAMES[i], fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    path_z100 = out_path.replace(".pdf", "_Z100.pdf")
    fig.savefig(path_z100, dpi=150)
    print(f"Saved {path_z100}")
    plt.close(fig)


def print_summary(results_z0, results_z100, theta_true_z0, theta_true_z100) -> None:
    """
    Print the approximate width (1σ equivalent) for each parameter
    by finding where delta_logL crosses -0.5.
    """
    def width_at_half(vals, dll):
        """Width of the region where dll > -0.5, in physical units."""
        above = vals[dll > -0.5]
        if len(above) < 2:
            return float("nan")
        return above[-1] - above[0]

    print("\n--- Likelihood profile widths (1σ-equivalent, delta_logL > -0.5) ---")
    print(f"{'Param':>12}  {'True_Z0':>12}  {'Width_Z0':>12}  {'True_Z100':>12}  {'Width_Z100':>12}")
    for i in range(8):
        vals_z0,  dll_z0  = results_z0[i]
        vals_z100, dll_z100 = results_z100[i]
        sc = SCALE[i]
        unit = "mm" if i < 4 else "mm" if i < 6 else "mm/s"
        w_z0   = width_at_half(vals_z0,   dll_z0)   * sc
        w_z100 = width_at_half(vals_z100, dll_z100) * sc
        tv_z0   = theta_true_z0[i]   * sc
        tv_z100 = theta_true_z100[i] * sc
        print(f"{PARAM_NAMES[i]:>12}  {tv_z0:>12.4f}  {w_z0:>12.4f}  {tv_z100:>12.4f}  {w_z100:>12.4f}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",      type=int, default=0)
    p.add_argument("--shot",     type=int, default=0)
    p.add_argument("--bins",     type=int, default=32)
    p.add_argument("--hermite",  type=int, default=5)
    p.add_argument("--ntheta",   type=int, default=512)
    p.add_argument("--npoints",  type=int, default=25,
                   help="sweep points per parameter")
    p.add_argument("--nsigma",   type=float, default=0.5,
                   help="sweep half-width as fraction of true value")
    p.add_argument("--out",      type=str, default="diagnose_eta_profiles.pdf")
    p.add_argument("--run-name", type=str,
                   default="R80_N50_A1000000_sig0.0_f0.3_ph0.0")
    p.add_argument("--psmap-z0",  type=str,
                   default=str(REPO_ROOT / "output-files" / "PSGRID4D_CONFOCAL_Z0.h5"))
    p.add_argument("--psmap-z100", type=str,
                   default=str(REPO_ROOT / "output-files" / "PSGRID4D_CONFOCAL_Z100.h5"))
    return p.parse_args()


def main():
    args  = parse_args()

    data_root = REPO_ROOT / "data" / args.run_name / f"run_{args.run:03d}"
    ds_z0   = ImageShotDataset(str(data_root / "Z0"   / "data_IMG.h5"))
    ds_z100 = ImageShotDataset(str(data_root / "Z100" / "data_IMG.h5"))

    shot = args.shot
    print(f"Run {args.run:03d}, shot {shot}  |  bins={args.bins}, "
          f"hermite={args.hermite}, n_theta={args.ntheta}")

    # ── true cloud parameters from metadata ──────────────────────────────────
    meta_z0   = ds_z0.meta(shot)
    meta_z100 = ds_z100.meta(shot)

    def meta_to_theta(meta):
        return np.array([meta["mu_x0"],  meta["mu_y0"],
                         meta["mu_vx0"], meta["mu_vy0"],
                         meta["sigma_x"],  meta["sigma_y"],
                         meta["sigma_vx"], meta["sigma_vy"]])

    theta_true_z0   = meta_to_theta(meta_z0)
    theta_true_z100 = meta_to_theta(meta_z100)

    print("True theta Z0  :", np.array2string(theta_true_z0,   precision=4))
    print("True theta Z100:", np.array2string(theta_true_z100, precision=4))

    # ── pixel grid for the downsampled image ─────────────────────────────────
    bins     = args.bins
    x_edges  = np.linspace(-ds_z0.half_range, ds_z0.half_range, bins + 1)
    y_edges  = x_edges.copy()

    # ── load PSMAPs and build models (once) ──────────────────────────────────
    print("Loading PSMAPs …")
    psmap_z0   = load_psmap(args.psmap_z0)
    psmap_z100 = load_psmap(args.psmap_z100)

    print("Building conditional image models (6 total) …")
    models_z0   = build_models(psmap_z0,   T_DET, x_edges, y_edges, args.hermite)
    models_z100 = build_models(psmap_z100, T_DET, x_edges, y_edges, args.hermite)
    print("  done.")

    # ── load and downsample one shot ─────────────────────────────────────────
    img_z0   = ds_z0[shot]    # (2, res, res) uint16
    img_z100 = ds_z100[shot]

    s0_z0   = downsample(img_z0[0],   bins)
    s1_z0   = downsample(img_z0[1],   bins)
    s0_z100 = downsample(img_z100[0], bins)
    s1_z100 = downsample(img_z100[1], bins)

    n_z0   = np.concatenate([s0_z0.ravel(),   s1_z0.ravel()])    # (2*n_pix,)
    n_z100 = np.concatenate([s0_z100.ravel(), s1_z100.ravel()])

    print(f"Shot {shot}: N_z0 = {int(n_z0.sum())}, N_z100 = {int(n_z100.sum())}")

    # ── log L at true params ──────────────────────────────────────────────────
    A_z0_true, Cc_z0_true, Cs_z0_true     = pixel_stats(*models_z0,   theta_true_z0)
    A_z100_true, Cc_z100_true, Cs_z100_true = pixel_stats(*models_z100, theta_true_z100)
    ll_true = per_shot_logL(n_z0, n_z100,
                            A_z0_true, Cc_z0_true, Cs_z0_true,
                            A_z100_true, Cc_z100_true, Cs_z100_true,
                            args.ntheta)
    print(f"log L at true params: {ll_true:.2f}")

    # ── parameter sweeps ──────────────────────────────────────────────────────
    print(f"\nSweeping Z0 parameters ({args.npoints} pts × 8 params) …")
    results_z0 = []
    for i in range(8):
        print(f"  {PARAM_NAMES[i]} …", end=" ", flush=True)
        vals, dll = sweep_parameter(
            i, theta_true_z0, theta_true_z100,
            n_z0, n_z100, models_z0, models_z100,
            args.npoints, args.nsigma, args.ntheta, sweep_z=0,
        )
        results_z0.append((vals, dll))
        peak_dll = np.nanmax(dll)
        print(f"peak delta_logL = {peak_dll:.2f}")

    print(f"\nSweeping Z100 parameters ({args.npoints} pts × 8 params) …")
    results_z100 = []
    for i in range(8):
        print(f"  {PARAM_NAMES[i]} …", end=" ", flush=True)
        vals, dll = sweep_parameter(
            i, theta_true_z0, theta_true_z100,
            n_z0, n_z100, models_z0, models_z100,
            args.npoints, args.nsigma, args.ntheta, sweep_z=100,
        )
        results_z100.append((vals, dll))
        peak_dll = np.nanmax(dll)
        print(f"peak delta_logL = {peak_dll:.2f}")

    # ── summary and plots ─────────────────────────────────────────────────────
    print_summary(results_z0, results_z100, theta_true_z0, theta_true_z100)
    plot_profiles(results_z0, results_z100,
                  theta_true_z0, theta_true_z100,
                  shot, args.run, args.out)


if __name__ == "__main__":
    main()
