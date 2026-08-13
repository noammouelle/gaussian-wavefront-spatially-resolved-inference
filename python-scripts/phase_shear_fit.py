"""
phase_shear_fit.py — Fit Gaussian × sinusoidal fringe to phase-shear images.

For each shot, fits the model

    n_g(x, y) = A · exp[-(x−μ_x)²/(2σ_x²) − (y−μ_y)²/(2σ_y²)]
              × (1 + C_cos·cos(κ_x x + κ_y y + γ_x x² + γ_y y²)
                   + C_sin·sin(κ_x x + κ_y y + γ_x x² + γ_y y²))

to the ground-state port (port 0) image of Z0 and Z100 independently.

Using C_cos/C_sin rather than C/φ avoids the cyclic-phase local-minimum problem.
Derived quantities:  C = √(C_cos²+C_sin²),  φ_0 = arctan2(C_sin, C_cos).

κ_x is initialised from the 'linear_phase_kappa' dataset attribute; κ_y, γ_x, γ_y
start at zero.  All parameters are then refined jointly by minimising the
Poisson log-likelihood.

Images are binned to --bins × --bins before fitting to control speed.
Default 128×128 gives ~8 pixels per 200 µm fringe period — well above Nyquist.

Usage
-----
    python python-scripts/phase_shear_fit.py
    python python-scripts/phase_shear_fit.py --data_root data/...kappa...both
    nohup python python-scripts/phase_shear_fit.py --data_root data/... \\
        > logs/phase_shear_fit.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'helpers'))
from helpers import ImageShotDataset   # noqa

DEFAULT_DATASET = (
    'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
    'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000'
    '_kappa3.14e+04_both'
)
DEFAULT_BINS     = 128    # image bins per side after downsampling
DEFAULT_MAX_SHOTS = None


# ── Image binning ──────────────────────────────────────────────────────────────

def bin_image(img, n_bins):
    """
    Sum-bin (res, res) image to (n_bins, n_bins) by integer block summation.
    Uses scipy.ndimage.zoom with order=1 then rescales to preserve total count.
    Falls back to block-sum when res is an integer multiple of n_bins.
    """
    res = img.shape[0]
    if res == n_bins:
        return img.astype(np.float64)
    if res % n_bins == 0:
        k = res // n_bins
        # Block-sum: reshape and sum
        return img.reshape(n_bins, k, n_bins, k).sum(axis=(1, 3)).astype(np.float64)
    # General case: zoom (area-conservative via order=1 + rescale)
    factor = n_bins / res
    out = zoom(img.astype(np.float64), factor, order=1)
    out *= img.sum() / out.sum()
    return out


def bin_pixel_centers(pixel_centers, n_bins):
    """Return n_bins pixel centers for the binned grid."""
    lo, hi = pixel_centers[0], pixel_centers[-1]
    dpix_half = (hi - lo) / (2 * (n_bins - 1))
    return np.linspace(lo, hi, n_bins)


# ── Fringe model ───────────────────────────────────────────────────────────────

def _phase(xx, yy, kappa_x, kappa_y, gamma_x, gamma_y):
    return kappa_x*xx + kappa_y*yy + gamma_x*xx**2 + gamma_y*yy**2


def model_image(xx, yy, log_A, mu_x, mu_y, log_sig_x, log_sig_y,
                C_cos, C_sin, kappa_x, kappa_y, gamma_x, gamma_y):
    A     = np.exp(log_A)
    sig_x = np.exp(log_sig_x)
    sig_y = np.exp(log_sig_y)
    gauss = A * np.exp(-0.5*((xx - mu_x)/sig_x)**2
                       -0.5*((yy - mu_y)/sig_y)**2)
    phi   = _phase(xx, yy, kappa_x, kappa_y, gamma_x, gamma_y)
    return gauss * (1.0 + C_cos*np.cos(phi) + C_sin*np.sin(phi))


# ── Initial guess ──────────────────────────────────────────────────────────────

def initial_guess(img_g, img_total, pixel_centers_binned, kappa_x_prior):
    """
    Estimate starting parameters.
    Gaussian params from moments of the total image.
    C_cos/C_sin from a linear solve on the normalised residual.
    kappa_x from dataset attribute.
    """
    pc   = pixel_centers_binned
    xx, yy = np.meshgrid(pc, pc, indexing='ij')
    dpix   = float(pc[1] - pc[0])

    total  = img_total.astype(np.float64).clip(1e-3)
    N_tot  = total.sum()
    px = total.sum(axis=1); py = total.sum(axis=0)
    mu_x = float((px * pc).sum() / N_tot)
    mu_y = float((py * pc).sum() / N_tot)
    var_x = float((px * (pc - mu_x)**2).sum() / N_tot)
    var_y = float((py * (pc - mu_y)**2).sum() / N_tot)
    sig_x = max(np.sqrt(var_x), dpix)
    sig_y = max(np.sqrt(var_y), dpix)
    # amplitude: A such that A * 2π σ_x σ_y / dpix² ≈ N_tot_g
    N_g   = img_g.sum()
    A     = N_g * dpix**2 / (2 * np.pi * sig_x * sig_y)

    # Linear solve for C_cos, C_sin at kappa_x = kappa_x_prior, others = 0
    gauss_env = A * np.exp(-0.5*((xx - mu_x)/sig_x)**2
                           -0.5*((yy - mu_y)/sig_y)**2)
    w     = (img_g.astype(np.float64) / gauss_env.clip(1e-6) - 1.0).ravel()
    phi   = kappa_x_prior * xx
    Bcos  = np.cos(phi).ravel()
    Bsin  = np.sin(phi).ravel()
    # Weighted least squares (weight by gauss_env to down-weight tails)
    wt    = gauss_env.ravel()
    BtW   = np.array([Bcos * wt, Bsin * wt])     # (2, n)
    BtWB  = BtW @ np.array([Bcos, Bsin]).T         # (2, 2)
    BtWy  = BtW @ w                                # (2,)
    try:
        cs = np.linalg.solve(BtWB, BtWy)
        C_cos0, C_sin0 = float(cs[0]), float(cs[1])
    except np.linalg.LinAlgError:
        C_cos0, C_sin0 = 0.3, 0.0

    return np.array([
        np.log(max(A, 1.0)),
        mu_x, mu_y,
        np.log(sig_x), np.log(sig_y),
        C_cos0, C_sin0,
        kappa_x_prior, 0.0,   # kappa_x, kappa_y
        0.0, 0.0,             # gamma_x, gamma_y
    ])


# ── Fitting ────────────────────────────────────────────────────────────────────

def fit_shot_image(img_g, img_total, pixel_centers_binned, kappa_x_prior):
    """
    Fit ground-state image img_g (n_bins, n_bins) to the Gaussian×fringe model.

    Returns a flat dict of fitted parameters and diagnostics.
    """
    EPS  = 1e-6
    pc   = pixel_centers_binned
    xx, yy = np.meshgrid(pc, pc, indexing='ij')

    p0 = initial_guess(img_g, img_total, pc, kappa_x_prior)

    n_obs = img_g.astype(np.float64)

    def neg_logL(p):
        n_mod = model_image(xx, yy, *p).clip(EPS)
        # Poisson NLL (ignoring constant factorial term)
        return float(np.sum(n_mod - n_obs * np.log(n_mod)))

    result = minimize(neg_logL, p0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})

    p = result.x
    log_A, mu_x, mu_y, log_sig_x, log_sig_y, C_cos, C_sin, kx, ky, gx, gy = p
    C   = float(np.sqrt(C_cos**2 + C_sin**2))
    phi = float(np.arctan2(C_sin, C_cos))

    return {
        'A':       float(np.exp(log_A)),
        'mu_x':    float(mu_x),
        'mu_y':    float(mu_y),
        'sig_x':   float(np.exp(log_sig_x)),
        'sig_y':   float(np.exp(log_sig_y)),
        'C_cos':   float(C_cos),
        'C_sin':   float(C_sin),
        'C':       C,
        'phi_0':   phi,
        'kappa_x': float(kx),
        'kappa_y': float(ky),
        'gamma_x': float(gx),
        'gamma_y': float(gy),
        'nll':     float(result.fun),
        'success': bool(result.success),
        'n_g':     float(n_obs.sum()),
        'p0':      p0.tolist(),
    }


# ── Per-run loop ───────────────────────────────────────────────────────────────

def _run_one(data_dir, n_bins, max_shots, log):
    data_dir = Path(data_dir)

    ds_z0   = ImageShotDataset(str(data_dir / 'Z0'   / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(data_dir / 'Z100' / 'data_IMG.h5'))

    with h5py.File(str(data_dir / 'Z0' / 'data_IMG.h5')) as f:
        kappa_x_prior = float(f.attrs.get('linear_phase_kappa', 0.0))
        f_signal          = float(f.attrs.get('signal_freq',  0.0))
        signal_amp_true   = float(f.attrs.get('signal_amp',   0.0))
        signal_phase_true = float(f.attrs.get('signal_phase', 0.0))
        meta_attrs = {}
        for k, v in f.attrs.items():
            try:
                meta_attrs[k] = float(v)
            except (TypeError, ValueError):
                meta_attrs[k] = str(v)

    As_true = signal_amp_true * np.cos(signal_phase_true)
    Ac_true = signal_amp_true * np.sin(signal_phase_true)
    log.info('κ_x_prior=%.1f  true As=%.4f Ac=%.4f  f=%.4f',
             kappa_x_prior, As_true, Ac_true, f_signal)

    shot_ids = list(range(ds_z0.n_shots))
    if max_shots is not None:
        shot_ids = shot_ids[:max_shots]

    # Build binned pixel grid once
    pc_binned = bin_pixel_centers(ds_z0.pixel_centers, n_bins)

    rows = []
    t_loop = time.perf_counter()

    for count, shot_id in enumerate(shot_ids, start=1):
        t0 = time.perf_counter()
        img0 = ds_z0[shot_id]    # (2, res, res): port0=g, port1=e
        img1 = ds_z100[shot_id]

        # Bin both ports
        g0 = bin_image(img0[0], n_bins)
        e0 = bin_image(img0[1], n_bins)
        g1 = bin_image(img1[0], n_bins)
        e1 = bin_image(img1[1], n_bins)

        fit_z0   = fit_shot_image(g0, g0 + e0, pc_binned, kappa_x_prior)
        fit_z100 = fit_shot_image(g1, g1 + e1, pc_binned, kappa_x_prior)

        meta0 = ds_z0.meta(shot_id)
        meta1 = ds_z100.meta(shot_id)

        rows.append({
            'shot':          shot_id,
            'fit_z0':        fit_z0,
            'fit_z100':      fit_z100,
            'n_e_z0':        float(img0[1].sum()),
            'n_e_z100':      float(img1[1].sum()),
            'delta_phi_true': float(meta1.get('delta_phi', float('nan'))),
            'theta_true_z0':  {k: float(v) for k, v in meta0.items()},
            'theta_true_z100':{k: float(v) for k, v in meta1.items()},
        })

        elapsed = time.perf_counter() - t_loop
        shot_t  = time.perf_counter() - t0
        eta_s   = (elapsed / count) * (len(shot_ids) - count)
        if count % 10 == 0 or count == len(shot_ids):
            log.info('[%3d/%d] shot=%4d  %.2fs/shot  ETA=%.0fm%.0fs  '
                     'Z0: C=%.3f φ=%.3f κ=%.0f  Z100: C=%.3f φ=%.3f κ=%.0f',
                     count, len(shot_ids), shot_id, shot_t,
                     eta_s // 60, eta_s % 60,
                     fit_z0['C'], fit_z0['phi_0'], fit_z0['kappa_x'],
                     fit_z100['C'], fit_z100['phi_0'], fit_z100['kappa_x'])

    log.info('Done in %.1fs', time.perf_counter() - t_loop)

    return {
        'rows':              rows,
        'shot_ids':          shot_ids,
        'f_signal':          f_signal,
        'As_true':           As_true,
        'Ac_true':           Ac_true,
        'signal_amp_true':   signal_amp_true,
        'signal_phase_true': signal_phase_true,
        'kappa_x_prior':     kappa_x_prior,
        'pc_binned':         pc_binned,
        'n_bins':            n_bins,
        'dataset_attrs':     meta_attrs,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bins',      type=int, default=DEFAULT_BINS,
                   help='Image bins per side after downsampling (default 128)')
    p.add_argument('--max_shots', type=int, default=DEFAULT_MAX_SHOTS)
    p.add_argument('--data_root', type=str, default='',
                   help='Dataset root containing run_NNN dirs (sweep mode)')
    p.add_argument('--data_dir',  type=str,
                   default=str(REPO / 'data' / DEFAULT_DATASET / 'run_000'))
    p.add_argument('--out_dir',   type=str, default='',
                   help='Output dir for sweep (default: results/phase_shear/)')
    p.add_argument('--out',       type=str, default='',
                   help='Output pkl path (single-run mode)')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)s  %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('phase_shear_fit')
    log.info('bins=%d  max_shots=%s', args.bins, args.max_shots)

    if args.data_root:
        data_root = Path(args.data_root)
        run_dirs  = sorted(data_root.glob('run_*'))
        out_dir   = Path(args.out_dir or str(REPO / 'results' / 'phase_shear'))
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info('Sweep: %d runs  ->  %s', len(run_dirs), out_dir)
    else:
        run_dirs = [Path(args.data_dir)]
        out_dir  = None

    for run_dir in run_dirs:
        log.info('===== %s =====', run_dir.name)
        payload = _run_one(run_dir, args.bins, args.max_shots, log)

        if out_dir is not None:
            out_path = out_dir / f'phase_shear_{run_dir.name}.pkl'
        else:
            out_path = Path(args.out or str(REPO / 'results' / 'phase_shear_fit.pkl'))

        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        log.info('Saved -> %s', out_path)


if __name__ == '__main__':
    main()
