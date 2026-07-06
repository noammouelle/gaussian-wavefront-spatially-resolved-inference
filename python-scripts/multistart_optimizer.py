"""
Multi-start L-BFGS-B optimizer for cloud-parameter likelihood.

Samples N_STARTS starting points via Latin Hypercube from the 8D cloud prior,
runs L-BFGS-B from each, and saves (theta_hat, logL_hat, n_fev, success) to a
pickle file that the companion notebook can load for plotting.

Results are written incrementally — safe to interrupt and re-load partial runs.

Usage
-----
  python python-scripts/multistart_optimizer.py               # defaults
  python python-scripts/multistart_optimizer.py --n_starts 50 --n_gh 10
  nohup python python-scripts/multistart_optimizer.py > logs/multistart.log 2>&1 &

Defaults target ~30-40 min total wall time on a single GPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scipy.special import ndtri

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'helpers'))
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, os.path.expanduser('~/local/aispy'))

from helpers import ImageShotDataset  # noqa: E402
from profile_cloud_nuisances import (  # noqa: E402
    SurrogatePixelACS, SemiAnalyticPixelACS, _joint_logL,
)

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cp_logsumexp
    USE_GPU = True
except ImportError:
    import numpy as cp  # type: ignore
    from scipy.special import logsumexp as cp_logsumexp  # type: ignore
    USE_GPU = False

from aispy.psmap import load_psmap, PSMAPSurrogate  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────
# Tune for ~30-40 min total on one GPU.
# BINS=32, N_GH=12: ~0.07s/logL call; 200 evals/run → ~14s/run → 100 starts ≈ 23 min.
DEFAULT_SHOT      = 0
DEFAULT_BINS      = 32
DEFAULT_N_GH      = 12
DEFAULT_N_THETA   = 128
DEFAULT_T_DET     = 3.8
DEFAULT_N_STARTS  = 100
DEFAULT_MAXITER   = 200
DEFAULT_SEED      = 42

PARAM_NAMES = ['mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0',
               'sigma_x0', 'sigma_y0', 'sigma_vx0', 'sigma_vy0']

# Prior centred on zero position/velocity, sigma params near 100 µm / 100 µm/s
PRIOR_MEAN = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
PRIOR_STD  = np.array([10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6])


# ── Encode / decode ────────────────────────────────────────────────────────────
# Log-transform sigma params so L-BFGS-B sees an unconstrained space.
def _encode(theta: np.ndarray) -> np.ndarray:
    return np.r_[theta[:4], np.log(np.maximum(theta[4:], 1e-12))]

def _decode(p: np.ndarray) -> np.ndarray:
    return np.r_[p[:4], np.exp(p[4:])]


def _build_bounds(half_range: float):
    hr = float(half_range)
    return [
        (-hr,  hr),                            # mu_x0
        (-hr,  hr),                            # mu_y0
        (-5e-3, 5e-3),                         # mu_vx0
        (-5e-3, 5e-3),                         # mu_vy0
        (np.log(1e-6), np.log(5e-3)),          # log sigma_x0
        (np.log(1e-6), np.log(5e-3)),          # log sigma_y0
        (np.log(1e-6), np.log(5e-3)),          # log sigma_vx0
        (np.log(1e-6), np.log(5e-3)),          # log sigma_vy0
    ]


def _rebin_2d(img: np.ndarray, b: int) -> np.ndarray:
    n = img.shape[0] // b
    return img.reshape(n, b, n, b).sum(axis=(1, 3))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shot',      type=int,   default=DEFAULT_SHOT)
    p.add_argument('--bins',      type=int,   default=DEFAULT_BINS,
                   help='Coarse grid size (default 32)')
    p.add_argument('--n_gh',      type=int,   default=DEFAULT_N_GH,
                   help='Gauss-Hermite order per velocity dim (default 12)')
    p.add_argument('--n_theta',   type=int,   default=DEFAULT_N_THETA)
    p.add_argument('--t_det',     type=float, default=DEFAULT_T_DET)
    p.add_argument('--n_starts',  type=int,   default=DEFAULT_N_STARTS)
    p.add_argument('--maxiter',   type=int,   default=DEFAULT_MAXITER)
    p.add_argument('--seed',      type=int,   default=DEFAULT_SEED)
    p.add_argument('--data_dir',  type=str,
                   default=str(REPO / 'data' /
                       'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
                       'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000' /
                       'run_000'))
    p.add_argument('--out',       type=str, default='',
                   help='Output pickle path (default: results/multistart_shot{N}.pkl)')
    args = p.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('multistart')

    out_path = args.out or str(REPO / 'results' /
                               f'multistart_shot{args.shot}_bins{args.bins}_ngh{args.n_gh}.pkl')
    log.info('Output → %s', out_path)
    log.info('Config: shot=%d  bins=%d  n_gh=%d  n_theta=%d  n_starts=%d  maxiter=%d',
             args.shot, args.bins, args.n_gh, args.n_theta, args.n_starts, args.maxiter)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    ds_z0   = ImageShotDataset(str(data_dir / 'Z0'   / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(data_dir / 'Z100' / 'data_IMG.h5'))
    log.info('Dataset: %d shots  res=%d  half_range=%.1f mm',
             ds_z0.n_shots, ds_z0.res, ds_z0.half_range * 1e3)

    b = ds_z0.res // args.bins
    edges = np.linspace(-ds_z0.half_range, ds_z0.half_range, args.bins + 1)

    img0 = np.asarray(ds_z0[args.shot],   dtype=np.float64)
    img1 = np.asarray(ds_z100[args.shot], dtype=np.float64)

    def _rb(img):
        return _rebin_2d(img, b).ravel()

    xp = cp if USE_GPU else np
    ng0 = xp.asarray(_rb(img0[0]));  ne0 = xp.asarray(_rb(img0[1]))
    ng1 = xp.asarray(_rb(img1[0]));  ne1 = xp.asarray(_rb(img1[1]))
    log.info('Z0  counts: %d (s0=%d, s1=%d)',
             int((ng0 + ne0).sum()), int(ng0.sum()), int(ne0.sum()))
    log.info('Z100 counts: %d', int((ng1 + ne1).sum()))

    # True params [mu_x0, mu_y0, mu_vx0, mu_vy0, sx, sy, svx, svy]
    def meta_to_theta(m):
        return np.array([m['mu_x0'], m['mu_y0'], m['mu_vx0'], m['mu_vy0'],
                         m['sigma_x'], m['sigma_y'], m['sigma_vx'], m['sigma_vy']])

    theta_true_z0   = meta_to_theta(ds_z0.meta(args.shot))
    theta_true_z100 = meta_to_theta(ds_z100.meta(args.shot))
    log.info('True Z0:   mu_x0=%.2f µm  mu_vx0=%.2f µm/s  sx=%.1f µm  svx=%.1f µm/s',
             theta_true_z0[0]*1e6, theta_true_z0[2]*1e6,
             theta_true_z0[4]*1e6, theta_true_z0[6]*1e6)
    log.info('True Z100: mu_x0=%.2f µm  mu_vx0=%.2f µm/s  sx=%.1f µm  svx=%.1f µm/s',
             theta_true_z100[0]*1e6, theta_true_z100[2]*1e6,
             theta_true_z100[4]*1e6, theta_true_z100[6]*1e6)

    # ── PSMAPs ────────────────────────────────────────────────────────────────
    log.info('Loading PSMAPs...')
    psmap_z0   = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_FINE_Z0.h5'))
    psmap_z100 = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_FINE_Z100.h5'))

    log.info('Building GPU evaluators (bins=%d, n_gh=%d)...', args.bins, args.n_gh)
    t0 = time.perf_counter()
    sur_z0   = PSMAPSurrogate(psmap_z0,   args.t_det, use_gpu=USE_GPU)
    sur_z100 = PSMAPSurrogate(psmap_z100, args.t_det, use_gpu=USE_GPU)
    base_z0   = SurrogatePixelACS(sur_z0,   args.t_det, edges, edges, n_quad=1)
    base_z100 = SurrogatePixelACS(sur_z100, args.t_det, edges, edges, n_quad=1)
    eval_z0   = SemiAnalyticPixelACS(base_z0,   n_gh=args.n_gh)
    eval_z100 = SemiAnalyticPixelACS(base_z100, n_gh=args.n_gh)
    log.info('Evaluators ready in %.1fs  n_bins=%d  gh_pts_per_pixel=%d',
             time.perf_counter() - t0, eval_z0.n_bins, eval_z0.n_v)

    # Pre-compute Z100 ACS at true params (held fixed — only Z0 is optimised)
    log.info('Pre-computing Z100 ACS at true params...')
    acs_z100_true = eval_z100.pixel_acs(theta_true_z100)
    log.info('Done')

    # Timing check at true params
    t0 = time.perf_counter()
    def _logL(theta_z0):
        acs0 = eval_z0.pixel_acs(theta_z0)
        return _joint_logL(ng0, ne0, *acs0,
                           ng1, ne1, *acs_z100_true,
                           args.n_theta, xp, cp_logsumexp)

    ll_true = _logL(theta_true_z0)
    t_eval = time.perf_counter() - t0
    log.info('logL at true params = %.2f  (%.3f s/call)', ll_true, t_eval)
    expected_s = t_eval * args.maxiter * args.n_starts
    log.info('Rough ETA for full run: %.0f s (%.1f min) at %d starts × %d max iters',
             expected_s, expected_s / 60, args.n_starts, args.maxiter)

    # ── LHS starts ────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    sampler = LatinHypercube(d=8, seed=rng)
    u = sampler.random(args.n_starts)
    z = ndtri(np.clip(u, 1e-6, 1 - 1e-6))
    theta_starts = PRIOR_MEAN + PRIOR_STD * z
    theta_starts[:, 4:] = np.abs(theta_starts[:, 4:]).clip(1e-7)
    log.info('Generated %d LHS starts from prior', args.n_starts)

    bounds = _build_bounds(ds_z0.half_range)

    def neg_logL(p):
        theta = _decode(p)
        if np.any(theta[4:] <= 0):
            return 1e300
        try:
            return float(-_logL(theta))
        except Exception:
            return 1e300

    # ── Optimisation loop ─────────────────────────────────────────────────────
    results = []
    t_loop_start = time.perf_counter()
    n_ok = 0

    for i in range(args.n_starts):
        th0 = theta_starts[i]
        p0  = _encode(th0)

        t_run = time.perf_counter()
        opt = minimize(
            neg_logL, p0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': args.maxiter, 'ftol': 1e-12, 'gtol': 1e-7},
        )
        elapsed_run = time.perf_counter() - t_run

        theta_hat = _decode(opt.x)
        logL_hat  = -float(opt.fun)
        if opt.success:
            n_ok += 1

        results.append({
            'theta_start': th0,
            'theta_hat':   theta_hat,
            'll_start':    float(-neg_logL(p0)),
            'll_hat':      logL_hat,
            'n_fev':       int(opt.nfev),
            'success':     bool(opt.success),
            'message':     opt.message,
            'wall_s':      float(elapsed_run),
        })

        # ETA
        elapsed_total = time.perf_counter() - t_loop_start
        avg_s = elapsed_total / (i + 1)
        eta_s = avg_s * (args.n_starts - i - 1)
        log.info('[%3d/%d] logL=%8.2f  Δtrue=%+7.2f  fev=%3d  ok=%s  '
                 'run=%.1fs  ETA=%.0fm%.0fs  '
                 'mu_x0=%+.2fµm  mu_vx0=%+.2fµm/s  sx=%.1fµm',
                 i + 1, args.n_starts,
                 logL_hat, logL_hat - ll_true,
                 opt.nfev, opt.success,
                 elapsed_run, eta_s // 60, eta_s % 60,
                 theta_hat[0] * 1e6, theta_hat[2] * 1e6, theta_hat[4] * 1e6)

        # Save incrementally after every run
        payload = {
            'results':          results,
            'theta_true_z0':    theta_true_z0,
            'theta_true_z100':  theta_true_z100,
            'll_true':          ll_true,
            'param_names':      PARAM_NAMES,
            'config': {
                'shot':     args.shot,
                'bins':     args.bins,
                'n_gh':     args.n_gh,
                'n_theta':  args.n_theta,
                'n_starts': args.n_starts,
                'maxiter':  args.maxiter,
                'seed':     args.seed,
                'data_dir': str(args.data_dir),
            },
        }
        with open(out_path, 'wb') as f:
            pickle.dump(payload, f)

    total_s = time.perf_counter() - t_loop_start
    log.info('Done. %d/%d converged. Total wall time: %.0f s (%.1f min)',
             n_ok, args.n_starts, total_s, total_s / 60)
    log.info('Best logL found: %.2f  (true: %.2f  Δ=%.2f)',
             max(r['ll_hat'] for r in results), ll_true,
             max(r['ll_hat'] for r in results) - ll_true)
    log.info('Results saved to %s', out_path)


if __name__ == '__main__':
    main()
