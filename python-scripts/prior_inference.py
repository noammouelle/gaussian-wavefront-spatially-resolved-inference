"""
prior_inference.py — 8D QMC prior marginalisation for signal inference.

Baseline companion to moment_inference.py.  Does NOT use final-position moments
to constrain the cloud.  Instead marginalises over the full 8D prior p0(eta)
directly.  Comparing results with moment_inference.py quantifies the information
contributed by the spatial moments.

Pipeline
--------
1. Draw n_eta samples from the 8D prior p0(eta) via Sobol QMC (scrambled,
   transformed through normal ppf).  These are fixed for the whole run.

2. Evaluate the spatially-integrated ACS at all n_eta points for Z0 and Z100
   in a single batched GPU call per AI.  This is O(n_eta), not O(n_shots*n_eta),
   because the prior does not change between shots.

3. Loop over shots: read total port counts (n_g, n_e) per AI.  No moment
   extraction is needed.

4. For a trial beta = (As, Ac):
   - dphi_i = As*sin(2*pi*f*i) + Ac*cos(2*pi*f*i)
   - For each phi on a uniform grid:
       log f_z0(phi)        = logsumexp_j [log w_j + Poisson logL for Z0 at phi]
       log f_z100(phi+dphi) = logsumexp_j [log w_j + Poisson logL for Z100]
   - logL_i = logsumexp_phi [log f_z0(phi) + log f_z100(phi+dphi)] - log K
   - logL = sum_i logL_i

5. Multi-start L-BFGS-B optimise logL(beta) over (As, Ac).

Usage
-----
    python python-scripts/prior_inference.py
    nohup python python-scripts/prior_inference.py --n_eta 512 > logs/prior_inference.log 2>&1 &
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
from scipy.optimize import minimize
from scipy.special import logsumexp as sp_logsumexp
from scipy.stats import norm as sp_norm
from scipy.stats.qmc import Sobol

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'helpers'))
sys.path.insert(0, str(REPO / 'python-scripts'))

from helpers import ImageShotDataset                                 # noqa
from profile_cloud_nuisances import SurrogatePixelACS               # noqa

try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    cp = None
    USE_GPU = False

try:
    from aispy.psmap import load_psmap, PSMAPSurrogate
except ImportError:
    sys.path.insert(0, str(REPO.parents[1] / 'local' / 'aispy'))
    from aispy.psmap import load_psmap, PSMAPSurrogate

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BINS      = 32       # pixel grid for SurrogatePixelACS construction only
DEFAULT_N_ETA     = 512      # prior QMC samples (prior integral nodes)
DEFAULT_N_QMC_INT = 2048     # cloud QMC samples per eta point for integrated ACS
DEFAULT_N_THETA   = 128      # phi grid points
DEFAULT_T_DET     = 3.8
DEFAULT_MAX_SHOTS = None
DEFAULT_GRID_N    = 31
DEFAULT_GRID_HALF = 0.2
DEFAULT_N_STARTS  = 8

# Gaussian prior on cloud parameters — must match data-generation parameters
TAU_MU_POS  = 10e-6     # prior std on mu_x0, mu_y0          [m]
TAU_MU_VEL  = 10e-6     # prior std on mu_vx0, mu_vy0        [m/s]
BAR_SIG_POS = 100e-6    # prior mean of sigma_x0, sigma_y0   [m]
BAR_SIG_VEL = 100e-6    # prior mean of sigma_vx0, sigma_vy0 [m/s]
SIG_SIG_POS = 10e-6     # prior std of sigma_x0              [m]
SIG_SIG_VEL = 10e-6     # prior std of sigma_vx0             [m/s]


# ── 8D prior sampler ─────────────────────────────────────────────────────────
def sample_prior_eta(n_eta, rng_seed=0):
    """
    Draw n_eta vectors from the 8D prior p0(eta) using scrambled Sobol QMC.

    eta columns: [mu_x0, mu_y0, mu_vx0, mu_vy0, sig_x0, sig_y0, sig_vx0, sig_vy0]

    Returns
    -------
    eta  : (n_eta, 8) float64
    log_w: (n_eta,)  float64  — uniform log weights = -log(n_eta)
    """
    sampler = Sobol(d=8, scramble=True, seed=rng_seed)
    # Sobol requires power-of-2; draw next power of 2 >= n_eta, then trim
    m = int(np.ceil(np.log2(max(n_eta, 1))))
    u = sampler.random_base2(m)[:n_eta]          # (n_eta, 8) in [0,1)^8
    z = sp_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))  # standard normal

    prior_means = np.array([
        0.0, 0.0, 0.0, 0.0,
        BAR_SIG_POS, BAR_SIG_POS, BAR_SIG_VEL, BAR_SIG_VEL,
    ])
    prior_stds = np.array([
        TAU_MU_POS, TAU_MU_POS, TAU_MU_VEL, TAU_MU_VEL,
        SIG_SIG_POS, SIG_SIG_POS, SIG_SIG_VEL, SIG_SIG_VEL,
    ])
    eta = prior_means[None, :] + prior_stds[None, :] * z   # (n_eta, 8)

    # Standard deviations must be positive
    eta[:, 4:] = np.clip(eta[:, 4:], 1e-8, None)

    log_w = np.full(n_eta, -np.log(n_eta))
    return eta.astype(np.float64), log_w


# ── Batched spatially-integrated ACS ─────────────────────────────────────────
def batched_integrated_acs(acs_obj, eta_batch, n_qmc, rng_seed=0):
    """
    Compute total (spatially integrated) ACS for a batch of eta vectors.

    Parameters
    ----------
    acs_obj   : SurrogatePixelACS — only _eval_fast and port-state masks used
    eta_batch : (N, 8) numpy float64
    n_qmc     : int — cloud QMC samples per eta point
    rng_seed  : int

    Returns
    -------
    (N, 6) numpy float64 — (A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e)
    """
    if not USE_GPU:
        raise RuntimeError('GPU (cupy) required')

    xp = cp
    N  = len(eta_batch)
    rng = np.random.default_rng(rng_seed)
    z_np = rng.standard_normal((n_qmc, 4)).astype(np.float64)
    z_g  = xp.asarray(z_np)             # (n_qmc, 4)

    eta_g  = xp.asarray(eta_batch, dtype=xp.float64)   # (N, 8)
    mu_g   = eta_g[:, :4]                               # (N, 4)
    sig_g  = eta_g[:, 4:]                               # (N, 4)

    # Cloud samples: (N, n_qmc, 4)
    pts = mu_g[:, None, :] + sig_g[:, None, :] * z_g[None, :, :]
    pts_flat = pts.reshape(N * n_qmc, 4)

    dphi_g, amp0_g, amp1_g = acs_obj._eval_fast(
        pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2], pts_flat[:, 3])

    inter  = acs_obj._port_inter[None]
    A_per  = amp0_g ** 2 + amp1_g ** 2
    Cc_per =  inter * 2.0 * amp0_g * amp1_g * xp.cos(dphi_g)
    Cs_per = -inter * 2.0 * amp0_g * amp1_g * xp.sin(dphi_g)

    s0, s1 = acs_obj.s0_g, acs_obj.s1_g

    def _avg(arr, mask):
        return arr[:, mask].sum(-1).reshape(N, n_qmc).mean(1)

    out = xp.stack([
        _avg(A_per,  s0), _avg(Cc_per, s0), _avg(Cs_per, s0),
        _avg(A_per,  s1), _avg(Cc_per, s1), _avg(Cs_per, s1),
    ], axis=1)   # (N, 6)

    return out.get()


# ── logL ──────────────────────────────────────────────────────────────────────
def log_f_ai(shot_dict, phi_shifted):
    """
    log f_z(phi) = logsumexp_j [log w_j + Poisson logL at phi_shifted]
    for one AI over the prior QMC quadrature.

    phi_shifted : (n_theta,) array
    Returns     : (n_theta,) array
    """
    EPS = 1e-300
    acs = shot_dict['acs_eta']   # (N_eta, 6)
    lw  = shot_dict['log_w']     # (N_eta,)
    n_g = shot_dict['n_g_tot']
    n_e = shot_dict['n_e_tot']

    c = np.cos(phi_shifted); s = np.sin(phi_shifted)   # (n_theta,)

    A_g  = acs[:, 0]; Cc_g = acs[:, 1]; Cs_g = acs[:, 2]
    A_e  = acs[:, 3]; Cc_e = acs[:, 4]; Cs_e = acs[:, 5]

    A_tot  = A_g + A_e
    Cc_tot = Cc_g + Cc_e
    Cs_tot = Cs_g + Cs_e

    totACS = (A_tot[:, None]
              + Cc_tot[:, None] * c[None]
              + Cs_tot[:, None] * s[None])             # (N_eta, n_theta)
    L_phi  = (n_g + n_e) / np.maximum(totACS, EPS)    # phi-dep normalisation

    lam_g = L_phi * np.maximum(
        A_g[:, None] + Cc_g[:, None] * c[None] + Cs_g[:, None] * s[None], EPS)
    lam_e = L_phi * np.maximum(
        A_e[:, None] + Cc_e[:, None] * c[None] + Cs_e[:, None] * s[None], EPS)

    ll = (n_g * np.log(lam_g) - lam_g +
          n_e * np.log(lam_e) - lam_e)   # (N_eta, n_theta)

    return sp_logsumexp(lw[:, None] + ll, axis=0)   # (n_theta,)


def total_logL(beta, precomp_list, f_signal, shot_idx_arr, n_theta):
    As, Ac = float(beta[0]), float(beta[1])
    phi  = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    dphi = (As * np.sin(2 * np.pi * f_signal * shot_idx_arr) +
            Ac * np.cos(2 * np.pi * f_signal * shot_idx_arr))

    total = 0.0
    for (sz0, sz100), dp in zip(precomp_list, dphi):
        lf0   = log_f_ai(sz0,   phi)
        lf100 = log_f_ai(sz100, phi + dp)
        total += float(sp_logsumexp(lf0 + lf100) - np.log(n_theta))
    return total


DEFAULT_DATASET = (
    'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
    'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000'
)


def _run_one(data_dir, acs_prior_z0, acs_prior_z100, log_w_prior, eta_prior, args, log):
    """Process one run directory using pre-computed prior ACS. Returns payload dict."""
    data_dir = Path(data_dir)
    ds_z0   = ImageShotDataset(str(data_dir / 'Z0'   / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(data_dir / 'Z100' / 'data_IMG.h5'))
    with h5py.File(str(data_dir / 'Z100' / 'data_IMG.h5')) as f:
        f_signal          = float(f.attrs['signal_freq'])
        signal_amp_true   = float(f.attrs['signal_amp'])
        signal_phase_true = float(f.attrs['signal_phase'])
    As_true = signal_amp_true * np.cos(signal_phase_true)
    Ac_true = signal_amp_true * np.sin(signal_phase_true)
    log.info('True: As=%.4f Ac=%.4f  (amp=%.3f phase=%.3f f=%.4f)',
             As_true, Ac_true, signal_amp_true, signal_phase_true, f_signal)

    shot_ids = list(range(ds_z0.n_shots))
    if args.max_shots is not None:
        shot_ids = shot_ids[:args.max_shots]
    log.info('Using %d/%d shots', len(shot_ids), ds_z0.n_shots)
    shot_idx_arr = np.array(shot_ids, dtype=np.float64)

    precomp_list = []
    diag_rows    = []

    for shot_id in shot_ids:
        img0 = ds_z0[shot_id]
        img1 = ds_z100[shot_id]

        n_g0_tot = float(img0[0].sum())
        n_e0_tot = float(img0[1].sum())
        n_g1_tot = float(img1[0].sum())
        n_e1_tot = float(img1[1].sum())

        sd_z0   = {'acs_eta': acs_prior_z0,  'log_w': log_w_prior,
                   'n_g_tot': n_g0_tot, 'n_e_tot': n_e0_tot}
        sd_z100 = {'acs_eta': acs_prior_z100, 'log_w': log_w_prior,
                   'n_g_tot': n_g1_tot, 'n_e_tot': n_e1_tot}
        precomp_list.append((sd_z0, sd_z100))

        meta0 = ds_z0.meta(shot_id)
        meta1 = ds_z100.meta(shot_id)
        diag_rows.append({
            'shot': shot_id,
            'n_g0_tot': n_g0_tot, 'n_e0_tot': n_e0_tot,
            'n_g1_tot': n_g1_tot, 'n_e1_tot': n_e1_tot,
            'theta_true_z0':   np.array([meta0['mu_x0'], meta0['mu_y0'], meta0['mu_vx0'],
                                         meta0['mu_vy0'], meta0['sigma_x'], meta0['sigma_y'],
                                         meta0['sigma_vx'], meta0['sigma_vy']]),
            'theta_true_z100': np.array([meta1['mu_x0'], meta1['mu_y0'], meta1['mu_vx0'],
                                         meta1['mu_vy0'], meta1['sigma_x'], meta1['sigma_y'],
                                         meta1['sigma_vx'], meta1['sigma_vy']]),
            'delta_phi_true': float(meta1['delta_phi']),
        })

    log.info('Port counts loaded for %d shots', len(shot_ids))

    def neg_logL(beta):
        return -total_logL(beta, precomp_list, f_signal, shot_idx_arr, args.n_theta)

    t0 = time.perf_counter()
    ll_true = -neg_logL((As_true, Ac_true))
    ll_zero = -neg_logL((0.0, 0.0))
    t_eval  = time.perf_counter() - t0
    log.info('logL(true)=%.2f  logL(0,0)=%.2f  delta=%.2f  (%.2f s/eval)',
             ll_true, ll_zero, ll_true - ll_zero, t_eval / 2)

    log.info('Running %d-start L-BFGS-B...', args.n_starts)
    t_opt = time.perf_counter()
    bounds = [(-args.grid_half, args.grid_half)] * 2
    rng_starts = np.random.default_rng(0)
    starts = [np.zeros(2)]
    starts += list(rng_starts.uniform(-args.grid_half, args.grid_half,
                                      size=(args.n_starts - 1, 2)))

    best_opt = None
    for k, b0 in enumerate(starts):
        opt = minimize(neg_logL, b0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300})
        log.info('  start %d/%d  As0=%.3f Ac0=%.3f  logL=%.2f  %s',
                 k + 1, args.n_starts, b0[0], b0[1], -float(opt.fun),
                 '✓' if opt.success else opt.message[:40])
        if best_opt is None or opt.fun < best_opt.fun:
            best_opt = opt
    log.info('Multi-start done in %.1fs', time.perf_counter() - t_opt)

    beta_hat = best_opt.x
    log.info('Optimised: As_hat=%.4f Ac_hat=%.4f  logL=%.2f  (true: As=%.4f Ac=%.4f)',
             beta_hat[0], beta_hat[1], -float(best_opt.fun), As_true, Ac_true)

    grid_ax   = np.linspace(-args.grid_half, args.grid_half, args.grid_n)
    logL_grid = np.full((args.grid_n, args.grid_n), np.nan)

    return {
        'diag_rows':         diag_rows,
        'shot_ids':          shot_ids,
        'f_signal':          f_signal,
        'As_true':           As_true,
        'Ac_true':           Ac_true,
        'signal_amp_true':   signal_amp_true,
        'signal_phase_true': signal_phase_true,
        'beta_hat':          beta_hat,
        'logL_hat':          -float(best_opt.fun),
        'logL_true':         ll_true,
        'logL_zero':         ll_zero,
        'opt_success':       bool(best_opt.success),
        'opt_message':       str(best_opt.message),
        'grid_ax':           grid_ax,
        'logL_grid':         logL_grid,
        'eta_prior':         eta_prior,
        'log_w_prior':       log_w_prior,
        'config':            vars(args),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bins',      type=int,   default=DEFAULT_BINS)
    p.add_argument('--n_eta',     type=int,   default=DEFAULT_N_ETA,
                   help='Number of Sobol QMC prior samples')
    p.add_argument('--n_qmc_int', type=int,   default=DEFAULT_N_QMC_INT,
                   help='Cloud QMC samples per eta point for integrated ACS')
    p.add_argument('--n_theta',   type=int,   default=DEFAULT_N_THETA)
    p.add_argument('--t_det',     type=float, default=DEFAULT_T_DET)
    p.add_argument('--max_shots', type=int,   default=DEFAULT_MAX_SHOTS)
    p.add_argument('--grid_n',    type=int,   default=DEFAULT_GRID_N)
    p.add_argument('--grid_half', type=float, default=DEFAULT_GRID_HALF)
    p.add_argument('--n_starts',  type=int,   default=DEFAULT_N_STARTS)
    p.add_argument('--data_root', type=str,   default='',
                   help='Dataset root containing run_NNN dirs. '
                        'Loops over all runs; PSMAPs and prior ACS computed once.')
    p.add_argument('--data_dir',  type=str,
                   default=str(REPO / 'data' / DEFAULT_DATASET / 'run_000'),
                   help='Single run dir (ignored when --data_root is given)')
    p.add_argument('--out_dir',   type=str,   default='',
                   help='Output dir for sweep results (default: results/mle_distributions/)')
    p.add_argument('--out',       type=str,   default='',
                   help='Output pkl path (single-run mode only)')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)s  %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('prior_inference')

    log.info('Config: n_eta=%d  n_qmc_int=%d  n_theta=%d  max_shots=%s',
             args.n_eta, args.n_qmc_int, args.n_theta, args.max_shots)

    # ── Resolve run directories ───────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root)
        run_dirs  = sorted(data_root.glob('run_*'))
        out_dir   = Path(args.out_dir or str(REPO / 'results' / 'mle_distributions'))
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info('Sweep: %d runs in %s  ->  %s', len(run_dirs), data_root, out_dir)
    else:
        run_dirs = [Path(args.data_dir)]
        out_dir  = None

    # ── Load PSMAPs once ──────────────────────────────────────────────────────
    log.info('Loading PSMAPs...')
    t0 = time.perf_counter()
    psmap_z0   = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_Z0.h5'))
    psmap_z100 = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_Z100.h5'))
    _ds_tmp = ImageShotDataset(str(run_dirs[0] / 'Z0' / 'data_IMG.h5'))
    edges = np.linspace(-_ds_tmp.half_range, _ds_tmp.half_range, args.bins + 1)
    del _ds_tmp
    sur_z0   = PSMAPSurrogate(psmap_z0,   args.t_det, use_gpu=USE_GPU)
    sur_z100 = PSMAPSurrogate(psmap_z100, args.t_det, use_gpu=USE_GPU)
    acs_z0   = SurrogatePixelACS(sur_z0,   args.t_det, edges, edges, n_quad=1)
    acs_z100 = SurrogatePixelACS(sur_z100, args.t_det, edges, edges, n_quad=1)
    log.info('Evaluators ready in %.1fs', time.perf_counter() - t0)

    # ── Sample prior and precompute ACS — once for all runs ──────────────────
    log.info('Sampling %d prior eta vectors (Sobol QMC)...', args.n_eta)
    eta_prior, log_w_prior = sample_prior_eta(args.n_eta, rng_seed=0)

    log.info('Precomputing prior ACS for Z0 (%d eta × %d cloud samples)...',
             args.n_eta, args.n_qmc_int)
    t0 = time.perf_counter()
    acs_prior_z0   = batched_integrated_acs(acs_z0,   eta_prior, args.n_qmc_int, rng_seed=1)
    log.info('  Z0  done in %.1fs', time.perf_counter() - t0)

    log.info('Precomputing prior ACS for Z100...')
    t0 = time.perf_counter()
    acs_prior_z100 = batched_integrated_acs(acs_z100, eta_prior, args.n_qmc_int, rng_seed=2)
    log.info('  Z100 done in %.1fs', time.perf_counter() - t0)

    # ── Loop over runs ────────────────────────────────────────────────────────
    for run_dir in run_dirs:
        log.info('===== %s =====', run_dir.name)
        payload = _run_one(run_dir, acs_prior_z0, acs_prior_z100,
                           log_w_prior, eta_prior, args, log)

        if out_dir is not None:
            out_path = out_dir / f'prior_{run_dir.name}.pkl'
        else:
            out_path = Path(args.out or str(REPO / 'results' / 'prior_inference.pkl'))

        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        log.info('Saved -> %s', out_path)


if __name__ == '__main__':
    main()
