"""
moment_inference.py — 4D Gauss-Hermite ridge integral for signal inference.

Pipeline (Section 5, notes/image_likelihood.tex)
-------------------------------------------------
1. Per shot per AI: compute total port counts (n_g, n_e) and port-summed
   moments (N_tot, mu_xf, mu_yf, sigma_xf^2, sigma_yf^2).

2. R=0 Kalman: treat sample moments as exact.  Each of the four 2D blocks
   (mu_x, mu_y, sigma^2_x, sigma^2_y) collapses from 2D to a 1D ridge
   manifold, yielding a 1D Gaussian prior on the free parameter along each
   ridge.  Total 8D nuisance space → 4D ridge.

3. 4D Gauss-Hermite quadrature over the ridge (n_gh_eta nodes per dim,
   n_gh_eta^4 total points).  For each quadrature point reconstruct the full
   8D eta vector, evaluate the spatially-integrated ACS (A, Cc, Cs per port
   per AI — 3 scalars each, no pixel binning) via a single batched PSMAP
   call, store with the GH weight.

4. For a trial beta = (As, Ac):
   - dphi_i = As*sin(2*pi*f*i) + Ac*cos(2*pi*f*i)
   - For each phi on a uniform grid:
       log f_z0(phi)        = logsumexp_j [log w_j + Poisson logL for Z0 at phi]
       log f_z100(phi+dphi) = logsumexp_j [log w_j + Poisson logL for Z100]
   - logL_i = logsumexp_phi [log f_z0(phi) + log f_z100(phi+dphi)] - log K
   - logL = sum_i logL_i

5. Grid scan and L-BFGS-B optimise logL(beta) over (As, Ac).

Usage
-----
    python python-scripts/moment_inference.py
    nohup python python-scripts/moment_inference.py --n_gh_eta 4 > logs/moment_inference.log 2>&1 &
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
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp as sp_logsumexp

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
DEFAULT_BINS      = 32       # only for moment extraction (image downsampling)
DEFAULT_N_GH_ETA  = 4        # GH order per ridge dimension (total = n_gh_eta^4)
DEFAULT_N_QMC_INT = 2048     # QMC samples per GH-eta point for integrated ACS
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


# ── Port-summed moments ───────────────────────────────────────────────────────
def port_summed_moments(img, pixel_centers):
    """img: (2, res, res). Returns N_tot, mu_x, mu_y, var_x, var_y."""
    total = img[0].astype(np.float64) + img[1].astype(np.float64)
    N_tot = float(total.sum())
    px = total.sum(axis=1); py = total.sum(axis=0)
    mu_x  = float((px * pixel_centers).sum() / N_tot)
    mu_y  = float((py * pixel_centers).sum() / N_tot)
    var_x = float((px * (pixel_centers - mu_x) ** 2).sum() / N_tot)
    var_y = float((py * (pixel_centers - mu_y) ** 2).sum() / N_tot)
    return N_tot, mu_x, mu_y, var_x, var_y


# ── 1D conditional prior on the free parameter along each ridge ───────────────
def ridge_prior_mean(mu_xf, T):
    """
    R=0: free param mu_x0 on the line mu_x0 + T*mu_vx0 = mu_xf.
    Prior: (mu_x0, mu_vx0) ~ N(0, diag(tau_pos^2, tau_vel^2)).
    Returns (mean, std) of 1D Gaussian prior on mu_x0 along the ridge.
    """
    tau_p2 = TAU_MU_POS ** 2
    tau_v2 = TAU_MU_VEL ** 2
    prec = 1.0 / tau_p2 + 1.0 / (T ** 2 * tau_v2)
    v = 1.0 / prec
    m = v * mu_xf / (T ** 2 * tau_v2)
    return m, np.sqrt(v)


def ridge_prior_var(sigma_xf2, T):
    """
    R=0: free param s_x0 = sigma_x0^2 on s_x0 + T^2*s_vx0 = sigma_xf^2.
    Linearised prior in s-space from Gaussian prior on sigma_x0.
    Returns (mean, std) of 1D Gaussian prior on s_x0.
    """
    mu_sp  = BAR_SIG_POS ** 2
    mu_sv  = BAR_SIG_VEL ** 2
    tau_sp2 = (2 * BAR_SIG_POS * SIG_SIG_POS) ** 2
    tau_sv2 = (2 * BAR_SIG_VEL * SIG_SIG_VEL) ** 2
    z    = sigma_xf2 - mu_sp - T ** 2 * mu_sv
    prec = 1.0 / tau_sp2 + 1.0 / (T ** 4 * tau_sv2)
    v    = 1.0 / prec
    m    = mu_sp + v * z / tau_sp2
    return m, np.sqrt(v)


def gh_nodes_logw(mean, std, n_gh):
    """GH nodes and log weights for N(mean, std^2). Returns (nodes, log_w) each (n_gh,)."""
    xi, wi = hermgauss(n_gh)
    nodes  = mean + std * np.sqrt(2.0) * xi
    log_w  = np.log(wi) - 0.5 * np.log(np.pi)
    return nodes, log_w


def build_gh_product_grid(m_mx, s_mx, m_my, s_my, m_sx, s_sx, m_sy, s_sy,
                           sigma_xf2, sigma_yf2, n_gh):
    """
    n_gh^4 GH product-grid points on the 4D ridge.

    Returns:
        eta_free : (n_gh^4, 4) — (mu_x0, mu_y0, s_x0, s_y0)
        log_w    : (n_gh^4,)
    """
    n_mx, lw_mx = gh_nodes_logw(m_mx, s_mx, n_gh)
    n_my, lw_my = gh_nodes_logw(m_my, s_my, n_gh)
    n_sx, lw_sx = gh_nodes_logw(m_sx, s_sx, n_gh)
    n_sy, lw_sy = gh_nodes_logw(m_sy, s_sy, n_gh)

    ii, jj, kk, ll = np.meshgrid(range(n_gh), range(n_gh),
                                   range(n_gh), range(n_gh), indexing='ij')
    ii = ii.ravel(); jj = jj.ravel(); kk = kk.ravel(); ll = ll.ravel()

    eta_free = np.column_stack([n_mx[ii], n_my[jj], n_sx[kk], n_sy[ll]])
    # Clip variance nodes to physically valid range [0, sigma_xf2]
    eta_free[:, 2] = np.clip(eta_free[:, 2], 0.0, float(sigma_xf2))
    eta_free[:, 3] = np.clip(eta_free[:, 3], 0.0, float(sigma_yf2))

    log_w = lw_mx[ii] + lw_my[jj] + lw_sx[kk] + lw_sy[ll]
    return eta_free, log_w


def reconstruct_eta(eta_free, mu_xf, mu_yf, sigma_xf2, sigma_yf2, T):
    """(N,4) free params → (N,8) full eta vectors."""
    mu_x0 = eta_free[:, 0]; mu_y0 = eta_free[:, 1]
    s_x0  = eta_free[:, 2]; s_y0  = eta_free[:, 3]

    mu_vx0 = (mu_xf  - mu_x0) / T
    mu_vy0 = (mu_yf  - mu_y0) / T
    s_vx0  = np.maximum((sigma_xf2 - s_x0) / T ** 2, 0.0)
    s_vy0  = np.maximum((sigma_yf2 - s_y0) / T ** 2, 0.0)

    return np.column_stack([
        mu_x0, mu_y0, mu_vx0, mu_vy0,
        np.sqrt(np.maximum(s_x0,  1e-30)),
        np.sqrt(np.maximum(s_y0,  1e-30)),
        np.sqrt(s_vx0),
        np.sqrt(s_vy0),
    ])


# ── Batched spatially-integrated ACS ─────────────────────────────────────────
def batched_integrated_acs(acs_obj, eta_batch, n_qmc, rng_seed=0):
    """
    Compute total (spatially integrated) ACS for a batch of eta vectors.

    Parameters
    ----------
    acs_obj   : SurrogatePixelACS — only _eval_fast and port-state masks used
    eta_batch : (N, 8) numpy float64
    n_qmc     : int — QMC cloud samples per eta point
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
    mu_g   = eta_g[:, :4]                              # (N, 4)
    sig_g  = eta_g[:, 4:]                              # (N, 4)

    # Cloud samples: (N, n_qmc, 4) — broadcast multiply
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
        # (N*n_qmc, ...) → sum over ports → reshape (N, n_qmc) → mean over n_qmc
        return arr[:, mask].sum(-1).reshape(N, n_qmc).mean(1)

    out = xp.stack([
        _avg(A_per,  s0), _avg(Cc_per, s0), _avg(Cs_per, s0),
        _avg(A_per,  s1), _avg(Cc_per, s1), _avg(Cs_per, s1),
    ], axis=1)     # (N, 6)

    return out.get()   # numpy


# ── Per-shot precompute ───────────────────────────────────────────────────────
def precompute_shot(mu_xf, mu_yf, sigma_xf2, sigma_yf2,
                    n_g_tot, n_e_tot,
                    acs_obj, n_gh_eta, n_qmc_int, t_det, rng_seed):
    """
    Returns dict:
        acs_gh  : (n_gh_eta^4, 6) integrated ACS at each GH point
        log_w   : (n_gh_eta^4,)   log GH weights
        n_g_tot : float            total ground-state count
        n_e_tot : float            total excited-state count
    """
    T = t_det
    m_mx, s_mx = ridge_prior_mean(mu_xf, T)
    m_my, s_my = ridge_prior_mean(mu_yf, T)
    m_sx, s_sx = ridge_prior_var(sigma_xf2, T)
    m_sy, s_sy = ridge_prior_var(sigma_yf2, T)
    m_sx = float(np.clip(m_sx, 0.0, sigma_xf2))
    m_sy = float(np.clip(m_sy, 0.0, sigma_yf2))

    eta_free, log_w = build_gh_product_grid(
        m_mx, s_mx, m_my, s_my, m_sx, s_sx, m_sy, s_sy,
        sigma_xf2, sigma_yf2, n_gh_eta)

    eta_full = reconstruct_eta(eta_free, mu_xf, mu_yf, sigma_xf2, sigma_yf2, T)
    acs_gh   = batched_integrated_acs(acs_obj, eta_full, n_qmc_int, rng_seed)

    return {'acs_gh': acs_gh, 'log_w': log_w,
            'n_g_tot': float(n_g_tot), 'n_e_tot': float(n_e_tot)}


# ── logL ──────────────────────────────────────────────────────────────────────
def log_f_ai(shot_dict, phi_shifted):
    """
    log f_z(phi) = logsumexp_j [log w_j + Poisson logL at phi_shifted]
    for one AI over the GH quadrature.

    phi_shifted : (n_theta,) array (phi or phi+dphi)
    Returns: (n_theta,) array
    """
    EPS = 1e-300
    acs = shot_dict['acs_gh']    # (N_gh, 6)
    lw  = shot_dict['log_w']     # (N_gh,)
    n_g = shot_dict['n_g_tot']
    n_e = shot_dict['n_e_tot']

    c = np.cos(phi_shifted); s = np.sin(phi_shifted)  # (n_theta,)

    A_g  = acs[:, 0]; Cc_g = acs[:, 1]; Cs_g = acs[:, 2]
    A_e  = acs[:, 3]; Cc_e = acs[:, 4]; Cs_e = acs[:, 5]

    A_tot  = A_g + A_e                                    # (N_gh,)
    Cc_tot = Cc_g + Cc_e                                  # (N_gh,)
    Cs_tot = Cs_g + Cs_e                                  # (N_gh,)

    # phi-dependent normalization: λ_g + λ_e = n_tot for every phi.
    # Required because Cc_tot ≠ 0 (complementarity violated by path pruning).
    totACS = (A_tot[:, None]
              + Cc_tot[:, None] * c[None]
              + Cs_tot[:, None] * s[None])                # (N_gh, n_theta)
    L_phi  = (n_g + n_e) / np.maximum(totACS, EPS)       # (N_gh, n_theta)

    lam_g = L_phi * np.maximum(
        A_g[:, None] + Cc_g[:, None] * c[None] + Cs_g[:, None] * s[None], EPS)
    lam_e = L_phi * np.maximum(
        A_e[:, None] + Cc_e[:, None] * c[None] + Cs_e[:, None] * s[None], EPS)

    ll = (n_g * np.log(lam_g) - lam_g +
          n_e * np.log(lam_e) - lam_e)    # (N_gh, n_theta)

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


def _run_one(data_dir, acs_z0, acs_z100, args, log):
    """Process one run directory. Returns the payload dict."""
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

    # ── Per-shot precompute ───────────────────────────────────────────────────
    precomp_list = []
    diag_rows    = []
    t_loop = time.perf_counter()

    for count, shot_id in enumerate(shot_ids, start=1):
        t_shot = time.perf_counter()
        img0 = ds_z0[shot_id]
        img1 = ds_z100[shot_id]

        N0, mux0_f, muy0_f, vx0_f, vy0_f = port_summed_moments(img0, ds_z0.pixel_centers)
        N1, mux1_f, muy1_f, vx1_f, vy1_f = port_summed_moments(img1, ds_z100.pixel_centers)

        n_g0_tot = float(img0[0].sum())
        n_e0_tot = float(img0[1].sum())
        n_g1_tot = float(img1[0].sum())
        n_e1_tot = float(img1[1].sum())

        sd_z0 = precompute_shot(
            mux0_f, muy0_f, vx0_f, vy0_f,
            n_g0_tot, n_e0_tot,
            acs_z0, args.n_gh_eta, args.n_qmc_int, args.t_det,
            rng_seed=count * 2)
        sd_z100 = precompute_shot(
            mux1_f, muy1_f, vx1_f, vy1_f,
            n_g1_tot, n_e1_tot,
            acs_z100, args.n_gh_eta, args.n_qmc_int, args.t_det,
            rng_seed=count * 2 + 1)

        precomp_list.append((sd_z0, sd_z100))

        meta0 = ds_z0.meta(shot_id)
        meta1 = ds_z100.meta(shot_id)
        diag_rows.append({
            'shot': shot_id,
            'N0': N0, 'mux0_f': mux0_f, 'muy0_f': muy0_f, 'vx0_f': vx0_f, 'vy0_f': vy0_f,
            'N1': N1, 'mux1_f': mux1_f, 'muy1_f': muy1_f, 'vx1_f': vx1_f, 'vy1_f': vy1_f,
            'theta_true_z0':   np.array([meta0['mu_x0'], meta0['mu_y0'], meta0['mu_vx0'],
                                         meta0['mu_vy0'], meta0['sigma_x'], meta0['sigma_y'],
                                         meta0['sigma_vx'], meta0['sigma_vy']]),
            'theta_true_z100': np.array([meta1['mu_x0'], meta1['mu_y0'], meta1['mu_vx0'],
                                         meta1['mu_vy0'], meta1['sigma_x'], meta1['sigma_y'],
                                         meta1['sigma_vx'], meta1['sigma_vy']]),
            'delta_phi_true': float(meta1['delta_phi']),
        })

        elapsed = time.perf_counter() - t_loop
        eta_s   = (elapsed / count) * (len(shot_ids) - count)
        if count % 10 == 0 or count == len(shot_ids):
            log.info('[%3d/%d] shot=%4d  shot_s=%.2f  ETA=%.0fm%.0fs',
                     count, len(shot_ids), shot_id,
                     time.perf_counter() - t_shot, eta_s // 60, eta_s % 60)

    log.info('Precompute done in %.1fs', time.perf_counter() - t_loop)

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
        'config':            vars(args),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bins',      type=int,   default=DEFAULT_BINS)
    p.add_argument('--n_gh_eta',  type=int,   default=DEFAULT_N_GH_ETA,
                   help='GH order per ridge dimension (n_gh_eta^4 total GH pts)')
    p.add_argument('--n_qmc_int', type=int,   default=DEFAULT_N_QMC_INT,
                   help='QMC samples per GH-eta point for integrated ACS')
    p.add_argument('--n_theta',   type=int,   default=DEFAULT_N_THETA)
    p.add_argument('--t_det',     type=float, default=DEFAULT_T_DET)
    p.add_argument('--max_shots', type=int,   default=DEFAULT_MAX_SHOTS)
    p.add_argument('--grid_n',    type=int,   default=DEFAULT_GRID_N)
    p.add_argument('--grid_half', type=float, default=DEFAULT_GRID_HALF)
    p.add_argument('--n_starts',  type=int,   default=DEFAULT_N_STARTS,
                   help='Number of L-BFGS-B restarts (includes start at 0)')
    p.add_argument('--data_root', type=str,   default='',
                   help='Dataset root containing run_NNN dirs. '
                        'Loops over all runs; PSMAPs loaded once.')
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
    log = logging.getLogger('moment_inference')

    log.info('Config: n_gh_eta=%d (%d GH pts/shot/AI)  n_qmc_int=%d  n_theta=%d  max_shots=%s',
             args.n_gh_eta, args.n_gh_eta ** 4, args.n_qmc_int, args.n_theta, args.max_shots)

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
    # Edges derived from first run; assumed identical across runs
    _ds_tmp = ImageShotDataset(str(run_dirs[0] / 'Z0' / 'data_IMG.h5'))
    edges = np.linspace(-_ds_tmp.half_range, _ds_tmp.half_range, args.bins + 1)
    del _ds_tmp
    sur_z0   = PSMAPSurrogate(psmap_z0,   args.t_det, use_gpu=USE_GPU)
    sur_z100 = PSMAPSurrogate(psmap_z100, args.t_det, use_gpu=USE_GPU)
    acs_z0   = SurrogatePixelACS(sur_z0,   args.t_det, edges, edges, n_quad=1)
    acs_z100 = SurrogatePixelACS(sur_z100, args.t_det, edges, edges, n_quad=1)
    log.info('Evaluators ready in %.1fs', time.perf_counter() - t0)

    # ── Loop over runs ────────────────────────────────────────────────────────
    for run_dir in run_dirs:
        log.info('===== %s =====', run_dir.name)
        payload = _run_one(run_dir, acs_z0, acs_z100, args, log)

        if out_dir is not None:
            out_path = out_dir / f'moment_{run_dir.name}.pkl'
        else:
            out_path = Path(args.out or str(REPO / 'results' / 'moment_inference.pkl'))

        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        log.info('Saved -> %s', out_path)


if __name__ == '__main__':
    main()
