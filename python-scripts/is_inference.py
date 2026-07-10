#!/usr/bin/env python
"""
is_inference.py — IS marginalisation over cloud nuisances for signal inference.

    L_i(β) = ∫ p(image_i | β, η) p(η) dη

              ≈ Σ_j  w̃_j · p(image_i | β, η_j) / q(η_j)
                 η_j ~ q(η)

Two proposals (--proposal):

  prior     : q = p(η),       w̃_j = 1/N (uniform)
              η shared across all shots — O(N_eta) ACS evaluations.

  posterior : q = p(η | M_i), w̃_j ∝ p(η_j) / p(η_j | M_i)
              where p(η | M_i) is the Kalman posterior given shot-i image moments.
              η per shot — O(N_shots × N_eta) ACS evaluations.

Likelihood: count-level Poisson on total (n_g, n_e) per AI per shot,
marginalised over the common laser phase on a uniform φ-grid (logsumexp),
then jointly optimised in (As, Ac).

Convergence diagnostics per shot (posterior mode only):
  ESS = [Σ exp(log w_j)]² / Σ exp(2 log w_j)   ∈ [1, N_eta]
  log-weight std — large values warn of poor IS efficiency

Signal uncertainty:
  Bootstrap over shots (resample with replacement, re-optimise β)
  → bootstrap std on (As, Ac).

Output pkl: same schema as map_inference.py — drop-in replacement.

Usage
-----
    # single run, posterior IS
    python python-scripts/is_inference.py

    # prior IS baseline (= prior_inference.py but with bootstrap CI)
    python python-scripts/is_inference.py --proposal prior

    # 20-run sweep
    DATA=data/R20_N200_...
    nohup python python-scripts/is_inference.py --data_root $DATA \\
        > logs/is_posterior_sweep.log 2>&1 &
    nohup python python-scripts/is_inference.py --data_root $DATA --proposal prior \\
        > logs/is_prior_sweep.log 2>&1 &
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
from scipy.stats.qmc import Sobol
from scipy.stats import norm as sp_norm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'helpers'))
sys.path.insert(0, str(REPO / 'python-scripts'))

from helpers import ImageShotDataset                     # noqa
from profile_cloud_nuisances import SurrogatePixelACS   # noqa

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

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_BINS        = 32
DEFAULT_N_ETA       = 512
DEFAULT_N_QMC_INT   = 512     # cloud QMC samples per η for ACS
DEFAULT_N_THETA     = 128
DEFAULT_T_DET       = 3.8
DEFAULT_N_STARTS    = 8
DEFAULT_MAX_SHOTS   = None
DEFAULT_GRID_N      = 31
DEFAULT_GRID_HALF   = 0.2
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_NUGGET      = 1e-3    # fraction of trace(P_eta)/10 added to P_post diagonal

TAU_MU_POS  = 10e-6
TAU_MU_VEL  = 10e-6
BAR_SIG_POS = 100e-6
BAR_SIG_VEL = 100e-6
SIG_SIG_POS = 10e-6
SIG_SIG_VEL = 10e-6

DEFAULT_DATASET = (
    'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
    'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000'
)

EPS = 1e-300

# ── Prior / Kalman machinery ──────────────────────────────────────────────────

def build_A(T):
    """4×10 linear observation matrix M = A η."""
    A = np.zeros((4, 10))
    A[0, 0] = 1.0; A[0, 1] = T
    A[1, 2] = 1.0; A[1, 3] = T
    A[2, 4] = 1.0; A[2, 5] = 2*T; A[2, 6] = T**2
    A[3, 7] = 1.0; A[3, 8] = 2*T; A[3, 9] = T**2
    return A

def build_prior(T):
    """10D Gaussian prior mean and covariance on η."""
    tau_V_pos = 2 * BAR_SIG_POS * SIG_SIG_POS
    tau_C     = BAR_SIG_POS * SIG_SIG_VEL + BAR_SIG_VEL * SIG_SIG_POS
    tau_V_vel = 2 * BAR_SIG_VEL * SIG_SIG_VEL
    m = np.array([0, 0, 0, 0,
                  BAR_SIG_POS**2, 0, BAR_SIG_VEL**2,
                  BAR_SIG_POS**2, 0, BAR_SIG_VEL**2])
    P = np.diag([TAU_MU_POS**2, TAU_MU_VEL**2,
                 TAU_MU_POS**2, TAU_MU_VEL**2,
                 tau_V_pos**2, tau_C**2, tau_V_vel**2,
                 tau_V_pos**2, tau_C**2, tau_V_vel**2])
    return m, P

def kalman_precompute(m_eta, P_eta, A):
    """Return (K, A_m_eta, P_post): Kalman gain, A@m_eta, posterior covariance."""
    S      = A @ P_eta @ A.T
    K      = P_eta @ A.T @ np.linalg.inv(S)
    P_post = P_eta - K @ A @ P_eta    # (10×10), rank-deficient with R=0
    return K, A @ m_eta, P_post

def kalman_map(M_i, m_eta, K, A_m_eta):
    """η̂_i = m_eta + K(M_i - A m_eta); clip variances to ≥0."""
    eta_hat = m_eta + K @ (M_i - A_m_eta)
    eta_hat[[4, 6, 7, 9]] = np.maximum(eta_hat[[4, 6, 7, 9]], 0.0)
    return eta_hat

def port_summed_moments(img, pixel_centers):
    """(N_tot, mu_x, mu_y, var_x, var_y) from a (2, res, res) image."""
    px    = img[0].sum(0) + img[1].sum(0)
    py    = img[0].sum(1) + img[1].sum(1)
    N_tot = float(px.sum())
    if N_tot < 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mu_x  = float((px * pixel_centers).sum() / N_tot)
    mu_y  = float((py * pixel_centers).sum() / N_tot)
    var_x = float((px * (pixel_centers - mu_x)**2).sum() / N_tot)
    var_y = float((py * (pixel_centers - mu_y)**2).sum() / N_tot)
    return N_tot, mu_x, mu_y, var_x, var_y


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_prior_8d(n_eta, seed=0):
    """
    Sobol QMC from 8D prior p(eta) in (mu_x0, mu_y0, mu_vx0, mu_vy0,
    sig_x0, sig_y0, sig_vx0, sig_vy0) parameterisation.
    Returns (N_eta, 8).
    """
    sampler = Sobol(d=8, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_eta, 1))))
    u = sampler.random_base2(m)[:n_eta]
    z = sp_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    means = np.array([0, 0, 0, 0,
                      BAR_SIG_POS, BAR_SIG_POS, BAR_SIG_VEL, BAR_SIG_VEL])
    stds  = np.array([TAU_MU_POS, TAU_MU_POS, TAU_MU_VEL, TAU_MU_VEL,
                      SIG_SIG_POS, SIG_SIG_POS, SIG_SIG_VEL, SIG_SIG_VEL])
    eta = (means + stds * z).astype(np.float64)
    eta[:, 4:] = np.clip(eta[:, 4:], 1e-8, None)
    return eta

def null_space_sampler(A, P_eta):
    """
    Precompute the null-space basis N_null (10×6) of A and the Cholesky factor
    L_gamma of the conditional prior covariance in the null-space basis.

    For R=0 Kalman: the posterior lives on the affine subspace {η : Aη = M_i}.
    Sampling η = η̂_i + N_null γ, γ ~ N(0, Σ_γ), gives the correct conditional
    prior p(η | Aη = M_i) with IS weights = 1 (uniform) — no weight collapse.

    Σ_γ = (N_null^T P_eta^{-1} N_null)^{-1}

    Returns (N_null (10,6), L_gamma (6,6)).
    """
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    # A is (4,10) rank-4, last 6 rows of Vt span null(A)
    N_null = Vt[4:].T                                    # (10,6)
    P_eta_inv = np.linalg.inv(P_eta)
    Sigma_gamma_inv = N_null.T @ P_eta_inv @ N_null      # (6,6)
    Sigma_gamma     = np.linalg.inv(Sigma_gamma_inv)
    Sigma_gamma     = 0.5 * (Sigma_gamma + Sigma_gamma.T)  # symmetrise
    L_gamma = np.linalg.cholesky(Sigma_gamma + 1e-15 * np.eye(6))
    return N_null, L_gamma

def sample_ridge(eta_hat, N_null, L_gamma, n_eta, rng):
    """
    Sample from the R=0 conditional prior p(η | Aη = A η̂).

    η_j = η̂ + N_null γ_j,  γ_j ~ N(0, Σ_γ)

    By construction A η_j = A η̂ = M_i exactly, so IS weights are uniform (log_w=0).
    Returns (samples (n_eta,10), log_w (n_eta,) = zeros).
    """
    z = rng.standard_normal((n_eta, 6))
    gamma   = (L_gamma @ z.T).T                          # (n_eta, 6)
    samples = eta_hat[None, :] + (N_null @ gamma.T).T    # (n_eta, 10)
    samples[:, [4, 6, 7, 9]] = np.maximum(samples[:, [4, 6, 7, 9]], 0.0)
    log_w = np.zeros(n_eta)
    return samples, log_w

def log_ess(log_weights):
    """log ESS = log[(Σ exp lw)² / Σ exp(2 lw)]."""
    lw_norm = log_weights - sp_logsumexp(log_weights)
    return -sp_logsumexp(2.0 * lw_norm)


# ── ACS computation ───────────────────────────────────────────────────────────

def _cloud_chol_10d(eta):
    """(10,) → (mu_4, L_4×4) in PSMAP ordering (x0, y0, vx0, vy0)."""
    mu = np.array([eta[0], eta[2], eta[1], eta[3]])
    Sigma = np.array([
        [eta[4],  0,       eta[5],  0      ],
        [0,       eta[7],  0,       eta[8] ],
        [eta[5],  0,       eta[6],  0      ],
        [0,       eta[8],  0,       eta[9] ],
    ])
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(Sigma), 1e-14)))
    return mu, L

def _acs_from_pts_gpu(pts_flat_g, acs_obj, N, n_qmc):
    """GPU batched ACS eval from (N*n_qmc, 4) CuPy array. Returns (N, 6) numpy."""
    xp = cp
    dphi_g, amp0_g, amp1_g = acs_obj._eval_fast(
        pts_flat_g[:, 0], pts_flat_g[:, 1],
        pts_flat_g[:, 2], pts_flat_g[:, 3])
    inter  = acs_obj._port_inter[None]
    A_per  = amp0_g**2 + amp1_g**2
    Cc_per =  inter * 2.0 * amp0_g * amp1_g * xp.cos(dphi_g)
    Cs_per = -inter * 2.0 * amp0_g * amp1_g * xp.sin(dphi_g)
    s0, s1 = acs_obj.s0_g, acs_obj.s1_g

    def _avg(arr, mask):
        return arr[:, mask].sum(-1).reshape(N, n_qmc).mean(-1)

    return xp.stack([
        _avg(A_per, s0), _avg(Cc_per, s0), _avg(Cs_per, s0),
        _avg(A_per, s1), _avg(Cc_per, s1), _avg(Cs_per, s1),
    ], axis=1).get()   # (N, 6) numpy

def acs_from_eta_8d(eta_8d, acs_obj, n_qmc, rng_seed=0):
    """
    ACS for (N_eta, 8) diagonal-sigma prior samples.
    Batched single GPU call via sigma*z sampling (no cross-terms).
    Returns (N_eta, 6).
    """
    if not USE_GPU:
        raise RuntimeError('GPU (cupy) required')
    rng   = np.random.default_rng(rng_seed)
    N     = len(eta_8d)
    z_g   = cp.asarray(rng.standard_normal((n_qmc, 4)).astype(np.float64))
    eta_g = cp.asarray(eta_8d, dtype=cp.float64)
    mu_g  = eta_g[:, :4]   # (N,4): mu_x, mu_y, mu_vx, mu_vy
    sig_g = eta_g[:, 4:]   # (N,4): sig_x, sig_y, sig_vx, sig_vy
    pts   = mu_g[:, None, :] + sig_g[:, None, :] * z_g[None, :, :]
    return _acs_from_pts_gpu(pts.reshape(N * n_qmc, 4), acs_obj, N, n_qmc)

def acs_from_eta_10d(eta_10d, acs_obj, n_qmc, rng_seed=0):
    """
    ACS for (N_eta, 10) full-covariance Kalman-posterior samples.
    Per-η Cholesky sampling; batched GPU call.
    Returns (N_eta, 6).
    """
    if not USE_GPU:
        raise RuntimeError('GPU (cupy) required')
    rng  = np.random.default_rng(rng_seed)
    N    = len(eta_10d)
    pts  = np.empty((N * n_qmc, 4), dtype=np.float64)
    for i, eta in enumerate(eta_10d):
        mu, L = _cloud_chol_10d(eta)
        z = rng.standard_normal((n_qmc, 4))
        pts[i*n_qmc:(i+1)*n_qmc] = mu + (L @ z.T).T
    return _acs_from_pts_gpu(cp.asarray(pts), acs_obj, N, n_qmc)


# ── IS-aware logL ─────────────────────────────────────────────────────────────

def log_f_ai_is(acs_eta, log_w, n_g, n_e, phi_shifted):
    """
    IS-weighted log f_z(phi) for one AI at one shot.

    Computes log[ Σ_j w̃_j · p(n_g, n_e | η_j, phi) ] for each phi in phi_shifted.

    acs_eta    : (N_eta, 6) — [A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e]
    log_w      : (N_eta,)  — raw IS log-weights (prior: all zeros)
    phi_shifted: (n_theta,)
    Returns    : (n_theta,)
    """
    c = np.cos(phi_shifted); s = np.sin(phi_shifted)   # (n_theta,)

    A_g  = acs_eta[:, 0]; Cc_g = acs_eta[:, 1]; Cs_g = acs_eta[:, 2]
    A_e  = acs_eta[:, 3]; Cc_e = acs_eta[:, 4]; Cs_e = acs_eta[:, 5]
    A_tot  = A_g + A_e
    Cc_tot = Cc_g + Cc_e
    Cs_tot = Cs_g + Cs_e

    totACS = (A_tot[:, None]
              + Cc_tot[:, None] * c[None]
              + Cs_tot[:, None] * s[None])              # (N_eta, n_theta)
    L_phi  = (n_g + n_e) / np.maximum(totACS, EPS)

    lam_g = L_phi * np.maximum(
        A_g[:, None] + Cc_g[:, None] * c[None] + Cs_g[:, None] * s[None], EPS)
    lam_e = L_phi * np.maximum(
        A_e[:, None] + Cc_e[:, None] * c[None] + Cs_e[:, None] * s[None], EPS)

    ll = n_g * np.log(lam_g) - lam_g + n_e * np.log(lam_e) - lam_e  # (N_eta, n_theta)

    # Normalized IS: logsumexp(lw + ll) - logsumexp(lw)
    return (sp_logsumexp(log_w[:, None] + ll, axis=0)
            - sp_logsumexp(log_w))                      # (n_theta,)

def total_logL(beta, precomp_list, f_signal, shot_idx_arr, n_theta):
    """Total IS marginal log-likelihood Σ_i log L_i(β)."""
    As, Ac = float(beta[0]), float(beta[1])
    phi  = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    dphi = (As * np.sin(2 * np.pi * f_signal * shot_idx_arr)
            + Ac * np.cos(2 * np.pi * f_signal * shot_idx_arr))
    total = 0.0
    for (sd_z0, sd_z100), dp in zip(precomp_list, dphi):
        lf0   = log_f_ai_is(sd_z0['acs'],   sd_z0['log_w'],
                             sd_z0['n_g'],   sd_z0['n_e'],   phi)
        lf100 = log_f_ai_is(sd_z100['acs'], sd_z100['log_w'],
                             sd_z100['n_g'], sd_z100['n_e'], phi + dp)
        total += float(sp_logsumexp(lf0 + lf100) - np.log(n_theta))
    return total


# ── Optimisation ──────────────────────────────────────────────────────────────

def optimise_beta(precomp_list, f_signal, shot_idx_arr, args, log):
    """Grid scan + multi-start L-BFGS-B over (As, Ac)."""
    gh = args.grid_half
    grid_As = np.linspace(-gh, gh, args.grid_n)
    grid_Ac = np.linspace(-gh, gh, args.grid_n)
    GAS, GAC = np.meshgrid(grid_As, grid_Ac, indexing='ij')

    def neg_logL(beta):
        return -total_logL(beta, precomp_list, f_signal, shot_idx_arr, args.n_theta)

    t0 = time.perf_counter()
    ll_grid = np.array([neg_logL((a, c)) for a, c in zip(GAS.ravel(), GAC.ravel())])
    flat_top = np.argsort(ll_grid.ravel())[:args.n_starts]
    starts   = [(GAS.ravel()[i], GAC.ravel()[i]) for i in flat_top]
    log.info('Grid scan done in %.1fs', time.perf_counter() - t0)

    best = None
    for k, (as0, ac0) in enumerate(starts):
        opt = minimize(neg_logL, [as0, ac0], method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12})
        if best is None or opt.fun < best.fun:
            best = opt
        log.info('  start %d/%d  As0=%.3f Ac0=%.3f  logL=%.2f  %s',
                 k+1, args.n_starts, as0, ac0, -float(opt.fun),
                 'OK' if opt.success else opt.message[:40])
    return best


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(precomp_list, f_signal, shot_idx_arr, beta0, args, log):
    """
    Resample shots with replacement n_bootstrap times, re-optimise β.
    Returns (bootstrap_std (2,), bootstrap_betas (n_bootstrap, 2)).
    """
    n_shots = len(precomp_list)
    rng     = np.random.default_rng(99)
    boots   = []

    def neg_sub(beta, sub, sub_arr):
        return -total_logL(beta, sub, f_signal, sub_arr, args.n_theta)

    for b in range(args.n_bootstrap):
        idx    = rng.integers(0, n_shots, n_shots)
        sub    = [precomp_list[i] for i in idx]
        sub_arr = shot_idx_arr[idx]
        opt = minimize(neg_sub, beta0, args=(sub, sub_arr), method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-10})
        boots.append(opt.x)
        if (b + 1) % 50 == 0:
            log.info('  bootstrap %d/%d', b + 1, args.n_bootstrap)

    boots = np.array(boots)
    return boots.std(axis=0), boots


# ── Run one data directory ────────────────────────────────────────────────────

def _run_one(data_dir, acs_z0, acs_z100,
             prior_acs_z0, prior_acs_z100,   # pre-computed for prior IS (or None)
             m_eta, P_eta, L_prior, K, A_m_eta, P_post, A,
             args, log):

    data_dir = Path(data_dir)
    ds_z0   = ImageShotDataset(str(data_dir / 'Z0'   / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(data_dir / 'Z100' / 'data_IMG.h5'))

    with h5py.File(str(data_dir / 'Z100' / 'data_IMG.h5')) as fh:
        f_signal          = float(fh.attrs['signal_freq'])
        signal_amp_true   = float(fh.attrs['signal_amp'])
        signal_phase_true = float(fh.attrs['signal_phase'])
    As_true = signal_amp_true * np.cos(signal_phase_true)
    Ac_true = signal_amp_true * np.sin(signal_phase_true)
    log.info('True: As=%.4f Ac=%.4f  (amp=%.3f phase=%.3f f=%.4f)',
             As_true, Ac_true, signal_amp_true, signal_phase_true, f_signal)

    shot_ids = list(range(ds_z0.n_shots))
    if args.max_shots is not None:
        shot_ids = shot_ids[:args.max_shots]
    shot_idx_arr = np.array(shot_ids, dtype=np.float64)
    log.info('Shots=%d  proposal=%s  n_eta=%d', len(shot_ids), args.proposal, args.n_eta)

    # Precompute null-space sampler once (shot-independent)
    if args.proposal == 'posterior':
        N_null, L_gamma = null_space_sampler(A, P_eta)
        log.info('Ridge sampler: null-space dim=%d, IS weights=uniform', N_null.shape[1])

    precomp_list  = []
    ess_vals      = []
    log_w_std_vals = []
    trace_P       = float(np.trace(P_eta))
    rng           = np.random.default_rng(42)

    for k, shot_id in enumerate(shot_ids):
        img0 = ds_z0[shot_id]
        img1 = ds_z100[shot_id]
        n_g0 = float(img0[0].sum()); n_e0 = float(img0[1].sum())
        n_g1 = float(img1[0].sum()); n_e1 = float(img1[1].sum())

        if args.proposal == 'prior':
            # ACS shared, IS weights uniform (log_w = 0)
            log_w = np.zeros(args.n_eta)
            sd_z0   = {'acs': prior_acs_z0,   'log_w': log_w, 'n_g': n_g0, 'n_e': n_e0}
            sd_z100 = {'acs': prior_acs_z100,  'log_w': log_w, 'n_g': n_g1, 'n_e': n_e1}
            ess_vals.append(float(args.n_eta))
            log_w_std_vals.append(0.0)

        else:  # posterior — ridge sampling (R=0 conditional prior, IS weights = 1)
            _, mu_xf0, mu_yf0, V_xf0, V_yf0 = port_summed_moments(img0, ds_z0.pixel_centers)
            _, mu_xf1, mu_yf1, V_xf1, V_yf1 = port_summed_moments(img1, ds_z100.pixel_centers)
            eta_hat_0 = kalman_map(np.array([mu_xf0, mu_yf0, V_xf0, V_yf0]),
                                   m_eta, K, A_m_eta)
            eta_hat_1 = kalman_map(np.array([mu_xf1, mu_yf1, V_xf1, V_yf1]),
                                   m_eta, K, A_m_eta)

            smp_0, log_w_0 = sample_ridge(eta_hat_0, N_null, L_gamma, args.n_eta, rng)
            smp_1, log_w_1 = sample_ridge(eta_hat_1, N_null, L_gamma, args.n_eta, rng)

            acs_0 = acs_from_eta_10d(smp_0, acs_z0,   args.n_qmc_int, rng_seed=k*2)
            acs_1 = acs_from_eta_10d(smp_1, acs_z100, args.n_qmc_int, rng_seed=k*2+1)

            sd_z0   = {'acs': acs_0, 'log_w': log_w_0, 'n_g': n_g0, 'n_e': n_e0}
            sd_z100 = {'acs': acs_1, 'log_w': log_w_1, 'n_g': n_g1, 'n_e': n_e1}

            ess_vals.append(float(args.n_eta))        # uniform weights → ESS = N_eta
            log_w_std_vals.append(0.0)

            if k % 20 == 0:
                log.info('  shot %d/%d  eta_hat_0=(%.2e,%.2e,%.2e,%.2e)',
                         k+1, len(shot_ids), eta_hat_0[0], eta_hat_0[2],
                         eta_hat_0[4], eta_hat_0[7])

        precomp_list.append((sd_z0, sd_z100))

    log.info('Precompute done.  mean_ESS=%.0f/%.0f  mean_log_w_std=%.3f',
             np.mean(ess_vals), args.n_eta, np.mean(log_w_std_vals))

    # ── Optimise (As, Ac) ─────────────────────────────────────────────────────
    best_opt  = optimise_beta(precomp_list, f_signal, shot_idx_arr, args, log)
    beta_hat  = best_opt.x
    logL_hat  = -float(best_opt.fun)
    logL_true = total_logL([As_true, Ac_true], precomp_list, f_signal, shot_idx_arr, args.n_theta)
    logL_zero = total_logL([0.0, 0.0],         precomp_list, f_signal, shot_idx_arr, args.n_theta)
    log.info('Optimised: As=%.4f Ac=%.4f  logL=%.2f  (true: As=%.4f Ac=%.4f)',
             beta_hat[0], beta_hat[1], logL_hat, As_true, Ac_true)

    # ── Bootstrap CI ─────────────────────────────────────────────────────────
    log.info('Running %d bootstrap resamples...', args.n_bootstrap)
    t0 = time.perf_counter()
    boot_std, boot_betas = bootstrap_ci(
        precomp_list, f_signal, shot_idx_arr, beta_hat, args, log)
    log.info('Bootstrap std: As=%.4f Ac=%.4f  (%.1fs)',
             boot_std[0], boot_std[1], time.perf_counter() - t0)

    return {
        # ── Signal estimate ─────────────────────────────────────────────────
        'beta_hat':          beta_hat,
        'As_true':           As_true,
        'Ac_true':           Ac_true,
        'f_signal':          f_signal,
        'signal_amp_true':   signal_amp_true,
        'signal_phase_true': signal_phase_true,
        'logL_hat':          logL_hat,
        'logL_true':         logL_true,
        'logL_zero':         logL_zero,
        # ── IS diagnostics ──────────────────────────────────────────────────
        'proposal':          args.proposal,
        'n_eta':             args.n_eta,
        'ess_per_shot':      np.array(ess_vals),
        'ess_mean':          float(np.mean(ess_vals)),
        'ess_min':           float(np.min(ess_vals)),
        'log_w_std_per_shot': np.array(log_w_std_vals),
        'log_w_std_mean':    float(np.mean(log_w_std_vals)),
        # ── Uncertainty ─────────────────────────────────────────────────────
        'bootstrap_std':     boot_std,
        'bootstrap_betas':   boot_betas,
        'shot_ids':          shot_ids,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--proposal',    choices=['prior', 'posterior'], default='posterior',
                   help='IS proposal: prior p(η) or Kalman posterior p(η|M_i)')
    p.add_argument('--n_eta',       type=int,   default=DEFAULT_N_ETA,
                   help='Number of IS samples per shot (posterior) or total (prior)')
    p.add_argument('--n_qmc_int',   type=int,   default=DEFAULT_N_QMC_INT,
                   help='Cloud QMC samples per η for ACS')
    p.add_argument('--n_theta',     type=int,   default=DEFAULT_N_THETA)
    p.add_argument('--bins',        type=int,   default=DEFAULT_BINS)
    p.add_argument('--t_det',       type=float, default=DEFAULT_T_DET)
    p.add_argument('--max_shots',   type=int,   default=DEFAULT_MAX_SHOTS)
    p.add_argument('--grid_n',      type=int,   default=DEFAULT_GRID_N)
    p.add_argument('--grid_half',   type=float, default=DEFAULT_GRID_HALF)
    p.add_argument('--n_starts',    type=int,   default=DEFAULT_N_STARTS)
    p.add_argument('--n_bootstrap', type=int,   default=DEFAULT_N_BOOTSTRAP)
    p.add_argument('--nugget',      type=float, default=DEFAULT_NUGGET,
                   help='Regularisation: fraction of trace(P_eta)/10 added to P_post diagonal')
    p.add_argument('--data_root',   type=str,   default='',
                   help='Sweep mode: glob run_* dirs under this path')
    p.add_argument('--data_dir',    type=str,
                   default=str(REPO / 'data' / DEFAULT_DATASET / 'run_000'),
                   help='Single run dir (ignored when --data_root given)')
    p.add_argument('--out_dir',     type=str,   default='',
                   help='Output dir for sweep (default: results/mle_distributions/)')
    p.add_argument('--out',         type=str,   default='',
                   help='Output pkl path (single-run mode only)')
    args = p.parse_args()

    tag = f'is_{args.proposal}'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)s  %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger(tag)
    log.info('proposal=%s  n_eta=%d  n_qmc_int=%d  n_theta=%d  nugget=%.0e',
             args.proposal, args.n_eta, args.n_qmc_int, args.n_theta, args.nugget)

    # ── Resolve run directories ───────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root)
        run_dirs  = sorted(data_root.glob('run_*'))
        out_dir   = Path(args.out_dir or str(REPO / 'results' / 'mle_distributions'))
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info('Sweep: %d runs  →  %s', len(run_dirs), out_dir)
    else:
        run_dirs = [Path(args.data_dir)]
        out_dir  = None

    # ── Load PSMAPs ───────────────────────────────────────────────────────────
    log.info('Loading PSMAPs...')
    t0 = time.perf_counter()
    psmap_z0   = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_Z0.h5'))
    psmap_z100 = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_Z100.h5'))
    _ds_tmp = ImageShotDataset(str(run_dirs[0] / 'Z0' / 'data_IMG.h5'))
    edges   = np.linspace(-_ds_tmp.half_range, _ds_tmp.half_range, args.bins + 1)
    del _ds_tmp
    sur_z0   = PSMAPSurrogate(psmap_z0,   args.t_det, use_gpu=USE_GPU)
    sur_z100 = PSMAPSurrogate(psmap_z100, args.t_det, use_gpu=USE_GPU)
    acs_z0   = SurrogatePixelACS(sur_z0,   args.t_det, edges, edges, n_quad=1)
    acs_z100 = SurrogatePixelACS(sur_z100, args.t_det, edges, edges, n_quad=1)
    log.info('PSMAPs ready in %.1fs', time.perf_counter() - t0)

    # ── Prior / Kalman setup ──────────────────────────────────────────────────
    A                    = build_A(args.t_det)
    m_eta, P_eta         = build_prior(args.t_det)
    K, A_m_eta, P_post   = kalman_precompute(m_eta, P_eta, A)
    L_prior              = np.linalg.cholesky(P_eta + 1e-30 * np.eye(10))

    # ── Prior IS: compute ACS once (shared across all shots and runs) ─────────
    prior_acs_z0 = prior_acs_z100 = None
    if args.proposal == 'prior':
        log.info('Pre-computing ACS for %d prior samples...', args.n_eta)
        t0 = time.perf_counter()
        eta_prior = sample_prior_8d(args.n_eta, seed=0)
        prior_acs_z0   = acs_from_eta_8d(eta_prior, acs_z0,   args.n_qmc_int, rng_seed=1)
        prior_acs_z100 = acs_from_eta_8d(eta_prior, acs_z100, args.n_qmc_int, rng_seed=2)
        log.info('Prior ACS done in %.1fs', time.perf_counter() - t0)

    # ── Sweep runs ────────────────────────────────────────────────────────────
    for run_dir in run_dirs:
        log.info('===== %s =====', run_dir.name)
        t_run = time.perf_counter()
        payload = _run_one(
            run_dir, acs_z0, acs_z100,
            prior_acs_z0, prior_acs_z100,
            m_eta, P_eta, L_prior, K, A_m_eta, P_post, A,
            args, log)

        if out_dir is not None:
            out_path = out_dir / f'{tag}_{run_dir.name}.pkl'
        elif args.out:
            out_path = Path(args.out)
        else:
            stem = run_dirs[0].parent.name
            out_path = REPO / 'results' / f'{tag}_{stem}.pkl'

        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        log.info('Saved → %s  (%.0fs)', out_path, time.perf_counter() - t_run)


if __name__ == '__main__':
    main()
