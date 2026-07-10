#!/usr/bin/env python
"""
full_image_is_inference.py — Full per-bin IS marginalisation for signal inference.

Uses the complete spatial image (per-bin Poisson counts) without profiling the
cloud nuisances.  Cloud parameters η are marginalised by importance sampling (IS)
from the moment posterior p(η | m_i):

    L_i(β) = ∫ p(image_i | β, η) p(η) dη

           ≈  1/M  Σ_j  w_j · p(image_i | β, η_j)
                η_j ~ q(η) = p(η | m_i, α)

where w_j = p(η_j)/q(η_j) corrects for using the (inflated) moment posterior
q instead of the flat prior p(η).

Differences from is_inference.py
---------------------------------
- Likelihood: full per-bin Poisson (all spatial bins) instead of 2-scalar total
- Proposal: finite-R 8D moment posterior p(η | m_i, α) instead of R=0 ridge
- IS weights: non-trivial 8D Gaussian ratio p(η)/q(η) (well-conditioned for R>0)
- ACS: per-bin (N_bins,6) via SemiAnalyticPixelACS 2D GH quadrature per sample

The IS weights w_j = p(η)/q(η) are β-independent and serve as a correction
factor; the function being integrated, p(image | β, η_j), carries the β-dependence.
The ESS = [Σ w_j]² / Σ w_j² diagnoses how much the full spatial image reshapes
p(η) relative to the moment-only posterior.

Usage
-----
    python python-scripts/full_image_is_inference.py

    python python-scripts/full_image_is_inference.py \\
        --n-eta 64 --bins 20 --n-gh 4 --n-theta 128 --max-shots 20

    # with a non-default run
    python python-scripts/full_image_is_inference.py --data_root /path/to/run
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
import scipy.linalg
from scipy.optimize import minimize
from scipy.special import logsumexp as sp_logsumexp, ndtr
from scipy.stats import norm as sp_norm
from scipy.stats.qmc import Sobol

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'helpers'))
sys.path.insert(0, str(REPO / 'python-scripts'))

from helpers import ImageShotDataset                            # noqa
from profile_cloud_nuisances import (                          # noqa
    SurrogatePixelACS, SemiAnalyticPixelACS, _downsample,
)

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

DEFAULT_BINS        = 32       # bins×bins spatial grid (2048/32=64 exact)
DEFAULT_N_ETA       = 64       # IS samples per shot per AI
DEFAULT_N_GH        = 4        # GH order per velocity dimension (4² = 16 pts/pixel)
DEFAULT_N_THETA     = 128      # phase-grid points
DEFAULT_T_DET       = 3.8      # detection time [s]
DEFAULT_N_STARTS    = 8
DEFAULT_MAX_SHOTS   = None
DEFAULT_GRID_N      = 21
DEFAULT_GRID_HALF   = 0.2
DEFAULT_ALPHA       = 4.0      # proposal covariance inflation factor
DEFAULT_CHUNK_BINS  = 64       # bins per GPU call in pixel_acs

# 8D prior parameters
TAU_MU_POS  = 10e-6    # prior std on μ_x0, μ_y0  [m]
TAU_MU_VEL  = 10e-6    # prior std on μ_vx0, μ_vy0 [m/s]
BAR_SIG_POS = 100e-6   # prior mean of σ_x0, σ_y0  [m]
BAR_SIG_VEL = 100e-6   # prior mean of σ_vx0, σ_vy0 [m/s]
SIG_SIG_POS = 10e-6    # prior std  of σ_x0, σ_y0  [m]
SIG_SIG_VEL = 10e-6    # prior std  of σ_vx0, σ_vy0 [m/s]

EPS = 1e-300

DEFAULT_DATASET = (
    'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
    'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000'
)

# 8D ordering: (μ_x0, μ_y0, μ_vx0, μ_vy0, σ_x0, σ_y0, σ_vx0, σ_vy0)
# Block indices:
#   mean-x:  (μ_x0, μ_vx0)  → [0, 2]
#   mean-y:  (μ_y0, μ_vy0)  → [1, 3]
#   sigma-x: (σ_x0, σ_vx0)  → [4, 6]
#   sigma-y: (σ_y0, σ_vy0)  → [5, 7]


# ── 8D prior ───────────────────────────────────────────────────────────────────

def build_prior_8d():
    """8D diagonal prior p(η) on (μ_x0, μ_y0, μ_vx0, μ_vy0, σ_x0, σ_y0, σ_vx0, σ_vy0)."""
    m = np.array([0.0, 0.0, 0.0, 0.0,
                  BAR_SIG_POS, BAR_SIG_POS, BAR_SIG_VEL, BAR_SIG_VEL])
    P_diag = np.array([TAU_MU_POS**2, TAU_MU_POS**2,
                       TAU_MU_VEL**2, TAU_MU_VEL**2,
                       SIG_SIG_POS**2, SIG_SIG_POS**2,
                       SIG_SIG_VEL**2, SIG_SIG_VEL**2])
    return m, P_diag   # (8,), (8,) — diagonal stored as vector


# ── 8D moment posterior (4 independent 2D Kalman blocks) ──────────────────────

def _kalman_2d_scalar_obs(m, P_diag, h, y, R):
    """
    2D Kalman update: prior N(m, diag(P_diag)), observation y = h^T θ + ε, ε~N(0,R).
    Returns (eta_hat, P_post) where P_post is the full (2,2) posterior covariance.
    """
    P = np.diag(P_diag)          # (2,2)
    Ph = P @ h                   # (2,)
    S = h @ Ph + R               # scalar
    K = Ph / S                   # (2,)  Kalman gain
    eta_hat = m + K * (y - h @ m)
    P_post  = P - np.outer(K, Ph)
    return eta_hat, P_post


def moment_posterior_8d(N_tot, mu_xf, mu_yf, V_xf, V_yf, T):
    """
    Compute 8D moment posterior from image statistics.

    Parameters
    ----------
    N_tot : float  — detected atom count
    mu_xf, mu_yf  — observed final CoM [m]
    V_xf, V_yf    — observed final variance [m²]
    T             — detection time [s]

    Returns
    -------
    eta_hat : (8,) posterior mean
    P_post  : (8,8) posterior covariance (block-diagonal)
    """
    # Mean block x: (μ_x0, μ_vx0) | μ_xf, h = [1, T]
    h_mu   = np.array([1.0, T])
    R_mu_x = max(V_xf / N_tot, 1e-30)
    eta_mu_x, P_mu_x = _kalman_2d_scalar_obs(
        np.array([0.0, 0.0]),
        np.array([TAU_MU_POS**2, TAU_MU_VEL**2]),
        h_mu, mu_xf, R_mu_x)

    # Mean block y
    R_mu_y = max(V_yf / N_tot, 1e-30)
    eta_mu_y, P_mu_y = _kalman_2d_scalar_obs(
        np.array([0.0, 0.0]),
        np.array([TAU_MU_POS**2, TAU_MU_VEL**2]),
        h_mu, mu_yf, R_mu_y)

    # Sigma block x: (σ_x0, σ_vx0) | V_xf via delta-method linearisation.
    #
    # The linearised observation model is:
    #   V_xf ≈ (σ̄_pos² + T²σ̄_vel²) + 2σ̄_pos(σ_x0 - σ̄_pos) + 2T²σ̄_vel(σ_vx0 - σ̄_vel)
    #        = h_σ^T σ  - (σ̄_pos² + T²σ̄_vel²)     with h_σ = (2σ̄_pos, 2T²σ̄_vel)
    #
    # Rearranging: y_eq ≡ V_xf + V̄_xf = h_σ^T σ  (standard form y = h^T θ).
    # This gives innovation = y_eq - h_σ^T m_prior = V_xf - V̄_xf (correct delta).
    V_bar = BAR_SIG_POS**2 + T**2 * BAR_SIG_VEL**2
    h_sig  = np.array([2*BAR_SIG_POS, 2*T**2*BAR_SIG_VEL])
    R_V_x  = max(2.0 * V_xf**2 / N_tot, 1e-30)
    eta_sig_x, P_sig_x = _kalman_2d_scalar_obs(
        np.array([BAR_SIG_POS, BAR_SIG_VEL]),
        np.array([SIG_SIG_POS**2, SIG_SIG_VEL**2]),
        h_sig, V_xf + V_bar, R_V_x)
    eta_sig_x = np.maximum(eta_sig_x, 1e-8)

    # Sigma block y
    R_V_y  = max(2.0 * V_yf**2 / N_tot, 1e-30)
    eta_sig_y, P_sig_y = _kalman_2d_scalar_obs(
        np.array([BAR_SIG_POS, BAR_SIG_VEL]),
        np.array([SIG_SIG_POS**2, SIG_SIG_VEL**2]),
        h_sig, V_yf + V_bar, R_V_y)
    eta_sig_y = np.maximum(eta_sig_y, 1e-8)

    # Assemble 8D mean (ordering: μ_x0, μ_y0, μ_vx0, μ_vy0, σ_x0, σ_y0, σ_vx0, σ_vy0)
    eta_hat = np.array([
        eta_mu_x[0],  eta_mu_y[0],   # μ_x0, μ_y0
        eta_mu_x[1],  eta_mu_y[1],   # μ_vx0, μ_vy0
        eta_sig_x[0], eta_sig_y[0],  # σ_x0, σ_y0
        eta_sig_x[1], eta_sig_y[1],  # σ_vx0, σ_vy0
    ])

    # Assemble 8D covariance (block-diagonal, non-zero blocks at [0,2], [1,3], [4,6], [5,7])
    P_post = np.zeros((8, 8))
    ix_mx  = np.ix_([0, 2], [0, 2])
    ix_my  = np.ix_([1, 3], [1, 3])
    ix_sx  = np.ix_([4, 6], [4, 6])
    ix_sy  = np.ix_([5, 7], [5, 7])
    P_post[ix_mx] = P_mu_x
    P_post[ix_my] = P_mu_y
    P_post[ix_sx] = P_sig_x
    P_post[ix_sy] = P_sig_y

    return eta_hat, P_post


def port_summed_moments(img, pixel_centers):
    """(N_tot, mu_x, mu_y, var_x, var_y) from a (2, res, res) image."""
    px    = img[0].sum(0) + img[1].sum(0)
    py    = img[0].sum(1) + img[1].sum(1)
    N_tot = float(px.sum())
    if N_tot < 1:
        return 1.0, 0.0, 0.0, (BAR_SIG_POS**2 + (DEFAULT_T_DET*BAR_SIG_VEL)**2), (BAR_SIG_POS**2 + (DEFAULT_T_DET*BAR_SIG_VEL)**2)
    mu_x  = float((px * pixel_centers).sum() / N_tot)
    mu_y  = float((py * pixel_centers).sum() / N_tot)
    var_x = float((px * (pixel_centers - mu_x)**2).sum() / N_tot)
    var_y = float((py * (pixel_centers - mu_y)**2).sum() / N_tot)
    return N_tot, mu_x, mu_y, max(var_x, 1e-15), max(var_y, 1e-15)


# ── IS machinery ───────────────────────────────────────────────────────────────

def sample_proposal_8d(eta_hat, P_post, alpha, n_eta, rng):
    """
    Sample from the α-inflated moment posterior q(η) = N(η̂, α × P_post).

    alpha > 1 inflates the covariance to give heavier tails, ensuring the
    proposal covers the prior tails (IS correctness).

    Returns
    -------
    samples : (n_eta, 8)
    L_q     : (8,8) Cholesky of α × P_post (for reuse in log-weight computation)
    """
    P_q = alpha * P_post
    P_q = 0.5 * (P_q + P_q.T)  # symmetrise
    P_q += 1e-18 * np.eye(8)    # numerical floor
    L_q = np.linalg.cholesky(P_q)
    z = rng.standard_normal((n_eta, 8))
    samples = eta_hat[None, :] + (L_q @ z.T).T  # (n_eta, 8)
    samples[:, 4:] = np.maximum(samples[:, 4:], 1e-8)  # clip σ to positive
    return samples, L_q


def _gauss_logpdf_diag(x, m, P_diag):
    """Log N(x; m, diag(P_diag)) for x of shape (N, 8)."""
    d = x - m[None, :]                       # (N, 8)
    return -0.5 * (d**2 / P_diag[None, :]).sum(1) - 0.5 * np.log(P_diag).sum() - 4 * np.log(2*np.pi)


def _gauss_logpdf_chol(x, m, L):
    """Log N(x; m, L L^T) for x of shape (N, 8), L Cholesky factor (8,8)."""
    d = x - m[None, :]                       # (N, 8)
    v = np.linalg.solve(L, d.T).T            # (N, 8): L^{-1} (x-m)
    return -0.5 * (v**2).sum(1) - np.log(np.diag(L)).sum() - 4 * np.log(2*np.pi)


def log_is_weights(samples, eta_hat, L_q, m_prior, P_prior_diag):
    """
    Compute IS log-weights: log w_j = log p(η_j) - log q(η_j).

    p(η) = N(m_prior, diag(P_prior_diag))  (8D diagonal prior)
    q(η) = N(eta_hat, L_q L_q^T)           (inflated moment posterior)
    """
    log_p = _gauss_logpdf_diag(samples, m_prior, P_prior_diag)  # (N_eta,)
    log_q = _gauss_logpdf_chol(samples, eta_hat, L_q)            # (N_eta,)
    return log_p - log_q


def log_ess(log_weights):
    """Effective sample size diagnostic: log ESS = log[(Σ exp lw)² / Σ exp(2 lw)]."""
    lw_norm = log_weights - sp_logsumexp(log_weights)
    return -sp_logsumexp(2.0 * lw_norm)


# ── Per-bin ACS precomputation ─────────────────────────────────────────────────

def pixel_acs_batch(eta_batch, semi_acs):
    """
    Per-bin ACS for N_eta cloud-parameter vectors.

    Calls SemiAnalyticPixelACS.pixel_acs in a loop (one GPU kernel set per η).
    This is safe for any N_eta and n_gh; see notes §9.2 for memory details.

    Parameters
    ----------
    eta_batch : (N_eta, 8) — cloud params (μ_x0, μ_y0, μ_vx0, μ_vy0, σ_x0, σ_y0, σ_vx0, σ_vy0)
    semi_acs  : SemiAnalyticPixelACS

    Returns
    -------
    acs_batch : (N_eta, n_bins, 6) numpy — [A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e] per bin per η
    """
    N_eta  = len(eta_batch)
    n_bins = semi_acs.n_bins
    result = np.empty((N_eta, n_bins, 6), dtype=np.float64)
    for j, eta in enumerate(eta_batch):
        acs_j = semi_acs.pixel_acs(eta)   # tuple of 6 GPU arrays, each (n_bins,)
        result[j] = np.stack([v.get() for v in acs_j], axis=1)  # (n_bins, 6)
    return result


# ── Per-bin IS-weighted logL ───────────────────────────────────────────────────

def log_f_ai_per_bin(acs_batch, log_w, n_g_b, n_e_b, cos_phi, sin_phi,
                     chunk_bins=64):
    """
    IS-weighted log f_z(φ) for one AI at one shot using full per-bin counts.

    For each phase φ_k:
      Λ_j(k) = N_obs / Σ_b [tot_ACS_bj(φ_k)]          (per-η normalisation)
      λ_g_bjk = Λ_j(k) × [A_g_bj + Cc_g_bj cos + Cs_g_bj sin]
      ll_jk   = Σ_b [n_g_b log λ_g_bjk - λ_g_bjk + n_e_b log λ_e_bjk - λ_e_bjk]
      log f(k) = logsumexp_j [log_w_j + ll_jk] - logsumexp_j [log_w_j]

    Parameters
    ----------
    acs_batch  : (N_eta, n_bins, 6) numpy
    log_w      : (N_eta,) IS log-weights (unnormalised)
    n_g_b, n_e_b : (n_bins,) per-bin observed counts (float)
    cos_phi, sin_phi : (K,) phase-grid values
    chunk_bins : bins processed per memory block

    Returns
    -------
    (K,) log IS-marginalised likelihood over phase
    """
    N_eta, N_bins, _ = acs_batch.shape
    K = len(cos_phi)

    A_g  = acs_batch[:, :, 0]   # (N_eta, N_bins)
    Cc_g = acs_batch[:, :, 1]
    Cs_g = acs_batch[:, :, 2]
    A_e  = acs_batch[:, :, 3]
    Cc_e = acs_batch[:, :, 4]
    Cs_e = acs_batch[:, :, 5]

    # Total ACS summed over bins → (N_eta,)
    A_tot_j  = (A_g + A_e).sum(1)
    Cc_tot_j = (Cc_g + Cc_e).sum(1)
    Cs_tot_j = (Cs_g + Cs_e).sum(1)
    N_obs = float(n_g_b.sum() + n_e_b.sum())

    # Normalisation: Λ_j(k) = N_obs / total_ACS_j(φ_k), shape (N_eta, K)
    total_jk = (A_tot_j[:, None]
                + Cc_tot_j[:, None] * cos_phi[None, :]
                + Cs_tot_j[:, None] * sin_phi[None, :])
    Lambda_jk = N_obs / np.maximum(total_jk, EPS)   # (N_eta, K)

    # Per-bin Poisson logL accumulated over chunks of bins
    ll_jk = np.zeros((N_eta, K), dtype=np.float64)
    for b0 in range(0, N_bins, chunk_bins):
        b1  = min(b0 + chunk_bins, N_bins)
        cb  = b1 - b0

        # (N_eta, cb, K) expected counts
        lam_g_c = Lambda_jk[:, None, :] * np.maximum(
            A_g [:, b0:b1, None] + Cc_g[:, b0:b1, None] * cos_phi[None, None, :]
            + Cs_g[:, b0:b1, None] * sin_phi[None, None, :], EPS)
        lam_e_c = Lambda_jk[:, None, :] * np.maximum(
            A_e [:, b0:b1, None] + Cc_e[:, b0:b1, None] * cos_phi[None, None, :]
            + Cs_e[:, b0:b1, None] * sin_phi[None, None, :], EPS)

        n_g_c = n_g_b[b0:b1]   # (cb,)
        n_e_c = n_e_b[b0:b1]

        ll_jk += (n_g_c[None, :, None] * np.log(lam_g_c) - lam_g_c
                  + n_e_c[None, :, None] * np.log(lam_e_c) - lam_e_c).sum(1)

    lf_k = sp_logsumexp(log_w[:, None] + ll_jk, axis=0) - sp_logsumexp(log_w)

    # Actual ESS: effective samples of the combined IS weight w_j * f_j(β),
    # phase-marginalized → (N_eta,) then standard ESS formula
    v_j = sp_logsumexp(log_w[:, None] + ll_jk, axis=1) - np.log(K)
    ess_actual = float(np.exp(log_ess(v_j)))

    return lf_k, ess_actual


def total_logL_per_bin(beta, precomp_list, f_signal, shot_idx_arr, n_theta,
                       chunk_bins=64):
    """Total IS-marginalised per-bin log-likelihood Σ_i log L_i(β).

    Returns (total_logL, mean_ess_actual) where mean_ess_actual is the
    mean over shots of ESS(w_j * f_j(β)), the true IS effective sample size.
    """
    As, Ac = float(beta[0]), float(beta[1])
    phi  = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    dphi = (As * np.sin(2 * np.pi * f_signal * shot_idx_arr)
            + Ac * np.cos(2 * np.pi * f_signal * shot_idx_arr))
    total    = 0.0
    ess_list = []
    for (sd_z0, sd_z100), dp in zip(precomp_list, dphi):
        c = np.cos(phi); s = np.sin(phi)
        lf0, ess0 = log_f_ai_per_bin(sd_z0['acs'],   sd_z0['log_w'],
                                      sd_z0['n_g_b'], sd_z0['n_e_b'],
                                      c, s, chunk_bins=chunk_bins)
        c100 = np.cos(phi + dp); s100 = np.sin(phi + dp)
        lf100, ess100 = log_f_ai_per_bin(sd_z100['acs'],   sd_z100['log_w'],
                                          sd_z100['n_g_b'], sd_z100['n_e_b'],
                                          c100, s100, chunk_bins=chunk_bins)
        total += float(sp_logsumexp(lf0 + lf100) - np.log(n_theta))
        ess_list.append(0.5 * (ess0 + ess100))
    return total, float(np.mean(ess_list))


# ── Optimisation ──────────────────────────────────────────────────────────────

def optimise_beta(precomp_list, f_signal, shot_idx_arr, args, log):
    """Grid scan + multi-start L-BFGS-B over (As, Ac)."""
    gh = args.grid_half
    gAs = np.linspace(-gh, gh, args.grid_n)
    gAc = np.linspace(-gh, gh, args.grid_n)
    GAS, GAC = np.meshgrid(gAs, gAc, indexing='ij')

    def neg_logL(beta):
        ll, _ = total_logL_per_bin(beta, precomp_list, f_signal, shot_idx_arr,
                                   args.n_theta, chunk_bins=DEFAULT_CHUNK_BINS)
        return -ll

    t0 = time.perf_counter()
    ll_grid  = np.array([neg_logL((a, c)) for a, c in zip(GAS.ravel(), GAC.ravel())])
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


# ── Run one data directory ────────────────────────────────────────────────────

def _run_one(data_dir, eval_z0, eval_z100, m_prior, P_prior_diag, T, args, log):
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
    log.info('Shots=%d  n_eta=%d  alpha=%.1f  bins=%d  n_gh=%d',
             len(shot_ids), args.n_eta, args.alpha, args.bins, args.n_gh)

    # Pixel edges for downsampling and moment computation
    edges   = np.linspace(-ds_z0.half_range, ds_z0.half_range, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])  # 1D — same for x and y

    rng          = np.random.default_rng(42)
    precomp_list = []
    ess_vals     = []

    for k, shot_id in enumerate(shot_ids):
        t_shot = time.perf_counter()
        img0   = _downsample(ds_z0[shot_id],   args.bins)   # (2, bins, bins)
        img1   = _downsample(ds_z100[shot_id], args.bins)

        n_g_b_0 = img0[0].ravel().astype(np.float64)   # (n_bins,)
        n_e_b_0 = img0[1].ravel().astype(np.float64)
        n_g_b_1 = img1[0].ravel().astype(np.float64)
        n_e_b_1 = img1[1].ravel().astype(np.float64)

        # ── Moment posterior for each AI ─────────────────────────────────
        N0, mu_xf0, mu_yf0, V_xf0, V_yf0 = port_summed_moments(img0, centers)
        N1, mu_xf1, mu_yf1, V_xf1, V_yf1 = port_summed_moments(img1, centers)

        eta_hat_0, P_post_0 = moment_posterior_8d(N0, mu_xf0, mu_yf0, V_xf0, V_yf0, T)
        eta_hat_1, P_post_1 = moment_posterior_8d(N1, mu_xf1, mu_yf1, V_xf1, V_yf1, T)

        # ── Sample from α-inflated proposal ──────────────────────────────
        smp_0, L_q_0 = sample_proposal_8d(eta_hat_0, P_post_0, args.alpha, args.n_eta, rng)
        smp_1, L_q_1 = sample_proposal_8d(eta_hat_1, P_post_1, args.alpha, args.n_eta, rng)

        # ── IS weights log w_j = log p(η_j) - log q(η_j) ─────────────────
        log_w_0 = log_is_weights(smp_0, eta_hat_0, L_q_0, m_prior, P_prior_diag)
        log_w_1 = log_is_weights(smp_1, eta_hat_1, L_q_1, m_prior, P_prior_diag)

        # ── Per-bin ACS for each IS sample ───────────────────────────────
        acs_0 = pixel_acs_batch(smp_0, eval_z0)    # (N_eta, n_bins, 6)
        acs_1 = pixel_acs_batch(smp_1, eval_z100)

        sd_z0   = {'acs': acs_0, 'log_w': log_w_0, 'n_g_b': n_g_b_0, 'n_e_b': n_e_b_0}
        sd_z100 = {'acs': acs_1, 'log_w': log_w_1, 'n_g_b': n_g_b_1, 'n_e_b': n_e_b_1}
        precomp_list.append((sd_z0, sd_z100))

        ess_0 = float(np.exp(log_ess(log_w_0)))
        ess_1 = float(np.exp(log_ess(log_w_1)))
        ess_vals.append(0.5 * (ess_0 + ess_1))

        if k % 10 == 0 or k < 3:
            log.info('  shot %d/%d  ESS_Z0=%.0f/%.0f  ESS_Z100=%.0f/%.0f  '
                     'eta_hat_Z0=(%.2e,%.2e,%.2e,%.2e)  (%.1fs)',
                     k+1, len(shot_ids), ess_0, args.n_eta, ess_1, args.n_eta,
                     eta_hat_0[0], eta_hat_0[1], eta_hat_0[4], eta_hat_0[6],
                     time.perf_counter() - t_shot)

    log.info('Precompute done.  mean_ESS_prior=%.1f/%.0f  (p/q only — see actual ESS below)',
             np.mean(ess_vals), args.n_eta)

    # ── Optimise (As, Ac) ─────────────────────────────────────────────────────
    best_opt  = optimise_beta(precomp_list, f_signal, shot_idx_arr, args, log)
    beta_hat  = best_opt.x
    logL_hat  = -float(best_opt.fun)
    logL_true, ess_actual_true = total_logL_per_bin(
        [As_true, Ac_true], precomp_list, f_signal, shot_idx_arr, args.n_theta)
    logL_zero, _               = total_logL_per_bin(
        [0.0, 0.0],         precomp_list, f_signal, shot_idx_arr, args.n_theta)
    # Actual ESS at β_hat: ESS(w_j * f_j(β_hat)), more meaningful than p/q ESS
    _, ess_actual_hat = total_logL_per_bin(
        beta_hat, precomp_list, f_signal, shot_idx_arr, args.n_theta)
    log.info('Optimised: As=%.4f Ac=%.4f  logL=%.2f  (true: As=%.4f Ac=%.4f)',
             beta_hat[0], beta_hat[1], logL_hat, As_true, Ac_true)
    log.info('  logL(true)=%.2f  logL(0,0)=%.2f  signal_LLR=%.2f',
             logL_true, logL_zero, logL_hat - logL_zero)
    log.info('  Actual ESS at beta_hat=%.1f/%.0f  at true_beta=%.1f/%.0f',
             ess_actual_hat, args.n_eta, ess_actual_true, args.n_eta)

    amp_hat   = float(np.sqrt(beta_hat[0]**2 + beta_hat[1]**2))
    phase_hat = float(np.arctan2(beta_hat[1], beta_hat[0]))

    return {
        'beta_hat':          beta_hat,
        'As_true':           As_true,
        'Ac_true':           Ac_true,
        'f_signal':          f_signal,
        'signal_amp_true':   signal_amp_true,
        'signal_phase_true': signal_phase_true,
        'logL_hat':          logL_hat,
        'logL_true':         logL_true,
        'logL_zero':         logL_zero,
        'amp_hat':           amp_hat,
        'phase_hat':         phase_hat,
        'mean_ess_actual':   ess_actual_hat,
        'mean_ess_prior':    float(np.mean(ess_vals)),
        'n_shots':           len(shot_ids),
        'n_eta':             args.n_eta,
        'alpha':             args.alpha,
        'bins':              args.bins,
        'n_gh':              args.n_gh,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

DEFAULT_PSMAP_Z0   = REPO / 'output-files' / 'PSGRID4D_CONFOCAL_Z0.h5'
DEFAULT_PSMAP_Z100 = REPO / 'output-files' / 'PSGRID4D_CONFOCAL_Z100.h5'


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--data_root', type=str, default='',
                   help='Dataset root containing run_NNN dirs. '
                        'Loops over all runs; PSMAPs loaded once.')
    p.add_argument('--data_dir',  type=str,
                   default=str(REPO / 'data' / DEFAULT_DATASET / 'run_000'),
                   help='Single run dir (ignored when --data_root is given)')
    p.add_argument('--out_dir',   type=str, default='',
                   help='Output dir for sweep results (default: results/full_image_is/)')
    p.add_argument('--out',       type=str, default='',
                   help='Output pkl path (single-run mode only)')
    p.add_argument('--psmap-z0',   type=Path, default=DEFAULT_PSMAP_Z0)
    p.add_argument('--psmap-z100', type=Path, default=DEFAULT_PSMAP_Z100)
    p.add_argument('--bins',      type=int,   default=DEFAULT_BINS,
                   help='Inference image resolution (bins×bins). (default: 20)')
    p.add_argument('--n-gh',      type=int,   default=DEFAULT_N_GH,
                   help='GH order per velocity dim; bins²×n_gh² evals per pixel_acs. (default: 4)')
    p.add_argument('--chunk-bins',type=int,   default=DEFAULT_CHUNK_BINS,
                   help='Bins per GPU call inside pixel_acs. (default: 64)')
    p.add_argument('--n-eta',     type=int,   default=DEFAULT_N_ETA,
                   help='IS samples per shot per AI. (default: 64)')
    p.add_argument('--alpha',     type=float, default=DEFAULT_ALPHA,
                   help='Proposal covariance inflation factor. (default: 4.0)')
    p.add_argument('--n-theta',   type=int,   default=DEFAULT_N_THETA,
                   help='Phase-grid points. (default: 128)')
    p.add_argument('--max-shots', type=int,   default=DEFAULT_MAX_SHOTS)
    p.add_argument('--n-starts',  type=int,   default=DEFAULT_N_STARTS)
    p.add_argument('--grid-n',    type=int,   default=DEFAULT_GRID_N)
    p.add_argument('--grid-half', type=float, default=DEFAULT_GRID_HALF)
    p.add_argument('--no-gpu',    action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)s  %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('full_image_is')

    log.info('bins=%d  n_gh=%d  n_eta=%d  alpha=%.1f  n_theta=%d  GPU=%s',
             args.bins, args.n_gh, args.n_eta, args.alpha, args.n_theta,
             USE_GPU and not args.no_gpu)

    # ── Resolve run directories ───────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root)
        run_dirs  = sorted(data_root.glob('run_*'))
        out_dir   = Path(args.out_dir or str(REPO / 'results' / 'full_image_is'))
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info('Sweep: %d runs in %s  ->  %s', len(run_dirs), data_root, out_dir)
    else:
        run_dirs = [Path(args.data_dir)]
        out_dir  = None

    # ── Load PSMAPs once ──────────────────────────────────────────────────────
    log.info('Loading PSMAPs...')
    t0 = time.perf_counter()
    sur_z0   = PSMAPSurrogate(load_psmap(str(args.psmap_z0)),   DEFAULT_T_DET, use_gpu=not args.no_gpu)
    sur_z100 = PSMAPSurrogate(load_psmap(str(args.psmap_z100)), DEFAULT_T_DET, use_gpu=not args.no_gpu)
    _ds_tmp  = ImageShotDataset(str(run_dirs[0] / 'Z0' / 'data_IMG.h5'))
    edges    = np.linspace(-_ds_tmp.half_range, _ds_tmp.half_range, args.bins + 1)
    del _ds_tmp
    _base_z0   = SurrogatePixelACS(sur_z0,   DEFAULT_T_DET, edges, edges, n_quad=1)
    _base_z100 = SurrogatePixelACS(sur_z100, DEFAULT_T_DET, edges, edges, n_quad=1)
    eval_z0    = SemiAnalyticPixelACS(_base_z0,   n_gh=args.n_gh, chunk_bins=args.chunk_bins)
    eval_z100  = SemiAnalyticPixelACS(_base_z100, n_gh=args.n_gh, chunk_bins=args.chunk_bins)
    log.info('Evaluators ready in %.1fs  n_bins=%d  n_v=%d',
             time.perf_counter() - t0, eval_z0.n_bins, eval_z0.n_v)

    m_prior, P_prior_diag = build_prior_8d()

    # ── Loop over runs ────────────────────────────────────────────────────────
    for run_dir in run_dirs:
        log.info('===== %s =====', run_dir.name)
        t0      = time.perf_counter()
        payload = _run_one(run_dir, eval_z0, eval_z100,
                           m_prior, P_prior_diag, DEFAULT_T_DET, args, log)
        payload['elapsed_s'] = time.perf_counter() - t0

        if out_dir is not None:
            out_path = out_dir / f'full_image_is_{run_dir.name}.pkl'
        else:
            out_path = Path(args.out or str(REPO / 'results' / 'full_image_is.pkl'))

        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        log.info('Saved -> %s  (%.1fs)', out_path, payload['elapsed_s'])


if __name__ == '__main__':
    main()
