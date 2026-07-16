"""
map_inference.py — Plug-in MAP signal inference with/without position moments.

Replaces the expensive η-marginalization with a single per-shot constrained MAP
estimate of the initial cloud moments, obtained analytically from the observed
final-image moments (Kalman update with R=0).

η ordering (10D):
    [μ_x0, μ_vx0, μ_y0, μ_vy0, V_x0, C_xv0, V_vx0, V_y0, C_yv0, V_vy0]

Ballistic map  M = A η  (4 measurements):
    μ_xf = μ_x0 + T μ_vx0
    μ_yf = μ_y0 + T μ_vy0
    V_xf = V_x0 + 2T C_xv0 + T² V_vx0
    V_yf = V_y0 + 2T C_yv0 + T² V_vy0

Constrained MAP:
    η̂_i = m_η + P_η Aᵀ (A P_η Aᵀ)⁻¹ (M_i − A m_η)

Three modes:
    --use_moments 1  : η̂_i from final image moments via Kalman update (default).
    --use_moments 0  : η̂_i = m_η for every shot — prior-mean baseline.
    --use_true_eta 1 : η̂_i = true initial cloud state from simulation metadata
                       (oracle / accuracy limit).

In all modes φ_i is still marginalised over a uniform phase grid (logsumexp).

Usage
-----
    # single run, position-resolved
    python python-scripts/map_inference.py

    # no-position-info baseline
    python python-scripts/map_inference.py --use_moments 0

    # oracle accuracy limit
    python python-scripts/map_inference.py --use_true_eta 1

    # sweep over all 20 runs, all modes, background
    DATA=data/R20_N200_...
    nohup python python-scripts/map_inference.py --data_root $DATA \\
        > logs/map_moment_sweep.log 2>&1 &
    nohup python python-scripts/map_inference.py --data_root $DATA --use_moments 0 \\
        > logs/map_prior_sweep.log 2>&1 &
    nohup python python-scripts/map_inference.py --data_root $DATA --use_true_eta 1 \\
        > logs/map_true_eta_sweep.log 2>&1 &
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

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_BINS      = 32
DEFAULT_N_QMC_INT = 2048     # cloud QMC samples per shot per AI for ACS
DEFAULT_N_THETA   = 128
# NOTE: n_theta=128 (grid spacing ~0.049 rad) is only adequate for the phi
# grid-marginalization ('grid' phi_method) at modest photon flux. At high flux
# (e.g. A~1e8) the per-shot phi posterior becomes far narrower than the grid
# spacing, and 'grid' silently returns a badly biased (beta) MAP estimate
# (measured ~12% bias in As at A=1e8, run_000 of R40_N50_A1e8, shrinking away
# only once n_theta is pushed to ~2000-8000). The 'laplace'/'point' phi_method
# paths solve phi continuously per shot and do not suffer this discretization
# bias, which is why they are the default and were used for all production
# results in results/mle_distributions_*. If you use --phi_method grid at high
# flux, verify convergence by checking that beta_hat is stable as --n_theta
# is increased.
DEFAULT_T_DET     = 3.8
DEFAULT_MAX_SHOTS = None
DEFAULT_GRID_N    = 31
DEFAULT_GRID_HALF = 0.2
DEFAULT_N_STARTS  = 8

# Prior on cloud moments — must match data-generation parameters
TAU_MU_POS  = 10e-6     # prior std on μ_x0, μ_y0          [m]
TAU_MU_VEL  = 10e-6     # prior std on μ_vx0, μ_vy0        [m/s]
BAR_SIG_POS = 100e-6    # prior mean of σ_x0, σ_y0         [m]
BAR_SIG_VEL = 100e-6    # prior mean of σ_vx0, σ_vy0       [m/s]
SIG_SIG_POS = 10e-6     # prior std of σ_x0                [m]
SIG_SIG_VEL = 10e-6     # prior std of σ_vx0               [m/s]

DEFAULT_DATASET = (
    'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
    'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000'
)


# ── Ballistic MAP machinery ────────────────────────────────────────────────────

def build_A(T):
    """
    4×10 linear observation matrix  M = A η.

    η : [μ_x0, μ_vx0, μ_y0, μ_vy0, V_x0, C_xv0, V_vx0, V_y0, C_yv0, V_vy0]
    M : [μ_xf, μ_yf, V_xf, V_yf]
    """
    A = np.zeros((4, 10))
    A[0, 0] = 1.0;  A[0, 1] = T                       # μ_xf
    A[1, 2] = 1.0;  A[1, 3] = T                       # μ_yf
    A[2, 4] = 1.0;  A[2, 5] = 2*T;  A[2, 6] = T**2   # V_xf
    A[3, 7] = 1.0;  A[3, 8] = 2*T;  A[3, 9] = T**2   # V_yf
    return A


def build_prior(T):
    """
    Prior mean m_η (10,) and diagonal covariance P_η (10×10).

    Variance block priors use the delta method:
        std(V_x0) ≈ 2 σ̄_pos σ_{σ,pos}
    Cross-covariance prior uses the natural scale σ̄_pos σ̄_vel.
    """
    tau_V_pos = 2 * BAR_SIG_POS * SIG_SIG_POS   # std of V_x0 = σ_x0²
    tau_V_vel = 2 * BAR_SIG_VEL * SIG_SIG_VEL   # std of V_vx0
    tau_C     = BAR_SIG_POS * BAR_SIG_VEL        # std of C_xv0

    m_eta = np.array([
        0.0, 0.0, 0.0, 0.0,                           # μ means
        BAR_SIG_POS**2, 0.0, BAR_SIG_VEL**2,          # x variance block
        BAR_SIG_POS**2, 0.0, BAR_SIG_VEL**2,          # y variance block
    ])
    P_eta = np.diag([
        TAU_MU_POS**2, TAU_MU_VEL**2,                 # μ_x0, μ_vx0
        TAU_MU_POS**2, TAU_MU_VEL**2,                 # μ_y0, μ_vy0
        tau_V_pos**2, tau_C**2, tau_V_vel**2,          # V_x0, C_xv0, V_vx0
        tau_V_pos**2, tau_C**2, tau_V_vel**2,          # V_y0, C_yv0, V_vy0
    ])
    return m_eta, P_eta


def precompute_kalman(m_eta, P_eta, A):
    """
    Precompute the Kalman gain K = P_η Aᵀ (A P_η Aᵀ)⁻¹  (10×4)
    and A @ m_eta for fast per-shot MAP evaluation.
    """
    S = A @ P_eta @ A.T          # (4×4) innovation covariance
    K = P_eta @ A.T @ np.linalg.inv(S)   # (10×4)
    return K, A @ m_eta          # K, A_m_eta


def eta_from_meta(meta):
    """Build true 10D η from simulation metadata (variances = σ², cross-terms = 0)."""
    return np.array([
        meta['mu_x0'],        # μ_x0
        meta['mu_vx0'],       # μ_vx0
        meta['mu_y0'],        # μ_y0
        meta['mu_vy0'],       # μ_vy0
        meta['sigma_x']**2,   # V_x0
        0.0,                  # C_xv0
        meta['sigma_vx']**2,  # V_vx0
        meta['sigma_y']**2,   # V_y0
        0.0,                  # C_yv0
        meta['sigma_vy']**2,  # V_vy0
    ])


def map_eta(M_i, m_eta, K, A_m_eta):
    """η̂_i = m_η + K (M_i − A m_η). Clips variances to positive."""
    eta = m_eta + K @ (M_i - A_m_eta)
    # variance components must be non-negative
    for idx in (4, 6, 7, 9):
        eta[idx] = max(eta[idx], 0.0)
    return eta


# ── Port-summed image moments ──────────────────────────────────────────────────

def port_summed_moments(img, pixel_centers):
    """img: (2, res, res). Returns N_tot, μ_x, μ_y, V_x, V_y."""
    total = img[0].astype(np.float64) + img[1].astype(np.float64)
    N_tot = float(total.sum())
    px = total.sum(axis=1); py = total.sum(axis=0)
    mu_x  = float((px * pixel_centers).sum() / N_tot)
    mu_y  = float((py * pixel_centers).sum() / N_tot)
    var_x = float((px * (pixel_centers - mu_x)**2).sum() / N_tot)
    var_y = float((py * (pixel_centers - mu_y)**2).sum() / N_tot)
    return N_tot, mu_x, mu_y, var_x, var_y


# ── ACS evaluation with full cloud covariance ──────────────────────────────────

def _cloud_cholesky(eta):
    """
    Extract cloud mean (4,) and Cholesky factor L (4×4) from η̂.
    PSMAP input ordering: (x0, y0, vx0, vy0).
    """
    mu = np.array([eta[0], eta[2], eta[1], eta[3]])   # [μ_x0, μ_y0, μ_vx0, μ_vy0]
    V_x0  = eta[4]; C_xv = eta[5]; V_vx0 = eta[6]
    V_y0  = eta[7]; C_yv = eta[8]; V_vy0 = eta[9]
    Sigma = np.array([
        [V_x0,  0.0,   C_xv,  0.0  ],
        [0.0,   V_y0,  0.0,   C_yv ],
        [C_xv,  0.0,   V_vx0, 0.0  ],
        [0.0,   C_yv,  0.0,   V_vy0],
    ])
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # Σ not PSD (numerical issue); fall back to diagonal
        L = np.diag(np.sqrt(np.maximum(np.diag(Sigma), 1e-14)))
    return mu, L


def batch_acs(eta_list, acs_obj, n_qmc, rng_seed=0):
    """
    Compute spatially-integrated ACS for a list of η̂ vectors in one GPU call.

    For each η̂, draws n_qmc cloud samples from N(μ, Σ) via Cholesky.
    All samples are stacked and evaluated in a single batched PSMAP call.

    Parameters
    ----------
    eta_list : list of (10,) arrays  (length n_shots)
    acs_obj  : SurrogatePixelACS
    n_qmc    : cloud samples per η̂
    rng_seed : int

    Returns
    -------
    (n_shots, 6) float64  — [A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e] per shot
    """
    if not USE_GPU:
        raise RuntimeError('GPU (cupy) required')
    n_shots = len(eta_list)
    rng = np.random.default_rng(rng_seed)

    # Build all cloud sample points: (n_shots * n_qmc, 4)
    pts_all = np.empty((n_shots * n_qmc, 4), dtype=np.float64)
    for i, eta in enumerate(eta_list):
        mu, L = _cloud_cholesky(eta)
        z = rng.standard_normal((n_qmc, 4))
        pts_all[i*n_qmc:(i+1)*n_qmc] = mu + (L @ z.T).T

    # Single batched GPU PSMAP call
    pts_g = cp.asarray(pts_all)
    dphi_g, amp0_g, amp1_g = acs_obj._eval_fast(
        pts_g[:, 0], pts_g[:, 1], pts_g[:, 2], pts_g[:, 3])

    inter  = acs_obj._port_inter[None]
    A_per  = amp0_g**2 + amp1_g**2
    Cc_per =  inter * 2.0 * amp0_g * amp1_g * cp.cos(dphi_g)
    Cs_per = -inter * 2.0 * amp0_g * amp1_g * cp.sin(dphi_g)

    s0, s1 = acs_obj.s0_g, acs_obj.s1_g

    def _avg(arr, mask):
        # (n_shots*n_qmc, n_ports) → select mask → (n_shots*n_qmc,)
        #   → (n_shots, n_qmc) → mean over n_qmc → (n_shots,)
        return arr[:, mask].sum(-1).reshape(n_shots, n_qmc).mean(-1)

    out = cp.stack([
        _avg(A_per, s0), _avg(Cc_per, s0), _avg(Cs_per, s0),
        _avg(A_per, s1), _avg(Cc_per, s1), _avg(Cs_per, s1),
    ], axis=1)   # (n_shots, 6)

    return out.get()   # numpy


def _gh_nodes_weights(n_order):
    """
    Tensor-product Gauss-Hermite nodes/weights for 4D N(0,I), normalised to sum=1.

    Returns
    -------
    T : (n_order**4, 4)  nodes in the standardised space (before Cholesky transform)
    W : (n_order**4,)    weights summing to 1
    """
    t1d, w1d = np.polynomial.hermite.hermgauss(n_order)
    grids = np.meshgrid(t1d, t1d, t1d, t1d, indexing='ij')
    T = np.stack([g.ravel() for g in grids], axis=1)          # (n^4, 4)
    W = np.ones(n_order**4)
    for g in np.meshgrid(w1d, w1d, w1d, w1d, indexing='ij'):
        W *= g.ravel()
    W /= np.pi**2   # normalise: π^(d/2) for d=4
    return T, W


def batch_acs_gh(eta_list, acs_obj, n_order, chunk_shots=None):
    """
    Compute spatially-integrated ACS using a tensor-product Gauss-Hermite rule.

    Exact for polynomial integrands up to degree (2*n_order - 1) per dimension.
    n_order=3 → 81 deterministic points/shot; n_order=4 → 256; n_order=12 →
    20,736 (4D tensor product over the cloud's full initial (x0,y0,vx0,vy0)
    state -- this integrates the WHOLE cloud unconditionally, unlike the 2D
    per-pixel conditional-velocity integral used elsewhere in this codebase,
    because this is the port-summed/spatially-integrated ACS, not a per-pixel
    one). At n_order=12 the (nP, 256, n_quad) Catmull-Rom gather tensor in
    SurrogatePixelACS._eval_fast is too large for all 200 shots in one GPU
    call (measured OOM at ~17GB for a single allocation, independent of
    --bins). chunk_shots processes eta_list in smaller batches and
    concatenates results -- purely a memory-management change, identical
    numerics to an unchunked call.

    Parameters
    ----------
    eta_list : list of (10,) arrays
    acs_obj  : SurrogatePixelACS
    n_order  : int  GH points per dimension
    chunk_shots : int or None  process at most this many shots per GPU call
        (None = all at once, the original behaviour)

    Returns
    -------
    (n_shots, 6) float64
    """
    if not USE_GPU:
        raise RuntimeError('GPU (cupy) required')
    if chunk_shots is not None and len(eta_list) > chunk_shots:
        chunks = [eta_list[i:i + chunk_shots] for i in range(0, len(eta_list), chunk_shots)]
        return np.concatenate([batch_acs_gh(c, acs_obj, n_order, chunk_shots=None)
                                for c in chunks], axis=0)

    T, W = _gh_nodes_weights(n_order)   # (n_gh, 4), (n_gh,)
    n_gh   = len(W)
    n_shots = len(eta_list)

    # Build sample points: x = mu + sqrt(2) * L @ t  for each GH node t
    pts_all = np.empty((n_shots * n_gh, 4), dtype=np.float64)
    for i, eta in enumerate(eta_list):
        mu, L = _cloud_cholesky(eta)
        pts_all[i*n_gh:(i+1)*n_gh] = mu + np.sqrt(2) * (L @ T.T).T

    pts_g  = cp.asarray(pts_all)
    W_g    = cp.asarray(W)   # (n_gh,)

    dphi_g, amp0_g, amp1_g = acs_obj._eval_fast(
        pts_g[:, 0], pts_g[:, 1], pts_g[:, 2], pts_g[:, 3])

    inter  = acs_obj._port_inter[None]
    A_per  = amp0_g**2 + amp1_g**2
    Cc_per =  inter * 2.0 * amp0_g * amp1_g * cp.cos(dphi_g)
    Cs_per = -inter * 2.0 * amp0_g * amp1_g * cp.sin(dphi_g)

    s0, s1 = acs_obj.s0_g, acs_obj.s1_g

    def _wavg(arr, mask):
        # (n_shots*n_gh, n_ports) → select port → (n_shots*n_gh,)
        #   → (n_shots, n_gh) → weighted sum over n_gh → (n_shots,)
        vals = arr[:, mask].sum(-1).reshape(n_shots, n_gh)  # (n_shots, n_gh)
        return (vals * W_g[None]).sum(-1)

    out = cp.stack([
        _wavg(A_per, s0), _wavg(Cc_per, s0), _wavg(Cs_per, s0),
        _wavg(A_per, s1), _wavg(Cc_per, s1), _wavg(Cs_per, s1),
    ], axis=1)   # (n_shots, 6)

    return out.get()


# ── logL ───────────────────────────────────────────────────────────────────────

def log_f_ai(shot_dict, phi_shifted):
    """
    log f_z(phi) = logsumexp_j [log w_j + Poisson logL at phi_shifted].

    With a single-point plug-in (N_pts=1, log_w=[0]), this reduces to just
    the Poisson logL at the MAP cloud estimate.

    phi_shifted : (n_theta,) — returns (n_theta,)
    """
    EPS = 1e-300
    acs = shot_dict['acs']     # (N_pts, 6)
    lw  = shot_dict['log_w']   # (N_pts,)
    n_g = shot_dict['n_g_tot']
    n_e = shot_dict['n_e_tot']

    c = np.cos(phi_shifted); s = np.sin(phi_shifted)   # (n_theta,)

    A_g  = acs[:, 0]; Cc_g = acs[:, 1]; Cs_g = acs[:, 2]
    A_e  = acs[:, 3]; Cc_e = acs[:, 4]; Cs_e = acs[:, 5]
    A_tot  = A_g + A_e
    Cc_tot = Cc_g + Cc_e
    Cs_tot = Cs_g + Cs_e

    totACS = (A_tot[:, None] + Cc_tot[:, None]*c[None] + Cs_tot[:, None]*s[None])
    L_phi  = (n_g + n_e) / np.maximum(totACS, EPS)

    lam_g = L_phi * np.maximum(A_g[:, None] + Cc_g[:, None]*c[None] + Cs_g[:, None]*s[None], EPS)
    lam_e = L_phi * np.maximum(A_e[:, None] + Cc_e[:, None]*c[None] + Cs_e[:, None]*s[None], EPS)

    ll = n_g*np.log(lam_g) - lam_g + n_e*np.log(lam_e) - lam_e   # (N_pts, n_theta)
    return sp_logsumexp(lw[:, None] + ll, axis=0)                  # (n_theta,)


def total_logL(beta, precomp_list, f_signal, shot_idx_arr, n_theta):
    As, Ac = float(beta[0]), float(beta[1])
    phi  = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    dphi = As*np.sin(2*np.pi*f_signal*shot_idx_arr) + Ac*np.cos(2*np.pi*f_signal*shot_idx_arr)
    total = 0.0
    for (sz0, sz100), dp in zip(precomp_list, dphi):
        lf0   = log_f_ai(sz0,   phi)
        lf100 = log_f_ai(sz100, phi + dp)
        total += float(sp_logsumexp(lf0 + lf100) - np.log(n_theta))
    return total


# ── Fast per-shot phi optimization (point-estimate / Laplace) ──────────────────
#
# Replaces the n_theta-grid marginalization over the per-shot nuisance phase phi
# with a direct optimization of the (already shown to be effectively unimodal)
# combined z0+z100 per-shot log-likelihood, plus an optional Laplace (curvature)
# correction that approximates the marginal integral. All shots are solved
# simultaneously via vectorized numpy ops (cheap: pure arithmetic on the
# precomputed ACS coefficients, no PSMAP re-evaluation).

def _ai_ll_batch(phi, acs_arr, n_g, n_e):
    """Per-shot Poisson log-likelihood at a single phi value per shot.
    phi, n_g, n_e : (n_shots,)      acs_arr : (n_shots, 6)
    """
    EPS = 1e-300
    A_g, Cc_g, Cs_g = acs_arr[:, 0], acs_arr[:, 1], acs_arr[:, 2]
    A_e, Cc_e, Cs_e = acs_arr[:, 3], acs_arr[:, 4], acs_arr[:, 5]
    A_tot, Cc_tot, Cs_tot = A_g + A_e, Cc_g + Cc_e, Cs_g + Cs_e
    c, s = np.cos(phi), np.sin(phi)
    totACS = A_tot + Cc_tot*c + Cs_tot*s
    L_phi  = (n_g + n_e) / np.maximum(totACS, EPS)
    lam_g  = L_phi * np.maximum(A_g + Cc_g*c + Cs_g*s, EPS)
    lam_e  = L_phi * np.maximum(A_e + Cc_e*c + Cs_e*s, EPS)
    return n_g*np.log(lam_g) - lam_g + n_e*np.log(lam_e) - lam_e


def _combined_ll_batch(phi, acs_z0_arr, n_g0, n_e0, acs_z100_arr, n_g1, n_e1, dphi):
    ll0 = _ai_ll_batch(phi,        acs_z0_arr,   n_g0, n_e0)
    ll1 = _ai_ll_batch(phi + dphi, acs_z100_arr, n_g1, n_e1)
    return ll0 + ll1


def _newton_polish(ll, phi0, n_newton, h):
    """Vectorized damped-Newton polish from a given (n_shots,) starting phi."""
    phi = phi0.copy()
    for _ in range(n_newton):
        f0, fp, fm = ll(phi), ll(phi + h), ll(phi - h)
        grad = (fp - fm) / (2*h)
        curv = -(fp - 2*f0 + fm) / h**2   # curvature of -ll (positive at a max of ll)
        curv_safe = np.where(curv > 1e-8, curv, 1e-8)
        step = np.clip(grad / curv_safe, -0.5, 0.5)
        phi = phi + step
    f0, fp, fm = ll(phi), ll(phi + h), ll(phi - h)
    ll_star = f0
    curv = -(fp - 2*f0 + fm) / h**2
    curv = np.maximum(curv, 1e-8)
    return phi, ll_star, curv


def solve_phi_batch(acs_z0_arr, n_g0, n_e0, acs_z100_arr, n_g1, n_e1, dphi,
                     n_scan=64, n_newton=12, h=1e-5, n_multistart=1):
    """
    Vectorized per-shot solve for the combined-likelihood optimum phi_i*.

    Historically this assumed the combined z0+z100 likelihood is "effectively
    unimodal" for this port-summed statistic (see module docstring) and used
    only ONE Newton polish from the single best coarse-scan point. That
    assumption was never verified here, and an analogous assumption was found
    to be WRONG for the full-pixel likelihood (joint_profile_inference.py:
    100% of a 50-shot sample is multimodal, 3-4 basins per shot). This is now
    checked directly: with n_multistart > 1, Newton-polish the top-n_multistart
    local maxima found by the coarse scan (not just the single best), and keep
    whichever converges to the highest log-likelihood per shot -- i.e. genuine
    multistart, not scan-then-polish-once. n_multistart=1 recovers the
    original behaviour exactly (for backward compatibility / comparison).

    Returns
    -------
    phi_star   : (n_shots,)  argmax location
    ll_star    : (n_shots,)  log-likelihood at the optimum
    curv       : (n_shots,)  curvature of -log-likelihood at the optimum (>=0)
    n_modes    : (n_shots,)  # of distinct local maxima found by the coarse scan
                 (diagnostic only -- multimodality census for this statistic)
    """
    n_shots = acs_z0_arr.shape[0]

    def ll(phi_val):
        return _combined_ll_batch(phi_val, acs_z0_arr, n_g0, n_e0,
                                   acs_z100_arr, n_g1, n_e1, dphi)

    scan = np.linspace(0.0, 2*np.pi, n_scan, endpoint=False)
    scan_ll = np.stack([ll(np.full(n_shots, p)) for p in scan], axis=0)  # (n_scan, n_shots)

    if n_multistart <= 1:
        best_idx = np.argmax(scan_ll, axis=0)
        best_phi = scan[best_idx]
        n_modes = np.ones(n_shots, dtype=int)  # not computed in single-start mode
        return _newton_polish(ll, best_phi, n_newton, h) + (n_modes,)

    # local maxima of the scan, per shot, periodic boundary (same logic as
    # diagnose_phi_multimodality.py's local_minima_idx, sign-flipped for a
    # log-likelihood maximum instead of an NLL minimum).
    is_max = (scan_ll >= np.roll(scan_ll, 1, axis=0)) & (scan_ll >= np.roll(scan_ll, -1, axis=0))
    n_modes = is_max.sum(axis=0)

    starts = np.empty((n_multistart, n_shots))
    for s in range(n_shots):
        idx = np.where(is_max[:, s])[0]
        if len(idx) == 0:
            idx = np.array([int(np.argmax(scan_ll[:, s]))])
        order = idx[np.argsort(-scan_ll[idx, s])]  # descending by ll
        top = order[:n_multistart]
        if len(top) < n_multistart:
            top = np.concatenate([top, np.repeat(top[-1], n_multistart - len(top))])
        starts[:, s] = scan[top]

    best_phi = np.full(n_shots, np.nan)
    best_ll = np.full(n_shots, -np.inf)
    best_curv = np.full(n_shots, np.nan)
    for k in range(n_multistart):
        phi_k, ll_k, curv_k = _newton_polish(ll, starts[k], n_newton, h)
        better = ll_k > best_ll
        best_phi = np.where(better, phi_k, best_phi)
        best_ll = np.where(better, ll_k, best_ll)
        best_curv = np.where(better, curv_k, best_curv)

    return best_phi, best_ll, best_curv, n_modes


def total_logL_fast(beta, acs_z0_arr, n_g0, n_e0, acs_z100_arr, n_g1, n_e1,
                     f_signal, shot_idx_arr, phi_method='laplace',
                     n_scan=64, n_newton=12, n_multistart=1):
    As, Ac = float(beta[0]), float(beta[1])
    dphi = As*np.sin(2*np.pi*f_signal*shot_idx_arr) + Ac*np.cos(2*np.pi*f_signal*shot_idx_arr)
    _, ll_star, curv, _ = solve_phi_batch(acs_z0_arr, n_g0, n_e0, acs_z100_arr, n_g1, n_e1,
                                           dphi, n_scan=n_scan, n_newton=n_newton,
                                           n_multistart=n_multistart)
    if phi_method == 'laplace':
        log_terms = ll_star + 0.5*np.log(2*np.pi) - 0.5*np.log(curv)
    elif phi_method == 'point':
        log_terms = ll_star
    else:
        raise ValueError(f'unknown phi_method {phi_method!r}')
    return float(np.sum(log_terms))


# ── Per-run inference ──────────────────────────────────────────────────────────

def _run_one(data_dir, acs_z0, acs_z100, m_eta, K, A_m_eta, args, log):
    """
    Process one run directory.  Returns payload dict.
    """
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
    log.info('Using %d/%d shots  use_moments=%s', len(shot_ids), ds_z0.n_shots, args.use_moments)
    shot_idx_arr = np.array(shot_ids, dtype=np.float64)

    # ── Compute η̂ per shot per AI ────────────────────────────────────────────
    t0 = time.perf_counter()
    eta_z0_list   = []
    eta_z100_list = []
    counts_z0     = []   # (n_g, n_e) per shot
    counts_z100   = []

    for shot_id in shot_ids:
        img0 = ds_z0[shot_id]
        img1 = ds_z100[shot_id]

        n_g0 = float(img0[0].sum()); n_e0 = float(img0[1].sum())
        n_g1 = float(img1[0].sum()); n_e1 = float(img1[1].sum())
        counts_z0.append((n_g0, n_e0))
        counts_z100.append((n_g1, n_e1))

        if args.use_true_eta:
            meta0 = ds_z0.meta(shot_id)
            meta1 = ds_z100.meta(shot_id)
            eta_z0_list.append(eta_from_meta(meta0))
            eta_z100_list.append(eta_from_meta(meta1))
        elif args.use_moments:
            _, mu_xf0, mu_yf0, V_xf0, V_yf0 = port_summed_moments(img0, ds_z0.pixel_centers)
            _, mu_xf1, mu_yf1, V_xf1, V_yf1 = port_summed_moments(img1, ds_z100.pixel_centers)
            eta_z0_list.append(map_eta(np.array([mu_xf0, mu_yf0, V_xf0, V_yf0]),
                                       m_eta, K, A_m_eta))
            eta_z100_list.append(map_eta(np.array([mu_xf1, mu_yf1, V_xf1, V_yf1]),
                                         m_eta, K, A_m_eta))
        else:
            eta_z0_list.append(m_eta)
            eta_z100_list.append(m_eta)

    log.info('η̂ computed in %.2fs', time.perf_counter() - t0)

    # ── Batched ACS precompute ────────────────────────────────────────────────
    t0 = time.perf_counter()
    def _compute_acs(eta_list, acs_obj, seed):
        if args.quad == 'gh':
            return batch_acs_gh(eta_list, acs_obj, args.gh_order,
                                 chunk_shots=getattr(args, 'gh_chunk_shots', None))
        else:
            return batch_acs(eta_list, acs_obj, args.n_qmc_int, rng_seed=seed)

    if args.use_moments or args.use_true_eta:
        acs_z0_arr   = _compute_acs(eta_z0_list,  acs_z0,   1)
        acs_z100_arr = _compute_acs(eta_z100_list, acs_z100, 2)
    else:
        # All shots identical → compute once (QMC: more samples; GH: same, deterministic)
        n_all = args.n_qmc_int * len(shot_ids) if args.quad == 'qmc' else args.n_qmc_int
        acs_z0_arr   = np.repeat(_compute_acs([m_eta], acs_z0,   1), len(shot_ids), axis=0)
        acs_z100_arr = np.repeat(_compute_acs([m_eta], acs_z100, 2), len(shot_ids), axis=0)
    log.info('ACS precompute done in %.1fs', time.perf_counter() - t0)

    # ── Assemble precomp_list ─────────────────────────────────────────────────
    precomp_list = []
    diag_rows    = []
    LOG_W_SINGLE = np.array([0.0])   # single plug-in point, weight = 1

    for k, shot_id in enumerate(shot_ids):
        n_g0, n_e0 = counts_z0[k]
        n_g1, n_e1 = counts_z100[k]

        sd_z0   = {'acs': acs_z0_arr[k:k+1],   'log_w': LOG_W_SINGLE,
                   'n_g_tot': n_g0, 'n_e_tot': n_e0}
        sd_z100 = {'acs': acs_z100_arr[k:k+1],  'log_w': LOG_W_SINGLE,
                   'n_g_tot': n_g1, 'n_e_tot': n_e1}
        precomp_list.append((sd_z0, sd_z100))

        meta0 = ds_z0.meta(shot_id)
        meta1 = ds_z100.meta(shot_id)
        diag_rows.append({
            'shot': shot_id,
            'eta_hat_z0':   eta_z0_list[k].copy(),
            'eta_hat_z100': eta_z100_list[k].copy(),
            'theta_true_z0':   np.array([meta0['mu_x0'], meta0['mu_y0'], meta0['mu_vx0'],
                                         meta0['mu_vy0'], meta0['sigma_x'], meta0['sigma_y'],
                                         meta0['sigma_vx'], meta0['sigma_vy']]),
            'theta_true_z100': np.array([meta1['mu_x0'], meta1['mu_y0'], meta1['mu_vx0'],
                                         meta1['mu_vy0'], meta1['sigma_x'], meta1['sigma_y'],
                                         meta1['sigma_vx'], meta1['sigma_vy']]),
            'delta_phi_true': float(meta1['delta_phi']),
        })

    phi_method = getattr(args, 'phi_method', 'grid')
    if phi_method == 'grid':
        def neg_logL(beta):
            return -total_logL(beta, precomp_list, f_signal, shot_idx_arr, args.n_theta)
    else:
        n_g0_arr = np.array([c[0] for c in counts_z0]);   n_e0_arr = np.array([c[1] for c in counts_z0])
        n_g1_arr = np.array([c[0] for c in counts_z100]); n_e1_arr = np.array([c[1] for c in counts_z100])
        n_scan   = getattr(args, 'phi_n_scan', 64)
        n_newton = getattr(args, 'phi_n_newton', 12)
        n_multistart = getattr(args, 'phi_n_multistart', 1)

        def neg_logL(beta):
            return -total_logL_fast(beta, acs_z0_arr, n_g0_arr, n_e0_arr,
                                     acs_z100_arr, n_g1_arr, n_e1_arr,
                                     f_signal, shot_idx_arr, phi_method=phi_method,
                                     n_scan=n_scan, n_newton=n_newton,
                                     n_multistart=n_multistart)

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


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bins',        type=int,   default=DEFAULT_BINS)
    p.add_argument('--n_qmc_int',   type=int,   default=DEFAULT_N_QMC_INT,
                   help='Cloud QMC samples per shot per AI for ACS (qmc mode only)')
    p.add_argument('--quad',        type=str,   default='qmc',
                   choices=['qmc', 'gh'],
                   help='Cloud integration method: qmc (default) or gh (Gauss-Hermite)')
    p.add_argument('--gh_order',    type=int,   default=3,
                   help='GH points per dimension (gh mode only); n^4 total, default 3 → 81 pts')
    p.add_argument('--gh_chunk_shots', type=int, default=None,
                   help='process at most this many shots per batch_acs_gh GPU call '
                        '(gh mode only; None = all shots at once, the original behaviour). '
                        'Needed at high gh_order (e.g. 12 -> 20,736 pts/shot) to avoid GPU OOM.')
    p.add_argument('--n_theta',     type=int,   default=DEFAULT_N_THETA)
    p.add_argument('--phi_method',  type=str,   default='laplace',
                   choices=['grid', 'laplace', 'point'],
                   help="per-shot nuisance-phase handling: 'laplace' (default) optimizes phi "
                        "continuously per shot with a curvature correction approximating the "
                        "marginal integral; 'point' is the same optimum without the correction; "
                        "'grid' marginalizes on a fixed n_theta grid and is ONLY safe when "
                        "n_theta is fine enough to resolve the per-shot phi posterior width -- "
                        "at high photon flux (e.g. A~1e8) the default n_theta=128 grid is far "
                        "too coarse and silently biases beta_hat (~12%% bias observed on As at "
                        "A=1e8 with n_theta=128; converges away only above n_theta~2000-8000)")
    p.add_argument('--phi_n_scan',   type=int, default=64,
                   help='coarse scan points to seed the phi optimum (laplace/point only)')
    p.add_argument('--phi_n_newton', type=int, default=12,
                   help='Newton polish iterations for the phi optimum (laplace/point only)')
    p.add_argument('--phi_n_multistart', type=int, default=1,
                   help='genuine multistart for the phi solve: Newton-polish the top-N '
                        'local maxima from the coarse scan and keep the best per shot, '
                        'instead of polishing only the single best scan point '
                        '(laplace/point only; 1 = original single-start behaviour)')
    p.add_argument('--t_det',       type=float, default=DEFAULT_T_DET)
    p.add_argument('--max_shots',   type=int,   default=DEFAULT_MAX_SHOTS)
    p.add_argument('--grid_n',      type=int,   default=DEFAULT_GRID_N)
    p.add_argument('--grid_half',   type=float, default=DEFAULT_GRID_HALF)
    p.add_argument('--n_starts',    type=int,   default=DEFAULT_N_STARTS)
    p.add_argument('--use_moments',  type=int, default=1,
                   help='1 = position-resolved MAP (default); 0 = prior-mean baseline')
    p.add_argument('--use_true_eta', type=int, default=0,
                   help='1 = oracle mode: plug in true η from metadata (accuracy limit)')
    p.add_argument('--data_root',   type=str,   default='',
                   help='Dataset root containing run_NNN dirs (sweep mode)')
    p.add_argument('--run_start',   type=int,   default=0,
                   help='First run index to process (inclusive, default 0)')
    p.add_argument('--run_end',     type=int,   default=-1,
                   help='Last run index to process (inclusive, default -1 = all)')
    p.add_argument('--data_dir',    type=str,
                   default=str(REPO / 'data' / DEFAULT_DATASET / 'run_000'),
                   help='Single run dir (ignored when --data_root is given)')
    p.add_argument('--out_dir',     type=str,   default='',
                   help='Output dir for sweep (default: results/mle_distributions/)')
    p.add_argument('--out',         type=str,   default='',
                   help='Output pkl path (single-run mode only)')
    args = p.parse_args()
    args.use_moments  = bool(args.use_moments)
    args.use_true_eta = bool(args.use_true_eta)
    if args.use_true_eta and args.use_moments:
        raise ValueError('--use_true_eta and --use_moments are mutually exclusive')

    if args.use_true_eta:
        tag = 'map_true_eta'
    elif args.use_moments:
        tag = 'map_moment'
    else:
        tag = 'map_prior'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)s  %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger(tag)

    quad_info = (f'gh_order={args.gh_order} ({args.gh_order**4} pts)'
                 if args.quad == 'gh' else f'n_qmc_int={args.n_qmc_int}')
    log.info('Mode: %s  quad=%s  %s  n_theta=%d  max_shots=%s',
             tag, args.quad, quad_info, args.n_theta, args.max_shots)

    # ── Resolve run directories ───────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root)
        run_dirs  = sorted(data_root.glob('run_*'))
        if args.run_end >= 0:
            run_dirs = run_dirs[args.run_start:args.run_end + 1]
        else:
            run_dirs = run_dirs[args.run_start:]
        out_dir   = Path(args.out_dir or str(REPO / 'results' / 'mle_distributions'))
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info('Sweep: %d runs in %s  ->  %s', len(run_dirs), data_root, out_dir)
    else:
        run_dirs = [Path(args.data_dir)]
        out_dir  = None

    # ── Load PSMAPs once ──────────────────────────────────────────────────────
    log.info('Loading PSMAPs...')
    t0 = time.perf_counter()
    psmap_z0   = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_FINE_Z0.h5'))
    psmap_z100 = load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_FINE_Z100.h5'))
    _ds_tmp = ImageShotDataset(str(run_dirs[0] / 'Z0' / 'data_IMG.h5'))
    edges = np.linspace(-_ds_tmp.half_range, _ds_tmp.half_range, args.bins + 1)
    del _ds_tmp
    sur_z0   = PSMAPSurrogate(psmap_z0,   args.t_det, use_gpu=USE_GPU)
    sur_z100 = PSMAPSurrogate(psmap_z100, args.t_det, use_gpu=USE_GPU)
    acs_z0   = SurrogatePixelACS(sur_z0,   args.t_det, edges, edges, n_quad=1)
    acs_z100 = SurrogatePixelACS(sur_z100, args.t_det, edges, edges, n_quad=1)
    log.info('Evaluators ready in %.1fs', time.perf_counter() - t0)

    # ── Build prior and Kalman gain (once) ────────────────────────────────────
    A       = build_A(args.t_det)
    m_eta, P_eta = build_prior(args.t_det)
    K, A_m_eta   = precompute_kalman(m_eta, P_eta, A)
    log.info('Prior m_η: V_x0=%.3g  V_vx0=%.3g  C_xv0=%.3g',
             m_eta[4], m_eta[6], m_eta[5])

    # ── Loop over runs ────────────────────────────────────────────────────────
    for run_dir in run_dirs:
        log.info('===== %s =====', run_dir.name)
        payload = _run_one(run_dir, acs_z0, acs_z100, m_eta, K, A_m_eta, args, log)

        if out_dir is not None:
            out_path = out_dir / f'{tag}_{run_dir.name}.pkl'
        else:
            out_path = Path(args.out or str(REPO / 'results' / f'{tag}.pkl'))

        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        log.info('Saved -> %s', out_path)


if __name__ == '__main__':
    main()
