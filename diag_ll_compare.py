#!/usr/bin/env python3
"""
diag_ll_compare.py — Side-by-side logL scan comparison: explorer vs custom.

Runs 5 diagnostic tests in sequence:

  TEST 1: ACS consistency — SemiAnalyticPixelACS.pixel_acs vs rates_ACS at 32 bins
  TEST 2: Normalization check — is sum(n_atoms * p_pix * (A_g+A_e)) = observed total?
  TEST 3: LogL scan at 32 bins — both methods, 100M dataset, Z0 only
  TEST 4: LogL scan at 32 bins — custom with self-normalisation (L0 trick)
  TEST 5: Full custom scan (256 bins) vs explorer (32 bins) head-to-head

Each test prints a summary. Pass/fail is printed at the end.
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/local/aispy'))
sys.path.insert(0, os.path.expanduser('~/aispp-sims/gaussian-wavefront-spatially-resolved-inference/helpers'))
sys.path.insert(0, os.path.expanduser('~/aispp-sims/gaussian-wavefront-spatially-resolved-inference/python-scripts'))

import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr

from helpers import ImageShotDataset
from aispy.psmap import load_psmap, PSMAPSurrogate
from profile_cloud_nuisances import SurrogatePixelACS, SemiAnalyticPixelACS

REPO = os.path.expanduser('~/aispp-sims/gaussian-wavefront-spatially-resolved-inference')

# ── Dataset: 100M atoms ───────────────────────────────────────────────────────
DATA100M = os.path.join(REPO, 'data',
    'R20_N200_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
    'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000/run_000')
Z0   = ImageShotDataset(os.path.join(DATA100M, 'Z0',   'data_IMG.h5'))
Z100 = ImageShotDataset(os.path.join(DATA100M, 'Z100', 'data_IMG.h5'))

# ── Dataset: 1M atoms (explorer) ─────────────────────────────────────────────
DATA1M = os.path.join(REPO, 'data',
    'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
    'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000/run_000')
Z0_1M = ImageShotDataset(os.path.join(DATA1M, 'Z0',   'data_IMG.h5'))

# ── PSMAPs ────────────────────────────────────────────────────────────────────
print('Loading PSMAPs...', flush=True)
psmap_z0 = load_psmap(os.path.join(REPO, 'output-files', 'PSGRID4D_CONFOCAL_FINE_Z0.h5'))
sur_z0   = PSMAPSurrogate(psmap_z0, t_det=3.8, use_gpu=True)
print('done')

# ── True params (shot 0) ──────────────────────────────────────────────────────
IDX = 0
# Custom ordering: [mu_x0, mu_vx0, mu_y0, mu_vy0, sigma_x, sigma_vx, sigma_y, sigma_vy]
theta_cust = np.array([Z0.mu_x0[IDX], Z0.mu_vx0[IDX], Z0.mu_y0[IDX], Z0.mu_vy0[IDX],
                       Z0.sigma_x[IDX], Z0.sigma_vx[IDX], Z0.sigma_y[IDX], Z0.sigma_vy[IDX]])
# Explorer/SemiAnalyticPixelACS ordering: [mu_x0, mu_y0, mu_vx0, mu_vy0, sx, sy, svx, svy]
theta_expl = np.array([Z0.mu_x0[IDX], Z0.mu_y0[IDX], Z0.mu_vx0[IDX], Z0.mu_vy0[IDX],
                       Z0.sigma_x[IDX], Z0.sigma_y[IDX], Z0.sigma_vx[IDX], Z0.sigma_vy[IDX]])

print(f'True mu_x0 = {Z0.mu_x0[IDX]*1e6:.2f} µm')

T_DET = 3.8
GH_N  = 20
N_PHI = 128
BIN32 = 32
BIN256 = 8   # factor (2048/8 = 256 bins)

# ── Shared GH setup ───────────────────────────────────────────────────────────
gh_nodes, gh_weights = hermgauss(GH_N)

# ── Helper: rebin counts ──────────────────────────────────────────────────────
def rebin(img, factor):
    n = img.shape[0] // factor
    return img.reshape(n, factor, n, factor).sum(axis=(1, 3))


# ── Custom implementation functions ──────────────────────────────────────────

def v_given_x_params(x, y, t, p):
    """p = [mux0, muvx0, muy0, muvy0, sx0, svx0, sy0, svy0]"""
    mux0, muvx0, muy0, muvy0, sx0, svx0, sy0, svy0 = p
    gain_x = t * svx0**2 / (sx0**2 + t**2 * svx0**2)
    gain_y = t * svy0**2 / (sy0**2 + t**2 * svy0**2)
    mu_vx = muvx0 + gain_x * (x - mux0 - t * muvx0)
    mu_vy = muvy0 + gain_y * (y - muy0 - t * muvy0)
    var_vx = sx0**2 * svx0**2 / (sx0**2 + t**2 * svx0**2)
    var_vy = sy0**2 * svy0**2 / (sy0**2 + t**2 * svy0**2)
    return mu_vx, mu_vy, var_vx, var_vy


def pixel_prob_pdf(x, y, t, p):
    """Gaussian PDF at detection plane."""
    mux0, muvx0, muy0, muvy0, sx0, svx0, sy0, svy0 = p
    mu_xf = mux0 + muvx0 * t;  mu_yf = muy0 + muvy0 * t
    var_xf = sx0**2 + (svx0*t)**2;  var_yf = sy0**2 + (svy0*t)**2
    px = np.exp(-0.5*(x-mu_xf)**2/var_xf) / np.sqrt(2*np.pi*var_xf)
    py = np.exp(-0.5*(y-mu_yf)**2/var_yf) / np.sqrt(2*np.pi*var_yf)
    return px * py


def pixel_prob_cdf(x_lo, x_hi, y_lo, y_hi, t, p):
    """Exact Gaussian CDF integral over pixel."""
    mux0, muvx0, muy0, muvy0, sx0, svx0, sy0, svy0 = p
    mu_xf = mux0 + muvx0 * t;  mu_yf = muy0 + muvy0 * t
    sxf = np.sqrt(sx0**2 + (svx0*t)**2)
    syf = np.sqrt(sy0**2 + (svy0*t)**2)
    Px = ndtr((x_hi - mu_xf)/sxf) - ndtr((x_lo - mu_xf)/sxf)
    Py = ndtr((y_hi - mu_yf)/syf) - ndtr((y_lo - mu_yf)/syf)
    return Px * Py


def rates_ACS(x, y, t_det, params, surrogate):
    """GH-integrate ACS over p(vx,vy|xf,yf) for all ports.
    Returns (A_g, C_g, S_g), (A_e, C_e, S_e) where each is (N_pix,).
    params order: [mux0, muvx0, muy0, muvy0, sx0, svx0, sy0, svy0]
    """
    mu_vx, mu_vy, var_vx, var_vy = v_given_x_params(x, y, t_det, params)
    N = len(x)
    vx_nodes = mu_vx[:, None] + np.sqrt(2*var_vx)*gh_nodes
    vy_nodes = mu_vy[:, None] + np.sqrt(2*var_vy)*gh_nodes
    vx_g = np.broadcast_to(vx_nodes[:,:,None], (N, GH_N, GH_N)).copy()
    vy_g = np.broadcast_to(vy_nodes[:,None,:], (N, GH_N, GH_N)).copy()
    x0 = x[:,None,None] - vx_g * t_det
    y0 = y[:,None,None] - vy_g * t_det

    dphi_all, a0_all, a1_all = surrogate.eval(
        x0.reshape(-1), y0.reshape(-1), vx_g.reshape(-1), vy_g.reshape(-1))

    weights = np.outer(gh_weights, gh_weights) / np.pi
    A_g = np.zeros(N); C_g = np.zeros(N); S_g = np.zeros(N)
    A_e = np.zeros(N); C_e = np.zeros(N); S_e = np.zeros(N)
    for pi in range(surrogate.nP):
        dphi = dphi_all[:, pi].reshape(N, GH_N, GH_N)
        a0   =   a0_all[:, pi].reshape(N, GH_N, GH_N)
        a1   =   a1_all[:, pi].reshape(N, GH_N, GH_N)
        inter = float(surrogate.port_interfering[pi])
        A_  = np.einsum('ij,pij->p', weights, a0**2 + a1**2)
        C_  = np.einsum('ij,pij->p', weights, inter * 2*a0*a1*np.cos(dphi))
        S_  = np.einsum('ij,pij->p', weights, inter * (-2*a0*a1*np.sin(dphi)))
        if surrogate.port_states[pi] == 0:
            A_g += A_; C_g += C_; S_g += S_
        else:
            A_e += A_; C_e += C_; S_e += S_
    return (A_g, C_g, S_g), (A_e, C_e, S_e)


def ll_image_custom_pdf(counts_g, counts_e, x, y, t_det, params,
                        pixel_area, n_atoms, surrogate):
    """Custom logL with PDF×area spatial weight."""
    (A_g, C_g, S_g), (A_e, C_e, S_e) = rates_ACS(x, y, t_det, params, surrogate)
    p_pix = pixel_prob_pdf(x, y, t_det, params) * pixel_area
    phi = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)
    lam_g = n_atoms * p_pix[:,None] * np.maximum(
        A_g[:,None] + C_g[:,None]*np.cos(phi) + S_g[:,None]*np.sin(phi), 1e-300)
    lam_e = n_atoms * p_pix[:,None] * np.maximum(
        A_e[:,None] + C_e[:,None]*np.cos(phi) + S_e[:,None]*np.sin(phi), 1e-300)
    ll = (counts_g[:,None]*np.log(lam_g) - lam_g +
          counts_e[:,None]*np.log(lam_e) - lam_e).sum(0)
    return logsumexp(ll) - np.log(N_PHI)


def ll_image_custom_cdf(counts_g, counts_e, x, y, x_edges, y_edges,
                        t_det, params, n_atoms, surrogate):
    """Custom logL with exact CDF spatial weight (should match explorer)."""
    (A_g, C_g, S_g), (A_e, C_e, S_e) = rates_ACS(x, y, t_det, params, surrogate)
    # pixel bounds (1D grid assumed symmetric, so x_edges apply to both)
    ix_all = np.arange(len(x)) // len(np.unique(y))  # assumes meshgrid ij ordering
    iy_all = np.arange(len(x)) % len(np.unique(y))
    # Recompute properly via CDF
    mux0, muvx0, muy0, muvy0, sx0, svx0, sy0, svy0 = params
    mu_xf = mux0 + muvx0 * t_det;  mu_yf = muy0 + muvy0 * t_det
    sxf = np.sqrt(sx0**2 + (svx0*t_det)**2)
    syf = np.sqrt(sy0**2 + (svy0*t_det)**2)
    Px = ndtr((x_edges[1:]-mu_xf)/sxf) - ndtr((x_edges[:-1]-mu_xf)/sxf)  # (BINS,)
    Py = ndtr((y_edges[1:]-mu_yf)/syf) - ndtr((y_edges[:-1]-mu_yf)/syf)
    Xc_idx = np.arange(len(Px))
    Yc_idx = np.arange(len(Py))
    Xidx, Yidx = np.meshgrid(Xc_idx, Yc_idx, indexing='ij')
    P_b = (Px[Xidx.ravel()] * Py[Yidx.ravel()])  # (N_pix,) — matches meshgrid ij

    phi = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)
    lam_g = n_atoms * P_b[:,None] * np.maximum(
        A_g[:,None] + C_g[:,None]*np.cos(phi) + S_g[:,None]*np.sin(phi), 1e-300)
    lam_e = n_atoms * P_b[:,None] * np.maximum(
        A_e[:,None] + C_e[:,None]*np.cos(phi) + S_e[:,None]*np.sin(phi), 1e-300)
    ll = (counts_g[:,None]*np.log(lam_g) - lam_g +
          counts_e[:,None]*np.log(lam_e) - lam_e).sum(0)
    return logsumexp(ll) - np.log(N_PHI)


def ll_image_selfnorm(counts_g, counts_e, x, y, t_det, params,
                      pixel_area, surrogate, cdf_P_b=None):
    """Same as explorer: self-normalise L0 = total_counts / sum_ACS."""
    (A_g, C_g, S_g), (A_e, C_e, S_e) = rates_ACS(x, y, t_det, params, surrogate)
    if cdf_P_b is not None:
        p_pix = cdf_P_b
    else:
        p_pix = pixel_prob_pdf(x, y, t_det, params) * pixel_area
    A_g_w = p_pix * A_g;  A_e_w = p_pix * A_e
    C_g_w = p_pix * C_g;  C_e_w = p_pix * C_e
    S_g_w = p_pix * S_g;  S_e_w = p_pix * S_e
    total_pred = float((A_g_w + A_e_w).sum())
    total_obs  = float(counts_g.sum() + counts_e.sum())
    L0 = total_obs / max(total_pred, 1e-300)
    phi = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)
    lam_g = L0 * np.maximum(A_g_w[:,None] + C_g_w[:,None]*np.cos(phi) + S_g_w[:,None]*np.sin(phi), 1e-300)
    lam_e = L0 * np.maximum(A_e_w[:,None] + C_e_w[:,None]*np.cos(phi) + S_e_w[:,None]*np.sin(phi), 1e-300)
    ll = (counts_g[:,None]*np.log(lam_g) - lam_g +
          counts_e[:,None]*np.log(lam_e) - lam_e).sum(0)
    return logsumexp(ll) - np.log(N_PHI)


# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('TEST 1: ACS consistency — SemiAnalyticPixelACS vs rates_ACS (32 bins)')
print('='*70)

BINS = BIN32
b = Z0.res // BINS
edges32 = np.linspace(-Z0.half_range, Z0.half_range, BINS+1)
centers32 = 0.5*(edges32[:-1]+edges32[1:])
Xc32, Yc32 = np.meshgrid(centers32, centers32, indexing='ij')
x_flat = Xc32.ravel(); y_flat = Yc32.ravel()
pixel_area32 = (centers32[1]-centers32[0])**2

# Build SemiAnalyticPixelACS evaluator
_base = SurrogatePixelACS(sur_z0, T_DET, edges32, edges32, n_quad=1)
eval32 = SemiAnalyticPixelACS(_base, n_gh=GH_N, chunk_bins=64)

# Explorer output (P_b * ACS)
print('  Computing SemiAnalyticPixelACS.pixel_acs...', end=' ', flush=True)
A_g_e, Cc_g_e, Cs_g_e, A_e_e, Cc_e_e, Cs_e_e = [v.get() for v in eval32.pixel_acs(theta_expl)]
print('done')

# Custom output (raw ACS, no spatial weight)
print('  Computing rates_ACS (custom)...', end=' ', flush=True)
(A_g_c, C_g_c, S_g_c), (A_e_c, C_e_c, S_e_c) = rates_ACS(x_flat, y_flat, T_DET, theta_cust, sur_z0)
print('done')

# Apply PDF×area spatial weight to custom output for comparison
p_pix32 = pixel_prob_pdf(x_flat, y_flat, T_DET, theta_cust) * pixel_area32
A_g_cw = p_pix32 * A_g_c;  A_e_cw = p_pix32 * A_e_c
Cc_g_cw = p_pix32 * C_g_c;  Cc_e_cw = p_pix32 * C_e_c
Cs_g_cw = p_pix32 * S_g_c;  Cs_e_cw = p_pix32 * S_e_c

# Apply exact CDF weight to custom output
mux0, muvx0, muy0, muvy0, sx0, svx0, sy0, svy0 = theta_cust
mu_xf = mux0 + muvx0*T_DET;  mu_yf = muy0 + muvy0*T_DET
sxf = np.sqrt(sx0**2 + (svx0*T_DET)**2)
syf = np.sqrt(sy0**2 + (svy0*T_DET)**2)
Px32 = ndtr((edges32[1:]-mu_xf)/sxf) - ndtr((edges32[:-1]-mu_xf)/sxf)
Py32 = ndtr((edges32[1:]-mu_yf)/syf) - ndtr((edges32[:-1]-mu_yf)/syf)
Xi, Yi = np.meshgrid(np.arange(BINS), np.arange(BINS), indexing='ij')
P_b32 = (Px32[Xi.ravel()] * Py32[Yi.ravel()])

A_g_ccdf  = P_b32 * A_g_c;  A_e_ccdf  = P_b32 * A_e_c
Cc_g_ccdf = P_b32 * C_g_c;  Cc_e_ccdf = P_b32 * C_e_c
Cs_g_ccdf = P_b32 * S_g_c;  Cs_e_ccdf = P_b32 * S_e_c

def rel_rms(a, b):
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-300) / 2
    return np.sqrt(np.mean(((a-b)/denom)**2))

print(f'  Explorer sum(A_g+A_e)           = {(A_g_e+A_e_e).sum():.6f}')
print(f'  Custom (CDF-weighted) sum(A_g+A_e) = {(A_g_ccdf+A_e_ccdf).sum():.6f}')
print(f'  Custom (PDF-weighted) sum(A_g+A_e) = {(A_g_cw+A_e_cw).sum():.6f}')
print(f'  Relative RMS(A_g): expl vs cust_CDF = {rel_rms(A_g_e, A_g_ccdf):.4f}')
print(f'  Relative RMS(A_g): expl vs cust_PDF = {rel_rms(A_g_e, A_g_cw):.4f}')
print(f'  Relative RMS(Cc_g): expl vs cust_CDF = {rel_rms(Cc_g_e, Cc_g_ccdf):.4f}')
print(f'  Relative RMS(Cc_g): expl vs cust_PDF = {rel_rms(Cc_g_e, Cc_g_cw):.4f}')

# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('TEST 2: Normalization check (100M dataset, 32 bins)')
print('='*70)

img100m = Z0[IDX]
ng_32 = rebin(img100m[0].astype(float), b)
ne_32 = rebin(img100m[1].astype(float), b)
observed_total = float(ng_32.sum() + ne_32.sum())
n_atoms = float(Z0.n_atoms_launched)

pred_pdf  = n_atoms * float((A_g_cw + A_e_cw).sum())
pred_cdf  = n_atoms * float((A_g_ccdf + A_e_ccdf).sum())
pred_expl_self = float(observed_total)  # explorer self-normalizes, so pred = obs by construction

print(f'  Observed total counts        = {observed_total:.0f}')
print(f'  n_atoms_launched             = {n_atoms:.0f}')
print(f'  sum(A_g+A_e) per atom (CDF)  = {(A_g_ccdf+A_e_ccdf).sum():.6f}  → pred = {pred_cdf:.0f}')
print(f'  sum(A_g+A_e) per atom (PDF)  = {(A_g_cw+A_e_cw).sum():.6f}  → pred = {pred_pdf:.0f}')
print(f'  Ratio pred_CDF/observed = {pred_cdf/observed_total:.4f}')
print(f'  Ratio pred_PDF/observed = {pred_pdf/observed_total:.4f}')

# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('TEST 3: LogL scan at 32 bins — explorer vs custom, 100M dataset, Z0 only')
print('='*70)

MUX0_SCAN = np.linspace(-30e-6, 30e-6, 21)
true_mux0 = Z0.mu_x0[IDX]

ng_32f = ng_32.ravel(); ne_32f = ne_32.ravel()

ll_expl_100m = []
ll_cust_pdf  = []
ll_cust_selfnorm = []

print('  Building SemiAnalyticPixelACS for 100M dataset scan...')

# Precompute ACS for explorer scan
def expl_logL(mux0_val):
    th = theta_expl.copy(); th[0] = mux0_val
    acs = [v.get() for v in eval32.pixel_acs(th)]
    A_g0, Cc_g0, Cs_g0, A_e0, Cc_e0, Cs_e0 = acs
    ng_gpu = ng_32f; ne_gpu = ne_32f
    L0 = observed_total / max(float((A_g0+A_e0).sum()), 1e-300)
    phi = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    lam_g = L0 * np.maximum(A_g0[:,None]+Cc_g0[:,None]*c+Cs_g0[:,None]*s, 1e-300)
    lam_e = L0 * np.maximum(A_e0[:,None]+Cc_e0[:,None]*c+Cs_e0[:,None]*s, 1e-300)
    ll = (ng_gpu[:,None]*np.log(lam_g)-lam_g + ne_gpu[:,None]*np.log(lam_e)-lam_e).sum(0)
    return float(logsumexp(ll) - np.log(N_PHI))

print('  Scanning...', flush=True)
for i, v in enumerate(MUX0_SCAN):
    th_c = theta_cust.copy(); th_c[0] = v
    lc_pdf = ll_image_custom_pdf(ng_32f, ne_32f, x_flat, y_flat, T_DET,
                                  th_c, pixel_area32, n_atoms, sur_z0)
    lc_sn  = ll_image_selfnorm(ng_32f, ne_32f, x_flat, y_flat, T_DET,
                                th_c, pixel_area32, sur_z0)
    le     = expl_logL(v)
    ll_cust_pdf.append(lc_pdf)
    ll_cust_selfnorm.append(lc_sn)
    ll_expl_100m.append(le)
    if i % 5 == 0:
        print(f'    {i+1}/{len(MUX0_SCAN)}  mux0={v*1e6:.1f}µm  expl={le:.1f}  cust_pdf={lc_pdf:.1f}  cust_sn={lc_sn:.1f}')

ll_e = np.array(ll_expl_100m); ll_e -= ll_e.max()
ll_p = np.array(ll_cust_pdf);  ll_p -= ll_p.max()
ll_s = np.array(ll_cust_selfnorm); ll_s -= ll_s.max()

def fwhm_idx(arr, xs):
    """Half-width at -0.5 (approximate FWHM in same units as xs)."""
    above = arr > -0.5
    if not above.any(): return float('nan')
    idxs = np.where(above)[0]
    return (xs[idxs[-1]] - xs[idxs[0]]) * 1e6  # µm

print(f'\n  Peak positions [µm] (true = {true_mux0*1e6:.2f}µm):')
print(f'    Explorer (self-norm, 100M):  {MUX0_SCAN[np.argmax(ll_e)]*1e6:.1f}µm  FWHM≈{fwhm_idx(ll_e, MUX0_SCAN):.1f}µm')
print(f'    Custom (PDF, abs n_atoms):   {MUX0_SCAN[np.argmax(ll_p)]*1e6:.1f}µm  FWHM≈{fwhm_idx(ll_p, MUX0_SCAN):.1f}µm')
print(f'    Custom (PDF, self-norm):     {MUX0_SCAN[np.argmax(ll_s)]*1e6:.1f}µm  FWHM≈{fwhm_idx(ll_s, MUX0_SCAN):.1f}µm')

# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('TEST 4: Does custom (256 bins) give sharper or broader than explorer (32 bins)?')
print('='*70)

# 256-bin custom
BINS256 = Z0.res // BIN256   # = 256
edges256 = Z0.edges[::BIN256]
centers256 = 0.5*(edges256[:-1]+edges256[1:])
Xc256, Yc256 = np.meshgrid(centers256, centers256, indexing='ij')
xf256 = Xc256.ravel(); yf256 = Yc256.ravel()
pixel_area256 = (centers256[1]-centers256[0])**2
ng_256 = rebin(img100m[0].astype(float), BIN256).ravel()
ne_256 = rebin(img100m[1].astype(float), BIN256).ravel()
obs256 = float(ng_256.sum() + ne_256.sum())
print(f'  256-bin total counts: {obs256:.0f}')
print(f'  256-bin pixel size: {(centers256[1]-centers256[0])*1e6:.1f}µm')

ll_cust256_sn  = []
print('  Scanning 256-bin custom (self-norm)...')
for i, v in enumerate(MUX0_SCAN):
    th_c = theta_cust.copy(); th_c[0] = v
    lc = ll_image_selfnorm(ng_256, ne_256, xf256, yf256, T_DET,
                           th_c, pixel_area256, sur_z0)
    ll_cust256_sn.append(lc)
    if i % 5 == 0:
        print(f'    {i+1}/{len(MUX0_SCAN)}  mux0={v*1e6:.1f}µm  cust256_sn={lc:.1f}')

ll_256s = np.array(ll_cust256_sn); ll_256s -= ll_256s.max()

print(f'\n  Summary (all using self-norm, Z0 only, 100M atoms):')
print(f'    Explorer  32 bins: peak={MUX0_SCAN[np.argmax(ll_e)]*1e6:.1f}µm  FWHM≈{fwhm_idx(ll_e, MUX0_SCAN):.1f}µm')
print(f'    Custom   256 bins: peak={MUX0_SCAN[np.argmax(ll_256s)]*1e6:.1f}µm  FWHM≈{fwhm_idx(ll_256s, MUX0_SCAN):.1f}µm')
print(f'    True value: {true_mux0*1e6:.2f}µm')

# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'TEST 1 - ACS RMS(A_g): expl vs cust_CDF = {rel_rms(A_g_e, A_g_ccdf):.4f}  '
      f'(expect < 0.01 if GH is well converged)')
print(f'TEST 2 - pred/obs ratio (CDF): {pred_cdf/observed_total:.4f}  '
      f'(expect ≈ 1.0 for correct normalization)')
print(f'TEST 3 - explorer vs custom_selfnorm FWHM: '
      f'{fwhm_idx(ll_e, MUX0_SCAN):.1f} vs {fwhm_idx(ll_s, MUX0_SCAN):.1f} µm')
print(f'TEST 4 - custom256_selfnorm FWHM: {fwhm_idx(ll_256s, MUX0_SCAN):.1f} µm '
      f'vs explorer {fwhm_idx(ll_e, MUX0_SCAN):.1f} µm')

import json
results = {
    't1_rms_Ag_cdf':  float(rel_rms(A_g_e, A_g_ccdf)),
    't2_pred_obs_ratio': float(pred_cdf/observed_total),
    't3_fwhm_expl': float(fwhm_idx(ll_e, MUX0_SCAN)),
    't3_fwhm_cust_pdf': float(fwhm_idx(ll_p, MUX0_SCAN)),
    't3_fwhm_cust_sn': float(fwhm_idx(ll_s, MUX0_SCAN)),
    't4_fwhm_cust256_sn': float(fwhm_idx(ll_256s, MUX0_SCAN)),
    'true_mux0_um': float(true_mux0*1e6),
    'peak_expl_um': float(MUX0_SCAN[np.argmax(ll_e)]*1e6),
    'peak_cust_pdf_um': float(MUX0_SCAN[np.argmax(ll_p)]*1e6),
    'peak_cust_sn_um': float(MUX0_SCAN[np.argmax(ll_s)]*1e6),
    'peak_cust256_sn_um': float(MUX0_SCAN[np.argmax(ll_256s)]*1e6),
}
print('\nJSON results:')
print(json.dumps(results, indent=2))
