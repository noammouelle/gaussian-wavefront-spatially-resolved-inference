#!/usr/bin/env python
"""
profile_cloud_nuisances.py — Per-shot GPU MLE of cloud nuisance parameters.

Fits 16 cloud parameters per shot (8 per AI, independently) by maximising the
phi-marginalised joint Poisson image log-likelihood:

    theta_hat = argmax_{theta} log L_i(theta)

    log L_i(theta) = logsumexp_k [ ll_Z0(theta_Z0, phi_k)
                                  + ll_Z100(theta_Z100, phi_k) ]
                   - log(K)

where
    theta = [theta_Z0 | theta_Z100]
    theta_ZX = [mu_x0, mu_y0, mu_vx0, mu_vy0,
                sigma_x0, sigma_y0, sigma_vx0, sigma_vy0]

Both AIs share the same phi_i per shot (common mode), so their log-likelihoods
are SUMMED before the logsumexp, not marginalised independently.  Each AI gets
independent cloud parameters (different atomic ensembles).

Implementation
--------------
All computation except the final scalar extraction runs on GPU:
  1. Gaussian phase-space weights: vectorised GPU ops (~3ms per theta eval).
  2. Six GPU bincounts → pixel-level (A, Cc, Cs) per state per AI.
  3. GPU logsumexp marginalises phi_i.

Spread parameters are log-transformed for unconstrained optimisation.

Output
------
JSONL (one JSON object per shot) at results/<stem>_cloud_mle.jsonl with fields:

    shot, logL_hat, logL_true, delta_logL,
    <param>_<ai>_hat, <param>_<ai>_true, <param>_<ai>_err
    (ai in {z0, z100}, param in {mu_x0, mu_y0, mu_vx0, mu_vy0,
                                  sigma_x0, sigma_y0, sigma_vx0, sigma_vy0})
    nfev, nit, success, message, elapsed_s, run_name, run_idx

Usage
-----
    python profile_cloud_nuisances.py [run_name] [options]

    python profile_cloud_nuisances.py --ntheta 256 --max-shots 5 --init-from-true
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "helpers"))

from helpers import ImageShotDataset  # noqa: E402
from psmap_fisher import _final_bin_indices  # noqa: E402

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cp_logsumexp
    _HAS_GPU = True
except ImportError:
    cp = None
    _HAS_GPU = False

try:
    from aispy.psmap import load_psmap, PSMAPSurrogate
except ImportError:
    sys.path.insert(0, str(REPO.parents[1] / "local" / "aispy"))
    from aispy.psmap import load_psmap, PSMAPSurrogate

# ── Constants ──────────────────────────────────────────────────────────────────
T_DET = 3.8  # detection time [s]

PARAM_NAMES = [
    "mu_x0", "mu_y0", "mu_vx0", "mu_vy0",
    "sigma_x0", "sigma_y0", "sigma_vx0", "sigma_vy0",
]
H5_KEYS = {
    "mu_x0": "mu_x0", "mu_y0": "mu_y0",
    "mu_vx0": "mu_vx0", "mu_vy0": "mu_vy0",
    "sigma_x0": "sigma_x", "sigma_y0": "sigma_y",
    "sigma_vx0": "sigma_vx", "sigma_vy0": "sigma_vy",
}

DEFAULT_RUN = (
    REPO / "data"
    / "R80_N50_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx309um_"
      "sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"
)
DEFAULT_PSMAP_Z0   = REPO / "output-files" / "PSGRID4D_CONFOCAL_Z0.h5"
DEFAULT_PSMAP_Z100 = REPO / "output-files" / "PSGRID4D_CONFOCAL_Z100.h5"


# ── GPU evaluator for one AI ───────────────────────────────────────────────────

class SurrogatePixelACS:
    """
    GPU evaluator using PSMAPSurrogate quadrilinear interpolation.

    Matches the data-generation model exactly: QMC points are drawn from the
    cloud Gaussian and port probabilities are obtained by interpolating the PSMAP
    with the same quadrilinear scheme used by PSMAPSurrogate.generate_atoms().

    Construction pre-generates N_quad Sobol points as standard-normal deviates
    on GPU.  Each call to pixel_acs() transforms them to the cloud Gaussian,
    calls sur._eval_gpu() once (phi-independent), then computes pixel-level
    (A, Cc, Cs) via the 3-point formula p(phi) = A + Cc·cos(phi) + Cs·sin(phi).
    """

    def __init__(self, surrogate: PSMAPSurrogate, t_det: float,
                 x_edges, y_edges, n_quad: int = 20_000):
        from scipy.stats.qmc import Sobol
        from scipy.special import ndtri

        self.sur   = surrogate
        self.t_det = float(t_det)

        nx = len(x_edges) - 1; ny = len(y_edges) - 1
        self.n_bins = nx * ny
        self.nx, self.ny = nx, ny
        self.x_edges_g = cp.asarray(x_edges, dtype=cp.float64)
        self.y_edges_g = cp.asarray(y_edges, dtype=cp.float64)

        # QMC points: Sobol in [0,1]^4 → standard normal deviates, kept on GPU
        sobol = Sobol(d=4, scramble=True, seed=0)
        n_pow2 = 1 << int(np.ceil(np.log2(n_quad)))  # round up to power-of-2
        u = sobol.random(n_pow2)[:n_quad]
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        self.z_g = cp.asarray(ndtri(u), dtype=cp.float64)   # (n_quad, 4)

        # Port state masks
        self.s0_g = cp.asarray(surrogate.port_states == 0)  # (nP,) bool
        self.s1_g = ~self.s0_g

        # ── Pre-extract 4D PSMAP grids onto GPU for vectorised interpolation ──
        # Stack all ports into (nP, nx, ny, nvx, nvy) tensors so the Python
        # for-loop over ports is eliminated; a single gather handles all ports.
        nP = surrogate.nP
        self._dphi_stack = cp.stack(surrogate._gpu_dphi_resid)  # (nP, nx, ny, nvx, nvy)
        self._amp0_stack = cp.stack(surrogate._gpu_amp0)
        self._amp1_stack = cp.stack(surrogate._gpu_amp1)
        self._dphi_linear = cp.asarray(surrogate._dphi_linear, dtype=cp.float64)  # (nP,5)
        self._port_inter  = cp.asarray(surrogate.port_interfering, dtype=cp.float64)
        self.nP = nP

        # Precompute corner offsets for 4D quadrilinear interpolation:
        # 16 corners = {0,1}^4 ordered by bitmask
        bits = cp.arange(16, dtype=cp.int32)
        self._bx  = ((bits >> 0) & 1).astype(cp.int32)
        self._by  = ((bits >> 1) & 1).astype(cp.int32)
        self._bvx = ((bits >> 2) & 1).astype(cp.int32)
        self._bvy = ((bits >> 3) & 1).astype(cp.int32)

        # Grid params for _find_cell
        self._x_lo  = float(surrogate._x_lo);   self._dx  = float(surrogate._dx)
        self._y_lo  = float(surrogate._y_lo);   self._dy  = float(surrogate._dy)
        self._vx_lo = float(surrogate._vx_lo);  self._dvx = float(surrogate._dvx)
        self._vy_lo = float(surrogate._vy_lo);  self._dvy = float(surrogate._dvy)
        self._nx_m1 = surrogate.nx - 1
        self._ny_m1 = surrogate.ny - 1
        self._nvx_m1 = surrogate.nvx - 1
        self._nvy_m1 = surrogate.nvy - 1

    def _find_cell(self, arr, lo, dx, n_m1):
        """GPU _find_cell: returns (idx, frac) on device."""
        raw = (arr - lo) / dx
        idx = cp.clip(raw.astype(cp.int32), 0, n_m1 - 1)
        tx  = cp.clip(raw - idx.astype(cp.float64), 0.0, 1.0)
        return idx, tx

    def _eval_fast(self, x0_g, y0_g, vx0_g, vy0_g):
        """
        Fully vectorised quadrilinear interpolation over all ports at once.

        No Python loops over ports or corners.  A single (16, n_quad) gather
        per field replaces the original 16-iteration Python for-loop.

        Returns dphi, amp0, amp1 each of shape (n_quad, nP).
        """
        ix,  tx  = self._find_cell(x0_g,  self._x_lo,  self._dx,  self._nx_m1)
        iy,  ty  = self._find_cell(y0_g,  self._y_lo,  self._dy,  self._ny_m1)
        ivx, tvx = self._find_cell(vx0_g, self._vx_lo, self._dvx, self._nvx_m1)
        ivy, tvy = self._find_cell(vy0_g, self._vy_lo, self._dvy, self._nvy_m1)

        # Corner weights: (16, n_quad)
        bx  = self._bx;  by  = self._by
        bvx = self._bvx; bvy = self._bvy
        bx_b  = bx [:, None].astype(bool)
        by_b  = by [:, None].astype(bool)
        bvx_b = bvx[:, None].astype(bool)
        bvy_b = bvy[:, None].astype(bool)
        w = (cp.where(bx_b,  tx[None],  1.0-tx[None])
           * cp.where(by_b,  ty[None],  1.0-ty[None])
           * cp.where(bvx_b, tvx[None], 1.0-tvx[None])
           * cp.where(bvy_b, tvy[None], 1.0-tvy[None]))  # (16, n_quad)

        # Corner grid indices: (16, n_quad)
        ix_c  = ix [None] + bx [:, None]
        iy_c  = iy [None] + by [:, None]
        ivx_c = ivx[None] + bvx[:, None]
        ivy_c = ivy[None] + bvy[:, None]

        # Batch gather over all ports: stacked grids are (nP, nx, ny, nvx, nvy)
        # Expand to (nP, 16, n_quad) via broadcast indexing
        nP = self.nP
        pi_idx = cp.arange(nP, dtype=cp.int32)[:, None, None]    # (nP, 1, 1)
        ix_e   = ix_c [None]; iy_e = iy_c [None]                 # (1, 16, n_quad)
        ivx_e  = ivx_c[None]; ivy_e = ivy_c[None]

        # Each gather: (nP, 16, n_quad) → weighted sum → (nP, n_quad)
        def _gather_sum(stack):
            corners = stack[pi_idx, ix_e, iy_e, ivx_e, ivy_e]    # (nP, 16, n_quad)
            return (corners * w[None]).sum(1)                      # (nP, n_quad)

        dphi_resid = _gather_sum(self._dphi_stack)   # (nP, n_quad)
        amp0_out   = _gather_sum(self._amp0_stack).T  # (n_quad, nP)
        amp1_out   = _gather_sum(self._amp1_stack).T

        # Add linear phase trend: _dphi_linear is (nP, 5) — [c0,cx,cy,cvx,cvy]
        c = self._dphi_linear  # (nP, 5)
        dphi_out = (dphi_resid
                    + c[:, 0:1]
                    + c[:, 1:2]*x0_g[None] + c[:, 2:3]*y0_g[None]
                    + c[:, 3:4]*vx0_g[None] + c[:, 4:5]*vy0_g[None]).T  # (n_quad, nP)

        return dphi_out, amp0_out, amp1_out

    def pixel_acs(self, theta):
        """
        theta : (8,) — [mu_x0, mu_y0, mu_vx0, mu_vy0, sx0, sy0, svx0, svy0]

        Returns six GPU arrays each of shape (n_bins,):
            A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e
        """
        mu    = cp.asarray(theta[:4], dtype=cp.float64)
        sigma = cp.asarray(theta[4:], dtype=cp.float64)

        # Transform QMC standard normals → cloud Gaussian
        pts   = self.z_g * sigma[None] + mu[None]   # (n_quad, 4)
        x0_g  = pts[:, 0]; y0_g  = pts[:, 1]
        vx0_g = pts[:, 2]; vy0_g = pts[:, 3]

        # Detection positions and pixel bins
        xf_g = x0_g + self.t_det * vx0_g
        yf_g = y0_g + self.t_det * vy0_g
        ix = cp.searchsorted(self.x_edges_g, xf_g, side='right') - 1
        iy = cp.searchsorted(self.y_edges_g, yf_g, side='right') - 1
        inside  = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        bin_idx = (ix * self.ny + iy).astype(cp.int64)

        # Interpolate (phi-independent) — fully vectorised, no Python loops
        dphi_g, amp0_g, amp1_g = self._eval_fast(x0_g, y0_g, vx0_g, vy0_g)
        # shapes: (n_quad, nP)

        # Phi-independent terms
        base = amp0_g**2 + amp1_g**2                              # (n_quad, nP)
        mod  = self._port_inter[None]*2.0*amp0_g*amp1_g           # (n_quad, nP)
        c_dp = cp.cos(dphi_g); s_dp = cp.sin(dphi_g)              # (n_quad, nP)

        # 3-phi evaluation in one batch: phi = [0, pi/2, pi]
        # cos(dphi+phi) = cos_dphi*cos(phi) - sin_dphi*sin(phi)
        # cos(phi):  [1,  0, -1]
        # sin(phi):  [0,  1,  0]
        cos_total = cp.stack([ c_dp,       -s_dp,       -c_dp], axis=0)  # (3,n,nP)
        prob3 = cp.maximum(base[None] + mod[None]*cos_total, 0.0)         # (3,n,nP)

        # Ground/excited state sums
        pg3 = prob3[:, :, self.s0_g].sum(-1)   # (3, n_quad)
        pe3 = prob3[:, :, self.s1_g].sum(-1)

        Am_g = 0.5*(pg3[0]+pg3[2]); Ac_g = 0.5*(pg3[0]-pg3[2]); As_g = pg3[1] - Am_g
        Am_e = 0.5*(pe3[0]+pe3[2]); Ac_e = 0.5*(pe3[0]-pe3[2]); As_e = pe3[1] - Am_e

        # Equal-weight MC quadrature over inside-detector points
        n_in = float(inside.sum())
        if n_in < 1:
            z = cp.zeros(self.n_bins, dtype=cp.float64)
            return z, z, z, z, z, z

        iw = inside.astype(cp.float64) / n_in
        bidx_in = bin_idx[inside]
        bc = lambda v: cp.bincount(bidx_in, weights=iw[inside]*v[inside],
                                   minlength=self.n_bins)
        return bc(Am_g), bc(Ac_g), bc(As_g), bc(Am_e), bc(Ac_e), bc(As_e)


# ── Joint phi-marginalised logL ────────────────────────────────────────────────

def _joint_logL(n_g0, n_e0, A_g0, Cc_g0, Cs_g0, A_e0, Cc_e0, Cs_e0,
                n_g1, n_e1, A_g1, Cc_g1, Cs_g1, A_e1, Cc_e1, Cs_e1,
                n_theta, xp, lse_fn):
    """
    logL = logsumexp_k[ ll_Z0(phi_k) + ll_Z100(phi_k) ] - log(K)

    The two AIs share the same phi per shot (common phase), so their
    log-likelihoods are SUMMED before marginalising.
    """
    tot0 = float((A_g0 + A_e0).sum())
    tot1 = float((A_g1 + A_e1).sum())
    L0 = float((n_g0 + n_e0).sum()) / max(tot0, 1e-300)
    L1 = float((n_g1 + n_e1).sum()) / max(tot1, 1e-300)

    phi = xp.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    c   = xp.cos(phi)
    s   = xp.sin(phi)

    def _ll_ai(L, A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e):
        lam_g = L * xp.maximum(A_g[None] + Cc_g[None]*c[:,None] + Cs_g[None]*s[:,None], 1e-300)
        lam_e = L * xp.maximum(A_e[None] + Cc_e[None]*c[:,None] + Cs_e[None]*s[:,None], 1e-300)
        return (n_g[None]*xp.log(lam_g) - lam_g + n_e[None]*xp.log(lam_e) - lam_e).sum(1)

    # closure over n_g/n_e via local name binding
    n_g, n_e = n_g0, n_e0
    ll0 = _ll_ai(L0, A_g0, Cc_g0, Cs_g0, A_e0, Cc_e0, Cs_e0)
    n_g, n_e = n_g1, n_e1
    ll1 = _ll_ai(L1, A_g1, Cc_g1, Cs_g1, A_e1, Cc_e1, Cs_e1)

    return float(lse_fn(ll0 + ll1) - np.log(n_theta))


def _per_ai_logL(n_g, n_e, A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e,
                 n_theta, xp, lse_fn):
    """phi-marginalised logL for a single AI (for diagnostics)."""
    tot = float((A_g + A_e).sum())
    L   = float((n_g + n_e).sum()) / max(tot, 1e-300)
    phi = xp.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    c, s = xp.cos(phi), xp.sin(phi)
    lam_g = L * xp.maximum(A_g[None] + Cc_g[None]*c[:,None] + Cs_g[None]*s[:,None], 1e-300)
    lam_e = L * xp.maximum(A_e[None] + Cc_e[None]*c[:,None] + Cs_e[None]*s[:,None], 1e-300)
    ll = (n_g[None]*xp.log(lam_g) - lam_g + n_e[None]*xp.log(lam_e) - lam_e).sum(1)
    return float(lse_fn(ll) - np.log(n_theta))


# ── Parameter encoding / decoding ─────────────────────────────────────────────

def _encode(theta_z0, theta_z100):
    """16-element param vector for the optimiser. Spreads in log-space."""
    def enc8(t):
        return np.r_[t[:4], np.log(np.maximum(t[4:], 1e-12))]
    return np.r_[enc8(theta_z0), enc8(theta_z100)]


def _decode(params):
    """Decode 16-element optimiser vector → (theta_z0, theta_z100)."""
    def dec8(p):
        return np.r_[p[:4], np.exp(p[4:])]
    return dec8(params[:8]), dec8(params[8:])


def _build_bounds(half_range,
                  sigma_pos_min=1e-6, sigma_pos_max=5e-3,
                  sigma_vel_min=1e-6, sigma_vel_max=5e-3,
                  com_vel_max=5e-3,
                  **_):
    """
    L-BFGS-B bounds on the 16-element parameter vector.

    COM position: ±half_range (physical detector size)
    COM velocity: ±com_vel_max [m/s]
    log(sigma_pos): [log(sigma_pos_min), log(sigma_pos_max)]
    log(sigma_vel): [log(sigma_vel_min), log(sigma_vel_max)]

    Returned as a list of (lo, hi) pairs for scipy.optimize.minimize.
    """
    hr = float(half_range)
    cv = float(com_vel_max)
    lp_lo, lp_hi = np.log(sigma_pos_min), np.log(sigma_pos_max)
    lv_lo, lv_hi = np.log(sigma_vel_min), np.log(sigma_vel_max)

    # 8 per AI: mu_x0, mu_y0, mu_vx0, mu_vy0, log_sx0, log_sy0, log_svx0, log_svy0
    per_ai = [
        (-hr,    hr),    # mu_x0
        (-hr,    hr),    # mu_y0
        (-cv,    cv),    # mu_vx0
        (-cv,    cv),    # mu_vy0
        (lp_lo, lp_hi),  # log sigma_x0
        (lp_lo, lp_hi),  # log sigma_y0
        (lv_lo, lv_hi),  # log sigma_vx0
        (lv_lo, lv_hi),  # log sigma_vy0
    ]
    return per_ai + per_ai   # Z0 + Z100


# ── Shot objective ─────────────────────────────────────────────────────────────

class ShotObjective:
    """Negative log-likelihood for one shot, callable by scipy.optimize."""

    def __init__(self, n_g_z0, n_e_z0, n_g_z100, n_e_z100,
                 eval_z0: SurrogatePixelACS, eval_z100: SurrogatePixelACS,
                 n_theta: int, xp, lse_fn):
        self.xp = xp
        self.n_g_z0   = xp.asarray(n_g_z0.astype(np.float64))
        self.n_e_z0   = xp.asarray(n_e_z0.astype(np.float64))
        self.n_g_z100 = xp.asarray(n_g_z100.astype(np.float64))
        self.n_e_z100 = xp.asarray(n_e_z100.astype(np.float64))
        self.eval_z0   = eval_z0
        self.eval_z100 = eval_z100
        self.n_theta   = n_theta
        self.lse_fn    = lse_fn

    def __call__(self, params):
        theta_z0, theta_z100 = _decode(params)
        if np.any(theta_z0[4:] <= 0) or np.any(theta_z100[4:] <= 0):
            return 1e300
        try:
            acs0 = self.eval_z0.pixel_acs(theta_z0)
            acs1 = self.eval_z100.pixel_acs(theta_z100)
        except Exception:
            return 1e300
        logL = _joint_logL(
            self.n_g_z0, self.n_e_z0, *acs0,
            self.n_g_z100, self.n_e_z100, *acs1,
            self.n_theta, self.xp, self.lse_fn,
        )
        return -logL

    def logL_true(self, theta_z0, theta_z100):
        return -self(_encode(theta_z0, theta_z100))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _downsample(img, bins):
    res = img.shape[-1]; b = res // bins
    if img.ndim == 2:
        return img.reshape(bins, b, bins, b).sum(axis=(1, 3)).astype(np.float64)
    return img.reshape(2, bins, b, bins, b).sum(axis=(2, 4)).astype(np.float64)


def _moment_init(img2d, centers, default_sigma_x0, default_sigma_y0,
                 default_sigma_vx0, default_sigma_vy0):
    """
    Image-moment initialisation.  Returns theta = [mu_x0≈mu_xf, mu_y0≈mu_yf,
    0, 0, sigma_x0_default, sigma_y0_default, sigma_vx0_default, sigma_vy0_default].
    """
    total = float(img2d.sum())
    if total <= 0:
        return np.array([0., 0., 0., 0.,
                         default_sigma_x0, default_sigma_y0,
                         default_sigma_vx0, default_sigma_vy0])
    nc, nr = img2d.shape
    xc = centers[:, None]; yc = centers[None, :]
    mu_xf = float((img2d * xc).sum() / total)
    mu_yf = float((img2d * yc).sum() / total)
    return np.array([mu_xf, mu_yf, 0., 0.,
                     default_sigma_x0, default_sigma_y0,
                     default_sigma_vx0, default_sigma_vy0])


def _setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_name", nargs="?", type=Path, default=DEFAULT_RUN)
    p.add_argument("--run-idx", type=int, default=0)
    p.add_argument("--psmap-z0",   type=Path, default=DEFAULT_PSMAP_Z0)
    p.add_argument("--psmap-z100", type=Path, default=DEFAULT_PSMAP_Z100)
    p.add_argument("--bins",    type=int,   default=32)
    p.add_argument("--ntheta",  type=int,   default=512)
    p.add_argument("--n-quad",  type=int,   default=20_000,
                   help="QMC quadrature points per theta eval. (default: 20000)")
    p.add_argument("--max-shots", type=int, default=None)
    p.add_argument("--shot-start", type=int, default=0)
    # Default sigma for warm-start (overridden by --init-from-true)
    p.add_argument("--sigma-x0",  type=float, default=100e-6)
    p.add_argument("--sigma-y0",  type=float, default=100e-6)
    p.add_argument("--sigma-vx0", type=float, default=309e-6)
    p.add_argument("--sigma-vy0", type=float, default=309e-6)
    # Optimiser
    p.add_argument("--maxiter",          type=int,   default=2000)
    p.add_argument("--fatol",            type=float, default=0.5,
                   help="logL convergence tolerance (default: 0.5)")
    # Spread bounds (prevents escaping along sigma degeneracy ridge)
    p.add_argument("--sigma-pos-max",    type=float, default=5e-3,
                   help="Upper bound on sigma_x0, sigma_y0 [m]. (default: 5mm)")
    p.add_argument("--sigma-vel-max",    type=float, default=5e-3,
                   help="Upper bound on sigma_vx0, sigma_vy0 [m/s]. (default: 5mm/s)")
    p.add_argument("--com-vel-max",      type=float, default=5e-3,
                   help="Bound on |mu_vx0|, |mu_vy0| [m/s]. (default: 5mm/s)")
    p.add_argument("--init-from-true", action="store_true",
                   help="Initialise from true metadata (oracle / debugging).")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--no-gpu", action="store_true")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    use_gpu = _HAS_GPU and not args.no_gpu
    xp      = cp if use_gpu else np
    lse_fn  = cp_logsumexp if use_gpu else \
              __import__('scipy.special', fromlist=['logsumexp']).logsumexp

    run_path = Path(args.run_name)
    run_stem = run_path.name
    out_path = args.output or (
        REPO / "results" / f"{run_stem}_run{args.run_idx:03d}_cloud_mle.jsonl"
    )
    log_path = REPO / "logs" / f"profile_cloud_{run_stem}_run{args.run_idx:03d}.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_path)

    logging.info("profile_cloud_nuisances.py  run=%s  run_idx=%d", run_stem, args.run_idx)
    logging.info("GPU=%s  bins=%d  ntheta=%d  n_quad=%d  maxiter=%d  n_params=16",
                 use_gpu, args.bins, args.ntheta, args.n_quad, args.maxiter)

    run_dir  = run_path / f"run_{args.run_idx:03d}"
    ds_z0    = ImageShotDataset(str(run_dir / "Z0"   / "data_IMG.h5"))
    ds_z100  = ImageShotDataset(str(run_dir / "Z100" / "data_IMG.h5"))
    logging.info("Loaded %d shots  res=%d  half_range=%.1f mm",
                 ds_z0.n_shots, ds_z0.res, ds_z0.half_range * 1e3)

    edges   = np.linspace(-ds_z0.half_range, ds_z0.half_range, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if ds_z0.res % args.bins:
        raise ValueError(f"Image res {ds_z0.res} not divisible by bins {args.bins}")

    logging.info("Loading PSMAPs and building GPU surrogate evaluators (n_quad=%d)...",
                 args.n_quad)
    t0 = time.perf_counter()
    sur_z0   = PSMAPSurrogate(load_psmap(str(args.psmap_z0)),   T_DET, use_gpu=use_gpu)
    sur_z100 = PSMAPSurrogate(load_psmap(str(args.psmap_z100)), T_DET, use_gpu=use_gpu)
    eval_z0   = SurrogatePixelACS(sur_z0,   T_DET, edges, edges, n_quad=args.n_quad)
    eval_z100 = SurrogatePixelACS(sur_z100, T_DET, edges, edges, n_quad=args.n_quad)
    logging.info("Evaluators ready in %.1fs  n_bins=%d",
                 time.perf_counter() - t0, eval_z0.n_bins)

    shot_ids = list(range(args.shot_start, ds_z0.n_shots))
    if args.max_shots is not None:
        shot_ids = shot_ids[:args.max_shots]

    logging.info("Fitting %d shots (16 params each)...", len(shot_ids))
    rows    = []
    t_start = time.perf_counter()

    for count, shot_id in enumerate(shot_ids, start=1):
        t_shot = time.perf_counter()

        # ── Data ────────────────────────────────────────────────────────
        img_z0  = _downsample(ds_z0[shot_id],   args.bins)   # (2, bins, bins)
        img_z100 = _downsample(ds_z100[shot_id], args.bins)
        n_g_z0, n_e_z0   = img_z0[0].ravel(),   img_z0[1].ravel()
        n_g_z100, n_e_z100 = img_z100[0].ravel(), img_z100[1].ravel()

        # ── True params ──────────────────────────────────────────────────
        meta_z0   = ds_z0.meta(shot_id)
        meta_z100 = ds_z100.meta(shot_id)
        theta_true_z0   = np.array([meta_z0[H5_KEYS[n]]   for n in PARAM_NAMES])
        theta_true_z100 = np.array([meta_z100[H5_KEYS[n]] for n in PARAM_NAMES])

        # ── Objective ────────────────────────────────────────────────────
        obj = ShotObjective(n_g_z0, n_e_z0, n_g_z100, n_e_z100,
                            eval_z0, eval_z100, args.ntheta, xp, lse_fn)

        # ── Initialisation ───────────────────────────────────────────────
        if args.init_from_true:
            t_init_z0   = theta_true_z0.copy()
            t_init_z100 = theta_true_z100.copy()
        else:
            t_init_z0   = _moment_init(
                img_z0[0] + img_z0[1], centers,
                args.sigma_x0, args.sigma_y0, args.sigma_vx0, args.sigma_vy0)
            t_init_z100 = _moment_init(
                img_z100[0] + img_z100[1], centers,
                args.sigma_x0, args.sigma_y0, args.sigma_vx0, args.sigma_vy0)

        p0 = _encode(t_init_z0, t_init_z100)   # (16,)

        # ── Optimise ─────────────────────────────────────────────────────
        bounds = _build_bounds(
            ds_z0.half_range,
            sigma_pos_max=args.sigma_pos_max,
            sigma_vel_max=args.sigma_vel_max,
            com_vel_max=args.com_vel_max,
        )
        # L-BFGS-B supports bounds; prevents optimizer escaping along
        # the flat sigma_x0/sigma_vx0 degeneracy ridge to unphysical values.
        ll0 = -float(obj(p0))
        opt = minimize(obj, p0, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": args.maxiter,
                                "ftol": args.fatol / max(abs(ll0), 1.0),
                                "gtol": 1e-8})
        theta_hat_z0, theta_hat_z100 = _decode(opt.x)
        logL_hat  = -float(opt.fun)
        logL_true = obj.logL_true(theta_true_z0, theta_true_z100)

        elapsed = time.perf_counter() - t_shot

        # ── Output row ───────────────────────────────────────────────────
        row = {
            "shot":       shot_id,
            "logL_hat":   logL_hat,
            "logL_true":  logL_true,
            "delta_logL": logL_hat - logL_true,
            "nfev":       int(opt.nfev),
            "nit":        int(opt.nit),
            "success":    bool(opt.success),
            "message":    str(opt.message),
            "elapsed_s":  round(elapsed, 2),
            "run_name":   run_stem,
            "run_idx":    args.run_idx,
        }
        for ai, (theta_hat, theta_true) in [
            ("z0",  (theta_hat_z0,  theta_true_z0)),
            ("z100",(theta_hat_z100, theta_true_z100)),
        ]:
            for name, v_hat, v_true in zip(PARAM_NAMES, theta_hat, theta_true):
                row[f"{name}_{ai}_hat"]  = float(v_hat)
                row[f"{name}_{ai}_true"] = float(v_true)
                row[f"{name}_{ai}_err"]  = float(v_hat - v_true)
        rows.append(row)

        if count % 5 == 0 or count == len(shot_ids):
            elapsed_tot = time.perf_counter() - t_start
            mean_delta = np.mean([r["delta_logL"] for r in rows])
            rmse_mux   = np.sqrt(np.mean([r["mu_x0_z0_err"]**2 for r in rows])) * 1e6
            rmse_sx    = np.sqrt(np.mean([r["sigma_x0_z0_err"]**2 for r in rows])) * 1e6
            logging.info(
                "Shot %3d/%d | %.1fs | delta_logL=%.1f | "
                "RMSE(mu_x0_z0)=%.1fµm RMSE(sigma_x0_z0)=%.1fµm",
                count, len(shot_ids), elapsed_tot, mean_delta, rmse_mux, rmse_sx,
            )

    # ── Write output ──────────────────────────────────────────────────────────
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    logging.info("Wrote %d rows to %s", len(rows), out_path)

    delta_lls = [r["delta_logL"] for r in rows]
    logging.info("delta_logL: mean=%.1f  std=%.1f  min=%.1f  max=%.1f",
                 np.mean(delta_lls), np.std(delta_lls),
                 np.min(delta_lls), np.max(delta_lls))
    for ai in ["z0", "z100"]:
        for name in PARAM_NAMES:
            errs = np.array([r[f"{name}_{ai}_err"] for r in rows])
            logging.info("  %-12s  %-4s  bias=%+.2g  rmse=%.2g",
                         name, ai, float(np.mean(errs)), float(np.sqrt(np.mean(errs**2))))


if __name__ == "__main__":
    main()
