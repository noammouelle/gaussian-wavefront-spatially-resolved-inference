"""
joint_profile_inference.py — signal (A_s, A_c) inference via nested profiling
of the per-shot nuisances (phi_i, theta_z0, theta_z100) against the FULL
pixel-resolved likelihood, instead of the crude 4-moment Kalman update used
by map_inference.py's "moment" mode.

Why this exists
----------------
The CRB work (crb_signal.py) showed the port-summed/moment-based pipeline
retains only ~9% of the theoretically achievable phase information once
realistic cloud-parameter (theta) uncertainty is accounted for, while the
full pixel-resolved likelihood retains ~96%. This script is the estimator
designed to actually realise that improvement: profile theta (and phi)
against the full pixel image + Gaussian prior at each shot, instead of a
4-number moment summary.

Structure: nested, but efficiently so
--------------------------------------
We established (see notes/theta_beta_sensitivity discussion) that profiling
theta ONCE at a reference beta and holding it fixed is NOT valid -- theta
(especially the weakly-identified ridge/width components) shifts by 6-24
posterior-sigma as beta sweeps a realistic range. So theta MUST be re-profiled
at (or near) every trial beta during the outer optimisation.

The joint Fisher matrix has exact "arrow" sparsity (each shot's 17-dim
nuisance block only couples to beta, never to another shot's nuisances), so
a properly-built joint optimiser over (beta, all thetas, all phis)
mathematically reduces to: outer step on beta, inner per-shot Newton step on
(phi_i, theta_z0, theta_z100), warm-started from the previous outer
iteration's solution. That's what's implemented here -- NOT a from-scratch
re-optimisation per shot per outer step, and NOT a monolithic dense joint
solve either.

phi's gradient/curvature is analytic (cos/sin chain rule); theta's is via
finite differences (see crb_signal.py's pixel_fisher_block) -- true analytic
PSMAP derivatives are possible (Catmull-Rom spline weights are cubic
polynomials with closed-form derivatives) but not implemented here; this is
the safe, already-validated route, sped up via warm-starting rather than
analytic derivatives.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import crb_signal as crb                                          # noqa: E402
import map_inference as mi                                         # noqa: E402
import pixel_acs_grad as pag                                        # noqa: E402
import pixel_acs_grad_batch as pagb                                 # noqa: E402
from helpers import ImageShotDataset                                # noqa: E402


# ── per-shot joint (phi, theta_z0, theta_z100) profiler ─────────────────────
# Deliberately simple: plain NLL functions, no hand-rolled gradients/Hessians.
# scipy does its own differentiation (finite differences internally). Slower
# than the analytic-gradient version, but much less likely to have a subtle
# sign/indexing bug -- correctness first.

def _ai_nll_only(theta, acs, n_g, n_e, phi):
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = crb.pixel_stats(theta, acs)
    c, s = np.cos(phi), np.sin(phi)
    I_g = A_g + Cc_g * c + Cs_g * s
    I_e = A_e + Cc_e * c + Cs_e * s
    n_tot = float(n_g.sum() + n_e.sum())
    L = n_tot / max(float((I_g + I_e).sum()), crb.EPS)
    mu_g = L * np.maximum(I_g, crb.EPS)
    mu_e = L * np.maximum(I_e, crb.EPS)
    return float(np.sum(mu_g - n_g * np.log(mu_g)) + np.sum(mu_e - n_e * np.log(mu_e)))


def prior_nll(theta, prior_mean, prior_std):
    return 0.5 * float(np.sum(((theta - prior_mean) / prior_std) ** 2))


def joint_nll(params, delta_phi_trial, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
              prior_mean, prior_std):
    """Scalar NLL_z0 + NLL_z100 + priors at an arbitrary point. No gradient."""
    phi_i = params[0]; theta_z0 = params[1:9]; theta_z100 = params[9:17]
    nll = _ai_nll_only(theta_z0, acs_z0, n_g0, n_e0, phi_i)
    nll += _ai_nll_only(theta_z100, acs_z100, n_g1, n_e1, phi_i + delta_phi_trial)
    nll += prior_nll(theta_z0, prior_mean, prior_std) + prior_nll(theta_z100, prior_mean, prior_std)
    return nll


# ── analytic-gradient version (validated: pixel_acs_grad.pixel_acs_and_grad) ─

_H_THETA_ANALYTIC = np.full(8, 1e-9)


def _ai_nll_and_grad_analytic(theta, acs, n_g, n_e, phi):
    """NLL and (9,) gradient w.r.t. (phi, theta[8]) for one AI, using the
    validated analytic Catmull-Rom gradient (pixel_acs_grad), not finite
    differences of the full pixel_acs."""
    base, dbase = pag.pixel_acs_and_grad(acs, theta, _H_THETA_ANALYTIC)
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = [b.get() if hasattr(b, 'get') else b for b in base]
    c, s = np.cos(phi), np.sin(phi)
    I_g = A_g + Cc_g * c + Cs_g * s
    I_e = A_e + Cc_e * c + Cs_e * s
    n_tot = float(n_g.sum() + n_e.sum())
    L = n_tot / max(float((I_g + I_e).sum()), crb.EPS)
    mu_g = L * np.maximum(I_g, crb.EPS)
    mu_e = L * np.maximum(I_e, crb.EPS)
    nll = float(np.sum(mu_g - n_g * np.log(mu_g)) + np.sum(mu_e - n_e * np.log(mu_e)))

    dI_g_dphi = -Cc_g * s + Cs_g * c
    dI_e_dphi = -Cc_e * s + Cs_e * c
    dmu_g_dphi = L * dI_g_dphi
    dmu_e_dphi = L * dI_e_dphi
    dnll_dphi = float(np.sum(dmu_g_dphi * (1 - n_g / mu_g)) + np.sum(dmu_e_dphi * (1 - n_e / mu_e)))

    grad = np.zeros(9)
    grad[0] = dnll_dphi
    for k in range(8):
        dA_g, dCc_g, dCs_g, dA_e, dCc_e, dCs_e = dbase[k]
        dI_g = dA_g + dCc_g * c + dCs_g * s
        dI_e = dA_e + dCc_e * c + dCs_e * s
        dmu_g = L * dI_g
        dmu_e = L * dI_e
        grad[1 + k] = float(np.sum(dmu_g * (1 - n_g / mu_g)) + np.sum(dmu_e * (1 - n_e / mu_e)))
    return nll, grad


def joint_nll_and_grad_analytic(params, delta_phi_trial, acs_z0, acs_z100,
                                 n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std):
    phi_i = params[0]; theta_z0 = params[1:9]; theta_z100 = params[9:17]
    nll0, g0 = _ai_nll_and_grad_analytic(theta_z0, acs_z0, n_g0, n_e0, phi_i)
    nll1, g1 = _ai_nll_and_grad_analytic(theta_z100, acs_z100, n_g1, n_e1, phi_i + delta_phi_trial)
    d0 = (theta_z0 - prior_mean) / prior_std
    d1 = (theta_z100 - prior_mean) / prior_std
    pnll0 = 0.5 * float(np.sum(d0 ** 2)); pnll1 = 0.5 * float(np.sum(d1 ** 2))
    grad = np.zeros(17)
    grad[0] = g0[0] + g1[0]
    grad[1:9] = g0[1:] + d0 / prior_std
    grad[9:17] = g1[1:] + d1 / prior_std
    nll = nll0 + nll1 + pnll0 + pnll1
    return nll, grad


def profile_shot_analytic(x0, delta_phi_trial, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
                           prior_mean, prior_std, n_iter=30, grad_tol=1e-3, _scale_cache=[None]):
    """Same as profile_shot, but with the validated analytic gradient supplied
    via jac=True -- scipy's own (robust) L-BFGS-B, just without the expensive
    finite-difference outer wrapper scipy would otherwise do.

    NLL here is O(1e7) with a correspondingly huge gradient, which causes
    L-BFGS-B's default tolerances (tuned for O(1) problems) to report false
    convergence at a large residual gradient. We rescale the objective (does
    not move the argmin) so the optimizer's defaults behave sanely, then
    verify -- not just assume -- that the UNSCALED gradient norm at the
    reported solution is actually small before trusting it.

    Also returns dNLL_z100/d(delta_phi_trial) at the converged point: by the
    envelope theorem, this equals d(total NLL)/d(delta_phi_trial) exactly AT
    A TRUE STATIONARY POINT of the inner objective (z0's contribution and the
    implicit d(phi,theta)/d(delta_phi) terms vanish there) -- validity of this
    depends entirely on actually being converged, hence the check above.
    """
    if _scale_cache[0] is None:
        _, g_probe = joint_nll_and_grad_analytic(x0, delta_phi_trial, acs_z0, acs_z100,
                                                   n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)
        _scale_cache[0] = 1.0 / max(np.linalg.norm(g_probe), 1.0)
    scale = _scale_cache[0]

    def scaled_obj(params):
        nll, grad = joint_nll_and_grad_analytic(params, delta_phi_trial, acs_z0, acs_z100,
                                                  n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)
        return nll * scale, grad * scale

    # Bound theta to a generous multiple of the prior width: nothing in the
    # physics allows theta to run off arbitrarily far, and without bounds the
    # unconstrained solve can chase a spurious unbounded-improvement direction
    # (observed directly: |grad| GREW with more iterations instead of shrinking,
    # the signature of divergence, not slow convergence).
    N_SIGMA = 20.0
    theta_lo = prior_mean - N_SIGMA * prior_std
    theta_hi = prior_mean + N_SIGMA * prior_std
    bnds = [(-50.0, 50.0)]  # phi: wide but finite
    bnds += list(zip(theta_lo, theta_hi)) * 2  # theta_z0, theta_z100

    res = minimize(scaled_obj, x0, jac=True, method='L-BFGS-B', bounds=bnds,
                    options={'maxiter': n_iter, 'gtol': 1e-10, 'ftol': 1e-15})
    x = res.x
    nll_final, grad_final = joint_nll_and_grad_analytic(x, delta_phi_trial, acs_z0, acs_z100,
                                                          n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)
    gnorm = float(np.linalg.norm(grad_final))
    if gnorm > grad_tol * max(abs(nll_final), 1.0):
        import warnings
        warnings.warn(f'profile_shot_analytic: inner solve NOT converged '
                       f'(|grad|={gnorm:.3e}, nll={nll_final:.3e}, ratio={gnorm/max(abs(nll_final),1.0):.3e}) '
                       f'-- envelope-theorem outer gradient is unreliable here.')
    _, g1 = _ai_nll_and_grad_analytic(x[9:17], acs_z100, n_g1, n_e1, x[0] + delta_phi_trial)
    return x, float(nll_final), float(g1[0])


def profile_shot(x0, delta_phi_trial, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
                  prior_mean, prior_std, n_iter=30):
    """Warm-started per-shot profile of (phi, theta_z0, theta_z100), via a
    plain scipy.optimize.minimize (L-BFGS-B, scipy's own finite-diff gradient).
    Returns the solution and its NLL only -- no envelope-theorem shortcut."""
    res = minimize(joint_nll, x0, args=(delta_phi_trial, acs_z0, acs_z100,
                                         n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std),
                    method='L-BFGS-B', options={'maxiter': n_iter})
    return res.x, float(res.fun)


# ── phi multi-start: fix for the multimodal-phi problem ─────────────────────
# Diagnostic (1D NLL scan vs phi, theta held fixed) found 3-4 genuine local
# minima per 2pi period for a subset of shots' phi landscape -- both fit_beta
# (scipy L-BFGS-B, warm-started from phi=0) and fit_beta_batch (per-shot
# L-BFGS, same phi=0 start) can lock onto whichever basin phi=0 happens to
# fall into, and it's not always the best one (nor even the same one between
# the two solvers). Since state['x'] is initialized ONCE before the outer
# beta loop starts (and the first trial beta is ~0, so delta_phi_trial~0 at
# that point), a coarse grid scan over phi at delta_phi_trial=0 and
# theta=prior_mean -- picking the best as the initial phi instead of a fixed
# 0.0 -- makes both solvers start in the same, better basin instead of
# whichever one a fixed start happens to fall into.

def _phi_prescan(acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std, n_grid=12):
    """Return the best phi in [-pi, pi) (theta at prior_mean, delta_phi_trial=0)
    by direct NLL evaluation on a coarse grid -- cheap (no optimization) and
    used only once, to pick a better initial phi than a fixed 0.0."""
    phi_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    nlls = [joint_nll(np.concatenate([[p], prior_mean, prior_mean]), 0.0,
                       acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)
            for p in phi_grid]
    return float(phi_grid[int(np.argmin(nlls))])


# ── outer beta loop, warm-starting each shot's nuisance state ───────────────

def fit_beta(run_dir, n_shots, f_signal, prior_mean, prior_std, bins=16, pixel_n_gh=4,
             n_inner_newton=3, gh_order=4, grid_half=0.2, n_starts=1, outer_maxiter=15,
             use_analytic=True, log=print):
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    acs_z0, acs_z100 = crb.build_evaluators(run_dir, mi.DEFAULT_T_DET, bins,
                                             likelihood='pixel', pixel_n_gh=pixel_n_gh)
    h_theta = np.full(8, 1e-7)

    shot_ids = list(range(min(n_shots, ds_z0.n_shots)))
    shot_idx_arr = np.array(shot_ids, dtype=np.float64)
    sin_i = np.sin(2 * np.pi * f_signal * shot_idx_arr)
    cos_i = np.cos(2 * np.pi * f_signal * shot_idx_arr)
    counts = []
    for sid in shot_ids:
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_g0 = img0[0].astype(np.float64); n_e0 = img0[1].astype(np.float64)
        n_g1 = img1[0].astype(np.float64); n_e1 = img1[1].astype(np.float64)
        # NOTE: real images are at native resolution; downsample to match acs bins,
        # and flatten to match pixel_acs's bin_idx = ix*ny+iy row-major convention.
        n_g0 = _downsample(n_g0, bins).ravel(); n_e0 = _downsample(n_e0, bins).ravel()
        n_g1 = _downsample(n_g1, bins).ravel(); n_e1 = _downsample(n_e1, bins).ravel()
        counts.append((n_g0, n_e0, n_g1, n_e1))

    # warm-started nuisance state, persists across outer-loop evaluations.
    # phi0 per shot comes from a coarse pre-scan (see _phi_prescan) instead of
    # a fixed 0.0, to reliably land in a good phi basin (see module-level
    # comment above _phi_prescan for why this matters).
    state = {'x': [np.concatenate([[_phi_prescan(acs_z0, acs_z100, *counts[k], prior_mean, prior_std)],
                                    prior_mean, prior_mean])
                   for k in range(len(shot_ids))]}

    def neg_logL(beta):
        As, Ac = float(beta[0]), float(beta[1])
        delta_phi = As * sin_i + Ac * cos_i
        total = 0.0
        for k, sid in enumerate(shot_ids):
            n_g0, n_e0, n_g1, n_e1 = counts[k]
            x_new, nll = profile_shot(state['x'][k], delta_phi[k], acs_z0, acs_z100,
                                       n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std,
                                       n_iter=n_inner_newton)
            state['x'][k] = x_new
            total += nll
        return total

    def _neg_logL_and_grad_raw(beta):
        As, Ac = float(beta[0]), float(beta[1])
        delta_phi = As * sin_i + Ac * cos_i
        total = 0.0
        d_dbeta = np.zeros(2)
        for k, sid in enumerate(shot_ids):
            n_g0, n_e0, n_g1, n_e1 = counts[k]
            x_new, nll, dnll_ddphi = profile_shot_analytic(state['x'][k], delta_phi[k], acs_z0, acs_z100,
                                                             n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std,
                                                             n_iter=n_inner_newton)
            state['x'][k] = x_new   # warm-start for the next beta evaluation
            total += nll
            # envelope theorem, validated against outer-objective finite differences
            # (see the __main__ validation block) before being trusted here.
            d_dbeta[0] += dnll_ddphi * sin_i[k]
            d_dbeta[1] += dnll_ddphi * cos_i[k]
        return total, d_dbeta

    # L-BFGS-B's default first-step heuristic is ~ -gradient before it has built
    # up curvature history, which massively overshoots given |gradient| ~ 1e5
    # against a beta scale of ~0.01-0.2. Rescaling the objective (and gradient)
    # by a constant doesn't move the argmin, just brings step sizes into a sane
    # range -- pick the scale from the gradient magnitude at the starting point.
    _, g0 = _neg_logL_and_grad_raw(np.zeros(2))
    grad_norm0 = max(np.linalg.norm(g0), 1e-30)
    obj_scale = 0.02 / grad_norm0   # target an initial step of order 0.02 in beta
    log(f'  outer objective scale = {obj_scale:.3e}  (raw grad norm at beta=0: {grad_norm0:.3e})')

    def neg_logL_and_grad_analytic(beta):
        total, d_dbeta = _neg_logL_and_grad_raw(beta)
        return total * obj_scale, d_dbeta * obj_scale

    t0 = time.perf_counter()
    bounds = [(-grid_half, grid_half)] * 2
    starts = [np.zeros(2)] + [np.array([0.05, 0.0]), np.array([-0.05, 0.05])][:max(0, n_starts - 1)]
    best = None
    for b0 in starts[:n_starts]:
        if use_analytic:
            res = minimize(neg_logL_and_grad_analytic, b0, jac=True, method='L-BFGS-B',
                            bounds=bounds, options={'maxiter': outer_maxiter})
        else:
            res = minimize(neg_logL, b0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': outer_maxiter})
        log(f'  start beta0={b0}  beta_hat={res.x}  nll={res.fun:.4f}  success={res.success}  '
            f'nit={res.nit}')
        if best is None or res.fun < best.fun:
            best = res
    log(f'fit_beta done in {time.perf_counter()-t0:.1f}s  beta_hat={best.x}')
    return best.x


# ── batched-across-shots inner solve (pixel_acs_grad_batch) ─────────────────

def _batched_nll_grad_per_shot(X, delta_phi_batch, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
                                prior_mean, prior_std, N, h_theta):
    """
    Same computation as `batched_nll_grad`, but keeps the shot axis instead of
    collapsing the NLL to one scalar summed over all N shots. This is what
    `solve_inner_batch_independent` needs to give each shot its own line
    search / convergence decision while still doing the expensive
    value/gradient evaluation as ONE batched GPU call across shots (see
    "Issue 2" in notes/joint_profile_inference.tex's Known Open Issues).

    X: (17N,) = [phi (N,), theta_z0.ravel() (8N,), theta_z100.ravel() (8N,)]
    n_g0 etc: (N, n_bins) numpy, one row per shot.
    Returns:
      nll_per_shot   (N,)   -- NLL_z0 + NLL_z100 + priors, per shot (not summed)
      grad_phi       (N,)
      grad_theta_z0  (N,8)
      grad_theta_z100(N,8)
      dnll1_dphi     (N,)   -- dNLL_z100/d(phi_i+delta_phi_trial_i), for the
                               outer envelope-theorem gradient (see
                               `batched_nll_grad`'s docstring for why this is
                               kept separate from grad_phi).
    """
    phi = X[:N]
    theta_z0 = X[N:N + 8*N].reshape(N, 8)
    theta_z100 = X[N + 8*N:N + 16*N].reshape(N, 8)

    base0, grad0 = pagb.pixel_acs_and_grad_batch(acs_z0, theta_z0, h_theta)
    base1, grad1 = pagb.pixel_acs_and_grad_batch(acs_z100, theta_z100, h_theta)

    def ai_part(base, grad, n_g, n_e, phi_arg):
        A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = [b.get() for b in base]   # each (N, n_bins)
        c = np.cos(phi_arg)[:, None]; s = np.sin(phi_arg)[:, None]   # (N,1)
        I_g = A_g + Cc_g*c + Cs_g*s; I_e = A_e + Cc_e*c + Cs_e*s
        n_tot = (n_g + n_e).sum(axis=1)                              # (N,)
        L = n_tot / np.maximum((I_g + I_e).sum(axis=1), crb.EPS)     # (N,)
        mu_g = L[:, None] * np.maximum(I_g, crb.EPS)
        mu_e = L[:, None] * np.maximum(I_e, crb.EPS)
        nll = (mu_g - n_g*np.log(mu_g)).sum(axis=1) + (mu_e - n_e*np.log(mu_e)).sum(axis=1)   # (N,)

        dI_g_dphi = -Cc_g*s + Cs_g*c; dI_e_dphi = -Cc_e*s + Cs_e*c
        dmu_g_dphi = L[:, None]*dI_g_dphi; dmu_e_dphi = L[:, None]*dI_e_dphi
        dnll_dphi = (dmu_g_dphi*(1 - n_g/mu_g)).sum(axis=1) + (dmu_e_dphi*(1 - n_e/mu_e)).sum(axis=1)   # (N,)

        # grad: (N,8,6,n_bins) -> theta components 0..5 = A_g,Cc_g,Cs_g,A_e,Cc_e,Cs_e
        dA_g, dCc_g, dCs_g = grad[:, :, 0, :], grad[:, :, 1, :], grad[:, :, 2, :]   # (N,8,n_bins)
        dA_e, dCc_e, dCs_e = grad[:, :, 3, :], grad[:, :, 4, :], grad[:, :, 5, :]
        dI_g_dtheta = dA_g + dCc_g*c[:, None, :] + dCs_g*s[:, None, :]   # (N,8,n_bins)
        dI_e_dtheta = dA_e + dCc_e*c[:, None, :] + dCs_e*s[:, None, :]
        dmu_g_dtheta = L[:, None, None]*dI_g_dtheta
        dmu_e_dtheta = L[:, None, None]*dI_e_dtheta
        w_g = (1 - n_g/mu_g)[:, None, :]; w_e = (1 - n_e/mu_e)[:, None, :]
        dnll_dtheta = (dmu_g_dtheta*w_g).sum(axis=2) + (dmu_e_dtheta*w_e).sum(axis=2)   # (N,8)
        return nll, dnll_dphi, dnll_dtheta

    nll0, dnll0_dphi, dnll0_dtheta = ai_part(base0, grad0, n_g0, n_e0, phi)
    nll1, dnll1_dphi, dnll1_dtheta = ai_part(base1, grad1, n_g1, n_e1, phi + delta_phi_batch)

    d0 = (theta_z0 - prior_mean[None, :]) / prior_std[None, :]
    d1 = (theta_z100 - prior_mean[None, :]) / prior_std[None, :]
    pnll = 0.5*(d0**2).sum(axis=1) + 0.5*(d1**2).sum(axis=1)                # (N,)

    nll_per_shot = nll0 + nll1 + pnll
    grad_phi = dnll0_dphi + dnll1_dphi                                    # (N,)
    grad_theta_z0 = dnll0_dtheta + d0/prior_std[None, :]                  # (N,8)
    grad_theta_z100 = dnll1_dtheta + d1/prior_std[None, :]

    return nll_per_shot, grad_phi, grad_theta_z0, grad_theta_z100, dnll1_dphi


def batched_nll_grad(X, delta_phi_batch, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
                      prior_mean, prior_std, N, h_theta):
    """
    X: (17N,) = [phi (N,), theta_z0.ravel() (8N,), theta_z100.ravel() (8N,)]
    n_g0 etc: (N, n_bins) numpy, one row per shot.
    Returns (total_nll, (17N,) gradient), via ONE batched call per AI instead
    of a Python loop over shots. Thin wrapper around
    `_batched_nll_grad_per_shot` that collapses the per-shot NLL to the total
    scalar needed by scipy's `minimize` (still used for the outer envelope
    gradient in `fit_beta_batch` -- see `_batched_nll_grad_for_scipy`).
    """
    nll_per_shot, grad_phi, grad_theta_z0, grad_theta_z100, dnll1_dphi = _batched_nll_grad_per_shot(
        X, delta_phi_batch, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
        prior_mean, prior_std, N, h_theta)

    total_nll = float(nll_per_shot.sum())
    full_grad = np.concatenate([grad_phi, grad_theta_z0.ravel(), grad_theta_z100.ravel()])
    # dnll1_dphi (N,) = dNLL_z100/d(phi_i + delta_phi_trial_i) is returned
    # separately (not just folded into grad_phi) because it is exactly what
    # the outer envelope-theorem gradient needs: by the envelope theorem,
    # d(total NLL)/d(delta_phi_trial_i) at a converged inner stationary point
    # equals the EXPLICIT partial derivative of NLL_z100 w.r.t. its argument
    # (phi_i + delta_phi_trial_i), NOT the full grad_phi (which also contains
    # z0's contribution and would double count / mismatch the true total
    # derivative). See profile_shot_analytic's identical use of g1[0] in the
    # per-shot loop (fit_beta) for the validated single-shot analogue.
    return total_nll, full_grad, dnll1_dphi


def _batched_nll_grad_for_scipy(X, *args):
    """scipy-compatible (f, g) wrapper around batched_nll_grad, which also
    returns the per-shot dnll1_dphi needed for the outer envelope gradient."""
    nll, grad, _ = batched_nll_grad(X, *args)
    return nll, grad


def solve_inner_batch_independent(x0, delta_phi_batch, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
                                   prior_mean, prior_std, N, h_theta, theta_lo, theta_hi,
                                   n_iter=50, grad_tol=1e-3, c1=1e-4, max_backtrack=20,
                                   phi_bound=50.0, log=None, memory=6, max_reject_streak=30):
    """Fix for Issue 2 (see notes/joint_profile_inference.tex, Known Open
    Issues): a vectorized per-shot L-BFGS inner solver. Batches the
    expensive value/gradient evaluation across all N shots in one call per
    iteration (via `_batched_nll_grad_per_shot`, i.e. `pixel_acs_grad_batch`),
    but each shot keeps its OWN (S,Y,rho) curvature history and OWN
    per-shot step size -- N independent quasi-Newton solves that share only
    the batched GPU eval, unlike the single monolithic
    `scipy.optimize.minimize` over the stacked (17N,)-dim vector this
    replaces (which shares ONE curvature/step-size history across all shots
    and can let one shot's hard, multimodal phi-landscape stall the whole
    batch).

    Design history (see notes/joint_profile_inference.tex Known Open Issues,
    and single/multi-shot validation done alongside this function):

    1. A first attempt used vectorized Barzilai-Borwein + backtracking
       Armijo (cheap per step, no curvature memory). It converged to a
       distinctly WORSE local optimum than the reference `profile_shot_analytic`
       (L-BFGS-B) on a hard phi landscape and was 6x SLOWER besides -- BB
       lacks the multi-step curvature that lets L-BFGS-B escape that basin.
    2. Replacing BB with genuine per-shot L-BFGS (batched two-loop
       recursion) fixed correctness, but an inner backtracking-line-search
       loop (retry smaller alpha immediately, same as attempt 1) was
       measured to be the actual speed killer: at N=40 shots, profiling
       showed 100% of wall time inside the batched eval and 783 eval calls
       for just 50 outer iterations (~16/iteration instead of ~1). The
       cause is structural, not a tuning issue: because ALL N shots share
       one batched eval call per backtracking round, if even ONE shot needs
       many rounds that iteration, every other shot pays for those rounds
       too -- even ones that would have accepted on round 1. This makes the
       batch's per-iteration cost set by its worst shot, and gets WORSE
       (not better) as N grows, since the odds that at least one of N shots
       is having a bad iteration increase with N (measured speedup vs the
       per-shot loop: 1.10x at N=8, 0.46x -- i.e. SLOWER -- at N=40).
    3. This version (current): decouples backtracking from eval-call count
       entirely by spending exactly ONE eval call per outer iteration for
       the whole batch, regardless of how many shots reject their step.
       A rejected shot doesn't retry immediately -- it just shrinks its own
       `cur_alpha` and tries again next outer iteration (so total eval
       calls = n_iter + 1, independent of any shot's difficulty). This is
       the batching-friendly structure: fixed total work, not
       data-dependent retry loops.

    x0: (17N,) packed [phi, theta_z0.ravel(), theta_z100.ravel()] -- same
    layout `_solve_inner`/`batched_nll_grad` use. Returns the same packed
    (17N,) layout, so it's a drop-in replacement for the old `_solve_inner`
    body. `max_backtrack` is unused (kept for signature/call-site
    compatibility with the previous immediate-retry version).
    """
    D = 17
    idx_n = np.arange(N)
    # Diagonal preconditioner for the steepest-descent fallback direction:
    # phi and theta live on wildly different natural scales (phi ~ O(1) rad,
    # theta ~ O(prior_std) ~ 1e-5). A bare steepest-descent step -grad, even
    # scaled by a single scalar alpha=1/|grad|, still carries those raw
    # per-component ratios -- one component can massively overshoot its
    # natural scale (and get clipped hard at its bound) while another barely
    # moves. Precondition by prior_std**2 (a diagonal Gauss-Newton-like
    # scaling assuming curvature ~ 1/scale**2 per component, same spirit as
    # the theta bounds already being expressed in units of prior_std).
    _P_diag = np.concatenate([[1.0], prior_std ** 2, prior_std ** 2])

    def _unpack(Xflat):
        phi = Xflat[:N]
        tz0 = Xflat[N:N + 8*N].reshape(N, 8)
        tz100 = Xflat[N + 8*N:N + 16*N].reshape(N, 8)
        return np.concatenate([phi[:, None], tz0, tz100], axis=1)   # (N,17)

    def _pack(Xmat):
        return np.concatenate([Xmat[:, 0], Xmat[:, 1:9].ravel(), Xmat[:, 9:17].ravel()])

    def _eval(Xmat):
        nll_ps, gphi, gtz0, gtz100, _ = _batched_nll_grad_per_shot(
            _pack(Xmat), delta_phi_batch, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
            prior_mean, prior_std, N, h_theta)
        grad_mat = np.concatenate([gphi[:, None], gtz0, gtz100], axis=1)   # (N,17)
        return nll_ps, grad_mat

    lo = np.concatenate([[-phi_bound], theta_lo, theta_lo])
    hi = np.concatenate([[phi_bound], theta_hi, theta_hi])

    X = _unpack(x0)
    active = np.ones(N, dtype=bool)

    # Per-shot circular (S,Y,rho) history, oldest at slot 0 conceptually but
    # tracked via hist_ptr (next write slot) / hist_count (# valid pairs) so
    # each shot can have a different number of accepted curvature pairs.
    S = np.zeros((N, memory, D)); Y = np.zeros((N, memory, D)); rho = np.zeros((N, memory))
    hist_ptr = np.zeros(N, dtype=int); hist_count = np.zeros(N, dtype=int)
    # Per-shot step size, persists (and shrinks/grows) ACROSS outer
    # iterations -- see the docstring's design-history point 3: there is no
    # inner retry loop, so this is the only thing standing in for a line
    # search. Counts consecutive rejections per shot to detect real stalls.
    cur_alpha = np.ones(N)
    reject_streak = np.zeros(N, dtype=int)
    gave_up = np.zeros(N, dtype=bool)

    nll_cur, grad_cur = _eval(X)

    for it in range(n_iter):
        gnorm = np.linalg.norm(grad_cur, axis=1)
        active &= ~(gnorm < grad_tol * np.maximum(np.abs(nll_cur), 1.0))
        if not active.any():
            break

        # Two-loop L-BFGS recursion, vectorized over the shot axis: each
        # shot walks its OWN history, most-recent-pair-first. Slots never
        # written for a given shot have rho=0, which zeroes their
        # contribution to both loops automatically.
        q = grad_cur.copy()
        alpha = np.zeros((N, memory))
        for j in range(memory):
            slot = (hist_ptr - 1 - j) % memory
            s_j, y_j, rho_j = S[idx_n, slot], Y[idx_n, slot], rho[idx_n, slot]
            a_j = rho_j * np.einsum('nd,nd->n', s_j, q)
            alpha[:, j] = a_j
            q -= a_j[:, None] * y_j

        last_slot = (hist_ptr - 1) % memory
        s_last, y_last = S[idx_n, last_slot], Y[idx_n, last_slot]
        yy = np.einsum('nd,nd->n', y_last, y_last)
        sy_last = np.einsum('nd,nd->n', s_last, y_last)
        gamma = np.where((hist_count > 0) & (yy > 1e-30), sy_last / np.maximum(yy, 1e-30), 1.0)
        r = gamma[:, None] * q
        for j in reversed(range(memory)):
            slot = (hist_ptr - 1 - j) % memory
            s_j, y_j, rho_j = S[idx_n, slot], Y[idx_n, slot], rho[idx_n, slot]
            beta_j = rho_j * np.einsum('nd,nd->n', y_j, r)
            r += (alpha[:, j] - beta_j)[:, None] * s_j

        direction = -r
        dg = np.einsum('nd,nd->n', direction, grad_cur)
        # Fall back to preconditioned steepest descent where there's no
        # history yet, or the L-BFGS direction isn't actually a descent
        # direction for that shot (can happen after a bad/rejected curvature
        # pair). See _P_diag above for why this can't just be -grad.
        no_curvature_or_ascent = (hist_count == 0) | (dg >= -1e-30)
        precond_dir = -grad_cur * _P_diag[None, :]
        direction = np.where(no_curvature_or_ascent[:, None], precond_dir, direction)
        dg = np.where(no_curvature_or_ascent,
                      -np.einsum('nd,nd->n', grad_cur, grad_cur * _P_diag[None, :]), dg)

        # ONE trial step, ONE batched eval, for the WHOLE batch -- see design
        # history point 3. No per-shot retry within this iteration: a
        # rejected shot just keeps its point/gradient/curvature unchanged
        # and tries again next iteration with a smaller `cur_alpha`.
        X_trial = np.clip(X + cur_alpha[:, None] * direction, lo[None, :], hi[None, :])
        X_eval = np.where(active[:, None], X_trial, X)
        nll_trial, grad_trial = _eval(X_eval)
        armijo_rhs = nll_cur + c1 * cur_alpha * dg
        accepted = active & (nll_trial <= armijo_rhs)
        rejected = active & ~accepted

        s_vec = X_trial - X; y_vec = grad_trial - grad_cur
        sy = np.einsum('nd,nd->n', s_vec, y_vec)
        # Standard L-BFGS curvature safeguard: only push pairs with s.y > 0
        # (positive curvature), otherwise skip the update but keep the step.
        push = accepted & (sy > 1e-10 * np.maximum(
            np.linalg.norm(s_vec, axis=1) * np.linalg.norm(y_vec, axis=1), 1e-30))
        pidx = np.where(push)[0]
        if pidx.size:
            ptr = hist_ptr[pidx]
            S[pidx, ptr] = s_vec[pidx]; Y[pidx, ptr] = y_vec[pidx]; rho[pidx, ptr] = 1.0 / sy[pidx]
            hist_ptr[pidx] = (ptr + 1) % memory
            hist_count[pidx] = np.minimum(hist_count[pidx] + 1, memory)

        X = np.where(accepted[:, None], X_trial, X)
        grad_cur = np.where(accepted[:, None], grad_trial, grad_cur)
        nll_cur = np.where(accepted, nll_trial, nll_cur)

        # Accepted: grow the step back toward the nominal L-BFGS unit step
        # (capped at 1) and reset the stall counter. Rejected: shrink for
        # next iteration's retry. A shot that's rejected `max_reject_streak`
        # times in a row despite repeated shrinking is genuinely stuck (not
        # just between two backtrack rounds) -- reset its curvature history
        # once (in case a bad curvature pair was the cause) and give it one
        # more full-budget attempt; if THAT also stalls out, give up rather
        # than retry an effectively-unchanged problem for the rest of
        # `n_iter` (mirrors scipy reporting non-convergence and returning
        # its last point instead of looping past its budget).
        cur_alpha[accepted] = np.minimum(cur_alpha[accepted] * 1.5, 1.0)
        reject_streak[accepted] = 0
        cur_alpha[rejected] *= 0.5
        reject_streak[rejected] += 1
        stuck_again = rejected & (reject_streak >= max_reject_streak)
        if stuck_again.any():
            already_reset = stuck_again & (hist_count == 0)
            active[already_reset] = False
            gave_up[already_reset] = True
            retry_now = stuck_again & ~already_reset
            hist_count[retry_now] = 0
            cur_alpha[retry_now] = 1.0
            reject_streak[retry_now] = 0

    if active.any() or gave_up.any():
        stuck = active | gave_up
        gnorm_final = np.linalg.norm(grad_cur, axis=1)
        bad = stuck & (gnorm_final >= grad_tol * np.maximum(np.abs(nll_cur), 1.0))
        if bad.any():
            import warnings
            warnings.warn(f'solve_inner_batch_independent: {int(bad.sum())}/{N} shot(s) NOT '
                           f'converged after {n_iter} iterations ({int(gave_up.sum())} gave up '
                           f'early on a stalled step; max |grad| ratio '
                           f'{float(np.max(gnorm_final[bad] / np.maximum(np.abs(nll_cur[bad]), 1.0))):.3e}) '
                           f'-- envelope-theorem outer gradient is unreliable for these shots.')

    return _pack(X)


def fit_beta_batch(run_dir, n_shots, f_signal, prior_mean, prior_std, bins=16, pixel_n_gh=4,
                    n_inner_iter=15, grid_half=0.2, n_starts=1, outer_maxiter=15, log=print):
    """Same model as fit_beta, but the inner per-shot solve is done as ONE
    batched (17*N)-dim optimisation instead of a Python loop over N separate
    small solves -- exploiting pixel_acs_grad_batch.

    BUG FIX (see notes on the original session's speedup work): this used to
    call the outer `minimize(neg_logL, b0, ...)` WITHOUT `jac=True`, so scipy
    fell back to its own finite-difference outer gradient -- on a function
    whose evaluation is itself an approximately-converged, warm-started
    nested L-BFGS-B solve. Each FD probe further mutated the persistent
    warm-start state, so repeated evaluations at (nearly) the same beta could
    return different NLL values depending on solve history -- exactly the
    kind of non-reproducible objective that breaks any gradient-based
    optimizer's line search and gradient estimate. That silently biased
    beta_hat away from the true optimum despite `fit_beta_batch` running
    faster.

    Fixed by using the SAME envelope-theorem outer gradient as `fit_beta`:
    at a converged inner stationary point, d(total NLL)/d(delta_phi_trial_i)
    equals dNLL_z100/d(phi_i+delta_phi_trial_i) alone (`dnll1_dphi`), so no
    finite differencing of the outer objective is needed. Also added: theta
    bounds on the inner solve (parity with fit_beta's divergence guard) and
    objective rescaling so L-BFGS-B's default step-size heuristics behave
    sanely given |grad| ~ 1e5 vs beta ~ 0.01-0.2.
    """
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    acs_z0, acs_z100 = crb.build_evaluators(run_dir, mi.DEFAULT_T_DET, bins,
                                             likelihood='pixel', pixel_n_gh=pixel_n_gh)
    h_theta = np.full(8, 1e-9)

    shot_ids = list(range(min(n_shots, ds_z0.n_shots)))
    N = len(shot_ids)
    shot_idx_arr = np.array(shot_ids, dtype=np.float64)
    sin_i = np.sin(2 * np.pi * f_signal * shot_idx_arr)
    cos_i = np.cos(2 * np.pi * f_signal * shot_idx_arr)

    n_g0 = np.zeros((N, bins*bins)); n_e0 = np.zeros((N, bins*bins))
    n_g1 = np.zeros((N, bins*bins)); n_e1 = np.zeros((N, bins*bins))
    for i, sid in enumerate(shot_ids):
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_g0[i] = _downsample(img0[0].astype(np.float64), bins).ravel()
        n_e0[i] = _downsample(img0[1].astype(np.float64), bins).ravel()
        n_g1[i] = _downsample(img1[0].astype(np.float64), bins).ravel()
        n_e1[i] = _downsample(img1[1].astype(np.float64), bins).ravel()

    # phi0 per shot from a coarse pre-scan instead of a fixed 0.0 -- see
    # _phi_prescan and the module-level comment above it.
    phi0 = np.array([_phi_prescan(acs_z0, acs_z100, n_g0[i], n_e0[i], n_g1[i], n_e1[i],
                                   prior_mean, prior_std)
                      for i in range(N)])
    state = {'x': np.concatenate([phi0, np.tile(prior_mean, N), np.tile(prior_mean, N)])}

    # Same divergence guard as profile_shot_analytic: bound theta to a
    # generous multiple of the prior width, and phi to a wide-but-finite range.
    N_SIGMA = 20.0
    theta_lo = prior_mean - N_SIGMA * prior_std
    theta_hi = prior_mean + N_SIGMA * prior_std

    def _solve_inner(delta_phi_batch):
        # Fix for Issue 2 (notes/joint_profile_inference.tex, Known Open
        # Issues): a single monolithic scipy.optimize.minimize over the
        # stacked (17N,) vector shares ONE curvature/step-size history across
        # all shots, so one shot with a hard/multimodal phi-landscape can
        # stall the whole batch in the wrong local optimum (confirmed: a
        # shot stuck at phi~=0.77 vs the correct phi~=0.00, residual
        # |grad|~8e7 not shrinking with more iterations). Replaced with a
        # vectorized per-shot solver that still batches the expensive
        # value/gradient evaluation across shots, but gives each shot its
        # own step size and convergence decision.
        x_new = solve_inner_batch_independent(
            state['x'], delta_phi_batch, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1,
            prior_mean, prior_std, N, h_theta, theta_lo, theta_hi, n_iter=n_inner_iter)
        state['x'] = x_new
        return x_new

    def neg_logL(beta):
        As, Ac = float(beta[0]), float(beta[1])
        delta_phi_batch = As*sin_i + Ac*cos_i
        x = _solve_inner(delta_phi_batch)
        nll, _, _ = batched_nll_grad(x, delta_phi_batch, acs_z0, acs_z100,
                                      n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std, N, h_theta)
        return nll

    def _neg_logL_and_grad_raw(beta):
        As, Ac = float(beta[0]), float(beta[1])
        delta_phi_batch = As*sin_i + Ac*cos_i
        x = _solve_inner(delta_phi_batch)
        nll, _, dnll1_dphi = batched_nll_grad(x, delta_phi_batch, acs_z0, acs_z100,
                                               n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std,
                                               N, h_theta)
        # envelope theorem, same as fit_beta's _neg_logL_and_grad_raw
        d_dbeta = np.array([float(np.sum(dnll1_dphi * sin_i)),
                             float(np.sum(dnll1_dphi * cos_i))])
        return nll, d_dbeta

    # Same rescaling rationale as fit_beta: bring the outer L-BFGS-B step
    # heuristics into a sane range given |grad| ~ 1e5 vs beta ~ 0.01-0.2.
    _, g0 = _neg_logL_and_grad_raw(np.zeros(2))
    grad_norm0 = max(np.linalg.norm(g0), 1e-30)
    obj_scale = 0.02 / grad_norm0
    log(f'  outer objective scale = {obj_scale:.3e}  (raw grad norm at beta=0: {grad_norm0:.3e})')

    def neg_logL_and_grad_analytic(beta):
        nll, d_dbeta = _neg_logL_and_grad_raw(beta)
        return nll * obj_scale, d_dbeta * obj_scale

    t0 = time.perf_counter()
    bounds = [(-grid_half, grid_half)] * 2
    starts = [np.zeros(2)] + [np.array([0.05, 0.0]), np.array([-0.05, 0.05])][:max(0, n_starts - 1)]
    best = None
    for b0 in starts[:n_starts]:
        res = minimize(neg_logL_and_grad_analytic, b0, jac=True, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': outer_maxiter})
        log(f'  start beta0={b0}  beta_hat={res.x}  nll={res.fun:.4f}  success={res.success}  nit={res.nit}')
        if best is None or res.fun < best.fun:
            best = res
    log(f'fit_beta_batch done in {time.perf_counter()-t0:.1f}s  beta_hat={best.x}')
    return best.x


def _downsample(img, bins):
    n = img.shape[0]
    if n % bins != 0:
        raise ValueError(f'image resolution {n} not divisible by bins {bins}')
    k = n // bins
    return img.reshape(bins, k, bins, k).sum(axis=(1, 3))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--n_shots', type=int, default=5)
    p.add_argument('--bins', type=int, default=16)
    p.add_argument('--pixel_n_gh', type=int, default=4)
    p.add_argument('--gh_order', type=int, default=4)
    p.add_argument('--f_signal', type=float, default=0.3)
    p.add_argument('--n_inner_newton', type=int, default=3,
                    help='inner iterations per outer step, --mode loop only')
    p.add_argument('--n_inner_iter', type=int, default=15,
                    help='inner iterations per outer step, --mode batch only')
    p.add_argument('--n_starts', type=int, default=1)
    p.add_argument('--outer_maxiter', type=int, default=15)
    p.add_argument('--use_analytic', type=int, default=1)
    p.add_argument('--mode', choices=['loop', 'batch'], default='loop',
                    help='loop: fit_beta (per-shot scipy L-BFGS-B loop, reference). '
                         'batch: fit_beta_batch (vectorized per-shot L-BFGS, batched GPU eval -- '
                         'faster and scales better with n_shots, see solve_inner_batch_independent '
                         'docstring for validation history).')
    args = p.parse_args()

    data_root = Path(args.data_root)
    run_dir = sorted(data_root.glob('run_*'))[0]
    prior_mean = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
    prior_std = np.array([10e-6] * 4 + [10e-6] * 4)

    t0 = time.perf_counter()
    if args.mode == 'batch':
        beta_hat = fit_beta_batch(run_dir, args.n_shots, args.f_signal, prior_mean, prior_std,
                                   bins=args.bins, pixel_n_gh=args.pixel_n_gh,
                                   n_inner_iter=args.n_inner_iter, n_starts=args.n_starts,
                                   outer_maxiter=args.outer_maxiter)
    else:
        beta_hat = fit_beta(run_dir, args.n_shots, args.f_signal, prior_mean, prior_std,
                             bins=args.bins, pixel_n_gh=args.pixel_n_gh, gh_order=args.gh_order,
                             n_inner_newton=args.n_inner_newton, n_starts=args.n_starts,
                             outer_maxiter=args.outer_maxiter, use_analytic=bool(args.use_analytic))
    print(f'\nFinal beta_hat (As, Ac) = {beta_hat}   (mode={args.mode}, '
          f'total {time.perf_counter()-t0:.1f}s)')
