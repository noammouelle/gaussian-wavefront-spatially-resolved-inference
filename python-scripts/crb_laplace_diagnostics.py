"""
crb_laplace_diagnostics.py — validity checks for the Laplace/quadratic
approximation used by crb_signal.py's theta-nuisance Schur complement.

Two checks, both against the *expected* (Asimov) log-posterior so results
reflect model curvature, not a particular noise realisation:

1. Ridge profile scan: walk the true expected log-posterior along the
   smallest- and largest-eigenvalue directions of H_theta (posterior
   precision) out to +/-3 posterior-sigma, and compare against the quadratic
   (Laplace) prediction from the Hessian at the true point.

2. Multistart joint (phi, theta_z0, theta_z100) MAP: many random starts,
   check convergence to a single dominant optimum (same pattern used earlier
   in this project to validate the phi-only Laplace approximation).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import crb_signal as crb                                          # noqa: E402
import map_inference as mi                                         # noqa: E402
from helpers import ImageShotDataset                                # noqa: E402


def asimov_counts(theta_true, phi_true, acs_obj, n_tot):
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = crb.pixel_stats(theta_true, acs_obj)
    c, s = np.cos(phi_true), np.sin(phi_true)
    I_g = A_g + Cc_g * c + Cs_g * s
    I_e = A_e + Cc_e * c + Cs_e * s
    L = n_tot / max(float((I_g + I_e).sum()), crb.EPS)
    n_g = L * np.maximum(I_g, crb.EPS)
    n_e = L * np.maximum(I_e, crb.EPS)
    return n_g, n_e, L


def expected_nll(theta_trial, phi, acs_obj, n_g_asimov, n_e_asimov, L):
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = crb.pixel_stats(theta_trial, acs_obj)
    c, s = np.cos(phi), np.sin(phi)
    mu_g = L * np.maximum(A_g + Cc_g * c + Cs_g * s, crb.EPS)
    mu_e = L * np.maximum(A_e + Cc_e * c + Cs_e * s, crb.EPS)
    return float(np.sum(mu_g - n_g_asimov * np.log(mu_g)) + np.sum(mu_e - n_e_asimov * np.log(mu_e)))


def prior_nll(theta_trial, prior_mean, prior_std):
    return float(0.5 * np.sum(((theta_trial - prior_mean) / prior_std) ** 2))


def ridge_profile_scan(theta_true, phi_true, acs_obj, n_tot, prior_mean, prior_std, h_theta,
                        k_range=(-3, -2, -1, 0, 1, 2, 3)):
    n_g_asimov, n_e_asimov, L = asimov_counts(theta_true, phi_true, acs_obj, n_tot)
    H9, _ = crb.pixel_fisher_block(theta_true, phi_true, acs_obj, n_tot, h_theta)
    H_theta = H9[1:, 1:].copy()
    H_theta_post = H_theta + np.diag(crb.theta_prior_precision(prior_std))

    eigvals, eigvecs = np.linalg.eigh(H_theta_post)
    base = expected_nll(theta_true, phi_true, acs_obj, n_g_asimov, n_e_asimov, L) \
        + prior_nll(theta_true, prior_mean, prior_std)

    out = {}
    for label, idx in [('ridge_min_eig', 0), ('stiff_max_eig', -1)]:
        lam = eigvals[idx]; v = eigvecs[:, idx]
        sigma = 1.0 / np.sqrt(max(lam, 1e-300))

        # theta_true is NOT generally the posterior MAP: the Asimov likelihood
        # gradient vanishes at theta_true by construction, but the PRIOR gradient
        # does not (theta_true is a random draw from the prior, not its mean).
        # So the correct local comparison is the full 2nd-order Taylor expansion
        # (linear + quadratic in k), not a pure quadratic centred at theta_true.
        h_k = 1e-3
        f_p = expected_nll(theta_true + h_k * sigma * v, phi_true, acs_obj, n_g_asimov, n_e_asimov, L) \
            + prior_nll(theta_true + h_k * sigma * v, prior_mean, prior_std)
        f_m = expected_nll(theta_true - h_k * sigma * v, phi_true, acs_obj, n_g_asimov, n_e_asimov, L) \
            + prior_nll(theta_true - h_k * sigma * v, prior_mean, prior_std)
        grad_k = (f_p - f_m) / (2 * h_k)   # d(NLL)/dk at k=0

        true_delta, quad_pred = [], []
        for k in k_range:
            theta_probe = theta_true + k * sigma * v
            nll = expected_nll(theta_probe, phi_true, acs_obj, n_g_asimov, n_e_asimov, L) \
                + prior_nll(theta_probe, prior_mean, prior_std)
            true_delta.append(nll - base)
            quad_pred.append(grad_k * k + 0.5 * k ** 2)
        out[label] = dict(eigval=float(lam), sigma=float(sigma), eigvec=v, grad_k=float(grad_k),
                           k_range=list(k_range), true_delta_nll=true_delta, quad_pred=quad_pred)
    return out


def multistart_joint_map(theta_z0_true, theta_z100_true, phi_z0_true, phi_z100_true,
                          acs_z0, acs_z100, n_tot0, n_tot1, prior_mean, prior_std,
                          n_starts=15, seed=0):
    # NOTE: joint_neg_log_posterior takes a single shared phi_i and adds it directly
    # to BOTH AIs' arguments; to keep the objective consistent with per_shot_information
    # (which evaluates each AI at its own true argument), we optimise over phi_i with
    # Z0 evaluated at phi_i and Z100 evaluated at phi_i + delta_phi_true, where
    # delta_phi_true = phi_z100_true - phi_z0_true is held fixed (it is a property of
    # the trial signal parameters, not something this diagnostic re-fits).
    delta_phi_true = phi_z100_true - phi_z0_true
    n_g0, n_e0, _ = asimov_counts(theta_z0_true, phi_z0_true, acs_z0, n_tot0)
    n_g1, n_e1, _ = asimov_counts(theta_z100_true, phi_z100_true, acs_z100, n_tot1)

    def nll_with_shift(params):
        phi_i = params[0]
        shifted = np.concatenate([[phi_i], params[1:9], params[9:17]])
        # Z0 at phi_i, Z100 at phi_i+delta_phi_true: build directly rather than
        # reusing joint_neg_log_posterior's single-phi convention.
        theta_z0_ = shifted[1:9]; theta_z100_ = shifted[9:17]

        def ai_nll(theta, acs, ng, ne, phi_):
            A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = crb.pixel_stats(theta, acs)
            c, s = np.cos(phi_), np.sin(phi_)
            I_g = A_g + Cc_g * c + Cs_g * s
            I_e = A_e + Cc_e * c + Cs_e * s
            ntot = float(ng.sum() + ne.sum())
            L = ntot / max(float((I_g + I_e).sum()), crb.EPS)
            mu_g = L * np.maximum(I_g, crb.EPS); mu_e = L * np.maximum(I_e, crb.EPS)
            return float(np.sum(mu_g - ng * np.log(mu_g)) + np.sum(mu_e - ne * np.log(mu_e)))

        nll = ai_nll(theta_z0_, acs_z0, n_g0, n_e0, phi_i)
        nll += ai_nll(theta_z100_, acs_z100, n_g1, n_e1, phi_i + delta_phi_true)
        nll += prior_nll(theta_z0_, prior_mean, prior_std) + prior_nll(theta_z100_, prior_mean, prior_std)
        return nll

    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_starts):
        phi0 = rng.uniform(0, 2 * np.pi) if i > 0 else phi_z0_true
        th0_z0 = theta_z0_true + (rng.normal(size=8) * prior_std * 2.0 if i > 0 else 0.0)
        th0_z100 = theta_z100_true + (rng.normal(size=8) * prior_std * 2.0 if i > 0 else 0.0)
        x0 = np.concatenate([[phi0], th0_z0, th0_z100])
        res = minimize(nll_with_shift, x0, method='L-BFGS-B',
                        options={'maxiter': 40, 'eps': 1e-8, 'ftol': 1e-9, 'gtol': 1e-5})
        results.append((res.fun, res.x, res.success))
        print(f'  start {i:2d}  phi0={phi0:.3f}  nll={res.fun:.6f}  success={res.success}', flush=True)
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--bins', type=int, default=32)
    p.add_argument('--pixel_n_gh', type=int, default=8)
    p.add_argument('--shot_id', type=int, default=0)
    p.add_argument('--n_starts', type=int, default=15)
    args = p.parse_args()

    data_root = Path(args.data_root)
    run_dir = sorted(data_root.glob('run_*'))[0]
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    acs_z0, acs_z100 = crb.build_evaluators(run_dir, mi.DEFAULT_T_DET, args.bins,
                                             likelihood='pixel', pixel_n_gh=args.pixel_n_gh)

    img0 = ds_z0[args.shot_id]; img1 = ds_z100[args.shot_id]
    n_tot0 = float(img0[0].sum() + img0[1].sum())
    n_tot1 = float(img1[0].sum() + img1[1].sum())
    meta0 = ds_z0.meta(args.shot_id); meta1 = ds_z100.meta(args.shot_id)
    theta_z0 = np.array([meta0[k] for k in crb.THETA_NAMES])
    theta_z100 = np.array([meta1[k] for k in crb.THETA_NAMES])
    phi_z0_true = float(meta0['phi0'])
    phi_z100_true = float(meta1['phi0'])   # already includes delta_phi_true

    prior_mean = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
    prior_std = np.array([10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6])
    h_theta = np.full(8, 1e-7)

    print(f'=== shot {args.shot_id}: ridge profile scan (Z0) ===')
    scan0 = ridge_profile_scan(theta_z0, phi_z0_true, acs_z0, n_tot0, prior_mean, prior_std, h_theta)
    for label, d in scan0.items():
        print(f'  [{label}] eigval={d["eigval"]:.4e} sigma={d["sigma"]:.4e}')
        print(f'    k       = {d["k_range"]}')
        print(f'    true dNLL = {[f"{v:.4f}" for v in d["true_delta_nll"]]}')
        print(f'    quad pred = {[f"{v:.4f}" for v in d["quad_pred"]]}')

    print(f'\n=== shot {args.shot_id}: ridge profile scan (Z100) ===')
    scan1 = ridge_profile_scan(theta_z100, phi_z100_true, acs_z100, n_tot1, prior_mean, prior_std, h_theta)
    for label, d in scan1.items():
        print(f'  [{label}] eigval={d["eigval"]:.4e} sigma={d["sigma"]:.4e}')
        print(f'    k       = {d["k_range"]}')
        print(f'    true dNLL = {[f"{v:.4f}" for v in d["true_delta_nll"]]}')
        print(f'    quad pred = {[f"{v:.4f}" for v in d["quad_pred"]]}')

    print(f'\n=== shot {args.shot_id}: multistart joint MAP ({args.n_starts} starts) ===')
    ms = multistart_joint_map(theta_z0, theta_z100, phi_z0_true, phi_z100_true, acs_z0, acs_z100,
                               n_tot0, n_tot1, prior_mean, prior_std, n_starts=args.n_starts)
    best = min(ms, key=lambda r: r[0])
    print(f'  best nll = {best[0]:.6f}')
    for fun, x, ok in ms:
        print(f'    delta_from_best={fun-best[0]:+.6f}  phi*={x[0]:.4f}')
