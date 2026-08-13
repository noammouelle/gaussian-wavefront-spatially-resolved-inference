"""
crb_signal.py — Cramér-Rao bound for the gradiometer signal parameters (A_s, A_c).

Computes the theoretical lower bound on Var(A_s, A_c) given the PSMAPs, cloud
priors, and pixel/bin geometry, accounting for the two per-shot nuisances:
the common laser phase phi_i (fully unknown, uniform prior) and the two
independent cloud-state vectors theta_z0, theta_z100 (Gaussian priors from
the dataset generation hyperparameters).

Mathematical structure (see notes/crb_signal.md for the full derivation)
--------------------------------------------------------------------------
For a single AI z, the port-summed Poisson likelihood reduces exactly to a
Binomial(n_tot,z, f_z) likelihood in the port fraction f_z = I_g/I_tot
(derived and empirically validated earlier in this project). For ANY pair of
parameters that affect the model only through f_z, the Fisher information is
the rank-1 outer product

    I_z(zeta_a, zeta_b) = [n_tot,z / (f_z (1-f_z))] * (df_z/d zeta_a)(df_z/d zeta_b)

Since z0 depends only on (phi_i, theta_z0) and z100 only on (phi_i, theta_z100)
(with phi_i shared and delta_phi_i(beta) = A_s sin(2*pi*f*t_i) + A_c cos(...)
added to z100's argument), the *joint* per-shot Fisher matrix over the 17
nuisance-plus-signal-relevant directions (phi_i, theta_z0 [8], theta_z100 [8])
is the SUM of two independent rank-1 pieces from z0 and z100. This is exact
given the port-summed reduction (not an extra approximation) and is cheap to
build from finite differences of the already-validated `batch_acs_gh` port
sums -- no new per-pixel machinery is needed.

A Gaussian prior on theta_z0, theta_z100 (matching the dataset generation
hyperparameters) is added to the corresponding diagonal blocks (a van Trees
/ Bayesian treatment, NOT a flat-prior nuisance elimination), then the 16
theta-directions are eliminated via a Schur complement (linear solve, no
explicit matrix inverse) to obtain the effective per-shot phase information
j_i = j_phi_eff. Because A_s, A_c only enter through the same phi-shift
mechanism at z100 (delta_phi_i is a *known* linear function of (A_s,A_c) via
sin/cos(2*pi*f*t_i)), the N-shot Fisher matrix for beta=(A_s,A_c) is then
exactly

    F_beta = sum_i j_i * x_i x_i^T,   x_i = [sin(2*pi*f*t_i), cos(2*pi*f*t_i)]

and CRB(beta) = F_beta^{-1}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import map_inference as mi                                      # noqa: E402
from helpers import ImageShotDataset                             # noqa: E402

try:
    from aispy.psmap import load_psmap, PSMAPSurrogate
except ImportError:
    sys.path.insert(0, str(REPO.parents[1] / 'local' / 'aispy'))
    from aispy.psmap import load_psmap, PSMAPSurrogate
from profile_cloud_nuisances import SurrogatePixelACS, SemiAnalyticPixelACS  # noqa: E402

THETA_NAMES = ('mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0', 'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy')
EPS = 1e-300


# ── theta -> port-summed (A,Cc,Cs) for both states, via the validated GH path ──

def theta_to_eta(theta):
    """[mu_x0,mu_y0,mu_vx0,mu_vy0,sx,sy,svx,svy] -> mi's 10-elem eta ordering."""
    mu_x0, mu_y0, mu_vx0, mu_vy0, sx, sy, svx, svy = theta
    return np.array([mu_x0, mu_vx0, mu_y0, mu_vy0, sx**2, 0.0, svx**2, sy**2, 0.0, svy**2])


def port_sums(theta, acs_obj, gh_order):
    """theta (8,) -> (A_g,Cc_g,Cs_g,A_e,Cc_e,Cs_e) port-summed scalars."""
    eta = theta_to_eta(theta)
    acs = mi.batch_acs_gh([eta], acs_obj, gh_order)[0]   # (6,)
    return acs   # A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e


def f_and_grad(theta, phi, acs_obj, gh_order, h_theta):
    """
    Port fraction f(phi;theta) = I_g/I_tot and its gradient w.r.t. (phi, theta[8]).

    Returns f, df_dphi, df_dtheta (8,), and n_tot-normalising ACS at theta.
    """
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = port_sums(theta, acs_obj, gh_order)
    A_tot, Cc_tot, Cs_tot = A_g + A_e, Cc_g + Cc_e, Cs_g + Cs_e

    c, s = np.cos(phi), np.sin(phi)
    I_g = A_g + Cc_g * c + Cs_g * s
    I_tot = A_tot + Cc_tot * c + Cs_tot * s
    f = I_g / max(I_tot, EPS)

    I_g_phi = -Cc_g * s + Cs_g * c
    I_tot_phi = -Cc_tot * s + Cs_tot * c
    df_dphi = (I_g_phi * I_tot - I_g * I_tot_phi) / max(I_tot, EPS) ** 2

    df_dtheta = np.zeros(8)
    for k in range(8):
        tp = theta.copy(); tp[k] += h_theta[k]
        tm = theta.copy(); tm[k] -= h_theta[k]
        Agp, Ccgp, Csgp, Aep, Ccep, Csep = port_sums(tp, acs_obj, gh_order)
        Agm, Ccgm, Csgm, Aem, Ccem, Csem = port_sums(tm, acs_obj, gh_order)
        Itot_p = (Agp + Aep) + (Ccgp + Ccep) * c + (Csgp + Csep) * s
        Itot_m = (Agm + Aem) + (Ccgm + Ccem) * c + (Csgm + Csem) * s
        Ig_p = Agp + Ccgp * c + Csgp * s
        Ig_m = Agm + Ccgm * c + Csgm * s
        fp = Ig_p / max(Itot_p, EPS)
        fm = Ig_m / max(Itot_m, EPS)
        df_dtheta[k] = (fp - fm) / (2 * h_theta[k])

    return f, df_dphi, df_dtheta, (A_g, A_e, A_tot)


# ── pixel-resolved alternative: full per-pixel Poisson Fisher block ─────────
#
# Unlike the port-summed reduction (rank-1 per AI, exact given the binomial
# structure), the pixel-resolved likelihood treats every (pixel, state) as an
# independent Poisson observation, so the (phi, theta[8]) Fisher block is a
# general rank-<=9 9x9 matrix per AI, built from the true per-pixel Jacobian.
# This is what actually makes the pixel/bin count matter.

def pixel_stats(theta, acs_obj):
    """theta (8,) -> (A_g,Cc_g,Cs_g,A_e,Cc_e,Cs_e), each (n_bins,) numpy arrays."""
    out = acs_obj.pixel_acs(theta)
    return tuple(np.asarray(o.get() if hasattr(o, 'get') else o) for o in out)


def pixel_fisher_block(theta, phi, acs_obj, n_tot, h_theta):
    """
    Full per-pixel Poisson Fisher block for one AI, over (phi, theta[8]).

    L (atom-number scale) is computed once at the baseline theta and held
    fixed while differentiating w.r.t. theta -- same convention as the
    L_iz-fixed treatment already used throughout this project (Eq. Lfix in
    notes/image_likelihood.tex; log_f_ai's L_phi in map_inference.py).

    Returns H (9,9), and (f-analog) diagnostics.
    """
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = pixel_stats(theta, acs_obj)
    c, s = np.cos(phi), np.sin(phi)
    I_g = A_g + Cc_g * c + Cs_g * s
    I_e = A_e + Cc_e * c + Cs_e * s
    I_tot_sum = float((I_g + I_e).sum())
    L = n_tot / max(I_tot_sum, EPS)

    mu_g = L * np.maximum(I_g, EPS)
    mu_e = L * np.maximum(I_e, EPS)

    dmu_g_dphi = L * (-Cc_g * s + Cs_g * c)
    dmu_e_dphi = L * (-Cc_e * s + Cs_e * c)

    n_bins = A_g.shape[0]
    J_g = np.zeros((n_bins, 9)); J_e = np.zeros((n_bins, 9))
    J_g[:, 0] = dmu_g_dphi; J_e[:, 0] = dmu_e_dphi

    for k in range(8):
        tp = theta.copy(); tp[k] += h_theta[k]
        tm = theta.copy(); tm[k] -= h_theta[k]
        Agp, Ccgp, Csgp, Aep, Ccep, Csep = pixel_stats(tp, acs_obj)
        Agm, Ccgm, Csgm, Aem, Ccem, Csem = pixel_stats(tm, acs_obj)
        Igp = Agp + Ccgp * c + Csgp * s;  Igm = Agm + Ccgm * c + Csgm * s
        Iep = Aep + Ccep * c + Csep * s;  Iem = Aem + Ccem * c + Csem * s
        J_g[:, 1 + k] = L * (Igp - Igm) / (2 * h_theta[k])
        J_e[:, 1 + k] = L * (Iep - Iem) / (2 * h_theta[k])

    Wg = 1.0 / np.maximum(mu_g, EPS)
    We = 1.0 / np.maximum(mu_e, EPS)
    H = (J_g * Wg[:, None]).T @ J_g + (J_e * We[:, None]).T @ J_e
    return H, {'mu_g_min': float(mu_g.min()), 'mu_e_min': float(mu_e.min()), 'L': float(L)}


# ── per-shot effective phase information (Schur complement over theta) ──────

def theta_prior_precision(prior_std):
    """prior_std: (8,) Gaussian prior std per theta component -> diagonal precision."""
    return 1.0 / np.asarray(prior_std) ** 2


def build_joint_fisher(theta_z0, theta_z100, phi_z0_true, phi_z100_true, n_tot0, n_tot1,
                        acs_z0, acs_z100, gh_order, prior_std, h_theta, likelihood='port'):
    """
    Full per-shot joint Fisher matrix H (17,17) over (phi, theta_z0[8], theta_z100[8]),
    WITH the theta prior precision already added (van Trees), WITHOUT any nuisance
    elimination. This is the single object everything else (beta-only CRB via Schur
    complement, or full nuisance covariance via direct inversion) is built from.

    likelihood : 'port' (rank-1-per-AI, port-summed, bin-count independent)
                 or 'pixel' (full per-pixel Poisson Fisher, bin-count matters)

    phi_z0_true, phi_z100_true : the true ARGUMENT each AI's model is
        evaluated at -- Z0 sees phi_i directly, Z100 sees phi_i + delta_phi_i.
        d(phi_i+delta_phi_i)/d(phi_i) = 1, so both AIs' d/dphi contributions
        combine directly once each is evaluated at its own correct argument.
    """
    H = np.zeros((17, 17))
    diag_extra = {}

    if likelihood == 'port':
        f0, df0_dphi, df0_dtheta, _ = f_and_grad(theta_z0, phi_z0_true, acs_z0, gh_order, h_theta)
        f1, df1_dphi, df1_dtheta, _ = f_and_grad(theta_z100, phi_z100_true, acs_z100, gh_order, h_theta)
        w0 = n_tot0 / max(f0 * (1 - f0), EPS)
        w1 = n_tot1 / max(f1 * (1 - f1), EPS)
        g0 = np.zeros(17); g0[0] = df0_dphi; g0[1:9] = df0_dtheta
        g1 = np.zeros(17); g1[0] = df1_dphi; g1[9:17] = df1_dtheta
        H = w0 * np.outer(g0, g0) + w1 * np.outer(g1, g1)   # rank <= 2
        diag_extra = {'f0': f0, 'f1': f1}
    elif likelihood == 'pixel':
        H0, d0 = pixel_fisher_block(theta_z0, phi_z0_true, acs_z0, n_tot0, h_theta)
        H1, d1 = pixel_fisher_block(theta_z100, phi_z100_true, acs_z100, n_tot1, h_theta)
        H[0, 0] += H0[0, 0] + H1[0, 0]
        H[0, 1:9] += H0[0, 1:]; H[1:9, 0] += H0[1:, 0]
        H[1:9, 1:9] += H0[1:, 1:]
        H[0, 9:17] += H1[0, 1:]; H[9:17, 0] += H1[1:, 0]
        H[9:17, 9:17] += H1[1:, 1:]
        diag_extra = {'pixel_z0': d0, 'pixel_z100': d1}
    else:
        raise ValueError(f'unknown likelihood {likelihood!r}')

    prior_prec = theta_prior_precision(prior_std)
    H[1:9, 1:9] += np.diag(prior_prec)
    H[9:17, 9:17] += np.diag(prior_prec)
    return H, diag_extra


def per_shot_information(theta_z0, theta_z100, phi_z0_true, phi_z100_true, n_tot0, n_tot1,
                          acs_z0, acs_z100, gh_order, prior_std, h_theta,
                          likelihood='port', return_diagnostics=False):
    """Returns j_phi_eff (scalar effective per-shot phase info, theta eliminated via
    Schur complement), and optionally diagnostics. See build_joint_fisher for H."""
    H, diag_extra = build_joint_fisher(theta_z0, theta_z100, phi_z0_true, phi_z100_true,
                                        n_tot0, n_tot1, acs_z0, acs_z100, gh_order,
                                        prior_std, h_theta, likelihood)
    H_pp = H[0, 0]
    H_pn = H[0, 1:]
    H_nn = H[1:, 1:]

    x = np.linalg.solve(H_nn, H_pn)      # H_nn^{-1} H_np  (solve, not explicit inverse)
    j_eff = max(H_pp - H_pn @ x, 0.0)

    if not return_diagnostics:
        return j_eff

    eigvals = np.linalg.eigvalsh(H_nn)
    diag = {
        'H_nn_eigvals': eigvals,
        'H_nn_cond': float(eigvals[-1] / max(eigvals[0], 1e-300)),
        'H_nn_min_eig': float(eigvals[0]),
        'j_eff': j_eff,
        'H_pp_no_prior_no_theta': float(H_pp),   # = oracle-limit info (theta known exactly)
        **diag_extra,
    }
    return j_eff, diag


def full_per_shot_covariance(H):
    """
    Full (17,17) covariance from direct inversion of the joint per-shot Fisher
    matrix H (phi, theta_z0[8], theta_z100[8]) -- no nuisance elimination, just
    the exact joint inverse. Useful as a theoretical floor for any *other*
    estimator of the nuisances (e.g. a learned image->theta predictor): no
    unbiased estimator using this image model can beat these numbers.

    Returns a dict with the physically meaningful sub-blocks.
    """
    cov = np.linalg.inv(H)
    return {
        'cov_full': cov,                      # (17,17), ordering: [phi, theta_z0, theta_z100]
        'var_phi': float(cov[0, 0]),
        'cov_theta_z0': cov[1:9, 1:9],         # (8,8)
        'cov_theta_z100': cov[9:17, 9:17],     # (8,8)
        'cov_theta_z0_z100': cov[1:9, 9:17],   # (8,8) cross-covariance between the two AIs' clouds
        'cov_phi_theta_z0': cov[0, 1:9],       # (8,) cross-covariance phi <-> theta_z0
        'cov_phi_theta_z100': cov[0, 9:17],
        'sigma_theta_z0': np.sqrt(np.diag(cov[1:9, 1:9])),     # (8,) marginal std per component
        'sigma_theta_z100': np.sqrt(np.diag(cov[9:17, 9:17])),
    }


# ── N-shot CRB assembly ──────────────────────────────────────────────────────

def crb_from_j(j_values, shot_idx, f_signal):
    x = np.stack([np.sin(2*np.pi*f_signal*shot_idx), np.cos(2*np.pi*f_signal*shot_idx)], axis=1)
    F = np.zeros((2, 2))
    for j, xi in zip(j_values, x):
        F += j * np.outer(xi, xi)
    cov = np.linalg.inv(F)
    return F, cov


def simple_scaling_crb(j_bar, n_shots):
    """Asymptotic scaling assuming shot times densely sample many signal periods,
    so <x x^T> -> 0.5*I_2 and F_beta ~ (n_shots * j_bar / 2) * I_2."""
    sigma = np.sqrt(2.0 / (n_shots * j_bar))
    return sigma, sigma


# ── driver: average over real shots from a generated dataset ────────────────

def build_evaluators(data_dir, t_det, bins, likelihood='port', pixel_n_gh=8, psmap_tag='CONFOCAL_FINE'):
    ds_tmp = ImageShotDataset(str(data_dir / 'Z0' / 'data_IMG.h5'))
    edges = np.linspace(-ds_tmp.half_range, ds_tmp.half_range, bins + 1)
    psmap_z0 = load_psmap(str(REPO / 'output-files' / f'PSGRID4D_{psmap_tag}_Z0.h5'))
    psmap_z100 = load_psmap(str(REPO / 'output-files' / f'PSGRID4D_{psmap_tag}_Z100.h5'))
    sur_z0 = PSMAPSurrogate(psmap_z0, t_det, use_gpu=mi.USE_GPU)
    sur_z100 = PSMAPSurrogate(psmap_z100, t_det, use_gpu=mi.USE_GPU)
    # n_quad=1 is fine: batch_acs_gh (port path) doesn't touch the QMC samples,
    # and SemiAnalyticPixelACS (pixel path) doesn't either -- both only borrow
    # _eval_fast/_port_inter/s0_g/s1_g/edges from this base object.
    base_z0 = SurrogatePixelACS(sur_z0, t_det, edges, edges, n_quad=1)
    base_z100 = SurrogatePixelACS(sur_z100, t_det, edges, edges, n_quad=1)
    if likelihood == 'pixel':
        acs_z0 = SemiAnalyticPixelACS(base_z0, n_gh=pixel_n_gh)
        acs_z100 = SemiAnalyticPixelACS(base_z100, n_gh=pixel_n_gh)
    else:
        acs_z0, acs_z100 = base_z0, base_z100
    return acs_z0, acs_z100


def run(data_root, t_det, bins, gh_order, prior_std, n_shots_eval, f_signal,
        likelihood='port', pixel_n_gh=8, out_json=None, psmap_tag='CONFOCAL_FINE'):
    data_root = Path(data_root)
    run_dir = sorted(data_root.glob('run_*'))[0]
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))

    acs_z0, acs_z100 = build_evaluators(run_dir, t_det, bins, likelihood, pixel_n_gh, psmap_tag)

    h_theta = np.array([1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7])  # SI units

    j_values, diags = [], []
    shot_ids = list(range(min(n_shots_eval, ds_z0.n_shots)))
    for sid in shot_ids:
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_tot0 = float(img0[0].sum() + img0[1].sum())
        n_tot1 = float(img1[0].sum() + img1[1].sum())
        meta0 = ds_z0.meta(sid); meta1 = ds_z100.meta(sid)
        theta_z0 = np.array([meta0[k] for k in THETA_NAMES])
        theta_z100 = np.array([meta1[k] for k in THETA_NAMES])
        phi_z0_true = float(meta0['phi0'])
        phi_z100_true = float(meta1['phi0'])   # already includes delta_phi_true (verified against raw data)

        j_eff, diag = per_shot_information(theta_z0, theta_z100, phi_z0_true, phi_z100_true,
                                            n_tot0, n_tot1, acs_z0, acs_z100, gh_order, prior_std,
                                            h_theta, likelihood=likelihood, return_diagnostics=True)
        j_values.append(j_eff)
        diags.append(diag)
        print(f'shot {sid:3d}  j_eff={j_eff:.5e}  j_oracle={diag["H_pp_no_prior_no_theta"]:.5e}  '
              f'H_nn_cond={diag["H_nn_cond"]:.3e}', flush=True)

    j_values = np.array(j_values)
    j_bar = j_values.mean()
    shot_idx_arr = np.array(shot_ids, dtype=np.float64)
    F, cov = crb_from_j(j_values, shot_idx_arr, f_signal)
    sigma_As, sigma_Ac = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    rho = cov[0, 1] / (sigma_As * sigma_Ac)
    sigma_simple, _ = simple_scaling_crb(j_bar, len(shot_ids))

    result = dict(j_bar=float(j_bar), j_std=float(j_values.std()),
                  sigma_As=float(sigma_As), sigma_Ac=float(sigma_Ac), rho=float(rho),
                  sigma_simple_scaling=float(sigma_simple), n_shots=len(shot_ids))
    print('\n=== CRB summary ===')
    for k, v in result.items():
        print(f'  {k} = {v}')

    if out_json:
        with open(out_json, 'w') as f:
            json.dump(dict(result=result, j_values=j_values.tolist()), f, indent=2)
        print(f'Saved -> {out_json}')

    return result, j_values, diags


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--t_det', type=float, default=mi.DEFAULT_T_DET)
    p.add_argument('--bins', type=int, default=mi.DEFAULT_BINS)
    p.add_argument('--gh_order', type=int, default=8)
    p.add_argument('--likelihood', type=str, default='port', choices=['port', 'pixel'])
    p.add_argument('--pixel_n_gh', type=int, default=8)
    p.add_argument('--n_shots_eval', type=int, default=20)
    p.add_argument('--f_signal', type=float, default=0.3)
    p.add_argument('--mu_pos_std', type=float, default=10e-6)
    p.add_argument('--mu_vel_std', type=float, default=10e-6)
    p.add_argument('--sig_pos_std', type=float, default=10e-6)
    p.add_argument('--sig_vel_std', type=float, default=10e-6)
    p.add_argument('--out_json', type=str, default='')
    p.add_argument('--psmap_tag', type=str, default='CONFOCAL_FINE',
                   help="PSMAP tag: reads output-files/PSGRID4D_<tag>_Z{0,100}.h5")
    args = p.parse_args()

    prior_std = np.array([args.mu_pos_std, args.mu_pos_std, args.mu_vel_std, args.mu_vel_std,
                           args.sig_pos_std, args.sig_pos_std, args.sig_vel_std, args.sig_vel_std])

    run(args.data_root, args.t_det, args.bins, args.gh_order, prior_std,
        args.n_shots_eval, args.f_signal, args.likelihood, args.pixel_n_gh,
        args.out_json or None, args.psmap_tag)


if __name__ == '__main__':
    main()
