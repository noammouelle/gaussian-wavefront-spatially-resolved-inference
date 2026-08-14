"""
theta_crb_vs_empirical.py — does a full-covariance Fisher/CRB predict theta's
actual empirical recovery scatter, for the confocal_random_zernike (aberrated
wavefront) dataset?

Background (see notes/theta_fisher_crb.tex for the full write-up): an earlier
attempt at this used only the DIAGONAL of a local finite-difference Hessian at
theta's MAP, and found it 66-220x tighter than the true empirical scatter
across 50 shots (notes/joint_profile_inference.tex, sibling worktree). That
was diagnosed as theta having a weakly-identified "ridge" direction (near-zero
Fisher eigenvalue) whose off-diagonal correlations a diagonal-only Hessian
discards by construction -- but whether the ridge is actually linear/quadratic
near the truth (a prerequisite for ANY local Fisher/CRB approach to be valid)
was flagged as an open question and never checked.

This script:
  1. Checks that prerequisite directly: profiles the real Poisson NLL along
     the smallest-Fisher-eigenvalue direction at the true (phi, theta_z0,
     theta_z100) point, for several real shots, and compares against the
     Fisher-implied quadratic.
  2. If that holds up, builds the FULL (not diagonal) per-shot theta
     covariance via crb_signal.py's build_joint_fisher(likelihood='pixel')
     + full_per_shot_covariance (both already implemented but never
     previously wired together for theta -- full_per_shot_covariance is
     dead code in crb_signal.py as of this writing).
  3. Compares the CRB-predicted theta RMSE against the actual empirical
     RMSE(theta_best - theta_true) from the already-completed portion of
     the confocal_random_zernike run.

Uses the EXACT same pixel-likelihood evaluator geometry (BINS_BEST=32,
TIGHT_HALF_RANGE, PIXEL_NGH=8) as
non_phase_shear/analysis/generate_kinematic_estimates.py's 'best' fit, so the
CRB is evaluated on the identical config that produced the empirical numbers
it's being compared against -- an apples-to-oranges bin/geometry mismatch
would invalidate the comparison before it starts.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import map_inference as mi                                            # noqa: E402
import crb_signal as crb                                               # noqa: E402
from profile_cloud_nuisances import SurrogatePixelACS, SemiAnalyticPixelACS  # noqa: E402
from helpers import ImageShotDataset                                    # noqa: E402

DATASET = ('R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
           'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000_psmapconfocal_random_zernike')
PSMAP_TAG = 'confocal_random_zernike'
LABEL = 'confocal_random_zernike'
N_RUNS_EMP, N_SHOTS_EMP = 20, 200

BINS_BEST = 32
TIGHT_HALF_RANGE = 1.54e-03
PIXEL_NGH = 8

PRIOR_MEAN = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
PRIOR_STD = np.array([10e-6] * 8)
H_THETA = np.array([1e-7] * 8)   # matches crb_signal.py's own convention (its main()'s default)
THETA_NAMES = crb.THETA_NAMES


def build_acs(psmap_tag=PSMAP_TAG):
    """Same evaluator geometry as generate_kinematic_estimates.py's 'best' fit
    (tight half-range, bins=32, pixel_n_gh=8) -- required for the CRB to be
    comparable to the empirical fits it's measured against."""
    psmap_z0 = mi.load_psmap(str(REPO / 'output-files' / f'PSGRID4D_{psmap_tag}_Z0.h5'))
    psmap_z100 = mi.load_psmap(str(REPO / 'output-files' / f'PSGRID4D_{psmap_tag}_Z100.h5'))
    sur_z0 = mi.PSMAPSurrogate(psmap_z0, mi.DEFAULT_T_DET, use_gpu=mi.USE_GPU)
    sur_z100 = mi.PSMAPSurrogate(psmap_z100, mi.DEFAULT_T_DET, use_gpu=mi.USE_GPU)
    edges = np.linspace(-TIGHT_HALF_RANGE, TIGHT_HALF_RANGE, BINS_BEST + 1)
    base_z0 = SurrogatePixelACS(sur_z0, mi.DEFAULT_T_DET, edges, edges, n_quad=1)
    base_z100 = SurrogatePixelACS(sur_z100, mi.DEFAULT_T_DET, edges, edges, n_quad=1)
    acs_z0 = SemiAnalyticPixelACS(base_z0, n_gh=PIXEL_NGH)
    acs_z100 = SemiAnalyticPixelACS(base_z100, n_gh=PIXEL_NGH)
    return acs_z0, acs_z100


def downsample_tight(img, half_range_native, bins, half_range_tight):
    """Identical to generate_kinematic_estimates.py's helper of the same name."""
    res = img.shape[0]
    native_edges = np.linspace(-half_range_native, half_range_native, res + 1)
    native_centers = 0.5 * (native_edges[:-1] + native_edges[1:])
    tight_edges = np.linspace(-half_range_tight, half_range_tight, bins + 1)
    ix = np.clip(np.searchsorted(tight_edges, native_centers) - 1, 0, bins - 1)
    outside = (native_centers < -half_range_tight) | (native_centers > half_range_tight)
    weight = img.astype(np.float64)
    mask = ~np.outer(outside, np.ones(res, dtype=bool)) & ~np.outer(np.ones(res, dtype=bool), outside)
    flat_ix = (ix[:, None] * bins + ix[None, :]).ravel()
    flat_w = np.where(mask, weight, 0.0).ravel()
    out_flat = np.bincount(flat_ix, weights=flat_w, minlength=bins * bins)
    return out_flat.reshape(bins, bins)


def load_shot(ds_z0, ds_z100, sid):
    img0 = ds_z0[sid]; img1 = ds_z100[sid]
    n_g0 = downsample_tight(img0[0].astype(np.float64), ds_z0.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
    n_e0 = downsample_tight(img0[1].astype(np.float64), ds_z0.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
    n_g1 = downsample_tight(img1[0].astype(np.float64), ds_z100.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
    n_e1 = downsample_tight(img1[1].astype(np.float64), ds_z100.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
    meta0 = ds_z0.meta(sid); meta1 = ds_z100.meta(sid)
    theta_z0 = np.array([meta0[k] for k in THETA_NAMES])
    theta_z100 = np.array([meta1[k] for k in THETA_NAMES])
    phi_z0_true = float(meta0['phi0'])
    phi_z100_true = float(meta1['phi0'])
    n_tot0 = float(n_g0.sum() + n_e0.sum())
    n_tot1 = float(n_g1.sum() + n_e1.sum())
    return dict(theta_z0=theta_z0, theta_z100=theta_z100, phi_z0=phi_z0_true, phi_z100=phi_z100_true,
                n_g0=n_g0, n_e0=n_e0, n_g1=n_g1, n_e1=n_e1,
                n_tot0=n_tot0, n_tot1=n_tot1)


def joint_pixel_nll(theta_z0, theta_z100, phi_i, delta_phi, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1):
    """Poisson NLL of the full joint model, as a function of (phi_i, theta_z0,
    theta_z100), matching exactly what build_joint_fisher's 'pixel' likelihood
    differentiates (z0 evaluated at phi_i, z100 at phi_i+delta_phi, delta_phi
    fixed/known from the data -- see build_joint_fisher's docstring)."""
    def _ai_nll(theta, acs_obj, phi_arg, n_g, n_e):
        A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = crb.pixel_stats(theta, acs_obj)
        c, s = np.cos(phi_arg), np.sin(phi_arg)
        I_g = A_g + Cc_g * c + Cs_g * s
        I_e = A_e + Cc_e * c + Cs_e * s
        n_tot = float(n_g.sum() + n_e.sum())
        L = n_tot / max(float((I_g + I_e).sum()), crb.EPS)
        mu_g = L * np.maximum(I_g, crb.EPS); mu_e = L * np.maximum(I_e, crb.EPS)
        return float(np.sum(mu_g - n_g * np.log(mu_g)) + np.sum(mu_e - n_e * np.log(mu_e)))
    nll = _ai_nll(theta_z0, acs_z0, phi_i, n_g0, n_e0) + _ai_nll(theta_z100, acs_z100, phi_i + delta_phi, n_g1, n_e1)
    d0 = (theta_z0 - PRIOR_MEAN) / PRIOR_STD
    d1 = (theta_z100 - PRIOR_MEAN) / PRIOR_STD
    nll += 0.5 * float(np.sum(d0 ** 2)) + 0.5 * float(np.sum(d1 ** 2))
    return nll


def ridge_check(shots, acs_z0, acs_z100, n_probe_sigma=(0.5, 1, 2, 4, 8), verbose=True):
    """Step 1: at the TRUE (phi, theta_z0, theta_z100) point, build the full
    17x17 Fisher matrix, find its smallest-eigenvalue direction (the ridge),
    and compare the real NLL profiled along it to the Fisher-implied
    quadratic. Returns a list of per-shot diagnostic dicts."""
    results = []
    for i, shot in enumerate(shots):
        H, _ = crb.build_joint_fisher(shot['theta_z0'], shot['theta_z100'], shot['phi_z0'], shot['phi_z100'],
                                       shot['n_tot0'], shot['n_tot1'], acs_z0, acs_z100, gh_order=None,
                                       prior_std=PRIOR_STD, h_theta=H_THETA, likelihood='pixel')
        eigvals, eigvecs = np.linalg.eigh(H)
        min_eig = eigvals[0]
        ridge_dir = eigvecs[:, 0]
        sigma_ridge = 1.0 / np.sqrt(max(min_eig, 1e-300))

        x0 = np.concatenate([[shot['phi_z0']], shot['theta_z0'], shot['theta_z100']])
        delta_phi = shot['phi_z100'] - shot['phi_z0']
        nll0 = joint_pixel_nll(shot['theta_z0'], shot['theta_z100'], shot['phi_z0'], delta_phi,
                                acs_z0, acs_z100, shot['n_g0'], shot['n_e0'], shot['n_g1'], shot['n_e1'])

        probe = {}
        for k in n_probe_sigma:
            step = k * sigma_ridge
            xp = x0 + step * ridge_dir
            nll_p = joint_pixel_nll(xp[1:9], xp[9:17], xp[0], delta_phi, acs_z0, acs_z100,
                                     shot['n_g0'], shot['n_e0'], shot['n_g1'], shot['n_e1'])
            quad_p = nll0 + 0.5 * min_eig * step ** 2
            probe[k] = dict(actual=nll_p - nll0, quadratic=0.5 * min_eig * step ** 2,
                             ratio=(nll_p - nll0) / max(0.5 * min_eig * step ** 2, 1e-300))
        results.append(dict(shot=i, min_eig=float(min_eig), sigma_ridge=float(sigma_ridge),
                             cond=float(eigvals[-1] / max(eigvals[0], 1e-300)), probe=probe))
        if verbose:
            print(f'shot {i}: min_eig={min_eig:.3e}  sigma_ridge={sigma_ridge:.3e}  '
                  f'cond={results[-1]["cond"]:.3e}')
            for k, v in probe.items():
                print(f'    {k:>4}sigma: actual_dNLL={v["actual"]:.4f}  quad_dNLL={v["quadratic"]:.4f}  '
                      f'ratio={v["ratio"]:.3f}')
    return results


def theta_crb(shots, acs_z0, acs_z100, verbose=True):
    """Step 2: full-covariance theta CRB at truth, per shot."""
    sigmas_z0, sigmas_z100 = [], []
    for i, shot in enumerate(shots):
        H, _ = crb.build_joint_fisher(shot['theta_z0'], shot['theta_z100'], shot['phi_z0'], shot['phi_z100'],
                                       shot['n_tot0'], shot['n_tot1'], acs_z0, acs_z100, gh_order=None,
                                       prior_std=PRIOR_STD, h_theta=H_THETA, likelihood='pixel')
        cov = crb.full_per_shot_covariance(H)
        sigmas_z0.append(cov['sigma_theta_z0'])
        sigmas_z100.append(cov['sigma_theta_z100'])
        if verbose:
            print(f'shot {i}: sigma_theta_z0={cov["sigma_theta_z0"]}')
    sigmas_z0 = np.array(sigmas_z0); sigmas_z100 = np.array(sigmas_z100)
    # aggregate: root-mean of predicted variances across shots (matches how the
    # empirical RMSE pools -- both are "typical per-shot spread", not a
    # shot-count-improving quantity; see the note's framing section)
    crb_rmse_z0 = np.sqrt((sigmas_z0 ** 2).mean(axis=0))
    crb_rmse_z100 = np.sqrt((sigmas_z100 ** 2).mean(axis=0))
    return crb_rmse_z0, crb_rmse_z100, sigmas_z0, sigmas_z100


def empirical_rmse(label=LABEL, n_runs=N_RUNS_EMP, n_shots=N_SHOTS_EMP, outlier_sigma=20.0, method='best'):
    """Step 3: load whatever portion of the checkpointed kinematic_estimates
    JSON exists, compute empirical RMSE(theta_<method> - theta_true).
    method: 'best' (pixel-likelihood, what the CRB models), 'moments'
    (closed-form Kalman-gain estimator), or 'null' (prior-mean only, no
    image information at all) -- pass 'moments'/'null' to see the CRB
    against the full spectrum of estimators, not just the one it's actually
    modeling.

    Also returns a robust RMSE with per-component outliers excluded (residual
    magnitude > outlier_sigma * MAD-based robust sigma). RMSE is quadratic in
    the residuals, so even a single catastrophic optimizer failure (L-BFGS-B
    landing in a bad basin for one shot) can dominate it -- report both rather
    than silently picking one, since the gap between them is itself
    informative (how much of the raw RMSE is a few bad fits vs. genuine
    typical scatter)."""
    path = REPO / 'non_phase_shear' / 'results' / f'kinematic_estimates_{label}_N{n_runs}_shots{n_shots}.json'
    with open(path) as f:
        data = json.load(f)
    residuals = []
    for run_name, run in data.items():
        for shot in run['shots']:
            true_z0 = np.array(shot['true_theta_z0']); est_z0 = np.array(shot[f'theta_{method}_z0'])
            true_z100 = np.array(shot['true_theta_z100']); est_z100 = np.array(shot[f'theta_{method}_z100'])
            residuals.append(est_z0 - true_z0)
            residuals.append(est_z100 - true_z100)
    residuals = np.array(residuals)
    rmse = np.sqrt((residuals ** 2).mean(axis=0))

    mad = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
    robust_sigma = 1.4826 * mad   # MAD -> sigma for a Gaussian
    is_outlier = np.abs(residuals) > outlier_sigma * robust_sigma[None, :]
    n_outliers = is_outlier.any(axis=1).sum()
    keep = ~is_outlier.any(axis=1)
    robust_rmse = np.sqrt((residuals[keep] ** 2).mean(axis=0))

    return rmse, robust_rmse, int(n_outliers), len(data), residuals.shape[0]


if __name__ == '__main__':
    t0 = time.perf_counter()
    print('Building pixel-likelihood evaluators (bins=32, tight range, pixel_n_gh=8)...')
    acs_z0, acs_z100 = build_acs()

    data_root = REPO / 'data' / DATASET
    run_dir = sorted(data_root.glob('run_*'))[0]
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))

    N_SHOTS_CHECK = 10
    print(f'\nLoading {N_SHOTS_CHECK} real shots from {run_dir.name}...')
    shots = [load_shot(ds_z0, ds_z100, i) for i in range(N_SHOTS_CHECK)]

    print('\n=== Step 1: ridge linearity/curvature check (at truth) ===')
    ridge_results = ridge_check(shots, acs_z0, acs_z100)

    print('\n=== Step 2: full-covariance theta CRB (at truth) ===')
    crb_rmse_z0, crb_rmse_z100, sig_z0, sig_z100 = theta_crb(shots, acs_z0, acs_z100)
    print(f'CRB RMSE (z0):   {dict(zip(THETA_NAMES, crb_rmse_z0))}')
    print(f'CRB RMSE (z100): {dict(zip(THETA_NAMES, crb_rmse_z100))}')

    print('\n=== Step 3: empirical RMSE from the (partial) real run ===')
    emp_rmse, emp_rmse_robust, n_outliers, n_runs_done, n_points = empirical_rmse()
    print(f'{n_runs_done} runs done, {n_points} (shot,z-slice) points, {n_outliers} outlier point(s) excluded '
          f'from the robust RMSE (>20 robust-sigma in any component)')
    print(f'Empirical RMSE (pooled z0+z100):        {dict(zip(THETA_NAMES, emp_rmse))}')
    print(f'Empirical RMSE, outliers excluded:      {dict(zip(THETA_NAMES, emp_rmse_robust))}')

    emp_moments, _, _, _, _ = empirical_rmse(method='moments')
    emp_null, _, _, _, _ = empirical_rmse(method='null')

    print('\n=== Comparison table: CRB (the pixel-likelihood floor) vs. the full estimator spectrum ===')
    crb_avg = 0.5 * (crb_rmse_z0 + crb_rmse_z100)
    for name, c, er, em, en in zip(THETA_NAMES, crb_avg, emp_rmse_robust, emp_moments, emp_null):
        print(f'  {name:10s}  CRB={c:.4e}  best={er:.4e} (x{er / c:.2f})  '
              f'moments={em:.4e} (x{em / c:.2f})  null={en:.4e} (x{en / c:.2f})')

    print(f'\nTotal time: {time.perf_counter() - t0:.1f}s')
