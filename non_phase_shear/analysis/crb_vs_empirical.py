"""
crb_vs_empirical.py -- Cramer-Rao bound (CRB) vs. empirical recovery scatter,
for theta (per-shot kinematic state) AND beta (the shared signal A_s,A_c),
for ANY dataset/label/psmap_tag already run through
generate_kinematic_estimates.py + beta_fits_from_kinematics.py.

Full theory/derivations: notes/theta_fisher_crb.tex. This script is the
runnable, parameterized version of what that note validates -- point it at
a new wavefront/sequence's already-fit results and it reproduces the same
tables for that configuration, not just the three datasets the note used.

What it computes, in one run:
  1. Ridge-linearity check (optional, --skip-ridge-check to skip): profiles
     the real Poisson NLL along the smallest-Fisher-eigenvalue theta
     direction at truth, compares to the Fisher-implied quadratic -- the
     prerequisite for any local Fisher/CRB approach to be valid at all.
  2. Full-covariance theta CRB (crb_signal.py's build_joint_fisher +
     full_per_shot_covariance, likelihood='pixel', evaluated at truth) vs.
     empirical RMSE(theta_<method> - theta_true), method in
     {best, moments, null}.
  3. Closed-form theoretical predictions for the null estimator (trivial --
     the prior width itself) and the moments estimator (linear propagation
     of Poisson pixel-counting noise through map_inference.py's fixed
     Kalman-gain map, plus a no-noise shrinkage-bias check) -- explains
     WHY moments/null under-perform the CRB, not just by how much.
  4. Beta CRB: crb_signal.py's per_shot_information Schur-complements theta
     out of the joint Fisher matrix, leaving effective per-shot phase
     information j_eff; crb_from_j/simple_scaling_crb propagate that to
     sigma(A_s), sigma(A_c) both as a rate-independent rad/sqrt(shot)
     number and at the dataset's actual N -- vs. empirical beta RMSE for
     best/moments/null.

Usage:
  python crb_vs_empirical.py <dataset> <n_runs> <n_shots> \
      --label <label> --psmap_tag <tag> [--n_shots_crb 10] [--skip-ridge-check]

<dataset>/<n_runs>/<n_shots>/--label/--psmap_tag have the exact same
meaning as generate_kinematic_estimates.py's (pass the SAME values used
there -- this script reads that stage's saved JSON for the empirical side).
--n_shots_crb controls how many real shots are sampled for the CRB side
(ridge check + full-covariance theta CRB + beta j_eff estimate) -- cheap
(~1 min for 10 shots on the pixel-likelihood evaluator), does not need to
scale with n_runs/n_shots.

If the dataset's source images are no longer on disk (deleted/regenerated
since the fit was run), falls back to reading true_theta/n_tot from the
saved kinematic_estimates JSON with a random phi substitute (phi is drawn
independently of theta by the generator, so this is a valid approximation
for the CRB's expectation over phi -- but not the exact per-shot phi that
fit actually saw). In that fallback mode the ridge check and the moments
Poisson-noise theory (both of which need the raw pixel images) are skipped
automatically, with a clear note explaining why.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import map_inference as mi                                              # noqa: E402
import crb_signal as crb                                                 # noqa: E402
from profile_cloud_nuisances import SurrogatePixelACS, SemiAnalyticPixelACS  # noqa: E402
from helpers import ImageShotDataset                                     # noqa: E402
from crb_report import print_comparison_table, save_json                 # noqa: E402

BINS_BEST = 32
TIGHT_HALF_RANGE = 1.54e-03
PIXEL_NGH = 8

PRIOR_MEAN = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
PRIOR_STD = np.array([10e-6] * 8)
H_THETA = np.array([1e-7] * 8)
THETA_NAMES = crb.THETA_NAMES

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('dataset', help='dataset directory name under data/ (the run tag), same as '
                                'generate_kinematic_estimates.py was given')
p.add_argument('n_runs', type=int)
p.add_argument('n_shots', type=int)
p.add_argument('--label', default=None, help='short tag matching what generate_kinematic_estimates.py '
                                              'used (default: dataset dir name)')
p.add_argument('--psmap_tag', default='CONFOCAL_FINE',
                help="PSMAP tag: reads output-files/PSGRID4D_<tag>_Z{0,100}.h5, must match the "
                     "--psmap_tag the empirical fit was run with")
p.add_argument('--n_shots_crb', type=int, default=10,
                help='number of real shots to sample for the CRB side (ridge check + theta CRB '
                     '+ beta j_eff) -- does not need to scale with n_runs/n_shots')
p.add_argument('--skip-ridge-check', action='store_true',
                help='skip the (slower) NLL-profiling linearity check -- use once already '
                     'validated for a given wavefront/sequence and just re-checking numbers')
p.add_argument('--f_signal', type=float, default=0.3, help='signal frequency, cycles/shot (must '
                                                             'match generate_data.py --signal_freq)')
p.add_argument('-v', '--verbose', action='store_true')
args = p.parse_args()
args.dataset = args.dataset.rstrip('/')
LABEL = args.label or args.dataset


def build_acs(psmap_tag):
    """Pixel-likelihood evaluator geometry -- MUST match generate_kinematic_estimates.py's
    'best' fit exactly (same BINS_BEST/TIGHT_HALF_RANGE/PIXEL_NGH) for the CRB to be
    comparable to the empirical numbers it's measured against."""
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


def load_shots_from_disk(ds_z0, ds_z100, n_shots):
    shots = []
    for sid in range(n_shots):
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_g0 = downsample_tight(img0[0].astype(np.float64), ds_z0.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        n_e0 = downsample_tight(img0[1].astype(np.float64), ds_z0.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        n_g1 = downsample_tight(img1[0].astype(np.float64), ds_z100.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        n_e1 = downsample_tight(img1[1].astype(np.float64), ds_z100.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        meta0 = ds_z0.meta(sid); meta1 = ds_z100.meta(sid)
        theta_z0 = np.array([meta0[k] for k in THETA_NAMES])
        theta_z100 = np.array([meta1[k] for k in THETA_NAMES])
        shots.append(dict(theta_z0=theta_z0, theta_z100=theta_z100,
                           phi_z0=float(meta0['phi0']), phi_z100=float(meta1['phi0']),
                           n_g0=n_g0, n_e0=n_e0, n_g1=n_g1, n_e1=n_e1,
                           n_tot0=float(n_g0.sum() + n_e0.sum()), n_tot1=float(n_g1.sum() + n_e1.sum()),
                           img0=img0, img1=img1, pixel_centers=ds_z0.pixel_centers))
    return shots


def load_shots_from_json_fallback(label, n_runs, n_shots, n_shots_crb, seed=0):
    """Used only when the source dataset's images are no longer on disk (see module
    docstring). phi is a random substitute -- phi is independent of theta in the
    generator, so this is valid in expectation but not the exact per-shot phi that
    fit actually used. Cannot supply pixel counts/images -- callers must skip the
    ridge check and the moments Poisson-noise theory in this mode."""
    path = OUT / 'results' / f'kinematic_estimates_{label}_N{n_runs}_shots{n_shots}.json'
    with open(path) as f:
        data = __import__('json').load(f)
    rng = np.random.default_rng(seed)
    shots = []
    count = 0
    for run in data.values():
        for shot in run['shots']:
            if count >= n_shots_crb:
                break
            theta_z0 = np.array(shot['true_theta_z0']); theta_z100 = np.array(shot['true_theta_z100'])
            n_tot0 = float(shot['n_g0'] + shot['n_e0']); n_tot1 = float(shot['n_g1'] + shot['n_e1'])
            shots.append(dict(theta_z0=theta_z0, theta_z100=theta_z100,
                               phi_z0=float(rng.uniform(0, 2 * np.pi)), phi_z100=float(rng.uniform(0, 2 * np.pi)),
                               n_tot0=n_tot0, n_tot1=n_tot1,
                               n_g0=None, n_e0=None, n_g1=None, n_e1=None, img0=None, img1=None))
            count += 1
        if count >= n_shots_crb:
            break
    return shots


def joint_pixel_nll(theta_z0, theta_z100, phi_i, delta_phi, acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1):
    """Poisson NLL of the full joint model, matching exactly what build_joint_fisher's
    'pixel' likelihood differentiates."""
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
    """At the TRUE (phi, theta_z0, theta_z100) point, build the full 17x17 Fisher
    matrix, find its smallest-eigenvalue direction (the ridge), and compare the
    real NLL profiled along it to the Fisher-implied quadratic -- the prerequisite
    for any local Fisher/CRB approach to be valid (see notes/theta_fisher_crb.tex
    Sec. 3 for why this matters and what "quadratic" vs. "banana-shaped" would
    each mean)."""
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
            probe[k] = dict(actual=nll_p - nll0, quadratic=0.5 * min_eig * step ** 2,
                             ratio=(nll_p - nll0) / max(0.5 * min_eig * step ** 2, 1e-300))
        results.append(dict(shot=i, min_eig=float(min_eig), sigma_ridge=float(sigma_ridge),
                             cond=float(eigvals[-1] / max(eigvals[0], 1e-300)), probe=probe))
        if verbose:
            print(f'  shot {i}: min_eig={min_eig:.3e}  sigma_ridge={sigma_ridge:.3e}  '
                  f'cond={results[-1]["cond"]:.3e}')
            for k, v in probe.items():
                print(f'      {k:>4}sigma: actual_dNLL={v["actual"]:.4f}  quad_dNLL={v["quadratic"]:.4f}  '
                      f'ratio={v["ratio"]:.3f}')
    return results


def theta_crb(shots, acs_z0, acs_z100, verbose=True):
    sigmas_z0, sigmas_z100 = [], []
    for i, shot in enumerate(shots):
        H, _ = crb.build_joint_fisher(shot['theta_z0'], shot['theta_z100'], shot['phi_z0'], shot['phi_z100'],
                                       shot['n_tot0'], shot['n_tot1'], acs_z0, acs_z100, gh_order=None,
                                       prior_std=PRIOR_STD, h_theta=H_THETA, likelihood='pixel')
        cov = crb.full_per_shot_covariance(H)
        sigmas_z0.append(cov['sigma_theta_z0'])
        sigmas_z100.append(cov['sigma_theta_z100'])
        if verbose:
            print(f'  shot {i}: sigma_theta_z0={cov["sigma_theta_z0"]}')
    sigmas_z0 = np.array(sigmas_z0); sigmas_z100 = np.array(sigmas_z100)
    crb_rmse_z0 = np.sqrt((sigmas_z0 ** 2).mean(axis=0))
    crb_rmse_z100 = np.sqrt((sigmas_z100 ** 2).mean(axis=0))
    return crb_rmse_z0, crb_rmse_z100, sigmas_z0, sigmas_z100


def empirical_theta_rmse(label, n_runs, n_shots, method, outlier_sigma=20.0):
    path = OUT / 'results' / f'kinematic_estimates_{label}_N{n_runs}_shots{n_shots}.json'
    with open(path) as f:
        data = __import__('json').load(f)
    residuals = []
    for run in data.values():
        for shot in run['shots']:
            residuals.append(np.array(shot[f'theta_{method}_z0']) - np.array(shot['true_theta_z0']))
            residuals.append(np.array(shot[f'theta_{method}_z100']) - np.array(shot['true_theta_z100']))
    residuals = np.array(residuals)
    mad = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
    robust_sigma = 1.4826 * mad
    keep = ~(np.abs(residuals) > outlier_sigma * robust_sigma[None, :]).any(axis=1)
    return np.sqrt((residuals[keep] ** 2).mean(axis=0)), len(data), residuals.shape[0], int((~keep).sum())


def null_theoretical_rmse():
    """RMSE(null) IS the prior width -- null always returns the prior mean, using
    no per-shot information at all, so its RMSE is exactly the population spread
    of true theta around the prior mean (see notes/theta_fisher_crb.tex Sec. 7.2
    for the delta-method derivation of the sigma_x/sigma_y rows)."""
    return np.array([mi.TAU_MU_POS, mi.TAU_MU_POS, mi.TAU_MU_VEL, mi.TAU_MU_VEL,
                      mi.SIG_SIG_POS, mi.SIG_SIG_POS, mi.SIG_SIG_VEL, mi.SIG_SIG_VEL])


def raw_moment_cov(px, centers, mu, V):
    N = px.sum()
    d = centers - mu
    mu3 = float((px * d ** 3).sum() / N)
    mu4 = float((px * d ** 4).sum() / N)
    return V / N, mu3 / N, (mu4 - V ** 2) / N


def moments_theta_cov(img, pixel_centers, K):
    """Cov(theta_moments) = K @ Cov(M) @ K.T -- exact, since theta_moments is an
    exact linear function of the raw image moments M (see notes/theta_fisher_crb.tex
    Sec. 7.3 for the full Poisson-thinning derivation)."""
    total = img[0].astype(np.float64) + img[1].astype(np.float64)
    px = total.sum(axis=1); py = total.sum(axis=0)
    N = total.sum()
    mu_x = float((px * pixel_centers).sum() / N); mu_y = float((py * pixel_centers).sum() / N)
    V_x = float((px * (pixel_centers - mu_x) ** 2).sum() / N)
    V_y = float((py * (pixel_centers - mu_y) ** 2).sum() / N)
    var_mux, cov_muVx, var_Vx = raw_moment_cov(px, pixel_centers, mu_x, V_x)
    var_muy, cov_muVy, var_Vy = raw_moment_cov(py, pixel_centers, mu_y, V_y)
    Cov_M = np.zeros((4, 4))
    Cov_M[0, 0] = var_mux; Cov_M[0, 2] = Cov_M[2, 0] = cov_muVx; Cov_M[2, 2] = var_Vx
    Cov_M[1, 1] = var_muy; Cov_M[1, 3] = Cov_M[3, 1] = cov_muVy; Cov_M[3, 3] = var_Vy
    return K @ Cov_M @ K.T


def moments_theory(shots, K):
    """Poisson-noise-only theoretical prediction for the moments estimator (needs
    raw images -- returns None if shots came from the JSON fallback)."""
    if shots[0]['img0'] is None:
        return None
    imgs = [s['img0'] for s in shots] + [s['img1'] for s in shots]
    sigmas = np.array([np.sqrt(np.clip(np.diag(moments_theta_cov(img, shots[0]['pixel_centers'], K)), 0, None))
                        for img in imgs])
    return np.sqrt((sigmas ** 2).mean(axis=0))


def j_eff_sample(shots, acs_z0, acs_z100):
    j_values = []
    for shot in shots:
        j = crb.per_shot_information(shot['theta_z0'], shot['theta_z100'], shot['phi_z0'], shot['phi_z100'],
                                      shot['n_tot0'], shot['n_tot1'], acs_z0, acs_z100, gh_order=None,
                                      prior_std=PRIOR_STD, h_theta=H_THETA, likelihood='pixel')
        j_values.append(j)
    return np.array(j_values)


def beta_crb(j_bar, n_shots, f_signal):
    sigma_per_shot, _ = crb.simple_scaling_crb(j_bar, 1)
    shot_idx = np.arange(n_shots, dtype=np.float64)
    j_const = np.full(n_shots, j_bar)
    F, cov = crb.crb_from_j(j_const, shot_idx, f_signal)
    return sigma_per_shot, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


def empirical_beta_rmse(label, n_runs, n_shots, method):
    path = OUT / 'results' / f'beta_fits_{label}_N{n_runs}_shots{n_shots}.json'
    with open(path) as f:
        data = __import__('json').load(f)
    rows = data[method]
    errs = np.array([[r['beta'][0] - r['As_true'], r['beta'][1] - r['Ac_true']] for r in rows])
    return np.sqrt((errs ** 2).mean(axis=0)), len(rows)


if __name__ == '__main__':
    t0 = time.perf_counter()
    print(f'dataset={args.dataset}  label={LABEL}  n_runs={args.n_runs}  n_shots={args.n_shots}  '
          f'psmap_tag={args.psmap_tag}  n_shots_crb={args.n_shots_crb}')

    print('\nBuilding pixel-likelihood evaluators...')
    acs_z0, acs_z100 = build_acs(args.psmap_tag)

    data_root = REPO / 'data' / args.dataset
    fallback = not data_root.exists()
    if fallback:
        print(f'\n*** {data_root} not found on disk -- falling back to true_theta/n_tot from the '
              f'saved kinematic_estimates JSON with a random phi substitute (see module docstring). '
              f'Ridge check and moments Poisson-noise theory are skipped in this mode. ***')
        shots = load_shots_from_json_fallback(LABEL, args.n_runs, args.n_shots, args.n_shots_crb)
    else:
        run_dir = sorted(data_root.glob('run_*'))[0]
        ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
        ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
        print(f'Loading {args.n_shots_crb} real shots from {run_dir.name}...')
        shots = load_shots_from_disk(ds_z0, ds_z100, args.n_shots_crb)

    if not fallback and not args.skip_ridge_check:
        print('\n=== Ridge linearity/curvature check (at truth) ===')
        ridge_check(shots, acs_z0, acs_z100, verbose=args.verbose)

    print('\n=== Full-covariance theta CRB (at truth) ===')
    crb_rmse_z0, crb_rmse_z100, _, _ = theta_crb(shots, acs_z0, acs_z100, verbose=args.verbose)
    crb_theta_avg = 0.5 * (crb_rmse_z0 + crb_rmse_z100)

    emp_best, n_runs_done, n_pts, n_out = empirical_theta_rmse(LABEL, args.n_runs, args.n_shots, 'best')
    emp_moments, *_ = empirical_theta_rmse(LABEL, args.n_runs, args.n_shots, 'moments')
    emp_null, *_ = empirical_theta_rmse(LABEL, args.n_runs, args.n_shots, 'null')
    print(f'\n{n_runs_done} runs, {n_pts} (shot,z-slice) points, {n_out} outlier(s) excluded (robust RMSE)')

    theta_rows = list(zip(THETA_NAMES, crb_theta_avg))
    theta_methods = {
        'best': dict(zip(THETA_NAMES, emp_best)),
        'moments (empirical)': dict(zip(THETA_NAMES, emp_moments)),
        'null (empirical)': dict(zip(THETA_NAMES, emp_null)),
        'null (theory, exact)': dict(zip(THETA_NAMES, null_theoretical_rmse())),
    }
    A = mi.build_A(mi.DEFAULT_T_DET)
    m_eta, P_eta = mi.build_prior(mi.DEFAULT_T_DET)
    K, A_m_eta = mi.precompute_kalman(m_eta, P_eta, A)
    mom_theory = moments_theory(shots, K)
    if mom_theory is not None:
        theta_methods['moments (Poisson theory)'] = dict(zip(THETA_NAMES, mom_theory))
    print_comparison_table(f'{LABEL}: theta CRB vs. the full estimator spectrum', theta_rows, theta_methods)

    print('\n=== Beta CRB ===')
    j_values = j_eff_sample(shots, acs_z0, acs_z100)
    j_bar = float(j_values.mean())
    j_rel_std = float(j_values.std() / j_bar)
    sigma_per_shot, crb_As, crb_Ac = beta_crb(j_bar, args.n_shots, args.f_signal)
    print(f'  j_bar={j_bar:.4e}  (rel. shot-to-shot spread {j_rel_std:.1%}, {len(shots)} sample shots)')
    print(f'  sigma_per_shot (rad/sqrt(shot)) = {sigma_per_shot:.4e}')
    print(f'  CRB(N={args.n_shots}): As={crb_As:.4e}  Ac={crb_Ac:.4e}')

    beta_rows = [('As', crb_As), ('Ac', crb_Ac)]
    beta_methods = {}
    for method in ('best', 'moments', 'null'):
        try:
            (rmse_As, rmse_Ac), n_beta_runs = empirical_beta_rmse(LABEL, args.n_runs, args.n_shots, method)
            beta_methods[method] = dict(As=rmse_As, Ac=rmse_Ac)
        except (FileNotFoundError, KeyError):
            pass
    if beta_methods:
        print_comparison_table(f'{LABEL}: beta CRB vs. empirical', beta_rows, beta_methods)

    out_path = OUT / 'results' / f'crb_vs_empirical_{LABEL}_N{args.n_runs}_shots{args.n_shots}.json'
    save_json(out_path, dict(
        dataset=args.dataset, label=LABEL, n_runs=args.n_runs, n_shots=args.n_shots,
        psmap_tag=args.psmap_tag, n_shots_crb=args.n_shots_crb, fallback_mode=fallback,
        theta_crb=dict(zip(THETA_NAMES, crb_theta_avg.tolist())),
        theta_methods={k: {n: float(v) for n, v in d.items()} for k, d in theta_methods.items()},
        beta_crb=dict(As=crb_As, Ac=crb_Ac, sigma_per_shot=sigma_per_shot, j_bar=j_bar, j_rel_std=j_rel_std),
        beta_methods=beta_methods,
    ))
    print(f'\nTotal time: {time.perf_counter() - t0:.1f}s')
