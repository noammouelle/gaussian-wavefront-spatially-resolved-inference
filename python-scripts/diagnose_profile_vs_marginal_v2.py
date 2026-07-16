"""
diagnose_profile_vs_marginal_v2.py — corrected profiled-vs-marginalized bias
test, fixing an under-resolution bug in diagnose_profile_vs_marginal.py.

BUG in v1: the 360-point phi grid (spacing ~0.0175 rad, chosen because it's
fine enough to correctly LOCATE basins, which are separated by 0.07-2.3 rad)
is far too coarse to INTEGRATE under each basin's likelihood peak. A direct
per-basin curvature probe found sigma_laplace ~ 0.0007 rad for a
representative shot -- ~25x finer than the grid spacing. That's physically
plausible (huge per-shot photon counts give very sharp within-basin phase
precision) but it means v1's grid-based logsumexp marginalization was
effectively sampling near-random points on unresolved spikes, not correctly
computing the integral -- so v1's "argmin shift = 0.0000" conclusion is not
trustworthy as stated and must be re-derived with a numerically sound method.

FIX: use the coarse grid only to LOCATE basins (valid -- basin separations
are two orders of magnitude larger than the within-basin width), then for
each basin fit a local quadratic (least-squares over a small fine-resolution
window) to get an accurate curvature kappa_k = d^2NLL/dphi^2 and refined
peak location/NLL. Marginalize via an ANALYTIC per-basin Laplace-mixture sum:

    Z(beta) = sum_k exp(-(NLL_k(beta) - NLL_min)) * sqrt(2*pi/kappa_k)
    NLL_marginalized(beta) = -log(Z(beta)) + NLL_min

This is itself a first working prototype of one of the candidate marginal
MLE methods (multi-start Laplace-mixture), not just a diagnostic fix.
"""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import crb_signal as crb                                          # noqa: E402
import map_inference as mi                                         # noqa: E402
import joint_profile_inference as jpi                              # noqa: E402
from helpers import ImageShotDataset                                # noqa: E402


def nll_at(acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std,
           delta_phi_trial, phi):
    return jpi.joint_nll(np.concatenate([[phi], prior_mean, prior_mean]),
                          delta_phi_trial, acs_z0, acs_z100,
                          n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)


def locate_basins(acs_z0, acs_z100, counts, prior_mean, prior_std, delta_phi_trial,
                   n_grid=360):
    """Coarse grid to locate basin centers (valid: separations >> grid spacing)."""
    phi_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    nlls = np.array([nll_at(acs_z0, acs_z100, *counts, prior_mean, prior_std,
                             delta_phi_trial, p) for p in phi_grid])
    n = len(nlls)
    is_min = np.array([nlls[i] <= nlls[(i - 1) % n] and nlls[i] <= nlls[(i + 1) % n]
                        for i in range(n)])
    idx = np.where(is_min)[0]
    if len(idx) > 1:
        merged, used = [], np.zeros(len(idx), dtype=bool)
        order = np.argsort(nlls[idx])
        for oi in order:
            if used[oi]:
                continue
            i0 = idx[oi]
            merged.append(i0)
            for oj in range(len(idx)):
                if used[oj]:
                    continue
                i1 = idx[oj]
                d = min(abs(i0 - i1), n - abs(i0 - i1))
                if d <= 3:
                    used[oj] = True
        idx = np.array(sorted(merged))
    return phi_grid[idx]


def refine_basin_laplace(acs_z0, acs_z100, counts, prior_mean, prior_std, delta_phi_trial,
                          phi0_coarse, window=0.006, n_local=41):
    """Fit a local quadratic around a coarse basin location to get an accurate
    peak location, NLL value, and curvature (hence Laplace sigma)."""
    dphis = np.linspace(-window, window, n_local)
    local_phis = phi0_coarse + dphis
    nlls_local = np.array([nll_at(acs_z0, acs_z100, *counts, prior_mean, prior_std,
                                   delta_phi_trial, p) for p in local_phis])
    # quadratic fit: NLL(dphi) = a*dphi^2 + b*dphi + c
    a, b, c = np.polyfit(dphis, nlls_local, 2)
    kappa = 2 * a
    if kappa <= 0:
        # fit failed to find a convex local minimum (window too wide/narrow,
        # or basin merged with a neighbor) -- fall back to the coarse point
        i0 = np.argmin(nlls_local)
        return phi0_coarse, nlls_local[i0], None, nlls_local, local_phis
    dphi_peak = -b / (2 * a)
    phi_refined = phi0_coarse + dphi_peak
    nll_refined = c - b ** 2 / (4 * a)
    sigma = 1.0 / np.sqrt(kappa)
    return phi_refined, nll_refined, sigma, nlls_local, local_phis


def marginalized_nll_laplace_mixture(acs_z0, acs_z100, counts, prior_mean, prior_std,
                                      delta_phi_trial, basins_coarse):
    """NLL_marginalized(beta) via an analytic per-basin Laplace-mixture sum,
    using LOCALLY-REFINED curvature (fixes the grid under-resolution bug)."""
    log_terms = []
    refined = []
    for phi0 in basins_coarse:
        phi_r, nll_r, sigma, _, _ = refine_basin_laplace(
            acs_z0, acs_z100, counts, prior_mean, prior_std, delta_phi_trial, phi0)
        if sigma is None:
            continue
        # log(exp(-nll_r) * sqrt(2*pi)*sigma) = -nll_r + 0.5*log(2*pi*sigma^2)
        log_terms.append(-nll_r + 0.5 * np.log(2 * np.pi * sigma ** 2))
        refined.append((phi_r, nll_r, sigma))
    log_Z = logsumexp(log_terms)
    return -log_Z, refined


def profiled_nll(acs_z0, acs_z100, counts, prior_mean, prior_std, delta_phi_trial, basins_coarse):
    """min over basins of the LOCALLY-REFINED nll (more accurate than the
    coarse-grid min alone, though for locating which basin wins it barely
    matters given the coarse grid's basin-location accuracy)."""
    best = None
    for phi0 in basins_coarse:
        phi_r, nll_r, sigma, _, _ = refine_basin_laplace(
            acs_z0, acs_z100, counts, prior_mean, prior_std, delta_phi_trial, phi0)
        if best is None or nll_r < best[1]:
            best = (phi_r, nll_r)
    return best[1], best[0]


def main():
    bins = 16
    pixel_n_gh = 4
    prior_mean = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
    prior_std = np.array([10e-6] * 8)

    data_root = REPO / 'data' / ('R80_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
                                  'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000')
    run_dir = sorted(data_root.glob('run_*'))[0]

    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    acs_z0, acs_z100 = crb.build_evaluators(run_dir, mi.DEFAULT_T_DET, bins,
                                             likelihood='pixel', pixel_n_gh=pixel_n_gh)

    def load_counts(sid):
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_g0 = jpi._downsample(img0[0].astype(np.float64), bins).ravel()
        n_e0 = jpi._downsample(img0[1].astype(np.float64), bins).ravel()
        n_g1 = jpi._downsample(img1[0].astype(np.float64), bins).ravel()
        n_e1 = jpi._downsample(img1[1].astype(np.float64), bins).ravel()
        return n_g0, n_e0, n_g1, n_e1

    near_degenerate = [72, 161, 156]
    typical = [0, 2, 33]
    shot_ids = near_degenerate + typical
    f_signal = 0.3
    direction = np.array([1.0, 0.5]) / np.hypot(1.0, 0.5)
    beta_scale = np.linspace(-0.2, 0.2, 11)

    # first: sanity-check the per-basin sigma at beta=0 for all 6 shots, to
    # report how far v1's grid under-resolved each one.
    print('=== Per-basin Laplace sigma sanity check (beta=0) ===')
    for sid in shot_ids:
        counts = load_counts(sid)
        basins = locate_basins(acs_z0, acs_z100, counts, prior_mean, prior_std, 0.0)
        sigmas = []
        for phi0 in basins:
            _, _, sigma, _, _ = refine_basin_laplace(acs_z0, acs_z100, counts, prior_mean,
                                                       prior_std, 0.0, phi0)
            if sigma is not None:
                sigmas.append(sigma)
        grid_spacing = 2 * np.pi / 360
        print(f'  shot {sid:3d}: {len(basins)} basins, sigma range=[{min(sigmas):.5f},{max(sigmas):.5f}] rad  '
              f'(grid spacing={grid_spacing:.5f} rad, under-resolution factor up to '
              f'{grid_spacing/min(sigmas):.1f}x)')

    print(f'\n=== Corrected profiled vs marginalized NLL(beta) sweep, shots={shot_ids} ===')
    t0 = time.perf_counter()
    profiled = {}
    marginal = {}
    for sid in shot_ids:
        counts = load_counts(sid)
        delta_phi = beta_scale * direction[0] * np.sin(2 * np.pi * f_signal * sid) + \
                    beta_scale * direction[1] * np.cos(2 * np.pi * f_signal * sid)
        prof = np.empty(len(beta_scale))
        marg = np.empty(len(beta_scale))
        for j, dphi_trial in enumerate(delta_phi):
            basins = locate_basins(acs_z0, acs_z100, counts, prior_mean, prior_std, dphi_trial)
            nll_p, _ = profiled_nll(acs_z0, acs_z100, counts, prior_mean, prior_std, dphi_trial, basins)
            nll_m, _ = marginalized_nll_laplace_mixture(acs_z0, acs_z100, counts, prior_mean,
                                                          prior_std, dphi_trial, basins)
            prof[j] = nll_p
            marg[j] = nll_m
        profiled[sid] = prof
        marginal[sid] = marg
        print(f'  shot {sid:3d}: profiled range=[{prof.min():.2f},{prof.max():.2f}]  '
              f'marginal range=[{marg.min():.2f},{marg.max():.2f}]')
    print(f'  ({time.perf_counter()-t0:.1f}s)')

    prof_sum = np.sum([profiled[sid] for sid in shot_ids], axis=0)
    marg_sum = np.sum([marginal[sid] for sid in shot_ids], axis=0)
    prof_argmin = beta_scale[int(np.argmin(prof_sum))]
    marg_argmin = beta_scale[int(np.argmin(marg_sum))]
    print(f'\nSummed over {len(shot_ids)} shots (CORRECTED marginalization):')
    print(f'  argmin(profiled beta_scale)     = {prof_argmin:.4f}')
    print(f'  argmin(marginalized beta_scale) = {marg_argmin:.4f}')
    print(f'  shift = {prof_argmin - marg_argmin:.4f}')

    prof_sum_nd = np.sum([profiled[sid] for sid in near_degenerate], axis=0)
    marg_sum_nd = np.sum([marginal[sid] for sid in near_degenerate], axis=0)
    prof_argmin_nd = beta_scale[int(np.argmin(prof_sum_nd))]
    marg_argmin_nd = beta_scale[int(np.argmin(marg_sum_nd))]
    print(f'\nNear-degenerate-only subset ({near_degenerate}) (CORRECTED marginalization):')
    print(f'  argmin(profiled beta_scale)     = {prof_argmin_nd:.4f}')
    print(f'  argmin(marginalized beta_scale) = {marg_argmin_nd:.4f}')
    print(f'  shift = {prof_argmin_nd - marg_argmin_nd:.4f}')

    out_dir = REPO / 'results' / 'phi_multimodality'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'profile_vs_marginal_v2.npz', beta_scale=beta_scale,
             **{f'profiled_{sid}': profiled[sid] for sid in shot_ids},
             **{f'marginal_{sid}': marginal[sid] for sid in shot_ids},
             prof_sum=prof_sum, marg_sum=marg_sum,
             prof_sum_nd=prof_sum_nd, marg_sum_nd=marg_sum_nd)
    print(f'\nSaved: {out_dir / "profile_vs_marginal_v2.npz"}')

    # compare against v1's (under-resolved) result for an explicit before/after,
    # using v1's OWN beta_scale array (different length than v2's) rather than
    # assuming they match.
    v1_path = REPO / 'results' / 'phi_multimodality' / 'profile_vs_marginal.npz'
    if v1_path.exists():
        v1 = np.load(v1_path)
        v1_beta_scale = v1['beta_scale']
        v1_prof_sum = v1['prof_sum']; v1_marg_sum = v1['marg_sum']
        v1_prof_argmin = v1_beta_scale[int(np.argmin(v1_prof_sum))]
        v1_marg_argmin = v1_beta_scale[int(np.argmin(v1_marg_sum))]
        print(f'\n=== Comparison to v1 (under-resolved grid marginalization) ===')
        print(f'  v1 shift (all 6)  = {v1_prof_argmin - v1_marg_argmin:.4f}')
        print(f'  v2 shift (all 6)  = {prof_argmin - marg_argmin:.4f}  (corrected)')


if __name__ == '__main__':
    main()
