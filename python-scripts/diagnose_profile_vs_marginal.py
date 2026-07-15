"""
diagnose_profile_vs_marginal.py — quantify bias in the profiled (point-estimate
phi) NLL(beta) vs a fine-grid marginalized-phi reference NLL(beta), for the
pixel-resolved image-likelihood estimator in joint_profile_inference.py.

Context: diagnose_phi_multimodality.py found phi's NLL landscape is
multimodal for 100% of a 50-shot sample (3-4 local minima, some with NLL gaps
between the top two basins as small as 665-32000 units out of ~1.5e7 total --
near-degenerate). This script tests directly whether that multimodality
actually biases beta estimation: does profiling (argmin over phi, what
fit_beta/fit_beta_batch do) produce a systematically different, and
specifically non-smooth/kinked, NLL(beta) than properly marginalizing phi
out, especially for the near-degenerate shots where small beta changes could
plausibly flip which basin is the argmin?

Methodology: theta held at prior_mean (same simplification used in
diagnose_phi_multimodality.py's fixed-theta scan, cross-validated there
against a theta-profiled scan -- multimodal/unimodal classification agreed
10/10). For a 1D sweep of trial beta along a fixed direction, and a
representative set of shots (near-degenerate multimodal + typical
multimodal), compute both:
  NLL_profiled(beta)     = min_phi NLL(phi, delta_phi_trial(beta))
  NLL_marginalized(beta) = -logsumexp_phi(-NLL(phi, delta_phi_trial(beta))) + log(dphi)
via direct grid evaluation (no optimization) at each beta -- cheap and exact
given the grid resolution already validated in diagnose_phi_multimodality.py.
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


def nll_grid_at_delta_phi(acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std,
                           delta_phi_trial, phi_grid):
    return np.array([jpi.joint_nll(np.concatenate([[p], prior_mean, prior_mean]),
                                    delta_phi_trial, acs_z0, acs_z100,
                                    n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)
                      for p in phi_grid])


def main():
    bins = 16
    pixel_n_gh = 4
    prior_mean = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
    prior_std = np.array([10e-6] * 8)
    n_grid = 360
    phi_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    dphi = phi_grid[1] - phi_grid[0]

    data_root = REPO / 'data' / ('R80_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
                                  'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000')
    run_dir = sorted(data_root.glob('run_*'))[0]

    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    acs_z0, acs_z100 = crb.build_evaluators(run_dir, mi.DEFAULT_T_DET, bins,
                                             likelihood='pixel', pixel_n_gh=pixel_n_gh)

    # near-degenerate multimodal shots (small top-2 NLL gap, from
    # diagnose_phi_multimodality.py's fixed-theta scan) + typical multimodal
    # shots, for contrast.
    near_degenerate = [72, 161, 156]
    typical = [0, 2, 33]
    shot_ids = near_degenerate + typical

    def load_counts(sid):
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_g0 = jpi._downsample(img0[0].astype(np.float64), bins).ravel()
        n_e0 = jpi._downsample(img0[1].astype(np.float64), bins).ravel()
        n_g1 = jpi._downsample(img1[0].astype(np.float64), bins).ravel()
        n_e1 = jpi._downsample(img1[1].astype(np.float64), bins).ravel()
        return n_g0, n_e0, n_g1, n_e1

    f_signal = 0.3
    direction = np.array([1.0, 0.5]) / np.hypot(1.0, 0.5)
    beta_scale = np.linspace(-0.2, 0.2, 21)

    print(f'=== profiled vs marginalized NLL(beta) sweep, shots={shot_ids} ===')
    t0 = time.perf_counter()
    profiled = {}   # shot_id -> (21,) NLL_profiled
    marginal = {}   # shot_id -> (21,) NLL_marginalized
    argmin_phi_profiled = {}  # shot_id -> (21,) argmin phi (to see basin switching)
    for sid in shot_ids:
        counts = load_counts(sid)
        As_i = beta_scale * direction[0]
        Ac_i = beta_scale * direction[1]
        delta_phi = As_i * np.sin(2 * np.pi * f_signal * sid) + Ac_i * np.cos(2 * np.pi * f_signal * sid)
        prof = np.empty(len(beta_scale))
        marg = np.empty(len(beta_scale))
        argphi = np.empty(len(beta_scale))
        for j, dphi_trial in enumerate(delta_phi):
            nlls = nll_grid_at_delta_phi(acs_z0, acs_z100, *counts, prior_mean, prior_std,
                                          dphi_trial, phi_grid)
            imin = int(np.argmin(nlls))
            prof[j] = nlls[imin]
            argphi[j] = phi_grid[imin]
            # stable marginalization: -log( sum_k exp(-nll_k) * dphi )
            marg[j] = -logsumexp(-nlls) - np.log(dphi)
        profiled[sid] = prof
        marginal[sid] = marg
        argmin_phi_profiled[sid] = argphi
        n_switches = int(np.sum(np.abs(np.diff(argphi)) > 0.5))
        print(f'  shot {sid:3d}: profiled NLL range=[{prof.min():.2f},{prof.max():.2f}]  '
              f'marginal NLL range=[{marg.min():.2f},{marg.max():.2f}]  '
              f'argmin-phi basin switches across sweep={n_switches}')
    print(f'  ({time.perf_counter()-t0:.1f}s)')

    # Sum across shots (emulates outer objective = sum_i NLL_i(beta))
    prof_sum = np.sum([profiled[sid] for sid in shot_ids], axis=0)
    marg_sum = np.sum([marginal[sid] for sid in shot_ids], axis=0)
    prof_argmin = beta_scale[int(np.argmin(prof_sum))]
    marg_argmin = beta_scale[int(np.argmin(marg_sum))]
    print(f'\nSummed over {len(shot_ids)} shots:')
    print(f'  argmin(profiled beta_scale)     = {prof_argmin:.4f}')
    print(f'  argmin(marginalized beta_scale) = {marg_argmin:.4f}')
    print(f'  shift = {prof_argmin - marg_argmin:.4f}  (beta_scale range is [-0.2, 0.2])')

    # near-degenerate-only subset sum (adversarial case)
    prof_sum_nd = np.sum([profiled[sid] for sid in near_degenerate], axis=0)
    marg_sum_nd = np.sum([marginal[sid] for sid in near_degenerate], axis=0)
    prof_argmin_nd = beta_scale[int(np.argmin(prof_sum_nd))]
    marg_argmin_nd = beta_scale[int(np.argmin(marg_sum_nd))]
    print(f'\nNear-degenerate-only subset ({near_degenerate}):')
    print(f'  argmin(profiled beta_scale)     = {prof_argmin_nd:.4f}')
    print(f'  argmin(marginalized beta_scale) = {marg_argmin_nd:.4f}')
    print(f'  shift = {prof_argmin_nd - marg_argmin_nd:.4f}')

    out_dir = REPO / 'results' / 'phi_multimodality'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'profile_vs_marginal.npz', beta_scale=beta_scale,
             **{f'profiled_{sid}': profiled[sid] for sid in shot_ids},
             **{f'marginal_{sid}': marginal[sid] for sid in shot_ids},
             prof_sum=prof_sum, marg_sum=marg_sum)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, sid in zip(axes.flat[:6], shot_ids):
        p = profiled[sid] - profiled[sid].min()
        m = marginal[sid] - marginal[sid].min()
        ax.plot(beta_scale, p, label='profiled', lw=2)
        ax.plot(beta_scale, m, label='marginalized', lw=2, ls='--')
        tag = 'near-degenerate' if sid in near_degenerate else 'typical'
        ax.set_title(f'shot {sid} ({tag})')
        ax.set_xlabel('beta_scale')
        ax.set_ylabel('NLL - min(NLL)')
        ax.legend(fontsize=8)

    ax = axes.flat[6]
    ax.plot(beta_scale, prof_sum - prof_sum.min(), label='profiled sum (6 shots)', lw=2)
    ax.plot(beta_scale, marg_sum - marg_sum.min(), label='marginalized sum (6 shots)', lw=2, ls='--')
    ax.axvline(prof_argmin, color='C0', ls=':', alpha=0.6)
    ax.axvline(marg_argmin, color='C1', ls=':', alpha=0.6)
    ax.set_title(f'Summed (all 6): argmin shift={prof_argmin-marg_argmin:.4f}')
    ax.set_xlabel('beta_scale'); ax.legend(fontsize=8)

    ax = axes.flat[7]
    ax.plot(beta_scale, prof_sum_nd - prof_sum_nd.min(), label='profiled sum (near-deg only)', lw=2)
    ax.plot(beta_scale, marg_sum_nd - marg_sum_nd.min(), label='marginalized sum (near-deg only)', lw=2, ls='--')
    ax.axvline(prof_argmin_nd, color='C0', ls=':', alpha=0.6)
    ax.axvline(marg_argmin_nd, color='C1', ls=':', alpha=0.6)
    ax.set_title(f'Summed (near-deg): argmin shift={prof_argmin_nd-marg_argmin_nd:.4f}')
    ax.set_xlabel('beta_scale'); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / 'profile_vs_marginal.png', dpi=150)
    print(f'\nSaved: {out_dir / "profile_vs_marginal.npz"}')
    print(f'Saved: {out_dir / "profile_vs_marginal.png"}')


if __name__ == '__main__':
    main()
