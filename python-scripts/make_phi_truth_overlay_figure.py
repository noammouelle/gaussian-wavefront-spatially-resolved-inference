"""
make_phi_truth_overlay_figure.py — overlay ground-truth phi on the landscape
plots, evaluated at each shot's ACTUAL true delta_phi (not the delta_phi_trial=0
placeholder used in diagnose_phi_multimodality.py's characterization scan).

Ground truth is stored directly in the H5 files: phi0 (per-shot true common
phase, same convention as this script's phi nuisance) and delta_phi (true
signal-induced phase shift between Z0 and Z100 -- nonzero, ~0.05-0.1 rad for
these shots, so NOT well approximated by the delta_phi_trial=0 used for the
pure multimodality characterization). This answers directly: for these single
noise realizations, does the true phi fall inside the global-best basin, or a
secondary one? That bears directly on whether CRB is achieved in practice
(distinct from whether multistart finds the true global optimum of the
REALIZED data -- see chat discussion, 2026-07-16).
"""
import sys
from pathlib import Path
import numpy as np
import h5py

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import crb_signal as crb
import map_inference as mi
import joint_profile_inference as jpi
from helpers import ImageShotDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
                      'legend.fontsize': 9, 'figure.dpi': 150})

bins = 16
pixel_n_gh = 4
prior_mean = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
prior_std = np.array([10e-6] * 8)
n_grid = 360
phi_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)

data_root = REPO / 'data' / ('R80_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
                              'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000')
run_dir = sorted(data_root.glob('run_*'))[0]

ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
acs_z0, acs_z100 = crb.build_evaluators(run_dir, mi.DEFAULT_T_DET, bins,
                                         likelihood='pixel', pixel_n_gh=pixel_n_gh)
out_dir = REPO / 'notes' / 'figures'


def load_counts(sid):
    img0 = ds_z0[sid]; img1 = ds_z100[sid]
    n_g0 = jpi._downsample(img0[0].astype(np.float64), bins).ravel()
    n_e0 = jpi._downsample(img0[1].astype(np.float64), bins).ravel()
    n_g1 = jpi._downsample(img1[0].astype(np.float64), bins).ravel()
    n_e1 = jpi._downsample(img1[1].astype(np.float64), bins).ravel()
    return n_g0, n_e0, n_g1, n_e1


def local_minima_idx(nlls):
    n = len(nlls)
    is_min = np.array([nlls[i] <= nlls[(i - 1) % n] and nlls[i] <= nlls[(i + 1) % n] for i in range(n)])
    idx = np.where(is_min)[0]
    if len(idx) <= 1:
        return idx
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
    return np.array(sorted(merged))


f0 = h5py.File(str(run_dir / 'Z0' / 'data_IMG.h5'), 'r')
f1 = h5py.File(str(run_dir / 'Z100' / 'data_IMG.h5'), 'r')
true_phi0 = f0['phi0'][:]
true_delta_phi = f1['delta_phi'][:]  # true delta_phi between Z0 and Z100

near_degenerate = [72, 161, 156]
typical = [0, 2, 189]
shot_ids = near_degenerate + typical
tags = ['near-degenerate'] * 3 + ['typical'] * 3

print('=== Truth-overlaid phi landscapes, evaluated at each shot\'s ACTUAL true delta_phi ===')
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
truth_in_global_best = []
for ax, sid, tag in zip(axes.flat, shot_ids, tags):
    counts = load_counts(sid)
    dphi_true = float(true_delta_phi[sid])
    nlls = np.array([jpi.joint_nll(np.concatenate([[p], prior_mean, prior_mean]), dphi_true,
                                    acs_z0, acs_z100, *counts, prior_mean, prior_std)
                      for p in phi_grid])
    ll = -(nlls - nlls.min())
    idx = local_minima_idx(nlls)
    order = np.argsort(nlls[idx])
    idx = idx[order]
    gap = float(nlls[idx[1]] - nlls[idx[0]]) if len(idx) > 1 else float('nan')

    phi_truth = ((true_phi0[sid] + np.pi) % (2 * np.pi)) - np.pi
    # which basin is truth closest to?
    dists = np.array([min(abs(phi_truth - phi_grid[i]), 2 * np.pi - abs(phi_truth - phi_grid[i])) for i in idx])
    closest_rank = int(np.argmin(dists))
    in_global_best = (closest_rank == 0)
    truth_in_global_best.append(in_global_best)

    ax.plot(phi_grid, ll, color='#2b6cb0', lw=1.8)
    ax.fill_between(phi_grid, ll, ll.min(), alpha=0.08, color='#2b6cb0')
    for rank, i0 in enumerate(idx):
        color = '#c53030' if rank == 0 else '#dd6b20' if rank == 1 else '#718096'
        ax.plot(phi_grid[i0], ll[i0], 'o', color=color, ms=9, zorder=5,
                 markeredgecolor='white', markeredgewidth=1.2)
    ax.axvline(phi_truth, color='#2f855a', lw=2.2, ls='-', zorder=4,
               label=f'true $\\phi_0$={phi_truth:.3f}')
    verdict = 'TRUTH IN GLOBAL-BEST BASIN' if in_global_best else f'truth in rank-{closest_rank} basin (NOT best!)'
    ax.set_title(f'shot {sid} ({tag}, gap={gap:,.0f})\n{verdict}', fontsize=10.5,
                 color='#2f855a' if in_global_best else '#c53030')
    ax.set_xlabel(r'$\phi$ (rad)')
    ax.set_ylabel(r'$-(\mathrm{NLL}-\mathrm{NLL}_{\min})$')
    ax.set_xlim(-np.pi, np.pi)
    ax.legend(loc='lower center', fontsize=8, framealpha=0.9)
    print(f'  shot {sid:3d}: true_delta_phi={dphi_true:.4f}  true_phi0={phi_truth:.4f}  '
          f'closest basin rank={closest_rank}  {"[BEST]" if in_global_best else "[NOT BEST]"}')

fig.suptitle('$\\phi$ NLL landscape at each shot\'s ACTUAL true delta_phi, with true $\\phi_0$ overlaid '
             '(green line)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_dir / 'fig_phi_truth_overlay.png', dpi=150)
print(f'\nsaved {out_dir / "fig_phi_truth_overlay.png"}')
print(f'\nTruth in global-best basin: {sum(truth_in_global_best)}/{len(shot_ids)} shots')
