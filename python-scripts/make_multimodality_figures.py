"""
make_multimodality_figures.py — publication/slide-quality figures documenting
phi's multimodal NLL landscape, for notes/joint_profile_inference.tex.

Produces (into notes/figures/):
  1. fig_phi_landscapes.png    -- likelihood-vs-phi curves for 6 representative
     shots (near-degenerate + typical multimodal), spanning the observed
     NLL-gap range, with REAL multi-start optimizer landing points overlaid
     for two of them (shows different starts genuinely land in different
     basins -- not a theoretical artifact of the grid).
  2. fig_laplace_blindspot.png -- direct visual of what a local-quadratic
     (Laplace/Fisher) approximation around the global optimum misses: the
     true multimodal landscape vs. the Gaussian a naive CRB/Fisher treatment
     would fit at the minimum.
  3. fig_multimodality_summary.png -- polished n_modes / gap / separation
     histograms across the full 40-shot sample (reuses
     results/phi_multimodality/scan_results.npz from
     diagnose_phi_multimodality.py).
  4. fig_profile_vs_marginal.png -- polished profiled-vs-marginalized NLL(beta)
     comparison (reuses results/phi_multimodality/profile_vs_marginal.npz
     from diagnose_profile_vs_marginal.py).

All curves use directly-computed NLL grids (no cached/interpolated values)
except where explicitly reusing the prior scripts' saved summary arrays.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import crb_signal as crb                                          # noqa: E402
import map_inference as mi                                         # noqa: E402
import joint_profile_inference as jpi                              # noqa: E402
from helpers import ImageShotDataset                                # noqa: E402
from diagnose_profile_vs_marginal_v2 import refine_basin_laplace    # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

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
out_dir.mkdir(parents=True, exist_ok=True)


def load_counts(sid):
    img0 = ds_z0[sid]; img1 = ds_z100[sid]
    n_g0 = jpi._downsample(img0[0].astype(np.float64), bins).ravel()
    n_e0 = jpi._downsample(img0[1].astype(np.float64), bins).ravel()
    n_g1 = jpi._downsample(img1[0].astype(np.float64), bins).ravel()
    n_e1 = jpi._downsample(img1[1].astype(np.float64), bins).ravel()
    return n_g0, n_e0, n_g1, n_e1


def nll_grid(sid):
    counts = load_counts(sid)
    return np.array([jpi.joint_nll(np.concatenate([[p], prior_mean, prior_mean]), 0.0,
                                    acs_z0, acs_z100, *counts, prior_mean, prior_std)
                      for p in phi_grid])


def local_minima_idx(nlls):
    n = len(nlls)
    is_min = np.array([nlls[i] <= nlls[(i - 1) % n] and nlls[i] <= nlls[(i + 1) % n]
                        for i in range(n)])
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


# ============================================================
# Figure 1: representative landscapes + multi-start landing points
# ============================================================
print('=== Figure 1: phi landscapes with multi-start optimizer landings ===')
near_degenerate = [72, 161, 156]     # gaps: 665, 21774, 31622
typical = [0, 2, 189]                # gaps: 252348, 976680, 1346205 (largest = "cleanest")
shot_ids = near_degenerate + typical
tags = ['near-degenerate'] * 3 + ['typical'] * 3

grids = {sid: nll_grid(sid) for sid in shot_ids}

# real multi-start optimizer runs on two representative shots: one
# near-degenerate (161), one typical (0) -- 12 evenly spaced phi0 starts,
# each a genuine profile_shot_analytic call (same solver fit_beta uses).
multistart_shots = [161, 0]
n_starts = 12
start_phis = np.linspace(-np.pi, np.pi, n_starts, endpoint=False)
landing = {}
for sid in multistart_shots:
    counts = load_counts(sid)
    finals = []
    for p0 in start_phis:
        x0 = np.concatenate([[p0], prior_mean, prior_mean])
        x, nll, _ = jpi.profile_shot_analytic(x0, 0.0, acs_z0, acs_z100, *counts,
                                                prior_mean, prior_std, n_iter=150)
        phi_final = ((x[0] + np.pi) % (2 * np.pi)) - np.pi
        finals.append((p0, phi_final, nll))
    landing[sid] = finals
    print(f'  shot {sid}: {len(set(round(f[1], 2) for f in finals))} distinct converged phi values '
          f'across {n_starts} starts')

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, sid, tag in zip(axes.flat, shot_ids, tags):
    nlls = grids[sid]
    ll = -(nlls - nlls.min())  # relative log-likelihood: 0 at the mode, negative elsewhere
    ax.plot(phi_grid, ll, color='#2b6cb0', lw=1.8)
    ax.fill_between(phi_grid, ll, ll.min() - 1, alpha=0.08, color='#2b6cb0')
    idx = local_minima_idx(nlls)
    order = np.argsort(nlls[idx])
    idx = idx[order]
    for rank, i0 in enumerate(idx):
        color = '#c53030' if rank == 0 else '#dd6b20' if rank == 1 else '#718096'
        ax.plot(phi_grid[i0], ll[i0], 'o', color=color, ms=9, zorder=5,
                 markeredgecolor='white', markeredgewidth=1.2)
    gap = float(nlls[idx[1]] - nlls[idx[0]]) if len(idx) > 1 else float('nan')
    if sid in landing:
        finals = landing[sid]
        for p0, pf, nll_f in finals:
            ax.plot(pf, 0.15, 'v', color='#2f855a', ms=7, alpha=0.85, zorder=6)
        ax.plot([], [], 'v', color='#2f855a', ms=7, label=f'{n_starts} optimizer landings')
    ax.set_title(f'shot {sid} ({tag}, gap$={gap:,.0f}$)', fontsize=12)
    ax.set_xlabel(r'$\phi$ (rad)')
    ax.set_ylabel(r'$-(\mathrm{NLL}-\mathrm{NLL}_{\min})$')
    ax.set_xlim(-np.pi, np.pi)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    if sid in landing:
        ax.legend(loc='lower center', fontsize=8, framealpha=0.9)
fig.suptitle(r'$\phi$ NLL landscape: local minima (red=best, orange=2nd, gray=other) '
             'and real multi-start optimizer landings', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_dir / 'fig_phi_landscapes.png', dpi=150)
print(f'  saved {out_dir / "fig_phi_landscapes.png"}')


# ============================================================
# Figure 2: Laplace/Fisher blind-spot illustration
#
# CORRECTED (v2): the 360-point full-period grid is fine enough to LOCATE
# basins (separated by 0.07-2.3 rad) but far too coarse to resolve their
# actual width -- a proper per-basin local refinement (see
# diagnose_profile_vs_marginal_v2.py) finds sigma ~ 0.0004-0.001 rad, i.e.
# 46-49x finer than the 0.0175 rad grid spacing. The coarse full-period
# curve therefore does NOT actually show each peak's true (needle-sharp)
# shape -- it interpolates between under-sampled points near the peak. This
# figure shows both: the coarse multi-basin view (correct for LOCATIONS),
# and a properly-resolved zoom on the true peak shape (correct for WIDTH),
# with the correctly-fit local quadratic/Laplace curve overlaid.
# ============================================================
print('\n=== Figure 2: Laplace approximation blind spot (corrected) ===')
sid_demo = 161  # near-degenerate
counts_demo = load_counts(sid_demo)
nlls = grids[sid_demo]
idx = local_minima_idx(nlls)
order = np.argsort(nlls[idx])
idx = idx[order]
true_ll = -(nlls - nlls.min())
phi0_coarse = phi_grid[idx[0]]

# proper local refinement (fixes the under-resolution bug)
phi_refined, nll_refined, sigma_laplace, nlls_local, local_phis = refine_basin_laplace(
    acs_z0, acs_z100, counts_demo, prior_mean, prior_std, 0.0, phi0_coarse,
    window=0.006, n_local=81)
local_ll = -(nlls_local - nlls.min())
gauss_ll_local = -(nll_refined - nlls.min()) - 0.5 * ((local_phis - phi_refined) / sigma_laplace) ** 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.plot(phi_grid, true_ll, color='#2b6cb0', lw=1.8, marker='.', ms=3)
ax.fill_between(phi_grid, true_ll, true_ll.min(), alpha=0.08, color='#2b6cb0')
for rank, ii in enumerate(idx):
    color = '#c53030' if rank == 0 else '#dd6b20'
    ax.plot(phi_grid[ii], true_ll[ii], 'o', color=color, ms=9, zorder=5,
             markeredgecolor='white', markeredgewidth=1.2)
grid_spacing = phi_grid[1] - phi_grid[0]
ax.axvspan(phi0_coarse - grid_spacing, phi0_coarse + grid_spacing, color='#c53030', alpha=0.15)
ax.set_xlim(-np.pi, np.pi)
ax.set_xlabel(r'$\phi$ (rad)')
ax.set_ylabel(r'$-(\mathrm{NLL}-\mathrm{NLL}_{\min})$')
ax.set_title(f'(a) Coarse 360-pt scan (spacing={grid_spacing:.4f} rad):\ncorrectly locates {len(idx)} basins, shot {sid_demo}')

ax = axes[1]
ax.plot(local_phis, local_ll, color='#2b6cb0', lw=2.2, marker='o', ms=3,
        label='true likelihood (locally refined, 81 pts)')
ax.plot(local_phis, gauss_ll_local, color='#c53030', lw=2, ls='--',
        label=fr'local quadratic fit ($\sigma$={sigma_laplace:.5f} rad)')
ax.axvline(phi0_coarse, color='gray', ls=':', lw=1.2,
           label=f'coarse-grid estimate (spacing={grid_spacing:.4f} rad)')
ax.set_xlabel(r'$\phi$ (rad)')
ax.set_title(f'(b) Zoom on the global-best basin:\ntrue peak width ({sigma_laplace:.4f} rad) is '
             f'{grid_spacing/sigma_laplace:.0f}x finer\nthan the coarse grid spacing')
ax.legend(fontsize=9, loc='lower center')
fig.suptitle('The coarse grid finds the right basins but cannot resolve their width -- '
             'this is what a naive Fisher/CRB local-curvature estimate would miss if computed at grid resolution',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(out_dir / 'fig_laplace_blindspot.png', dpi=150)
print(f'  saved {out_dir / "fig_laplace_blindspot.png"}  '
      f'(corrected sigma_laplace={sigma_laplace:.5f} rad, under-resolution factor={grid_spacing/sigma_laplace:.1f}x)')

# mass balance across ALL basins via the analytic Laplace-mixture (correct
# integration method -- see diagnose_profile_vs_marginal_v2.py), NOT a grid
# Riemann sum (which cannot resolve sub-grid-width peaks).
log_terms = []
basin_info = []
for i0 in idx:
    phi0_c = phi_grid[i0]
    phi_r, nll_r, sigma_r, _, _ = refine_basin_laplace(acs_z0, acs_z100, counts_demo,
                                                         prior_mean, prior_std, 0.0, phi0_c)
    if sigma_r is None:
        continue
    log_terms.append(-(nll_r - nlls.min()) + 0.5 * np.log(2 * np.pi * sigma_r ** 2))
    basin_info.append((phi_r, nll_r, sigma_r))
log_terms = np.array(log_terms)
from scipy.special import logsumexp as _lse
log_Z = _lse(log_terms)
mass_fracs = np.exp(log_terms - log_Z)
print(f'  basin mass fractions (Laplace-mixture, correctly integrated): '
      + ', '.join(f'{f:.3e}' for f in mass_fracs))
print(f'  (all non-best basins together contribute {1 - mass_fracs.max():.3e} of total mass -- '
      f'negligible at this NLL-gap scale, but now via a numerically sound calculation)')


# ============================================================
# Figure 3: polished summary histograms (reuse task #2 results)
# ============================================================
print('\n=== Figure 3: multimodality summary (reusing scan_results.npz) ===')
scan = np.load(REPO / 'results' / 'phi_multimodality' / 'scan_results.npz')
n_modes_arr = scan['fixed_n_modes']
gaps = scan['fixed_gaps']
seps = scan['fixed_seps']
gaps_valid = gaps[np.isfinite(gaps)]
seps_valid = seps[np.isfinite(seps)]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
counts_, bins_, _ = ax.hist(n_modes_arr, bins=np.arange(0.5, n_modes_arr.max() + 1.5),
                             rwidth=0.75, color='#2b6cb0', edgecolor='white')
for c, b in zip(counts_, bins_):
    if c > 0:
        ax.text(b + 0.5, c + 0.5, f'{int(c)}', ha='center', fontsize=10)
ax.set_xlabel('# local minima in NLL($\\phi$)')
ax.set_ylabel('# shots')
frac_multi = float(np.mean(n_modes_arr > 1))
ax.set_title(f'Mode count (n={len(n_modes_arr)} shots)\n{frac_multi:.0%} multimodal')

ax = axes[1]
ax.hist(gaps_valid, bins=20, color='#dd6b20', edgecolor='white')
ax.set_xlabel('NLL gap: best vs. 2nd-best basin')
ax.set_title(f'Severity (median={np.median(gaps_valid):,.0f})')
ax.axvline(np.median(gaps_valid), color='k', ls='--', lw=1)

ax = axes[2]
ax.hist(seps_valid, bins=20, color='#38a169', edgecolor='white')
ax.set_xlabel(r'$\phi$ separation (rad), top-2 basins')
ax.set_title(f'Basin separation (median={np.median(seps_valid):.2f} rad)')
ax.axvline(np.median(seps_valid), color='k', ls='--', lw=1)

fig.suptitle(r'$\phi$ multimodality across a 40-shot representative sample '
             '(fixed-theta scan, cross-validated against a theta-profiled scan)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(out_dir / 'fig_multimodality_summary.png', dpi=150)
print(f'  saved {out_dir / "fig_multimodality_summary.png"}')


# ============================================================
# Figure 4: polished profiled-vs-marginalized comparison (reuse task #3 results)
# ============================================================
print('\n=== Figure 4: profiled vs marginalized NLL(beta) (reusing profile_vs_marginal_v2.npz, '
      'corrected Laplace-mixture marginalization) ===')
pvm = np.load(REPO / 'results' / 'phi_multimodality' / 'profile_vs_marginal_v2.npz')
beta_scale = pvm['beta_scale']
prof_sum = pvm['prof_sum']; marg_sum = pvm['marg_sum']
nd_shots = [72, 161, 156]
prof_sum_nd = np.sum([pvm[f'profiled_{sid}'] for sid in nd_shots], axis=0)
marg_sum_nd = np.sum([pvm[f'marginal_{sid}'] for sid in nd_shots], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(beta_scale, prof_sum - prof_sum.min(), color='#2b6cb0', lw=2.2, marker='o', ms=4,
        label='profiled (point-estimate $\\phi$)')
ax.plot(beta_scale, marg_sum - marg_sum.min(), color='#c53030', lw=2.2, ls='--', marker='s', ms=4,
        label='marginalized (corrected Laplace-mixture)')
prof_argmin = beta_scale[int(np.argmin(prof_sum))]
marg_argmin = beta_scale[int(np.argmin(marg_sum))]
ax.axvline(prof_argmin, color='#2b6cb0', ls=':', alpha=0.6)
ax.axvline(marg_argmin, color='#c53030', ls=':', alpha=0.6)
ax.set_title(f'All 6 shots: argmin shift = {prof_argmin - marg_argmin:.4f}')
ax.set_xlabel(r'$\beta_\mathrm{scale}$')
ax.set_ylabel(r'summed NLL $-$ min')
ax.legend(fontsize=10)

ax = axes[1]
ax.plot(beta_scale, prof_sum_nd - prof_sum_nd.min(), color='#2b6cb0', lw=2.2, marker='o', ms=4,
        label='profiled (point-estimate $\\phi$)')
ax.plot(beta_scale, marg_sum_nd - marg_sum_nd.min(), color='#c53030', lw=2.2, ls='--', marker='s', ms=4,
        label='marginalized (corrected Laplace-mixture)')
prof_argmin_nd = beta_scale[int(np.argmin(prof_sum_nd))]
marg_argmin_nd = beta_scale[int(np.argmin(marg_sum_nd))]
ax.axvline(prof_argmin_nd, color='#2b6cb0', ls=':', alpha=0.6)
ax.axvline(marg_argmin_nd, color='#c53030', ls=':', alpha=0.6)
ax.set_title(f'Near-degenerate-only (gaps 665-31622): argmin shift = {prof_argmin_nd - marg_argmin_nd:.4f}')
ax.set_xlabel(r'$\beta_\mathrm{scale}$')
ax.legend(fontsize=10)

fig.suptitle(r'Profiled vs. marginalized $\phi$: effect on the outer $\beta$ objective '
             f'(sweep range $\\pm${beta_scale.max():.2f}, matching fit_beta bounds)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(out_dir / 'fig_profile_vs_marginal.png', dpi=150)
print(f'  saved {out_dir / "fig_profile_vs_marginal.png"}')

print('\nAll figures saved to', out_dir)
