"""
diagnose_phi_multimodality.py — characterize the fraction and severity of
phi multimodality across a representative sample of shots.

Context: notes/joint_profile_inference.tex's Known Open Issues documents that
a couple of shots, found by hand, showed 3-4 genuine local minima in
NLL(phi) per 2*pi period (theta held at prior_mean, delta_phi_trial=0 -- the
same point _phi_prescan uses to seed the outer-loop warm start). This script
extends that to a representative sample, quantifying:

  1. how many local minima each shot's NLL(phi) landscape has,
  2. the NLL gap between the best and second-best basin (severity: a small
     gap means point-estimate profiling can flip basins under small
     perturbations of beta/theta -- exactly the failure mode motivating this
     investigation),
  3. the phi separation between competing basins.

Two independent scans are run per shot to cross-check the finding isn't an
artifact of holding theta fixed:
  (a) theta held at prior_mean (cheap, matches _phi_prescan's own scan),
  (b) theta re-profiled (scipy L-BFGS-B) at each grid phi (expensive, but
      answers "does letting theta relax wash out the phi multimodality").
"""
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
import joint_profile_inference as jpi                              # noqa: E402
from helpers import ImageShotDataset                                # noqa: E402


def scan_theta_fixed(acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std,
                      delta_phi_trial, n_grid=360):
    phi_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    nlls = np.array([jpi.joint_nll(np.concatenate([[p], prior_mean, prior_mean]),
                                    delta_phi_trial, acs_z0, acs_z100,
                                    n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)
                      for p in phi_grid])
    return phi_grid, nlls


def scan_theta_profiled(acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std,
                         delta_phi_trial, n_grid=48, n_iter=25):
    """At each grid phi, optimize theta_z0/theta_z100 only (phi held fixed at
    the grid point), warm-started from prior_mean, then from the previous
    grid point's solution (cheap continuation). More expensive than the
    fixed-theta scan so uses a coarser grid."""
    phi_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    nlls = np.empty(n_grid)
    x_theta = np.concatenate([prior_mean, prior_mean])

    def obj(theta2, phi):
        return jpi.joint_nll(np.concatenate([[phi], theta2]), delta_phi_trial,
                              acs_z0, acs_z100, n_g0, n_e0, n_g1, n_e1, prior_mean, prior_std)

    for i, p in enumerate(phi_grid):
        res = minimize(obj, x_theta, args=(p,), method='L-BFGS-B',
                        options={'maxiter': n_iter})
        x_theta = res.x
        nlls[i] = res.fun
    return phi_grid, nlls


def local_minima(phi_grid, nlls):
    """Indices of local minima on the periodic grid, plus a basic
    de-duplication: two grid points that are both "locally minimal" but
    within 2 grid-steps of each other (flat/noisy bottom) count once."""
    n = len(nlls)
    is_min = np.array([nlls[i] <= nlls[(i - 1) % n] and nlls[i] <= nlls[(i + 1) % n]
                        for i in range(n)])
    idx = np.where(is_min)[0]
    if len(idx) <= 1:
        return idx
    # merge minima closer than 3 grid steps (periodic)
    merged = []
    used = np.zeros(len(idx), dtype=bool)
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


def summarize(phi_grid, nlls, label):
    idx = local_minima(phi_grid, nlls)
    vals = nlls[idx]
    order = np.argsort(vals)
    idx = idx[order]
    vals = vals[order]
    n_modes = len(idx)
    gap = float(vals[1] - vals[0]) if n_modes > 1 else float('inf')
    sep = None
    if n_modes > 1:
        d = abs(phi_grid[idx[0]] - phi_grid[idx[1]])
        sep = float(min(d, 2 * np.pi - d))
    return dict(label=label, n_modes=n_modes, best_phi=float(phi_grid[idx[0]]),
                best_nll=float(vals[0]), gap=gap, sep=sep,
                mode_phis=[float(phi_grid[i]) for i in idx],
                mode_nlls=[float(v) for v in vals])


def main():
    n_shots_fixed = 40      # cheap scan: representative sample
    n_shots_profiled = 10   # expensive scan: smaller cross-check subset
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

    # deterministic, spread-out sample across the 200-shot dataset (not just
    # the first N, to avoid any ordering artifact)
    rng = np.random.default_rng(0)
    all_ids = rng.choice(ds_z0.n_shots, size=n_shots_fixed, replace=False)
    all_ids.sort()

    def load_counts(sid):
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_g0 = jpi._downsample(img0[0].astype(np.float64), bins).ravel()
        n_e0 = jpi._downsample(img0[1].astype(np.float64), bins).ravel()
        n_g1 = jpi._downsample(img1[0].astype(np.float64), bins).ravel()
        n_e1 = jpi._downsample(img1[1].astype(np.float64), bins).ravel()
        return n_g0, n_e0, n_g1, n_e1

    print(f'=== Fixed-theta scan (n_grid=360) over {n_shots_fixed} shots, ids={list(all_ids)} ===')
    fixed_results = []
    t0 = time.perf_counter()
    for sid in all_ids:
        counts = load_counts(int(sid))
        phi_grid, nlls = scan_theta_fixed(acs_z0, acs_z100, *counts, prior_mean, prior_std,
                                           delta_phi_trial=0.0, n_grid=360)
        s = summarize(phi_grid, nlls, f'shot{sid}_fixed')
        s['shot_id'] = int(sid)
        fixed_results.append(s)
        tag = 'MULTIMODAL' if s['n_modes'] > 1 else 'unimodal'
        gap_str = f"gap={s['gap']:.3f}" if s['n_modes'] > 1 else ''
        print(f"  shot {sid:3d}: n_modes={s['n_modes']}  {tag}  {gap_str}")
    print(f'  ({time.perf_counter()-t0:.1f}s)')

    multi = [r for r in fixed_results if r['n_modes'] > 1]
    frac = len(multi) / len(fixed_results)
    print(f'\nFraction multimodal (fixed-theta, n={n_shots_fixed}): {frac:.2%} ({len(multi)}/{len(fixed_results)})')
    if multi:
        gaps = np.array([r['gap'] for r in multi])
        seps = np.array([r['sep'] for r in multi])
        print(f'  gap (NLL units) over multimodal shots: min={gaps.min():.3f} median={np.median(gaps):.3f} max={gaps.max():.3f}')
        print(f'  phi separation (rad) over multimodal shots: min={seps.min():.3f} median={np.median(seps):.3f} max={seps.max():.3f}')

    print(f'\n=== Theta-profiled cross-check scan (n_grid=48) over {n_shots_profiled} shots ===')
    profiled_ids = all_ids[:n_shots_profiled]
    profiled_results = []
    t0 = time.perf_counter()
    for sid in profiled_ids:
        counts = load_counts(int(sid))
        phi_grid, nlls = scan_theta_profiled(acs_z0, acs_z100, *counts, prior_mean, prior_std,
                                              delta_phi_trial=0.0, n_grid=48)
        s = summarize(phi_grid, nlls, f'shot{sid}_profiled')
        s['shot_id'] = int(sid)
        profiled_results.append(s)
        tag = 'MULTIMODAL' if s['n_modes'] > 1 else 'unimodal'
        gap_str = f"gap={s['gap']:.3f}" if s['n_modes'] > 1 else ''
        print(f"  shot {sid:3d}: n_modes={s['n_modes']}  {tag}  {gap_str}")
    print(f'  ({time.perf_counter()-t0:.1f}s)')

    multi_p = [r for r in profiled_results if r['n_modes'] > 1]
    frac_p = len(multi_p) / len(profiled_results)
    print(f'\nFraction multimodal (theta-profiled, n={n_shots_profiled}): {frac_p:.2%} ({len(multi_p)}/{len(profiled_results)})')

    # cross-check agreement: for the shots scanned both ways, does the
    # fixed-theta scan's multimodal/unimodal call match the profiled one?
    fixed_by_id = {r['shot_id']: r for r in fixed_results}
    agree = sum(1 for r in profiled_results if (r['n_modes'] > 1) == (fixed_by_id[r['shot_id']]['n_modes'] > 1))
    print(f'Fixed-theta vs theta-profiled multimodal/unimodal AGREEMENT: {agree}/{len(profiled_results)}')

    # save results + plot
    out_dir = REPO / 'results' / 'phi_multimodality'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'scan_results.npz',
             fixed_shot_ids=[r['shot_id'] for r in fixed_results],
             fixed_n_modes=[r['n_modes'] for r in fixed_results],
             fixed_gaps=[r['gap'] for r in fixed_results],
             fixed_seps=[r['sep'] if r['sep'] is not None else np.nan for r in fixed_results],
             profiled_shot_ids=[r['shot_id'] for r in profiled_results],
             profiled_n_modes=[r['n_modes'] for r in profiled_results])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    n_modes_arr = np.array([r['n_modes'] for r in fixed_results])
    axes[0].hist(n_modes_arr, bins=np.arange(0.5, n_modes_arr.max() + 1.5), rwidth=0.8)
    axes[0].set_xlabel('# local minima in NLL(phi)')
    axes[0].set_ylabel('# shots')
    axes[0].set_title(f'Mode count, fixed-theta scan (n={n_shots_fixed})')

    if multi:
        axes[1].hist(gaps, bins=20)
        axes[1].set_xlabel('NLL gap: best vs 2nd-best basin')
        axes[1].set_title('Severity (multimodal shots only)')

        axes[2].hist(seps, bins=20)
        axes[2].set_xlabel('phi separation (rad) between top-2 basins')
        axes[2].set_title('Basin separation (multimodal shots only)')
    fig.tight_layout()
    fig.savefig(out_dir / 'multimodality_summary.png', dpi=150)
    print(f'\nSaved: {out_dir / "scan_results.npz"}')
    print(f'Saved: {out_dir / "multimodality_summary.png"}')

    # a few example landscape plots (worst-gap and best-gap multimodal cases)
    if multi:
        multi_sorted = sorted(multi, key=lambda r: r['gap'])
        examples = [multi_sorted[0], multi_sorted[-1]] if len(multi_sorted) > 1 else multi_sorted
        fig2, axes2 = plt.subplots(1, len(examples), figsize=(6 * len(examples), 4), squeeze=False)
        for ax, r in zip(axes2[0], examples):
            sid = r['shot_id']
            counts = load_counts(sid)
            phi_grid, nlls = scan_theta_fixed(acs_z0, acs_z100, *counts, prior_mean, prior_std, 0.0, n_grid=360)
            ax.plot(phi_grid, nlls - nlls.min())
            for mp, mn in zip(r['mode_phis'], r['mode_nlls']):
                ax.axvline(mp, color='r', ls='--', alpha=0.5)
            ax.set_title(f'shot {sid}: n_modes={r["n_modes"]} gap={r["gap"]:.3f}')
            ax.set_xlabel('phi (rad)')
            ax.set_ylabel('NLL - min(NLL)')
        fig2.tight_layout()
        fig2.savefig(out_dir / 'example_landscapes.png', dpi=150)
        print(f'Saved: {out_dir / "example_landscapes.png"}')


if __name__ == '__main__':
    main()
