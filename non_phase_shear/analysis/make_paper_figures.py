"""
Generates the figures and tables for the kinematic-estimation comparison:
  - Residual plots: theta_hat - theta_true, all 8 kinematic params,
    4 methods x len(labels) datasets, pooled across all runs/shots.
  - Scatter plots: (As_hat, Ac_hat) per run, 4 methods x len(labels) datasets.
  - RMSE tables (kinematic params + beta), printed as text.

Usage: python make_paper_figures.py [labels...] [--n_runs N] [--n_shots N]
Defaults to labels 1e6 1e8, n_runs=10, n_shots=50 (the originally-reproduced
comparison) if no args given. labels must match --label used in
generate_kinematic_estimates.py / beta_fits_from_kinematics.py.
Reads: results/{kinematic_estimates,beta_fits}_<label>_N<n_runs>_shots<n_shots>.json
Writes: figures/*.png, results/paper_tables.txt
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('labels', nargs='*', default=['1e6', '1e8'],
                help='dataset labels to compare (default: 1e6 1e8)')
p.add_argument('--n_runs', type=int, default=10)
p.add_argument('--n_shots', type=int, default=50)
p.add_argument('--verbose', '-v', action='store_true',
                help='print resolved config up front and per-panel diagnostics '
                     '(axis ranges, points in/out of range) while building the figures')
args = p.parse_args()
VERBOSE = args.verbose

THETA_NAMES = ['mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0', 'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy']
THETA_LABELS = [r'$\mu_{x0}$', r'$\mu_{y0}$', r'$\mu_{vx0}$', r'$\mu_{vy0}$',
                r'$\sigma_x$', r'$\sigma_y$', r'$\sigma_{vx}$', r'$\sigma_{vy}$']
METHODS = ['null', 'moments', 'best', 'oracle']
METHOD_LABELS = {'null': 'Null (prior only)', 'moments': 'MAP-moments (Kalman)',
                  'best': 'Pixel-likelihood (bins=32, tight range)', 'oracle': 'Oracle (true $\\theta$)'}
METHOD_COLORS = {'null': 'gray', 'moments': 'tab:orange', 'best': 'tab:blue', 'oracle': 'tab:green'}
DATASETS = args.labels
_PRETTY = {'1e6': r'$10^6$ atoms', '1e8': r'$10^8$ atoms'}   # known nice labels; anything else displays as-is
DATASET_LABELS = {d: _PRETTY.get(d, d) for d in DATASETS}
N_RUNS, N_SHOTS = args.n_runs, args.n_shots


def load_kinematics(dataset):
    with open(REPO / 'results' / f'kinematic_estimates_{dataset}_N{N_RUNS}_shots{N_SHOTS}.json') as f:
        return json.load(f)


def load_beta_fits(dataset):
    with open(REPO / 'results' / f'beta_fits_{dataset}_N{N_RUNS}_shots{N_SHOTS}.json') as f:
        return json.load(f)


def compute_residuals(kin_data, method):
    """Returns (n_shots_total, 8) residuals, pooling z0 and z100 and all runs/shots."""
    residuals = []
    for run_name, run in kin_data.items():
        for shot in run['shots']:
            true_z0 = np.array(shot['true_theta_z0']); est_z0 = np.array(shot[f'theta_{method}_z0'])
            true_z100 = np.array(shot['true_theta_z100']); est_z100 = np.array(shot[f'theta_{method}_z100'])
            residuals.append(est_z0 - true_z0)
            residuals.append(est_z100 - true_z100)
    return np.array(residuals)


def compute_rmse_table(kin_data, method):
    res = compute_residuals(kin_data, method)
    return np.sqrt((res**2).mean(axis=0))


if VERBOSE:
    print(f'labels={DATASETS}  n_runs={N_RUNS}  n_shots={N_SHOTS}')

# ============ Figure 1: residual plots (one figure per dataset) ============
for dataset in DATASETS:
    t0 = time.perf_counter()
    kin = load_kinematics(dataset)
    res_by_method = {m: compute_residuals(kin, m) for m in METHODS}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for k in range(8):
        ax = axes.flat[k]
        # per-panel range: robust (0.5-99.5 percentile) over methods that
        # actually vary (excludes oracle, which is a delta function at 0
        # and would otherwise not affect the scale) -- null/moments/best
        all_vals = np.concatenate([res_by_method[m][:, k] for m in ['null', 'moments', 'best']])
        lo, hi = np.percentile(all_vals, [0.5, 99.5])
        pad = 0.1 * (hi - lo)
        xlim = (lo - pad, hi + pad)
        for method in METHODS:
            res = res_by_method[method][:, k]
            in_range = res[(res >= xlim[0]) & (res <= xlim[1])]
            if VERBOSE:
                print(f'  {dataset} {THETA_NAMES[k]} [{method}]: xlim=({xlim[0]:.3e},{xlim[1]:.3e})  '
                      f'{len(in_range)}/{len(res)} points in range')
            ax.hist(in_range, bins=50, range=xlim, histtype='step', density=True, linewidth=1.8,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_xlim(xlim)
        ax.set_title(THETA_LABELS[k], fontsize=13)
        ax.axvline(0, color='k', linewidth=0.7, linestyle=':')
        ax.set_xlabel('estimate $-$ truth  [m or m/s]')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    axes.flat[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'Kinematic parameter residuals, {DATASET_LABELS[dataset]} (pooled over {N_RUNS} runs $\\times$ {N_SHOTS} shots $\\times$ 2 slices)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / f'residuals_{dataset}.png', dpi=150)
    plt.close(fig)
    timing = f'  ({time.perf_counter() - t0:.1f}s)' if VERBOSE else ''
    print(f'Saved residuals_{dataset}.png{timing}', flush=True)

# ============ Figure 2: beta scatter plots (one figure per dataset) ============
for dataset in DATASETS:
    t0 = time.perf_counter()
    beta_data = load_beta_fits(dataset)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    for i, method in enumerate(METHODS):
        ax = axes[i]
        rows = beta_data[method]
        As_true = rows[0]['As_true']; Ac_true = rows[0]['Ac_true']
        As_hat = [r['beta'][0] for r in rows]; Ac_hat = [r['beta'][1] for r in rows]
        ax.scatter(As_hat, Ac_hat, color=METHOD_COLORS[method], s=50, alpha=0.8, zorder=3)
        ax.scatter([As_true], [Ac_true], color='red', marker='*', s=250, zorder=5, label='truth')
        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xlabel('$A_s$');
        if i == 0: ax.set_ylabel('$A_c$')
        ax.axhline(Ac_true, color='red', linewidth=0.5, alpha=0.3)
        ax.axvline(As_true, color='red', linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'$(\\hat A_s, \\hat A_c)$ across {N_RUNS} independent runs, {DATASET_LABELS[dataset]}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / f'beta_scatter_{dataset}.png', dpi=150)
    plt.close(fig)
    timing = f'  ({time.perf_counter() - t0:.1f}s)' if VERBOSE else ''
    print(f'Saved beta_scatter_{dataset}.png{timing}', flush=True)

# ============ Tables ============
lines = []
lines.append('=== Kinematic parameter RMSE ===')
for dataset in DATASETS:
    kin = load_kinematics(dataset)
    lines.append(f'\n{DATASET_LABELS[dataset]}:')
    header = f'{"param":10s} ' + ' '.join(f'{m:>12s}' for m in METHODS)
    lines.append(header)
    for k in range(8):
        row = f'{THETA_NAMES[k]:10s} '
        for method in METHODS:
            rmse = compute_rmse_table(kin, method)[k]
            row += f'{rmse:12.4e} '
        lines.append(row)

lines.append('\n=== Beta RMSE ===')
for dataset in DATASETS:
    beta_data = load_beta_fits(dataset)
    lines.append(f'\n{DATASET_LABELS[dataset]}:')
    for method in METHODS:
        rows = beta_data[method]
        errs = np.array([[r['beta'][0]-r['As_true'], r['beta'][1]-r['Ac_true']] for r in rows])
        rmse = np.sqrt((errs**2).mean(axis=0))
        lines.append(f'  {method:10s}: RMSE(As)={rmse[0]:.6f}  RMSE(Ac)={rmse[1]:.6f}')

table_text = '\n'.join(lines)
print(table_text)
with open(REPO / 'results' / 'paper_tables.txt', 'w') as f:
    f.write(table_text)
print('\nDONE')
