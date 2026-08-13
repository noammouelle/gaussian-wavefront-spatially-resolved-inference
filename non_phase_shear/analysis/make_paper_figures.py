"""
Generates the figures and tables for the kinematic-estimation paper:
  - Residual plots: theta_hat - theta_true, all 8 kinematic params,
    4 methods x 2 atom counts, pooled across all runs/shots.
  - Scatter plots: (As_hat, Ac_hat) per run, 4 methods x 2 atom counts.
  - RMSE tables (kinematic params + beta), printed as both text and LaTeX.

Usage: python make_paper_figures.py
Reads: results/kinematic_estimates_{1e6,1e8}_N10_shots50.json
       results/beta_fits_{1e6,1e8}_N10_shots50.json
Writes: notes/figures/*.png, results/paper_tables.tex
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

THETA_NAMES = ['mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0', 'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy']
THETA_LABELS = [r'$\mu_{x0}$', r'$\mu_{y0}$', r'$\mu_{vx0}$', r'$\mu_{vy0}$',
                r'$\sigma_x$', r'$\sigma_y$', r'$\sigma_{vx}$', r'$\sigma_{vy}$']
METHODS = ['null', 'moments', 'best', 'oracle']
METHOD_LABELS = {'null': 'Null (prior only)', 'moments': 'MAP-moments (Kalman)',
                  'best': 'Pixel-likelihood (bins=32, tight range)', 'oracle': 'Oracle (true $\\theta$)'}
METHOD_COLORS = {'null': 'gray', 'moments': 'tab:orange', 'best': 'tab:blue', 'oracle': 'tab:green'}
DATASETS = ['1e6', '1e8']
DATASET_LABELS = {'1e6': r'$10^6$ atoms', '1e8': r'$10^8$ atoms'}
N_RUNS, N_SHOTS = 10, 50


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


# ============ Figure 1: residual plots (one figure per dataset) ============
for dataset in DATASETS:
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
    print(f'Saved residuals_{dataset}.png', flush=True)

# ============ Figure 2: beta scatter plots (one figure per dataset) ============
for dataset in DATASETS:
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
    print(f'Saved beta_scatter_{dataset}.png', flush=True)

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
