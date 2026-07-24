"""
2D scatter + density-contour plots of kinematic parameter residual pairs,
focused on the pairs implicated in the paper's core mechanism: the moment
method's structural degeneracy V_xf = V_x0 + 2T*C_xv0 + T^2*V_vx0 couples
position spread and velocity spread (and position/velocity mean similarly
via the linear part of the same equation) together. If that degeneracy
shows up as a real effect, residuals in (mu_x0, mu_vx0) and (sigma_x,
sigma_vx) should show visible correlation for 'moments' that is absent (or
much weaker) for 'best' (which does not go through that reduced-moment
bottleneck at all).

Usage: python make_2d_kinematic_plots.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

THETA_NAMES = ['mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0', 'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy']
THETA_LABELS = {n: l for n, l in zip(THETA_NAMES,
    [r'$\mu_{x0}$', r'$\mu_{y0}$', r'$\mu_{vx0}$', r'$\mu_{vy0}$',
     r'$\sigma_x$', r'$\sigma_y$', r'$\sigma_{vx}$', r'$\sigma_{vy}$'])}
METHODS = ['null', 'moments', 'best']   # oracle is a delta function at 0, not useful in 2D
METHOD_LABELS = {'null': 'Null (prior only)', 'moments': 'MAP-moments (Kalman)',
                  'best': 'Pixel-likelihood (bins=32, tight range)'}
METHOD_COLORS = {'null': 'gray', 'moments': 'tab:orange', 'best': 'tab:blue'}
DATASETS = ['1e6', '1e8']
DATASET_LABELS = {'1e6': r'$10^6$ atoms', '1e8': r'$10^8$ atoms'}
N_RUNS, N_SHOTS = 10, 50

# Pairs to plot: the ones physically coupled by the moment method's
# detection-plane-variance degeneracy (V_xf/V_yf mixing mean/variance
# terms), plus the sigma-sigma pair since that's where the paper's main
# effect lives.
PAIRS = [
    ('mu_x0', 'mu_vx0'), ('mu_y0', 'mu_vy0'),
    ('sigma_x', 'sigma_vx'), ('sigma_y', 'sigma_vy'),
]


def load_kinematics(dataset):
    with open(REPO / 'results' / f'kinematic_estimates_{dataset}_N{N_RUNS}_shots{N_SHOTS}.json') as f:
        return json.load(f)


def compute_residuals(kin_data, method):
    residuals = []
    for run_name, run in kin_data.items():
        for shot in run['shots']:
            true_z0 = np.array(shot['true_theta_z0']); est_z0 = np.array(shot[f'theta_{method}_z0'])
            true_z100 = np.array(shot['true_theta_z100']); est_z100 = np.array(shot[f'theta_{method}_z100'])
            residuals.append(est_z0 - true_z0)
            residuals.append(est_z100 - true_z100)
    return np.array(residuals)


for dataset in DATASETS:
    kin = load_kinematics(dataset)
    res_by_method = {m: compute_residuals(kin, m) for m in METHODS}

    fig, axes = plt.subplots(len(PAIRS), len(METHODS), figsize=(5 * len(METHODS), 4.5 * len(PAIRS)))
    for row, (px, py) in enumerate(PAIRS):
        ix, iy = THETA_NAMES.index(px), THETA_NAMES.index(py)
        # shared, robust axis limits across methods for this row (so panels are comparable)
        all_x = np.concatenate([res_by_method[m][:, ix] for m in METHODS])
        all_y = np.concatenate([res_by_method[m][:, iy] for m in METHODS])
        xlo, xhi = np.percentile(all_x, [0.5, 99.5]); ylo, yhi = np.percentile(all_y, [0.5, 99.5])
        xpad = 0.15 * (xhi - xlo); ypad = 0.15 * (yhi - ylo)
        xlim = (xlo - xpad, xhi + xpad); ylim = (ylo - ypad, yhi + ypad)

        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            x = res_by_method[method][:, ix]; y = res_by_method[method][:, iy]
            in_range = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
            x_r, y_r = x[in_range], y[in_range]

            ax.scatter(x_r, y_r, s=4, alpha=0.25, color=METHOD_COLORS[method])
            # density contours (skip if degenerate, e.g. too few unique points)
            try:
                xy = np.vstack([x_r, y_r])
                kde = gaussian_kde(xy)
                xx, yy = np.mgrid[xlim[0]:xlim[1]:80j, ylim[0]:ylim[1]:80j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors=METHOD_COLORS[method], linewidths=1.2)
            except Exception as e:
                ax.text(0.5, 0.5, f'(contour failed: {e})', transform=ax.transAxes, fontsize=7, ha='center')

            corr = np.corrcoef(x_r, y_r)[0, 1]
            ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
            ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(f'{THETA_LABELS[px]} residual')
            ax.set_ylabel(f'{THETA_LABELS[py]} residual')
            ax.set_title(f'{METHOD_LABELS[method]}\n(Pearson r = {corr:.3f})', fontsize=10)
            ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    fig.suptitle(f'Joint residual structure, {DATASET_LABELS[dataset]} (pooled over {N_RUNS} runs $\\times$ {N_SHOTS} shots $\\times$ 2 slices)', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / f'residuals_2d_{dataset}.png', dpi=140)
    plt.close(fig)
    print(f'Saved residuals_2d_{dataset}.png', flush=True)

print('DONE')
