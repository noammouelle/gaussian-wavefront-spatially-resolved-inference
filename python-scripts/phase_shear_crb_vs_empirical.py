"""
phase_shear_crb_vs_empirical.py -- Fisher/CRB for the phase-shear readout
(phase_shear_fit.py), for ANY dataset already run through phase_shear_fit.py,
analogous to non_phase_shear/analysis/crb_vs_empirical.py but for a
genuinely different model: phase_shear_fit.py fits a Gaussian x fringe
model directly to the ground-state port image (model_image: 11 params --
log_A, mu_x, mu_y, log_sig_x, log_sig_y, C_cos, C_sin, kappa_x, kappa_y,
gamma_x, gamma_y), independently at Z0 and Z100, via a plain (no prior)
Poisson MLE. This does NOT reuse crb_signal.py's PSMAP-based machinery at
all -- a fresh Fisher matrix for this specific closed-form model.

Full theory/derivation, including the rad/sqrt(shot) unit conversion and
the systematic-bias-vs-noise-floor caveat below: notes/theta_fisher_crb.tex
Sec. 8.

Method
------
1. Poisson Fisher matrix (11,11) for model_image at a given parameter point,
   via finite differences (cheap here -- model_image is a closed-form
   elementwise numpy expression on an n_bins x n_bins grid, no PSMAP lookup,
   unlike the kinematic pipeline).
2. Full inversion (no prior term -- phase_shear_fit.py's neg_logL has none),
   delta-method the (C_cos,C_sin) 2x2 sub-covariance into Var(phi_0)
   (phi_0 = atan2(C_sin,C_cos)).
3. phi_0^Z0 and phi_0^Z100 are estimated INDEPENDENTLY (unlike the kinematic
   pipeline's shared phi_i with a known z100 offset) -- Delta_phi =
   phi_0^Z100 - phi_0^Z0, so Var(Delta_phi) = Var(phi_0^Z0) + Var(phi_0^Z100).
   The effective per-shot Fisher information for Delta_phi (entering the
   SAME beta Fisher/CRB formula as crb_vs_empirical.py's j_eff, via
   standard weighted-linear-regression information for a single noisy
   Delta_phi_i observation) is j_eff = 1/Var(Delta_phi).
4. IMPORTANT CAVEAT, not swept under the rug: this CRB models pure Poisson
   shot noise for a model assumed to be exactly correctly specified. Real
   phase-shear data can show a SEPARATE systematic bias (a
   "wavefront-averaging effect" depending on cloud position/shape) that
   must be regressed out and may not fully cancel even after correction --
   that is a model-specification effect, not something a CRB (which
   assumes the model IS correct) can predict or account for. The always-
   computed "raw" empirical residual therefore only ever LOWER-BOUNDS how
   far off "raw vs. CRB" can be attributed to systematic bias vs. genuine
   estimator inefficiency; --published_tex (optional) adds a fairer,
   systematic-corrected comparison when available.

Usage
-----
  python phase_shear_crb_vs_empirical.py --data_root data/<dataset> \
      --out_dir phase_shear/results/<label> --label <label> \
      [--n_runs 3] [--n_shots_crb 15] [--published_tex phase_shear/paper/phase_shear_results.tex]

--data_root/--out_dir mirror phase_shear_fit.py's own flags exactly (this
script needs both: the raw images -- via the same fitted-parameter pickles,
which already embed everything the Fisher matrix needs, so no re-reading of
images is actually required -- and the fitted .pkl runs for the empirical
side). --published_tex is optional: if given and it parses (matches this
repo's notes/phase-shear-results table convention), additionally prints the
systematic-corrected (fitted-feature/oracle) comparison rows; otherwise just
prints CRB vs. the always-available raw residual, with a note that no
systematic-correction comparison is available for this dataset.
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))

from phase_shear_fit import model_image                                 # noqa: E402
import crb_signal as crb                                                 # noqa: E402
from crb_report import print_comparison_table, save_json                 # noqa: E402

PARAM_NAMES = ['log_A', 'mu_x', 'mu_y', 'log_sig_x', 'log_sig_y',
               'C_cos', 'C_sin', 'kappa_x', 'kappa_y', 'gamma_x', 'gamma_y']
IDX_C = [5, 6]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--data_root', required=True, help='data/<dataset> the phase_shear_fit.py run used '
                                                    '(unused for the CRB itself, kept for symmetry '
                                                    'with phase_shear_fit.py and future extensions)')
p.add_argument('--out_dir', required=True, help='phase_shear/results/<label> -- the fitted .pkl runs '
                                                  'phase_shear_fit.py wrote (both the CRB and the '
                                                  'empirical side are computed from these)')
p.add_argument('--label', default=None, help='short tag for the printed table / output JSON '
                                              '(default: out_dir\'s basename)')
p.add_argument('--n_runs', type=int, default=3, help='number of .pkl runs to load')
p.add_argument('--n_shots_crb', type=int, default=15, help='shots per run to sample for the Fisher/CRB '
                                                             'side -- 11-param finite-diff Fisher x '
                                                             'n_bins^2 pixels is not free')
p.add_argument('--published_tex', default=None, help='optional path to a notes/phase-shear-results-style '
                                                       '.tex table (see phase_shear/paper/phase_shear_results.tex) '
                                                       'for the systematic-corrected comparison rows')
args = p.parse_args()
LABEL = args.label or Path(args.out_dir).name


def params_from_fit(fit):
    return np.array([
        np.log(fit['A']), fit['mu_x'], fit['mu_y'],
        np.log(fit['sig_x']), np.log(fit['sig_y']),
        fit['C_cos'], fit['C_sin'],
        fit['kappa_x'], fit['kappa_y'], fit['gamma_x'], fit['gamma_y'],
    ])


def phase_shear_fisher(p, xx, yy):
    """(11,11) Poisson Fisher matrix at params p, finite-difference Jacobian
    (relative step per parameter, floored -- these params span wildly
    different scales: log_A~O(7), mu_x~O(1e-5), kappa_x~O(3e4))."""
    h = np.maximum(np.abs(p) * 1e-4, 1e-6)
    mu0 = model_image(xx, yy, *p).clip(1e-6)
    npix = mu0.size
    J = np.zeros((npix, len(p)))
    for k in range(len(p)):
        pp = p.copy(); pp[k] += h[k]
        pm = p.copy(); pm[k] -= h[k]
        mup = model_image(xx, yy, *pp).clip(1e-6)
        mum = model_image(xx, yy, *pm).clip(1e-6)
        J[:, k] = ((mup - mum) / (2 * h[k])).ravel()
    W = 1.0 / mu0.ravel()
    return (J * W[:, None]).T @ J


def shot_var_delta_phi(row, pc_binned):
    xx, yy = np.meshgrid(pc_binned, pc_binned, indexing='ij')
    variances = []
    for key in ('fit_z0', 'fit_z100'):
        fit = row[key]
        p = params_from_fit(fit)
        H = phase_shear_fisher(p, xx, yy)
        cov = np.linalg.inv(H)
        cov_cc = cov[np.ix_(IDX_C, IDX_C)]
        C_cos, C_sin = fit['C_cos'], fit['C_sin']
        C2 = C_cos ** 2 + C_sin ** 2
        grad = np.array([-C_sin / C2, C_cos / C2])
        variances.append(float(grad @ cov_cc @ grad))
    return variances[0] + variances[1], variances[0], variances[1]


def load_runs(pkl_dir, n_runs):
    paths = sorted(Path(pkl_dir).glob('phase_shear_run_*.pkl'))[:n_runs]
    if not paths:
        raise FileNotFoundError(f'no phase_shear_run_*.pkl files found under {pkl_dir}')
    runs = []
    for path in paths:
        with open(path, 'rb') as f:
            runs.append(pickle.load(f))
    return runs


def empirical_delta_phi_std(runs):
    """Exactly the residual definition used in
    phase_shear/notebooks/shear_mle_distributions.ipynb (verified to
    reproduce its published number, 13.58 vs. 13.84 mrad for 1e6 -- the
    small remaining difference is presumably a different run subset, not a
    formula mismatch). Two non-obvious pieces matter: (1) est_raw is
    NEGATED relative to the naive dphi_fit - delta_phi_true (a sign
    convention in how the notebook defines its "estimate" of the injected
    signal), and (2) the mean is removed via a proper circular-mean unwrap
    THEN a second, ordinary (non-circular) mean-subtraction on the unwrapped
    values -- not a single naive wrapped subtraction, which is what an
    earlier, wrong version of this function did (gave ~142 mrad instead of
    ~14 mrad -- a 10x error from missing exactly this)."""
    dphi_raw, delta_true = [], []
    for run in runs:
        for row in run['rows']:
            dphi_raw.append(row['fit_z100']['phi_0'] - row['fit_z0']['phi_0'])
            delta_true.append(row['delta_phi_true'])
    dphi_raw = np.array(dphi_raw); delta_true = np.array(delta_true)

    mean_angle = np.angle(np.exp(1j * dphi_raw).mean())
    dphi_circ = ((dphi_raw - mean_angle + np.pi) % (2 * np.pi)) - np.pi + mean_angle
    est_raw = -(dphi_circ - dphi_circ.mean())
    res_raw = est_raw - delta_true
    return float(res_raw.std()), len(res_raw)


def parse_published_tex(path, label):
    """Best-effort parser for this repo's phase-shear-results table
    convention (see phase_shear/paper/phase_shear_results.tex): one
    \\multirow[t]{4}{*}{$10^N$ atoms} block per dataset, followed by 4 rows
    (raw / fitted-feature in-sample / fitted-feature leave-one-run-out /
    oracle), columns (residual std [mrad], beta RMSE [mrad] or ---).
    Matches label's exponent digit (e.g. '1e6' -> '6') against the table's
    $10^N$ atoms header. Returns None (with a printed note) if the file
    doesn't parse or no matching dataset block is found -- never raises,
    since this comparison is optional."""
    m = re.search(r'1e(\d+)', label)
    if not m:
        print(f'  (--published_tex given, but --label "{label}" has no 1eN pattern to match '
              f'against a $10^N$ atoms table block -- skipping)')
        return None
    exponent = m.group(1)
    try:
        text = Path(path).read_text()
    except OSError as e:
        print(f'  (--published_tex {path} could not be read: {e} -- skipping)')
        return None

    block_re = re.compile(
        r'\\multirow\[t\]\{4\}\{\*\}\{\$10\^' + exponent + r'\$ atoms\}(.*?)(?=\\multirow|\\bottomrule)',
        re.DOTALL)
    block = block_re.search(text)
    if not block:
        print(f'  (--published_tex {path}: no "$10^{exponent}$ atoms" block found -- skipping)')
        return None
    row_re = re.compile(r'&\s*([\d.]+|---)\s*&\s*([\d.]+|---)\s*\\\\')
    rows = row_re.findall(block.group(1))
    if len(rows) < 4:
        print(f'  (--published_tex {path}: expected 4 rows, found {len(rows)} -- skipping)')
        return None
    keys = ['raw', 'fit_is', 'fit_oos', 'oracle']
    resid = {k: float(v[0]) for k, v in zip(keys, rows)}
    beta = {f'beta_{k}': (float(v[1]) if v[1] != '---' else None) for k, v in zip(keys, rows)}
    return dict(**resid, **beta)


if __name__ == '__main__':
    runs = load_runs(args.out_dir, args.n_runs)
    pc_binned = runs[0]['pc_binned']
    f_signal = runs[0]['f_signal']

    j_values, var_z0_list, var_z100_list = [], [], []
    for run in runs:
        for row in run['rows'][:args.n_shots_crb]:
            var_dphi, var_z0, var_z100 = shot_var_delta_phi(row, pc_binned)
            j_values.append(1.0 / var_dphi)
            var_z0_list.append(var_z0); var_z100_list.append(var_z100)

    j_bar = float(np.mean(j_values))
    sigma_dphi_crb = np.sqrt(1.0 / j_bar)
    emp_std_raw, n_emp = empirical_delta_phi_std(runs)

    sigma_per_shot, _ = crb.simple_scaling_crb(j_bar, 1)
    n_shots_per_run = len(runs[0]['rows'])
    shot_idx = np.arange(n_shots_per_run, dtype=np.float64)
    j_const = np.full(n_shots_per_run, j_bar)
    _, cov = crb.crb_from_j(j_const, shot_idx, f_signal)
    crb_As, crb_Ac = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    crb_beta_avg = 0.5 * (crb_As + crb_Ac) * 1e3

    print(f'{LABEL}: j_bar={j_bar:.4e} (from {len(j_values)} shots across {len(runs)} runs)  '
          f'mean Var(phi0_z0)={np.mean(var_z0_list):.4e}  mean Var(phi0_z100)={np.mean(var_z100_list):.4e}')
    print(f'  sigma_per_shot (beta) = {sigma_per_shot:.4e} rad/sqrt(shot)')

    dphi_methods = {'raw (this dataset, always available)': {'Delta_phi': emp_std_raw * 1e3}}
    beta_methods = {}
    if args.published_tex:
        pub = parse_published_tex(args.published_tex, LABEL)
        if pub is not None:
            dphi_methods = {
                'raw': {'Delta_phi': pub['raw']},
                'fitted-feature (IS)': {'Delta_phi': pub['fit_is']},
                'fitted-feature (OOS)': {'Delta_phi': pub['fit_oos']},
                'oracle': {'Delta_phi': pub['oracle']},
            }
            beta_methods = {name: {'As': v, 'Ac': v} for name, v in
                             [('raw', pub['beta_raw']), ('fitted-feature (IS)', pub['beta_fit_is']),
                              ('oracle', pub['beta_oracle'])] if v is not None}

    print_comparison_table(f'{LABEL}: Delta_phi CRB vs. empirical', [('Delta_phi', sigma_dphi_crb * 1e3)],
                            dphi_methods, unit=' mrad')
    if beta_methods:
        print_comparison_table(f'{LABEL}: beta CRB vs. empirical (published, systematic-corrected)',
                                [('As', crb_As * 1e3), ('Ac', crb_Ac * 1e3)], beta_methods, unit=' mrad')
    else:
        print(f'\n  (no --published_tex systematic-correction comparison available for {LABEL} -- '
              f'CRB(beta) = {crb_beta_avg:.3f} mrad; only the raw Delta_phi residual above is directly '
              f'comparable, and per the module docstring caveat, raw includes an unmodeled systematic '
              f'so a large raw/CRB ratio should not be read as pure estimator inefficiency)')

    save_json(REPO / 'phase_shear' / 'results' / f'crb_vs_empirical_{LABEL}.json', dict(
        label=LABEL, data_root=args.data_root, out_dir=args.out_dir, n_runs=len(runs),
        n_shots_crb=args.n_shots_crb, j_bar=j_bar, sigma_per_shot=sigma_per_shot,
        crb_delta_phi_mrad=sigma_dphi_crb * 1e3, crb_beta_As_mrad=crb_As * 1e3, crb_beta_Ac_mrad=crb_Ac * 1e3,
        empirical_raw_mrad=emp_std_raw * 1e3, n_emp_shots=n_emp,
        dphi_methods=dphi_methods, beta_methods=beta_methods,
    ))
