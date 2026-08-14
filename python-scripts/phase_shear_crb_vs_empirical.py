"""
phase_shear_crb_vs_empirical.py — Fisher/CRB for the phase-shear readout
(phase_shear_fit.py), analogous to theta_crb_vs_empirical.py /
beta_crb_vs_empirical.py but for a genuinely different model: phase_shear_fit.py
fits a Gaussian x fringe model directly to the ground-state port image
(model_image: 11 params -- log_A, mu_x, mu_y, log_sig_x, log_sig_y, C_cos,
C_sin, kappa_x, kappa_y, gamma_x, gamma_y), independently at Z0 and Z100, via
a plain (no prior) Poisson MLE. This does NOT reuse crb_signal.py's PSMAP-based
machinery at all -- a fresh Fisher matrix for this specific closed-form model.

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
   SAME beta Fisher/CRB formula as beta_crb_vs_empirical.py's j_eff, via
   standard weighted-linear-regression information for a single noisy
   Delta_phi_i observation) is j_eff = 1/Var(Delta_phi).
4. IMPORTANT CAVEAT, not swept under the rug: this CRB models pure Poisson
   shot noise for a model assumed to be exactly correctly specified. The
   empirical phase_shear_results.tex results show a SEPARATE systematic
   bias (a "wavefront-averaging effect" depending on cloud position/shape)
   that must be regressed out and does not fully cancel even after
   correction -- that is a model-specification effect, not something a CRB
   (which assumes the model IS correct) can predict or account for. The
   fairest comparison is against the RESIDUAL after regression-based
   systematic correction (oracle or fitted-feature), not the raw number,
   since regression removes (most of) the systematic and leaves something
   closer to the pure noise floor this CRB models.
"""
import sys
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))

from phase_shear_fit import model_image                                 # noqa: E402
import crb_signal as crb                                                 # noqa: E402

PARAM_NAMES = ['log_A', 'mu_x', 'mu_y', 'log_sig_x', 'log_sig_y',
               'C_cos', 'C_sin', 'kappa_x', 'kappa_y', 'gamma_x', 'gamma_y']
IDX_C = [5, 6]


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
    H = (J * W[:, None]).T @ J
    return H


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
    runs = []
    for p in paths:
        with open(p, 'rb') as f:
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


# Published, already-validated numbers from phase_shear/paper/phase_shear_results.tex
# (a degree-1 regression of Delta_phi on fitted/true cloud-shear features,
# removing the position/shape-dependent systematic this CRB does NOT model --
# see module docstring). Not re-derived here -- referenced directly since
# re-implementing that regression pipeline is out of scope for this check.
PUBLISHED = {
    '1e6': dict(raw=13.84, fit_is=6.65, fit_oos=7.22, oracle=7.07,
                beta_raw=5.35, beta_fit=2.63, beta_oracle=2.64),
    '1e8': dict(raw=13.51, fit_is=1.18, fit_oos=1.85, oracle=1.08,
                beta_raw=5.35, beta_fit=0.46, beta_oracle=0.33),
}

if __name__ == '__main__':
    N_SHOTS_CRB = 15   # per run, kept small: 11-param finite-diff Fisher x n_bins^2=128^2 pixels is not free

    for label, pkl_dir in [('1e6', REPO / 'phase_shear' / 'results' / 'phase_shear_1e6'),
                            ('1e8', REPO / 'phase_shear' / 'results' / 'phase_shear_1e8')]:
        if not pkl_dir.exists():
            print(f'{label}: {pkl_dir} not found, skipping')
            continue
        runs = load_runs(pkl_dir, n_runs=3)
        pc_binned = runs[0]['pc_binned']
        f_signal = runs[0]['f_signal']

        j_values = []
        var_z0_list, var_z100_list = [], []
        n_done = 0
        for run in runs:
            for row in run['rows'][:N_SHOTS_CRB]:
                var_dphi, var_z0, var_z100 = shot_var_delta_phi(row, pc_binned)
                j_values.append(1.0 / var_dphi)
                var_z0_list.append(var_z0); var_z100_list.append(var_z100)
                n_done += 1

        j_values = np.array(j_values)
        j_bar = float(j_values.mean())
        sigma_dphi_crb = np.sqrt(1.0 / j_bar)   # per-shot Delta_phi CRB std

        emp_std_raw, n_emp = empirical_delta_phi_std(runs)

        sigma_per_shot, _ = crb.simple_scaling_crb(j_bar, 1)
        n_shots_per_run = len(runs[0]['rows'])
        shot_idx = np.arange(n_shots_per_run, dtype=np.float64)
        j_const = np.full(n_shots_per_run, j_bar)
        F, cov = crb.crb_from_j(j_const, shot_idx, f_signal)
        crb_As, crb_Ac = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

        print(f'\n=== {label} ===')
        print(f'  j_bar={j_bar:.4e} (from {n_done} shots)  '
              f'mean Var(phi0_z0)={np.mean(var_z0_list):.4e}  mean Var(phi0_z100)={np.mean(var_z100_list):.4e}')
        print(f'  CRB(Delta_phi) per-shot std      = {sigma_dphi_crb*1e3:.2f} mrad')
        print(f'  Empirical Delta_phi RAW resid std = {emp_std_raw*1e3:.2f} mrad  ({n_emp} shots, '
              f'includes uncorrected systematic -- see module docstring caveat)')
        print(f'  ratio (empirical raw / CRB)       = {emp_std_raw/sigma_dphi_crb:.3f}')
        print(f'  sigma_per_shot (beta)             = {sigma_per_shot:.4e} rad/sqrt(shot)')
        print(f'  CRB(beta), N={n_shots_per_run} shots/run    : As={crb_As*1e3:.3f} mrad  Ac={crb_Ac*1e3:.3f} mrad')

        pub = PUBLISHED[label]
        print(f'  --- vs. published (phase_shear_results.tex), all in mrad ---')
        print(f'  Delta_phi resid std : raw={pub["raw"]:.2f} (x{pub["raw"]/(sigma_dphi_crb*1e3):.2f})  '
              f'fitted-feat(IS)={pub["fit_is"]:.2f} (x{pub["fit_is"]/(sigma_dphi_crb*1e3):.2f})  '
              f'fitted-feat(OOS)={pub["fit_oos"]:.2f} (x{pub["fit_oos"]/(sigma_dphi_crb*1e3):.2f})  '
              f'oracle={pub["oracle"]:.2f} (x{pub["oracle"]/(sigma_dphi_crb*1e3):.2f})')
        crb_beta_avg = 0.5 * (crb_As + crb_Ac) * 1e3
        print(f'  beta RMSE           : raw={pub["beta_raw"]:.2f} (x{pub["beta_raw"]/crb_beta_avg:.2f})  '
              f'fitted-feat={pub["beta_fit"]:.2f} (x{pub["beta_fit"]/crb_beta_avg:.2f})  '
              f'oracle={pub["beta_oracle"]:.2f} (x{pub["beta_oracle"]/crb_beta_avg:.2f})')
