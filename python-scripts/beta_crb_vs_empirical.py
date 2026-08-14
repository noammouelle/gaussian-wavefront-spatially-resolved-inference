"""
beta_crb_vs_empirical.py — propagate the (already-validated, see
theta_crb_vs_empirical.py / theta_crb_gaussian_baseline.py) per-shot theta
Fisher information through to a beta=(A_s,A_c) CRB, in two forms:

  1. Per-shot / asymptotic: sigma_per_shot = sqrt(2/j_bar) [rad], i.e.
     sigma(N) = sigma_per_shot / sqrt(N) for any shot count N (dense-sampling
     approximation, crb_signal.py's simple_scaling_crb). This is a
     rate-independent number -- convert to rad/sqrt(Hz) yourself once you
     have an actual shot repetition rate: ASD = sigma_per_shot * sqrt(T_rep).
  2. Finite-N, matching the real experiment: crb_from_j using the REAL
     shot-index (sin/cos) pattern at the SAME N used in the existing
     beta_fits_*.json runs (50 for 1e6/1e8, 200 for confocal_random_zernike),
     so it's directly comparable to the empirical beta RMSE already measured
     there.

j_eff (crb_signal.py's per_shot_information, likelihood='pixel') already
Schur-complements theta OUT of the joint (phi,theta_z0,theta_z100) Fisher
matrix -- this IS the theta-uncertainty-propagated-into-phase-information
step this whole line of work has been building toward. Uses a representative
per-dataset sample of shots (not all N) to estimate j_bar/j_std (checked for
being tight enough that a per-shot-constant approximation is reasonable),
then applies the REAL x_i=[sin,cos](2*pi*f*i) pattern for the requested N --
avoids the O(N) cost of a full per-shot Fisher matrix for N=200 while still
using the real experimental time-sampling structure, not just the asymptotic
limit.
"""
import sys
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import crb_signal as crb                                                # noqa: E402
from theta_crb_vs_empirical import build_acs, PRIOR_STD, H_THETA        # noqa: E402
from theta_crb_gaussian_baseline import shots_from_real_data, shots_from_json_random_phi  # noqa: E402

F_SIGNAL = 0.3   # cycles/shot, matches every dataset's --signal_freq


def j_eff_sample(shots, acs_z0, acs_z100):
    j_values = []
    for shot in shots:
        j = crb.per_shot_information(shot['theta_z0'], shot['theta_z100'], shot['phi_z0'], shot['phi_z100'],
                                      shot['n_tot0'], shot['n_tot1'], acs_z0, acs_z100, gh_order=None,
                                      prior_std=PRIOR_STD, h_theta=H_THETA, likelihood='pixel')
        j_values.append(j)
    return np.array(j_values)


def beta_crb(j_bar, n_shots, f_signal=F_SIGNAL):
    """Both the asymptotic per-shot number and the finite-N number using the
    real shot-index sin/cos pattern (with j approximated as constant = j_bar
    across shots -- checked against j's actual sample spread before trusting
    this)."""
    sigma_per_shot, _ = crb.simple_scaling_crb(j_bar, 1)
    shot_idx = np.arange(n_shots, dtype=np.float64)
    j_const = np.full(n_shots, j_bar)
    F, cov = crb.crb_from_j(j_const, shot_idx, f_signal)
    sigma_As, sigma_Ac = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return sigma_per_shot, sigma_As, sigma_Ac


def empirical_beta_rmse(label, n_runs, n_shots, method='best'):
    path = REPO / 'non_phase_shear' / 'results' / f'beta_fits_{label}_N{n_runs}_shots{n_shots}.json'
    with open(path) as f:
        data = json.load(f)
    rows = data[method]
    errs = np.array([[r['beta'][0] - r['As_true'], r['beta'][1] - r['Ac_true']] for r in rows])
    rmse = np.sqrt((errs ** 2).mean(axis=0))
    return rmse[0], rmse[1], len(rows)


if __name__ == '__main__':
    N_SHOTS_CRB_SAMPLE = 30   # representative sample to estimate j_bar/j_std

    configs = [
        dict(name='1e6 (unaberrated, phi substituted)', psmap_tag='CONFOCAL_FINE',
             loader=lambda: shots_from_json_random_phi('1e6', N_SHOTS_CRB_SAMPLE, seed=1),
             beta_label='1e6', n_runs=10, n_shots=50),
        dict(name='1e8 (unaberrated, real data)', psmap_tag='CONFOCAL_FINE',
             loader=lambda: shots_from_real_data(
                 REPO / 'data' / 'R40_N50_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
                                  'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000', N_SHOTS_CRB_SAMPLE),
             beta_label='1e8', n_runs=10, n_shots=50),
        dict(name='confocal_random_zernike (aberrated, real data)', psmap_tag='confocal_random_zernike',
             loader=lambda: shots_from_real_data(
                 REPO / 'data' / 'R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
                                  'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000_'
                                  'psmapconfocal_random_zernike', N_SHOTS_CRB_SAMPLE),
             beta_label='confocal_random_zernike', n_runs=20, n_shots=200),
    ]

    print(f'{"dataset":45s} {"j_bar":>12s} {"j_rel_std":>10s}   '
          f'{"sig/sqrt(shot)":>14s} {"CRB(N) As":>10s} {"CRB(N) Ac":>10s}   '
          f'{"emp As":>10s} {"emp Ac":>10s}   {"ratio As":>9s} {"ratio Ac":>9s}')
    for cfg in configs:
        acs_z0, acs_z100 = build_acs(cfg['psmap_tag'])
        shots = cfg['loader']()
        j_values = j_eff_sample(shots, acs_z0, acs_z100)
        j_bar = float(j_values.mean())
        j_rel_std = float(j_values.std() / j_bar)

        sigma_per_shot, crb_As, crb_Ac = beta_crb(j_bar, cfg['n_shots'])
        emp_As, emp_Ac, n_runs_avail = empirical_beta_rmse(cfg['beta_label'], cfg['n_runs'], cfg['n_shots'], method='best')
        mom_As, mom_Ac, _ = empirical_beta_rmse(cfg['beta_label'], cfg['n_runs'], cfg['n_shots'], method='moments')
        null_As, null_Ac, _ = empirical_beta_rmse(cfg['beta_label'], cfg['n_runs'], cfg['n_shots'], method='null')

        print(f'{cfg["name"]:45s} {j_bar:12.4e} {j_rel_std:10.3f}   '
              f'{sigma_per_shot:14.4e} {crb_As:10.4e} {crb_Ac:10.4e}')
        print(f'{"":45s} best   : As={emp_As:.4e} (x{emp_As / crb_As:.2f})  Ac={emp_Ac:.4e} (x{emp_Ac / crb_Ac:.2f})')
        print(f'{"":45s} moments: As={mom_As:.4e} (x{mom_As / crb_As:.2f})  Ac={mom_Ac:.4e} (x{mom_Ac / crb_Ac:.2f})')
        print(f'{"":45s} null   : As={null_As:.4e} (x{null_As / crb_As:.2f})  Ac={null_Ac:.4e} (x{null_Ac / crb_Ac:.2f})')
        print(f'{"":45s} (j from {len(shots)} sample shots; empirical from {n_runs_avail} independent runs)')
