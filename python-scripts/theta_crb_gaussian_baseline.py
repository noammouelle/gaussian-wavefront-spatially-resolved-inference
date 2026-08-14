"""
theta_crb_gaussian_baseline.py — same full-covariance theta Fisher/CRB check
as theta_crb_vs_empirical.py, but for the unaberrated (CONFOCAL_FINE) '1e6'
and '1e8' datasets already fit and saved in non_phase_shear/results/. No new
data generation -- reuses the existing kinematic_estimates_{1e6,1e8}_N10_shots50.json
checkpoints.

Source data availability differs between the two:
  - '1e8' matches data/R40_N50_A100000000_..._f0.3000 exactly (1e8 atoms, 50
    shots/run, standard prior -- confirmed by matching true_theta_z0/n_tot
    against that directory's actual attrs). Real phi/theta/n_tot read
    directly from the h5 files, same as theta_crb_vs_empirical.py.
  - '1e6' has NO matching dataset on disk (checked: the only 1e6-atom,
    50-shots/run directory present, R80_N50_..._sigVx309um, has a DIFFERENT
    cloud-width prior than what the saved true_theta values actually show --
    its source data was apparently deleted/regenerated since that fit was
    run). Its true_theta and n_tot ARE saved in the kinematic_estimates JSON
    (both are used by build_joint_fisher's 'pixel' likelihood -- only the
    true phi is not saved anywhere). Since phi is drawn independently of
    theta, ~Uniform(0,2pi) (phi0_mode='random' in the generator, unrelated
    to theta's own draw), this substitutes random phi draws in its place.
    Averaged over many shots this should be a good approximation -- but it
    is an approximation, not the exact per-shot value used in the real fit,
    and is reported as such.
"""
import sys
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import crb_signal as crb                                                # noqa: E402
from helpers import ImageShotDataset                                     # noqa: E402
from theta_crb_vs_empirical import (                                     # noqa: E402
    build_acs, downsample_tight, theta_crb, empirical_rmse,
    THETA_NAMES, BINS_BEST, TIGHT_HALF_RANGE,
)

PSMAP_TAG_BASELINE = 'CONFOCAL_FINE'


def shots_from_real_data(data_root, n_shots):
    run_dir = sorted(Path(data_root).glob('run_*'))[0]
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    shots = []
    for sid in range(n_shots):
        meta0 = ds_z0.meta(sid); meta1 = ds_z100.meta(sid)
        theta_z0 = np.array([meta0[k] for k in THETA_NAMES])
        theta_z100 = np.array([meta1[k] for k in THETA_NAMES])
        img0 = ds_z0[sid]; img1 = ds_z100[sid]
        n_tot0 = float(img0[0].sum() + img0[1].sum())
        n_tot1 = float(img1[0].sum() + img1[1].sum())
        shots.append(dict(theta_z0=theta_z0, theta_z100=theta_z100,
                           phi_z0=float(meta0['phi0']), phi_z100=float(meta1['phi0']),
                           n_tot0=n_tot0, n_tot1=n_tot1))
    return shots


def shots_from_json_random_phi(label, n_shots, seed=0):
    """theta_true and n_tot come from the saved fit; phi is a random draw
    (see module docstring) since the source data no longer exists."""
    path = REPO / 'non_phase_shear' / 'results' / f'kinematic_estimates_{label}_N10_shots50.json'
    with open(path) as f:
        data = json.load(f)
    rng = np.random.default_rng(seed)
    shots = []
    count = 0
    for run_name, run in data.items():
        for shot in run['shots']:
            if count >= n_shots:
                break
            theta_z0 = np.array(shot['true_theta_z0'])
            theta_z100 = np.array(shot['true_theta_z100'])
            n_tot0 = float(shot['n_g0'] + shot['n_e0'])
            n_tot1 = float(shot['n_g1'] + shot['n_e1'])
            phi_z0 = float(rng.uniform(0, 2 * np.pi))
            phi_z100 = float(rng.uniform(0, 2 * np.pi))   # independent draw: delta_phi is tiny (~0.1 rad signal), ignored here
            shots.append(dict(theta_z0=theta_z0, theta_z100=theta_z100,
                               phi_z0=phi_z0, phi_z100=phi_z100, n_tot0=n_tot0, n_tot1=n_tot1))
            count += 1
        if count >= n_shots:
            break
    return shots


if __name__ == '__main__':
    N_SHOTS_CHECK = 10
    acs_z0, acs_z100 = build_acs(PSMAP_TAG_BASELINE)

    print('=== 1e8 (real data: data/R40_N50_A100000000_..._f0.3000) ===')
    shots_1e8 = shots_from_real_data(
        REPO / 'data' / 'R40_N50_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_'
                          'sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000',
        N_SHOTS_CHECK)
    crb_z0_1e8, crb_z100_1e8, _, _ = theta_crb(shots_1e8, acs_z0, acs_z100, verbose=False)
    emp_1e8, emp_1e8_robust, n_out_1e8, n_runs_1e8, n_pts_1e8 = empirical_rmse('1e8', 10, 50)
    emp_1e8_mom, *_ = empirical_rmse('1e8', 10, 50, method='moments')
    emp_1e8_null, *_ = empirical_rmse('1e8', 10, 50, method='null')
    crb_avg_1e8 = 0.5 * (crb_z0_1e8 + crb_z100_1e8)
    print(f'{n_runs_1e8} runs, {n_pts_1e8} points, {n_out_1e8} outlier(s) excluded')
    for name, c, e, em, en in zip(THETA_NAMES, crb_avg_1e8, emp_1e8_robust, emp_1e8_mom, emp_1e8_null):
        print(f'  {name:10s}  CRB={c:.4e}  best={e:.4e} (x{e/c:.2f})  moments={em:.4e} (x{em/c:.2f})  '
              f'null={en:.4e} (x{en/c:.2f})')

    print('\n=== 1e6 (approximation: JSON true_theta/n_tot + random phi -- see docstring) ===')
    shots_1e6 = shots_from_json_random_phi('1e6', N_SHOTS_CHECK, seed=0)
    crb_z0_1e6, crb_z100_1e6, _, _ = theta_crb(shots_1e6, acs_z0, acs_z100, verbose=False)
    emp_1e6, emp_1e6_robust, n_out_1e6, n_runs_1e6, n_pts_1e6 = empirical_rmse('1e6', 10, 50)
    emp_1e6_mom, *_ = empirical_rmse('1e6', 10, 50, method='moments')
    emp_1e6_null, *_ = empirical_rmse('1e6', 10, 50, method='null')
    crb_avg_1e6 = 0.5 * (crb_z0_1e6 + crb_z100_1e6)
    print(f'{n_runs_1e6} runs, {n_pts_1e6} points, {n_out_1e6} outlier(s) excluded')
    for name, c, e, em, en in zip(THETA_NAMES, crb_avg_1e6, emp_1e6_robust, emp_1e6_mom, emp_1e6_null):
        print(f'  {name:10s}  CRB={c:.4e}  best={e:.4e} (x{e/c:.2f})  moments={em:.4e} (x{em/c:.2f})  '
              f'null={en:.4e} (x{en/c:.2f})')

    print('\n=== For reference: same comparison, aberrated confocal_random_zernike (already done) ===')
    print('  (see notes/theta_fisher_crb.tex Table 1 -- all components matched to +-5%)')
