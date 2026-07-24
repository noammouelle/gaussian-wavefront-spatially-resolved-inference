"""
Consumes the saved kinematic_estimates_<dataset>_N{n}_shots{s}.json (from
generate_kinematic_estimates.py) and runs the beta-fit pipeline
(batch_acs_gh + total_logL_fast + L-BFGS-B, phi_method='point') for all 4
methods (null, moments, best, oracle), across all runs. No re-fitting of
theta needed -- pure reuse of already-computed, saved estimates. Saves
beta_hat per method per run, and prints RMSE tables.

Usage: python beta_fits_from_kinematics.py <1e6|1e8> <n_runs> <n_shots>
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))
import map_inference as mi
from helpers import ImageShotDataset

DATASET_KEY = sys.argv[1]
N_RUNS = int(sys.argv[2])
N_SHOTS = int(sys.argv[3])

GH_ORDER = 12
GH_CHUNK = 20
BINS_BETA = 16

in_path = REPO / 'results' / f'kinematic_estimates_{DATASET_KEY}_N{N_RUNS}_shots{N_SHOTS}.json'
with open(in_path) as f:
    data = json.load(f)
run_names = sorted(data.keys())
print(f'Loaded {len(run_names)} runs from {in_path}', flush=True)

DATASETS = {
    '1e6': 'R80_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000',
    '1e8': 'R40_N50_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000',
}
data_root = REPO / 'data' / DATASETS[DATASET_KEY]
run_dir0 = sorted(data_root.glob('run_*'))[0]

psmap_z0 = mi.load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_FINE_Z0.h5'))
psmap_z100 = mi.load_psmap(str(REPO / 'output-files' / 'PSGRID4D_CONFOCAL_FINE_Z100.h5'))
_ds_tmp = ImageShotDataset(str(run_dir0 / 'Z0' / 'data_IMG.h5'))
edges = np.linspace(-_ds_tmp.half_range, _ds_tmp.half_range, BINS_BETA + 1)
sur_z0 = mi.PSMAPSurrogate(psmap_z0, mi.DEFAULT_T_DET, use_gpu=mi.USE_GPU)
sur_z100 = mi.PSMAPSurrogate(psmap_z100, mi.DEFAULT_T_DET, use_gpu=mi.USE_GPU)
acs_beta_z0 = mi.SurrogatePixelACS(sur_z0, mi.DEFAULT_T_DET, edges, edges, n_quad=1)
acs_beta_z100 = mi.SurrogatePixelACS(sur_z100, mi.DEFAULT_T_DET, edges, edges, n_quad=1)


def theta_to_eta(theta):
    return np.array([theta[0], theta[2], theta[1], theta[3],
                      theta[4]**2, 0.0, theta[6]**2,
                      theta[5]**2, 0.0, theta[7]**2])


def fit_beta_given_eta(eta_z0_list, eta_z100_list, n_g0_arr, n_e0_arr, n_g1_arr, n_e1_arr,
                        f_signal, shot_idx_arr, n_starts=8, grid_half=0.2, seed=0):
    acs_z0_arr = mi.batch_acs_gh(eta_z0_list, acs_beta_z0, GH_ORDER, chunk_shots=GH_CHUNK)
    acs_z100_arr = mi.batch_acs_gh(eta_z100_list, acs_beta_z100, GH_ORDER, chunk_shots=GH_CHUNK)

    def neg_logL(beta):
        return -mi.total_logL_fast(beta, acs_z0_arr, n_g0_arr, n_e0_arr, acs_z100_arr, n_g1_arr, n_e1_arr,
                                    f_signal, shot_idx_arr, phi_method='point', n_scan=64, n_newton=12)

    rng = np.random.default_rng(seed)
    starts = [np.zeros(2)] + list(rng.uniform(-grid_half, grid_half, size=(max(0, n_starts - 1), 2)))
    best = None
    for b0 in starts[:n_starts]:
        res = minimize(neg_logL, b0, method='L-BFGS-B', bounds=[(-grid_half, grid_half)]*2,
                       options={'maxiter': 300})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


METHODS = ['null', 'moments', 'best', 'oracle']
results = {m: [] for m in METHODS}

for run_name in run_names:
    run = data[run_name]
    shots = run['shots']
    n_g0_arr = np.array([s['n_g0'] for s in shots]); n_e0_arr = np.array([s['n_e0'] for s in shots])
    n_g1_arr = np.array([s['n_g1'] for s in shots]); n_e1_arr = np.array([s['n_e1'] for s in shots])
    shot_idx_arr = np.arange(len(shots), dtype=np.float64)

    for method in METHODS:
        eta_z0_list = [theta_to_eta(np.array(s[f'theta_{method}_z0'])) for s in shots]
        eta_z100_list = [theta_to_eta(np.array(s[f'theta_{method}_z100'])) for s in shots]
        beta_hat = fit_beta_given_eta(eta_z0_list, eta_z100_list, n_g0_arr, n_e0_arr, n_g1_arr, n_e1_arr,
                                       run['f_signal'], shot_idx_arr)
        results[method].append(dict(run=run_name, As_true=run['As_true'], Ac_true=run['Ac_true'],
                                     beta=beta_hat.tolist()))
        print(f'{DATASET_KEY} {run_name} [{method}]: beta_hat={beta_hat}  '
              f'true=({run["As_true"]:.5f},{run["Ac_true"]:.5f})', flush=True)

out_path = REPO / 'results' / f'beta_fits_{DATASET_KEY}_N{N_RUNS}_shots{N_SHOTS}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n=== RMSE summary, {DATASET_KEY}, N_RUNS={N_RUNS} ===')
for method in METHODS:
    rows = results[method]
    errs = np.array([[r['beta'][0]-r['As_true'], r['beta'][1]-r['Ac_true']] for r in rows])
    rmse = np.sqrt((errs**2).mean(axis=0))
    print(f'  {method:10s}: RMSE(As)={rmse[0]:.6f}  RMSE(Ac)={rmse[1]:.6f}')
print('DONE')
