import sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))
import joint_profile_inference as jpi

data_root = REPO / 'data' / 'R80_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000'
run_dir = sorted(data_root.glob('run_*'))[0]
prior_mean = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
prior_std = np.array([10e-6] * 8)

for N in [8, 40]:
    print(f'\n########## N={N} ##########')
    t0 = time.perf_counter()
    beta_loop = jpi.fit_beta(run_dir, N, 0.3, prior_mean, prior_std, bins=16, pixel_n_gh=4,
                              n_inner_newton=3, n_starts=1, outer_maxiter=15, use_analytic=True, log=lambda *a: None)
    t_loop = time.perf_counter() - t0
    print(f'  LOOP  beta_hat={beta_loop}  time={t_loop:.2f}s')

    t0 = time.perf_counter()
    beta_batch = jpi.fit_beta_batch(run_dir, N, 0.3, prior_mean, prior_std, bins=16, pixel_n_gh=4,
                                     n_inner_iter=200, n_starts=1, outer_maxiter=15, log=lambda *a: None)
    t_batch = time.perf_counter() - t0
    print(f'  BATCH beta_hat={beta_batch}  time={t_batch:.2f}s')
    print(f'  beta diff = {beta_loop - beta_batch}   speedup = {t_loop/t_batch:.2f}x')
