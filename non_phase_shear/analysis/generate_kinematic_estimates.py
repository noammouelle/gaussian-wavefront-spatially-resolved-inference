"""
Generate and save theta (kinematic parameter) estimates for all 4 methods
(null, moments, pixel-likelihood best-config, oracle), across N_RUNS runs x
N_SHOTS shots, for a given dataset. Also saves raw counts needed for the
downstream beta-fit step, so nothing needs recomputing.

Usage: python generate_kinematic_estimates.py <dataset_dir_under_data/> <n_runs> <n_shots> [--label LABEL]

<dataset_dir_under_data/> is the exact directory name generate_data.py
created under data/ (its --run_name, i.e. the run's tag). --label sets the
short tag used to name output files under results/ (defaults to the full
dataset directory name if omitted -- pass a short one for readability,
e.g. --label 1e6).
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np
import h5py
from scipy.special import logsumexp
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]   # repo root: python-scripts/, helpers/, data/, output-files/
OUT  = Path(__file__).resolve().parent.parent  # non_phase_shear/: results/, figures/
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))
import crb_signal as crb
import map_inference as mi
import pixel_acs_grad as pag
from profile_cloud_nuisances import SurrogatePixelACS, SemiAnalyticPixelACS
from helpers import ImageShotDataset
from crb_signal import THETA_NAMES
from run_manifest import log_run

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('dataset', help='dataset directory name under data/ (the run tag)')
p.add_argument('n_runs', type=int)
p.add_argument('n_shots', type=int)
p.add_argument('--label', default=None, help='short tag for output filenames (default: dataset dir name)')
p.add_argument('--psmap_tag', default='CONFOCAL_FINE',
                help="PSMAP tag: reads output-files/PSGRID4D_<tag>_Z{0,100}.h5 "
                     "(default: the analytic confocal beam; use a tag from "
                     "phase_space_grids.py --tag for an arbitrary-wavefront PSMAP)")
args = p.parse_args()

N_RUNS = args.n_runs
N_SHOTS = args.n_shots
LABEL = args.label or args.dataset

prior_mean = np.array([0., 0., 0., 0., 100e-6, 100e-6, 100e-6, 100e-6])
prior_std = np.array([10e-6] * 8)
_H_THETA = np.full(8, 1e-9)

BINS_BEST = 32
TIGHT_HALF_RANGE = 1.54e-03
PIXEL_NGH = 8
M_PHI = 2000

data_root = REPO / 'data' / args.dataset
run_dirs = sorted(data_root.glob('run_*'))[:N_RUNS]
assert run_dirs, f'no run_* dirs found under {data_root}'

# --- best-config (pixel-likelihood, bins=32, tight range) evaluators ---
edges_tight = np.linspace(-TIGHT_HALF_RANGE, TIGHT_HALF_RANGE, BINS_BEST + 1)
psmap_z0 = mi.load_psmap(str(REPO / 'output-files' / f'PSGRID4D_{args.psmap_tag}_Z0.h5'))
psmap_z100 = mi.load_psmap(str(REPO / 'output-files' / f'PSGRID4D_{args.psmap_tag}_Z100.h5'))
sur_z0 = mi.PSMAPSurrogate(psmap_z0, mi.DEFAULT_T_DET, use_gpu=mi.USE_GPU)
sur_z100 = mi.PSMAPSurrogate(psmap_z100, mi.DEFAULT_T_DET, use_gpu=mi.USE_GPU)
base_z0_tight = SurrogatePixelACS(sur_z0, mi.DEFAULT_T_DET, edges_tight, edges_tight, n_quad=1)
base_z100_tight = SurrogatePixelACS(sur_z100, mi.DEFAULT_T_DET, edges_tight, edges_tight, n_quad=1)
acs_z0_best = SemiAnalyticPixelACS(base_z0_tight, n_gh=PIXEL_NGH)
acs_z100_best = SemiAnalyticPixelACS(base_z100_tight, n_gh=PIXEL_NGH)

# --- moments (Kalman-gain) prior/gain setup ---
A_mat = mi.build_A(mi.DEFAULT_T_DET)
m_eta, P_eta = mi.build_prior(mi.DEFAULT_T_DET)
K, A_m_eta = mi.precompute_kalman(m_eta, P_eta, A_mat)


def eta_to_theta(eta):
    """(10,) eta -> (8,) theta-equivalent, discarding cross terms (matches
    the convention already used earlier this session)."""
    return np.array([eta[0], eta[2], eta[1], eta[3],
                      np.sqrt(max(eta[4], 0)), np.sqrt(max(eta[7], 0)),
                      np.sqrt(max(eta[6], 0)), np.sqrt(max(eta[9], 0))])


def downsample_tight(img, half_range_native, bins, half_range_tight):
    res = img.shape[0]
    native_edges = np.linspace(-half_range_native, half_range_native, res + 1)
    native_centers = 0.5 * (native_edges[:-1] + native_edges[1:])
    tight_edges = np.linspace(-half_range_tight, half_range_tight, bins + 1)
    ix = np.clip(np.searchsorted(tight_edges, native_centers) - 1, 0, bins - 1)
    outside = (native_centers < -half_range_tight) | (native_centers > half_range_tight)
    weight = img.astype(np.float64)
    mask = ~np.outer(outside, np.ones(res, dtype=bool)) & ~np.outer(np.ones(res, dtype=bool), outside)
    flat_ix = (ix[:, None] * bins + ix[None, :]).ravel()
    flat_w = np.where(mask, weight, 0.0).ravel()
    out_flat = np.bincount(flat_ix, weights=flat_w, minlength=bins*bins)
    return out_flat.reshape(bins, bins)


def phi_marginal_nll(theta, acs, n_g, n_e, phi_grid, dphi):
    base, dbase = pag.pixel_acs_and_grad(acs, theta, _H_THETA)
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = [b.get() if hasattr(b, 'get') else b for b in base]
    c = np.cos(phi_grid); s = np.sin(phi_grid)
    I_g = A_g[None, :] + Cc_g[None, :]*c[:, None] + Cs_g[None, :]*s[:, None]
    I_e = A_e[None, :] + Cc_e[None, :]*c[:, None] + Cs_e[None, :]*s[:, None]
    n_tot = float(n_g.sum() + n_e.sum())
    L = n_tot / np.maximum((I_g + I_e).sum(axis=1), crb.EPS)
    mu_g = L[:, None] * np.maximum(I_g, crb.EPS)
    mu_e = L[:, None] * np.maximum(I_e, crb.EPS)
    nll_m = (mu_g - n_g[None, :]*np.log(mu_g)).sum(axis=1) + \
            (mu_e - n_e[None, :]*np.log(mu_e)).sum(axis=1)
    return -logsumexp(-nll_m, b=dphi)


PHI_GRID = np.linspace(0, 2*np.pi, M_PHI, endpoint=False)
DPHI = 2*np.pi / M_PHI


def fit_theta_best(acs, n_g, n_e):
    def neg_logL_prior(theta):
        mnll = phi_marginal_nll(theta, acs, n_g, n_e, PHI_GRID, DPHI)
        return mnll + 0.5 * float(np.sum(((theta - prior_mean) / prior_std) ** 2))
    res = minimize(neg_logL_prior, prior_mean.copy(), method='L-BFGS-B', options={'maxiter': 200, 'eps': 1e-8})
    return res.x


out_path = OUT / 'results' / f'kinematic_estimates_{LABEL}_N{N_RUNS}_shots{N_SHOTS}.json'
out_path.parent.mkdir(parents=True, exist_ok=True)

all_runs = {}
t_start = time.perf_counter()
for run_dir in run_dirs:
    run_name = run_dir.name
    t_run0 = time.perf_counter()
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    with h5py.File(str(run_dir / 'Z100' / 'data_IMG.h5')) as f:
        f_signal = float(f.attrs['signal_freq'])
        signal_amp_true = float(f.attrs['signal_amp'])
        signal_phase_true = float(f.attrs['signal_phase'])
    As_true = signal_amp_true * np.cos(signal_phase_true)
    Ac_true = signal_amp_true * np.sin(signal_phase_true)

    shots = []
    for i in range(N_SHOTS):
        img0 = ds_z0[i]; img1 = ds_z100[i]
        n_g0_full = img0[0].astype(np.float64); n_e0_full = img0[1].astype(np.float64)
        n_g1_full = img1[0].astype(np.float64); n_e1_full = img1[1].astype(np.float64)

        meta0 = ds_z0.meta(i); meta1 = ds_z100.meta(i)
        true_theta_z0 = np.array([meta0[k] for k in THETA_NAMES])
        true_theta_z100 = np.array([meta1[k] for k in THETA_NAMES])

        # NULL
        theta_null_z0 = prior_mean.copy(); theta_null_z100 = prior_mean.copy()

        # MOMENTS (Kalman gain)
        _, mu_xf0, mu_yf0, V_xf0, V_yf0 = mi.port_summed_moments(img0, ds_z0.pixel_centers)
        _, mu_xf1, mu_yf1, V_xf1, V_yf1 = mi.port_summed_moments(img1, ds_z100.pixel_centers)
        eta_m_z0 = mi.map_eta(np.array([mu_xf0, mu_yf0, V_xf0, V_yf0]), m_eta, K, A_m_eta)
        eta_m_z100 = mi.map_eta(np.array([mu_xf1, mu_yf1, V_xf1, V_yf1]), m_eta, K, A_m_eta)
        theta_moments_z0 = eta_to_theta(eta_m_z0)
        theta_moments_z100 = eta_to_theta(eta_m_z100)

        # ORACLE
        theta_oracle_z0 = true_theta_z0.copy(); theta_oracle_z100 = true_theta_z100.copy()

        # BEST CONFIG (expensive)
        n_g0_t = downsample_tight(n_g0_full, ds_z0.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        n_e0_t = downsample_tight(n_e0_full, ds_z0.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        n_g1_t = downsample_tight(n_g1_full, ds_z100.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        n_e1_t = downsample_tight(n_e1_full, ds_z100.half_range, BINS_BEST, TIGHT_HALF_RANGE).ravel()
        theta_best_z0 = fit_theta_best(acs_z0_best, n_g0_t, n_e0_t)
        theta_best_z100 = fit_theta_best(acs_z100_best, n_g1_t, n_e1_t)

        shots.append(dict(
            n_g0=float(n_g0_full.sum()), n_e0=float(n_e0_full.sum()),
            n_g1=float(n_g1_full.sum()), n_e1=float(n_e1_full.sum()),
            true_theta_z0=true_theta_z0.tolist(), true_theta_z100=true_theta_z100.tolist(),
            theta_null_z0=theta_null_z0.tolist(), theta_null_z100=theta_null_z100.tolist(),
            theta_moments_z0=theta_moments_z0.tolist(), theta_moments_z100=theta_moments_z100.tolist(),
            theta_best_z0=theta_best_z0.tolist(), theta_best_z100=theta_best_z100.tolist(),
            theta_oracle_z0=theta_oracle_z0.tolist(), theta_oracle_z100=theta_oracle_z100.tolist(),
        ))

    t_run = time.perf_counter() - t_run0
    print(f'{LABEL} {run_name}: t={t_run:.0f}s  (total elapsed {time.perf_counter()-t_start:.0f}s)', flush=True)
    all_runs[run_name] = dict(As_true=As_true, Ac_true=Ac_true, f_signal=f_signal, shots=shots)

    with open(out_path, 'w') as f:
        json.dump(all_runs, f)

print(f'Saved to {out_path}')
log_run(REPO, 'kinematic_estimates', LABEL, dataset=args.dataset,
        n_runs=N_RUNS, n_shots=N_SHOTS, bins_best=BINS_BEST,
        tight_half_range=TIGHT_HALF_RANGE, psmap_tag=args.psmap_tag, output=str(out_path))
print('DONE')
