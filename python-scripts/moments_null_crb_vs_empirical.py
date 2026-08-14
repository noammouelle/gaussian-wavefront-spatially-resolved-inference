"""
moments_null_crb_vs_empirical.py — theoretical (not just empirical) per-shot
error predictions for the 'moments' and 'null' theta estimators, completing
the same CRB-vs-empirical comparison theta_crb_vs_empirical.py already did
for 'best'.

null
----
theta_null is map_eta() with the innovation forced to zero -- i.e. it ALWAYS
returns the prior mean, completely ignoring the image. It uses no
per-shot information, so there is nothing to "propagate": the predicted
RMSE(null) is exactly the population spread of the true theta around the
prior mean, which by construction of the simulator IS the prior width in
map_inference.py:
    mu_x0, mu_y0, mu_vx0, mu_vy0    -> TAU_MU_POS, TAU_MU_VEL exactly
    sigma_x, sigma_y, sigma_vx, sigma_vy -> delta-method sqrt(V): since
        V ~ (BAR_SIG^2 + tau_V * z) for prior draws and sigma=sqrt(V),
        std(sigma) ~= tau_V/(2*BAR_SIG) = SIG_SIG_POS/VEL exactly, because
        build_prior() DEFINES tau_V_pos = 2*BAR_SIG_POS*SIG_SIG_POS for
        exactly this reason (see its docstring).
No new code needed beyond reading these constants off map_inference.py.

moments
-------
theta_moments = map_eta(M, m_eta, K, A_m_eta) = m_eta + K @ (M - A@m_eta) is
an EXACT LINEAR function of the raw 4-vector image moments M = [mu_xf, mu_yf,
V_xf, V_yf] (see map_inference.py's own build_A/precompute_kalman/map_eta).
So its covariance is exactly
    Cov(theta_moments) = K @ Cov(M) @ K.T
-- no finite differences, no Fisher matrix, just the propagation of the
raw-moment measurement noise through the SAME closed-form linear map already
used to produce the estimate.

Cov(M) comes from Poisson pixel-counting statistics. The port-summed pixel
histogram along x (row sums, similarly y) is a sum of independent Poisson
counts n_i ~ Poisson(lambda_i), i=1..bins, with N = sum(n_i). This is the
standard Poisson-thinning / compound-Poisson identity: conditional on N, the
vector (n_i) is EXACTLY Multinomial(N, p_i) with p_i = lambda_i / sum(lambda).
So, conditional on the realized N (which is what we observe, and use), mu_x
and V_x behave exactly like the sample mean/variance of N iid draws from the
pixel distribution p(x). The standard (shape-agnostic, not Gaussian-only)
finite-sample results for a weighted mean/variance estimator are:
    Var(mu)    = V / N
    Cov(mu, V) = mu3 / N          (mu3 = 3rd central moment of p(x))
    Var(V)     = (mu4 - V^2) / N  (mu4 = 4th central moment of p(x))
computed directly from each shot's OWN measured marginal histogram (not
assumed Gaussian, since the PSF-blurred image marginal need not be exactly
so) -- these reduce to the familiar Var(mu)=V/N, Var(V)=2V^2/N only in the
Gaussian limit (mu3=0, mu4=3V^2).
The extra scatter from N itself being random (rather than fixed) is
second-order in 1/N (Var(mu) = E[Var(mu|N)] + Var(E[mu|N]), and the second
term vanishes because E[mu|N] = mu_true regardless of N) -- dropped.
Cross-covariance between the (mu_x,V_x) block and the (mu_y,V_y) block is
exactly zero: x and y central moments of the SAME image are computed from
different marginals (row sums vs column sums) of correlated pixel counts,
so this is an approximation, not exact -- but is the standard leading-order
treatment and is checked against the empirical result below.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python-scripts'))
sys.path.insert(0, str(REPO / 'helpers'))

import map_inference as mi                                            # noqa: E402
from helpers import ImageShotDataset                                   # noqa: E402
from theta_crb_vs_empirical import (                                   # noqa: E402
    DATASET, THETA_NAMES, N_RUNS_EMP, N_SHOTS_EMP, LABEL, empirical_rmse,
)

N_SHOTS_CHECK = 10


def null_theoretical_rmse():
    """Trivial: RMSE(null) IS the prior width, per-component."""
    return np.array([mi.TAU_MU_POS, mi.TAU_MU_POS, mi.TAU_MU_VEL, mi.TAU_MU_VEL,
                      mi.SIG_SIG_POS, mi.SIG_SIG_POS, mi.SIG_SIG_VEL, mi.SIG_SIG_VEL])


def raw_moment_cov(px, centers, mu, V):
    """Var(mu), Cov(mu,V), Var(V) for the weighted mean/variance of one 1D
    marginal histogram, from its own measured central moments (shape-agnostic
    -- not assuming Gaussian)."""
    N = px.sum()
    d = centers - mu
    mu3 = float((px * d ** 3).sum() / N)
    mu4 = float((px * d ** 4).sum() / N)
    var_mu = V / N
    cov_mu_V = mu3 / N
    var_V = (mu4 - V ** 2) / N
    return var_mu, cov_mu_V, var_V


def moments_theta_cov(img, pixel_centers, K):
    """Cov(theta_moments) = K @ Cov(M) @ K.T, M=[mu_xf,mu_yf,V_xf,V_yf]."""
    total = img[0].astype(np.float64) + img[1].astype(np.float64)
    px = total.sum(axis=1); py = total.sum(axis=0)
    N = total.sum()
    mu_x = float((px * pixel_centers).sum() / N)
    mu_y = float((py * pixel_centers).sum() / N)
    V_x = float((px * (pixel_centers - mu_x) ** 2).sum() / N)
    V_y = float((py * (pixel_centers - mu_y) ** 2).sum() / N)

    var_mux, cov_muVx, var_Vx = raw_moment_cov(px, pixel_centers, mu_x, V_x)
    var_muy, cov_muVy, var_Vy = raw_moment_cov(py, pixel_centers, mu_y, V_y)

    Cov_M = np.zeros((4, 4))
    Cov_M[0, 0] = var_mux; Cov_M[0, 2] = Cov_M[2, 0] = cov_muVx; Cov_M[2, 2] = var_Vx
    Cov_M[1, 1] = var_muy; Cov_M[1, 3] = Cov_M[3, 1] = cov_muVy; Cov_M[3, 3] = var_Vy

    return K @ Cov_M @ K.T


def moments_crb(shots_imgs, pixel_centers, K, verbose=True):
    sigmas = []
    for i, img in enumerate(shots_imgs):
        cov = moments_theta_cov(img, pixel_centers, K)
        sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
        sigmas.append(sigma)
        if verbose:
            print(f'shot {i}: sigma_theta_moments={sigma}')
    sigmas = np.array(sigmas)
    return np.sqrt((sigmas ** 2).mean(axis=0)), sigmas


if __name__ == '__main__':
    print('=== null: theoretical RMSE (= prior width, no derivation needed) ===')
    null_theory = null_theoretical_rmse()
    for name, v in zip(THETA_NAMES, null_theory):
        print(f'  {name:10s}  theory={v:.4e}')

    print('\n=== moments: theoretical RMSE via K @ Cov(M) @ K.T (Poisson moment propagation) ===')
    A = mi.build_A(mi.DEFAULT_T_DET)
    m_eta, P_eta = mi.build_prior(mi.DEFAULT_T_DET)
    K, A_m_eta = mi.precompute_kalman(m_eta, P_eta, A)

    data_root = REPO / 'data' / DATASET
    run_dir = sorted(data_root.glob('run_*'))[0]
    ds_z0 = ImageShotDataset(str(run_dir / 'Z0' / 'data_IMG.h5'))
    ds_z100 = ImageShotDataset(str(run_dir / 'Z100' / 'data_IMG.h5'))
    imgs = [ds_z0[i] for i in range(N_SHOTS_CHECK)] + [ds_z100[i] for i in range(N_SHOTS_CHECK)]

    moments_theory, sig_all = moments_crb(imgs, ds_z0.pixel_centers, K)

    print('\n=== Comparison: theoretical vs. empirical, null & moments ===')
    emp_null, _, _, _, _ = empirical_rmse(LABEL, N_RUNS_EMP, N_SHOTS_EMP, method='null')
    emp_moments, _, _, _, _ = empirical_rmse(LABEL, N_RUNS_EMP, N_SHOTS_EMP, method='moments')
    for name, tn, en, tm, em in zip(THETA_NAMES, null_theory, emp_null, moments_theory, emp_moments):
        print(f'  {name:10s}  null: theory={tn:.4e} emp={en:.4e} (x{en / tn:.2f})   '
              f'moments: theory={tm:.4e} emp={em:.4e} (x{em / tm:.2f})')
