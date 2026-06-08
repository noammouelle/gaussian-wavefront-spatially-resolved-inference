"""
diag_zero_jitter.py — Diagnose amplitude bias in zero-jitter runs.

Tests:
  1. Fringe residuals vs shot noise (is noise model correct?)
  2. Higher harmonic content in fringe (model misspecification?)
  3. MLE recovery across all zero-jitter datasets
  4. Likelihood at true params vs MLE params
  5. MCMC convergence diagnostics

Run: conda run -n aispy_env python3 diag_zero_jitter.py
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/aispp-sims/gaussian-wavefront-spatially-resolved-inference/helpers/'))

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
import emcee

ROOT = os.path.expanduser('~/aispp-sims/gaussian-wavefront-spatially-resolved-inference/data')

ZERO_JITTER_RUNS = [
    'N100_A100000_muXStd0.0um_muVxStd0.0um_sigX100um_sigVx309um_sigXStd0.0um_sigVxStd0.0um_phi0random_sig_A0.100_f0.3000',
    'N100_A1000000_muXStd0.0um_muVxStd0.0um_sigX100um_sigVx309um_sigXStd0.0um_sigVxStd0.0um_phi0random_sig_A0.100_f0.3000',
    'N100_A10000000_muXStd0.0um_muVxStd0.0um_sigX100um_sigVx309um_sigXStd0.0um_sigVxStd0.0um_phi0random_sig_A0.100_f0.3000',
]

TRUE_AS    = 0.1 * np.sin(0.5)   # signal_amp * sin(signal_phase) but wait...
# delta_phi[i] = 0.1 * sin(2π*0.3*i + 0.5)
# = 0.1 * [sin(0.5)*cos(2π*0.3*i) + cos(0.5)*sin(2π*0.3*i)]
# In model: dphi = phi0 + As*sin(2π*f*t) + Ac*cos(2π*f*t)
# So As = 0.1*cos(0.5), Ac = 0.1*sin(0.5)
TRUE_AMP   = 0.1
TRUE_PHASE = 0.5
TRUE_AS    = TRUE_AMP * np.cos(TRUE_PHASE)   # = 0.0878
TRUE_AC    = TRUE_AMP * np.sin(TRUE_PHASE)   # = 0.0479
TRUE_FREQ  = 0.3
NTHETA     = 512


# ─── helper: load shot counts from HDF5 ───────────────────────────────────────

def load_counts(h5path):
    with h5py.File(h5path) as f:
        shot_idx = f['shot_index'][:]
        states   = f['states'][:]
        phi0     = f['phi0'][:]
        delta_phi = f['delta_phi'][:]
        sigma_x  = f['sigma_x'][:]
        mu_x0    = f['mu_x0'][:]
    n_shots = len(phi0)
    nground = np.zeros(n_shots, dtype=np.int64)
    ntotal  = np.zeros(n_shots, dtype=np.int64)
    for i in range(n_shots):
        mask = shot_idx == i
        s = states[mask]
        ntotal[i]  = len(s)
        nground[i] = int(np.sum(s == 0))
    return nground, ntotal, phi0, delta_phi, sigma_x, mu_x0


# ─── helper: sinusoidal + cos/sin fringe fit ──────────────────────────────────

def fit_fringe_fourier(phi0, ground_frac, harmonics=2):
    """
    Fit P(phi) = A + sum_{k=1}^{harmonics} [a_k cos(k*phi) + b_k sin(k*phi)]
    Returns (A, coeffs_cos, coeffs_sin) for harmonics 1..harmonics.
    """
    N = len(phi0)
    cols = [np.ones(N)]
    for k in range(1, harmonics+1):
        cols.append(np.cos(k * phi0))
        cols.append(np.sin(k * phi0))
    X = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(X, ground_frac, rcond=None)
    pred = X @ coeffs
    resid = ground_frac - pred
    A = coeffs[0]
    cos_k = coeffs[1::2]
    sin_k = coeffs[2::2]
    return A, cos_k, sin_k, pred, resid


# ─── helper: marginal log-likelihood ──────────────────────────────────────────

def total_marginal_ll(n1, n2, N1, N2, A1, C1, A2, C2, dphi_arr, ntheta=NTHETA):
    """Sum over shots of log ∫ dθ/2π p_Z0(n1|θ) p_Z100(n2|θ+δφ)"""
    eps = 1e-12
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)

    p1 = np.clip(A1 + 0.5*C1*np.cos(theta), eps, 1-eps)    # (ntheta,)
    ll_z0 = n1[:, None]*np.log(p1) + (N1[:, None]-n1[:, None])*np.log1p(-p1)  # (nshots, ntheta)

    dphi_arr = np.asarray(dphi_arr)
    p2 = np.clip(A2 + 0.5*C2*np.cos(theta[None, :] + dphi_arr[:, None]), eps, 1-eps)
    ll_z100 = n2[:, None]*np.log(p2) + (N2[:, None]-n2[:, None])*np.log1p(-p2)

    per_shot = logsumexp(ll_z0 + ll_z100, axis=1) - np.log(ntheta)
    return float(np.sum(per_shot))


def signal_dphi(phi0, As, Ac, f, t):
    return phi0 + As*np.sin(2*np.pi*f*t) + Ac*np.cos(2*np.pi*f*t)


def run_mle(n1, n2, N1, N2, t, A1_0, C1_0, A2_0, C2_0, f=TRUE_FREQ, verbose=True):
    """Full 7-parameter MLE: A1, A2, C1, C2, phi0, As, Ac"""
    amp_bound = np.pi

    def neg_ll(p):
        A1, A2, C1, C2, phi0, As, Ac = p
        # physical check
        eps = 1e-8
        if not (A1-C1/2 > eps and A1+C1/2 < 1-eps and
                A2-C2/2 > eps and A2+C2/2 < 1-eps and
                0 <= phi0 <= 2*np.pi and
                -amp_bound <= As <= amp_bound and
                -amp_bound <= Ac <= amp_bound):
            return 1e300
        dphi = signal_dphi(phi0, As, Ac, f, t)
        ll = total_marginal_ll(n1, n2, N1, N2, A1, C1, A2, C2, dphi)
        return -ll if np.isfinite(ll) else 1e300

    # Initial estimates
    x1 = n1 / N1
    x2 = n2 / N2

    # starts: vary As, Ac around zero
    starts = [
        [A1_0, A2_0, C1_0, C2_0, np.pi, 0.0,   0.0  ],
        [A1_0, A2_0, C1_0, C2_0, np.pi, 0.05,  0.0  ],
        [A1_0, A2_0, C1_0, C2_0, np.pi, -0.05, 0.0  ],
        [A1_0, A2_0, C1_0, C2_0, np.pi, 0.0,   0.05 ],
        [A1_0, A2_0, C1_0, C2_0, np.pi, 0.0,  -0.05 ],
        # also start near true
        [A1_0, A2_0, C1_0, C2_0, np.pi, TRUE_AS, TRUE_AC],
    ]

    best = None
    for x0 in starts:
        res = minimize(neg_ll, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-6})
        if best is None or res.fun < best.fun:
            best = res

    A1m, A2m, C1m, C2m, phi0m, Asm, Acm = best.x
    amp_mle   = np.sqrt(Asm**2 + Acm**2)
    phase_mle = np.arctan2(Acm, Asm)

    if verbose:
        print(f"  MLE converged: {best.success}, logL = {-best.fun:.2f}")
        print(f"  As_mle = {Asm:.5f}  (true {TRUE_AS:.5f})")
        print(f"  Ac_mle = {Acm:.5f}  (true {TRUE_AC:.5f})")
        print(f"  amp_mle = {amp_mle:.6f}  (true {TRUE_AMP:.6f})  "
              f"delta = {amp_mle - TRUE_AMP:+.6f}")
        print(f"  phase_mle = {phase_mle:.5f}  (true {TRUE_PHASE:.5f})  "
              f"delta = {phase_mle - TRUE_PHASE:+.5f}")

    return best.x, amp_mle, phase_mle, -best.fun


# ─── helper: ll at true params ────────────────────────────────────────────────

def ll_at_true(n1, n2, N1, N2, A1, C1, A2, C2, t):
    dphi_true = signal_dphi(0.0, TRUE_AS, TRUE_AC, TRUE_FREQ, t)
    # phi0 is marginalized over, so we need phi0 to absorb the constant offset
    # Use phi0=0 plus the model — but actually the fringe has a phase offset.
    # Let's grid-search phi0 to find the best logL at (TRUE_AS, TRUE_AC).
    best_ll = -np.inf
    best_phi0 = 0.0
    for phi0_try in np.linspace(0, 2*np.pi, 128, endpoint=False):
        dphi = signal_dphi(phi0_try, TRUE_AS, TRUE_AC, TRUE_FREQ, t)
        ll = total_marginal_ll(n1, n2, N1, N2, A1, C1, A2, C2, dphi)
        if ll > best_ll:
            best_ll = ll
            best_phi0 = phi0_try
    return best_ll, best_phi0


# ─── main diagnostic loop ─────────────────────────────────────────────────────

for run in ZERO_JITTER_RUNS:
    z0path  = os.path.join(ROOT, run, 'Z0',   'data_PROB.h5')
    z100path = os.path.join(ROOT, run, 'Z100', 'data_PROB.h5')

    if not os.path.exists(z0path):
        print(f"\n[SKIP] {run} — file not found")
        continue

    print(f"\n{'='*70}")
    print(f"Run: {run}")
    print('='*70)

    n1, N1, phi0_z0,   dp_z0,  sx_z0,  mx_z0  = load_counts(z0path)
    n2, N2, phi0_z100, dp_z100, sx_z100, mx_z100 = load_counts(z100path)

    t = np.arange(len(n1), dtype=float)
    delta_phi_true = 0.1 * np.sin(2*np.pi*0.3*t + 0.5)

    # ── 1. Basic stats ─────────────────────────────────────────────────────
    print(f"\n[1] Basic stats")
    print(f"  Z0:   {N1.mean():.0f} atoms/shot  (range {N1.min():.0f}–{N1.max():.0f})")
    print(f"  Z100: {N2.mean():.0f} atoms/shot  (range {N2.min():.0f}–{N2.max():.0f})")
    print(f"  sigma_x Z0 range:  {sx_z0.min()*1e6:.2f}–{sx_z0.max()*1e6:.2f} um  "
          f"(all same: {np.allclose(sx_z0, sx_z0[0])})")
    print(f"  sigma_x Z100 range: {sx_z100.min()*1e6:.2f}–{sx_z100.max()*1e6:.2f} um  "
          f"(all same: {np.allclose(sx_z100, sx_z100[0])})")
    print(f"  mu_x0 Z0 range:   {mx_z0.min()*1e6:.2f}–{mx_z0.max()*1e6:.2f} um  "
          f"(all same: {np.allclose(mx_z0, 0.0)})")

    # ── 2. Fringe residuals ────────────────────────────────────────────────
    print(f"\n[2] Fringe residuals (should be shot-noise only with zero jitter)")
    gf0   = n1 / N1
    gf100 = n2 / N2

    A1_0, cos1, sin1, pred0, resid0 = fit_fringe_fourier(phi0_z0,   gf0,   harmonics=3)
    A2_0, cos2, sin2, pred2, resid2 = fit_fringe_fourier(phi0_z100, gf100, harmonics=3)

    C1_0 = 2 * np.sqrt(cos1[0]**2 + sin1[0]**2)
    C2_0 = 2 * np.sqrt(cos2[0]**2 + sin2[0]**2)

    sigma_shot_z0   = np.sqrt(gf0   * (1-gf0)   / N1)
    sigma_shot_z100 = np.sqrt(gf100 * (1-gf100) / N2)

    noise_z0   = np.std(resid0)
    noise_z100 = np.std(resid2)

    print(f"  Z0   residual std: {noise_z0:.4e}  shot noise: {sigma_shot_z0.mean():.4e}  "
          f"ratio: {noise_z0/sigma_shot_z0.mean():.2f}x")
    print(f"  Z100 residual std: {noise_z100:.4e}  shot noise: {sigma_shot_z100.mean():.4e}  "
          f"ratio: {noise_z100/sigma_shot_z100.mean():.2f}x")
    print(f"  Z0   contrast C1 = {C1_0:.4f}")
    print(f"  Z100 contrast C2 = {C2_0:.4f}")

    # ── 3. Higher harmonic content ─────────────────────────────────────────
    print(f"\n[3] Higher harmonic content (cos amplitudes relative to fundamental)")
    for k in range(1, 4):
        idx = k - 1
        amp_z0   = 2 * np.sqrt(cos1[idx]**2 + sin1[idx]**2) if idx < len(cos1) else 0.0
        amp_z100 = 2 * np.sqrt(cos2[idx]**2 + sin2[idx]**2) if idx < len(cos2) else 0.0
        print(f"  Harmonic {k}: Z0 amp={amp_z0:.5f}  Z100 amp={amp_z100:.5f}  "
              f"(k{k}/k1 Z0={amp_z0/C1_0:.4f}  Z100={amp_z100/C2_0:.4f})")

    # ── 4. MLE recovery ───────────────────────────────────────────────────
    print(f"\n[4] MLE recovery (true amp=0.1, As={TRUE_AS:.4f}, Ac={TRUE_AC:.4f})")
    mle_params, amp_mle, phase_mle, mle_logL = run_mle(
        n1.astype(float), n2.astype(float),
        N1.astype(float), N2.astype(float),
        t, A1_0, C1_0, A2_0, C2_0, verbose=True
    )
    A1m, A2m, C1m, C2m, phi0m, Asm, Acm = mle_params

    # ── 5. Log-likelihood at true vs MLE ──────────────────────────────────
    print(f"\n[5] Log-likelihood comparison")
    print(f"  logL(MLE) = {mle_logL:.4f}")

    # ll at true (As, Ac), optimizing phi0
    ll_true, phi0_best_true = ll_at_true(
        n1.astype(float), n2.astype(float),
        N1.astype(float), N2.astype(float),
        A1m, C1m, A2m, C2m, t
    )
    print(f"  logL(true As,Ac, best phi0, MLE A,C) = {ll_true:.4f}")
    print(f"  delta logL (MLE - true) = {mle_logL - ll_true:.4f}  "
          f"(>0 means MLE is genuinely better)")

    # also try true params completely
    dphi_true_full = signal_dphi(0.0, TRUE_AS, TRUE_AC, TRUE_FREQ, t)
    ll_true_naive = total_marginal_ll(
        n1.astype(float), n2.astype(float),
        N1.astype(float), N2.astype(float),
        A1m, C1m, A2m, C2m, dphi_true_full
    )
    print(f"  logL(true As,Ac, phi0=0, MLE A,C) = {ll_true_naive:.4f}")

    # ── 6. ntheta sensitivity ─────────────────────────────────────────────
    print(f"\n[6] ntheta sensitivity (MLE logL as function of ntheta)")
    dphi_mle = signal_dphi(phi0m, Asm, Acm, TRUE_FREQ, t)
    for nt in [64, 128, 256, 512, 1024, 2048]:
        ll_nt = total_marginal_ll(
            n1.astype(float), n2.astype(float),
            N1.astype(float), N2.astype(float),
            A1m, C1m, A2m, C2m, dphi_mle, ntheta=nt
        )
        print(f"  ntheta={nt:5d}  logL={ll_nt:.4f}")

    # ── 7. Fisher-based posterior width estimate ──────────────────────────
    print(f"\n[7] Expected posterior width (Fisher information)")
    # Estimate from Hessian at MLE using finite differences
    eps_fd = 1e-5
    def ll7(p):
        A1, A2, C1, C2, phi0, As, Ac = p
        if not (A1-C1/2 > 1e-8 and A1+C1/2 < 1-1e-8 and
                A2-C2/2 > 1e-8 and A2+C2/2 < 1-1e-8 and
                0 <= phi0 <= 2*np.pi):
            return -1e300
        dphi = signal_dphi(phi0, As, Ac, TRUE_FREQ, t)
        return total_marginal_ll(n1.astype(float), n2.astype(float),
                                  N1.astype(float), N2.astype(float),
                                  A1, C1, A2, C2, dphi)

    # Diagonal Hessian for As and Ac only (indices 5, 6)
    for idx_p, pname in [(5, 'As'), (6, 'Ac')]:
        p_plus  = mle_params.copy(); p_plus[idx_p]  += eps_fd
        p_minus = mle_params.copy(); p_minus[idx_p] -= eps_fd
        d2 = (ll7(p_plus) + ll7(p_minus) - 2*mle_logL) / eps_fd**2
        if d2 < 0:
            sigma = 1.0 / np.sqrt(-d2)
            print(f"  sigma_{pname} (Fisher) ≈ {sigma:.2e} rad")
        else:
            print(f"  sigma_{pname} (Fisher): curvature not negative ({d2:.2e})")

    sigma_amp = np.sqrt(Asm**2/(Asm**2+Acm**2) * (1/abs((ll7(mle_params+np.array([0,0,0,0,0,eps_fd,0])) + ll7(mle_params-np.array([0,0,0,0,0,eps_fd,0])) - 2*mle_logL)/eps_fd**2)) +
                        Acm**2/(Asm**2+Acm**2) * (1/abs((ll7(mle_params+np.array([0,0,0,0,0,0,eps_fd])) + ll7(mle_params-np.array([0,0,0,0,0,0,eps_fd])) - 2*mle_logL)/eps_fd**2)))

    print(f"  Expected sigma_amp ≈ {sigma_amp:.2e} rad")
    print(f"  Bias = {amp_mle - TRUE_AMP:+.6f} rad  ({abs(amp_mle-TRUE_AMP)/sigma_amp:.1f}σ from truth)")

    print()


print("\n=== SUMMARY ===")
print("Key question: does amp_mle converge to 0.1 with more atoms?")
print("If bias/sigma_amp shrinks as 1/sqrt(N_atoms) → statistical fluctuation")
print("If bias stays constant → systematic misspecification or optimizer bug")
