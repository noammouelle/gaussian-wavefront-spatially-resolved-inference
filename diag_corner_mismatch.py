"""
Diagnostic: why do corner-plot centers not match MLE estimates?

Tests:
  1. Batch vs scalar likelihood consistency
  2. MLE is at a true optimum (verify gradient numerically)
  3. Derived (amp, phase) bias: Rice distribution effect
  4. Short MCMC vs MLE agreement with synthetic data
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize, check_grad
import emcee

rng = np.random.default_rng(42)

# ── simulation parameters (mirror the data-generation setup) ──────────────────
TRUE_A1   = 0.50
TRUE_A2   = 0.52
TRUE_C1   = 0.94
TRUE_C2   = 0.76
TRUE_PHI0 = 1.93
TRUE_AS   = 0.10   # gives amp ≈ 0.10 rad at f = 0.3
TRUE_AC   = 0.00
TRUE_F    = 0.30
TRUE_AMP  = np.sqrt(TRUE_AS**2 + TRUE_AC**2)

NSHOTS   = 200          # smaller than the real data for speed
N_ATOMS  = 10_000       # per shot per arm
NTHETA   = 512
AMP_BOUND = np.pi

# ── generate synthetic observations ──────────────────────────────────────────
t_arr = np.arange(NSHOTS, dtype=float)
true_dphi = TRUE_PHI0 + TRUE_AS * np.sin(2*np.pi*TRUE_F*t_arr) \
                       + TRUE_AC * np.cos(2*np.pi*TRUE_F*t_arr)

N1_arr = np.full(NSHOTS, N_ATOMS, dtype=float)
N2_arr = np.full(NSHOTS, N_ATOMS, dtype=float)

theta_i = rng.uniform(0, 2*np.pi, NSHOTS)          # latent phases

p1_true = TRUE_A1 + 0.5 * TRUE_C1 * np.cos(theta_i)
p2_true = TRUE_A2 + 0.5 * TRUE_C2 * np.cos(theta_i + true_dphi)

n1_arr = rng.binomial(N_ATOMS, p1_true).astype(float)
n2_arr = rng.binomial(N_ATOMS, p2_true).astype(float)


# ── likelihood implementation ─────────────────────────────────────────────────

theta_grid = np.linspace(0, 2*np.pi, NTHETA, endpoint=False)   # (ntheta,)

def log_binomial_noconst(n, N, p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return n * np.log(p) + (N - n) * np.log1p(-p)

def valid7(p):
    A1, A2, C1, C2, phi0, As, Ac = p
    return (
        0 < A1 - C1/2 and A1 + C1/2 < 1 and
        0 < A2 - C2/2 and A2 + C2/2 < 1 and
        0 <= phi0 <= 2*np.pi and
        -AMP_BOUND <= As <= AMP_BOUND and
        -AMP_BOUND <= Ac <= AMP_BOUND
    )

def total_ll_scalar(params, f=TRUE_F):
    """Scalar version – one parameter vector at a time."""
    A1, A2, C1, C2, phi0, As, Ac = params
    if not valid7(params):
        return -np.inf

    dphi = phi0 + As*np.sin(2*np.pi*f*t_arr) + Ac*np.cos(2*np.pi*f*t_arr)
    # shapes: (nshots,1) and (1,ntheta)
    th = theta_grid[None, :]           # (1, ntheta)
    dp = dphi[:, None]                 # (nshots, 1)

    p1 = A1 + 0.5*C1*np.cos(th)       # (1, ntheta)  – broadcasts over shots
    p2 = A2 + 0.5*C2*np.cos(th + dp)  # (nshots, ntheta)

    ll1 = log_binomial_noconst(n1_arr[:, None], N1_arr[:, None], p1)
    ll2 = log_binomial_noconst(n2_arr[:, None], N2_arr[:, None], p2)

    per_shot = logsumexp(ll1 + ll2, axis=1) - np.log(NTHETA)
    return float(np.sum(per_shot))

def total_ll_batch(params_batch, f=TRUE_F):
    """
    Vectorized batch version – mirrors the notebook's
    total_marginal_ll_sinusoid_batch exactly (NumPy only).
    Returns array of shape (n_params,).
    """
    params_batch = np.atleast_2d(np.asarray(params_batch, dtype=float))
    out = np.full(params_batch.shape[0], -np.inf)

    ok = np.array([valid7(p) for p in params_batch])
    if not np.any(ok):
        return out

    p = params_batch[ok]                     # (nbatch, 7)
    A1   = p[:, 0][None, None, :]            # (1, 1, nbatch)
    A2   = p[:, 1][None, None, :]
    C1   = p[:, 2][None, None, :]
    C2   = p[:, 3][None, None, :]
    phi0 = p[:, 4][None, None, :]
    As   = p[:, 5][None, None, :]
    Ac   = p[:, 6][None, None, :]

    tt = t_arr[:, None, None]                # (nshots, 1, 1)
    th = theta_grid[None, :, None]           # (1, ntheta, 1)

    dphi  = phi0 + As*np.sin(2*np.pi*f*tt) + Ac*np.cos(2*np.pi*f*tt)
    prob1 = A1 + 0.5*C1*np.cos(th)          # (1, ntheta, nbatch)
    prob2 = A2 + 0.5*C2*np.cos(th + dphi)   # (nshots, ntheta, nbatch)

    ll1 = log_binomial_noconst(n1_arr[:, None, None], N1_arr[:, None, None], prob1)
    ll2 = log_binomial_noconst(n2_arr[:, None, None], N2_arr[:, None, None], prob2)

    logLi = logsumexp(ll1 + ll2, axis=1) - np.log(NTHETA)
    logL  = np.sum(logLi, axis=0)

    out[ok] = logL
    return out


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: scalar vs batch consistency
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1 – scalar vs batch consistency")

params_test = np.array([TRUE_A1, TRUE_A2, TRUE_C1, TRUE_C2,
                         TRUE_PHI0, TRUE_AS, TRUE_AC])

ll_scalar = total_ll_scalar(params_test)
ll_batch  = total_ll_batch(params_test[None, :])[0]

print(f"  scalar logL = {ll_scalar:.6f}")
print(f"  batch  logL = {ll_batch:.6f}")
print(f"  difference  = {abs(ll_scalar - ll_batch):.2e}")
assert abs(ll_scalar - ll_batch) < 1e-6, "FAIL: scalar vs batch mismatch!"
print("  PASS")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: batch at multiple params – does order matter?
# ═══════════════════════════════════════════════════════════════════════
print()
print("TEST 2 – batch order independence")

p_a = params_test + np.array([0.01, 0, 0, 0, 0, 0, 0])
p_b = params_test + np.array([0, 0.02, 0, 0, 0, 0, 0])
p_c = params_test

batch_ab  = total_ll_batch(np.stack([p_a, p_b]))
batch_ba  = total_ll_batch(np.stack([p_b, p_a]))
batch_abc = total_ll_batch(np.stack([p_a, p_b, p_c]))

print(f"  [a,b] = {batch_ab}")
print(f"  [b,a] = {batch_ba}")
print(f"  [a,b,c][0] = {batch_abc[0]:.6f}  (should match [a,b][0] = {batch_ab[0]:.6f})")
print(f"  [a,b,c][1] = {batch_abc[1]:.6f}  (should match [a,b][1] = {batch_ab[1]:.6f})")
err_order = max(abs(batch_ab[0] - batch_ba[1]),
                abs(batch_ab[1] - batch_ba[0]),
                abs(batch_abc[0] - batch_ab[0]),
                abs(batch_abc[1] - batch_ab[1]))
if err_order > 1e-6:
    print(f"  FAIL: order matters! max diff = {err_order:.2e}")
else:
    print("  PASS")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: MLE optimization – does it recover truth?
# ═══════════════════════════════════════════════════════════════════════
print()
print("TEST 3 – MLE optimization vs truth")

bounds7 = [
    (1e-4, 1 - 1e-4),
    (1e-4, 1 - 1e-4),
    (1e-4, 1.999),
    (1e-4, 1.999),
    (0.0, 2*np.pi),
    (-AMP_BOUND, AMP_BOUND),
    (-AMP_BOUND, AMP_BOUND),
]

def neg_ll(params):
    val = total_ll_batch(params[None, :])[0]
    return -val if np.isfinite(val) else 1e300

best_res = None
for _ in range(5):
    x0 = params_test + 1e-2 * rng.normal(size=7)
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds7,
                   options={"maxiter": 1000})
    if best_res is None or res.fun < best_res.fun:
        best_res = res

mle_params = best_res.x
mle_logL   = -best_res.fun

labels7 = ["A1", "A2", "C1", "C2", "phi0", "As", "Ac"]
print(f"  MLE logL = {mle_logL:.4f}")
print(f"  True logL = {total_ll_scalar(params_test):.4f}")
print(f"  MLE params:")
for name, mle_v, true_v in zip(labels7, mle_params, params_test):
    print(f"    {name:6s}: mle={mle_v:8.5f}  truth={true_v:8.5f}  diff={mle_v-true_v:+.5f}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Is the MLE actually at the MAP/posterior peak?
#         Perturb the MLE and see if log-prob decreases.
# ═══════════════════════════════════════════════════════════════════════
print()
print("TEST 4 – MLE is a local maximum (numerical gradient check)")

ll_at_mle = total_ll_scalar(mle_params)
directions = np.eye(7)
h = 1e-4

all_higher = []
for i, d in enumerate(directions):
    ll_plus  = total_ll_scalar(mle_params + h * d)
    ll_minus = total_ll_scalar(mle_params - h * d)
    grad_i = (ll_plus - ll_minus) / (2 * h)
    # at a maximum, gradient should be near 0; also ll at +/-h should be < mle
    is_max = ll_plus < ll_at_mle + 1e-4 and ll_minus < ll_at_mle + 1e-4
    if not is_max:
        all_higher.append((labels7[i], ll_plus - ll_at_mle, ll_minus - ll_at_mle))
    print(f"  d{labels7[i]:6s}: grad={grad_i:+.4f}  "
          f"Δll(+h)={ll_plus-ll_at_mle:+.4f}  Δll(-h)={ll_minus-ll_at_mle:+.4f}  "
          f"{'LOCAL MAX?' if not is_max else 'ok'}")

if all_higher:
    print(f"  >> MLE is NOT at a local maximum in {len(all_higher)} direction(s)!")
    for name, dp, dm in all_higher:
        print(f"     {name}: Δll(+h)={dp:+.4f}, Δll(-h)={dm:+.4f}")
else:
    print("  MLE appears to be at a local maximum in all 7 directions.")


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Rice/amplitude bias – derived quantity shift
# ═══════════════════════════════════════════════════════════════════════
print()
print("TEST 5 – derived quantity bias (Rice distribution)")

# Simulate what MCMC gives for (As, Ac) near the MLE with Gaussian uncertainty
As_mle = mle_params[5]
Ac_mle = mle_params[6]

# Estimate uncertainty from curvature (finite differences)
dAs2 = (total_ll_scalar(mle_params + h*directions[5]) +
         total_ll_scalar(mle_params - h*directions[5]) -
         2*ll_at_mle) / h**2
sigma_As = np.sqrt(-1.0 / dAs2) if dAs2 < 0 else np.nan

dAc2 = (total_ll_scalar(mle_params + h*directions[6]) +
         total_ll_scalar(mle_params - h*directions[6]) -
         2*ll_at_mle) / h**2
sigma_Ac = np.sqrt(-1.0 / dAc2) if dAc2 < 0 else np.nan

print(f"  MLE: As={As_mle:.4f}, Ac={Ac_mle:.4f}")
print(f"  MLE: amp={np.sqrt(As_mle**2 + Ac_mle**2):.4f}, "
      f"phase={np.arctan2(Ac_mle, As_mle):.4f}")
print(f"  Approx 1σ uncertainty: σ_As≈{sigma_As:.4f}, σ_Ac≈{sigma_Ac:.4f}")

# Simulate samples as if MCMC is Gaussian around MLE
if np.isfinite(sigma_As) and np.isfinite(sigma_Ac):
    n_sim = 50_000
    As_sim = rng.normal(As_mle, sigma_As, n_sim)
    Ac_sim = rng.normal(Ac_mle, sigma_Ac, n_sim)
    amp_sim   = np.sqrt(As_sim**2 + Ac_sim**2)
    phase_sim = np.arctan2(Ac_sim, As_sim)
    amp_mle   = np.sqrt(As_mle**2 + Ac_mle**2)
    phase_mle = np.arctan2(Ac_mle, As_mle)

    print(f"  Simulated sample median(amp)   = {np.median(amp_sim):.4f}  "
          f"(MLE amp = {amp_mle:.4f}, bias = {np.median(amp_sim)-amp_mle:+.4f})")
    print(f"  Simulated sample median(phase) = {np.median(phase_sim):.4f}  "
          f"(MLE phase = {phase_mle:.4f}, bias = {np.median(phase_sim)-phase_mle:+.4f})")
    if abs(np.median(amp_sim) - amp_mle) > 0.01:
        print("  >> Rice bias is non-trivial! amp distribution center ≠ MLE amp.")
    else:
        print("  Rice bias is small for this signal strength.")
else:
    print("  Could not estimate uncertainty (MLE likely not at optimum)")


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Short MCMC – do sample medians agree with MLE?
# ═══════════════════════════════════════════════════════════════════════
print()
print("TEST 6 – short MCMC vs MLE (with synthetic data)")

def log_prob_vectorized(params_batch):
    params_batch = np.atleast_2d(np.asarray(params_batch, dtype=float))
    out = np.full(params_batch.shape[0], -np.inf)
    ok = np.array([valid7(p) for p in params_batch])
    if not np.any(ok):
        return out
    ll = total_ll_batch(params_batch[ok])
    ok_idx = np.flatnonzero(ok)
    good = np.isfinite(ll)
    out[ok_idx[good]] = ll[good]  # flat prior → log_prob = ll
    return out

nwalkers = 32
nsteps   = 800
burn     = 200
thin     = 4
ndim     = 7
scales   = np.array([1e-3]*7)

p0 = np.empty((nwalkers, ndim))
for w in range(nwalkers):
    while True:
        trial = mle_params + scales * rng.normal(size=ndim)
        if valid7(trial):
            p0[w] = trial
            break

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_vectorized,
                                 vectorize=True)
sampler.run_mcmc(p0, nsteps, progress=True)

samples = sampler.get_chain(discard=burn, thin=thin, flat=True)

print(f"\n  MCMC samples: {len(samples)}")
print(f"  {'param':8s}  {'MLE':>10s}  {'med(post)':>10s}  {'diff':>10s}")
for i, name in enumerate(labels7):
    med = np.median(samples[:, i])
    diff = med - mle_params[i]
    flag = "  <-- MISMATCH" if abs(diff) > 0.02 else ""
    print(f"  {name:8s}  {mle_params[i]:10.5f}  {med:10.5f}  {diff:+10.5f}{flag}")

# Also check derived quantities
As_samp    = samples[:, 5]
Ac_samp    = samples[:, 6]
amp_samp   = np.sqrt(As_samp**2 + Ac_samp**2)
phase_samp = np.arctan2(Ac_samp, As_samp)

amp_mle   = np.sqrt(mle_params[5]**2 + mle_params[6]**2)
phase_mle = np.arctan2(mle_params[6], mle_params[5])

print(f"\n  Derived:")
print(f"  {'amp':8s}  {amp_mle:10.5f}  {np.median(amp_samp):10.5f}  "
      f"{np.median(amp_samp)-amp_mle:+10.5f}  <-- Rice bias?")
print(f"  {'phase':8s}  {phase_mle:10.5f}  {np.median(phase_samp):10.5f}  "
      f"{np.median(phase_samp)-phase_mle:+10.5f}")

# ── autocorrelation times ──────────────────────────────────────────────
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"\n  Autocorrelation times: {np.round(tau, 1)}")
    print(f"  Effective samples:     {np.round(len(samples) / tau, 0)}")
    if np.any(tau > (nsteps - burn) / thin / 5):
        print("  >> WARNING: chain may not be converged (tau > n_eff/5)!")
except Exception as e:
    print(f"  (autocorr estimate failed: {e})")

print()
print("Done.")
