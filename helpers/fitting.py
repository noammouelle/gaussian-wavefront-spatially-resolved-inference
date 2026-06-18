"""
fitting.py — MLE fitting routines for the MAGIS-100 sinusoidal phase signal.

Two-stage MLE strategy (robust against the two-basin phi0 landscape):

  Stage 1 — coarse phi0 grid search (ntheta=512, fast)
    Sweeps phi0 over n_phi0_grid equally-spaced values in [0, 2π],
    with a few (As, Ac) starting values at each.  All 7 parameters are
    free — this is NOT a scan with fixed A/C.  Finds the correct basin.

  Stage 2 — fine polish (adaptive ntheta)
    Refines the best Stage-1 solution with tight convergence criteria.

WHY NOT use the null-fit phi0 as a starting point:
    The null model absorbs the signal's variance into phi0 and C2, shifting
    phi0 to the wrong basin (typically ~π away from the true fringe offset).
    Starting the 7-param MLE from there lands at a boundary minimum
    (As ≈ 0, wrong phase).  The grid search avoids this entirely.

As ≥ 0 is enforced in valid_params7() to break the exact π-phase degeneracy
L(δφ) = L(−δφ).  This makes the recovered phase unique in (−π/2, π/2].

Usage
-----
    from likelihood import LikelihoodEvaluator
    from fitting import fit_mle, fit_from_datasets, FitResult

    # From raw count arrays
    ev = LikelihoodEvaluator(n1, n2, N1, N2)
    result = fit_mle(ev, f=0.3)
    print(result.amp, result.phase)   # 0.0995, 0.505

    # From LazyShotDataset objects (convenience)
    result = fit_from_datasets(Z0, Z100, f=0.3)

    # Loop over many runs to build MLE distribution
    results = [fit_from_datasets(load_run(i, 'Z0'), load_run(i, 'Z100'), f=0.3)
               for i in range(n_runs)]
    amps = np.array([r.amp for r in results])
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from likelihood import LikelihoodEvaluator, FeatureConditionedLikelihoodEvaluator, adaptive_ntheta

_AMP_BOUND_DEFAULT = np.pi


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """
    MLE estimates for the 7-parameter sinusoidal differential-phase model.

    Attributes
    ----------
    A1, A2 : float   Fringe DC offsets (ground-state probability at mid-fringe)
    C1, C2 : float   Fringe contrasts (peak-to-peak amplitude / 2)
    phi0   : float   Fringe phase offset [rad]
    As, Ac : float   Signal sine/cosine amplitudes [rad]
    amp    : float   Signal amplitude  = sqrt(As²+Ac²)  [rad]
    phase  : float   Signal phase      = arctan2(Ac, As) ∈ (−π/2, π/2] with As≥0
    logL   : float   Log-likelihood at MLE
    ntheta : int     Theta quadrature points used
    f      : float   Signal frequency [cycles per time unit]
    converged : bool Nelder-Mead convergence flag (Stage 2)
    """
    A1: float
    A2: float
    C1: float
    C2: float
    phi0: float
    As: float
    Ac: float
    amp: float
    phase: float
    logL: float
    ntheta: int
    f: float
    converged: bool
    feature_names: tuple = ()
    feature_names_z0: tuple = ()
    feature_names_z100: tuple = ()
    feature_names_phase: tuple = ()
    feature_nuisance: tuple = ()
    beta_phi: tuple = ()
    beta_A1: tuple = ()
    beta_A2: tuple = ()
    beta_C1: tuple = ()
    beta_C2: tuple = ()
    beta_phi_prior_std: float = np.nan
    beta_A_prior_std: float = np.nan
    beta_C_prior_std: float = np.nan
    beta_penalty: float = 0.0
    log_posterior: float = np.nan


# ── parameter validity ────────────────────────────────────────────────────────

def valid_params7(p, amp_bound=_AMP_BOUND_DEFAULT):
    """
    Return True iff (A1,A2,C1,C2,phi0,As,Ac) lie in the physical region.

    Enforces:
    - 0 < A ± C/2 < 1  (fringe probabilities stay in (0,1))
    - 0 ≤ phi0 ≤ 2π
    - 0 ≤ As ≤ amp_bound  (As ≥ 0 breaks the π-phase degeneracy)
    - |Ac| ≤ amp_bound
    """
    A1, A2, C1, C2, phi0, As, Ac = p
    return (
        0 < A1 - C1/2 and A1 + C1/2 < 1 and
        0 < A2 - C2/2 and A2 + C2/2 < 1 and
        0.0 <= phi0 <= 2 * np.pi and
        0.0 <= As <= amp_bound and
        -amp_bound <= Ac <= amp_bound
    )


def valid_params5(p):
    """Return True iff (A1,A2,C1,C2,phi0) are physically valid."""
    A1, A2, C1, C2, phi0 = p
    return (
        0 < A1 - C1/2 and A1 + C1/2 < 1 and
        0 < A2 - C2/2 and A2 + C2/2 < 1 and
        0.0 <= phi0 <= 2 * np.pi
    )


# ── initial-estimate helpers ──────────────────────────────────────────────────

def _ols_ac(x, min_c=1e-3):
    """Estimate fringe A and C from a ground-fraction array via OLS."""
    A = float(np.mean(x))
    C = float(2 * np.sqrt(2) * np.std(x))
    C = float(np.clip(C, min_c, 0.90 * 2 * min(A, 1 - A)))
    return A, C


# ── null fit ──────────────────────────────────────────────────────────────────

def fit_null(ev: LikelihoodEvaluator):
    """
    5-parameter null-model MLE: constant differential phase.

    Fits (A1, A2, C1, C2, phi0) to maximise the marginal log-likelihood
    with δφ_i = phi0 for all shots.

    Parameters
    ----------
    ev : LikelihoodEvaluator

    Returns
    -------
    A1, A2, C1, C2, phi0 : float  — MLE estimates
    logL : float                   — log-likelihood at MLE
    """
    x1 = ev.n1_np / ev.N1_np
    x2 = ev.n2_np / ev.N2_np
    A1_0, C1_0 = _ols_ac(x1)
    A2_0, C2_0 = _ols_ac(x2)
    rho    = float(np.clip(np.corrcoef(x1 - A1_0, x2 - A2_0)[0, 1], -1.0, 1.0))
    phi0_0 = float(np.arccos(rho))

    def neg_nll(p):
        if not valid_params5(p):
            return 1e300
        v = ev.null_ll(*p)
        return -v if np.isfinite(v) else 1e300

    best = None
    for ph in np.unique([phi0_0, 2 * np.pi - phi0_0, 0.0, np.pi]):
        r = minimize(neg_nll,
                     np.array([A1_0, A2_0, C1_0, C2_0, ph]),
                     method='Nelder-Mead',
                     options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-3})
        if best is None or r.fun < best.fun:
            best = r

    A1, A2, C1, C2, phi0 = best.x
    return float(A1), float(A2), float(C1), float(C2), float(phi0), -float(best.fun)


# ── signal MLE ────────────────────────────────────────────────────────────────

def fit_mle(
    ev: LikelihoodEvaluator,
    f: float,
    amp_bound: float = _AMP_BOUND_DEFAULT,
    n_phi0_grid: int = 8,
    as_starts: tuple = (0.03, 0.07, 0.10),
    fast: bool = False,
) -> FitResult:
    """
    Two-stage MLE for the 7-parameter sinusoidal differential-phase model.

    Stage 1: coarse phi0 grid search (ntheta=512) to escape the wrong basin.
    Stage 2: fine polish with adaptive ntheta from the best Stage-1 result.

    Parameters
    ----------
    ev : LikelihoodEvaluator
        Pre-built evaluator (holds data on device).
    f : float
        Signal frequency [cycles per time unit].
    amp_bound : float
        Upper bound on |As|, |Ac|.  Default: π.
    n_phi0_grid : int
        Number of phi0 starting values swept in Stage 1.  Default: 8.
    as_starts : tuple of float
        As starting values tried at each phi0 grid point.  Default: (0.03, 0.07, 0.10).
    fast : bool
        Speed-optimised mode for repeated fitting (bootstrap / multi-run studies).
        Uses fewer grid points, fewer perturbations, and looser tolerances.
        Accuracy loss is typically < 0.1% on amp/phase.  Default: False.

    Returns
    -------
    FitResult
    """
    if fast:
        n_phi0_grid = min(n_phi0_grid, 4)
        as_starts   = (0.07,)
        maxiter_coarse, xatol_coarse, fatol_coarse = 800,  1e-4, 1e-2
        maxiter_fine,   xatol_fine,   fatol_fine   = 1000, 1e-6, 1e-4
        fine_perturbs = np.array([[0, 0, 0, 0, 0, 0.0, 0.0],
                                  [0, 0, 0, 0, 0, 0.05, 0.0]])
    else:
        maxiter_coarse, xatol_coarse, fatol_coarse = 1500, 1e-5, 1e-3
        maxiter_fine,   xatol_fine,   fatol_fine   = 2000, 1e-8, 1e-5
        fine_perturbs = np.array([
            [0, 0, 0, 0, 0,  0.00,  0.00],
            [0, 0, 0, 0, 0,  0.02,  0.00],
            [0, 0, 0, 0, 0,  0.05,  0.00],
            [0, 0, 0, 0, 0,  0.00,  0.02],
            [0, 0, 0, 0, 0,  0.00, -0.02],
        ])

    # OLS-fringe initial A/C estimates (used as starting values for A1/A2/C1/C2)
    A1_0, C1_0 = _ols_ac(ev.n1_np / ev.N1_np)
    A2_0, C2_0 = _ols_ac(ev.n2_np / ev.N2_np)

    def neg_nll_coarse(p):
        if not valid_params7(p, amp_bound):
            return 1e300
        v = ev.signal_ll(*p, f=f, coarse=True)
        return -v if np.isfinite(v) else 1e300

    def neg_nll_fine(p):
        if not valid_params7(p, amp_bound):
            return 1e300
        v = ev.signal_ll(*p, f=f, coarse=False)
        return -v if np.isfinite(v) else 1e300

    # ── Stage 1: coarse phi0 grid (ntheta=512) ────────────────────────────────
    # The null-fit phi0 is NOT used — it typically lies in the wrong basin.
    # Sweeping [0, 2π] guarantees we find the correct one.
    phi0_grid   = np.linspace(0, 2 * np.pi, n_phi0_grid, endpoint=False)
    best_coarse = None

    for phi0_init in phi0_grid:
        for As_init in as_starts:
            x0 = np.array([A1_0, A2_0, C1_0, C2_0, phi0_init, float(As_init), 0.0])
            r  = minimize(neg_nll_coarse, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter_coarse,
                                   'xatol': xatol_coarse,
                                   'fatol': fatol_coarse})
            if best_coarse is None or r.fun < best_coarse.fun:
                best_coarse = r

    x0_fine    = best_coarse.x.copy()
    x0_fine[5] = abs(x0_fine[5])   # ensure As ≥ 0 entering Stage 2

    # ── Stage 2: fine polish (adaptive ntheta) ────────────────────────────────
    best_fine = None
    for dp in fine_perturbs:
        start    = np.asarray(x0_fine + dp, dtype=float)
        start[5] = max(start[5], 0.0)
        r = minimize(neg_nll_fine, start, method='Nelder-Mead',
                     options={'maxiter': maxiter_fine,
                               'xatol': xatol_fine,
                               'fatol': fatol_fine})
        if best_fine is None or r.fun < best_fine.fun:
            best_fine = r

    A1, A2, C1, C2, phi0, As, Ac = best_fine.x
    amp   = float(np.sqrt(As**2 + Ac**2))
    phase = float(np.arctan2(Ac, As))

    return FitResult(
        A1=float(A1), A2=float(A2), C1=float(C1), C2=float(C2),
        phi0=float(phi0), As=float(As), Ac=float(Ac),
        amp=amp, phase=phase,
        logL=-float(best_fine.fun),
        ntheta=ev.ntheta,
        f=float(f),
        converged=bool(best_fine.success),
    )


# ── feature-conditioned signal MLE ───────────────────────────────────────────

_FEATURE_NUISANCE_ORDER = ("phase", "offset", "contrast")
_FEATURE_BETA_FIELDS = ("beta_phi", "beta_A1", "beta_A2", "beta_C1", "beta_C2")


def normalize_feature_nuisance(feature_nuisance):
    """Return a validated tuple of enabled nuisance blocks."""
    if feature_nuisance is None:
        return _FEATURE_NUISANCE_ORDER
    if isinstance(feature_nuisance, str):
        feature_nuisance = (feature_nuisance,)
    requested = tuple(dict.fromkeys(feature_nuisance))
    unknown = sorted(set(requested) - set(_FEATURE_NUISANCE_ORDER))
    if unknown:
        raise ValueError(
            f"Unknown feature nuisance block(s): {unknown}. "
            f"Choose from {_FEATURE_NUISANCE_ORDER}."
        )
    if not requested:
        raise ValueError("At least one feature nuisance block must be enabled")
    return tuple(name for name in _FEATURE_NUISANCE_ORDER if name in requested)


def fit_feature_conditioned_mle(
    ev: FeatureConditionedLikelihoodEvaluator,
    f: float,
    feature_names_z0=(),
    feature_names_z100=(),
    feature_names=(),
    feature_nuisance=None,
    beta_phi_prior_std: float = 0.3,
    beta_A_prior_std: float = 0.03,
    beta_C_prior_std: float = 0.3,
    amp_bound: float = _AMP_BOUND_DEFAULT,
    n_phi0_grid: int = 8,
    as_starts: tuple = (0.03, 0.07, 0.10),
    fast: bool = False,
) -> FitResult:
    """
    Two-stage MAP fit for the nonlinear site-local feature model.

    With standardized selected features s0_i from Z0 and s100_i from Z100,
    the enabled nuisance blocks are

        phase:    dpsi_i = beta_phi @ [s0_i, s100_i]
        offset:   A1_i = A1 + beta_A1 @ s0_i
                  A2_i = A2 + beta_A2 @ s100_i
        contrast: C1_i = C1 * exp(beta_C1 @ s0_i)
                  C2_i = C2 * exp(beta_C2 @ s100_i)

    The likelihood still marginalizes the common per-shot theta_i exactly as in
    the count-only model.  Gaussian penalties are applied to each enabled beta
    block, so the optimizer maximizes a MAP objective while the returned logL is
    the unpenalized likelihood at the optimum.
    """
    if ev.n_features_z0 < 1 or ev.n_features_z100 < 1:
        raise ValueError("feature-conditioned fitting requires at least one feature per site")
    if beta_phi_prior_std <= 0 or beta_A_prior_std <= 0 or beta_C_prior_std <= 0:
        raise ValueError("All beta prior std values must be positive")
    nuisance = normalize_feature_nuisance(feature_nuisance)

    if fast:
        n_phi0_grid = min(n_phi0_grid, 4)
        as_starts = (0.07,)
        maxiter_coarse, xatol_coarse, fatol_coarse = 1800, 1e-4, 1e-2
        maxiter_fine, xatol_fine, fatol_fine = 2200, 1e-6, 1e-4
    else:
        maxiter_coarse, xatol_coarse, fatol_coarse = 3500, 1e-5, 1e-3
        maxiter_fine, xatol_fine, fatol_fine = 5000, 1e-8, 1e-5

    A1_0, C1_0 = _ols_ac(ev.n1_np / ev.N1_np)
    A2_0, C2_0 = _ols_ac(ev.n2_np / ev.N2_np)
    nfeat_z0 = ev.n_features_z0
    nfeat_z100 = ev.n_features_z100
    nfeat_phase = ev.n_features_phase

    active_blocks = []
    if "phase" in nuisance:
        active_blocks.append(("beta_phi", nfeat_phase, beta_phi_prior_std))
    if "offset" in nuisance:
        active_blocks.extend([
            ("beta_A1", nfeat_z0, beta_A_prior_std),
            ("beta_A2", nfeat_z100, beta_A_prior_std),
        ])
    if "contrast" in nuisance:
        active_blocks.extend([
            ("beta_C1", nfeat_z0, beta_C_prior_std),
            ("beta_C2", nfeat_z100, beta_C_prior_std),
        ])

    n_params = 7 + sum(n for _name, n, _prior_std in active_blocks)
    maxiter_coarse = max(maxiter_coarse, (250 if fast else 400) * n_params)
    maxiter_fine = max(maxiter_fine, (400 if fast else 650) * n_params)

    def zero_betas():
        return {
            "beta_phi": np.zeros(nfeat_phase, dtype=float),
            "beta_A1": np.zeros(nfeat_z0, dtype=float),
            "beta_A2": np.zeros(nfeat_z100, dtype=float),
            "beta_C1": np.zeros(nfeat_z0, dtype=float),
            "beta_C2": np.zeros(nfeat_z100, dtype=float),
        }

    def split(p):
        p = np.asarray(p, dtype=float)
        base = p[:7]
        beta = zero_betas()
        cursor = 7
        for name, n_features, _prior_std in active_blocks:
            beta[name] = p[cursor:cursor + n_features]
            cursor += n_features
        if cursor != len(p):
            raise ValueError("Parameter vector has the wrong length")
        return base, beta

    def pack(base, beta=None):
        pieces = [np.asarray(base, dtype=float)]
        beta = beta or {}
        defaults = zero_betas()
        for name, _n_features, _prior_std in active_blocks:
            pieces.append(np.asarray(beta.get(name, defaults[name]), dtype=float))
        return np.concatenate(pieces)

    def penalty(beta):
        total = 0.0
        for name, _n_features, prior_std in active_blocks:
            values = np.asarray(beta[name], dtype=float)
            total += 0.5 * float(np.sum((values / prior_std) ** 2))
        return total

    def valid_feature_params(p):
        try:
            base, beta = split(p)
        except ValueError:
            return False
        if not valid_params7(base, amp_bound):
            return False
        return all(np.all(np.isfinite(values)) for values in beta.values())

    def eval_ll(base, beta, coarse):
        A1, A2, C1, C2, phi0, As, Ac = base
        return ev.signal_ll_feature(
            A1,
            A2,
            C1,
            C2,
            phi0,
            As,
            Ac,
            beta_phi=beta["beta_phi"] if "phase" in nuisance else None,
            f=f,
            coarse=coarse,
            beta_A1=beta["beta_A1"] if "offset" in nuisance else None,
            beta_A2=beta["beta_A2"] if "offset" in nuisance else None,
            beta_C1=beta["beta_C1"] if "contrast" in nuisance else None,
            beta_C2=beta["beta_C2"] if "contrast" in nuisance else None,
        )

    def neg_logpost(p, coarse):
        if not valid_feature_params(p):
            return 1e300
        base, beta = split(p)
        ll = eval_ll(base, beta, coarse=coarse)
        if not np.isfinite(ll):
            return 1e300
        return -ll + penalty(beta)

    phi0_grid = np.linspace(0, 2 * np.pi, n_phi0_grid, endpoint=False)
    best_coarse = None
    for phi0_init in phi0_grid:
        for As_init in as_starts:
            base0 = np.array([A1_0, A2_0, C1_0, C2_0, phi0_init, float(As_init), 0.0])
            x0 = pack(base0)
            r = minimize(
                lambda p: neg_logpost(p, coarse=True),
                x0,
                method='Nelder-Mead',
                options={'maxiter': maxiter_coarse, 'xatol': xatol_coarse, 'fatol': fatol_coarse},
            )
            if best_coarse is None or r.fun < best_coarse.fun:
                best_coarse = r

    x0_fine = best_coarse.x.copy()
    x0_fine[5] = abs(x0_fine[5])
    perturbations = [np.zeros_like(x0_fine)]
    for amount in ((0.02, 0.0), (0.05, 0.0), (0.0, 0.02), (0.0, -0.02)):
        dp = np.zeros_like(x0_fine)
        dp[5], dp[6] = amount
        perturbations.append(dp)

    best_fine = None
    for dp in perturbations:
        start = x0_fine + dp
        start[5] = max(start[5], 0.0)
        r = minimize(
            lambda p: neg_logpost(p, coarse=False),
            start,
            method='Nelder-Mead',
            options={'maxiter': maxiter_fine, 'xatol': xatol_fine, 'fatol': fatol_fine},
        )
        if best_fine is None or r.fun < best_fine.fun:
            best_fine = r

    base, beta = split(best_fine.x)
    A1, A2, C1, C2, phi0, As, Ac = base
    logL = eval_ll(base, beta, coarse=False)
    beta_penalty = penalty(beta)
    amp = float(np.sqrt(As**2 + Ac**2))
    phase = float(np.arctan2(Ac, As))

    feature_names = tuple(feature_names) if feature_names else tuple(feature_names_z100)
    feature_names_phase = tuple(f"Z0:{name}" for name in feature_names_z0) + tuple(
        f"Z100:{name}" for name in feature_names_z100
    )
    return FitResult(
        A1=float(A1), A2=float(A2), C1=float(C1), C2=float(C2),
        phi0=float(phi0), As=float(As), Ac=float(Ac),
        amp=amp, phase=phase,
        logL=float(logL),
        ntheta=ev.ntheta,
        f=float(f),
        converged=bool(best_fine.success),
        feature_names=feature_names,
        feature_names_z0=tuple(feature_names_z0),
        feature_names_z100=tuple(feature_names_z100),
        feature_names_phase=feature_names_phase,
        feature_nuisance=tuple(nuisance),
        beta_phi=tuple(float(x) for x in beta["beta_phi"]),
        beta_A1=tuple(float(x) for x in beta["beta_A1"]),
        beta_A2=tuple(float(x) for x in beta["beta_A2"]),
        beta_C1=tuple(float(x) for x in beta["beta_C1"]),
        beta_C2=tuple(float(x) for x in beta["beta_C2"]),
        beta_phi_prior_std=float(beta_phi_prior_std),
        beta_A_prior_std=float(beta_A_prior_std),
        beta_C_prior_std=float(beta_C_prior_std),
        beta_penalty=float(beta_penalty),
        log_posterior=float(logL - beta_penalty),
    )


def fit_feature_conditioned_from_datasets(
    Z0,
    Z100,
    features_z0,
    features_z100,
    f: float,
    feature_names_z0=(),
    feature_names_z100=(),
    feature_names=(),
    feature_nuisance=None,
    use_gpu: bool = True,
    ntheta: int = None,
    beta_phi_prior_std: float = 0.3,
    beta_A_prior_std: float = 0.03,
    beta_C_prior_std: float = 0.3,
    feature_mean_z0=None,
    feature_scale_z0=None,
    feature_mean_z100=None,
    feature_scale_z100=None,
    **kwargs,
) -> FitResult:
    """Build a FeatureConditionedLikelihoodEvaluator from datasets and fit."""
    c0 = Z0.state_counts()
    c100 = Z100.state_counts()

    n1 = np.asarray(c0[0], dtype=float)
    n2 = np.asarray(c100[0], dtype=float)
    N1 = np.asarray(c0[0] + c0[1], dtype=float)
    N2 = np.asarray(c100[0] + c100[1], dtype=float)

    ev = FeatureConditionedLikelihoodEvaluator(
        n1,
        n2,
        N1,
        N2,
        features_z0,
        features_z100,
        use_gpu=use_gpu,
        ntheta=ntheta,
        feature_mean_z0=feature_mean_z0,
        feature_scale_z0=feature_scale_z0,
        feature_mean_z100=feature_mean_z100,
        feature_scale_z100=feature_scale_z100,
    )
    return fit_feature_conditioned_mle(
        ev,
        f=f,
        feature_names_z0=feature_names_z0,
        feature_names_z100=feature_names_z100,
        feature_names=feature_names,
        feature_nuisance=feature_nuisance,
        beta_phi_prior_std=beta_phi_prior_std,
        beta_A_prior_std=beta_A_prior_std,
        beta_C_prior_std=beta_C_prior_std,
        **kwargs,
    )


# ── convenience wrapper ───────────────────────────────────────────────────────

def fit_from_datasets(Z0, Z100, f: float, use_gpu: bool = True,
                      ntheta: int = None, **kwargs) -> FitResult:
    """
    Build a LikelihoodEvaluator from LazyShotDataset objects and fit.

    Parameters
    ----------
    Z0, Z100 : ImageShotDataset  (or any object with a .state_counts() method
                                  returning a DataFrame with columns 0 and 1)
    f : float
        Signal frequency [cycles per time unit].
    use_gpu : bool
        Pass False to force CPU inference.
    **kwargs
        Forwarded to fit_mle: amp_bound, n_phi0_grid, as_starts.

    Returns
    -------
    FitResult

    Example
    -------
    >>> from helpers import ImageShotDataset
    >>> from fitting import fit_from_datasets
    >>> Z0  = ImageShotDataset('data/run_000/Z0/data_IMG.h5')
    >>> Z100 = ImageShotDataset('data/run_000/Z100/data_IMG.h5')
    >>> r = fit_from_datasets(Z0, Z100, f=0.3)
    >>> print(r.amp, r.phase)
    """
    c0   = Z0.state_counts()    # DataFrame, columns 0 (ground) and 1 (excited)
    c100 = Z100.state_counts()

    n1 = np.asarray(c0[0],   dtype=float)
    n2 = np.asarray(c100[0], dtype=float)
    N1 = np.asarray(c0[0]   + c0[1],   dtype=float)
    N2 = np.asarray(c100[0] + c100[1], dtype=float)

    ev = LikelihoodEvaluator(n1, n2, N1, N2, use_gpu=use_gpu, ntheta=ntheta)
    return fit_mle(ev, f=f, **kwargs)
