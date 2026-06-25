"""
image_fitting.py — Two-parameter MLE for the position-resolved gradiometer image
likelihood.

Signal model
------------
With the signal frequency f fixed and known, the only free parameters are

    β = (As, Ac)

where the per-shot differential phase is

    δφ_i = As·sin(2π·f·t_i) + Ac·cos(2π·f·t_i).

The signal amplitude and phase are

    A  = √(As² + Ac²),  φ = arctan2(Ac, As) ∈ (−π/2, π/2] with As ≥ 0.

The symmetry L(As, Ac) = L(−As, −Ac) (a full 2π rotation of the signal) is
broken by enforcing As ≥ 0.

Two-stage optimisation
----------------------
The 2-parameter landscape is smooth but can have a local maximum near the null
(As ≈ 0) when the signal is weak.  We use a two-stage strategy:

  Stage 1 — Coarse multi-start scan (coarse theta grid, loose tolerances):
    Try n_as_starts starting values of As (with Ac=0) plus the exact null
    (As=Ac=0).  Uses the faster coarse=True likelihood (fixed 512-point theta)
    so each Nelder-Mead run is cheap.  Identifies the basin of the global max.

  Stage 2 — Fine polish (adaptive theta grid, tight tolerances):
    Re-run Nelder-Mead from the best Stage-1 solution, plus a few perturbations,
    using the full ntheta grid.  Returns the polished MLE.

Compared to the 7-parameter count-only fitter (fitting.py), this is much
simpler: no phi0 grid scan, only 2 parameters, typically converges in < 50 iters.

Usage
-----
    from image_likelihood import ImageLikelihoodEvaluator
    from image_fitting import fit_image_mle, ImageFitResult

    ev = ImageLikelihoodEvaluator(ds_z0, ds_z100, psmap_z0, psmap_z100,
                                   sigma_vx=3.09e-4, f=0.3)
    result = fit_image_mle(ev)
    print(result.amp, result.phase)

    # Or from datasets directly:
    result = fit_image_from_datasets(ds_z0, ds_z100, psmap_z0, psmap_z100,
                                     sigma_vx=3.09e-4, f=0.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

from image_likelihood import ImageLikelihoodEvaluator

_AMP_BOUND = np.pi   # hard upper bound on |As|, |Ac| [rad]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ImageFitResult:
    """
    MLE estimate for the 2-parameter sinusoidal differential-phase signal model.

    Attributes
    ----------
    As, Ac     : float  Signal sine/cosine amplitudes [rad]; As ≥ 0 by convention.
    amp        : float  Signal amplitude  = √(As² + Ac²)  [rad].
    phase      : float  Signal phase      = arctan2(Ac, As) ∈ (−π/2, π/2].
    logL       : float  Marginal log-likelihood at the MLE (β̂).
    logL_null  : float  Marginal log-likelihood at the null (As=Ac=0).
    delta_logL : float  logL − logL_null (positive = data prefer signal).
    f          : float  Fixed signal frequency [cycles per time unit].
    ntheta     : int    Theta quadrature points at the fine stage.
    converged  : bool   Nelder-Mead convergence flag (Stage 2 best start).

    Optimiser diagnostics (all from Stage 2 best run)
    --------------------------------------------------
    optimizer_status, optimizer_message, optimizer_nit, optimizer_nfev,
    optimizer_objective : scipy.OptimizeResult fields.

    Stage 1 / Stage 2 diagnostics
    ------------------------------
    n_coarse_starts    : int    Total Stage-1 starting points tried.
    fine_start_As, fine_start_Ac : float  Stage-2 starting point.
    fine_start_objectives : tuple  Objective values at each Stage-2 start.
    """
    As: float
    Ac: float
    amp: float
    phase: float
    logL: float
    logL_null: float
    delta_logL: float
    f: float
    ntheta: int
    converged: bool
    # Optimiser diagnostics
    optimizer_status: int = -1
    optimizer_message: str = ""
    optimizer_nit: int = 0
    optimizer_nfev: int = 0
    optimizer_objective: float = float("nan")
    # Multi-start diagnostics
    n_coarse_starts: int = 0
    fine_start_As: float = float("nan")
    fine_start_Ac: float = float("nan")
    fine_start_objectives: tuple = field(default_factory=tuple)


# ── Parameter validity ────────────────────────────────────────────────────────

def _valid_params(p, amp_bound=_AMP_BOUND):
    """
    Return True iff (As, Ac) lie in the physical region.

    Enforces:
        0 ≤ As ≤ amp_bound   (As ≥ 0 breaks the amplitude-phase degeneracy)
        |Ac| ≤ amp_bound
    """
    As, Ac = p
    return 0.0 <= As <= amp_bound and -amp_bound <= Ac <= amp_bound


# ── MLE ───────────────────────────────────────────────────────────────────────

def fit_image_mle(
    ev: ImageLikelihoodEvaluator,
    amp_bound: float = _AMP_BOUND,
    as_starts: tuple = (0.0, 0.03, 0.07, 0.12, 0.20),
    fast: bool = False,
) -> ImageFitResult:
    """
    Two-stage MLE for the 2-parameter sinusoidal signal model.

    Parameters
    ----------
    ev : ImageLikelihoodEvaluator
        Pre-built evaluator (holds pixel statistics on device).
    amp_bound : float
        Hard upper bound on |As|, |Ac| [rad].  Default π.
    as_starts : tuple of float
        As starting values for Stage 1.  Each is tried with Ac=0.
        The null (0, 0) is always included automatically.
    fast : bool
        Use fewer starts, looser tolerances, coarser theta at Stage 2.
        Accuracy loss is typically < 0.5% on amp/phase.  Good for bootstrap.

    Returns
    -------
    ImageFitResult
    """
    # Tolerances
    if fast:
        maxiter_coarse, xatol_coarse, fatol_coarse = 400,  1e-4, 1e-2
        maxiter_fine,   xatol_fine,   fatol_fine   = 500,  1e-6, 1e-4
        fine_perturbs = ((0.0, 0.0), (0.03, 0.0))
    else:
        maxiter_coarse, xatol_coarse, fatol_coarse = 800,  1e-5, 1e-3
        maxiter_fine,   xatol_fine,   fatol_fine   = 1500, 1e-8, 1e-5
        fine_perturbs = (
            ( 0.00,  0.00),
            ( 0.03,  0.00),
            ( 0.00,  0.03),
            ( 0.00, -0.03),
        )

    # ── objective wrappers ────────────────────────────────────────────────────

    def neg_ll_coarse(p):
        """Negative log-likelihood on the coarse theta grid (fast)."""
        if not _valid_params(p, amp_bound):
            return 1e300
        v = ev.signal_ll(float(p[0]), float(p[1]), coarse=True)
        return -v if np.isfinite(v) else 1e300

    def neg_ll_fine(p):
        """Negative log-likelihood on the fine (adaptive) theta grid."""
        if not _valid_params(p, amp_bound):
            return 1e300
        v = ev.signal_ll(float(p[0]), float(p[1]), coarse=False)
        return -v if np.isfinite(v) else 1e300

    # ── Stage 1: coarse multi-start ───────────────────────────────────────────
    # Sweep the supplied As starting values with Ac=0.
    # With only 2 parameters, this is very fast even with many starts.
    all_as_starts = sorted(set([0.0] + list(as_starts)))  # always include null
    best_coarse   = None

    for As_init in all_as_starts:
        x0 = np.array([float(As_init), 0.0])
        r  = minimize(
            neg_ll_coarse, x0,
            method="Nelder-Mead",
            options={
                "maxiter": maxiter_coarse,
                "xatol":   xatol_coarse,
                "fatol":   fatol_coarse,
            },
        )
        if best_coarse is None or r.fun < best_coarse.fun:
            best_coarse = r

    x0_fine     = best_coarse.x.copy()
    x0_fine[0]  = abs(x0_fine[0])  # enforce As ≥ 0

    # ── Stage 2: fine polish from multiple perturbations ─────────────────────
    # Re-optimise from the Stage-1 solution plus small perturbations in (As, Ac)
    # to reduce the risk of stopping at a plateau near the true maximum.
    best_fine       = None
    fine_objectives = []

    for d_As, d_Ac in fine_perturbs:
        start    = x0_fine + np.array([d_As, d_Ac])
        start[0] = max(start[0], 0.0)   # keep As ≥ 0
        r = minimize(
            neg_ll_fine, start,
            method="Nelder-Mead",
            options={
                "maxiter": maxiter_fine,
                "xatol":   xatol_fine,
                "fatol":   fatol_fine,
            },
        )
        fine_objectives.append(float(r.fun))
        if best_fine is None or r.fun < best_fine.fun:
            best_fine = r

    As, Ac    = float(best_fine.x[0]), float(best_fine.x[1])
    As        = abs(As)   # guarantee As ≥ 0 after polish
    amp       = float(np.hypot(As, Ac))
    phase     = float(np.arctan2(Ac, As))
    logL      = -float(best_fine.fun)
    logL_null = ev.null_ll()

    return ImageFitResult(
        As=As, Ac=Ac,
        amp=amp, phase=phase,
        logL=logL,
        logL_null=logL_null,
        delta_logL=logL - logL_null,
        f=ev.f,
        ntheta=ev.ntheta,
        converged=bool(best_fine.success),
        optimizer_status=int(best_fine.status),
        optimizer_message=str(best_fine.message),
        optimizer_nit=int(best_fine.nit),
        optimizer_nfev=int(best_fine.nfev),
        optimizer_objective=float(best_fine.fun),
        n_coarse_starts=len(all_as_starts),
        fine_start_As=float(x0_fine[0]),
        fine_start_Ac=float(x0_fine[1]),
        fine_start_objectives=tuple(fine_objectives),
    )


# ── Convenience wrapper ────────────────────────────────────────────────────────

def fit_image_from_datasets(
    ds_z0,
    ds_z100,
    psmap_z0: dict,
    psmap_z100: dict,
    sigma_vx: float,
    f: float,
    sigma_x: float | None = None,
    sigma_vy: float | None = None,
    sigma_y: float | None = None,
    bins: int = 32,
    hermite_order: int = 5,
    ntheta: int | None = None,
    t=None,
    use_gpu: bool = True,
    verbose: bool = True,
    **kwargs,
) -> ImageFitResult:
    """
    Build an ImageLikelihoodEvaluator and fit in one call.

    Parameters
    ----------
    ds_z0, ds_z100 : ImageShotDataset
    psmap_z0, psmap_z100 : dict  from aispy.psmap.load_psmap
    sigma_vx : float  initial velocity spread [m/s]
    f : float  signal frequency [cycles per shot]
    sigma_x : float or None  initial position spread [m]
    **kwargs : forwarded to fit_image_mle (amp_bound, as_starts, fast)

    Returns
    -------
    ImageFitResult
    """
    ev = ImageLikelihoodEvaluator(
        ds_z0, ds_z100, psmap_z0, psmap_z100,
        sigma_vx=sigma_vx,
        sigma_x=sigma_x,
        sigma_vy=sigma_vy,
        sigma_y=sigma_y,
        bins=bins,
        hermite_order=hermite_order,
        ntheta=ntheta,
        f=f,
        t=t,
        use_gpu=use_gpu,
        verbose=verbose,
    )
    return fit_image_mle(ev, **kwargs)
