"""
likelihood.py — GPU-aware marginal likelihood evaluator for MAGIS-100.

The per-shot marginal log-likelihood integrates out the common laser phase θ:

    log L = Σ_i  log ∫_0^{2π} dθ/(2π)  p_Z0(n1_i|θ) · p_Z100(n2_i|θ + δφ_i)

The θ integral is discretised to `ntheta` equally-spaced quadrature points.
ntheta must scale with √N_atoms_per_shot to keep quadrature error small —
use `adaptive_ntheta()` to compute it automatically.

Usage
-----
    from likelihood import LikelihoodEvaluator, FeatureConditionedLikelihoodEvaluator, adaptive_ntheta

    ev = LikelihoodEvaluator(n1, n2, N1, N2)
    print(ev)  # LikelihoodEvaluator(n_shots=100, ntheta=2048, backend='cupy')

    logL      = ev.signal_ll(A1, A2, C1, C2, phi0, As, Ac, f=0.3)
    logL_null = ev.null_ll(A1, A2, C1, C2, phi0)
"""

import math
import numpy as np
from scipy.special import logsumexp as _np_logsumexp

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as _cp_logsumexp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


# ── quadrature helper ─────────────────────────────────────────────────────────

def adaptive_ntheta(N_mean: float) -> int:
    """
    Minimum ntheta for accurate theta quadrature.

    The theta integrand peak has width ~1/(C·√N), so the grid must satisfy
    ntheta ≥ 2·√N_mean.  Result is rounded up to the next power of 2
    and floored at 512.

    Parameters
    ----------
    N_mean : float
        Mean detected atoms per shot (use the smaller arm, Z100).

    Returns
    -------
    int  — always a power of 2, minimum 512.
    """
    nmin = max(512, int(2.0 * float(N_mean) ** 0.5))
    return 2 ** math.ceil(math.log2(nmin))


# ── evaluator ─────────────────────────────────────────────────────────────────

class LikelihoodEvaluator:
    """
    Stateful evaluator that holds shot data on the GPU (or CPU) and exposes
    log-likelihood methods for use with scipy optimizers.

    Data is transferred to the compute device once at construction time.
    Multiple optimizer calls reuse the same device arrays, avoiding
    redundant host↔device transfers.

    Parameters
    ----------
    n1, n2 : array-like (n_shots,)
        Ground-state counts for Z0 and Z100.
    N1, N2 : array-like (n_shots,)
        Total atom counts for Z0 and Z100.
    t : array-like (n_shots,), optional
        Shot time / index array.  Defaults to arange(n_shots).
    use_gpu : bool
        Use CuPy backend if available.  Falls back to NumPy silently.
    ntheta : int, optional
        Theta quadrature points.  Defaults to adaptive_ntheta(N2.mean()).

    Attributes
    ----------
    ntheta : int
    n_shots : int
    backend : str  — 'cupy' or 'numpy'
    n1_np, n2_np, N1_np, N2_np : numpy arrays (CPU copies for initial estimates)
    """

    def __init__(self, n1, n2, N1, N2, t=None, use_gpu=True, ntheta=None):
        n1 = np.asarray(n1, dtype=float).ravel()
        n2 = np.asarray(n2, dtype=float).ravel()
        N1 = np.asarray(N1, dtype=float).ravel()
        N2 = np.asarray(N2, dtype=float).ravel()
        assert len(n1) == len(n2) == len(N1) == len(N2), \
            "n1, n2, N1, N2 must all have the same length"

        self.n_shots = len(n1)
        t_np = (np.arange(self.n_shots, dtype=float)
                if t is None else np.asarray(t, dtype=float).ravel())

        # NumPy copies kept on CPU for initial-guess calculations in fitting.py
        self.n1_np = n1.copy()
        self.n2_np = n2.copy()
        self.N1_np = N1.copy()
        self.N2_np = N2.copy()
        self.t_np  = t_np.copy()

        if ntheta is None:
            ntheta = adaptive_ntheta(N2.mean())
        self.ntheta = ntheta

        # Backend
        self.xp      = cp if (use_gpu and _HAS_CUPY) else np
        self._lse    = _cp_logsumexp if self.xp is cp else _np_logsumexp
        self.backend = 'cupy' if self.xp is cp else 'numpy'

        # Transfer to device once
        xp = self.xp
        self._n1 = xp.asarray(n1,   dtype=xp.float64)
        self._n2 = xp.asarray(n2,   dtype=xp.float64)
        self._N1 = xp.asarray(N1,   dtype=xp.float64)
        self._N2 = xp.asarray(N2,   dtype=xp.float64)
        self._t  = xp.asarray(t_np, dtype=xp.float64)

        # Fine grid (adaptive ntheta) and coarse grid (fixed 512 for fast scans)
        self._theta_fine   = xp.linspace(0, 2 * xp.pi, ntheta, endpoint=False)
        self._theta_coarse = xp.linspace(0, 2 * xp.pi, 512,    endpoint=False)

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _log_binom(xp, n, N, p, eps=1e-12):
        """Log-binomial PMF without combinatorial constants (fine for MLE)."""
        p = xp.clip(p, eps, 1.0 - eps)
        return n * xp.log(p) + (N - n) * xp.log1p(-p)

    def _to_float(self, x):
        return float(x.get()) if self.xp is cp else float(x)

    def _core_ll(self, A1, C1, A2, C2, dphi, theta, n_th):
        """
        Marginal log-likelihood for given fringe parameters and dphi.

        dphi  : device array (n_shots,)  — per-shot differential phase
        theta : device array (n_th,)     — quadrature grid
        n_th  : int                       — len(theta), for normalisation
        """
        xp = self.xp

        # p1: fringe probability at Z0, same across all shots
        # p2: fringe probability at Z100, varies per shot via dphi
        p1 = A1 + 0.5 * C1 * xp.cos(theta)                            # (n_th,)
        p2 = A2 + 0.5 * C2 * xp.cos(theta[None, :] + dphi[:, None])   # (n_shots, n_th)

        ll1 = self._log_binom(xp, self._n1[:, None], self._N1[:, None], p1)
        ll2 = self._log_binom(xp, self._n2[:, None], self._N2[:, None], p2)

        return self._to_float(
            xp.sum(self._lse(ll1 + ll2, axis=1) - xp.log(n_th))
        )

    # ── public API ────────────────────────────────────────────────────────────

    def null_ll(self, A1, A2, C1, C2, phi0):
        """
        Log-likelihood for the null (constant differential phase) model.

        δφ_i = phi0  for all shots.

        Parameters
        ----------
        A1, A2, C1, C2, phi0 : float
        """
        xp   = self.xp
        dphi = xp.full(self.n_shots, float(phi0), dtype=xp.float64)
        return self._core_ll(float(A1), float(C1), float(A2), float(C2),
                             dphi, self._theta_fine, self.ntheta)

    def signal_ll(self, A1, A2, C1, C2, phi0, As, Ac, f, coarse=False):
        """
        Log-likelihood for the sinusoidal signal model.

        δφ_i = phi0 + As·sin(2π·f·t_i) + Ac·cos(2π·f·t_i)

        Parameters
        ----------
        A1, A2, C1, C2, phi0, As, Ac : float
        f : float
            Signal frequency in cycles per time unit.
        coarse : bool
            Use the fixed 512-point theta grid instead of the adaptive one.
            Faster but less accurate — only use for the coarse phi0 grid scan.
        """
        xp   = self.xp
        tpf  = 2.0 * float(xp.pi) * float(f)
        dphi = (float(phi0)
                + float(As) * xp.sin(tpf * self._t)
                + float(Ac) * xp.cos(tpf * self._t))
        if coarse:
            return self._core_ll(float(A1), float(C1), float(A2), float(C2),
                                 dphi, self._theta_coarse, 512)
        return self._core_ll(float(A1), float(C1), float(A2), float(C2),
                             dphi, self._theta_fine, self.ntheta)

    def __repr__(self):
        return (f'LikelihoodEvaluator(n_shots={self.n_shots}, '
                f'ntheta={self.ntheta}, backend={self.backend!r})')

class FeatureConditionedLikelihoodEvaluator(LikelihoodEvaluator):
    """
    Marginal likelihood with site-local image-summary-conditioned nuisance terms.

    For each shot i, Z0 has standardized feature vector s0_i and Z100 has
    standardized feature vector s100_i.  The nonlinear model is

        p0_i(theta) = A0_i + 0.5*C0_i*cos(theta)
        p1_i(theta) = A1_i + 0.5*C1_i*cos(theta + dphi_signal_i + dpsi_i)

    with optional feature channels

        dpsi_i = beta_phi @ [s0_i, s100_i]
        A0_i   = A0 + beta_A0 @ s0_i
        A1_i   = A1 + beta_A1 @ s100_i
        C0_i   = C0 * exp(beta_C0 @ s0_i)
        C1_i   = C1 * exp(beta_C1 @ s100_i)

    The Z0 phase nuisance is absorbed into the marginalized common theta_i
    gauge, so the fitted phase correction is differential and can use both
    sites' selected features.  Offset and contrast corrections are local:
    Z0 image summaries never correct Z100 offset/contrast, and vice versa.
    """

    def __init__(
        self,
        n1,
        n2,
        N1,
        N2,
        features_z0,
        features_z100,
        t=None,
        use_gpu=True,
        ntheta=None,
        feature_mean_z0=None,
        feature_scale_z0=None,
        feature_mean_z100=None,
        feature_scale_z100=None,
    ):
        super().__init__(n1, n2, N1, N2, t=t, use_gpu=use_gpu, ntheta=ntheta)

        z0, mean_z0, scale_z0 = self._standardize_features(
            features_z0, feature_mean_z0, feature_scale_z0, "features_z0"
        )
        z100, mean_z100, scale_z100 = self._standardize_features(
            features_z100, feature_mean_z100, feature_scale_z100, "features_z100"
        )
        if z0.shape[0] != self.n_shots:
            raise ValueError(f"features_z0 has {z0.shape[0]} rows but likelihood has {self.n_shots} shots")
        if z100.shape[0] != self.n_shots:
            raise ValueError(f"features_z100 has {z100.shape[0]} rows but likelihood has {self.n_shots} shots")

        self.features_z0_np = np.asarray(features_z0, dtype=float).copy()
        self.features_z100_np = np.asarray(features_z100, dtype=float).copy()
        self.feature_mean_z0 = mean_z0.copy()
        self.feature_scale_z0 = scale_z0.copy()
        self.feature_mean_z100 = mean_z100.copy()
        self.feature_scale_z100 = scale_z100.copy()
        self.standardized_features_z0_np = z0.copy()
        self.standardized_features_z100_np = z100.copy()
        self.n_features_z0 = z0.shape[1]
        self.n_features_z100 = z100.shape[1]
        self.n_features_phase = self.n_features_z0 + self.n_features_z100
        self._features_z0 = self.xp.asarray(z0, dtype=self.xp.float64)
        self._features_z100 = self.xp.asarray(z100, dtype=self.xp.float64)
        self._features_phase = self.xp.concatenate([self._features_z0, self._features_z100], axis=1)

    @staticmethod
    def _standardize_features(features, feature_mean, feature_scale, name):
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features[:, None]
        if features.ndim != 2:
            raise ValueError(f"{name} must be a 2D array with shape (n_shots, n_features)")
        if features.shape[1] < 1:
            raise ValueError(f"{name} must contain at least one feature column")
        if not np.all(np.isfinite(features)):
            raise ValueError(f"{name} contains NaN or inf")
        if feature_mean is None:
            feature_mean = features.mean(axis=0)
        else:
            feature_mean = np.asarray(feature_mean, dtype=float)
        if feature_scale is None:
            feature_scale = features.std(axis=0)
        else:
            feature_scale = np.asarray(feature_scale, dtype=float)
        feature_scale = np.where(feature_scale > 0, feature_scale, 1.0)
        if feature_mean.shape[0] != features.shape[1] or feature_scale.shape[0] != features.shape[1]:
            raise ValueError(f"{name} standardization length does not match feature columns")
        return (features - feature_mean) / feature_scale, feature_mean, feature_scale

    def _as_feature_beta(self, beta, name, n_features):
        beta = self.xp.asarray(beta, dtype=self.xp.float64).ravel()
        if beta.shape[0] != n_features:
            raise ValueError(f"{name} has length {beta.shape[0]}, expected {n_features}")
        return beta

    def _feature_linear_z0(self, beta, name):
        return self._features_z0 @ self._as_feature_beta(beta, name, self.n_features_z0)

    def _feature_linear_z100(self, beta, name):
        return self._features_z100 @ self._as_feature_beta(beta, name, self.n_features_z100)

    def _feature_linear_phase(self, beta, name):
        return self._features_phase @ self._as_feature_beta(beta, name, self.n_features_phase)

    def _zero_shot_vector(self):
        return self.xp.zeros(self.n_shots, dtype=self.xp.float64)

    def _log_binom_checked(self, n, N, p):
        """Log-binomial PMF without clipping; invalid probabilities return None."""
        xp = self.xp
        if bool(self._to_float(xp.any((p <= 0.0) | (p >= 1.0)))):
            return None
        return n * xp.log(p) + (N - n) * xp.log1p(-p)

    def _core_feature_ll(
        self,
        A1,
        C1,
        A2,
        C2,
        dphi_signal,
        theta,
        n_th,
        beta_phi=None,
        beta_A1=None,
        beta_A2=None,
        beta_C1=None,
        beta_C2=None,
    ):
        xp = self.xp
        zero = self._zero_shot_vector()
        dpsi = zero if beta_phi is None else self._feature_linear_phase(beta_phi, "beta_phi")
        delta_A1 = zero if beta_A1 is None else self._feature_linear_z0(beta_A1, "beta_A1")
        delta_A2 = zero if beta_A2 is None else self._feature_linear_z100(beta_A2, "beta_A2")
        log_C1 = zero if beta_C1 is None else self._feature_linear_z0(beta_C1, "beta_C1")
        log_C2 = zero if beta_C2 is None else self._feature_linear_z100(beta_C2, "beta_C2")

        A1_i = float(A1) + delta_A1
        A2_i = float(A2) + delta_A2
        C1_i = float(C1) * xp.exp(log_C1)
        C2_i = float(C2) * xp.exp(log_C2)
        dphi = dphi_signal + dpsi

        p1 = A1_i[:, None] + 0.5 * C1_i[:, None] * xp.cos(theta[None, :])
        p2 = A2_i[:, None] + 0.5 * C2_i[:, None] * xp.cos(theta[None, :] + dphi[:, None])

        ll1 = self._log_binom_checked(self._n1[:, None], self._N1[:, None], p1)
        if ll1 is None:
            return -np.inf
        ll2 = self._log_binom_checked(self._n2[:, None], self._N2[:, None], p2)
        if ll2 is None:
            return -np.inf

        return self._to_float(xp.sum(self._lse(ll1 + ll2, axis=1) - xp.log(n_th)))

    def signal_ll_feature(
        self,
        A1,
        A2,
        C1,
        C2,
        phi0,
        As,
        Ac,
        beta_phi=None,
        f=0.3,
        coarse=False,
        beta_A1=None,
        beta_A2=None,
        beta_C1=None,
        beta_C2=None,
    ):
        """Log-likelihood for the site-local feature-conditioned signal model."""
        xp = self.xp
        tpf = 2.0 * float(xp.pi) * float(f)
        dphi_signal = (
            float(phi0)
            + float(As) * xp.sin(tpf * self._t)
            + float(Ac) * xp.cos(tpf * self._t)
        )
        kwargs = dict(
            beta_phi=beta_phi,
            beta_A1=beta_A1,
            beta_A2=beta_A2,
            beta_C1=beta_C1,
            beta_C2=beta_C2,
        )
        if coarse:
            return self._core_feature_ll(
                A1, C1, A2, C2, dphi_signal, self._theta_coarse, 512, **kwargs
            )
        return self._core_feature_ll(
            A1, C1, A2, C2, dphi_signal, self._theta_fine, self.ntheta, **kwargs
        )

    def __repr__(self):
        return (f'FeatureConditionedLikelihoodEvaluator(n_shots={self.n_shots}, '
                f'n_features_z0={self.n_features_z0}, n_features_z100={self.n_features_z100}, n_features_phase={self.n_features_phase}, '
                f'ntheta={self.ntheta}, backend={self.backend!r})')

