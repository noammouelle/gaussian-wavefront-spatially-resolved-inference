"""
image_likelihood.py — Position-resolved Poisson marginal likelihood for MAGIS-100
gradiometer image data.

Mathematical overview
---------------------
Two atom interferometers (AIs) at source heights z=0 m and z=100 m are operated
simultaneously as a gradiometer. For each shot i and each AI z the camera records
two (bins × bins) images: n_izsb counts in pixel b for state s ∈ {g, e}.

The expected count at pixel b under total laser+signal phase Φ is

    λ_izsb(Φ) = Λ_iz · p_izsb(Φ)

where Λ_iz is the effective number of launched atoms (fixed from the observed
total count) and p_izsb(Φ) is the per-atom detection probability in pixel b:

    p_izsb(Φ) = A_izsb + Cc_izsb · cos(Φ) + Cs_izsb · sin(Φ)

The three pixel statistics (A, Cc, Cs) are pre-computed ONCE from the PSMAP and
the per-shot cloud shape via Gauss-Hermite quadrature (see `_extract_pixel_stats`).
This exploits the fact that any port-probability-derived quantity is exactly
affine in cos(Φ) and sin(Φ), so three evaluations at Φ ∈ {0, π/2, π} suffice to
determine the decomposition exactly.

Phase structure
---------------
The effective total phase for each AI is:

    Φ_iz(φ_i, β) = φ_i + ψ_z(δφ_i),
        ψ_z=0  = 0               (Z0 sees only the common laser phase)
        ψ_z=100= δφ_i            (Z100 additionally sees the differential signal)
    δφ_i = As·sin(2π·f·t_i) + Ac·cos(2π·f·t_i)

where φ_i ~ Uniform(0, 2π) is the unobserved common laser phase and β = (As, Ac)
are the two science parameters (signal frequency f is fixed/known).

Phase marginalisation
---------------------
φ_i is integrated out by uniform quadrature on n_theta equally-spaced points:

    log L_i(β) = logsumexp_k [log L_i(β, φ_k)] − log(n_theta),
    log L(β)   = Σ_i log L_i(β)

The conditional per-shot log-likelihood is a sparse sum over occupied pixels:

    log L_i(β, φ_k) = Σ_{z,s,b} [n_izsb · log λ_izsb(Φ_iz,k) − λ_izsb(Φ_iz,k)]

Atom count normalisation
------------------------
The atom count Λ_iz is fixed per shot per AI using the observed total count:

    Λ_iz = N^+_iz / p̄_det,iz
    N^+_iz    = Σ_{s,b} n_izsb        (observed total downsampled counts)
    p̄_det,iz  = Σ_{s,b} A_izsb        (DC detection probability, Φ-independent)

Cloud nuisance
--------------
The cloud COM (μ_x^f, μ_y^f) in final-position space is estimated per shot per AI
from a weighted centroid of the sum image.  The cloud spreads (σ_x0, σ_vx, ...)
are fixed at their calibration/generation values — the shot-to-shot spread
variation is ~10% and has negligible impact on the science inference.

Because we observe the atom cloud after free-flight propagation, we cannot
separate μ_x0 from μ_vx0 independently.  We therefore absorb the full COM into
μ_x0 (set μ_vx0 = 0), giving μ_x^f = μ_x0 and σ_x^f² = σ_x0² + T²_det·σ_vx².

Usage
-----
    from image_likelihood import ImageLikelihoodEvaluator, adaptive_ntheta
    from aispy.psmap import load_psmap

    psmap_z0  = load_psmap('PSGRID4D_CONFOCAL_Z0.h5')
    psmap_z100 = load_psmap('PSGRID4D_CONFOCAL_Z100.h5')
    ev = ImageLikelihoodEvaluator(
            ds_z0, ds_z100, psmap_z0, psmap_z100,
            sigma_vx=3.09e-4, sigma_x=100e-6, f=0.3)
    print(ev)
    logL = ev.signal_ll(As=0.1, Ac=0.0)
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp as _np_logsumexp

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as _cp_logsumexp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

# Resolve path to sibling helper modules
_HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
if _HELPER_DIR not in sys.path:
    sys.path.insert(0, _HELPER_DIR)

from psmap_fisher import PSMAPConditionalImageModel  # noqa: E402
from likelihood import adaptive_ntheta               # noqa: E402

# Detection time — must match generate_data.py
T_DET = 3.8  # s


# ── Image utilities ────────────────────────────────────────────────────────────

def _downsample_image(image, bins):
    """
    Block-sum downsample a (res, res) uint16 image to (bins, bins).

    Each output pixel is the sum of a (res//bins × res//bins) block of input
    pixels.  The input resolution must be divisible by `bins`.

    Parameters
    ----------
    image : (res, res) array-like
    bins  : int — output resolution per axis

    Returns
    -------
    (bins, bins) float64 array
    """
    image = np.asarray(image, dtype=np.float64)
    res = image.shape[0]
    if res % bins:
        raise ValueError(
            f"Image resolution {res} is not divisible by bins={bins}."
        )
    block = res // bins
    return image.reshape(bins, block, bins, block).sum(axis=(1, 3))


def _downsampled_edges(ds, bins):
    """
    Return bin edges for the downsampled pixel grid that matches dataset `ds`.

    Parameters
    ----------
    ds   : ImageShotDataset (must have .half_range and .res attributes)
    bins : int

    Returns
    -------
    edges : (bins+1,) float64 array  [m]
    """
    if ds.res % bins:
        raise ValueError(
            f"Dataset resolution {ds.res} is not divisible by bins={bins}."
        )
    return np.linspace(-ds.half_range, ds.half_range, bins + 1)


# ── Cloud COM estimation ───────────────────────────────────────────────────────

def _estimate_cloud_com(sum_image, x_centers, y_centers):
    """
    Estimate the final-position cloud centre-of-mass from the sum image.

    Uses a weighted centroid (method-of-moments estimator).  The sum image
    n^+_b = Σ_s n_b is proportional to the cloud density at each pixel under
    the small-pixel approximation, so the centroid directly estimates
    (μ_x^f, μ_y^f).

    Parameters
    ----------
    sum_image : (bins, bins) array — downsampled sum over both states
    x_centers : (bins,) array — pixel centre x-coordinates [m]
    y_centers : (bins,) array — pixel centre y-coordinates [m]

    Returns
    -------
    mu_xf, mu_yf : float [m]
        Weighted centroid. Returns (0.0, 0.0) for an empty image.
    """
    total = float(sum_image.sum())
    if total <= 0.0:
        return 0.0, 0.0
    mu_xf = float((sum_image * x_centers[:, None]).sum() / total)
    mu_yf = float((sum_image * y_centers[None, :]).sum() / total)
    return mu_xf, mu_yf


def _build_cloud_theta(mu_xf, mu_yf, sigma_x, sigma_y, sigma_vx, sigma_vy):
    """
    Pack cloud parameters into the 8-element theta array expected by
    PSMAPConditionalImageModel.detected_probabilities.

    We absorb the full final-position COM into the initial-space position mean
    (μ_x0 = μ_x^f) and set μ_vx0 = 0.  This is exact for
    PSMAPConditionalImageModel because the model uses only the final-position
    mean (μ_x0 + T_det·μ_vx0) and the conditional velocity moments, which
    depend on the spread parameters but not on how the COM is split between
    position and velocity.

    Parameters
    ----------
    mu_xf, mu_yf : float — estimated final-position COM [m]
    sigma_x  : float — initial position spread σ_x0 [m]
    sigma_y  : float — initial position spread σ_y0 [m]
    sigma_vx : float — initial velocity spread σ_vx0 [m/s]
    sigma_vy : float — initial velocity spread σ_vy0 [m/s]

    Returns
    -------
    theta : (8,) float64 array  — [mu_x0, mu_y0, mu_vx0, mu_vy0,
                                    sigma_x0, sigma_y0, sigma_vx0, sigma_vy0]
    """
    return np.array([mu_xf, mu_yf, 0.0, 0.0,
                     sigma_x, sigma_y, sigma_vx, sigma_vy],
                    dtype=np.float64)


# ── Pixel statistics pre-computation ──────────────────────────────────────────

def _extract_pixel_stats(model_0, model_90, model_180, theta):
    """
    Extract per-pixel (A, Cc, Cs) via exact 3-point phase evaluation.

    The port-probability formula gives:

        p_s,b(Φ) = A_sb + Cc_sb·cos(Φ) + Cs_sb·sin(Φ)

    exactly (it is a sum of constant + cos + sin at each grid node, and
    integration over the Gaussian cloud preserves this structure).  Three
    evaluations at Φ = 0, π/2, π uniquely determine the three coefficients:

        A  = ½(p(0) + p(π))
        Cc = ½(p(0) − p(π))
        Cs = p(π/2) − A

    Parameters
    ----------
    model_0, model_90, model_180 : PSMAPConditionalImageModel
        Built at phi0 = 0, π/2, π respectively for the same PSMAP.
    theta : (8,) array
        Initial-space cloud parameters for this shot.

    Returns
    -------
    A, Cc, Cs : (2, n_pix) float64 arrays
        First axis = state index (0 = ground, 1 = excited).
        Values are detection probabilities per launched atom per pixel.
    """
    p0   = model_0.detected_probabilities(theta)    # (2·n_pix,)
    p90  = model_90.detected_probabilities(theta)   # (2·n_pix,)
    p180 = model_180.detected_probabilities(theta)  # (2·n_pix,)

    A  = 0.5 * (p0 + p180)
    Cc = 0.5 * (p0 - p180)
    Cs = p90 - A

    n_pix = len(p0) // 2
    # Returned shape (2, n_pix): row 0 = ground state, row 1 = excited state.
    return A.reshape(2, n_pix), Cc.reshape(2, n_pix), Cs.reshape(2, n_pix)


# ── Evaluator ──────────────────────────────────────────────────────────────────

class ImageLikelihoodEvaluator:
    """
    Stateful position-resolved likelihood evaluator for gradiometer images.

    Construction
    ------------
    Calling the constructor performs ALL expensive pre-computation:
      1. Downsample images to (bins × bins) pixels.
      2. Estimate the cloud COM per shot per AI from the sum image centroid.
      3. Build 3 PSMAPConditionalImageModel objects per AI (one per phase
         point 0, π/2, π) — 6 models in total.
      4. Call detected_probabilities once per shot × AI × phase point (6 × n_shots
         evaluations) and extract per-pixel (A, Cc, Cs) statistics.
      5. Compute the atom-count normalisation Λ_iz per shot per AI.
      6. Transfer all statistics to the compute device (GPU or CPU).

    Evaluation
    ----------
    `signal_ll(As, Ac)` is the bottleneck called by the optimiser at each
    iteration.  It batches the n_theta phase-grid points to keep GPU memory
    bounded, accumulates the per-shot conditional log-likelihoods, and returns
    the full marginal log-likelihood as a Python float.

    Parameters
    ----------
    ds_z0, ds_z100 : ImageShotDataset
        Image datasets for the two AIs.  Must have matching n_shots and the same
        half_range / res (pixel-aligned).
    psmap_z0, psmap_z100 : dict
        PSMAP dictionaries from aispy.psmap.load_psmap.
    sigma_vx : float
        Known initial velocity spread σ_vx0 = σ_vy0 [m/s], applied to both AIs.
        Used to compute the conditional velocity distribution at each final pixel.
    sigma_x : float or None
        Known initial position spread σ_x0 = σ_y0 [m].
        Defaults to deriving it from ds_z0.n_atoms_launched or 100 µm if unknown.
        Explicitly provide this when you know the generation value.
    sigma_vy : float or None
        If None, uses sigma_vx for both axes (isotropic).
    sigma_y : float or None
        If None, uses sigma_x for both axes (isotropic).
    bins : int
        Downsampled image resolution per axis.  The original image must have
        resolution divisible by bins.  Default 32 → 1024 pixels per AI per state.
    hermite_order : int
        Gauss-Hermite quadrature order for the velocity integral.  n_GH² nodes
        per pixel.  Order 5 achieves accuracy equivalent to ~10^4 MC samples
        with zero variance (purely deterministic).
    ntheta : int or None
        Phase grid size for marginalising φ_i.  Defaults to adaptive_ntheta
        (power of 2 ≥ 2·√N_detected, minimum 512).
    f : float
        Signal frequency [cycles per shot index].  Fixed/known.
    t : array-like (n_shots,) or None
        Shot time/index array.  Defaults to arange(n_shots).
    use_gpu : bool
        Use CuPy for the per-evaluation GPU kernel if available.
    verbose : bool
        Print a progress bar during pre-computation.

    Attributes
    ----------
    n_shots : int
    bins : int
    n_pix : int  — bins²
    ntheta : int
    n_theta_coarse : int  — 512, used for fast optimizer scans
    backend : str  — 'cupy' or 'numpy'
    """

    # Phase grid points used for 3-point coefficient extraction
    _PHASE_POINTS = (0.0, 0.5 * math.pi, math.pi)

    def __init__(
        self,
        ds_z0,
        ds_z100,
        psmap_z0,
        psmap_z100,
        sigma_vx: float,
        sigma_x: float | None = None,
        sigma_vy: float | None = None,
        sigma_y: float | None = None,
        bins: int = 32,
        hermite_order: int = 5,
        ntheta: int | None = None,
        f: float = 0.3,
        t=None,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        t_total = time.perf_counter()

        # ── validate inputs ───────────────────────────────────────────────────
        if ds_z0.n_shots != ds_z100.n_shots:
            raise ValueError(
                f"Z0 has {ds_z0.n_shots} shots but Z100 has {ds_z100.n_shots}."
            )
        if abs(ds_z0.half_range - ds_z100.half_range) > 1e-9:
            raise ValueError(
                f"Z0 half_range={ds_z0.half_range} ≠ Z100 half_range={ds_z100.half_range}. "
                "Datasets must share the same pixel grid."
            )
        if ds_z0.res != ds_z100.res:
            raise ValueError(
                f"Z0 res={ds_z0.res} ≠ Z100 res={ds_z100.res}."
            )
        if ds_z0.res % bins:
            raise ValueError(
                f"Image resolution {ds_z0.res} is not divisible by bins={bins}."
            )

        self.n_shots  = ds_z0.n_shots
        self.bins     = bins
        self.n_pix    = bins * bins
        self.f        = float(f)

        # Isotropic spreads if not specified
        sigma_vy = sigma_vy if sigma_vy is not None else sigma_vx
        # Default sigma_x: recover from cloud final spread
        if sigma_x is None:
            # σ_x^f² = σ_x0² + T²·σ_vx², so σ_x0 = sqrt(σ_xf² - T²σ_vx²).
            # Estimate σ_xf from the dataset half_range ≈ 5σ_xf.
            sigma_xf_est = ds_z0.half_range / 5.0
            sigma_x0_sq  = max(sigma_xf_est**2 - (T_DET * sigma_vx)**2, (1e-5)**2)
            sigma_x = math.sqrt(sigma_x0_sq)
        sigma_y = sigma_y if sigma_y is not None else sigma_x

        self._sigma_x  = float(sigma_x)
        self._sigma_y  = float(sigma_y)
        self._sigma_vx = float(sigma_vx)
        self._sigma_vy = float(sigma_vy)

        # ── pixel grid ────────────────────────────────────────────────────────
        edges = _downsampled_edges(ds_z0, bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.edges   = edges
        self.centers = centers

        # ── backend ───────────────────────────────────────────────────────────
        self.xp      = cp if (use_gpu and _HAS_CUPY) else np
        self._lse    = _cp_logsumexp if self.xp is cp else _np_logsumexp
        self.backend = 'cupy' if self.xp is cp else 'numpy'

        # ── shot times ────────────────────────────────────────────────────────
        t_np = (np.arange(self.n_shots, dtype=np.float64)
                if t is None else np.asarray(t, dtype=np.float64).ravel())
        if len(t_np) != self.n_shots:
            raise ValueError(
                f"t has length {len(t_np)}, expected {self.n_shots}."
            )
        self._t = self.xp.asarray(t_np)

        # ── ntheta ────────────────────────────────────────────────────────────
        # Count detected atoms for adaptive quadrature sizing.
        # adaptive_ntheta expects total atoms per shot (both states) at the
        # smaller arm (Z100). state_counts() reads the HDF5 efficiently.
        sc_z100  = ds_z100.state_counts()
        mean_det = float((sc_z100[0] + sc_z100[1]).mean())
        if ntheta is None:
            ntheta = adaptive_ntheta(mean_det)
        self.ntheta         = ntheta
        self.n_theta_coarse = 512
        self._theta_fine   = self.xp.linspace(0, 2 * math.pi, ntheta,     endpoint=False)
        self._theta_coarse = self.xp.linspace(0, 2 * math.pi, 512,        endpoint=False)

        # ── build PSMAP models at 3 phase points (one set per AI) ─────────────
        # PSMAPConditionalImageModel.from_psmap builds scipy interpolators from
        # the PSMAP node values at a fixed phi0.  We need 3 × 2 = 6 models.
        if verbose:
            print("Building PSMAP conditional image models (6 total)…", end=" ", flush=True)
        t0 = time.perf_counter()

        def _make_models(psmap, label):
            """Build 3 PSMAPConditionalImageModel objects at phi0=0, π/2, π."""
            ms = []
            for phi0 in self._PHASE_POINTS:
                m = PSMAPConditionalImageModel.from_psmap(
                    psmap, T_DET, phi0, edges, edges,
                    hermite_order=hermite_order,
                )
                ms.append(m)
            return tuple(ms)  # (model_phi0, model_phi90, model_phi180)

        models_z0  = _make_models(psmap_z0,  "Z0")
        models_z100 = _make_models(psmap_z100, "Z100")
        if verbose:
            print(f"done  ({time.perf_counter()-t0:.1f}s)")

        # ── load and downsample images ─────────────────────────────────────────
        if verbose:
            print("Loading and downsampling images…", end=" ", flush=True)
        t0 = time.perf_counter()

        def _load_downsampled(ds):
            """Return (n_shots, 2, bins, bins) downsampled count arrays."""
            imgs = np.empty((ds.n_shots, 2, bins, bins), dtype=np.float64)
            with __import__('h5py').File(ds.path) as fh:
                for i in range(ds.n_shots):
                    imgs[i, 0] = _downsample_image(fh['images_s0'][i], bins)
                    imgs[i, 1] = _downsample_image(fh['images_s1'][i], bins)
            return imgs

        imgs_z0  = _load_downsampled(ds_z0)   # (n_shots, 2, bins, bins)
        imgs_z100 = _load_downsampled(ds_z100)
        if verbose:
            print(f"done  ({time.perf_counter()-t0:.1f}s)")

        # ── pre-compute pixel statistics per shot per AI ───────────────────────
        # For each shot i and AI z:
        #   1. Estimate cloud COM from the sum image (weighted centroid).
        #   2. Pack into 8-element theta array (initial-space cloud params).
        #   3. Call PSMAPConditionalImageModel.detected_probabilities at 3 phases.
        #   4. Extract (A, Cc, Cs) per pixel per state.
        #   5. Normalise by observed total count → Λ_iz.
        #
        # Storage layout: (n_shots, 2_state, n_pix) for each of A, Cc, Cs.
        # Separate arrays per AI to simplify the GPU kernel.

        A_z0   = np.empty((self.n_shots, 2, self.n_pix), dtype=np.float64)
        Cc_z0  = np.empty_like(A_z0)
        Cs_z0  = np.empty_like(A_z0)
        A_z100  = np.empty_like(A_z0)
        Cc_z100 = np.empty_like(A_z0)
        Cs_z100 = np.empty_like(A_z0)

        Lambda_z0  = np.empty(self.n_shots, dtype=np.float64)
        Lambda_z100 = np.empty(self.n_shots, dtype=np.float64)

        if verbose:
            print(
                f"Pre-computing pixel statistics "
                f"({self.n_shots} shots × 2 AIs × 3 phase pts)…"
            )
        t_precomp = time.perf_counter()

        for i in range(self.n_shots):
            if verbose and (i % max(1, self.n_shots // 20) == 0):
                frac = i / self.n_shots
                w    = 30
                bar  = "█" * int(w * frac) + "░" * (w - int(w * frac))
                elapsed = time.perf_counter() - t_precomp
                eta = (elapsed / max(frac, 1e-9)) * (1 - frac) if frac > 0 else 0
                print(
                    f"  [{bar}] shot {i:4d}/{self.n_shots}"
                    f"  {elapsed:.0f}s elapsed  ETA {eta:.0f}s   ",
                    end="\r", flush=True,
                )

            for (ai_idx, imgs, models, A_arr, Cc_arr, Cs_arr, Lambda_arr) in [
                (0, imgs_z0,  models_z0,  A_z0,  Cc_z0,  Cs_z0,  Lambda_z0),
                (1, imgs_z100, models_z100, A_z100, Cc_z100, Cs_z100, Lambda_z100),
            ]:
                # Sum image (n_pix flattened) for COM estimation
                sum_image = imgs[i, 0] + imgs[i, 1]       # (bins, bins)

                mu_xf, mu_yf = _estimate_cloud_com(sum_image, centers, centers)
                theta = _build_cloud_theta(
                    mu_xf, mu_yf,
                    self._sigma_x, self._sigma_y,
                    self._sigma_vx, self._sigma_vy,
                )

                # Extract pixel statistics from 3-point GH quadrature
                A_sb, Cc_sb, Cs_sb = _extract_pixel_stats(*models, theta)
                # A_sb, Cc_sb, Cs_sb: (2_state, n_pix) each

                A_arr[i]  = A_sb          # store for this shot
                Cc_arr[i] = Cc_sb
                Cs_arr[i] = Cs_sb

                # Atom count normalisation:
                # Λ = N^+ / Σ_{s,b} A_sb  (DC detection probability)
                n_plus   = float(sum_image.sum())
                p_det_bar = float(A_sb.sum())
                Lambda_arr[i] = n_plus / max(p_det_bar, 1e-300)

        if verbose:
            elapsed = time.perf_counter() - t_precomp
            bar = "█" * 30
            print(
                f"  [{bar}] shot {self.n_shots:4d}/{self.n_shots}"
                f"  {elapsed:.0f}s elapsed  done             "
            )

        # ── observed counts ───────────────────────────────────────────────────
        # Flatten spatial dims to n_pix for efficient GPU indexing.
        # n_z0[i, s, b] = count of state-s atoms in pixel b for shot i, Z0 AI.
        n_z0  = imgs_z0.reshape(self.n_shots, 2, self.n_pix)
        n_z100 = imgs_z100.reshape(self.n_shots, 2, self.n_pix)

        # ── transfer to device ────────────────────────────────────────────────
        if verbose:
            print("Transferring pixel statistics to device…", end=" ", flush=True)
        t0 = time.perf_counter()
        xp = self.xp
        self._A_z0   = xp.asarray(A_z0,   dtype=xp.float64)
        self._Cc_z0  = xp.asarray(Cc_z0,  dtype=xp.float64)
        self._Cs_z0  = xp.asarray(Cs_z0,  dtype=xp.float64)
        self._n_z0   = xp.asarray(n_z0,   dtype=xp.float64)
        self._L_z0   = xp.asarray(Lambda_z0,  dtype=xp.float64)

        self._A_z100  = xp.asarray(A_z100,  dtype=xp.float64)
        self._Cc_z100 = xp.asarray(Cc_z100, dtype=xp.float64)
        self._Cs_z100 = xp.asarray(Cs_z100, dtype=xp.float64)
        self._n_z100  = xp.asarray(n_z100,  dtype=xp.float64)
        self._L_z100  = xp.asarray(Lambda_z100, dtype=xp.float64)
        if verbose:
            print(f"done  ({time.perf_counter()-t0:.1f}s)")

        # Summary
        self._n_atoms_mean = float(Lambda_z0.mean() + Lambda_z100.mean()) / 2
        if verbose:
            elapsed = time.perf_counter() - t_total
            print(
                f"\nImageLikelihoodEvaluator ready in {elapsed:.1f}s | "
                f"n_shots={self.n_shots} | bins={bins}×{bins}={self.n_pix} px | "
                f"ntheta={self.ntheta} | backend={self.backend}"
            )

    # ── Internal GPU kernel ────────────────────────────────────────────────────

    def _core_ll(self, delta_phi, theta_grid, n_theta, theta_batch_size=64):
        """
        Compute the full marginal log-likelihood for a given delta_phi vector.

        This is the inner loop called by signal_ll. It loops over batches of
        theta grid points to avoid materialising the full
        (n_shots × n_theta × 2_state × n_pix) tensor in memory at once.

        The computation for each batch:

            Φ_z0_k   = theta_k               (same for all shots)
            Φ_z100_k = theta_k + delta_phi_i  (shot-dependent)

            λ_z,i,s,b(Φ) = Λ_iz · [A_iz,sb + Cc_iz,sb · cos(Φ) + Cs_iz,sb · sin(Φ)]

            ll_isk = Σ_{s,b} [n_b · log(λ_b) − λ_b]   (occupancy sum over pixels)

        Then logsumexp over theta → marginal per-shot ll → sum over shots.

        Parameters
        ----------
        delta_phi : device array (n_shots,) — per-shot differential phase
        theta_grid : device array (n_theta,)
        n_theta : int
        theta_batch_size : int — number of theta values per GPU batch

        Returns
        -------
        float  — marginal log-likelihood
        """
        xp   = self.xp
        n_s  = self.n_shots

        # Accumulate log-likelihood over theta: ll_all[i, k] = log L_i(β, φ_k)
        ll_all = xp.empty((n_s, n_theta), dtype=xp.float64)

        for k_start in range(0, n_theta, theta_batch_size):
            k_end  = min(k_start + theta_batch_size, n_theta)
            bsize  = k_end - k_start
            th_b   = theta_grid[k_start:k_end]   # (bsize,)

            # ── Z0: phase = theta_k (same across all shots) ────────────────
            # Shapes: A_z0 (shots, 2, n_pix)
            #         cos_phi_z0 (bsize,) → broadcast to (1, bsize, 1, 1)
            cos_z0 = xp.cos(th_b)          # (bsize,)
            sin_z0 = xp.sin(th_b)

            lam_z0 = self._L_z0[:, None, None, None] * (
                self._A_z0[:, None, :, :]
                + self._Cc_z0[:, None, :, :] * cos_z0[None, :, None, None]
                + self._Cs_z0[:, None, :, :] * sin_z0[None, :, None, None]
            )
            # lam_z0: (shots, bsize, 2, n_pix)
            lam_z0 = xp.maximum(lam_z0, 1e-300)

            ll_z0 = (
                self._n_z0[:, None, :, :] * xp.log(lam_z0) - lam_z0
            ).sum(axis=(2, 3))   # (shots, bsize)

            # ── Z100: phase = theta_k + delta_phi_i (shot-dependent) ───────
            # Phi_z100: (shots, bsize) = theta_k[None,:] + delta_phi[:,None]
            phi_z100 = th_b[None, :] + delta_phi[:, None]    # (shots, bsize)
            cos_z100 = xp.cos(phi_z100)   # (shots, bsize)
            sin_z100 = xp.sin(phi_z100)

            lam_z100 = self._L_z100[:, None, None, None] * (
                self._A_z100[:, None, :, :]
                + self._Cc_z100[:, None, :, :] * cos_z100[:, :, None, None]
                + self._Cs_z100[:, None, :, :] * sin_z100[:, :, None, None]
            )
            # lam_z100: (shots, bsize, 2, n_pix)
            lam_z100 = xp.maximum(lam_z100, 1e-300)

            ll_z100 = (
                self._n_z100[:, None, :, :] * xp.log(lam_z100) - lam_z100
            ).sum(axis=(2, 3))   # (shots, bsize)

            ll_all[:, k_start:k_end] = ll_z0 + ll_z100

        # Phase marginalisation: sum_i logsumexp_k [ll_i,k] − log(n_theta)
        logL = float(
            xp.sum(self._lse(ll_all, axis=1) - math.log(n_theta))
        )
        return logL

    # ── Public API ─────────────────────────────────────────────────────────────

    def signal_ll(self, As: float, Ac: float, coarse: bool = False) -> float:
        """
        Marginal log-likelihood for the sinusoidal signal model.

        Computes

            log L(β) = Σ_i logsumexp_k [log L_i(β, φ_k)] − log(n_theta)

        where δφ_i = As·sin(2π·f·t_i) + Ac·cos(2π·f·t_i) and φ_k is a uniform
        grid of n_theta values in [0, 2π).

        Parameters
        ----------
        As, Ac : float  — signal sine/cosine amplitudes [rad]
        coarse : bool   — use the fixed 512-point grid (faster, less accurate).
                          Only for coarse optimiser scans.

        Returns
        -------
        float  — log L(As, Ac)
        """
        xp   = self.xp
        tpf  = 2.0 * math.pi * self.f
        dphi = (float(As) * xp.sin(tpf * self._t)
                + float(Ac) * xp.cos(tpf * self._t))   # (n_shots,)

        if coarse:
            return self._core_ll(dphi, self._theta_coarse, 512)
        return self._core_ll(dphi, self._theta_fine, self.ntheta)

    def null_ll(self) -> float:
        """
        Marginal log-likelihood for the null model (As=0, Ac=0).

        Under the null there is no differential signal: δφ_i = 0 for all shots
        and the two AIs share the same common phase φ_i.  This is the reference
        value for computing the log Bayes factor / likelihood-ratio statistic.

        Returns
        -------
        float  — log L(As=0, Ac=0)
        """
        return self.signal_ll(As=0.0, Ac=0.0)

    def __repr__(self) -> str:
        return (
            f"ImageLikelihoodEvaluator("
            f"n_shots={self.n_shots}, "
            f"bins={self.bins}×{self.bins}, "
            f"ntheta={self.ntheta}, "
            f"backend={self.backend!r})"
        )
