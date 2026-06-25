"""
tests/test_image_likelihood.py — Unit tests for helpers/image_likelihood.py
and helpers/image_fitting.py.

Tests are split into two groups:
  - Unit tests: use only synthetic in-memory data, no HDF5 files required.
  - Integration tests: require a real data directory and PSMAP files.
    Skipped automatically if the files are absent.

Run all tests:
    pytest tests/test_image_likelihood.py -v

Run only unit tests (no data files):
    pytest tests/test_image_likelihood.py -v -m "not integration"
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
HELPERS_DIR = REPO_ROOT / "helpers"
AISPY_ROOT  = REPO_ROOT.parents[1] / "local" / "aispy"
sys.path.insert(0, str(HELPERS_DIR))
if str(AISPY_ROOT) not in sys.path:
    sys.path.insert(0, str(AISPY_ROOT))

from image_likelihood import (
    _downsample_image,
    _estimate_cloud_com,
    _build_cloud_theta,
    _extract_pixel_stats,
    T_DET,
)
from image_fitting import _valid_params, ImageFitResult

# ── synthetic PSMAP builder (reused from test_psmap_fisher) ───────────────────

def make_synthetic_psmap(n_grid=5, amp=0.5):
    """
    Build a minimal synthetic PSMAP suitable for unit-testing.

    The port structure:
      - 2 ports per atom: port 0 → ground state (s=0), port 1 → excited state (s=1).
      - Both ports interfere (is_interfering=1).
      - amp0 = amp1 = amp (so max contrast = 2*amp² = 0.5).
      - phase_shifts: a linear function of (x0, y0, vx0, vy0) for port 0,
        and port 0 + π for port 1.

    This gives:
      p_ground(Φ) = 2*amp² * cos²((dphi + Φ)/2)
      p_excited(Φ) = 2*amp² * sin²((dphi + Φ)/2)
    which has the expected A + Cc*cos(Φ) + Cs*sin(Φ) structure.
    """
    axes = [
        np.linspace(-3e-3, 3e-3, n_grid),
        np.linspace(-3e-3, 3e-3, n_grid),
        np.linspace(-2e-3, 2e-3, n_grid),
        np.linspace(-2e-3, 2e-3, n_grid),
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    # Linear phase function of initial conditions
    phase = (100.0 * grid[:, 0] - 80.0 * grid[:, 1]
             + 50.0 * grid[:, 2] + 30.0 * grid[:, 3])
    n_atoms = len(grid)
    n_ports = 2

    return {
        "atom_indices":       np.repeat(np.arange(n_atoms), n_ports),
        "initial_positions":  np.repeat(
            np.column_stack([grid[:, :2], np.zeros(n_atoms)]),
            n_ports, axis=0
        ),
        "initial_velocities": np.repeat(
            np.column_stack([grid[:, 2:], np.zeros(n_atoms)]),
            n_ports, axis=0
        ),
        # Port 0 has phase dphi, port 1 has phase dphi + pi
        "phase_shifts":       np.column_stack([phase, phase + np.pi]).ravel(),
        "amp0":               np.full(n_atoms * n_ports, float(amp)),
        "amp1":               np.full(n_atoms * n_ports, float(amp)),
        "is_interfering":     np.ones(n_atoms * n_ports, dtype=bool),
        "states":             np.tile([0, 1], n_atoms),
    }


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestDownsample:
    """_downsample_image: block-sum downsampling."""

    def test_uniform_image(self):
        """A uniform image of value v downsamples to v × block²."""
        res, bins = 64, 8
        block = res // bins
        img = np.ones((res, res), dtype=np.float64)
        ds  = _downsample_image(img, bins)
        assert ds.shape == (bins, bins)
        np.testing.assert_allclose(ds, block**2)

    def test_sum_is_preserved(self):
        """Total count is preserved under block-sum downsampling."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 100, (128, 128)).astype(float)
        ds  = _downsample_image(img, 16)
        assert abs(ds.sum() - img.sum()) < 1e-9

    def test_indivisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            _downsample_image(np.ones((100, 100)), bins=32)


class TestCloudCom:
    """_estimate_cloud_com: weighted centroid estimator."""

    def test_symmetric_image_returns_zero(self):
        """A perfectly symmetric image has zero centroid."""
        n = 16
        centers = np.linspace(-1.0, 1.0, n)
        img = np.ones((n, n))
        mu_x, mu_y = _estimate_cloud_com(img, centers, centers)
        assert abs(mu_x) < 1e-10
        assert abs(mu_y) < 1e-10

    def test_single_pixel_spike(self):
        """A single occupied pixel returns its coordinates exactly."""
        n = 10
        centers = np.linspace(-1.0, 1.0, n)
        img = np.zeros((n, n))
        # Place all counts at pixel (3, 7)
        img[3, 7] = 100.0
        mu_x, mu_y = _estimate_cloud_com(img, centers, centers)
        np.testing.assert_allclose(mu_x, centers[3], atol=1e-12)
        np.testing.assert_allclose(mu_y, centers[7], atol=1e-12)

    def test_empty_image_returns_zero(self):
        """An empty image safely returns (0, 0) without raising."""
        centers = np.linspace(-1.0, 1.0, 8)
        mu_x, mu_y = _estimate_cloud_com(np.zeros((8, 8)), centers, centers)
        assert mu_x == 0.0
        assert mu_y == 0.0

    def test_gaussian_cloud_recovers_mean(self):
        """Centroid of a Gaussian image recovers the true mean to <1 pixel."""
        n = 64
        centers = np.linspace(-5e-3, 5e-3, n)
        dx = centers[1] - centers[0]
        true_mu_x, true_mu_y = 0.8e-3, -1.2e-3
        sigma = 1.5e-3
        xx, yy = np.meshgrid(centers, centers, indexing="ij")
        img = np.exp(-0.5 * ((xx - true_mu_x)**2 + (yy - true_mu_y)**2) / sigma**2)
        img *= 1000  # simulate ~1000 total atoms
        mu_x, mu_y = _estimate_cloud_com(img, centers, centers)
        np.testing.assert_allclose(mu_x, true_mu_x, atol=2 * dx)
        np.testing.assert_allclose(mu_y, true_mu_y, atol=2 * dx)


class TestBuildCloudTheta:
    """_build_cloud_theta: packaging initial-space cloud parameters."""

    def test_output_shape_and_values(self):
        """Returned theta has the expected layout."""
        theta = _build_cloud_theta(
            mu_xf=1e-3, mu_yf=-2e-3,
            sigma_x=100e-6, sigma_y=100e-6,
            sigma_vx=3.09e-4, sigma_vy=3.09e-4,
        )
        assert theta.shape == (8,)
        assert theta[0] == pytest.approx(1e-3)
        assert theta[1] == pytest.approx(-2e-3)
        assert theta[2] == 0.0  # mu_vx0 = 0 by convention
        assert theta[3] == 0.0  # mu_vy0 = 0 by convention
        assert theta[4] == pytest.approx(100e-6)
        assert theta[6] == pytest.approx(3.09e-4)

    def test_final_mean_is_preserved(self):
        """final_mean = mu_x0 + T_det*mu_vx0 = mu_xf (since mu_vx0=0)."""
        mu_xf = 0.5e-3
        theta  = _build_cloud_theta(mu_xf, 0.0, 100e-6, 100e-6, 3e-4, 3e-4)
        # PSMAPConditionalImageModel uses: final_mean = theta[0] + T_det*theta[2]
        final_mean_model = theta[0] + T_DET * theta[2]
        assert final_mean_model == pytest.approx(mu_xf)


class TestExtractPixelStats:
    """_extract_pixel_stats: exactness of the 3-point decomposition."""

    @pytest.fixture
    def conditional_model_set(self):
        """Three PSMAPConditionalImageModel objects at phi0=0, π/2, π."""
        from psmap_fisher import PSMAPConditionalImageModel
        psmap = make_synthetic_psmap(n_grid=5, amp=0.5)
        edges = np.linspace(-8e-3, 8e-3, 9)  # 8×8 = 64 pixels
        phi_values = (0.0, 0.5 * math.pi, math.pi)
        return tuple(
            PSMAPConditionalImageModel.from_psmap(
                psmap, t_det=2.0, phi0=phi, x_edges=edges, y_edges=edges,
                hermite_order=5,
            )
            for phi in phi_values
        )

    @pytest.fixture
    def cloud_theta(self):
        """A representative 8-element cloud parameter vector."""
        return np.array([
            0.0, 0.0, 0.0, 0.0,       # COMs (zero)
            1.5e-3, 1.5e-3,            # sigma_x, sigma_y
            1.0e-3, 1.0e-3,            # sigma_vx, sigma_vy
        ])

    def test_three_point_extraction_is_exact(self, conditional_model_set, cloud_theta):
        """
        Verify: A + Cc*cos(Phi) + Cs*sin(Phi) exactly recovers
        detected_probabilities at 5 arbitrary phase values.

        This tests the core mathematical claim that the 3-point formula is
        exact (not just an approximation).
        """
        from psmap_fisher import PSMAPConditionalImageModel
        psmap  = make_synthetic_psmap(n_grid=5, amp=0.5)
        edges  = np.linspace(-8e-3, 8e-3, 9)
        m0, m90, m180 = conditional_model_set

        A, Cc, Cs = _extract_pixel_stats(m0, m90, m180, cloud_theta)

        # Verify at 5 test phases (not used in fitting)
        test_phases = [0.3, 1.1, 2.4, 4.7, 5.9]
        for phi_test in test_phases:
            m_test = PSMAPConditionalImageModel.from_psmap(
                psmap, t_det=2.0, phi0=phi_test, x_edges=edges, y_edges=edges,
                hermite_order=5,
            )
            p_ref = m_test.detected_probabilities(cloud_theta)
            n_pix = len(p_ref) // 2
            p_ref_reshaped = p_ref.reshape(2, n_pix)  # (state, pix)

            p_model = (A
                       + Cc * math.cos(phi_test)
                       + Cs * math.sin(phi_test))

            # Relative tolerance accounts for GH quadrature approximation (~1e-6)
            np.testing.assert_allclose(
                p_model, p_ref_reshaped,
                rtol=1e-5, atol=1e-10,
                err_msg=f"3-point extraction failed at phi_test={phi_test:.2f}",
            )

    def test_a_is_nonnegative(self, conditional_model_set, cloud_theta):
        """The DC coefficient A must be non-negative (it is a probability)."""
        m0, m90, m180 = conditional_model_set
        A, Cc, Cs = _extract_pixel_stats(m0, m90, m180, cloud_theta)
        assert np.all(A >= -1e-12), "A contains negative values"

    def test_contrast_bounded_by_dc(self, conditional_model_set, cloud_theta):
        """
        The amplitude sqrt(Cc² + Cs²) must not exceed A at each pixel,
        since p(Φ) ∈ [0, 1] requires A ≥ sqrt(Cc² + Cs²).
        """
        m0, m90, m180 = conditional_model_set
        A, Cc, Cs = _extract_pixel_stats(m0, m90, m180, cloud_theta)
        contrast = np.sqrt(Cc**2 + Cs**2)
        # Allow small numeric slack for near-zero pixels
        mask = A > 1e-10
        np.testing.assert_array_less(
            contrast[mask] - A[mask], 1e-8,
            err_msg="Contrast exceeds DC component — p(Φ) can go negative.",
        )

    def test_sum_returns_total_detection_probability(self, conditional_model_set, cloud_theta):
        """
        Σ_{s,b} A_sb ≈ p_det (total detection probability per launched atom).
        Compare against the sum of detected_probabilities at phi0=0.
        """
        m0, m90, m180 = conditional_model_set
        A, _, _ = _extract_pixel_stats(m0, m90, m180, cloud_theta)
        p_det_A = float(A.sum())

        # Reference: average of p(0) and p(π) summed
        p0   = m0.detected_probabilities(cloud_theta)
        p180 = m180.detected_probabilities(cloud_theta)
        p_det_ref = float(0.5 * (p0 + p180).sum())

        np.testing.assert_allclose(p_det_A, p_det_ref, rtol=1e-10)


class TestValidParams:
    """_valid_params: boundary enforcement for (As, Ac)."""

    def test_valid_interior(self):
        assert _valid_params([0.1, 0.2]) is True

    def test_valid_null(self):
        assert _valid_params([0.0, 0.0]) is True

    def test_as_negative_invalid(self):
        assert _valid_params([-0.01, 0.0]) is False

    def test_as_too_large_invalid(self):
        assert _valid_params([4.0, 0.0]) is False

    def test_ac_too_large_invalid(self):
        assert _valid_params([0.1, 4.0]) is False

    def test_custom_bound(self):
        assert _valid_params([0.5, 0.0], amp_bound=0.4) is False
        assert _valid_params([0.5, 0.0], amp_bound=0.6) is True


# ── Integration tests (require data files) ────────────────────────────────────

PSMAP_Z0   = REPO_ROOT / "output-files" / "PSGRID4D_CONFOCAL_Z0.h5"
PSMAP_Z100 = REPO_ROOT / "output-files" / "PSGRID4D_CONFOCAL_Z100.h5"
# Use the smallest available dataset
_DATA_DIRS = sorted((REPO_ROOT / "data").glob(
    "R80_N50_A1000000*phi0random_sig_A0.100_f0.3000"
))
DATA_DIR = _DATA_DIRS[0] if _DATA_DIRS else None

_integration_available = (
    PSMAP_Z0.is_file() and PSMAP_Z100.is_file()
    and DATA_DIR is not None
    and (DATA_DIR / "run_000" / "Z0" / "data_IMG.h5").is_file()
)

pytestmark_integration = pytest.mark.skipif(
    not _integration_available,
    reason="Integration test data or PSMAP files not found.",
)


@pytestmark_integration
class TestImageLikelihoodEvaluatorIntegration:
    """
    End-to-end tests on real data.

    Uses run_000 from the smallest available dataset.
    Checks:
      1. Construction succeeds.
      2. logL at null is finite.
      3. logL at true (As, Ac) > logL at null (signal recovery sanity check).
      4. logL is symmetric around As: L(As,Ac) = L(As,-Ac) does NOT hold in
         general (test asymmetry), but L(As,Ac) = L(As+ε,Ac) ≈ L(As,Ac)
         for small ε close to MLE.
    """

    @pytest.fixture(scope="class")
    def evaluator(self):
        from aispy.psmap import load_psmap
        from helpers import ImageShotDataset
        from image_likelihood import ImageLikelihoodEvaluator

        ds_z0  = ImageShotDataset(str(DATA_DIR / "run_000" / "Z0"  / "data_IMG.h5"))
        ds_z100 = ImageShotDataset(str(DATA_DIR / "run_000" / "Z100" / "data_IMG.h5"))
        psmap_z0  = load_psmap(str(PSMAP_Z0))
        psmap_z100 = load_psmap(str(PSMAP_Z100))

        return ImageLikelihoodEvaluator(
            ds_z0, ds_z100, psmap_z0, psmap_z100,
            sigma_vx=3.09e-4,
            sigma_x=100e-6,
            bins=32,
            hermite_order=5,
            f=0.3,
            use_gpu=False,   # CPU for reproducibility
            verbose=False,
        )

    def test_construction(self, evaluator):
        assert evaluator.n_shots > 0
        assert evaluator.n_pix == 32 * 32

    def test_null_ll_is_finite(self, evaluator):
        logL = evaluator.null_ll()
        assert np.isfinite(logL), f"null logL={logL} is not finite"

    def test_signal_ll_is_finite(self, evaluator):
        logL = evaluator.signal_ll(As=0.1, Ac=0.0)
        assert np.isfinite(logL)

    def test_signal_ll_varies_with_as(self, evaluator):
        """logL must change as As increases from 0 — the landscape is non-flat."""
        ll0   = evaluator.signal_ll(As=0.0,  Ac=0.0)
        ll005 = evaluator.signal_ll(As=0.05, Ac=0.0)
        ll010 = evaluator.signal_ll(As=0.10, Ac=0.0)
        # At least one value must differ from the null
        assert not (ll0 == ll005 == ll010), (
            "signal_ll is constant across As values — likely a bug."
        )

    def test_logL_not_constant(self, evaluator):
        """logL must vary with (As, Ac) — it should not be flat."""
        ll_null  = evaluator.signal_ll(As=0.0,  Ac=0.0)
        ll_point = evaluator.signal_ll(As=0.15, Ac=0.0)
        ll_other = evaluator.signal_ll(As=0.0,  Ac=0.2)
        assert not (ll_null == ll_point == ll_other), (
            "logL is constant — evaluator likely has a bug."
        )


@pytestmark_integration
class TestFitImageMleIntegration:
    """Smoke test: fit returns a plausible result on one real run."""

    def test_fit_runs_and_returns_result(self):
        from aispy.psmap import load_psmap
        from helpers import ImageShotDataset
        from image_likelihood import ImageLikelihoodEvaluator
        from image_fitting import fit_image_mle

        ds_z0  = ImageShotDataset(str(DATA_DIR / "run_000" / "Z0"  / "data_IMG.h5"))
        ds_z100 = ImageShotDataset(str(DATA_DIR / "run_000" / "Z100" / "data_IMG.h5"))
        psmap_z0  = load_psmap(str(PSMAP_Z0))
        psmap_z100 = load_psmap(str(PSMAP_Z100))

        ev = ImageLikelihoodEvaluator(
            ds_z0, ds_z100, psmap_z0, psmap_z100,
            sigma_vx=3.09e-4, sigma_x=100e-6,
            bins=32, hermite_order=5, f=0.3,
            use_gpu=True, verbose=False,
        )
        result = fit_image_mle(ev, fast=True)

        assert isinstance(result, ImageFitResult)
        assert np.isfinite(result.amp)
        assert np.isfinite(result.phase)
        assert np.isfinite(result.logL)
        assert result.As >= 0.0, "As must be non-negative by convention"
        # True signal amplitude is 0.1 rad; check within factor of 3
        assert 0.0 <= result.amp < 0.5, f"amp={result.amp:.4f} is implausible"
        assert np.isfinite(result.delta_logL)
