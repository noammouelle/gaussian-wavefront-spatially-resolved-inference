"""Deterministic cloud-image predictions and Fisher information from a PSMAP.

The PSMAP is a single-atom forward model on a regular transverse phase-space
grid. This module integrates that model against a Gaussian cloud distribution,
without Monte Carlo atom sampling, and differentiates the result analytically
using Gaussian score functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PARAMETER_NAMES = (
    "mu_x0",
    "mu_y0",
    "mu_vx0",
    "mu_vy0",
    "sigma_x0",
    "sigma_y0",
    "sigma_vx0",
    "sigma_vy0",
)


def _trapezoid_weights(axis):
    axis = np.asarray(axis, dtype=float)
    if len(axis) == 1:
        return np.ones(1)
    weights = np.empty_like(axis)
    weights[0] = 0.5 * (axis[1] - axis[0])
    weights[-1] = 0.5 * (axis[-1] - axis[-2])
    weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return weights


def _normal_pdf(values, mean, sigma):
    z = (values - mean) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sigma)


def _port_probabilities(amp0, amp1, phase, interfering, phi0):
    return (
        amp0**2
        + amp1**2
        + interfering * 2.0 * amp0 * amp1 * np.cos(phase + phi0)
    )


def _psmap_nodes_and_state_probabilities(psmap, phi0):
    atom_indices = np.asarray(psmap["atom_indices"])
    unique_atoms, first, counts = np.unique(atom_indices, return_index=True, return_counts=True)
    if not np.all(counts == counts[0]):
        raise ValueError("Every PSMAP atom must have the same number of output ports")
    n_atoms = len(unique_atoms)
    n_ports = int(counts[0])

    def by_atom(values):
        return np.asarray(values).reshape(n_atoms, n_ports)

    initial_positions = np.asarray(psmap["initial_positions"])[first]
    initial_velocities = np.asarray(psmap["initial_velocities"])[first]
    coordinates = np.column_stack([
        initial_positions[:, 0], initial_positions[:, 1],
        initial_velocities[:, 0], initial_velocities[:, 1],
    ])
    port_probability = np.maximum(_port_probabilities(
        by_atom(psmap["amp0"]), by_atom(psmap["amp1"]),
        by_atom(psmap["phase_shifts"]),
        by_atom(psmap["is_interfering"]).astype(float), phi0,
    ), 0.0)
    states = by_atom(psmap["states"])
    ground_probability = np.sum(port_probability * (states == 0), axis=1)
    excited_probability = np.sum(port_probability * (states != 0), axis=1)
    return coordinates, ground_probability, excited_probability


def _final_bin_indices(coordinates, t_det, x_edges, y_edges):
    xf = coordinates[:, 0] + float(t_det) * coordinates[:, 2]
    yf = coordinates[:, 1] + float(t_det) * coordinates[:, 3]
    ix = np.searchsorted(x_edges, xf, side="right") - 1
    iy = np.searchsorted(y_edges, yf, side="right") - 1
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    bin_index = np.full(len(coordinates), -1, dtype=np.int64)
    bin_index[inside] = ix[inside] * ny + iy[inside]
    return bin_index, nx * ny


@dataclass(frozen=True)
class FisherResult:
    """Fisher matrix and Cramer-Rao diagnostics in physical parameter units."""

    parameter_names: tuple[str, ...]
    fisher: np.ndarray
    covariance: np.ndarray
    standard_deviations: np.ndarray
    correlation: np.ndarray
    eigenvalues_scaled: np.ndarray
    rank: int
    condition_number: float
    grid_mass: float
    detected_probability: float
    observation_model: str
    n_atoms: float


class PSMAPImageModel:
    """Expected state-resolved 2D images for Gaussian initial clouds."""

    def __init__(self, coordinates, quadrature_weights, ground_probability,
                 excited_probability, bin_index, n_image_bins):
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.quadrature_weights = np.asarray(quadrature_weights, dtype=float)
        self.ground_probability = np.asarray(ground_probability, dtype=float)
        self.excited_probability = np.asarray(excited_probability, dtype=float)
        self.bin_index = np.asarray(bin_index, dtype=np.int64)
        self.n_image_bins = int(n_image_bins)

        n_nodes = len(self.coordinates)
        expected_shapes = {
            "coordinates": (n_nodes, 4),
            "quadrature_weights": (n_nodes,),
            "ground_probability": (n_nodes,),
            "excited_probability": (n_nodes,),
            "bin_index": (n_nodes,),
        }
        for name, shape in expected_shapes.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} has shape {np.shape(getattr(self, name))}; expected {shape}")

    @classmethod
    def from_psmap(cls, psmap, t_det, phi0, x_edges, y_edges):
        # Native-grid quadrature, mainly useful for tests or broad clouds.
        coordinates, ground_probability, excited_probability = _psmap_nodes_and_state_probabilities(psmap, phi0)
        axes = [np.unique(coordinates[:, index]) for index in range(4)]
        axis_weights = [_trapezoid_weights(axis) for axis in axes]
        indices = [np.searchsorted(axis, coordinates[:, index]) for index, axis in enumerate(axes)]
        quadrature_weights = np.prod(np.column_stack([
            weight[index] for weight, index in zip(axis_weights, indices)
        ]), axis=1)
        bin_index, n_image_bins = _final_bin_indices(coordinates, t_det, x_edges, y_edges)
        return cls(coordinates, quadrature_weights, ground_probability, excited_probability, bin_index, n_image_bins)

    @classmethod
    def from_psmap_qmc(cls, psmap, t_det, phi0, x_edges, y_edges,
                       reference_theta, n_samples=2**17, seed=0):
        # Sobol importance quadrature concentrated on the nominal cloud.
        from scipy.interpolate import RegularGridInterpolator
        from scipy.stats import norm, qmc

        reference_theta = np.asarray(reference_theta, dtype=float)
        if reference_theta.shape != (8,) or np.any(reference_theta[4:] <= 0):
            raise ValueError("reference_theta must contain four means and four positive spreads")
        if n_samples <= 0 or n_samples & (n_samples - 1):
            raise ValueError("n_samples must be a positive power of two for Sobol balance")

        nodes, ground_nodes, excited_nodes = _psmap_nodes_and_state_probabilities(psmap, phi0)
        axes = tuple(np.unique(nodes[:, index]) for index in range(4))
        grid_shape = tuple(len(axis) for axis in axes)
        grid_indices = tuple(np.searchsorted(axis, nodes[:, index]) for index, axis in enumerate(axes))
        ground_grid = np.empty(grid_shape)
        excited_grid = np.empty(grid_shape)
        ground_grid[grid_indices] = ground_nodes
        excited_grid[grid_indices] = excited_nodes
        options = dict(bounds_error=True, method="linear")
        ground_interpolator = RegularGridInterpolator(axes, ground_grid, **options)
        excited_interpolator = RegularGridInterpolator(axes, excited_grid, **options)

        unit_samples = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(int(np.log2(n_samples)))
        unit_samples = np.clip(unit_samples, np.finfo(float).eps, 1 - np.finfo(float).eps)
        coordinates = reference_theta[:4] + norm.ppf(unit_samples) * reference_theta[4:]
        inside_map = np.ones(n_samples, dtype=bool)
        for index, axis in enumerate(axes):
            inside_map &= (coordinates[:, index] >= axis[0]) & (coordinates[:, index] <= axis[-1])
        coordinates = coordinates[inside_map]
        if not len(coordinates):
            raise ValueError("No QMC samples lie inside the PSMAP bounds")

        ground_probability = ground_interpolator(coordinates)
        excited_probability = excited_interpolator(coordinates)
        proposal_density = np.prod([
            _normal_pdf(coordinates[:, index], reference_theta[index], reference_theta[index + 4])
            for index in range(4)
        ], axis=0)
        quadrature_weights = 1.0 / (n_samples * proposal_density)
        bin_index, n_image_bins = _final_bin_indices(coordinates, t_det, x_edges, y_edges)
        return cls(coordinates, quadrature_weights, ground_probability, excited_probability, bin_index, n_image_bins)

    def _weights_and_scores(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (8,):
            raise ValueError("theta must contain the eight parameters in PARAMETER_NAMES order")

        means = theta[:4]
        sigmas = theta[4:]
        if np.any(sigmas <= 0):
            raise ValueError("All Gaussian spreads must be positive")

        density = np.ones(len(self.coordinates))
        for index in range(4):
            density *= _normal_pdf(self.coordinates[:, index], means[index], sigmas[index])
        raw_weights = self.quadrature_weights * density
        grid_mass = raw_weights.sum()
        if not np.isfinite(grid_mass) or grid_mass <= 0:
            raise ValueError("Gaussian cloud has negligible or invalid mass on the PSMAP grid")
        weights = raw_weights / grid_mass

        centered = self.coordinates - means
        raw_scores = np.column_stack(
            [
                centered[:, index] / sigmas[index] ** 2
                for index in range(4)
            ]
            + [
                -1.0 / sigmas[index] + centered[:, index] ** 2 / sigmas[index] ** 3
                for index in range(4)
            ]
        )
        scores = raw_scores - weights @ raw_scores
        weight_jacobian = weights[:, None] * scores
        return weights, weight_jacobian, float(grid_mass)

    def probabilities_and_jacobian(self, theta, observation_model="launched"):
        """Return category probabilities and their analytic parameter Jacobian.

        ``launched`` includes a final category for undetected atoms and atoms
        outside the image ROI. It assumes the launched atom count is known.

        ``detected_conditional`` conditions on atoms detected inside the image
        ROI, discarding information from total detection efficiency.
        """
        weights, weight_jacobian, grid_mass = self._weights_and_scores(theta)
        inside = self.bin_index >= 0
        bins = self.bin_index[inside]

        def aggregate(values):
            return np.bincount(
                bins, weights=values[inside], minlength=self.n_image_bins
            )

        ground = aggregate(weights * self.ground_probability)
        excited = aggregate(weights * self.excited_probability)
        probabilities_detected = np.concatenate([ground, excited])

        jacobian_parts = []
        for parameter_index in range(len(PARAMETER_NAMES)):
            derivative_weights = weight_jacobian[:, parameter_index]
            jacobian_parts.append(
                np.concatenate(
                    [
                        aggregate(derivative_weights * self.ground_probability),
                        aggregate(derivative_weights * self.excited_probability),
                    ]
                )
            )
        jacobian_detected = np.column_stack(jacobian_parts)
        detected_probability = float(probabilities_detected.sum())

        if observation_model == "launched":
            probabilities = np.append(probabilities_detected, 1.0 - detected_probability)
            jacobian = np.vstack([jacobian_detected, -jacobian_detected.sum(axis=0)])
        elif observation_model == "detected_conditional":
            if detected_probability <= 0:
                raise ValueError("No atoms are detected inside the image ROI")
            detected_derivative = jacobian_detected.sum(axis=0)
            probabilities = probabilities_detected / detected_probability
            jacobian = (
                jacobian_detected * detected_probability
                - probabilities_detected[:, None] * detected_derivative
            ) / detected_probability**2
        else:
            raise ValueError("observation_model must be 'launched' or 'detected_conditional'")

        return probabilities, jacobian, grid_mass, detected_probability

    def fisher_information(self, theta, n_atoms, parameter_scales,
                           observation_model="launched", probability_floor=1e-15,
                           rcond=1e-10):
        """Calculate the multinomial Fisher information and Cramer-Rao bound."""
        probabilities, jacobian, grid_mass, detected_probability = (
            self.probabilities_and_jacobian(theta, observation_model)
        )
        valid = probabilities > probability_floor
        fisher = float(n_atoms) * (
            jacobian[valid].T @ (jacobian[valid] / probabilities[valid, None])
        )

        scales = np.asarray(parameter_scales, dtype=float)
        if scales.shape != (8,) or np.any(scales <= 0):
            raise ValueError("parameter_scales must contain eight positive values")
        scale_matrix = np.diag(scales)
        fisher_scaled = scale_matrix @ fisher @ scale_matrix
        eigenvalues = np.linalg.eigvalsh(fisher_scaled)
        largest = max(float(eigenvalues[-1]), 0.0)
        threshold = rcond * largest
        rank = int(np.sum(eigenvalues > threshold))
        positive = eigenvalues[eigenvalues > threshold]
        condition_number = float(positive[-1] / positive[0]) if len(positive) else np.inf

        covariance_scaled = np.linalg.pinv(fisher_scaled, rcond=rcond, hermitian=True)
        covariance = scale_matrix @ covariance_scaled @ scale_matrix
        standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard_deviations, standard_deviations)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )

        return FisherResult(
            parameter_names=PARAMETER_NAMES,
            fisher=fisher,
            covariance=covariance,
            standard_deviations=standard_deviations,
            correlation=correlation,
            eigenvalues_scaled=eigenvalues,
            rank=rank,
            condition_number=condition_number,
            grid_mass=grid_mass,
            detected_probability=detected_probability,
            observation_model=observation_model,
            n_atoms=float(n_atoms),
        )



class PSMAPConditionalImageModel:
    # Smooth final-image model using conditional Gaussian quadrature.
    def __init__(self, axes, ground_grid, excited_grid, t_det, x_edges, y_edges,
                 hermite_order=12, interp_method="linear"):
        from scipy.interpolate import RegularGridInterpolator
        options = dict(bounds_error=False, fill_value=None, method=interp_method)
        self.ground_interpolator = RegularGridInterpolator(axes, ground_grid, **options)
        self.excited_interpolator = RegularGridInterpolator(axes, excited_grid, **options)
        self.t_det = float(t_det)
        self.x_edges, self.y_edges = np.asarray(x_edges), np.asarray(y_edges)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.pixel_area = np.outer(np.diff(self.x_edges), np.diff(self.y_edges)).ravel()
        nodes, weights = np.polynomial.hermite.hermgauss(hermite_order)
        self.hermite_nodes = np.sqrt(2.0) * nodes
        self.hermite_weights = np.outer(weights, weights) / np.pi

    @classmethod
    def from_psmap(cls, psmap, t_det, phi0, x_edges, y_edges, hermite_order=12, interp_method="linear"):
        nodes, ground_nodes, excited_nodes = _psmap_nodes_and_state_probabilities(psmap, phi0)
        axes = tuple(np.unique(nodes[:, index]) for index in range(4))
        shape = tuple(len(axis) for axis in axes)
        indices = tuple(np.searchsorted(axis, nodes[:, index]) for index, axis in enumerate(axes))
        ground_grid, excited_grid = np.empty(shape), np.empty(shape)
        ground_grid[indices], excited_grid[indices] = ground_nodes, excited_nodes
        return cls(axes, ground_grid, excited_grid, t_det, x_edges, y_edges, hermite_order, interp_method)

    @staticmethod
    def _conditional_parameters(mu_x0, mu_v0, sigma_x0, sigma_v0, t_det):
        final_mean = mu_x0 + t_det * mu_v0
        final_variance = sigma_x0**2 + t_det**2 * sigma_v0**2
        covariance = t_det * sigma_v0**2
        slope = covariance / final_variance
        conditional_variance = sigma_v0**2 - covariance**2 / final_variance
        return final_mean, final_variance, slope, conditional_variance

    def detected_probabilities(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (8,) or np.any(theta[4:] <= 0):
            raise ValueError("theta must contain four means and four positive spreads")
        x_final, y_final = np.meshgrid(self.x_centers, self.y_centers, indexing="ij")
        xf, yf = x_final.ravel(), y_final.ravel()
        mx, varx, slopex, cond_var_vx = self._conditional_parameters(theta[0], theta[2], theta[4], theta[6], self.t_det)
        my, vary, slopey, cond_var_vy = self._conditional_parameters(theta[1], theta[3], theta[5], theta[7], self.t_det)
        density = _normal_pdf(xf, mx, np.sqrt(varx)) * _normal_pdf(yf, my, np.sqrt(vary))
        mean_vx, mean_vy = theta[2] + slopex * (xf - mx), theta[3] + slopey * (yf - my)
        order = len(self.hermite_nodes)
        shape = (len(xf), order, order)
        vx = np.broadcast_to(mean_vx[:, None, None] + np.sqrt(cond_var_vx) * self.hermite_nodes[None, :, None], shape)
        vy = np.broadcast_to(mean_vy[:, None, None] + np.sqrt(cond_var_vy) * self.hermite_nodes[None, None, :], shape)
        xf_grid, yf_grid = np.broadcast_to(xf[:, None, None], shape), np.broadcast_to(yf[:, None, None], shape)
        points = np.column_stack([(xf_grid - self.t_det * vx).ravel(), (yf_grid - self.t_det * vy).ravel(), vx.ravel(), vy.ravel()])
        ground_conditional = np.sum(self.ground_interpolator(points).reshape(shape) * self.hermite_weights, axis=(1, 2))
        excited_conditional = np.sum(self.excited_interpolator(points).reshape(shape) * self.hermite_weights, axis=(1, 2))
        ground = density * np.maximum(ground_conditional, 0) * self.pixel_area
        excited = density * np.maximum(excited_conditional, 0) * self.pixel_area
        return np.concatenate([ground, excited])

    def probabilities_and_jacobian(self, theta, derivative_steps, observation_model="launched"):
        theta = np.asarray(theta, dtype=float)
        detected = self.detected_probabilities(theta)
        jacobian = np.empty((len(detected), len(theta)))
        for index, step in enumerate(derivative_steps):
            plus, minus = theta.copy(), theta.copy()
            plus[index] += step
            minus[index] -= step
            jacobian[:, index] = (self.detected_probabilities(plus) - self.detected_probabilities(minus)) / (2 * step)
        detected_probability = float(detected.sum())
        if observation_model == "launched":
            return np.append(detected, 1.0 - detected_probability), np.vstack([jacobian, -jacobian.sum(axis=0)]), detected_probability
        if observation_model == "detected_conditional":
            derivative = jacobian.sum(axis=0)
            conditional_jacobian = (jacobian * detected_probability - detected[:, None] * derivative) / detected_probability**2
            return detected / detected_probability, conditional_jacobian, detected_probability
        raise ValueError("observation_model must be launched or detected_conditional")

    def fisher_information(self, theta, n_atoms, parameter_scales, derivative_steps=None, observation_model="launched", probability_floor=1e-15, rcond=1e-10):
        scales = np.asarray(parameter_scales, dtype=float)
        derivative_steps = scales * 1e-2 if derivative_steps is None else np.asarray(derivative_steps)
        probabilities, jacobian, detected_probability = self.probabilities_and_jacobian(theta, derivative_steps, observation_model)
        if observation_model == "launched" and probabilities[-1] < 0:
            raise ValueError("Integrated detected probability exceeds one")
        valid = probabilities > probability_floor
        fisher = float(n_atoms) * (jacobian[valid].T @ (jacobian[valid] / probabilities[valid, None]))
        scale_matrix = np.diag(scales)
        fisher_scaled = scale_matrix @ fisher @ scale_matrix
        eigenvalues = np.linalg.eigvalsh(fisher_scaled)
        threshold = rcond * max(float(eigenvalues[-1]), 0.0)
        rank = int(np.sum(eigenvalues > threshold))
        positive = eigenvalues[eigenvalues > threshold]
        condition_number = float(positive[-1] / positive[0]) if len(positive) else np.inf
        covariance_scaled = np.linalg.pinv(fisher_scaled, rcond=rcond, hermitian=True)
        covariance = scale_matrix @ covariance_scaled @ scale_matrix
        standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard_deviations, standard_deviations)
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
        return FisherResult(PARAMETER_NAMES, fisher, covariance, standard_deviations, correlation, eigenvalues, rank, condition_number, np.nan, detected_probability, observation_model, float(n_atoms))
