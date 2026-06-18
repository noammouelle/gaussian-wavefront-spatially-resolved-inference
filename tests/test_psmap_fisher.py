import numpy as np

from helpers.psmap_fisher import PARAMETER_NAMES, PSMAPConditionalImageModel, PSMAPImageModel


def make_synthetic_psmap():
    axes = [
        np.linspace(-3e-3, 3e-3, 5),
        np.linspace(-3e-3, 3e-3, 5),
        np.linspace(-2e-3, 2e-3, 5),
        np.linspace(-2e-3, 2e-3, 5),
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    phase = 450 * grid[:, 0] - 300 * grid[:, 1] + 700 * grid[:, 2] + 250 * grid[:, 3]
    n_atoms = len(grid)
    n_ports = 2

    initial_positions = np.repeat(
        np.column_stack([grid[:, :2], np.zeros(n_atoms)]), n_ports, axis=0
    )
    initial_velocities = np.repeat(
        np.column_stack([grid[:, 2:], np.zeros(n_atoms)]), n_ports, axis=0
    )
    phase_shifts = np.column_stack([phase, phase + np.pi]).ravel()
    return {
        "atom_indices": np.repeat(np.arange(n_atoms), n_ports),
        "initial_positions": initial_positions,
        "initial_velocities": initial_velocities,
        "phase_shifts": phase_shifts,
        "amp0": np.full(n_atoms * n_ports, 0.5),
        "amp1": np.full(n_atoms * n_ports, 0.5),
        "is_interfering": np.ones(n_atoms * n_ports, dtype=bool),
        "states": np.tile([0, 1], n_atoms),
    }


def make_model():
    edges = np.linspace(-8e-3, 8e-3, 9)
    return PSMAPImageModel.from_psmap(
        make_synthetic_psmap(), t_det=2.0, phi0=0.7, x_edges=edges, y_edges=edges
    )


def test_analytic_probability_jacobian_matches_finite_difference():
    model = make_model()
    theta = np.array([1e-4, -2e-4, 5e-5, -8e-5, 1.2e-3, 1.1e-3, 8e-4, 7e-4])
    probabilities, jacobian, _, _ = model.probabilities_and_jacobian(theta)

    for index in range(len(PARAMETER_NAMES)):
        step = 1e-7
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference = (
            model.probabilities_and_jacobian(plus)[0]
            - model.probabilities_and_jacobian(minus)[0]
        ) / (2 * step)
        np.testing.assert_allclose(jacobian[:, index], finite_difference, rtol=2e-4, atol=2e-7)

    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(jacobian.sum(axis=0), 0.0, atol=1e-10)


def test_crlb_scales_with_inverse_square_root_atom_number():
    model = make_model()
    theta = np.array([1e-4, -2e-4, 5e-5, -8e-5, 1.2e-3, 1.1e-3, 8e-4, 7e-4])
    scales = np.array([1e-3] * 4 + [1e-3] * 4)
    result_1 = model.fisher_information(theta, 10_000, scales)
    result_4 = model.fisher_information(theta, 40_000, scales)

    np.testing.assert_allclose(
        result_4.standard_deviations,
        result_1.standard_deviations / 2,
        rtol=1e-10,
        atol=1e-15,
    )
    assert result_1.rank > 0



def test_conditional_image_model_produces_finite_fisher_bound():
    theta = np.array([0, 0, 0, 0, 3e-4, 3e-4, 2e-4, 2e-4])
    edges = np.linspace(-8e-3, 8e-3, 17)
    model = PSMAPConditionalImageModel.from_psmap(
        make_synthetic_psmap(), 2.0, 0.7, edges, edges, hermite_order=8
    )
    result = model.fisher_information(theta, 10_000, np.full(8, 1e-4))

    assert result.rank > 0
    assert 0 < result.detected_probability <= 1
    assert np.all(np.isfinite(result.standard_deviations))
