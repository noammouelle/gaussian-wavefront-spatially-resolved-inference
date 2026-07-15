import numpy as np

from helpers.response_regime import (
    HarmonicResponse, generator_fields, orbit_diagnostics,
    poisson_phase_information, profiled_information,
)


def grid_response(fn, n=17, t=2.0):
    x = np.linspace(-1, 1, n); v = np.linspace(-.5, .5, n)
    xx, vv = np.meshgrid(x, v, indexing="ij")
    z = np.c_[xx.ravel(), np.zeros(xx.size), vv.ravel(), np.zeros(xx.size)]
    return HarmonicResponse(z, np.full(xx.size, .55), fn(xx+t*vv).ravel(), 0)


def test_final_position_response_has_zero_ballistic_generator():
    response = grid_response(lambda xf: .2 + .1*xf + .05j*xf**2)
    assert generator_fields(response, 2.0)["generator_violation"] < 1e-10


def test_global_amplitude_has_no_shape_information():
    a = np.array([1., 2., 4.]); q = (.2+.1j)*a
    info = poisson_phase_information(a, q, n_atoms=100)
    assert orbit_diagnostics(a, q)["count_only_distance"] < 1e-12
    np.testing.assert_allclose(info["shape"], 0, atol=1e-12)


def test_strong_carrier_has_uniform_phase_information():
    x = np.linspace(0, 2*np.pi, 128, endpoint=False)
    info = poisson_phase_information(np.full(128, .55), .35*np.exp(1j*x), n_atoms=1000)
    assert info["min_median_ratio"] > .99
    assert orbit_diagnostics(np.full(128, .55), .35*np.exp(1j*x))["axis_ratio"] > .99


def test_profile_removes_phase_tangent():
    mu = np.array([3., 4., 5.]); phase = np.array([1., -2., .5])
    value, _, _ = profiled_information(2.5*phase, phase, mu)
    assert value < 1e-12


def test_orthogonal_deformation_survives_profile():
    mu = np.ones(3); phase = np.array([1., -1., 0.]); nuisance = np.array([1., 1., -2.])
    value, raw, _ = profiled_information(nuisance, phase, mu)
    np.testing.assert_allclose(value, raw)
    assert value > 0
