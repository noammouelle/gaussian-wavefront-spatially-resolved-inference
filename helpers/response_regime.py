"""Geometry and information diagnostics for first-harmonic PSMAP responses.

The response convention is ``r(z, phi) = a(z) + Re[q(z) exp(i phi)]``.
Functions in the first half are distribution independent; cloud projection and
Poisson information functions explicitly require weights and pixel bins.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HarmonicResponse:
    coordinates: np.ndarray  # x0,y0,vx0,vy0 in SI units
    a: np.ndarray
    q: np.ndarray
    state: int

    def evaluate(self, phi):
        return self.a + np.real(self.q * np.exp(1j * phi))


def harmonic_from_psmap(psmap, state=0):
    """Extract an output-state response using the project's native PSMAP dict."""
    atom = np.asarray(psmap["atom_indices"])
    _, first, counts = np.unique(atom, return_index=True, return_counts=True)
    if not np.all(counts == counts[0]):
        raise ValueError("PSMAP atoms have unequal port counts")
    n, p = len(first), int(counts[0])
    reshape = lambda key: np.asarray(psmap[key]).reshape(n, p)
    pos, vel = np.asarray(psmap["initial_positions"])[first], np.asarray(psmap["initial_velocities"])[first]
    coordinates = np.c_[pos[:, :2], vel[:, :2]]
    select = reshape("states") == state
    amp0, amp1 = reshape("amp0"), reshape("amp1")
    dc = amp0**2 + amp1**2
    contrast = reshape("is_interfering").astype(float) * 2 * amp0 * amp1
    # cos(delta + phi) = Re[exp(i delta) exp(i phi)]
    phasor = contrast * np.exp(1j * reshape("phase_shifts"))
    return HarmonicResponse(coordinates, np.sum(dc * select, axis=1), np.sum(phasor * select, axis=1), state)


def fit_phase_harmonics(samples, phases, max_harmonic=1):
    """Least-squares Fourier fit; returns coefficients and relative RMS residual."""
    phases, y = np.asarray(phases), np.asarray(samples)
    cols = [np.ones_like(phases)]
    for k in range(1, max_harmonic + 1):
        cols += [np.cos(k * phases), np.sin(k * phases)]
    design = np.column_stack(cols)
    coef = np.linalg.lstsq(design, y.reshape(len(phases), -1), rcond=None)[0]
    fitted = (design @ coef).reshape(y.shape)
    residual = np.sqrt(np.mean((y - fitted)**2)) / max(np.sqrt(np.mean((y - y.mean(axis=0))**2)), 1e-15)
    return coef, fitted, float(residual)


def regular_grid(response):
    axes = tuple(np.unique(response.coordinates[:, i]) for i in range(4))
    shape = tuple(map(len, axes))
    idx = tuple(np.searchsorted(axes[i], response.coordinates[:, i]) for i in range(4))
    a, q = np.empty(shape), np.empty(shape, complex)
    a[idx], q[idx] = response.a, response.q
    return axes, a, q


def generator_fields(response, t_det):
    """Return Gx/Gy fields and a dimensionless RMS generator violation."""
    axes, a, q = regular_grid(response)
    def gradients(values):
        out = []
        for dim, axis in enumerate(axes):
            if len(axis) == 1:
                out.append(np.zeros_like(values))
            else:
                out.append(np.gradient(values, axis, axis=dim, edge_order=min(2, len(axis)-1)))
        return out
    da, dqr, dqi = gradients(a), gradients(q.real), gradients(q.imag)
    ga = (da[2] - t_det * da[0], da[3] - t_det * da[1])
    gq = ((dqr[2] - t_det*dqr[0]) + 1j*(dqi[2] - t_det*dqi[0]),
          (dqr[3] - t_det*dqr[1]) + 1j*(dqi[3] - t_det*dqi[1]))
    # Scale derivatives by their coordinate displacement for unit invariance.
    sx, sy, svx, svy = [np.ptp(x) for x in axes]
    fibre = np.mean((np.abs(gq[0])*svx)**2 + (np.abs(gq[1])*svy)**2)
    total = np.mean(sum((np.abs(g)*s)**2 for g, s in zip(dqr, (sx,sy,svx,svy))) +
                    sum((np.abs(g)*s)**2 for g, s in zip(dqi, (sx,sy,svx,svy))))
    return {"axes": axes, "Ga": ga, "Gq": gq,
            "generator_violation": float(np.sqrt(fibre / max(total, 1e-30)))}


def final_bin_indices(coordinates, t_det, x_edges, y_edges):
    xf = coordinates[:, 0] + t_det * coordinates[:, 2]
    yf = coordinates[:, 1] + t_det * coordinates[:, 3]
    ix, iy = np.searchsorted(x_edges, xf, side="right")-1, np.searchsorted(y_edges, yf, side="right")-1
    nx, ny = len(x_edges)-1, len(y_edges)-1
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    out = np.full(len(xf), -1, int); out[ok] = ix[ok]*ny + iy[ok]
    return out, nx*ny


def fibre_symmetrize(values, bins):
    """Equal-weight conditional mean on approximate final-position fibres."""
    values = np.asarray(values); valid = bins >= 0
    n = int(bins[valid].max()+1) if np.any(valid) else 0
    count = np.bincount(bins[valid], minlength=n)
    if np.iscomplexobj(values):
        sums = np.bincount(bins[valid], values[valid].real, minlength=n) + 1j*np.bincount(bins[valid], values[valid].imag, minlength=n)
    else: sums = np.bincount(bins[valid], values[valid], minlength=n)
    means = sums / np.maximum(count, 1)
    out = values.copy(); out[valid] = means[bins[valid]]
    return out


def fibre_variation(values, bins):
    sym = fibre_symmetrize(values, bins)
    valid = bins >= 0
    return float(np.sum(np.abs(values[valid]-sym[valid])**2) / max(np.sum(np.abs(values[valid]-np.mean(values[valid]))**2), 1e-30))


def aggregate_harmonics(a, q, bins, weights=None, n_bins=None):
    valid = bins >= 0; w = np.ones(len(a)) if weights is None else np.asarray(weights)
    n_bins = int(bins[valid].max()+1) if n_bins is None else n_bins
    agg = lambda v: np.bincount(bins[valid], weights=(w*v)[valid], minlength=n_bins)
    return agg(a), agg(q.real) + 1j*agg(q.imag)


def orbit_diagnostics(a, q, phases=None):
    phases = np.linspace(0, 2*np.pi, 181, endpoint=False) if phases is None else np.asarray(phases)
    mu = a[None, :] + np.real(q[None, :] * np.exp(1j*phases[:, None]))
    s = np.linalg.svd(mu-mu.mean(0), compute_uv=False)
    gamma = np.vdot(a, q) / max(np.vdot(a, a).real, 1e-30)
    count_only_distance = np.linalg.norm(q-gamma*a) / max(np.linalg.norm(q), 1e-30)
    coherence = abs(np.sum(q)) / max(np.sum(abs(q)), 1e-30)
    total = mu.sum(1); pi = mu / np.maximum(total[:, None], 1e-15)
    shape_change = np.mean(np.sum((pi-pi.mean(0))**2, axis=1))
    return {"axis_ratio": float(s[1]/s[0]) if len(s)>1 and s[0]>0 else 0.0,
            "singular_values": s[:2], "coherence": float(coherence),
            "count_only_distance": float(count_only_distance), "normalized_shape_variance": float(shape_change),
            "orbit": mu, "phases": phases}


def poisson_phase_information(a, q, n_atoms=1.0, phases=None, floor=1e-15):
    phases = np.linspace(0, 2*np.pi, 181, endpoint=False) if phases is None else np.asarray(phases)
    e = np.exp(1j*phases[:, None]); mu = n_atoms*(a[None,:] + np.real(q[None,:]*e))
    dmu = n_atoms*np.real(1j*q[None,:]*e)
    image = np.sum(dmu*dmu/np.maximum(mu, floor), axis=1)
    lam, dlam = mu.sum(1), dmu.sum(1)
    count = dlam*dlam/np.maximum(lam, floor); shape = np.maximum(image-count, 0)
    return {"phases": phases, "image": image, "count": count, "shape": shape,
            "minimum": float(image.min()), "median": float(np.median(image)),
            "min_median_ratio": float(image.min()/max(np.median(image),floor)),
            "shape_fraction": float(np.mean(shape/np.maximum(image,floor))),
            "worst_phase_sigma": float(1/np.sqrt(max(image.min(),floor)))}


def profiled_information(j_u, tangents, mu, floor=1e-15):
    """Poisson-metric information after projecting out tangent columns."""
    j, T, w = np.asarray(j_u), np.atleast_2d(tangents).T if np.asarray(tangents).ndim == 1 else np.asarray(tangents), 1/np.maximum(mu, floor)
    raw = float(j @ (w*j)); gram = T.T @ (w[:,None]*T)
    cross = T.T @ (w*j); removed = float(cross @ np.linalg.pinv(gram) @ cross)
    return max(raw-removed, 0.0), raw, removed


def apply_fibre_strength(response, bins, alpha):
    af, qf = fibre_symmetrize(response.a, bins), fibre_symmetrize(response.q, bins)
    return HarmonicResponse(response.coordinates, af+alpha*(response.a-af), qf+alpha*(response.q-qf), response.state)


def apply_shear(response, t_det, kappa, axis="x"):
    i, vi = (0,2) if axis == "x" else (1,3)
    xf = response.coordinates[:, i] + t_det*response.coordinates[:, vi]
    return HarmonicResponse(response.coordinates, response.a, response.q*np.exp(1j*kappa*xf), response.state)


def classify_regime(generator_violation, hidden_snr, phase_info, orbit, thresholds=None):
    cfg = {"generator": .05, "hidden_snr": 1., "min_info": 25., "axis_ratio": .2, "min_median": .2}
    if thresholds: cfg.update(thresholds)
    if generator_violation < cfg["generator"]:
        return "I"
    if hidden_snr < cfg["hidden_snr"]:
        return "I/mixed"
    robust = phase_info["minimum"] >= cfg["min_info"] and orbit["axis_ratio"] >= cfg["axis_ratio"] and phase_info["min_median_ratio"] >= cfg["min_median"]
    return "III" if robust else "II"
