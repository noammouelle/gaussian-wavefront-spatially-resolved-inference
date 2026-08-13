"""
generate_wavefront.py — build the down/up beam fields for an interpolated
(wtype='interpolated') ais++ beam, via aisoptics.

Physical model: a perfect, unaberrated Gaussian beam travels down from the
laser to the retroreflecting mirror at z=mirror_z, picks up a wavefront
aberration on reflection (mirror surface imperfections), and travels back
up as the aberrated beam:

    down beam = GaussianBeam(...)                      (no perturbation, sampled directly)
    up beam   = GaussianBeam(...) * exp(i * aberration), evaluated AT THE MIRROR
                PLANE ONLY, then actually propagated (Fresnel/paraxial) to
                every other z the atom might be at when a pulse fires.

Why propagate rather than evaluate the aberration directly at every (x,y,z)
the way the old version of this script did (and the way ais++'s own native
wtype=confocal zernikecoeff_N still does): a wavefront aberration imprinted
at a mirror is a property of that one plane. Its effect away from the
mirror is not the same pattern copied unchanged to every z -- it diffracts.
Evaluating the aberration as a function of (x,y) alone, independent of z,
silently assumes it doesn't (this is exactly what ais++'s GetZernikePhase
does: rho is computed from pos[0],pos[1] only, never pos[2]). Building the
aberration at one reference plane and propagating it with aisoptics'
ParaxialPropagator fixes that -- the resulting field genuinely depends on
how far the atom is from the mirror, the way a real aberrated wavefront
would.

Two aberration bases, combinable and summed at the mirror plane before
propagation:
  --mode     Fourier mode in the transverse plane: amp*cos(qx*x+qy*y+phase)
  --zernike  Zernike polynomial term, Noll-indexed, same convention as
             ais++'s zernikecoeff_N (see aisoptics.ZernikeAberration
             docstring) -- so a --zernike run here and the same coefficient
             passed to ais++'s native wtype=confocal zernikecoeff_N should
             agree closely right at the mirror and diverge with distance;
             that divergence is the diffraction the native path drops.

Both fields are sampled/propagated onto the same 3D grid and exported in
the HDF5 layout ais++'s `wtype=interpolated` requires (x, y, z, phase,
amplitude, export_format=aispp_current_interpolation_hdf5).

CAVEAT (found the hard way -- see the module docstring's "verified" note
below): the propagator is FFT-based (both ParaxialPropagator and
AngularSpectrumPropagator), which implicitly assumes periodic boundary
conditions on the transverse plane. If the field's amplitude hasn't
decayed to ~0 by the edge of --xlim/--ylim, the FFT wraps around and
silently corrupts the propagated field (no error, no warning -- it just
looks like noise once you inspect it). This script checks the imprinted
field's amplitude at the grid edge relative to its peak and warns if it
isn't small; --beam_radius (Zernike aperture) can be much smaller than
--xlim/--ylim (that's fine, even required -- Zernike is only defined for
rho<=1), but --xlim/--ylim themselves must stay set by the GAUSSIAN BEAM's
waist, not by --beam_radius. The defaults (xlim=ylim=+/-0.03,
waist~0.01) satisfy this (edge amplitude ~1e-4 of peak).

Usage
-----
    python generate_wavefront.py --tag my_wavefront \\
        --zernike noll=4 amp=0.05 \\
        --mode qx=1571 qy=0 amp=0.1 phase=0

    # defaults match the beam config in phase_space_grids.py (waist, wavelength)
    python generate_wavefront.py --tag flat_mirror   # no aberration: up == down (sanity check)

Writes <out_dir>/<tag>_down.h5 and <out_dir>/<tag>_up.h5, and logs a
runs_manifest.jsonl entry (stage='generate_wavefront').

See convergence_check.py before trusting a given --nx/--ny/--nz for a real
PSMAP run -- there is no default resolution that is correct for every
aberration; it depends on the spatial frequencies in --mode/--zernike, and
propagation adds its own resolution requirement (the FFT-based propagator
needs the transverse grid fine/wide enough to represent the propagated
field without aliasing).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'helpers'))
from run_manifest import log_run

sys.path.insert(0, str(REPO.parent / 'local' / 'aispy'))
from aispy.utils import kz as AISPY_KZ  # noqa: E402

from aisoptics import Backend, GaussianBeam, GridSpec, AISPPExporter, FieldPlane
from aisoptics import ParaxialPropagator, AngularSpectrumPropagator
from aisoptics.fields.perturbations import (
    FourierModePerturbation, FourierPerturbation, ZernikeAberration, ZernikeSum,
)
from aisoptics.grids.field_grid import FieldGrid
from aisoptics.io.metadata import decode_metadata

DEFAULT_WAVELENGTH = 2 * np.pi / float(AISPY_KZ)   # matches aispy's Sr-87 clock kz
DEFAULT_ZR = 450.085                                # m, matches phase_space_grids.py
DEFAULT_WAIST = np.sqrt(2 * DEFAULT_ZR / float(AISPY_KZ))
DEFAULT_BEAM_RADIUS = 0.03   # m, aperture radius for Zernike rho = r/beam_radius

PROPAGATORS = {'paraxial': ParaxialPropagator, 'angular_spectrum': AngularSpectrumPropagator}


def parse_mode(spec):
    """Parse '--mode qx=1571 qy=0 amp=0.1 phase=0' style key=value tokens."""
    kv = dict(tok.split('=') for tok in spec)
    return FourierModePerturbation(
        qx=float(kv.get('qx', 0.0)),
        qy=float(kv.get('qy', 0.0)),
        amplitude=float(kv['amp']),
        phase=float(kv.get('phase', 0.0)),
    )


def parse_zernike(spec, beam_radius):
    """Parse '--zernike noll=4 amp=0.05' style key=value tokens."""
    kv = dict(tok.split('=') for tok in spec)
    return ZernikeAberration(
        noll_index=int(kv['noll']),
        amplitude=float(kv['amp']),
        beam_radius=beam_radius,
    )


def random_zernike_terms(n, rms, beam_radius, min_noll=4, max_noll=20, split='equal', seed=None):
    """n random, distinct Noll-indexed ZernikeAberration terms whose combined
    phase RMS over the aperture equals `rms` (radians).

    Relies on aisoptics' Zmn being RMS-normalised to 1 over the unit disk
    (Noll convention -- verified numerically against a Monte-Carlo sample of
    the unit disk, see the terminal output of this function for a rerun of
    that check on request) and on distinct Noll terms being orthogonal there,
    so a single term's phase RMS is 2*pi*amplitude and combined RMS adds in
    quadrature: rms = 2*pi*sqrt(sum(amplitude_i**2)).

    min_noll/max_noll default to 4..20 -- excludes piston (1, a constant,
    physically inert) and tip/tilt (2, 3, indistinguishable from a beam
    pointing offset rather than a genuine wavefront distortion). Pass
    min_noll=1 if you actually want those included.

    split='equal' gives every term the same RMS contribution; 'dirichlet'
    draws a random variance split (Dirichlet(1,...,1)) so some terms
    dominate and others are near-negligible -- more representative of a
    real, uneven aberration budget.
    """
    rng = np.random.default_rng(seed)
    candidates = np.arange(min_noll, max_noll + 1)
    if n > len(candidates):
        raise ValueError(f'n={n} exceeds available distinct Noll indices in '
                          f'[{min_noll},{max_noll}] ({len(candidates)})')
    noll_idx = rng.choice(candidates, size=n, replace=False)
    if split == 'equal':
        weights = np.full(n, 1.0 / n)
    elif split == 'dirichlet':
        weights = rng.dirichlet(np.ones(n))
    else:
        raise ValueError(f"split must be 'equal' or 'dirichlet', got {split!r}")
    signs = rng.choice([-1.0, 1.0], size=n)
    amplitudes = signs * (rms / (2.0 * np.pi)) * np.sqrt(weights)
    terms = [ZernikeAberration(noll_index=int(j), amplitude=float(a), beam_radius=beam_radius)
              for j, a in zip(noll_idx, amplitudes)]
    achieved_rms = 2.0 * np.pi * np.sqrt(np.sum(amplitudes ** 2))
    print(f'random Zernike terms (target rms={rms:g}, achieved={achieved_rms:.6g}): ' +
          ', '.join(f'noll={t.noll_index} amp={t.amplitude:.6g}' for t in terms))
    return terms


def load_aispp_field(path):
    """Load a field this script wrote (AISPPExporter's flattened x/y/z +
    phase/amplitude HDF5 layout) back into an aisoptics.FieldGrid -- e.g.
    for visualising exactly what's on disk (what ais++'s wtype=interpolated
    will actually read) rather than re-deriving it analytically. Used by
    optics/notebooks/wavefront_visualization.ipynb's LOAD_FROM_FILE mode.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        x, y, z = f['x'][:], f['y'][:], f['z'][:]
        metadata = decode_metadata(f.attrs.get('metadata'))
        shape = tuple(metadata.get('array_shape_before_flattening', (len(x), len(y), len(z))))
        order = metadata.get('flatten_order', 'C')
        phase = f['phase'][:].reshape(shape, order=order)
        amplitude = f['amplitude'][:].reshape(shape, order=order)
    grid = GridSpec(x=x, y=y, z=z)
    complex_values = amplitude * np.exp(1j * phase)
    return FieldGrid(grid, complex_values=complex_values, backend=Backend('numpy'))


def parse_zernike_random(spec, beam_radius):
    """Parse '--zernike_random n=5 rms=0.1 [seed=1] [min_noll=4] [max_noll=20]
    [split=equal|dirichlet]' style key=value tokens."""
    kv = dict(tok.split('=') for tok in spec)
    return random_zernike_terms(
        n=int(kv['n']), rms=float(kv['rms']), beam_radius=beam_radius,
        min_noll=int(kv.get('min_noll', 4)), max_noll=int(kv.get('max_noll', 20)),
        split=kv.get('split', 'equal'),
        seed=int(kv['seed']) if 'seed' in kv else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True, help='output filename stem and run_manifest label')
    p.add_argument('--out_dir', default=str(REPO / 'optics' / 'fields'))
    p.add_argument('--wavelength', type=float, default=DEFAULT_WAVELENGTH, help='m (default: aispy Sr-87 clock kz)')
    p.add_argument('--waist', type=float, default=DEFAULT_WAIST, help='m (default: matches phase_space_grids.py)')
    p.add_argument('--focus_z', type=float, default=0.0, help='m, beam focus position')
    p.add_argument('--mirror_z', type=float, default=None,
                    help='m, reference plane the aberration is imprinted at and propagated from '
                         '(default: same as --focus_z)')
    p.add_argument('--beam_radius', type=float, default=DEFAULT_BEAM_RADIUS,
                    help='m, aperture radius for Zernike rho=r/beam_radius (default: 0.03)')
    p.add_argument('--propagator', choices=list(PROPAGATORS), default='paraxial',
                    help="paraxial (Fresnel, valid for this beam's small divergence angle "
                         "out to ~100m -- default) or angular_spectrum (more general, no "
                         "small-angle approximation, costlier)")
    p.add_argument('--mode', nargs='+', action='append', default=[],
                    help="one Fourier aberration mode as 'qx=<rad/m> qy=<rad/m> amp=<rad> phase=<rad>' "
                         "(qx/qy/phase default 0). Repeat for multiple modes.")
    p.add_argument('--zernike', nargs='+', action='append', default=[],
                    help="one Zernike aberration term as 'noll=<index> amp=<rad>' "
                         "(same convention as ais++'s zernikecoeff_N). Repeat for multiple terms.")
    p.add_argument('--zernike_random', nargs='+', default=None,
                    help="generate n random Zernike terms with combined phase rms=<rad>, as "
                         "'n=<count> rms=<rad> [seed=<int>] [min_noll=4] [max_noll=20] "
                         "[split=equal|dirichlet]'. Combines with any explicit --zernike terms.")
    p.add_argument('--xlim', type=float, nargs=2, default=(-0.03, 0.03), help='m')
    p.add_argument('--ylim', type=float, nargs=2, default=(-0.03, 0.03), help='m')
    p.add_argument('--zlim', type=float, nargs=2, default=(-5.0, 20.0), help='m')
    p.add_argument('--nx', type=int, default=9)
    p.add_argument('--ny', type=int, default=9)
    p.add_argument('--nz', type=int, default=401)
    args = p.parse_args()

    mirror_z = args.focus_z if args.mirror_z is None else args.mirror_z

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = Backend("numpy")
    grid = GridSpec.from_bounds(xlim=tuple(args.xlim), ylim=tuple(args.ylim), zlim=tuple(args.zlim),
                                 shape=(args.nx, args.ny, args.nz))

    modes = [parse_mode(m) for m in args.mode]
    zterms = [parse_zernike(z, args.beam_radius) for z in args.zernike]
    if args.zernike_random is not None:
        zterms = zterms + parse_zernike_random(args.zernike_random, args.beam_radius)
    print(f'{len(modes)} Fourier mode(s), {len(zterms)} Zernike term(s): ' +
          (', '.join(f'Fourier(qx={m.qx:g},qy={m.qy:g},amp={m.amplitude:g},phase={m.phase:g})' for m in modes) +
           (' ' if modes and zterms else '') +
           ', '.join(f'Zernike(noll={z.noll_index},n={z.n},m={z.m},amp={z.amplitude:g})' for z in zterms)
           if (modes or zterms) else '(none -- up beam will match down beam exactly)'))

    down_beam = GaussianBeam(wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z,
                              propagation_direction="-z", backend=backend)
    up_beam_ref = GaussianBeam(wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z,
                                propagation_direction="+z", backend=backend)

    down_path = out_dir / f'{args.tag}_down.h5'
    up_path = out_dir / f'{args.tag}_up.h5'

    field_down = down_beam.sample(grid)
    AISPPExporter().export_total_field(str(down_path), field_down)
    print(f'wrote {down_path}')

    # --- up beam: build the aberration at the mirror plane, propagate the rest ---
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    perfect_at_mirror = up_beam_ref.complex_amplitude(X, Y, mirror_z * np.ones_like(X))

    aberration_phase = np.zeros_like(X)
    if zterms:
        aberration_phase = aberration_phase + ZernikeSum(zterms).value(X, Y, mirror_z)
    if modes:
        aberration_phase = aberration_phase + FourierPerturbation(modes=modes).value(X, Y, mirror_z)

    aberrated_at_mirror = perfect_at_mirror * np.exp(1j * aberration_phase)

    # FFT-based propagation assumes periodic boundaries: if the field hasn't
    # decayed by the grid edge, it silently wraps around and corrupts the
    # result (no error). Warn rather than fail, since a deliberately
    # aggressive/small grid might be an intentional (if risky) choice.
    edge_amp = np.abs(np.concatenate([aberrated_at_mirror[0, :], aberrated_at_mirror[-1, :],
                                       aberrated_at_mirror[:, 0], aberrated_at_mirror[:, -1]])).max()
    peak_amp = np.abs(aberrated_at_mirror).max()
    edge_ratio = edge_amp / peak_amp if peak_amp > 0 else 0.0
    if edge_ratio > 1e-3:
        print(f'WARNING: field amplitude at the --xlim/--ylim edge is {edge_ratio:.2e} of the peak '
              f'(want << 1e-3). FFT-based propagation assumes periodic boundaries -- this edge '
              f'amplitude is large enough that the propagated field is likely corrupted by '
              f'wrap-around aliasing. Widen --xlim/--ylim relative to --waist (NOT --beam_radius).')

    plane = FieldPlane(x=grid.x, y=grid.y, values=aberrated_at_mirror, z0=mirror_z,
                        wavelength=args.wavelength, backend=backend)

    propagator = PROPAGATORS[args.propagator]()
    field_up = propagator.propagate_to_many(plane, grid.z)
    AISPPExporter().export_total_field(str(up_path), field_up)
    print(f'wrote {up_path}  (propagated from mirror_z={mirror_z} via {args.propagator})')

    log_run(REPO, 'generate_wavefront', args.tag,
            wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z, mirror_z=mirror_z,
            beam_radius=args.beam_radius, propagator=args.propagator,
            modes=[(m.qx, m.qy, m.amplitude, m.phase) for m in modes],
            zernike_terms=[(z.noll_index, z.amplitude) for z in zterms],
            zernike_random_spec=args.zernike_random,
            xlim=list(args.xlim), ylim=list(args.ylim), zlim=list(args.zlim),
            nx=args.nx, ny=args.ny, nz=args.nz,
            down_file=str(down_path), up_file=str(up_path))
    print('DONE')


if __name__ == '__main__':
    main()
