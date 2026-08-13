"""
convergence_check.py — grid self-convergence check for a wavefront field,
before trusting it for a real PSMAP generation run.

Builds the UP (aberrated) field the same way generate_wavefront.py does --
aberration imprinted at --mirror_z, then propagated (Fresnel/paraxial) to
the rest of the grid -- at several increasing grid resolutions, and uses
aisoptics.GridConvergenceValidator to check that coarser grids agree with
the finest one in their overlap region. This validates self-consistency
under refinement -- it does NOT prove the finest grid is physically exact,
only that you've resolved the spatial structure of the field you asked for
(no default resolution is correct for every choice of --mode/--zernike; a
high-spatial-frequency aberration needs a finer grid than a smooth one, and
propagation adds its own resolution requirement on top).

Usage
-----
    python convergence_check.py --zernike noll=4 amp=0.05 \\
        --nxy 5 9 17 33 --nz 41 81 161 321

    # sanity check with no aberration (should converge trivially / near-exactly)
    python convergence_check.py --nxy 5 9 17 --nz 21 41 81

--nxy and --nz must be the same length -- entry i is one grid resolution
(nx=ny=nxy[i], nz=nz[i]), sorted automatically from coarsest to finest.

--verbose/-v adds: per-resolution build time and edge-amplitude ratio as
each grid is built, the full beam/grid config, and an adjacent-resolution
comparison table (grid i vs. grid i+1) in addition to the default
everything-vs-finest one -- useful for telling genuine convergence apart
from a coincidental match to whichever grid was picked as reference.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'helpers'))
from run_manifest import log_run

sys.path.insert(0, str(REPO.parent / 'local' / 'aispy'))
from aispy.utils import kz as AISPY_KZ  # noqa: E402

from aisoptics import Backend, GaussianBeam, GridSpec, FieldPlane, GridConvergenceValidator
from aisoptics import ParaxialPropagator, AngularSpectrumPropagator
from aisoptics.fields.perturbations import FourierPerturbation, ZernikeSum

from generate_wavefront import (
    parse_mode, parse_zernike, parse_zernike_random,
    DEFAULT_WAVELENGTH, DEFAULT_WAIST, DEFAULT_BEAM_RADIUS, PROPAGATORS,
)


def amplitude_weighted_phase_rms(report, reference_grid, core_threshold=1e-2):
    """Re-derive phase_rms from report's per-point samples/errors, but
    weighted by the REFERENCE grid's local field intensity (and, separately,
    restricted to a "core" mask where amplitude exceeds core_threshold *
    peak amplitude).

    Why: GridConvergenceValidator's raw phase_rms is an unweighted average
    over every sampled point in the grids' overlap region, including the
    beam's far wings where amplitude is numerically ~0. The phase of a
    near-zero complex number is dominated by roundoff/FFT artifacts, not
    physical signal -- so those points contribute noise to phase_rms that
    does not need to shrink monotonically with resolution (can even grow,
    since finer grids place more points deeper into that meaningless-phase
    region). The intensity-weighted and core-masked versions here suppress
    that contamination and better reflect what the grid resolution actually
    does to the physically meaningful (non-negligible amplitude) part of
    the field.
    """
    ref_amp = reference_grid.amplitude
    peak_amp = float(np.max(ref_amp))
    out = []
    for row in report.rows:
        comp = row['comparison']
        pts = report.samples[comp]
        phase_err = report.errors[comp]['phase_error']
        amp = reference_grid.evaluate(pts[:, 0], pts[:, 1], pts[:, 2], quantity='amplitude', bounds_policy='clip')
        amp = np.asarray(amp)
        weight = amp ** 2
        intensity_weighted_rms = float(np.sqrt(np.sum(weight * phase_err ** 2) / np.sum(weight))) if np.sum(weight) > 0 else float('nan')
        core_mask = amp > core_threshold * peak_amp
        core_rms = float(np.sqrt(np.mean(phase_err[core_mask] ** 2))) if core_mask.any() else float('nan')
        out.append({'comparison': comp, 'phase_rms_intensity_weighted': intensity_weighted_rms,
                     'phase_rms_core': core_rms, 'core_fraction': float(core_mask.mean())})
    return out


def build_up_field(grid, wavelength, waist, focus_z, mirror_z, modes, zterms, propagator_name, backend):
    """Same construction as generate_wavefront.py's up beam: aberration
    imprinted at the mirror plane, then propagated to the rest of grid.z."""
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    up_beam_ref = GaussianBeam(wavelength=wavelength, waist=waist, focus_z=focus_z,
                                propagation_direction="+z", backend=backend)
    perfect_at_mirror = up_beam_ref.complex_amplitude(X, Y, mirror_z * np.ones_like(X))

    aberration_phase = np.zeros_like(X)
    if zterms:
        aberration_phase = aberration_phase + ZernikeSum(zterms).value(X, Y, mirror_z)
    if modes:
        aberration_phase = aberration_phase + FourierPerturbation(modes=modes).value(X, Y, mirror_z)

    aberrated_at_mirror = perfect_at_mirror * np.exp(1j * aberration_phase)

    edge_amp = np.abs(np.concatenate([aberrated_at_mirror[0, :], aberrated_at_mirror[-1, :],
                                       aberrated_at_mirror[:, 0], aberrated_at_mirror[:, -1]])).max()
    peak_amp = np.abs(aberrated_at_mirror).max()
    edge_ratio = edge_amp / peak_amp if peak_amp > 0 else 0.0

    plane = FieldPlane(x=grid.x, y=grid.y, values=aberrated_at_mirror, z0=mirror_z,
                        wavelength=wavelength, backend=backend)
    field_up = PROPAGATORS[propagator_name]().propagate_to_many(plane, grid.z)
    return field_up, edge_ratio


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wavelength', type=float, default=DEFAULT_WAVELENGTH)
    p.add_argument('--waist', type=float, default=DEFAULT_WAIST)
    p.add_argument('--focus_z', type=float, default=0.0)
    p.add_argument('--mirror_z', type=float, default=None, help='default: same as --focus_z')
    p.add_argument('--beam_radius', type=float, default=DEFAULT_BEAM_RADIUS)
    p.add_argument('--propagator', choices=list(PROPAGATORS), default='paraxial')
    p.add_argument('--mode', nargs='+', action='append', default=[],
                    help="see generate_wavefront.py --mode; checks the UP (aberrated) field")
    p.add_argument('--zernike', nargs='+', action='append', default=[],
                    help="see generate_wavefront.py --zernike")
    p.add_argument('--zernike_random', nargs='+', default=None,
                    help="see generate_wavefront.py --zernike_random")
    p.add_argument('--xlim', type=float, nargs=2, default=(-0.03, 0.03))
    p.add_argument('--ylim', type=float, nargs=2, default=(-0.03, 0.03))
    p.add_argument('--zlim', type=float, nargs=2, default=(-5.0, 20.0))
    p.add_argument('--nxy', type=int, nargs='+', required=True, help='nx=ny at each resolution to test')
    p.add_argument('--nz', type=int, nargs='+', required=True, help='nz at each resolution to test (same length as --nxy)')
    p.add_argument('--out', default=str(REPO / 'optics' / 'convergence_report.png'))
    p.add_argument('--verbose', '-v', action='store_true',
                    help='print per-resolution build time, edge-amplitude ratio, and the full '
                         'pairwise comparison table (not just the vs-finest summary)')
    args = p.parse_args()

    if len(args.nxy) != len(args.nz):
        p.error('--nxy and --nz must have the same length')
    mirror_z = args.focus_z if args.mirror_z is None else args.mirror_z
    order = np.argsort(args.nxy)
    nxy_list = [args.nxy[i] for i in order]
    nz_list = [args.nz[i] for i in order]

    backend = Backend("numpy")
    modes = [parse_mode(m) for m in args.mode]
    zterms = [parse_zernike(z, args.beam_radius) for z in args.zernike]
    if args.zernike_random is not None:
        zterms = zterms + parse_zernike_random(args.zernike_random, args.beam_radius)

    print(f'{len(modes)} Fourier mode(s), {len(zterms)} Zernike term(s), {len(nxy_list)} resolutions: ' +
          ', '.join(f'({nxy}x{nxy}x{nz})' for nxy, nz in zip(nxy_list, nz_list)))

    if args.verbose:
        print(f'wavelength={args.wavelength:.6g} m  waist={args.waist:.6g} m  focus_z={args.focus_z:g} m  '
              f'mirror_z={mirror_z:g} m  beam_radius={args.beam_radius:g} m  propagator={args.propagator}')
        print(f'xlim={tuple(args.xlim)}  ylim={tuple(args.ylim)}  zlim={tuple(args.zlim)}')

    grids, labels = [], []
    worst_edge_ratio = 0.0
    for nxy, nz in zip(nxy_list, nz_list):
        grid_spec = GridSpec.from_bounds(xlim=tuple(args.xlim), ylim=tuple(args.ylim), zlim=tuple(args.zlim),
                                          shape=(nxy, nxy, nz))
        t0 = time.perf_counter()
        field_up, edge_ratio = build_up_field(grid_spec, args.wavelength, args.waist, args.focus_z, mirror_z,
                                               modes, zterms, args.propagator, backend)
        elapsed = time.perf_counter() - t0
        if args.verbose:
            print(f'  {nxy}x{nxy}x{nz}: built in {elapsed:.2f}s, edge amplitude ratio = {edge_ratio:.2e}')
        worst_edge_ratio = max(worst_edge_ratio, edge_ratio)
        grids.append(field_up)
        labels.append(f'{nxy}x{nxy}x{nz}')

    if worst_edge_ratio > 1e-3:
        print(f'\nWARNING: field amplitude at the --xlim/--ylim edge reaches {worst_edge_ratio:.2e} of the '
              f'peak (want << 1e-3) for at least one resolution tested. FFT-based propagation assumes '
              f'periodic boundaries -- results below may be corrupted by wrap-around aliasing rather than '
              f'reflecting real convergence behaviour. Widen --xlim/--ylim relative to --waist.\n')

    validator = GridConvergenceValidator(grids, labels=labels, reference='finest')
    report = validator.compare_to_reference()
    weighted_rows = amplitude_weighted_phase_rms(report, grids[-1])
    print()
    print(report.summary())
    print('\nCorrected phase_rms (weighted by the finest grid\'s local field intensity, and '
          'restricted to a "core" mask where amplitude > 1% of peak):')
    for row in weighted_rows:
        print(f"  {row['comparison']}: phase_rms_intensity_weighted={row['phase_rms_intensity_weighted']:.6e}, "
              f"phase_rms_core={row['phase_rms_core']:.6e} (core_fraction={row['core_fraction']:.3f} of sampled points)")
    print('\nNote: both amplitude_relative_rms above AND the raw phase_rms above can look huge near '
          'the beam\'s low-amplitude wings -- the phase of a numerically ~0 complex amplitude is '
          'dominated by roundoff/FFT artifacts, not physical signal, and does not need to shrink '
          'monotonically with resolution (can even grow, since finer grids sample more points deeper '
          'into that meaningless-phase region). This is a metric artifact of the far tails, not '
          'evidence the grid is bad -- the corrected phase_rms above is what actually reflects '
          'resolution quality in the illuminated region that matters for the PSMAP dphi.')

    if args.verbose and len(grids) > 1:
        pairwise = validator.pairwise_convergence()
        print('\nAdjacent-resolution comparison (each grid vs. the next-finer one, not vs. the '
              'finest overall) -- a smoothly shrinking phase_rms down this table, not just the '
              'vs-finest one above, is stronger evidence of genuine convergence rather than a '
              'coincidental match to whichever grid happened to be picked as reference:')
        print(pairwise.summary())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    report.plot_metric('phase_rms', ax=axes[0])
    axes[0].set_title('Raw phase RMS error vs. finest grid\n(contaminated by low-amplitude tail phase noise)')
    axes[0].set_yscale('log')
    report.plot_metric('amplitude_relative_rms', ax=axes[1])
    axes[1].set_title('Amplitude relative RMS error vs. finest grid')
    axes[1].set_yscale('log')
    axes[2].plot([r['comparison'] for r in weighted_rows], [r['phase_rms_core'] for r in weighted_rows],
                 'o-', label='core (amp > 1% peak)')
    axes[2].plot([r['comparison'] for r in weighted_rows], [r['phase_rms_intensity_weighted'] for r in weighted_rows],
                 's-', label='intensity-weighted')
    axes[2].set_title('Corrected phase RMS vs. finest grid')
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=8)
    axes[2].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f'\nSaved {args.out}')

    log_run(REPO, 'convergence_check', labels[-1],
            wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z, mirror_z=mirror_z,
            propagator=args.propagator, modes=[(m.qx, m.qy, m.amplitude, m.phase) for m in modes],
            zernike_terms=[(z.noll_index, z.amplitude) for z in zterms],
            zernike_random_spec=args.zernike_random,
            resolutions=labels, phase_rms=[row['phase_rms'] for row in report.rows],
            amplitude_relative_rms=[row['amplitude_relative_rms'] for row in report.rows],
            phase_rms_intensity_weighted=[row['phase_rms_intensity_weighted'] for row in weighted_rows],
            phase_rms_core=[row['phase_rms_core'] for row in weighted_rows],
            worst_edge_ratio=worst_edge_ratio, out=args.out)
    print('DONE')


if __name__ == '__main__':
    main()
