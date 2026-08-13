"""
convergence_check.py — grid self-convergence check for a wavefront field,
before trusting it for a real PSMAP generation run.

Samples the SAME field (Gaussian + aberration modes, same convention as
generate_wavefront.py) at several increasing grid resolutions, and uses
aisoptics.GridConvergenceValidator to check that coarser grids agree with
the finest one in their overlap region. This validates self-consistency
under refinement -- it does NOT prove the finest grid is physically exact,
only that you've resolved the spatial structure of the field you asked for
(no default resolution is correct for every choice of --mode; a
high-spatial-frequency aberration needs a finer grid than a smooth one).

Usage
-----
    python convergence_check.py --mode qx=1571 qy=0 amp=0.15 phase=0 \\
        --nxy 5 9 17 33 --nz 41 81 161 321

    # sanity check with no aberration (should converge trivially / near-exactly)
    python convergence_check.py --nxy 5 9 17 --nz 21 41 81

--nxy and --nz must be the same length -- entry i is one grid resolution
(nx=ny=nxy[i], nz=nz[i]), sorted automatically from coarsest to finest.
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

from aisoptics import Backend, GaussianBeam, GridSpec, CompositeField, GridConvergenceValidator
from aisoptics.fields.perturbations import FourierModePerturbation, FourierPerturbation

from generate_wavefront import parse_mode, DEFAULT_WAVELENGTH, DEFAULT_WAIST


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wavelength', type=float, default=DEFAULT_WAVELENGTH)
    p.add_argument('--waist', type=float, default=DEFAULT_WAIST)
    p.add_argument('--focus_z', type=float, default=0.0)
    p.add_argument('--mode', nargs='+', action='append', default=[],
                    help="see generate_wavefront.py --mode; checks the UP (aberrated) field")
    p.add_argument('--xlim', type=float, nargs=2, default=(-0.03, 0.03))
    p.add_argument('--ylim', type=float, nargs=2, default=(-0.03, 0.03))
    p.add_argument('--zlim', type=float, nargs=2, default=(-5.0, 20.0))
    p.add_argument('--nxy', type=int, nargs='+', required=True, help='nx=ny at each resolution to test')
    p.add_argument('--nz', type=int, nargs='+', required=True, help='nz at each resolution to test (same length as --nxy)')
    p.add_argument('--out', default=str(REPO / 'optics' / 'convergence_report.png'))
    args = p.parse_args()

    if len(args.nxy) != len(args.nz):
        p.error('--nxy and --nz must have the same length')
    order = np.argsort(args.nxy)
    nxy_list = [args.nxy[i] for i in order]
    nz_list = [args.nz[i] for i in order]

    backend = Backend("numpy")
    beam = GaussianBeam(wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z,
                         propagation_direction="+z", backend=backend)
    modes = [parse_mode(m) for m in args.mode]
    field_obj = CompositeField(reference=beam, phase_perturbation=FourierPerturbation(modes=modes), mode="exact") \
        if modes else beam

    print(f'{len(modes)} aberration mode(s), {len(nxy_list)} resolutions: ' +
          ', '.join(f'({nxy}x{nxy}x{nz})' for nxy, nz in zip(nxy_list, nz_list)))

    grids, labels = [], []
    for nxy, nz in zip(nxy_list, nz_list):
        grid_spec = GridSpec.from_bounds(xlim=tuple(args.xlim), ylim=tuple(args.ylim), zlim=tuple(args.zlim),
                                          shape=(nxy, nxy, nz))
        grids.append(field_obj.sample(grid_spec))
        labels.append(f'{nxy}x{nxy}x{nz}')

    validator = GridConvergenceValidator(grids, labels=labels, reference='finest')
    report = validator.compare_to_reference()
    print()
    print(report.summary())
    print('\nNote: amplitude_relative_rms/max can look huge near the beam\'s low-'
          'amplitude wings (relative error blows up dividing by ~0 amplitude) --\n'
          'this is a metric artifact of the far tails, not evidence the grid is bad.'
          ' phase_rms is the metric that matters for this pipeline (it\'s what'
          '\nfeeds the PSMAP dphi), and is not sensitive to this issue.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    report.plot_metric('phase_rms', ax=axes[0])
    axes[0].set_title('Phase RMS error vs. finest grid')
    report.plot_metric('amplitude_relative_rms', ax=axes[1])
    axes[1].set_title('Amplitude relative RMS error vs. finest grid')
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f'\nSaved {args.out}')

    log_run(REPO, 'convergence_check', labels[-1],
            wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z,
            modes=[(m.qx, m.qy, m.amplitude, m.phase) for m in modes],
            resolutions=labels, phase_rms=[row['phase_rms'] for row in report.rows],
            amplitude_relative_rms=[row['amplitude_relative_rms'] for row in report.rows],
            out=args.out)
    print('DONE')


if __name__ == '__main__':
    main()
