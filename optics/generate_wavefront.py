"""
generate_wavefront.py — build the down/up beam fields for an interpolated
(wtype='interpolated') ais++ beam, via aisoptics.

Physical model: a perfect, unaberrated Gaussian beam travels down from the
laser to the retroreflecting mirror at z=0, picks up a wavefront aberration
on reflection (mirror surface imperfections), and travels back up as the
aberrated beam. Concretely:

    down beam = GaussianBeam(...)                          (no perturbation)
    up beam   = CompositeField(GaussianBeam(...), phase_perturbation=<modes>)

The aberration is currently a sum of Fourier modes in the transverse (x, y)
plane -- see --mode below. Both fields are sampled on the same 3D grid and
exported in the HDF5 layout ais++'s `wtype=interpolated` requires
(x, y, z, phase, amplitude, export_format=aispp_current_interpolation_hdf5).

Usage
-----
    python generate_wavefront.py --tag my_wavefront \\
        --mode qx=1571 qy=0 amp=0.1 phase=0 \\
        --mode qx=0 qy=1571 amp=0.05 phase=1.2

    # defaults match the beam config in phase_space_grids.py (waist, wavelength)
    python generate_wavefront.py --tag flat_mirror   # no --mode: up == down (sanity check)

Writes <out_dir>/<tag>_down.h5 and <out_dir>/<tag>_up.h5, and logs a
runs_manifest.jsonl entry (stage='generate_wavefront').

See convergence_check.py before trusting a given --nx/--ny/--nz for a real
PSMAP run -- there is no default resolution that is correct for every
aberration; it depends on the spatial frequencies you put in --mode.
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

from aisoptics import Backend, GaussianBeam, GridSpec, AISPPExporter, CompositeField
from aisoptics.fields.perturbations import FourierModePerturbation, FourierPerturbation

DEFAULT_WAVELENGTH = 2 * np.pi / float(AISPY_KZ)   # matches aispy's Sr-87 clock kz
DEFAULT_ZR = 450.085                                # m, matches phase_space_grids.py
DEFAULT_WAIST = np.sqrt(2 * DEFAULT_ZR / float(AISPY_KZ))


def parse_mode(spec):
    """Parse '--mode qx=1571 qy=0 amp=0.1 phase=0' style key=value tokens."""
    kv = dict(tok.split('=') for tok in spec)
    return FourierModePerturbation(
        qx=float(kv.get('qx', 0.0)),
        qy=float(kv.get('qy', 0.0)),
        amplitude=float(kv['amp']),
        phase=float(kv.get('phase', 0.0)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True, help='output filename stem and run_manifest label')
    p.add_argument('--out_dir', default=str(REPO / 'optics' / 'fields'))
    p.add_argument('--wavelength', type=float, default=DEFAULT_WAVELENGTH, help='m (default: aispy Sr-87 clock kz)')
    p.add_argument('--waist', type=float, default=DEFAULT_WAIST, help='m (default: matches phase_space_grids.py)')
    p.add_argument('--focus_z', type=float, default=0.0, help='m, focus position (mirror plane)')
    p.add_argument('--mode', nargs='+', action='append', default=[],
                    help="one Fourier mode as 'qx=<rad/m> qy=<rad/m> amp=<rad> phase=<rad>' "
                         "(qx/qy/phase default 0). Repeat --mode for multiple modes. "
                         "Omit entirely for a flat (unaberrated) mirror.")
    p.add_argument('--xlim', type=float, nargs=2, default=(-0.03, 0.03), help='m')
    p.add_argument('--ylim', type=float, nargs=2, default=(-0.03, 0.03), help='m')
    p.add_argument('--zlim', type=float, nargs=2, default=(-5.0, 20.0), help='m')
    p.add_argument('--nx', type=int, default=9)
    p.add_argument('--ny', type=int, default=9)
    p.add_argument('--nz', type=int, default=401)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = Backend("numpy")
    grid = GridSpec.from_bounds(xlim=tuple(args.xlim), ylim=tuple(args.ylim), zlim=tuple(args.zlim),
                                 shape=(args.nx, args.ny, args.nz))

    modes = [parse_mode(m) for m in args.mode]
    print(f'{len(modes)} aberration mode(s): ' +
          (', '.join(f'(qx={m.qx:g}, qy={m.qy:g}, amp={m.amplitude:g}, phase={m.phase:g})' for m in modes)
           if modes else '(none -- up beam will match down beam exactly)'))

    down_beam = GaussianBeam(wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z,
                              propagation_direction="-z", backend=backend)
    up_beam_ref = GaussianBeam(wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z,
                                propagation_direction="+z", backend=backend)

    down_path = out_dir / f'{args.tag}_down.h5'
    up_path = out_dir / f'{args.tag}_up.h5'

    field_down = down_beam.sample(grid)
    AISPPExporter().export_total_field(str(down_path), field_down)
    print(f'wrote {down_path}')

    if modes:
        perturbation = FourierPerturbation(modes=modes)
        up_field_obj = CompositeField(reference=up_beam_ref, phase_perturbation=perturbation, mode="exact")
    else:
        up_field_obj = up_beam_ref
    field_up = up_field_obj.sample(grid)
    AISPPExporter().export_total_field(str(up_path), field_up)
    print(f'wrote {up_path}')

    log_run(REPO, 'generate_wavefront', args.tag,
            wavelength=args.wavelength, waist=args.waist, focus_z=args.focus_z,
            modes=[(m.qx, m.qy, m.amplitude, m.phase) for m in modes],
            xlim=list(args.xlim), ylim=list(args.ylim), zlim=list(args.zlim),
            nx=args.nx, ny=args.ny, nz=args.nz,
            down_file=str(down_path), up_file=str(up_path))
    print('DONE')


if __name__ == '__main__':
    main()
