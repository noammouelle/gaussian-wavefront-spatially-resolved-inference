"""
phase_space_grids.py — ais++ psgrid input files for a 4D transverse phase-space scan.

Uses the same interferometer parameters as one_atom_traj.py (lmt_order=229,
loopnumber=2, T=1.0 s, MAGIS-100 geometry) but in initmode=psgrid to sweep
the full 4D transverse phase space (x0, y0, vx0, vy0) at fixed vz0=0.

Two files are produced — one per longitudinal position in MAGIS-100:
  z0 = 0 m    (bottom source)
  z0 = 100 m  (top source)

Usage
-----
    cd python-scripts/
    python phase_space_grids.py [options]

    ais++ -i ../input-files/PSGRID4D_Z0.aisi   -o ../output-files/PSGRID4D_Z0.h5
    ais++ -i ../input-files/PSGRID4D_Z100.aisi  -o ../output-files/PSGRID4D_Z100.h5

Options
-------
  --nx, --ny      Grid points in x0 and y0 (default: 25)
  --nvx, --nvy    Grid points in vx0 and vy0 (default: 25)
  --xrange        Half-width of x0 and y0 grid [m] (default: 1e-3)
  --vxrange       Half-width of vx0 and vy0 grid [m/s] (default: 3.09e-3)
"""

import argparse
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'local', 'aispy'))
from aispy.utils import AISFlow, pi, hbar, kz

# ── interferometer parameters (match one_atom_traj.py) ────────────────────────
LMT_ORDER          = 301
INTERROGATION_TIME = mp.mpf('1.11')             # s  per arm
LOOP_NUMBER        = 2                          # double-loop MZ
RABI_FREQ          = 2 * pi * mp.mpf('1e3')   # rad/s
DT_LMT             = mp.mpf('1e-7')            # s
DETECTION_TIME     = 4 * INTERROGATION_TIME - mp.mpf('0.44')  # s
VZ0                = 1.91 * mp.mpf('9.81')            # m/s

ZR         = 450.085               # m  Rayleigh range
BEAM_WAIST = mp.sqrt(2 * ZR / kz)

# ── MAGIS-100 longitudinal positions ──────────────────────────────────────────
Z0_VALUES = [0.0, 100.0]   # m

# ── cloud placeholder values (required by ais++ parser, ignored in psgrid mode) ──
CLOUD_SIGMA_X = 100e-6   # m
CLOUD_TRANSTEMP = 1e-9   # K

# ── default grid half-widths ──────────────────────────────────────────────────
DEFAULT_XRANGE  = 1e-3     # m    (±1 mm)
DEFAULT_VXRANGE = 3.09e-3  # m/s  (±3.09 mm/s, ≈ ±10 σ_v at 1 nK for Sr-87)


def build_param_dict(xgrid, ygrid, zgrid, vxgrid, vygrid, vzgrid):
    vz0 = vzgrid[0]
    return {
        'cloud_params': {
            'initmode':     'psgrid',
            'xgrid':        xgrid,
            'ygrid':        ygrid,
            'zgrid':        zgrid,
            'vxgrid':       vxgrid,
            'vygrid':       vygrid,
            'vzgrid':       vzgrid,
            # Gaussian fields required by parser even in psgrid mode
            'natoms':       1,
            'initialstate': 0,
            'sigma':        CLOUD_SIGMA_X,
            'transtemp':    CLOUD_TRANSTEMP,
            'longtemp':     0,
            'x0':           [0.0, 0.0, float(zgrid[0])],
            'v0':           [0.0, 0.0, float(vz0)],
        },
        'potential_params': {
            'utype': 'linear_pot',
        },
        'sequence_params': {
            't_init':             mp.mpf('0.0'),
            'detectiontime':      DETECTION_TIME,
            'interrogation_time': [INTERROGATION_TIME],
            'lmt_order':          LMT_ORDER,
            'dt_lmt':             DT_LMT,
            'automaticdetuning':  1,
            'frequencychirp':     0,
            'kchirp':             0,
            'ultranarrow':        True,
            'sequencename':       'MZ',
            'loopnumber':         LOOP_NUMBER,
        },
        'pulse_params': {
            'rabi_freq':      RABI_FREQ,
            'wtype':          'gaussian',
            'phi0':           0,
            'kx_psr':         0,
            'ky_psr':         0,
            'waist':          BEAM_WAIST,
            'focallength':    0,
            'zupwardlaser':   0,
            'zdownwardlaser': 0,
            'beam_radius':    1.0,
            'baseline':       0.0,
            'zernike_params': {},
        },
        'simulation_params': {
            'amplitudethreshold': 0,
            'coherencelength':    1.0,
            'usemcbranching':     0,
            'ignoredetuning':     0,
            'usestaticapprox':    0,
            'seed':               -1,
            'usedetvolselection': 0,
            'usepathselection':   1,
            'xdet':               [-3e-2, 3e-2],
            'ydet':               [-3e-2, 3e-2],
            'zdet':               [-3e-2, 3e-2],
            'gslqagabserr':       1e-12,
            'gslqagrelerr':       1e-12,
            'gslkinodeabserr':    1e-9,
            'gslkinoderelerr':    0,
            'gslpulseodeabserr':  1e-9,
            'gslpulseoderelerr':  0,
            'ultrafast':          1,
        },
        'io_params': {
            'printprobs':       0,
            'printwavepackets': 0,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nx',      type=int,   default=25)
    p.add_argument('--ny',      type=int,   default=25)
    p.add_argument('--nvx',     type=int,   default=25)
    p.add_argument('--nvy',     type=int,   default=25)
    p.add_argument('--xrange',  type=float, default=DEFAULT_XRANGE,
                   help='Half-width of x0 and y0 grid [m] (default: 1e-3)')
    p.add_argument('--vxrange', type=float, default=DEFAULT_VXRANGE,
                   help='Half-width of vx0 and vy0 grid [m/s] (default: 3.09e-3)')
    args = p.parse_args()

    for name, val in [('nx', args.nx), ('ny', args.ny),
                      ('nvx', args.nvx), ('nvy', args.nvy)]:
        if val % 2 == 0:
            p.error(f'--{name} must be odd so that 0 is a grid point (got {val})')

    xgrid  = [-args.xrange,  args.xrange,  args.nx]
    ygrid  = [-args.xrange,  args.xrange,  args.ny]
    vxgrid = [-args.vxrange, args.vxrange, args.nvx]
    vygrid = [-args.vxrange, args.vxrange, args.nvy]
    vzgrid = [VZ0, VZ0, 1]  # fixed launch velocity

    n_atoms = args.nx * args.ny * args.nvx * args.nvy

    script_dir = os.path.dirname(os.path.abspath(__file__))
    indir = os.path.join(script_dir, '..', 'input-files')
    os.makedirs(indir, exist_ok=True)

    print(f'4D transverse phase-space grid')
    print(f'  Grid             : {args.nx} × {args.ny} × {args.nvx} × {args.nvy}'
          f' = {n_atoms:,} atoms per file')
    print(f'  x0, y0 range     : ±{args.xrange*1e6:.1f} µm')
    print(f'  vx0, vy0 range   : ±{args.vxrange*1e3:.3f} mm/s')
    print(f'  vz0              : {float(VZ0):.3f} m/s')
    print(f'  lmt_order        : {LMT_ORDER},  loopnumber : {LOOP_NUMBER}')
    print(f'  Detection time   : {float(DETECTION_TIME):.3f} s')
    print()

    stems = {z0: f'PSGRID4D_Z{int(z0)}' for z0 in Z0_VALUES}

    for z0 in Z0_VALUES:
        zgrid  = [z0, z0, 1]
        stem   = stems[z0]

        param_dict = build_param_dict(xgrid, ygrid, zgrid, vxgrid, vygrid, vzgrid)
        AISFlow(param_dict, stem, indir)

        aisi = os.path.join(indir, stem + '.aisi')
        print(f'  z0 = {z0:5.0f} m  →  {aisi}')

    print()
    print('Run ais++ with:')
    for z0, stem in stems.items():
        aisi  = os.path.join(indir, stem + '.aisi')
        h5out = os.path.join(script_dir, '..', 'output-files', stem + '.h5')
        print(f'  ais++ -i {aisi} \\')
        print(f'        -o {h5out}')


if __name__ == '__main__':
    main()
