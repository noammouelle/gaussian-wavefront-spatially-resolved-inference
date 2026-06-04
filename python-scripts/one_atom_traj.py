"""
one_atom_traj.py — ais++ input file for a single-atom MAGIS-100-like trajectory.

Parameters from arXiv:2506.0911 (SOTA pulse efficiencies):
  - lmt_order  = 300   (~optimal for 1800 total pulses (butterfly-crossing sequence))
  - T          = 1.11 s (interrogation time per arm; peak GW sensitivity at 0.3 Hz)
  - vz0        = 1.91*9.81 m/s (launch velocity; atom reaches apex at t ≈ T)
  - loopnumber = 2    (double-loop Mach-Zehnder, insensitive to Coriolis)

printtrajectory is enabled so ais++ writes a _TRAJ.h5 file alongside the main
output.  Load it with aispy.trajectory.plot_trajectory to visualise the
spacetime diagram and check that the parameters are reasonable.

Usage
-----
    cd python-scripts/
    python one_atom_traj.py [--lmt_order 229] [--T 1.0] [--vz0 9.81]

    ais++ -i input-files/MAGIS100_TRAJ.aisi -o output-files/MAGIS100_TRAJ.h5

    python - <<'EOF'
    from aispy.trajectory import plot_trajectory
    fig, axes = plot_trajectory("output-files/MAGIS100_TRAJ_TRAJ.h5")
    fig.savefig("trajectory.png", dpi=150, bbox_inches="tight")
    EOF
"""

import argparse
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'local', 'aispy'))
from aispy.utils import AISFlow, pi, hbar, kz

# ── default MAGIS-100 parameters ──────────────────────────────────────────────
LMT_ORDER          = 301                        # must be odd
INTERROGATION_TIME = mp.mpf('1.11')              # s  per arm
LOOP_NUMBER        = 2                           # double-loop MZ for Coriolis insensitivity
RABI_FREQ          = 2 * pi * mp.mpf('1e3')    # rad/s  (1 kHz Rabi)
DT_LMT             = mp.mpf('1e-7')             # s  LMT sub-pulse spacing
VZ0                = 1.91 * mp.mpf('9.81')             # m/s  launch velocity

# derived timing
DETECTION_TIME = 4 * INTERROGATION_TIME - mp.mpf('0.44')

# ── beam parameters ───────────────────────────────────────────────────────────
ZR          = 450.085              # m  Rayleigh range (w0 = 1 cm at Sr-87 clock λ)
BEAM_WAIST  = mp.sqrt(2 * ZR / kz)

def build_param_dict(lmt_order, interrogation_time, vz0):
    """Return a full ais++ parameter dict for one-atom trajectory mode."""
    return {
        'cloud_params': {
            'natoms':       1,
            'initialstate': 0,
            'sigma':        0e-6,     # transverse cloud radius (m)
            'longtemp':     0,
            'transtemp':    0e-9,       # 1 nK
            'x0':           [0.0, 0.0, 0.0],
            'v0':           [0.0, 0.0, float(vz0)],
        },
        'potential_params': {
            'utype': 'linear_pot',
        },
        'sequence_params': {
            't_init':             mp.mpf('0.0'),
            'detectiontime':      DETECTION_TIME,
            'interrogation_time': [interrogation_time],
            'lmt_order':          lmt_order,
            'dt_lmt':             DT_LMT * (lmt_order != 1),
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
            'beam_radius':    1.0, # obsolete (no hard cutoff in Gaussian beam)
            'baseline':       0.0, # obsolete
            'zernike_params': {}, # obsolete
        },
        'simulation_params': {
            'amplitudethreshold': 0,
            'coherencelength':    1.0,      # large → no coherence cutoff
            'usemcbranching':     0,
            'ignoredetuning':     0,
            'usestaticapprox':    0,
            'seed':               42,
            'usedetvolselection': 0,
            'usepathselection':   1,
            'xdet':               [-3e-2, 3e-2], # all obsolete
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
            'printtrajectory':  1,      # write _TRAJ.h5 alongside main output
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lmt_order', type=int,   default=LMT_ORDER,
                   help=f'LMT order (odd integer, default: {LMT_ORDER})')
    p.add_argument('--T',         type=float, default=float(INTERROGATION_TIME),
                   help=f'Interrogation time per arm [s] (default: {float(INTERROGATION_TIME)})')
    p.add_argument('--vz0',       type=float, default=float(VZ0),
                   help=f'Launch velocity vz0 [m/s] (default: {float(VZ0)})')
    p.add_argument('--stem',      type=str,   default='MAGIS100_TRAJ',
                   help='Output file stem (default: MAGIS100_TRAJ)')
    args = p.parse_args()

    if args.lmt_order % 2 == 0:
        p.error('--lmt_order must be odd (ultranarrow MZ requirement)')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    indir  = os.path.join(script_dir, '../input-files')
    os.makedirs(indir,  exist_ok=True)

    T     = mp.mpf(str(args.T))
    vz0   = mp.mpf(str(args.vz0))
    n     = args.lmt_order

    dt_pi  = float(mp.pi / RABI_FREQ)
    t_lmt  = 2 * (n - 1) * (float(DT_LMT) + dt_pi)

    print(f'MAGIS-100 single-atom trajectory')
    print(f'  lmt_order     : {n}  ({n - 1} π-kicks per LMT block)')
    print(f'  T (per arm)   : {args.T} s   →  f_peak = {1 / (3 * args.T):.3f} Hz')
    print(f'  vz0           : {args.vz0} m/s')
    print(f'  Detection t   : {DETECTION_TIME} s')
    print()

    param_dict = build_param_dict(n, T, vz0)
    AISFlow(param_dict, args.stem, indir)


if __name__ == '__main__':
    main()
