"""
generate_data.py — synthetic shot datasets via PSMAPSurrogate.

Uses the 4D phase-space maps (PSGRID4D_Z0.h5 and PSGRID4D_Z100.h5) to
generate Gaussian-cloud atom interferometer distributions with shot-to-shot
variation in cloud COM and width, plus a configurable per-shot phase offset.

For each shot the cloud parameters are drawn independently for each z0 source:
  - COM (mu_x0, mu_y0, mu_vx0, mu_vy0): sampled from N(0, mu_*_std)
  - Width (sigma_*): sampled from N(mean, std), clipped to positive
The phase offset phi0[i] is shared between the two z0 values — it is
laser-determined and therefore common to both sources in a given shot.

Differential signal (dark matter / GW)
---------------------------------------
When --signal_amp is nonzero a sinusoidal differential phase is injected into
the TOP interferometer (Z100) only:

    phi0_Z100[i] = phi0[i] + delta_phi[i]
    delta_phi[i] = signal_amp * sin(2π * signal_freq * i + signal_phase)

where i is the shot index (0-based).  phi0_Z0 is unchanged, so the signal
appears only in the differential channel (Z100 − Z0).

Frequency conversion
    If shots are acquired at repetition rate f_rep [Hz], a signal_freq of
    f_s [cycles/shot] corresponds to a physical frequency of f_s * f_rep [Hz].
    Example: f_rep = 1 Hz, signal_freq = 0.05 → 0.05 Hz signal.

Both HDF5 files store delta_phi (the signal array, zero for Z0) so that
analysis code can remove or study the injected signal.

Output
------
    ../data/{run_name}/Z0/data_PROB.h5
    ../data/{run_name}/Z100/data_PROB.h5

Both files are loadable with aispy.analysis.load_data.  Each file contains:
  - All atom-level data (states, positions, velocities, probabilities) for
    all shots concatenated along axis 0.
  - A shot_index dataset identifying which shot each atom belongs to.
  - Per-shot metadata: phi0, delta_phi, mu_*, sigma_* arrays of length n_shots.

Shots are streamed one at a time into an open HDF5 file using resizable
datasets, so peak RAM is O(n_atoms) rather than O(n_shots * n_atoms).

Usage
-----
    python generate_data.py [options]
    python generate_data.py --n_shots 200 --n_atoms 5000
    python generate_data.py --phi0_mode linear --run_name sweep_phi0
    python generate_data.py --signal_amp 0.3 --signal_freq 0.05 --signal_phase 0.0

Options
-------
  --n_shots          Number of shots (default: 100)
  --n_atoms          Atoms per shot (default: 10000)
  --run_name         Output subdirectory name (auto-generated if omitted)
  --seed             RNG seed (default: 0)
  --phi0_mode        Phase offset schedule: random | linear | fixed (default: random)

  -- Differential signal (injected into Z100 only):
  --signal_amp       Amplitude of delta_phi [rad] (default: 0 = no signal)
  --signal_freq      Frequency [cycles/shot] (default: 0)
  --signal_phase     Phase offset of signal [rad] (default: 0)

  -- COM distribution (drawn fresh each shot, independently per z0):
  --mu_x_std         Std of x0/y0 COM [m]   (default: 5e-4)
  --mu_vx_std        Std of vx0/vy0 COM [m/s] (default: 5e-4)

  -- Cloud width distribution:
  --sigma_x_mean / --sigma_x_std    mean/std of σ_x, σ_y [m]   (default: 100e-6 / 20e-6)
  --sigma_vx_mean / --sigma_vx_std  mean/std of σ_vx, σ_vy [m/s] (default: 3.09e-4 / 5e-5)
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', '..', 'local', 'aispy'))
from aispy.psmap import load_psmap, PSMAPSurrogate

# ── detection time (must match phase_space_grids.py: 4*T - 0.2) ──────────────
T_DET = 3.8   # s

# ── MAGIS-100 source heights ──────────────────────────────────────────────────
Z0_VALUES = {0: 'Z0', 100: 'Z100'}


def _make_surrogate(psmap_dir, z0):
    fname = os.path.join(psmap_dir, f'PSGRID4D_Z{z0}.h5')
    return PSMAPSurrogate(load_psmap(fname), t_det=T_DET, use_gpu=False)


def _sample_phi0(rng, n_shots, mode):
    if mode == 'random':
        return rng.uniform(0, 2 * np.pi, n_shots)
    if mode == 'linear':
        return np.linspace(0, 2 * np.pi, n_shots, endpoint=False)
    return np.zeros(n_shots)   # 'fixed'


def _sample_cloud_params(rng, n_shots, args):
    """Draw COM and width parameters for one set of n_shots clouds."""
    mu_x0  = rng.normal(0, args.mu_x_std,  n_shots)
    mu_y0  = rng.normal(0, args.mu_x_std,  n_shots)
    mu_vx0 = rng.normal(0, args.mu_vx_std, n_shots)
    mu_vy0 = rng.normal(0, args.mu_vx_std, n_shots)

    sigma_x  = np.clip(rng.normal(args.sigma_x_mean,  args.sigma_x_std,  n_shots), 1e-7, None)
    sigma_y  = np.clip(rng.normal(args.sigma_x_mean,  args.sigma_x_std,  n_shots), 1e-7, None)
    sigma_vx = np.clip(rng.normal(args.sigma_vx_mean, args.sigma_vx_std, n_shots), 1e-9, None)
    sigma_vy = np.clip(rng.normal(args.sigma_vx_mean, args.sigma_vx_std, n_shots), 1e-9, None)

    return mu_x0, mu_y0, mu_vx0, mu_vy0, sigma_x, sigma_y, sigma_vx, sigma_vy


def _fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.0f}s'
    m, s = divmod(int(seconds), 60)
    return f'{m}m{s:02d}s'


def _iter_shots(surrogate, cloud_params, phi0, n_atoms, z0_m, rng):
    """
    Yield (states, pos, vel, probs, shot_idx) for each shot in sequence.
    Prints an in-place progress bar.  Peak RAM = O(n_atoms), not O(n_shots*n_atoms).
    """
    (mu_x0, mu_y0, mu_vx0, mu_vy0,
     sigma_x, sigma_y, sigma_vx, sigma_vy) = cloud_params
    n_shots = len(phi0)

    t_start    = time.perf_counter()
    shot_times = []

    for i in range(n_shots):
        t0 = time.perf_counter()

        df = surrogate.generate_atoms(
            mu_x0=mu_x0[i],   mu_y0=mu_y0[i],
            mu_vx0=mu_vx0[i], mu_vy0=mu_vy0[i],
            sigma_x=sigma_x[i],   sigma_y=sigma_y[i],
            sigma_vx=sigma_vx[i], sigma_vy=sigma_vy[i],
            phi0=float(phi0[i]),
            natoms=n_atoms,
            rng=rng,
        )
        n_det = len(df)

        states   = df['state'].values.astype(np.int8)
        pos      = np.column_stack([df['xf'].values, df['yf'].values,
                                    np.full(n_det, z0_m)])
        vel      = np.column_stack([df['vxf'].values, df['vyf'].values,
                                    np.zeros(n_det)])
        probs    = df['prob_s0'].values
        shot_idx = np.full(n_det, i, dtype=np.int32)

        shot_times.append(time.perf_counter() - t0)
        elapsed  = time.perf_counter() - t_start
        avg_shot = elapsed / (i + 1)
        eta      = avg_shot * (n_shots - i - 1)
        bar_w    = 20
        filled   = int(bar_w * (i + 1) / n_shots)
        bar      = '█' * filled + '░' * (bar_w - filled)
        print(f'  [{bar}] {i+1:4d}/{n_shots}'
              f'  {avg_shot*1e3:.0f} ms/shot'
              f'  elapsed {_fmt_time(elapsed)}'
              f'  ETA {_fmt_time(eta)}   ',
              end='\r', flush=True)

        yield states, pos, vel, probs, shot_idx

    total = time.perf_counter() - t_start
    print(f'  [{"█"*bar_w}] {n_shots}/{n_shots}'
          f'  {np.mean(shot_times)*1e3:.0f} ms/shot'
          f'  total {_fmt_time(total)}         ')


def _init_h5(f, phi0, delta_phi, cloud_params, signal_params, n_atoms_hint):
    """
    Prepare an open HDF5 file for streaming writes.

    Writes per-shot metadata upfront (small, O(n_shots)) and creates
    empty resizable datasets for the atom-level arrays.  Call _append_shot
    once per shot to fill them.
    """
    (mu_x0, mu_y0, mu_vx0, mu_vy0,
     sigma_x, sigma_y, sigma_vx, sigma_vy) = cloud_params

    # chunk size: one typical shot's worth of atoms (h5py rounds to powers of 2)
    c1 = (n_atoms_hint,)
    c2 = (n_atoms_hint, 3)

    # resizable atom-level datasets (initially empty)
    f.create_dataset('states',           shape=(0,),   maxshape=(None,),   dtype=np.int8,    chunks=c1, compression='gzip')
    f.create_dataset('positions',        shape=(0, 3), maxshape=(None, 3), dtype=np.float64, chunks=c2, compression='gzip')
    f.create_dataset('velocities',       shape=(0, 3), maxshape=(None, 3), dtype=np.float64, chunks=c2, compression='gzip')
    f.create_dataset('probabilities',    shape=(0,),   maxshape=(None,),   dtype=np.float64, chunks=c1, compression='gzip')
    f.create_dataset('phaseShifts',      shape=(0,),   maxshape=(None,),   dtype=np.float64, chunks=c1, compression='gzip')
    f.create_dataset('phaseShiftErrors', shape=(0,),   maxshape=(None,),   dtype=np.float64, chunks=c1, compression='gzip')
    f.create_dataset('interferingFlag',  shape=(0,),   maxshape=(None,),   dtype=np.int8,    chunks=c1, compression='gzip')
    f.create_dataset('shot_index',       shape=(0,),   maxshape=(None,),   dtype=np.int32,   chunks=c1, compression='gzip')

    # per-shot metadata — O(n_shots), written upfront
    f.create_dataset('phi0',      data=phi0)
    f.create_dataset('delta_phi', data=delta_phi)
    f.create_dataset('mu_x0',     data=mu_x0)
    f.create_dataset('mu_y0',     data=mu_y0)
    f.create_dataset('mu_vx0',    data=mu_vx0)
    f.create_dataset('mu_vy0',    data=mu_vy0)
    f.create_dataset('sigma_x',   data=sigma_x)
    f.create_dataset('sigma_y',   data=sigma_y)
    f.create_dataset('sigma_vx',  data=sigma_vx)
    f.create_dataset('sigma_vy',  data=sigma_vy)

    f.attrs['signal_amp']   = signal_params['amp']
    f.attrs['signal_freq']  = signal_params['freq']
    f.attrs['signal_phase'] = signal_params['phase']


def _append_shot(f, states, positions, velocities, probabilities, shot_index):
    """Extend the resizable atom datasets with one shot's data."""
    n = len(states)

    for name, data in [
        ('states',        states),
        ('positions',     positions),
        ('velocities',    velocities),
        ('probabilities', probabilities),
        ('shot_index',    shot_index),
    ]:
        ds  = f[name]
        old = ds.shape[0]
        ds.resize(old + n, axis=0)
        ds[old:old + n] = data

    for name in ('phaseShifts', 'phaseShiftErrors'):
        ds  = f[name]
        old = ds.shape[0]
        ds.resize(old + n, axis=0)
        ds[old:old + n] = 0.0

    ds  = f['interferingFlag']
    old = ds.shape[0]
    ds.resize(old + n, axis=0)
    ds[old:old + n] = 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_shots',  type=int,   default=100)
    p.add_argument('--n_atoms',  type=int,   default=10_000)
    p.add_argument('--run_name', type=str,   default=None)
    p.add_argument('--seed',     type=int,   default=0)
    p.add_argument('--phi0_mode', choices=['random', 'linear', 'fixed'],
                   default='random')

    p.add_argument('--signal_amp',   type=float, default=0.0,
                   help='Amplitude of sinusoidal differential phase injected into Z100 [rad]')
    p.add_argument('--signal_freq',  type=float, default=0.0,
                   help='Signal frequency [cycles/shot]. f_Hz = signal_freq * f_rep_Hz.')
    p.add_argument('--signal_phase', type=float, default=0.0,
                   help='Signal phase offset [rad]')

    p.add_argument('--mu_x_std',       type=float, default=10e-6)
    p.add_argument('--mu_vx_std',      type=float, default=10e-6)
    p.add_argument('--sigma_x_mean',   type=float, default=100e-6)
    p.add_argument('--sigma_x_std',    type=float, default=10e-6)
    p.add_argument('--sigma_vx_mean',  type=float, default=3.09e-4)
    p.add_argument('--sigma_vx_std',   type=float, default=10e-6)

    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    psmap_dir  = os.path.join(script_dir, '..', 'output-files')
    data_root  = os.path.join(script_dir, '..', 'data')

    signal_tag = (f'_sig_A{args.signal_amp:.3f}_f{args.signal_freq:.4f}'
                  if args.signal_amp != 0.0 else '')
    if args.run_name is None:
        args.run_name = (
            f'N{args.n_shots}_A{args.n_atoms}'
            f'_muXStd{args.mu_x_std*1e6:.1f}um'
            f'_muVxStd{args.mu_vx_std*1e6:.1f}um'
            f'_sigX{args.sigma_x_mean*1e6:.0f}um'
            f'_sigVx{args.sigma_vx_mean*1e6:.0f}um'
            f'_sigXStd{args.sigma_x_std*1e6:.1f}um'
            f'_sigVxStd{args.sigma_vx_std*1e6:.1f}um'
            f'_phi0{args.phi0_mode}'
            f'{signal_tag}'
        )

    rng = np.random.default_rng(args.seed)

    # Sinusoidal differential signal injected into Z100 only.
    # delta_phi[i] = signal_amp * sin(2π * signal_freq * i + signal_phase)
    # Frequency conversion: f_Hz = signal_freq [cycles/shot] * f_rep [Hz]
    shot_indices = np.arange(args.n_shots)
    delta_phi = (args.signal_amp
                 * np.sin(2 * np.pi * args.signal_freq * shot_indices
                          + args.signal_phase))
    signal_params = {'amp': args.signal_amp,
                     'freq': args.signal_freq,
                     'phase': args.signal_phase}

    print(f'Run : {args.run_name}')
    print(f'  {args.n_shots} shots × {args.n_atoms} atoms'
          f' = {args.n_shots * args.n_atoms:,} atoms per z0')
    print(f'  phi0 mode : {args.phi0_mode}')
    if args.signal_amp != 0.0:
        print(f'  signal    : amp={args.signal_amp:.3f} rad'
              f'  freq={args.signal_freq:.4f} cyc/shot'
              f'  phase={args.signal_phase:.3f} rad')
        print(f'             (f_Hz = {args.signal_freq:.4f} × f_rep_Hz)')
    print()

    # phi0 is laser-determined → shared across z0 sources within a shot
    phi0 = _sample_phi0(rng, args.n_shots, args.phi0_mode)

    # load surrogates up front (expensive init)
    print('Loading surrogates …', end=' ', flush=True)
    t0 = time.perf_counter()
    surrogates = {z0: _make_surrogate(psmap_dir, z0) for z0 in Z0_VALUES}
    print(f'done  ({_fmt_time(time.perf_counter() - t0)})\n')

    t_total = time.perf_counter()

    for z0_m, z0_label in Z0_VALUES.items():
        print(f'z0 = {z0_m} m  ({z0_label})')
        cloud_params = _sample_cloud_params(rng, args.n_shots, args)

        # delta_phi is added to the top interferometer (Z100) only
        is_top      = (z0_m == 100)
        phi0_eff    = phi0 + delta_phi if is_top else phi0
        delta_phi_z = delta_phi        if is_top else np.zeros(args.n_shots)

        out = os.path.join(data_root, args.run_name, z0_label, 'data_PROB.h5')
        os.makedirs(os.path.dirname(out), exist_ok=True)

        with h5py.File(out, 'w') as f:
            _init_h5(f, phi0_eff, delta_phi_z, cloud_params, signal_params,
                     n_atoms_hint=args.n_atoms)
            for states, pos, vel, probs, shot_idx in _iter_shots(
                    surrogates[z0_m], cloud_params, phi0_eff,
                    args.n_atoms, float(z0_m), rng):
                _append_shot(f, states, pos, vel, probs, shot_idx)

        print(f'  → {out}  ({os.path.getsize(out)/1e6:.1f} MB)\n')

    print(f'Done.  Total wall time: {_fmt_time(time.perf_counter() - t_total)}')


if __name__ == '__main__':
    main()
