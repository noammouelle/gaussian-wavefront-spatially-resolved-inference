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

Multiple independent runs
--------------------------
Use --n_runs to generate several repeated experiments at the same parameters.
Each run gets an independent RNG derived from the master --seed via
numpy.random.SeedSequence.spawn(), so run k always gets the same noise
regardless of how the job was split or resumed.

Directory layout with n_runs > 1:

    ../data/{run_name}/run_000/Z0/data_IMG.h5
    ../data/{run_name}/run_000/Z100/data_IMG.h5
    ../data/{run_name}/run_001/Z0/data_IMG.h5
    ...

Use --run_start K to resume a partial job starting at run index K without
re-simulating earlier runs.

Output format
-------------
Each data_IMG.h5 file contains:
  - images_s0 : (n_shots, res, res) uint16  — state-0 atom count per pixel
  - images_s1 : (n_shots, res, res) uint16  — state-1 atom count per pixel
  - Per-shot metadata: phi0, delta_phi, mu_*, sigma_* arrays of length n_shots
  - File-level attrs: image_half_range, image_res, z0_m, n_atoms_launched,
                      signal_amp, signal_freq, signal_phase

The image bin edges are symmetric: linspace(-half_range, half_range, res+1),
where half_range is computed from the cloud parameters at generation time
(5σ cloud body + 3σ COM jitter, rounded up to the nearest mm).  This makes
all files for the same experiment pixel-aligned by construction.

Usage
-----
    python generate_data.py [options]
    python generate_data.py --n_shots 200 --n_atoms 5000
    python generate_data.py --n_runs 100 --n_shots 100 --n_atoms 10000
    python generate_data.py --n_runs 100 --run_start 50   # resume from run 50
    python generate_data.py --signal_amp 0.1 --signal_freq 0.3 --signal_phase 0.5

Options
-------
  --n_shots          Number of shots per run (default: 100)
  --n_atoms          Atoms per shot (default: 10000)
  --n_runs           Number of independent runs (default: 1)
  --run_start        First run index to simulate, for resuming (default: 0)
  --run_name         Output subdirectory name (auto-generated if omitted)
  --seed             Master RNG seed; child seeds are derived via SeedSequence (default: 0)
  --phi0_mode        Phase offset schedule: random | linear | fixed (default: random)
  --image_res        Image resolution in pixels (default: 2048)

  -- Differential signal (injected into Z100 only):
  --signal_amp       Amplitude of delta_phi [rad] (default: 0 = no signal)
  --signal_freq      Frequency [cycles/shot] (default: 0)
  --signal_phase     Phase offset of signal [rad] (default: 0)

  -- COM distribution (drawn fresh each shot, independently per z0):
  --mu_x_std         Std of x0/y0 COM [m]   (default: 10e-6)
  --mu_vx_std        Std of vx0/vy0 COM [m/s] (default: 10e-6)

  -- Cloud width distribution:
  --sigma_x_mean / --sigma_x_std    mean/std of σ_x, σ_y [m]   (default: 100e-6 / 10e-6)
  --sigma_vx_mean / --sigma_vx_std  mean/std of σ_vx, σ_vy [m/s] (default: 3.09e-4 / 10e-6)
"""

import argparse
import math
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
    fname = os.path.join(psmap_dir, f'PSGRID4D_CONFOCAL_Z{z0}.h5')
    return PSMAPSurrogate(load_psmap(fname), t_det=T_DET, use_gpu=True)


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


def _compute_image_half_range(args):
    """
    Compute the half-range for image binning from the cloud parameters.

    Uses 5σ of the expected final cloud width plus 3σ of COM jitter,
    rounded up to the nearest mm.  This is deterministic from the CLI
    args, so all files in the same experiment share the same bin edges.
    """
    sigma_xf  = math.sqrt(args.sigma_x_mean**2 + (args.sigma_vx_mean * T_DET)**2)
    sigma_com = math.sqrt(args.mu_x_std**2      + (args.mu_vx_std     * T_DET)**2)
    half_range = 5.0 * sigma_xf + 3.0 * sigma_com
    return math.ceil(half_range / 1e-3) * 1e-3   # round up to nearest mm


def _fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.0f}s'
    m, s = divmod(int(seconds), 60)
    return f'{m}m{s:02d}s'


def _init_h5(f, phi0, delta_phi, cloud_params, signal_params,
             n_atoms, half_range, res, z0_m):
    """
    Prepare an open HDF5 file for image output.

    Pre-allocates image datasets for all shots (known size) and writes
    per-shot metadata upfront.  Call _write_shot_image once per shot.
    """
    n_shots = len(phi0)
    (mu_x0, mu_y0, mu_vx0, mu_vy0,
     sigma_x, sigma_y, sigma_vx, sigma_vy) = cloud_params

    chunk = (1, res, res)
    f.create_dataset('images_s0', shape=(n_shots, res, res), dtype=np.uint16,
                     chunks=chunk, compression='gzip', compression_opts=4)
    f.create_dataset('images_s1', shape=(n_shots, res, res), dtype=np.uint16,
                     chunks=chunk, compression='gzip', compression_opts=4)

    # per-shot metadata
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

    f.attrs['image_half_range'] = float(half_range)
    f.attrs['image_res']        = int(res)
    f.attrs['z0_m']             = float(z0_m)
    f.attrs['n_atoms_launched'] = int(n_atoms)
    f.attrs['signal_amp']       = signal_params['amp']
    f.attrs['signal_freq']      = signal_params['freq']
    f.attrs['signal_phase']     = signal_params['phase']


def _write_shot_image(f, shot_i, img_s0, img_s1):
    """Write pre-computed state images for one shot into an open HDF5 file."""
    f['images_s0'][shot_i] = img_s0
    f['images_s1'][shot_i] = img_s1


def _iter_shots(surrogate, cloud_params, phi0, n_atoms, edges, rng):
    """
    Yield (shot_i, img_s0, img_s1) uint16 images for each shot.

    Uses the GPU image path (_image_edges): atoms are sampled, interpolated,
    and histogrammed entirely on the GPU; only the two (res, res) uint16
    images are transferred to CPU (~8 MB vs ~1.6 GB at 10^8 atoms).
    Falls back to CPU histogramming when CuPy is unavailable.
    Prints an in-place progress bar.
    """
    (mu_x0, mu_y0, mu_vx0, mu_vy0,
     sigma_x, sigma_y, sigma_vx, sigma_vy) = cloud_params
    n_shots = len(phi0)

    t_start    = time.perf_counter()
    shot_times = []

    for i in range(n_shots):
        t0 = time.perf_counter()

        img_s0, img_s1 = surrogate.generate_atoms(
            mu_x0=mu_x0[i],   mu_y0=mu_y0[i],
            mu_vx0=mu_vx0[i], mu_vy0=mu_vy0[i],
            sigma_x=sigma_x[i],   sigma_y=sigma_y[i],
            sigma_vx=sigma_vx[i], sigma_vy=sigma_vy[i],
            phi0=float(phi0[i]),
            natoms=n_atoms,
            rng=rng,
            _image_edges=edges,
        )

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

        yield i, img_s0, img_s1

    total = time.perf_counter() - t_start
    print(f'  [{"█"*bar_w}] {n_shots}/{n_shots}'
          f'  {np.mean(shot_times)*1e3:.0f} ms/shot'
          f'  total {_fmt_time(total)}         ')


def _simulate_run(run_idx, rng, args, surrogates, data_root,
                  delta_phi, signal_params, half_range):
    """Simulate one run and write its HDF5 image files. Returns wall time."""
    t_run = time.perf_counter()

    edges = np.linspace(-half_range, half_range, args.image_res + 1)

    # Laser phase — shared across both arms within each shot, re-drawn each run
    phi0 = _sample_phi0(rng, args.n_shots, args.phi0_mode)

    for z0_m, z0_label in Z0_VALUES.items():
        print(f'  {z0_label}')

        cloud_params = _sample_cloud_params(rng, args.n_shots, args)

        is_top   = (z0_m == 100)
        phi0_eff = phi0 + delta_phi if is_top else phi0
        dphi_z   = delta_phi        if is_top else np.zeros(args.n_shots)

        out = os.path.join(data_root, args.run_name,
                           f'run_{run_idx:03d}', z0_label, 'data_IMG.h5')
        os.makedirs(os.path.dirname(out), exist_ok=True)

        with h5py.File(out, 'w') as f:
            _init_h5(f, phi0_eff, dphi_z, cloud_params, signal_params,
                     args.n_atoms, half_range, args.image_res, float(z0_m))
            for shot_i, img_s0, img_s1 in _iter_shots(
                    surrogates[z0_m], cloud_params, phi0_eff,
                    args.n_atoms, edges, rng):
                _write_shot_image(f, shot_i, img_s0, img_s1)

        print(f'  → {out}  ({os.path.getsize(out)/1e6:.1f} MB)')

    return time.perf_counter() - t_run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_shots',   type=int,   default=100)
    p.add_argument('--n_atoms',   type=int,   default=10_000)
    p.add_argument('--n_runs',    type=int,   default=1)
    p.add_argument('--run_start', type=int,   default=0)
    p.add_argument('--run_name',  type=str,   default=None)
    p.add_argument('--seed',      type=int,   default=0)
    p.add_argument('--phi0_mode', choices=['random', 'linear', 'fixed'],
                   default='random')
    p.add_argument('--image_res', type=int,   default=2048,
                   help='Image resolution in pixels (default: 2048)')

    p.add_argument('--signal_amp',   type=float, default=0.0)
    p.add_argument('--signal_freq',  type=float, default=0.0)
    p.add_argument('--signal_phase', type=float, default=0.0)

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
            f'R{args.n_runs}'
            f'_N{args.n_shots}_A{args.n_atoms}'
            f'_muXStd{args.mu_x_std*1e6:.1f}um'
            f'_muVxStd{args.mu_vx_std*1e6:.1f}um'
            f'_sigX{args.sigma_x_mean*1e6:.0f}um'
            f'_sigVx{args.sigma_vx_mean*1e6:.0f}um'
            f'_sigXStd{args.sigma_x_std*1e6:.1f}um'
            f'_sigVxStd{args.sigma_vx_std*1e6:.1f}um'
            f'_phi0{args.phi0_mode}'
            f'{signal_tag}'
        )

    seed_seq    = np.random.SeedSequence(args.seed)
    n_total     = args.run_start + args.n_runs
    child_seeds = seed_seq.spawn(n_total)

    shot_indices = np.arange(args.n_shots)
    delta_phi = (args.signal_amp
                 * np.sin(2 * np.pi * args.signal_freq * shot_indices
                          + args.signal_phase))
    signal_params = {'amp':   args.signal_amp,
                     'freq':  args.signal_freq,
                     'phase': args.signal_phase}

    half_range = _compute_image_half_range(args)
    pixel_um   = 2 * half_range / args.image_res * 1e6

    print(f'Run name : {args.run_name}')
    print(f'  {args.n_runs} run(s) × {args.n_shots} shots × {args.n_atoms} atoms'
          f' = {args.n_runs * args.n_shots * args.n_atoms:,} atoms total per z0')
    print(f'  run indices : {args.run_start} – {args.run_start + args.n_runs - 1}')
    print(f'  phi0 mode   : {args.phi0_mode}')
    print(f'  image       : {args.image_res}×{args.image_res}'
          f'  window ±{half_range*1e3:.0f}mm  pixel={pixel_um:.1f}µm')
    if args.signal_amp != 0.0:
        print(f'  signal      : amp={args.signal_amp:.3f} rad'
              f'  freq={args.signal_freq:.4f} cyc/shot'
              f'  phase={args.signal_phase:.3f} rad')
    print()

    print('Loading surrogates …', end=' ', flush=True)
    t0 = time.perf_counter()
    surrogates = {z0: _make_surrogate(psmap_dir, z0) for z0 in Z0_VALUES}
    print(f'done  ({_fmt_time(time.perf_counter() - t0)})\n')

    t_global   = time.perf_counter()
    run_times  = []

    for local_i, run_idx in enumerate(range(args.run_start,
                                            args.run_start + args.n_runs)):
        eta_str = ''
        if run_times:
            avg_run = float(np.mean(run_times))
            eta_str = f'  ETA {_fmt_time(avg_run * (args.n_runs - local_i))}'

        print(f'── run {run_idx:03d}  ({local_i + 1}/{args.n_runs}){eta_str}')

        rng     = np.random.default_rng(child_seeds[run_idx])
        elapsed = _simulate_run(run_idx, rng, args, surrogates,
                                data_root, delta_phi, signal_params, half_range)
        run_times.append(elapsed)
        print(f'  run {run_idx:03d} finished in {_fmt_time(elapsed)}\n')

    total = time.perf_counter() - t_global
    print(f'All {args.n_runs} run(s) done.  Total wall time: {_fmt_time(total)}'
          f'  ({_fmt_time(total / args.n_runs)} / run)')


if __name__ == '__main__':
    main()
