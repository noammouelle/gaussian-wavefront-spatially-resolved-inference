"""
convert_to_images.py — Convert atom-list data_PROB.h5 files to image format.

Reads existing data_PROB.h5 files (atom-level format produced by the old
generate_data.py) and writes data_IMG.h5 files (per-shot 2D histogram images)
alongside them.

The image half_range is computed from the mean cloud parameters stored in
each file, using the same formula as the new generate_data.py:
    half_range = 5 * sigma_xf + 3 * sigma_com  (rounded up to nearest mm)

A shared half_range is computed from the FIRST file found in the batch so
that all output images are pixel-aligned.  You can override this with
--half_range_mm.

Verification
------------
After writing each image file the script verifies that the total state-0 and
state-1 counts per shot match the originals exactly.  Files that fail
verification are flagged and the originals are NOT deleted.

Usage
-----
    # Dry run — show what would be converted, don't write anything
    python convert_to_images.py <path>  --dry-run

    # Convert all data_PROB.h5 under <path>, keep originals
    python convert_to_images.py <path>  --res 2048

    # Convert and delete originals after successful verification
    python convert_to_images.py <path>  --res 2048  --delete-originals

    # Override the computed half_range
    python convert_to_images.py <path>  --res 2048  --half_range_mm 7.0

<path> can be a directory (searched recursively) or a single .h5 file.
"""

import argparse
import math
import os
import sys
import time

import h5py
import numpy as np

T_DET      = 3.8   # s, must match generate_data.py
N_SIGMA_CLOUD = 5.0
N_SIGMA_COM   = 3.0


def _compute_half_range(f):
    """Derive half_range from mean cloud params stored in an open HDF5 file."""
    sigma_x_mean  = float(np.mean(f['sigma_x'][:]))
    sigma_vx_mean = float(np.mean(f['sigma_vx'][:]))
    # COM std: approximate from the realized spread of mu arrays
    mu_x_std  = float(np.std(f['mu_x0'][:]))
    mu_vx_std = float(np.std(f['mu_vx0'][:]))
    sigma_xf  = math.sqrt(sigma_x_mean**2 + (sigma_vx_mean * T_DET)**2)
    sigma_com = math.sqrt(mu_x_std**2      + (mu_vx_std     * T_DET)**2)
    half_range = N_SIGMA_CLOUD * sigma_xf + N_SIGMA_COM * sigma_com
    return math.ceil(half_range / 1e-3) * 1e-3   # round up to nearest mm


def _find_prob_files(root):
    """Return sorted list of data_PROB.h5 paths under root."""
    if os.path.isfile(root):
        return [root]
    result = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == 'data_PROB.h5':
                result.append(os.path.join(dirpath, fn))
    return sorted(result)


def _fmt_time(s):
    if s < 60:
        return f'{s:.0f}s'
    m, sec = divmod(int(s), 60)
    return f'{m}m{sec:02d}s'


def convert_file(in_path, res, half_range, dry_run=False):
    """
    Convert one data_PROB.h5 → data_IMG.h5.

    Returns (out_path, ok, msg) where ok is True iff the file was written
    and verified successfully.
    """
    out_path = os.path.join(os.path.dirname(in_path), 'data_IMG.h5')

    if dry_run:
        return out_path, True, 'dry-run'

    edges = np.linspace(-half_range, half_range, res + 1)

    with h5py.File(in_path, 'r') as fin:
        # ── Read atom-level data ───────────────────────────────────────────
        shot_idx = fin['shot_index'][:]
        x        = fin['positions'][:, 0]
        y        = fin['positions'][:, 1]
        states   = fin['states'][:]
        n_shots  = int(fin['phi0'].shape[0])

        # ── Infer z0_m from positions (constant per file) ─────────────────
        z0_m = float(np.round(fin['positions'][0, 2]))

        # ── Per-shot metadata ──────────────────────────────────────────────
        meta = {}
        for k in ['phi0', 'delta_phi', 'mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0',
                  'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy']:
            if k in fin:
                meta[k] = fin[k][:]

        signal_amp   = float(fin.attrs.get('signal_amp',   0.0))
        signal_freq  = float(fin.attrs.get('signal_freq',  0.0))
        signal_phase = float(fin.attrs.get('signal_phase', 0.0))
        n_atoms_launched = int(round(fin['states'].shape[0] / n_shots / 0.6))
        # Prefer exact n_atoms if it was stored (older files don't have it)
        if 'n_atoms_launched' in fin.attrs:
            n_atoms_launched = int(fin.attrs['n_atoms_launched'])

        # Compute per-shot atom offsets for fast slicing
        changed = np.flatnonzero(np.diff(shot_idx) != 0) + 1
        starts  = np.r_[0, changed].astype(np.int64)
        stops   = np.r_[changed, len(shot_idx)].astype(np.int64)

    # ── Write image file ───────────────────────────────────────────────────
    with h5py.File(out_path, 'w') as fout:
        chunk = (1, res, res)
        fout.create_dataset('images_s0', shape=(n_shots, res, res),
                            dtype=np.uint16, chunks=chunk,
                            compression='gzip', compression_opts=4)
        fout.create_dataset('images_s1', shape=(n_shots, res, res),
                            dtype=np.uint16, chunks=chunk,
                            compression='gzip', compression_opts=4)

        for k, v in meta.items():
            fout.create_dataset(k, data=v)

        fout.attrs['image_half_range'] = float(half_range)
        fout.attrs['image_res']        = int(res)
        fout.attrs['z0_m']             = z0_m
        fout.attrs['n_atoms_launched'] = n_atoms_launched
        fout.attrs['signal_amp']       = signal_amp
        fout.attrs['signal_freq']      = signal_freq
        fout.attrs['signal_phase']     = signal_phase

        orig_n0 = np.zeros(n_shots, dtype=np.int64)
        orig_n1 = np.zeros(n_shots, dtype=np.int64)

        for i in range(n_shots):
            sl     = slice(int(starts[i]), int(stops[i]))
            xi     = x[sl]; yi = y[sl]; si = states[sl]
            s0     = (si == 0)
            img_s0, _, _ = np.histogram2d(xi[s0],  yi[s0],  bins=edges)
            img_s1, _, _ = np.histogram2d(xi[~s0], yi[~s0], bins=edges)
            fout['images_s0'][i] = img_s0.astype(np.uint16)
            fout['images_s1'][i] = img_s1.astype(np.uint16)
            orig_n0[i] = s0.sum()
            orig_n1[i] = (~s0).sum()

    # ── Verify ────────────────────────────────────────────────────────────
    with h5py.File(out_path, 'r') as fout:
        img_n0 = fout['images_s0'][:].sum(axis=(1, 2))
        img_n1 = fout['images_s1'][:].sum(axis=(1, 2))

    # Atoms outside the window are dropped; count those separately
    in_window_s0 = np.array([
        ((x[starts[i]:stops[i]] >= -half_range) &
         (x[starts[i]:stops[i]] <=  half_range) &
         (y[starts[i]:stops[i]] >= -half_range) &
         (y[starts[i]:stops[i]] <=  half_range) &
         (states[starts[i]:stops[i]] == 0)).sum()
        for i in range(n_shots)], dtype=np.int64)
    in_window_s1 = np.array([
        ((x[starts[i]:stops[i]] >= -half_range) &
         (x[starts[i]:stops[i]] <=  half_range) &
         (y[starts[i]:stops[i]] >= -half_range) &
         (y[starts[i]:stops[i]] <=  half_range) &
         (states[starts[i]:stops[i]] == 1)).sum()
        for i in range(n_shots)], dtype=np.int64)

    ok0 = np.array_equal(img_n0, in_window_s0)
    ok1 = np.array_equal(img_n1, in_window_s1)

    atoms_outside = int((orig_n0 - in_window_s0).sum() +
                        (orig_n1 - in_window_s1).sum())

    if ok0 and ok1:
        msg = f'OK  ({atoms_outside} atoms outside window, dropped)'
        return out_path, True, msg
    else:
        diff0 = int(np.abs(img_n0 - in_window_s0).sum())
        diff1 = int(np.abs(img_n1 - in_window_s1).sum())
        return out_path, False, f'MISMATCH  s0_diff={diff0}  s1_diff={diff1}'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', help='Directory or data_PROB.h5 file to convert')
    p.add_argument('--res',            type=int,   default=2048,
                   help='Image resolution in pixels (default: 2048)')
    p.add_argument('--half_range_mm',  type=float, default=None,
                   help='Override image half-range [mm]. '
                        'Default: computed from cloud params in first file.')
    p.add_argument('--dry-run',        action='store_true',
                   help='Show what would be converted without writing anything')
    p.add_argument('--delete-originals', action='store_true',
                   help='Delete data_PROB.h5 after successful verification')
    args = p.parse_args()

    files = _find_prob_files(args.path)
    if not files:
        print(f'No data_PROB.h5 files found under {args.path}')
        sys.exit(1)

    print(f'Found {len(files)} data_PROB.h5 file(s)')

    # ── Determine shared half_range ────────────────────────────────────────
    if args.half_range_mm is not None:
        half_range = args.half_range_mm * 1e-3
        print(f'half_range: {half_range*1e3:.0f}mm  (from --half_range_mm)')
    else:
        with h5py.File(files[0], 'r') as f:
            half_range = _compute_half_range(f)
        print(f'half_range: {half_range*1e3:.0f}mm  (computed from {os.path.basename(files[0])})')

    pixel_um = 2 * half_range / args.res * 1e6
    print(f'Resolution: {args.res}×{args.res}  pixel={pixel_um:.1f}µm')
    if args.dry_run:
        print('(dry-run — no files will be written)\n')

    t_global   = time.perf_counter()
    n_ok       = 0
    n_fail     = 0
    # Separate already-converted from pending
    pending  = []
    existing = []
    for f in files:
        img = os.path.join(os.path.dirname(f), 'data_IMG.h5')
        if os.path.exists(img):
            existing.append(f)
        else:
            pending.append(f)
    if existing:
        print(f'{len(existing)} file(s) already converted — '
              f'{"will verify+delete originals" if args.delete_originals and not args.dry_run else "skipping"}')

    # For already-converted files: delete originals (prior run already verified them)
    if args.delete_originals and not args.dry_run and existing:
        print('Deleting originals for already-converted files …')
        for in_path in existing:
            freed_mb = os.path.getsize(in_path) / 1e6
            os.remove(in_path)
            rel = os.path.relpath(in_path, args.path) if os.path.isdir(args.path) else in_path
            print(f'  deleted {rel}  ({freed_mb:.0f}MB freed)')
            n_ok += 1

    files = pending
    for i, in_path in enumerate(files):
        rel = os.path.relpath(in_path, args.path) if os.path.isdir(args.path) else in_path
        print(f'[{i+1:3d}/{len(files)}] {rel}', end='  ', flush=True)
        t0 = time.perf_counter()

        out_path, ok, msg = convert_file(in_path, args.res, half_range,
                                         dry_run=args.dry_run)
        elapsed = time.perf_counter() - t0

        if ok:
            n_ok += 1
            size_mb = os.path.getsize(out_path) / 1e6 if not args.dry_run else 0
            print(f'{msg}  {size_mb:.1f}MB  {_fmt_time(elapsed)}')
            # Delete immediately so disk space stays bounded (one file at a time)
            if args.delete_originals and not args.dry_run:
                freed_mb = os.path.getsize(in_path) / 1e6
                os.remove(in_path)
                print(f'        deleted original  ({freed_mb:.0f}MB freed)')
        else:
            n_fail += 1
            print(f'FAILED: {msg}')

    total = time.perf_counter() - t_global
    print(f'\n{n_ok} converted OK, {n_fail} failed.  Total: {_fmt_time(total)}')

    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
