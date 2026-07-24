import sys, os
sys.path.insert(0, os.path.expanduser('~/local/aispy'))

import h5py
import numpy as np
import pandas as pd

T_DET = 3.8   # detection time [s], must match generate_data.py


class ShotDataset:
    """
    Multi-shot dataset loaded from a data_PROB.h5 file.

    Usage
    -----
    ds = ShotDataset('../data/run/Z0/data_PROB.h5')

    ds.n_shots          # int
    ds.phi0             # (n_shots,) array of phase offsets
    ds.mu_x0            # (n_shots,) array of COM x0
    ds[i]               # DataFrame of atoms in shot i
    ds[i].state         # binary outcomes for shot i
    ds[3:7]             # DataFrame of shots 3–6 combined
    ds.meta(i)          # dict of all per-shot params for shot i
    """

    _META_KEYS = ['phi0', 'mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0',
                  'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy']

    def __init__(self, path):
        with h5py.File(path) as f:
            shot_idx      = f['shot_index'][:]
            atom_data = {
                'x':     f['positions'][:, 0],
                'y':     f['positions'][:, 1],
                'state': f['states'][:].astype(int),
                'prob':  f['probabilities'][:],
                'shot':  shot_idx,
            }
            if 'velocities' in f:
                atom_data['vx'] = f['velocities'][:, 0]
                atom_data['vy'] = f['velocities'][:, 1]
            self._df = pd.DataFrame(atom_data)
            for k in self._META_KEYS:
                setattr(self, k, f[k][:])

        self.n_shots = len(self.phi0)
        # precompute per-shot boolean masks for fast __getitem__
        self._masks = [shot_idx == i for i in range(self.n_shots)]

    def __len__(self):
        return self.n_shots

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            shots = range(*idx.indices(self.n_shots))
            mask  = np.isin(self._df['shot'].values, list(shots))
        else:
            mask = self._masks[idx]
        return self._df[mask].reset_index(drop=True)

    def __iter__(self):
        for i in range(self.n_shots):
            yield self[i]

    def meta(self, i):
        return {k: getattr(self, k)[i] for k in self._META_KEYS}


class LazyShotDataset:
    """
    Memory-light multi-shot dataset backed by a data_PROB.h5 file.

    This class is intended for large files where constructing one all-atom
    Pandas DataFrame would be too expensive. It reads metadata and the compact
    shot_index array once, then pulls atom columns from HDF5 only for requested
    shots.
    """

    _META_KEYS = ['phi0', 'delta_phi', 'mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0',
                  'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy']

    def __init__(self, path):
        self.path = path

        with h5py.File(path) as f:
            self.n_atoms_total = int(f['states'].shape[0])
            shot_idx = f['shot_index'][:]

            for k in self._META_KEYS:
                if k in f:
                    setattr(self, k, f[k][:])

        self.n_shots = len(self.phi0)
        self._starts, self._stops = self._bounds_from_sorted_index(shot_idx)

    @staticmethod
    def _bounds_from_sorted_index(shot_idx):
        if len(shot_idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        changed = np.flatnonzero(np.diff(shot_idx) != 0) + 1
        starts = np.r_[0, changed].astype(np.int64)
        stops = np.r_[changed, len(shot_idx)].astype(np.int64)
        return starts, stops

    def __len__(self):
        return self.n_shots

    def _shot_slice(self, idx):
        if idx < 0:
            idx += self.n_shots
        if idx < 0 or idx >= self.n_shots:
            raise IndexError(idx)
        return slice(int(self._starts[idx]), int(self._stops[idx]))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            shots = range(*idx.indices(self.n_shots))
            return pd.concat([self[i] for i in shots], ignore_index=True)

        s = self._shot_slice(idx)
        with h5py.File(self.path) as f:
            atom_data = {
                'x':     f['positions'][s, 0],
                'y':     f['positions'][s, 1],
                'state': f['states'][s].astype(int),
                'prob':  f['probabilities'][s],
                'shot':  np.full(s.stop - s.start, idx, dtype=np.int32),
            }
            if 'velocities' in f:
                atom_data['vx'] = f['velocities'][s, 0]
                atom_data['vy'] = f['velocities'][s, 1]
            return pd.DataFrame(atom_data)

    def __iter__(self):
        for i in range(self.n_shots):
            yield self[i]

    def meta(self, i):
        return {k: getattr(self, k)[i]
                for k in self._META_KEYS
                if hasattr(self, k)}

    def state_counts(self):
        """Return a DataFrame indexed by shot with columns 0 and 1."""
        # Bulk-read all states in one HDF5 call, then split by shot.
        # ~10x faster than the per-shot slice loop for large files with
        # compressed chunks, because gzip decompression happens in one pass.
        with h5py.File(self.path) as f:
            all_states = f['states'][:]

        n_per_shot = (self._stops - self._starts).astype(np.int64)
        shot_idx   = np.repeat(np.arange(self.n_shots, dtype=np.int32), n_per_shot)
        n0 = np.bincount(shot_idx[all_states == 0], minlength=self.n_shots)
        n1 = np.bincount(shot_idx[all_states == 1], minlength=self.n_shots)
        return pd.DataFrame({0: n0, 1: n1})


class ImageShotDataset:
    """
    Shot dataset backed by a data_IMG.h5 file (2D histogram format).

    Each shot is stored as two (res, res) uint16 images: one for each
    output state.  Images are pixel-aligned across all shots and files
    of the same experiment (shared bin edges from generation parameters).

    Usage
    -----
    ds = ImageShotDataset('../data/run/Z0/data_IMG.h5')

    ds.n_shots          # int
    ds.half_range       # float [m] — half the image window
    ds.res              # int — pixels per side
    ds.edges            # (res+1,) bin edges [m]
    ds.pixel_centers    # (res,) pixel centre positions [m]
    ds.phi0             # (n_shots,) phase offsets [rad]
    ds[i]               # (2, res, res) uint16 — images_s0, images_s1
    ds[3:7]             # (4, 2, res, res) uint16
    ds.state_counts()   # DataFrame with columns {0, 1}, n_shots rows
    ds.meta(i)          # dict of per-shot cloud params
    """

    _META_KEYS = ['phi0', 'delta_phi', 'mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0',
                  'sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy']

    def __init__(self, path):
        self.path = path
        with h5py.File(path) as f:
            self.half_range     = float(f.attrs['image_half_range'])
            self.res            = int(f.attrs['image_res'])
            self.z0_m           = float(f.attrs['z0_m'])
            self.n_atoms_launched = int(f.attrs['n_atoms_launched'])
            for k in self._META_KEYS:
                if k in f:
                    setattr(self, k, f[k][:])

        self.n_shots     = len(self.phi0)
        self.edges       = np.linspace(-self.half_range, self.half_range,
                                       self.res + 1)
        self.pixel_size  = 2 * self.half_range / self.res
        self.pixel_centers = 0.5 * (self.edges[:-1] + self.edges[1:])

    def __len__(self):
        return self.n_shots

    def __getitem__(self, idx):
        """Return (2, res, res) uint16 or (n, 2, res, res) for a slice."""
        with h5py.File(self.path) as f:
            if isinstance(idx, slice):
                s0 = f['images_s0'][idx]
                s1 = f['images_s1'][idx]
                return np.stack([s0, s1], axis=1)
            s0 = f['images_s0'][idx]
            s1 = f['images_s1'][idx]
        return np.stack([s0, s1], axis=0)

    def __iter__(self):
        for i in range(self.n_shots):
            yield self[i]

    def meta(self, i):
        return {k: getattr(self, k)[i]
                for k in self._META_KEYS if hasattr(self, k)}

    def state_counts(self):
        """
        Total detected atoms per shot per state.

        Returns a DataFrame with columns {0, 1} and n_shots rows.
        Equivalent to summing all pixel counts, but reads the full image
        arrays in one HDF5 call for efficiency.
        """
        with h5py.File(self.path) as f:
            n0 = f['images_s0'][:].sum(axis=(1, 2))
            n1 = f['images_s1'][:].sum(axis=(1, 2))
        return pd.DataFrame({0: n0.astype(np.int64),
                              1: n1.astype(np.int64)})

    def detection_efficiency(self):
        """Mean fraction of launched atoms that were detected."""
        counts = self.state_counts()
        total  = (counts[0] + counts[1]).mean()
        return float(total) / self.n_atoms_launched

