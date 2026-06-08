import sys, os
sys.path.insert(0, os.path.expanduser('~/local/aispy'))

import h5py
import numpy as np
import pandas as pd


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
            self._df = pd.DataFrame({
                'x':     f['positions'][:, 0],
                'y':     f['positions'][:, 1],
                'vx':    f['velocities'][:, 0],
                'vy':    f['velocities'][:, 1],
                'state': f['states'][:].astype(int),
                'prob':  f['probabilities'][:],
                'shot':  shot_idx,
            })
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
            return pd.DataFrame({
                'x':     f['positions'][s, 0],
                'y':     f['positions'][s, 1],
                'vx':    f['velocities'][s, 0],
                'vy':    f['velocities'][s, 1],
                'state': f['states'][s].astype(int),
                'prob':  f['probabilities'][s],
                'shot':  np.full(s.stop - s.start, idx, dtype=np.int32),
            })

    def __iter__(self):
        for i in range(self.n_shots):
            yield self[i]

    def meta(self, i):
        return {k: getattr(self, k)[i]
                for k in self._META_KEYS
                if hasattr(self, k)}

    def state_counts(self):
        """Return a DataFrame indexed by shot with columns 0 and 1."""
        counts = np.zeros((self.n_shots, 2), dtype=np.int64)

        with h5py.File(self.path) as f:
            states = f['states']
            for i in range(self.n_shots):
                s = self._shot_slice(i)
                shot_states = states[s]
                counts[i] = np.bincount(shot_states.astype(np.int8),
                                        minlength=2)[:2]

        return pd.DataFrame(counts, columns=[0, 1])

