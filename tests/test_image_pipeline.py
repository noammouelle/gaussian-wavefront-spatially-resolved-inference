"""
Tests for the image-based data pipeline:
  - ImageShotDataset (helpers.py)
  - convert_to_images.py conversion logic
  - generate_data._compute_image_half_range
"""

import math
import sys
import os

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-scripts'))

from helpers.helpers import ImageShotDataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_image_h5(path, n_shots=5, res=64, half_range=5e-3, seed=42):
    """Write a minimal data_IMG.h5 file with synthetic images."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(-half_range, half_range, res + 1)
    n_atoms = 1000

    with h5py.File(path, 'w') as f:
        imgs_s0 = np.zeros((n_shots, res, res), dtype=np.uint16)
        imgs_s1 = np.zeros((n_shots, res, res), dtype=np.uint16)
        for i in range(n_shots):
            x  = rng.normal(0, half_range / 5, n_atoms)
            y  = rng.normal(0, half_range / 5, n_atoms)
            st = rng.integers(0, 2, n_atoms)
            s0 = (st == 0)
            h0, _, _ = np.histogram2d(x[s0],  y[s0],  bins=edges)
            h1, _, _ = np.histogram2d(x[~s0], y[~s0], bins=edges)
            imgs_s0[i] = h0.astype(np.uint16)
            imgs_s1[i] = h1.astype(np.uint16)

        f.create_dataset('images_s0', data=imgs_s0,
                         chunks=(1, res, res), compression='gzip')
        f.create_dataset('images_s1', data=imgs_s1,
                         chunks=(1, res, res), compression='gzip')
        f.create_dataset('phi0',      data=rng.uniform(0, 2*np.pi, n_shots))
        f.create_dataset('delta_phi', data=np.zeros(n_shots))
        for k in ('mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0'):
            f.create_dataset(k, data=rng.normal(0, 1e-5, n_shots))
        for k in ('sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy'):
            f.create_dataset(k, data=rng.uniform(5e-5, 2e-4, n_shots))

        f.attrs['image_half_range'] = float(half_range)
        f.attrs['image_res']        = int(res)
        f.attrs['z0_m']             = 0.0
        f.attrs['n_atoms_launched'] = n_atoms
        f.attrs['signal_amp']       = 0.0
        f.attrs['signal_freq']      = 0.0
        f.attrs['signal_phase']     = 0.0

    return path


def make_prob_h5(path, n_shots=5, n_per_shot=200, seed=42):
    """Write a minimal data_PROB.h5 file with synthetic atom-list data."""
    rng = np.random.default_rng(seed)
    half_range = 5e-3

    with h5py.File(path, 'w') as f:
        shot_idx = np.repeat(np.arange(n_shots, dtype=np.int32), n_per_shot)
        x        = rng.normal(0, half_range / 5, n_shots * n_per_shot)
        y        = rng.normal(0, half_range / 5, n_shots * n_per_shot)
        z        = np.zeros(n_shots * n_per_shot)
        states   = rng.integers(0, 2, n_shots * n_per_shot, dtype=np.int8)

        f.create_dataset('positions',  data=np.column_stack([x, y, z]))
        f.create_dataset('states',     data=states)
        f.create_dataset('shot_index', data=shot_idx)
        f.create_dataset('phi0',       data=rng.uniform(0, 2*np.pi, n_shots))
        f.create_dataset('delta_phi',  data=np.zeros(n_shots))
        for k in ('mu_x0', 'mu_y0', 'mu_vx0', 'mu_vy0'):
            f.create_dataset(k, data=rng.normal(0, 1e-5, n_shots))
        for k in ('sigma_x', 'sigma_y', 'sigma_vx', 'sigma_vy'):
            f.create_dataset(k, data=rng.uniform(5e-5, 2e-4, n_shots))

        f.attrs['signal_amp']   = 0.0
        f.attrs['signal_freq']  = 0.0
        f.attrs['signal_phase'] = 0.0

    return path, half_range


# ── ImageShotDataset tests ────────────────────────────────────────────────────

class TestImageShotDataset:
    def test_attrs_loaded(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', n_shots=5, res=64)
        ds = ImageShotDataset(p)
        assert ds.n_shots == 5
        assert ds.res == 64
        assert ds.half_range == pytest.approx(5e-3)
        assert ds.z0_m == 0.0
        assert ds.n_atoms_launched == 1000

    def test_edges_and_pixel_centers(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', res=64, half_range=5e-3)
        ds = ImageShotDataset(p)
        assert len(ds.edges) == 65
        assert ds.edges[0]  == pytest.approx(-5e-3)
        assert ds.edges[-1] == pytest.approx( 5e-3)
        assert len(ds.pixel_centers) == 64
        assert ds.pixel_size == pytest.approx(10e-3 / 64)

    def test_single_shot_shape(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', n_shots=5, res=64)
        ds = ImageShotDataset(p)
        img = ds[0]
        assert img.shape == (2, 64, 64)
        assert img.dtype == np.uint16

    def test_slice_shape(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', n_shots=5, res=64)
        ds = ImageShotDataset(p)
        imgs = ds[1:4]
        assert imgs.shape == (3, 2, 64, 64)

    def test_state_counts_consistency(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', n_shots=5, res=64)
        ds = ImageShotDataset(p)
        counts = ds.state_counts()
        assert counts.shape == (5, 2)
        for i in range(5):
            img = ds[i]
            assert counts[0].iloc[i] == img[0].sum()
            assert counts[1].iloc[i] == img[1].sum()

    def test_meta_keys(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', n_shots=5, res=64)
        ds = ImageShotDataset(p)
        m = ds.meta(0)
        assert 'phi0' in m
        assert 'mu_x0' in m
        assert isinstance(m['phi0'], float)

    def test_iter_yields_n_shots(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', n_shots=4, res=32)
        ds = ImageShotDataset(p)
        items = list(ds)
        assert len(items) == 4
        assert all(img.shape == (2, 32, 32) for img in items)

    def test_detection_efficiency_range(self, tmp_path):
        p = make_image_h5(tmp_path / 'data_IMG.h5', n_shots=5, res=64)
        ds = ImageShotDataset(p)
        eff = ds.detection_efficiency()
        assert 0.0 < eff <= 1.0


# ── Conversion tests ──────────────────────────────────────────────────────────

class TestConvertToImages:
    def test_counts_preserved(self, tmp_path):
        """Image pixel sums must equal original in-window atom counts."""
        from convert_to_images import convert_file

        in_path, half_range = make_prob_h5(tmp_path / 'data_PROB.h5',
                                           n_shots=4, n_per_shot=300)
        out_path, ok, msg = convert_file(str(in_path), res=64,
                                         half_range=half_range)
        assert ok, f'Conversion failed: {msg}'
        assert os.path.exists(out_path)

        # Verify round-trip counts
        with h5py.File(in_path)  as fin, \
             h5py.File(out_path) as fout:
            shot_idx  = fin['shot_index'][:]
            x, y      = fin['positions'][:, 0], fin['positions'][:, 1]
            states    = fin['states'][:]
            n_shots   = int(fin['phi0'].shape[0])
            edges     = np.linspace(-half_range, half_range, 65)
            in_win    = ((x >= -half_range) & (x <= half_range) &
                         (y >= -half_range) & (y <= half_range))

            for i in range(n_shots):
                mask = (shot_idx == i) & in_win
                expected_n0 = int((states[mask] == 0).sum())
                expected_n1 = int((states[mask] == 1).sum())
                got_n0 = int(fout['images_s0'][i].sum())
                got_n1 = int(fout['images_s1'][i].sum())
                assert got_n0 == expected_n0, f'shot {i} s0 mismatch'
                assert got_n1 == expected_n1, f'shot {i} s1 mismatch'

    def test_metadata_preserved(self, tmp_path):
        """Metadata arrays (phi0, mu_x0, etc.) survive conversion."""
        from convert_to_images import convert_file

        in_path, half_range = make_prob_h5(tmp_path / 'data_PROB.h5')
        out_path, ok, _ = convert_file(str(in_path), res=64,
                                       half_range=half_range)
        assert ok

        with h5py.File(in_path)  as fin, \
             h5py.File(out_path) as fout:
            np.testing.assert_array_equal(fin['phi0'][:], fout['phi0'][:])
            np.testing.assert_array_equal(fin['mu_x0'][:], fout['mu_x0'][:])

    def test_image_attrs_written(self, tmp_path):
        """data_IMG.h5 must carry the image geometry attrs."""
        from convert_to_images import convert_file

        in_path, half_range = make_prob_h5(tmp_path / 'data_PROB.h5')
        out_path, ok, _ = convert_file(str(in_path), res=64,
                                       half_range=half_range)
        assert ok

        ds = ImageShotDataset(out_path)
        assert ds.half_range == pytest.approx(half_range)
        assert ds.res == 64

    def test_dry_run_no_write(self, tmp_path):
        """--dry-run must not write any file."""
        from convert_to_images import convert_file

        in_path, half_range = make_prob_h5(tmp_path / 'data_PROB.h5')
        out_path, ok, msg = convert_file(str(in_path), res=64,
                                         half_range=half_range, dry_run=True)
        assert ok
        assert msg == 'dry-run'
        assert not os.path.exists(out_path)


# ── _compute_image_half_range tests ──────────────────────────────────────────

class TestComputeHalfRange:
    def test_default_params_gives_7mm(self):
        """Default cloud params should yield 7mm half_range."""
        from generate_data import _compute_image_half_range

        class Args:
            sigma_x_mean  = 100e-6
            sigma_x_std   = 10e-6
            sigma_vx_mean = 3.09e-4
            sigma_vx_std  = 10e-6
            mu_x_std      = 10e-6
            mu_vx_std     = 10e-6

        hr = _compute_image_half_range(Args())
        assert hr == pytest.approx(7e-3, abs=0.5e-3)

    def test_rounded_to_mm(self):
        """Result must always be a multiple of 1mm."""
        from generate_data import _compute_image_half_range

        class Args:
            sigma_x_mean  = 80e-6
            sigma_x_std   = 5e-6
            sigma_vx_mean = 2.5e-4
            sigma_vx_std  = 5e-6
            mu_x_std      = 8e-6
            mu_vx_std     = 8e-6

        hr = _compute_image_half_range(Args())
        assert hr * 1e3 == pytest.approx(round(hr * 1e3), abs=1e-9)

    def test_larger_cloud_gives_larger_range(self):
        from generate_data import _compute_image_half_range

        class SmallArgs:
            sigma_x_mean = 50e-6;  sigma_x_std = 5e-6
            sigma_vx_mean = 1e-4;  sigma_vx_std = 5e-6
            mu_x_std = 5e-6;       mu_vx_std = 5e-6

        class LargeArgs:
            sigma_x_mean = 200e-6; sigma_x_std = 10e-6
            sigma_vx_mean = 5e-4;  sigma_vx_std = 10e-6
            mu_x_std = 20e-6;      mu_vx_std = 20e-6

        assert (_compute_image_half_range(LargeArgs()) >
                _compute_image_half_range(SmallArgs()))
