"""Streaming result round trips and independent synthetic geometry checks."""
import json

import numpy as np
import pytest
from scipy.ndimage import label

from platesim.tile3d.grid import Grid3D
from platesim.tile3d.result import TileResult, TileResultWriter
from fixtures3d.make_fixtures import box_step, half_filled_trench, via_with_void


def test_round_trip(tmp_path):
    grid = Grid3D.from_um((8, 12, 16), 4)
    bare = np.arange(24, dtype=np.float32).reshape(grid.shape) * grid.dx
    writer = TileResultWriter(tmp_path, grid, bare, meta={'backend': 'synthetic'})
    for i, t in enumerate([0., .125, 2.]):
        fields = {'phi': bare + i} if i != 1 else None
        assert writer.add_frame(t, bare - i, fields, {'charge_C_cm2': i}) == i
        path = tmp_path / 'frames' / f'psi_{i:04d}.npz'
        assert path.exists()  # before close: no retained frame buffer
        with np.load(path) as saved:
            assert set(saved.files) == {'psi', 't_s'}
            assert saved['psi'].dtype == np.float32
            assert saved['psi'].flags.c_contiguous
            assert saved['t_s'].dtype == np.float64
            assert saved['t_s'].shape == ()
    writer.close({'summary_text': 'fixture'})
    result = TileResult(tmp_path)
    assert result.grid == grid
    np.testing.assert_array_equal(result.psi_si, bare)
    np.testing.assert_array_equal(result.times_s, [0, .125, 2])
    assert result.times_s.dtype == np.float64
    for i in range(3):
        np.testing.assert_array_equal(result.psi(i), bare - i)
    np.testing.assert_array_equal(result.psi(), bare - 2)
    np.testing.assert_array_equal(result.field('phi', 0), bare)
    np.testing.assert_array_equal(result.field('phi'), bare + 2)
    assert result.field('phi', 1) is None
    assert result.field('ux') is None
    assert result.series == {'t_s': [0, .125, 2], 'charge_C_cm2': [0, 1, 2]}
    assert result.manifest == {'backend': 'synthetic', 'summary_text': 'fixture'}
    assert json.loads((tmp_path / 'series.json').read_text()) == result.series
    with np.load(tmp_path / 'static.npz') as static:
        assert static.files == ['psi_si']
    with np.load(tmp_path / 'fields' / 'phi_0000.npz') as field:
        assert set(field.files) == {'phi', 't_s'}
    with pytest.raises(IndexError):
        result.psi(3)
    with pytest.raises(RuntimeError):
        writer.add_frame(3, bare)


def test_validation_and_sparse_series(tmp_path):
    grid = Grid3D.from_um((8, 8, 8), 4)
    psi = np.ones(grid.shape)
    writer = TileResultWriter(tmp_path, grid, psi)
    with pytest.raises(ValueError):
        writer.add_frame(0, psi[:, :, 0])
    with pytest.raises(ValueError):
        writer.add_frame(np.nan, psi)
    with pytest.raises(ValueError):
        writer.add_frame(0, psi, {'../escape': psi})
    writer.add_frame(0, psi, scalars={'a': 1})
    with pytest.raises(ValueError):
        writer.add_frame(0, psi)
    writer.add_frame(1, psi, scalars={'b': 2})
    writer.close()
    result = TileResult(tmp_path)
    assert result.series == {'t_s': [0, 1], 'a': [1, None], 'b': [None, 2]}
    with pytest.raises(FileExistsError):
        TileResultWriter(tmp_path, grid, psi)


def _crossing(coords, values, near):
    hits = np.flatnonzero(values[:-1] * values[1:] < 0)
    roots = [coords[i] - values[i] * (coords[i+1] - coords[i]) /
             (values[i+1] - values[i]) for i in hits]
    return min(roots, key=lambda v: abs(v - near))


@pytest.mark.parametrize('make', [box_step, half_filled_trench, via_with_void])
def test_fixture_answers(tmp_path, make):
    result, answers = make(tmp_path)
    grid, psi = result.grid, result.psi()
    assert max(grid.shape) <= 48
    assert grid.dx == pytest.approx(4e-4)
    assert len(result.times_s) >= 3
    assert np.all(np.diff(result.times_s) > 0)
    assert np.all(result.psi(0) == result.psi_si)
    x, y, z = [c * 1e4 for c in grid.centers()]
    p = psi.astype(float) * 1e4
    field = _crossing(z, p[0, 0, :], answers['field_um'])
    floor = _crossing(z, p[len(x)//2, len(y)//2, :], -80 + answers['floor_um'])
    k = np.argmin(abs(z + 24))
    wall = 48 - _crossing(x, p[:, len(y)//2, k], 48 - answers['wall_um'])
    assert field == pytest.approx(answers['field_um'], abs=4)
    assert floor + 80 == pytest.approx(answers['floor_um'], abs=4)
    assert wall == pytest.approx(answers['wall_um'], abs=4)
    volume = np.count_nonzero((result.psi_si > 0) & (psi < 0)) * (grid.dx * 1e4)**3
    assert volume == pytest.approx(answers['deposited_cu_volume_um3'], rel=.03)
    # A sign-only mask or distance in grid cells cannot pass these cm-distance checks.
    np.testing.assert_allclose(psi[0, 0, z > 16],
                               (z[z > 16] - answers['field_um']) * 1e-4, atol=1e-9)
    for i in range(1, len(result.times_s)):
        assert np.all((result.psi(i - 1) <= 0) <= (result.psi(i) <= 0))
    if make is via_with_void:
        labels, _ = label(psi > 0)
        bulk = np.unique(labels[:, :, -1])
        pocket = (labels > 0) & ~np.isin(labels, bulk)
        assert pocket.any()
        assert pocket.sum() * 4**3 == pytest.approx(answers['void_volume_um3'], rel=.03)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        centroid = [a[pocket].mean() for a in xyz]
        np.testing.assert_allclose(centroid, answers['void_centroid_um'], atol=4)
        # Exact Euclidean distance outside a capped cylinder, not max/min CSG.
        i, j, k = len(x)//2, len(y)//2, np.argmin(abs(z + 40))
        expected = min(36 - np.hypot(x[i], y[j]), z[k] + 64, -16 - z[k])
        assert p[i, j, k] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize('make', [box_step, half_filled_trench, via_with_void])
def test_fixture_euclidean_distance_in_solid(tmp_path, make):
    result, answers = make(tmp_path)
    grid = result.grid
    x, y, z = grid.centers()
    # Sample inside the solid below and outside the cavity's bottom edge.
    # Nearest front point is on the edge, so distance is a Euclidean hypotenuse.
    if make is via_with_void:
        i, j, k = np.argmin(abs(x - 46e-4)), len(y)//2, np.argmin(abs(z + 78e-4))
        radial = np.hypot(x[i], y[j]) - 36e-4
        expected = -np.hypot(radial, z[k] + 64e-4)
    else:
        wall = (48 - answers['wall_um']) * 1e-4
        floor = (-80 + answers['floor_um']) * 1e-4
        i, j, k = np.argmin(abs(x - 54e-4)), len(y)//2, np.argmin(abs(z + 90e-4))
        expected = -np.hypot(x[i] - wall, z[k] - floor)
    assert result.psi()[i, j, k] == pytest.approx(expected, abs=1e-9)


def test_storage_conversion_empty_and_numeric_order(tmp_path):
    grid = Grid3D.from_um((8, 8, 8), 4)
    source = np.asfortranarray(np.arange(8, dtype=np.float64).reshape(grid.shape))
    writer = TileResultWriter(tmp_path, grid, source)
    empty = TileResult(tmp_path)
    assert empty.times_s.size == 0
    with pytest.raises(IndexError):
        empty.psi()
    for i in range(12):
        writer.add_frame(float(i), source + i, fields={'c_cu': source / (i+1)})
    writer.close()
    result = TileResult(tmp_path)
    for i in range(12):
        np.testing.assert_array_equal(result.psi(i), (source + i).astype(np.float32))
        np.testing.assert_array_equal(result.field('c_cu', i), (source / (i+1)).astype(np.float32))
    with pytest.raises(ValueError):
        result.field('../bad')
