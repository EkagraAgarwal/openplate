"""Contract tests for the cell-centred grid (cm internally)."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from platesim.tile3d.grid import Grid3D


def test_centers_mesh_and_round_trip():
    grid = Grid3D.from_um((8, 12, 16), 4, origin_um=(-4, 0, -8))
    assert grid.shape == (2, 3, 4)
    assert grid.dx == pytest.approx(4e-4)
    assert grid.periodic == (True, True, False)
    assert grid.bc['z_lo'] == 'bottom'
    assert grid.bc['z_hi'] == 'bulk'
    for got, expected in zip(grid.centers(), ([-2, 2], [2, 6, 10], [-6, -2, 2, 6])):
        np.testing.assert_allclose(got, np.array(expected) * 1e-4)
    x, y, z = grid.meshgrid()
    assert x.shape == y.shape == z.shape == grid.shape
    np.testing.assert_array_equal(x[:, 0, 0], grid.centers()[0])
    np.testing.assert_array_equal(y[0, :, 0], grid.centers()[1])
    np.testing.assert_array_equal(z[0, 0, :], grid.centers()[2])
    assert grid.cell_volume == pytest.approx((4e-4) ** 3)
    assert Grid3D.from_dict(json.loads(json.dumps(grid.to_dict()))) == grid
    with pytest.raises(FrozenInstanceError):
        grid.dx = 1
    encoded = grid.to_dict()
    encoded['bc']['x_lo'] = 'halo'
    assert grid.bc['x_lo'] == 'periodic'


@pytest.mark.parametrize('update', [
    {'bad': 'halo'}, {'x_lo': 'wrong'}, {'x_lo': 'bottom'},
    {'y_hi': 'bulk'}, {'x_lo': 'halo'}, {'z_lo': 'periodic'},
])
def test_invalid_boundaries(update):
    bc = Grid3D.from_um((8, 8, 8), 4).to_dict()['bc']
    bc.update(update)
    with pytest.raises(ValueError, match='bc|periodic'):
        Grid3D((2, 2, 2), 4e-4, (0, 0, 0), bc)


def test_missing_boundary_and_valid_alternatives():
    with pytest.raises(ValueError, match='bc'):
        Grid3D((2, 2, 2), 1, (0, 0, 0), {})
    bc = {f'{a}_{side}': 'symmetry' for a in 'xyz' for side in ('lo', 'hi')}
    bc['x_lo'] = 'halo'
    assert Grid3D((2, 2, 2), 1, (0, 0, 0), bc).periodic == (False,) * 3
    bc.update(z_lo='periodic', z_hi='periodic')
    assert Grid3D((2, 2, 2), 1, (0, 0, 0), bc).periodic[2]


@pytest.mark.parametrize('shape,dx,origin', [
    ((0, 2, 2), 1, (0, 0, 0)), ((2.5, 2, 2), 1, (0, 0, 0)),
    ((2, 2), 1, (0, 0, 0)), ((2, 2, 2), 0, (0, 0, 0)),
    ((2, 2, 2), np.nan, (0, 0, 0)), ((2, 2, 2), 1, (0, np.inf, 0)),
])
def test_invalid_geometry(shape, dx, origin):
    with pytest.raises(ValueError):
        Grid3D(shape, dx, origin)


def test_from_um_rejects_nonintegral_cells():
    with pytest.raises(ValueError):
        Grid3D.from_um((9, 8, 8), 4)
