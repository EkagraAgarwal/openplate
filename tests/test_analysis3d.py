"""Synthetic analysis acceptance tests; no measured data."""
import base64
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from skimage.measure import marching_cubes

from tests.fixtures3d.make_fixtures import box_step, half_filled_trench, via_with_void
from platesim.analysis3d import (height_map, thickness_map, cross_section,
                                feature_report, void_report, planarity,
                                export_surface, export_volume)


@pytest.mark.parametrize('factory', [box_step, half_filled_trench, via_with_void])
def test_fixture_answers(factory, tmp_path):
    result, known = factory(tmp_path / factory.__name__)
    report = feature_report(result)
    for key in ('field_um', 'floor_um', 'wall_um'):
        assert report[key] == pytest.approx(known[key], abs=4.)
    assert report['deposited_cu_volume_um3'] == pytest.approx(
        known['deposited_cu_volume_um3'], rel=.03)
    if known['field_um']:
        assert report['floor_over_field'] == pytest.approx(
            known['floor_um'] / known['field_um'], abs=.15)
    else:
        assert report['floor_over_field'] is None
    voids = void_report(result.psi(), result.grid)
    assert voids['count'] == (1 if factory is via_with_void else 0)
    if voids['count']:
        assert voids['volumes_um3'][0] == pytest.approx(known['void_volume_um3'], rel=.05)
        np.testing.assert_allclose(voids['centroids_um'][0], known['void_centroid_um'], atol=4)
        assert report['closure_time_s'] == 1.
    else:
        assert report['closure_time_s'] is None
    if factory is half_filled_trench:
        np.testing.assert_allclose(report['fill_fraction'], [0, .25, .5], atol=.03)
        assert report['wall_um'] == pytest.approx(0, abs=.05)
    print(factory.__name__, {k: report[k] for k in known if k in report},
          'known=', known, 'fill=', report['fill_fraction'], voids)


def test_maps_sections(tmp_path):
    result, _ = box_step(tmp_path / 'box')
    psi, bare, grid = result.psi(), result.psi_si, result.grid
    heights = height_map(psi, grid)['height_um']
    assert heights[0, 0] == pytest.approx(8, abs=.01)
    assert heights[24, 24] == pytest.approx(-72, abs=.01)
    assert thickness_map(psi, bare, grid)['thickness_um'][24, 24] == pytest.approx(8)
    section = cross_section(psi, bare, grid, [(-90, 0), (90, 0)], n_samples=181)
    outlines = np.concatenate(section['cu_outline'])
    assert outlines[:, 1].min() == pytest.approx(-72, abs=.1)
    assert outlines[:, 1].max() == pytest.approx(8, abs=.1)
    assert section['psi'].shape == (181, grid.shape[2])
    diagonal = cross_section(psi, bare, grid, [(-40, -40), (0, 0), (40, -40)])
    assert diagonal['s_um'][-1] == pytest.approx(80 * np.sqrt(2))
    assert planarity(np.ones((4, 5))) == dict(step_um=0., tir_um=0., std_um=0.)


def test_exports(tmp_path):
    result, _ = box_step(tmp_path / 'box')
    path = tmp_path / 'surface.stl'
    assert export_surface(result.psi(), result.grid, path)['kind'] == 'stl'
    count = struct.unpack('<I', path.read_bytes()[80:84])[0]
    expected = len(marching_cubes(result.psi(), level=0)[1])
    assert count == expected > 0
    assert path.stat().st_size == 84 + 50 * count
    artifact = export_volume(result, tmp_path / 'volume.vti')
    root = ET.parse(artifact['path']).getroot()
    assert root.find('ImageData').attrib['WholeExtent'] == '0 48 0 48 0 40'
    arrays = root.findall('.//DataArray')
    raw = base64.b64decode(arrays[0].text)
    assert struct.unpack('<I', raw[:4])[0] == np.prod(result.grid.shape) * 4
    assert len(artifact['frames']) == len(result.times_s)


@pytest.mark.parametrize('axis', [0, 1])
def test_periodic_voids(axis):
    from platesim.tile3d.grid import Grid3D
    grid = Grid3D.from_um((24, 24, 24), 4)
    psi = -np.ones(grid.shape)
    psi[:, :, -1] = 1
    psi = np.moveaxis(psi, axis, 0)
    psi[0, 2, 2] = psi[-1, 2, 2] = 1
    psi = np.moveaxis(psi, 0, axis)
    report = void_report(psi, grid)
    assert report['count'] == 1
    assert report['volumes_um3'][0] == pytest.approx(128)
    view = np.moveaxis(psi, axis, 0)
    view[-1, 2, 2:] = 1
    assert void_report(psi, grid)['count'] == 0


@pytest.mark.parametrize('periodic_x', [False, True])
def test_void_report_x_boundary_and_bulk_wrap(periodic_x):
    from platesim.tile3d.grid import Grid3D
    side = 'periodic' if periodic_x else 'symmetry'
    grid = Grid3D.from_um((24, 24, 24), 4,
                          bc={'x_lo': side, 'x_hi': side,
                              'y_lo': 'symmetry', 'y_hi': 'symmetry',
                              'z_lo': 'bottom', 'z_hi': 'bulk'})
    psi = -np.ones(grid.shape)
    psi[0, 2, 2] = psi[-1, 2, 2] = 1
    report = void_report(psi, grid)
    assert report['count'] == (1 if periodic_x else 2)
    np.testing.assert_allclose(report['volumes_um3'], [128] if periodic_x else [64, 64])
    assert report['centroids_um'].shape == (report['count'], 3)
    assert report['seam_flags'] == [True] * report['count']

    psi[-1, 2, 2:] = 1
    report = void_report(psi, grid)
    assert report['count'] == (0 if periodic_x else 1)
    np.testing.assert_allclose(report['volumes_um3'], [] if periodic_x else [64])
    assert report['centroids_um'].shape == (report['count'], 3)
    assert report['seam'] is (not periodic_x)


def test_missing_crossing_and_interpolation():
    from platesim.tile3d.grid import Grid3D
    grid = Grid3D.from_um((16, 16, 24), 4, origin_um=(0, 0, -12))
    _, _, z = grid.meshgrid()
    assert np.allclose(height_map(z - 1.25e-4, grid)['height_um'], 1.25)
    assert np.isnan(height_map(np.ones(grid.shape), grid)['height_um']).all()


def test_trilinear_sampling_and_region():
    from platesim.tile3d.grid import Grid3D
    grid = Grid3D.from_um((32, 32, 32), 4, origin_um=(-16, -16, -16),
                          bc={'x_lo': 'symmetry', 'x_hi': 'symmetry',
                              'y_lo': 'symmetry', 'y_hi': 'symmetry',
                              'z_lo': 'bottom', 'z_hi': 'bulk'})
    x, y, z = grid.meshgrid()
    psi = z + .2 * x - .3 * y
    section = cross_section(psi, psi, grid, [(-10, -8), (3, 5), (10, -2)], 101)
    arc = [0, np.hypot(13, 13), np.hypot(13, 13) + np.hypot(7, 7)]
    xs = np.interp(section['s_um'], arc, [-10, 3, 10])
    ys = np.interp(section['s_um'], arc, [-8, 5, -2])
    expected = (section['z_um'][None, :] + .2 * xs[:, None] - .3 * ys[:, None]) * 1e-4
    np.testing.assert_allclose(section['psi'], expected, atol=1e-15)
    cropped = height_map(psi, grid, region_um=(-4, 4, -8, 8))
    assert cropped['height_um'].shape == (2, 4)
    with pytest.raises(ValueError, match='outside'):
        cross_section(psi, psi, grid, [(0, 0), (30, 0)])
    with pytest.raises(ValueError, match='nonzero'):
        cross_section(psi, psi, grid, [(0, 0), (0, 0)])


def test_optional_export_errors(tmp_path, monkeypatch):
    import builtins
    from platesim.tile3d.grid import Grid3D
    grid = Grid3D.from_um((16, 16, 16), 4, origin_um=(0, 0, -8))
    _, _, psi = grid.meshgrid()
    real_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name in ('pyvista', 'trimesh'):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    for fmt, package in [('vtp', 'pyvista'), ('gltf', 'trimesh')]:
        with pytest.raises(ImportError, match=package):
            export_surface(psi, grid, tmp_path / f'surface.{fmt}', fmt=fmt)


def test_flat_result_has_no_feature(tmp_path):
    from platesim.tile3d.grid import Grid3D
    from platesim.tile3d.result import TileResult, TileResultWriter
    grid = Grid3D.from_um((16, 16, 32), 4, origin_um=(0, 0, -16))
    _, _, bare = grid.meshgrid()
    writer = TileResultWriter(tmp_path / 'flat', grid, bare)
    writer.add_frame(0., bare)
    writer.add_frame(1., bare - 3e-4)
    writer.close()
    report = feature_report(TileResult(tmp_path / 'flat'))
    assert report['field_um'] == pytest.approx(3)
    assert report['floor_um'] is None
    assert report['wall_um'] is None
    assert report['floor_over_field'] is None
    assert report['closure_time_s'] is None
    np.testing.assert_array_equal(report['fill_fraction'], [0, 0])


def test_gltf_when_available(tmp_path):
    import json
    pytest.importorskip('trimesh')
    from platesim.tile3d.grid import Grid3D
    grid = Grid3D.from_um((16, 16, 16), 4, origin_um=(0, 0, -8))
    _, _, psi = grid.meshgrid()
    artifact = export_surface(psi, grid, tmp_path / 'surface.gltf', fmt='gltf')
    assert artifact['kind'] == 'gltf'
    data = json.loads((tmp_path / 'surface.gltf').read_text())
    assert data['meshes']
    assert all(buffer['uri'].startswith('data:') for buffer in data['buffers'])


def test_surface_normals_point_into_electrolyte(tmp_path):
    from platesim.tile3d.grid import Grid3D
    grid = Grid3D.from_um((16, 16, 16), 4, origin_um=(0, 0, -8))
    _, _, psi = grid.meshgrid()
    path = tmp_path / 'plane.stl'
    export_surface(psi, grid, path)
    normal = struct.unpack('<fff', path.read_bytes()[84:96])
    assert normal[2] > .99
