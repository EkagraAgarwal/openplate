"""Rendering checks on synthetic (not measured) geometries."""
import numpy as np
import imageio.v2 as iio
import pytest

from tests.fixtures3d.make_fixtures import box_step, half_filled_trench, via_with_void
from platesim.render3d import (render_surface, render_height_map,
                              render_cross_section, render_growth_animation)
from platesim.render3d._geom import height_map, vertical_slice

SIZE = (400, 300)


def check_image(path, copper=False):
    assert path.exists()
    rgb = iio.imread(path)[..., :3].astype(float)
    assert rgb.shape[:2] == SIZE[::-1]
    assert rgb.std() > 10
    if copper:
        assert np.count_nonzero((rgb[..., 0] > rgb[..., 1] * 1.25) &
                                (rgb[..., 1] > rgb[..., 2] * 1.2)) > 30


@pytest.fixture(params=[box_step, half_filled_trench, via_with_void])
def result(request, tmp_path):
    return request.param(tmp_path / 'result')[0]


def test_matplotlib_outputs(result, tmp_path):
    for camera in ('iso', 'top', 'front'):
        p = tmp_path / f'{camera}.png'
        # A front view of the uncut half-filled trench physically hides the Cu.
        cut = {'axis': 'y', 'position_um': 0} if camera == 'front' else None
        a = render_surface(result, out_png=p, camera=camera, size=SIZE,
                           renderer='matplotlib', cut=cut)
        b = render_surface(result, out_png=p, camera=camera, size=SIZE,
                           renderer='matplotlib', cut=cut)
        assert a['camera'] == b['camera']
        assert a['renderer'] == 'matplotlib'
        check_image(p, copper=True)
    p = tmp_path / 'height.png'
    render_height_map(**height_map(result.psi(), result.grid), out_png=p, size=SIZE)
    check_image(p, copper=True)
    p = tmp_path / 'section.png'
    section = vertical_slice(result.psi(), result.psi_si, result.grid, 'x', 0)
    render_cross_section(section, p, sem_overlay=[[0, -40], [20, -20]], size=SIZE)
    check_image(p, copper=True)
    p = tmp_path / 'growth.gif'
    a = render_growth_animation(result, p, renderer='matplotlib', size=SIZE)
    frames = iio.mimread(p)
    assert len(frames) == len(result.times_s) == a['frames']
    for frame in frames:
        assert frame.shape[:2] == SIZE[::-1] and frame.std() > 10
    rgb = frames[-1][..., :3].astype(float)
    assert ((rgb[..., 0] > 1.25*rgb[..., 1]) &
            (rgb[..., 1] > 1.2*rgb[..., 2])).sum() > 30


def test_pyvista(result, tmp_path):
    p = tmp_path / 'vtk.png'
    a = render_surface(result, out_png=p, size=SIZE, renderer='pyvista',
                       cut={'axis': 'y', 'position_um': 0}, z_exaggeration=1.5)
    if a['renderer'] != 'pyvista':
        pytest.skip('Off-screen OpenGL unavailable: ' + a['fallback_reason'])
    print('\nVTK capabilities: ' + '\n'.join(a['opengl'].splitlines()[:3]))
    check_image(p, copper=True)
    b = render_surface(result, out_png=p, size=SIZE, renderer='pyvista',
                       cut={'axis': 'y', 'position_um': 0}, z_exaggeration=1.5)
    assert a['camera'] == b['camera']


def test_context_failure_falls_back(tmp_path, monkeypatch):
    from platesim.render3d import surface
    result = box_step(tmp_path / 'result')[0]
    def broken(*args, **kwargs):
        raise RuntimeError('forced context failure')
    monkeypatch.setattr(surface, '_isolated_pyvista', broken)
    p = tmp_path / 'fallback.png'
    a = render_surface(result, out_png=p, size=SIZE,
                       cut={'axis': 'x', 'position_um': 0})
    assert a['renderer'] == 'matplotlib'
    assert 'forced context failure' in a['fallback_reason']
    check_image(p, copper=True)


def test_geometry_and_validation(tmp_path):
    result = half_filled_trench(tmp_path / 'result')[0]
    h = height_map(result.psi(), result.grid)['height_um']
    assert h[24, 24] == pytest.approx(-40)
    assert h[0, 0] == pytest.approx(0)
    with pytest.raises(ValueError):
        render_surface(result, out_png=tmp_path / 'bad.png', camera='bad')


@pytest.mark.parametrize('mode', ['exit', 'timeout'])
def test_native_failure_isolation(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    from platesim.render3d import surface
    result = box_step(tmp_path / 'result')[0]
    def fail(*args, **kwargs):
        if mode == 'timeout':
            raise surface.subprocess.TimeoutExpired('vtk', 20)
        return SimpleNamespace(returncode=-1073741819, stderr='driver failed')
    monkeypatch.setattr(surface.subprocess, 'run', fail)
    p = tmp_path / 'isolated.png'
    a = render_surface(result, out_png=p, size=SIZE)
    assert a['renderer'] == 'matplotlib' and a['fallback_reason']
    check_image(p, copper=True)
    assert not list(tmp_path.glob('.vtk_*'))


def test_contour_fill_matches_analytic_deposit_area(tmp_path, monkeypatch):
    from shapely.geometry import Polygon, GeometryCollection
    from platesim.render3d import plots
    result = box_step(tmp_path / 'box')[0]
    section = vertical_slice(result.psi(), result.psi_si, result.grid, 'x', 0)
    captured = {}
    def capture(fig, path, title):
        ax = fig.axes[0]
        # The first two collections are filled final solid and bare silicon.
        def region(collection):
            geometry = GeometryCollection()
            for path in collection.get_paths():
                for ring in path.to_polygons():
                    geometry = geometry.symmetric_difference(Polygon(ring))
            return geometry
        final, bare = map(region, ax.collections[:2])
        captured['area'] = final.difference(bare).area
        assert not any(type(c).__name__ == 'QuadMesh' for c in ax.collections)
        return {}
    monkeypatch.setattr(plots, 'save', capture)
    render_cross_section(section, tmp_path / 'unused.png')
    # 8 um field/floor growth over the sampled width plus two 80 um walls.
    width = np.ptp(section['s_um'])
    assert captured['area'] == pytest.approx(width*8 + 2*80*8, rel=.02)


def test_surface_defaults_to_final_interface(tmp_path, monkeypatch):
    from platesim.render3d import surface
    result = box_step(tmp_path / 'box')[0]
    def capture(psi, bare, grid, path, options):
        assert options['show_si'] is False
        np.testing.assert_array_equal(psi, result.psi())
        return {}
    monkeypatch.setattr(surface, '_isolated_pyvista', capture)
    assert render_surface(result, out_png=tmp_path / 'final.png')['renderer'] == 'pyvista'


def test_matplotlib_final_front_has_no_silicon_speckles(tmp_path, monkeypatch):
    from platesim.render3d import surface
    result = half_filled_trench(tmp_path / 'box')[0]
    def capture(fig, path, title):
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        colors = ax.collections[0].get_facecolors()
        from matplotlib.colors import to_rgba
        np.testing.assert_allclose(colors, np.tile(to_rgba(surface.COPPER), (len(colors), 1)))
        assert all('µm' in label for label in (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()))
        return {}
    monkeypatch.setattr(surface, 'save', capture)
    render_surface(result, out_png=tmp_path / 'final.png', renderer='matplotlib')


def test_bare_surface_is_optional_transparent_and_cut(tmp_path, monkeypatch):
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from platesim.render3d import surface
    result = box_step(tmp_path / 'box')[0]
    plot_surface = Axes3D.plot_surface
    seen = []
    def plot(ax, x, y, z, *args, **kwargs):
        if kwargs.get('alpha') == .25:
            assert np.isnan(z[x > 0]).all()
            assert np.isfinite(z[x < 0]).any()
            seen.append(True)
        return plot_surface(ax, x, y, z, *args, **kwargs)
    monkeypatch.setattr(Axes3D, 'plot_surface', plot)
    def capture(fig, path, title):
        ax = fig.axes[0]
        assert len(ax.collections) == 3  # cut face, optional bare, final front
        bare = ax.collections[1]
        assert bare.get_alpha() == .25
        assert seen == [True]
        return {}
    monkeypatch.setattr(surface, 'save', capture)
    render_surface(result, out_png=tmp_path / 'overlay.png', renderer='matplotlib',
                   show_si=True, cut={'axis': 'x', 'position_um': 0})
