"""Fixed cameras and a subprocess boundary around native OpenGL drivers."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from ._geom import (COPPER, SILICON, camera_parameters, height_map, mesh,
                    vertical_slice)
from .plots import figure, save


def render_surface(result, index=-1, out_png=None, camera='iso', size=(1280, 960),
                   show_si=False, cut=None, *, z_exaggeration=1., renderer='auto'):
    """Render the final electrolyte interface, optionally overlaying bare Si.

    ``renderer`` is auto, pyvista, or matplotlib. Even explicit pyvista requests
    fall back safely; inspect renderer/fallback_reason to detect this. Camera
    coordinates are µm with z scaled; axis tick labels remain physical µm.
    """
    if camera not in ('iso', 'top', 'front'):
        raise ValueError('camera must be iso, top or front')
    if renderer not in ('auto', 'pyvista', 'matplotlib'):
        raise ValueError('renderer must be auto, pyvista or matplotlib')
    if not np.isfinite(z_exaggeration) or z_exaggeration <= 0:
        raise ValueError('z_exaggeration must be positive and finite')
    if len(size) != 2 or any(int(n) != n or n <= 0 for n in size):
        raise ValueError('size must contain two positive pixel counts')
    if out_png is None:
        raise ValueError('out_png is required')
    if cut is not None:
        if cut.get('axis') not in ('x', 'y') or not np.isfinite(cut['position_um']):
            raise ValueError('cut requires axis x/y and finite position_um')
        d = 'xy'.index(cut['axis'])
        low = result.grid.origin[d]*1e4
        if not low < cut['position_um'] < low+result.grid.shape[d]*result.grid.dx*1e4:
            raise ValueError('cut position must be inside the grid')
    path = Path(out_png)
    path.parent.mkdir(parents=True, exist_ok=True)
    psi = result.psi(index)
    cam = camera_parameters(result.grid, camera, z_exaggeration)
    title = f'Tile surface, t = {result.times_s[index]:g} s'
    options = dict(camera=cam, size=list(size), show_si=show_si, cut=cut, title=title)
    reason = None
    if renderer != 'matplotlib':
        try:
            info = _isolated_pyvista(psi, result.psi_si, result.grid, path, options)
            return dict(kind='png', path=str(path), title=title, camera=cam,
                        renderer='pyvista', **info)
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
    artifact = _matplotlib(psi, result.psi_si, result.grid, path, options)
    artifact.update(camera=cam, renderer='matplotlib')
    if reason:
        artifact['fallback_reason'] = reason
    return artifact


def _isolated_pyvista(psi, bare, grid, path, options):
    # Native driver access violations cannot be caught with Python try/except.
    with TemporaryDirectory(prefix='.vtk_', dir=path.resolve().parent) as temp:
        folder = Path(temp)
        np.savez(folder/'fields.npz', psi=psi, bare=bare)
        job = dict(grid=grid.to_dict(), out=str((folder/'render.png').resolve()), **options)
        (folder/'job.json').write_text(json.dumps(job), encoding='utf-8')
        command = ('from platesim.render3d.surface import _pv_worker; '
                   'import sys; _pv_worker(sys.argv[1])')
        proc = subprocess.run([sys.executable, '-c', command, str(folder)],
                              capture_output=True, text=True, timeout=20)
        if proc.returncode:
            raise RuntimeError(f'VTK subprocess exit {proc.returncode}: {proc.stderr[-1200:]}')
        info = json.loads((folder/'info.json').read_text(encoding='utf-8'))
        import imageio.v2 as iio
        image = iio.imread(folder/'render.png')
        if image.shape[:2] != tuple(options['size'][::-1]) or image[..., :3].std() < 2:
            raise RuntimeError('VTK returned a blank or incorrectly sized image')
        (folder/'render.png').replace(path)
        return info


def _pv_worker(folder):
    import pyvista as pv
    from platesim.tile3d.grid import Grid3D
    folder = Path(folder)
    job = json.loads((folder/'job.json').read_text(encoding='utf-8'))
    grid = Grid3D.from_dict(job['grid'])
    with np.load(folder/'fields.npz') as data:
        psi, bare = data['psi'], data['bare']
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=job['size'])
    try:
        plotter.set_background('white')
        # The final front alone: the Cu/Si intersection would also create a
        # buried underside sheet, coincident with bare Si (z-fighting).
        fields = [(psi, COPPER, 1.)]
        if job['show_si']:
            fields.insert(0, (bare, SILICON, .25))
        for field, color, opacity in fields:
            points, faces = mesh(field, grid)
            if not len(faces):
                continue
            poly = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)))
            if job['cut']:
                cut = job['cut']
                dim = 'xy'.index(cut['axis'])
                origin, normal = np.zeros(3), np.zeros(3)
                origin[dim], normal[dim] = cut['position_um'], 1
                poly = poly.clip(normal=normal, origin=origin, invert=True)
            if poly.n_points:
                plotter.add_mesh(poly, color=color, smooth_shading=False,
                                 ambient=.5, diffuse=.5, opacity=opacity)
        # Fill the exposed cut with material-coloured tiles (also exposes voids).
        if job['cut']:
            cut = job['cut']
            sec = vertical_slice(psi, bare, grid, cut['axis'], cut['position_um'])
            s, z = np.meshgrid(sec['s_um'], sec['z_um'], indexing='ij')
            fixed = np.full_like(s, cut['position_um'])
            xyz = (fixed, s, z) if cut['axis'] == 'x' else (s, fixed, z)
            plane = pv.StructuredGrid(*(a[..., None] for a in xyz))
            material = np.where(sec['psi_si'] <= 0, 1, np.where(sec['psi'] <= 0, 2, 0))
            plane['material'] = material.ravel(order='F')
            for code, color in ((1, SILICON), (2, COPPER)):
                if code == 1 and not job['show_si']:
                    continue
                part = plane.threshold([code-.1, code+.1], scalars='material',
                                       preference='point', all_scalars=True)
                if part.n_cells:
                    plotter.add_mesh(part, color=color, lighting=False)
        cam = job['camera']
        plotter.set_scale(zscale=cam['z_exaggeration'])
        plotter.camera_position = [cam['position'], cam['focal_point'], cam['view_up']]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = cam['parallel_scale']
        plotter.show_bounds(xtitle='x (um)', ytitle='y (um)', ztitle='z (um)',
                            bounds=tuple(v for o, n in zip(grid.origin, grid.shape)
                                         for v in (o*1e4, (o+n*grid.dx)*1e4)),
                            font_size=10, color='black')
        plotter.add_text(job['title'], font_size=10, color='black')
        plotter.show(screenshot=job['out'], auto_close=False)
        capabilities = plotter.render_window.ReportCapabilities()
        (folder/'info.json').write_text(json.dumps({'opengl': capabilities}), encoding='utf-8')
    finally:
        plotter.close()


def _matplotlib(psi, bare, grid, path, options):
    from matplotlib.colors import to_rgba
    fig = figure(options['size'])
    ax = fig.add_subplot(projection='3d')
    h = height_map(psi, grid)
    bare_h = height_map(bare, grid)['height_um']
    x, y = np.meshgrid(h['x_um'], h['y_um'], indexing='ij')
    z = h['height_um'].copy()
    colors = np.broadcast_to(to_rgba(COPPER), z.shape+(4,))
    if options['cut']:
        cut = options['cut']
        removed = (x if cut['axis'] == 'x' else y) > cut['position_um']
        z[removed] = np.nan
        bare_h[removed] = np.nan
        sec = vertical_slice(psi, bare, grid, cut['axis'], cut['position_um'])
        s, zz = np.meshgrid(sec['s_um'], sec['z_um'], indexing='ij')
        fixed = np.full_like(s, cut['position_um'])
        xx, yy = (fixed, s) if cut['axis'] == 'x' else (s, fixed)
        material = np.where(sec['psi_si'] <= 0, 1, np.where(sec['psi'] <= 0, 2, 0))
        palette = np.array([to_rgba('white', 0), to_rgba(SILICON), to_rgba(COPPER)])
        if not options['show_si']:
            palette[1, 3] = 0
        ax.plot_surface(xx, yy, zz, facecolors=palette[material], shade=False,
                        rstride=1, cstride=1)
    if options['show_si']:
        ax.plot_surface(x, y, bare_h, color=SILICON, alpha=.25,
                        shade=False, rstride=1, cstride=1)
    ax.plot_surface(x, y, z, facecolors=colors, shade=False, rstride=1, cstride=1)
    cam = options['camera']
    elevation, azimuth = {'iso': (32, -50), 'top': (90, -90), 'front': (0, -90)}[cam['preset']]
    ax.view_init(elev=elevation, azim=azimuth)
    low = np.array(grid.origin)*1e4
    high = low+np.array(grid.shape)*grid.dx*1e4
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
           xlabel='x (µm)', ylabel='y (µm)', zlabel='z (µm)', title=options['title'])
    ax.set_box_aspect((high-low)*[1, 1, cam['z_exaggeration']])
    return save(fig, path, options['title'])
