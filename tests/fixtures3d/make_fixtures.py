"""Analytic synthetic geometry, not measured data.

All geometry calculations use cm. Dimensions below are synthetic choices [J].
Rectangular growth is uniform on the planar faces with sharp junctions (not
rounded isotropic-offset corners). Answers refer to the final saved frame.
Thickness keys field_um/floor_um/wall_um are normal displacements away from
edges; deposited_cu_volume_um3 excludes pre-existing silicon/seed.
"""
import numpy as np

from platesim.tile3d.grid import Grid3D
from platesim.tile3d.result import TileResult, TileResultWriter


def _grid():
    # [J] 192 x 192 x 160 um window, 80 um deep features, 4 um cells.
    return Grid3D.from_um((192, 192, 160), 4, origin_um=(-96, -96, -96))


def _rect_distance(grid, half_width, floor, field):
    """Exact signed distance to the union of six rectangular surface patches."""
    x, y, z = grid.meshgrid()
    ax, ay = np.abs(x), np.abs(y)
    qx, qy = np.maximum(ax - half_width, 0), np.maximum(ay - half_width, 0)
    dz = np.maximum(np.maximum(floor - z, z - field), 0)
    bottom = np.sqrt(qx*qx + qy*qy + (z - floor)**2)
    wall_x = np.sqrt((ax - half_width)**2 + qy*qy + dz*dz)
    wall_y = np.sqrt((ay - half_width)**2 + qx*qx + dz*dz)
    lip = np.maximum(np.minimum(half_width - ax, half_width - ay), 0)
    top = np.hypot(lip, z - field)
    distance = np.minimum.reduce([bottom, wall_x, wall_y, top])
    liquid = (z > field) | ((ax < half_width) & (ay < half_width) & (z > floor))
    return np.where(liquid, distance, -distance)


def _open_via(grid, radius, floor):
    """Exact distance to disk floor, finite cylindrical wall and exterior field."""
    x, y, z = grid.meshgrid()
    rho = np.hypot(x, y)
    bottom = np.hypot(np.maximum(rho - radius, 0), z - floor)
    wall = np.hypot(rho - radius, np.maximum(np.maximum(floor-z, z), 0))
    top = np.hypot(np.maximum(radius - rho, 0), z)
    distance = np.minimum.reduce([bottom, wall, top])
    return np.where((z > 0) | ((rho < radius) & (z > floor)), distance, -distance)


def _closed_via(grid, radius, floor, ceiling, field):
    """Exact distance to two disjoint electrolyte regions: bulk and capped cylinder."""
    x, y, z = grid.meshgrid()
    radial = np.hypot(x, y) - radius
    axial = np.abs(z - (floor + ceiling) / 2) - (ceiling - floor) / 2
    cylinder = np.hypot(np.maximum(radial, 0), np.maximum(axial, 0))
    cylinder += np.minimum(np.maximum(radial, axial), 0)
    return np.maximum(z - field, -cylinder)


def _save(target, grid, bare, frames, answers, name):
    writer = TileResultWriter(target, grid, bare, meta={'backend': 'analytic', 'fixture': name})
    for t_s, psi in enumerate(frames):
        volume = np.count_nonzero((bare > 0) & (psi < 0)) * grid.cell_volume
        writer.add_frame(float(t_s), psi, scalars={'deposited_cu_volume_cm3': volume})
    writer.close({'summary_text': 'Synthetic geometry; no measured data.', 'answers': answers})
    return TileResult(target), answers


def box_step(target):
    """96 x 96 x 80 um rectangular trench; 8 um uniform planar-face growth."""
    grid = _grid()
    bare = _rect_distance(grid, 48e-4, -80e-4, 0.)
    t = 8e-4
    area, opening = (192e-4)**2, (96e-4)**2
    volume = area*t + 80e-4 * (opening - (96e-4-2*t)**2)
    answers = dict(field_um=8., floor_um=8., wall_um=8.,
                   deposited_cu_volume_um3=volume * 1e12)
    frames = (_rect_distance(grid, 48e-4-s, -80e-4+s, s) for s in (0., t/2, t))
    return _save(target, grid, bare, frames, answers, 'box_step')


def half_filled_trench(target):
    """Same trench, bottom-up fill to half depth; no field or wall plating."""
    grid = _grid()
    bare = _rect_distance(grid, 48e-4, -80e-4, 0.)
    answers = dict(field_um=0., floor_um=40., wall_um=0.,
                   deposited_cu_volume_um3=96.**2 * 40.)
    frames = (_rect_distance(grid, 48e-4, floor, 0.) for floor in (-80e-4, -60e-4, -40e-4))
    return _save(target, grid, bare, frames, answers, 'half_filled_trench')


def via_with_void(target):
    """Radius 48 um, depth 80 um via; a sealed radius 36, height 48 um pocket.

    The final void spans z=-64..-16 um; a solid lid seals the mouth, field
    overburden is 8 um; floor deposition is 16 um, wall deposition 12 um.
    """
    grid = _grid()
    bare = _open_via(grid, 48e-4, -80e-4)
    pocket = np.pi * (36e-4)**2 * 48e-4
    volume = (192e-4)**2 * 8e-4 + np.pi*(48e-4)**2 * 80e-4 - pocket
    answers = dict(field_um=8., floor_um=16., wall_um=12.,
                   deposited_cu_volume_um3=volume * 1e12,
                   void_volume_um3=pocket * 1e12, void_centroid_um=[0., 0., -40.])
    def frames():
        yield bare
        yield _closed_via(grid, 40e-4, -72e-4, -8e-4, 4e-4)
        yield _closed_via(grid, 36e-4, -64e-4, -16e-4, 8e-4)
    return _save(target, grid, bare, frames(), answers, 'via_with_void')
