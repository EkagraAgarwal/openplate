"""Private geometry in micrometres; independent of analysis3d."""
import numpy as np

COPPER = '#b87333'
SILICON = '#929292'


def height_map(psi, grid):
    """Highest solid-to-liquid zero crossing; NaN for empty columns.

    Fully solid columns use the upper cell boundary (domain-limited height).
    """
    x, y, z = (c * 1e4 for c in grid.centers())
    a = np.asarray(psi)
    solid = a <= 0
    k = np.max(np.where(solid, np.arange(a.shape[2]), -1), axis=2)
    lo = np.clip(k, 0, len(z)-1)
    hi = np.minimum(lo+1, len(z)-1)
    v0 = np.take_along_axis(a, lo[..., None], 2)[..., 0]
    v1 = np.take_along_axis(a, hi[..., None], 2)[..., 0]
    fraction = np.divide(-v0, v1-v0, out=np.zeros_like(v0, dtype=float),
                         where=v1 != v0)
    height = z[lo] + fraction * (z[hi]-z[lo])
    height[k < 0] = np.nan
    height[k == len(z)-1] = z[-1] + grid.dx*5e3
    return dict(height_um=height, x_um=x, y_um=y)


def vertical_slice(psi, psi_si, grid, axis, position_um):
    """Linear interpolation on a vertical x/y-normal plane, clamped at centres."""
    coords = [c * 1e4 for c in grid.centers()]
    dim = 'xy'.index(axis)
    q = (position_um-coords[dim][0]) / (grid.dx*1e4)
    q = np.clip(q, 0, grid.shape[dim]-1)
    lo, hi = int(np.floor(q)), int(np.ceil(q))
    def sample(a):
        return (1-(q-lo))*np.take(a, lo, dim)+(q-lo)*np.take(a, hi, dim)
    return dict(s_um=coords[1-dim], z_um=coords[2],
                psi=sample(psi), psi_si=sample(psi_si))


def mesh(psi, grid):
    from skimage.measure import marching_cubes
    a = np.asarray(psi)
    if not (a.min() < 0 < a.max()):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices, faces, _, _ = marching_cubes(a, level=0, spacing=(grid.dx*1e4,)*3)
    vertices += np.array(grid.origin)*1e4 + grid.dx*5e3
    return vertices, faces


def camera_parameters(grid, preset, exaggeration):
    low = np.array(grid.origin)*1e4
    high = low + np.array(grid.shape)*grid.dx*1e4
    low[2] *= exaggeration
    high[2] *= exaggeration
    center = (low+high)/2
    span = float(np.max(high-low))
    direction = {'iso': (1.5, -1.8, 1.4), 'top': (0, 0, 3),
                 'front': (0, -3, 0)}[preset]
    return dict(preset=preset, position=(center+span*np.array(direction)).tolist(),
                focal_point=center.tolist(), view_up=[0, 1, 0] if preset == 'top' else [0, 0, 1],
                parallel_scale=span*.78, z_exaggeration=float(exaggeration))
