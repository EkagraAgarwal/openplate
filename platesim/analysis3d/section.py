"""Trilinear vertical sections parameterized by polyline arc length."""
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

from .maps import _check


def cross_section(psi, psi_si, grid, polyline_um, n_samples=None):
    points = np.asarray(polyline_um, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('polyline_um must contain at least two x-y points')
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.isfinite(points).all() or np.any(lengths <= 0):
        raise ValueError('polyline vertices must be finite and segments nonzero')
    arc = np.r_[0., np.cumsum(lengths)]
    n = int(n_samples) if n_samples is not None else int(np.ceil(arc[-1] / (grid.dx * 1e4))) + 1
    if n < 2:
        raise ValueError('n_samples must be at least two')
    s = np.linspace(0, arc[-1], n)
    xy = np.column_stack([np.interp(s, arc, points[:, a]) for a in range(2)]) * 1e-4
    nz = grid.shape[2]
    coords = np.empty((3, n, nz))
    for a in range(2):
        q = (xy[:, a] - grid.origin[a]) / grid.dx - .5
        if grid.periodic[a]:
            # Explicit padding provides interpolation across periodic seams.
            q %= grid.shape[a]
        elif np.any((q < -.5) | (q > grid.shape[a] - .5)):
            raise ValueError('polyline lies outside the grid')
        coords[a] = q[:, None] + 1
    coords[2] = np.arange(nz)[None, :] + 1
    sampled = []
    for field in (psi, psi_si):
        padded = _check(field, grid)
        for a in range(3):
            widths = [(0, 0)] * 3
            widths[a] = (1, 1)
            padded = np.pad(padded, widths, mode='wrap' if grid.periodic[a] else 'edge')
        sampled.append(map_coordinates(padded, coords, order=1, prefilter=False))
    z = grid.centers()[2] * 1e4
    def contours(field):
        return [np.column_stack((np.interp(c[:, 0], np.arange(n), s),
                                 np.interp(c[:, 1], np.arange(nz), z)))
                for c in find_contours(field, 0)]
    return dict(s_um=s, z_um=z, psi=sampled[0], psi_si=sampled[1],
                cu_outline=contours(sampled[0]), si_outline=contours(sampled[1]))
