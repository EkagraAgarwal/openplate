"""Vertical surface measurements. Missing crossings are NaN, never extrapolated."""
import numpy as np


def _check(psi, grid):
    psi = np.asarray(psi, dtype=float)
    if psi.shape != grid.shape or not np.isfinite(psi).all():
        raise ValueError(f'psi must be finite with shape {grid.shape}')
    return psi


def _height(psi, grid, upper=True):
    p = _check(psi, grid)
    if p.shape[-1] < 2:
        return np.full(p.shape[:2], np.nan)
    crossing = (p[..., :-1] <= 0) & (p[..., 1:] > 0)
    if upper:
        index = p.shape[-1] - 2 - np.argmax(crossing[..., ::-1], axis=-1)
    else:
        index = np.argmax(crossing, axis=-1)
    lo = np.take_along_axis(p, index[..., None], -1)[..., 0]
    hi = np.take_along_axis(p, (index + 1)[..., None], -1)[..., 0]
    fraction = np.divide(-lo, hi - lo, out=np.zeros_like(lo), where=hi != lo)
    z = grid.centers()[2]
    return np.where(crossing.any(axis=-1), z[index] + fraction * grid.dx, np.nan)


def planarity(height_um):
    """Step is the median upper-half minus lower-half height; TIR is max-min."""
    h = np.asarray(height_um, dtype=float)
    h = np.sort(h[np.isfinite(h)])
    if not h.size:
        return dict(step_um=np.nan, tir_um=np.nan, std_um=np.nan)
    mid = max(1, h.size // 2)
    return dict(step_um=float(np.median(h[-mid:]) - np.median(h[:mid])),
                tir_um=float(np.ptp(h)), std_um=float(np.std(h)))


def height_map(psi, grid, region_um=None):
    """Topmost solid-to-electrolyte crossing, linearly interpolated in z.

    region_um, when supplied, is (xmin, xmax, ymin, ymax), inclusive.
    """
    h = _height(psi, grid) * 1e4
    x, y, _ = (axis * 1e4 for axis in grid.centers())
    if region_um is not None:
        xmin, xmax, ymin, ymax = region_um
        ix, iy = (x >= xmin) & (x <= xmax), (y >= ymin) & (y <= ymax)
        h, x, y = h[np.ix_(ix, iy)], x[ix], y[iy]
    return dict(height_um=h, x_um=x, y_um=y, stats=planarity(h))


def thickness_map(psi, psi_si, grid):
    """Vertical top-envelope displacement, not normal wall thickness."""
    out = height_map(psi, grid)
    out['thickness_um'] = out['height_um'] - _height(psi_si, grid) * 1e4
    out['stats'] = planarity(out['thickness_um'])
    return out
