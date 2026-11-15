"""Cell-centred (x, y) sampling; mask distances are in micrometres."""
import numpy as np


def rasterize(layout, layer, window_um, dx_um):
    """Area-balanced binary raster: largest cell-overlap fractions win.

    Rounding the total covered area to the nearest cell bounds area error by
    half a cell, including six-cell-wide circles and off-grid rectangles where
    centre-only sampling cannot meet a 2% tolerance. Boundary cells remain
    binary, not fractional. Exact intersections are processed in bounded chunks.
    """
    from shapely import box, intersection, area
    window = np.asarray(window_um, dtype=float)
    if window.shape != (4,) or not np.isfinite(window).all():
        raise ValueError("window_um: use finite (x_lo, y_lo, x_hi, y_hi)")
    if not np.isfinite(dx_um) or dx_um <= 0:
        raise ValueError(f"dx_um={dx_um}: use finite spacing > 0")
    size = (window[2:]-window[:2])/dx_um
    if np.any(size <= 0) or not np.allclose(size, np.rint(size), atol=1e-8, rtol=0):
        raise ValueError("window_um: positive extents must be integer multiples of dx_um")
    nx, ny = np.rint(size).astype(int)
    geometry = layout.geometry(layer)
    coverage = np.zeros(nx*ny, dtype=float)
    for start in range(0, nx*ny, 65536):
        indices = np.arange(start, min(start+65536, nx*ny))
        x = window[0]+(indices//ny)*dx_um
        y = window[1]+(indices % ny)*dx_um
        cells = box(x, y, x+dx_um, y+dx_um)
        coverage[indices] = area(intersection(cells, geometry))/dx_um**2
    count = int(round(float(coverage.sum(dtype=np.float64))))
    result = np.zeros(nx*ny, dtype=bool)
    if count:
        # Stable sorting makes equal-overlap selections reproducible.
        result[np.argsort(-coverage, kind="stable")[:count]] = True
    return result.reshape(nx, ny)


def signed_distance_2d(mask, dx_um):
    """Sub-cell fast-marching distance; uniform masks have signed infinite distance.

    The boundary is halfway between opposing pixel centres. No artificial wall
    is introduced at the edge of a cropped window.
    """
    import skfmm
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("mask: supply a nonempty 2-D array")
    if not np.isfinite(dx_um) or dx_um <= 0:
        raise ValueError(f"dx_um={dx_um}: use finite spacing > 0")
    if mask.all() or not mask.any():
        return np.full(mask.shape, np.inf if mask.all() else -np.inf)
    return np.asarray(skfmm.distance(np.where(mask, 1., -1.), dx=float(dx_um)))
