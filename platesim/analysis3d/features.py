"""Geometry-only diagnostics, independent of stored scalar answers."""
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt

from .maps import _height
from .voids import _regions


def _wall_thickness(psi, bare, grid, region):
    z = grid.centers()[2]
    depths = z[np.any(region, axis=(0, 1))]
    if not depths.size:
        return None
    low, high = np.min(depths), min(0., np.max(depths) + grid.dx / 2)
    middle = (z > low + .25 * (high - low)) & (z < low + .75 * (high - low))
    vertical_gradient = np.gradient(psi, grid.dx, axis=2)
    values = []
    for axis in (0, 1):
        p = np.moveaxis(bare, axis, 0)
        cross = (p[:-1] * p[1:] < 0) & middle[None, None, :]
        i, j, k = np.nonzero(cross)
        if not len(i):
            continue
        f = -p[i, j, k] / (p[i + 1, j, k] - p[i, j, k])
        coords = np.array([i + f, j, k]) if axis == 0 else np.array([j, i + f, k])
        # Restrict to walls bounding the selected feature, not unrelated solids.
        selected = map_coordinates(region.astype(float), coords, order=0) > 0
        # Floor/lid-dominated distances are not lateral wall thickness [J].
        selected &= np.abs(map_coordinates(vertical_gradient, coords, order=1)) < .5
        values.extend((-map_coordinates(psi, coords[:, selected], order=1) * 1e4).tolist())
    return float(np.median(np.maximum(values, 0))) if values else None


def feature_report(result, feature_mask=None):
    """Final face thicknesses and per-frame occupancy-based fill/volume.

    Floor uses the lowest upward crossing, so a sealed lid is not mistaken
    for floor deposition. Field/floor medians exclude opening edges. Closure
    is the first saved frame with no feature electrolyte connected to bulk.
    Volumes/fill use cell-centre occupancy (not sub-cell quadrature). Wall
    thickness is a signed-distance proxy at mid-depth bare wall crossings;
    strongly tilted, fully buried or floor/lid-dominated walls are excluded.
    Undefined measurements and ratios return None, rather than infinity.
    """
    grid, bare = result.grid, result.psi_si
    z = grid.centers()[2]
    region = (bare > 0) & (z[None, None, :] < 0)
    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=bool)
        if mask.shape == grid.shape[:2]:
            mask = np.broadcast_to(mask[..., None], grid.shape)
        if mask.shape != grid.shape:
            raise ValueError('feature_mask must have grid shape or x-y shape')
        region &= mask
    opening = region.any(axis=2)
    # Interior columns avoid wall-intersection artifacts in floor measurements.
    interior = opening & (distance_transform_edt(opening) >= max(2., .5 * distance_transform_edt(opening).max()))
    field = ~((bare > 0) & (z[None, None, :] < 0)).any(axis=2)
    base = _height(bare, grid, upper=False)
    fills, volumes = [], []
    closure = None
    for index, time in enumerate(result.times_s):
        psi = result.psi(index)
        deposited = (bare > 0) & (psi <= 0)
        volumes.append(float(deposited.sum(dtype=np.float64) * grid.cell_volume * 1e12))
        fills.append(float((deposited & region).sum() / region.sum()) if region.any() else 0.)
        if closure is None and region.any():
            labels, trapped = _regions(psi, grid)
            connected = (labels > 0) & ~np.isin(labels, trapped)
            if not np.any(connected & region):
                closure = float(time)
    if not len(result.times_s):
        raise ValueError('feature_report requires at least one frame')
    psi = result.psi()
    top = _height(psi, grid)
    floor = _height(psi, grid, upper=False)
    def median(values, mask):
        v = values[mask & np.isfinite(values)]
        return float(np.median(v) * 1e4) if v.size else None
    field_um = median(top - base, field)
    floor_um = median(floor - base, interior)
    return dict(field_um=field_um, floor_um=floor_um,
                wall_um=_wall_thickness(psi, bare, grid, region),
                floor_over_field=(floor_um / field_um if field_um and floor_um is not None else None),
                overburden_um=median(top, field), fill_fraction=np.array(fills),
                t_s=np.array(result.times_s), closure_time_s=closure,
                deposited_cu_volume_um3=volumes[-1], deposited_cu_volume_series_um3=np.array(volumes))
