"""Bare silicon geometry, independent of spec3d and tile3d."""
import numpy as np

from .raster import signed_distance_2d


def _rounded_intersection(wall, floor, radius, angle):
    """Distance to an eroded then dilated wedge (a circular floor fillet)."""
    if radius == 0:
        return np.minimum(wall, floor)
    u, v = wall-radius, floor-radius
    cosine, sine = np.cos(angle), np.sin(angle)
    corner = -np.hypot((u-v*cosine)/sine, v)
    inside = (u >= 0) & (v >= 0)
    distance = np.where(inside, np.minimum(u, v), corner)
    distance = np.where((u < 0) & (v-u*cosine >= 0), u, distance)
    distance = np.where((v < 0) & (u-v*cosine >= 0), v, distance)
    return distance+radius


def etch_to_psi(mask, dx_um, etch_layer, z_extent_um):
    """Return float32 psi in cm and a plain cell-centred grid description.

    z_extent_um is (z_low, z_high), on cell faces. XY origin is (0, 0):
    a bare mask cannot carry the raster window origin. Connected components
    define features for depth_vs_width; width is twice their largest inscribed
    radius. Side boundaries of the returned grid are symmetry, not periodic.
    Seed coverage is intentionally left to the subsequent seed/plating model.
    """
    import skfmm
    from scipy.ndimage import label
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("mask: supply a nonempty 2-D array")
    if not np.isfinite(dx_um) or dx_um <= 0:
        raise ValueError(f"dx_um={dx_um}: use finite spacing > 0")
    extent = np.asarray(z_extent_um, dtype=float)
    if extent.shape != (2,) or not np.isfinite(extent).all() or extent[1] <= extent[0]:
        raise ValueError(f"z_extent_um={z_extent_um}: use finite (low, high), low < high")
    cells = (extent[1]-extent[0])/dx_um
    if not np.isclose(cells, round(cells), atol=1e-8, rtol=0):
        raise ValueError("z_extent_um: extent must be an integer multiple of dx_um")
    depth = float(etch_layer.get("depth_um", 0))
    angle_deg = float(etch_layer.get("sidewall_deg", 90))
    radius = float(etch_layer.get("bottom_radius_um", 0))
    if not np.isfinite(depth) or depth < 0:
        raise ValueError(f"etch.depth_um={depth}: use a finite depth >= 0")
    if not np.isfinite(angle_deg) or not 0 < angle_deg <= 90:
        raise ValueError(f"etch.sidewall_deg={angle_deg}: use 0 < angle <= 90")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError(f"etch.bottom_radius_um={radius}: use a finite radius >= 0")
    table = etch_layer.get("depth_vs_width")
    if table is not None:
        table = np.asarray(table, dtype=float)
        if (table.ndim != 2 or table.shape[1] != 2 or len(table) == 0
                or not np.isfinite(table).all() or np.any(table < 0)
                or np.any(np.diff(table[:, 0]) <= 0)):
            raise ValueError("etch.depth_vs_width: use increasing nonnegative [width, depth] pairs")
    z = extent[0]+(np.arange(int(round(cells)))+.5)*dx_um
    shape = (*mask.shape, len(z))
    phi = np.broadcast_to(z, shape).copy()
    components, count = label(mask)
    angle = np.deg2rad(angle_deg)
    for feature in range(1, count+1):
        component = components == feature
        distance = signed_distance_2d(component, dx_um)
        feature_depth = depth
        if table is not None:
            width = 2*np.max(distance[component])
            feature_depth = float(np.interp(width, table[:, 0], table[:, 1]))
        if radius > feature_depth:
            raise ValueError("etch.bottom_radius_um: reduce radius to at most the local depth")
        wall = distance[:, :, None]*np.sin(angle)+z[None, None, :]*np.cos(angle)
        floor = z[None, None, :]+feature_depth
        cavity = _rounded_intersection(wall, floor, radius, angle)
        phi = np.maximum(phi, cavity)
    # One 3-D fast-marching pass restores physical distance after CSG operations.
    # A wholly solid/electrolyte crop has no zero contour; retain its analytic field.
    if phi.min() < 0 < phi.max():
        phi = np.asarray(skfmm.distance(phi, dx=float(dx_um)))
    bc = {"x_lo": "symmetry", "x_hi": "symmetry", "y_lo": "symmetry",
          "y_hi": "symmetry", "z_lo": "bottom", "z_hi": "bulk"}
    grid = dict(shape=shape, dx=float(dx_um)*1e-4,
                origin=(0., 0., float(extent[0])*1e-4), bc=bc)
    return (phi*1e-4).astype(np.float32), grid
