"""Shared geometry and field containers for platesim.trenchflow (feature-scale flow + particles).

This file is the CONTRACT between the flow solver (cavity_lbm.py), the continuum transport solver
(transport.py) and the Brownian particle solver (particles.py). Do not change names or conventions.

Conventions
- 2-D cross-section of a long trench, x across the trench, z normal to the wafer.
- z = 0 is the original Si field surface (wafer plane); the trench floor is at z = -depth.
- Arrays are indexed [iz, ix]; iz = 0 is the BOTTOM row; cell (iz, ix) is centred at (x_um[ix], z_um[iz]).
- The domain is periodic in x with period pitch_um; the trench is centred in the domain.
- solid[iz, ix] is True in Si or Cu, False in electrolyte.
- Lengths in um in names ending _um; velocities in cm/s; diffusivity in cm2/s; time in s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CavityGrid:
    dx_um: float
    x_um: np.ndarray          # (nx,) cell centres, 0 at the left edge of the periodic domain
    z_um: np.ndarray          # (nz,) cell centres, increasing upward, 0 = wafer plane
    solid: np.ndarray         # (nz, nx) bool
    width_um: float
    depth_um: float
    pitch_um: float
    height_um: float          # electrolyte height above the wafer plane (top boundary at z = height_um)
    deposit_um: float = 0.0   # conformal Cu thickness already included in `solid`

    @property
    def shape(self):
        return self.solid.shape

    @property
    def x_left_um(self):
        """x of the left Si trench wall."""
        return 0.5 * (self.pitch_um - self.width_um)


@dataclass
class FlowField:
    grid: CavityGrid
    ux: np.ndarray            # (nz, nx) cm/s, 0 inside solid
    uz: np.ndarray            # (nz, nx) cm/s, 0 inside solid
    shear_rate_s: float       # imposed shear rate far above the wafer, 1/s
    converged: bool = True
    info: dict = field(default_factory=dict)


def _dist_to_si(xx, zz, x0, width, depth, pitch):
    """Distance from electrolyte points to the Si surface of the periodic trench array (negative inside Si)."""
    xr = (xx - x0) % pitch                     # 0..pitch, trench occupies 0 < xr < width
    inside = (zz > 0) | ((xr > 0) & (xr < width) & (zz > -depth))
    # field surface: z = 0 for width <= xr <= pitch
    dx_f = np.where(xr < width, np.minimum(xr, width - xr), 0.0)     # horizontal distance to the field strip
    d_field = np.hypot(dx_f, zz)
    # walls: x = 0 and x = width for -depth <= z <= 0
    dz_w = np.where(zz > 0, zz, np.where(zz < -depth, -depth - zz, 0.0))
    d_wall = np.minimum(np.hypot(xr, dz_w), np.hypot(xr - width, dz_w))
    # floor: z = -depth for 0 <= xr <= width
    dx_fl = np.where(xr < 0, -xr, np.where(xr > width, xr - width, 0.0))
    d_floor = np.hypot(dx_fl, zz + depth)
    d = np.minimum(np.minimum(d_field, d_wall), d_floor)
    return np.where(inside, d, -d)


def trench_grid(width_um, depth_um, pitch_um=None, height_um=None, dx_um=4.0, deposit_um=0.0) -> CavityGrid:
    """Periodic trench cross-section with an optional conformal Cu deposit of thickness deposit_um.

    Defaults: pitch = width + max(4 * depth, width) (isolated-trench-like), height = max(3 * depth, 250 um).
    """
    pitch_um = float(pitch_um) if pitch_um else float(width_um + max(4.0 * depth_um, width_um))
    height_um = float(height_um) if height_um else float(max(3.0 * depth_um, 250.0))
    nx = int(round(pitch_um / dx_um))
    nz_below = int(np.ceil((depth_um + 2 * dx_um) / dx_um))
    nz_above = int(round(height_um / dx_um))
    x = (np.arange(nx) + 0.5) * dx_um
    z = (np.arange(-nz_below, nz_above) + 0.5) * dx_um
    xx, zz = np.meshgrid(x, z)
    x0 = 0.5 * (nx * dx_um - width_um)
    d = _dist_to_si(xx, zz, x0, float(width_um), float(depth_um), nx * dx_um)
    solid = d <= float(deposit_um)
    return CavityGrid(dx_um=float(dx_um), x_um=x, z_um=z, solid=solid, width_um=float(width_um),
                      depth_um=float(depth_um), pitch_um=nx * dx_um, height_um=nz_above * dx_um,
                      deposit_um=float(deposit_um))


def couette_flow(grid: CavityGrid, shear_rate_s: float) -> FlowField:
    """Analytic test field: ux = shear * z above the wafer plane (z > deposit), zero below. Not a solution
    inside the trench; used by unit tests of the transport and particle solvers over a FLAT wall."""
    zz = np.broadcast_to(grid.z_um[:, None], grid.shape)
    ux = np.where(grid.solid, 0.0, shear_rate_s * np.clip(zz - grid.deposit_um, 0.0, None) * 1e-4)
    return FlowField(grid=grid, ux=ux, uz=np.zeros(grid.shape), shear_rate_s=float(shear_rate_s),
                     info={"kind": "couette"})


def flat_grid(pitch_um=200.0, height_um=200.0, dx_um=4.0) -> CavityGrid:
    """Flat wall (no trench): two solid rows below z = 0. For analytic validation cases."""
    nx = int(round(pitch_um / dx_um))
    nz_above = int(round(height_um / dx_um))
    x = (np.arange(nx) + 0.5) * dx_um
    z = (np.arange(-2, nz_above) + 0.5) * dx_um
    solid = np.broadcast_to((z < 0)[:, None], (len(z), nx)).copy()
    return CavityGrid(dx_um=float(dx_um), x_um=x, z_um=z, solid=solid, width_um=0.0, depth_um=0.0,
                      pitch_um=nx * dx_um, height_um=nz_above * dx_um)


def surface_cells(grid: CavityGrid):
    """Electrolyte cells that touch solid through a face. Returns (iz, ix, n_faces) arrays; n_faces is the
    number of solid face-neighbours (1 on flat surfaces, 2 in concave corners). x is periodic."""
    s = grid.solid
    nb = np.zeros(s.shape, int)
    nb += np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1)
    nb[1:] += s[:-1]
    nb[:-1] += s[1:]
    m = (~s) & (nb > 0)
    iz, ix = np.nonzero(m)
    return iz, ix, nb[iz, ix]
