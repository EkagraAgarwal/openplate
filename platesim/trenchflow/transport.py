"""Steady advection-diffusion-reaction transport of Cu2+ in a trench cross-section.

Cell-centred finite volume on a :class:`platesim.trenchflow.geometry.CavityGrid`, dimensionless
concentration c (c = 1 in the bulk), per unit depth. Model in the electrolyte cells:

    u . grad(c) = D * laplacian(c)                       (electrolyte)
    c = 1                                               (top boundary z = grid.height_um,
                                                         half a cell above the last row)
    periodic in x
    -D * dc/dn = k * c                                  (every electrolyte/solid face,
                                                         first-order surface reaction)

Discretisation choices ([J] = derivation/judgement, no fitted constants)
- Unknowns are the electrolyte cells only (flattened row-major, ``iz * nx + ix``); solid cells are
  eliminated from the matrix and returned as NaN.
- Diffusion: face conductance ``D * dx / dx = D`` cm2/s (unit depth, dx = grid.dx_um * 1e-4 cm);
  the x direction wraps periodically.
- Advection is written in the conservative SIMPLE-style form ``sum_faces F_f * (c_f - c_P)`` with
  ``F_f = u_face * dx`` [cm2/s] the face flux (arithmetic mean of the two cell velocities; faces with
  a solid cell on either side are dropped, i.e. their face velocity is 0). Written this way every
  row of the advection operator sums to zero, so a uniform field is preserved exactly even when the
  supplied velocity field is only approximately divergence free (the remaining residual is
  ``-c_P * div(u)`` [J]).
- First-order upwind (default): ``c_f = c_P`` on outflow faces, ``c_N`` on inflow faces. Scheme
  ``"hybrid"`` uses central differencing while the face Peclet number ``|F| / D <= 2`` and upwind
  above that [J] (boundedness of the upwind branch, accuracy of the central branch).
- Reaction: the wall is half a cell from the cell centre, so one solid face is a series resistance
  ``1/k + (dx/2)/D``, giving the face conductance ``g = dx / (1/k + (dx/2)/D)`` [cm2/s] lumped on
  the diagonal. ``k -> inf`` gives ``g = 2 D`` and ``c_wall = 0`` (mass-transfer limit).
- Top boundary: Dirichlet value half a cell away -> conductance ``2 D`` and a source ``2 D * c_bulk``.
  Bottom of the array is solid (no boundary condition needed).

Outputs of :func:`solve_concentration` use flux_cell [cm2/s] (total wall flux of the cell per unit
depth) and surf_flux_density [cm/s] (mean flux density over the cell's solid faces); there is
deliberately no key called ``"flux"``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import surface_cells

_UM_TO_CM = 1e-4
_TINY = 1e-300
_SCHEMES = ("upwind", "hybrid")


def _inverse_k(k_cm_s, shape):
    """1/k per cell: 0 where k = inf (mass-transfer limit), inf where k = 0 (no reaction)."""
    k = np.broadcast_to(np.asarray(k_cm_s, dtype=float), shape)
    inv = np.full(shape, np.inf)
    pos = np.isfinite(k) & (k > 0.0)
    inv[pos] = 1.0 / k[pos]
    inv[np.isinf(k)] = 0.0
    return inv


def _solid_face_counts(solid):
    """Number of solid face-neighbours per cell (same count as geometry.surface_cells)."""
    s = np.asarray(solid, bool)
    nb = np.zeros(s.shape, dtype=np.int64)
    nb += np.roll(s, 1, axis=1).astype(np.int64) + np.roll(s, -1, axis=1).astype(np.int64)
    nb[1:] += s[:-1].astype(np.int64)
    nb[:-1] += s[1:].astype(np.int64)
    return nb


def _build_index(elyt):
    """Flat cell index -> unknown index (-1 for solid cells)."""
    idx = -np.ones(elyt.size, dtype=np.int64)
    idx[elyt.ravel()] = np.arange(int(elyt.sum()), dtype=np.int64)
    return idx


def _faces(grid, flow, idx, dx):
    """Interior electrolyte/electrolyte faces: unknowns (iP, iN) and signed face flux F [cm2/s].

    F > 0 means the face velocity points from P to N. Faces touching a solid cell are dropped
    because their face velocity is 0 by the geometry contract [J].
    """
    s = grid.solid
    elyt = ~s
    nz, nx = s.shape

    # x-directed faces: one per (iz, ix), partner (iz, ix + 1 mod nx) -> nx faces per row (periodic)
    iz_a = np.repeat(np.arange(nz), nx)
    ix_a = np.tile(np.arange(nx), nz)
    ix_b = (ix_a + 1) % nx
    okx = elyt[iz_a, ix_a] & elyt[iz_a, ix_b]

    # z-directed faces: partner (iz + 1, ix)
    iz_c = np.repeat(np.arange(nz - 1), nx) if nz > 1 else np.empty(0, dtype=np.int64)
    ix_c = np.tile(np.arange(nx), max(nz - 1, 0))
    okz = elyt[iz_c, ix_c] & elyt[iz_c + 1, ix_c] if nz > 1 else np.zeros(0, bool)

    n_fx = int(okx.sum())
    n_fz = int(okz.sum())
    if flow is None:
        F = np.zeros(n_fx + n_fz)
    else:
        ux = np.asarray(flow.ux, dtype=float)
        uz = np.asarray(flow.uz, dtype=float)
        Fx = 0.5 * (ux[iz_a, ix_a] + ux[iz_a, ix_b])[okx] * dx      # cm/s * cm -> cm2/s
        Fz = 0.5 * (uz[iz_c, ix_c] + uz[iz_c + 1, ix_c])[okz] * dx
        F = np.concatenate([Fx, Fz])
    iP = np.concatenate([idx[(iz_a * nx + ix_a)[okx]], idx[(iz_c * nx + ix_c)[okz]]])
    iN = np.concatenate([idx[(iz_a * nx + ix_b)[okx]], idx[((iz_c + 1) * nx + ix_c)[okz]]])
    return iP, iN, F


def _face_conductances(F, D, scheme):
    """Per-face conductances (P->N, N->P).

    Upwind: ``D + max(-F, 0)`` for P (the donor cell gets the flux on its diagonal) and
    ``D + max(F, 0)`` for N. Both directions carry the full diffusion conductance, so every row of
    the matrix sums to zero. Hybrid replaces the advective part by central differencing when
    ``|F| <= 2 D`` [J].
    """
    ud_p = D + np.maximum(-F, 0.0)
    ud_n = D + np.maximum(F, 0.0)
    if scheme == "upwind":
        return ud_p, ud_n
    central = np.abs(F) <= 2.0 * D
    return np.where(central, D - 0.5 * F, ud_p), np.where(central, D + 0.5 * F, ud_n)


def _region_masks(grid, iz, ix):
    """Boolean groups of surface cells: field, trench floor, trench wall (see region_means)."""
    s = grid.solid
    x = np.asarray(grid.x_um)[np.asarray(ix)]
    z = np.asarray(grid.z_um)[np.asarray(iz)]
    iz = np.asarray(iz)
    ix = np.asarray(ix)
    tol = 1e-6 * max(abs(float(grid.dx_um)), 1.0)
    xl = float(grid.x_left_um)
    xr = xl + float(grid.width_um)
    dep = float(grid.deposit_um)

    below_solid = np.zeros(iz.size, bool)
    has_below = iz > 0
    # [J] the bottom row of the array cannot have a solid neighbour below it
    below_solid[has_below] = s[iz[has_below] - 1, ix[has_below]]

    in_trench = (x > xl - tol) & (x < xr + tol)
    near_trench = (x > xl - dep - tol) & (x < xr + dep + tol)
    field = (z > dep + tol) & ~near_trench
    floor = in_trench & below_solid
    wall = in_trench & ~below_solid
    return field, floor, wall


def solve_concentration(grid, flow=None, D_cm2_s=5.6e-6, k_cm_s=1e-3, scheme="upwind"):
    """Steady Cu2+ concentration field, wall flux and surface-cell statistics.

    Parameters
    ----------
    grid : CavityGrid
    flow : FlowField or None
        Face velocities in cm/s (``ux``, ``uz``), zero in solid. ``None`` = pure diffusion.
    D_cm2_s : float
        Cu2+ diffusivity (default 5.6e-6 cm2/s, typical aqueous value [J]).
    k_cm_s : float or array of grid.shape
        First-order surface reaction rate, cm/s. ``numpy.inf`` = mass-transfer limit (c_wall = 0),
        ``0`` = inert wall.
    scheme : {"upwind", "hybrid"}

    Returns
    -------
    dict with keys ``c``, ``flux_cell``, ``surf_iz``, ``surf_ix``, ``surf_flux_density``,
    ``surf_x_um``, ``surf_z_um``, ``total_wall_flux``, ``top_flux``, ``balance_error``,
    ``delta_eff_um``, ``k_cm_s``, ``D_cm2_s``, ``scheme``.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
    if flow is not None and flow.grid.shape != grid.shape:
        raise ValueError("flow field shape does not match grid")

    s = grid.solid
    nz, nx = s.shape
    elyt = ~s
    ne = int(elyt.sum())
    if ne == 0:
        raise ValueError("grid has no electrolyte cells")

    dx = float(grid.dx_um) * _UM_TO_CM
    D = float(D_cm2_s)
    idx = _build_index(elyt)

    # ---- interior faces, conductances, matrix --------------------------------
    iP, iN, F = _faces(grid, flow, idx, dx)
    a_p, a_n = _face_conductances(F, D, scheme)
    diag = np.bincount(iP, a_p, minlength=ne) + np.bincount(iN, a_n, minlength=ne)

    # ---- wall reaction (series resistance 1/k + (dx/2)/D per solid face) -----
    inv_k = _inverse_k(k_cm_s, (nz, nx))
    g_wall = dx / (0.5 * dx / D + inv_k)                     # cm2/s per solid face
    n_faces = _solid_face_counts(s)
    diag += (n_faces * g_wall)[elyt]                          # ravel order == idx order

    # ---- top Dirichlet boundary, half a cell above the last row -------------
    rhs = np.zeros(ne)
    top_mask = elyt[-1, :]
    if np.any(top_mask):
        g_top = 2.0 * D                                      # D * dx / (dx/2)
        top_unknowns = idx[(nz - 1) * nx + np.flatnonzero(top_mask)]
        diag[top_unknowns] += g_top
        rhs[top_unknowns] += g_top                           # c_bulk = 1

    rows = np.concatenate([np.arange(ne, dtype=np.int64), iP, iN])
    cols = np.concatenate([np.arange(ne, dtype=np.int64), iN, iP])
    data = np.concatenate([diag, -a_p, -a_n])
    A = sp.coo_matrix((data, (rows, cols)), shape=(ne, ne)).tocsr()
    c_flat = spla.spsolve(A, rhs)
    if not np.all(np.isfinite(c_flat)):
        raise RuntimeError("linear solve produced non-finite concentrations")

    # ---- unpack, wall fluxes ------------------------------------------------
    cval = np.zeros((nz, nx))
    cval[elyt] = c_flat
    c = np.where(elyt, cval, np.nan)

    flux_cell = n_faces * g_wall * cval                      # cm2/s per unit depth
    flux_cell = np.where(elyt, flux_cell, np.nan)

    iz_s, ix_s, nf_s = surface_cells(grid)
    cd = cval[iz_s, ix_s]
    gw = g_wall[iz_s, ix_s]
    fd = gw * cd / dx                                       # mean flux density over the solid faces, cm/s

    total_wall_flux = float(np.sum(nf_s * gw * cd))
    top_flux = float(np.sum(2.0 * D * (1.0 - cval[-1, np.flatnonzero(top_mask)]))) if np.any(top_mask) else 0.0
    balance_error = abs(top_flux - total_wall_flux) / max(abs(top_flux), _TINY)

    # ---- effective Nernst layer for the field surface -----------------------
    field, _, _ = _region_masks(grid, iz_s, ix_s)
    inv_k_s = inv_k[iz_s, ix_s]
    c_wall_s = np.zeros_like(fd)                                  # [J] k = 0 -> undefined, use 0
    known = np.isfinite(inv_k_s) & np.isfinite(fd)
    c_wall_s[known] = fd[known] * inv_k_s[known]                  # c_wall = flux_density / k
    if np.any(field) and np.mean(fd[field]) > 0.0:
        delta_eff_um = float(D * (1.0 - np.mean(c_wall_s[field])) / np.mean(fd[field]) / _UM_TO_CM)
    else:
        delta_eff_um = float("nan")

    return {
        "c": c,
        "flux_cell": flux_cell,
        "surf_iz": iz_s,
        "surf_ix": ix_s,
        "surf_flux_density": fd,
        "surf_x_um": np.asarray(grid.x_um)[ix_s],
        "surf_z_um": np.asarray(grid.z_um)[iz_s],
        "total_wall_flux": total_wall_flux,
        "top_flux": top_flux,
        "balance_error": balance_error,
        "delta_eff_um": delta_eff_um,
        "k_cm_s": k_cm_s,
        "D_cm2_s": D,
        "scheme": scheme,
    }


def region_means(result, grid):
    """Mean wall flux density (cm/s) over the field surface, the trench floor and the trench walls.

    Groups of surface cells (from :func:`platesim.trenchflow.geometry.surface_cells`):
    - ``field``: surface cells with z > deposit_um (the field plane) and x outside
      [x_left - deposit_um, x_left + width_um + deposit_um].
    - ``floor``: surface cells inside the open trench whose solid neighbour is BELOW them.
    - ``wall``: the remaining surface cells inside the open trench (the side walls).

    Returns
    -------
    dict with ``field``, ``floor``, ``wall`` (mean flux density, cm/s, NaN for an empty group),
    ``floor_over_field``, ``wall_over_field`` and the counts ``n_field``, ``n_floor``, ``n_wall``.
    """
    iz = np.asarray(result["surf_iz"])
    ix = np.asarray(result["surf_ix"])
    fd = np.asarray(result["surf_flux_density"], dtype=float)
    field, floor, wall = _region_masks(grid, iz, ix)

    def _mean(m):
        return float(np.mean(fd[m])) if np.any(m) else float("nan")

    m_field, m_floor, m_wall = _mean(field), _mean(floor), _mean(wall)
    with np.errstate(invalid="ignore", divide="ignore"):
        floor_over_field = m_floor / m_field if m_field else float("nan")
        wall_over_field = m_wall / m_field if m_field else float("nan")
    return {
        "field": m_field,
        "floor": m_floor,
        "wall": m_wall,
        "floor_over_field": float(floor_over_field),
        "wall_over_field": float(wall_over_field),
        "n_field": int(field.sum()),
        "n_floor": int(floor.sum()),
        "n_wall": int(wall.sum()),
    }


def sherwood_scan(grid, flow, D_cm2_s, k_values):
    """Run :func:`solve_concentration` for every k and collect :func:`region_means`.

    Returns a list of dicts, one per k, each holding the region means plus ``k_cm_s``,
    ``flux_field``, ``delta_eff_um``, ``balance_error`` and the dimensionless Sherwood number
    ``sherwood_field`` = flux_field * depth / D, using the feature depth as the length scale [J].
    """
    out = []
    depth_cm = float(grid.depth_um) * _UM_TO_CM
    for k in k_values:
        res = solve_concentration(grid, flow, D_cm2_s=D_cm2_s, k_cm_s=k)
        rm = region_means(res, grid)
        row = dict(rm)
        row["k_cm_s"] = float(k) if np.isfinite(k) else float("inf")
        row["flux_field"] = rm["field"]
        row["delta_eff_um"] = res["delta_eff_um"]
        row["balance_error"] = res["balance_error"]
        with np.errstate(invalid="ignore", divide="ignore"):
            row["sherwood_field"] = (
                float(rm["field"] * depth_cm / float(D_cm2_s)) if np.isfinite(rm["field"]) else float("nan")
            )
        out.append(row)
    return out
