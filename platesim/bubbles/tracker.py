"""One-way-coupled Lagrangian micro-bubble tracker in a 2-D trench cross-section (plan item E2a).

A point bubble is advected by the liquid velocity field (:class:`platesim.trenchflow.geometry.
FlowField`, cm/s, ``[iz, ix]``) plus its own terminal slip velocity, grows or shrinks by
Epstein-Plesset diffusion (:func:`platesim.bubbles.physics.diffusive_growth_rate_um_s`), sticks when
it reaches a solid cell and detaches once it is larger than the Fritz radius. There is NO two-way
coupling: the flow field is a frozen input and the bubbles do not feel each other.

Gravity is a parameter, ``gravity_dir`` (x, z), because the coupon hangs vertically:
- ``(0, -1)``: gravity towards -z (wafer face up in a horizontal cell) -> bubbles rise towards +z.
- ``(+1, 0)``: gravity along +x (wafer face vertical, trench axis vertical) -> bubbles rise in -x.
- ``(0, 0)``: gravity out of the cross-section -> no in-plane slip, the bubble follows the liquid.

Units follow platesim.trenchflow: positions and radii in um, velocities in cm/s, time in s, the grid
periodic in x with pitch ``grid.pitch_um``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from . import physics

if TYPE_CHECKING:                                    # pragma: no cover - typing only
    from ..trenchflow.geometry import FlowField


def _bilinear(field, x, z, grid):
    """Bilinear interpolation of a cell-centred field at positions (um); x periodic, z clamped.

    Cell (iz, ix) is centred at ``(x_um[ix], z_um[iz])`` and the spacing is ``dx_um`` in both
    directions (the geometry contract), so the weights are the fractional offsets from the four
    surrounding cell centres. Because ``FlowField.ux/uz`` are exactly 0 in solid cells, the
    interpolation already sees a no-slip wall one cell away from the interface.
    """
    dx = float(grid.dx_um)
    nx, nz = grid.x_um.size, grid.z_um.size
    x_lo = grid.x_um[0] - 0.5 * dx
    z_lo = grid.z_um[0] - 0.5 * dx
    fx = (np.asarray(x, float) - x_lo) / dx - 0.5
    fz = (np.asarray(z, float) - z_lo) / dx - 0.5
    iz0 = np.clip(np.floor(fz).astype(np.int64), 0, nz - 1)
    tz = np.clip(fz - iz0, 0.0, 1.0)
    iz1 = np.minimum(iz0 + 1, nz - 1)
    fxi = np.floor(fx).astype(np.int64)
    ix0 = fxi % nx
    tx = fx - fxi
    ix1 = (ix0 + 1) % nx
    return ((1.0 - tz) * ((1.0 - tx) * field[iz0, ix0] + tx * field[iz0, ix1])
            + tz * ((1.0 - tx) * field[iz1, ix0] + tx * field[iz1, ix1]))


class _Contacts:
    """Nearest solid-cell-centre lookup, periodic in x, for the adhesion test.

    A bubble of radius R touches the solid when the nearest solid cell centre is closer than
    ``R + dx/2``: the cell is a dx x dx square, so its inscribed circle has radius dx/2 and the
    spherical envelope of the bubble reaches into it. This is a one-cell-resolution surrogate for a
    signed distance function [J].
    """

    def __init__(self, grid):
        self.grid = grid
        self.half_cell = 0.5 * float(grid.dx_um)
        s = grid.solid
        if s is None or not np.any(s):
            self.tree = None
            return
        iz, ix = np.nonzero(s)
        x = grid.x_um[ix]
        z = grid.z_um[iz]
        pitch = float(grid.pitch_um)
        self.tree = cKDTree(np.column_stack([np.concatenate([x, x - pitch, x + pitch]),
                                            np.concatenate([z, z, z])]))
        self._x = x
        self._z = z
        self._ix = ix
        self._iz = iz
        self._pitch = pitch

    def nearest(self, x, z):
        """Distance (um) to the nearest solid cell centre and the index of that cell."""
        if self.tree is None:
            return float("inf"), -1
        d, k = self.tree.query([float(x), float(z)])
        k = int(k) % max(self._x.size, 1)
        return float(d), k

    def cell(self, k):
        return int(self._iz[k]), int(self._ix[k])

    def touches(self, x, z, R_um):
        d, k = self.nearest(x, z)
        return bool(d <= float(R_um) + self.half_cell), d, k


def track_bubbles(flow: "FlowField | None", R0_um, x0_um, z0_um, t_end_s, dt_s,
                  gravity_dir=(0.0, -1.0),
                  supersat=0.0, theta_deg=45.0, sigma=physics.SIGMA_WATER_N_M,
                  drho=physics.DRHO_KG_M3, mu=physics.MU_LIQUID_PA_S,
                  rho_l=physics.RHO_LIQUID_KG_M3, D_m2_s=physics.D_H2_WATER_M2_S,
                  k_H=physics.K_H_H2_MOL_L_ATM, p_atm_Pa=physics.P_ATM_PA, T_C=25.0,
                  attach=True, slip_model="hadamard", cfl=0.5, r_min_um=0.05,
                  max_substeps=400):
    """Track point bubbles in a frozen 2-D flow field; one-way coupling only.

    Each free bubble moves with ``u_liquid(x, z) + u_slip`` where ``u_liquid`` is the bilinear
    interpolation of ``flow.ux/uz`` (cm/s) and ``u_slip`` is the terminal rise velocity along
    ``-gravity_dir`` from :func:`platesim.bubbles.physics.rise_velocity`
    (``slip_model='hadamard'`` by default: a gas bubble has a mobile interface). x is periodic with
    ``flow.grid.pitch_um``; z is bounded by the grid. With ``flow=None`` the liquid is quiescent, the
    domain is unbounded and no wall exists (pure rise / pure growth test case).

    Radius: every substep integrates dR/dt from
    :func:`platesim.bubbles.physics.diffusive_growth_rate_um_s` with ``c_inf = (1 + supersat) c_sat``
    and the interface pressure ``p_atm + 2 sigma/R``. A bubble that shrinks to ``r_min_um`` is marked
    dissolved and frozen.

    Adhesion: a free bubble whose envelope reaches a solid cell (see :class:`_Contacts`) sticks; its
    position is frozen while it keeps growing. It detaches when ``R > R_fritz`` (Fritz radius for
    ``theta_deg``), and is pushed one contact radius off the wall so it does not re-stick at once.

    Time stepping: an explicit Euler step with internal substepping that keeps the displacement below
    ``cfl * dx`` per substep (``cfl=0.5``, capped at ``max_substeps`` substeps per output interval, so
    a very fast bubble on a coarse grid is under-resolved rather than silently slow). Output is
    recorded at every ``dt_s``.

    Returns a dict: ``t_s`` (n_rec,), ``x_um``/``z_um``/``R_um`` (n_rec, n_bubbles),
    ``attached`` (n_rec, n_bubbles) bool, ``events`` (list of dicts with ``t_s``, ``bubble``,
    ``kind`` in attach/detach/dissolved/left_domain, and the position), ``fritz_radius_um``,
    ``reynolds`` / ``re_gt_1`` (the INITIAL rise Reynolds number per bubble, from
    :func:`platesim.bubbles.physics.rise_velocity`) and the ``warnings`` raised by them, plus the
    ``settings`` used.
    """
    if float(dt_s) <= 0.0:
        raise ValueError("dt_s must be positive")
    if float(t_end_s) < 0.0:
        raise ValueError("t_end_s must be non-negative")
    if slip_model not in ("hadamard", "stokes"):
        raise ValueError("slip_model must be 'hadamard' or 'stokes'")

    x, z, R = (a.astype(float).copy() for a in np.broadcast_arrays(
        np.atleast_1d(np.asarray(x0_um, dtype=float)),
        np.atleast_1d(np.asarray(z0_um, dtype=float)),
        np.atleast_1d(np.asarray(R0_um, dtype=float))))
    nb = int(R.size)
    if np.any(R <= 0.0):
        raise ValueError("R0_um must be positive")

    gv = np.asarray(gravity_dir, dtype=float).ravel()
    if gv.size != 2:
        raise ValueError("gravity_dir must be a 2-vector (x, z)")
    gn = float(np.hypot(gv[0], gv[1]))
    slip_dir = -gv / gn if gn > 0.0 else np.zeros(2)
    grid = getattr(flow, "grid", None)
    contacts = _Contacts(grid) if grid is not None else None
    pitch = float(grid.pitch_um) if grid is not None else 0.0
    z_lo = float(grid.z_um[0]) if grid is not None else -np.inf
    z_hi = float(grid.z_um[-1]) if grid is not None else np.inf
    R_fritz = physics.fritz_detachment_radius_um(theta_deg, sigma, drho) if sigma > 0.0 else 0.0
    rise0 = [physics.rise_velocity(r, drho=drho, mu=mu, rho_l=rho_l) for r in R]
    warnings = [w for w in (rv["warning"] for rv in rise0) if w]

    def slip_cm_s(r_um):
        rv = physics.rise_velocity(r_um, drho=drho, mu=mu, rho_l=rho_l)
        return (rv["U_hr_m_s"] if slip_model == "hadamard" else rv["U_stokes_m_s"]) * 1e2

    n_rec = int(round(float(t_end_s) / float(dt_s))) + 1
    times = np.arange(n_rec) * float(dt_s)
    X = np.empty((n_rec, nb))
    Z = np.empty((n_rec, nb))
    RR = np.empty((n_rec, nb))
    ATT = np.zeros((n_rec, nb), bool)
    alive = np.ones(nb, bool)
    attached = np.zeros(nb, bool)
    events = []
    for i in range(nb):
        if contacts is not None and attach and contacts.touches(x[i], z[i], R[i])[0]:
            attached[i] = True
            events.append({"t_s": 0.0, "bubble": i, "kind": "attach",
                           "x_um": float(x[i]), "z_um": float(z[i]), "R_um": float(R[i])})

    t = 0.0
    for k in range(n_rec):
        X[k], Z[k], RR[k], ATT[k] = x, z, R, attached
        if k == n_rec - 1:
            break
        dt_out = float(times[k + 1] - times[k])
        vx = _bilinear(flow.ux, x, z, grid) if grid is not None else np.zeros(nb)
        vz = _bilinear(flow.uz, x, z, grid) if grid is not None else np.zeros(nb)
        free = alive & ~attached
        speed = np.zeros(nb)
        for i in np.nonzero(free)[0]:
            speed[i] = math.hypot(vx[i] + slip_dir[0] * slip_cm_s(R[i]),
                                  vz[i] + slip_dir[1] * slip_cm_s(R[i]))
        if grid is not None and speed.max() > 0.0:
            n_sub = int(min(max_substeps,
                            max(1, math.ceil(speed.max() * 1e4 * dt_out
                                             / (cfl * float(grid.dx_um))))))
        else:
            n_sub = 1
        dt_sub = dt_out / n_sub
        for _ in range(n_sub):
            t_next = t + dt_sub
            for i in np.nonzero(alive)[0]:
                dR = physics.diffusive_growth_rate_um_s(
                    R[i], supersat=supersat, D_m2_s=D_m2_s, k_H=k_H, p_atm_Pa=p_atm_Pa,
                    sigma=sigma, T_C=T_C, t_s=t_next)
                R[i] += dR * dt_sub
                if R[i] <= r_min_um:
                    R[i] = r_min_um
                    alive[i] = False
                    attached[i] = False
                    events.append({"t_s": t_next, "bubble": i, "kind": "dissolved",
                                   "x_um": float(x[i]), "z_um": float(z[i]), "R_um": float(R[i])})
            if grid is not None:
                vx = _bilinear(flow.ux, x, z, grid)
                vz = _bilinear(flow.uz, x, z, grid)
            for i in np.nonzero(alive & ~attached)[0]:
                u = slip_cm_s(R[i])
                x[i] = (x[i] + (vx[i] + slip_dir[0] * u) * dt_sub * 1e4)
                z[i] = z[i] + (vz[i] + slip_dir[1] * u) * dt_sub * 1e4
                if grid is not None:
                    x[i] %= pitch
                    if z[i] > z_hi:
                        z[i] = z_hi
                        alive[i] = False
                        events.append({"t_s": t_next, "bubble": i, "kind": "left_domain",
                                       "x_um": float(x[i]), "z_um": float(z[i]),
                                       "R_um": float(R[i])})
                    elif z[i] < z_lo:
                        z[i] = z_lo
            t = t_next
        # adhesion and detachment happen at the end of the output interval
        for i in np.nonzero(alive & ~attached)[0]:
            if contacts is not None and attach and contacts.touches(x[i], z[i], R[i])[0]:
                attached[i] = True
                events.append({"t_s": t, "bubble": i, "kind": "attach",
                               "x_um": float(x[i]), "z_um": float(z[i]), "R_um": float(R[i])})
        for i in np.nonzero(alive & attached)[0]:
            if contacts is None or grid is None or R_fritz <= 0.0 or R[i] <= R_fritz:
                continue
            attached[i] = False
            d, kk = contacts.nearest(x[i], z[i])
            iz, ix = contacts.cell(kk)
            cx, cz = float(grid.x_um[ix]), float(grid.z_um[iz])
            dxv, dzv = x[i] - cx, z[i] - cz
            nrm = math.hypot(dxv, dzv) or 1.0
            target = R[i] + 0.75 * float(grid.dx_um)
            x[i] = (cx + dxv / nrm * target) % pitch
            z[i] = cz + dzv / nrm * target
            events.append({"t_s": t, "bubble": i, "kind": "detach",
                           "x_um": float(x[i]), "z_um": float(z[i]), "R_um": float(R[i])})

    return {
        "t_s": times,
        "x_um": X,
        "z_um": Z,
        "R_um": RR,
        "attached": ATT,
        "events": events,
        "fritz_radius_um": float(R_fritz),
        "reynolds": np.array([rv["reynolds"] for rv in rise0]),
        "re_gt_1": np.array([rv["re_gt_1"] for rv in rise0], dtype=bool),
        "warnings": warnings,
        "settings": {"dt_s": float(dt_s), "gravity_dir": tuple(gv.tolist()),
                     "supersat": float(supersat), "theta_deg": float(theta_deg),
                     "sigma": float(sigma), "slip_model": slip_model, "attach": bool(attach),
                     "n_bubbles": nb, "grid": grid is not None},
    }
