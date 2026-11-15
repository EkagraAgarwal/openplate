"""Bubble <-> plating-front coupling (plan item E2, coupling half).

``platesim.bubbles.physics`` / ``tracker`` (E2a) are the physics half: point bubbles in a *frozen*
flow, one-way only.  This module is the other half: the small set of fields that lets the level-set
feature engine (``platesim.feature.engine``) see the bubbles, and lets the bubbles see the engine.

Four entry points:

``attached_bubbles``
    The attached subset of a tracker result (or of a plain list of tuples), as :class:`Bubble`.
``shading_mask``
    Per-cell multiplicative factor in [0, 1] for the electrode reaction on the level-set grid:
    0 under the wetted contact disc of an attached bubble (radius ``R sin(theta)``), 1 elsewhere.
``h2_nucleation_sites``
    Seed bubbles where the H2 supersaturation of the side-reaction current exceeds a threshold,
    with a seeded RNG.
``step_bubbles``
    One engine step of growth / advection / adhesion / detachment for a persistent bubble list.

Conventions (inherited from ``platesim.feature.engine`` and ``platesim.bubbles``)
----------------------------------------------------------------------------------
* Level-set arrays are ``(nr, nz)`` cell-centred, ``psi`` is a signed distance in **cm**,
  ``psi > 0`` in the electrolyte and ``psi < 0`` in metal; the plating front is ``psi = 0``.
* Grid spacing ``dx`` is in **cm**; all bubble geometry (position, radius, footprint) is in **um**.
* Cell ``[i, j]`` is centred at ``origin_cm + (i, j) * dx``.  The default origin,
  ``(dx/2, dx/2)``, is the level-set convention ``x_i = (i + 0.5) dx`` for a grid that starts at 0;
  ``platesim.feature.engine.levelset.Grid`` starts at ``z_lo = -nbot*dx``, so pass
  ``origin_cm=(g.rc[0, 0], g.zc[0, 0])`` when wiring it in.
* ``periodic=True`` wraps the x direction (``trench_full``: pitch = ``nr * dx``).
* Currents are in A/cm2, concentrations in mol/cm3, potentials in V (platesim conventions).
* Nothing here is imported by an existing module: this file is standalone until the lead wires it
  into ``platesim/feature/engine/sim.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..core.constants import F
from . import physics

__all__ = [
    "Bubble",
    "attached_bubbles",
    "shading_mask",
    "h2_nucleation_sites",
    "step_bubbles",
]

_DEFAULT_THETA_DEG = 45.0


# --------------------------------------------------------------------------------------------- bubble
@dataclass
class Bubble:
    """One Lagrangian bubble in the (x, z) cross-section.

    Positions and radius in um, contact angle in degrees (the gas-side angle of the spherical cap,
    as in :func:`platesim.bubbles.physics.contact_disc_radius_um`).  Mutable on purpose:
    :func:`step_bubbles` advances a persistent list in place.
    """

    x_um: float
    z_um: float
    R_um: float
    attached: bool = False
    theta_deg: float = _DEFAULT_THETA_DEG
    alive: bool = True
    age_s: float = 0.0

    @property
    def contact_radius_um(self) -> float:
        """Radius of the wetted contact disc, ``R sin(theta)`` (um); 0 for a vanished bubble."""
        if self.R_um <= 0.0:
            return 0.0
        return float(physics.contact_disc_radius_um(self.R_um, self.theta_deg))

    def copy(self) -> "Bubble":
        return replace(self)


def _bubble_from_row(x, z, r, att, theta_deg) -> Bubble:
    return Bubble(float(x), float(z), float(r), bool(att), float(theta_deg))


def _from_tracker(res: dict, index: int = -1, theta_deg=None) -> list[Bubble]:
    """Rows of a ``track_bubbles`` result -> bubbles; ``index`` selects the recorded time (2-D arrays)."""
    X = np.asarray(res["x_um"], dtype=float)
    Z = np.asarray(res["z_um"], dtype=float)
    R = np.asarray(res["R_um"], dtype=float)
    A = np.asarray(res.get("attached", np.zeros_like(R)), dtype=bool)
    if theta_deg is None:
        theta_deg = res.get("settings", {}).get("theta_deg", _DEFAULT_THETA_DEG)
    if X.ndim == 2:
        k = int(index) % X.shape[0]
        X, Z, R, A = X[k], Z[k], R[k], A[k]
    return [_bubble_from_row(X[i], Z[i], R[i], A[i], theta_deg) for i in range(X.size)]


def _as_bubbles(source, index: int = -1, theta_deg=None) -> list[Bubble]:
    """Normalise the accepted bubble inputs to a list of :class:`Bubble`.

    Accepted forms:
      * ``None`` -> ``[]``
      * a ``track_bubbles`` result dict (uses ``x_um/z_um/R_um/attached`` and ``settings['theta_deg']``)
      * a :class:`Bubble` or an iterable of them
      * an iterable of ``(x_um, z_um, R_um, attached)`` or ``(x_um, z_um, R_um, attached, theta_deg)``
    """
    if source is None:
        return []
    if isinstance(source, Bubble):
        b = source.copy()
        if theta_deg is not None:
            b.theta_deg = float(theta_deg)
        return [b]
    if isinstance(source, dict):
        return _from_tracker(source, index, theta_deg)
    out: list[Bubble] = []
    for item in source:
        if isinstance(item, Bubble):
            b = item.copy()
            if theta_deg is not None:
                b.theta_deg = float(theta_deg)
            out.append(b)
            continue
        vals = list(item)
        if len(vals) >= 5:
            th = vals[4] if theta_deg is None else theta_deg
        elif len(vals) == 4:
            th = _DEFAULT_THETA_DEG if theta_deg is None else theta_deg
        else:
            raise ValueError("bubble rows must be (x_um, z_um, R_um, attached[, theta_deg])")
        out.append(_bubble_from_row(vals[0], vals[1], vals[2], vals[3], th))
    return out


def attached_bubbles(source, index: int = -1, theta_deg=None) -> list[Bubble]:
    """The bubbles currently attached to the front.

    ``source`` is a ``track_bubbles`` result (``index`` picks the recorded time, default the last
    frame) or a list of ``(x_um, z_um, R_um, attached, theta_deg)`` rows / :class:`Bubble` objects.
    Returns fresh copies, so mutating them cannot corrupt the tracker history.
    """
    return [b for b in _as_bubbles(source, index, theta_deg) if b.attached]


# --------------------------------------------------------------------------------------------- shading
def _psi_grad(psi: np.ndarray, dx: float, periodic: bool):
    """Central-difference gradient of psi, periodic x halo / edge pad in z (as levelset._pad)."""
    p = np.pad(psi, 1, mode="edge")
    if periodic:
        p[0, 1:-1] = psi[-1, :]
        p[-1, 1:-1] = psi[0, :]
    gr = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * dx)
    gz = (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * dx)
    return gr, gz


def shading_mask(psi, grid_dx_cm: float, bubbles, periodic: bool = True, *,
                 origin_cm: Sequence[float] | None = None, band_cells: float = 1.0,
                 soft_edge_um: float | None = None, silhouette_shade: float = 0.0,
                 index: int = -1) -> np.ndarray:
    """Per-cell multiplicative factor for the electrode reaction under attached bubbles.

    The returned ``(nr, nz)`` array multiplies the local reaction current (or the interface speed
    ``v = Mfac * i_s`` in ``sim.run``): 0 under an attached bubble's wetted contact disc, 1 elsewhere.

    Geometry: a sphere of radius ``R`` at contact angle ``theta`` hides a disc of radius
    ``r_c = R sin(theta)`` on the electrode [FT].  For every cell within ``band_cells * dx`` of the
    front the vector from the bubble centre is split into the local normal (from ``grad psi``) and
    tangent components; the cell is shaded when the tangential offset is below ``r_c`` and the cell
    lies in the bubble's normal span ``[-R, 0]`` from its centre.  This handles a
    flat floor, a sidewall and a moving (non-planar) front with one rule, and the footprint on a flat
    front is the expected ``2 R sin(theta)`` within one cell.

    Parameters
    ----------
    psi : (nr, nz) array
        Signed distance in cm, ``> 0`` electrolyte.
    grid_dx_cm : float
        Cell size in cm.
    bubbles : tracker result / list of :class:`Bubble` / list of tuples
        See :func:`_as_bubbles`.  Only ``attached`` bubbles shade.
    periodic : bool
        Wrap the tangential offset in x over the pitch ``nr * dx`` (default True, ``trench_full``).
    origin_cm : (x, z), optional
        Coordinate of the *centre* of cell ``[0, 0]`` in cm.  Defaults to ``(dx/2, dx/2)``.
    band_cells : float
        Only cells with ``|psi| <= band_cells * dx`` receive shading; the rest stay 1.  Default 1:
        the one-cell strip that carries the interface source in the diffuse-boundary solver.
    soft_edge_um : float, optional
        Width of the anti-aliased disc edge (um).  Default ``dx`` expressed in um, i.e. one cell;
        pass 0 for a hard step.
    silhouette_shade : float
        Optional partial shading by the bubble *silhouette* for the diffusive path: in the annulus
        ``r_c <= r_t < R`` the factor becomes ``1 - silhouette_shade`` (0 = off, the default).
        The cap shadow is wider than the contact disc, so a value of ~0.2-0.5 is the [J] order of
        magnitude for "the cap partially blocks the diffusion layer"; it is not a validated number.
    index : int
        Time index when ``bubbles`` is a tracker result (default -1, last frame).

    Returns
    -------
    (nr, nz) float array in [0, 1], exactly 1 where no attached bubble projects.
    """
    psi = np.asarray(psi, dtype=float)
    if psi.ndim != 2:
        raise ValueError("psi must be a 2-D (nr, nz) signed-distance array")
    dx = float(grid_dx_cm)
    if dx <= 0.0:
        raise ValueError("grid_dx_cm must be positive")
    nr, nz = psi.shape
    if origin_cm is None:
        origin_cm = (0.5 * dx, 0.5 * dx)
    bubs = [b for b in _as_bubbles(bubbles, index) if b.attached and b.R_um > 0.0]
    if not bubs:
        return np.ones((nr, nz), dtype=float)

    band = np.abs(psi) <= float(band_cells) * dx
    if not band.any():
        return np.ones((nr, nz), dtype=float)

    dx_um = dx * 1e4
    soft = dx_um if soft_edge_um is None else max(float(soft_edge_um), 0.0)
    pitch_um = float(nr) * dx_um if periodic else 0.0
    gr, gz = _psi_grad(psi, dx, periodic)
    gm = np.hypot(gr, gz)
    gm = np.where(gm > 0.0, gm, 1.0)
    ii, jj = np.nonzero(band)
    x_um = (float(origin_cm[0]) + ii * dx) * 1e4
    z_um = (float(origin_cm[1]) + jj * dx) * 1e4
    n_x, n_z = gr[ii, jj] / gm[ii, jj], gz[ii, jj] / gm[ii, jj]

    cov = np.zeros(x_um.shape, dtype=float)
    tol = (float(band_cells) + 0.5) * dx_um            # band cells lie this far from the wall

    def _disc(radius_um: float, v_t: np.ndarray) -> np.ndarray:
        if radius_um <= 0.0:
            return np.zeros_like(v_t)
        if soft > 0.0:
            return np.clip(0.5 + (radius_um - v_t) / soft, 0.0, 1.0)
        return (v_t <= radius_um).astype(float)

    for b in bubs:
        du = x_um - b.x_um
        if periodic:
            du = du - pitch_um * np.round(du / pitch_um)
        dz = z_um - b.z_um
        v_n = du * n_x + dz * n_z                      # normal offset (negative on the wall side)
        v_t = np.abs(du * n_z - dz * n_x)              # tangential offset, i.e. |v x n| in 2-D
        # the bubble spans v_n in [-R, 0] (the tracker freezes the centre at the sphere-touch distance
        # R, the cap's contact plane is at -R cos(theta)); a cell shades only if it lies inside that
        # span, not on a distant wall that merely happens to have a small tangential offset
        side = np.abs(v_n) <= b.R_um + tol
        c = _disc(b.contact_radius_um, v_t)
        if silhouette_shade > 0.0:
            c = np.maximum(c, float(silhouette_shade) * _disc(b.R_um, v_t))
        cov = np.maximum(cov, np.where(side, c, 0.0))

    out = np.ones((nr, nz), dtype=float)
    out[ii, jj] = 1.0 - np.clip(cov, 0.0, 1.0)
    return out


# ------------------------------------------------------------------------------------------- nucleation
def _as_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def h2_nucleation_sites(j_h2_field, supersat_threshold: float = 0.0, rng=None, *,
                        delta_cm: float = 65e-4, D_h2_cm2_s: float | None = None,
                        c_sat_mol_cm3: float | None = None, c_bulk_mol_cm3: float = 0.0,
                        rate_per_cell: float = 1.0, R0_um: float = 1.0,
                        theta_deg: float = _DEFAULT_THETA_DEG, dx_cm: float | None = None,
                        origin_cm: Sequence[float] | None = None, periodic: bool = True,
                        max_sites: int | None = None) -> dict:
    """Seed bubbles where the H2 side-reaction current super-saturates the wall layer.

    The steady 1-D Nernst layer around an H2-evolving wall gives [FT]

        c_s = c_bulk + j_H2 delta / (2 F D_H2)      (mol/cm3; j in A/cm2, delta in cm, D in cm2/s)

    and the supersaturation ``S - 1 = c_s / c_sat - 1`` follows from Henry's law at 1 atm
    (``c_sat = 0.78 mol/m3`` for H2 in water, [J] as in ``physics.henry_concentration_mol_m3``).

    A cell whose supersaturation exceeds ``supersat_threshold`` is a candidate; each candidate is
    seeded independently with probability ``p = 1 - exp(-rate_per_cell (S - 1 - threshold))``, drawn
    from ``rng``.  Same seed -> same sites.  Radii are drawn around ``R0_um`` (0.5-1.5 x, uniform):
    ``R0_um`` is the [J] critical-nucleus scale of the model, not a measured value.

    Parameters
    ----------
    j_h2_field : (nr, nz) array
        Local H2 (HER) current density, A/cm2.  Take it from
        :func:`platesim.efficiency.side_reactions.faradaic_efficiency` (``j_H2_A_cm2``) spread over
        the active interface cells, or from any other side-reaction model.
    supersat_threshold : float
        Dimensionless ``S - 1`` the wall layer must exceed.  Default 0 (saturation).
    rng : Generator | int | None
        Seeded RNG (``np.random.default_rng``); ``None`` draws fresh entropy.
    delta_cm, D_h2_cm2_s, c_sat_mol_cm3, c_bulk_mol_cm3
        Diffusion-layer thickness (cm), H2 diffusivity (cm2/s, default 5e-5), Henry solubility at
        1 atm (mol/cm3, default 7.8e-7) and the far-field dissolved H2 (mol/cm3).
    rate_per_cell : float
        Seeding-rate prefactor of the exponential probability above.  Large values approach
        "every super-threshold cell seeds".
    R0_um : float
        Mean seeded radius (um); radii are uniform in ``[0.5, 1.5] R0_um``.
    dx_cm, origin_cm, periodic
        Optional level-set geometry.  When ``dx_cm`` is given the seeded cell centres are also
        returned as ``x_um`` / ``z_um`` (um), ready for :class:`Bubble`.
    max_sites : int, optional
        Keep only the ``max_sites`` most supersaturated sites.

    Returns
    -------
    dict with
        ``i``, ``j`` (int arrays, seeded cells), ``supersat`` (at the seeded cells),
        ``c_s_mol_cm3``, ``R0_um``, ``x_um`` / ``z_um`` (float arrays, or ``None`` without
        ``dx_cm``), ``supersat_field`` (the full ``(nr, nz)`` field, to feed :func:`step_bubbles`),
        ``threshold``, ``c_sat_mol_cm3``, ``n_candidates``.
    """
    j = np.asarray(j_h2_field, dtype=float)
    if j.ndim != 2:
        raise ValueError("j_h2_field must be a 2-D (nr, nz) array")
    if delta_cm <= 0.0:
        raise ValueError("delta_cm must be positive")
    if float(rate_per_cell) < 0.0:
        raise ValueError("rate_per_cell must be non-negative")
    D = physics.D_H2_WATER_M2_S * 1e4 if D_h2_cm2_s is None else float(D_h2_cm2_s)
    if D <= 0.0:
        raise ValueError("D_h2_cm2_s must be positive")
    c_sat = (physics.K_H_H2_MOL_L_ATM * 1e-3 if c_sat_mol_cm3 is None
             else float(c_sat_mol_cm3))
    if c_sat <= 0.0:
        raise ValueError("c_sat_mol_cm3 must be positive")

    c_s = float(c_bulk_mol_cm3) + np.maximum(j, 0.0) * float(delta_cm) / (2.0 * F * D)
    supersat = c_s / c_sat - 1.0
    cand = supersat > float(supersat_threshold)
    ii, jj = np.nonzero(cand)
    s_c = supersat[ii, jj]
    n_cand = int(ii.size)

    gen = _as_rng(rng)
    if n_cand:
        p = 1.0 - np.exp(-float(rate_per_cell) * np.maximum(s_c - float(supersat_threshold), 0.0))
        keep = gen.random(n_cand) < np.clip(p, 0.0, 1.0)
        ii, jj, s_c = ii[keep], jj[keep], s_c[keep]
    if max_sites is not None and ii.size > int(max_sites):
        order = np.argsort(s_c)[::-1][: int(max_sites)]
        ii, jj, s_c = ii[order], jj[order], s_c[order]

    n = int(ii.size)
    if n and float(R0_um) > 0.0:
        R0 = float(R0_um) * (0.5 + gen.random(n))
    else:
        R0 = np.full(n, float(R0_um))

    if dx_cm is not None and n:
        dx = float(dx_cm)
        ox, oz = (0.5 * dx, 0.5 * dx) if origin_cm is None else (float(origin_cm[0]),
                                                                float(origin_cm[1]))
        x_um = (ox + ii * dx) * 1e4
        z_um = (oz + jj * dx) * 1e4
    else:
        x_um = None
        z_um = None

    return {
        "i": ii,
        "j": jj,
        "supersat": s_c,
        "c_s_mol_cm3": c_s[ii, jj],
        "R0_um": R0,
        "theta_deg": float(theta_deg),
        "x_um": x_um,
        "z_um": z_um,
        "supersat_field": supersat,
        "threshold": float(supersat_threshold),
        "c_sat_mol_cm3": c_sat,
        "n_candidates": n_cand,
    }


# ------------------------------------------------------------------------------------------ time step
def _bilinear_grid(field, x_um, z_um, grid):
    """Bilinear interpolation of a cell-centred ``FlowField`` array; x periodic, z clamped.

    Same rule as ``platesim.bubbles.tracker``: cell (iz, ix) is centred at
    ``(grid.x_um[ix], grid.z_um[iz])`` with spacing ``grid.dx_um`` in both directions.
    """
    dx = float(grid.dx_um)
    nx, nz = grid.x_um.size, grid.z_um.size
    fx = (float(x_um) - grid.x_um[0]) / dx
    fz = (float(z_um) - grid.z_um[0]) / dx
    i0 = int(math.floor(fx))
    t = fx - i0
    j0 = int(np.clip(math.floor(fz), 0, nz - 1))
    tz = float(np.clip(fz - j0, 0.0, 1.0))
    j1 = min(j0 + 1, nz - 1)
    ia, ib = i0 % nx, (i0 + 1) % nx
    return float((1.0 - tz) * ((1.0 - t) * field[j0, ia] + t * field[j0, ib])
                 + tz * ((1.0 - t) * field[j1, ia] + t * field[j1, ib]))


def _bilinear_ls(field, x_um, z_um, dx, origin_cm, periodic):
    """Bilinear sample of an ``(nr, nz)`` level-set field; x periodic, z clamped.  cm -> um."""
    field = np.asarray(field, dtype=float)
    nr, nz = field.shape
    dx_um = float(dx) * 1e4
    fx = (float(x_um) - float(origin_cm[0]) * 1e4) / dx_um
    fz = (float(z_um) - float(origin_cm[1]) * 1e4) / dx_um
    i0 = int(math.floor(fx))
    t = fx - i0
    j0 = int(np.clip(math.floor(fz), 0, nz - 1))
    tz = float(np.clip(fz - j0, 0.0, 1.0))
    j1 = min(j0 + 1, nz - 1)
    if periodic:
        ia, ib = i0 % nr, (i0 + 1) % nr
    else:
        ia, ib = int(np.clip(i0, 0, nr - 1)), int(np.clip(i0 + 1, 0, nr - 1))
    return float((1.0 - tz) * ((1.0 - t) * field[ia, j0] + t * field[ib, j0])
                 + tz * ((1.0 - t) * field[ia, j1] + t * field[ib, j1]))


def step_bubbles(bubbles, dt: float, flow=None, supersat_field=0.0, *,
                 gravity_dir: Sequence[float] = (0.0, -1.0), slip_model: str = "hadamard",
                 psi=None, dx_cm: float | None = None, origin_cm: Sequence[float] | None = None,
                 periodic: bool = True, attach: bool = True,
                 sigma: float = physics.SIGMA_WATER_N_M, drho: float = physics.DRHO_KG_M3,
                 mu: float = physics.MU_LIQUID_PA_S, rho_l: float = physics.RHO_LIQUID_KG_M3,
                 D_m2_s: float = physics.D_H2_WATER_M2_S, k_H: float = physics.K_H_H2_MOL_L_ATM,
                 p_atm_Pa: float = physics.P_ATM_PA, T_C: float = 25.0,
                 r_min_um: float = 0.05) -> dict:
    """Advance a persistent bubble list by one engine step (thin wrapper over ``physics``).

    Per step and per live bubble:

    1. **Growth** - ``dR/dt`` from :func:`platesim.bubbles.physics.diffusive_growth_rate_um_s` with
       the local supersaturation (constant, callable ``f(x_um, z_um)``, or an ``(nr, nz)`` field
       bilinearly sampled).  A bubble shrinking to ``r_min_um`` dissolves (``alive=False``).
    2. **Motion** (free bubbles only) - liquid velocity plus the terminal slip along
       ``-gravity_dir`` from :func:`platesim.bubbles.physics.rise_velocity`, exactly as
       ``track_bubbles`` does.  ``flow`` may be ``None`` (quiescent), a ``trenchflow`` FlowField
       (``ux/uz`` + ``.grid``), or a ``(ux, uz)`` pair of ``(nr, nz)`` arrays on the level-set grid
       (then ``dx_cm`` is required).  The level-set grid wraps x when ``periodic``.
    3. **Adhesion** - only when ``psi`` is given: a free bubble whose centre is within
       ``R + dx/2`` of the front sticks (``psi`` is the signed distance, so this is exactly the
       one-cell-resolution surrogate the tracker uses).
    4. **Detachment** - ``R > R_fritz(theta)`` from
       :func:`platesim.bubbles.physics.fritz_detachment_radius_um`.  When ``psi`` is given the
       detached bubble is pushed ``R + 0.75 dx`` off the wall along the local normal.

    ``bubbles`` is updated in place.  Returns
    ``{"bubbles", "events", "dt_s", "supersat"}``; events mirror ``track_bubbles``:
    ``kind`` in ``attach/detach/dissolved`` plus the position and radius.
    """
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    dt = float(dt)
    blist = bubbles if isinstance(bubbles, list) else list(bubbles)
    for b in blist:
        if not isinstance(b, Bubble):
            raise TypeError("step_bubbles works on a list of Bubble objects")
    if slip_model not in ("hadamard", "stokes"):
        raise ValueError("slip_model must be 'hadamard' or 'stokes'")
    if psi is not None and dx_cm is None:
        raise ValueError("psi requires dx_cm (and optionally origin_cm) to know the grid geometry")
    dx_v = None if dx_cm is None else float(dx_cm)
    if dx_v is not None and origin_cm is None:
        origin_cm = (0.5 * dx_v, 0.5 * dx_v)
    org = ((float(origin_cm[0]), float(origin_cm[1])) if origin_cm is not None else (0.0, 0.0))

    gv = np.asarray(gravity_dir, dtype=float).ravel()
    if gv.size != 2:
        raise ValueError("gravity_dir must be a 2-vector (x, z)")
    gn = float(np.hypot(gv[0], gv[1]))
    slip_dir = -gv / gn if gn > 0.0 else np.zeros(2)

    # ---- flow sampler
    grid = getattr(flow, "grid", None)
    array_flow: tuple[np.ndarray, np.ndarray] | None = None
    if flow is not None and grid is None:
        try:
            ux, uz = flow
        except (TypeError, ValueError):
            raise TypeError("flow must be None, a FlowField, or a (ux, uz) pair of (nr, nz) arrays")
        if dx_v is None:
            raise ValueError("array flow requires dx_cm")
        array_flow = (np.asarray(ux, dtype=float), np.asarray(uz, dtype=float))
    psi_arr = None if psi is None else np.asarray(psi, dtype=float)
    if psi_arr is not None:
        if psi_arr.ndim != 2:
            raise ValueError("psi must be a 2-D (nr, nz) array")

    def sample_supersat(x, z):
        if callable(supersat_field):
            return float(supersat_field(x, z))
        if np.ndim(supersat_field) == 0:
            return float(supersat_field)
        if dx_v is None:
            raise ValueError("an array supersat_field requires dx_cm")
        return _bilinear_ls(supersat_field, x, z, dx_v, org, periodic)

    def sample_psi(x, z):
        if dx_v is None or psi_arr is None:
            raise ValueError("psi sampling requires psi and dx_cm")
        return _bilinear_ls(psi_arr, x, z, dx_v, org, periodic)

    def slip_cm_s(r_um):
        rv = physics.rise_velocity(r_um, drho=drho, mu=mu, rho_l=rho_l)
        return (rv["U_hr_m_s"] if slip_model == "hadamard" else rv["U_stokes_m_s"]) * 1e2

    events: list[dict] = []
    supersat_used = np.zeros(len(blist))
    for k, b in enumerate(blist):
        if not b.alive:
            continue
        s = sample_supersat(b.x_um, b.z_um)
        supersat_used[k] = s

        # 1) growth
        dR = physics.diffusive_growth_rate_um_s(
            b.R_um, supersat=s, D_m2_s=D_m2_s, k_H=k_H, p_atm_Pa=p_atm_Pa, sigma=sigma,
            T_C=T_C, t_s=b.age_s + dt)
        b.R_um += dR * dt
        b.age_s += dt
        if b.R_um <= r_min_um:
            b.R_um = r_min_um
            b.alive = False
            b.attached = False
            events.append({"kind": "dissolved", "bubble": k, "x_um": b.x_um, "z_um": b.z_um,
                           "R_um": b.R_um})
            continue

        # 2) motion of a free bubble
        if not b.attached:
            if flow is None:
                vx = vz = 0.0
            elif grid is not None:
                vx = _bilinear_grid(flow.ux, b.x_um, b.z_um, grid)
                vz = _bilinear_grid(flow.uz, b.x_um, b.z_um, grid)
            elif array_flow is not None and dx_v is not None:
                vx = _bilinear_ls(array_flow[0], b.x_um, b.z_um, dx_v, org, periodic)
                vz = _bilinear_ls(array_flow[1], b.x_um, b.z_um, dx_v, org, periodic)
            else:
                vx = vz = 0.0
            u = slip_cm_s(b.R_um)
            b.x_um += (vx + slip_dir[0] * u) * dt * 1e4
            b.z_um += (vz + slip_dir[1] * u) * dt * 1e4
            if grid is not None:
                pitch = float(grid.pitch_um)
                b.x_um %= pitch
                b.z_um = float(np.clip(b.z_um, grid.z_um[0], grid.z_um[-1]))
            elif dx_v is not None and psi_arr is not None:
                dxu = dx_v * 1e4
                z_lo = org[1] * 1e4
                b.z_um = float(np.clip(b.z_um, z_lo, z_lo + (psi_arr.shape[1] - 1) * dxu))
                if periodic:
                    b.x_um %= psi_arr.shape[0] * dxu

        # 3) adhesion (front proximity from psi, one-cell tolerance)
        if attach and not b.attached and psi_arr is not None and dx_v is not None:
            if sample_psi(b.x_um, b.z_um) <= b.R_um + 0.5 * dx_v * 1e4:
                b.attached = True
                events.append({"kind": "attach", "bubble": k, "x_um": b.x_um, "z_um": b.z_um,
                               "R_um": b.R_um})

        # 4) detachment at the Fritz radius
        if b.attached:
            r_fritz = physics.fritz_detachment_radius_um(b.theta_deg, sigma, drho) if sigma > 0.0 else 0.0
            if r_fritz > 0.0 and b.R_um > r_fritz:
                b.attached = False
                if psi_arr is not None and dx_v is not None:
                    dxu = dx_v * 1e4
                    i = int(np.clip(round((b.x_um - org[0] * 1e4) / dxu), 0, psi_arr.shape[0] - 1))
                    j = int(np.clip(round((b.z_um - org[1] * 1e4) / dxu), 0, psi_arr.shape[1] - 1))
                    gr, gz = _psi_grad(psi_arr, dx_v, bool(periodic))
                    gm = math.hypot(gr[i, j], gz[i, j]) or 1.0
                    nrm = (gr[i, j] / gm, gz[i, j] / gm)
                    target = b.R_um + 0.75 * dxu
                    push = target - sample_psi(b.x_um, b.z_um)
                    b.x_um += nrm[0] * push
                    b.z_um += nrm[1] * push
                events.append({"kind": "detach", "bubble": k, "x_um": b.x_um, "z_um": b.z_um,
                               "R_um": b.R_um})

    return {"bubbles": blist, "events": events, "dt_s": dt, "supersat": supersat_used}
