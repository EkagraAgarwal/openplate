"""Thermal estimate for a plating run: Joule heating and temperature rise (plan v2 D7).

Two models, both cheap enough to run once per plating run:

1. ``LumpedThermal`` -- one well-mixed bath temperature ("lumped beaker"):

       C dT/dt = Q - UA (T - T_inf),      Q = I V_cell = I^2 R

   All electrical work done in the cell ends up as heat (ohmic + activation; the
   reversible part stored in the deposit is < 10 % of V_cell for an acid copper
   bath [J]).  Loss to ambient is Newtonian, ``UA = h A`` over the beaker surface.
   ``T(t)`` is available in closed form and as a fixed-step RK4 integration; the
   steady rise is ``dT_inf = Q/UA`` and the time constant ``tau = C/UA``.

2. ``solve_heat_2d`` -- steady 2-D heat conduction on a rectangular cell
   cross-section (the gap between the electrodes), finite volume on a uniform
   grid with scipy.sparse, volumetric Joule source ``q = |j|^2 / kappa`` from a
   given current-density field (scalar or ``(ny, nx)`` array), Dirichlet / Robin /
   insulated walls.  Returns the field together with max / mean temperature rise
   and the discrete energy balance (source vs boundary flux).

``temperature_rise_report`` bundles the lumped answer for one run with the
temperature-induced change of the bath transport/kinetic properties (``D_Cu``,
``kappa``, ``i0``) obtained from the existing ``platesim.core.bath.Bath``
T-dependence -- nothing in this module re-implements those formulas.

Units follow the repository convention: cm, s, A, V, K, mol; power in W, energy
in J.  ``_L``, ``_m2K``, ``_cm3``, ``_h`` suffixes mark anything else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..core.bath import Bath

# ---------------------------------------------------------------- material data
# Water at 20-25 C.  The acid copper electrolyte is within ~1 % of water for all
# three numbers (65 g/L CuSO4 + 8 g/L H2SO4), so water values are used [J].
WATER_RHO_G_CM3 = 0.9982      # [FT] NIST Chemistry WebBook SRD 69, water at 20 C (1.000 at 4 C)
WATER_CP_J_G_K = 4.1813       # [FT] NIST Chemistry WebBook SRD 69, cp of liquid water at 20 C
WATER_K_W_CM_K = 6.00e-3      # [FT] NIST: 0.598 W/(m K) at 20 C, 0.606 at 25 C -> 6.0e-3 W/(cm K)

# Natural convection from a vertical glass wall to still air: 5-10 W/(m2 K)
# (Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, ch. 9 order of
# magnitude).  The chosen value is a lower bound for a real beaker because
# evaporation from the free surface and conduction into the bench add a factor
# 2-5 -- so the lumped dT below is an UPPER bound [J].
H_AIR_NATURAL_W_M2K = 10.0    # [J] upper-bound-ish convective coefficient, W/(m2 K)

W_M2K_TO_W_CM2K = 1.0e-4      # 1 W/(m2 K) = 1e-4 W/(cm2 K)


# ---------------------------------------------------------------- helpers

def beaker_fill_height_cm(volume_L: float, beaker_id_cm: float = 9.0) -> float:
    """Liquid height of ``volume_L`` litres in a cylinder of inner diameter ``beaker_id_cm``."""
    if volume_L <= 0.0:
        raise ValueError(f"volume_L must be > 0, got {volume_L!r}")
    if beaker_id_cm <= 0.0:
        raise ValueError(f"beaker_id_cm must be > 0, got {beaker_id_cm!r}")
    floor_cm2 = math.pi * (beaker_id_cm / 2.0) ** 2
    return volume_L * 1000.0 / floor_cm2


def beaker_surface_area_cm2(volume_L: float, beaker_id_cm: float = 9.0) -> float:
    """Heat-loss area of a beaker: wetted side wall + bottom + free surface.

    The bottom conducts into the bench and the free surface loses heat by
    evaporation; both are lumped into the same Newtonian coefficient [J].
    """
    floor_cm2 = math.pi * (beaker_id_cm / 2.0) ** 2
    side_cm2 = math.pi * beaker_id_cm * beaker_fill_height_cm(volume_L, beaker_id_cm)
    return side_cm2 + 2.0 * floor_cm2


def ohmic_resistance_ohm(kappa_S_cm: float, gap_cm: float, area_cm2: float) -> float:
    """Electrolyte resistance between two parallel electrodes, R = gap / (kappa A)."""
    if kappa_S_cm <= 0.0:
        raise ValueError(f"kappa_S_cm must be > 0, got {kappa_S_cm!r}")
    if gap_cm <= 0.0 or area_cm2 <= 0.0:
        raise ValueError(f"gap_cm and area_cm2 must be > 0, got {gap_cm!r}, {area_cm2!r}")
    return gap_cm / (kappa_S_cm * area_cm2)


def volumetric_source(j_A_cm2, kappa_S_cm: float) -> float | np.ndarray:
    """Joule source density q = |j|^2 / kappa in W/cm3 (j in A/cm2, kappa in S/cm)."""
    if kappa_S_cm <= 0.0:
        raise ValueError(f"kappa_S_cm must be > 0, got {kappa_S_cm!r}")
    j = np.asarray(j_A_cm2, dtype=float)
    q = j * j / float(kappa_S_cm)
    return float(q) if q.ndim == 0 else q


# ---------------------------------------------------------------- lumped model

@dataclass
class LumpedThermal:
    """One-well-mixed-volume heat balance ``C dT/dt = Q - UA (T - T_inf)``.

    Attributes (SI-adjacent, cm-based): ``Q_W`` heating power, ``UA_W_K`` total
    loss conductance, ``C_J_K`` total heat capacity (bath + beaker), ``T0_C``
    initial bath temperature, ``T_inf_C`` ambient temperature.
    """

    Q_W: float
    UA_W_K: float
    C_J_K: float
    T0_C: float
    T_inf_C: float

    def __post_init__(self) -> None:
        if self.UA_W_K <= 0.0:
            raise ValueError(f"UA_W_K must be > 0, got {self.UA_W_K!r}")
        if self.C_J_K <= 0.0:
            raise ValueError(f"C_J_K must be > 0, got {self.C_J_K!r}")

    # ---- lumped scalars
    @property
    def tau_s(self) -> float:
        """Thermal time constant C/UA (s)."""
        return self.C_J_K / self.UA_W_K

    @property
    def dT_steady_K(self) -> float:
        """Steady temperature rise Q/UA (K)."""
        return self.Q_W / self.UA_W_K

    @property
    def T_steady_C(self) -> float:
        return self.T_inf_C + self.dT_steady_K

    # ---- closed form
    def T_analytic(self, t_s) -> float | np.ndarray:
        """Exact solution T(t) = T_inf + dT_inf + (T0 - T_inf - dT_inf) exp(-t/tau)."""
        t = np.asarray(t_s, dtype=float)
        d_inf = self.dT_steady_K
        d0 = self.T0_C - self.T_inf_C
        T = self.T_inf_C + d_inf + (d0 - d_inf) * np.exp(-t / self.tau_s)
        return float(T) if T.ndim == 0 else T

    def dT_at(self, t_s) -> float:
        """Analytic temperature rise above ambient at time ``t_s`` (K)."""
        return float(self.T_analytic(t_s)) - self.T_inf_C

    # ---- numerical integration (used to cross-check the closed form)
    def T_history(self, t_end_s: float, n_steps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Classical RK4 integration of the ODE; returns ``(t_s, T_C)`` arrays."""
        t_end_s = float(t_end_s)
        n_steps = int(n_steps)
        if t_end_s <= 0.0 or n_steps < 1:
            raise ValueError(f"t_end_s and n_steps must be positive, got {t_end_s!r}, {n_steps!r}")
        t = np.linspace(0.0, t_end_s, n_steps + 1)
        T = np.empty_like(t)
        T[0] = self.T0_C
        dt = t_end_s / n_steps
        Q, UA, C, Ti = self.Q_W, self.UA_W_K, self.C_J_K, self.T_inf_C

        def f(TT: float) -> float:
            return (Q - UA * (TT - Ti)) / C

        for k in range(n_steps):
            k1 = f(T[k])
            k2 = f(T[k] + 0.5 * dt * k1)
            k3 = f(T[k] + 0.5 * dt * k2)
            k4 = f(T[k] + dt * k3)
            T[k + 1] = T[k] + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return t, T


def lumped_parameters(
    I_A: float,
    V_cell: float | None = None,
    volume_L: float = 1.0,
    *,
    R_ohm: float | None = None,
    h_W_m2K: float = H_AIR_NATURAL_W_M2K,
    area_cm2: float | None = None,
    beaker_id_cm: float = 9.0,
    extra_heat_capacity_J_K: float = 0.0,
    C_J_K: float | None = None,
    T0_C: float = 22.0,
    T_inf_C: float | None = None,
) -> LumpedThermal:
    """Build the lumped parameters of one plating run.

    Heating power is ``I_A * V_cell`` when ``V_cell`` is given, otherwise
    ``I_A**2 * R_ohm``.  Loss area defaults to the beaker wall + bottom + free
    surface of ``volume_L`` litres in a ``beaker_id_cm`` beaker; the heat
    capacity defaults to the bath water plus ``extra_heat_capacity_J_K`` (glass
    beaker, typically 50-150 J/K [J]).  ``T_inf_C`` defaults to ``T0_C``.
    """
    if I_A < 0.0:
        raise ValueError(f"I_A must be >= 0, got {I_A!r}")
    if V_cell is not None:
        Q_W = abs(float(I_A) * float(V_cell))
    elif R_ohm is not None:
        Q_W = float(I_A) ** 2 * float(R_ohm)
    else:
        raise ValueError("either V_cell or R_ohm must be given")

    if area_cm2 is None:
        area_cm2 = beaker_surface_area_cm2(volume_L, beaker_id_cm)
    UA_W_K = float(h_W_m2K) * W_M2K_TO_W_CM2K * float(area_cm2)

    if C_J_K is None:
        C_J_K = (WATER_RHO_G_CM3 * WATER_CP_J_G_K * volume_L * 1000.0
                 + float(extra_heat_capacity_J_K))

    return LumpedThermal(Q_W=Q_W, UA_W_K=UA_W_K, C_J_K=float(C_J_K),
                         T0_C=float(T0_C),
                         T_inf_C=float(T0_C if T_inf_C is None else T_inf_C))


# ---------------------------------------------------------------- 2-D steady FV

_WALL_KINDS = ("dirichlet", "insulated", "robin")
_WALLS = ("left", "right", "bottom", "top")


def _wall_spec(spec, h_default_W_m2K: float) -> tuple[str, float | None]:
    """Normalise a wall specification to ``(kind, h_W_cm2K)``."""
    if isinstance(spec, str):
        kind = spec.lower()
        if kind not in _WALL_KINDS:
            raise ValueError(f"wall spec must be one of {_WALL_KINDS}, got {spec!r}")
        if kind == "robin":
            return "robin", float(h_default_W_m2K) * W_M2K_TO_W_CM2K
        return kind, None
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(f"wall spec tuple must be (kind, h_W_m2K), got {spec!r}")
        kind = str(spec[0]).lower()
        if kind != "robin":
            raise ValueError(f"only ('robin', h) tuples are accepted, got {spec!r}")
        return "robin", float(spec[1]) * W_M2K_TO_W_CM2K
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return "robin", float(spec) * W_M2K_TO_W_CM2K
    raise TypeError(f"unsupported wall spec {spec!r}")


def _normalise_walls(walls: dict | None, h_default_W_m2K: float) -> dict:
    out: dict[str, tuple[str, float | None]] = {k: ("dirichlet", None) for k in _WALLS}
    for key, spec in (walls or {}).items():
        if key not in _WALLS:
            raise ValueError(f"unknown wall {key!r}; expected a subset of {_WALLS}")
        kind, h_cm2K = _wall_spec(spec, h_default_W_m2K)
        if kind == "robin":
            h = 0.0 if h_cm2K is None else float(h_cm2K)
            if h <= 0.0:
                kind, h_cm2K = "insulated", None
            elif math.isinf(h):
                kind, h_cm2K = "dirichlet", None
            else:
                h_cm2K = h
        out[key] = (kind, h_cm2K)
    return out


@dataclass
class HeatField2D:
    """Steady 2-D temperature field on a rectangular cell section.

    ``T_C``/``dT_K`` are ``(ny, nx)`` cell-centred arrays; ``x_cm``/``y_cm`` are
    the cell centres.  ``source_W`` and ``boundary_flux_W`` are per cm of depth
    (the section is 2-D), and both ``energy_balance_rel`` (source vs boundary
    flux) and ``discrete_residual_rel`` (per-cell finite-volume residual) are
    reported as relative numbers.
    """

    x_cm: np.ndarray
    y_cm: np.ndarray
    T_C: np.ndarray
    dT_K: np.ndarray
    q_W_cm3: np.ndarray
    T_ref_C: float
    T_wall_C: float
    T_inf_C: float
    k_W_cmK: float
    Lx_cm: float
    Ly_cm: float
    source_W: float
    boundary_flux_W: float
    energy_balance_rel: float
    discrete_residual_rel: float
    walls: dict

    @property
    def dT_max_K(self) -> float:
        return float(np.max(self.dT_K))

    @property
    def dT_min_K(self) -> float:
        return float(np.min(self.dT_K))

    @property
    def dT_mean_K(self) -> float:
        return float(np.mean(self.dT_K))

    def to_dict(self) -> dict:
        return {
            "Lx_cm": float(self.Lx_cm),
            "Ly_cm": float(self.Ly_cm),
            "k_W_cmK": float(self.k_W_cmK),
            "nx": int(self.T_C.shape[1]),
            "ny": int(self.T_C.shape[0]),
            "T_wall_C": float(self.T_wall_C),
            "T_inf_C": float(self.T_inf_C),
            "dT_max_K": self.dT_max_K,
            "dT_mean_K": self.dT_mean_K,
            "dT_min_K": self.dT_min_K,
            "source_W_per_cm": float(self.source_W),
            "boundary_flux_W_per_cm": float(self.boundary_flux_W),
            "energy_balance_rel": float(self.energy_balance_rel),
            "discrete_residual_rel": float(self.discrete_residual_rel),
            "walls": {k: v[0] for k, v in self.walls.items()},
        }


def solve_heat_2d(
    q_W_cm3,
    Lx_cm: float,
    Ly_cm: float,
    *,
    nx: int = 40,
    ny: int = 40,
    k_W_cmK: float = WATER_K_W_CM_K,
    T_wall_C: float = 22.0,
    T_inf_C: float | None = None,
    walls: dict | None = None,
    h_W_m2K: float = H_AIR_NATURAL_W_M2K,
) -> HeatField2D:
    """Steady 2-D heat conduction on a rectangular cell section (finite volume).

    Parameters
    ----------
    q_W_cm3
        Volumetric Joule source, W/cm3: a scalar (uniform) or an ``(ny, nx)``
        array co-located with the cell centres (``volumetric_source`` builds it
        from a current-density field).
    Lx_cm, Ly_cm, nx, ny
        Section size and uniform cell counts (cells, not nodes).
    k_W_cmK
        Electrolyte thermal conductivity, W/(cm K) (uniform).
    T_wall_C
        Dirichlet wall temperature (the bath bulk temperature).
    walls
        Optional ``{"left"|"right"|"bottom"|"top": spec}``; a missing wall is
        ``"dirichlet"``.  ``spec`` is ``"dirichlet"``, ``"insulated"``,
        ``"robin"`` (uses ``h_W_m2K``) or a float/tuple giving an explicit
        ``h`` in W/(m2 K).  Robin walls relax to ``T_inf_C``.
    h_W_m2K
        Convective coefficient for ``"robin"`` walls, W/(m2 K).

    Returns
    -------
    HeatField2D

    The linear system is the 5-point finite-volume stencil on a uniform grid;
    the face conductance is ``k * face_length / cell_span`` and a Dirichlet/Robin
    wall adds the half-cell conduction resistance in series with the wall
    resistance, so the discrete energy balance closes to round-off.
    """
    Lx_cm = float(Lx_cm)
    Ly_cm = float(Ly_cm)
    if Lx_cm <= 0.0 or Ly_cm <= 0.0:
        raise ValueError(f"Lx_cm and Ly_cm must be > 0, got {Lx_cm!r}, {Ly_cm!r}")
    nx, ny = int(nx), int(ny)
    if nx < 2 or ny < 2:
        raise ValueError(f"nx and ny must be >= 2, got {nx}, {ny}")
    k = float(k_W_cmK)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"k_W_cmK must be finite and > 0, got {k_W_cmK!r}")
    if T_inf_C is None:
        T_inf_C = float(T_wall_C)

    spec = _normalise_walls(walls, h_W_m2K)

    q = np.asarray(q_W_cm3, dtype=float)
    if q.ndim == 0:
        q = np.full((ny, nx), float(q))
    elif q.shape != (ny, nx):
        raise ValueError(f"q_W_cm3 must be scalar or shape {(ny, nx)}, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_W_cm3 contains non-finite values")

    dx, dy = Lx_cm / nx, Ly_cm / ny
    cell_cm2 = dx * dy

    # ---- face list: (owner, neighbour | -1 for boundary, G [W/K per cm depth],
    #                  wall temperature used by a boundary face | NaN for interior)
    faces: list[tuple[int, int, float, float]] = []
    Gx_int = k * dy / dx            # interior x-face
    Gy_int = k * dx / dy            # interior y-face
    for iy in range(ny):
        for ix in range(nx):
            P = iy * nx + ix
            # x faces
            for step, wall in ((-1, "left"), (+1, "right")):
                nb = ix + step
                if 0 <= nb < nx:
                    faces.append((P, iy * nx + nb, Gx_int, math.nan))
                    continue
                kind, h_cm2K = spec[wall]
                if kind == "insulated":
                    continue
                G = 2.0 * Gx_int      # half cell to the wall plane
                if kind == "robin":
                    G = 1.0 / (1.0 / G + 1.0 / (float(h_cm2K or 0.0) * dy))
                    faces.append((P, -1, G, float(T_inf_C)))
                else:
                    faces.append((P, -1, G, float(T_wall_C)))
            # y faces
            for step, wall in ((-1, "bottom"), (+1, "top")):
                nb = iy + step
                if 0 <= nb < ny:
                    faces.append((P, nb * nx + ix, Gy_int, math.nan))
                    continue
                kind, h_cm2K = spec[wall]
                if kind == "insulated":
                    continue
                G = 2.0 * Gy_int
                if kind == "robin":
                    G = 1.0 / (1.0 / G + 1.0 / (float(h_cm2K or 0.0) * dx))
                    faces.append((P, -1, G, float(T_inf_C)))
                else:
                    faces.append((P, -1, G, float(T_wall_C)))

    if not any(nb < 0 for _, nb, _, _ in faces):
        raise ValueError("all walls insulated: the steady problem has no solution for q != 0")

    n = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.zeros(n)
    rhs = (q * cell_cm2).reshape(-1).astype(float).copy()
    for P, nb, G, T_bc in faces:
        diag[P] += G
        if nb < 0:
            rhs[P] += G * T_bc
        else:
            rows.append(P)
            cols.append(nb)
            vals.append(-G)
    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(diag.tolist())
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    Tflat = spla.spsolve(A, rhs)
    T = np.asarray(Tflat, dtype=float).reshape(ny, nx)

    # ---- energy balance + per-cell residual, recomputed from the face list
    residual = (q * cell_cm2).reshape(-1).astype(float).copy()
    flux_out = 0.0
    for P, nb, G, T_bc in faces:
        if nb < 0:
            f = G * (Tflat[P] - T_bc)
            flux_out += f
        else:
            f = G * (Tflat[P] - Tflat[nb])
        residual[P] -= f
    source_W = float(np.sum(q) * cell_cm2)
    scale = max(abs(source_W), 1.0e-300)
    max_res = float(np.max(np.abs(residual)))
    return HeatField2D(
        x_cm=(np.arange(nx) + 0.5) * dx,
        y_cm=(np.arange(ny) + 0.5) * dy,
        T_C=T,
        dT_K=T - float(T_wall_C),
        q_W_cm3=q,
        T_ref_C=float(T_wall_C),
        T_wall_C=float(T_wall_C),
        T_inf_C=float(T_inf_C),
        k_W_cmK=k,
        Lx_cm=Lx_cm,
        Ly_cm=Ly_cm,
        source_W=source_W,
        boundary_flux_W=float(flux_out),
        energy_balance_rel=abs(source_W - flux_out) / scale,
        discrete_residual_rel=max_res / scale,
        walls=spec,
    )


def cell_section_heat(
    j_A_cm2,
    kappa_S_cm: float,
    Lx_cm: float,
    Ly_cm: float,
    *,
    nx: int = 40,
    ny: int = 40,
    k_W_cmK: float = WATER_K_W_CM_K,
    T_bulk_C: float = 22.0,
    walls: dict | None = None,
    h_W_m2K: float = H_AIR_NATURAL_W_M2K,
) -> HeatField2D:
    """Convenience wrapper: current density -> q = |j|^2/kappa -> 2-D temperature.

    ``j_A_cm2`` may be a scalar, an ``(ny, nx)`` array, or a 1-D profile of
    length ``nx`` (current density varying only across the gap, tiled over y).
    """
    j = np.asarray(j_A_cm2, dtype=float)
    if j.ndim == 1:
        if j.size != nx:
            raise ValueError(f"1-D j_A_cm2 must have length nx={nx}, got {j.size}")
        j = np.tile(j, (ny, 1))
    return solve_heat_2d(volumetric_source(j, kappa_S_cm), Lx_cm, Ly_cm,
                         nx=nx, ny=ny, k_W_cmK=k_W_cmK,
                         T_wall_C=T_bulk_C, walls=walls, h_W_m2K=h_W_m2K)


# ---------------------------------------------------------------- bath response

def bath_property_shift(T_C: float, dT_K: float = 0.0, bath: Bath | None = None) -> dict:
    """Re-evaluate D_Cu, kappa and i0 at ``T_C + dT_K`` with the existing Bath laws.

    Uses ``Bath.at()`` so the T-dependence is the one already used by the solver
    (Stokes-Einstein for D_Cu, linear kappa_per_K for kappa, Arrhenius Ea_i0 for
    i0).  Returns absolute values and relative changes in percent, ready to be
    recorded in a run log ("dT = x K, kappa +y %").
    """
    b0 = (bath or Bath()).at(float(T_C))
    b1 = b0.at(float(T_C) + float(dT_K))

    def rel(a: float, b: float) -> float:
        return 100.0 * (b / a - 1.0) if a != 0.0 else 0.0

    return {
        "T0_C": float(b0.T_C),
        "T1_C": float(b1.T_C),
        "dT_K": float(dT_K),
        "D_cu_cm2_s": float(b0.D_cu),
        "D_cu_cm2_s_new": float(b1.D_cu),
        "D_cu_rel_pct": rel(b0.D_cu, b1.D_cu),
        "kappa_S_cm": float(b0.kappa),
        "kappa_S_cm_new": float(b1.kappa),
        "kappa_rel_pct": rel(b0.kappa, b1.kappa),
        "i0_cu_A_cm2": float(b0.i0_cu),
        "i0_cu_A_cm2_new": float(b1.i0_cu),
        "i0_cu_rel_pct": rel(b0.i0_cu, b1.i0_cu),
    }


# ---------------------------------------------------------------- report

def temperature_rise_report(
    I_A: float,
    V_cell: float | None = None,
    volume_L: float = 1.0,
    hours: float = 4.5,
    *,
    R_ohm: float | None = None,
    T0_C: float = 22.0,
    T_inf_C: float | None = None,
    h_W_m2K: float = H_AIR_NATURAL_W_M2K,
    beaker_id_cm: float = 9.0,
    area_cm2: float | None = None,
    extra_heat_capacity_J_K: float = 0.0,
    C_J_K: float | None = None,
    bath: Bath | None = None,
    j_A_cm2=None,
    kappa_S_cm: float | None = None,
    section_Lx_cm: float = 4.0,
    section_Ly_cm: float = 2.0,
    nx: int = 40,
    ny: int = 40,
) -> dict:
    """Per-run temperature-rise record (lumped model + optional 2-D section).

    Heating is ``I_A * V_cell`` (or ``I_A**2 * R_ohm`` when ``V_cell`` is None).
    The lumped result is reported as the steady rise, the rise after ``hours``
    (closed form, exact) and its RK4 cross-check, plus the thermal time constant.
    The bath properties are then re-evaluated at ``T0_C + dT_end`` with
    ``bath_property_shift`` so the run can log "dT = x K, kappa +y %".

    If ``kappa_S_cm`` and ``j_A_cm2`` are both given, a 2-D steady cell-section
    solve (``section_Lx_cm`` x ``section_Ly_cm``, all walls Dirichlet at
    ``T0_C + dT_end/2`` ... see ``solve_heat_2d``) is added under ``"section"``.
    """
    if hours < 0.0:
        raise ValueError(f"hours must be >= 0, got {hours!r}")
    lt = lumped_parameters(I_A, V_cell, volume_L, R_ohm=R_ohm, h_W_m2K=h_W_m2K,
                          area_cm2=area_cm2, beaker_id_cm=beaker_id_cm,
                          extra_heat_capacity_J_K=extra_heat_capacity_J_K,
                          C_J_K=C_J_K, T0_C=T0_C, T_inf_C=T_inf_C)

    t_s = float(hours) * 3600.0
    dT_end = lt.dT_at(t_s) if t_s > 0.0 else 0.0
    _, T_rk4 = lt.T_history(t_s, n_steps=max(1, int(min(2000, max(50, t_s / 5.0))))) \
        if t_s > 0.0 else (None, np.array([lt.T0_C]))
    dT_rk4 = float(T_rk4[-1]) - lt.T_inf_C

    rep = {
        "model": "lumped_newton_cooling",
        "I_A": float(I_A),
        "V_cell_V": None if V_cell is None else float(V_cell),
        "R_ohm": None if R_ohm is None else float(R_ohm),
        "power_source": "I*V_cell" if V_cell is not None else "I^2*R_ohm",
        "P_W": float(lt.Q_W),
        "volume_L": float(volume_L),
        "fill_height_cm": beaker_fill_height_cm(volume_L, beaker_id_cm),
        "area_cm2": float(area_cm2 if area_cm2 is not None
                          else beaker_surface_area_cm2(volume_L, beaker_id_cm)),
        "h_W_m2K": float(h_W_m2K),
        "UA_W_K": float(lt.UA_W_K),
        "C_J_K": float(lt.C_J_K),
        "tau_h": lt.tau_s / 3600.0,
        "hours": float(hours),
        "T0_C": float(lt.T0_C),
        "T_inf_C": float(lt.T_inf_C),
        "dT_steady_K": float(lt.dT_steady_K),
        "dT_end_K": float(dT_end),
        "dT_end_rk4_K": float(dT_rk4),
        "T_end_C": float(lt.T0_C + dT_end),
    }
    rep.update(bath_property_shift(lt.T0_C, dT_end, bath=bath))

    if kappa_S_cm is not None and j_A_cm2 is not None:
        # walls at the mean bath temperature of the run; the section is a
        # local worst-case heat path, not the whole beaker
        T_wall = lt.T0_C + 0.5 * dT_end
        hf = cell_section_heat(j_A_cm2, kappa_S_cm, section_Lx_cm, section_Ly_cm,
                               nx=nx, ny=ny, T_bulk_C=T_wall)
        sec = hf.to_dict()
        sec["j_A_cm2_max"] = float(np.max(np.abs(np.asarray(j_A_cm2, dtype=float))))
        rep["section"] = sec

    rep["summary_line"] = (
        f"dT = {rep['dT_end_K']:.2f} K after {hours:g} h "
        f"(steady {rep['dT_steady_K']:.2f} K), kappa {rep['kappa_rel_pct']:+.1f} %, "
        f"D {rep['D_cu_rel_pct']:+.1f} %, i0 {rep['i0_cu_rel_pct']:+.1f} %"
    )
    return rep


__all__ = [
    "H_AIR_NATURAL_W_M2K",
    "WATER_CP_J_G_K",
    "WATER_K_W_CM_K",
    "WATER_RHO_G_CM3",
    "HeatField2D",
    "LumpedThermal",
    "bath_property_shift",
    "beaker_fill_height_cm",
    "beaker_surface_area_cm2",
    "cell_section_heat",
    "lumped_parameters",
    "ohmic_resistance_ohm",
    "solve_heat_2d",
    "temperature_rise_report",
    "volumetric_source",
]
