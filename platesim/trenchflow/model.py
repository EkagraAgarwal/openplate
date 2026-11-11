"""Coupled trench growth model: quasi-steady flow + Cu2+ transport, marched galvanostatically.

Quick estimator. The reference model is the feature engine with geometry='trench_full'.

Physical picture
----------------
The coupon sits in a stirred beaker. At feature scale the bulk flow is a shear flow with wall shear
rate ``G`` (1/s) and the Cu2+ concentration reaches its bulk value at a Nernst-layer height ``delta``
above the wafer. For a given trench (width, depth) and a given stage of the deposit we build

    grid = trench_grid(width, depth, height_um=delta_um, dx_um=dx, deposit_um=dep)

solve the flow ``solve_flow(grid, G)``, then the Cu2+ field with a first-order surface reaction of rate
constant ``k``: ``solve_concentration(grid, flow, D, k)``.

The metal is one equipotential and the ohmic drop inside a ~100 um feature is small (Wagner number >> 1)
[J], so the same ``k`` applies on every surface; the local growth rate is proportional to the local
flux density ``k * c_wall``. The cell is galvanostatic, so ``k`` is not known a priori: it is the value
for which the FIELD flux density equals the local field current density,

    flux_field = j_field / (2 F c_bulk)                                            [C]

``flux_field`` is monotonic in ``k``, so that scalar equation is solved with
``scipy.optimize.brentq`` on ``log10(k)`` between 1e-7 and 1e2 cm/s. If ``j_field`` exceeds the
mass-transfer limit ``2 F c_bulk D / delta`` a ``ValueError`` is raised quoting that limiting current
density.

Growth marching
---------------
Stages ``s = 0..n-1`` cover field thickness from 0 to ``T_field`` in equal steps ``dT``. At the
midpoint thickness ``t_s`` the geometry uses the conformal deposit

    dep = t_s * 0.5 * (1 + r_prev)                                                 [J]

where ``r_prev`` is the previous stage's ``floor_over_field`` (1.0 for the first stage): the geometry
is approximated as conformal at the mean of the field and floor thickness. Then

    floor    += floor_over_field    * dT
    wall_up  += wall_up_over_field  * dT
    wall_down+= wall_down_over_field* dT

Upstream wall = the trench wall at smaller x (the flow goes in +x), downstream = the wall at larger x.
``transport.region_means`` gives the field, floor and combined wall means; the two walls are separated
here from ``surf_x_um`` (split at the trench centre ``x_left + width/2``) and the floor is identified by
the solid cell below each surface cell (the same rule as ``transport._region_masks``).

Constants come from the feature engine so both models agree: ``Params()``, ``derive(p, T_C)``,
``D = ph.D_c`` (includes the migration enhancement), ``c_bulk = p.c_bulk``, ``F`` from
``platesim.core.constants`` and ``ph.Mfac`` (cm/s per A/cm2) for the Faraday thickness rate.

Every public entry point takes optional ``p: Params`` and ``ph: Phys`` keyword arguments: pass the same
objects the feature engine uses (``ph = derive(p, T_C)``) and the two models run on identical constants
(``ph.D_c``, ``ph.Mfac``, ``p.c_bulk``). ``ph.transport == "tertiary"`` sets ``ph.D_c = D_cu`` because
that path carries migration in an explicit potential field. The quick model is 1-D and has no potential
field, so it must keep the migration enhancement: it uses ``E_mig * D_cu`` and records the choice in the
returned ``'D_source'`` entry. The default (no ``p``/``ph``) is the enhanced set and is bit-identical to
the pre-``p``/``ph`` behaviour.

There are no parameters fitted to the trench data in this model. The only measured input is the field
thickness ``field_um``, which fixes ``j_field``.

Tags: [J] judgement/derivation, [C] calculation, [FT] full-text literature, [M] measured.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from platesim.core.constants import F
from platesim.feature.engine.params import Params, derive

from .cavity_lbm import eddy_summary, solve_flow
from .geometry import trench_grid
from .transport import region_means, solve_concentration

_UM_TO_CM = 1e-4
LEVEQUE = 0.538          # Sh_x = 0.538 (G x^2 / D)^(1/3); the same prefactor as platesim/flow/lbm.py [FT]
_MEAN_PREFACTOR = 1.5 * LEVEQUE   # 0.807: mean of x^(-1/3) over a plate of length L is 1.5 L^(-1/3) [C]
K_LO, K_HI = 1e-7, 1e2  # cm/s bracket for the galvanostatic k solve [J]


def shear_from_delta(delta_um: float, D_cm2_s: float, L_cm: float = 2.0) -> float:
    """Wall shear rate G (1/s) whose Leveque mean mass-transfer coefficient gives ``delta_um``.

    Leveque over a plate of length L: the mean mass-transfer coefficient is

        k_mean = 1.5 * 0.538 * D^(2/3) * G^(1/3) * L^(-1/3)                         [FT]

    (the 1.5 is the mean of x^(-1/3) over 0 < x < L; the 0.538 prefactor is the local Leveque one used
    already in platesim/flow/lbm.py, Newman, Electrochemical Systems). Requiring k_mean = D/delta:

        G = ((D/delta) * L^(1/3) / (0.807 * D^(2/3)))**3                            [C]
    """
    D = float(D_cm2_s)
    delta_cm = float(delta_um) * _UM_TO_CM
    if D <= 0.0 or delta_cm <= 0.0:
        raise ValueError("D_cm2_s and delta_um must be positive")
    k_mean = D / delta_cm
    return float((k_mean * float(L_cm) ** (1.0 / 3.0) / (_MEAN_PREFACTOR * D ** (2.0 / 3.0))) ** 3)


def _constants(T_C, p=None, ph=None):
    """(D cm2/s, c_bulk mol/cm3, Mfac cm/s per A/cm2, D_source) of the quick 1-D model.

    ``ph = derive(p, T_C)`` is used when ``ph`` is omitted.  On the tertiary path ``ph.D_c = D_cu``
    (migration lives in the explicit potential field of the feature engine); this model is 1-D and has
    no potential field, so it keeps the lumped enhancement ``E_mig * D_cu`` and says so in ``D_source``.
    """
    if ph is None:
        p = p if p is not None else Params()
        ph = derive(p, T_C)
    elif p is None:
        p = Params()            # c_bulk only; pass p when it differs from the default bath
    if ph.transport == "tertiary":
        D = float(p.E_mig * ph.D_cu)
        src = "tertiary:E_mig*D_cu (1-D quick model has no potential field; migration lumped)"
    else:
        D = float(ph.D_c)
        src = "enhanced:ph.D_c"
    return D, float(p.c_bulk), float(ph.Mfac), src


def model_constants(T_C, p=None, ph=None):
    """(D_cm2_s with migration, c_bulk mol/cm3, Mfac cm/s per A/cm2) from the feature engine.

    Optional ``p``/``ph`` (``ph = derive(p, T_C)``) let the caller reuse the constants of the feature
    engine instead of a fresh default ``Params()``.  See :func:`_constants` for the tertiary path.
    """
    D, c_bulk, mfac, _ = _constants(T_C, p=p, ph=ph)
    return D, c_bulk, mfac


def _limit_current_mA_cm2(D_cm2_s, c_bulk, delta_um):
    """Mass-transfer limiting current density of the field surface, mA/cm2: 2 F c_bulk D / delta."""
    return float(2.0 * F * c_bulk * D_cm2_s / (float(delta_um) * _UM_TO_CM) * 1e3)


def _wall_fluxes(result, grid):
    """Mean wall flux density (cm/s) of the upstream and downstream trench walls.

    Wall cells = surface cells inside the open trench that are not floor cells (a floor cell has a solid
    cell directly below it, as in transport.region_masks). Split at the trench centre x. Returns
    ``(mean_up, mean_down, n_up, n_down)``; NaN when a group is empty.
    """
    iz = np.asarray(result["surf_iz"])
    ix = np.asarray(result["surf_ix"])
    fd = np.asarray(result["surf_flux_density"], dtype=float)
    x = np.asarray(result["surf_x_um"], dtype=float)
    s = grid.solid
    tol = 1e-6 * max(abs(float(grid.dx_um)), 1.0)
    xl = float(grid.x_left_um)
    xr = xl + float(grid.width_um)

    below = np.zeros(iz.size, bool)
    has = iz > 0
    below[has] = s[iz[has] - 1, ix[has]]           # bottom array row cannot have a solid neighbour below
    in_trench = (x > xl - tol) & (x < xr + tol)
    wall = in_trench & ~below
    xc = xl + 0.5 * float(grid.width_um)
    up = wall & (x < xc)
    down = wall & (x > xc)

    def _mean(m):
        return float(np.mean(fd[m])) if np.any(m) else float("nan")

    return _mean(up), _mean(down), int(up.sum()), int(down.sum())


def stage_fields(width_um, depth_um, delta_um, shear_rate_s, j_field_mA_cm2, deposit_um, T_C=22.0,
                 dx_um=4.0, use_flow=True, p=None, ph=None):
    """Full fields of one stage: grid, flow, concentration, k and the region fluxes.

    Same inputs as :func:`stage`; used by :mod:`platesim.trenchflow.validate` to dump fields for figures.
    ``p``/``ph`` (``ph = derive(p, T_C)``) select the constants instead of a fresh default ``Params()``.

    The grid height is ``delta_um`` [J: the Nernst layer is the transport boundary layer]. Late stages
    can push the conformal deposit above that height; the domain is then extended to
    ``deposit_um + 2 dx`` so the top Dirichlet row stays in the electrolyte [J] (numerical guard, not a
    physical layer thickness).
    """
    D, c_bulk, _, d_source = _constants(T_C, p=p, ph=ph)
    j_field_mA_cm2 = float(j_field_mA_cm2)
    i_lim = _limit_current_mA_cm2(D, c_bulk, delta_um)
    if j_field_mA_cm2 >= i_lim:
        raise ValueError(
            f"current density {j_field_mA_cm2:.3g} mA/cm2 exceeds the mass-transfer limit of the field "
            f"surface, i_lim_field = {i_lim:.3g} mA/cm2 (2 F c_bulk D / delta)")
    if j_field_mA_cm2 <= 0.0:
        raise ValueError("j_field_mA_cm2 must be positive")

    dx = float(dx_um)
    height = max(float(delta_um), float(deposit_um) + 2.0 * dx)
    grid = trench_grid(float(width_um), float(depth_um), height_um=height, dx_um=dx,
                       deposit_um=float(deposit_um))
    flow = solve_flow(grid, float(shear_rate_s)) if use_flow else None
    target_flux = j_field_mA_cm2 * 1e-3 / (2.0 * F * c_bulk)        # cm/s, j / (2 F c_bulk) [C]
    k, res = solve_k(grid, flow, D, target_flux)
    rm = region_means(res, grid)
    wall_up, wall_down, n_up, n_down = _wall_fluxes(res, grid)
    field = rm["field"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "grid": grid,
            "flow": flow,
            "c_result": res,
            "k_cm_s": float(k),
            "field_flux_cm_s": float(field),
            "floor_over_field": float(rm["floor"] / field),
            "wall_up_over_field": float(wall_up / field),
            "wall_down_over_field": float(wall_down / field),
            "i_lim_field_mA_cm2": i_lim,
            "n_up": n_up,
            "n_down": n_down,
            "eddy": eddy_summary(flow) if flow is not None else None,
            "grid_shape": tuple(int(v) for v in grid.shape),
            "D_source": d_source,
        }
    return out


def solve_k(grid, flow, D_cm2_s, target_field_flux_cm_s):
    """Solve the galvanostatic condition for the surface rate constant k.

    ``target_field_flux_cm_s`` = ``j_field / (2 F c_bulk)`` (cm/s). The mean field flux density from
    :func:`transport.region_means` increases monotonically with k, so ``brentq`` on ``log10(k)`` over
    [1e-7, 1e2] cm/s is used. Returns ``(k_cm_s, transport_result_at_that_k)``.

    Raises ``ValueError`` when no k in the bracket reaches the target (i.e. the target is at or above the
    mass-transfer limit of the geometry) or when the target is below the flux of the smallest k.
    """
    D = float(D_cm2_s)
    target = float(target_field_flux_cm_s)
    if not target > 0.0:
        raise ValueError("target_field_flux_cm_s must be positive")

    def gap(logk):
        res = solve_concentration(grid, flow, D_cm2_s=D, k_cm_s=10.0 ** logk)
        return region_means(res, grid)["field"] - target

    g_lo, g_hi = gap(np.log10(K_LO)), gap(np.log10(K_HI))
    if g_hi < 0.0:
        raise ValueError(
            f"target field flux {target:.4g} cm/s is above the achievable flux at k = {K_HI:g} cm/s "
            f"({target - g_hi:.4g} cm/s, mass-transfer limited)")
    if g_lo > 0.0:
        res = solve_concentration(grid, flow, D_cm2_s=D, k_cm_s=K_LO)
        return float(K_LO), res
    logk = brentq(gap, np.log10(K_LO), np.log10(K_HI), xtol=1e-6, rtol=1e-10)
    k = float(10.0 ** logk)
    res = solve_concentration(grid, flow, D_cm2_s=D, k_cm_s=k)
    return k, res


def stage(width_um, depth_um, delta_um, shear_rate_s, j_field_mA_cm2, deposit_um, T_C=22.0, dx_um=4.0,
          use_flow=True, p=None, ph=None):
    """One quasi-steady growth stage; scalar result of :func:`stage_fields`.

    Returns ``{'k_cm_s', 'floor_over_field', 'wall_up_over_field', 'wall_down_over_field',
    'field_flux_cm_s', 'i_lim_field_mA_cm2', 'eddy', 'grid_shape', 'D_source'}``.
    """
    f = stage_fields(width_um, depth_um, delta_um, shear_rate_s, j_field_mA_cm2, deposit_um, T_C=T_C,
                     dx_um=dx_um, use_flow=use_flow, p=p, ph=ph)
    return {
        "k_cm_s": f["k_cm_s"],
        "floor_over_field": f["floor_over_field"],
        "wall_up_over_field": f["wall_up_over_field"],
        "wall_down_over_field": f["wall_down_over_field"],
        "field_flux_cm_s": f["field_flux_cm_s"],
        "i_lim_field_mA_cm2": f["i_lim_field_mA_cm2"],
        "eddy": f["eddy"],
        "grid_shape": f["grid_shape"],
        "D_source": f["D_source"],
    }


def predict_site(site, delta_um, shear_rate_s=None, n_stages=5, dx_um=None, use_flow=True, field_um=None,
                 p=None, ph=None):
    """March one SEM site of ``trench_sem.json`` through ``n_stages`` galvanostatic stages.

    The measured field thickness is an INPUT: it fixes the local field current density

        j_field = field_um * 1e-4 / (Mfac * hours * 3600)      [A/cm2]                [C]

    (reported in mA/cm2). ``shear_rate_s = None`` -> ``shear_from_delta(delta_um, D)``. ``dx_um = None``
    -> 4.0 um for width < 400 um, else 6.0 um.  ``p``/``ph`` (``ph = derive(p, T_C)``) select the
    constants instead of a fresh default ``Params()``; the same objects are forwarded to every stage.

    Returns ``{'id', 'field_um', 'j_field_mA_cm2', 'delta_um', 'shear_rate_s', 'floor_um', 'wall_up_um',
    'wall_down_um', 'floor_over_field', 'wall_asymmetry', 'stages', 'use_flow', 'dx_um', 'D_source'}``; each
    stage dict holds the stage() scalars plus ``'t_field_um'`` (midpoint field thickness) and ``'deposit_um'``.
    """
    width = float(site["width_um"])
    depth = float(site["depth_um"])
    hours = float(site["hours"])
    T_C = float(site.get("T_C", 22.0))
    D, _, mfac, d_source = _constants(T_C, p=p, ph=ph)

    field_um = float(site["field_um"]["best"]) if field_um is None else float(field_um)
    j_field = field_um * _UM_TO_CM / (mfac * hours * 3600.0)          # A/cm2
    j_field_mA = j_field * 1e3
    shear = shear_from_delta(delta_um, D) if shear_rate_s is None else float(shear_rate_s)
    if dx_um is None:
        dx_um = 4.0 if width < 400.0 else 6.0
    dx_um = float(dx_um)

    dT = field_um / float(n_stages)
    r_prev = 1.0
    floor = wall_up = wall_down = 0.0
    stages = []
    for s in range(int(n_stages)):
        t_field = (s + 0.5) * dT
        dep = t_field * 0.5 * (1.0 + r_prev)                          # conformal at mean of field/floor [J]
        st = stage(width, depth, delta_um, shear, j_field_mA, dep, T_C=T_C, dx_um=dx_um, use_flow=use_flow,
                   p=p, ph=ph)
        floor += st["floor_over_field"] * dT
        wall_up += st["wall_up_over_field"] * dT
        wall_down += st["wall_down_over_field"] * dT
        r_prev = st["floor_over_field"]
        row = dict(st)
        row["t_field_um"] = float(t_field)
        row["deposit_um"] = float(dep)
        stages.append(row)

    with np.errstate(invalid="ignore", divide="ignore"):
        asymmetry = float(wall_down / wall_up) if wall_up else float("nan")
    return {
        "id": str(site.get("id", "")),
        "field_um": field_um,
        "j_field_mA_cm2": float(j_field_mA),
        "delta_um": float(delta_um),
        "shear_rate_s": float(shear),
        "dx_um": dx_um,
        "floor_um": float(floor),
        "wall_up_um": float(wall_up),
        "wall_down_um": float(wall_down),
        "floor_over_field": float(floor / field_um),
        "wall_asymmetry": asymmetry,
        "stages": stages,
        "use_flow": bool(use_flow),
        "D_source": d_source,
    }
