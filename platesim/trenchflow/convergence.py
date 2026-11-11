"""Grid convergence, marching-sensitivity and LBM tau-clamp checks for the coupled trench model.

Task 2 of ``plan.md`` (wave 1). Nothing here changes physics; it measures how much the numerical
settings of ``platesim.trenchflow`` move the answer.

1. **Grid convergence** (:func:`convergence_row`): one quasi-steady stage at deposit 0 for every
   ``dx``, at exactly the conditions ``validate.py`` uses for the "with flow" case:
   ``delta = 64 um``, ``G = shear_from_delta(64, D)`` (``model.py:72``),
   ``j_field = field_um * 1e-4 / (Mfac * hours * 3600)`` (``predict_site``, ``model.py:260``),
   ``T`` from the site, ``use_flow=True``.  Reported per grid: floor/field, wall/field (upstream,
   downstream and their mean), ``k``, the LBM convergence flag/iterations from ``flow.info``, the
   eddy penetration depth and the wall time.  :func:`richardson` gives the observed order and the
   Richardson-extrapolated floor/field from the three finest grids plus the ``dx = 4`` error.

2. **Marching sweep** (:func:`march_sweep`): full :func:`platesim.trenchflow.model.predict_site`
   at dx 6/4/3 and n_stages 3/5/9; the spread (max - min) of the final floor/field ratio is the
   model's sensitivity to the stage decomposition.  This sweep is expensive (the wide site at
   dx = 3 needs minutes per stage); ``quick=True`` exercises it once (one site, dx 8, 2 stages).
   The full wave-3 sweep (both sites, dx 8..2, 3/5/9 stages) took 4697.8 s (~78 min) on this host.

3. **LBM tau clamp** (:func:`tau_clamp_check`): the same physical Reynolds number on the same grid
   with two lattice lid velocities.  ``cavity_lbm._scaling`` (``cavity_lbm.py:96``) clamps
   ``tau <= TAU_MAX`` and lowers ``u_lid`` to keep ``Re``; with the Trench 1 geometry both of the
   requested lid velocities clamp to the *same* lattice lid, so the check is degenerate.  The
   degenerate pair is still run (it proves the clamp is deterministic) and an unclamped pair at the
   same ``Re`` but a smaller lattice lid -- where ``tau`` genuinely differs -- supplies the
   informative number.  Result for the narrow site at dx 4: 5.5 % mean, 25 % on cells above 5 % of
   the trench speed, 81 % at a near-stagnant corner -- **the clamp is not benign at this Re**
   (``tau_clamp_benign = False``), so the "Stokes flow is Re-independent, the clamp is benign"
   argument in ``cavity_lbm.py:99-105`` is not confirmed by this cavity.

4. **Particles versus continuum** (:func:`particle_cross_check`): a 120 x 60 um trench, dx 6, LBM
   flow, 20-40 % surface depletion; floor/field from :func:`platesim.trenchflow.particles.simulate`
   against :func:`platesim.trenchflow.transport.solve_concentration`.  Agreement is checked as
   ``|p - c| <= max(3 * se, 0.08 * |c|)``.

No parameter is fitted, no existing module is edited; the only output is ``convergence.json`` for a
full run and ``convergence_quick.json`` for ``--quick`` (quick mode must never overwrite the
production artifact), both next to this file.  Files are written as strict JSON: non-finite values
(NaN observed order, undefined spread) become ``null``, while the returned dict keeps the NaN.
``quick=True`` (CLI ``--quick``) keeps every step to seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from scipy.optimize import brentq

from .cavity_lbm import TAU_MAX, penetration_depth_um, solve_flow
from .geometry import trench_grid
from .model import model_constants, predict_site, shear_from_delta, stage_fields
from .particles import region_means as particle_region_means
from .particles import simulate as simulate_particles
from .transport import region_means, solve_concentration

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
SEM_JSON = _ROOT / "data" / "trench_sem.json"
CONVERGENCE_JSON = _HERE / "convergence.json"          # production artifact (full sweep)
QUICK_JSON = _HERE / "convergence_quick.json"          # quick-mode artifact (never overwrites the above)

DELTA_UM = 64.0            # central Nernst layer of the Trench 1 run, as in validate.py:40
NU_CM2_S = 0.0114          # cavity_lbm.solve_flow default kinematic viscosity
DX_LIST = (8.0, 6.0, 4.0, 3.0, 2.0)
MARCH_DX = (6.0, 4.0, 3.0)
MARCH_STAGES = (3, 5, 9)
DUST = 1e-9


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _strict(obj: Any) -> Any:
    """Copy of ``obj`` with every non-finite float replaced by ``None`` (strict RFC JSON).

    The in-memory result keeps NaN, which is the honest value for e.g. a non-monotone Richardson
    order; only the file on disk is sanitised, so ``json.loads`` on it never sees a bare ``NaN``.
    """
    if isinstance(obj, dict):
        return {str(k): _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _finite(x):
    """float(x) or NaN; keeps a JSON-serialisable placeholder for missing/undefined values."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _order_extrap(dx, f):
    """Observed order p and the Richardson limit f0 from three grids with any refinement ratio.

    ``f(h) = f0 + C h^p``; ``p`` solves

        (f1 - f3) / (f2 - f3) = (h1^p - h3^p) / (h2^p - h3^p)      (h1 > h2 > h3)

    with ``brentq`` on ``0.05 <= p <= 10`` and then ``f0 = f3 - (f2 - f3) h3^p / (h2^p - h3^p)``.
    This is the general three-point form (the log-ratio shortcut only holds for a constant
    refinement ratio h1/h2 = h2/h3, which neither 8/6/4 nor 6/4/3 satisfies).  Returns
    ``(nan, nan)`` when the three values are not monotone in ``dx`` or the bracket does not contain
    a sign change - a non-monotone sequence must not crash the study.
    """
    dx = np.asarray(dx, dtype=float)
    f = np.asarray(f, dtype=float)
    nan = float("nan")
    if dx.size < 3 or f.size < 3 or not np.all(np.isfinite(f)) or dx[-2] == dx[-1]:
        return nan, nan
    d1, d2, d3 = float(dx[-3]), float(dx[-2]), float(dx[-1])
    f1, f2, f3 = float(f[-3]), float(f[-2]), float(f[-1])
    if f1 == f3 or f2 == f3:
        return nan, nan

    def g(p):
        return ((f1 - f3) / (f2 - f3)
                - (d1 ** p - d3 ** p) / (d2 ** p - d3 ** p))

    lo, hi = 0.05, 10.0
    try:
        if not np.isfinite(g(lo)) or not np.isfinite(g(hi)) or g(lo) * g(hi) > 0.0:
            return nan, nan
        p = float(brentq(g, lo, hi))
    except (ValueError, ZeroDivisionError, OverflowError):
        return nan, nan
    c = (f2 - f3) / (d2 ** p - d3 ** p)
    return p, float(f3 - c * d3 ** p)


def richardson(rows, key="floor_over_field", dx_probe=4.0):
    """Observed order and Richardson extrapolation of ``key`` from the three finest grids.

    Uses :func:`_order_extrap` (general refinement ratio).  ``p`` and the extrapolated value are NaN
    when the three values are not monotone in ``dx``; ``dx_probe`` selects the row whose value is
    compared with the extrapolated one (``dx = 4 um`` by default).
    """
    out = {
        "key": key,
        "dxs_used_um": [],
        "values_used": [],
        "observed_order": float("nan"),
        "extrapolated": float("nan"),
        "dx_probe_um": None,
        "dx_probe_minus_extrapolated": float("nan"),
        "dx_probe_error_rel": float("nan"),
        "note": "",
    }
    pts = [r for r in rows if r.get("dx_um") is not None and np.isfinite(_finite(r.get(key)))]
    pts.sort(key=lambda r: -float(r["dx_um"]))
    if len(pts) < 3:
        out["note"] = f"needs at least 3 grids, got {len(pts)}"
    else:
        p1, p2, p3 = pts[-3], pts[-2], pts[-1]
        h1, h2, h3 = float(p1["dx_um"]), float(p2["dx_um"]), float(p3["dx_um"])
        f1, f2, f3 = _finite(p1[key]), _finite(p2[key]), _finite(p3[key])
        out["dxs_used_um"] = [h1, h2, h3]
        out["values_used"] = [f1, f2, f3]
        p, f0 = _order_extrap([h1, h2, h3], [f1, f2, f3])
        out["observed_order"] = p
        out["extrapolated"] = f0
        if not np.isfinite(p):
            out["note"] = ("non-monotone or degenerate sequence: observed order and "
                           "extrapolation set to NaN")
        elif not (0.0 < p <= 10.0):
            out["note"] = f"observed order {p:.3g} outside [0.05, 10]"


    probe = [r for r in rows if r.get("dx_um") is not None
             and abs(float(r["dx_um"]) - float(dx_probe)) < 1e-9]
    if probe and np.isfinite(_finite(probe[0].get(key))):
        fp = _finite(probe[0][key])
        out["dx_probe_um"] = float(dx_probe)
        if np.isfinite(out["extrapolated"]):
            out["dx_probe_minus_extrapolated"] = fp - out["extrapolated"]
            if abs(out["extrapolated"]) > 0.0:
                out["dx_probe_error_rel"] = (fp - out["extrapolated"]) / out["extrapolated"]
    return out


# ----------------------------------------------------------------------------------------------
# 1. grid convergence at fixed conditions (one stage, deposit 0, with flow)
# ----------------------------------------------------------------------------------------------
def site_conditions(site, delta_um=DELTA_UM):
    """The conditions ``validate.py`` derives for the "with flow" case of one SEM site.

    ``delta_um`` is the central Nernst layer, ``G = shear_from_delta(delta, D)`` (``model.py:72``)
    and ``j_field = field_um * 1e-4 / (Mfac * hours * 3600)`` (``predict_site``, ``model.py:260``),
    i.e. the field current density that would deposit the measured field thickness in ``hours``.
    """
    T_C = float(site.get("T_C", 22.0))
    D, c_bulk, mfac = model_constants(T_C)
    field_um = float(site["field_um"]["best"])
    hours = float(site["hours"])
    j_field_mA = field_um * 1e-4 / (mfac * hours * 3600.0) * 1e3
    return {
        "id": str(site.get("id", "")),
        "width_um": float(site["width_um"]),
        "depth_um": float(site["depth_um"]),
        "T_C": T_C,
        "hours": hours,
        "field_um": field_um,
        "delta_um": float(delta_um),
        "shear_rate_s": float(shear_from_delta(delta_um, D)),
        "j_field_mA_cm2": float(j_field_mA),
        "D_cm2_s": float(D),
        "c_bulk_mol_cm3": float(c_bulk),
        "Mfac_cm_s_per_A_cm2": float(mfac),
    }


def convergence_row(site, dx_um, cond=None):
    """One stage at ``deposit = 0`` with flow: ratios, k, LBM bookkeeping, penetration, wall time."""
    cond = site_conditions(site) if cond is None else cond
    t0 = time.time()
    fld = stage_fields(cond["width_um"], cond["depth_um"], cond["delta_um"], cond["shear_rate_s"],
                       cond["j_field_mA_cm2"], 0.0, T_C=cond["T_C"], dx_um=float(dx_um),
                       use_flow=True)
    wall_s = time.time() - t0
    flow = fld["flow"]
    info = dict(flow.info or {})
    eddy = fld.get("eddy") or {}
    pen = eddy.get("penetration_um")
    if pen is None or not np.isfinite(_finite(pen)):
        pen = penetration_depth_um(flow)
    up = _finite(fld["wall_up_over_field"])
    down = _finite(fld["wall_down_over_field"])
    with np.errstate(invalid="ignore"):
        wall_mean = 0.5 * (up + down) if np.isfinite(up) and np.isfinite(down) else float("nan")
    return {
        "dx_um": float(dx_um),
        "floor_over_field": _finite(fld["floor_over_field"]),
        "wall_up_over_field": up,
        "wall_down_over_field": down,
        "wall_over_field": float(wall_mean),
        "wall_asymmetry": _finite(
            fld["wall_down_over_field"] / fld["wall_up_over_field"]
            if fld["wall_up_over_field"] else float("nan")),
        "k_cm_s": _finite(fld["k_cm_s"]),
        "field_flux_cm_s": _finite(fld["field_flux_cm_s"]),
        "i_lim_field_mA_cm2": _finite(fld["i_lim_field_mA_cm2"]),
        "lbm_converged": bool(info.get("converged", flow.converged)),
        "lbm_steps": int(info.get("steps", 0)),
        "lbm_tau": _finite(info.get("tau")),
        "lbm_u_lid_lb": _finite(info.get("u_lid_lb")),
        "lbm_re": _finite(info.get("re")),
        "lbm_re_clamped": bool(info.get("re_clamped", False)),
        "penetration_um": _finite(pen),
        "n_up": int(fld.get("n_up", 0)),
        "n_down": int(fld.get("n_down", 0)),
        "grid_shape": [int(v) for v in fld["grid_shape"]],
        "balance_error": _finite(fld["c_result"]["balance_error"]),
        "wall_s": float(wall_s),
    }


# ----------------------------------------------------------------------------------------------
# 2. full marching sweep (predict_site) - sensitivity to dx and to the stage decomposition
# ----------------------------------------------------------------------------------------------
def march_sweep(sites, dx_values=MARCH_DX, stage_values=MARCH_STAGES, use_flow=True,
                delta_um=DELTA_UM):
    """Full :func:`predict_site` march for every (dx, n_stages) pair; spread of the final ratio.

    This is the expensive part of the study (each row re-solves the flow and the k scalar equation
    for every stage), so ``run(quick=True)`` calls it with one site, dx 8 and 2 stages only.
    """
    out = {
        "dx_values_um": [float(v) for v in dx_values],
        "stage_values": [int(v) for v in stage_values],
        "delta_um": float(delta_um),
        "use_flow": bool(use_flow),
        "sites": [],
        "notes": [
            "spread = max - min of the final floor_over_field over the whole (dx, n_stages) grid",
            "predict_site marches floor += floor_over_field * dT with the conformal midpoint geometry "
            "dep = t_field * 0.5 * (1 + previous ratio) [J]",
        ],
    }
    for site in sites:
        rows = []
        t_site = time.time()
        for dx in dx_values:
            for n in stage_values:
                t0 = time.time()
                try:
                    pred = predict_site(site, float(delta_um), None, n_stages=int(n),
                                        dx_um=float(dx), use_flow=use_flow)
                except ValueError as exc:                 # above the mass-transfer limit / no k
                    rows.append({"dx_um": float(dx), "n_stages": int(n), "error": str(exc),
                                 "wall_s": float(time.time() - t0)})
                    continue
                rows.append({
                    "dx_um": float(dx),
                    "n_stages": int(n),
                    "floor_over_field": _finite(pred["floor_over_field"]),
                    "floor_um": _finite(pred["floor_um"]),
                    "wall_up_um": _finite(pred["wall_up_um"]),
                    "wall_down_um": _finite(pred["wall_down_um"]),
                    "wall_asymmetry": _finite(pred["wall_asymmetry"]),
                    "j_field_mA_cm2": _finite(pred["j_field_mA_cm2"]),
                    "k_first_cm_s": _finite(pred["stages"][0]["k_cm_s"]) if pred["stages"] else None,
                    # predict_site/stage expose only the eddy summary, not flow.info; this flag
                    # therefore says nothing about LBM convergence (flow.converged is not returned)
                    "eddy_summary_available_all": bool(
                        all(st.get("eddy") is not None for st in pred["stages"])),
                    "wall_s": float(time.time() - t0),
                })
        vals = [r["floor_over_field"] for r in rows if "floor_over_field" in r]
        enough = len(vals) >= 2
        entry = {
            "id": str(site.get("id", "")),
            "width_um": float(site["width_um"]),
            "depth_um": float(site["depth_um"]),
            "rows": rows,
            # a spread needs at least two runs; one (dx, n_stages) point has no spread
            "spread_defined": bool(enough),
            "spread_floor_over_field": float(max(vals) - min(vals)) if enough else float("nan"),
            "min_floor_over_field": float(min(vals)) if enough else float("nan"),
            "max_floor_over_field": float(max(vals)) if enough else float("nan"),
            "wall_s": float(time.time() - t_site),
        }
        if not enough:
            entry["note"] = (f"spread undefined: {len(vals)} usable run(s), need at least 2 "
                            "(quick mode runs one dx x one stage count)")
        out["sites"].append(entry)
    return out


# ----------------------------------------------------------------------------------------------
# 3. LBM tau-clamp check: two lattice settings at the same physical Reynolds number
# ----------------------------------------------------------------------------------------------
def _velocity_pair(flow_a, flow_b, grid, label, note="", sig_frac=0.05):
    """Relative difference of the two lattice velocity fields over the open trench cells.

    ``max_rel_diff`` is the literal ``max |u1 - u2| / max(|u1|, |u2|)`` over every open trench cell
    (the task definition); ``max_rel_diff_sig`` repeats it above ``sig_frac`` (default 5 %) of the
    largest trench speed, so that a deep stagnant corner where both speeds are ~1e-5 cm/s cannot
    dominate the report.
    """
    below = (grid.z_um[:, None] < 0.0) & (~grid.solid)
    s1 = np.hypot(flow_a.ux, flow_a.uz)
    s2 = np.hypot(flow_b.ux, flow_b.uz)
    denom = np.maximum(s1, s2)
    valid = below & (denom > 0.0)
    base = {"label": label, "note": note,
            "tau": [_finite(flow_a.info.get("tau")), _finite(flow_b.info.get("tau"))],
            "u_lid_lb": [_finite(flow_a.info.get("u_lid_lb")),
                         _finite(flow_b.info.get("u_lid_lb"))]}
    if not np.any(valid):
        return base | {"n_cells": 0, "max_rel_diff": float("nan"), "mean_rel_diff": float("nan"),
                       "max_rel_diff_sig": float("nan"), "max_abs_diff_cm_s": float("nan"),
                       "re": [], "re_clamped": [], "steps": [], "converged": []}
    rel = np.abs(s1 - s2)[valid] / denom[valid]
    absd = np.abs(s1 - s2)[valid]
    u_max = float(max(s1[valid].max(), s2[valid].max()))
    sig = denom[valid] > float(sig_frac) * u_max
    return base | {
        "n_cells": int(valid.sum()),
        "max_rel_diff": float(rel.max()),
        "mean_rel_diff": float(rel.mean()),
        "max_rel_diff_sig": float(rel[sig].max()) if np.any(sig) else float("nan"),
        "n_cells_sig": int(sig.sum()),
        "sig_threshold_frac": float(sig_frac),
        "max_abs_diff_cm_s": float(absd.max()),
        "u_max_trench_cm_s": u_max,
        "re": [_finite(flow_a.info.get("re")), _finite(flow_b.info.get("re"))],
        "re_clamped": [bool(flow_a.info.get("re_clamped", False)),
                       bool(flow_b.info.get("re_clamped", False))],
        "steps": [int(flow_a.info.get("steps", 0)), int(flow_b.info.get("steps", 0))],
        "converged": [bool(flow_a.converged), bool(flow_b.converged)],
    }


def tau_clamp_check(site, dx_um=4.0, u_lid_values=(0.05, 0.10), cond=None,
                    sig_threshold_frac=0.05, benign_threshold=0.05):
    """Same grid, same physical Re, two lattice lid velocities; velocity difference in the trench.

    ``_scaling`` (``cavity_lbm.py:96``) keeps ``Re = u_lid * N_H / nu_lb`` matched and clamps
    ``tau`` to ``[TAU_MIN, TAU_MAX]``.  For the Trench 1 cavity the required ``nu_lb`` exceeds
    ``TAU_MAX`` for both requested lids, so both are lowered to the *same* lattice lid and the
    requested pair is degenerate (it still checks that the clamp is deterministic).  When that
    happens an extra pair at the same ``Re`` but below the clamp limit (0.5 and 0.9 of the largest
    unclamped lid) is run; there ``tau`` differs and the velocity difference is informative.

    What this measures is only the **velocity field**: ``max_rel_diff`` over all open trench cells,
    ``max_rel_diff_sig`` over cells above ``sig_threshold_frac`` (5 %) of the largest trench speed,
    and ``tau_clamp_benign`` = ``max_rel_diff_sig < benign_threshold``.  The effect of that velocity
    difference on the floor/field **flux ratio** is not measured here - a ~5 % mean / 25 %
    significant-cell velocity change does not translate one-to-one into the transport answer, so
    the boolean should be read as "the clamp alters the flow field", not "the deposit prediction
    moves by 25 %".
    """
    cond = site_conditions(site) if cond is None else cond
    grid = trench_grid(cond["width_um"], cond["depth_um"], height_um=cond["delta_um"],
                       dx_um=float(dx_um))
    u1, u2 = float(u_lid_values[0]), float(u_lid_values[1])
    t0 = time.time()
    fa = solve_flow(grid, cond["shear_rate_s"], nu_cm2_s=NU_CM2_S, u_lid_lb=u1)
    fb = solve_flow(grid, cond["shear_rate_s"], nu_cm2_s=NU_CM2_S, u_lid_lb=u2)
    pairs = [_velocity_pair(fa, fb, grid, f"requested u_lid_lb {u1:g} vs {u2:g}",
                            sig_frac=float(sig_threshold_frac))]

    info = fa.info or {}
    same = abs(_finite(info.get("u_lid_lb")) - _finite((fb.info or {}).get("u_lid_lb"))) < 1e-12
    degenerate = bool(same)
    if degenerate:
        re = _finite(info.get("re"))
        n_h = _finite(info.get("n_fluid_rows"))
        nu_max = (TAU_MAX - 0.5) / 3.0
        u_limit = re * nu_max / n_h if (np.isfinite(re) and np.isfinite(n_h) and n_h > 0) else float("nan")
        if np.isfinite(u_limit) and u_limit > 0.0:
            hi, lo = 0.9 * u_limit, 0.45 * u_limit
            fc = solve_flow(grid, cond["shear_rate_s"], nu_cm2_s=NU_CM2_S, u_lid_lb=lo)
            fd = solve_flow(grid, cond["shear_rate_s"], nu_cm2_s=NU_CM2_S, u_lid_lb=hi)
            pairs.append(_velocity_pair(
                fc, fd, grid, f"unclamped pair, u_lid_lb {lo:.4g} vs {hi:.4g}",
                note="same Re, tau differs because nu_lb is below the TAU_MAX clamp",
                sig_frac=float(sig_threshold_frac)))
    informative = pairs[1:] or pairs
    rels = [_finite(p["max_rel_diff"]) for p in informative]
    rels = [v for v in rels if np.isfinite(v)]
    sigs = [_finite(p.get("max_rel_diff_sig")) for p in pairs]
    sigs = [v for v in sigs if np.isfinite(v)]
    max_rel = float(max(rels)) if rels else float("nan")
    max_rel_sig = float(max(sigs)) if sigs else float("nan")
    benign = bool(np.isfinite(max_rel_sig) and max_rel_sig < float(benign_threshold))
    sig_pct = 100.0 * float(sig_threshold_frac)

    if degenerate and len(pairs) > 1:
        ip = pairs[1]
        interpretation = (
            f"requested u_lid_lb pair is degenerate: both clamp to u_lid_lb = "
            f"{pairs[0]['u_lid_lb'][0]:.4g} with tau = {pairs[0]['tau'][0]:.3g} (TAU_MAX), so the "
            f"requested comparison shows no difference by construction. The unclamped pair at the "
            f"same Re (tau {ip['tau'][0]:.3g} vs {ip['tau'][1]:.3g}) gives a velocity field that "
            f"differs by {100.0 * ip['mean_rel_diff']:.1f} % mean and "
            f"{100.0 * ip['max_rel_diff_sig']:.1f} % on cells above {sig_pct:.0f} % of the trench "
            f"speed (max {100.0 * ip['max_rel_diff']:.1f} % at a near-stagnant corner). The effect "
            f"on the floor/field flux ratio is NOT measured here. tau_clamp_benign = "
            f"{str(benign).lower()} (threshold {100.0 * float(benign_threshold):.0f} % on significant "
            f"cells), i.e. the clamp {'does not measurably alter' if benign else 'does alter'} the "
            f"trench flow at this Re.")
    else:
        interpretation = (
            f"the two lattice settings differ (tau {pairs[0]['tau'][0]:.3g} vs "
            f"{pairs[0]['tau'][1]:.3g}) at the same Re; mean velocity difference "
            f"{100.0 * pairs[0]['mean_rel_diff']:.1f} %, "
            f"{100.0 * pairs[0]['max_rel_diff_sig']:.1f} % above {sig_pct:.0f} % of the trench "
            f"speed. The effect on floor/field is not measured; tau_clamp_benign = "
            f"{str(benign).lower()}.")
    return {
        "id": cond["id"],
        "dx_um": float(dx_um),
        "shear_rate_s": float(cond["shear_rate_s"]),
        "grid_shape": [int(v) for v in grid.shape],
        "degenerate_requested_pair": degenerate,
        "requested_max_rel_diff": _finite(pairs[0]["max_rel_diff"]),
        "max_rel_diff": max_rel,
        "max_rel_diff_sig": max_rel_sig,
        "sig_threshold_frac": float(sig_threshold_frac),
        "benign_threshold": float(benign_threshold),
        "tau_clamp_benign": benign,
        "interpretation": interpretation,
        "pairs": pairs,
        "wall_s": float(time.time() - t0),
    }


# ----------------------------------------------------------------------------------------------
# 4. particles versus continuum in a real trench
# ----------------------------------------------------------------------------------------------
def _k_for_wall_concentration(grid, flow, D_cm2_s, c_wall_target, scheme="upwind", iterations=4,
                              k0=2e-4):
    """k (cm/s) giving the requested mean surface concentration on the field plane.

    ``transport.solve_concentration`` defines ``c_wall = flux_density / k`` for a first-order
    reaction, so the field-plane condition ``c_wall = target`` is the fixed point ``k = flux/k`` ->
    ``k <- region_means(...)["field"] / target``.  ``scheme`` is forwarded, so the converged k is the
    one used by the final continuum solve.
    """
    k = float(k0)
    for _ in range(int(iterations)):
        res = solve_concentration(grid, flow, D_cm2_s=float(D_cm2_s), k_cm_s=k, scheme=scheme)
        flux_field = region_means(res, grid)["field"]
        if not np.isfinite(flux_field) or flux_field <= 0.0:
            break
        k = float(flux_field / float(c_wall_target))
    return k


def particle_cross_check(width_um=120.0, depth_um=60.0, dx_um=6.0, height_um=90.0,
                         delta_um=128.0, c_wall_target=0.7, use_flow=True, scheme="upwind",
                         n_ref_per_um2=0.25, t_burn_s=30.0, t_sample_s=30.0, seed=0):
    """Brownian tracers against the continuum solution in a real 120 x 60 um trench.

    The surface rate constant is chosen by :func:`_k_for_wall_concentration` so that the field-plane
    surface concentration is ``c_wall_target`` (0.7 -> 30 % depletion).  Floor/field and wall/field
    from :func:`platesim.trenchflow.particles.simulate` are compared with
    :func:`platesim.trenchflow.transport.solve_concentration`; the pass rule is
    ``|p - c| <= max(3 * se, 0.08 * |c|)`` with ``se`` the Poisson standard error of the particle
    ratio (``particles.region_means`` divides by ``sqrt(absorbed count)``).

    Chosen settings (the cost is dominated by the diffusion-limited time step, so the sampling time
    has to cover ~1.5 diffusion times of the 90 um column, 30 s burn + 30 s sample; 0.25 particles
    per um2 then gives ~2000 field and ~900 floor absorption events in ~15 s of wall time).  The
    earlier, shorter settings (8 s / 15 s) left the trench un-equilibrated and biased floor/field
    low by 10 %.
    """
    T_C = 22.0
    D, c_bulk, _ = model_constants(T_C)
    shear = float(shear_from_delta(delta_um, D)) if use_flow else 0.0
    grid = trench_grid(float(width_um), float(depth_um), height_um=float(height_um),
                       dx_um=float(dx_um))
    t0 = time.time()
    flow = solve_flow(grid, shear, nu_cm2_s=NU_CM2_S) if use_flow else None
    k = _k_for_wall_concentration(grid, flow, D, c_wall_target, scheme=scheme)
    cres = solve_concentration(grid, flow, D_cm2_s=D, k_cm_s=k, scheme=scheme)
    crm = region_means(cres, grid)
    pres = simulate_particles(grid, flow, D_cm2_s=D, k_cm_s=k,
                              n_ref_per_um2=float(n_ref_per_um2), t_burn_s=float(t_burn_s),
                              t_sample_s=float(t_sample_s), seed=int(seed))
    prm = particle_region_means(pres, grid)

    def _cmp(key):
        c_val = _finite(crm[key])
        p_val = _finite(prm[key])
        a, b = key.split("_over_")
        se_rel = float("nan")
        n_a, n_b = prm.get(a + "_counts"), prm.get(b + "_counts")
        if n_a and n_b and n_a > 0 and n_b > 0 and np.isfinite(p_val):
            se_rel = math.sqrt(1.0 / n_a + 1.0 / n_b)
        se_abs = abs(p_val) * se_rel if np.isfinite(se_rel) else float("nan")
        diff = p_val - c_val if np.isfinite(p_val) and np.isfinite(c_val) else float("nan")
        tol = max(3.0 * se_abs, 0.08 * abs(c_val)) if np.isfinite(diff) else float("nan")
        return {
            "continuum": c_val,
            "particles": p_val,
            "difference": diff,
            "particle_se_rel": se_rel,
            "particle_se_abs": se_abs,
            "tolerance": tol,
            "agree": bool(np.isfinite(diff) and np.isfinite(tol) and abs(diff) <= tol),
        }

    comparisons = {key: _cmp(key) for key in ("floor_over_field", "wall_over_field")}
    # surface depletion actually realised on each region: c_wall = flux / k
    depletion = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in ("field", "floor", "wall"):
            cw = crm[name] / k if np.isfinite(_finite(crm[name])) else float("nan")
            depletion[name] = float(1.0 - cw) if np.isfinite(cw) else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        cw_field = crm["field"] / k
    ok = all(v["agree"] for v in comparisons.values())
    delta_eff = _finite(cres["delta_eff_um"])
    delta_note = (
        f"delta_um = {float(delta_um):g} is a SHEAR input only: G = shear_from_delta(delta_um, D) "
        f"({shear:.2f} 1/s) is chosen so the Leveque layer fits the 90 um domain; it is not the "
        f"Nernst-consistent delta of this case. The delta realised by the continuum solution is "
        f"delta_eff_um = {delta_eff:.1f} um (D (1 - c_wall) / flux on the field plane), and the "
        f"surface depletion actually imposed is c_wall_field = {_finite(crm['field'] / k):.3f}."
    )
    return {
        "settings": {
            "width_um": float(width_um), "depth_um": float(depth_um), "dx_um": float(dx_um),
            "height_um": float(grid.height_um), "delta_um": float(delta_um),
            "delta_note": delta_note,
            "shear_rate_s": float(shear), "use_flow": bool(use_flow), "scheme": scheme,
            "T_C": T_C, "D_cm2_s": float(D), "k_cm_s": float(k),
            "c_wall_target": float(c_wall_target),
            "n_ref_per_um2": float(n_ref_per_um2), "t_burn_s": float(t_burn_s),
            "t_sample_s": float(t_sample_s), "seed": int(seed),
            "grid_shape": [int(v) for v in grid.shape],
            "lbm_steps": int((flow.info or {}).get("steps", 0)) if flow is not None else 0,
            "lbm_converged": bool(flow.converged) if flow is not None else True,
            "lbm_tau": _finite((flow.info or {}).get("tau")) if flow is not None else None,
        },
        "continuum": {key: _finite(crm[key]) for key in
                      ("field", "floor", "wall", "floor_over_field", "wall_over_field")
                      } | {"k_cm_s": float(k), "balance_error": _finite(cres["balance_error"]),
                           "delta_eff_um": _finite(cres["delta_eff_um"])},
        "particles": {key: _finite(prm[key]) for key in
                      ("field", "floor", "wall", "floor_over_field", "wall_over_field")}
                     | {"field_counts": float(prm.get("field_counts", 0.0)),
                        "floor_counts": float(prm.get("floor_counts", 0.0)),
                        "wall_counts": float(prm.get("wall_counts", 0.0)),
                        "n_particles_mean": _finite(pres["n_particles_mean"]),
                        "steps": int(pres["steps"]), "dt_s": float(pres["dt_s"]),
                        "P_abs": _finite(pres["P_abs"]),
                        "particle_wall_s": float(pres["wall_s"])},
        "depletion": depletion,
        "c_wall_field": _finite(cw_field),
        "comparisons": comparisons,
        "agree": bool(ok),
        "wall_s": float(time.time() - t0),
    }


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def run(dx_list=DX_LIST, quick=False, site_ids=None, march_dx=None, march_stages=None,
        tau_site=None, cross_check=True, out_path=None, write=True):
    """Run the convergence study; writes a JSON artifact and returns the same dict.

    Output: ``convergence_quick.json`` when ``quick=True`` (it must never overwrite the production
    ``convergence.json``) and ``convergence.json`` otherwise; ``out_path`` overrides both and
    ``write=False`` writes nothing (tests use that, or ``out_path`` into a tmp directory).

    ``quick=True`` keeps every part to seconds: the narrow site, dx (8, 6, 4), the marching sweep
    once (dx 8, 2 stages) and the tau-clamp check at dx 4 on the narrow site.  The full sweep (both
    sites, dx down to 2, predict_site at dx 6/4/3 x 3/5/9 stages) takes about 78 min on this host.
    """
    t0 = time.time()
    all_sites = json.loads(SEM_JSON.read_text())["sites"]
    if site_ids:
        wanted = {str(s) for s in site_ids}
        sites = [s for s in all_sites if str(s["id"]) in wanted]
        if not sites:
            raise ValueError(f"no site matched {sorted(wanted)}")
    else:
        sites = all_sites
    dxs = [8.0, 6.0, 4.0] if quick else [float(v) for v in dx_list]
    if quick:
        sites = [min(sites, key=lambda s: float(s["width_um"]))]   # the narrow site is the cheap one
        m_dx = [8.0] if march_dx is None else [float(v) for v in march_dx]
        m_st = [2] if march_stages is None else [int(v) for v in march_stages]
    else:
        m_dx = [float(v) for v in (MARCH_DX if march_dx is None else march_dx)]
        m_st = [int(v) for v in (MARCH_STAGES if march_stages is None else march_stages)]
    # plan item 3: the tau-clamp check is done on the NARROW site at dx 4
    tau = (min(sites, key=lambda s: float(s["width_um"])) if tau_site is None
           else next((s for s in all_sites if str(s["id"]) == str(tau_site)), sites[0]))

    notes = [
        f"delta = {DELTA_UM:g} um (central Nernst layer, as validate.py); G = shear_from_delta(delta, D)",
        "j_field = field_um * 1e-4 / (Mfac * hours * 3600), the field current density implied by the "
        "measured field thickness (predict_site, model.py:260)",
        "single quasi-steady stage at deposit = 0 with flow; no parameter is fitted",
        "observed order and Richardson extrapolation use the three finest grids of dx_list; "
        "non-monotone sequences give NaN, not an exception",
        "marching spread = max - min of the final floor_over_field over the (dx, n_stages) grid",
        "LBM tau clamp: _scaling keeps Re matched and clamps tau to [TAU_MIN, TAU_MAX]; the requested "
        "u_lid_lb pair is degenerate in this cavity, so an unclamped pair at the same Re is added",
        "full sweep = both sites, grid dx 8/6/4/3/2, march dx 6/4/3 x stages 3/5/9; --quick "
        "exercises every part cheaply",
    ]

    entries = []
    for site in sites:
        cond = site_conditions(site)
        rows, t_site = [], time.time()
        for dx in dxs:
            row = convergence_row(site, dx, cond)
            rows.append(row)
            print(f"  {cond['id']:16s} dx={dx:>4g} um  floor/field={row['floor_over_field']:.4f}  "
                  f"wall/field={row['wall_over_field']:.4f}  k={row['k_cm_s']:.3e}  "
                  f"lbm={'ok' if row['lbm_converged'] else 'NO'}/{row['lbm_steps']}  "
                  f"pen={row['penetration_um']:.1f} um  {row['wall_s']:.1f} s", flush=True)
        entries.append({
            "id": cond["id"],
            "conditions": cond,
            "rows": rows,
            "richardson_floor_over_field": richardson(rows, "floor_over_field"),
            "richardson_wall_over_field": richardson(rows, "wall_over_field"),
            "wall_s": float(time.time() - t_site),
        })

    march = march_sweep(sites, m_dx, m_st) if m_dx and m_st else None
    tau_res = tau_clamp_check(tau) if tau is not None else None
    particles = particle_cross_check() if cross_check else None

    out = {
        "quick": bool(quick),
        "constants": {
            "delta_um": DELTA_UM,
            "nu_cm2_s": NU_CM2_S,
            "dx_list_um": [float(v) for v in dxs],
            "march_dx_um": [float(v) for v in m_dx],
            "march_stages": [int(v) for v in m_st],
            "sites": [str(s["id"]) for s in sites],
            "tau_site": str(tau.get("id", "")) if tau is not None else None,
        },
        "sites": entries,
        "march": march,
        "tau_clamp": tau_res,
        "particle_cross_check": particles,
        "notes": notes,
        "runtime_s": float(time.time() - t0),
    }
    out = _jsonable(out)
    if write:
        path = Path(out_path) if out_path else (QUICK_JSON if quick else CONVERGENCE_JSON)
        out["output_json"] = str(path)
        # strict JSON: non-finite in-memory values (NaN order/spread) become null on disk
        path.write_text(json.dumps(_strict(out), indent=2, allow_nan=False))
    else:
        out["output_json"] = None
    return out


def _print_table(res):
    """Readable table: per site and dx the ratios, k, LBM bookkeeping, penetration and wall time."""
    c = res["constants"]
    print("=== platesim.trenchflow grid convergence (task 2) ===")
    print(f"delta = {c['delta_um']:g} um   dx = {c['dx_list_um']}   quick = {res['quick']}")
    hdr = (f"{'site':16s} {'dx':>4s} {'flr/fld':>8s} {'wUp/fld':>8s} {'wDn/fld':>8s} "
           f"{'wMean':>8s} {'k [cm/s]':>10s} {'lbm':>7s} {'pen um':>7s} {'wall s':>7s}")
    for site in res["sites"]:
        cond = site["conditions"]
        print("")
        print(f"{site['id']}  ({cond['width_um']:.0f} x {cond['depth_um']:.0f} um, "
              f"{cond['hours']:.1f} h)  G = {cond['shear_rate_s']:.2f} 1/s  "
              f"j_field = {cond['j_field_mA_cm2']:.2f} mA/cm2  T = {cond['T_C']:.1f} C")
        print(hdr)
        for r in site["rows"]:
            lbm = f"{'ok' if r['lbm_converged'] else 'NO'}/{r['lbm_steps']}"
            print(f"{site['id']:16s} {r['dx_um']:4g} {r['floor_over_field']:8.4f} "
                  f"{r['wall_up_over_field']:8.4f} {r['wall_down_over_field']:8.4f} "
                  f"{r['wall_over_field']:8.4f} {r['k_cm_s']:10.3e} {lbm:>7s} "
                  f"{r['penetration_um']:7.2f} {r['wall_s']:7.1f}")
        rch = site["richardson_floor_over_field"]
        print(f"  observed order p = {rch['observed_order']:.3f}   extrapolated floor/field = "
              f"{rch['extrapolated']:.4f}   dx={rch['dx_probe_um']} minus extrapolated = "
              f"{rch['dx_probe_minus_extrapolated']:+.4f}")

    if res.get("march"):
        print("")
        print("--- marching sweep: spread of the final floor/field ---")
        for s in res["march"]["sites"]:
            if s.get("spread_defined") and np.isfinite(_finite(s.get("spread_floor_over_field"))):
                spread = (f"spread = {s['spread_floor_over_field']:.4f}   "
                          f"[{s['min_floor_over_field']:.4f}, {s['max_floor_over_field']:.4f}]")
            else:
                spread = "spread = undefined (fewer than 2 runs)"
            print(f"  {s['id']:16s} {spread}   ({len(s['rows'])} runs, {s['wall_s']:.1f} s)")
            for r in s["rows"]:
                val = r.get("floor_over_field")
                txt = f"{val:.4f}" if val is not None and np.isfinite(val) else str(r.get("error"))
                print(f"    dx {r['dx_um']:4g} um  n_stages {r['n_stages']:2d}  floor/field {txt}"
                      f"  ({r['wall_s']:.1f} s)")

    tau_res = res.get("tau_clamp")
    if tau_res:
        print("")
        print(f"--- LBM tau clamp: {tau_res['id']}, dx = {tau_res['dx_um']:g} um, grid "
              f"{tau_res['grid_shape']} ---")
        for p in tau_res["pairs"]:
            print(f"  {p['label']:52s} tau {p['tau']} u_lid {p['u_lid_lb']}  "
                  f"max rel diff {p['max_rel_diff']:.3e}  mean {p['mean_rel_diff']:.3e}  "
                  f"sig {p.get('max_rel_diff_sig', float('nan')):.3e}  cells {p['n_cells']}")
        print(f"  tau_clamp_benign = {tau_res['tau_clamp_benign']}  "
              f"(sig threshold {100.0 * tau_res.get('sig_threshold_frac', 0.05):.0f} % of trench "
              f"speed, benign threshold {100.0 * tau_res.get('benign_threshold', 0.05):.0f} %)")
        print(f"  {tau_res['interpretation']}")

    pc = res.get("particle_cross_check")
    if pc:
        st = pc["settings"]
        print("")
        print(f"--- particles vs continuum: {st['width_um']:.0f} x {st['depth_um']:.0f} um, dx "
              f"{st['dx_um']:g}, k = {st['k_cm_s']:.3e} cm/s, c_wall field = "
              f"{pc['c_wall_field']:.3f} ---")
        for key, cmp in pc["comparisons"].items():
            print(f"  {key:16s} continuum {cmp['continuum']:.4f}  particles {cmp['particles']:.4f} "
                  f"  diff {cmp['difference']:+.4f}  se {cmp['particle_se_abs']:.4f}  "
                  f"tol {cmp['tolerance']:.4f}  agree {cmp['agree']}")
        print(f"  agree overall: {pc['agree']}   ({pc['wall_s']:.1f} s)")
    print("")
    out_path = res.get("output_json")
    print(f"wrote {out_path}" if out_path else "not written (write=False)")
    print(f"runtime {res['runtime_s']:.1f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Grid convergence, marching and LBM tau-clamp checks for the trench flow model")
    ap.add_argument("--quick", action="store_true",
                    help="one site, dx (8,6,4), march dx 8 x 2 stages, under a minute")
    ap.add_argument("--dx", nargs="+", type=float, default=None, help="override the dx list (um)")
    ap.add_argument("--sites", nargs="+", default=None, help="site ids from data/trench_sem.json")
    ap.add_argument("--march-dx", nargs="+", type=float, default=None)
    ap.add_argument("--march-stages", nargs="+", type=int, default=None)
    ap.add_argument("--no-march", action="store_true", help="skip the predict_site sweep")
    ap.add_argument("--no-particles", action="store_true", help="skip the particle cross-check")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default trenchflow/convergence.json for a full run, "
                         "trenchflow/convergence_quick.json for --quick)")
    args = ap.parse_args(argv)
    res = run(dx_list=args.dx if args.dx else DX_LIST, quick=args.quick, site_ids=args.sites,
              march_dx=([] if args.no_march else args.march_dx),
              march_stages=([] if args.no_march else args.march_stages),
              cross_check=not args.no_particles, out_path=args.out)
    _print_table(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



