"""Grid and time-step convergence of the level-set feature-fill engine (additive-free Trench 1).

Per case (the two Trench 1 SEM sites, pulse-reverse 15 mA/cm2 net, efficiency 0.82, delta 64 um) and
per grid spacing dx this reports the field and floor thickness, floor/field, two distinct Faraday
numbers, the Cu2+ mass balance (result["cu_balance_rel"]) and the wall clock:
  faraday_rel_err       engine numerical check: deposited volume vs charge passed
                        (result["meta"]["faraday"], audit 0.2-0.8 %);
  field_vs_faraday_rel  physical deficit of the field thickness against the ideal planar deposit
                        Mfac * j_net * t (faraday_um).
The two checks are complementary: faraday_rel_err is the level-set deposit vs charge, while
cu_balance_rel is the transport flux through the top plane vs charge (it never uses the level set).
From the three finest grids it reports the observed order and the Richardson-extrapolated
floor/field, and the dx = 4 value minus that limit.  One time-step study repeats the narrow site at
dx = 4 with the CFL number halved and quartered.

CLI:  python -m platesim.feature.convergence [--quick] [--dx 8 6 4 3] [--cases trench1_wide ...]
      [--out platesim/feature/convergence.json]

--quick runs dx = (8, 6) for 0.5 h (well under a minute) and skips the time-step study; the full
sweep is 4.5 h at dx = (8, 6, 4, 3) and is meant to be run once per physics change.

Deviation from "reuse calibrate.predict": predict() accepts the dx override but drops
cu_balance_rel and Params.cfl, so _predict_full() below rebuilds exactly the same Params / recipe
from the same calibrate helpers (calibrate._full_theta, calibrate._build_recipe, calibrate.F_OPEN)
and calls run_case(..., return_sim=True) once per (case, dx).  tests/test_feature_convergence.py
asserts that it reproduces calibrate.predict() on field and floor.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import brentq

from platesim.feature import calibrate as cal
from platesim.feature.engine import Params, run_case
from platesim.feature.engine.params import derive

EFFICIENCY = 0.82
DELTA_UM = 64.0
HOURS = 4.5
THETA = dict(efficiency=EFFICIENCY, delta_um=DELTA_UM)     # additive-free; i0_u, k_mix from FREE_PARAMS
CFL_LIST = (0.5, 0.25, 0.125)                              # base / halved / quartered
TSTEP_SITE = "trench1_narrow"
TSTEP_DX = 4.0
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(PKG_DIR, "convergence.json")


# ---------------------------------------------------------------------------------- one run
def _predict_full(site: dict, theta: dict, dx_um: float | None, cfl: float = 0.5) -> dict:
    """calibrate.predict()'s run, plus cu_balance_rel and Params.cfl (predict() exposes neither).

    Params / recipe construction is copied verbatim from calibrate.predict() so the two agree on
    field and floor; there is no cache here, one run_case call per (case, dx)."""
    th = cal._full_theta(theta)
    hours = float(site["hours"])
    rec, warnings = cal._build_recipe(site, use_pr=True)
    p = Params(geometry="trench", H_um=float(site["depth_um"]), suppressor=False,
               efficiency=th["efficiency"], delta_um=th["delta_um"], i0_u=th["i0_u"], k_mix=th["k_mix"],
               dx_um=None if dx_um is None else float(dx_um), cfl=float(cfl),
               # [C] headroom as in calibrate.predict: ~21 um/h of field growth + 40 um of margin
               max_overburden_um=40.0 + 21.0 * hours)
    sid = str(site.get("id") or f"{site.get('width_um')}x{site.get('depth_um')}um")
    dx_tag = "default" if dx_um is None else f"{float(dx_um):g}"      # dx_um=None -> engine rule
    t0 = time.time()
    res, sim = run_case(float(site["width_um"]), float(site["T_C"]), rec, p, f_open=cal.F_OPEN,
                        label=f"{sid}_dx{dx_tag}_cfl{float(cfl):g}", out=None, return_sim=True)
    wall = time.time() - t0

    field = cal._last(res["series"], "face_thickness_um")
    floor = cal._last(res["series"], "floor_height_um")
    if field is None:
        field = float(res["result"].get("overburden_um") or 0.0)
        warnings.append("field height unavailable from the series; used result['overburden_um']")
    if floor is None:
        floor = 0.0
        warnings.append("floor height unavailable from the run; used 0 um")
    stop = res["result"].get("stop_reason")
    if stop == "overburden_exceeded":
        warnings.append("run stopped at the overburden limit: predicted thickness is a lower bound")

    t_s = float(res["series"]["t_s"][-1])
    mfac = derive(p, float(site["T_C"])).Mfac
    far = mfac * float(site["j_mA_cm2"]) * 1e-3 * t_s / 1e-4          # pure-Faraday thickness, um
    # engine numerical Faraday check (deposit volume vs charge passed), as quoted in the audit (0.2-0.8 %)
    fa = res["meta"].get("faraday") or {}
    dep, chg = float(fa.get("deposited_volume_cm3") or 0.0), float(fa.get("charge_volume_cm3") or 0.0)
    far_err = float(dep / chg - 1.0) if chg > 0.0 else None
    cu_rel = res["result"].get("cu_balance_rel")
    if cu_rel is None:
        cu_rel = getattr(sim, "cu_balance_rel", None)
    return dict(case=sid, dx_um=float(sim.g.dx_um), dx_requested=None if dx_um is None else float(dx_um),
                cfl=float(cfl), hours=hours, t_end_s=t_s,
                grid=[int(sim.g.nr), int(sim.g.nz)], steps=int(res["meta"]["steps"]),
                field_um=float(field), floor_um=float(floor),
                ratio=float(floor / field) if field > 0 else float("nan"),
                faraday_um=float(far),
                faraday_rel_err=far_err,                                  # engine: deposit vs charge
                field_vs_faraday_rel=float(field / far - 1.0) if far > 0 else None,   # physical deficit
                cu_balance_rel=None if cu_rel is None else float(cu_rel),
                stop_reason=stop, outcome=res["result"].get("outcome"),
                wall_s=float(wall), warnings=list(warnings))


# ---------------------------------------------------------------------------------- convergence
def _order_extrap(dx, f) -> tuple[float | None, float | None]:
    """Observed order p and the Richardson limit f(dx -> 0) from three grids (any refinement ratio).

    f(dx) = f0 + C dx^p; p solves (f1-f3)/(f2-f3) = (dx1^p - dx3^p)/(dx2^p - dx3^p), then
    f0 = f3 - (f2-f3) dx3^p / (dx2^p - dx3^p).  None when the three values are not monotone.
    """
    dx = np.asarray(dx, float)
    f = np.asarray(f, float)
    if len(f) < 3 or not np.all(np.isfinite(f)) or dx[-2] == dx[-1]:
        return None, None
    d1, d2, d3 = dx[-3], dx[-2], dx[-1]
    f1, f2, f3 = f[-3], f[-2], f[-1]
    if f1 == f3 or f2 == f3:
        return None, None

    def g(p):
        return (f1 - f3) / (f2 - f3) - (d1 ** p - d3 ** p) / (d2 ** p - d3 ** p)

    lo, hi = 0.05, 10.0
    try:
        if g(lo) * g(hi) > 0:
            return None, None
        p = float(brentq(g, lo, hi))
    except (ValueError, ZeroDivisionError):
        return None, None
    c = (f2 - f3) / (d2 ** p - d3 ** p)
    return p, float(f3 - c * d3 ** p)


def _case_summary(rows: list[dict]) -> dict:
    """Observed order + Richardson limit per quantity, from the three finest grids present."""
    def req(r):
        return r["dx_requested"] if r.get("dx_requested") is not None else r["dx_um"]

    rows = sorted(rows, key=lambda r: -req(r))                  # coarse -> fine
    dx = [r["dx_um"] for r in rows]                             # effective dx (grid-clamped)
    out: dict = {"dx_requested": [req(r) for r in rows], "dx_um_effective": dx, "n_grids": len(rows)}
    for key in ("field_um", "floor_um", "ratio"):
        p, lim = _order_extrap(dx, [r[key] for r in rows])
        gap = None
        if lim is not None:
            at4 = next((r[key] for r in rows if abs(req(r) - 4.0) < 1e-9), None)
            gap = None if at4 is None else float(at4 - lim)
        out[key] = {"order": p, "richardson": lim, "dx4_minus_extrap": gap}
    if len(rows) < 3:
        out["note"] = "fewer than three grids: no order / Richardson estimate"
    elif out["ratio"]["order"] is None:
        out["note"] = ("the three finest grids are not consistent with one power law "
                       "f0 + C dx^p (order / Richardson not resolvable at this precision)")
    else:
        out["note"] = None
    return out


def _timestep_study(site: dict, dx_um: float) -> dict:
    """CFL halved and quartered at one grid: does floor/field move with the time step?"""
    rows = [_predict_full(site, THETA, dx_um, cfl=cfl) for cfl in CFL_LIST]
    base = rows[0]["ratio"]
    for r in rows:
        r["ratio_rel_to_base"] = float(r["ratio"] / base - 1.0) if base else None
    spread = max(r["ratio"] for r in rows) - min(r["ratio"] for r in rows)
    return dict(case=str(site.get("id")), dx_um=float(rows[0]["dx_um"]), cfl_list=list(CFL_LIST),
                runs=rows, ratio_spread=float(spread),
                converged=bool(spread < 0.01),
                note="base cfl = 0.5; halved 0.25; quartered 0.125 (Params.cfl)")


# ---------------------------------------------------------------------------------- driver
def _select_sites(cases) -> list[dict]:
    known = {str(s.get("id")): s for s in cal.load_sites()}
    if cases is None:
        return list(known.values())
    if isinstance(cases, (str, dict)):
        cases = [cases]
    out = []
    for c in cases:
        if isinstance(c, dict):
            out.append(c)
        elif str(c) in known:
            out.append(known[str(c)])
        else:
            raise ValueError(f"unknown case {c!r}; known: {sorted(known)}")
    return out


def _clean(obj):
    """numpy -> python so json.dump works without a default hook."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def run(cases=None, dx_list=(8, 6, 4, 3), quick=False, out=None) -> dict:
    """Sweep dx for the Trench 1 sites; quick=True = dx (8, 6) and 0.5 h (under one minute).

    Writes the JSON to `out` when given (the CLI passes DEFAULT_OUT; run(quick=True) in the tests
    passes nothing so the production JSON is not overwritten).
    """
    t_all = time.time()
    sites = _select_sites(cases)
    dxs = (8, 6) if quick else tuple(float(d) for d in dx_list)
    hours = 0.5 if quick else HOURS
    payload: dict = {
        "generated_unix": float(t_all),
        "quick": bool(quick),
        "conditions": dict(efficiency=EFFICIENCY, delta_um=DELTA_UM, hours=hours, suppressor=False,
                           f_open=cal.F_OPEN, waveform="pulse-reverse, 15 mA/cm2 net",
                           theta=cal._full_theta(THETA), dx_list=[float(d) for d in dxs]),
        "runs": {}, "convergence": {}, "timestep": {}, "wall_s": None,
    }
    for site in sites:
        s = dict(site)
        s["hours"] = hours
        sid = str(s.get("id") or f"{s.get('width_um')}x{s.get('depth_um')}um")
        rows = [_predict_full(s, THETA, float(dx)) for dx in dxs]
        payload["runs"][sid] = rows
        payload["convergence"][sid] = _case_summary(rows)

    if quick:
        payload["timestep"] = dict(skipped=True, reason="--quick: time-step study runs only in the full sweep")
    else:
        try:
            site = _select_sites([TSTEP_SITE])[0]
        except ValueError:
            site = sites[0]
        payload["timestep"] = _timestep_study(dict(site, hours=hours), TSTEP_DX)

    payload["wall_s"] = float(time.time() - t_all)
    payload = _clean(payload)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
    return payload


# ---------------------------------------------------------------------------------- CLI
_HEAD_ITEMS = (("case", 16), ("dx", 5), ("cfl", 6), ("field", 9), ("floor", 9), ("fl/fi", 7),
               ("F_err%", 8), ("field-F%", 9), ("cubal%", 8), ("wall_s", 7))


def _row(cells) -> str:
    parts = []
    for i, (name, width) in enumerate(_HEAD_ITEMS):
        cell = cells[i]
        parts.append(str(cell).ljust(width) if i < 2 else str(cell).rjust(width))
    return "  ".join(parts).rstrip()


def _print_table(payload: dict) -> None:
    print(_row([n for n, _ in _HEAD_ITEMS]))
    for sid, rows in payload["runs"].items():
        for r in rows:
            cells = [sid, f"{r['dx_requested'] if r.get('dx_requested') else r['dx_um']:g}", f"{r['cfl']:g}",
                     f"{r['field_um']:8.2f}", f"{r['floor_um']:8.2f}",
                     f"{r['ratio']:7.4f}",
                     "n/a" if r["faraday_rel_err"] is None else f"{100 * r['faraday_rel_err']:8.3f}",
                     "n/a" if r["field_vs_faraday_rel"] is None else f"{100 * r['field_vs_faraday_rel']:9.2f}",
                     "n/a" if r["cu_balance_rel"] is None else f"{100 * r['cu_balance_rel']:8.3f}",
                     f"{r['wall_s']:7.1f}"]
            print(_row(cells))
    for sid, conv in payload["convergence"].items():
        r = conv["ratio"]
        print(f"{sid}: order floor/field = {r['order']}  richardson = {r['richardson']}  "
              f"dx4 - extrap = {r['dx4_minus_extrap']}")
    ts = payload["timestep"]
    if ts.get("skipped"):
        print(f"time-step study: skipped ({ts['reason']})")
    else:
        print(f"time-step study {ts['case']} dx={ts['dx_um']:g}: ratio spread {ts['ratio_spread']:.3e} "
              f"(converged={ts['converged']})")
        for tr in ts["runs"]:
            print(f"  cfl={tr['cfl']:g}  field={tr['field_um']:.3f}  floor={tr['floor_um']:.3f}  "
                  f"ratio={tr['ratio']:.4f}  wall={tr['wall_s']:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m platesim.feature.convergence",
        description="Grid / time-step convergence of the level-set feature-fill engine (Trench 1, additive-free).")
    ap.add_argument("--quick", action="store_true",
                    help="dx (8, 6), 0.5 h, no time-step study (< 1 min)")
    ap.add_argument("--dx", nargs="+", type=float, default=list((8, 6, 4, 3)), help="grid spacings, um")
    ap.add_argument("--cases", nargs="+", default=None, help="site ids from data/trench_sem.json")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output JSON (default {DEFAULT_OUT})")
    a = ap.parse_args(argv)
    payload = run(cases=a.cases, dx_list=tuple(a.dx), quick=a.quick, out=a.out)
    _print_table(payload)
    print(f"wrote {a.out}  ({payload['wall_s']:.1f} s)")
    return payload


if __name__ == "__main__":
    main()
