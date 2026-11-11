"""Validation suite for the feature-fill port (trench geometry + via regression).

Run from the platesim project root:   python -m platesim.feature.validate_trench [faraday seam regression]

  faraday    (a) trench DC conformal (no suppressor): deposited area per unit length vs integrated
               charge within 2 %  (research README V1 analogue, planar units)
  seam       (b) conformal trench, no suppressor: the pocket pinches when the walls have grown
               ~w/2 each (face thickness at closure ~ w/2), classified 'seam'  (README V4 analogue)
  regression (c) axisymmetric via, one short case (d = 100 um, 40 C, plan_ramp truncated to 20 min):
               the ported engine matches the original research engine (imported read-only from
               D:/Workspace/Semiconductors/research/via_fill_sim) within 0.5 % in fill fraction

Numbers are printed;  --json PATH  also writes them as JSON.  The same three checks back
tests/test_feature.py (the seam case is marked slow there)."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

from .engine import Params, dc, run_case
from .engine.recipes import Recipe, plan_ramp
from .engine.sim import ViaSim

# Optional: path to the original (private) via-fill research engine for the regression check.
# Leave unset to skip that check.
RESEARCH_DIR = os.environ.get("PLATESIM_RESEARCH_DIR", "")

# shared test-case parameters (documented for the report)
FARADAY_CASE = dict(d_um=30.0, T_C=25.0, j_mA_cm2=2.0, tmax_h=1.0, f_open=0.2)   # no suppressor, DC
SEAM_CASE = dict(d_um=30.0, T_C=25.0, j_mA_cm2=2.5, tmax_h=45.0, f_open=0.2)     # no suppressor, ohmic off
VIA_REG_CASE = dict(d_um=100.0, T_C=40.0, t_end_s=1200.0, f_open=0.05)           # plan_ramp truncated 20 min


def truncate(rec: Recipe, t_end_s: float) -> Recipe:
    """Copy of a recipe cut at t_end_s (works for the ported and the research Recipe classes)."""
    out, t = [], 0.0
    for s in rec.steps:
        if t >= t_end_s:
            break
        out.append(dataclasses.replace(s, duration_s=min(s.duration_s, t_end_s - t)))
        t += s.duration_s
    return type(rec)(rec.name + f"_trunc{t_end_s:g}", out)


# ----------------------------------------------------------------------------------------------- (a) Faraday
def v_faraday() -> dict:
    """Deposited area per unit length vs M/(2 F rho) x integrated net charge, DC conformal trench."""
    c = FARADAY_CASE
    p = Params(geometry="trench", suppressor=False)
    t0 = time.time()
    res = run_case(c["d_um"], c["T_C"], dc(c["j_mA_cm2"], c["tmax_h"]), p,
                   f_open=c["f_open"], label="trench_faraday")
    fd = res["meta"]["faraday"]
    rel = fd["deposited_volume_cm3"] / fd["charge_volume_cm3"] - 1.0
    out = dict(test="faraday_trench", rel_err=float(rel), target=0.02,
               deposited_per_length_cm2=fd["deposited_volume_cm3"],
               charge_per_length_cm2=fd["charge_volume_cm3"],
               max_balance_error=res["meta"]["max_balance_error"],
               steps=res["meta"]["steps"], stop=res["result"]["stop_reason"],
               wall_s=round(time.time() - t0, 2), **c)
    out["pass"] = bool(abs(rel) < 0.02)
    return out


# ----------------------------------------------------------------------------------------------- (b) seam
def v_seam() -> dict:
    """Conformal trench (no suppressor, ohmic off) must pinch as a seam when the walls have grown
    ~w/2 = a each (face thickness at closure ~ w/2, as in README V4 for the via)."""
    c = SEAM_CASE
    p = Params(geometry="trench", suppressor=False, ohmic=False)
    t0 = time.time()
    res = run_case(c["d_um"], c["T_C"], dc(c["j_mA_cm2"], c["tmax_h"]), p,
                   f_open=c["f_open"], label="trench_seam")
    r, s = res["result"], res["series"]
    w2 = c["d_um"] / 2.0                                    # wall growth at pinch ~ w/2
    face_at_closure = None
    if r["voids"]:
        t_void = r["voids"][0]["t_formed_s"]
        ts = [t for t in s["t_s"] if t is not None]
        fs = [f for t, f in zip(s["t_s"], s["face_thickness_um"]) if t is not None]
        face_at_closure = float(np.interp(t_void, ts, fs))
    big = max(r["voids"], key=lambda v: v["volume_frac"]) if r["voids"] else None
    out = dict(test="seam_trench", target_wall_um=w2,
               face_thickness_at_closure_um=None if face_at_closure is None else round(face_at_closure, 2),
               outcome=r["outcome"], stop=r["stop_reason"],
               voids=r["voids"], fill_final=r["fill_fraction_final"],
               steps=res["meta"]["steps"], wall_s=round(time.time() - t0, 2), **c)
    grown_ok = face_at_closure is not None and 0.6 * w2 <= face_at_closure <= 1.4 * w2
    out["pass"] = bool(big is not None and big["cls"] == "seam" and grown_ok)
    return out


# ----------------------------------------------------------------------------------------------- (c) via regression
def research_modules():
    """Import the original research engine (read-only) as top-level modules; None if unavailable."""
    if not os.path.isdir(RESEARCH_DIR):
        return None
    if RESEARCH_DIR not in sys.path:
        sys.path.insert(0, RESEARCH_DIR)
    import params as r_params                      # noqa: F401  (research engine)
    import recipes as r_recipes
    import sim as r_sim
    return r_params, r_recipes, r_sim


def v_via_regression() -> dict:
    """Ported engine vs research engine, axisymmetric via, short plan_ramp: fill fraction within 0.5 %."""
    mods = research_modules()
    if mods is None:
        return {"test": "via_regression", "pass": None, "skipped": f"missing {RESEARCH_DIR}"}
    r_params, r_recipes, r_sim = mods
    c = VIA_REG_CASE

    t0 = time.time()
    sim_r = r_sim.ViaSim(c["d_um"], c["T_C"], r_params.Params(), c["f_open"],
                         truncate(r_recipes.plan_ramp(c["T_C"]), c["t_end_s"]))
    raw_r = sim_r.run()
    wall_r = time.time() - t0

    t0 = time.time()
    sim_p = ViaSim(c["d_um"], c["T_C"], Params(), c["f_open"],
                   truncate(plan_ramp(c["T_C"]), c["t_end_s"]))
    raw_p = sim_p.run()
    wall_p = time.time() - t0

    t_r = np.asarray(raw_r["series"]["t_s"], float)
    f_r = np.asarray(raw_r["series"]["fill_fraction"], float)
    f_p = np.asarray(raw_p["series"]["fill_fraction"], float)
    at = [float(np.interp(x, t_r, f_r)) for x in (600.0, c["t_end_s"])]
    at_p = [float(np.interp(x, np.asarray(raw_p["series"]["t_s"], float), f_p))
            for x in (600.0, c["t_end_s"])]
    rels = [p_ / r_ - 1.0 if r_ > 0 else 0.0 for p_, r_ in zip(at_p, at)]
    rel_final = raw_p["fill_final"] / raw_r["fill_final"] - 1.0
    out = dict(test="via_regression", research_fill_final=raw_r["fill_final"],
               port_fill_final=raw_p["fill_final"], rel_err_final=float(rel_final),
               fill_at_600s=[round(x, 6) for x in at], fill_at_tend=[round(x, 6) for x in at_p],
               rel_err_series=[float(x) for x in rels],
               steps_research=raw_r["steps"], steps_port=raw_p["steps"],
               wall_research_s=round(wall_r, 2), wall_port_s=round(wall_p, 2),
               target=0.005, **c)
    out["pass"] = bool(abs(rel_final) < 0.005 and max(abs(x) for x in rels) < 0.005)
    return out


TESTS = dict(faraday=v_faraday, seam=v_seam, regression=v_via_regression)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tests", nargs="*", default=[], help="subset of: faraday seam regression (default all)")
    ap.add_argument("--json", default="", help="write results to this JSON path")
    a = ap.parse_args(argv)
    names = [n for n in a.tests if n in TESTS] or list(TESTS)
    results, ok = {}, True
    for name in names:
        t0 = time.time()
        r = TESTS[name]()
        r["test_wall_s"] = round(time.time() - t0, 2)
        results[name] = r
        flag = r.get("pass")
        if flag is not None:                     # None = skipped (research reference unavailable)
            ok = ok and bool(flag)
        print(f"== {name}: pass={flag}")
        print(json.dumps(r, indent=1, default=float))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print("wrote", a.json)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
