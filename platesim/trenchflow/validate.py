"""Validation of the coupled trench growth model (model.py) against the Trench 1 SEM data.

For every site of ``platesim/data/trench_sem.json`` this script

- predicts the deposit with the coupled flow + transport model at the central Nernst layer
  ``delta = 64 um`` [C: coupon calibration 78 +/- 22 um at 20 % stir, scaled by (30/20)^-0.5 for the
  30 % stir of the trench run], with the shear rate from ``shear_from_delta``;
- repeats the prediction without flow at the same delta;
- scans the sensitivity to ``delta`` (shear recomputed from each delta) and to a multiplicative factor on
  the shear rate at fixed delta = 64 um;
- compares ``floor_over_field`` with the measured ratio and its range
  ``[floor lo / field hi, floor hi / field lo]`` [M];
- lists two reference models: Faraday/conformal (ratio 1.0 by definition) and the level-set engine values
  from ``platesim/data/trench_calibration.json`` (``ratio_pred``).

Writes ``validation.json`` next to this file and, for figures, ``out/<site id>_dep<dep>.npz`` snapshots of
the first stage (deposit 0) and of the stage nearest to 40 um deposit.

No parameter of the coupled model is fitted to the trench data; the only measured input is the field
thickness used to set the field current density.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .model import model_constants, predict_site, shear_from_delta, stage_fields

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
SEM_JSON = _ROOT / "data" / "trench_sem.json"
CALIB_JSON = _ROOT / "data" / "trench_calibration.json"
VALIDATION_JSON = _HERE / "validation.json"
OUT_DIR = _HERE / "out"

DELTA_CENTRAL_UM = 64.0
SNAPSHOT_DEP_UM = 40.0


def _jsonable(obj):
    """Recursively convert numpy scalars/tuples/arrays into JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _measured(site):
    """Measured thicknesses, best-estimate ratios and their ranges [M]."""
    fb, flo, fhi = site["field_um"]["best"], site["field_um"]["lo"], site["field_um"]["hi"]
    gb, glo, ghi = site["floor_um"]["best"], site["floor_um"]["lo"], site["floor_um"]["hi"]
    return {
        "field_um": float(fb),
        "floor_um": float(gb),
        "field_range_um": [float(flo), float(fhi)],
        "floor_range_um": [float(glo), float(ghi)],
        "ratio_best": float(gb / fb),
        "ratio_range": [float(glo / fhi), float(ghi / flo)],
    }


def _summary(pred):
    """Compact JSON-safe summary of a predict_site result."""
    return {
        "delta_um": pred["delta_um"],
        "shear_rate_s": pred["shear_rate_s"],
        "dx_um": pred["dx_um"],
        "j_field_mA_cm2": pred["j_field_mA_cm2"],
        "field_um": pred["field_um"],
        "floor_um": pred["floor_um"],
        "wall_up_um": pred["wall_up_um"],
        "wall_down_um": pred["wall_down_um"],
        "floor_over_field": pred["floor_over_field"],
        "wall_asymmetry": pred["wall_asymmetry"],
        "k_cm_s": [st["k_cm_s"] for st in pred["stages"]],
        "eddy_first_stage": pred["stages"][0]["eddy"] if pred["stages"] else None,
        "use_flow": pred["use_flow"],
    }


def _save_snapshot(site, delta_um, shear_rate_s, j_field_mA_cm2, deposit_um, dx_um, T_C, path):
    """Write the flow and concentration fields of one stage to an npz for figures."""
    f = stage_fields(site["width_um"], site["depth_um"], delta_um, shear_rate_s, j_field_mA_cm2,
                     deposit_um, T_C=T_C, dx_um=dx_um, use_flow=True)
    g, flow, res = f["grid"], f["flow"], f["c_result"]
    np.savez_compressed(
        path,
        x_um=g.x_um, z_um=g.z_um, solid=g.solid, ux=flow.ux, uz=flow.uz, c=res["c"],
        k_cm_s=np.float64(f["k_cm_s"]), delta_um=np.float64(delta_um),
        shear_rate_s=np.float64(shear_rate_s),
    )
    return str(path)


def run(delta_values=(45.0, 64.0, 90.0), shear_factors=(0.3, 1.0, 3.0), n_stages=5, quick=False):
    """Run the full validation; writes validation.json (+ out/*.npz) and returns the same dict."""
    t0 = time.time()
    sites = json.loads(SEM_JSON.read_text())["sites"]
    if quick:
        n_stages = 2
    calib = json.loads(CALIB_JSON.read_text()) if CALIB_JSON.exists() else None
    calib_ratio = {row["id"]: float(row["ratio_pred"]) for row in calib["table"]} if calib else {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    notes = [
        f"central Nernst layer delta = {DELTA_CENTRAL_UM:g} um [C]: coupon calibration 78 +/- 22 um at "
        "20 % stir, scaled by (30/20)^-0.5 for the 30 % stir of the trench run",
        "no free parameters are fitted to the trench data; the measured field thickness is an input that "
        "fixes j_field = field_um / (Mfac * hours * 3600)",
        "measured ratio range = [floor lo / field hi, floor hi / field lo] [M]",
        "grid height is the Nernst layer delta_um; it is extended to deposit + 2 dx when the late deposit "
        "approaches that height [J]",
        "Faraday/conformal reference gives ratio 1.0 by definition",
        f"stage march: {n_stages} stages, deposit treated as conformal at 0.5*(1 + previous floor/field) [J]",
    ]
    if quick:
        notes.append("quick mode: 2 stages, dx = 8 um, central and no-flow cases only")

    entries = []
    for site in sites:
        T_C = float(site.get("T_C", 22.0))
        D, c_bulk, _ = model_constants(T_C)
        dx = 8.0 if quick else None
        measured = _measured(site)

        flow = predict_site(site, DELTA_CENTRAL_UM, None, n_stages=n_stages, dx_um=dx, use_flow=True)
        noflow = predict_site(site, DELTA_CENTRAL_UM, None, n_stages=n_stages, dx_um=dx, use_flow=False)

        sens_delta, sens_shear = [], []
        if not quick:
            for d in delta_values:
                p = predict_site(site, float(d), None, n_stages=n_stages, dx_um=dx, use_flow=True)
                sens_delta.append({"delta_um": float(d), "shear_rate_s": p["shear_rate_s"],
                                   **_summary(p)})
            for fac in shear_factors:
                p = predict_site(site, DELTA_CENTRAL_UM, flow["shear_rate_s"] * float(fac),
                                 n_stages=n_stages, dx_um=dx, use_flow=True)
                sens_shear.append({"shear_factor": float(fac), **_summary(p)})

        lo, hi = measured["ratio_range"]
        inside = bool(lo <= flow["floor_over_field"] <= hi)

        # snapshots: initial state (deposit 0) and the stage nearest to 40 um deposit
        paths = []
        stages = flow["stages"]
        near = min(stages, key=lambda st: abs(st["deposit_um"] - SNAPSHOT_DEP_UM))
        for dep in sorted({0.0, round(float(near["deposit_um"]), 3)}):
            path = OUT_DIR / f"{site['id']}_dep{round(dep)}.npz"
            _save_snapshot(site, DELTA_CENTRAL_UM, flow["shear_rate_s"], flow["j_field_mA_cm2"], dep,
                           float(flow["dx_um"]), T_C, path)
            paths.append(str(path))

        entries.append({
            "id": site["id"],
            "width_um": float(site["width_um"]),
            "depth_um": float(site["depth_um"]),
            "hours": float(site["hours"]),
            "measured": measured,
            "faraday_ratio": 1.0,
            "levelset_ratio": calib_ratio.get(site["id"]),
            "noflow": _summary(noflow),
            "flow": _summary(flow),
            "sens_delta": sens_delta,
            "sens_shear": sens_shear,
            "inside_measured_range": inside,
            "snapshots": paths,
        })

    out = {
        "sites": entries,
        "constants": {"D_cm2_s": float(model_constants(22.0)[0]), "c_bulk_mol_cm3": float(model_constants(22.0)[1]),
                      "T_C": 22.0, "n_stages": int(n_stages), "delta_central_um": DELTA_CENTRAL_UM},
        "notes": notes,
        "runtime_s": float(time.time() - t0),
    }
    VALIDATION_JSON.write_text(json.dumps(_jsonable(out), indent=2))
    return _jsonable(out)


def _print_table(res):
    """Readable table of the central predictions, sensitivities and references."""
    c = res["constants"]
    print("=== platesim.trenchflow coupled growth model vs Trench 1 SEM ===")
    print(f"D_c = {c['D_cm2_s']:.4e} cm2/s   c_bulk = {c['c_bulk_mol_cm3']:.4e} mol/cm3   "
          f"T = {c['T_C']:.1f} C   n_stages = {c['n_stages']}")
    for s in res["sites"]:
        m = s["measured"]
        print("")
        print(f"{s['id']}  ({s['width_um']:.0f} x {s['depth_um']:.0f} um, {s['hours']:.1f} h)")
        print(f"  j_field = {s['flow']['j_field_mA_cm2']:.2f} mA/cm2   "
              f"delta = {s['flow']['delta_um']:.0f} um   G = {s['flow']['shear_rate_s']:.1f} 1/s")
        print(f"  measured   ratio = {m['ratio_best']:.3f}   range [{m['ratio_range'][0]:.3f}, "
              f"{m['ratio_range'][1]:.3f}]   (field {m['field_um']:.0f} um, floor {m['floor_um']:.0f} um)")
        for name, val in (("flow", s["flow"]), ("noflow", s["noflow"])):
            print(f"  {name:9s}  ratio = {val['floor_over_field']:.3f}   field {val['field_um']:.1f} um   "
                  f"floor {val['floor_um']:.1f} um   wall up/down {val['wall_up_um']:.1f}/"
                  f"{val['wall_down_um']:.1f} um   asym {val['wall_asymmetry']:.2f}   "
                  f"k {val['k_cm_s'][0]:.2e}..{val['k_cm_s'][-1]:.2e} cm/s")
        ls = s["levelset_ratio"]
        print(f"  refs       faraday 1.000   levelset {'n/a' if ls is None else f'{ls:.3f}'}   "
              f"inside measured range: {'yes' if s['inside_measured_range'] else 'no'}")
        if s["sens_delta"]:
            txt = "  ".join(f"{d['delta_um']:.0f}um:{d['floor_over_field']:.3f}" for d in s["sens_delta"])
            print(f"  sens delta {txt}")
        if s["sens_shear"]:
            txt = "  ".join(f"x{d['shear_factor']:.1f}:{d['floor_over_field']:.3f}" for d in s["sens_shear"])
            print(f"  sens shear {txt}")
    print("")
    print(f"wrote {VALIDATION_JSON}")
    print(f"runtime {res['runtime_s']:.1f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate the coupled trench growth model against Trench 1 SEM")
    ap.add_argument("--quick", action="store_true", help="2 stages, dx 8 um, central + no-flow only")
    args = ap.parse_args(argv)
    res = run(quick=args.quick)
    _print_table(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
