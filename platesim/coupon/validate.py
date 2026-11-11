"""Leave-one-out / leave-setup-out validation of the coupon calibration and a
blind prediction of the trench coupon.

`calibrate.calibrate_joint()` fits 6 parameters to the 8 planar DOE runs plus the
position-resolved SEM points and saves calibration.json.  That fit has never been
tested on data it did not see.  This module does three things:

1. leave_one_out(): for every DOE run i, refit on the other 7 runs (and only the
   SEM points of those runs), then predict run i.  Nothing of run i enters the fit.
2. leave_setup_out(): fit on the 6 "base" runs only and predict the two "imp1"
   runs (taped + baffles + 5x5 cm anode).  The reverse direction is not possible:
   2 runs cannot support a 6-parameter fit.
3. trench_coupon_prediction(): blind prediction for the trench coupon "Trench 1"
   [M] at 15 mA/cm2, 270 min, 30 % stir, 22 C with the SAVED calibration (no
   fitting), compared with the measured field thicknesses of that coupon.

Nothing here writes calibration.json: every fit uses save=False.

Units: cm, s, A inside the pipeline; `_um`, `_mA_cm2`, `_min` suffixes otherwise.
Row residuals follow the calibrate.py convention: resid = measured - predicted.
Tags: [M] measured (team data), [J] judgement, [C] calc.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..efficiency.model import calibration_runs
from .calibrate import calibrate_joint, predict_table, theta_from_calibration
from .pipeline import DEFAULT_THETA, load_sem_points, simulate_run

VALIDATION_PATH = Path(__file__).resolve().parent / "validation.json"

THETA_KEYS = ("delta0_um", "n_exp", "rpm_per_pct", "f_back", "L0_um",
              "kappa_scale", "sigma_um", "flow_dir")

# trench coupon "Trench 1" [M]: 15 mA/cm2, 4.5 h, 30 % stir, 22 C, imp1 hardware
TRENCH_RUN = {"setup": "imp1", "j_mA_cm2": 15.0, "t_min": 270.0,
              "stir_pct": 30.0, "T_C": 22.0}
TRENCH_MEASURED = {
    "min_um": 39.5,
    "max_um": 91.0,
    # [M] 39.5 um at a no-trench site; 69-80 um next to the narrow trench,
    # 79-91 um next to the wide trench -> representative sites below
    "sites_um": [39.5, 69.0, 82.0],
    "faraday_um": 89.3,          # [C] 100 % Faraday at 15 mA/cm2, 270 min
}
TRENCH_SITE_LABELS = ["no_trench", "narrow_trench", "wide_trench"]


# ---------------------------------------------------------------------------
# small statistics helpers (also used by tests/test_coupon_validate.py)
# ---------------------------------------------------------------------------
def _finite(values) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            out.append(v)
    return out


def rms(values) -> float | None:
    """RMS of the finite values (None when nothing is left)."""
    vals = _finite(values)
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else None


def max_abs(values) -> float | None:
    """Largest |value| over the finite values (None when nothing is left)."""
    vals = _finite(values)
    return float(np.max(np.abs(vals))) if vals else None


def mean_bias(values) -> float | None:
    """Mean of the finite values (residual convention: measured - predicted)."""
    vals = _finite(values)
    return float(np.mean(vals)) if vals else None


def theta_slim(theta: dict | None) -> dict:
    """The physics-facing part of theta, JSON-safe (drops bookkeeping keys)."""
    th = {**DEFAULT_THETA, **(theta or {})}
    return {k: (th[k] if isinstance(th[k], str) else float(th[k]))
            for k in THETA_KEYS if k in th}


def make_row(run: dict, pred_um: float, pred_full_um: float | None = None,
             theta_fit: dict | None = None, budget_hit: bool = False) -> dict:
    """One validation row.  resid = measured - predicted (calibrate.py convention)."""
    meas = float(run["actual_um"])
    pred = float(pred_um)
    return {
        "id": int(run["id"]),
        "setup": run.get("setup"),
        "j_mA_cm2": float(run["j_mA_cm2"]),
        "t_min": float(run["t_min"]),
        "stir_pct": (None if run.get("stir_pct") is None
                     else float(run["stir_pct"])),
        "measured_um": meas,
        "pred_loo_um": pred,
        "pred_full_um": None if pred_full_um is None else float(pred_full_um),
        "resid_loo_um": meas - pred,
        "resid_full_um": (None if pred_full_um is None
                          else meas - float(pred_full_um)),
        "theta_loo": theta_slim(theta_fit) if theta_fit is not None else None,
        "budget_hit": bool(budget_hit),
    }


def _progress_printer(prefix: str = ""):
    """Compact stderr progress: only milestones, not every residual evaluation."""
    t0 = time.perf_counter()

    def _p(frac, msg=""):
        if msg.startswith("joint fit") or msg.startswith("fit done"):
            print(f"{prefix}[{100.0 * float(frac):5.1f}% {time.perf_counter() - t0:6.0f} s] "
                  f"{msg}", flush=True)

    return _p


def _full_predictions(runs: list[dict], resolution: str) -> tuple[dict, dict]:
    """(id -> predicted mean um, theta) from the SAVED full-data calibration."""
    theta, cal = theta_from_calibration()
    doe_rows, _ = predict_table(theta, runs=runs, sem_points=[], resolution=resolution)
    preds = {int(r["id"]): float(r["predicted_um"]) for r in doe_rows}
    return preds, {"theta": theta, "calibration_found": cal is not None,
                   "calibration_created": (cal or {}).get("created")}


# ---------------------------------------------------------------------------
# 1. leave-one-out
# ---------------------------------------------------------------------------
def leave_one_out(runs: list[dict] | None = None,
                  sem_points: list[dict] | None = None,
                  max_nfev: int = 25,
                  time_budget_s: float = 300.0,
                  resolution: str = "coarse",
                  progress=None,
                  ids: list[int] | None = None) -> dict:
    """Refit without run i, predict run i (DOE mean thickness). One fit per run.

    The held-out run is removed from BOTH data sets: its DOE row and every SEM
    point whose `run_id` is i (SEM points carry the run they belong to).  The
    fit starts from the saved full-data theta, so any difference between
    pred_loo_um and pred_full_um is the information run i carried.
    """
    t_start = time.perf_counter()
    runs = list(runs) if runs is not None else calibration_runs()
    sem_points = load_sem_points() if sem_points is None else list(sem_points)
    if not runs:
        raise ValueError("no DOE runs with measured thickness")
    wanted = None if ids is None else {int(i) for i in ids}
    targets = [r for r in runs if wanted is None or int(r["id"]) in wanted]
    if not targets:
        raise ValueError(f"none of ids={sorted(wanted or [])} are in the calibration runs")

    pred_full, full_info = _full_predictions(runs, resolution)
    theta0 = full_info["theta"]

    rows: list[dict] = []
    fit_log: list[dict] = []
    notes: list[str] = []
    for r in targets:
        rid = int(r["id"])
        train_runs = [x for x in runs if int(x["id"]) != rid]
        train_sem = [p for p in sem_points if p.get("run_id") != rid]
        if progress:
            progress(0.0, f"LOO run {rid}: refit on {len(train_runs)} DOE runs + "
                          f"{len(train_sem)} SEM points")
        t_fit = time.perf_counter()
        fit = calibrate_joint(runs=train_runs, sem_points=train_sem, theta0=theta0,
                              resolution=resolution, max_nfev=max_nfev,
                              time_budget_s=time_budget_s, uncertainties=False,
                              save=False, progress=progress)
        theta_fit = {**DEFAULT_THETA, **fit["theta"]}
        sim = simulate_run(r, theta_fit, resolution=resolution)
        row = make_row(r, sim["mean_thickness_um"], pred_full.get(rid),
                       theta_fit, fit.get("budget_hit", False))
        row["fit_rms_um"] = fit.get("rms_um")
        row["fit_n_fev"] = fit.get("n_fev")
        row["fit_runtime_s"] = round(time.perf_counter() - t_fit, 1)
        rows.append(row)
        fit_log.append({"id": rid, "rms_um": fit.get("rms_um"),
                        "n_fev": fit.get("n_fev"),
                        "runtime_s": row["fit_runtime_s"],
                        "budget_hit": fit.get("budget_hit", False),
                        "error": fit.get("error")})

    notes = [
        "resid = measured - predicted (same sign convention as calibrate.py)",
        "each fit starts from the saved full-data theta; the held-out run is absent "
        "from both the DOE rows and the SEM points (filtered on run_id) [C]",
        "one SEM point set is small (8 points: 6 on run 1, 2 on run 2): removing "
        "run 1 also removes its 6 SEM constraints, so that LOO theta is weaker "
        "than the full fit [J]",
        "means only: the LOO test does not check the position-resolved prediction "
        "shape, only the coupon mean thickness [J]",
    ]
    if not full_info["calibration_found"]:
        notes.append("no valid calibration.json found: pred_full_um was computed "
                     "with DEFAULT_THETA, not the saved fit")
    if any(r["budget_hit"] for r in rows):
        notes.append("at least one LOO fit hit its time/solve budget - its theta is "
                     "the best point seen, not a converged fit")
    if any(f["error"] for f in fit_log):
        notes.append("at least one LOO fit reported a least_squares error (see fit_log)")

    return {
        "rows": rows,
        "rms_loo_um": rms([r["resid_loo_um"] for r in rows]),
        "rms_full_um": rms([r["resid_full_um"] for r in rows]),
        "max_abs_loo_um": max_abs([r["resid_loo_um"] for r in rows]),
        "mean_bias_loo_um": mean_bias([r["resid_loo_um"] for r in rows]),
        "faraday_rms_um": rms([float(r["actual_um"]) - float(r["expected_um"])
                               for r in targets
                               if r.get("expected_um") is not None]),
        "max_nfev": int(max_nfev),
        "time_budget_s": float(time_budget_s),
        "resolution": resolution,
        "n_fits": len(rows),
        "theta_full": theta_slim(theta0),
        "calibration_found": full_info["calibration_found"],
        "calibration_created": full_info["calibration_created"],
        "fit_log": fit_log,
        "runtime_s": round(time.perf_counter() - t_start, 1),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 2. leave-setup-out (base -> imp1)
# ---------------------------------------------------------------------------
def leave_setup_out(runs: list[dict] | None = None,
                    sem_points: list[dict] | None = None,
                    max_nfev: int = 25,
                    time_budget_s: float = 300.0,
                    resolution: str = "coarse",
                    progress=None) -> dict:
    """Fit on the 6 'base' runs only, predict the 2 'imp1' runs.

    The reverse direction (imp1 -> base) is not attempted: two runs cannot
    constrain six free parameters, so the fit would just return the prior.
    """
    t_start = time.perf_counter()
    runs = list(runs) if runs is not None else calibration_runs()
    sem_points = load_sem_points() if sem_points is None else list(sem_points)
    base_runs = [r for r in runs if r.get("setup") == "base"]
    imp_runs = [r for r in runs if r.get("setup") == "imp1"]
    if not base_runs:
        raise ValueError("no 'base' runs to fit on")
    if not imp_runs:
        raise ValueError("no 'imp1' runs to predict (the held-out setup)")

    base_ids = {int(r["id"]) for r in base_runs}
    train_sem = [p for p in sem_points if p.get("run_id") in base_ids]
    pred_full, full_info = _full_predictions(runs, resolution)

    if progress:
        progress(0.0, f"setup hold-out: fit on {len(base_runs)} base runs + "
                      f"{len(train_sem)} SEM points, predict {len(imp_runs)} imp1 runs")
    fit = calibrate_joint(runs=base_runs, sem_points=train_sem,
                          theta0=full_info["theta"], resolution=resolution,
                          max_nfev=max_nfev, time_budget_s=time_budget_s,
                          uncertainties=False, save=False, progress=progress)
    theta_base = {**DEFAULT_THETA, **fit["theta"]}

    rows: list[dict] = []
    for r in imp_runs:
        sim = simulate_run(r, theta_base, resolution=resolution)
        rows.append(make_row(r, sim["mean_thickness_um"], pred_full.get(int(r["id"])),
                             theta_base, fit.get("budget_hit", False)))

    notes = [
        "fit uses only setup == 'base' (untaped 1:1 anode, no baffles, 20 % stir); "
        "the imp1 runs (taped, baffles, 5x5 cm anode, 30 % stir) are never seen [C]",
        "reverse direction (imp1 -> base) is not attempted: 2 runs < 6 parameters [J]",
        "the base-only fit cannot identify the imp1-specific hardware (f_back_imp is "
        "fixed at 0 and the anode size is a constant), so this tests how far the "
        "shared physics (flow + O2 + L0) extrapolates across a hardware change [J]",
    ]
    if fit.get("budget_hit"):
        notes.append("the base-only fit hit its time/solve budget - theta is the best "
                     "point seen, not a converged fit")
    if not full_info["calibration_found"]:
        notes.append("no valid calibration.json found: pred_full_um uses DEFAULT_THETA")

    return {
        "rows": rows,
        "rms_um": rms([r["resid_loo_um"] for r in rows]),
        "rms_full_um": rms([r["resid_full_um"] for r in rows]),
        "max_abs_um": max_abs([r["resid_loo_um"] for r in rows]),
        "mean_bias_um": mean_bias([r["resid_loo_um"] for r in rows]),
        "n_base_runs": len(base_runs),
        "n_imp_runs": len(rows),
        "n_base_sem_points": len(train_sem),
        "fit_rms_um": fit.get("rms_um"),
        "fit_n_fev": fit.get("n_fev"),
        "theta_base": theta_slim(theta_base),
        "budget_hit": bool(fit.get("budget_hit", False)),
        "max_nfev": int(max_nfev),
        "time_budget_s": float(time_budget_s),
        "runtime_s": round(time.perf_counter() - t_start, 1),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 3. blind trench-coupon prediction
# ---------------------------------------------------------------------------
def trench_coupon_prediction(theta: dict | None = None,
                             resolution: str = "coarse") -> dict:
    """Predict the trench coupon "Trench 1" [M] with the saved calibration.

    No fitting happens here: theta is the saved full-data calibration (or
    DEFAULT_THETA when calibration.json is missing).  Only the thickness-map
    range can be compared with the measurement, because the positions of the SEM
    sites on the coupon are not known.
    """
    cal_found = True
    if theta is None:
        theta, cal = theta_from_calibration()
        cal_found = cal is not None
    else:
        theta = {**DEFAULT_THETA, **theta}
        cal = None

    sim = simulate_run(dict(TRENCH_RUN), theta, resolution=resolution)
    th_map = np.asarray(sim["maps"]["thickness_um"], float)
    flat = th_map[np.isfinite(th_map)]
    p5, p50, p95 = (float(v) for v in np.percentile(flat, [5.0, 50.0, 95.0]))
    tmin = float(flat.min())
    tmax = float(flat.max())

    pred = {
        "mean_um": float(sim["mean_thickness_um"]),
        "min_um": tmin,
        "max_um": tmax,
        "p5_um": p5,
        "p50_um": p50,
        "p95_um": p95,
        "max_min_ratio": (tmax / tmin) if tmin > 0 else None,
        "faraday_um": float(sim["faraday_um"]),
        "apparent_eff": (float(sim["apparent_eff"])
                         if sim.get("apparent_eff") is not None else None),
        "edge_center_ratio": sim.get("edge_center_ratio"),
        "std_over_mean_pct": sim.get("std_over_mean_pct"),
        "n_grid_points": int(flat.size),
    }
    inside = {label: bool(tmin <= float(v) <= tmax)
              for label, v in zip(TRENCH_SITE_LABELS, TRENCH_MEASURED["sites_um"])}
    inside["all"] = bool(all(inside[k] for k in TRENCH_SITE_LABELS))

    notes = [
        "the positions of the SEM sites on the coupon are NOT known, so only the "
        "predicted min-max range can be compared with the measured range "
        "(39.5-91 um); a site inside/outside the range is not a point-wise test [J]",
        "the trench run was pulse-reverse (9/0.5/0.5 ms) while this coupon model is "
        "DC: PR changes the local concentration at the trench mouth and is expected "
        "to raise the trench-mouth thickness relative to DC [J]",
        "the model has no trench geometry: it predicts the planar coupon field, so "
        "69-91 um next to the trenches is compared with the planar field maximum, not "
        "with a feature-fill solution [J]",
        "predicted percentiles are over the coupon face grid (endpoint-inclusive, "
        "41x41 at the coarse mesh), not area-weighted [C]",
        "the predicted max sits on the endpoint-inclusive boundary/corner nodes, "
        "where the cell solve has an edge singularity: the p5-p95 band is the "
        "robust non-uniformity measure, the min-max ratio is the pessimistic one [C]",
        f"Faraday at 100 % = {TRENCH_MEASURED['faraday_um']} um [C]; the measured "
        "coupon spans ~2.1-2.3x, i.e. the field non-uniformity dominates",
    ]
    if not cal_found:
        notes.append("no valid calibration.json found: theta is DEFAULT_THETA")

    return {
        "run": dict(TRENCH_RUN),
        "pred": pred,
        "measured": dict(TRENCH_MEASURED),
        "inside_range": inside,
        "theta": theta_slim(theta),
        "calibration_found": cal_found,
        "calibration_created": (cal or {}).get("created"),
        "resolution": resolution,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fnum(v, width: int = 7, nd: int = 2) -> str:
    if v is None:
        return f"{'-':>{width}}"
    return f"{float(v):>{width}.{nd}f}"


def _print_loo_table(res: dict) -> None:
    print()
    print("=" * 96)
    print("LEAVE-ONE-OUT: refit on 7 DOE runs (minus the held-out run's SEM points), "
          "predict it")
    print("=" * 96)
    print(f"{'id':>3} {'setup':>5} {'j':>5} {'t/min':>6} {'stir':>5} "
          f"{'meas':>7} {'LOO':>7} {'full':>7} {'r_LOO':>7} {'r_full':>7} "
          f"{'nfev':>4} {'s':>5}")
    for r in res["rows"]:
        print(f"{r['id']:>3} {str(r['setup']):>5} {r['j_mA_cm2']:>5.0f} "
              f"{r['t_min']:>6.0f} {_fnum(r['stir_pct'], 5, 0)} "
              f"{_fnum(r['measured_um'], 7)} {_fnum(r['pred_loo_um'], 7)} "
              f"{_fnum(r['pred_full_um'], 7)} {_fnum(r['resid_loo_um'], 7)} "
              f"{_fnum(r['resid_full_um'], 7)} "
              f"{(r.get('fit_n_fev') or 0):>4d} {_fnum(r.get('fit_runtime_s'), 5, 0)}")
    print("-" * 96)
    print(f"rms LOO      {_fnum(res['rms_loo_um'], 7)} um    "
          f"rms full-fit {_fnum(res['rms_full_um'], 7)} um")
    print(f"max |LOO|    {_fnum(res['max_abs_loo_um'], 7)} um    "
          f"mean bias    {_fnum(res['mean_bias_loo_um'], 7)} um  "
          f"(measured - predicted)")
    print(f"plain Faraday rms (measured - expected) {_fnum(res['faraday_rms_um'], 7)} um")
    for n in res["notes"]:
        print(f"  note: {n}")


def _print_setup_table(res: dict) -> None:
    print()
    print("=" * 96)
    print(f"LEAVE-SETUP-OUT: fit on {res['n_base_runs']} 'base' runs only, predict "
          f"{res['n_imp_runs']} 'imp1' runs")
    print("=" * 96)
    print(f"{'id':>3} {'setup':>5} {'j':>5} {'t/min':>6} {'stir':>5} "
          f"{'meas':>7} {'base-fit':>9} {'full':>7} {'r_base':>7} {'r_full':>7}")
    for r in res["rows"]:
        print(f"{r['id']:>3} {str(r['setup']):>5} {r['j_mA_cm2']:>5.0f} "
              f"{r['t_min']:>6.0f} {_fnum(r['stir_pct'], 5, 0)} "
              f"{_fnum(r['measured_um'], 7)} {_fnum(r['pred_loo_um'], 9)} "
              f"{_fnum(r['pred_full_um'], 7)} {_fnum(r['resid_loo_um'], 7)} "
              f"{_fnum(r['resid_full_um'], 7)}")
    print("-" * 96)
    print(f"rms held-out setup {_fnum(res['rms_um'], 7)} um    "
          f"rms full-fit on the same runs {_fnum(res['rms_full_um'], 7)} um    "
          f"fit rms on base {_fnum(res['fit_rms_um'], 7)} um")
    for n in res["notes"]:
        print(f"  note: {n}")


def _print_trench_table(res: dict) -> None:
    p, m = res["pred"], res["measured"]
    print()
    print("=" * 96)
    print("TRENCH COUPON 'Trench 1' [M]: blind prediction, saved calibration, no fitting")
    print(f"  run: setup={res['run']['setup']}  j={res['run']['j_mA_cm2']:g} mA/cm2  "
          f"t={res['run']['t_min']:g} min  stir={res['run']['stir_pct']:g} %  "
          f"T={res['run']['T_C']:g} C")
    print("=" * 96)
    print(f"{'quantity':<22} {'predicted':>10} {'measured':>10}")
    print(f"{'mean thickness (um)':<22} {_fnum(p['mean_um'], 10)}")
    print(f"{'min (um)':<22} {_fnum(p['min_um'], 10)} {_fnum(m['min_um'], 10)}")
    print(f"{'max (um)':<22} {_fnum(p['max_um'], 10)} {_fnum(m['max_um'], 10)}")
    print(f"{'p5 (um)':<22} {_fnum(p['p5_um'], 10)}")
    print(f"{'p50 (um)':<22} {_fnum(p['p50_um'], 10)}")
    print(f"{'p95 (um)':<22} {_fnum(p['p95_um'], 10)}")
    print(f"{'max/min':<22} {_fnum(p['max_min_ratio'], 10)} "
          f"{_fnum(m['max_um'] / m['min_um'], 10)}")
    print(f"{'Faraday 100 % (um)':<22} {_fnum(p['faraday_um'], 10)} "
          f"{_fnum(m['faraday_um'], 10)}")
    print("-" * 96)
    for label, site in zip(TRENCH_SITE_LABELS, m["sites_um"]):
        print(f"  measured site {label:<14} {site:6.1f} um  "
              f"inside predicted range: {res['inside_range'][label]}")
    print("  the coupon position of each SEM site is unknown -> only the range is "
          "comparable")
    for n in res["notes"]:
        print(f"  note: {n}")


def main(argv: list[str] | None = None) -> int:
    """python -m platesim.coupon.validate [--quick] [--out path] [--resolution r]"""
    ap = argparse.ArgumentParser(
        prog="python -m platesim.coupon.validate",
        description="Leave-one-out validation of the coupon calibration + blind "
                    "trench-coupon prediction (never writes calibration.json).")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: max_nfev=6, 60 s per fit, LOO on runs 1 and 7 only")
    ap.add_argument("--out", default=str(VALIDATION_PATH),
                    help=f"output JSON path (default {VALIDATION_PATH})")
    ap.add_argument("--resolution", default="coarse",
                    choices=["coarse", "medium", "fine"])
    args = ap.parse_args(argv)

    quick = bool(args.quick)
    max_nfev = 6 if quick else 25
    budget = 60.0 if quick else 300.0
    ids = [1, 7] if quick else None
    t0 = time.perf_counter()

    print(f"coupon calibration validation ({'QUICK' if quick else 'FULL'}): "
          f"max_nfev={max_nfev}, time_budget={budget:g} s/fit, "
          f"resolution={args.resolution}")
    prog = _progress_printer("  ")
    loo = leave_one_out(max_nfev=max_nfev, time_budget_s=budget,
                        resolution=args.resolution, progress=prog, ids=ids)
    lso = leave_setup_out(max_nfev=max_nfev, time_budget_s=budget,
                          resolution=args.resolution, progress=prog)
    trench = trench_coupon_prediction(resolution=args.resolution)

    _print_loo_table(loo)
    _print_setup_table(lso)
    _print_trench_table(trench)
    total = time.perf_counter() - t0
    print()
    print("=" * 96)
    print(f"total runtime {total:.0f} s")

    payload = {
        "model": "coupon_pipeline_validation",
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "quick": quick,
        "resolution": args.resolution,
        "leave_one_out": loo,
        "leave_setup_out": lso,
        "trench_coupon": trench,
        "runtime_s": round(total, 1),
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())


__all__ = ["VALIDATION_PATH", "TRENCH_RUN", "TRENCH_MEASURED", "leave_one_out",
           "leave_setup_out", "main", "make_row", "max_abs", "mean_bias", "rms",
           "theta_slim", "trench_coupon_prediction"]
