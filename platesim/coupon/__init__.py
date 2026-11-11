"""platesim.coupon: coupled coupon pipeline (flow -> cell -> efficiency) for the
2x2 cm Si coupon + joint calibration to the team's planar DOE (runs 1-8) and the
position-resolved SEM points (Oct 7 corners, 9/18 mA top/bottom/mid).

Model id "coupon_pipeline".  pipeline.simulate_run() couples flow.correlation
(delta map with f_back/stray bookkeeping and flow_dir), cell.solver (front
fraction + local j map) and the efficiency physics (local O2 parasitic +
startup L0); calibrate.calibrate_joint() fits {delta0_um, n_exp, rpm_per_pct,
f_back, L0_um, kappa_scale, sigma_um} with scipy least_squares on memoised
solves and saves platesim/coupon/calibration.json (rough Jacobian
uncertainties, residual table, consistency check against efficiency H4's
f_stray_base ~0.31).
"""
from __future__ import annotations

from ..efficiency.model import calibration_runs, load_doe
from .calibrate import (
    CALIBRATION_PATH, calibrate_joint, load_calibration, predict_table,
    theta_from_calibration,
)
from .pipeline import (
    ASSUMPTIONS as PIPELINE_ASSUMPTIONS, DEFAULT_THETA, load_sem_points,
    simulate_run, solve_cache_stats,
)

MODEL_SPEC = {
    "id": "coupon_pipeline",
    "title": "Coupled coupon pipeline (flow -> cell -> efficiency)",
    "description": (
        "Nernst-layer delta map (stir % -> rpm -> delta, baffles per setup, "
        "flow direction) feeds the 3-D beaker-cell FEM solve; the local j map "
        "is converted to a true local efficiency (O2 parasitic at the local "
        "delta) and a thickness map with startup loss L0. Calibrated jointly "
        "to planar DOE runs 1-8 (mean thickness) and the position-resolved SEM "
        "points; reports fitted theta with rough Jacobian uncertainties, "
        "residuals and a consistency check against the efficiency module's H4 "
        "stray fraction."
    ),
    "params": [
        {"name": "j_mA_cm2", "type": "float", "default": 12.0, "min": 1.0, "max": 60.0,
         "unit": "mA/cm2", "help": "set current density on the ~4 cm2 coupon "
                                   "(I_total = j * 4 cm2)."},
        {"name": "t_min", "type": "float", "default": 60.0, "min": 5.0, "max": 240.0,
         "unit": "min", "help": "plating time for the thickness map."},
        {"name": "stir_pct", "type": "float", "default": 30.0, "min": 0.0, "max": 100.0,
         "unit": "% dial", "help": "stirrer dial percent (W9 deck setting); "
                                   "rpm = rpm_per_pct * stir_pct."},
        {"name": "setup", "type": "enum", "choices": ["base", "imp1", "imp2"],
         "default": "imp1",
         "help": "base: untaped 1:1 anode, no baffles; imp1: taped+baffles+5x5 cm anode; "
                 "imp2: imp1 + 40% stir (runs 9-10 pending)."},
        {"name": "T_C", "type": "float", "default": 22.0, "min": 15.0, "max": 60.0,
         "unit": "degC", "help": "bath temperature (kappa, i0 and O2 limits rescaled)."},
        {"name": "use_calibration", "type": "bool", "default": True,
         "help": "use platesim/coupon/calibration.json if present, else default theta."},
        {"name": "resolution", "type": "enum", "choices": ["coarse", "medium", "fine"],
         "default": "coarse",
         "help": "FEM mesh resolution (coarse ~0.6 s per solve, needed for calibration)."},
        {"name": "mode", "type": "enum", "choices": ["simulate", "calibrate"],
         "default": "simulate",
         "help": "simulate: forward pipeline for one run (+ DOE/SEM comparison); "
                 "calibrate: joint fit to runs 1-8 + SEM points (writes calibration.json, "
                 "budgeted to < 10 min)."},
    ],
}


def _theta_summary(theta: dict, std: dict | None = None) -> dict:
    out = {}
    for k in ("delta0_um", "n_exp", "rpm_per_pct", "f_back", "L0_um",
              "kappa_scale", "sigma_um", "flow_dir"):
        v = theta.get(k)
        if isinstance(v, str):
            out[k] = v
        elif v is not None:
            out[k] = round(float(v), 4)
        else:
            out[k] = None
    if std:
        for k, v in std.items():
            out[f"{k}_std"] = round(float(v), 4)
    return out


def _heat(plot_title: str, sim: dict, key: str, zlabel: str) -> dict:
    maps = sim["maps"]
    return {"kind": "heatmap", "title": plot_title,
            "x": maps["x_cm"], "y": maps["z_cm"], "z": maps[key], "zlabel": zlabel}


def _pred_vs_meas_plot(doe_rows: list[dict]) -> dict:
    xs = [r["id"] for r in doe_rows]
    meas, pred = [], []
    for r in doe_rows:
        m, q = r.get("measured_um"), r.get("predicted_um")
        if m is not None:
            meas.append(round(float(m), 3))
        if q is not None:
            pred.append(round(float(q), 3))
    return {
        "kind": "line",
        "title": "Predicted vs measured thickness - DOE runs 1-8 (markers)",
        "x": xs,
        "series": [
            {"name": "measured SEM (um)", "y": meas, "mode": "markers"},
            {"name": "pipeline (um)", "y": pred, "mode": "markers"},
        ],
        "xlabel": "run id", "ylabel": "mean thickness (um)",
    }


def _current_split_bar(sim: dict) -> dict:
    return {
        "kind": "bar",
        "title": f"Current split (I_total) - {sim['run']['setup']}, "
                 f"j = {sim['run']['j_mA_cm2']:g} mA/cm2",
        "categories": ["front (coupon)", "back+edges", "stray (clip)"],
        "series": [{"name": "share of I_total",
                    "y": [round(sim["front_fraction"], 4),
                          round(sim["back_edge_fraction"], 4),
                          round(sim["stray_fraction"], 4)]}],
    }


def _loss_breakdown_bar(sim: dict) -> dict:
    return {
        "kind": "bar",
        "title": "Loss breakdown vs Faraday at set j (coupon mean, um)",
        "categories": ["current diverted", "O2 parasitic", "startup L0"],
        "series": [{"name": "thickness loss (um)",
                    "y": [round(sim["loss_current_um"], 3),
                          round(sim["loss_o2_um"], 3),
                          round(sim["loss_l0_um"], 3)]}],
    }


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    p = params or {}
    mode = str(p.get("mode", "simulate"))
    resolution = str(p.get("resolution", "coarse"))
    use_calibration = bool(p.get("use_calibration", True))
    warnings: list[str] = []
    assumptions = list(PIPELINE_ASSUMPTIONS)

    run_user = {
        "j_mA_cm2": float(p.get("j_mA_cm2", 12.0)),
        "t_min": float(p.get("t_min", 60.0)),
        "stir_pct": float(p.get("stir_pct", 30.0)),
        "setup": str(p.get("setup", "imp1")),
        "T_C": float(p.get("T_C", 22.0)),
    }

    theta = None
    theta_source = None
    cal_result = None
    doe_rows: list[dict] | None = None
    sem_rows: list[dict] | None = None

    if mode == "calibrate":
        progress(0.02, "joint calibration: flow -> cell -> efficiency")
        cal_result = calibrate_joint(resolution=resolution, progress=progress)
        theta = {**DEFAULT_THETA, **cal_result["theta"]}
        theta_source = "joint calibration (this run)"
        doe_rows = [r for r in cal_result["residuals"] if r["source"] == "doe"]
        sem_rows = [r for r in cal_result["residuals"] if r["source"] == "sem"]
        if cal_result["budget_hit"]:
            warnings.append("calibration stopped at its time/solve budget - fitted values "
                            "are from the best point seen (raise the budget for a tighter fit)")
        worst = cal_result.get("max_abs_residual_um") or 0.0
        if worst > 2.0 * max(theta.get("sigma_um", 0.5), 0.05):
            warnings.append(f"largest residual {worst:.2f} um exceeds 2*sigma_um - check "
                            "the residual table (imp runs may be over-predicted: no CC->CV "
                            "fallback in this pipeline)")
        if cal_result.get("error"):
            warnings.append(f"least_squares reported: {cal_result['error']}")
    elif mode == "simulate":
        theta, cal = theta_from_calibration() if use_calibration else (dict(DEFAULT_THETA), None)
        if use_calibration and cal is None:
            warnings.append("use_calibration=True but platesim/coupon/calibration.json is "
                            "missing/invalid - running with default theta; set mode=calibrate "
                            "to fit the team data")
            theta_source = "defaults (no calibration.json)"
        else:
            theta_source = (f"calibration.json ({cal.get('created', 'unknown date')})"
                            if cal else "defaults")
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'simulate' or 'calibrate'")

    # ---- main demo run -------------------------------------------------------
    progress(0.35, f"simulate {run_user['setup']} @ {run_user['j_mA_cm2']:g} mA/cm2")
    sim = simulate_run(run_user, theta, resolution=resolution)

    # ---- DOE/SEM comparison table (shared solves where possible) -------------
    if doe_rows is None:
        progress(0.7, "DOE runs 1-8 + SEM point predictions")
        doe_rows, sem_rows = predict_table(theta, resolution=resolution)
    assert doe_rows is not None and sem_rows is not None  # narrowed for type checkers

    # ---- plots ----------------------------------------------------------------
    plots = [
        _heat("Current density on coupon front face", sim, "j_mA_cm2", "j (mA/cm2)"),
        _heat(f"Deposit thickness after {run_user['t_min']:g} min", sim, "thickness_um",
              "thickness (um)"),
        _heat("Nernst-layer thickness delta(x, y) over the coupon", sim, "delta_um",
              "delta (um)"),
        _pred_vs_meas_plot(doe_rows),
        _current_split_bar(sim),
        _loss_breakdown_bar(sim),
    ]

    # ---- summary --------------------------------------------------------------
    sem_preds = {r["label"]: round(r["predicted_um"], 3) for r in sim["sem_points"]}
    summary = {
        "mode": mode,
        "theta_source": theta_source,
        **{k: v for k, v in run_user.items()},
        "rpm": sim["rpm"],
        "delta_mean_um": round(sim["delta_mean_um"], 2),
        "front_fraction": round(sim["front_fraction"], 4),
        "back_edge_fraction": round(sim["back_edge_fraction"], 4),
        "stray_fraction": round(sim["stray_fraction"], 4),
        "j_mean_mA_cm2": round(sim["j_mean_mA_cm2"], 3),
        "std_over_mean_pct": round(sim["std_over_mean_pct"], 2)
                             if sim["std_over_mean_pct"] is not None else None,
        "edge_center_ratio": round(sim["edge_center_ratio"], 3)
                             if sim["edge_center_ratio"] is not None else None,
        "V_cell_V": round(sim["V_cell_V"], 3),
        "faraday_um": round(sim["faraday_um"], 3),
        "mean_thickness_um": round(sim["mean_thickness_um"], 3),
        "apparent_eff": round(sim["apparent_eff"], 3),
        "eff_charge": round(sim["eff_charge"], 3),
        "loss_current_um": round(sim["loss_current_um"], 3),
        "loss_o2_um": round(sim["loss_o2_um"], 3),
        "loss_l0_um": round(sim["loss_l0_um"], 3),
        "sem_point_predictions_um": sem_preds,
        "theta": _theta_summary(theta),
        "rms_um_doe_sem": round(
            sum(r["residual_um"] ** 2 for r in (doe_rows + sem_rows)
                if r.get("residual_um") is not None)
            / max(1, sum(1 for r in (doe_rows + sem_rows)
                         if r.get("residual_um") is not None)), 4) ** 0.5,
        "solve_cache": solve_cache_stats(),
    }
    if cal_result is not None:
        summary.update({
            "calibration_rms_um": cal_result["rms_um"],
            "calibration_rms_doe_um": cal_result["rms_doe_um"],
            "calibration_rms_sem_um": cal_result["rms_sem_um"],
            "calibration_runtime_s": cal_result["runtime_s"],
            "calibration_n_solves": cal_result["n_solves"],
            "calibration_n_fev": cal_result["n_fev"],
            "calibration_converged": cal_result["converged"],
            "h4_check": cal_result["h4_check"],
        })
        assumptions += cal_result.get("assumptions", [])

    warnings += [w for w in sim["warnings"] if w not in warnings]
    warnings.append("runs 9-10 (40% stir, imp2) are still pending [M deck]")
    if run_user["setup"] == "base":
        warnings.append("base setup is untaped in the data: partial backside activity is "
                        "modelled via f_back; taped=False (front_fraction ~0.48) is the "
                        "fully-active upper bound [spec]")
    progress(1.0, "done")

    return {
        "summary": summary,
        "plots": plots,
        "warnings": warnings,
        "assumptions": assumptions,
        "residuals": doe_rows + sem_rows,
        "sim": {k: sim[k] for k in ("run", "rpm", "front_fraction", "geometry",
                                    "delta_mean_um", "solve")},
    }


__all__ = ["MODEL_SPEC", "run", "simulate_run", "calibrate_joint",
           "load_calibration", "theta_from_calibration", "DEFAULT_THETA",
           "predict_table", "load_sem_points", "CALIBRATION_PATH"]
