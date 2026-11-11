"""platesim.efficiency: planar Cu current-efficiency loss model + Bayesian calibration
to the team's planar DOE (runs 1-8 in platesim/data/planar_doe.json; runs 9-10 pending).

Model id: "planar_efficiency". See model.py for the forward model and calibrate.py for
MAP/MCMC calibration and H1-H4 comparison. Calibration results cache to
platesim/efficiency/calibration_results.json.
"""
from __future__ import annotations

from ..core.bath import Bath
from .model import (
    HYPOTHESES, default_theta, load_doe, predict,
)
from .calibrate import (
    calibrate, compare_hypotheses, load_cache, predict_pending_runs, true_efficiency_curve,
)

MODEL_SPEC = {
    "id": "planar_efficiency",
    "title": "Planar Cu current-efficiency loss model",
    "description": (
        "Lumped loss model (stray conductors, CC->CV fallback with anode passivation, "
        "O2/H2 parasitic currents, startup loss, area/SEM bookkeeping) calibrated "
        "Bayesianly to the team's 10-run planar DOE; predicts pending runs 9-10 and "
        "true gravimetric efficiency vs current density."
    ),
    "params": [
        {"name": "mode", "type": "enum", "choices": ["predict", "calibrate"], "default": "predict",
         "help": "predict: forward model for one run (uses cached H4 MAP fit if present). "
                 "calibrate: fit H1-H4 to DOE runs 1-8 and compare."},
        {"name": "hypothesis", "type": "enum", "choices": ["H1", "H2", "H3", "H4"], "default": "H4",
         "help": "H1 O2-only; H2 +stray/area bookkeeping; H3 +CV fallback/anode passivation; H4 combo."},
        {"name": "j_mA_cm2", "type": "float", "default": 12.0, "min": 1.0, "max": 40.0,
         "unit": "mA/cm2", "help": "set cathode current density on the ~4 cm2 coupon."},
        {"name": "t_min", "type": "float", "default": 60.0, "min": 5.0, "max": 240.0, "unit": "min",
         "help": "plating time."},
        {"name": "stir_pct", "type": "float", "default": 30.0, "min": 0.0, "max": 100.0,
         "unit": "% dial", "help": "stirrer dial percent (W9 deck setting)."},
        {"name": "setup", "type": "enum", "choices": ["base", "imp1", "imp2"], "default": "imp1",
         "help": "base: untaped 1:1 anode; imp1: taped+baffles+larger anode; imp2: 40% stir improvement #2."},
        {"name": "T_C", "type": "float", "default": 22.0, "min": 15.0, "max": 60.0, "unit": "degC",
         "help": "bath temperature."},
        {"name": "method", "type": "enum", "choices": ["map", "mcmc"], "default": "map",
         "help": "calibration method: penalized-LS MAP, or MCMC (emcee; Metropolis fallback)."},
    ],
}

_LOSS_CATEGORIES = ["stray current", "CC/CV fallback", "O2 parasitic", "H2 (over j_lim)",
                    "startup L0", "area error"]


def _theta_for_predict(hypothesis: str) -> dict:
    """Cached MAP theta if available, else prior medians."""
    cache = load_cache()
    for key, r in cache.get("hypotheses", {}).items():
        if key.startswith(f"{hypothesis}:") and r.get("theta_map"):
            return r["theta_map"]
    return default_theta(hypothesis)


def _eff_vs_j_plot(theta: dict, hypothesis: str, bath: Bath, t_min: float = 60.0) -> dict:
    """Apparent efficiency vs j, one series per setup (stir at each setup's DOE setting)."""
    j_grid = [2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 30.0, 40.0]
    setup_stir = {"base": 20.0, "imp1": 30.0, "imp2": 40.0}
    series = []
    for setup, stir in setup_stir.items():
        ys = []
        for j in j_grid:
            run = {"j_mA_cm2": j, "t_min": t_min, "stir_pct": stir, "setup": setup}
            ys.append(round(predict(run, theta, bath=bath, hypothesis=hypothesis)["apparent_eff"], 4))
        series.append({"name": setup, "y": ys})
    return {"kind": "line", "title": f"Apparent efficiency vs j (t = {t_min:g} min, calibrated theta)",
            "x": j_grid, "series": series, "xlabel": "j (mA/cm2)", "ylabel": "apparent efficiency"}


def _measured_vs_predicted_plot(predictions: list[dict]) -> dict:
    xs = [p["id"] for p in predictions]
    return {
        "kind": "line",
        "title": "Measured vs predicted deposit thickness (DOE runs 1-8)",
        "x": xs,
        "series": [
            {"name": "measured SEM (um)", "y": [p["measured_um"] for p in predictions],
             "mode": "markers"},
            {"name": "model MAP (um)", "y": [p["predicted_um"] for p in predictions],
             "mode": "markers"},
        ],
        "xlabel": "run id", "ylabel": "thickness (um)",
    }


def _loss_breakdown_bar(run: dict, theta: dict, hypothesis: str, bath: Bath) -> dict:
    p = predict(run, theta, bath=bath, hypothesis=hypothesis)
    b = p["breakdown"]
    ys = [round(b["loss_stray_um"], 3), round(b["loss_cc_um"], 3), round(b["loss_o2_um"], 3),
          round(b["loss_h2_um"], 3), round(b["loss_l0_um"], 3), round(b["loss_area_um"], 3)]
    return {"kind": "bar",
            "title": f"Loss breakdown for run j={run['j_mA_cm2']:g} mA/cm2, "
                     f"t={run['t_min']:g} min, {run.get('setup','base')} (um vs Faraday)",
            "categories": _LOSS_CATEGORIES,
            "series": [{"name": "thickness loss (um)", "y": ys}]}


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    mode = params.get("mode", "predict")
    hypothesis = params.get("hypothesis", "H4")
    method = params.get("method", "map")
    bath = Bath().at(float(params.get("T_C", 22.0)))
    warnings: list[str] = []
    assumptions = [
        "coupon A_meas = 4 cm2 [M user 2026-10-28]; DOE expected_um uses the same area",
        "anode area = 1x (base) or 2x (imp1/imp2) coupon area [M deck improvement #1; J ratio value]",
        "cell ohmic resistance = gap/(kappa*sqrt(A_anode*A_cath)), gap = 4 cm [J]",
        "delta(stir) = delta0*(stir/20)^(-0.5) [FT Avdeef 2022]; n = 0.5 fixed",
        "true cathodic efficiency of acid Cu 95-100 % [FT/AB report 06]; losses here are "
        "mostly bookkeeping [C KB sec 2]",
    ]

    run_user = {"j_mA_cm2": float(params.get("j_mA_cm2", 12.0)),
                "t_min": float(params.get("t_min", 60.0)),
                "stir_pct": float(params.get("stir_pct", 30.0)),
                "setup": str(params.get("setup", "imp1")),
                "T_C": float(params.get("T_C", 22.0))}

    if mode == "predict":
        progress(0.1, "forward model")
        theta = _theta_for_predict(hypothesis)
        p = predict(run_user, theta, bath=bath, hypothesis=hypothesis)
        b = p["breakdown"]
        plots = [_eff_vs_j_plot(theta, hypothesis, bath),
                 _loss_breakdown_bar(run_user, theta, hypothesis, bath)]
        summary = {
            "thickness_um": round(p["thickness_um"], 3),
            "apparent_eff": round(p["apparent_eff"], 3),
            "true_grav_eff": round(p["true_grav_eff"], 3),
            "delivered_charge_C": round(p["delivered_charge_C"], 4),
            "faraday_um": round(b["faraday_um"], 3),
            "loss_stray_um": round(b["loss_stray_um"], 3),
            "loss_cc_um": round(b["loss_cc_um"], 3),
            "loss_o2_um": round(b["loss_o2_um"], 3),
            "loss_h2_um": round(b["loss_h2_um"], 3),
            "loss_l0_um": round(b["loss_l0_um"], 3),
            "loss_area_um": round(b["loss_area_um"], 3),
            "delta_um": round(b["delta_um"], 2),
            "j_o2_mA_cm2": round(b["j_o2_mA_cm2"], 4),
            "V_req_mean_V": round(b["V_req_mean_V"], 3),
            "theta_source": "cached MAP" if _theta_source_is_map(hypothesis) else "prior medians",
            "hypothesis": hypothesis,
        }
        if summary["theta_source"] == "prior medians":
            warnings.append("no calibration cached yet - predict mode used prior medians; "
                            "run mode=calibrate first for fitted parameters")
        return {"summary": summary, "plots": plots, "warnings": warnings,
                "assumptions": assumptions}

    if mode == "calibrate":
        progress(0.05, f"MAP fit {hypothesis}")
        res = calibrate(hypothesis, method=method)
        progress(0.4, "comparison table")
        comparison = compare_hypotheses(method="map")   # BIC table always from MAP
        theta = res["theta_map"]
        pending = predict_pending_runs(hypothesis=hypothesis, method="map", bath=bath)
        progress(0.7, "efficiency curves")
        doe = load_doe()
        runs8 = [r for r in doe["runs"] if r.get("actual_um") is not None and r["id"] <= 8]

        best = comparison[0]
        second = comparison[1] if len(comparison) > 1 else None
        d_bic = (second["bic"] - best["bic"]) if second else float("nan")
        if d_bic < 6:
            warnings.append(
                f"only {len(runs8)} data points: {best['hypothesis']} vs {second['hypothesis'] if second else '-'} "
                f"separated by dBIC={d_bic:.1f} - hypotheses are weakly distinguishable; "
                "do not over-interpret the ranking")

        summary = {
            "best_hypothesis": best["hypothesis"],
            "best_bic": best["bic"],
            "best_rms_um": best["rms_um"],
            f"{hypothesis}_loglik": res.get("loglik"),
            f"{hypothesis}_bic": res.get("bic"),
            f"{hypothesis}_rms_um": res.get("rms_um"),
            "n_params": res.get("n_params"),
            "mcmc_backend": res.get("mcmc_backend", "n/a (map)"),
            "waic": res.get("waic"),
            "run9_um": pending[0]["predicted_um"] if pending else None,
            "run9_lo_um": pending[0]["lo_um"] if pending else None,
            "run9_hi_um": pending[0]["hi_um"] if pending else None,
            "run10_um": pending[1]["predicted_um"] if len(pending) > 1 else None,
            "run10_lo_um": pending[1]["lo_um"] if len(pending) > 1 else None,
            "run10_hi_um": pending[1]["hi_um"] if len(pending) > 1 else None,
            "comparison_table": comparison,
            "theta_map": theta,
        }
        if res.get("samples_summary"):
            summary["samples_summary"] = res["samples_summary"]

        plots = [
            _measured_vs_predicted_plot(res["predictions"]),
            _eff_vs_j_plot(theta, hypothesis, bath),
            _loss_breakdown_bar(runs8[0], theta, hypothesis, bath),
        ]
        warnings.append("runs 9-10 (40 % stir, imp2) are predictions - results still pending")
        warnings.append("PSU current accuracy is only a few % at small currents [M deck limitations]")
        return {"summary": summary, "plots": plots, "warnings": warnings,
                "assumptions": assumptions,
                "predictions_run9_10": pending,
                "true_efficiency_curves": true_efficiency_curve(hypothesis=hypothesis, bath=bath)}

    raise ValueError(f"unknown mode {mode!r}; use 'predict' or 'calibrate'")


def _theta_source_is_map(hypothesis: str) -> bool:
    cache = load_cache()
    return any(k.startswith(f"{hypothesis}:") and r.get("theta_map")
               for k, r in cache.get("hypotheses", {}).items())
