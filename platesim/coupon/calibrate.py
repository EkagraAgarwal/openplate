"""Joint calibration of the coupon pipeline (flow -> cell -> efficiency) to team data.

Fit theta = {delta0_um, n_exp (fixed at 0.5 by default), rpm_per_pct, f_back,
L0_um, kappa_scale, sigma_um} to BOTH
  (a) planar DOE runs 1-8 mean SEM thickness (efficiency.model.calibration_runs), and
  (b) position-resolved SEM points (Oct 7 corners -> run 1, j 9; 9/18 mA top/bottom/mid;
      planar2_top_right-short.tif excluded - anomalous/likely shorted [M]), by
scipy.optimize.least_squares (penalised MAP: data residuals / sigma_um + weakly
informative priors from efficiency.PARAM_SPECS where names are shared).

Speed (target < 10 min full fit):
- solve_cell memoised on quantised inputs (pipeline.QUANT); DOE runs sharing
  (setup, stir, j) share one solve (runs 3/4, 2/5, 7/8) and SEM points reuse the
  run 1/2 solves; L0/sigma/sigma probes need no solve at all;
- wall-clock + solve budget (time_budget_s / max_solves) aborts least_squares
  gracefully (_BudgetExceeded) and keeps the best point seen.

Uncertainties are rough 1-sigma values from the finite-difference Jacobian at the
solution (cov = s^2 (J'J)^-1, s^2 = 2*cost/(m-n), mapped to physical units by the
delta method). Consistency check: fitted f_back vs the efficiency module's H4
stray fraction f_stray_base (~0.31).

Units: cm, s, A; param suffixes as in efficiency.model.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..core.bath import Bath
from ..efficiency.calibrate import load_cache as _load_eff_cache
from ..efficiency.model import (
    PARAM_SPECS as EFF_SPECS, calibration_runs, load_doe,
)
from .pipeline import (
    DEFAULT_THETA, flow_dir_deg, load_sem_points, quantised_theta,
    simulate_run, solve_cache_stats,
)

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"
CALIBRATION_VERSION = 1

# fit order (n_exp only when fix_n=False; flow_dir_deg only when fit_flow_dir=True)
FREE_ORDER = ["delta0_um", "n_exp", "rpm_per_pct", "f_back", "L0_um",
              "kappa_scale", "sigma_um"]

# transform: log (positive), logit (bounded), tanh (bounded, 0 -> mid).
# priors for delta0/L0/sigma are REUSED from efficiency.PARAM_SPECS (same physics,
# same sources); new params get weak priors tagged here.
COUPON_SPECS: dict[str, dict] = {
    "delta0_um": {"transform": "log", "prior": EFF_SPECS["delta0_um"]["prior"],
                  "help": "Nernst layer at rpm_ref=300 (efficiency prior; sources there)"},
    "L0_um": {"transform": "log", "prior": EFF_SPECS["L0_um"]["prior"],
              "help": "startup thickness loss (efficiency prior)"},
    "sigma_um": {"transform": "log", "prior": EFF_SPECS["sigma_um"]["prior"],
                 "help": "SEM thickness noise floor (efficiency prior)"},
    "rpm_per_pct": {"transform": "log", "prior": ("norm", 15.0, 5.0),
                    "help": "rpm per % dial [J core.cell.STIR_PCT_TO_RPM 20%->300 rpm guess]"},
    "f_back": {"transform": "logit", "bounds": (0.0, 1.0),
               "prior": ("truncnorm", 0.30, 0.15, 0.0, 1.0),
               "help": "active fraction of back+edge area, base setup [J, mirrors "
                       "efficiency H4 f_stray_base prior (0.30 +- 0.15)]"},
    "kappa_scale": {"transform": "log", "prior": ("norm", 1.0, 0.3),
                    "help": "multiplier on Bath.kappa_22 [C +-20% KB]"},
    "n_exp": {"transform": "tanh", "bounds": (0.1, 1.2), "prior": ("norm", 0.5, 0.1),
              "help": "delta ~ rpm^(-n) [FT Avdeef 2022 n = 0.505]"},
    "flow_dir_deg": {"transform": "tanh", "bounds": (-180.0, 180.0), "prior": None,
                     "help": "Leveque direction in degrees (0 = +x) [spec]"},
}

DATA_SIGMA_FLOOR = 0.05     # um
FD_REL_STEP = 0.02          # finite-difference step in unconstrained space (>= 3x QUANT)


class _BudgetExceeded(Exception):
    """Raised inside a residual evaluation when the time/solve budget is spent."""


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def _to_u(name: str, x: float) -> float:
    spec = COUPON_SPECS[name]
    tr = spec["transform"]
    x = float(x)
    if tr == "log":
        return math.log(max(x, 1e-12))
    if tr == "logit":
        lo, hi = spec["bounds"]
        p = min(max((x - lo) / (hi - lo), 1e-9), 1 - 1e-9)
        return math.log(p / (1 - p))
    if tr == "tanh":
        lo, hi = spec["bounds"]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return math.atanh(min(max((x - mid) / half, -1 + 1e-12), 1 - 1e-12))
    raise ValueError(tr)


def _from_u(name: str, u: float) -> float:
    spec = COUPON_SPECS[name]
    tr = spec["transform"]
    u = min(max(float(u), -30.0), 30.0)
    if tr == "log":
        return math.exp(u)
    if tr == "logit":
        lo, hi = spec["bounds"]
        return lo + (hi - lo) / (1.0 + math.exp(-u))
    if tr == "tanh":
        lo, hi = spec["bounds"]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return mid + half * math.tanh(u)
    raise ValueError(tr)


def _dx_du(name: str, x: float) -> float:
    """d(physical)/d(unconstrained) at x - delta-method factor for uncertainties."""
    spec = COUPON_SPECS[name]
    tr = spec["transform"]
    if tr == "log":
        return abs(float(x))
    if tr == "logit":
        lo, hi = spec["bounds"]
        p = min(max((x - lo) / (hi - lo), 0.0), 1.0)
        return (hi - lo) * p * (1.0 - p)
    if tr == "tanh":
        lo, hi = spec["bounds"]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        t = min(max((x - mid) / half, -1.0), 1.0)
        return abs(half * (1.0 - t * t))
    return 1.0


def free_params(fix_n: bool = True, fit_flow_dir: bool = False,
                fit_params: list[str] | None = None) -> list[str]:
    if fit_params is not None:
        free = [n for n in FREE_ORDER if n in fit_params]
        if fit_flow_dir and "flow_dir_deg" in fit_params:
            free.append("flow_dir_deg")
        return free
    free = [n for n in FREE_ORDER if not (n == "n_exp" and fix_n)]
    if fit_flow_dir:
        free.append("flow_dir_deg")
    return free


def theta_from_u(u, free: list[str], base_theta: dict) -> dict:
    th = dict(base_theta)
    for name, ui in zip(free, u):
        x = _from_u(name, float(ui))
        if name == "flow_dir_deg":
            th["flow_dir"] = x
        else:
            th[name] = x
    return th


def u_from_theta(theta: dict, free: list[str]) -> np.ndarray:
    out = []
    for name in free:
        if name == "flow_dir_deg":
            out.append(_to_u(name, flow_dir_deg(theta.get("flow_dir", "+x"))))
        else:
            out.append(_to_u(name, float(theta.get(name, DEFAULT_THETA[name]))))
    return np.asarray(out, float)


def _prior_residual(name: str, x: float, scale: float) -> float:
    spec = COUPON_SPECS[name]
    p = spec.get("prior")
    if p is None:
        return 0.0
    if p[0] == "lognormal":
        r = (math.log(max(float(x), 1e-12)) - p[1]) / p[2]
    elif p[0] in ("norm", "truncnorm"):
        r = (float(x) - p[1]) / p[2]
    elif p[0] == "halfnormal":
        r = float(x) / p[1]
    else:
        raise ValueError(p[0])
    return r / max(scale, 1e-6)


# ---------------------------------------------------------------------------
# forward table
# ---------------------------------------------------------------------------
def _run_by_id(run_id, runs: list[dict]) -> dict | None:
    for r in runs:
        if r.get("id") == run_id:
            return r
    for r in load_doe().get("runs", []):
        if r.get("id") == run_id:
            return r
    return None


def predict_table(theta: dict, runs: list[dict] | None = None,
                  sem_points: list[dict] | None = None, bath: Bath | None = None,
                  resolution: str = "coarse") -> tuple[list[dict], list[dict]]:
    """Residual table rows for (a) DOE means and (b) SEM position points.

    DOE rows are mean-thickness predictions; SEM rows are bilinear point
    predictions on the run's thickness map (runs 1/2 maps already simulated for
    the DOE rows, so the points ride on the same solve).
    """
    runs = list(runs) if runs is not None else calibration_runs()
    sem_points = load_sem_points() if sem_points is None else list(sem_points)
    sims: dict = {}
    doe_rows: list[dict] = []
    for r in runs:
        sim = simulate_run(r, theta, resolution=resolution, bath=bath)
        sims[r["id"]] = sim
        meas = float(r["actual_um"])
        pred = float(sim["mean_thickness_um"])
        doe_rows.append({
            "source": "doe", "id": r["id"], "label": f"run {r['id']}",
            "setup": r.get("setup"), "j_mA_cm2": r["j_mA_cm2"],
            "t_min": r["t_min"], "stir_pct": r.get("stir_pct"),
            "measured_um": meas, "predicted_um": pred, "residual_um": meas - pred,
        })
    sem_rows: list[dict] = []
    for pt in sem_points:
        sim = sims.get(pt["run_id"])
        if sim is None:
            r = _run_by_id(pt["run_id"], runs)
            if r is None:
                raise ValueError(f"SEM point {pt['label']} references unknown run "
                                 f"{pt['run_id']}")
            sim = simulate_run(r, theta, resolution=resolution, bath=bath)
            sims[pt["run_id"]] = sim
        pred = None
        for row in sim["sem_points"]:
            if row["label"] == pt["label"]:
                pred = float(row["predicted_um"])
                break
        meas = pt["measured_um"]
        sem_rows.append({
            "source": "sem", "id": pt["run_id"], "label": pt["label"],
            "setup": "base" if pt["run_id"] in (1, 2) else None,
            "x_cm": pt["x_cm"], "y_cm": pt["y_cm"],
            "measured_um": meas,
            "predicted_um": pred,
            "residual_um": (meas - pred) if (pred is not None and meas is not None) else None,
            "confidence": pt.get("confidence"),
        })
    return doe_rows, sem_rows


def _data_residuals(doe_rows, sem_rows, sigma: float) -> list[float]:
    out = []
    for row in list(doe_rows) + list(sem_rows):
        if row["measured_um"] is None or row["predicted_um"] is None:
            raise ValueError(f"missing prediction for {row['label']}")
        out.append((row["measured_um"] - row["predicted_um"]) / sigma)
    return out


# ---------------------------------------------------------------------------
# MAP fit
# ---------------------------------------------------------------------------
def calibrate_joint(runs: list[dict] | None = None,
                    sem_points: list[dict] | None = None,
                    theta0: dict | None = None,
                    bath: Bath | None = None,
                    resolution: str = "coarse",
                    max_nfev: int = 40,
                    fix_n: bool = True,
                    flow_dir=None,
                    fit_flow_dir: bool = False,
                    fit_params: list[str] | None = None,
                    prior_scale: float = 1.0,
                    time_budget_s: float = 420.0,
                    max_solves: int = 500,
                    uncertainties: bool = True,
                    save: bool = True,
                    path: str | Path | None = None,
                    progress=None) -> dict:
    """Joint MAP fit of theta to DOE means + SEM position points.

    Returns {theta, std, free_params, residuals, rms_um, h4_check, n_fev,
    n_solves, runtime_s, budget_hit, ...}; saves calibration.json unless
    save=False.  Smoke tests: pass a 2-run subset, small max_nfev/time budget,
    save=False, uncertainties=False.
    """
    t0 = time.perf_counter()
    base_theta = {**DEFAULT_THETA, **(theta0 or {})}
    if flow_dir is not None:
        base_theta["flow_dir"] = flow_dir
    runs = list(runs) if runs is not None else calibration_runs()
    sem_points = load_sem_points() if sem_points is None else list(sem_points)
    if not runs:
        raise ValueError("no DOE runs with measured thickness to fit")
    bath = bath or Bath()
    free = free_params(fix_n=fix_n, fit_flow_dir=fit_flow_dir, fit_params=fit_params)
    if not free:
        raise ValueError("no free parameters")

    solves0 = solve_cache_stats()["misses"]
    state: dict = {"n_forward": 0, "n_data": None, "best": None,
                   "last_u": None, "last_f": None, "failures": []}
    if progress:
        progress(0.02, f"joint fit: {len(free)} free params, {len(runs)} DOE runs, "
                       f"{len(sem_points)} SEM points, {resolution} mesh")

    def _check_budget():
        if time.perf_counter() - t0 > time_budget_s:
            raise _BudgetExceeded("time budget")
        if solve_cache_stats()["misses"] - solves0 > max_solves:
            raise _BudgetExceeded("solve budget")

    def _residual(u: np.ndarray) -> np.ndarray:
        state["n_forward"] += 1
        _check_budget()
        theta = theta_from_u(u, free, base_theta)
        try:
            doe_rows, sem_rows = predict_table(theta, runs=runs, sem_points=sem_points,
                                               bath=bath, resolution=resolution)
            sigma = max(float(theta["sigma_um"]), DATA_SIGMA_FLOOR)
            f = np.asarray(_data_residuals(doe_rows, sem_rows, sigma) +
                           [_prior_residual(n, float(theta[n]), prior_scale) for n in free],
                           dtype=float)
        except _BudgetExceeded:
            raise
        except Exception as exc:                       # solver/Newton failure: penalise
            if len(state["failures"]) < 3:
                state["failures"].append(f"{type(exc).__name__}: {exc}")
            m = state["n_data"] or (len(runs) + len(sem_points) + len(free))
            f = np.full(m, 1e3)
            doe_rows, sem_rows = [], []
        if not np.all(np.isfinite(f)):
            m = state["n_data"] or f.size
            f = np.full(m, 1e3)
        state["n_data"] = f.size
        cost = 0.5 * float(f @ f)
        if state["best"] is None or cost < state["best"]["cost"]:
            state["best"] = {"u": np.array(u, dtype=float), "cost": cost, "f": np.array(f),
                             "doe_rows": doe_rows, "sem_rows": sem_rows,
                             "theta": theta_from_u(u, free, base_theta)}
        state["last_u"] = np.array(u, dtype=float)
        state["last_f"] = np.array(f, dtype=float)
        if progress and state["n_forward"] % 3 == 0:
            est = max(1, (max_nfev + 1) * (len(free) + 1))
            progress(min(0.9, 0.05 + 0.85 * state["n_forward"] / est),
                     f"residual eval {state['n_forward']} (cost {cost:.2f})")
        return f

    def _fd_jac(u: np.ndarray) -> np.ndarray:
        if state["last_u"] is not None and np.array_equal(state["last_u"], u):
            f0 = state["last_f"]
        else:
            f0 = _residual(u)
        J = np.empty((f0.size, u.size), dtype=float)
        for k in range(u.size):
            h = FD_REL_STEP * max(1.0, abs(float(u[k])))
            up = np.array(u, dtype=float)
            up[k] += h
            J[:, k] = (_residual(up) - f0) / h
        return J

    u0 = u_from_theta(base_theta, free)
    _residual(u0)                                   # baseline (also fixes m, validates pipeline)
    sol = None
    budget_hit = False
    error = None
    try:
        sol = least_squares(_residual, u0, jac=_fd_jac, method="trf",  # type: ignore[arg-type]
                            x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5,
                            max_nfev=max_nfev)
    except _BudgetExceeded:
        budget_hit = True
    except Exception as exc:                        # keep best point seen
        error = f"{type(exc).__name__}: {exc}"

    # pick the better of (sol.x, best-seen)
    u_fit = state["best"]["u"]
    converged = False
    status = "budget" if budget_hit else ("error" if error else "unknown")
    if sol is not None:
        try:
            f_sol = _residual(np.asarray(sol.x, dtype=float))
            if 0.5 * float(f_sol @ f_sol) <= state["best"]["cost"]:
                u_fit = np.asarray(sol.x, dtype=float)
                converged = True
                status = f"least_squares status {sol.status}: {sol.message}"
        except _BudgetExceeded:
            budget_hit = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    theta_fit = theta_from_u(u_fit, free, base_theta)
    # re-materialise the best rows for theta_fit (cache makes this cheap when
    # theta_fit is the tracked best; otherwise one fresh set of solves)
    if state["best"] is not None and np.array_equal(state["best"]["u"], u_fit):
        doe_rows, sem_rows = state["best"]["doe_rows"], state["best"]["sem_rows"]
        cost = state["best"]["cost"]
    else:
        doe_rows, sem_rows = predict_table(theta_fit, runs=runs, sem_points=sem_points,
                                           bath=bath, resolution=resolution)
        sigma = max(float(theta_fit["sigma_um"]), DATA_SIGMA_FLOOR)
        f = np.asarray(_data_residuals(doe_rows, sem_rows, sigma) +
                       [_prior_residual(n, float(theta_fit[n]), prior_scale) for n in free])
        cost = 0.5 * float(f @ f)
    residuals = doe_rows + sem_rows

    # ---- rough uncertainties from the Jacobian at the solution ---------------
    std = None
    if uncertainties and time.perf_counter() - t0 < 0.85 * time_budget_s:
        try:
            J = _fd_jac(u_fit)
            m, n = J.shape
            if m > n:
                s2 = 2.0 * cost / (m - n)
                cov = s2 * np.linalg.pinv(J.T @ J, rcond=1e-10)
                std_u = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                std = {}
                for k, name in enumerate(free):
                    x = theta_fit["flow_dir"] if name == "flow_dir_deg" else theta_fit[name]
                    std[name] = abs(_dx_du(name, float(x))) * float(std_u[k])
        except _BudgetExceeded:
            budget_hit = True
        except Exception:
            std = None

    # ---- fit statistics ------------------------------------------------------
    res_vals = np.array([r["residual_um"] for r in residuals
                         if r.get("residual_um") is not None], float)
    rms = float(np.sqrt(np.mean(res_vals ** 2))) if res_vals.size else float("nan")
    doe_res = np.array([r["residual_um"] for r in doe_rows], float)
    sem_res = np.array([r["residual_um"] for r in sem_rows], float)
    rms_doe = float(np.sqrt(np.mean(doe_res ** 2))) if doe_res.size else float("nan")
    rms_sem = float(np.sqrt(np.mean(sem_res ** 2))) if sem_res.size else float("nan")

    # ---- consistency with efficiency H4 stray fraction ------------------------
    h4 = h4_consistency(theta_fit, runs=runs, bath=bath, resolution=resolution)

    n_solves = solve_cache_stats()["misses"] - solves0
    runtime = time.perf_counter() - t0
    if progress:
        progress(1.0, f"fit done in {runtime:.0f} s, rms {rms:.2f} um")

    result = {
        "model": "coupon_pipeline",
        "version": CALIBRATION_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "theta": {k: (v if isinstance(v, str) else round(float(v), 5))
                  for k, v in theta_fit.items() if k in set(FREE_ORDER) | {
                      "flow_dir", "anode_base_cm", "anode_imp_cm", "f_back_imp"}},
        "std": ({k: round(float(v), 4) for k, v in std.items()} if std else None),
        "free_params": free,
        "fix_n": fix_n,
        "fit_flow_dir": fit_flow_dir,
        "resolution": resolution,
        "prior_scale": prior_scale,
        "n_params": len(free),
        "n_data_doe": len(doe_rows),
        "n_data_sem": len(sem_rows),
        "residuals": residuals,
        "rms_um": round(rms, 3),
        "rms_doe_um": round(rms_doe, 3),
        "rms_sem_um": round(rms_sem, 3),
        "max_abs_residual_um": round(float(np.max(np.abs(res_vals))), 3)
                               if res_vals.size else None,
        "h4_check": h4,
        "n_fev": state["n_forward"],
        "n_solves": n_solves,
        "runtime_s": round(runtime, 1),
        "budget_hit": budget_hit,
        "converged": converged,
        "status": status,
        "solver_errors": state["failures"],
        "error": error,
        "solve_cache": solve_cache_stats(),
        "assumptions": [
            "joint MAP: data residuals (DOE means + SEM points) / sigma_um + weak priors "
            "from efficiency.PARAM_SPECS where names are shared [C]",
            "SEM top_right corner excluded (anomalous/likely shorted) [M]",
            "Oct 7 corner points assigned to run 1 (j 9, base) per sem_measurements "
            "run_id_guess [M]",
            "uncertainties are rough 1-sigma values from the finite-difference Jacobian "
            "(delta0_um and rpm_per_pct are expected to be strongly correlated: only two "
            "stir settings exist) [J]",
        ],
    }
    if save:
        out_path = Path(path) if path is not None else CALIBRATION_PATH
        out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def h4_consistency(theta_fit: dict, runs: list[dict] | None = None,
                   bath: Bath | None = None, resolution: str = "coarse") -> dict:
    """Compare fitted f_back with the efficiency module's H4 f_stray_base (~0.31).

    The quantities are NOT identical by construction: H4's f_stray_base is a
    current fraction of the set current, while f_back is an active-area fraction
    of back+edges.  Both are reported: f_back itself, the implied base-setup
    current loss from the fitted pipeline, and H4's value.
    """
    f_h4 = None
    try:
        cache = _load_eff_cache()
        for key, r in sorted(cache.get("hypotheses", {}).items()):
            if key.startswith("H4:") and r.get("theta_map") is not None:
                f_h4 = float(r["theta_map"].get("f_stray_base"))
    except Exception:
        f_h4 = None
    f_back = float(theta_fit.get("f_back", DEFAULT_THETA["f_back"]))

    implied_loss = None
    try:
        runs = list(runs) if runs is not None else calibration_runs()
        base_runs = [r for r in runs if r.get("setup") == "base"] or runs
        sim = simulate_run(base_runs[0], theta_fit, resolution=resolution, bath=bath)
        implied_loss = float(sim["loss_fraction"])
    except Exception:
        implied_loss = None

    out = {
        "f_back_area": round(f_back, 4),
        "implied_current_loss_base": round(implied_loss, 4) if implied_loss is not None else None,
        "f_stray_base_h4": round(f_h4, 4) if f_h4 is not None else None,
        "note": "f_back is an active back+edge AREA fraction; H4 f_stray_base is a current "
                "fraction - compare it with implied_current_loss_base, not with f_back "
                "directly [spec consistency check]",
    }
    if f_h4 is not None and implied_loss is not None:
        d = implied_loss - f_h4
        out["delta_current_loss"] = round(d, 4)
        out["consistent"] = bool(abs(d) <= 0.15)   # ~ half the H4 prior sd
    return out


def load_calibration(path: str | Path = CALIBRATION_PATH) -> dict | None:
    """Load calibration.json (None if missing/stale/unreadable)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("model") != "coupon_pipeline" or data.get("version") != CALIBRATION_VERSION:
        return None
    return data


def theta_from_calibration(path: str | Path = CALIBRATION_PATH) -> tuple[dict, dict | None]:
    """(theta, calibration dict or None) - defaults when no fit is cached."""
    cal = load_calibration(path)
    if cal is None:
        return dict(DEFAULT_THETA), None
    return {**DEFAULT_THETA, **cal["theta"]}, cal


__all__ = [
    "CALIBRATION_PATH", "CALIBRATION_VERSION", "COUPON_SPECS", "FREE_ORDER",
    "calibrate_joint", "free_params", "h4_consistency", "load_calibration",
    "predict_table", "theta_from_calibration", "theta_from_u", "u_from_theta",
]
