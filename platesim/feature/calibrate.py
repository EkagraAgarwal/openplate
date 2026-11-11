"""Calibrate the trench feature-fill model against Cu thicknesses measured on SEM cross-sections.

What is fitted: a small subset of the level-set feature-fill parameters (FREE_PARAMS below)
- current efficiency, diffusion boundary layer delta_um, unsuppressed exchange current density i0_u,
  eddy well-mixed depth k_mix - to two measured thicknesses per site, the Cu thickness on the field
  (face_thickness_um) and on the trench floor (floor_height_um).

Model assumptions of this harness:
- suppressor=False: the bath is treated as ADDITIVE-FREE, so the model grows conformally (no S-NDR
  suppressor/accelerator gradient and therefore no bottom-up fill). Measurements that need bottom-up
  behaviour are outside this model. [J]
- efficiency scales the growth velocity and the Cu2+ sink/source (Params.efficiency -> Phys.Mfac and
  the Cu2+ consumption terms in engine/fields.py); it does not change the potential field or the
  transport coefficients. [C]
- f_open is fixed at 0.2 (duty of the measured trench array) and the temperature/current/duration of
  each site come from the SEM record; only theta is free.

Identifiability caveat: each site contributes two numbers (field, floor), so N sites give 2N data.
fit() raises for n_free >= n_data, and reports identifiable=False (with the Jacobian rank) whenever
the requested parameters are not jointly determined by the data. The Trench 1 SEM record is a single
sample (two sites, same wafer), so at most one parameter is identifiable from it; any theta reported
for a larger free set is a least-squares compromise, NOT a measurement, and is not validated on
independent data.

CLI:
    python -m platesim.feature.calibrate --free efficiency [delta_um ...] [--dx 4] [--no-pr]
        [--out path.json]
The fitted theta and a table of measured vs predicted thicknesses are printed; the same dict is
written to platesim/data/trench_calibration.json by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import least_squares

from platesim.feature.engine import Params, dc, run_case

try:                       # the pulse-reverse helper is owned by another worker (engine/recipes.py)
    from platesim.feature.engine.recipes import pulse_reverse
except ImportError:        # keep working on a DC approximation, flagged in the output
    pulse_reverse = None

# ---------------------------------------------------------------------------------- configuration
FREE_PARAMS = {  # name: (lower, upper, log_scale, default)
    "efficiency": (0.3, 1.0, False, 1.0),
    "delta_um": (10.0, 300.0, True, 80.0),
    "i0_u": (1e-4, 2e-2, True, 2.0e-3),
    "k_mix": (0.0, 1.6, False, 1.0),
}

F_OPEN = 0.2                     # [M] open fraction of the trench array (team layout)
SEM_SIGMA_FLOOR_UM = 3.0         # [J] floor on the SEM reading uncertainty, um
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PKG_DIR, "data", "trench_sem.json")
DEFAULT_OUT = os.path.join(_PKG_DIR, "data", "trench_calibration.json")

_CACHE: dict = {}                # (site id, rounded theta, dx_um, use_pr) -> predict() dict
_CACHE_KEYS = tuple(sorted(FREE_PARAMS))


# ---------------------------------------------------------------------------------- data loading
def load_sites(path: str | None = None) -> list[dict]:
    """Load the measured trench sites (see platesim/data/trench_sem.json)."""
    with open(path or DATA_PATH, encoding="utf-8") as fh:
        doc = json.load(fh)
    sites = list(doc["sites"])
    for s in sites:
        for k in ("width_um", "depth_um", "T_C", "j_mA_cm2", "hours", "field_um", "floor_um"):
            if k not in s:
                raise ValueError(f"site {s.get('id', '?')!r} is missing {k!r}")
    return sites


# ---------------------------------------------------------------------------------- simulation
def _full_theta(theta: dict | None) -> dict:
    """Fill a partial theta with the FREE_PARAMS defaults (unknown names are a hard error)."""
    out = {k: float(FREE_PARAMS[k][3]) for k in _CACHE_KEYS}
    for k, v in (theta or {}).items():
        if k not in FREE_PARAMS:
            raise ValueError(f"unknown calibration parameter {k!r}; known: {sorted(FREE_PARAMS)}")
        out[k] = float(v)
    return out


def _cache_key(site: dict, th: dict, dx_um: float | None, use_pr: bool):
    site_id = site.get("id") or f"{site.get('width_um')}x{site.get('depth_um')}um"
    rounded = tuple(float(f"{th[k]:.12g}") for k in _CACHE_KEYS)
    return (site_id, rounded, None if dx_um is None else round(float(dx_um), 6), bool(use_pr))


def _last(series: dict, key: str) -> float | None:
    for v in reversed(series.get(key) or []):
        if v is not None:
            return float(v)
    return None


def _build_recipe(site: dict, use_pr: bool):
    """t_max_h = measured hours; pulse-reverse when the site has a 'pr' block and the helper exists."""
    j = float(site["j_mA_cm2"])
    hours = float(site["hours"])
    pr = site.get("pr")
    warnings: list[str] = []
    if pr and use_pr:
        if pulse_reverse is not None:
            kw = {k: float(pr[k]) for k in ("t_f_ms", "t_r_ms", "t_off_ms", "rr") if k in pr}
            try:
                return pulse_reverse(j, hours, **kw), warnings
            except TypeError:            # helper present but with another signature -> DC fallback
                pass
        warnings.append("pulse-reverse treated as DC")
    return dc(j, hours), warnings


def predict(site: dict, theta: dict, dx_um: float | None = None, use_pr: bool = True) -> dict:
    """Run one trench simulation for a site at a (partial) parameter set.

    Returns {"field_um", "floor_um", "ratio", "outcome", "warnings"}; results are cached by
    (site id, rounded theta, dx_um, use_pr) so a repeated fit evaluation is free.
    """
    th = _full_theta(theta)
    key = _cache_key(site, th, dx_um, use_pr)
    hit = _CACHE.get(key)
    if hit is not None:
        return dict(hit, warnings=list(hit["warnings"]))

    hours = float(site["hours"])
    rec, warnings = _build_recipe(site, use_pr)
    p = Params(geometry="trench", H_um=float(site["depth_um"]), suppressor=False,
               efficiency=th["efficiency"], delta_um=th["delta_um"], i0_u=th["i0_u"], k_mix=th["k_mix"],
               dx_um=None if dx_um is None else float(dx_um),
               # [J] headroom so the run is not truncated: ~21 um/h of field growth + 40 um of margin
               max_overburden_um=40.0 + 21.0 * hours)
    res: dict = run_case(  # type: ignore[assignment]
        float(site["width_um"]), float(site["T_C"]), rec, p, f_open=F_OPEN,
        label=str(key[0]), out=None)

    field = _last(res["series"], "face_thickness_um")
    floor = _last(res["series"], "floor_height_um")
    if field is None:
        field = float(res["result"].get("overburden_um") or 0.0)
    if floor is None:
        floor = 0.0
        warnings.append("floor height unavailable from the run; used 0 um")
    stop = res["result"].get("stop_reason")
    if stop == "overburden_exceeded":
        warnings.append("run stopped at the overburden limit: predicted thickness is a lower bound")

    out = {"field_um": float(field), "floor_um": float(floor),
           "ratio": float(floor / field) if field > 0 else float("nan"),
           "outcome": res["result"]["outcome"], "warnings": warnings}
    _CACHE[key] = dict(out, warnings=list(warnings))
    return out


# ---------------------------------------------------------------------------------- residuals
def _sigma(site: dict, which: str) -> float:
    """Reading uncertainty of one measured thickness: half the lo/hi span, floored at 3 um. [J]"""
    m = site[f"{which}_um"]
    return max((float(m["hi"]) - float(m["lo"])) * 0.5, SEM_SIGMA_FLOOR_UM)


def residuals(theta: dict, sites: list[dict], dx_um: float | None = None,
              use_pr: bool = True) -> list[float]:
    """Two residuals per site: (pred - best)/sigma for the field and for the floor thickness."""
    out: list[float] = []
    for s in sites:
        pred = predict(s, theta, dx_um, use_pr)
        out.append((pred["field_um"] - float(s["field_um"]["best"])) / _sigma(s, "field"))
        out.append((pred["floor_um"] - float(s["floor_um"]["best"])) / _sigma(s, "floor"))
    return out


# ---------------------------------------------------------------------------------- fitting
def _to_x(name: str, value: float) -> float:
    return math.log(float(value)) if FREE_PARAMS[name][2] else float(value)


def _from_x(name: str, x: float) -> float:
    return float(math.exp(x)) if FREE_PARAMS[name][2] else float(x)


def fit(sites: list[dict], free: list[str], dx_um: float | None = None, use_pr: bool = True,
        max_nfev: int = 40) -> dict:
    """Least-squares fit of `free` (names from FREE_PARAMS) to the two thicknesses per site.

    The optimiser works in a transformed variable (log for log_scale parameters) with the
    FREE_PARAMS bounds; the returned theta is on the physical scale. See the module docstring for
    the identifiability caveat: with 2 residuals per site a single sample rarely determines more
    than one parameter.
    """
    sites = list(sites)
    free = [str(n) for n in free]
    if not free:
        raise ValueError("free must name at least one parameter from FREE_PARAMS")
    bad = [n for n in free if n not in FREE_PARAMS]
    if bad:
        raise ValueError(f"unknown parameter(s) {bad}; known: {sorted(FREE_PARAMS)}")
    if len(set(free)) != len(free):
        raise ValueError(f"duplicate parameter names in free: {free}")

    n_data = 2 * len(sites)
    n_free = len(free)
    if n_free >= n_data:
        raise ValueError(f"cannot fit {n_free} parameters to {n_data} numbers "
                         f"({len(sites)} site(s) x 2 residuals): free fewer parameters or add sites")

    lb = np.array([_to_x(n, FREE_PARAMS[n][0]) for n in free], float)
    ub = np.array([_to_x(n, FREE_PARAMS[n][1]) for n in free], float)
    x0 = np.array([min(max(_to_x(n, FREE_PARAMS[n][3]), lb[i]), ub[i]) for i, n in enumerate(free)], float)

    def theta_of(x) -> dict:
        return {n: _from_x(n, xi) for n, xi in zip(free, x)}

    def fun(x):
        r = np.asarray(residuals(theta_of(x), sites, dx_um, use_pr), float)
        return np.where(np.isfinite(r), r, 1.0e3)     # [J] non-finite residual guard

    opt = least_squares(fun, x0, bounds=(lb, ub), max_nfev=int(max_nfev), method="trf",
                        # [J] 1 % relative finite-difference step: run_case rounds every series value to
                        # 1e-6 um, so scipy's default ~1.5e-8 step is below the output resolution and
                        # would return a zero Jacobian (and a false "gtol" convergence) at the start.
                        diff_step=1.0e-2)
    theta = _full_theta(theta_of(opt.x))
    r = fun(opt.x)
    rms_sigma = float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan")

    J = np.asarray(opt.jac, float)
    if J.size and np.all(np.isfinite(J)):
        # rank of the COLUMN-NORMALISED Jacobian: the log/linear variables are not comparable in
        # magnitude, and a near-zero column means the parameter does not move the data at all. [J]
        norms = np.linalg.norm(J, axis=0)
        norms[norms == 0.0] = 1.0
        Jn = J / norms
        sv = np.linalg.svd(Jn, compute_uv=False)
        rank = int(np.sum(sv > 1e-6 * max(float(sv[0]), 1e-300)))
    else:
        rank = 0
    identifiable = bool(n_data > n_free and rank == n_free)

    param_sd: dict[str, float | None] = {n: None for n in free}
    if identifiable:
        try:
            cov = np.linalg.inv(J.T @ J) * float(r @ r) / max(n_data - n_free, 1)
            sd_x = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            for i, n in enumerate(free):
                # [C] delta method: sd(theta) = theta * sd(log theta) for a log-scaled parameter
                param_sd[n] = float(theta[n] * sd_x[i]) if FREE_PARAMS[n][2] else float(sd_x[i])
        except np.linalg.LinAlgError:
            param_sd = {n: None for n in free}

    table = []
    for s in sites:
        pred = predict(s, theta, dx_um, use_pr)
        f_meas = float(s["field_um"]["best"])
        fl_meas = float(s["floor_um"]["best"])
        table.append({
            "id": str(s.get("id", "")),
            "field_meas": round(f_meas, 3), "field_pred": round(pred["field_um"], 3),
            "floor_meas": round(fl_meas, 3), "floor_pred": round(pred["floor_um"], 3),
            "ratio_meas": round(fl_meas / f_meas, 4) if f_meas > 0 else None,
            "ratio_pred": round(pred["ratio"], 4) if np.isfinite(pred["ratio"]) else None,
        })

    warnings: list[str] = []
    samples = {str(s.get("sample") or s.get("id") or i) for i, s in enumerate(sites)}
    if len(samples) < 3:
        warnings.append("single-sample fit: parameters are not validated on independent data")
    for s in sites:
        for w in predict(s, theta, dx_um, use_pr)["warnings"]:
            msg = f"{s.get('id', '?')}: {w}"
            if msg not in warnings:
                warnings.append(msg)
    if not identifiable:
        warnings.append(f"Jacobian rank {rank} < {n_free}: parameters are not jointly identifiable "
                        "from this data; the reported theta is one of many equally good solutions")

    return {"theta": theta, "free": free, "n_data": n_data, "n_free": n_free,
            "rms_sigma": rms_sigma, "table": table, "identifiable": identifiable,
            "warnings": warnings, "param_sd": param_sd}


# ---------------------------------------------------------------------------------- CLI
_HEADER = [("id", 22), ("field_meas", 10), ("field_pred", 10), ("floor_meas", 10),
           ("floor_pred", 10), ("ratio_meas", 10), ("ratio_pred", 10)]


def _row(cells, header=False) -> str:
    parts = []
    for (name, width), cell in zip(_HEADER, cells):
        if header or cell is None or isinstance(cell, str):
            text = "" if cell is None else str(cell)
        else:
            text = f"{cell:g}"
        parts.append(text.ljust(width) if name == "id" else text.rjust(width))
    return "  ".join(parts).rstrip()


def _print_result(out: dict) -> None:
    print(f"sites: {len(out['table'])}  data: {out['n_data']}  free: {out['n_free']}  "
          f"rms: {out['rms_sigma']:.2f} sigma")
    print(_row([n for n, _ in _HEADER], header=True))
    for r in out["table"]:
        print(_row([r["id"], r["field_meas"], r["field_pred"], r["floor_meas"], r["floor_pred"],
                    r["ratio_meas"], r["ratio_pred"]]))
    print("theta:")
    for name in FREE_PARAMS:
        sd = out["param_sd"].get(name)
        mark = "*" if name in out["free"] else " "
        sd_txt = "" if sd is None else f"  +/- {sd:.3g}"
        print(f" {mark} {name:<10s} {out['theta'][name]:.6g}{sd_txt}")
    print(f"identifiable: {out['identifiable']}")
    for w in out["warnings"]:
        print(f"warning: {w}")


def main(argv=None):
    """CLI: fit the requested FREE_PARAMS to the SEM record and write a calibration JSON."""
    ap = argparse.ArgumentParser(
        prog="python -m platesim.feature.calibrate",
        description="Fit trench fill parameters to SEM-measured Cu thicknesses (additive-free model).")
    ap.add_argument("--free", nargs="+", required=True, choices=sorted(FREE_PARAMS),
                    help="parameters to fit (subset of FREE_PARAMS)")
    ap.add_argument("--data", default=None, help=f"sites JSON (default {DATA_PATH})")
    ap.add_argument("--dx", type=float, default=None, help="grid spacing, um (default: engine rule)")
    ap.add_argument("--no-pr", action="store_true", help="treat pulse-reverse sites as DC")
    ap.add_argument("--max-nfev", type=int, default=40, help="least_squares budget (default 40)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"calibration JSON (default {DEFAULT_OUT})")
    a = ap.parse_args(argv)

    sites = load_sites(a.data)
    use_pr = not a.no_pr
    try:
        out = fit(sites, a.free, dx_um=a.dx, use_pr=use_pr, max_nfev=a.max_nfev)
    except ValueError as exc:
        raise SystemExit(f"calibration failed: {exc}")

    _print_result(out)
    payload = {"fitted_with": "python -m platesim.feature.calibrate",
               "data": a.data or DATA_PATH, "dx_um": a.dx, "use_pr": use_pr, **out}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
