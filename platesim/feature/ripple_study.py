"""Ripple diagnosis sweep for the level-set feature-fill engine (plan Task 7, wave 4).

Is the rippled field of the 4.5 h `trench_full` runs real or an artefact?  One variable at a time on
top of the baseline case of `tmp/ripple_probe.py`: geometry trench_full, additive-free
(suppressor=False), pulse-reverse 15 mA/cm2 net, 4.5 h, delta 64 um, dx 8 um, f_open 0.2.

Metrics come from the FINAL contour only (surface_stats), so the sweep does not depend on the engine
diagnostics that another worker adds to `series` (field_mean_um, ...): the contour is resampled onto
the column centres x_i = (i + 0.5) dx and every statistic is an interpolated crossing of the
polylines, never a nearest-point pick.
  field_*_um   thickness = upper crossing - H over the columns far from both walls,
               |x - pitch/2| > a + 150 um (outer quarter of each field side if the pitch leaves none);
               the flat-wall test (f_open = 1, no field region) measures the thickness from the
               electrode plane z = 0 instead, where the surface starts;
  shoulder_max_um  same thickness over a < |x - pitch/2| <= a + 150 um (the corner region);
  floor_*_um   LOWEST crossing over the central 40 % of the trench width, |x - pitch/2| < 0.4 a;
  wall_up/down_um  horizontal deposit on the left / right wall at mid depth z = H/2;
  ratio_mean   floor_mean / field_mean;  ripple_wavelength_um = largest non-DC peak of the FFT of the
               detrended field profile (NaN when the field is flatter than 0.05 um).

CLI:  python -m platesim.feature.ripple_study [--quick] [--groups base,shear,...] [--sites narrow,wide]
          [--workers 6] [--out platesim/feature/ripple_study.json] [--list] [--table] [--force]

The driver runs the selected cases in a ProcessPoolExecutor (longest cases first: wide before narrow,
small dx first) and rewrites the --out JSON atomically after every finished case, so an interrupted
sweep resumes where it stopped (a name is skipped when its stored result has error == None;
--force re-runs everything).  --list prints the case names, --table prints the Markdown summary of
the stored JSON without running anything.  --quick = 0.5 h, narrow site, dx 8 (smoke test).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import tempfile
import time
import traceback

import numpy as np

from platesim.feature.calibrate import load_sites
from platesim.feature.engine import Params, pulse_reverse, run_case
from platesim.feature.engine.params import derive
from platesim.trenchflow.model import shear_from_delta

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(PKG_DIR, "ripple_study.json")

HOURS = 4.5                     # [J] same duration as the wave-3 trench_full runs
QUICK_HOURS = 0.5
J_MA = 15.0                     # net mA/cm2 of the pulse-reverse waveform
F_OPEN = 0.2                    # [M] open fraction of the measured trench array (calibrate.F_OPEN)
PR = dict(t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)
DELTA_UM = 64.0                 # [J] central Nernst layer of the baseline (calibrate default)
DX_UM = 8.0
WALL_BAND_UM = 150.0            # "far from the wall" band of the field metrics
FLAT = dict(d_um=227.0, H_um=88.0)      # flat-wall test: no trench, same pitch and depth as narrow

# Baseline Params of every case; run_one() adds H_um from the site and the per-case overrides.
BASELINE = dict(geometry="trench_full", suppressor=False, ohmic=True, dx_um=DX_UM, delta_um=DELTA_UM,
                k_mix=0.0, max_overburden_um=140.0)
SHEAR_LIST = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
GROUPS = ("base", "legacy", "flat", "shear", "dx", "delta", "reverse", "update", "current", "fopen",
          "noohmic", "eff", "tertiary")


# ---------------------------------------------------------------------------------- sites / grid
def _sites() -> dict:
    """The two Trench 1 SEM sites keyed by 'narrow' / 'wide' (smallest / largest width)."""
    sites = load_sites()
    return {"narrow": min(sites, key=lambda s: float(s["width_um"])),
            "wide": max(sites, key=lambda s: float(s["width_um"]))}


def _contour(res: dict) -> list:
    """The final psi = 0 polylines of a run_case result."""
    frames = res.get("frames") or []
    if not frames:
        raise ValueError("result has no frames")
    lines = frames[-1].get("contour") or []
    if not lines:
        raise ValueError("the final frame has an empty contour")
    return lines


def _grid(res: dict, contour: list) -> dict:
    """Grid scalars of the run (R_um = pitch, a_um = half width, H_um, dx_um) with fallbacks."""
    grid = (res.get("meta") or {}).get("grid") or {}
    dx = grid.get("dx_um")
    if dx is None:
        nr = grid.get("nr")
        dx = (float(grid["R_um"]) / float(nr)) if (grid.get("R_um") and nr) else DX_UM
    dx = float(dx)
    R = grid.get("R_um")
    if R is None:
        xmax = max((float(np.max(np.asarray(ln, float)[:, 0])) for ln in contour if len(ln)), default=0.0)
        R = xmax + 0.5 * dx
    return dict(R_um=float(R), a_um=float(grid.get("a_um") or 0.0), H_um=float(grid.get("H_um") or 0.0),
                dx_um=dx)


def _last_finite(series: dict, key: str):
    """Last finite value of a series key (None when the key is absent or empty)."""
    for v in reversed(series.get(key) or []):
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


# ---------------------------------------------------------------------------------- contour -> surface
def _column_crossings(contour: list, xs: np.ndarray):
    """Upper (max) and lower (min) psi = 0 crossing of every column centre in `xs`.

    Linear interpolation along every polyline segment, one column range per segment (vertical
    segments carry no single-valued crossing and are skipped; surface_stats handles the walls by a
    horizontal cut at z = H/2).  Columns with no crossing are NaN."""
    n = xs.size
    upper = np.full(n, -np.inf)
    lower = np.full(n, np.inf)
    for ln in contour:
        P = np.asarray(ln, float)
        if P.ndim != 2 or P.shape[0] < 2:
            continue
        xa, za, xb, zb = P[:-1, 0], P[:-1, 1], P[1:, 0], P[1:, 1]
        dxs = xb - xa
        for i in np.nonzero(np.abs(dxs) > 1e-12)[0]:
            lo, hi = (xa[i], xb[i]) if xa[i] < xb[i] else (xb[i], xa[i])
            i0 = int(np.searchsorted(xs, lo, side="left"))      # endpoints included, so a column that
            i1 = int(np.searchsorted(xs, hi, side="right"))     # falls on a polyline vertex is not lost
            if i1 <= i0:
                continue
            t = (xs[i0:i1] - xa[i]) / dxs[i]
            z = za[i] + t * (zb[i] - za[i])
            np.maximum(upper[i0:i1], z, out=upper[i0:i1])
            np.minimum(lower[i0:i1], z, out=lower[i0:i1])
    upper[np.isneginf(upper)] = np.nan
    lower[np.isposinf(lower)] = np.nan
    return upper, lower


def _surface(res: dict):
    """(contour, grid, xs, upper, lower) of the final frame, resampled on the column centres."""
    contour = _contour(res)
    g = _grid(res, contour)
    n = max(int(round(g["R_um"] / g["dx_um"])), 1)
    xs = (np.arange(n) + 0.5) * g["dx_um"]
    upper, lower = _column_crossings(contour, xs)
    return contour, g, xs, upper, lower


def _x_at_z(contour: list, z_cut: float) -> list:
    """x of every crossing of the horizontal line z = z_cut with the contour (vertical segments included,
    so a wall that has not moved yet still reports its own x)."""
    out: list[float] = []
    for ln in contour:
        P = np.asarray(ln, float)
        if P.ndim != 2 or P.shape[0] < 2:
            continue
        xa, za, xb, zb = P[:-1, 0], P[:-1, 1], P[1:, 0], P[1:, 1]
        dz = zb - za
        span = ((za - z_cut) * (zb - z_cut)) <= 0.0
        if not span.any():
            continue
        flat = dz == 0.0
        t = np.where(flat, 0.5, (z_cut - za) / np.where(flat, 1.0, dz))
        x = np.where(flat, 0.5 * (xa + xb), xa + t * (xb - xa))
        out.extend(float(v) for v in x[span])
    return out


def _wall_um(contour: list, g: dict):
    """(left, right) horizontal deposit thickness at mid depth z = H/2; NaN where no crossing exists.

    Only crossings inside the trench width count (a crossing of the far field surface would otherwise
    be mistaken for a wall); a wall that has not moved reports 0."""
    xc, a = 0.5 * g["R_um"], g["a_um"]
    if not (a > 0.0 and g["H_um"] > 0.0):
        return float("nan"), float("nan")
    xs = _x_at_z(contour, 0.5 * g["H_um"])
    left = [x for x in xs if xc - a <= x <= xc]
    right = [x for x in xs if xc < x <= xc + a]
    up = (max(left) - (xc - a)) if left else float("nan")
    down = ((xc + a) - min(right)) if right else float("nan")
    return float(up), float(down)


def _ripple_um(xs: np.ndarray, thick: np.ndarray, sel: np.ndarray, xc: float, pitch: float, dx: float):
    """Dominant wavelength of the detrended field profile (NaN below 0.05 um of thickness spread)."""
    m = sel & np.isfinite(thick)
    n = int(m.sum())
    if n < 8:
        return float("nan")
    order = np.argsort(np.mod(xs[m] - xc, pitch))
    t = thick[m][order]
    if float(np.std(t)) < 0.05:
        return float("nan")
    k = np.arange(n, dtype=float)
    t = t - np.polyval(np.polyfit(k, t, 1), k)          # detrend: remove the mean slope
    spec = np.abs(np.fft.rfft(t))
    if spec.size < 3:
        return float("nan")
    kpk = int(np.argmax(spec[1:])) + 1                  # largest non-DC peak
    if kpk == 0 or spec[kpk] <= 0.0:
        return float("nan")
    return float(n * dx / kpk)


# ---------------------------------------------------------------------------------- statistics
def _stats(res: dict, contour: list, g: dict, xs: np.ndarray, upper: np.ndarray, lower: np.ndarray,
           flat: bool = False) -> dict:
    """The surface statistics of one run (see the module docstring for the definitions)."""
    H, R, a, dx = g["H_um"], g["R_um"], g["a_um"], g["dx_um"]
    xc = 0.5 * R
    off = np.abs(xs - xc)
    # flat electrode (f_open = 1): the whole pitch is the cavity, so the surface starts at z = 0 and the
    # 'field level' H does not exist; thickness is measured from the electrode plane z = 0.
    thick = upper - (0.0 if flat else H)
    if flat:                                            # flat electrode: no trench, all columns are field
        sel_field = np.ones(xs.size, bool)
        sel_shoulder = np.zeros(xs.size, bool)
        sel_floor = np.zeros(xs.size, bool)
    else:
        sel_field = off > a + WALL_BAND_UM
        if not sel_field.any():                         # pitch too small for the 150 um band
            sel_field = off > a + 0.25 * (0.5 * R - a)
        sel_shoulder = (off > a) & (off <= a + WALL_BAND_UM)
        sel_floor = off < 0.4 * a
    t_field = thick[sel_field & np.isfinite(thick)]
    t_sh = thick[sel_shoulder & np.isfinite(thick)]
    z_floor = lower[sel_floor & np.isfinite(lower)]
    field_mean = float(t_field.mean()) if t_field.size else float("nan")
    floor_mean = float(z_floor.mean()) if z_floor.size else float("nan")
    ratio = float(floor_mean / field_mean) if (math.isfinite(field_mean) and field_mean != 0.0
                                              and math.isfinite(floor_mean)) else float("nan")
    wall_up, wall_down = _wall_um(contour, g)
    series = res.get("series") or {}
    result = res.get("result") or {}
    meta = res.get("meta") or {}
    face = _last_finite(series, "face_thickness_um")
    floor_legacy = _last_finite(series, "floor_height_um")
    return dict(
        field_mean_um=field_mean,
        field_min_um=float(t_field.min()) if t_field.size else float("nan"),
        field_max_um=float(t_field.max()) if t_field.size else float("nan"),
        field_std_um=float(t_field.std()) if t_field.size else float("nan"),
        shoulder_max_um=float(t_sh.max()) if t_sh.size else float("nan"),
        floor_mean_um=floor_mean,
        floor_min_um=float(z_floor.min()) if z_floor.size else float("nan"),
        floor_max_um=float(z_floor.max()) if z_floor.size else float("nan"),
        wall_up_um=wall_up,
        wall_down_um=wall_down,
        ratio_mean=ratio,
        ripple_wavelength_um=_ripple_um(xs, thick, sel_field, xc, R, dx),
        face_thickness_um=face,
        floor_height_um=floor_legacy,
        ratio_legacy=(float(floor_legacy / face) if (face not in (None, 0.0) and face > 0.0
                                                     and floor_legacy is not None) else None),
        cu_balance_rel=result.get("cu_balance_rel"),
        cu_balance_adv_rel=result.get("cu_balance_adv_rel"),     # added by the other worker
        wall_s=meta.get("wall_s"),
        steps=meta.get("steps"),
    )


def _surface_and_stats(res: dict, flat: bool = False):
    """(stats, profile) of the final contour; profile = column-resampled surface rounded to 0.01 um."""
    contour, g, xs, upper, lower = _surface(res)
    stats = _stats(res, contour, g, xs, upper, lower, flat=flat)
    profile = dict(
        x_um=[round(float(v), 2) for v in xs],
        upper_um=[None if not math.isfinite(v) else round(float(v), 2) for v in upper],
        lower_um=[None if not math.isfinite(v) else round(float(v), 2) for v in lower])
    return stats, profile


def surface_stats(res: dict, flat: bool = False) -> dict:
    """Field / floor / wall / ripple statistics of one run_case result (final contour only).

    flat=True treats every column as field (the f_open=1 flat-electrode test, where the 150 um band
    next to the "wall" leaves no column at all)."""
    return _surface_and_stats(res, flat=flat)[0]


# ---------------------------------------------------------------------------------- one case
def run_one(case: dict) -> dict:
    """Run one sweep case; never raises (the traceback is returned in 'error')."""
    case = dict(case or {})
    try:
        flat = bool(case.get("flat", False))
        site = _sites()[str(case.get("site", "narrow"))]
        params = dict(BASELINE)
        params["H_um"] = float(site["depth_um"])
        params.update({k: v for k, v in (case.get("params") or {}).items() if v is not None})
        d_um, T_C, f_open = float(site["width_um"]), float(site["T_C"]), float(case.get("f_open", F_OPEN))
        if flat:
            d_um, f_open = float(FLAT["d_um"]), 1.0
            params["H_um"] = float(FLAT["H_um"])
        p = Params(**params)
        rec = pulse_reverse(float(case.get("j", J_MA)), float(case.get("hours", HOURS)), **PR)
        res = run_case(d_um, T_C, rec, p, f_open=f_open, label=str(case.get("name", "")), out=None)
        stats, profile = _surface_and_stats(res, flat=flat)
        return dict(case=case, stats=stats, profile=profile, error=None)
    except BaseException:                                # noqa: BLE001 - a sweep records, never raises
        return dict(case=case, stats=None, profile=None, error=traceback.format_exc())


# ---------------------------------------------------------------------------------- case list
def _tag(v) -> str:
    """Filesystem-safe tag of a float ('G3', 'G0p1', 'j5')."""
    return f"{float(v):g}".replace("-", "m").replace(".", "p")


def _base_shear(T_C: float) -> float:
    """Wall shear rate (1/s) the engine derives from the baseline delta_um (sim.py, shear_from_delta)."""
    return float(shear_from_delta(DELTA_UM, derive(Params(), float(T_C)).D_c))


def build_cases(quick: bool = False) -> list[dict]:
    """The sweep as a list of cases; quick = 0.5 h, narrow site, dx 8 only (smoke test)."""
    hours = QUICK_HOURS if quick else HOURS
    sites = ("narrow",) if quick else ("narrow", "wide")
    cases: list[dict] = []

    def add(group: str, name: str, site: str, **kw) -> None:
        c = dict(name=name, site=site, group=group, hours=hours, j=J_MA, f_open=F_OPEN, params={})
        c.update(kw)
        cases.append(c)

    for s in sites:
        add("base", f"base_{s}", s)                                            # baseline (adv_form default)
    for s in sites:
        add("legacy", f"legacy_{s}", s, params={"adv_form": "divergence"})     # legacy conservative form
    for dx in ((8.0,) if quick else (8.0, 4.0)):
        add("flat", f"flat_dx{_tag(dx)}", "narrow", flat=True, params={"dx_um": dx})
    for s in sites:
        for g in SHEAR_LIST:
            add("shear", f"shear_{s}_G{_tag(g)}", s, params={"flow_shear_s": float(g)})
    for dx in ((8.0,) if quick else (8.0, 6.0, 4.0, 3.0)):
        add("dx", f"dx_narrow_{_tag(dx)}", "narrow", params={"dx_um": dx})
    for dx in (() if quick else (8.0, 6.0, 4.0)):
        add("dx", f"dx_wide_{_tag(dx)}", "wide", params={"dx_um": dx})
    for s in sites:
        for d in (32.0, 64.0, 128.0):
            add("delta", f"delta_{s}_{_tag(d)}", s, params={"delta_um": float(d)})
    add("reverse", "reverse_narrow", "narrow",                                # minus the baseline shear
        params={"flow_shear_s": -_base_shear(float(_sites()["narrow"]["T_C"]))})
    add("update", "update_narrow", "narrow", params={"flow_every": 2, "flow_move_cells": 0.25})
    # same charge passed as 15 mA/cm2 for `hours`
    for j in (5.0, 10.0):
        add("current", f"current_narrow_j{_tag(j)}", "narrow", j=j, hours=hours * J_MA / j)
    for fo in (0.1, 0.4):
        add("fopen", f"fopen_narrow_{_tag(fo)}", "narrow", f_open=fo)
    add("noohmic", "noohmic_narrow", "narrow", params={"ohmic": False})
    for s in sites:
        add("eff", f"eff_{s}", s, params={"efficiency": 0.828})
    add("tertiary", "tertiary_narrow", "narrow", params={"transport": "tertiary"})
    add("tertiary", "tertiary_narrow_noflow", "narrow",
        params={"transport": "tertiary", "flow_shear_s": 0.0})
    if quick:
        keep = []
        for c in cases:
            dx = float((c.get("params") or {}).get("dx_um", DX_UM))
            if c["site"] == "narrow" and dx == 8.0:
                keep.append(c)
        cases = keep
    return cases


def _select(cases: list[dict], groups=None, sites=None) -> list[dict]:
    """Filter by group / site names (None or empty = keep everything)."""
    out = list(cases)
    if groups:
        out = [c for c in out if c["group"] in set(groups)]
    if sites:
        out = [c for c in out if c["site"] in set(sites)]
    return out


# ---------------------------------------------------------------------------------- reporting
def _cell(v, nd: int = 2) -> str:
    """Markdown cell of a number ('-' for None / NaN)."""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v).replace("|", "/")
    return "-" if not math.isfinite(f) else f"{f:.{nd}f}"


def _oneline(text: str) -> str:
    return (text or "").strip().splitlines()[-1].replace("|", "/") if text else ""


def _line(r: dict) -> str:
    """One console line per finished case."""
    name = str((r.get("case") or {}).get("name", "?"))
    if r.get("error"):
        return f"{name}: ERROR {_oneline(r['error'])}"
    s = r.get("stats") or {}
    return (f"{name}: field {_cell(s.get('field_mean_um'))}/{_cell(s.get('field_min_um'))}/"
            f"{_cell(s.get('field_max_um'))} um  floor {_cell(s.get('floor_mean_um'))}  "
            f"ratio {_cell(s.get('ratio_mean'), 4)}  shoulder {_cell(s.get('shoulder_max_um'))}  "
            f"cu {_cell(s.get('cu_balance_rel'), 5)}  adv {_cell(s.get('cu_balance_adv_rel'), 5)}  "
            f"wall {_cell(s.get('wall_s'), 1)} s")


def summary_table(results: dict) -> str:
    """Markdown table of a {name: run_one result} mapping, sorted by group then name."""
    rows = sorted((results or {}).items(),
                  key=lambda kv: (str(((kv[1] or {}).get("case") or {}).get("group", "")), str(kv[0])))
    head = ("| case | group | field mean | field min | field max | floor mean | ratio | shoulder max "
            "| cu bal | cu adv | wall s | ripple um |")
    lines = [head, "|" + "---|" * 12]
    for name, r in rows:
        r = r or {}
        case = r.get("case") or {}
        group = str(case.get("group", "-"))
        if r.get("error"):
            lines.append(f"| {name} | {group} | " + " | ".join(["ERROR"] * 10)
                         + f" | {_oneline(r['error'])} |")
            continue
        s = r.get("stats") or {}
        lines.append("| " + " | ".join([
            str(name), group,
            _cell(s.get("field_mean_um")), _cell(s.get("field_min_um")), _cell(s.get("field_max_um")),
            _cell(s.get("floor_mean_um")), _cell(s.get("ratio_mean"), 4), _cell(s.get("shoulder_max_um")),
            _cell(s.get("cu_balance_rel"), 5), _cell(s.get("cu_balance_adv_rel"), 5),
            _cell(s.get("wall_s"), 1), _cell(s.get("ripple_wavelength_um"), 1)]) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- resumable JSON
def _jsonable(x):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> python scalars."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    return x


def _load(path: str) -> dict:
    """Existing payload ({created, results}) or a fresh one."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict) and isinstance(doc.get("results"), dict):
            doc.setdefault("created", time.time())
            return doc
    except (OSError, ValueError):
        pass
    return dict(created=time.time(), results={})


def _write(path: str, payload: dict) -> None:
    """Atomic rewrite: temp file in the same folder, then os.replace."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(payload), fh, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _done(r) -> bool:
    return bool(r) and r.get("error") is None and bool(r.get("stats"))


def _priority(case: dict):
    """Longest cases first: wide before narrow, small dx before large, more hours before fewer."""
    dx = float((case.get("params") or {}).get("dx_um", DX_UM))
    return (0 if case.get("site") == "wide" else 1, dx, -float(case.get("hours", HOURS)))


# ---------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m platesim.feature.ripple_study",
        description="Ripple diagnosis sweep of the trench_full level-set engine (plan Task 7, wave 4).")
    ap.add_argument("--quick", action="store_true", help="0.5 h, narrow site, dx 8 only (< 1 min)")
    ap.add_argument("--groups", default="", help=f"comma separated subset of {','.join(GROUPS)}")
    ap.add_argument("--sites", default="", help="comma separated: narrow,wide")
    ap.add_argument("--workers", type=int, default=6, help="ProcessPoolExecutor workers (default 6)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output JSON (default {DEFAULT_OUT})")
    ap.add_argument("--list", action="store_true", help="print the selected case names and exit")
    ap.add_argument("--table", action="store_true", help="print the Markdown table of --out and exit")
    ap.add_argument("--force", action="store_true", help="re-run cases that already have a result")
    a = ap.parse_args(argv)

    cases = _select(build_cases(a.quick),
                    [g.strip() for g in a.groups.split(",") if g.strip()],
                    [s.strip() for s in a.sites.split(",") if s.strip()])
    if a.list:
        for c in cases:
            print(f"{c['name']}\t{c['group']}\t{c['site']}", flush=True)
        return cases
    if a.table:
        payload = _load(a.out)
        print(summary_table(payload["results"]))
        return payload

    payload = _load(a.out)
    results = payload["results"]
    todo = [c for c in cases if a.force or not _done(results.get(c["name"]))]
    todo.sort(key=_priority)
    if not todo:
        print(f"nothing to do: {len(cases)} case(s) selected, all stored in {a.out}", flush=True)
        print(summary_table(results))
        return payload
    print(f"{len(todo)} of {len(cases)} case(s) to run, {len(results)} stored, workers={a.workers}, "
          f"-> {a.out}", flush=True)
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
        futs = {ex.submit(run_one, c): c for c in todo}
        for fut in concurrent.futures.as_completed(futs):
            case = futs[fut]
            try:
                r = fut.result()
            except BaseException:                        # noqa: BLE001 - a dead worker must not end the sweep
                r = dict(case=dict(case), stats=None, profile=None, error=traceback.format_exc())
            results[str((r.get("case") or {}).get("name") or case["name"])] = r
            _write(a.out, payload)                       # resumable: after EVERY finished case
            print(f"[{len(results)}] {_line(r)}", flush=True)
    print(f"done in {time.time() - t0:.0f} s -> {a.out}", flush=True)
    print(summary_table(results), flush=True)
    return payload


if __name__ == "__main__":
    main()




