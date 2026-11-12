"""Coupled coupon pipeline: flow -> cell -> efficiency for the 2x2 cm Si coupon.

Stages (each reuses an existing platesim module, no physics duplicated):
1. flow: rpm = rpm_per_pct * stir_pct -> flow.correlation.delta_map with
   MassTransferParams(delta0_um, n) (+baffles per setup), then re-directed by
   theta["flow_dir"] so corner gradients can be fitted (Leveque profile is +x by
   construction in delta_map; rotate_flow remaps it without touching the mean).
2. cell: cell.solver.solve_cell(geom, bath, I = j*4 cm2, delta_map) -> front
   current fraction + local j(x, y) map.  Untaped back/edges are modelled as
   taped=True + stray_area = f_back * (back + edge area) [spec]; solve_cell with
   taped=False (fully active back, front_fraction ~0.48 at 12 mA/cm2 [C]) is the
   upper bound that motivates f_back in [0, 1].
3. efficiency: local true efficiency from the O2 parasitic limit with the LOCAL
   delta (bath.j_lim_o2, same formulation as efficiency.model.predict:
   frac_o2 = j_o2/j_front, H2 clipped above bath.j_lim_cu) and a uniform startup
   loss L0; thickness = Faraday(j_local, eta_local) - L0.

Solve memoisation: solve_cell results are cached on quantised solve-relevant
inputs (see QUANT) so a calibration reuses solves across DOE runs that share
(setup, stir, j) and across repeated parameter points; L0/sigma probes cost no
solve at all.  Shared constants/priors come from efficiency.model (A_MEAS_CM2,
UM_PER_A_S_CM2, prior_median, PARAM_SPECS).

Units: cm, s, A inside code; param suffixes _um, _mA_cm2, _min as in CONTRACTS.
Tags: [FT] full text, [AB] abstract, [SH] secondhand, [C] calc, [J] judgement,
[M] measured (team data), [spec] task specification.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..cell.mesh import COUPON_T_CM
from ..cell.solver import GRID_N, march_cell, solve_cell
from ..core.bath import Bath
from ..core.cell import CellGeometry
from ..core.constants import thickness_um
from ..efficiency.model import A_MEAS_CM2, prior_median
from ..flow.correlation import MassTransferParams, delta_map

SEM_PATH = Path(__file__).resolve().parents[1] / "data" / "sem_measurements.json"

COUPON_W_CM = 2.0            # [M user 2026-10-28] 2 x 2 = 4 cm2 (A_MEAS_CM2)
COUPON_H_CM = 2.0

# filename -> (label, x_cm, y_cm): position on the coupon in the CENTRED frame
# (x across width, y up; corners at +-0.8 cm [spec]; top/bottom cross-sections
# sample the top/bottom half of the coupon -> +-0.5 cm [J spec ambiguity]).
SEM_POSITIONS: dict[str, tuple[str, float, float]] = {
    "planar2_top_left.tif":      ("top_left",   -0.8,  0.8),
    "planar2_top_left_2.tif":    ("top_left_2", -0.8,  0.8),
    "planar2_bottom_left.tif":   ("bottom_left", -0.8, -0.8),
    "planar2_bottom_right.tif":  ("bottom_right", 0.8, -0.8),
    "planar2_9mA_top.tif":       ("run1_top",     0.0,  0.5),
    "planar2_9mA_bottom.tif":    ("run1_bottom",  0.0, -0.5),
    "planar2_18mA_mid.tif":      ("run2_mid",     0.0,  0.0),
    "planar2_18mA_bottom.tif":   ("run2_bottom",  0.0, -0.5),
}

# excluded images -> why (kept visible so the calibration table can explain them)
SEM_EXCLUDED: dict[str, str] = {
    "planar2_top_right-short.tif": "anomalous 1.5-2.0 um, likely shorted during plating, "
                                   "low-medium confidence [M sem_measurements notes]",
    "planar2_9mA_seed.tif":        "high-mag seed/stack zoom: not a full-film thickness [M]",
    "useless.tif":                 "no complete layer stack in frame [M]",
}

FLOW_DIRS = {"+x": 0.0, "-x": 180.0, "+y": 90.0, "-y": 270.0}

# solve-relevant quantisation for the memoise key (step must stay well below the
# calibration finite-difference step: h_u = 0.02*max(1,|u|) -> dx >= 3x each step)
QUANT: dict[str, float] = {
    "delta0_um": 0.2,        # um
    "n_exp": 0.001,          # -
    "rpm_per_pct": 0.05,     # rpm per % dial
    "kappa_scale": 0.005,    # -
    "f_back": 0.001,         # fraction of back+edge area
}

DEFAULT_THETA: dict = {
    "delta0_um": 80.0,                    # prior mean of efficiency.PARAM_SPECS
                                          # [C Fenech 1959 70-100 um unstirred; FT Avdeef 2022 33-47 um @300 rpm]
    "n_exp": 0.5,                         # [FT Avdeef 2022 exponent 0.505]
    "rpm_per_pct": 15.0,                  # [J] core.cell.STIR_PCT_TO_RPM: 20% -> 300 rpm
    "f_back": 0.3,                        # [J] efficiency H4 f_stray_base ~0.31 (calibration_results.json)
    "L0_um": prior_median("L0_um"),       # half-normal median of efficiency prior [same sources]
    "kappa_scale": 1.0,                   # [C] kappa +-20% (Bath.kappa_22, KB)
    "sigma_um": prior_median("sigma_um"), # [M] SEM noise floor, efficiency prior median
    "flow_dir": "+x",                     # [spec] Leveque direction; +x/-x/+y/-y or degrees
    "anode_base_cm": 2.0,                 # [M user] 1:1 anode -> 2x2 cm
    "anode_imp_cm": 5.0,                  # [J] KB sec 7 proposal 5x5 cm (clipped by beaker)
    "f_back_imp": 0.0,                    # [J] taped imp setups: no back/edge leakage modelled
}

ASSUMPTIONS = [
    "coupon 2x2 cm = 4 cm2 [M user 2026-10-28]; SEM points: corners +-0.8 cm, "
    "top/bottom cross-sections at +-0.5 cm, mid at centre [spec]",
    "untaped back+edges modelled as taped=True + stray_area = f_back*(back 4.0 + edges 0.4 cm2) "
    "[spec]; fully active back (taped=False) gives front_fraction ~0.48 at 12 mA/cm2 [C] = upper bound",
    "base: 1:1 anode (2x2 cm), no baffles [M user/deck]; imp1/imp2: 5x5 cm anode (clipped by the "
    "9 cm ID beaker to 3.69 cm width [M solver warning]) + baffles (baffle_factor 0.8 [J])",
    "rpm = rpm_per_pct * stir_pct [J W9 stir%->rpm mapping unknown; core.cell.STIR_PCT_TO_RPM "
    "20%->300 rpm is a guess and a calibration target]",
    "delta(x) is the Leveque profile of flow.correlation.delta_map normalised to delta_mean_um; "
    "flow_dir rotates it (+x/-x/+y/-y or degrees) [spec]",
    "local true efficiency = 1 - j_lim_o2(delta_local)/j_local - H2 above j_lim_cu, the same "
    "formulation as efficiency.model.predict; bath.j_lim_o2 / j_lim_cu reused [C]",
    "uniform startup loss L0 (um) subtracted from every grid cell, clipped at 0 [as efficiency.model]",
    "bath conductivity scaled by kappa_scale [C Bath.kappa_22 +-20%]",
    "solve inputs quantised for memoisation (delta0 0.2 um, rpm/pct 0.05, f_back 0.001, "
    "n 0.001, kappa 0.005) [spec: calibration must finish < 10 min]",
    "coupon means (thickness, j, eff) use trapezoidal endpoint-half grid weights: the "
    "endpoint-inclusive face grid over-weights the hot edge columns by ~2 % with plain "
    "averaging [C plain/trapezoid vs integrated = 1.024/1.005]",
    "no CC->CV fallback / anode film here (efficiency H3 terms absent) [J]: imp runs may be "
    "over-predicted relative to the efficiency module's H4 fit",
    "runs 9-10 (40% stir, imp2) pending [M deck]",
    "SEM planar2_top_right-short.tif excluded: anomalous 1.5-2.0 um, likely shorted [M]",
]


# ---------------------------------------------------------------------------
# SEM points
# ---------------------------------------------------------------------------
def load_sem_points(path: str | Path = SEM_PATH) -> list[dict]:
    """Position-resolved SEM thickness points (excluded images are dropped)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[dict] = []
    for m in data.get("measurements", []):
        f = m.get("file")
        if f not in SEM_POSITIONS or m.get("thickness_um_mean") is None:
            continue
        label, x_cm, y_cm = SEM_POSITIONS[f]
        out.append({
            "label": label,
            "file": f,
            "run_id": m.get("run_id_guess"),
            "x_cm": x_cm,
            "y_cm": y_cm,
            "measured_um": float(m["thickness_um_mean"]),
            "manual_mean_um": m.get("manual_mean_um"),
            "confidence": m.get("confidence"),
        })
    return out


# ---------------------------------------------------------------------------
# geometry + flow
# ---------------------------------------------------------------------------
def back_edge_areas(w: float = COUPON_W_CM, h: float = COUPON_H_CM) -> tuple[float, float]:
    """(back cm2, 4x sidewall cm2) of the Si coupon [J: Si 0.05 cm, mesh.COUPON_T_CM]."""
    return w * h, (2.0 * w + 2.0 * h) * COUPON_T_CM


def build_geometry(setup: str, theta: dict) -> tuple[CellGeometry, dict]:
    """Per-setup cell geometry + bookkeeping.

    base: untaped coupon -> partial backside activity as taped=True with
    stray_area = f_back * (back + edge area) [spec]; imp1/imp2: taped, bigger anode.
    """
    setup = str(setup)
    a_back, a_edge = back_edge_areas()
    if setup == "base":
        side = float(theta.get("anode_base_cm", theta.get("anode_cm", 2.0)))   # 1:1 [M]
        f_back = min(max(float(theta.get("f_back", 0.0)), 0.0), 1.0)
        stray = f_back * (a_back + a_edge)
        attributed = "back+edges"
    elif setup in ("imp1", "imp2"):
        side = float(theta.get("anode_imp_cm", theta.get("anode_cm", 5.0)))    # [J KB sec 7]
        f_imp = min(max(float(theta.get("f_back_imp", 0.0)), 0.0), 1.0)
        stray = f_imp * (a_back + a_edge)
        attributed = "stray (clip)" if f_imp > 0 else "none (taped)"
    else:
        raise ValueError(f"unknown setup {setup!r}; use base/imp1/imp2")
    geom = CellGeometry(
        coupon_w=COUPON_W_CM, coupon_h=COUPON_H_CM,
        anode_w=side, anode_h=side,
        taped=True, stray_area=round(stray, 6),
    )
    book = {
        "setup": setup, "baffles": setup != "base",
        "back_area_cm2": a_back, "edge_area_cm2": a_edge,
        "stray_area_cm2": geom.stray_area,
        "anode_area_cm2": side * side,
        "loss_attribution": attributed,
    }
    return geom, book


def flow_dir_deg(flow_dir) -> float:
    """'+x'/'-x'/'+y'/'-y' or an angle in degrees (0 = +x, CCW toward +y)."""
    if isinstance(flow_dir, str):
        if flow_dir not in FLOW_DIRS:
            raise ValueError(f"flow_dir must be one of {sorted(FLOW_DIRS)} or a number of "
                             f"degrees, got {flow_dir!r}")
        return FLOW_DIRS[flow_dir]
    return float(flow_dir)


def build_delta_map(geom: CellGeometry, bath: Bath, rpm: float, theta: dict,
                    baffles: bool, nx: int = GRID_N, ny: int = GRID_N) -> dict:
    """delta(x, y) on the coupon: delta_map output re-directed by theta['flow_dir']."""
    p = MassTransferParams(delta0_um=float(theta["delta0_um"]), n=float(theta["n_exp"]))
    dmap = delta_map(geom, bath, rpm, p, baffles=baffles, nx=nx, ny=ny)
    return rotate_flow(dmap, theta.get("flow_dir", "+x"))


def rotate_flow(dmap: dict, flow_dir) -> dict:
    """Rotate the Leveque profile to `flow_dir`, preserving the area mean.

    delta_map builds the profile along +x (uniform in y); for +x the original
    dict is returned untouched (exact reuse).  For any other direction the same
    (s - s0 + l0)^(1/3) profile is evaluated along s = x*cos + y*sin with
    l0 = 0.05 * projected extent, normalised to the mean from delta_map.
    """
    deg = flow_dir_deg(flow_dir)
    if deg % 360.0 == 0.0:
        return dmap
    x = np.asarray(dmap["x"], float)
    y = np.asarray(dmap["y"], float)
    th = math.radians(deg)
    ct, st = math.cos(th), math.sin(th)
    X, Y = np.meshgrid(x, y)                       # rows = y (bottom -> top)
    S = X * ct + Y * st
    w_ext = float(x[-1] - x[0])
    h_ext = float(y[-1] - y[0])
    l0 = 0.05 * (w_ext * abs(ct) + h_ext * abs(st))
    s0 = float(S.min())
    raw = np.cbrt(S - s0 + l0)
    scale = float(dmap["mean_um"]) / float(raw.mean())
    d = scale * raw
    return {
        "x": x.tolist(), "y": y.tolist(), "delta_um": d.tolist(),
        "mean_um": float(dmap["mean_um"]), "k_m_cm_s": float(dmap["k_m_cm_s"]),
        "flow_dir_deg": deg,
    }


# ---------------------------------------------------------------------------
# memoised cell solve
# ---------------------------------------------------------------------------
SOLVE_CACHE: dict[str, dict] = {}
SOLVE_STATS = {"hits": 0, "misses": 0}


def clear_solve_cache() -> None:
    SOLVE_CACHE.clear()
    SOLVE_STATS["hits"] = 0
    SOLVE_STATS["misses"] = 0


def solve_cache_stats() -> dict:
    return {"entries": len(SOLVE_CACHE), "hits": SOLVE_STATS["hits"],
            "misses": SOLVE_STATS["misses"]}


def _q(value: float, step: float) -> float:
    return round(round(float(value) / step) * step, 6)


def _solve_cached(geom: CellGeometry, bath: Bath, I_A: float, dmap: dict,
                  resolution: str, key_extra: dict) -> dict:
    """solve_cell memoised on quantised solve-relevant inputs [spec]."""
    payload = {
        "geom": geom.to_dict(),
        "I_A": round(float(I_A), 6),
        "resolution": resolution,
        "kappa_22": round(float(bath.kappa_22), 8),
        "T_C": round(float(bath.T_C), 3),
        **key_extra,
    }
    key = json.dumps(payload, sort_keys=True)
    hit = SOLVE_CACHE.get(key)
    if hit is not None:
        SOLVE_STATS["hits"] += 1
        return hit
    SOLVE_STATS["misses"] += 1
    res = solve_cell(geom, bath, I_A, t_s=3600.0, efficiency=1.0,
                     delta_um=float(dmap["mean_um"]), delta_map=dmap,
                     suppressed=True, resolution=resolution)
    entry = {
        "front_fraction": float(res.front_fraction),
        "loss_fraction": float(res.loss_fraction),
        "j_grid": np.asarray(res.j_mA_cm2, float),
        "x_cm": np.asarray(res.x_cm, float),
        "z_cm": np.asarray(res.z_cm, float),
        "solve_time_s": float(res.solve_time_s),
        "n_newton": int(res.n_newton),
        "V_cell_V": float(res.V_cell_V),
        "warnings": list(res.warnings),
    }
    SOLVE_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# gridded sampling + local efficiency
# ---------------------------------------------------------------------------
def _sampler(x, y, F):
    """Bilinear sampler F(x, y) on an ascending rectilinear grid -> callable."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    F = np.asarray(F, float)
    nx, ny = len(x), len(y)
    sx = (nx - 1) / (x[-1] - x[0])
    sy = (ny - 1) / (y[-1] - y[0])

    def f(xq, yq):
        xq = np.asarray(xq, float)
        yq = np.asarray(yq, float)
        fx = np.clip((xq - x[0]) * sx, 0.0, nx - 1.0)
        fy = np.clip((yq - y[0]) * sy, 0.0, ny - 1.0)
        ix = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        tx = fx - ix
        ty = fy - iy
        return (F[iy, ix] * (1 - tx) * (1 - ty) + F[iy, ix + 1] * tx * (1 - ty)
                + F[iy + 1, ix] * (1 - tx) * ty + F[iy + 1, ix + 1] * tx * ty)

    return f


def local_efficiency(j_mA_cm2, delta_um, bath: Bath) -> np.ndarray:
    """True local cathodic efficiency: 1 - O2 parasitic - H2 above j_lim_Cu.

    Same formulation as efficiency.model.predict (frac_o2 = j_lim_o2/j_front,
    frac_h2 clipped above bath.j_lim_cu); delta enters both limits locally.
    """
    d_cm = np.maximum(np.asarray(delta_um, float), 1e-6) * 1e-4
    j = np.maximum(np.asarray(j_mA_cm2, float), 1e-9)
    # j_lim_* are annotated scalar but are pure arithmetic: vectorised here on purpose
    j_o2_mA = bath.j_lim_o2(d_cm) * 1e3            # type: ignore[arg-type]  A/cm2 -> mA/cm2
    j_lim_mA = bath.j_lim_cu(d_cm) * 1e3           # type: ignore[arg-type]
    frac_o2 = np.minimum(1.0, j_o2_mA / j)
    frac_h2 = np.clip((j - j_lim_mA) / j, 0.0, 1.0)
    return np.clip(1.0 - frac_o2 - frac_h2, 0.0, 1.0)


def _trap_weights(n: int) -> np.ndarray:
    """Trapezoid (endpoint-half) weights for a uniform endpoint-inclusive grid."""
    w = np.ones(n, dtype=float)
    if n > 1:
        w[0] = w[-1] = 0.5
    return w


def _delta_on_grid(dmap: dict, x_cm: np.ndarray, z_cm: np.ndarray,
                   center_z: float) -> np.ndarray:
    """delta on the j grid (rows: bottom->top, cols: left->right)."""
    dx = np.asarray(dmap["x"], float)
    dy = np.asarray(dmap["y"], float) - 0.0        # y from bottom (0)
    dd = np.asarray(dmap["delta_um"], float)
    y_rel = dy - (z_cm[-1] - z_cm[0]) / 2.0        # to centred frame
    if (dd.shape == (len(z_cm), len(x_cm))
            and np.allclose(dx, x_cm, atol=1e-9)
            and np.allclose(y_rel, z_cm - center_z, atol=1e-9)):
        return dd
    f = _sampler(dx, y_rel, dd)
    X, Z = np.meshgrid(x_cm, z_cm)
    return np.asarray(f(X, Z - center_z), float)


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
def quantised_theta(theta: dict) -> dict:
    """Theta with solve-relevant params snapped to the memoisation quanta [spec]."""
    th = dict(theta)
    for k, step in QUANT.items():
        th[k] = _q(th[k], step)
    return th


def simulate_run(run: dict, theta: dict | None = None, resolution: str = "coarse",
                 bath: Bath | None = None, progress=None, pr: dict | None = None,
                 march: bool | dict = False) -> dict:
    """One coupled simulation: flow -> cell -> efficiency.

    run: {j_mA_cm2, t_min, stir_pct, setup, T_C?} (+ optional id for SEM points).
    theta: see DEFAULT_THETA; missing keys fall back to the defaults.
    pr: optional pulse-reverse waveform {'t_f_ms', 't_r_ms', 't_off_ms', 'rr'} for the
        NET current (F3, see cell.solver.solve_cell); None keeps the DC solve.
    march: False (default) = one steady cell solve; True = F4 time march with the
        default 6 stages; a dict is forwarded to cell.solver.march_cell
        (e.g. {"n_stages": 8, "seed_R_sq0": 0.1}).
    Returns a JSON-serialisable dict with maps (j, thickness, delta, eta),
    current split, loss breakdown and predicted SEM-point thicknesses.
    """
    t_start = time.perf_counter()
    th = {**DEFAULT_THETA, **(theta or {})}
    setup = str(run.get("setup", "base"))
    j_set = float(run["j_mA_cm2"])
    t_s = float(run.get("t_min", 60.0)) * 60.0
    stir = float(run.get("stir_pct", 30.0))
    T_C = float(run.get("T_C", th.get("T_C", 22.0)))

    base_bath = (bath or Bath()).at(T_C)
    kappa_scale = float(th["kappa_scale"])
    bath_used = replace(base_bath, kappa_22=base_bath.kappa_22 * kappa_scale)

    th_q = quantised_theta(th)
    rpm = th_q["rpm_per_pct"] * stir
    geom, book = build_geometry(setup, th_q)
    baffles = book["baffles"]

    # ---- stage 1: flow ------------------------------------------------------
    if progress:
        progress(0.1, f"flow: {stir:g}% stir -> {rpm:.0f} rpm, "
                      f"{'baffles' if baffles else 'no baffles'}, delta map")
    dmap = build_delta_map(geom, bath_used, rpm, th_q, baffles)

    # ---- stage 2: cell ------------------------------------------------------
    if progress:
        progress(0.3, f"cell: solve at j = {j_set:g} mA/cm2 ({resolution})")
    I_A = j_set * 1e-3 * geom.coupon_area
    key_extra = {
        "delta0_um": th_q["delta0_um"], "n_exp": th_q["n_exp"],
        "rpm": round(rpm, 4), "kappa_scale": th_q["kappa_scale"],
        "flow_dir": flow_dir_deg(th.get("flow_dir", "+x")),
        "baffles": baffles,
    }
    solved = _solve_cached(geom, bath_used, I_A, dmap, resolution, key_extra)
    j_grid = solved["j_grid"]
    x_cm, z_cm = solved["x_cm"], solved["z_cm"]

    # ---- optional F3 pulse reverse / F4 time march --------------------------
    # Both bypass the steady-solve memo: the cache key carries (setup, stir, j)
    # only, and neither the waveform nor the deposit history belongs in it.
    march_info: dict | None = None
    march_final_um: np.ndarray | None = None
    if pr is not None or march:
        if progress:
            progress(0.4, "cell: " + ("F4 time march" if march else "F3 pulse reverse"))
        if march:
            m_kw = dict(march) if isinstance(march, dict) else {}
            mr = march_cell(geom, bath_used, I_A, t_s / 3600.0, pr=pr,
                            delta_um=float(dmap["mean_um"]), delta_map=dmap,
                            suppressed=True, resolution=resolution, **m_kw)
            last = mr.stages[-1]
            solved = {
                "front_fraction": last.front_fraction, "loss_fraction": last.loss_fraction,
                "j_grid": np.asarray(last.j_mA_cm2, float),
                "x_cm": np.asarray(mr.x_cm, float), "z_cm": np.asarray(mr.z_cm, float),
                "solve_time_s": float(mr.solve_time_s), "n_newton": int(mr.n_newton_total),
                "V_cell_V": float(mr.V_cell_last_V), "warnings": list(mr.warnings),
            }
            march_final_um = np.asarray(mr.thickness_um, float)   # pure Faraday, um
            march_info = {
                "n_stages": mr.n_stages, "hours": mr.hours,
                "seed_R_sq0": mr.seed_R_sq0, "seed_t_um": mr.seed_t_um,
                "R_s_mean_ohm_sq": float(np.mean(mr.R_s_ohm_sq))
                if mr.R_s_ohm_sq is not None else None,
                "mass_g": mr.mass_g, "mass_faraday_g": mr.mass_faraday_g,
                "mass_balance": mr.mass_balance,
                "V_cell_first_V": mr.V_cell_first_V, "V_cell_last_V": mr.V_cell_last_V,
                "stages": [{"index": s.index, "t_end_h": s.t_end_h,
                            "near_far_thickness_ratio": s.near_far_thickness_ratio,
                            "edge_center_thickness_ratio": s.edge_center_thickness_ratio,
                            "near_far_j_ratio": s.near_far_j_ratio,
                            "R_s_mean_ohm_sq": s.R_s_mean_ohm_sq,
                            "mass_balance": (s.mass_g / s.mass_faraday_g)
                            if s.mass_faraday_g else None}
                           for s in mr.stages],
            }
        else:
            res = solve_cell(geom, bath_used, I_A, t_s=3600.0, delta_um=float(dmap["mean_um"]),
                             delta_map=dmap, suppressed=True, resolution=resolution, pr=pr)
            solved = {
                "front_fraction": float(res.front_fraction),
                "loss_fraction": float(res.loss_fraction),
                "j_grid": np.asarray(res.j_mA_cm2, float),
                "x_cm": np.asarray(res.x_cm, float), "z_cm": np.asarray(res.z_cm, float),
                "solve_time_s": float(res.solve_time_s), "n_newton": int(res.n_newton),
                "V_cell_V": float(res.V_cell_V), "warnings": list(res.warnings),
            }
        j_grid = solved["j_grid"]
        x_cm, z_cm = solved["x_cm"], solved["z_cm"]

    # ---- stage 3: efficiency + thickness ------------------------------------
    if progress:
        progress(0.7, "efficiency: local O2 parasitic + startup loss")
    delta_grid = _delta_on_grid(dmap, x_cm, z_cm, geom.coupon_center_z)
    eta = local_efficiency(j_grid, delta_grid, bath_used)
    L0 = max(float(th["L0_um"]), 0.0)
    if march_final_um is None:
        # vectorised on purpose: Faraday law is elementwise (see efficiency.model)
        faraday_grid = np.asarray(thickness_um(j_grid, t_s, eta), float)  # type: ignore[arg-type]
    else:
        # the march already integrated Faraday over the run at 100 % efficiency;
        # the local efficiency enter the map exactly as in the DC path
        faraday_grid = march_final_um * np.asarray(eta, float)
    th_grid = np.maximum(faraday_grid - L0, 0.0)

    # ---- metrics ------------------------------------------------------------
    # coupon means use trapezoidal (endpoint-half) weights: the face grid is
    # endpoint-inclusive and the edge columns carry ~40 % more current than the
    # interior, so a plain arithmetic mean over-weights the boundary nodes by
    # ~2 % [C: plain mean / integrated = 1.024, trapezoid / integrated = 1.005].
    w_x = _trap_weights(len(x_cm))
    w_z = _trap_weights(len(z_cm))
    W = np.outer(w_z, w_x)
    W_sum = float(W.sum())

    def _wmean(a) -> float:
        return float((np.asarray(a, float) * W).sum() / W_sum)

    mean_j = _wmean(j_grid)
    j_weighted = float((j_grid * W).sum())
    eff_charge = float((j_grid * eta * W).sum() / j_weighted) if j_weighted > 0 else 0.0
    faraday_ideal = float(thickness_um(j_set, t_s, 1.0))     # 100 % at SET j
    # charge that reached the front face, at 100 % efficiency (mean_j = I_front/A_front)
    charge_front_um = float(thickness_um(mean_j, t_s, 1.0))
    gross_mean = _wmean(faraday_grid)
    mean_th = _wmean(th_grid)
    var_j = _wmean((j_grid - mean_j) ** 2)

    loss_current_um = faraday_ideal - charge_front_um
    loss_o2_um = charge_front_um - gross_mean
    loss_l0_um = gross_mean - mean_th

    half_w, half_h = COUPON_W_CM / 2, COUPON_H_CM / 2
    su = np.abs(x_cm) / half_w
    sv = np.abs(z_cm - geom.coupon_center_z) / half_h
    SU, SV = np.meshgrid(su, sv)
    edge_mask = (SU >= 0.85) | (SV >= 0.85)
    centre_mask = (SU <= 0.40) & (SV <= 0.40)
    j_edge = float(j_grid[edge_mask].mean()) if edge_mask.any() else float("nan")
    j_centre = float(j_grid[centre_mask].mean()) if centre_mask.any() else float("nan")

    # current split: front vs back+edges vs stray (bookkeeping attribution [J])
    loss = solved["loss_fraction"]
    if setup == "base":
        frac_back_edge, frac_stray = loss, 0.0
    else:
        frac_back_edge, frac_stray = 0.0, loss

    # ---- predicted SEM points for this run ----------------------------------
    sem_rows = []
    if run.get("id") is not None:
        f_th = _sampler(x_cm, z_cm - geom.coupon_center_z, th_grid)
        for pt in load_sem_points():
            if pt["run_id"] != run["id"]:
                continue
            pred = float(f_th(pt["x_cm"], pt["y_cm"]))
            meas = pt["measured_um"]
            sem_rows.append({
                "label": pt["label"], "file": pt["file"], "run_id": pt["run_id"],
                "x_cm": pt["x_cm"], "y_cm": pt["y_cm"],
                "measured_um": meas, "predicted_um": pred,
                "residual_um": (meas - pred) if meas is not None else None,
                "confidence": pt.get("confidence"),
            })

    warnings = list(solved["warnings"])
    if setup == "base" and book["stray_area_cm2"] > 0:
        warnings.append(
            f"base setup: {100 * th_q['f_back']:.0f}% of the back+edge area "
            f"({book['stray_area_cm2']:.2f} cm2) treated as a stray conductor "
            f"-> {100 * loss:.1f}% of I_total misses the front face")
    if progress:
        progress(1.0, "done")

    out = {
        "run": {"j_mA_cm2": j_set, "t_min": t_s / 60.0, "stir_pct": stir,
                "setup": setup, "T_C": T_C, "id": run.get("id")},
        "rpm": round(rpm, 2),
        "theta": {k: (v if isinstance(v, str) else float(v)) for k, v in th.items()},
        "geometry": book,
        "delta_mean_um": float(dmap["mean_um"]),
        "V_cell_V": solved["V_cell_V"],
        "front_fraction": solved["front_fraction"],
        "loss_fraction": loss,
        "back_edge_fraction": frac_back_edge,
        "stray_fraction": frac_stray,
        "j_mean_mA_cm2": mean_j,
        "j_max_mA_cm2": float(j_grid.max()),
        "j_min_mA_cm2": float(j_grid.min()),
        "std_over_mean_pct": round(100.0 * math.sqrt(var_j) / mean_j, 3)
                             if mean_j else None,
        "edge_center_ratio": (j_edge / j_centre) if j_centre else None,
        "faraday_um": faraday_ideal,
        "gross_mean_um": gross_mean,
        "mean_thickness_um": mean_th,
        "min_thickness_um": float(th_grid.min()),
        "max_thickness_um": float(th_grid.max()),
        "apparent_eff": mean_th / faraday_ideal if faraday_ideal > 0 else None,
        "eff_charge": eff_charge,
        "loss_current_um": loss_current_um,
        "loss_o2_um": loss_o2_um,
        "loss_l0_um": loss_l0_um,
        "sem_points": sem_rows,
        "maps": {
            "x_cm": [round(float(v), 6) for v in x_cm],
            "z_cm": [round(float(v), 6) for v in z_cm],
            "j_mA_cm2": [[round(float(v), 5) for v in row] for row in j_grid],
            "thickness_um": [[round(float(v), 5) for v in row] for row in th_grid],
            "delta_um": [[round(float(v), 4) for v in row] for row in delta_grid],
            "eta_true": [[round(float(v), 5) for v in row] for row in eta],
        },
        "solve": {"resolution": resolution,
                  "solve_time_s": solved["solve_time_s"],
                  "n_newton": solved["n_newton"],
                  "cache": solve_cache_stats(),
                  "total_s": round(time.perf_counter() - t_start, 3)},
        "warnings": warnings,
        "assumptions": list(ASSUMPTIONS),
    }
    if pr is not None:
        out["pr"] = dict(pr)
    if march_info is not None:
        out["march"] = march_info
        out["run"]["march"] = {"n_stages": march_info["n_stages"],
                               "seed_R_sq0": march_info["seed_R_sq0"]}
    return out


__all__ = [
    "ASSUMPTIONS", "DEFAULT_THETA", "QUANT", "SEM_EXCLUDED", "SEM_POSITIONS",
    "SOLVE_CACHE", "SOLVE_STATS", "build_delta_map", "build_geometry",
    "clear_solve_cache", "flow_dir_deg", "load_sem_points", "local_efficiency",
    "quantised_theta", "rotate_flow", "simulate_run", "solve_cache_stats",
]
