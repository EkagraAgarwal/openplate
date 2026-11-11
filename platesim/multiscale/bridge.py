"""Coupon-to-trench bridge: local coupon conditions -> feature / trenchflow prediction.

Two scale-bridging steps, both built ONLY on public functions of the existing
modules (no physics is duplicated and nothing is re-fitted here):

1. :func:`site_conditions` samples the coupled coupon pipeline
   (``coupon.pipeline.simulate_run``: flow -> cell -> efficiency) at a position on
   the 2x2 cm coupon face and returns the local current density, Nernst-layer
   thickness, true efficiency and coupon Faraday thickness at that position.
   The pipeline solve is run ONCE per ``(run, theta, resolution)`` and memoised, so
   a 25-position grid costs one coupon solve, not 25.

2. :func:`predict_trench` feeds those local conditions into one of the two feature
   scale models:
   ``levelset`` -> ``feature.calibrate.predict`` (level-set fill, DC or the site's
   pulse-reverse block), ``flow`` -> ``trenchflow.model.predict_site`` (quasi-steady
   flow + Cu2+ transport, marched galvanostatically).

Conventions and caveats
-----------------------
* Map frame. ``simulate_run`` returns ``maps`` on the cell-solve grid: ``x_cm``
  across the coupon width, ``z_cm`` the ABSOLUTE beaker height
  (``coupon_center_z +- COUPON_H_CM/2``, ``cell.solver`` x_grid/z_grid). The SEM
  positions are given in the CENTRED coupon frame ``(x_cm, y_cm)`` with ``y_cm``
  up, so the sampler maps ``y_cm -> z_cm - coupon_center_z``. The centre is taken
  from the returned map endpoints (``0.5*(z_cm[0] + z_cm[-1])``), which equals
  ``core.cell.CellGeometry().coupon_center_z`` = 4.0 cm for the coupon pipeline.
  Samples are bilinearly interpolated and CLIPPED to the face: no extrapolation.
* Waveform. The coupon pipeline is DC (one steady state, see
  ``coupon/validate.py``); ``feature.calibrate.predict`` applies each SEM site's
  own pulse-reverse block (9/0.5/0.5 ms, rr = 2), which the flow model cannot do.
* The level-set prediction is cached inside ``feature.calibrate`` by
  ``(site id, theta, dx_um, use_pr)``. Every coupon position therefore gets a
  UNIQUE site id: without it all positions would silently return the first
  position's result.
* No parameter is fitted to the trench data anywhere in this module.

Units are in the key names (``_cm``, ``_um``, ``_mA_cm2``).
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..coupon.calibrate import theta_from_calibration
from ..coupon.pipeline import simulate_run
from ..feature.calibrate import predict as _feature_predict
from ..trenchflow.model import model_constants, predict_site

DEFAULT_RUN = {"setup": "imp1", "j_mA_cm2": 15.0, "t_min": 270.0,
               "stir_pct": 30.0, "T_C": 22.0}

# (run, theta, resolution) -> simulate_run() result. One coupon solve per key.
SIM_CACHE: dict[str, dict] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def clear_cache() -> None:
    """Drop the memoised coupon solves (mainly for tests)."""
    SIM_CACHE.clear()
    _CACHE_STATS["hits"] = 0
    _CACHE_STATS["misses"] = 0


def cache_stats() -> dict:
    return {"entries": len(SIM_CACHE), **_CACHE_STATS}


def _sim_key(run: dict, theta: dict, resolution: str) -> str:
    payload = {
        "run": {k: (v if isinstance(v, str) or v is None else float(v))
                for k, v in dict(run).items()},
        "theta": {k: (v if isinstance(v, str) else float(v)) for k, v in theta.items()},
        "resolution": str(resolution),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _simulate_cached(run: dict, theta: dict, resolution: str) -> dict:
    """simulate_run memoised on (run, theta, resolution)."""
    key = _sim_key(run, theta, resolution)
    hit = SIM_CACHE.get(key)
    if hit is not None:
        _CACHE_STATS["hits"] += 1
        return hit
    _CACHE_STATS["misses"] += 1
    sim = simulate_run(dict(run), theta, resolution=resolution)
    SIM_CACHE[key] = sim
    return sim


def _bilinear(x, y, F, xq: float, yq: float) -> float:
    """Bilinear sample of F(x, y) (rows = y, ascending) at one point.

    The query is clipped to the grid extent: outside the face the edge value is
    returned (no extrapolation).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    F = np.asarray(F, float)
    nx, ny = int(x.size), int(y.size)
    if nx < 2 or ny < 2:
        raise ValueError("grid must have at least 2 points per axis")
    xq = float(np.clip(float(xq), x[0], x[-1]))
    yq = float(np.clip(float(yq), y[0], y[-1]))
    fx = (xq - x[0]) / (x[-1] - x[0]) * (nx - 1)
    fy = (yq - y[0]) / (y[-1] - y[0]) * (ny - 1)
    ix = min(max(int(math.floor(fx)), 0), nx - 2)
    iy = min(max(int(math.floor(fy)), 0), ny - 2)
    tx, ty = fx - ix, fy - iy
    return float(F[iy, ix] * (1 - tx) * (1 - ty) + F[iy, ix + 1] * tx * (1 - ty)
                 + F[iy + 1, ix] * (1 - tx) * ty + F[iy + 1, ix + 1] * tx * ty)


def coupon_centre_z_cm(sim: dict) -> float:
    """Height of the coupon centre in the map frame (= mean of the z endpoints)."""
    z = np.asarray(sim["maps"]["z_cm"], float)
    return 0.5 * float(z[0] + z[-1])


def face_extent_cm(sim: dict) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of the coupon face in the centred frame."""
    x = np.asarray(sim["maps"]["x_cm"], float)
    z = np.asarray(sim["maps"]["z_cm"], float)
    zc = 0.5 * float(z[0] + z[-1])
    return (float(x[0]), float(x[-1]), float(z[0] - zc), float(z[-1] - zc))


# ---------------------------------------------------------------------------
# 1. local coupon conditions
# ---------------------------------------------------------------------------
def site_conditions(run: dict | None = None, x_cm: float = 0.0, y_cm: float = 0.0,
                    theta: dict | None = None, resolution: str = "coarse") -> dict:
    """Local conditions on the coupon face at ``(x_cm, y_cm)`` [centred frame, y up].

    ``run``: ``{setup, j_mA_cm2, t_min, stir_pct, T_C}``; defaults to the Trench 1
    coupon run.  ``theta``: coupon calibration; ``None`` -> the SAVED calibration
    (``coupon.calibrate.theta_from_calibration``), falling back to the pipeline
    defaults when calibration.json is absent.

    One ``coupon.pipeline.simulate_run`` per ``(run, theta, resolution)`` is shared
    by every position (memoised in :data:`SIM_CACHE`).

    Returns
    -------
    dict with keys ``x_cm``, ``y_cm``, ``j_mA_cm2``, ``delta_um``, ``efficiency``,
    ``coupon_thickness_um``, ``front_fraction``, ``delta_mean_um``, ``rpm``,
    ``j_mean_mA_cm2``.  ``coupon_thickness_um`` is the planar coupon Faraday
    thickness at that position (the "flat site" model); the trench models start
    from ``j_mA_cm2``/``delta_um``/``efficiency`` instead.
    """
    run = dict(DEFAULT_RUN if run is None else run)
    th = theta_from_calibration()[0] if theta is None else {**theta}
    sim = _simulate_cached(run, th, resolution)

    maps = sim["maps"]
    x = np.asarray(maps["x_cm"], float)
    z = np.asarray(maps["z_cm"], float)
    zc = 0.5 * float(z[0] + z[-1])
    y = z - zc                       # centred frame: z_cm -> y_cm

    j = _bilinear(x, y, maps["j_mA_cm2"], x_cm, y_cm)
    delta = _bilinear(x, y, maps["delta_um"], x_cm, y_cm)
    eta = _bilinear(x, y, maps["eta_true"], x_cm, y_cm)
    thick = _bilinear(x, y, maps["thickness_um"], x_cm, y_cm)

    x_lo, x_hi, y_lo, y_hi = face_extent_cm(sim)
    return {
        "x_cm": float(np.clip(float(x_cm), x_lo, x_hi)),
        "y_cm": float(np.clip(float(y_cm), y_lo, y_hi)),
        "j_mA_cm2": j,
        "delta_um": delta,
        "efficiency": eta,
        "coupon_thickness_um": thick,
        "front_fraction": float(sim["front_fraction"]),
        "delta_mean_um": float(sim["delta_mean_um"]),
        "rpm": float(sim["rpm"]),
        "j_mean_mA_cm2": float(sim["j_mean_mA_cm2"]),
    }


# ---------------------------------------------------------------------------
# 2. feature-scale prediction at one coupon position
# ---------------------------------------------------------------------------
def _unique_site(site: dict, site_id: str, j_mA_cm2: float) -> dict:
    """Copy of the SEM site with the local current and a position-unique id.

    The unique id is essential: ``feature.calibrate.predict`` caches by
    ``(site id, theta, dx_um, use_pr)``, so a shared id would return the first
    position's result for every other position.
    """
    out = dict(site)
    out["j_mA_cm2"] = float(j_mA_cm2)
    out["id"] = site_id
    return out


def predict_trench(site: dict, conditions: dict, model: str = "levelset",
                   quick: bool = False, k_mix: float = 1.0) -> dict:
    """Predict field/floor/wall thickness of one trench ``site`` at local conditions.

    ``conditions`` is a :func:`site_conditions` dict; ``model`` is ``"levelset"``
    (``feature.calibrate.predict``, level-set fill) or ``"flow"``
    (``trenchflow.model.predict_site``, quasi-steady flow + Cu2+ transport).
    ``quick``: dx = 8 um and 2 flow stages (smoke-test settings); ``False`` uses
    the production defaults of each model.  ``k_mix`` is the level-set eddy-mixing
    surrogate (``feature.engine.params`` ``k_mix``: 1.0 = the ``feature.calibrate``
    default, 0.0 = no eddy mixing); the flow model has no such term and ignores it.

    levelset
        theta = {efficiency, delta_um, k_mix} from the coupon position; the site
        keeps its measured duration/geometry and applies its own pulse-reverse
        block.  The coupon pipeline itself is DC, so the field current entering the
        feature model is the DC local current (the pulse changes the local
        concentration at the trench mouth, not the average current).
        NOTE: ``k_mix`` is a MODEL SWITCH, not a fitted parameter - at ``k_mix > 0``
        the eddy-diffusivity surrogate mixes the trench, which suppresses the
        additive-free bottom-up (floor/field) signal.  ``k_mix = 0`` is the
        calibrate-predictor configuration with eddy mixing switched off:
        ``feature.calibrate.predict`` builds ``Params(suppressor=False, ...)``, so
        the suppressor is already off and ``ohmic`` is LEFT AT ITS DEFAULT ``True``.
        The independent reference computation (Task 4) additionally sets
        ``ohmic=False``, so the ``k_mix=0`` columns differ from that reference by
        the ohmic term only - they are not the same configuration.
    flow
        The measured field thickness is not used: it is derived from the local
        current,
        ``field_um = j_mA_cm2*1e-3 * Mfac * efficiency * hours*3600 / 1e-4`` [C],
        i.e. the absolute field thickness is linear in the coupon efficiency.
        ``predict_site`` then solves its galvanostatic rate constant ``k`` from the
        field current implied by that ``field_um`` (``j_field = field_um/(Mfac t)``),
        so the transport problem is run at the EFFICIENCY-SCALED field current
        ``j_local*efficiency``.  The efficiency factor does NOT cancel: a lower
        efficiency lowers ``j_field`` and therefore ``k``, and the flux ratios
        (``floor_over_field``, wall asymmetry) shift weakly with ``k`` through the
        concentration field.  The ratios are therefore mildly, not exactly,
        insensitive to the efficiency assumption.

    Returns ``{model, site_id, x_cm, y_cm, j_mA_cm2, efficiency, delta_um, quick,
    k_mix, field_um, floor_um, floor_over_field, wall_um}`` plus, for ``flow``,
    ``wall_up_um``/``wall_down_um``/``j_field_mA_cm2``, and ``warnings``.  A flow
    solve rejected by ``predict_site`` (``j_field`` above the mass-transfer limit)
    returns the thicknesses as ``None`` with ``error`` set and a warning, so a
    high-current sweep does not abort.
    ``wall_um`` is ``None`` for the level-set run (that model has no wall split).
    """
    model = str(model).lower()
    j_local = float(conditions["j_mA_cm2"])
    hours = float(site["hours"])
    # Position- AND condition-unique site id: feature.calibrate caches on
    # (site id, theta, dx_um, use_pr), so the id must carry the local current and
    # duration too - a different j at the same position/theta would otherwise be
    # served the cached result of the first current (theta carries k_mix).
    site_id = (f"{site.get('id', '')}_{conditions['x_cm']:.3f}_"
               f"{conditions['y_cm']:.3f}_j{j_local:.3f}_h{hours:g}")
    base = {
        "model": model,
        "site_id": site_id,
        "x_cm": float(conditions["x_cm"]),
        "y_cm": float(conditions["y_cm"]),
        "j_mA_cm2": j_local,
        "efficiency": float(conditions["efficiency"]),
        "delta_um": float(conditions["delta_um"]),
        "quick": bool(quick),
        "k_mix": (float(k_mix) if model == "levelset" else None),
    }

    if model == "levelset":
        site2 = _unique_site(site, site_id, j_local)
        theta = {"efficiency": float(conditions["efficiency"]),
                 "delta_um": float(conditions["delta_um"]),
                 "k_mix": float(k_mix)}
        out = _feature_predict(site2, theta, dx_um=(8.0 if quick else None))
        field = float(out["field_um"])
        floor = float(out["floor_um"])
        return {**base,
                "field_um": field,
                "floor_um": floor,
                "floor_over_field": (floor / field) if field > 0 else float("nan"),
                "wall_um": None,
                "outcome": out.get("outcome"),
                "error": None,
                "warnings": list(out.get("warnings") or [])}

    if model == "flow":
        T_C = float(site.get("T_C", 22.0))
        mfac = float(model_constants(T_C)[2])
        field = (j_local * 1e-3 * mfac * float(conditions["efficiency"])
                 * hours * 3600.0 / 1e-4)
        try:
            out = predict_site(site, delta_um=float(conditions["delta_um"]),
                               n_stages=(2 if quick else 5),
                               dx_um=(8.0 if quick else None),
                               field_um=field, use_flow=True)
        except ValueError as exc:            # above the mass-transfer limit
            return {**base, "field_um": field, "floor_um": None,
                    "floor_over_field": None, "wall_um": None,
                    "wall_up_um": None, "wall_down_um": None,
                    "j_field_mA_cm2": j_local * float(conditions["efficiency"]),
                    "outcome": None, "error": f"ValueError: {exc}",
                    "warnings": [f"flow solve rejected: {exc}"]}
        floor = float(out["floor_um"])
        up = float(out["wall_up_um"])
        down = float(out["wall_down_um"])
        return {**base,
                "field_um": field,
                "floor_um": floor,
                "floor_over_field": (float(out["floor_over_field"])
                                     if field > 0 else float("nan")),
                "wall_um": 0.5 * (up + down),
                "wall_up_um": up,
                "wall_down_um": down,
                "j_field_mA_cm2": float(out["j_field_mA_cm2"]),
                "outcome": None,
                "error": None,
                "warnings": []}

    raise ValueError(f"model must be 'levelset' or 'flow', got {model!r}")


__all__ = ["DEFAULT_RUN", "SIM_CACHE", "cache_stats", "clear_cache",
           "coupon_centre_z_cm", "face_extent_cm", "predict_trench",
           "site_conditions"]
