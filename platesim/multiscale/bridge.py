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
* Two-way coupling (Plan F1, ``two_way=True``, default OFF -> the one-way path
  above and its numbers are unchanged). A trench array has more true surface area
  per projected area than a flat plane, so it draws more current locally.  One
  quick ``stokes`` feature stage gives the region flux ratios, from which
  :func:`effective_area_factor` builds ``A_eff = I_pitch/(j_flat*pitch)``;
  :func:`coupon_area_map` iterates ``j_field = j_coupon*A_eff/<A_eff>`` over the
  coupon to a fixed point, and the feature model is then run at ``j_field``.
  ``<A_eff>`` is current-weighted, which conserves the coupon total current.  No
  parameter is fitted in the two-way path either.
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
from ..trenchflow.model import model_constants, predict_site, shear_from_delta, stage

DEFAULT_RUN = {"setup": "imp1", "j_mA_cm2": 15.0, "t_min": 270.0,
               "stir_pct": 30.0, "T_C": 22.0}

# (run, theta, resolution) -> simulate_run() result. One coupon solve per key.
SIM_CACHE: dict[str, dict] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}

# Two-way coupling caches (only populated when two_way=True).
_A_EFF_CACHE: dict[str, dict] = {}
_AREA_MAP_CACHE: dict[str, dict] = {}


def clear_cache() -> None:
    """Drop the memoised coupon solves and two-way factors (mainly for tests)."""
    SIM_CACHE.clear()
    _A_EFF_CACHE.clear()
    _AREA_MAP_CACHE.clear()
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
# 1b. two-way coupling: the trench array feeds its own area back to the coupon
# ---------------------------------------------------------------------------
TWO_WAY_TOL = 1e-3          # fixed-point stop: max |change| of A_eff between sweeps
TWO_WAY_MAX_ITER = 10       # fixed-point iteration cap
DEFAULT_F_OPEN = 0.2        # open-area fraction of the trench array


def effective_area_factor(site: dict, f_open: float, conditions: dict,
                          dx_um: float | None = None, deposit_um: float = 0.0,
                          flow_solver: str = "stokes", p=None, ph=None) -> dict:
    """Effective area (true area x local efficiency) factor ``A_eff`` of a trench array.

    One pitch of the array (``pitch = width_um / f_open``) has the projected field
    surface ``pitch - width``, the floor ``width`` and two walls of height ``depth``
    (per unit length into the page; the model geometry is rectangular).  A quick
    feature solve (:func:`platesim.trenchflow.model.stage`, ``flow_solver='stokes'``,
    one direct sparse flow solve, deposit 0 = the as-etched trench) gives the mean
    flux density of each region RELATIVE to the field surface, ``r_floor``,
    ``r_up``, ``r_down``.  The current drawn by one pitch at a field current density
    ``j_flat`` is therefore

        I_pitch = (pitch - width)*j_flat + width*r_floor*j_flat
                  + depth*(r_up + r_down)*j_flat                            [C]

    and

        A_eff = I_pitch / (j_flat * pitch)
              = 1 + [width*(r_floor - 1) + depth*(r_up + r_down)] / pitch    [C]

    i.e. the true-surface-area gain of the walls and floor times their local
    transport efficiency (all surfaces share one k: Wagner number >> 1).  ``A_eff``
    is 1 for no trenches (``depth == 0`` or ``f_open -> 0``: the trench area is a
    vanishing fraction of the pitch) and grows with the wall area per pitch.

    Conventions and caveats
    -----------------------
    * ``conditions`` supplies ``j_mA_cm2`` and ``delta_um``; ``T_C`` comes from
      ``site``.  The shear rate is ``shear_from_delta`` (the same closure as
      ``trenchflow.model.predict_site``).
    * The region ratios are local concentration ratios, so they are only weakly
      dependent on the periodic-cell pitch.  They are taken from the model's own
      cell (default pitch ``width + max(4*depth, width)``) while the AREA weighting
      uses the array pitch ``width/f_open``: adding far-field area contributes
      exactly 1:1 to both ``I_pitch`` and ``j_flat*pitch``, which is what the
      ``1 + (...) / pitch`` form expresses.  A different pitch is not passed to
      ``stage`` because the model has no such argument.
    * ``deposit_um`` defaults to 0 (the as-etched area is the largest one).
    * ``j`` at or above the mass-transfer limit of the field surface makes
      ``stage`` raise; the factor then falls back to 1.0 (no coupling at that
      position) and the reason is reported in ``warnings``.

    Returns ``{a_eff, pitch_um, j_mA_cm2, delta_um, shear_rate_s,
    floor_over_field, wall_up_over_field, wall_down_over_field, k_cm_s,
    flow_solver, grid_shape, warnings}``.  No parameter is fitted here.
    """
    width = float(site["width_um"])
    depth = float(site["depth_um"])
    f = float(f_open)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"f_open must be in (0, 1], got {f_open!r}")
    T_C = float(site.get("T_C", 22.0))
    j = float(conditions["j_mA_cm2"])
    delta = float(conditions["delta_um"])
    dx = (4.0 if width < 400.0 else 6.0) if dx_um is None else float(dx_um)
    pitch = width / f

    base = {"a_eff": 1.0, "pitch_um": pitch, "j_mA_cm2": j, "delta_um": delta,
            "shear_rate_s": None, "floor_over_field": None,
            "wall_up_over_field": None, "wall_down_over_field": None,
            "k_cm_s": None, "flow_solver": str(flow_solver), "grid_shape": None,
            "warnings": []}
    if depth <= 0.0:
        base["warnings"].append("no trench (depth_um = 0): effective area factor = 1.0")
        return base
    if delta <= 0.0 or j <= 0.0:
        base["warnings"].append("non-positive j/delta: effective area factor = 1.0")
        return base

    key = (f"{width!r}|{depth!r}|{T_C!r}|{f!r}|{j:.12g}|{delta:.12g}|{dx!r}|"
           f"{float(deposit_um)!r}|{str(flow_solver)}")
    hit = _A_EFF_CACHE.get(key)
    if hit is not None:
        return hit

    D = float(model_constants(T_C)[0])
    shear = float(shear_from_delta(delta, D))
    try:
        st = stage(width, depth, delta, shear, j, float(deposit_um), T_C=T_C,
                   dx_um=dx, use_flow=True, p=p, ph=ph,
                   flow_solver=str(flow_solver))
    except ValueError as exc:          # above the mass-transfer limit of the field
        base["shear_rate_s"] = shear
        base["warnings"].append(f"stage solve rejected: {exc}; effective area factor = 1.0")
        _A_EFF_CACHE[key] = base
        return base

    rf = float(st["floor_over_field"])
    ru = float(st["wall_up_over_field"])
    rd = float(st["wall_down_over_field"])
    out = {**base, "shear_rate_s": shear, "floor_over_field": rf,
           "wall_up_over_field": ru, "wall_down_over_field": rd,
           "k_cm_s": float(st["k_cm_s"]), "flow_solver": str(st["flow_solver"]),
           "grid_shape": tuple(st["grid_shape"])}
    if not all(np.isfinite([rf, ru, rd])):
        out["warnings"].append("empty trench region in the stage solve; factor = 1.0")
        _A_EFF_CACHE[key] = out
        return out

    i_pitch = (pitch - width) + width * rf + depth * (ru + rd)
    out["a_eff"] = float(i_pitch / pitch)
    _A_EFF_CACHE[key] = out
    return out


def rescale_coupon_currents(j_coupon_mA_cm2, a_eff,
                            weights=None) -> tuple[np.ndarray, float]:
    """Galvanostatic rescaling of a coupon current map by the array area factor.

    The array region draws ``A_eff`` times more current per projected area than a
    flat surface, so the local field current density that the feature model must
    see is ``j_coupon * A_eff / <A_eff>``.  ``<A_eff>`` is the mean over the coupon
    weighted by the coupon current itself,

        <A_eff> = sum(w*j_coupon*A_eff) / sum(w*j_coupon),

    which equals the area-weighted mean when the coupon current is uniform and is
    the normalisation that conserves the coupon total current exactly:
    ``sum(w*j_field) == sum(w*j_coupon)`` to round-off.  ``weights`` are the cell
    areas (``None`` = uniform).

    Returns ``(j_field_mA_cm2, a_eff_mean)``.
    """
    j = np.asarray(j_coupon_mA_cm2, float)
    a = np.asarray(a_eff, float)
    if j.shape != a.shape:
        raise ValueError(f"j and a_eff must have the same shape, got {j.shape} and {a.shape}")
    w = np.ones_like(j) if weights is None else np.asarray(weights, float)
    if w.shape != j.shape:
        raise ValueError(f"weights must have the shape of j, got {w.shape} and {j.shape}")
    total = float((w * j).sum())
    if not total > 0.0:
        raise ValueError("the coupon total current must be positive")
    mean = float((w * j * a).sum()) / total
    if not mean > 0.0:
        raise ValueError("the current-weighted mean A_eff must be positive")
    return j * a / mean, mean


def coupon_area_map(site: dict, f_open: float = DEFAULT_F_OPEN, run: dict | None = None,
                    theta: dict | None = None, resolution: str = "coarse", n: int = 3,
                    tol: float = TWO_WAY_TOL, max_iter: int = TWO_WAY_MAX_ITER,
                    dx_um: float | None = None, p=None, ph=None) -> dict:
    """Coupon-scale two-way map: ``A_eff`` and the rescaled local current density.

    The coupon face is sampled on an ``n x n`` grid of cell CENTRES (equal cell
    areas, so a plain mean is the area-weighted mean), the coupon conditions at each
    node come from :func:`site_conditions` (one memoised coupon solve for the whole
    map), and the coupon <-> feature exchange is iterated to a fixed point:

    1. ``A_eff`` at the coupon current (one quick ``stokes`` stage per node);
    2. ``j_field = j_coupon * A_eff / <A_eff>`` (:func:`rescale_coupon_currents`);
    3. ``A_eff`` again at the rescaled field current (the flux ratios change weakly
       with the galvanostatic ``k``);
    4. repeat until ``max|A_eff_new - A_eff| < tol`` or ``max_iter`` sweeps.

    The feedback is negative (a larger ``A_eff`` raises ``j_field``, a larger current
    depletes the trench and lowers ``A_eff``), so the sweep contracts.  The coupon
    efficiency and Nernst thickness are NOT updated: the coupon solve is
    galvanostatic and the same total current is only redistributed inside the face.

    Returns ``{x_cm, y_cm, weights, j_coupon_mA_cm2, j_field_mA_cm2, a_eff,
    delta_um, efficiency, a_eff_mean, a_eff_mean_area, iterations, history,
    converged, f_open, tol, max_iter, width_um, depth_um, run, resolution, n}``.
    ``a_eff_mean`` is the current-weighted mean used for the normalisation,
    ``a_eff_mean_area`` its unweighted (area) counterpart, and ``history`` is one
    ``{iteration, max_change, a_eff_mean}`` dict per sweep.  The map is memoised on
    ``(site geometry, f_open, run, theta, resolution, n, tol, max_iter, dx)``.
    """
    run = dict(DEFAULT_RUN if run is None else run)
    th = theta_from_calibration()[0] if theta is None else {**theta}
    n = int(n)
    if n < 2:
        raise ValueError("n must be >= 2 (the sampler is bilinear)")
    f = float(f_open)
    key = "|".join((_sim_key(run, th, resolution), json.dumps(
        {"n": n, "width": float(site["width_um"]), "depth": float(site["depth_um"]),
         "T_C": float(site.get("T_C", 22.0)), "f_open": f, "tol": float(tol),
         "max_iter": int(max_iter),
         "dx_um": None if dx_um is None else float(dx_um)},
        sort_keys=True, default=str)))
    hit = _AREA_MAP_CACHE.get(key)
    if hit is not None:
        return hit

    sim = _simulate_cached(run, th, resolution)
    x_lo, x_hi, y_lo, y_hi = face_extent_cm(sim)
    xs = x_lo + (np.arange(n) + 0.5) * (x_hi - x_lo) / n
    ys = y_lo + (np.arange(n) + 0.5) * (y_hi - y_lo) / n

    j_c = np.empty((n, n))
    delta = np.empty((n, n))
    eff = np.empty((n, n))
    a_eff = np.empty((n, n))
    conds = [[{} for _ in range(n)] for _ in range(n)]
    for iy in range(n):
        for ix in range(n):
            c = site_conditions(run, float(xs[ix]), float(ys[iy]), theta=th,
                                resolution=resolution)
            conds[iy][ix] = c
            j_c[iy, ix] = float(c["j_mA_cm2"])
            delta[iy, ix] = float(c["delta_um"])
            eff[iy, ix] = float(c["efficiency"])
            a_eff[iy, ix] = effective_area_factor(
                site, f, c, dx_um=dx_um, p=p, ph=ph)["a_eff"]

    weights = np.ones_like(j_c)
    j_field, mean = rescale_coupon_currents(j_c, a_eff, weights)
    history: list[dict] = []
    converged = False
    for it in range(int(max_iter)):
        a_new = np.empty_like(a_eff)
        for iy in range(n):
            for ix in range(n):
                c2 = {**conds[iy][ix], "j_mA_cm2": float(j_field[iy, ix])}
                a_new[iy, ix] = effective_area_factor(
                    site, f, c2, dx_um=dx_um, p=p, ph=ph)["a_eff"]
        change = float(np.max(np.abs(a_new - a_eff)))
        a_eff = a_new
        j_field, mean = rescale_coupon_currents(j_c, a_eff, weights)
        history.append({"iteration": it + 1, "max_change": change,
                        "a_eff_mean": float(mean)})
        if change < float(tol):
            converged = True
            break

    out = {"x_cm": xs, "y_cm": ys, "weights": weights,
           "j_coupon_mA_cm2": j_c, "j_field_mA_cm2": j_field, "a_eff": a_eff,
           "delta_um": delta, "efficiency": eff,
           "a_eff_mean": float(mean), "a_eff_mean_area": float(a_eff.mean()),
           "iterations": len(history), "history": history,
           "converged": bool(converged), "f_open": f, "tol": float(tol),
           "max_iter": int(max_iter), "width_um": float(site["width_um"]),
           "depth_um": float(site["depth_um"]), "run": run,
           "resolution": str(resolution), "n": n}
    _AREA_MAP_CACHE[key] = out
    return out


def _check_area_map(amap: dict, site: dict, f_open: float) -> None:
    """Reject a two-way map built for another trench geometry / open fraction."""
    for key, want in (("width_um", float(site["width_um"])),
                      ("depth_um", float(site["depth_um"]))):
        got = float(amap[key])
        if abs(got - want) > 1e-9 * max(1.0, abs(want)):
            raise ValueError(f"two_way_map {key}={got} does not match the site ({want})")
    got_f = float(amap["f_open"])
    if abs(got_f - float(f_open)) > 1e-12:
        raise ValueError(f"two_way_map f_open={got_f} does not match f_open={f_open}")


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
                   quick: bool = False, k_mix: float = 1.0, two_way: bool = False,
                   f_open: float = DEFAULT_F_OPEN, two_way_map: dict | None = None,
                   two_way_n: int = 3, two_way_tol: float = TWO_WAY_TOL,
                   two_way_max_iter: int = TWO_WAY_MAX_ITER) -> dict:
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
    two_way
        ``two_way=False`` (default) is the one-way path above and is bit-identical
        to it: ``conditions`` is used unchanged and :func:`effective_area_factor` is
        never called.  ``two_way=True`` adds the array -> coupon feedback: the array
        draws ``A_eff`` times more current per projected area, so the feature model
        is run at ``j_coupon * A_eff / <A_eff>``, where ``<A_eff>`` is the
        current-weighted mean over the coupon (:func:`coupon_area_map` iterated to
        ``max|dA| < two_way_tol``, at most ``two_way_max_iter`` sweeps).  The
        normalisation conserves the coupon total current exactly.  ``two_way_map``
        lets a caller reuse a map built once for this site (``f_open`` and the
        geometry are validated); otherwise a map is built from :data:`DEFAULT_RUN`
        with an ``two_way_n x two_way_n`` coupon grid.

    Returns ``{model, site_id, x_cm, y_cm, j_mA_cm2, efficiency, delta_um, quick,
    k_mix, field_um, floor_um, floor_over_field, wall_um}`` plus, for ``flow``,
    ``wall_up_um``/``wall_down_um``/``j_field_mA_cm2``, and ``warnings``.  A flow
    solve rejected by ``predict_site`` (``j_field`` above the mass-transfer limit)
    returns the thicknesses as ``None`` with ``error`` set and a warning, so a
    high-current sweep does not abort.
    ``wall_um`` is ``None`` for the level-set run (that model has no wall split).
    The two-way keys are always present: ``two_way``, ``a_eff`` (local factor or
    ``None``), ``a_eff_mean``, ``j_coupon_mA_cm2``, ``two_way_iterations``,
    ``two_way_history`` (per-sweep ``{iteration, max_change, a_eff_mean}``) and
    ``two_way_converged``.
    """
    model = str(model).lower()
    j_coupon = float(conditions["j_mA_cm2"])
    hours = float(site["hours"])
    # One-way by default: `conditions` is used verbatim and no feature solve is
    # started for the area factor, so the default path is bit-identical.
    two_way_info = {"two_way": False, "a_eff": None, "a_eff_mean": None,
                    "j_coupon_mA_cm2": j_coupon, "two_way_iterations": 0,
                    "two_way_history": [], "two_way_converged": None}
    j_local = j_coupon
    if two_way:
        amap = two_way_map if two_way_map is not None else coupon_area_map(
            site, f_open=f_open, n=two_way_n, tol=two_way_tol,
            max_iter=two_way_max_iter)
        _check_area_map(amap, site, f_open)
        a_here = _bilinear(amap["x_cm"], amap["y_cm"], amap["a_eff"],
                           conditions["x_cm"], conditions["y_cm"])
        # j_coupon * A_eff / <A_eff>: the caller's own local current is rescaled by
        # the local factor and the coupon normalisation; the map only supplies the
        # (smooth, weakly varying) factor and its current-weighted mean.
        j_local = j_coupon * a_here / float(amap["a_eff_mean"])
        two_way_info = {"two_way": True, "a_eff": float(a_here),
                        "a_eff_mean": float(amap["a_eff_mean"]),
                        "j_coupon_mA_cm2": j_coupon,
                        "two_way_iterations": int(amap["iterations"]),
                        "two_way_history": list(amap["history"]),
                        "two_way_converged": bool(amap["converged"])}
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
        **two_way_info,
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


__all__ = ["DEFAULT_F_OPEN", "DEFAULT_RUN", "SIM_CACHE", "TWO_WAY_MAX_ITER",
           "TWO_WAY_TOL", "cache_stats", "clear_cache", "coupon_area_map",
           "coupon_centre_z_cm", "effective_area_factor", "face_extent_cm",
           "predict_trench", "rescale_coupon_currents", "site_conditions"]
