"""Global (Sobol / variance-based) sensitivity of the trench floor/field ratio (improvement plan F6).

Question: which uncertain input dominates ``stage(...)['floor_over_field']`` of the quick trench model
(:mod:`platesim.trenchflow.model`), i.e. where should the next measurement go?  The answer is a
*global* one: :func:`sobol_indices` decomposes the output variance into first-order contributions
``S1`` (variance explained by one input alone) and total-order contributions ``ST`` (that input plus
every interaction it takes part in), with bootstrap confidence intervals.  ``ST - S1`` is the share of
the input that only acts through interactions.

Inputs (all uniform unless marked, all judged ranges ``[J]``, not fitted):

    name              range          scale        note
    delta_um          32 - 128 um    log-uniform  Nernst layer [J: coupon calibration 78 +/- 22 um]
    shear_factor      0.01 - 3       log-uniform  multiplies ``shear_from_delta(delta, D)`` [J]
    j_field_mA_cm2    10 - 20        linear       field current density [J]
    efficiency        0.8 - 1.0      linear       current efficiency, ``Params(efficiency=..)`` [J]
    T_C               20 - 26 C      linear       bath temperature [J]
    deposit_um        0 - 40 um      linear       stage midpoint deposit (conformal growth) [J]

``shear_factor`` turns the delta-consistent Leveque shear into "no cross-flow" (0.01, essentially
stagnant) up to strong cross-flow (3).  ``shear_factor = 0`` (no flow at all, ``use_flow=False``) is a
separate reference: :func:`trench_ratio_sensitivity` with ``use_flow=False``.

The model is evaluated with ``flow_solver="stokes"`` (direct sparse steady solve, ~10x faster than the
D2Q9 lattice at the same grid) and ``dx_um = 8``.

A structural caveat the indices expose: ``stage`` receives ``j_field`` directly and never uses
``ph.Mfac``, the only quantity ``efficiency`` enters (``derive`` -> ``Mfac = M_CU/(2 F RHO_CU) *
efficiency``).  ``efficiency`` is therefore *exactly* inert in this stage model and its ``S1``/``ST``
are 0 by construction; it would matter only in the marched :func:`predict_site` (where ``j_field`` is
derived from the measured field thickness via ``Mfac``).

Estimators (Saltelli et al. 2010, Comput. Phys. Commun. 181, 259; Jansen 1999):

    S1_i = mean_j[ f(B)_j * ( f(A_B^i)_j - f(A)_j ) ] / V
    ST_i = mean_j[ ( f(A)_j - f(A_B^i)_j )^2 ] / (2 V)

with ``A``, ``B`` the two base matrices, ``A_B^i`` = ``A`` with column ``i`` taken from ``B`` and
``V = var([f(A), f(B)])``.  ``N = n_base * (d + 2)`` model evaluations.  The design is one scrambled
:class:`scipy.stats.qmc.Sobol` sequence in ``2 d`` dimensions split by column into ``A`` and ``B``
(SALib convention; a row-wise half/half split of a ``d``-dimensional Sobol sequence is degenerate in
its first direction).  Both estimators are evaluated on the **centred** outputs ``f - mean(f)``: they
are shift-invariant in expectation, but the ``mean * spread`` term otherwise dominates their sampling
variance when the coefficient of variation of the output is small (here ~8 %), which makes ``S1``
unresolvable at ``n_base = 64``.  Confidence intervals are percentile bootstrap CIs over the ``n_base``
rows (pure numpy; the model evaluations themselves are a Python loop / process pool).

No parameter is fitted and no other module is modified.  SALib is not used (not a dependency).
"""
from __future__ import annotations

import os

# set before numpy/scipy (and inherited by every worker) so each model evaluation stays single-threaded
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import functools
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

from platesim.feature.engine.params import Params, derive

from .geometry import trench_grid
from .model import shear_from_delta, stage

# name, low, high, log-uniform?  (the single source of truth for the input table above)
INPUTS = (
    ("delta_um", 32.0, 128.0, True),
    ("shear_factor", 0.01, 3.0, True),
    ("j_field_mA_cm2", 10.0, 20.0, False),
    ("efficiency", 0.8, 1.0, False),
    ("T_C", 20.0, 26.0, False),
    ("deposit_um", 0.0, 40.0, False),
)
INPUT_NAMES = tuple(row[0] for row in INPUTS)
BOUNDS = np.array([[row[1], row[2]] for row in INPUTS], dtype=float)
LOG_MASK = np.array([row[3] for row in INPUTS], dtype=bool)

# site id -> (width_um, depth_um); the same two sites as platesim/data/trench_sem.json (Trench 1)
SITES = {"narrow": (227.0, 88.0), "wide": (944.0, 97.0)}
DX_UM = 8.0


def unit_to_physical(u, bounds=BOUNDS, log_mask=LOG_MASK):
    """Map the unit cube ``u`` (..., d) to physical inputs; log-uniform dims interpolate in log space."""
    u = np.asarray(u, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    log_mask = np.asarray(log_mask, dtype=bool)
    lo, hi = bounds[:, 0], bounds[:, 1]
    lin = lo + u * (hi - lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        log = lo * (hi / lo) ** u
    return np.where(log_mask, log, lin)


def sobol_design(n_base, d, seed):
    """Saltelli design: ``(A, B, AB)`` with ``A``/``B`` (n_base, d) and ``AB`` (d, n_base, d).

    One scrambled Sobol sequence in ``2 * d`` dimensions is drawn and split **by column**: columns
    ``0..d-1`` give ``A``, columns ``d..2d-1`` give ``B`` (the SALib convention).  Splitting a
    ``d``-dimensional sequence by *rows* instead would be wrong: the two halves of the van der Corput
    direction of a Sobol sequence are nearly identical.  ``AB[i]`` is ``A`` with column ``i`` replaced
    by ``B`` column ``i``.  Fixed ``seed`` => reproducible design.
    """
    n_base = int(n_base)
    d = int(d)
    if n_base < 2:
        raise ValueError(f"n_base must be >= 2, got {n_base}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # scipy warns when n_base is not a power of two
        pts = qmc.Sobol(d=2 * d, scramble=True, seed=int(seed)).random(n_base)
    a, b = pts[:, :d], pts[:, d:]
    ab = np.repeat(a[None, :, :], d, axis=0)
    for i in range(d):
        ab[i, :, i] = b[:, i]
    return a, b, ab


def _estimators(y_a, y_b, y_ab):
    """S1 / ST of the Saltelli 2010 (first-order) and Jansen (total) estimators, and the variance V.

    The inputs are expected to be **centred** on the sample mean (see :func:`sobol_indices`): both
    estimators are shift-invariant in expectation, but centring removes the ``mean * spread`` term that
    otherwise dominates their sampling variance when the coefficient of variation is small.
    """
    n = y_a.size
    v = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"model output has zero or non-finite variance (V = {v!r}); no indices defined")
    s1 = (y_b * (y_ab - y_a)).mean(axis=1) / v
    st = (0.5 * (y_a - y_ab) ** 2).mean(axis=1) / v
    return s1, st, v


def _bootstrap(y_a, y_b, y_ab, n_boot, seed, conf=0.95):
    """Percentile bootstrap CIs of S1/ST, resampling the ``n`` base rows with replacement.

    Vectorised: every resample is one matmul, so the bookkeeping cost is independent of the model.
    Returns ``(s1, st)`` with each an array of shape ``(2, d)`` = [low, high] percentiles.
    """
    n = int(y_a.size)
    rng = np.random.default_rng(int(seed) + 7919)
    idx = rng.integers(0, n, size=(int(n_boot), n))                  # (B, n) resampled row ids
    counts = np.zeros(idx.shape, dtype=float)                        # (B, n) multiplicity of each row
    np.add.at(counts, (np.repeat(np.arange(idx.shape[0]), n), idx.ravel()), 1.0)
    c1 = y_b[None, :] * (y_ab - y_a[None, :])                        # (d, n) first-order contribution
    c2 = 0.5 * (y_a[None, :] - y_ab) ** 2                            # (d, n) total-order contribution
    m1 = (c1 @ counts.T) / n                                         # (d, B) resample means
    m2 = (c2 @ counts.T) / n
    s1 = counts @ y_a
    s2 = counts @ y_b
    sq = counts @ (y_a ** 2 + y_b ** 2)
    n2 = 2 * n
    vb = (sq - (s1 + s2) ** 2 / n2) / (n2 - 1)                       # (B,) variance of each resample
    with np.errstate(divide="ignore", invalid="ignore"):
        s1_b = m1 / vb[None, :]
        st_b = m2 / vb[None, :]
    pct = [50.0 * (1.0 - float(conf)), 50.0 * (1.0 + float(conf))]
    ci1 = np.nanpercentile(s1_b, pct, axis=1)
    cit = np.nanpercentile(st_b, pct, axis=1)
    return ci1, cit


_WORKER_FN = None


def _worker_init(fn):
    """Process-pool initializer: install the (picklable) model function and pin BLAS to one thread."""
    global _WORKER_FN
    _WORKER_FN = fn
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _worker_call(row):
    fn = _WORKER_FN
    if fn is None:
        raise RuntimeError("worker initializer did not install a model function")
    return float(fn(row))


def evaluate_rows(model_fn, x, workers=1):
    """Evaluate ``model_fn`` on every row of ``x`` (n, d), in-process or in a process pool."""
    n = int(x.shape[0])
    workers = int(workers or 1)
    if workers > 1 and n > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                     initargs=(model_fn,)) as ex:
                out = list(ex.map(_worker_call, x, chunksize=max(1, n // (4 * workers))))
            return np.asarray(out, dtype=float)
        except Exception as exc:                     # e.g. a closure/lambda model_fn is not picklable
            warnings.warn(f"process pool unavailable ({type(exc).__name__}: {exc}); "
                          "evaluating serially", RuntimeWarning, stacklevel=2)
    return np.asarray([model_fn(row) for row in x], dtype=float)


def sobol_indices(model_fn, bounds, n_base=64, seed=0, log_mask=None, names=None, n_boot=1000,
                  conf=0.95, workers=1):
    """First- and total-order Sobol indices of a scalar model, with bootstrap 95 % CIs.

    ``model_fn`` takes one physical input row (length ``d``) and returns a float, so any model works
    (the trench stage is a Python loop).  ``bounds`` is the ``(d, 2)`` physical input box, uniform
    unless ``log_mask[i]`` is True (then log-uniform).  Returns a dict of numpy arrays/lists:
    ``names``, ``S1``, ``S1_lo``, ``S1_hi``, ``ST``, ``ST_lo``, ``ST_hi``, ``mean_y``, ``std_y``,
    ``var_y``, ``n_base``, ``n_eval``, ``seed``, ``n_boot``, ``conf``.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")
    d = int(bounds.shape[0])
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("every bound must satisfy low <= high")
    if log_mask is None:
        log_mask = np.zeros(d, dtype=bool)
    log_mask = np.asarray(log_mask, dtype=bool)
    if log_mask.shape != (d,):
        raise ValueError(f"log_mask must have shape ({d},), got {log_mask.shape}")
    if np.any(log_mask & (bounds[:, 0] <= 0.0)):
        raise ValueError("log-uniform dimensions need strictly positive lower bounds")
    names = [f"x{i}" for i in range(d)] if names is None else list(names)
    if len(names) != d:
        raise ValueError(f"names has {len(names)} entries for d = {d}")

    a, b, ab = sobol_design(n_base, d, seed)
    unit = np.concatenate([a, b, ab.reshape(-1, d)], axis=0)         # (n_base * (d + 2), d)
    x = unit_to_physical(unit, bounds, log_mask)
    y = evaluate_rows(model_fn, x, workers=workers)
    if y.shape != (x.shape[0],):
        raise ValueError(f"model_fn must return a scalar per row, got shape {y.shape}")
    bad = ~np.isfinite(y)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"model returned {int(bad.sum())} non-finite values; first at input "
                         f"{dict(zip(names, x[i].tolist()))} -> {y[i]!r}")

    n = int(n_base)
    y_a, y_b = y[:n], y[n:2 * n]
    y_ab = y[2 * n:].reshape(d, n)
    mu = float(np.mean(np.concatenate([y_a, y_b])))
    y_a, y_b, y_ab = y_a - mu, y_b - mu, y_ab - mu        # see _estimators: variance reduction only
    s1, st, v = _estimators(y_a, y_b, y_ab)
    ci1, cit = _bootstrap(y_a, y_b, y_ab, n_boot, seed, conf=conf)
    return {
        "names": names,
        "S1": s1,
        "S1_lo": ci1[0],
        "S1_hi": ci1[1],
        "ST": st,
        "ST_lo": cit[0],
        "ST_hi": cit[1],
        "ST_minus_S1": st - s1,
        "mean_y": mu,
        "std_y": float(np.sqrt(v)),
        "var_y": float(v),
        "min_y": float(np.min(y)),
        "max_y": float(np.max(y)),
        "n_base": n,
        "n_eval": int(x.shape[0]),
        "seed": int(seed),
        "n_boot": int(n_boot),
        "conf": float(conf),
        "bounds": bounds.tolist(),
        "log_mask": [bool(v_) for v_ in log_mask],
    }


def _stage_ratio(row, spec):
    """One model evaluation: ``floor_over_field`` of a single quasi-steady stage.

    ``row`` is the physical input row ``(delta_um, shear_factor, j_field_mA_cm2, efficiency, T_C,
    deposit_um)``.  ``spec`` (a plain dict, so a ``functools.partial`` of this function pickles for the
    process pool) holds ``width_um``, ``depth_um``, ``dx_um`` and ``use_flow``.
    """
    delta, factor, j_field, efficiency, t_c, deposit = (float(v) for v in row)
    p = Params(efficiency=efficiency)
    ph = derive(p, t_c)
    shear = 0.0
    if spec["use_flow"] and factor > 0.0:
        shear = factor * shear_from_delta(delta, float(ph.D_c))
    out = stage(spec["width_um"], spec["depth_um"], delta, shear, j_field, deposit, T_C=t_c,
                dx_um=spec["dx_um"], use_flow=spec["use_flow"], p=p, ph=ph, flow_solver="stokes")
    return float(out["floor_over_field"])


def site_grid_shape(width_um, depth_um, dx_um=DX_UM, delta_um=128.0, deposit_um=40.0):
    """Grid shape at the largest requested stage; used to confirm the trench fits the dx = 8 um grid."""
    g = trench_grid(float(width_um), float(depth_um), height_um=max(delta_um, deposit_um + 2 * dx_um),
                    dx_um=float(dx_um), deposit_um=float(deposit_um))
    return tuple(int(v) for v in g.shape)


def trench_ratio_sensitivity(site="narrow", n_base=64, seed=0, workers=1, use_flow=True,
                             dx_um=DX_UM, n_boot=1000, conf=0.95):
    """Sobol S1/ST of the trench floor/field ratio for one site, over the input table at the top.

    ``site`` is ``"narrow"`` (227 x 88 um) or ``"wide"`` (944 x 97 um).  ``use_flow=False`` pins the
    shear factor at 0 and skips the flow solve: the "no cross-flow" reference run (``ST`` of the shear
    factor is then 0 by construction).  ``workers > 1`` evaluates the sample in a process pool.
    """
    key = str(site).strip().lower()
    if key not in SITES:
        raise ValueError(f"site must be one of {tuple(SITES)}, got {site!r}")
    width, depth = SITES[key]
    spec = {"width_um": float(width), "depth_um": float(depth), "dx_um": float(dx_um),
            "use_flow": bool(use_flow)}
    bounds = BOUNDS.copy()
    log_mask = LOG_MASK.copy()
    if not use_flow:
        bounds[1] = (0.0, 0.0)                       # shear_factor pinned: no flow at all
        log_mask[1] = False
    t0 = time.perf_counter()
    res = sobol_indices(functools.partial(_stage_ratio, spec=spec), bounds, n_base=n_base, seed=seed,
                        log_mask=log_mask, names=list(INPUT_NAMES), n_boot=n_boot, conf=conf,
                        workers=workers)
    wall = time.perf_counter() - t0
    res.update({
        "site": key,
        "width_um": float(width),
        "depth_um": float(depth),
        "dx_um": float(dx_um),
        "use_flow": bool(use_flow),
        "flow_solver": "stokes",
        "grid_shape": site_grid_shape(width, depth, dx_um),
        "wall_s": float(wall),
        "s_per_eval": float(wall / max(1, res["n_eval"])),
        "dominant_input": str(INPUT_NAMES[int(np.argmax(res["ST"]))]),
    })
    return res


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays into strict-JSON types (non-finite -> null)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _print_table(res):
    """S1/ST table with 95 % CIs plus the run summary."""
    print(f"=== Sobol sensitivity of floor/field: site {res['site']} "
          f"({res['width_um']:.0f} x {res['depth_um']:.0f} um), dx {res['dx_um']:.0f} um, "
          f"solver {res['flow_solver']}, flow {res['use_flow']} ===")
    print(f"grid {res['grid_shape'][0]} x {res['grid_shape'][1]} cells   "
          f"n_base {res['n_base']}   evaluations {res['n_eval']}   seed {res['seed']}   "
          f"ratio {res['mean_y']:.4f} +/- {res['std_y']:.4f} (min {res['min_y']:.4f}, "
          f"max {res['max_y']:.4f})")
    print(f"{'input':<16}{'S1':>8}  {'S1 95% CI':<18}{'ST':>8}  {'ST 95% CI':<18}{'ST-S1':>8}")
    for i, name in enumerate(res["names"]):
        s1 = f"[{res['S1_lo'][i]:+.3f}, {res['S1_hi'][i]:+.3f}]"
        st = f"[{res['ST_lo'][i]:+.3f}, {res['ST_hi'][i]:+.3f}]"
        tag = " (log)" if res["log_mask"][i] else ""
        print(f"{name + tag:<16}{res['S1'][i]:>8.3f}  {s1:<18}{res['ST'][i]:>8.3f}  {st:<18}"
              f"{res['ST'][i] - res['S1'][i]:>8.3f}")
    print(f"dominant input: {res['dominant_input']}   wall {res['wall_s']:.1f} s "
          f"({res['s_per_eval']:.3f} s per evaluation)")
    print(f"sum S1 = {float(np.sum(res['S1'])):.3f}   sum ST = {float(np.sum(res['ST'])):.3f} "
          f"(> 1 means interactions)")


def main(argv=None):
    """CLI: ``python -m platesim.trenchflow.sensitivity --site narrow --n-base 64 --workers 6``."""
    ap = argparse.ArgumentParser(
        description="Global (Sobol) sensitivity of the trench floor/field ratio (F6)")
    ap.add_argument("--site", default="narrow", choices=["narrow", "wide", "both"])
    ap.add_argument("--n-base", type=int, default=64, help="base sample size (N = n_base * (d + 2))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dx", type=float, default=DX_UM, help="grid spacing in um (default 8)")
    ap.add_argument("--no-flow", action="store_true", help="reference run: no cross-flow at all")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the CIs")
    ap.add_argument("--out", default=None, help="write the results to this JSON file")
    args = ap.parse_args(argv)

    sites = ["narrow", "wide"] if args.site == "both" else [args.site]
    results = {}
    t0 = time.perf_counter()
    for site in sites:
        res = trench_ratio_sensitivity(site=site, n_base=args.n_base, seed=args.seed,
                                       workers=args.workers, use_flow=not args.no_flow,
                                       dx_um=args.dx, n_boot=args.n_boot)
        results[site] = res
        _print_table(res)
        print("")
    total = time.perf_counter() - t0
    print(f"total wall {total:.1f} s")
    if args.out:
        payload = {"task": "F6 sobol sensitivity of trench floor/field",
                   "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "total_wall_s": total, "results": results}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(payload), fh, indent=2, allow_nan=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


