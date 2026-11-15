"""Richardson grid-error estimates for the feature-fill comparison tables (plan v2 item C6).

Every comparison table (scheme x site x recipe) should carry a grid-error column instead of a single
number at one dx.  Given the same quantity at 2-4 grid spacings, :func:`richardson` returns

    extrapolated   f0 from  f(dx) = f0 + C dx^p  (Richardson, any refinement ratios)
    p_observed     observed order; None with two grids (order not resolvable) or non-monotone data
    gci_fine       grid convergence index of the finest value (Roache/Celik):
                   Fs * |f_fine - f0| / |f_fine|, Fs = 1.25 for 3+ grids, 3.0 for two grids
    error_fine     |f_fine - f0| in the units of the quantity (absolute)
    monotone       whether the used values are monotone in dx
    note           what was assumed, used or flagged, human readable

Three or more grids: the order solves Celik et al. (2008) Eq. (3.6) in its iterative form,

    p = (1 / ln r21) * | ln|(f3 - f2) / (f2 - f1)| + ln((r21^p - 1) / (r32^p - 1)) |,

with r21 = h2/h1 and r32 = h3/h2 (h1 the finest).  Unlike the log-ratio shortcut, which is exact only
for a constant refinement ratio, this stays valid for r21 != r32 (e.g. 8/6/4 or 6/4/3); the iterate is
then polished on the equivalent three-point equation (single positive root, see
:func:`_three_point_order`) so p and f0 are accurate to floating point.  f0 is finally taken from the
two finest grids, which minimises the round-off amplification of the coarse point.

More than three grids: only the three finest enter the estimate (``dx_used``/``values_used`` say which).
Two grids: ``p_assumed`` supplies the order (2.0 = second order when None) and Fs = 3.0 per Roache.
Oscillatory / non-monotone convergence has no meaningful p: the estimate falls back to the finest value
with the spread of the finest values as the error, ``monotone=False`` and a note.  The same fallback is
used for a monotone sequence that is not a power law (no positive root in [0.05, 10]).

:func:`grid_error_column` applies the estimate to comparison-table rows, grouped by the other keys
(scheme, site, recipe, ...), adding ``extrapolated`` and ``grid_err`` (= ``gci_fine``, a relative
fraction) to every row of the group.

CLI:  uv run python -m platesim.feature.grid_error file.json [--p-assumed 2] [--json]
      file.json is {"dx": value}, e.g. {"8": 0.882, "4": 0.872, "2": 0.873, "1": 0.87089};
      "-" reads stdin.  Prints the extrapolated value, observed order, GCI and the note.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Iterable, Mapping, Sequence

FS_MULTI = 1.25           # Roache safety factor, three or more grids (Celik et al. 2008: Fs = 1.25)
FS_TWO = 3.0              # two grids: no observed order, so the conservative Fs = 3.0
P_DEFAULT = 2.0           # order assumed for two grids when the caller supplies none
P_MIN, P_MAX = 0.05, 10.0  # admissible observed-order window

__all__ = ["richardson", "grid_error_column", "main"]


# ------------------------------------------------------------------------------------ helpers
def _finite(x) -> float | None:
    """float(x) when it is finite, else None; tolerant of JSON strings/ints and of NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _usable(values: Mapping) -> list[tuple[float, float]]:
    """(dx, value) pairs with dx > 0 and both finite, sorted coarse -> fine."""
    pts: list[tuple[float, float]] = []
    for dx, val in values.items():
        d, f = _finite(dx), _finite(val)
        if d is None or f is None or d <= 0.0:
            continue
        pts.append((d, f))
    pts.sort(key=lambda p: -p[0])
    return pts


def _celik_order(dx1: float, f1: float, dx2: float, f2: float, dx3: float, f3: float,
                 tol: float = 1e-14, itmax: int = 500) -> float | None:
    """Observed order by the Celik et al. (2008) Eq. (3.6) fixed point, ``dx1 > dx2 > dx3``.

    ``r21 = h2/h1``, ``r32 = h3/h2`` with h1 the finest grid, so both ratios are > 1 and the form
    stays valid for non-uniform refinement.  Returns None when an increment vanishes (no order can be
    inferred), when the increments change sign (oscillatory: the caller flags it), or when the
    iteration leaves the admissible window.
    """
    r21 = dx2 / dx3                       # h2 / h1
    r32 = dx1 / dx2                       # h3 / h2
    num = f3 - f2
    den = f2 - f1
    if num == 0.0 or den == 0.0:
        return None
    if (num > 0.0) != (den > 0.0):
        return None                       # s = -1: oscillatory, no real p
    ln_r21 = math.log(r21)
    if not (ln_r21 > 0.0) or not (r32 > 1.0):
        return None
    term = math.log(abs(num / den))
    p = P_DEFAULT
    for _ in range(itmax):
        try:
            q = math.log((r21 ** p - 1.0) / (r32 ** p - 1.0))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        p_new = abs(term + q) / ln_r21    # the outer modulus is part of Celik's definition
        if not math.isfinite(p_new) or p_new <= 0.0 or p_new > 1.0e3:
            return None
        if abs(p_new - p) <= tol:
            return p_new
        p = p_new
    return p


def _three_point_order(dx1: float, f1: float, dx2: float, f2: float, dx3: float, f3: float,
                       tol_iters: int = 200) -> float | None:
    """Solve the three-point equation ``(f1-f3)/(f2-f3) = (dx1^p-dx3^p)/(dx2^p-dx3^p)``.

    This is the same power law as Celik's Eq. (3.6) but without the modulus, so it has at most one
    root in the window (the right-hand side increases monotonically with p for dx1 > dx2 > dx3);
    plain bisection to floating-point resolution.  None when there is no sign change in
    [P_MIN, P_MAX] -- i.e. the data are not consistent with one power law.
    """
    if f1 == f3 or f2 == f3:
        return None

    def residual(p: float) -> float:
        return ((f1 - f3) / (f2 - f3)
                - (dx1 ** p - dx3 ** p) / (dx2 ** p - dx3 ** p))

    try:
        lo, g_lo = P_MIN, residual(P_MIN)
        hi, g_hi = P_MAX, residual(P_MAX)
    except (OverflowError, ZeroDivisionError):
        return None
    if not (math.isfinite(g_lo) and math.isfinite(g_hi)) or g_lo * g_hi > 0.0:
        return None
    if g_lo == 0.0:
        return P_MIN
    if g_hi == 0.0:
        return P_MAX
    for _ in range(tol_iters):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        g_mid = residual(mid)
        if g_mid == 0.0:
            return mid
        if (g_mid > 0.0) == (g_lo > 0.0):
            lo, g_lo = mid, g_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _observed_order(dx1: float, f1: float, dx2: float, f2: float,
                    dx3: float, f3: float) -> float | None:
    """Celik iterative order, polished on the three-point equation; None if not a power law."""
    p_celik = _celik_order(dx1, f1, dx2, f2, dx3, f3)
    p_root = _three_point_order(dx1, f1, dx2, f2, dx3, f3)
    if p_root is not None:
        return p_root
    if p_celik is not None and P_MIN <= p_celik <= P_MAX:
        return p_celik
    return None


def _extrapolate(dx_coarse: float, f_coarse: float, dx_fine: float, f_fine: float,
                 p: float) -> float | None:
    """Richardson limit f0 from two grids of the model f = f0 + C dx^p."""
    if not (p > 0.0):
        return None
    try:
        den = (dx_coarse / dx_fine) ** p - 1.0
    except (OverflowError, ZeroDivisionError):
        return None
    if not (den > 0.0) or not math.isfinite(den):
        return None
    lim = f_fine + (f_fine - f_coarse) / den
    return lim if math.isfinite(lim) else None


def _set_error(out: dict, f_fine: float, error_abs: float) -> None:
    """Fill error_fine / error_fine_rel / gci_fine for an absolute error on the finest value."""
    out["error_fine"] = float(error_abs)
    if f_fine == 0.0:
        out["error_fine_rel"] = None
        out["gci_fine"] = None
        return
    rel = float(error_abs / abs(f_fine))
    out["error_fine_rel"] = rel
    out["gci_fine"] = float(out["fs"] * rel)


# ------------------------------------------------------------------------------------ estimate
def richardson(values: Mapping[float, float], p_assumed: float | None = None) -> dict:
    """Richardson extrapolation, observed order and a Roache/Celik GCI for one quantity.

    ``values`` maps grid spacing dx to the quantity at that spacing (2-4 grids; more than three uses
    the three finest).  ``p_assumed`` is the order used when only two grids are available (default
    2.0 = second order).  Returns the dict described in the module docstring; it never raises for a
    non-monotone or degenerate sequence -- it falls back and says so in ``note``.
    """
    if p_assumed is not None:
        p_assumed = _finite(p_assumed)
        if p_assumed is None or p_assumed <= 0.0:
            raise ValueError(f"p_assumed must be a positive finite order, got {p_assumed!r}")

    pts = _usable(values)
    n = len(pts)
    used = pts[-3:]                                   # the three finest (all of them if n < 3)
    out: dict = {
        "extrapolated": None, "p_observed": None, "gci_fine": None, "error_fine": None,
        "error_fine_rel": None, "monotone": None, "note": "",
        "fs": FS_MULTI if n >= 3 else FS_TWO,
        "n_grids": n,
        "dx_used": [p[0] for p in used],
        "values_used": [p[1] for p in used],
    }

    if n == 0:
        out["note"] = "no usable grids: need at least two positive dx with finite values"
        return out

    if n == 1:
        out["extrapolated"] = used[0][1]
        out["monotone"] = True
        _set_error(out, used[0][1], 0.0)
        out["note"] = "single grid: the value is returned unchanged, no grid error can be estimated"
        return out

    if n == 2:
        (dx_c, f_c), (dx_f, f_f) = pts
        out["monotone"] = True                        # two points cannot oscillate
        if f_c == f_f:
            out["extrapolated"] = f_f
            _set_error(out, f_f, 0.0)
            out["note"] = "two grids with identical values: nothing to extrapolate"
            return out
        p = P_DEFAULT if p_assumed is None else float(p_assumed)
        lim = _extrapolate(dx_c, f_c, dx_f, f_f, p)
        if lim is None:
            out["note"] = f"two grids but the extrapolation is undefined at p = {p:g}"
            return out
        out["extrapolated"] = lim
        _set_error(out, f_f, abs(f_f - lim))
        out["note"] = ("two grids: observed order not resolvable, p = "
                       + (f"{p:g} supplied" if p_assumed is not None else f"{p:g} assumed (default)"))
        return out

    (dx1, f1), (dx2, f2), (dx3, f3) = used            # dx1 > dx2 > dx3 (coarse -> fine)
    d12, d23 = f2 - f1, f3 - f2
    out["monotone"] = bool(d12 != 0.0 and d23 != 0.0 and (d12 > 0.0) == (d23 > 0.0))
    ratio_note = (f"r21 = {dx2 / dx3:.4g}, r32 = {dx1 / dx2:.4g}"
                  + ("" if abs(dx2 / dx3 - dx1 / dx2) <= 1e-12 * (dx2 / dx3) else " (non-uniform)"))
    extra = f"; the three finest of {n} grids are used" if n > 3 else ""

    if not out["monotone"]:
        spread = max(f1, f2, f3) - min(f1, f2, f3)
        out["extrapolated"] = f3
        _set_error(out, f3, spread)
        out["note"] = ("non-monotone (oscillatory) convergence in the 3 finest grids: no observed "
                       "order, extrapolated = the finest value and the error is the spread of the "
                       f"finest values ({spread:.6g}){extra}")
        return out

    p = _observed_order(dx1, f1, dx2, f2, dx3, f3)
    lim = None if p is None else _extrapolate(dx2, f2, dx3, f3, p)
    if lim is None:
        spread = max(f1, f2, f3) - min(f1, f2, f3)
        out["extrapolated"] = f3
        _set_error(out, f3, spread)
        out["note"] = ("monotone in dx but not consistent with one power law f0 + C dx^p (no observed "
                       "order in [0.05, 10]): extrapolated = the finest value and the error is the "
                       f"spread of the finest values ({spread:.6g}){extra}")
        return out

    out["p_observed"] = p
    out["extrapolated"] = lim
    _set_error(out, f3, abs(f3 - lim))
    out["note"] = f"observed order p = {p:.6g} from the 3 finest grids ({ratio_note}){extra}"
    return out


# ------------------------------------------------------------------------------------ table column
def _freeze(value) -> str:
    """Stable, hashable group-key component (table cells may be None, floats, lists)."""
    return repr(value)


def grid_error_column(rows: Iterable[Mapping], key_dx: str = "dx", key_val: str = "ratio",
                      group_by: Sequence[str] = ()) -> list[dict]:
    """Add ``extrapolated`` and ``grid_err`` to comparison-table rows, one estimate per group.

    ``rows`` are dicts carrying at least ``key_dx`` and ``key_val``.  ``group_by`` lists the columns
    that identify one convergence series (e.g. ``("scheme", "site")``); an empty tuple groups by every
    key present except ``key_dx``/``key_val``.  Rows with a non-finite or non-positive dx, or a
    non-finite value, are left in place with ``grid_err = None`` but do not enter the estimate.

    Returns NEW row dicts (the input is not mutated).  ``grid_err`` is the group's ``gci_fine`` (a
    relative fraction, safety factor included); it is None for a group with fewer than two usable
    grids.  Two rows in one group claiming the same dx with different values are an error.

    Raises ValueError on a conflicting duplicate dx within a group.
    """
    out = [dict(r) for r in rows]
    keys = tuple(group_by)
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(out):
        if keys:
            gid = tuple(_freeze(row.get(k)) for k in keys)
        else:
            # group by every other column; the two output columns never split a group, so the
            # function can be re-applied to its own output
            gid = tuple(sorted((k, _freeze(row.get(k))) for k in row
                               if k not in (key_dx, key_val, "extrapolated", "grid_err")))
        groups.setdefault(gid, []).append(i)

    for idx in groups.values():
        series: dict[float, float] = {}
        for i in idx:
            d = _finite(out[i].get(key_dx))
            v = _finite(out[i].get(key_val))
            if d is None or v is None or d <= 0.0:
                continue
            if d in series and series[d] != v:
                raise ValueError(f"conflicting values at dx = {d:g} in one group: "
                                 f"{series[d]!r} vs {v!r}")
            series[d] = v
        est = richardson(series) if len(series) >= 2 else None
        for i in idx:
            out[i]["extrapolated"] = None if est is None else est["extrapolated"]
            out[i]["grid_err"] = None if est is None else est["gci_fine"]
    return out


# ------------------------------------------------------------------------------------ CLI
def _print_estimate(est: dict) -> None:
    dx_used = ", ".join(f"{d:g}" for d in est["dx_used"])
    print(f"grids: {est['n_grids']} (dx = {dx_used} used)")
    if est["extrapolated"] is not None:
        print(f"extrapolated = {est['extrapolated']:.9g}")
    if est["values_used"]:
        print(f"finest value = {est['values_used'][-1]:.9g}")
    print("observed order = " + ("-" if est["p_observed"] is None else f"{est['p_observed']:.6g}"))
    if est["error_fine"] is not None:
        print(f"error(fine) = {est['error_fine']:.6g}")
    if est["gci_fine"] is not None:
        print(f"GCI(fine) = {100.0 * est['gci_fine']:.3g} % (Fs = {est['fs']:g})")
    if est["note"]:
        print(f"note: {est['note']}")


def main(argv: Sequence[str] | None = None) -> int:
    """python -m platesim.feature.grid_error <file.json> [--p-assumed P] [--json]."""
    ap = argparse.ArgumentParser(
        prog="python -m platesim.feature.grid_error",
        description="Richardson extrapolation and Roache/Celik GCI from 2-4 grid spacings.")
    ap.add_argument("path", help="JSON object {dx: value}; '-' reads stdin")
    ap.add_argument("--p-assumed", type=float, default=None,
                    help="order used when only two grids are given (default 2 = second order)")
    ap.add_argument("--json", action="store_true", help="print the full estimate as JSON")
    args = ap.parse_args(argv)

    try:
        if args.path == "-":
            raw = json.load(sys.stdin)
        else:
            with open(args.path, encoding="utf-8") as fh:
                raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"grid_error: cannot read {args.path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(raw, dict):
        print("grid_error: expected a JSON object {dx: value}", file=sys.stderr)
        return 2
    try:
        est = richardson(raw, p_assumed=args.p_assumed)
    except ValueError as exc:
        print(f"grid_error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(est, indent=2))
    else:
        _print_estimate(est)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
