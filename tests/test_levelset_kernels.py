"""Level-set kernels: WENO5 + TVD-RK3 accuracy and the narrow band (plan v2 C3 + B4).

The exact solution of  psi_t + v |grad psi| = 0  with a constant v is the initial front displaced along its
normal by v*T (a Minkowski sum with a disk of radius v*T), so both cases below have a closed-form answer:

  * an expanding circle:      radius R = r0 + v*T, measured on a unit-spacing node grid at 64 rays with a
    quartic (5th order) interpolation of the numerical field -- the cubic-interpolated variant bottoms out
    at its own interpolation error (~1e-7) and could not resolve the WENO5 error;
  * an expanding square:      the square grown by v*T with rounded corners; the area error is measured as
    -closed_integral(psi ds) along the exact boundary polyline, which is the first-order area change and
    needs no sub-cell volume formula (the linear-ramp volume of levelset.electrolyte_fraction has a ~1e-2
    systematic bias of its own at this resolution and would swamp both schemes).

The narrow-band path must reproduce the full-grid path inside the band exactly (the update set is dilated by
the RK3 stencil radius), the numba kernels must reproduce the numpy ones, and the default advect() must stay
bit-identical.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from platesim.feature.engine import ls_kernels as K
from platesim.feature.engine.levelset import (Grid, NARROW_CELLS, advect, initial_psi, solid_volume,
                                             volume_correct)

V, T, CFL = 0.1, 2.0, 0.4          # normal speed, final time, CFL of every convergence run
R0 = 0.15                          # initial circle radius (unit domain)


# --------------------------------------------------------------------------------- exact test geometries
def _nodes(n):
    return np.arange(n + 1) * (1.0 / n)


def _circle(n, r0=R0):
    """Exact signed distance to a circle of radius r0 around the centre of the unit square (node grid)."""
    x = _nodes(n)
    X, Z = np.meshgrid(x, x, indexing="ij")
    return np.hypot(X - 0.5, Z - 0.5) - r0


def _square(n, a=0.2):
    """Exact signed distance to an axis-aligned square of half width a centred in the unit square."""
    x = _nodes(n)
    X, Z = np.meshgrid(x, x, indexing="ij")
    ex, ez = np.abs(X - 0.5) - a, np.abs(Z - 0.5) - a
    return np.hypot(np.maximum(ex, 0), np.maximum(ez, 0)) + np.minimum(np.maximum(ex, ez), 0.0)


def _box_sdf(x, z, a, g):
    """Signed distance to square(a) grown by g (the exact Minkowski sum)."""
    ex, ez = abs(x) - a, abs(z) - a
    return math.hypot(max(ex, 0.0), max(ez, 0.0)) + min(max(ex, ez), 0.0) - g


def _interp(psi):
    """5th-order (quintic) interpolant of a node field on the unit square."""
    dx = 1.0 / (psi.shape[0] - 1)
    x = np.arange(psi.shape[0]) * dx
    return RegularGridInterpolator((x, x), psi, method="quintic")


def _advance(psi, scheme, v=V, t_end=T, cfl=CFL):
    """Integrate psi_t + v|grad psi| = 0 up to t_end with dt collapsed to an integer number of steps."""
    dx = 1.0 / (psi.shape[0] - 1)
    nt = int(math.ceil(t_end * v / (cfl * dx)))
    dt = t_end / nt
    vv = np.full_like(psi, v)
    for _ in range(nt):
        psi = advect(psi, vv, dt, dx, periodic=True, scheme=scheme)
    return psi


def _radius_error(psi, radius, nray=64):
    """-(psi) averaged over the exact front: one Newton step of a distance field == the radius error."""
    th = np.arange(nray) * 2 * math.pi / nray
    pts = np.stack([0.5 + radius * np.cos(th), 0.5 + radius * np.sin(th)], axis=1)
    return -float(_interp(psi)(pts).mean())


def _area_error(psi, a, g, n=360):
    """-closed_integral(psi ds) along the exact Minkowski boundary (first-order area deficit/excess)."""
    th = np.arange(n) * 2 * math.pi / n
    r = np.array([brentq(lambda rr: _box_sdf(rr * math.cos(t), rr * math.sin(t), a, g), 0.0, 1.0) for t in th])
    P = np.stack([0.5 + r * np.cos(th), 0.5 + r * np.sin(th)], axis=1)
    Q = np.roll(P, -1, axis=0)
    L = np.hypot(*(Q - P).T)
    f = _interp(psi)
    return -float((L * (f(P) + f(Q)) / 2.0).sum())


def _order(e_coarse, e_fine):
    return math.log(abs(e_coarse) / abs(e_fine)) / math.log(2.0)


# --------------------------------------------------------------------------------- (a) circle
def test_circle_expansion_radius_error_godunov1_vs_weno5():
    """Radius error against r0 + v*T at two grids: godunov1 is first order, WENO5 is far more accurate."""
    err = {}
    for scheme in ("godunov1", "weno5"):
        err[scheme] = [_radius_error(_advance(_circle(n), scheme), R0 + V * T) for n in (48, 96)]
    print("\n[a] circle radius error (R = %.3f):" % (R0 + V * T))
    for scheme, e in err.items():
        print(f"    {scheme:9s} n= 48 err={e[0]:+.4e}   n= 96 err={e[1]:+.4e}   order={_order(*e):.2f}")
    g, w = err["godunov1"], err["weno5"]
    assert 0.8 < _order(*g) < 1.3, f"godunov1 must stay first order, got {g}"
    assert _order(*w) > 1.5, f"weno5 observed order {_order(*w):.2f} is not > 1.5 ({w})"
    for gc, wc in zip(g, w):
        assert abs(wc) < 0.01 * abs(gc), f"weno5 {wc:.3e} is not clearly below godunov1 {gc:.3e}"
        assert wc < 0.0, f"the front must lag the exact circle (numerical radius below exact), got {wc:.3e}"
        assert abs(wc) < 0.1 * R0, "the measured radius must have grown to ~r0 + v*T"


# --------------------------------------------------------------------------------- (b) square / corners
def test_square_expansion_corner_area_error_godunov1_vs_weno5():
    """Area error against the exact Minkowski sum (square grown by v*T with rounded corners)."""
    a, g = 0.2, V * T
    err = {}
    for scheme in ("godunov1", "weno5"):
        err[scheme] = [_area_error(_advance(_square(n, a), scheme), a, g) for n in (48, 96)]
    print("\n[b] square area error (a = %.2f, corner radius %.2f):" % (a, g))
    for scheme, e in err.items():
        print(f"    {scheme:9s} n= 48 err={e[0]:+.4e}   n= 96 err={e[1]:+.4e}")
    gd, wn = err["godunov1"], err["weno5"]
    for gc, wc in zip(gd, wn):
        assert wc < 0.0, f"the numerical front must lag the exact Minkowski boundary, got {wc:.3e}"
        assert abs(wc) < 0.5 * abs(gc), f"weno5 corner error {wc:.3e} vs godunov1 {gc:.3e} is not 2x better"
    # the smeared corner is a real, grid-dependent error for the first-order scheme
    assert abs(gd[1]) < abs(gd[0]), "godunov1 area error must shrink with the grid"


# --------------------------------------------------------------------------------- WENO5 operator order
def test_weno5_operator_is_fifth_order_in_a_smooth_region():
    """The Hamilton-Jacobi operator itself (internal, no time stepping):

    psi = 0.8x + 0.1 sin(2 pi x) + 0.9z + 0.1 cos(2 pi z) has strictly positive psi_x, psi_z, so the Godunov
    Hamiltonian is exactly |grad psi| and the WENO5 upwind derivatives can be compared with the analytic
    value.  A linear scheme must not be able to do this (the legacy first-order one gives order 1).
    """
    def err(n, m=8):
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        X, Z = np.meshgrid(x, x, indexing="ij")
        psi = 0.8 * X + 0.1 * np.sin(2 * np.pi * X) + 0.9 * Z + 0.1 * np.cos(2 * np.pi * Z)
        H = K._weno5_rhs_full(psi, np.ones_like(psi), dx, False)          # H(psi) == |grad psi| here
        ref = np.hypot(0.8 + 0.2 * np.pi * np.cos(2 * np.pi * X), 0.9 - 0.2 * np.pi * np.sin(2 * np.pi * Z))
        sl = (slice(m, -m), slice(m, -m))                                # interior: the edges are padded
        return float(np.abs(H[sl] - ref[sl]).max() / np.abs(ref[sl]).max())

    e = [err(n) for n in (32, 64, 128)]
    print("\n[op] WENO5 |grad psi| error: " + "  ".join(f"n={n} {v:.3e}" for n, v in zip((32, 64, 128), e)))
    assert _order(e[0], e[1]) > 4.5 and _order(e[1], e[2]) > 4.5, f"operator order {_order(e[0], e[1]):.2f}"


# --------------------------------------------------------------------------------- (c) narrow band
def test_narrow_band_matches_the_full_grid_inside_the_band():
    """|psi| < band*dx: band == full to 1e-12; farther cells are returned untouched."""
    n, band = 160, 8.0
    dx = 1.0 / n
    psi0 = _circle(n)
    vv = np.full_like(psi0, V)
    dt = CFL * dx / V
    m = np.abs(psi0) < band * dx
    assert m.sum() > 200
    for scheme, halo in (("godunov1", 0), ("weno5", 3 * K.STENCIL)):
        full = advect(psi0, vv, dt, dx, periodic=True, scheme=scheme)
        nb = advect(psi0, vv, dt, dx, periodic=True, scheme=scheme, band_cells=band)
        np.testing.assert_allclose(nb[m], full[m], rtol=0, atol=1e-12,
                                   err_msg=f"{scheme}: narrow band differs from the full grid in the band")
        outside = ~K.dilate_band(m, halo, True)
        np.testing.assert_array_equal(nb[outside], psi0[outside])
        assert np.abs(nb[m] - psi0[m]).max() > 1e-6, f"{scheme}: nothing was advanced in the band"
    # an empty band is a no-op
    far = psi0 + 100.0 * dx
    np.testing.assert_array_equal(advect(far, vv, dt, dx, periodic=True, scheme="weno5", band_cells=band), far)
    # ... and the scheme name is validated on every path
    with pytest.raises(ValueError):
        advect(psi0, vv, dt, dx, periodic=True, scheme="godunov2")


# --------------------------------------------------------------------------------- (d) numba == numpy
@pytest.mark.skipif(not K.numba_available(), reason="numba is an optional extra (uv run --with numba)")
def test_numba_kernels_match_the_numpy_fallback():
    n = 96
    dx = 1.0 / n
    psi = _circle(n) + 0.3 * dx * np.sin(9.0 * _nodes(n))[:, None]
    # (nr, 1) speed field: the legacy operator broadcasts it, so the jitted kernels must too
    vv = V * (1.0 + 0.5 * np.cos(7.0 * _nodes(n))[:, None])
    dt = CFL * dx / V
    for v_field in (vv, np.broadcast_to(vv, psi.shape).copy()):
        for band in (None, 8.0):
            a = K.advect_weno5(psi, v_field, dt, dx, True, band, backend="numpy")
            b = K.advect_weno5(psi, v_field, dt, dx, True, band, backend="numba")
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-12,
                                       err_msg=f"weno5 band={band} shape={np.shape(v_field)}")
        a = K.advect_godunov1_band(psi, v_field, dt, dx, True, 8.0, backend="numpy")
        b = K.advect_godunov1_band(psi, v_field, dt, dx, True, 8.0, backend="numba")
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12, err_msg="godunov1 band")
    with pytest.raises(ValueError):
        K.advect_weno5(psi, vv, dt, dx, True, None, backend="cuda")


# --------------------------------------------------------------------------------- (e) default path
def test_default_advect_is_bit_identical():
    """No kwargs (today's call in sim.py) == explicit defaults == the pre-B4 expression, bit for bit."""
    g = Grid(30.0, 100.0, 0.3, 20.0, dx_um=2.0, geometry="trench_full")
    psi = initial_psi(g)
    v = 0.8e-7 * (1.0 + 0.2 * np.cos(5.0 * g.rc)) * np.ones_like(psi)
    dt = 0.2 * g.dx / v.max()
    base = advect(psi, v, dt, g)                                  # legacy positional call
    np.testing.assert_array_equal(base, advect(psi, v, dt, g, scheme="godunov1", band_cells=None))
    # the pre-B4 body, written out here: any reordering of the arithmetic would show up as a bit difference
    p = np.pad(psi, 1, mode="edge")
    p[0, 1:-1], p[-1, 1:-1] = psi[-1, :], psi[0, :]               # periodic x (trench_full)
    c = p[1:-1, 1:-1]
    dx = g.dx
    dxm, dxp = (c - p[:-2, 1:-1]) / dx, (p[2:, 1:-1] - c) / dx
    dzm, dzp = (c - p[1:-1, :-2]) / dx, (p[1:-1, 2:] - c) / dx
    gpos = np.sqrt(np.maximum(dxm, 0) ** 2 + np.minimum(dxp, 0) ** 2
                   + np.maximum(dzm, 0) ** 2 + np.minimum(dzp, 0) ** 2)
    gneg = np.sqrt(np.minimum(dxm, 0) ** 2 + np.maximum(dxp, 0) ** 2
                   + np.minimum(dzm, 0) ** 2 + np.maximum(dzp, 0) ** 2)
    ref = psi - dt * (np.maximum(v, 0) * gpos + np.minimum(v, 0) * gneg)
    np.testing.assert_array_equal(base, ref)
    # the dx + periodic spelling and a zero velocity are still exact identities
    np.testing.assert_array_equal(base, advect(psi, v, dt, g.dx, periodic=g.periodic_x))
    np.testing.assert_array_equal(advect(psi, np.zeros_like(psi), dt, g), psi)
    # the Grid form works for the new scheme too (dx / periodic_x are taken from the grid)
    np.testing.assert_array_equal(advect(psi, v, dt, g, scheme="weno5"),
                                  advect(psi, v, dt, g.dx, periodic=g.periodic_x, scheme="weno5"))


# --------------------------------------------------------------------------------- (f) volume correction
def test_volume_correct_hits_the_target_inside_the_band():
    g = Grid(30.0, 100.0, 0.3, 20.0, dx_um=2.0, geometry="trench_full")
    psi = initial_psi(g)
    v0 = solid_volume(psi, g)
    m = np.abs(psi) < NARROW_CELLS * g.dx
    far = ~m
    for fac in (1.01, 0.99):
        target = v0 * fac
        out = volume_correct(psi, target, g)
        assert abs(solid_volume(out, g) / target - 1.0) < 1e-9
        np.testing.assert_array_equal(out[far], psi[far])                 # far field untouched
        d = (out - psi)[m]
        np.testing.assert_allclose(d, d[0], rtol=0, atol=1e-18)           # one constant shift
        assert np.sign(d[0]) == -np.sign(fac - 1.0)      # ef grows with psi, so more solid => psi down
    # already at the target: identity; unreachable target: capped at the band half width
    np.testing.assert_array_equal(volume_correct(psi, v0, g), psi)
    capped = volume_correct(psi, 10.0 * v0, g)
    assert (capped - psi).min() >= -NARROW_CELLS * g.dx - 1e-18
    assert (capped - psi).max() <= 0.0
    assert solid_volume(capped, g) > v0                       # as much solid as the band allows
    assert solid_volume(capped, g) < 10.0 * v0                # ... but not the target


# --------------------------------------------------------------------------------- (g) timing report
def test_timing_report_300x150():
    """Per-call cost on a 300 x 150 grid (z by r): full grid vs narrow band, numpy vs numba."""
    nr, nz, dx = 150, 300, 2e-4
    r = np.hypot(np.subtract.outer(np.arange(nr) - nr / 2, np.zeros(nz)),
                 np.subtract.outer(np.zeros(nr), np.arange(nz) - nz / 2))
    psi = (r - min(nr, nz) / 4.0) * dx
    v = np.full_like(psi, 1e-6)
    dt = 0.1 * dx
    band = NARROW_CELLS
    reps = 5

    def timeit(fn, *a, **kw):
        fn(*a, **kw)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*a, **kw)
        return (time.perf_counter() - t0) / reps * 1e3

    rows = [("godunov1 full", lambda: advect(psi, v, dt, dx, True))]
    rows += [(f"godunov1 band {band:.0f}", lambda: advect(psi, v, dt, dx, True, band_cells=band))]
    for backend in ("numpy", "numba"):
        if backend == "numba" and not K.numba_available():
            continue
        rows += [
            (f"weno5 full {backend}", lambda b=backend: K.advect_weno5(psi, v, dt, dx, True, None, b)),
            (f"weno5 band {backend}", lambda b=backend: K.advect_weno5(psi, v, dt, dx, True, band, b)),
            (f"godunov1 band {backend}",
             lambda b=backend: K.advect_godunov1_band(psi, v, dt, dx, True, band, b)),
        ]
    times = {}
    print(f"\n[g] 300 x 150 grid ({nr * nz} cells, {int((np.abs(psi) < band * dx).sum())} in the band):")
    for name, fn in rows:
        times[name] = timeit(fn)
        print(f"    {name:22s} {times[name]:7.3f} ms/call")
    if K.numba_available():
        assert times["weno5 band numba"] < times["weno5 full numba"] < times["weno5 full numpy"]
    assert times[f"weno5 band numpy"] < times["weno5 full numpy"]
    assert times[f"godunov1 band {band:.0f}"] < 0.25 * times["weno5 full numpy"]
