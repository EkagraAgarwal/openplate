"""Phase 3 step 1: front curvature and the CEAC carried-coverage kernel (pure functions)."""
import math
import time
from dataclasses import replace

import numpy as np
import pytest

from platesim.feature.engine.params import Params
from platesim.tile3d import levelset3d as ls
from platesim.tile3d.additives3d import ceac_carry_step

UM = 1e-4                       # um -> cm
PER = (False, False, False)


@pytest.fixture(scope="module", autouse=True)
def bounded_numba_threads():
    """Keep the synthetic parallel kernels below a reproducible thread budget."""
    try:
        import numba
    except ImportError:
        yield
        return
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    try:
        yield
    finally:
        numba.set_num_threads(previous)


def _mesh(R_um, dx_um, half_span=1.3, per_y=False, ny=None):
    """Centred cell grid; ``half_span`` is in units of R.  Returns (X,Y,Z), dx, periodic."""
    dx = dx_um * UM
    half = half_span * R_um * UM
    n = int(round(2.0 * half / dx))
    a = (np.arange(n) + 0.5) * dx - half
    b = a if ny is None else (np.arange(ny) + 0.5) * dx - half
    X, Y, Z = np.meshgrid(a, b, a, indexing="ij")
    periodic = (False, True, False) if per_y else PER
    return (X, Y, Z), dx, periodic


# (a) front curvature: sphere +2/R, cylinder +1/R, plane 0, clip at 1/(2 dx)
def test_front_curvature_sphere_cylinder_plane_and_clip():
    R_um, dx_um = 40.0, 4.0
    R = R_um * UM
    (X, Y, Z), dx, _ = _mesh(R_um, dx_um)

    psi_s = np.sqrt(X*X + Y*Y + Z*Z) - R
    k = ls.front_curvature(psi_s, dx, PER)
    front = np.abs(psi_s) < dx
    assert front.any()
    assert abs(float(np.mean(k[front])) / (2.0 / R) - 1.0) < 0.05

    (X2, Y2, Z2), dx2, per_y = _mesh(R_um, dx_um, per_y=True, ny=8)
    psi_c = np.sqrt(X2*X2 + Z2*Z2) - R
    kc = ls.front_curvature(psi_c, dx2, per_y)
    front_c = np.abs(psi_c) < dx2
    assert abs(float(np.mean(kc[front_c])) / (1.0 / R) - 1.0) < 0.05

    kf = ls.front_curvature(Z.copy(), dx, PER)
    assert np.max(np.abs(kf)) < 1e-8

    # tiny sphere: raw curvature exceeds the resolution guard; clip caps it.
    Rt = 2.0 * dx
    psi_t = np.sqrt(X*X + Y*Y + Z*Z) - Rt
    raw = ls.front_curvature(psi_t, dx, PER, clip=False)
    clipped = ls.front_curvature(psi_t, dx, PER, clip=True)
    assert np.max(np.abs(raw)) > 0.5 / dx
    assert np.all(np.abs(clipped) <= 0.5 / dx + 1e-9)


# (b) flat front, constant v: the exact 0-D Langmuir closed form
def test_ceac_flat_front_matches_zerod_closed_form():
    (X, Y, Z), dx, _ = _mesh(40.0, 4.0)
    psi = Z.copy()
    v, dt, theta0 = 1e-5, 1.0, 0.5
    p = Params()
    psi_new = ls.advect(psi, v, dt, dx, PER, backend="numpy")
    theta = ceac_carry_step(np.full(psi.shape, theta0), psi, psi_new, v, dt, dx, PER, p)
    rate = p.k_acc * p.c_acc + p.k_inc_a * v
    theta_inf = p.k_acc * p.c_acc / rate
    expect = theta_inf + (theta0 - theta_inf) * math.exp(-rate * dt)
    assert theta.shape == psi.shape
    np.testing.assert_allclose(theta[np.abs(psi) < dx], expect, rtol=0.0, atol=1e-10)


# (c) sign: growing solid dilutes, shrinking cavity enriches
def test_ceac_sign_growing_sphere_vs_shrinking_cavity():
    (X, Y, Z), dx, _ = _mesh(40.0, 4.0)
    R = 40.0 * UM
    v, dt, theta0 = 1e-4, 1.0, 0.5
    p = replace(Params(), k_acc=0.0, k_inc_a=0.0)
    th = np.full(X.shape, theta0)

    psi_s = np.sqrt(X*X + Y*Y + Z*Z) - R            # kappa = +2/R
    out_s = ceac_carry_step(th, psi_s, ls.advect(psi_s, v, dt, dx, PER, backend="numpy"),
                            v, dt, dx, PER, p)
    psi_c = R - np.sqrt(X*X + Y*Y + Z*Z)            # kappa = -2/R
    out_c = ceac_carry_step(th, psi_c, ls.advect(psi_c, v, dt, dx, PER, backend="numpy"),
                            v, dt, dx, PER, p)
    assert float(np.mean(out_s[np.abs(psi_s) < dx])) < theta0
    assert float(np.mean(out_c[np.abs(psi_c) < dx])) > theta0


# (d) the key CEAC test: surface amount theta*A is conserved with k_a = k_inc = 0.
def _cavity_run(shape, dx_um, R0_um=60.0, R1_um=30.0, v=1e-5, backend="numba"):
    """Shrink an electrolyte cavity; return the drift of theta*R^exponent."""
    dx = dx_um * UM
    half = 1.3 * R0_um * UM
    n = int(round(2.0 * half / dx))
    a = (np.arange(n) + 0.5) * dx - half
    if shape == "sphere":
        X, Y, Z = np.meshgrid(a, a, a, indexing="ij")
        r = np.sqrt(X*X + Y*Y + Z*Z)
        per, expo = PER, 2
        rad = lambda V: (3.0*V/(4.0*math.pi))**(1.0/3.0)
    else:                                            # y-extruded cylinder, periodic y
        ny = 8
        b = (np.arange(ny) + 0.5) * dx - half
        X, Y, Z = np.meshgrid(a, b, a, indexing="ij")
        r = np.sqrt(X*X + Z*Z)
        per, expo = (False, True, False), 1
        length = ny * dx
        rad = lambda V: math.sqrt(V/(math.pi*length))

    psi = R0_um * UM - r                             # electrolyte cavity inside solid
    p = replace(Params(), k_acc=0.0, k_inc_a=0.0)
    theta = np.full(psi.shape, 0.1)
    dt = 0.3 * dx / v                                # CFL = 0.3
    vol = lambda: float(np.sum(ls.electrolyte_fraction(psi, dx))) * dx**3
    prod0 = 0.1 * rad(vol())**expo
    t0, steps = time.perf_counter(), 0
    while rad(vol()) > R1_um * UM:
        kappa = ls.front_curvature(psi, dx, per)     # from psi_pre
        psi_new = ls.advect(psi, v, dt, dx, per, backend=backend)
        theta = ceac_carry_step(theta, psi, psi_new, v, dt, dx, per, p, kappa=kappa)
        psi = psi_new
        steps += 1
        if steps % 4 == 0:
            psi = ls.reinit(psi, dx, per, backend=backend)
    R_end = rad(vol())
    theta_end = float(np.mean(theta[np.abs(psi) < dx]))
    drift = 100.0 * (theta_end * R_end**expo / prod0 - 1.0)
    return dict(drift=drift, theta=theta_end, R_um=R_end/UM, steps=steps,
                wall=time.perf_counter() - t0)


@pytest.mark.slow
def test_cavity_carry_conservation():
    """theta*R^2 (sphere) / theta*R (cylinder) within 3% at R = 30 um.

    At dx = 4 um the sphere is at only R/dx ~ 7.5 and its volume quadrature and
    discrete curvature give ~3.8% drift; this is reported, not hidden.  The
    dx = 2 um rerun converges to ~1.9%, which is what the 3% assertion checks.
    The cylinder is within 3% already at dx = 4 um.
    """
    results = {}
    for shape, dxu in (("sphere", 4.0), ("sphere", 2.0), ("cylinder", 4.0)):
        results[(shape, dxu)] = _cavity_run(shape, dxu)
        r = results[(shape, dxu)]
        exponent = 2 if shape == "sphere" else 1
        print(f"\n{shape} dx={dxu:g}um: theta*R^{exponent} drift={r['drift']:+.2f}% "
              f"theta={r['theta']:.4f} R={r['R_um']:.2f}um "
              f"steps={r['steps']} wall={r['wall']:.1f}s", flush=True)
    assert abs(results[("sphere", 2.0)]["drift"]) < 3.0
    assert abs(results[("cylinder", 4.0)]["drift"]) < 3.0
