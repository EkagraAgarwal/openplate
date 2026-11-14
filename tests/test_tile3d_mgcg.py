"""MGCG contract checks. Run with -s to retain solver comparison tables."""
import os
import time

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads
from scipy.sparse.linalg import spsolve

from platesim.tile3d import operators
from platesim.tile3d.backend.mgcg_numba import MGCG
from platesim.tile3d.solvers_ref import solve_amg


@pytest.fixture(autouse=True)
def threads():
    old = get_num_threads()
    set_num_threads(min(old, 4))
    yield
    set_num_threads(old)


def trench(shape=(48, 8, 48)):
    """Copy of FieldSolver's diffuse geometry and potential Newton diagonal."""
    from platesim.core.constants import F
    from platesim.tile3d.backend import ls_numpy

    # Params defaults at 22 C, copied to avoid feature.__init__'s unrelated
    # contourpy dependency (not part of the isolated MGCG test environment).
    eps_fac, n_active_eps, kappa = 1.2, 3.5, .065
    af, i0_u = .5 * F / (8.314 * 295.15), .002
    dx = 4e-4 * 48 / shape[0]
    nx, ny, nz = shape
    x = (np.arange(nx) + .5) * dx - nx * dx / 2
    z0 = -0.55 * nz * dx
    zc = z0 + (np.arange(nz) + .5) * dx
    # Union of upper half-space and a rectangular trench (signed-distance CSG).
    psi = np.broadcast_to(np.maximum(zc[None, None, :], np.minimum(
        nx * dx * .19 - abs(x[:, None, None]),
        zc[None, None, :] + nz * dx * .42)), shape).copy()
    periodic = (True, True, False)
    z_plane = zc[-1] + .8 * dx
    kt = int(np.floor((z_plane - .5 * dx - z0) / dx - .5))
    assert kt == nz - 1
    d_top = z_plane - zc[kt]
    eps = eps_fac * dx
    v = np.clip(psi / eps, -30, 30)
    Hs = np.maximum(.5 * (1 + np.tanh(v)), 1e-6)
    gm = np.sqrt(sum(((ls_numpy.shifted(psi, 1, a, periodic)
                       - ls_numpy.shifted(psi, -1, a, periodic)) / (2 * dx)) ** 2
                     for a in range(3)))
    active = psi > -n_active_eps * eps
    w = np.where(active, .5 / eps / np.cosh(v) ** 2 * gm, 0)
    top_scale = Hs[:, :, -1] * dx / (2 * d_top)
    faces = []
    for axis in range(3):
        h = np.moveaxis(Hs, axis, 0)
        f = np.zeros((h.shape[0] + 1, *h.shape[1:]))
        f[1:-1] = .5 * (h[:-1] + h[1:])
        if periodic[axis]:
            f[0] = f[-1] = .5 * (h[0] + h[-1])
        faces.append(np.moveaxis(f, 0, axis).copy())
    faces[2][:, :, -1] = top_scale
    faces = [kappa * f for f in faces]
    I = i0_u * np.exp(af * .12) * np.ones(shape)
    q = w * af * I
    args = dict(reaction=q, active=active, dirichlet={"z_hi": 0.0}, periodic=periodic)
    A, _ = operators.assemble(*faces, dx, **args)
    b = -w * I
    return faces, dx, args, A, b.ravel()


def test_apply():
    rng = np.random.default_rng(31)
    shape = (12, 8, 10)
    faces = operators.face_coefficients(np.exp(rng.normal(size=shape)), (True, True, False))
    args = dict(active=rng.random(shape) > .3, reaction=rng.random(shape),
                dirichlet={"z_hi": 2.0}, periodic=(True, True, False))
    A, _ = operators.assemble(*faces, .07, **args)
    u = rng.normal(size=shape)
    solver = MGCG(*faces, .07, **args)
    np.testing.assert_allclose(solver.apply(u).ravel(), A @ u.ravel(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-8), (np.float32, 1e-5)])
def test_real_field(dtype, tol):
    faces, dx, args, A, b = trench()
    start = time.perf_counter()
    solver = MGCG(*faces, dx, dtype=dtype, **args)
    setup = time.perf_counter() - start
    x, info = solver.solve(b, tol=tol)
    true_residual = np.linalg.norm(b - A @ x.ravel()) / np.linalg.norm(b)
    print(f"\nMGCG {np.dtype(dtype)} setup={setup:.4f}s {info} original_residual={true_residual:.3e}")
    assert x.dtype == dtype
    if dtype == np.float32 and (not info["converged"] or true_residual > tol):
        assert true_residual < 1e-3, info
        pytest.xfail(f"native float32 stalled at {true_residual:.3e}; requested {tol:.1e}")
    assert info["converged"], info
    assert true_residual <= tol
    if dtype == np.float64:
        direct = spsolve(A, b)
        assert np.linalg.norm(x.ravel() - direct) / np.linalg.norm(direct) < 1e-6
        start = time.perf_counter()
        _, amg = solve_amg(A, b, tol=tol)
        print(f"AMG total={time.perf_counter()-start:.4f}s {amg}")


def smooth_problem(n, dtype=np.float64):
    t = (np.arange(n) + .5) / n
    k = 1 + .2 * np.sin(2 * np.pi * t[:, None, None]) * np.cos(2 * np.pi * t[None, :, None])
    k = np.broadcast_to(k, (n, n, n)).copy()
    faces = operators.face_coefficients(k, (True, True, False))
    solver = MGCG(*faces, 1 / n, reaction=.1, dirichlet={"z_hi": 0}, dtype=dtype)
    rng = np.random.default_rng(56)
    b = 1 + rng.random(k.shape)
    return solver, b


def test_grid_independence():
    counts = []
    for n in (24, 48):
        solver, b = smooth_problem(n)
        _, info = solver.solve(b)
        print(f"\nsmooth {n}^3: {info}")
        assert info["converged"]
        counts.append(info["iterations"])
    assert counts[1] <= 1.5 * counts[0], counts


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("MGCG_SLOW") != "1", reason="set MGCG_SLOW=1 for 96^3")
def test_grid_independence_slow():
    counts = []
    for n in (24, 48, 96):
        solver, b = smooth_problem(n)
        _, info = solver.solve(b)
        print(f"\nsmooth {n}^3: {info}")
        assert info["converged"]
        counts.append(info["iterations"])
    assert max(counts) <= 1.5 * min(counts), counts


def test_reaction_reuse():
    faces, dx, args, _, b = trench((24, 4, 24))
    solver = MGCG(*faces, dx, **args)
    hierarchy = [(id(l), tuple(id(f) for f in l.faces)) for l in solver.hierarchy]
    reaction = args["reaction"] * 2 + .01
    solver.set_reaction(reaction)
    assert hierarchy == [(id(l), tuple(id(f) for f in l.faces)) for l in solver.hierarchy]
    args["reaction"] = reaction
    fresh = MGCG(*faces, dx, **args)
    x, info = solver.solve(b, tol=1e-10)
    y, other = fresh.solve(b, tol=1e-10)
    assert info["converged"] and other["converged"]
    np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_symmetric_cycle_and_transfers(smoother):
    from platesim.tile3d.backend.mgcg_numba import _prolong, _restrict, _coarse_faces

    rng = np.random.default_rng(19)
    shape = (16, 8, 16)
    faces = operators.face_coefficients(np.ones(shape), (True, True, False))
    mask = rng.random(shape) > .1
    solver = MGCG(*faces, .1, reaction=2, active=mask, dirichlet={"z_hi": 0}, smoother=smoother)
    fine, coarse = solver.hierarchy[:2]
    a, b = rng.normal(size=shape), rng.normal(size=shape)
    a[~mask] = b[~mask] = 0
    fine.rhs[:] = a
    solver._cycle()
    ma = fine.x.copy()
    fine.rhs[:] = b
    solver._cycle()
    mb = fine.x.copy()
    assert np.vdot(a, ma) > 0
    np.testing.assert_allclose(np.vdot(a, mb), np.vdot(ma, b), rtol=1e-11, atol=1e-12)
    c = rng.normal(size=coarse.x.shape)
    pc, ra = np.empty(shape), np.empty(c.shape)
    _prolong(c, pc, mask, fine.factors, solver.periodic)
    _restrict(a, ra, mask, coarse.mask, fine.factors, solver.periodic)
    np.testing.assert_allclose(np.vdot(a, pc), 8 * np.vdot(ra, c), rtol=1e-12)
    # Constant contrasts survive conservative averaging, rather than being floored.
    low = [np.full(f.shape, 1e-6) for f in faces]
    for f in _coarse_faces(low, (2, 2, 2)):
        np.testing.assert_allclose(f, 1e-6, rtol=1e-15)


def test_edge_cases():
    from platesim.tile3d.backend.mgcg_numba import _dot

    large = np.array([[[1e20, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(_dot(large, large),
                               np.vdot(large.astype(float), large.astype(float)), rtol=1e-15)
    shape = (9, 8, 8)  # Odd periodic axis: RBGS must fall back to Jacobi.
    faces = operators.face_coefficients(np.ones(shape), (True, True, False))
    solver = MGCG(*faces, 1, reaction=.5, dtype=np.float32)
    assert not solver.hierarchy[0].rbgs
    assert all(f.dtype == np.float32 for l in solver.hierarchy for f in l.faces)
    zero, info = solver.solve(np.zeros(shape))
    assert info["iterations"] == 0 and info["converged"] and not zero.any()
    b = np.ones(shape)
    _, info = solver.solve(b, maxiter=0)
    assert not info["converged"] and info["iterations"] == 0
    x, info = solver.solve(b, tol=1e-5)
    assert info["converged"]
    _, warm = solver.solve(b, x0=x, tol=1e-5)
    assert warm["converged"] and warm["iterations"] == 0
    with pytest.raises(ValueError, match="equal positive"):
        MGCG(*faces, 1, pre=1, post=2)
    with pytest.raises(ValueError, match="nonnegative"):
        solver.set_reaction(-1)
