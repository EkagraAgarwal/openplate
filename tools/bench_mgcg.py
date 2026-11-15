"""Isolated MGCG/SA-AMG timings; no project environment or lockfile required.

Example (set PYTHONPATH=. first):
uv run --no-project --python 3.12 --with numpy --with scipy --with numba
    --with pyamg python tools/bench_mgcg.py

JIT is warmed separately. AMG setup includes fine-grid sparse assembly. Both
storage precisions are tested, including Krylov vectors. Residuals in
the table are independently evaluated against the original float64 operator.
"""
from __future__ import annotations

import argparse
import gc
from time import perf_counter

import numpy as np
from numba import get_num_threads, set_num_threads

from platesim.tile3d import operators
from platesim.tile3d.backend.mgcg_numba import MGCG


def problem(shape, trench=False):
    nx, ny, nz = shape
    periodic = (True, True, False)
    dx = 4e-4 * 48 / nx if trench else 1 / nx
    if trench:
        # FieldSolver Hs/active/top_scale and potential Newton q=w*af*I.
        from platesim.core.constants import F
        from platesim.tile3d.backend import ls_numpy
        x = (np.arange(nx) + .5) * dx - nx * dx / 2
        z = (np.arange(nz) + .5 - .55 * nz) * dx
        psi = np.broadcast_to(np.maximum(z[None, None, :], np.minimum(
            nx * dx * .19 - abs(x[:, None, None]), z[None, None, :] + nz * dx * .42)), shape).copy()
        eps = 1.2 * dx
        v = np.clip(psi / eps, -30, 30)
        Hs = np.maximum(.5 * (1 + np.tanh(v)), 1e-6)
        active = psi > -3.5 * eps
        gm = np.sqrt(sum(((ls_numpy.shifted(psi, 1, a, periodic)
                           - ls_numpy.shifted(psi, -1, a, periodic)) / (2*dx))**2 for a in range(3)))
        w = np.where(active, .5 / eps / np.cosh(v)**2 * gm, 0)
        faces = []
        for axis in range(3):
            h = np.moveaxis(Hs, axis, 0)
            f = np.zeros((h.shape[0]+1, *h.shape[1:]))
            f[1:-1] = .5 * (h[:-1] + h[1:])
            if periodic[axis]:
                f[0] = f[-1] = .5 * (h[0] + h[-1])
            faces.append(np.moveaxis(f, 0, axis).copy())
        faces[2][:, :, -1] = Hs[:, :, -1] / 1.6  # d_top = .8 dx
        faces = [.065 * f for f in faces]
        af = .5 * F / (8.314 * 295.15)
        I = .002 * np.exp(af * .12)
        reaction = w * af * I
        b = -w * I
    else:
        t = (np.arange(nx) + .5) / nx
        k = np.broadcast_to(1 + .2 * np.sin(2*np.pi*t[:, None, None])
                            * np.cos(2*np.pi*t[None, :, None]), shape).copy()
        faces = operators.face_coefficients(k, periodic)
        active, reaction = None, .1
        b = 1 + np.random.default_rng(56).random(shape)
    args = dict(reaction=reaction, active=active, dirichlet={"z_hi": 0}, periodic=periodic)
    return faces, dx, args, b


def run_case(name, shape, trench):
    from scipy.sparse.linalg import cg
    import pyamg

    faces, dx, args, b = problem(shape, trench)
    reference = MGCG(*faces, dx, **args)
    norm = np.linalg.norm(b.ravel())
    rows = []
    for dtype, tol in ((np.float64, 1e-8), (np.float32, 1e-5)):
        start = perf_counter()
        solver = MGCG(*faces, dx, dtype=dtype, **args)
        setup = perf_counter() - start
        x, info = solver.solve(b, tol=tol)
        residual = np.linalg.norm((b-reference.apply(x.astype(float))).ravel()) / norm
        rows.append((name, "MGCG", np.dtype(dtype).name, setup, info["seconds"],
                     info["iterations"], residual, info["converged"] and residual <= tol))
        del solver, x
        gc.collect()
    del reference
    gc.collect()
    for dtype, tol in ((np.float64, 1e-8), (np.float32, 1e-5)):
        start = perf_counter()
        A64, _ = operators.assemble(*faces, dx, **args)
        A = A64.astype(dtype)
        hierarchy = pyamg.smoothed_aggregation_solver(
            A, symmetry="symmetric", strength="symmetric",
            presmoother=("gauss_seidel", {"sweep": "symmetric"}),
            postsmoother=("gauss_seidel", {"sweep": "symmetric"}))
        setup = perf_counter() - start
        count = [0]
        start = perf_counter()
        x, status = cg(A, b.ravel().astype(dtype), rtol=tol, atol=0, maxiter=200,
                       M=hierarchy.aspreconditioner(),
                       callback=lambda _: count.__setitem__(0, count[0]+1))
        seconds = perf_counter()-start
        residual = np.linalg.norm(b.ravel()-A64 @ x.astype(float)) / norm
        rows.append((name, "SA-AMG", np.dtype(dtype).name, setup, seconds,
                     count[0], residual, status == 0 and residual <= tol))
        del hierarchy, A, A64, x
        gc.collect()
    for case, method, dtype, setup, seconds, it, res, ok in rows:
        print(f"{case:18} {method:7} {dtype:7} {setup:9.3f} {seconds:9.3f} "
              f"{setup+seconds:9.3f} {it:5d} {res:11.3e} {str(ok):5}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "trench", "smooth"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smooth-size", type=int, default=160)
    parser.add_argument("--trench-shape", nargs=3, type=int, default=(128, 16, 96))
    args = parser.parse_args()
    set_num_threads(min(args.threads, get_num_threads()))
    start = perf_counter()
    for dtype in (np.float64, np.float32):
        faces, dx, kw, b = problem((8, 8, 8))
        MGCG(*faces, dx, dtype=dtype, **kw).solve(b, tol=1e-5 if dtype == np.float32 else 1e-8)
    print(f"Numba threads={get_num_threads()}; JIT warmup={perf_counter()-start:.3f}s", flush=True)
    print("case               solver  storage   setup_s   solve_s   total_s iters    residual    ok", flush=True)
    if args.case in ("all", "trench"):
        shape = tuple(args.trench_shape)
        run_case("trench " + "x".join(map(str, shape)), shape, True)
    if args.case in ("all", "smooth"):
        run_case(f"smooth {args.smooth_size}^3", (args.smooth_size,)*3, False)


if __name__ == "__main__":
    main()
