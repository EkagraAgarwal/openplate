"""MGCG field-pattern benchmark; run only after device parity tests pass.

PYTHONPATH=. uv run --no-project --python 3.12 --with numpy --with scipy \
  --with numba --with pyopencl python tools/bench_mgcg_gpu.py

Use --sizes 48x56x48 --repeats 2 --skip256 for a short smoke run.
No sparse fine-grid matrix is assembled. No result contains machine paths.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import set_num_threads

from platesim.tile3d import operators
from platesim.tile3d.backend.mgcg_numba import MGCG
from platesim.tile3d.backend.mgcg_opencl import MGCG_CL, available_devices


def trench(shape):
    """test_tile3d_mgcg.trench, without its expensive sparse reference matrix."""
    from platesim.core.constants import F
    from platesim.tile3d.backend import ls_numpy
    periodic = (True, True, False)
    dx = 4e-4 * 48 / shape[0]
    nx, _ny, nz = shape
    x = (np.arange(nx) + .5)*dx - nx*dx/2
    z0 = -.55*nz*dx
    zc = z0 + (np.arange(nz)+.5)*dx
    psi = np.broadcast_to(np.maximum(zc[None, None, :], np.minimum(
        nx*dx*.19-abs(x[:, None, None]), zc[None, None, :]+nz*dx*.42)), shape).copy()
    eps = 1.2*dx
    v = np.clip(psi/eps, -30, 30)
    Hs = np.maximum(.5*(1+np.tanh(v)), 1e-6)
    gm = np.sqrt(sum(((ls_numpy.shifted(psi, 1, a, periodic)
                      - ls_numpy.shifted(psi, -1, a, periodic))/(2*dx))**2 for a in range(3)))
    active = psi > -3.5*eps
    w = np.where(active, .5/eps/np.cosh(v)**2*gm, 0)
    faces = []
    for axis in range(3):
        h = np.moveaxis(Hs, axis, 0)
        f = np.zeros((h.shape[0]+1, *h.shape[1:]))
        f[1:-1] = .5*(h[:-1]+h[1:])
        if periodic[axis]:
            f[0] = f[-1] = .5*(h[0]+h[-1])
        faces.append(np.moveaxis(f, 0, axis).copy())
    faces[2][:, :, -1] = Hs[:, :, -1]/1.6
    faces = [.065*f for f in faces]
    af = .5*F/(8.314*295.15)
    I = .002*np.exp(af*.12)
    args = {"reaction": w*af*I, "active": active, "dirichlet": {"z_hi": 0.}, "periodic": periodic}
    return faces, dx, args, (-w*I).ravel()


def smooth(shape):
    n = shape[0]
    t = (np.arange(n)+.5)/n
    k = np.broadcast_to(1+.2*np.sin(2*np.pi*t[:, None, None])
                        * np.cos(2*np.pi*t[None, :, None]), shape).copy()
    faces = operators.face_coefficients(k, (True, True, False))
    return faces, 1/n, {"reaction": .1, "dirichlet": {"z_hi": 0}}, np.ones(shape)


def measure(cls, faces, dx, args, b, repeats, **extra):
    start = perf_counter()
    solver = cls(*faces, dx, **args, **extra)
    setup = perf_counter()-start
    # Warm JIT/device helper kernels; exclude one untimed solve for all backends.
    _, warm = solver.solve(b, tol=1e-8)
    if not warm["converged"]:
        raise RuntimeError(f"warm solve failed: {warm}")
    start = perf_counter()
    _, info = solver.solve(b, tol=1e-8)
    elapsed = perf_counter()-start
    if not info["converged"]:
        raise RuntimeError(f"solve failed: {info}")
    reaction = args["reaction"]
    start = perf_counter()
    counts = []
    for i in range(repeats):
        solver.set_reaction(reaction*(1+.01*(i % 5)))
        _, repeated = solver.solve(b, tol=1e-8)
        if not repeated["converged"]:
            raise RuntimeError(f"repeated solve failed: {repeated}")
        counts.append(repeated["iterations"])
    repeat_seconds = perf_counter()-start
    return setup, elapsed, info["iterations"], repeat_seconds, f"{min(counts)}–{max(counts)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["48x56x48", "128x128x96", "192x192x128"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--skip256", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("bench_mgcg_gpu_results.md"))
    opts = parser.parse_args()
    if opts.repeats < 1:
        parser.error("--repeats must be positive")
    set_num_threads(opts.threads)
    devices = available_devices()
    names = [d.name for d in devices]
    lines = ["# GPU-resident MGCG benchmark", "",
             "Tolerance: 1e-8 true relative residual; zero initial guess for every solve.",
             f"CPU: Numba, {opts.threads} threads. Devices: {', '.join(names)}.",
             "Setup includes hierarchy, coarse factor/inverse, and OpenCL context/program construction.",
             "Solve timings exclude one warm-up solve; repeated timing includes set_reaction and all transfers.",
             "The reaction cycles through 1.00–1.04 times the initial field; no warm-starting.", "",
             f"| Field | Shape | Backend | Setup s | Solve s | Iterations | {opts.repeats} updates + solves s | Repeat iterations |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    opts.output.write_text("\n".join(lines)+"\n", encoding="utf-8")
    # Warm CPU compilation independently of measured setup/solve.
    f, dx, a, b = trench((12, 8, 12))
    MGCG(*f, dx, **a).solve(b)
    cases = [("trench", tuple(map(int, s.split("x")))) for s in opts.sizes]
    if not opts.skip256:
        if any("gfx1102" in d.name and d.global_mem_size >= 6*1024**3 for d in devices):
            cases.append(("smooth", (256, 256, 256)))
        else:
            lines.insert(7, "256^3 omitted: discrete GPU reports less than 6 GiB available device capacity.")
    for case_index, (field, shape) in enumerate(cases):
        faces, dx, args, b = (trench if field == "trench" else smooth)(shape)
        backends = [("Numba f64", MGCG, {})]
        for name in ("gfx1102", "gfx1035"):
            if name == "gfx1035" and case_index != 0:
                continue
            if any(name in n for n in names):
                backends.extend((f"{name} {dtype}", MGCG_CL, {"device": name, "dtype": dtype})
                                for dtype in ("float64", "mixed"))
        for label, cls, extra in backends:
            gc.collect()
            print(f"Measuring {field} {shape} {label}", flush=True)
            try:
                setup, solve, count, repeated, counts = measure(cls, faces, dx, args, b, opts.repeats, **extra)
                row = f"| {field} | {'x'.join(map(str, shape))} | {label} | {setup:.4f} | {solve:.4f} | {count} | {repeated:.4f} | {counts} |"
            except Exception as exc:  # noqa: BLE001 - retain a result row for every backend failure
                # Record allocation/build/convergence failures without losing
                # completed rows (notably the optional 256^3 memory probe).
                row = f"| {field} | {'x'.join(map(str, shape))} | {label} | failed: {type(exc).__name__}: {exc} | — | — | — | — |"
            lines.append(row)
            print(row, flush=True)
            opts.output.write_text("\n".join(lines)+"\n", encoding="utf-8")
        del faces, args, b
        gc.collect()
    print(f"Wrote {opts.output.name}", flush=True)


if __name__ == "__main__":
    main()
