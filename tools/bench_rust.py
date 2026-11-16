"""Isolated lane-E benchmark. Run from repo root; see rust/README.md.

Kernels below are copies of mgcg_numba._kernel and ls_numba.rhs, compiled
both serially and with parallel=True; never change engine dispatch.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

os.environ.setdefault("RAYON_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("rust/numba-cache").resolve()))

import numpy as np
import numba
from numba import njit, prange
import platesim_rs as rs
from platesim.tile3d.backend import ls_numpy, ls_numba, mgcg_numba as mg


def smooth_kernel(u, b, fx, fy, fz, base, diag, periodic, out, color, omega):
    nx, ny, nz = u.shape
    for ij in prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nz):
            if color >= 0 and (i + j + k) % 2 != color:
                continue
            v = np.float64(u[i, j, k])
            a = np.float64(base[i, j, k]) * v
            if i > 0:
                a += fx[i, j, k] * (v - u[i-1, j, k])
            elif periodic[0] and nx > 1:
                a += fx[0, j, k] * (v - u[nx-1, j, k])
            if i + 1 < nx:
                a += fx[i+1, j, k] * (v - u[i+1, j, k])
            elif periodic[0] and nx > 1:
                a += fx[0, j, k] * (v - u[0, j, k])
            if j > 0:
                a += fy[i, j, k] * (v - u[i, j-1, k])
            elif periodic[1] and ny > 1:
                a += fy[i, 0, k] * (v - u[i, ny-1, k])
            if j + 1 < ny:
                a += fy[i, j+1, k] * (v - u[i, j+1, k])
            elif periodic[1] and ny > 1:
                a += fy[i, 0, k] * (v - u[i, 0, k])
            if k > 0:
                a += fz[i, j, k] * (v - u[i, j, k-1])
            elif periodic[2] and nz > 1:
                a += fz[i, j, 0] * (v - u[i, j, nz-1])
            if k + 1 < nz:
                a += fz[i, j, k+1] * (v - u[i, j, k+1])
            elif periodic[2] and nz > 1:
                a += fz[i, j, 0] * (v - u[i, j, 0])
            if omega == 0:
                out[i, j, k] = a
            else:
                out[i, j, k] = v + omega * (b[i, j, k] - a) / diag[i, j, k]


_left = njit(inline="always")(ls_numpy.weno_left)


@njit(inline="always")
def _index(i, n, periodic):
    if periodic:
        return i % n
    return min(max(i, 0), n-1)


def weno_kernel(psi, speed, dx, periodic, high_order):
    nx, ny, nz = psi.shape
    out = np.empty_like(psi)
    for flat in prange(nx*ny*nz):
        i, j, k = flat//(ny*nz), (flat//nz) % ny, flat % nz
        gp, gn = 0.0, 0.0
        for axis in range(3):
            nodes = np.empty(7, dtype=np.float64)
            for q in range(7):
                o = q-3
                ii = _index(i+o, nx, periodic[0]) if axis == 0 else i
                jj = _index(j+o, ny, periodic[1]) if axis == 1 else j
                kk = _index(k+o, nz, periodic[2]) if axis == 2 else k
                nodes[q] = psi[ii, jj, kk]
            if high_order:
                d0 = (nodes[1]-nodes[0])/dx
                d1 = (nodes[2]-nodes[1])/dx
                d2 = (nodes[3]-nodes[2])/dx
                d3 = (nodes[4]-nodes[3])/dx
                d4 = (nodes[5]-nodes[4])/dx
                d5 = (nodes[6]-nodes[5])/dx
                dm = _left(d0, d1, d2, d3, d4)
                dp = -_left(-d5, -d4, -d3, -d2, -d1)
            else:
                dm = (nodes[3]-nodes[2])/dx
                dp = (nodes[4]-nodes[3])/dx
            gp += max(dm, 0.0)**2 + min(dp, 0.0)**2
            gn += min(dm, 0.0)**2 + max(dp, 0.0)**2
        v = speed[i, j, k]
        out[i, j, k] = max(v, 0.0)*np.sqrt(gp) + min(v, 0.0)*np.sqrt(gn)
    return out


SMOOTH = {"numba-serial": njit(smooth_kernel), "numba-parallel": njit(parallel=True)(smooth_kernel)}
WENO = {"numba-serial": njit(weno_kernel), "numba-parallel": njit(parallel=True)(weno_kernel)}


def level(shape, periodic, seed=42):
    rng = np.random.default_rng(seed)
    faces = []
    for axis in range(3):
        dims = list(shape)
        dims[axis] += 1
        face = rng.uniform(0.1, 2.0, dims)
        v = np.moveaxis(face, axis, 0)
        if periodic[axis]:
            v[-1] = v[0]
        else:
            v[0] = v[-1] = 0
        faces.append(face)
    base = rng.uniform(0.2, 1.0, shape)
    diag = base.copy()
    for axis, face in enumerate(faces):
        v, d = np.moveaxis(face, axis, 0), np.moveaxis(diag, axis, 0)
        if shape[axis] > 1:
            d[:] += v[:-1] + v[1:]
    # Representative masked identity rows: remove incident conductances.
    mask = rng.random(shape) > 0.08
    for axis, face in enumerate(faces):
        v, m = np.moveaxis(face, axis, 0), np.moveaxis(mask, axis, 0)
        v[1:-1] *= m[:-1] & m[1:]
        if periodic[axis]:
            v[0] *= m[0] & m[-1]
            v[-1] = v[0]
    base[~mask] = 1
    diag[:] = base
    for axis, face in enumerate(faces):
        if shape[axis] > 1:
            np.moveaxis(diag, axis, 0)[:] += np.moveaxis(face, axis, 0)[:-1] + np.moveaxis(face, axis, 0)[1:]
    return SimpleNamespace(x=rng.normal(size=shape), rhs=rng.normal(size=shape),
                           faces=tuple(faces), base=base, diag=diag, work=np.empty(shape),
                           rbgs=not any(p and n % 2 for p, n in zip(periodic, shape)))


def smooth_call(name, l, periodic, sweeps=2, reverse=False):
    if name.startswith("rust"):
        rs.smooth(l.x, l.rhs, *l.faces, l.base, l.diag, l.work, periodic,
                  sweeps, l.rbgs, reverse, name == "rust-rayon")
    elif name == "numba-engine-1":
        mg.MGCG._smooth(SimpleNamespace(periodic=periodic), l, sweeps, reverse)  # type: ignore[arg-type]
    else:
        if any(p and n % 2 for p, n in zip(periodic, l.x.shape)):
            sweeps *= 2
        kernel = SMOOTH[name]
        for _ in range(sweeps):
            if l.rbgs:
                for color in ((1, 0) if reverse else (0, 1)):
                    kernel(l.x, l.rhs, *l.faces, l.base, l.diag, periodic, l.x, color, 1.0)
            else:
                kernel(l.x, l.rhs, *l.faces, l.base, l.diag, periodic, l.work, -1, 2/3)
                l.x[:] = l.work
    return l.x


def weno_call(name, psi, speed, dx, periodic):
    if name.startswith("rust"):
        return rs.rhs(psi, speed, dx, periodic, name == "rust-rayon")
    if name == "numpy":
        return ls_numpy.rhs(psi, speed, dx, periodic, True)
    if name == "numba-engine-1":
        return ls_numba.rhs(psi, speed, dx, periodic, True)
    return WENO[name](psi, speed, dx, periodic, True)


def assert_equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    return float(np.max(np.abs(actual-expected)))


def check_contracts():
    """Small boundary, singleton, odd-periodic, reverse, Jacobi and zero-sweep cases."""
    numba.set_num_threads(4)
    for shape in [(8, 4, 6), (7, 4, 5), (1, 2, 1), (2, 1, 3)]:
        for periodic in [(False, False, False), (True, True, False), (True, True, True)]:
            for rbgs in [False, True]:
                l = level(shape, periodic)
                l.rbgs &= rbgs
                initial = l.x.copy()
                for sweeps, reverse in [(0, False), (1, False), (3, True)]:
                    l.x[:] = initial
                    expected = smooth_call("numba-engine-1", l, periodic, sweeps, reverse).copy()
                    for name in [*SMOOTH, "rust-single", "rust-rayon"]:
                        l.x[:] = initial
                        assert_equal(smooth_call(name, l, periodic, sweeps, reverse), expected)
            rng = np.random.default_rng(5)
            # physical dx and signed-distance-scale psi, plus nonsmooth noise
            for psi in [rng.normal(size=shape)*8e-4, np.zeros(shape), np.ones(shape)*8e-4]:
                speed = rng.normal(size=shape)*2e-5
                expected = weno_call("numpy", psi, speed, 8e-4, periodic)
                for name in [*WENO, "numba-engine-1", "rust-single", "rust-rayon"]:
                    assert_equal(weno_call(name, psi, speed, 8e-4, periodic), expected)
    # Explicit layout and shape rejection; no silent copy at the Rust boundary.
    a = np.ones((4, 4, 4))
    for bad in [a[:, :, ::2], a[:2]]:
        try:
            rs.rhs(a, bad, 1.0, (False,)*3)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid layout/shape accepted")


def measure(call, reset, repeats, inner):
    reset()
    call()  # all JIT compilation and initial pool startup excluded
    samples = []
    for _ in range(repeats):
        reset()  # outside timed region; identical trajectory per batch
        start = perf_counter()
        for _ in range(inner):
            call()
        samples.append((perf_counter()-start)*1000/inner)
    return {"ms": min(samples), "samples_ms": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--output", default="rust/results.json")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.repeats < 5 or args.inner < 1 or args.threads < 1:
        p.error("need repeats>=5, inner>=1, threads>=1")
    if rs.threads() != args.threads:
        p.error("set RAYON_NUM_THREADS to --threads before starting Python")
    check_contracts()
    print("Contract checks passed", flush=True)
    if args.check_only:
        return
    rows = []
    names = ["numba-serial", "numba-engine-1", "numba-parallel", "rust-single", "rust-rayon"]
    for shape in [(590, 4, 48), (142, 4, 47), (128, 128, 64)]:
        periodic = (True, True, False)
        l = level(shape, periodic)
        initial = l.x.copy()
        rng = np.random.default_rng(13)
        grid = np.indices(shape)
        dx = 8e-4  # 8 um in cm
        psi = dx*(0.3*grid[0]+0.1*grid[1]+0.5*grid[2]+0.2*rng.normal(size=shape))
        speed = rng.uniform(-2e-5, 3e-5, shape)
        for kernel, variants in [("smooth", names), ("weno", ["numpy", *names])]:
            if kernel == "smooth":
                numba.set_num_threads(1)
                l.x[:] = initial
                expected = smooth_call("numba-engine-1", l, periodic).copy()
            else:
                expected = weno_call("numpy", psi, speed, dx, periodic)
            for name in variants:
                numba.set_num_threads(1 if name == "numba-engine-1" else args.threads)
                if kernel == "smooth":
                    def smooth_reset():
                        l.x[:] = initial
                    def smooth_run():
                        return smooth_call(name, l, periodic)
                    reset, call = smooth_reset, smooth_run
                else:
                    def weno_run():
                        return weno_call(name, psi, speed, dx, periodic)
                    reset, call = lambda: None, weno_run
                reset()
                error = assert_equal(call(), expected)
                row = dict(shape=shape, kernel=kernel, variant=name, max_abs_error=error,
                           **measure(call, reset, args.repeats, args.inner))
                rows.append(row)
                print(f"{shape!s:18} {kernel:6} {name:16} {row['ms']:10.4f} ms  err={error:.2g}", flush=True)
    metadata = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                    numpy=np.__version__, numba=numba.__version__, threads=args.threads,
                    rayon_threads=rs.threads(), repeats=args.repeats, inner=args.inner,
                    logical_cpus=os.cpu_count(), timing="min batch mean; warm; reset outside timer")
    Path(args.output).write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2)+"\n")


if __name__ == "__main__":
    main()
