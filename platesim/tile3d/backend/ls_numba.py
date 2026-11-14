"""Parallel Numba CPU kernels, same public signatures as ls_numpy.

Import this optional module only after backend selection. Compilation is lazy;
no on-disk cache is written into the shared checkout.
"""
import numpy as np
from numba import njit, prange

from .ls_numpy import finish, weno_left

_left = njit(inline="always")(weno_left)


@njit(inline="always")
def _index(i, n, periodic):
    if periodic:
        return i % n
    return min(max(i, 0), n-1)


@njit(parallel=True)
def rhs(psi, speed, dx, periodic, high_order):
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


@njit(parallel=True)
def _stage(base, current, h, dt, a, b):
    out = np.empty_like(base)
    for i in prange(base.size):
        out.flat[i] = a*base.flat[i] + b*(current.flat[i]-dt*h.flat[i])
    return out


def advect_godunov1(psi, speed, dt, dx, periodic, band_cells=None):
    speed = np.ascontiguousarray(np.broadcast_to(speed, psi.shape), dtype=np.float64)
    out = _stage(psi, psi, rhs(psi, speed, dx, periodic, False), dt, 0.0, 1.0)
    return finish(psi, out, dx, band_cells)


def advect_weno5(psi, speed, dt, dx, periodic, band_cells=None):
    speed = np.ascontiguousarray(np.broadcast_to(speed, psi.shape), dtype=np.float64)
    u1 = _stage(psi, psi, rhs(psi, speed, dx, periodic, True), dt, 0.0, 1.0)
    u2 = _stage(psi, u1, rhs(u1, speed, dx, periodic, True), dt, 0.75, 0.25)
    out = _stage(psi, u2, rhs(u2, speed, dx, periodic, True), dt, 1/3, 2/3)
    return finish(psi, out, dx, band_cells)
