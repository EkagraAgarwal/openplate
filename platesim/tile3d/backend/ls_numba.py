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


@njit(inline="always")
def _axis_weno(p0, p1, p2, p3, p4, p5, p6, inv_dx, high_order, gp, gn):
    """Accumulate the up/down-wind gradient squares of one axis from seven nodes."""
    if high_order:
        d0 = (p1-p0)*inv_dx
        d1 = (p2-p1)*inv_dx
        d2 = (p3-p2)*inv_dx
        d3 = (p4-p3)*inv_dx
        d4 = (p5-p4)*inv_dx
        d5 = (p6-p5)*inv_dx
        dm = _left(d0, d1, d2, d3, d4)
        dp = -_left(-d5, -d4, -d3, -d2, -d1)
    else:
        dm = (p3-p2)*inv_dx
        dp = (p4-p3)*inv_dx
    return gp + max(dm, 0.0)**2 + min(dp, 0.0)**2, gn + min(dm, 0.0)**2 + max(dp, 0.0)**2


@njit(parallel=True)
def rhs(psi, speed, dx, periodic, high_order):
    # Seven explicit scalars per axis: allocating a 7-node workspace inside the
    # cell loop costs more than the whole WENO stencil on thin tiles.
    nx, ny, nz = psi.shape
    inv_dx = 1.0/dx
    out = np.empty_like(psi)
    for flat in prange(nx*ny*nz):
        i, j, k = flat//(ny*nz), (flat//nz) % ny, flat % nz
        gp, gn = 0.0, 0.0
        if periodic[0]:
            i0, i1, i2, i3, i4, i5, i6 = (_index(i-3, nx, True), _index(i-2, nx, True),
                                          _index(i-1, nx, True), i, _index(i+1, nx, True),
                                          _index(i+2, nx, True), _index(i+3, nx, True))
        else:
            i0, i1, i2, i3, i4, i5, i6 = (_index(i-3, nx, False), _index(i-2, nx, False),
                                          _index(i-1, nx, False), i, _index(i+1, nx, False),
                                          _index(i+2, nx, False), _index(i+3, nx, False))
        gp, gn = _axis_weno(psi[i0, j, k], psi[i1, j, k], psi[i2, j, k], psi[i3, j, k],
                            psi[i4, j, k], psi[i5, j, k], psi[i6, j, k], inv_dx, high_order, gp, gn)
        if periodic[1]:
            j0, j1, j2, j3, j4, j5, j6 = (_index(j-3, ny, True), _index(j-2, ny, True),
                                          _index(j-1, ny, True), j, _index(j+1, ny, True),
                                          _index(j+2, ny, True), _index(j+3, ny, True))
        else:
            j0, j1, j2, j3, j4, j5, j6 = (_index(j-3, ny, False), _index(j-2, ny, False),
                                          _index(j-1, ny, False), j, _index(j+1, ny, False),
                                          _index(j+2, ny, False), _index(j+3, ny, False))
        gp, gn = _axis_weno(psi[i, j0, k], psi[i, j1, k], psi[i, j2, k], psi[i, j3, k],
                            psi[i, j4, k], psi[i, j5, k], psi[i, j6, k], inv_dx, high_order, gp, gn)
        if periodic[2]:
            k0, k1, k2, k3, k4, k5, k6 = (_index(k-3, nz, True), _index(k-2, nz, True),
                                          _index(k-1, nz, True), k, _index(k+1, nz, True),
                                          _index(k+2, nz, True), _index(k+3, nz, True))
        else:
            k0, k1, k2, k3, k4, k5, k6 = (_index(k-3, nz, False), _index(k-2, nz, False),
                                          _index(k-1, nz, False), k, _index(k+1, nz, False),
                                          _index(k+2, nz, False), _index(k+3, nz, False))
        gp, gn = _axis_weno(psi[i, j, k0], psi[i, j, k1], psi[i, j, k2], psi[i, j, k3],
                            psi[i, j, k4], psi[i, j, k5], psi[i, j, k6], inv_dx, high_order, gp, gn)
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
