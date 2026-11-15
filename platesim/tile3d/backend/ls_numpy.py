"""3-D Hamilton-Jacobi kernels, cm and s; positive speed grows solid.

WENO5 reconstructs one-sided derivatives, with the 2-D engine's Godunov
Hamiltonian and Shu-Osher TVD-RK3. No fast-math approximations are used.
"""
import numpy as np


def shifted(a, offset, axis, periodic):
    """a[i+offset] with wrap or zero-gradient edge padding."""
    indices = np.arange(a.shape[axis]) + offset
    if periodic[axis]:
        indices %= a.shape[axis]
    else:
        indices = np.clip(indices, 0, a.shape[axis]-1)
    return np.take(a, indices, axis=axis)


def weno_left(v1, v2, v3, v4, v5):
    """Jiang-Peng reconstruction, scalar or array; derivatives are dimensionless."""
    s1 = 13/12*(v1-2*v2+v3)**2 + 0.25*(v1-4*v2+3*v3)**2
    s2 = 13/12*(v2-2*v3+v4)**2 + 0.25*(v2-v4)**2
    s3 = 13/12*(v3-2*v4+v5)**2 + 0.25*(3*v3-4*v4+v5)**2
    a1, a2, a3 = 0.1/(1e-6+s1)**2, 0.6/(1e-6+s2)**2, 0.3/(1e-6+s3)**2
    p1 = v1/3 - 7/6*v2 + 11/6*v3
    p2 = -v2/6 + 5/6*v3 + v4/3
    p3 = v3/3 + 5/6*v4 - v5/6
    return (a1*p1+a2*p2+a3*p3)/(a1+a2+a3)


def rhs(psi, speed, dx, periodic, high_order):
    """Positive Hamiltonian; psi_t = -H. Speed may vary in sign."""
    gp, gn = np.zeros_like(psi), np.zeros_like(psi)
    for axis in range(3):
        if high_order:
            nodes = [shifted(psi, o, axis, periodic) for o in range(-3, 4)]
            diffs = [(nodes[j+1]-nodes[j])/dx for j in range(6)]
            dm = weno_left(*diffs[:5])
            dp = -weno_left(*[-d for d in diffs[:0:-1]])
        else:
            dm = (psi-shifted(psi, -1, axis, periodic))/dx
            dp = (shifted(psi, 1, axis, periodic)-psi)/dx
        gp += np.maximum(dm, 0)**2 + np.minimum(dp, 0)**2
        gn += np.minimum(dm, 0)**2 + np.maximum(dp, 0)**2
    return np.maximum(speed, 0)*np.sqrt(gp) + np.minimum(speed, 0)*np.sqrt(gn)


def finish(psi, out, dx, band_cells):
    """Only the original strict band is returned updated (contract section 4)."""
    if band_cells is None:
        return out
    return np.where(np.abs(psi) < band_cells*dx, out, psi)


def advect_godunov1(psi, speed, dt, dx, periodic, band_cells=None):
    """First-order Godunov plus forward Euler, with an optional strict band."""
    return finish(psi, psi-dt*rhs(psi, speed, dx, periodic, False), dx, band_cells)


def advect_weno5(psi, speed, dt, dx, periodic, band_cells=None):
    """WENO5/TVD-RK3; full intermediate stages preserve all band dependencies."""
    u1 = psi-dt*rhs(psi, speed, dx, periodic, True)
    u2 = 0.75*psi + 0.25*(u1-dt*rhs(u1, speed, dx, periodic, True))
    out = psi/3 + (2/3)*(u2-dt*rhs(u2, speed, dx, periodic, True))
    return finish(psi, out, dx, band_cells)
