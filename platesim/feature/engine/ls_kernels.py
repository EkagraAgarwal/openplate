"""High-order and narrow-band level-set kernels for the feature engine (plan v2 items B4 + C3).

Conventions are those of platesim.feature.engine.levelset:
  * psi = signed distance, > 0 in the electrolyte, < 0 in metal; cell-centred uniform grid, shape (nr, nz),
    axis 0 = r (via) or x (trench), axis 1 = z;
  * the interface obeys  psi_t + v |grad psi| = 0  with v the normal-speed extension field and the SAME sign
    convention as levelset.advect: v > 0 grows metal into the electrolyte, so psi decreases;
  * boundaries are levelset._pad's: edge replication in z, periodic wrap in x when periodic=True
    (geometry 'trench_full').  Here they are implemented by clamping (z) / wrapping (x) the stencil indices,
    which is exactly what _pad does to a 1-cell halo.

Two ingredients beyond the legacy first-order Godunov + forward-Euler operator of levelset.advect:

  * ``advect_weno5``: Jiang-Peng WENO5 upwind derivatives (6-point, 5th order) inside the same Godunov
    Hamiltonian split that levelset.advect uses, advanced with the third-order strong-stability-preserving
    TVD Runge-Kutta scheme (Shu-Osher).  Cuts the first-order smearing of a curving front and of the
    convex/concave corners at a feature mouth (C3).

  * narrow-band variants: the update is evaluated only at cells with |psi| < band_cells*dx, the remaining
    cells keep their value (B4).  For the RK3 scheme the update set is the band dilated by STENCIL cells per
    stage (3 stages), so the band result is *identical* to the full-grid result inside the band: every value
    a band cell needs at any stage is itself updated.  Values farther than 3*STENCIL cells outside the band
    are untouched, as advertised.

numba is an optional extra (``uv run --with numba``): every kernel has a pure-numpy implementation and the
jitted ones reproduce it to ~1e-15.  ``backend="auto"`` uses numba when it imports, else numpy.
"""
from __future__ import annotations

import types

import numpy as np
import scipy.ndimage as ndi

STENCIL = 3                        # WENO5 ghost cells per side (nodes i-3 .. i+3)
WENO_EPS = 1e-6                    # Jiang-Peng smoothness regulariser (psi/dx is O(1), so this is scale free)
_D1, _D2, _D3 = 0.1, 0.6, 0.3      # ideal weights of the three left-biased 5th-order candidates
DEFAULT_BAND_CELLS = 8.0           # == levelset.NARROW_CELLS (the FMM narrow band)
_OFF = (-3, -2, -1, 0, 1, 2, 3)    # stencil offsets of the 7 nodes a WENO5 derivative needs
_NB = None                         # lazy numba kernel cache; None = not tried, False = unavailable

BACKENDS = ("auto", "numpy", "numba")


# ------------------------------------------------------------------------------------------- small helpers
def _dx_periodic(dx_or_g, periodic=None):
    """Accept a Grid (duck-typed: .dx / .periodic_x) or a plain dx with an explicit periodic flag."""
    d = getattr(dx_or_g, "dx", None)
    if d is not None:
        per = getattr(dx_or_g, "periodic_x", False) if periodic is None else bool(periodic)
        return float(d), bool(per)
    return float(dx_or_g), bool(periodic)


def _pad(psi, k, periodic):
    """k-cell halo: edge (mirror) in z, periodic x wrap when the grid says so (levelset._pad, widened)."""
    p = np.pad(psi, k, mode="edge")
    if periodic:
        p[:k, k:-k] = psi[-k:, :]
        p[-k:, k:-k] = psi[:k, :]
    return p


def _ix(i, n, per):
    """Stencil index with _pad's boundary rule: wrap in x when periodic, else clamp (edge pad)."""
    if per:
        return i % n
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i


def band_index(psi, band_cells, dx):
    """Index lists (ii, jj) of the narrow band |psi| < band_cells*dx."""
    return np.nonzero(np.abs(psi) < float(band_cells) * dx)


def dilate_band(mask, k, periodic):
    """Grow a boolean mask by k cells (square stencil); wraps in x when periodic, clamped in z."""
    if k <= 0:
        return mask
    mode = ("wrap", "nearest") if periodic else "nearest"
    return ndi.maximum_filter(mask.astype(np.uint8), size=(2 * k + 1, 2 * k + 1), mode=mode) > 0


# ------------------------------------------------------------------------------------------- WENO5 stencil
def _weno_left(v1, v2, v3, v4, v5, eps=WENO_EPS):
    """Jiang-Peng WENO5 combination of the three left-biased derivative candidates.

    v1..v5 are consecutive differences (u_k - u_{k-1})/dx of the 6-node stencil i-3..i+2.  When the three
    smoothness indicators agree (smooth data) the weights are the ideal (0.1, 0.6, 0.3), which reproduces the
    5th-order one-sided derivative (-2, 15, -60, 20, 30, -3)/(60 dx) exactly; where a candidate straddles a
    kink its indicator grows and its weight collapses, so the operator stays essentially non-oscillatory
    across the interface kinks (front corners) instead of ringing like a plain central/one-sided stencil.

    The same function is called with numpy arrays (vectorised) and from the numba kernels with scalars.
    """
    s1 = 13.0 / 12.0 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    s2 = 13.0 / 12.0 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    s3 = 13.0 / 12.0 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2
    p1 = v1 / 3.0 - 7.0 / 6.0 * v2 + 11.0 / 6.0 * v3
    p2 = -v2 / 6.0 + 5.0 / 6.0 * v3 + v4 / 3.0
    p3 = v3 / 3.0 + 5.0 / 6.0 * v4 - v5 / 6.0
    a1 = _D1 / (eps + s1) ** 2
    a2 = _D2 / (eps + s2) ** 2
    a3 = _D3 / (eps + s3) ** 2
    return (a1 * p1 + a2 * p2 + a3 * p3) / (a1 + a2 + a3)


def _weno_dm_dp(nm3, nm2, nm1, n0, np1, np2, np3, ir):
    """(left-biased, right-biased) WENO5 derivatives at the centre node, from nodes x_{i-3} .. x_{i+3}.

    ir is the reciprocal grid spacing 1/dx (hoisted out of the per-cell loop: 10 divisions per cell would
    otherwise dominate a 2-D step).  The right-biased derivative is the mirrored left-biased one, i.e. the
    same reconstruction run on the reversed stencil with the sign flipped.
    """
    v1 = (nm2 - nm3) * ir
    v2 = (nm1 - nm2) * ir
    v3 = (n0 - nm1) * ir
    v4 = (np1 - n0) * ir
    v5 = (np2 - np1) * ir
    g1 = (np2 - np3) * ir
    g2 = (np1 - np2) * ir
    g3 = (n0 - np1) * ir
    g4 = (nm1 - n0) * ir
    g5 = (nm2 - nm1) * ir
    return _weno_left(v1, v2, v3, v4, v5), -_weno_left(g1, g2, g3, g4, g5)


def _ham(dmx, dpx, dmz, dpz, v):
    """Godunov Hamiltonian of psi_t + v|grad psi| = 0 -- levelset.advect's split, term for term."""
    gp = np.sqrt(np.maximum(dmx, 0.0) ** 2 + np.minimum(dpx, 0.0) ** 2
                 + np.maximum(dmz, 0.0) ** 2 + np.minimum(dpz, 0.0) ** 2)
    gn = np.sqrt(np.minimum(dmx, 0.0) ** 2 + np.maximum(dpx, 0.0) ** 2
                 + np.minimum(dmz, 0.0) ** 2 + np.maximum(dpz, 0.0) ** 2)
    return np.maximum(v, 0.0) * gp + np.minimum(v, 0.0) * gn


# ------------------------------------------------------------------------------------------- numpy kernels
def _weno5_rhs_full(psi, v, dx, per):
    """The Godunov Hamiltonian H(psi) on the whole grid (slice views of a 3-cell halo, no index lists).

    psi_t = -H, so the RK3 stages below subtract dt*H.
    """
    u = _pad(psi, STENCIL, per)
    nx, nz = u.shape
    ir = 1.0 / dx
    ax = [u[STENCIL + o:nx - STENCIL + o, STENCIL:-STENCIL] for o in _OFF]
    az = [u[STENCIL:-STENCIL, STENCIL + o:nz - STENCIL + o] for o in _OFF]
    dmx, dpx = _weno_dm_dp(*ax, ir)
    dmz, dpz = _weno_dm_dp(*az, ir)
    return _ham(dmx, dpx, dmz, dpz, v)


def _weno5_rhs_band(psi, v, dx, per, ii, jj):
    """H(psi) at the cells (ii, jj) only.

    The stencil is read through 1-D gathers on the flattened halo (base +- k cells, base +- k rows) instead of
    2-D advanced indexing, which is markedly cheaper for a thin band.
    """
    u = _pad(psi, STENCIL, per)
    nzp = u.shape[1]
    flat = u.reshape(-1)
    base = (ii + STENCIL) * nzp + (jj + STENCIL)
    ir = 1.0 / dx
    dmx, dpx = _weno_dm_dp(*[flat[base + o * nzp] for o in _OFF], ir)
    dmz, dpz = _weno_dm_dp(*[flat[base + o] for o in _OFF], ir)
    return _ham(dmx, dpx, dmz, dpz, v[ii, jj])


def _rk3_full(psi, v, dt, dx, per):
    """TVD-RK3 (Shu-Osher) on the whole grid: u1 = u0 + dt L(u0), u2 = 3/4 u0 + 1/4 (u1 + dt L(u1)),
    u = 1/3 u0 + 2/3 (u2 + dt L(u2)) with L = -H."""
    u1 = psi - dt * _weno5_rhs_full(psi, v, dx, per)
    u2 = 0.75 * psi + 0.25 * (u1 - dt * _weno5_rhs_full(u1, v, dx, per))
    return psi / 3.0 + (2.0 / 3.0) * (u2 - dt * _weno5_rhs_full(u2, v, dx, per))


def _rk3_band(psi, v, dt, dx, per, ii, jj):
    """TVD-RK3 restricted to the cells (ii, jj) == band dilated by 3*STENCIL; all other cells keep psi."""
    u0 = psi
    u1 = psi.copy()
    u2 = psi.copy()
    out = psi.copy()
    u1[ii, jj] = u0[ii, jj] - dt * _weno5_rhs_band(u0, v, dx, per, ii, jj)
    u2[ii, jj] = 0.75 * u0[ii, jj] + 0.25 * (u1[ii, jj] - dt * _weno5_rhs_band(u1, v, dx, per, ii, jj))
    out[ii, jj] = u0[ii, jj] / 3.0 + (2.0 / 3.0) * (u2[ii, jj] - dt * _weno5_rhs_band(u2, v, dx, per, ii, jj))
    return out


def _godunov1_band(psi, v, dt, dx, per, ii, jj):
    """levelset.advect's first-order Godunov + forward Euler step, evaluated only at (ii, jj).

    Same expression order as levelset.advect, so inside the band the numbers are bit-identical.
    """
    u = _pad(psi, 1, per)
    a, b = ii + 1, jj + 1
    c = u[a, b]
    dmx, dpx = (c - u[a - 1, b]) / dx, (u[a + 1, b] - c) / dx
    dmz, dpz = (c - u[a, b - 1]) / dx, (u[a, b + 1] - c) / dx
    out = psi.copy()
    out[ii, jj] = psi[ii, jj] - dt * _ham(dmx, dpx, dmz, dpz, v[ii, jj])
    return out


# ------------------------------------------------------------------------------------------- numba kernels
def _jitted(njit, fn, env=None):
    """Compile fn with numba, optionally re-binding its module globals.

    A plain Python helper called from nopython code cannot be resolved (numba reports "Untyped global name"),
    so a copy of the function is built whose globals map the helper to its compiled dispatcher.  This keeps
    one single source of the WENO/Hamiltonian formulas for both backends."""
    ns = dict(fn.__globals__)
    if env:
        ns.update(env)
    return njit(types.FunctionType(fn.__code__, ns, fn.__name__, fn.__defaults__, fn.__closure__))


def numba_available():
    """True when the optional numba extra imports (kernels are compiled on first use)."""
    return _numba_kernels() is not None


def _numba_kernels():
    """Compile (once per process) and return the jitted kernels, or None when numba is not importable."""
    global _NB
    if _NB is not None:
        return _NB or None
    try:
        from numba import njit
    except Exception:                                  # numba is an optional extra
        _NB = False
        return None
    dm_dp = _jitted(njit, _weno_dm_dp, {"_weno_left": _jitted(njit, _weno_left)})
    ham = _jitted(njit, _ham)
    ix = _jitted(njit, _ix)

    @njit(cache=False)
    def ham_cell(u, v, a, b, ir, per):
        nr, nz = u.shape
        im3, im2, im1 = ix(a - 3, nr, per), ix(a - 2, nr, per), ix(a - 1, nr, per)
        ip1, ip2, ip3 = ix(a + 1, nr, per), ix(a + 2, nr, per), ix(a + 3, nr, per)
        jm3, jm2, jm1 = ix(b - 3, nz, False), ix(b - 2, nz, False), ix(b - 1, nz, False)
        jp1, jp2, jp3 = ix(b + 1, nz, False), ix(b + 2, nz, False), ix(b + 3, nz, False)
        dmx, dpx = dm_dp(u[im3, b], u[im2, b], u[im1, b], u[a, b], u[ip1, b], u[ip2, b], u[ip3, b], ir)
        dmz, dpz = dm_dp(u[a, jm3], u[a, jm2], u[a, jm1], u[a, b], u[a, jp1], u[a, jp2], u[a, jp3], ir)
        return ham(dmx, dpx, dmz, dpz, v[a, b])

    @njit(cache=False)
    def weno5_band(u0, v, dt, dx, per, ii, jj):
        ir = 1.0 / dx
        u1 = u0.copy()
        u2 = u0.copy()
        out = u0.copy()
        for k in range(ii.size):
            a, b = ii[k], jj[k]
            u1[a, b] = u0[a, b] - dt * ham_cell(u0, v, a, b, ir, per)
        for k in range(ii.size):
            a, b = ii[k], jj[k]
            u2[a, b] = 0.75 * u0[a, b] + 0.25 * (u1[a, b] - dt * ham_cell(u1, v, a, b, ir, per))
        for k in range(ii.size):
            a, b = ii[k], jj[k]
            out[a, b] = u0[a, b] / 3.0 + (2.0 / 3.0) * (u2[a, b] - dt * ham_cell(u2, v, a, b, ir, per))
        return out

    @njit(cache=False)
    def weno5_full(u0, v, dt, dx, per):
        nr, nz = u0.shape
        ir = 1.0 / dx
        u1 = u0.copy()
        u2 = u0.copy()
        out = u0.copy()
        for a in range(nr):
            for b in range(nz):
                u1[a, b] = u0[a, b] - dt * ham_cell(u0, v, a, b, ir, per)
        for a in range(nr):
            for b in range(nz):
                u2[a, b] = 0.75 * u0[a, b] + 0.25 * (u1[a, b] - dt * ham_cell(u1, v, a, b, ir, per))
        for a in range(nr):
            for b in range(nz):
                out[a, b] = u0[a, b] / 3.0 + (2.0 / 3.0) * (u2[a, b] - dt * ham_cell(u2, v, a, b, ir, per))
        return out

    @njit(cache=False)
    def godunov1_band(u0, v, dt, dx, per, ii, jj):
        nr, nz = u0.shape
        out = u0.copy()
        for k in range(ii.size):
            a, b = ii[k], jj[k]
            am, ap = ix(a - 1, nr, per), ix(a + 1, nr, per)
            bm, bp = ix(b - 1, nz, False), ix(b + 1, nz, False)
            c = u0[a, b]
            dmx, dpx = (c - u0[am, b]) / dx, (u0[ap, b] - c) / dx
            dmz, dpz = (c - u0[a, bm]) / dx, (u0[a, bp] - c) / dx
            out[a, b] = c - dt * ham(dmx, dpx, dmz, dpz, v[a, b])
        return out

    _NB = dict(weno5=weno5_band, weno5_full=weno5_full, godunov1=godunov1_band)
    return _NB


# ------------------------------------------------------------------------------------------- public API
def _as_field(v, shape):
    """Broadcast the speed field to the grid shape.

    sim.py passes full (nr, nz) arrays; levelset.advect also accepts a (nr, 1) or scalar field through numpy
    broadcasting, and the numba kernels index v[a, b] directly, so every public kernel normalises first
    (a no-op view when the shape already matches)."""
    if np.shape(v) == shape:
        return v
    return np.ascontiguousarray(np.broadcast_to(v, shape))


def _use_numba(backend):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "numpy":
        return False
    kernels = _numba_kernels()
    if kernels is None:
        if backend == "numba":
            raise RuntimeError("backend='numba' requested but numba is not importable "
                               "(install the optional extra: uv run --with numba)")
        return False
    return True


def advect_weno5(psi, v, dt, dx, periodic=None, band_cells=None, backend="auto"):
    """One WENO5 + TVD-RK3 step of  psi_t + v |grad psi| = 0  (v > 0 grows metal, psi decreases).

    psi, v : (nr, nz) signed distance and normal-speed field (same conventions as levelset.advect)
    dx     : grid spacing (a Grid is accepted too, as in levelset.advect)
    band_cells : None -> whole grid; a number -> update only |psi| < band_cells*dx (cells outside keep psi;
             the update set is dilated by 3*STENCIL cells so the band matches the full grid inside the band)
    backend : 'auto' (numba when importable) | 'numpy' | 'numba'

    Unlike the first-order operator this is *not* a monotone single step, so the CFL limit is tighter in
    practice: keep dt*max|v| <= cfl*dx with cfl <= 0.5 (sim.py already does).
    """
    dx, per = _dx_periodic(dx, periodic)
    v = _as_field(v, psi.shape)
    nb = _use_numba(backend)
    if band_cells is None:
        if nb:
            return _numba_kernels()["weno5_full"](psi, v, dt, dx, per)
        return _rk3_full(psi, v, dt, dx, per)
    upd = dilate_band(np.abs(psi) < float(band_cells) * dx, 3 * STENCIL, per)
    ii, jj = np.nonzero(upd)
    if ii.size == 0:
        return psi.copy()
    if nb:
        return _numba_kernels()["weno5"](psi, v, dt, dx, per, ii, jj)
    return _rk3_band(psi, v, dt, dx, per, ii, jj)


def advect_godunov1_band(psi, v, dt, dx, periodic=None, band_cells=DEFAULT_BAND_CELLS, backend="auto"):
    """levelset.advect's first-order Godunov + forward-Euler step restricted to |psi| < band_cells*dx.

    Inside the band the numbers are bit-identical to the full-grid step (same expressions, same inputs);
    outside the band psi is returned unchanged.
    """
    dx, per = _dx_periodic(dx, periodic)
    v = _as_field(v, psi.shape)
    ii, jj = band_index(psi, band_cells, dx)
    if ii.size == 0:
        return psi.copy()
    if _use_numba(backend):
        return _numba_kernels()["godunov1"](psi, v, dt, dx, per, ii, jj)
    return _godunov1_band(psi, v, dt, dx, per, ii, jj)
