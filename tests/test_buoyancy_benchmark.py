"""Natural-convection benchmark: differentially heated square cavity (de Vahl Davis 1983).

The MAC Stokes solver of :mod:`platesim.trenchflow.stokes` is driven by a Boussinesq body force and
coupled to a steady scalar transport solver written here, on the same grid and the same staggered
face velocities.  That is the same physics chain the feature engine uses for solutal buoyancy
(concentration -> body force -> Stokes flow), so the benchmark checks both the body-force term of
``solve_stokes`` and the sign/scale conventions the engine feeds it.

Problem
-------
Square cavity of side L, solid walls all round (bottom row and both side columns solid, the top row a
stationary lid because shear = 0), ``T = +1/2`` on the left (hot) wall and ``-1/2`` on the right (cold)
wall, adiabatic top and bottom.  Non-dimensionalised with L, the thermal diffusivity alpha and
Delta T = 1:

    u.grad u + grad p = Pr lap u + Ra Pr T e_z,     u.grad T = lap T,
    Ra = g beta Delta T L^3 / (nu alpha),           Pr = nu / alpha.

The solver integrates the dimensional Stokes/Oseen equation ``-(nu) lap u + grad p + (u.grad)u = f``
with ``nu = Pr``, ``alpha = 1`` and ``L = 1 cm`` (so ``u`` in cm/s is already the dimensionless
velocity in units of alpha/L), and the body force

    f_z = Ra * Pr * T        [cm/s2]

which is exactly ``g beta Delta T T`` for ``g beta Delta T = Ra nu alpha / L^3``.  Only Ra remains.

Nu is the average Nusselt number on the hot wall from the one-sided wall gradient
``Nu = -(dT/dx)|wall * L / Delta T = 2 n (1/2 - T_1)`` (T_1 = first fluid cell, wall half a cell
away, L = n dx), averaged over the wall.

Benchmark [FT]: de Vahl Davis, Int. J. Numer. Methods Fluids 3 (1983) 249, table I: Nu = 1.118 at
Ra = 1e3 and Nu = 2.243 at Ra = 1e4 (Pr = 0.71), u_max = 3.649 / 19.617 and v_max = 3.697 / 19.617 in
units of alpha/L.  At least one Oseen (inertia) pass per flow solve is required at Pr = 0.71: pure
Stokes flow (no inertia) gives Nu -> 1 at both Ra.

Numerics: first-order upwind advection of T on the MAC faces (conservative), central diffusion, Picard
iteration between flow and T.  Upwind numerical diffusion acts along the flow, i.e. along the walls,
so it does not thicken the thermal boundary layer (whose cross-stream Peclet number is small); the
remaining error is the O(1/n) wall-gradient/boundary-layer resolution error.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from platesim.trenchflow.geometry import CavityGrid
from platesim.trenchflow.stokes import solve_stokes

PR = 0.71                      # Prandtl number of air [FT]
ALPHA = 1.0                    # thermal diffusivity, cm2/s (unit of the non-dimensionalisation)
NU = PR                        # cm2/s, so that Pr = nu/alpha
NU_REF = {1e3: 1.118, 1e4: 2.243}          # de Vahl Davis 1983, table I
U_REF = {1e3: 3.649, 1e4: 19.617}
V_REF = {1e3: 3.697, 1e4: 19.617}


def _box(n):
    """n x n fluid cells inside solid walls: bottom row and both side columns solid, top row the lid.

    dx = L/n with L = 1 cm (dx_um = 1e4/n) so that lengths, nu and the body force share one unit
    system: mu = Pr, alpha = 1, f_z = Ra Pr T.
    """
    d = 1e4 / n
    nx, nz = n + 2, n + 1
    x = (np.arange(nx) + 0.5) * d
    z = (np.arange(nz) + 0.5) * d
    solid = np.zeros((nz, nx), bool)
    solid[0, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    return CavityGrid(dx_um=d, x_um=x, z_um=z, solid=solid, width_um=n * d, depth_um=0.0,
                      pitch_um=nx * d, height_um=nz * d)


def _t_step(g, T, ufx, ufz, alpha):
    """One steady solve of u.grad T = alpha lap T on the MAC faces, first-order upwind advection.

    ``alpha`` is the diffusivity in cell units (alpha_phys / dx): the advective face coefficient is the
    face velocity itself and the diffusive one alpha, so their ratio is the cell Peclet number.
    Dirichlet T = +-1/2 on the side walls (the face is half a cell away: 2 alpha), adiabatic top and
    bottom (both face fluxes are zero there because the walls are no-slip).
    """
    nz, nx = g.shape
    fluid = ~g.solid
    fk, fj = np.nonzero(fluid)
    idx = np.full((nz, nx), -1, np.int64)
    idx[fk, fj] = np.arange(fk.size)
    jj, kk = np.arange(nx), np.arange(nz)
    jp = (jj + 1) % nx
    # face connectivity masks and neighbour unknown index
    f_r, f_l = fluid & np.roll(fluid, -1, axis=1), fluid & np.roll(fluid, 1, axis=1)
    f_t, f_b = np.zeros((nz, nx), bool), np.zeros((nz, nx), bool)
    f_t[:-1] = fluid[:-1] & fluid[1:]
    f_b[1:] = fluid[1:] & fluid[:-1]
    nb_r, nb_l = np.roll(idx, -1, axis=1), np.roll(idx, 1, axis=1)
    nb_t, nb_b = np.full((nz, nx), -1, np.int64), np.full((nz, nx), -1, np.int64)
    nb_t[:-1], nb_b[1:] = idx[1:], idx[:-1]
    # outward face fluxes of every cell (unit face area), and the wall condition of that face
    faces = ((ufx[:, jp], f_r, nb_r, -0.5),          # right face -> cold wall
             (-ufx[:, jj], f_l, nb_l, 0.5),          # left face -> hot wall
             (ufz[1:], f_t, nb_t, None),             # top face (adiabatic)
             (-ufz[:nz], f_b, nb_b, None))           # bottom face (adiabatic)

    diag = np.zeros((nz, nx))
    rhs = np.zeros((nz, nx))
    rows, cols, vals = [], [], []
    for F, op, nb, tw in faces:
        F = np.where(fluid, F, 0.0)
        diag += np.maximum(F, 0.0)                       # outflow: upwind value is T_P
        diag += np.where(op, alpha, 0.0)                 # interior diffusive face
        coef = np.where(op, np.minimum(F, 0.0) - alpha, 0.0)     # inflow from Q + diffusion from Q
        m = op
        rows.append(idx[m])
        cols.append(nb[m])
        vals.append(coef[m])
        if tw is not None:                               # Dirichlet wall, half a cell away
            wall = fluid & ~op
            diag[wall] += 2.0 * alpha
            rhs[wall] += (2.0 * alpha - np.minimum(F[wall], 0.0)) * tw
    rows.append(idx[fluid])
    cols.append(idx[fluid])
    vals.append(diag[fluid])
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(fk.size, fk.size)).tocsr()
    out = np.zeros((nz, nx))
    out[fluid] = spsolve(A, rhs[fluid])
    return out


def _cavity(n, ra, itmax=400, tol=1e-7):
    """Picard iteration between the Stokes/Oseen flow and the steady temperature field.

    Returns (Nu_hot, u_max, v_max, iterations, seconds); u_max/v_max are in units of alpha/L.
    """
    g = _box(n)
    nz, nx = g.shape
    fluid = ~g.solid
    t0 = time.time()
    T = np.zeros((nz, nx))
    xc = (np.arange(1, nx - 1) + 0.5) * g.dx_um        # cell centres, 1 um = 1e-4 cm
    T[:, 1:nx - 1] = 0.5 - (xc - g.dx_um) / (n * g.dx_um)      # conduction profile between the walls
    fb = ra * PR / (n * g.dx_um * 1e-4) ** 3            # Ra Pr / L^3, f_z = fb * T
    zero = np.zeros((nz, nx))
    uprev, it = None, 0
    for it in range(itmax):
        flow = solve_stokes(g, 0.0, nu_cm2_s=NU, oseen_passes=3 if it else 0,
                            body_force=(zero, fb * T))
        Tn = _t_step(g, T, flow.info["u_face_x"], flow.info["u_face_z"], alpha=ALPHA / (g.dx_um * 1e-4))
        dT = float(np.abs(Tn - T)[fluid].max())
        u = np.concatenate([flow.ux[fluid], flow.uz[fluid]])
        du = 0.0 if uprev is None else float(np.abs(u - uprev).max() / max(np.abs(u).max(), 1e-300))
        T, uprev = Tn, u
        if dT < tol and du < tol:
            break
    h = 1.0 / n
    nu_hot = float(np.mean((0.5 - T[1:-1, 1]) / (0.5 * h)))
    return nu_hot, float(np.abs(flow.ux).max()), float(np.abs(flow.uz).max()), it, time.time() - t0


def _check(n, ra):
    """Run the coupled solve and compare Nu with the benchmark (within 5 %)."""
    nu_hot, umax, vmax, it, wall = _cavity(n, ra)
    nu_ref, u_ref, v_ref = NU_REF[ra], U_REF[ra], V_REF[ra]
    err = 100.0 * (nu_hot - nu_ref) / nu_ref
    print(f"\ncavity {n}x{n} Ra={ra:g} Pr={PR}: Nu_hot = {nu_hot:.4f} (de Vahl Davis {nu_ref:.3f}, "
          f"{err:+.2f} %), u_max = {umax:.3f} ({u_ref:.3f}), v_max = {vmax:.3f} ({v_ref:.3f}), "
          f"{it} Picard iterations, {wall:.1f} s")
    assert abs(err) <= 5.0, f"Nu = {nu_hot:.4f} is {err:+.1f} % off the benchmark {nu_ref}"
    return nu_hot


def test_natural_convection_ra_1e3():
    """Ra = 1e3, Pr = 0.71: steady conduction-dominated cavity, Nu = 1.118 [FT]."""
    _check(40, 1e3)


def test_natural_convection_ra_1e4():
    """Ra = 1e4, Pr = 0.71: boundary-layer regime, Nu = 2.243 [FT] (needs the Oseen inertia term)."""
    _check(40, 1e4)
