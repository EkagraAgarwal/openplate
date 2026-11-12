"""Brownian-dynamics particle (Lagrangian tracer) solver for Cu2+ transport into a trench.

WHAT THIS IS -- AND WHAT IT IS NOT
----------------------------------
This is a *tracer* solver of the same advection-diffusion-reaction problem that the continuum solver
``platesim.trenchflow.transport`` solves.  It is **not** a molecular-dynamics simulation of individual
ions: every computational particle stands for a large number of Cu2+ ions, so the particle cloud is a
stochastic sample of the *continuum* concentration field c(x, z) = n_particles / n_ref, and the measured
absorption rate at the copper surface is the surface flux density.  The value of the method is that it
is a completely independent discretisation of the same PDE (no mesh, no upwinding), so agreement with
the continuum solver is a genuine cross-check, and it gives particle paths for visualisation.

MODEL (one explicit time step dt, lengths in um, velocities in cm/s)
    x  <- x + ux(x,z)*dt*1e4 + sqrt(2*D*dt)*1e4 * N(0,1)
    z  <- z + uz(x,z)*dt*1e4 + sqrt(2*D*dt)*1e4 * N(0,1)
``ux``, ``uz`` come from the flow field (``flow=None`` -> pure diffusion).  With
``interp="bilinear"`` the velocity at (x, z) is the bilinear interpolation of the four
cell-centred velocities surrounding that point, and a neighbour that is solid contributes zero
velocity (no-slip), so the flow decays linearly to the wall inside the surface cell.  With
``interp="nearest"`` (default, see ``simulate``) the velocity is taken from the nearest cell (the cell of the position at the
start of the step); that older rule lets the flow slip by up to half a cell at the wall.
This is the Euler-Maruyama / random-walk solution of dc/dt + u.grad c = D lap c with a reaction
boundary; the 1e4 converts cm to um (D in cm2/s, sqrt(2 D dt) in cm).

Boundaries
- x is periodic with period ``grid.pitch_um`` (the trench array repeats).
- top reservoir: the bulk concentration is imposed at z = ``grid.height_um`` by keeping the particle
  density in the source layer [height - L_src, height] at the reference density n_ref (particles per
  um2) -- every step the layer is emptied and refilled uniformly with Poisson(n_ref*L_src*pitch)
  particles.  This realises the Dirichlet condition c = 1 at the top boundary.  Then c = n/n_ref.
- solid wall (a step that ends in a solid cell): the particle hit the Cu/Si surface.  With probability
  P_abs it is absorbed (deposited) and removed from the system.  Otherwise:

  * ``wall="specular"``: the step is reflected specularly back into the electrolyte about
    the cell face it crossed -- the face between the last fluid cell and the first solid cell -- with
    only the velocity component normal to that face reversed.  The reflected point is the mirror
    image of the step end, so the reflecting plane is the physical surface rather than the particle's
    old position.  If the mirror image is itself in a solid cell (the particle entered through a
    concave corner) the step is undone instead (old position).
  * ``wall="return"`` (default, see ``simulate``): the older rule, put the particle back where the step started.  That pushes
    every non-absorbed encounter a whole step away from the wall.

  For a first-order surface reaction with rate constant k (cm/s):

      P_abs = k * sqrt(pi * dt / D)          (clipped to [0, 1]; k = inf -> 1)

  [FT] R. Erban and S. J. Chapman, Phys. Biol. 4 (2007) 16-28, "Reactive boundary conditions for
  stochastic simulations of reaction-diffusion processes": the partially absorbing boundary condition
  D dc/dn = k c is implemented as a reaction probability per encounter; the expression above is their
  first-order form, valid for P_abs << 1, i.e. rms step << D/k and << dx.

  ``simulate(..., reactive="second")`` switches to their second-order scheme, the (a)+(b) rule of
  their Sec. 2.2 (the (b) part is the Andrews-Bray correction): a step that ends inside the wall
  reacts with probability ``p = k sqrt(pi dt/D) / 2``, *and* a step that stayed in the electrolyte
  but touched a wall face and came back reacts with probability ``exp(-d_old d_new / (D dt)) p``,
  where d_old/d_new are the distances of the two ends of the step from that face.  Both parts give
  the same leading-order Robin constant -- ``K = p sqrt(D/(pi dt))`` for (a) alone and
  ``K = 2 p sqrt(D/(pi dt))`` with (a)+(b) -- so the flux is unchanged to first order and the
  O(sqrt(dt)) boundary-layer error drops out (measured: the 1-D slab flux matches
  ``1/(1/k + H_eff/D)`` to 0.1 % with the layer-corrected height, and the first-order and
  second-order rules agree to 0.8 %, less than the counting noise of the comparison).
  ``reactive="first"`` (default) is the old rule, bit-identical to the pre-E1b code; the (b) term is
  derived for specular reflection, so use it with ``wall="specular"``.  Cost: the (b) term is
  computed only for particles in surface cells and roughly doubles the numpy wall time.

  A caveat that dominates the absolute accuracy of the 1-D benchmark: the top reservoir imposes
  c = 1 over a layer of thickness ``l_src = max(3 sigma, dx)``, i.e. effectively at its lower edge,
  so the tracer solves a column that is ``l_src`` shorter than the grid height.  On the flat wall at
  dx 4 (H = 80, k = 1e-3) the flux reads +3.1 % against ``1/(1/k + H/D)`` and -0.1 % against
  ``1/(1/k + (H - l_src)/D)``; at dx 8 the same pair is +5.6 % and -0.9 %.  It is common to every
  wall/interp rule, so it does not affect the ranking of the options or a ratio of two regions.

Accuracy note: the two wall rules differ only inside one rms step of the wall, and neither shifts the
absorbing plane by more than that.  What limits the flat-wall accuracy is common to both: the wall
concentration is taken at the first cell centre rather than at the face (an O(dx) boundary layer) and
the run has a finite burn-in.  Measured on the flat-wall case of tests/test_trenchflow_particles.py
(H = 80 um, dx = 4 um, rms step 1.4 um, k = 1e-3 cm/s): both rules land within 5 % of the analytic
1/(1/k + H/D) and the two fluxes differ from each other by about 1-2 %, i.e. less than the cell-size
bias they share, so the wall rule is not the accuracy limit at that resolution.  Where the difference
is resolved (coarse grids, rms step/H of several per cent)
``wall="return"`` biases the flux a few per cent *below* ``wall="specular"``, because undoing the step
acts like an extra push away from the wall.  The earlier note in this file claimed a ~0.6*sigma/H low
bias for that rule; on a flat wall at H = 80 um that is at the 1 % level and is not what the run above
measures, so treat the wall-rule error as "a per cent or two at practical resolutions" and refine dt
(smaller rms step) if a flat-wall validation needs better.

Migration drift (optional)
--------------------------
``simulate(..., phi=..., z_ion=2)`` adds the Nernst-Planck migration velocity

    u_mig = -z_ion F D / (R T) grad(phi)          [cm/s]   (phi in V, grad(phi) in V/cm)

to the flow velocity.  ``phi`` is the potential on the grid, shape (nz, nx), in V; its gradient is a
central difference of ``phi`` (periodic in x, one-sided on the first and last row, V/um -> V/cm) and is
sampled *bilinearly* at the particle whatever ``interp`` does for the flow.  ``phi=None`` (default)
leaves the solver bit-identical to the pre-drift code.  Only the drift is added: the potential does
not change the boundary rules or ``P_abs``, and the drift counts towards the CFL limit of ``dt``.

Backends
--------
``backend="numpy"`` (default) is the vectorised path written out above and is bit-identical to the
solver before ``backend`` existed.  ``backend="numba"`` runs the same algorithm in one compiled kernel
(numba is an optional extra, imported lazily; when it is missing the call falls back to numpy with a
RuntimeWarning).  The kernel updates the particles in a ``prange`` loop with a counter-based RNG
stream keyed by (seed, step, particle slot) -- reproducible and independent of the thread count -- and
counts absorptions and snapshots in per-thread buffers.  Tagged tracks (``n_tracks``) exist only on
the numpy path.  The two backends draw from different random streams, so they agree statistically
(within counting noise), not bit for bit.  Speed (this host, 16 threads): ~5x the numpy rate at
8 x 10^3 particles, ~8x at 1.3 x 10^5 (12962 steps in 31 s = 53 M particle-steps/s), but the
parallel-region overhead per step makes it *slower* than numpy below ~5 x 10^3 particles, and 4
threads beat 16 at ~3 x 10^4 (the step is transcendental-bound, so SMT hurts).  The E1b high
statistics run (16 particles/um2 in the 120 x 60 um trench, 5 x 10^5 particles, se 0.005) is 160 s.

Flux estimator: every absorption is recorded in the electrolyte surface cell the particle came FROM
(the cell of its position before the step), accumulated only after the burn-in.  For a surface cell
with f solid faces (face_length = f * dx_um)

    flux_density [um/s] = absorbed_count / (n_ref * t_sample * f * dx_um),   [cm/s] = 1e-4 * [um/s]

Conventions (see geometry.py): arrays are [iz, ix] with iz = 0 the bottom row, z = 0 the original wafer
plane, solid True = Si or Cu, x periodic with period pitch_um.
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np

from platesim.core.constants import F, R_GAS

from .geometry import CavityGrid, FlowField, surface_cells  # noqa: F401  (FlowField: type of `flow`)

__all__ = ["grid_from_levelset", "region_means", "simulate"]

_CM_TO_UM = 1e4
_UM_PER_CM = 1e-4
_KELVIN = 273.15
_MASK64 = (1 << 64) - 1
_TWOPI = 6.283185307179586


def _cell_of(z, x, ze0, dx, nx, nz):
    """Nearest-cell indices [iz, ix] of continuous positions (um); x periodic with period nx*dx.

    Cell (iz, ix) is centred at (x_um[ix], z_um[iz]), so its edges are at ..., ze0 + k*dx (z = 0 is
    an edge for every grid built by geometry.py).  A position on an edge belongs to the upper cell.
    """
    iz = np.floor((z - ze0) / dx).astype(np.int64)
    np.clip(iz, 0, nz - 1, out=iz)
    ix = np.floor(x / dx).astype(np.int64) % nx
    return iz, ix


def _bilinear(field, z, x, ze0, dx, nz, nx):
    """Bilinear interpolation of a cell-centred field at continuous positions (um).

    Cell (iz, ix) is centred at ((ix + 0.5) * dx, ze0 + (iz + 0.5) * dx), so the interpolation
    weights are the fractional offsets from the four surrounding cell centres.  x is periodic with
    period nx * dx; z is clamped to the outermost row, so a point at or beyond the first/last cell
    centre takes that cell's value (the physics above the top and inside the wall is set by the
    boundary rules, not by the interpolation).  ``field`` is (nz, nx).
    """
    fz = (z - ze0) / dx - 0.5
    iz0 = np.clip(np.floor(fz).astype(np.int64), 0, nz - 1)
    tz = np.clip(fz - iz0, 0.0, 1.0)
    iz1 = np.minimum(iz0 + 1, nz - 1)
    fx = x / dx - 0.5
    ix_lo = np.floor(fx).astype(np.int64)
    ix0 = ix_lo % nx
    tx = fx - ix_lo
    ix1 = (ix0 + 1) % nx
    return ((1.0 - tz) * ((1.0 - tx) * field[iz0, ix0] + tx * field[iz0, ix1])
            + tz * ((1.0 - tx) * field[iz1, ix0] + tx * field[iz1, ix1]))


def _velocity_at(ux, uz, z, x, ze0, dx, nz, nx, interp="bilinear"):
    """Velocity (cm/s) at continuous positions; ``ux``/``uz`` must already be 0 in solid (no-slip)."""
    if interp == "nearest":
        iz, ix = _cell_of(z, x, ze0, dx, nx, nz)
        return ux[iz, ix], uz[iz, ix]
    return (_bilinear(ux, z, x, ze0, dx, nz, nx),
            _bilinear(uz, z, x, ze0, dx, nz, nx))


def _drift_velocity(phi, dx_um, nz, nx, D_cm2_s, z_ion=2, T_C=22.0):
    """Nernst-Planck migration velocity (wx, wz) in cm/s from the potential ``phi`` (V) on the grid.

        u_mig = -z_ion F D / (R T) grad(phi)                                        [C]

    ``D`` in cm2/s and ``F / (R T)`` in 1/V give a mobility in cm2/(s V); the gradient of ``phi`` is
    taken in V/cm, so the result is cm/s.  The gradient is a central difference of the cell-centred
    ``phi`` (periodic in x with period ``nx``, one-sided on the first and last row) in V/um scaled by
    1e4.  The returned fields are meant to be bilinearly interpolated at a particle, exactly like the
    flow (``_bilinear``): for a linear potential the interpolated field is the exact constant drift.
    ``phi`` must be finite on the whole grid (the drift is not masked in solid, so a potential that is
    flat inside the metal gives zero drift there, as it should).
    """
    phi = np.asarray(phi, dtype=float)
    if phi.shape != (nz, nx):
        raise ValueError(f"phi must have the grid shape {(nz, nx)}, got {phi.shape}")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi must be finite on the whole grid")
    dx = float(dx_um)
    gz = np.empty_like(phi)                                   # dphi/dz, V/um
    if nz > 2:
        gz[1:-1] = (phi[2:] - phi[:-2]) / (2.0 * dx)
    else:
        gz[1:-1] = 0.0
    gz[0] = (phi[1] - phi[0]) / dx
    gz[-1] = (phi[-1] - phi[-2]) / dx
    gx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)   # dphi/dx, V/um
    mob = float(z_ion) * float(D_cm2_s) * F / (R_GAS * (float(T_C) + _KELVIN))   # cm2/(s V)
    fac = -mob * _CM_TO_UM                                    # V/um -> V/cm
    return fac * gx, fac * gz


def _specular_reflect(x_old, z_old, x_new, z_new, iz0, ix0, iz1, ix1, ze0, dx, pitch):
    """Mirror the step (x_old, z_old) -> (x_new, z_new) about the wall face it crossed.

    The step started in the fluid cell (iz0, ix0) and ended in the solid cell (iz1, ix1).  The
    crossed face is the face of the solid cell towards the start of the step: z = ze0 + iz1*dx when
    moving up, z = ze0 + (iz1 + 1)*dx when moving down, x = ix1*dx when moving +x and
    x = (ix1 + 1)*dx when moving -x (x faces in the unwrapped frame, result wrapped into
    [0, pitch)).  When both indices changed (a diagonal step into a corner cell) the face crossed
    last -- the one the path entered through -- is used.  Only the component normal to the face is
    reversed, so the tangential component of the step is kept.

    Returns (x, z); the caller must still check the result for solid and fall back to the old
    position when the mirror image is solid too.
    """
    dz = z_new - z_old
    ddx = (x_new - x_old + 0.5 * pitch) % pitch - 0.5 * pitch   # shortest signed x step
    changed_z = iz1 != iz0
    changed_x = ix1 != ix0
    with np.errstate(divide="ignore", invalid="ignore"):
        z_face = ze0 + np.where(dz > 0.0, iz1, iz1 + 1) * dx
        x_face = (ix1 + (ddx <= 0.0)) * dx
        t_z = np.where(changed_z, (z_face - z_old) / dz, np.inf)
        t_x = np.where(changed_x, (x_face - x_old) / ddx, np.inf)
    use_z = changed_z & (~changed_x | (t_z >= t_x))
    z_out = np.where(use_z, 2.0 * z_face - z_new, z_new)
    x_out = np.where(use_z, x_new, (2.0 * x_face - (x_old + ddx)) % pitch)
    return x_out, z_out


def _unwrapped_dx(x_new, x_old, pitch):
    """Shortest signed x step between two positions (um), periodic with period ``pitch``."""
    return (np.asarray(x_new) - np.asarray(x_old) + 0.5 * pitch) % pitch - 0.5 * pitch


def _wall_touch_prob(x_old, z_old, dx_step, dz_step, iz0, ix0, solid, surf, ze0, dx, nz, nx, D_dt):
    """P(the step touched a wall and came back) -- the Andrews-Bray term (b) of the reactive BC.

    Erban & Chapman, Phys. Biol. 4 (2007) 16-28, Sec. 2.2: for the Euler-Maruyama random walk
    (their "position jump process II") the probability that a molecule hit the plane ``x = 0`` at
    least once during a step that starts at ``x_old > 0`` and ends at ``x_new > 0`` is
    ``exp(-x_old * x_new / (D dt))``.  Applying that to every solid face of the cell the step
    started in (distances normal to the face, um; ``D_dt`` = ``D * dt`` in um2) and combining the
    faces as ``1 - prod(1 - r_face)`` gives the probability of touching at least one wall while
    ending in the fluid.  A step that crossed a face has ``d_new < 0`` on that face and gets
    ``r_face = 0``: the crossing branch of the boundary rule already handles it.

    Only particles whose cell has a solid face neighbour can have a non-negligible probability
    (``r < exp(-2)`` once either distance exceeds one rms step, and the rms step is 0.35 dx), so
    the caller can restrict the work to the surface cells.  Returns an array of length
    ``x_old.size`` (0 outside those cells).
    """
    r = np.zeros(np.size(x_old), dtype=float)
    if r.size == 0:
        return r
    idx = np.flatnonzero(surf[iz0, ix0])
    if idx.size == 0:
        return r
    iz = iz0[idx]
    ix = ix0[idx]
    xs = np.asarray(x_old, dtype=float)[idx]
    zs = np.asarray(z_old, dtype=float)[idx]
    dxs = np.asarray(dx_step, dtype=float)[idx]
    dzs = np.asarray(dz_step, dtype=float)[idx]
    stay = np.ones(idx.size, dtype=float)                    # prod(1 - r_face)

    def _face(d_old, d_new):
        """Fold one face in: only steps that stay on the fluid side of it can have touched it."""
        keep = (d_old > 0.0) & (d_new > 0.0)
        if keep.any():
            q = np.exp(-d_old[keep] * d_new[keep] / D_dt)
            stay[keep] *= 1.0 - q

    below = (iz > 0) & solid[np.maximum(iz - 1, 0), ix]
    with np.errstate(invalid="ignore"):
        face_z = ze0 + iz * dx
        _face(np.where(below, zs - face_z, -1.0), np.where(below, zs - face_z + dzs, -1.0))
        above = (iz < nz - 1) & solid[np.minimum(iz + 1, nz - 1), ix]
        face_z = ze0 + (iz + 1) * dx
        _face(np.where(above, face_z - zs, -1.0), np.where(above, face_z - zs - dzs, -1.0))
        left = solid[iz, (ix - 1) % nx]
        _face(np.where(left, xs - ix * dx, -1.0), np.where(left, xs - ix * dx + dxs, -1.0))
        right = solid[iz, (ix + 1) % nx]
        _face(np.where(right, (ix + 1) * dx - xs, -1.0),
              np.where(right, (ix + 1) * dx - xs - dxs, -1.0))
    r[idx] = 1.0 - stay
    return r


def _surface_mask(solid, nx, nz):
    """True in fluid cells with at least one solid face neighbour (periodic in x)."""
    nb = np.zeros((nz, nx), dtype=np.int8)
    nb += np.roll(solid, 1, axis=1) + np.roll(solid, -1, axis=1)
    nb[1:] += solid[:-1]
    nb[:-1] += solid[1:]
    return (~solid) & (nb > 0)


def _splitmix64(k):
    """splitmix64 mixing function: a bijection on 64 bits with full avalanche, used as a hash RNG."""
    k = (k + 0x9E3779B97F4A7C15) & _MASK64
    k = ((k ^ (k >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    k = ((k ^ (k >> 27)) * 0x94D049BB133111EB) & _MASK64
    return k ^ (k >> 31)


def _u01(k):
    """Uniform in (0, 1) with 53 bits from a 64-bit hash value (never exactly 0 or 1)."""
    return (float((k >> 11) & 0x1FFFFFFFFFFFFF) + 0.5) * (1.0 / 9007199254740992.0)


def _cell_of_1(z, ze0, dx, nz):
    """Scalar twin of :func:`_cell_of`'s z lookup (the x part is ``int(x / dx) % nx``)."""
    iz = int((z - ze0) / dx)
    if iz < 0:
        iz = 0
    elif iz > nz - 1:
        iz = nz - 1
    return iz


def _bilinear_1(field, z, x, ze0, dx, nz, nx):
    """Scalar twin of :func:`_bilinear` (same clamping and periodic wrap)."""
    fz = (z - ze0) / dx - 0.5
    iz0 = int(math.floor(fz))
    if iz0 < 0:
        iz0 = 0
    elif iz0 > nz - 1:
        iz0 = nz - 1
    tz = fz - iz0
    if tz < 0.0:
        tz = 0.0
    elif tz > 1.0:
        tz = 1.0
    iz1 = iz0 + 1 if iz0 + 1 < nz else nz - 1
    fx = x / dx - 0.5
    ixl = int(math.floor(fx))
    ix0 = ixl % nx
    tx = fx - ixl
    ix1 = (ix0 + 1) % nx
    return ((1.0 - tz) * ((1.0 - tx) * field[iz0, ix0] + tx * field[iz0, ix1])
            + tz * ((1.0 - tx) * field[iz1, ix0] + tx * field[iz1, ix1]))


def _specular_1(xo, zo, xn, zn, iz0, ix0, iz1, ix1, ze0, dx, pitch):
    """Scalar twin of :func:`_specular_reflect` for one particle."""
    dz = zn - zo
    ddx = (xn - xo + 0.5 * pitch) % pitch - 0.5 * pitch          # shortest signed x step
    changed_z = iz1 != iz0
    changed_x = ix1 != ix0
    z_face = 0.0
    x_face = 0.0
    t_z = math.inf
    t_x = math.inf
    if changed_z:
        z_face = ze0 + (iz1 if dz > 0.0 else iz1 + 1) * dx
        if dz != 0.0:
            t_z = (z_face - zo) / dz
    if changed_x:
        x_face = (ix1 + (1 if ddx <= 0.0 else 0)) * dx
        if ddx != 0.0:
            t_x = (x_face - xo) / ddx
    if changed_z and (not changed_x or t_z >= t_x):
        return xn, 2.0 * z_face - zn
    return (2.0 * x_face - (xo + ddx)) % pitch, zn


_NB = None                      # lazy numba kernel; None = not tried, False = numba unavailable


def _nb_kernels():
    """Compile (once) and return the numba tracer kernel, or None when numba cannot be imported."""
    global _NB
    if _NB is not None:
        return _NB or None
    try:                                                  # numba is an optional extra
        from numba import get_num_threads, get_thread_id, njit, prange
    except Exception:
        _NB = False
        return None
    h64 = njit(inline="always")(_splitmix64)
    u01 = njit(inline="always")(_u01)
    cell = njit(inline="always")(_cell_of_1)
    bil = njit(inline="always")(_bilinear_1)
    spec = njit(inline="always")(_specular_1)

    @njit(cache=False, parallel=True)
    def tracer(x, z, dead, solid, surf, ux, uz, wx, wz, nnew, dx, pitch, ze0, z_top, sigma, dt,
               p_abs, l_src, n_burn, n_samp, nz, nx, n_alive0, d_dt, use_flow, use_bilinear,
               use_specular, use_drift, use_ab, seed):
        """One call = the whole time loop; see ``_simulate_numba`` for the meaning of the arrays."""
        nth = get_num_threads()
        cnt = np.zeros((nth, nz * nx), np.int64)          # absorptions per thread and cell
        cacc = np.zeros((nth, nz * nx), np.float64)       # concentration snapshots, per thread
        cap = x.size
        n_alive = n_alive0
        n_snap = 0
        n_sum = 0
        steps = n_burn + n_samp
        u64 = np.uint64
        for step in range(steps):
            sampling = step >= n_burn
            base = h64(u64(seed) + u64(step) * u64(0x9E3779B97F4A7C15))
            for i in prange(n_alive):
                tid = get_thread_id()
                xo = x[i]
                zo = z[i]
                iz0 = cell(zo, ze0, dx, nz)               # cell the step came from
                ix0 = int(xo / dx) % nx
                hv = h64(base + u64(i))
                hv2 = h64(hv)
                hv3 = h64(hv2)
                r2 = -2.0 * math.log(u01(hv))
                ang = _TWOPI * u01(hv2)
                norm1 = math.sqrt(r2) * math.cos(ang)
                norm2 = math.sqrt(r2) * math.sin(ang)
                uxp = 0.0
                uzp = 0.0
                if use_flow:
                    if use_bilinear:
                        uxp = bil(ux, zo, xo, ze0, dx, nz, nx)
                        uzp = bil(uz, zo, xo, ze0, dx, nz, nx)
                    else:
                        uxp = ux[iz0, ix0]
                        uzp = uz[iz0, ix0]
                if use_drift:
                    uxp += bil(wx, zo, xo, ze0, dx, nz, nx)
                    uzp += bil(wz, zo, xo, ze0, dx, nz, nx)
                xn = xo + (uxp * dt) * 10000.0 + sigma * norm1
                zn = zo + (uzp * dt) * 10000.0 + sigma * norm2
                xn = xn % pitch
                if zn > z_top:
                    zn = z_top
                iz1 = cell(zn, ze0, dx, nz)
                ix1 = int(xn / dx) % nx
                if solid[iz1, ix1]:
                    if p_abs >= 1.0:
                        absorbed = True
                    elif p_abs > 0.0:
                        absorbed = u01(hv3) < p_abs
                    else:
                        absorbed = False
                    if absorbed:
                        if sampling:
                            cnt[tid, iz0 * nx + ix0] += 1
                        dead[i] = 1
                    elif use_specular:
                        xr, zr = spec(xo, zo, xn, zn, iz0, ix0, iz1, ix1, ze0, dx, pitch)
                        if solid[cell(zr, ze0, dx, nz), int(xr / dx) % nx]:
                            xr = xo                            # concave corner: undo the step
                            zr = zo
                        xn = xr
                        zn = zr
                    else:
                        xn = xo                                # return rule: undo the step
                        zn = zo
                elif use_ab:
                    # Andrews-Bray term (b): the continuous path inside the step may have touched a
                    # wall face and come back.  r = 1 - prod(1 - exp(-d_old d_new /(D dt))) over the
                    # solid faces of the starting cell (see particles._wall_touch_prob).
                    stay = 1.0
                    touched = False
                    dzs = zn - zo
                    dxs = (xn - xo + 0.5 * pitch) % pitch - 0.5 * pitch
                    if surf[iz0, ix0]:
                        if iz0 > 0 and solid[iz0 - 1, ix0]:    # wall below
                            d0 = zo - (ze0 + iz0 * dx)
                            d1 = d0 + dzs
                            if d0 > 0.0 and d1 > 0.0:
                                stay *= 1.0 - math.exp(-d0 * d1 / d_dt)
                                touched = True
                        if iz0 < nz - 1 and solid[iz0 + 1, ix0]:   # wall above
                            d0 = ze0 + (iz0 + 1) * dx - zo
                            d1 = d0 - dzs
                            if d0 > 0.0 and d1 > 0.0:
                                stay *= 1.0 - math.exp(-d0 * d1 / d_dt)
                                touched = True
                        ddx = int(xo / dx) % nx
                        if solid[iz0, (ddx - 1) % nx]:         # wall on the left
                            d0 = xo - ddx * dx
                            d1 = d0 + dxs
                            if d0 > 0.0 and d1 > 0.0:
                                stay *= 1.0 - math.exp(-d0 * d1 / d_dt)
                                touched = True
                        if solid[iz0, (ddx + 1) % nx]:         # wall on the right
                            d0 = (ddx + 1) * dx - xo
                            d1 = d0 - dxs
                            if d0 > 0.0 and d1 > 0.0:
                                stay *= 1.0 - math.exp(-d0 * d1 / d_dt)
                                touched = True
                    if touched and u01(h64(hv3)) < (1.0 - stay) * p_abs:
                        if sampling:
                            cnt[tid, iz0 * nx + ix0] += 1
                        dead[i] = 1
                if dead[i] == 0 and zn >= z_top - l_src:
                    dead[i] = 1                                # top reservoir: layer is renewed
                if dead[i] == 0:
                    x[i] = xn
                    z[i] = zn
            # ---- serial: compact the survivors to [0, n_alive)
            j = 0
            for i in range(n_alive):
                if dead[i] == 0:
                    if j != i:
                        x[j] = x[i]
                        z[j] = z[i]
                    dead[j] = 0                            # the slot now holds a live particle
                    j += 1
            for i in range(j, n_alive):
                dead[i] = 0
            n_alive = j
            # ---- serial: refill the source layer to its reference density
            nb = nnew[step]
            if nb > 0:
                if n_alive + nb > cap:
                    raise RuntimeError("tracer exceeded the preallocated capacity; raise max_particles")
                sbase = h64(u64(seed) + u64(step + 1) * u64(0xD1B54A32D192ED03))
                for k in range(nb):
                    hk = h64(sbase + u64(k))
                    x[n_alive + k] = u01(hk) * pitch
                    z[n_alive + k] = z_top - l_src * u01(h64(hk))
                n_alive += nb
            # ---- statistics
            if sampling:
                for i in prange(n_alive):
                    tid = get_thread_id()
                    cacc[tid, cell(z[i], ze0, dx, nz) * nx + int(x[i] / dx) % nx] += 1.0
                n_snap += 1
                n_sum += n_alive
        out_cnt = np.zeros(nz * nx, np.int64)
        out_c = np.zeros(nz * nx, np.float64)
        for t in range(nth):
            for c in range(nz * nx):
                out_cnt[c] += cnt[t, c]
                out_c[c] += cacc[t, c]
        return out_cnt, out_c, n_snap, n_sum, n_alive

    _NB = tracer
    return tracer


def grid_from_levelset(psi, g, width_um=None, depth_um=None, deposit_um=0.0):
    """Build the tracer's :class:`~platesim.trenchflow.geometry.CavityGrid` from a level set.

    ``psi`` is the feature engine's signed distance, shape (nr, nz) in the engine's (x, z) order, with
    ``psi > 0`` in the electrolyte and ``psi <= 0`` in metal, and ``g`` its
    :class:`~platesim.feature.engine.levelset.Grid` (duck-typed here: ``dx_um``, ``rc``, ``zc``, ``H``,
    ``z_top``, ``d_um``, ``H_um``, ``R_um``).  The mapping is the one
    :meth:`platesim.feature.engine.sim.ViaSim.cavity_flow` already uses for the flow solve, so a tracer
    run on the returned grid sees exactly the geometry the continuum model solved on:

    * ``solid = (psi <= 0).T``  -- engine (nr, nz) -> CavityGrid (nz, nx), engine lengths in cm
      (``UM = 1e-4``) converted to um;
    * ``z_um = (g.zc[0] - g.H) / UM``: z = 0 is the wafer plane, z < 0 inside the trench;
    * ``height_um = (g.z_top - g.H) / UM``, ``pitch_um = g.R_um``.

    ``width_um``/``depth_um`` default to the engine's ``d_um``/``H_um`` and only need overriding for a
    geometry whose feature width/depth differ from the engine's (a partially filled feature keeps them).
    ``deposit_um`` is bookkeeping only (the deposit is already inside ``psi``).
    """
    psi = np.asarray(psi, dtype=float)
    if psi.ndim != 2 or psi.shape != (int(g.nr), int(g.nz)):
        raise ValueError(f"psi must have the engine grid shape {(int(g.nr), int(g.nz))}, "
                         f"got {psi.shape}")
    if not np.all(np.isfinite(psi)):
        raise ValueError("psi must be finite")
    dx_um = float(g.dx_um)
    return CavityGrid(
        dx_um=dx_um,
        x_um=np.asarray(g.rc[:, 0], dtype=float) / _UM_PER_CM,
        z_um=(np.asarray(g.zc[0], dtype=float) - float(g.H)) / _UM_PER_CM,
        solid=(psi <= 0.0).T.copy(),
        width_um=float(g.d_um if width_um is None else width_um),
        depth_um=float(g.H_um if depth_um is None else depth_um),
        pitch_um=float(g.R_um),
        height_um=(float(g.z_top) - float(g.H)) / _UM_PER_CM,
        deposit_um=float(deposit_um),
    )


def _initial_particles(grid, n_ref, init, ze0, dx, rng, max_particles):
    """Start from c = z/height above the wafer plane (empty trench) to shorten the burn-in.

    ``init='linear'`` is the steady profile of the pure-diffusion problem, i.e. close to the converged
    answer, so the burn-in only has to fix the wall boundary layer.  ``init='empty'`` starts with no
    particles at all (the reservoir fills the domain by diffusion).
    """
    if init == "empty":
        return np.empty(0), np.empty(0)
    H = float(grid.height_um)
    prof = np.clip(grid.z_um, 0.0, H)[:, None] / H          # [C] linear profile, 0 below the wafer plane
    c0 = np.where(grid.solid, 0.0, prof)                    # trench empty -> no particles in Si/Cu
    mean_n = n_ref * c0 * dx * dx                           # expected particles per cell
    cnt = rng.poisson(mean_n)
    iz, ix = np.nonzero(cnt)
    k = cnt[iz, ix]
    n = int(k.sum())
    if n > max_particles:
        raise ValueError(f"initial condition needs {n} particles > max_particles={max_particles}; "
                         "reduce n_ref_per_um2")
    izr = np.repeat(iz, k)
    ixr = np.repeat(ix, k)
    z = ze0 + (izr + rng.random(n)) * dx
    x = (ixr + rng.random(n)) * dx
    return x, z


def simulate(grid, flow=None, D_cm2_s=5.6e-6, k_cm_s=1e-3, n_ref_per_um2=0.02, t_burn_s=None,
             t_sample_s=None, dt_s=None, seed=0, n_tracks=0, max_particles=2_000_000,
             init="linear", interp="nearest", wall="return", backend="numpy", phi=None, z_ion=2,
             T_C=22.0, reactive="first") -> dict:
    """Run the Brownian-tracer solver on ``grid``; see the module docstring for the model.

    ``interp`` selects how the flow is sampled at a particle: ``"bilinear"`` interpolates the four
    surrounding cell-centred velocities with no-slip on solid neighbours, ``"nearest"`` (default) is
    the nearest-cell sampling.  ``wall`` selects the treatment of a step that ends in a solid cell
    without being absorbed: ``"specular"`` mirrors it about the crossed cell face, ``"return"``
    (default) puts it back at the previous position.  The remaining arguments are unchanged.

    ``phi`` (shape (nz, nx), V) switches on the Nernst-Planck migration drift
    ``-z_ion F D / (R T) grad(phi)`` added to the flow velocity (``z_ion``, ``T_C``); ``phi=None``
    (default) runs the plain advection-diffusion problem.  ``backend`` picks the implementation:
    ``"numpy"`` (default) is the vectorised path, ``"numba"`` the compiled kernel with the same
    algorithm and a different (counter-based) random stream -- see the module docstring.
    ``reactive`` picks the partially absorbing boundary rule: ``"first"`` (default, the old
    first-order Erban-Chapman probability) or ``"second"``, their (a)+(b) two-part scheme with the
    Andrews-Bray correction (see the module docstring; use it with ``wall="specular"``).

    The defaults stay on the older rules on purpose, and the E1b high-statistics runs say why.  The
    original cross-check (0.25 particles/um2, se(floor/field) 0.029) gave 0.726 / 0.772 / 0.697 /
    0.669 for nearest+return / nearest+specular / bilinear+return / bilinear+specular against 0.736
    for the continuum: the 0.669 is a 2.3 se fluctuation, not a property of bilinear interpolation.
    Repeating the same case at 4 particles/um2 (se 0.007) gives 0.7264 / 0.7233 / 0.7171 / 0.7159,
    i.e. nearest+return stays closest and bilinear is ~1 se lower; at 16 particles/um2
    (se 0.0036, numba backend, 160 s) nearest+return reads 0.7218 +- 0.0036 and wall/field
    0.7992 +- 0.0041 against 0.8169.  So all four rules agree within counting noise and all sit
    ~1.5-2 % below the continuum; that common offset is not the wall rule:
      * the continuum is a first-order upwind FV solve, whose numerical diffusion (u dx/2 ~ 20 % of
        D at dx 6) smooths the field-to-floor gradient; the "hybrid" scheme moves floor/field from
        0.7361 to 0.7305, i.e. 0.6 % towards the particles;
      * the tracer's source layer (thickness max(3 sigma, dx) = 6.3 um here) imposes c = 1 one layer
        below the continuum's plane; truncating the continuum domain by that layer moves floor/field
        from 0.7361 to 0.7338;
      * the 1-D flat-wall benchmark shows the reaction rule itself is good to <1 % (first order) and
        <0.6 % (second order) once the same layer correction is applied.
    Those account for roughly half of the offset; the rest is not identified (candidates: the
    staircase wall orientation of the corner surface cells, and the tracer's own O(sqrt(dt)) wall
    layer, sigma = 0.35 dx = 2.1 um here).  Switch the default only if a comparison against a
    *converged* continuum (higher-order advection, matched top plane) resolves a difference; neither
    of those is true today.

    Returns a dict with the surface-cell arrays (same order as ``geometry.surface_cells``), the flux
    density per surface cell (cm/s), the raw absorbed counts, the time-averaged concentration estimate
    c (NaN in solid), run bookkeeping ('n_particles_mean', 'steps', 'dt_s', 't_sample_s', 'P_abs',
    'interp', 'wall', 'backend', 'migration', 'wall_s') and the tagged tracks (n_tracks, n_saved, 2)
    or None.
    """
    key = str(backend).strip().lower()
    if key not in ("numpy", "numba"):
        raise ValueError("backend must be 'numpy' or 'numba'")
    rkey = str(reactive).strip().lower()
    if rkey not in ("first", "second"):
        raise ValueError("reactive must be 'first' or 'second'")
    if key == "numba":
        if _nb_kernels() is None:
            warnings.warn("backend='numba' requested but numba is not importable; using numpy",
                          RuntimeWarning, stacklevel=2)
            key = "numpy"
        elif int(n_tracks) > 0:
            raise ValueError("n_tracks > 0 is only supported with backend='numpy'")
    if key == "numba":
        return _simulate_numba(grid, flow, D_cm2_s=D_cm2_s, k_cm_s=k_cm_s,
                               n_ref_per_um2=n_ref_per_um2, t_burn_s=t_burn_s, t_sample_s=t_sample_s,
                               dt_s=dt_s, seed=seed, n_tracks=n_tracks, max_particles=max_particles,
                               init=init, interp=interp, wall=wall, phi=phi, z_ion=z_ion, T_C=T_C,
                               reactive=rkey)
    return _simulate_numpy(grid, flow, D_cm2_s=D_cm2_s, k_cm_s=k_cm_s, n_ref_per_um2=n_ref_per_um2,
                           t_burn_s=t_burn_s, t_sample_s=t_sample_s, dt_s=dt_s, seed=seed,
                           n_tracks=n_tracks, max_particles=max_particles, init=init, interp=interp,
                           wall=wall, phi=phi, z_ion=z_ion, T_C=T_C, reactive=rkey)


def _simulate_numpy(grid, flow=None, D_cm2_s=5.6e-6, k_cm_s=1e-3, n_ref_per_um2=0.02,
                    t_burn_s=None, t_sample_s=None, dt_s=None, seed=0, n_tracks=0,
                    max_particles=2_000_000, init="linear", interp="nearest", wall="return",
                    phi=None, z_ion=2, T_C=22.0, reactive="first") -> dict:
    """The vectorised tracer (``simulate(..., backend="numpy")``); bit-identical to the pre-E1b code."""
    t_wall0 = time.perf_counter()
    if init not in ("linear", "empty"):
        raise ValueError("init must be 'linear' or 'empty'")
    if interp not in ("bilinear", "nearest"):
        raise ValueError("interp must be 'bilinear' or 'nearest'")
    if wall not in ("specular", "return"):
        raise ValueError("wall must be 'specular' or 'return'")
    if reactive not in ("first", "second"):
        raise ValueError("reactive must be 'first' or 'second'")
    g = grid
    dx = float(g.dx_um)
    nz, nx = g.shape
    pitch = float(g.pitch_um)
    D = float(D_cm2_s)
    k = float(k_cm_s)
    n_ref = float(n_ref_per_um2)
    if D <= 0.0 or n_ref <= 0.0:
        raise ValueError("D_cm2_s and n_ref_per_um2 must be positive")
    solid = np.asarray(g.solid, dtype=bool)
    ze0 = float(g.z_um[0]) - 0.5 * dx
    z_top = ze0 + nz * dx                                   # = height_um for grids from geometry.py
    rng = np.random.default_rng(seed)
    if phi is None:
        drift = None
    else:
        drift = _drift_velocity(phi, dx, nz, nx, D, z_ion, T_C)

    # ---- times -------------------------------------------------------------
    H_cm = (float(g.height_um) + float(g.depth_um)) * 1e-4  # deepest path to steady state, cm [C]
    if t_burn_s is None:
        t_burn = (0.5 if init == "linear" else 1.5) * H_cm ** 2 / D   # [C] diffusion time
    else:
        t_burn = float(t_burn_s)
    t_sample = t_burn if t_sample_s is None else float(t_sample_s)
    if t_burn < 0.0 or t_sample <= 0.0:
        raise ValueError("t_burn_s must be >= 0 and t_sample_s > 0")

    # ---- time step ---------------------------------------------------------
    dt_diff = (0.35 * dx / _CM_TO_UM) ** 2 / (2.0 * D)      # rms free step = 0.35 dx [J]
    umax = 0.0
    if flow is not None:
        if tuple(flow.grid.shape) != (nz, nx) or flow.ux.shape != (nz, nx):
            raise ValueError("flow field shape does not match grid")
        umax = float(max(np.abs(flow.ux).max(), np.abs(flow.uz).max()) * _CM_TO_UM)
    if drift is not None:                                   # the drift also counts towards the CFL limit
        umax = max(umax, float(max(np.abs(drift[0]).max(), np.abs(drift[1]).max()) * _CM_TO_UM))
    dt_adv = np.inf if umax <= 0.0 else 0.5 * dx / umax      # |u|*dt <= 0.5 dx [J]
    dt = float(dt_s) if dt_s is not None else float(min(dt_diff, dt_adv))
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    sigma_um = float(np.sqrt(2.0 * D * dt) * _CM_TO_UM)     # rms free step, um
    n_burn = int(round(t_burn / dt))
    n_samp = max(1, int(round(t_sample / dt)))
    steps = n_burn + n_samp
    t_samp_eff = n_samp * dt                                # sampling time actually simulated, s

    if np.isinf(k):
        p_abs = 1.0
        p_first = 1.0
    else:
        # [FT] Erban & Chapman, Phys. Biol. 4 (2007) 16-28: reaction probability per encounter.
        p_first = float(np.clip(k * np.sqrt(np.pi * dt / D), 0.0, 1.0))
        # Their (a)+(b) scheme (the Andrews-Bray correction (b) is applied below, to steps that
        # touched a wall and came back) carries half the per-touch probability: with both terms the
        # Robin constant is K = 2 p sqrt(D/(pi dt)), versus K = p sqrt(D/(pi dt)) for (a) alone.
        p_abs = min(0.5 * k * np.sqrt(np.pi * dt / D), 1.0) if reactive == "second" else p_first
    surf = _surface_mask(solid, nx, nz) if reactive == "second" else None
    D_dt_um2 = D * _CM_TO_UM * _CM_TO_UM * dt              # D dt in um2 (for the Andrews-Bray term)

    # ---- velocity field with no-slip in solid (used by the bilinear stencil) --
    if flow is None:
        ux_ns = uz_ns = None
    else:
        ux_ns = np.where(solid, 0.0, np.asarray(flow.ux, dtype=float))
        uz_ns = np.where(solid, 0.0, np.asarray(flow.uz, dtype=float))

    l_src = max(3.0 * sigma_um, dx)                         # >= ~3 rms steps, so fully renewed [J]
    n_src = n_ref * l_src * pitch                           # mean particles in the source layer
    # ---- initial state -----------------------------------------------------
    x, z = _initial_particles(g, n_ref, init, ze0, dx, rng, max_particles)

    n_tracks = int(n_tracks)
    if n_tracks < 0:
        raise ValueError("n_tracks must be >= 0")
    save_every = max(1, steps // 200)
    n_saved = steps // save_every + 1
    tracks = None
    tag_slot = np.full(x.size, -1, dtype=np.int64)            # per-particle track slot, -1 = untagged
    n_tagged = 0
    if n_tracks > 0:
        tracks = np.full((n_tracks, n_saved, 2), np.nan)
        if x.size:
            idx = np.unique(np.linspace(0, x.size - 1, num=min(n_tracks, x.size)).astype(np.int64))
            tag_slot[idx] = np.arange(idx.size, dtype=np.int64)   # spread over the column
            n_tagged = int(idx.size)

    abs_counts = np.zeros(nz * nx, dtype=np.int64)          # absorptions per cell, sample phase only
    c_acc = np.zeros(nz * nx, dtype=np.float64)             # sum of per-snapshot concentrations
    n_snap = 0
    n_particles_sum = 0
    n_abs_total = 0

    # ---- time loop (one Python loop over steps; everything else vectorised) --
    for step in range(steps):
        sampling = step >= n_burn
        n = x.size
        x_old = x
        z_old = z
        iz0, ix0 = _cell_of(z_old, x_old, ze0, dx, nx, nz)  # cell "come from" (nearest cell)
        xi = rng.standard_normal(2 * n)
        if flow is None:
            xn = x_old + sigma_um * xi[:n]
            zn = z_old + sigma_um * xi[n:]
        else:
            ux_p, uz_p = _velocity_at(ux_ns, uz_ns, z_old, x_old, ze0, dx, nz, nx, interp)
            xn = x_old + (ux_p * dt) * _CM_TO_UM + sigma_um * xi[:n]
            zn = z_old + (uz_p * dt) * _CM_TO_UM + sigma_um * xi[n:]
        if drift is not None:                                # migration: bilinear gradient of phi
            xn = xn + (_bilinear(drift[0], z_old, x_old, ze0, dx, nz, nx) * dt) * _CM_TO_UM
            zn = zn + (_bilinear(drift[1], z_old, x_old, ze0, dx, nz, nx) * dt) * _CM_TO_UM
        xn %= pitch                                         # periodic in x
        np.minimum(zn, z_top, out=zn)                       # reservoir reflects anything above the top
        iz1, ix1 = _cell_of(zn, xn, ze0, dx, nx, nz)
        hit = solid[iz1, ix1]

        # ---- solid encounter: absorb with P_abs, else reflect (specular) / return to the start
        absorbed = None
        if hit.any():
            hi = np.flatnonzero(hit)
            if p_abs >= 1.0:
                absorbed = hi
            elif p_abs > 0.0:
                absorbed = hi[rng.random(hi.size) < p_abs]
            else:
                absorbed = hi[:0]
            reflect = hit.copy()
            reflect[absorbed] = False
            if reflect.any():
                ri = np.flatnonzero(reflect)
                if wall == "specular":
                    xr, zr = _specular_reflect(x_old[ri], z_old[ri], xn[ri], zn[ri],
                                               iz0[ri], ix0[ri], iz1[ri], ix1[ri],
                                               ze0, dx, pitch)
                    izr, ixr = _cell_of(zr, xr, ze0, dx, nx, nz)
                    bad = solid[izr, ixr]                   # concave corner: the mirror is solid too
                    if bad.any():
                        xr[bad] = x_old[ri][bad]
                        zr[bad] = z_old[ri][bad]
                    xn[ri] = xr
                    zn[ri] = zr
                else:                                      # a step that would enter the wall is undone
                    xn[ri] = x_old[ri]
                    zn[ri] = z_old[ri]
        # ---- Andrews-Bray term (b): a step that touched a wall and came back can still react
        if reactive == "second" and p_abs > 0.0:
            bi = np.flatnonzero(~hit)
            if bi.size:
                dx_step = _unwrapped_dx(xn[bi], x_old[bi], pitch)
                r_touch = _wall_touch_prob(x_old[bi], z_old[bi], dx_step, zn[bi] - z_old[bi],
                                           iz0[bi], ix0[bi], solid, surf, ze0, dx, nz, nx,
                                           D_dt_um2)
                react = bi[rng.random(bi.size) < r_touch * p_abs]
                absorbed = react if absorbed is None else np.concatenate([absorbed, react])
        if absorbed is not None and absorbed.size:
            if sampling:                                    # record in the cell it came from
                abs_counts += np.bincount(iz0[absorbed] * nx + ix0[absorbed], minlength=nz * nx)
                n_abs_total += int(absorbed.size)
            keep = np.ones(n, dtype=bool)
            keep[absorbed] = False
            xn = xn[keep]
            zn = zn[keep]
            if n_tracks:
                tag_slot = tag_slot[keep]

        # ---- top reservoir: keep the source layer at the reference density (Dirichlet c = 1)
        if x.size:
            src = zn >= (z_top - l_src)
            if src.any():
                keep = ~src
                xn = xn[keep]
                zn = zn[keep]
                if n_tracks:
                    tag_slot = tag_slot[keep]
        n_new = int(rng.poisson(n_src))
        if n_new:
            if xn.size + n_new > max_particles:
                raise RuntimeError(f"particle count exceeded max_particles={max_particles}")
            xn = np.concatenate([xn, rng.random(n_new) * pitch])
            zn = np.concatenate([zn, z_top - l_src * rng.random(n_new)])
            if n_tracks:
                tag_slot = np.concatenate([tag_slot, np.full(n_new, -1, dtype=np.int64)])
        x = xn
        z = zn

        # ---- statistics ----------------------------------------------------
        if sampling:
            i_s, j_s = _cell_of(z, x, ze0, dx, nx, nz)
            c_acc += np.bincount(i_s * nx + j_s, minlength=nz * nx)
            n_snap += 1
            n_particles_sum += int(x.size)
        if tracks is not None and step % save_every == 0:
            j = step // save_every
            if j < n_saved:
                if n_tagged < n_tracks:                 # init='empty': tag particles once they exist
                    cand = np.flatnonzero((tag_slot < 0) & (z < z_top - l_src))
                    take = int(min(cand.size, n_tracks - n_tagged))
                    if take:
                        tag_slot[cand[:take]] = n_tagged + np.arange(take, dtype=np.int64)
                        n_tagged += take
                sel = np.flatnonzero(tag_slot >= 0)
                if sel.size:
                    slots = tag_slot[sel]
                    tracks[slots, j, 0] = x[sel]
                    tracks[slots, j, 1] = z[sel]
    # ---- results -----------------------------------------------------------
    wall_s = time.perf_counter() - t_wall0
    c = c_acc.reshape(nz, nx) / (n_ref * dx * dx * max(n_snap, 1))   # c = n / n_ref, time averaged
    c[solid] = np.nan

    s_iz, s_ix, s_nf = surface_cells(g)
    cnt = abs_counts.reshape(nz, nx)[s_iz, s_ix]
    face_um = s_nf * dx
    flux_um_s = cnt / (n_ref * t_samp_eff * face_um)          # um/s (2-D: per um of trench length)
    return {
        "surf_iz": s_iz,
        "surf_ix": s_ix,
        "surf_x_um": np.asarray(g.x_um)[s_ix],
        "surf_z_um": np.asarray(g.z_um)[s_iz],
        "surf_flux_density": flux_um_s * 1e-4,                # cm/s
        "surf_counts": cnt,
        "c": c,
        "n_particles_mean": float(n_particles_sum / max(n_snap, 1)),
        "steps": int(steps),
        "dt_s": dt,
        "t_sample_s": t_samp_eff,
        "P_abs": p_abs,
        "P_abs_first_order": p_first,
        "interp": interp,
        "wall": wall,
        "reactive": reactive,
        "backend": "numpy",
        "migration": drift is not None,
        "wall_s": wall_s,
        "tracks": tracks,
        "x_final_um": x,                                     # final particle positions (for checks/plots)
        "z_final_um": z,
        # bookkeeping (not required by the caller contract, handy for reports/CLI)
        "t_burn_s": n_burn * dt,
        "n_ref_per_um2": n_ref,
        "k_cm_s": k,
        "D_cm2_s": D,
        "sigma_um": sigma_um,
        "n_absorbed": int(n_abs_total),                       # all absorption events (see surf_counts)
    }


def _simulate_numba(grid, flow=None, D_cm2_s=5.6e-6, k_cm_s=1e-3, n_ref_per_um2=0.02,
                    t_burn_s=None, t_sample_s=None, dt_s=None, seed=0, n_tracks=0,
                    max_particles=2_000_000, init="linear", interp="nearest", wall="return",
                    phi=None, z_ion=2, T_C=22.0, reactive="first") -> dict:
    """Compiled-kernel twin of :func:`_simulate_numpy` (``simulate(..., backend="numba")``).

    Same model, same time step, same boundary rules; only the random stream differs.  The whole time
    loop runs inside one ``prange`` kernel: every particle carries a counter-based random stream keyed
    by ``(seed, step, slot)``, so the draws do not depend on the thread count (a run is reproducible
    on any machine), and absorptions/snapshots are accumulated in per-thread buffers that are summed
    at the end.  Particles live in a fixed preallocated array of ``capacity`` slots: the prange runs
    over the live prefix ``[0, n_alive)``, dead particles are compacted serially each step and the
    reservoir layer is refilled from a Poisson count drawn up front with numpy.  The capacity is
    ``1.35 * n_ref * dx^2 * n_fluid + margins`` (the count of a full c = 1 domain plus 6 sigma), so a
    run that would need more raises RuntimeError instead of overflowing.
    """
    kern = _nb_kernels()
    if kern is None:
        raise RuntimeError("numba is not importable")
    t_wall0 = time.perf_counter()
    if init not in ("linear", "empty"):
        raise ValueError("init must be 'linear' or 'empty'")
    if interp not in ("bilinear", "nearest"):
        raise ValueError("interp must be 'bilinear' or 'nearest'")
    if wall not in ("specular", "return"):
        raise ValueError("wall must be 'specular' or 'return'")
    if reactive not in ("first", "second"):
        raise ValueError("reactive must be 'first' or 'second'")
    g = grid
    dx = float(g.dx_um)
    nz, nx = g.shape
    pitch = float(g.pitch_um)
    D = float(D_cm2_s)
    k = float(k_cm_s)
    n_ref = float(n_ref_per_um2)
    if D <= 0.0 or n_ref <= 0.0:
        raise ValueError("D_cm2_s and n_ref_per_um2 must be positive")
    solid = np.ascontiguousarray(np.asarray(g.solid, dtype=bool))
    ze0 = float(g.z_um[0]) - 0.5 * dx
    z_top = ze0 + nz * dx                                   # = height_um for grids from geometry.py
    rng = np.random.default_rng(seed)
    drift = None if phi is None else _drift_velocity(phi, dx, nz, nx, D, z_ion, T_C)

    # ---- times, time step, reaction probability (identical formulas to the numpy path) --------
    H_cm = (float(g.height_um) + float(g.depth_um)) * 1e-4
    t_burn = (0.5 if init == "linear" else 1.5) * H_cm ** 2 / D if t_burn_s is None else float(t_burn_s)
    t_sample = t_burn if t_sample_s is None else float(t_sample_s)
    if t_burn < 0.0 or t_sample <= 0.0:
        raise ValueError("t_burn_s must be >= 0 and t_sample_s > 0")
    dt_diff = (0.35 * dx / _CM_TO_UM) ** 2 / (2.0 * D)      # rms free step = 0.35 dx [J]
    umax = 0.0
    if flow is not None:
        if tuple(flow.grid.shape) != (nz, nx) or flow.ux.shape != (nz, nx):
            raise ValueError("flow field shape does not match grid")
        umax = float(max(np.abs(flow.ux).max(), np.abs(flow.uz).max()) * _CM_TO_UM)
    if drift is not None:
        umax = max(umax, float(max(np.abs(drift[0]).max(), np.abs(drift[1]).max()) * _CM_TO_UM))
    dt_adv = np.inf if umax <= 0.0 else 0.5 * dx / umax      # |u|*dt <= 0.5 dx [J]
    dt = float(dt_s) if dt_s is not None else float(min(dt_diff, dt_adv))
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    sigma_um = float(np.sqrt(2.0 * D * dt) * _CM_TO_UM)
    n_burn = int(round(t_burn / dt))
    n_samp = max(1, int(round(t_sample / dt)))
    steps = n_burn + n_samp
    t_samp_eff = n_samp * dt
    if np.isinf(k):
        p_abs = 1.0
        p_first = 1.0
    else:
        p_first = float(np.clip(k * np.sqrt(np.pi * dt / D), 0.0, 1.0))
        p_abs = min(0.5 * k * np.sqrt(np.pi * dt / D), 1.0) if reactive == "second" else p_first
    surf = (np.ascontiguousarray(_surface_mask(solid, nx, nz)) if reactive == "second"
            else np.zeros((nz, nx), dtype=bool))
    d_dt = D * _CM_TO_UM * _CM_TO_UM * dt                   # D dt in um2 (Andrews-Bray term)

    zeros = np.zeros((nz, nx), dtype=float)
    if flow is None:
        ux = uz = zeros
    else:
        ux = np.ascontiguousarray(np.where(solid, 0.0, np.asarray(flow.ux, dtype=float)))
        uz = np.ascontiguousarray(np.where(solid, 0.0, np.asarray(flow.uz, dtype=float)))
    wx = zeros if drift is None else np.ascontiguousarray(drift[0])
    wz = zeros if drift is None else np.ascontiguousarray(drift[1])

    l_src = max(3.0 * sigma_um, dx)                         # >= ~3 rms steps, so fully renewed [J]
    n_src = n_ref * l_src * pitch                           # mean particles in the source layer
    x, z = _initial_particles(g, n_ref, init, ze0, dx, rng, max_particles)

    # ---- preallocate: capacity = a c = 1 domain + 6 sigma of counting noise + the source layer --
    n_fluid = int((~solid).sum())
    n_ub = n_ref * dx * dx * n_fluid
    cap = int(min(float(max_particles),
                  max(1.35 * n_ub + 6.0 * math.sqrt(n_ub) + 6.0 * math.sqrt(n_src) + 64.0,
                      4.0 * n_src, 1024.0)))
    if x.size > cap:
        raise ValueError(f"initial condition needs {x.size} particles > capacity {cap}; "
                         "reduce n_ref_per_um2 or raise max_particles")
    xb = np.zeros(cap, dtype=float)
    zb = np.zeros(cap, dtype=float)
    dead = np.zeros(cap, dtype=np.uint8)
    xb[:x.size] = x
    zb[:x.size] = z
    nnew = rng.poisson(n_src, size=steps).astype(np.int64)   # reservoir refills, one per step

    t_kernel = time.perf_counter()
    counts, cacc, n_snap, n_sum, n_alive = kern(
        xb, zb, dead, solid, surf, ux, uz, wx, wz, nnew, dx, pitch, ze0, z_top, sigma_um, dt,
        p_abs, l_src, n_burn, n_samp, nz, nx, int(x.size), d_dt, flow is not None,
        interp == "bilinear", wall == "specular", drift is not None, reactive == "second",
        np.uint64(seed))
    kernel_s = time.perf_counter() - t_kernel
    wall_s = time.perf_counter() - t_wall0
    n_snap = int(n_snap)
    n_alive = int(n_alive)

    c = cacc.reshape(nz, nx) / (n_ref * dx * dx * max(n_snap, 1))   # c = n / n_ref, time averaged
    c[solid] = np.nan
    s_iz, s_ix, s_nf = surface_cells(g)
    cnt = counts.reshape(nz, nx)[s_iz, s_ix]
    face_um = s_nf * dx
    flux_um_s = cnt / (n_ref * t_samp_eff * face_um)          # um/s (2-D: per um of trench length)
    return {
        "surf_iz": s_iz,
        "surf_ix": s_ix,
        "surf_x_um": np.asarray(g.x_um)[s_ix],
        "surf_z_um": np.asarray(g.z_um)[s_iz],
        "surf_flux_density": flux_um_s * 1e-4,                # cm/s
        "surf_counts": cnt,
        "c": c,
        "n_particles_mean": float(n_sum / max(n_snap, 1)),
        "steps": int(steps),
        "dt_s": dt,
        "t_sample_s": t_samp_eff,
        "P_abs": p_abs,
        "P_abs_first_order": p_first,
        "interp": interp,
        "wall": wall,
        "reactive": reactive,
        "backend": "numba",
        "migration": drift is not None,
        "wall_s": wall_s,
        "tracks": None,
        "x_final_um": xb[:n_alive],                           # final particle positions (for checks/plots)
        "z_final_um": zb[:n_alive],
        # bookkeeping (not required by the caller contract, handy for reports/CLI)
        "t_burn_s": n_burn * dt,
        "n_ref_per_um2": n_ref,
        "k_cm_s": k,
        "D_cm2_s": D,
        "sigma_um": sigma_um,
        "n_absorbed": int(cnt.sum()),                         # all absorption events (see surf_counts)
        "kernel_s": kernel_s,                                 # time inside the compiled kernel
        "capacity": int(cap),
        "n_snaps": int(n_snap),
    }


def region_means(result: dict, grid) -> dict:
    """Mean flux density over the field, the trench floor and the trench walls.

    'field'  surface cells with z above the deposit level and x outside the trench,
    'floor'  surface cells inside the open trench whose solid neighbour is below them,
    'wall'   the other surface cells inside the trench (side walls, and wall/floor corners counts as
             floor since their solid neighbour is below),
    plus the ratios to the field value and a Poisson standard error mean/sqrt(total absorbed count)
    for each group [C].
    """
    iz = np.asarray(result["surf_iz"], dtype=np.int64)
    ix = np.asarray(result["surf_ix"], dtype=np.int64)
    flux = np.asarray(result["surf_flux_density"], dtype=float)
    cnt = np.asarray(result["surf_counts"], dtype=float)
    x = np.asarray(grid.x_um, dtype=float)[ix]
    z = np.asarray(grid.z_um, dtype=float)[iz]

    inside = np.zeros(iz.size, dtype=bool)
    if float(grid.width_um) > 0.0:
        xl = float(grid.x_left_um)
        inside = (x > xl) & (x < xl + float(grid.width_um))
    below_solid = np.zeros(iz.size, dtype=bool)
    m = iz > 0                                                 # iz = 0 has no cell below
    below_solid[m] = np.asarray(grid.solid, dtype=bool)[iz[m] - 1, ix[m]]
    field = (~inside) & (z > float(grid.deposit_um))
    floor = inside & below_solid
    wall = inside & ~below_solid

    out: dict = {}
    for name, mask in (("field", field), ("floor", floor), ("wall", wall)):
        n_cells = int(mask.sum())
        tot = float(cnt[mask].sum())
        if n_cells == 0:
            mean = float("nan")
        else:
            mean = float(flux[mask].mean())
        out[name] = mean
        out[name + "_counts"] = tot
        out[name + "_cells"] = n_cells
        out[name + "_se"] = mean / np.sqrt(tot) if (tot > 0.0 and n_cells > 0) else float("nan")
    for a, b in (("floor", "field"), ("wall", "field")):
        base = out[b]
        out[a + "_over_" + b] = out[a] / base if base > 0.0 else float("nan")
    return out


def _cli(argv=None) -> int:
    """python -m platesim.trenchflow.particles --width 227 --depth 88 --k 1e-3 --dx 4"""
    from .geometry import trench_grid

    p = argparse.ArgumentParser(description="Brownian-tracer Cu2+ transport into a trench (2-D)")
    p.add_argument("--width", type=float, default=227.0, help="trench width, um")
    p.add_argument("--depth", type=float, default=88.0, help="trench depth, um")
    p.add_argument("--pitch", type=float, default=None, help="array pitch, um (default from trench_grid)")
    p.add_argument("--height", type=float, default=None, help="electrolyte height, um")
    p.add_argument("--dx", type=float, default=4.0, help="grid spacing, um")
    p.add_argument("--k", type=float, default=1e-3, help="surface reaction rate constant, cm/s")
    p.add_argument("--D", type=float, default=5.6e-6, help="Cu2+ diffusivity, cm2/s")
    p.add_argument("--n-ref", type=float, default=0.05, help="particles per um2 of bulk reference density")
    p.add_argument("--t-burn", type=float, default=5.0,
                   help="burn-in time, s (solver default: 0.5 H^2/D, slow for a large trench)")
    p.add_argument("--t-sample", type=float, default=5.0, help="sampling time, s (default = burn-in)")
    p.add_argument("--tracks", type=int, default=0, help="number of tagged particles to record")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    g = trench_grid(a.width, a.depth, pitch_um=a.pitch, height_um=a.height, dx_um=a.dx)
    t_diff = 0.5 * ((g.height_um + g.depth_um) * 1e-4) ** 2 / a.D      # [C] diffusion time of the cavity
    if a.t_burn < 0.5 * t_diff:
        print(f"note: --t-burn {a.t_burn:g} s is well below the diffusion time 0.5 H^2/D = {t_diff:.1f} s;"
              " the profile (and the fluxes below) is still transient.  Increase --t-burn for a"
              " converged answer.")
    res = simulate(g, flow=None, D_cm2_s=a.D, k_cm_s=a.k, n_ref_per_um2=a.n_ref,
                   t_burn_s=a.t_burn, t_sample_s=a.t_sample, seed=a.seed, n_tracks=a.tracks)
    rm = region_means(res, g)
    print(f"grid {g.shape[1]}x{g.shape[0]} cells, dx={g.dx_um:g} um, pitch={g.pitch_um:g} um, "
          f"height={g.height_um:g} um")
    print(f"steps {res['steps']}, dt={res['dt_s']:.3e} s, rms step {res['sigma_um']:.2f} um, "
          f"P_abs={res['P_abs']:.3g}, mean particles {res['n_particles_mean']:.0f}, "
          f"wall {res['wall_s']:.1f} s")
    print(f"sampled {res['t_sample_s']:.2f} s, absorbed {int(res['surf_counts'].sum())} particles")
    print(f"{'region':6s} {'flux [cm/s]':>12s} {'se':>10s} {'counts':>8s} {'cells':>6s} "
          f"{'ratio/field':>12s}")
    for name in ("field", "floor", "wall"):
        ratio = rm.get(name + "_over_field", float("nan"))
        print(f"{name:6s} {rm[name]:12.3e} {rm[name + '_se']:10.2e} {rm[name + '_counts']:8.0f} "
              f"{rm[name + '_cells']:6d} {ratio:12.3f}")
    if res["tracks"] is not None:
        print(f"tracks: {res['tracks'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


