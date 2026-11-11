"""D2Q9 lattice-Boltzmann shear flow over a periodic trench (feature-scale, platesim.trenchflow).

Physical picture: the stirred beaker drives an electrolyte shear flow u_x = G * z far above the wafer
(G = wall shear rate, 10-200 1/s). Over a 100-1000 um wide, ~90 um deep trench the cavity Reynolds number
G * w^2 / nu is 0.05-200, so the flow is laminar and steady [C].

Scheme (all [J]):
- D2Q9 BGK collision, SRT. tau is selected for Reynolds matching, see _scaling().
- Full-way bounce-back on solid nodes == halfway bounce-back (wall lies halfway between the last fluid
  node centre and the solid node centre). On a flat wall that puts the wall at z = 0 while the first
  fluid row centre sits at z = dx/2; the top lid sits exactly at z = height_um (one virtual solid row is
  appended above the grid and halfway bounce-back puts the wall at (height + dx/2 + (height - dx/2))/2).
- Moving wall: bounce-back with the Ladd momentum correction, f_j(x_w) = f_i(x_w) - 6 w_i (c_i . u_w),
  j = opposite(i), applied only to the appended lid row. rho_w = 1 (weakly compressible, Ma <~ 0.1).
- x is periodic (np.roll); z is bounded, so streaming uses slicing with zero inflow at the open faces.
- Initial field: the analytic Couette profile of the open region, which removes most of the spin-up.

Units: lattice units inside the solver, cm/s and um outside (geometry.Conventions). The converged
lattice field is rescaled by (u_top / u_lid_lb); for a Reynolds-matched steady laminar flow that is the
exact similarity scaling, and it is also exact for Stokes flow where inertia is negligible [C].

Sign of the driving: G = 0 is the only resting case. For G < 0 the flow is the mirror image of the G > 0
flow: the Reynolds scaling uses |G| (identical Re, tau, u_lid), the lid is driven in -x, and the lattice
field is mirror-mapped back (ux -> -ux[:, ::-1], uz -> uz[:, ::-1]). Solving the mirrored geometry with
the lid in +x and mapping back is exact by symmetry (bit-identical mirror), and it leaves every
arithmetic operation of the G > 0 path untouched.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .geometry import CavityGrid, FlowField, trench_grid

# --- D2Q9 lattice ----------------------------------------------------------------
C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=np.int64)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

_CX = C[:, 0][:, None, None].astype(np.float64)
_CZ = C[:, 1][:, None, None].astype(np.float64)
_W = W[:, None, None]

TAU_MIN = 0.51          # keep nu_lb > 0 by a safe margin
TAU_MAX = 2.0           # heavier relaxation than this is not a useful lattice
LID_MAX = 0.1           # low-Mach ceiling for a lid velocity
LID_DEFAULT = 0.05


def _equilibrium(rho, ux, uz):
    """Second-order equilibrium f_i = w_i rho (1 + 3 c.u + 9/2 (c.u)^2 - 3/2 u^2)."""
    cu = 3.0 * (_CX * ux + _CZ * uz)
    usq = 1.5 * (ux * ux + uz * uz)
    return _W * rho * (1.0 + cu + 0.5 * cu * cu - usq)


def _macro(f):
    """Density and velocity; uz is the velocity normal to the wafer plane."""
    rho = f.sum(0)
    ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
    uz = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
    return rho, ux, uz


def _stream(fpost):
    """Stream all 9 populations: periodic in x (axis 1), one-sided (zero inflow) in z (axis 0)."""
    out = np.empty_like(fpost)
    for i in range(9):
        cx, cz = int(C[i, 0]), int(C[i, 1])
        g = fpost[i]
        if cx:
            g = np.roll(g, cx, axis=1)
        if cz > 0:
            out[i, 1:, :] = g[:-1, :]
            out[i, 0, :] = 0.0
        elif cz < 0:
            out[i, :-1, :] = g[1:, :]
            out[i, -1, :] = 0.0
        else:
            out[i] = g
    return out


def _step(f, solid, tau, u_wall=0.0):
    """One BGK collide + stream. `solid` includes the appended moving-lid row as its last row."""
    rho = f.sum(0)
    ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
    uz = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
    feq = _equilibrium(rho, ux, uz)
    fpost = f - (f - feq) * (1.0 / tau)
    # no-slip: reflect the populations that just arrived at solid nodes (full-way bounce-back)
    fpost[:, solid] = f[OPP][:, solid]
    if u_wall:
        # moving lid: momentum correction on the populations that travel back into the fluid
        for i in (5, 6, 7, 8):
            fpost[OPP[i], -1, :] = f[i, -1, :] - 6.0 * W[i] * (C[i, 0] * u_wall)
    return _stream(fpost), ux, uz


def _scaling(grid, shear_rate_s, nu_cm2_s, u_lid_lb, tau_guess):
    """Reynolds matching: nu_lb = u_lid * N_H / Re, tau = 3 nu_lb + 1/2.

    Re = u_top H / nu must be identical in lattice and physical units, because the normalised velocity
    field is a function of Re and geometry only (steady laminar flow). u_lid_lb is the free parameter:
    if the required tau leaves [TAU_MIN, TAU_MAX] we rescale u_lid so that Re stays matched, which is
    what actually fixes the physics; tau then follows from the rescaled u_lid. For our regime
    (Re ~ 0.05-200, N_H ~ 20-70) the required tau is almost always > TAU_MAX: a lattice cannot be
    viscous enough at u_lid <= 0.1, so tau is clamped at TAU_MAX and u_lid is lowered to keep Re. With
    Re << 1 the flow is the Stokes solution, which does not vary with Re at all, so the clamp is benign.

    Only |shear_rate_s| enters: Re, tau and u_lid are identical for +G and -G, and the returned
    ``u_top`` is the speed magnitude at z = height_um (the caller re-applies the sign).
    """
    h_um = grid.height_um - grid.deposit_um
    h_cm = max(h_um, 1e-12) * 1e-4
    u_top = abs(float(shear_rate_s)) * h_cm      # cm/s at z = height_um (speed magnitude)
    n_h = float(max(1, int(np.count_nonzero(grid.z_um >= grid.deposit_um))))
    re = u_top * h_cm / nu_cm2_s
    nu_min = (TAU_MIN - 0.5) / 3.0
    nu_max = (TAU_MAX - 0.5) / 3.0
    u_lid = float(u_lid_lb)
    re_clamped = False
    if not np.isfinite(re) or re <= 0.0:
        nu_lb = min(max((tau_guess - 0.5) / 3.0, nu_min), nu_max)
    else:
        nu_lb = u_lid * n_h / re
        if nu_lb > nu_max:
            nu_lb = nu_max
            u_lid = min(u_lid, re * nu_lb / n_h)
            re_clamped = True
        elif nu_lb < nu_min:
            nu_lb = nu_min
            u_lid = re * nu_lb / n_h
            re_clamped = u_lid > LID_MAX
    if u_lid > LID_MAX:
        u_lid = LID_MAX
        re_clamped = True
    u_lid = max(u_lid, 1e-9)
    tau = 3.0 * nu_lb + 0.5
    re_sim = u_lid * n_h / nu_lb if nu_lb > 0 else 0.0
    return {"u_top": u_top, "h_um": h_um, "re": float(re), "re_sim": float(re_sim),
            "nu_lb": float(nu_lb), "tau": float(tau), "u_lid_lb": float(u_lid),
            "n_fluid_rows": n_h, "re_clamped": bool(re_clamped)}


def solve_flow(grid: CavityGrid, shear_rate_s: float, nu_cm2_s: float = 0.0114, tau: float = 0.9,
               max_steps: int = 200000, tol: float = 1e-5, check_every: int = 500,
               u_lid_lb: float = 0.05) -> FlowField:
    """Steady D2Q9 BGK flow over `grid` driven by the far-field shear `shear_rate_s` [1/s].

    nu_cm2_s: kinematic viscosity, default 0.0114 cm2/s [J: dilute CuSO4/H2SO4, ~22 C].
    tau: fallback/initial guess only; the relaxation time used is the one from _scaling().
    Returns a FlowField with ux, uz in cm/s, exactly 0.0 in solid cells.

    shear_rate_s == 0 (or non-finite) gives a fluid at rest. A negative shear rate gives the mirror
    image of the positive-shear flow: the same speed magnitude with the lid moving in -x. The scaling
    uses |G|, the lattice problem is solved on the mirrored geometry with the lid in +x, and the field
    is mapped back (ux -> -ux[:, ::-1], uz -> uz[:, ::-1]).
    """
    t0 = time.time()
    sc = _scaling(grid, abs(shear_rate_s), nu_cm2_s, u_lid_lb, tau)
    u_lid = sc["u_lid_lb"]
    u_top = sc["u_top"]                          # speed magnitude at z = height_um
    mirror = bool(np.isfinite(shear_rate_s) and shear_rate_s < 0.0)
    if mirror:
        u_top = -u_top                           # report the lid speed with the sign of G
    nz, nx = grid.shape

    key = dict(steps=0, tau=sc["tau"], u_lid_lb=u_lid, re=sc["re"], re_sim=sc["re_sim"],
               max_mach=0.0, wall_s=0.0, residual=float("inf"), u_top_cm_s=u_top,
               nu_lb=sc["nu_lb"], re_clamped=sc["re_clamped"], n_fluid_rows=sc["n_fluid_rows"],
               scheme="D2Q9-BGK/srt, halfway bounce-back, lid momentum correction")

    if not np.isfinite(shear_rate_s) or shear_rate_s == 0.0:      # no driving: trivially at rest
        z = np.zeros(grid.shape)
        return FlowField(grid=grid, ux=z.copy(), uz=z.copy(), shear_rate_s=float(shear_rate_s),
                         converged=True, info=dict(key, wall_s=time.time() - t0, residual=0.0))

    # lattice domain: the grid plus one virtual solid row for the moving lid at z = height_um.
    # G < 0 is solved on the x-mirrored geometry with the lid still moving in +x; the feedback is a
    # pure index reversal of the solid mask, so the negative-shear result is the exact mirror of the
    # positive-shear one (the lattice scheme is reflection-equivariant about x).
    solid = np.ones((nz + 1, nx), bool)
    solid[:nz] = grid.solid[:, ::-1] if mirror else grid.solid
    zz = np.broadcast_to(grid.z_um[:, None], grid.shape)

    # start from the analytic Couette profile of the open region (exact solution without the trench)
    ux0 = np.zeros((nz + 1, nx))
    ux0[:nz] = np.where(solid[:nz], 0.0,
                        u_lid * np.clip(zz, 0.0, grid.height_um) / max(grid.height_um, 1e-30))
    f = _equilibrium(np.ones_like(ux0), ux0, np.zeros_like(ux0))
    ux_prev, uz_prev = ux0[:nz], np.zeros(grid.shape)
    ux = uz = np.zeros((nz + 1, nx))
    ref = max(abs(u_lid_lb), 1e-12)
    residual, converged, steps = float("inf"), False, 0

    while steps < max_steps:
        n = min(int(check_every), max_steps - steps)
        for _ in range(n):
            f, ux, uz = _step(f, solid, sc["tau"], u_lid)
        steps += n
        if not (np.isfinite(f).all() and np.isfinite(ux).all()):
            raise FloatingPointError("LBM diverged; raise TAU_MIN or lower u_lid_lb")
        du = max(np.abs(ux[:nz] - ux_prev).max(), np.abs(uz[:nz] - uz_prev).max())
        residual = float(du / ref)
        ux_prev, uz_prev = ux[:nz].copy(), uz[:nz].copy()
        if residual < tol:
            converged = True
            break

    # lattice -> physical: exact similarity scaling for a Reynolds-matched steady laminar flow
    scale = sc["u_top"] / u_lid                  # speed magnitude (sign is carried by ux_f/uz_f)
    if mirror:                                   # mirror back into the original x orientation
        ux_f, uz_f = -ux_prev[:, ::-1], uz_prev[:, ::-1]
    else:
        ux_f, uz_f = ux_prev, uz_prev
    ux_cm = np.where(grid.solid, 0.0, ux_f * scale)
    uz_cm = np.where(grid.solid, 0.0, uz_f * scale)
    speed = np.where(grid.solid, 0.0, np.hypot(ux_f, uz_f))
    key.update(steps=steps, wall_s=time.time() - t0, residual=residual, converged=converged,
               max_mach=float(speed.max() * np.sqrt(3.0)), dx_um=grid.dx_um)
    return FlowField(grid=grid, ux=ux_cm, uz=uz_cm, shear_rate_s=float(shear_rate_s),
                     converged=bool(converged), info=key)


def penetration_depth_um(flow: FlowField, frac: float = 0.01) -> float:
    """Depth (um below the wafer plane, positive) where |u| on the trench centreline first drops below
    `frac` times the outer-flow speed at 100 um above the wall, u(100 um) = G * 100 um [C].

    Returns grid.depth_um - deposit_um when the flow never falls below the threshold (deep penetration),
    and 0.0 for a flat grid. Solid rows are skipped (their |u| is identically 0)."""
    g = flow.grid
    available = g.depth_um - g.deposit_um
    if available <= 0.0 or g.width_um <= 0.0:
        return 0.0
    ix = int(np.argmin(np.abs(g.x_um - 0.5 * g.pitch_um)))       # trench centre column
    speed = np.hypot(flow.ux[:, ix], flow.uz[:, ix])
    thresh = frac * abs(flow.shear_rate_s) * 100.0 * 1e-4         # cm/s (speed magnitude)
    rows = np.nonzero((g.z_um < 0.0) & (~g.solid[:, ix]))[0]
    for iz in rows[np.argsort(-g.z_um[rows])]:                    # scan downwards from z = 0
        if speed[iz] < thresh:
            return float(min(-g.z_um[iz], available))
    return float(available)


def eddy_summary(flow: FlowField) -> dict:
    """Trench-interior speed, recirculation flag and outer-flow penetration depth.

    Recirculation is flow *opposite to the lid*, so the threshold is sign(G) * ux < -1e-3 |u_top|; the
    flag is therefore identical for +G and -G (the two flows are mirror images)."""
    g = flow.grid
    below = (g.z_um[:, None] < 0.0) & (~g.solid)
    if not below.any():
        return {"u_max_in_trench_cm_s": 0.0, "has_recirculation": False, "penetration_um": 0.0}
    speed = np.hypot(flow.ux, flow.uz)
    sign = float(np.sign(flow.shear_rate_s))                     # lid direction for the recirculation test
    u_top = abs(flow.shear_rate_s) * (g.height_um - g.deposit_um) * 1e-4
    return {
        "u_max_in_trench_cm_s": float(speed[below].max()),
        "has_recirculation": bool((sign * flow.ux[below] < -1e-3 * u_top).any()),
        "penetration_um": float(penetration_depth_um(flow)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="D2Q9 shear flow over a periodic trench")
    p.add_argument("--width", type=float, default=227.0, help="trench width (um)")
    p.add_argument("--depth", type=float, default=88.0, help="trench depth (um)")
    p.add_argument("--shear", type=float, default=80.0, help="wall shear rate G (1/s)")
    p.add_argument("--dx", type=float, default=4.0, help="lattice spacing (um)")
    p.add_argument("--pitch", type=float, default=None, help="x period (um), default as trench_grid")
    p.add_argument("--height", type=float, default=None, help="electrolyte height (um)")
    p.add_argument("--nu", type=float, default=0.0114, help="kinematic viscosity (cm2/s)")
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--tol", type=float, default=1e-5)
    a = p.parse_args(argv)
    g = trench_grid(a.width, a.depth, pitch_um=a.pitch, height_um=a.height, dx_um=a.dx)
    flow = solve_flow(g, a.shear, nu_cm2_s=a.nu, max_steps=a.max_steps, tol=a.tol)
    print(f"grid: {g.shape[1]} x {g.shape[0]} nodes, dx = {g.dx_um} um, "
          f"pitch = {g.pitch_um:.0f} um, height = {g.height_um:.0f} um")
    print(f"shear G = {a.shear} 1/s, nu = {a.nu} cm2/s, u_top = {flow.info['u_top_cm_s']:.4g} cm/s")
    for k in ("steps", "tau", "u_lid_lb", "re", "re_sim", "max_mach", "wall_s", "residual", "re_clamped"):
        print(f"  {k:12s} {flow.info[k]}")
    print("converged:", flow.converged)
    s = eddy_summary(flow)
    print("eddy_summary:")
    for k, v in s.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":        # pragma: no cover
    raise SystemExit(main())
