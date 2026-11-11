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
``ux``, ``uz`` are sampled at the particle's nearest cell (``flow=None`` -> pure diffusion).
This is the Euler-Maruyama / random-walk solution of dc/dt + u.grad c = D lap c with a reaction
boundary; the 1e4 converts cm to um (D in cm2/s, sqrt(2 D dt) in cm).

Boundaries
- x is periodic with period ``grid.pitch_um`` (the trench array repeats).
- top reservoir: the bulk concentration is imposed at z = ``grid.height_um`` by keeping the particle
  density in the source layer [height - L_src, height] at the reference density n_ref (particles per
  um2) -- every step the layer is emptied and refilled uniformly with Poisson(n_ref*L_src*pitch)
  particles.  This realises the Dirichlet condition c = 1 at the top boundary.  Then c = n/n_ref.
- solid wall (a step that ends in a solid cell): the particle hit the Cu/Si surface.  With probability
  P_abs it is absorbed (deposited) and removed from the system, otherwise it is reflected -- the
  simplest robust rule, return to the position at the start of the step.  For a first-order surface
  reaction with rate constant k (cm/s):

      P_abs = k * sqrt(pi * dt / D)          (clipped to [0, 1]; k = inf -> 1)

  [FT] R. Erban and S. J. Chapman, Phys. Biol. 4 (2007) 16-28, "Reactive boundary conditions for
  stochastic simulations of reaction-diffusion processes": the partially absorbing boundary condition
  D dc/dn = k c is implemented as a reaction probability per encounter; the expression above is their
  first-order form, valid for P_abs << 1, i.e. rms step << D/k and << dx.

Accuracy note [C]: the "return to previous position" reflection offsets the effective absorbing plane
by ~0.6 * rms_step below the true wall (a boundary layer of one step thickness is compressed into the
reflecting rule rather than resolved), which biases the measured flux low by ~0.6*sigma/H -- about 1 %
for a 1 um rms step and H = 80 um.  If a flat-wall validation shows more bias than that, shorten dt
(smaller rms step) rather than change the physics.

Flux estimator: every absorption is recorded in the electrolyte surface cell the particle came FROM
(the cell of its position before the step), accumulated only after the burn-in.  For a surface cell
with f solid faces (face_length = f * dx_um)

    flux_density [um/s] = absorbed_count / (n_ref * t_sample * f * dx_um),   [cm/s] = 1e-4 * [um/s]

Conventions (see geometry.py): arrays are [iz, ix] with iz = 0 the bottom row, z = 0 the original wafer
plane, solid True = Si or Cu, x periodic with period pitch_um.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .geometry import CavityGrid, FlowField, surface_cells  # noqa: F401  (FlowField: type of `flow`)

__all__ = ["simulate", "region_means"]

_CM_TO_UM = 1e4


def _cell_of(z, x, ze0, dx, nx, nz):
    """Nearest-cell indices [iz, ix] of continuous positions (um); x periodic with period nx*dx.

    Cell (iz, ix) is centred at (x_um[ix], z_um[iz]), so its edges are at ..., ze0 + k*dx (z = 0 is
    an edge for every grid built by geometry.py).  A position on an edge belongs to the upper cell.
    """
    iz = np.floor((z - ze0) / dx).astype(np.int64)
    np.clip(iz, 0, nz - 1, out=iz)
    ix = np.floor(x / dx).astype(np.int64) % nx
    return iz, ix


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
             init="linear") -> dict:
    """Run the Brownian-tracer solver on ``grid``; see the module docstring for the model.

    Returns a dict with the surface-cell arrays (same order as ``geometry.surface_cells``), the flux
    density per surface cell (cm/s), the raw absorbed counts, the time-averaged concentration estimate
    c (NaN in solid), run bookkeeping ('n_particles_mean', 'steps', 'dt_s', 't_sample_s', 'P_abs',
    'wall_s') and the tagged tracks (n_tracks, n_saved, 2) or None.
    """
    t_wall0 = time.perf_counter()
    if init not in ("linear", "empty"):
        raise ValueError("init must be 'linear' or 'empty'")
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
    else:
        # [FT] Erban & Chapman, Phys. Biol. 4 (2007) 16-28: reaction probability per encounter.
        p_abs = float(np.clip(k * np.sqrt(np.pi * dt / D), 0.0, 1.0))

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
            xn = x_old + (flow.ux[iz0, ix0] * dt) * _CM_TO_UM + sigma_um * xi[:n]
            zn = z_old + (flow.uz[iz0, ix0] * dt) * _CM_TO_UM + sigma_um * xi[n:]
        xn %= pitch                                         # periodic in x
        np.minimum(zn, z_top, out=zn)                       # reservoir reflects anything above the top
        iz1, ix1 = _cell_of(zn, xn, ze0, dx, nx, nz)
        hit = solid[iz1, ix1]

        # ---- solid encounter: absorb with P_abs, else return to the previous position
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
            if reflect.any():                               # a step that would enter the wall is undone
                xn[reflect] = x_old[reflect]
                zn[reflect] = z_old[reflect]
            if absorbed.size:
                if sampling:                                # record in the cell it came from
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


