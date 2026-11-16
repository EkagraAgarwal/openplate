"""Time loop of one 3-D tile (physics level L1): quasi-steady fields, explicit level-set advance.

Each step: solve the fields on the current front (fields.FieldSolver), extend the growth speed off the
front, rescale it so the deposited volume rate equals the Faraday rate of the set current exactly,
advance the level set, and write frames as they are produced.  Units: cm, s, A.
"""
from __future__ import annotations

import time

import numpy as np

from . import levelset3d as ls
from .fields import FieldSolver
from .grid import Grid3D
from .result import TileResult, TileResultWriter


def top_height(psi, grid: Grid3D):
    """Height (cm) of the first front crossing coming down from the top, per (x, y) column."""
    z = grid.centers()[2]
    solid = psi <= 0
    k = psi.shape[2] - 1 - np.argmax(solid[:, :, ::-1], axis=2)        # highest solid cell
    k = np.clip(k, 0, psi.shape[2] - 2)
    ii, jj = np.meshgrid(np.arange(psi.shape[0]), np.arange(psi.shape[1]), indexing="ij")
    a, b = psi[ii, jj, k], psi[ii, jj, k + 1]                          # a <= 0 < b
    frac = np.where(b > a, -a / np.where(b > a, b - a, 1.0), 0.0)
    h = z[k] + np.clip(frac, 0.0, 1.0) * grid.dx
    return np.where(solid.any(axis=2), h, z[0] - 0.5 * grid.dx)


def run_tile(psi_si, grid: Grid3D, j_net_mA_cm2, duration_s, delta_um=64.0, params=None, T_C=22.0,
             run_dir=None, n_frames=10, dt_max=120.0, cfl=0.3, reinit_every=4, scheme="weno5",
             backend="auto", field_mask=None, progress=None, meta=None, volume_correct=False, flow=None,
             linear="auto", pulse=None, frame_callback=None):
    """Plate the bare geometry psi_si for duration_s at the net current density j_net_mA_cm2.

    pulse: None (legacy) or dict(t_f_ms, t_r_ms, t_off_ms, j_f_mA_cm2,
    j_r_mA_cm2, model='resolved'|'net'). Reverse j is a positive magnitude;
    the waveform average must equal j_net_mA_cm2. model='net' ignores timing.
    frame_callback: optional (time_s, psi, scalars) observer at every saved frame;
    psi is a borrowed array (copy it if retaining). Resolved summary['pulse']
    contains the last frozen-front solve's phase/charging diagnostics.
    field_mask: bool (nx, ny) columns that count as "field" for the bulk-plane height; default: columns
    whose bare surface lies within one cell of the highest bare surface.
    flow: None (no convection) or dict with any of shear_rate_s (0), shear_dir ((1, 0), in-plane),
    gravity_dir (None = no buoyancy; e.g. (0, -1, 0) for gravity along -y), beta_L_mol (0.15, solutal
    expansion per mol/L Cu2+, the 2-D engine's default [J]: ~1 M CuSO4 is ~15 % denser than water),
    nu_cm2_s (0.0114), update_every (5 steps), lid_um (None: the moving lid sits at the bulk plane;
    a number: fixed lid height above the initial mean field height, e.g. the 2-D engine's domain top -
    the grid must reach it, and flow then crosses the bulk plane), axial_net_flux_zero (None, passed to
    solve_stokes3d).  Steady Stokes (platesim.flow3d), recomputed every update_every steps from the
    latest Cu2+ field (lagged one solve) and used to advect Cu2+ in the field solve.
    Returns a TileResult when run_dir is given, else a dict with the final psi and the series."""
    # None/model='net' retains the exact legacy numerical path.
    solver = FieldSolver(params, T_C, linear=linear, pulse=pulse)
    fl = None if flow is None else dict(dict(shear_rate_s=0.0, shear_dir=(1.0, 0.0), gravity_dir=None,
                                             beta_L_mol=0.15, nu_cm2_s=0.0114, update_every=5, lid_um=None,
                                             axial_net_flux_zero=None), **flow)
    flow_faces, flow_info = None, {}
    p, ph = solver.p, solver.ph
    dx, per = grid.dx, grid.periodic
    z0 = grid.origin[2]
    j = j_net_mA_cm2 * 1e-3
    area = grid.shape[0] * grid.shape[1] * dx * dx
    psi = np.asarray(psi_si, dtype=np.float64).copy()
    h0 = top_height(psi, grid)
    if field_mask is None:
        field_mask = h0 > h0.max() - dx
    solid0 = ls.solid_volume(psi, dx)
    writer = None
    if run_dir is not None:
        info = dict(backend="numpy/" + backend, physics="L1", dx_um=dx * 1e4, T_C=T_C,
                    j_net_mA_cm2=j_net_mA_cm2, delta_um=delta_um, duration_s=duration_s)
        info.update(meta or {})
        writer = TileResultWriter(run_dir, grid, psi.astype(np.float32), meta=info)
    frame_times = np.linspace(0.0, duration_s, max(int(n_frames), 2))
    series = {k: [] for k in ("t_s", "charge_C_cm2", "solid_volume_cm3", "faraday_volume_cm3", "V",
                              "cu_balance_rel", "ct_min_front", "outer_iterations", "flow_umax_um_s")}
    t, step, state, next_frame, charge = 0.0, 0, None, 0, 0.0
    wall0 = time.time()

    def save(out):
        nonlocal next_frame
        vol = ls.solid_volume(psi, dx) - solid0
        row = dict(charge_C_cm2=charge, solid_volume_cm3=vol,
                   faraday_volume_cm3=charge * area * ph.Mfac, V=out["V"],
                   cu_balance_rel=out["info"]["cu_balance_rel"],
                   ct_min_front=out["info"]["ct_min_front"],
                   outer_iterations=float(out["info"]["outer"]),
                   flow_umax_um_s=float(flow_info.get("umax_cm_s", 0.0)) * 1e4)
        series["t_s"].append(t)
        for key, value in row.items():
            series[key].append(value)
        if writer is not None:
            fields = dict(c_cu=(out["ct"] * p.c_bulk).astype(np.float32), phi=out["phi"].astype(np.float32),
                          speed=out["speed"].astype(np.float32))
            writer.add_frame(t, psi.astype(np.float32), fields=fields, scalars=row)
        if frame_callback is not None:
            frame_callback(t, psi, row)
        next_frame += 1

    while True:
        z_plane = float(np.mean(top_height(psi, grid)[field_mask])) + delta_um * 1e-4
        if fl is not None and (flow_faces is None or step % int(fl["update_every"]) == 0):
            # flow from the latest Cu2+ field (one solve behind); the very first one from a no-flow solve
            if state is None or flow_faces is None:
                prev = solver.solve(psi, dx, z0, z_plane, j, periodic=per, state=state, flow_faces=flow_faces)
                state, ct_prev = prev["state"], prev["ct"]
            from ..flow3d import buoyancy_force, solve_stokes3d
            body = None
            if fl["gravity_dir"] is not None:
                body = buoyancy_force(ct_prev, p.c_bulk * 1e3, fl["beta_L_mol"], tuple(fl["gravity_dir"]))
            z_lid = z_plane if fl["lid_um"] is None else float(np.mean(h0[field_mask])) + fl["lid_um"] * 1e-4
            fsol = solve_stokes3d(psi, dx, z0, z_lid, periodic=per, shear_rate_s=fl["shear_rate_s"],
                                  shear_dir=tuple(fl["shear_dir"]), body_force=body, nu_cm2_s=fl["nu_cm2_s"],
                                  axial_net_flux_zero=fl["axial_net_flux_zero"])
            flow_faces = fsol["faces"]
            flow_info = dict(umax_cm_s=max(float(np.max(np.abs(fsol[k]))) for k in ("ux", "uy", "uz")))
        out = solver.solve(psi, dx, z0, z_plane, j, periodic=per, state=state, flow_faces=flow_faces)
        state, ct_prev = out["state"], out["ct"]
        if next_frame < len(frame_times) and t >= frame_times[next_frame] - 1e-9:
            save(out)
        if t >= duration_s - 1e-9:
            break
        # growth speed: extend the interface value off the front, then enforce the Faraday volume rate
        speed = ls.extend(psi, out["speed"], dx, periodic=per)
        rate = float(np.sum(out["w"] * speed) * dx ** 3)              # cm3/s the front would sweep
        target = j * area * ph.Mfac                                    # cm3/s the current deposits
        speed = speed * (target / rate) if rate > 0 else speed
        vmax = float(np.max(np.abs(speed[np.abs(psi) < 3 * dx]))) if step >= 0 else 0.0
        dt = min(dt_max, cfl * dx / max(vmax, 1e-30), duration_s - t)
        if next_frame < len(frame_times):
            dt = min(dt, max(frame_times[next_frame] - t, 1e-9))
        psi = ls.advect(psi, speed, dt, dx, periodic=per, scheme=scheme, backend=backend)
        t += dt
        charge += j * dt
        step += 1
        if step % reinit_every == 0:
            psi = ls.reinit(psi, dx, periodic=per)
        if volume_correct:
            # Off by default: it hides the scheme's own volume drift (reported as faraday_error_rel).
            psi = ls.volume_correct(psi, solid0 + charge * area * ph.Mfac, dx)
        if progress is not None:
            progress(min(t / duration_s, 1.0), f"step {step}, t = {t:.0f} s")
    summary = dict(steps=step, wall_s=time.time() - wall0,
                   faraday_error_rel=(series["solid_volume_cm3"][-1] / series["faraday_volume_cm3"][-1] - 1.0
                                       if series["faraday_volume_cm3"][-1] > 0 else 0.0))
    if 'pulse' in out['info']:
        summary['pulse'] = out['info']['pulse']
    if writer is not None:
        writer.close(dict(summary=summary))
        return TileResult(run_dir)
    return dict(psi=psi, series=series, summary=summary, grid=grid)
