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
             backend="auto", field_mask=None, progress=None, meta=None, volume_correct=False):
    """Plate the bare geometry psi_si for duration_s at the net current density j_net_mA_cm2.

    field_mask: bool (nx, ny) columns that count as "field" for the bulk-plane height; default: columns
    whose bare surface lies within one cell of the highest bare surface.
    Returns a TileResult when run_dir is given, else a dict with the final psi and the series."""
    solver = FieldSolver(params, T_C)
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
                              "cu_balance_rel", "ct_min_front", "outer_iterations")}
    t, step, state, next_frame, charge = 0.0, 0, None, 0, 0.0
    wall0 = time.time()

    def save(out):
        nonlocal next_frame
        vol = ls.solid_volume(psi, dx) - solid0
        row = dict(charge_C_cm2=charge, solid_volume_cm3=vol,
                   faraday_volume_cm3=charge * area * ph.Mfac, V=out["V"],
                   cu_balance_rel=out["info"]["cu_balance_rel"],
                   ct_min_front=out["info"]["ct_min_front"],
                   outer_iterations=float(out["info"]["outer"]))
        series["t_s"].append(t)
        for key, value in row.items():
            series[key].append(value)
        if writer is not None:
            fields = dict(c_cu=(out["ct"] * p.c_bulk).astype(np.float32), phi=out["phi"].astype(np.float32),
                          speed=out["speed"].astype(np.float32))
            writer.add_frame(t, psi.astype(np.float32), fields=fields, scalars=row)
        next_frame += 1

    while True:
        z_plane = float(np.mean(top_height(psi, grid)[field_mask])) + delta_um * 1e-4
        out = solver.solve(psi, dx, z0, z_plane, j, periodic=per, state=state)
        state = out["state"]
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
    if writer is not None:
        writer.close(dict(summary=summary))
        return TileResult(run_dir)
    return dict(psi=psi, series=series, summary=summary, grid=grid)
