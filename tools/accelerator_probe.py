"""Transported-accelerator probe: floor/field of a narrow and a wide trench (3-D additives phase 2).

Question: a *prescribed* floor-vs-field exchange-current factor (``tools/i0_factor_probe.py``) lowers
floor/field in a narrow trench but lowers it just as much in a wide one.  A transport-limited accelerator
(consumed at the front, depleted where the supply path is long) should act mainly in narrow, deep
trenches and barely under a wide-open field.  This tool runs that model and reports floor/field for a
narrow and a wide generic trench at 3-4 accelerator strengths.  Nothing here is fitted to measured data.

Model (``additives='acc'``, ``platesim.tile3d``): steady bulk diffusion of the normalised accelerator
``alpha = c_a/c_acc`` with a surface consumption sink ``Gamma_acc k_acc (1-theta_a) alpha`` and the
quasi-steady coverage ``theta_a = k_acc c_acc alpha / (k_acc c_acc alpha + k_inc_a v)``; the coverage
multiplies the local exchange current by ``1 + b_acc theta_a``.  The strength knob is a multiplier on
``Gamma_acc`` (the new, uncalibrated adsorbate capacity): it changes only how transport-limited the
supply is, not the equilibrium coverage.

Geometry (generic round numbers): rectangular trench, pitch = width / 0.2 (open fraction 0.2), 4 periodic y
cells, dx 8 um, delta 64 um, field surface at z = 0.  floor/field is the mean grown thickness of the
floor columns over the far-field columns at the end of the run.

Usage::

    python -m tools.accelerator_probe                     # strengths 0 (off), 0.25, 0.5, 1, 2
    python -m tools.accelerator_probe --strength 1        # one strength, both geometries
    python -m tools.accelerator_probe --cases narrow --hours 1.5
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from platesim.feature.engine.params import Params
from platesim.tile3d import levelset3d as ls
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.sim import run_tile, top_height

UM = 1e-4
MFAC = 3.676e-5                     # cm3/C, growth per charge (Params.derive Mfac at efficiency 1)
DX_UM, DELTA_UM, NY, F_OPEN = 8.0, 64.0, 4, 0.2
GAMMA_ACC0 = 1.0e-10                # mol/cm2, Params.Gamma_acc default (the strength reference)

# name: (width_um, depth_um, j_mA_cm2, hours) -- the two public synthetic trenches of the phase-2 scope
CASES = {
    "narrow": dict(width_um=200.0, depth_um=80.0, j_mA_cm2=12.0, hours=4.0),
    "wide": dict(width_um=800.0, depth_um=80.0, j_mA_cm2=12.0, hours=4.0),
}
TARGETS = {}  # no measured targets in the public tool


def geometry(width_um=200.0, depth_um=80.0, j_mA_cm2=12.0, hours=4.0, dx_um=DX_UM, ny=NY):
    """(psi, grid, field_mask, floor_mask) of one synthetic trench, cm.

    The z extent holds the whole deposit history (field thickness = j*hours*Mfac/f_open) plus a delta
    layer and 40 um of headroom for the moving Dirichlet plane.
    """
    pitch_um = width_um / F_OPEN
    nx = int(round(pitch_um / dx_um))
    nx += nx % 2                                     # keep the trench centred on a cell edge
    z_lo_um = -(depth_um + dx_um)
    z_lo_um = -dx_um * int(np.ceil(abs(z_lo_um) / dx_um))
    field_um = (j_mA_cm2 * 1e-3) * (hours * 3600.0) * MFAC / (1.0 - F_OPEN) * 1e4
    nz = int(np.ceil((field_um + DELTA_UM + 40.0 - z_lo_um) / dx_um))
    grid = Grid3D((nx, ny, nz), dx_um * UM, (-0.5 * nx * dx_um * UM, 0.0, z_lo_um * UM))
    x, _, z = grid.meshgrid()
    half = 0.5 * width_um * UM
    psi = np.where(z > 0.0, z, np.where(np.abs(x) < half,
                                       np.minimum(z + depth_um * UM, half - np.abs(x)), z))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    field_mask = np.abs(x[:, :, 0]) >= (half + 0.15 * pitch_um * UM)
    floor_mask = np.abs(x[:, :, 0]) <= 0.25 * width_um * UM
    return psi, grid, field_mask, floor_mask


def floor_over_field(psi0, psi1, grid, field_mask, floor_mask):
    grown = (top_height(psi1, grid) - top_height(psi0, grid)) / UM
    with np.errstate(invalid="ignore", divide="ignore"):
        floor = float(np.nanmean(np.where(floor_mask, grown, np.nan)))
        field = float(np.nanmean(np.where(field_mask, grown, np.nan)))
    return dict(floor_um=floor, field_um=field, ratio=floor / field)


def run_case(name, strength, gamma_acc=GAMMA_ACC0, hours=None, n_frames=3, linear="auto", progress=None,
             ny=None, dx_um=None, **over):
    """One trench at one accelerator strength (multiplier on Gamma_acc; 0 = additive-free)."""
    c = dict(CASES[name])
    if hours is not None:
        c["hours"] = float(hours)
    ny = NY if ny is None else int(ny)
    dx_um = DX_UM if dx_um is None else float(dx_um)
    psi0, grid, field_mask, floor_mask = geometry(c["width_um"], c["depth_um"], c["j_mA_cm2"], c["hours"],
                                                  dx_um=dx_um, ny=ny)
    if strength == 0.0:
        p = Params(suppressor=False)
        mode = "off"
    else:
        p = Params(additives="acc", suppressor=False, Gamma_acc=float(strength) * gamma_acc, **over)
        mode = "acc"
    t0 = time.time()
    out = run_tile(psi0, grid, c["j_mA_cm2"], c["hours"] * 3600.0, delta_um=DELTA_UM, params=p,
                   field_mask=field_mask, n_frames=n_frames, linear=linear, progress=progress)
    wall = time.time() - t0
    row = floor_over_field(psi0, out["psi"], grid, field_mask, floor_mask)
    row.update(case=name, mode=mode, strength=float(strength),
               Gamma_acc=float(gamma_acc * strength if strength else 0.0), hours=c["hours"],
               wall_s=wall, steps=out["summary"]["steps"],
               info=out["summary"].get("additives"))
    if mode == "acc":
        ii, jj, kk = _front_cells(out["psi"])
        th_a = out["fields"]["theta_a"][ii, jj, kk]
        a_acc = out["fields"]["a_acc"][ii, jj, kk]
        row["theta_a_floor"] = float(np.mean(th_a[~field_mask]))
        row["theta_a_field"] = float(np.mean(th_a[field_mask]))
        row["a_floor"] = float(np.mean(a_acc[~field_mask]))
        row["a_field"] = float(np.mean(a_acc[field_mask]))
        row["i0_floor"] = float(np.mean(out["fields"]["i0_factor"][ii, jj, kk][~field_mask]))
        row["i0_field"] = float(np.mean(out["fields"]["i0_factor"][ii, jj, kk][field_mask]))
        row["outer"] = float(out["series"]["outer_iterations"][-1])
        row["additives"] = out["summary"].get("additives")
    return row


def _front_cells(psi):
    from platesim.tile3d.additives3d import front_cells
    return front_cells(psi)


def _take(a, ii, jj, kk):
    return None if a is None else a[ii, jj, kk]


def _fmt(row):
    tgt = TARGETS.get(row["case"])
    verdict = ""
    if tgt is not None and row["mode"] != "off":
        verdict = "  in-target" if tgt[0] <= row["ratio"] <= tgt[1] else "  OUT"
    acc = ""
    if row["mode"] == "acc":
        acc = (f"  theta_a {row['theta_a_floor']:.4f}/{row['theta_a_field']:.4f}"
               f"  a {row['a_floor']:.4f}/{row['a_field']:.4f}")
    return (f"{row['case']:6s} strength {row['strength']:4.2f} ({row['mode']:3s}): "
            f"floor {row['floor_um']:7.2f} um  field {row['field_um']:7.2f} um  "
            f"floor/field {row['ratio']:.4f}{verdict}{acc}  "
            f"[{row['steps']} steps, {row['wall_s']:.0f} s]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strength", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0],
                    help="multipliers on Params.Gamma_acc (0 = additive-free baseline)")
    ap.add_argument("--cases", nargs="+", default=["narrow", "wide"], choices=sorted(CASES))
    ap.add_argument("--hours", type=float, default=None, help="override the plated time (h)")
    ap.add_argument("--dx", type=float, default=None, help="override dx_um")
    ap.add_argument("--ny", type=int, default=None, help="periodic y cells (default 4)")
    ap.add_argument("--linear", default="auto", choices=("auto", "direct", "amg", "mg"))
    ap.add_argument("--json", default=None, help="write the rows to this JSON file")
    args = ap.parse_args(argv)
    if args.dx is not None:
        global DX_UM
        DX_UM = float(args.dx)
    ny = NY if args.ny is None else int(args.ny)
    rows = []
    print(f"dx {DX_UM} um, delta {DELTA_UM} um, {ny} periodic y cells, gamma_acc0 {GAMMA_ACC0:.2e} mol/cm2")
    for name in args.cases:
        c = CASES[name]
        print(f"\n== {name}: {c['width_um']:.0f} x {c['depth_um']:.0f} um, pitch "
              f"{c['width_um'] / F_OPEN:.0f} um, j {c['j_mA_cm2']} mA/cm2, {args.hours or c['hours']} h", flush=True)
        for s in args.strength:
            row = run_case(name, s, hours=args.hours, linear=args.linear, ny=ny, dx_um=DX_UM,
                           progress=lambda f, m: None)
            rows.append(row)
            print(_fmt(row), flush=True)
            inf = row.get("additives")
            if inf:
                print(f"       outer {row['outer']:.0f} passes, additive converged "
                      f"{inf.get('converged')} (cov {inf.get('cov_change'):.1e}, "
                      f"acc {inf.get('acc_change', 0):.1e}), a_balance {inf.get('a_balance_rel'):+.1e}, "
                      f"a_min_front {inf.get('a_min_front'):.4f}, "
                      f"i0 {row.get('i0_floor', float('nan')):.2f}/{row.get('i0_field', float('nan')):.2f}",
                      flush=True)
    print("\n== summary (floor/field)")
    for row in rows:
        tgt = TARGETS.get(row["case"])
        flag = ""
        if tgt is not None and row["mode"] != "off":
            flag = "in" if tgt[0] <= row["ratio"] <= tgt[1] else "OUT"
        print(f"{row['case']:6s} {row['strength']:5.2f}  {row['ratio']:.4f}  {flag}")
    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([{k: v for k, v in r.items() if k != "info"} | {"info": r["info"]} for r in rows],
                      fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
