"""Prescribed floor-vs-field exchange-current factor on a public synthetic trench.

Step 0 of the 3-D additives port as a first-class engine call: the factor is
passed through ``run_tile(kinetics_factor=...)`` instead of patching the solver.  The map is

    f(z) = f_deep      for z below (current mean field top - cut_um)
         = 1           otherwise

an *effective, uncalibrated* stand-in for a floor-depleted accelerator or a field-side suppressor
breakdown: a differential, not a uniform suppression.  The point of the probe is the sign and the
magnitude of the floor/field response.

Geometry (generic round numbers, not a measured sample): 200 um wide x 80 um deep rectangular
trench, field surface at z = 0, pitch 200 / 0.2 = 1000 um (open fraction 0.2), 4 periodic y cells,
dx 8 um, delta 64 um, 12 mA/cm2 for 4 h.  floor/field is the mean grown thickness of the floor columns
(|x| <= 57 um) over the far-field columns (|x| >= 300 um).

Usage::

    python -m tools.i0_factor_probe                 # f = 1.0, 0.85 with cut 25 um
    python -m tools.i0_factor_probe 1.0 0.85 0.7    # explicit f values
    python -m tools.i0_factor_probe --cut 15        # different depth cut (um)
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
WIDTH_UM, DEPTH_UM, F_OPEN = 200.0, 80.0, 0.2
DX_UM, DELTA_UM, J_MA, HOURS = 8.0, 64.0, 12.0, 4.0


def geometry():
    """(psi, grid, field_mask) of the synthetic trench, cm."""
    pitch_um = WIDTH_UM / F_OPEN
    nx = int(round(pitch_um / DX_UM))
    nx += nx % 2                                            # keep the trench centred on a cell edge
    ny = 4
    # z in [-96, 192] um: the field grows ~64 um (Mfac*j*4 h) plus the open-fraction excess, so the
    # moving Dirichlet plane (field top + delta) stays inside the domain (run_tile raises otherwise).
    nz = 36
    grid = Grid3D((nx, ny, nz), DX_UM * UM, (-0.5 * nx * DX_UM * UM, 0.0, -96 * UM))
    x, _, z = grid.meshgrid()
    half = 0.5 * WIDTH_UM * UM
    psi = np.where(z > 0.0, z, np.where(np.abs(x) < half,
                                       np.minimum(z + DEPTH_UM * UM, half - np.abs(x)), z))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    field_mask = np.abs(x[:, :, 0]) >= 300 * UM
    return psi, grid, field_mask


def floor_over_field(psi0, psi1, grid):
    x = grid.centers()[0]
    grown = (top_height(psi1, grid) - top_height(psi0, grid)) / UM
    floor = float(np.mean(grown[np.abs(x) <= 57 * UM]))
    field = float(np.mean(grown[np.abs(x) >= 300 * UM]))
    return dict(floor_um=floor, field_um=field, ratio=floor / field)


def run_factor(f_deep, cut_um=25.0, n_frames=3, linear="auto", progress=None):
    psi0, grid, field_mask = geometry()

    def factor(z_cm, z_plane_cm, field_top_cm):
        return np.where(z_cm < field_top_cm - cut_um * UM, f_deep, 1.0)

    t0 = time.time()
    out = run_tile(psi0, grid, J_MA, HOURS * 3600.0, delta_um=DELTA_UM, params=Params(),
                   field_mask=field_mask, n_frames=n_frames, kinetics_factor=None
                   if f_deep == 1.0 else factor, linear=linear, progress=progress)
    row = floor_over_field(psi0, out["psi"], grid)
    row.update(f_deep=f_deep, cut_um=cut_um, wall_s=time.time() - t0, steps=out["summary"]["steps"])
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("factors", nargs="*", type=float, default=None)
    ap.add_argument("--cut", type=float, default=25.0, help="depth below the field top (um)")
    ap.add_argument("--linear", default="auto", choices=("auto", "direct", "amg", "mg"))
    args = ap.parse_args(argv)
    factors = args.factors or [1.0, 0.85]
    psi0, grid, _ = geometry()
    print(f"grid {grid.shape} dx {DX_UM} um  pitch {WIDTH_UM / F_OPEN:.0f} um  "
          f"j {J_MA} mA/cm2  delta {DELTA_UM} um  {HOURS} h")
    for f in factors:
        row = run_factor(f, cut_um=args.cut, linear=args.linear)
        print(f"f={row['f_deep']:.3f} cut={row['cut_um']:.0f} um: floor {row['floor_um']:.3f} um  "
              f"field {row['field_um']:.3f} um  floor/field {row['ratio']:.4f}  "
              f"({row['steps']} steps, {row['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
