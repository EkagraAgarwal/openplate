"""Synthetic L1 V6/V7 comparisons; run from root with PYTHONPATH=.

No solver patches, volume correction, additives, flow or measured data. Default
runs every case; --fast runs only the narrow 0.5 h trench and step-zero check.
Grid rounds requested dx to fit the exact 2-D pitch; both solvers use that dx.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import contourpy
import numpy as np
from scipy.ndimage import map_coordinates

from platesim.feature.engine import levelset as ls2
from platesim.feature.engine.fields import StepGeom, StepSolver
from platesim.feature.engine.params import Params
from platesim.feature.engine.recipes import dc
from platesim.feature.engine.sim import ViaSim
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d import levelset3d as ls3
from platesim.tile3d.sim import run_tile, top_height


def setup(width=227., depth=88., dx=8., hours=.5, via=False):
    # [C] Sufficient headroom even for 4.5 h at the imposed Faraday rate.
    p = Params(suppressor=False, geometry="via" if via else "trench_full",
               H_um=depth, dx_um=dx, delta_um=64., flow_shear_s=0., k_mix=0.,
               efficiency=1., transport="enhanced", max_overburden_um=200.)
    sim = ViaSim(width, 22., p, .05 if via else .2, dc(15., hours))
    g = sim.g
    if not via:
        psi = np.repeat(ls2.initial_psi(g, p.corner_cells)[:, None, :], 4, axis=1)
        grid = Grid3D(psi.shape, g.dx, (0., 0., g.z_lo - g.H))
        field = np.broadcast_to((np.abs(g.rc[:, 0] - g.x_c) > g.a)[:, None], psi.shape[:2])
    else:
        # Quarter square: symmetry at the via centre and half-pitch boundaries
        # is the reflection reduction of a full square periodic unit cell.
        half_side = np.sqrt(np.pi) * g.R / 2
        n = int(round(half_side / g.dx))
        bc = dict(x_lo="symmetry", x_hi="symmetry", y_lo="symmetry",
                  y_hi="symmetry", z_lo="bottom", z_hi="bulk")
        grid = Grid3D((n, n, g.nz), g.dx, (0., 0., g.z_lo - g.H), bc)
        x, y, _ = grid.centers()
        r = np.hypot(x[:, None], y[None, :])
        # Revolve the reference rounded initial radial distance, not a voxel mask.
        psi = np.empty(grid.shape)
        for k in range(g.nz):
            psi[:, :, k] = np.interp(r, g.rc[:, 0], sim.psi[:, k])
        field = r > g.a
    return sim, psi, grid, field


def heights(psi, g):
    return np.array([ls2.column_crossing(col, g.zc[0], upper=True)[1] for col in psi])


def row(name, a, b, target=None, relative=False):
    difference = (b / a - 1) if relative else b - a
    return dict(quantity=name, two_d=float(a), three_d=float(b), difference=float(difference),
                target=target, difference_kind="relative" if relative else "absolute",
                passed=None if target is None else bool(abs(difference) <= target))


@lru_cache(maxsize=None)
def compare_trench(width=227., depth=88., dx=8., hours=.5):
    sim, psi0, grid, field = setup(width, depth, dx, hours)
    g = sim.g
    h0 = heights(sim.psi, g)
    raw2 = sim.run("vs3d")
    raw3 = run_tile(psi0, grid, 15., hours * 3600, params=sim.p, field_mask=field,
                    n_frames=2)
    if raw2['stop_reason'] != 'recipe_end' or abs(raw2['t_end'] - hours * 3600) > 1e-6:
        raise RuntimeError(f"2-D stopped early: {raw2['stop_reason']} at {raw2['t_end']} s")
    h2 = heights(sim.psi, g)
    h3 = top_height(raw3['psi'], grid).mean(axis=1) + g.H
    off = np.abs(g.rc[:, 0] - g.x_c)
    far = off > g.a + 150e-4
    if not far.any():
        far = off > g.a + .25 * (.5 * g.R - g.a)
    centre = g.nr // 2
    f2, f3 = ((h - h0)[far].mean() * 1e4 for h in (h2, h3))
    b2, b3 = ((h[centre] - h0[centre]) * 1e4 for h in (h2, h3))
    error = (h3 - h2) * 1e4
    volume3 = (ls3.solid_volume(raw3['psi'], grid.dx) - ls3.solid_volume(psi0, grid.dx)) / (4 * grid.dx)
    rows = [row("field_um", f2, f3, .005, True), row("floor_centre_um", b2, b3),
            row("floor/field", b2 / f2, b3 / f3, .003),
            row("volume_um2_per_unit_length", raw2['solid_gain'] * 1e8, volume3 * 1e8),
            row("profile_max_um", 0., np.max(np.abs(error))),
            row("profile_rms_um", 0., np.sqrt(np.mean(error ** 2)))]
    return dict(case=f"trench_{width:g}x{depth:g}_dx{dx:g}_{hours:g}h", rows=rows,
                actual_dx_um=g.dx_um, shape=list(grid.shape),
                faraday_error_2d=raw2['faraday_error_rel'],
                faraday_error_3d=raw3['summary']['faraday_error_rel'])


@lru_cache(maxsize=None)
def compare_step0():
    sim, psi, grid, field = setup()
    g, p = sim.g, sim.p
    plane2 = sim.face_height() + 64e-4
    geo = StepGeom(g, sim.psi, plane2, p)
    solver = StepSolver(geo, p, sim.ph, np.zeros_like(sim.psi))
    ctrl = dict(mode="I", I_t=.015 * g.area_cell, duty_f=1., duty_r=0., rr=0., pr=False)
    out2 = solver.solve(sim.st, ctrl, 120.)
    plane3 = float(top_height(psi, grid)[field].mean()) + 64e-4
    out3 = FieldSolver(p, 22.).solve(psi, grid.dx, grid.origin[2], plane3, .015)
    current2 = np.zeros_like(sim.psi)
    current2[geo.act] = out2['i_f']
    # Sample ALL psi=0 contour segments, including vertical walls, with identical
    # bilinear interpolation. Column heights alone would omit those walls.
    lines = contourpy.contour_generator(x=g.rc[:, 0], y=g.zc[0], z=sim.psi.T).lines(0.)
    points = np.concatenate(lines)
    coords = np.array([points[:, 0] / g.dx - .5,
                       (points[:, 1] - g.z_lo) / g.dx - .5])
    arrays = (("surface_c/c_bulk", sim.st['ctil'], out3['ct'][:, 0, :]),
              ("surface_j_A_cm2", current2, out3['i'][:, 0, :]))
    rows, profiles = [], {}
    for name, a, b in arrays:
        a, b = (map_coordinates(v, coords, order=1, mode="nearest") for v in (a, b))
        rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-30)
        rows.append(row(name + "_max_relative", 0., rel.max(), .01))
        profiles[name] = dict(two_d=a.tolist(), three_d=b.tolist())
    return dict(case="narrow_step0", rows=rows, profiles=profiles,
                front_x_um=(points[:, 0] * 1e4).tolist(),
                front_z_um=((points[:, 1] - g.H) * 1e4).tolist(),
                bulk_plane_difference_um=(plane3 + g.H - plane2) * 1e4,
                solver_info_2d=out2['info'], solver_info_3d=out3['info'])


@lru_cache(maxsize=None)
def compare_via():
    sim, psi0, grid, field = setup(100., 100., 4., .5, via=True)
    g = sim.g
    h0 = heights(sim.psi, g)
    h03 = top_height(psi0, grid)
    raw2 = sim.run("via_vs3d")
    raw3 = run_tile(psi0, grid, 15., 1800., params=sim.p, field_mask=field, n_frames=2)
    if raw2['stop_reason'] != 'recipe_end':
        raise RuntimeError(f"2-D via stopped early: {raw2['stop_reason']}")
    x, y, z = grid.centers()
    r = np.hypot(x[:, None], y[None, :])
    # Same fractional-cell fill convention as Grid.via_w, normalised by the
    # actual initial electrolyte volume (not nominal pi*a^2*H).
    radial = r < g.a + .5 * grid.dx
    wz = np.clip((np.minimum(z + .5 * grid.dx, 0.) -
                  np.maximum(z - .5 * grid.dx, -g.H)) / grid.dx, 0., 1.)
    weight = radial[:, :, None] * wz[None, None, :]
    e0 = ls3.electrolyte_fraction(psi0, grid.dx)
    e = ls3.electrolyte_fraction(raw3['psi'], grid.dx)
    fill3 = float(np.sum((e0 - e) * weight) / np.sum(e0 * weight))
    h2 = heights(sim.psi, g)
    h3 = top_height(raw3['psi'], grid)
    f2 = float((h2[-1] - h0[-1]) * 1e4)
    f3 = float((h3 - h03)[r > g.a + 100e-4].mean() * 1e4)
    b2 = float((h2[0] - h0[0]) * 1e4)
    b3 = float((h3[0, 0] - h03[0, 0]) * 1e4)
    return dict(case="via_100x100_dx4_0.5h", rows=[
        row("fill_fraction", raw2['fill_final'], fill3, .01, True),
        row("field_um", f2, f3, .005, True), row("floor_centre_um", b2, b3),
        row("floor/field", b2 / f2, b3 / f3)], actual_dx_um=g.dx_um,
        shape=list(grid.shape), square_area_rounding_relative=4 * (grid.shape[0] * grid.dx) ** 2 / g.area_cell - 1,
        faraday_error_2d=raw2['faraday_error_rel'], faraday_error_3d=raw3['summary']['faraday_error_rel'])


def print_table(result):
    print(f"\n{result['case']}", flush=True)
    print("quantity | 2-D | 3-D | difference | target | pass", flush=True)
    for r in result['rows']:
        target = "report only" if r['target'] is None else f"{r['target']:.6g}"
        passed = "n/a" if r['passed'] is None else str(r['passed'])
        print(f"{r['quantity']} | {r['two_d']:.8g} | {r['three_d']:.8g} | "
              f"{r['difference']:+.6g} ({r['difference_kind']}) | {target} | {passed}", flush=True)


def annotate(result):
    """Keep interpretation with the measurements, explicitly distinguishing hypotheses."""
    notes = ["Synthetic data; DC 15 mA/cm2, 22 C, delta 64 um, efficiency 1; no flow/additives."]
    if result['case'] == 'narrow_step0':
        notes += ["Surface samples include floor, field and vertical walls; differences are pointwise relative maxima.",
                  "Default 2-D edge and 3-D field-mean bulk planes differ slightly after initial corner rounding."]
    else:
        notes += ["Field/floor thicknesses subtract the actual initial psi=0 heights, not nominal silicon heights.",
                  "Native volume quadratures differ: 2-D normal-aware linear ramp; 3-D cosine Heaviside.",
                  "Front profile uses the highest psi=0 crossing in each column; max/RMS have no specified target."]
        failed = [r['quantity'] for r in result['rows'] if r['passed'] is False]
        if failed:
            notes += ["Target gaps: " + ', '.join(failed) + ". Step-zero trench fields meet 1%; growth discrepancies are consistent with different velocity extension/rescaling, Godunov1 versus WENO5, bulk-plane rules, and redistancing. These mechanisms have not been individually isolated by ablations."]
        if result['case'].startswith('via'):
            notes += ["Via is a quarter of a square periodic cell, not the axisymmetric circular cell; lateral supply geometry differs.",
                      "Equal-area square side is rounded to an integer number of the SAME effective dx used by 2-D; square_area_rounding_relative records the unavoidable area mismatch.",
                      "Via fill uses each engine's native volume fraction and approximate boundary-cell feature weights; the 1% result includes this quadrature uncertainty."]
    result['diagnosis'] = notes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--annotate-existing", action="store_true",
                        help="Print and annotate the saved measurements without rerunning solvers")
    args = parser.parse_args()
    dest = Path(__file__).with_name("compare_3d_2d_results.json")
    if args.annotate_existing:
        results = [annotate(r) for r in json.loads(dest.read_text(encoding="utf-8"))]
        for result in results:
            print_table(result)
            for note in result['diagnosis']:
                print(note)
        dest.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        return
    results = []
    jobs = [compare_step0, compare_trench]
    if not args.fast:
        jobs += [lambda w=w, d=d, dx=dx, h=h: compare_trench(w, d, dx, h)
                 for w, d in ((227., 88.), (944., 97.))
                 for dx, h in ((8., .5), (8., 1.), (4., 1.), (8., 4.5))
                 if (w, dx, h) != (227., 8., .5)]
        jobs += [compare_via]
    for job in jobs:
        result = annotate(job())
        print_table(result)
        results.append(result)
        dest.write_text(json.dumps(results, indent=2, allow_nan=False, default=lambda v: v.item()), encoding="utf-8")


if __name__ == "__main__":
    main()
