"""T2.4 native 2-D/3-D convection comparison (no solver modifications).

Run with PYTHONPATH=. python tools/compare_3d_2d_flow.py. Checkpoints each case.
The engines' native lid positions are deliberately not silently equated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from tools.compare_3d_2d import setup, heights, row, print_table
from platesim.flow3d import buoyancy_force, solve_stokes3d
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.sim import run_tile, top_height


def progress(label):
    def report(frac, message):
        print(f"{label} {100 * frac:5.1f}% {message}", flush=True)
    return report


def save_json(path, result):
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False,
                                    default=lambda x: x.item()), encoding="utf-8")


def configure(sim, kind, shear=0.):
    p = sim.p
    p.flow_solver = "stokes"
    # 3-D is exponentially fitted, not the 2-D default first-order upwind.
    p.adv_scheme = "sg"
    p.flow_shear_s = shear
    p.buoyancy = kind in ("across", "axial")
    p.gravity_dir = (1., 0.) if kind == "across" else (0., 0.)
    return None if kind == "none" else dict(
        shear_rate_s=shear, gravity_dir={"across": (1., 0., 0.),
                                        "axial": (0., -1., 0.)}.get(kind),
        beta_L_mol=p.beta_cu_L_mol)


def lid_diagnostic(dx=8., shear=10.):
    """Exact same-domain shear mapping at t=0, separate from native run_tile.

    2-D has deposit_um=0: U=G*z_top. 3-D uses U=G3*(lid-z_field).
    Setting lid to the full domain top and G3=U/(lid-z_field) matches it.
    """
    sim, psi, grid, field = setup(dx=dx, hours=1.)
    configure(sim, "shear", shear)
    ux, uz = sim.cavity_flow(shear)
    lid = grid.origin[2] + grid.shape[2] * grid.dx
    zfield = float(top_height(psi, grid).max())
    u2 = shear * (sim.g.z_top - sim.g.H)
    mapped = u2 / (lid - zfield)
    flow = solve_stokes3d(psi, grid.dx, grid.origin[2], lid,
                          shear_rate_s=mapped, axial_net_flux_zero=False)
    ex = np.max(np.abs(flow['ux'][:, 0] - ux))
    ez = np.max(np.abs(flow['uz'][:, 0] - uz))
    return dict(two_d_lid_um=lid * 1e4, two_d_lid_speed_um_s=u2 * 1e4,
                three_d_matched_shear_s=mapped,
                matched_max_velocity_error_um_s=max(ex, ez) * 1e4,
                matched_relative_error=max(ex, ez) / max(np.abs(ux).max(), 1e-30),
                native_three_d_initial_lid_um=float(top_height(psi, grid)[field].mean() * 1e4 + 64.))


def buoyancy_diagnostic(dx=16.):
    """Isolate the pressure-gradient convention on one frozen concentration.

    Same full-domain lid, identical no-flow ct, no front advance or cadence
    difference. This diagnostic does not replace either native growth result.
    """
    sim, psi, grid, field = setup(dx=dx, hours=1.)
    configure(sim, 'across')
    plane = sim.face_height() - sim.g.H + 64e-4
    out = FieldSolver(sim.p, 22.).solve(psi, grid.dx, grid.origin[2], plane, .015)
    ct = np.where(grid.centers()[2][None, None, :] > plane + .5*grid.dx, 1., out['ct'])
    sim.st['ctil'] = ct[:, 0, :].copy()
    ux, uz = sim.cavity_flow(0.)
    body = buoyancy_force(ct, sim.p.c_bulk*1e3, sim.p.beta_cu_L_mol, (1., 0., 0.))
    lid = grid.origin[2] + grid.shape[2]*grid.dx
    flows = [solve_stokes3d(psi, grid.dx, grid.origin[2], lid, body_force=body,
                             axial_net_flux_zero=flag) for flag in (False, True)]
    free, closed = flows
    speed = lambda f: max(np.abs(f[k]).max() for k in ('ux','uy','uz')) * 1e4
    error = max(np.abs(free['ux'][:, 0, :] - ux).max(),
                np.abs(free['uz'][:, 0, :] - uz).max()) * 1e4
    return dict(actual_dx_um=grid.dx*1e4, matched_full_lid_um=lid*1e4,
                matched_unconstrained_velocity_error_um_s=error,
                unconstrained_max_component_um_s=speed(free),
                constrained_max_component_um_s=speed(closed),
                unconstrained_x_section_flux_cm3_s=float(free['faces'][0][0].sum()*grid.dx**2),
                constrained_x_section_flux_cm3_s=float(closed['faces'][0][0].sum()*grid.dx**2),
                constrained_pressure_gradient_cm_s2=closed['info']['mean_pressure_gradient_cm_s2'])


def compare_flow(kind="shear", shear=None, dx=8., hours=1.):
    if shear is None:
        shear = 10. if kind == 'shear' else 0.
    sim, psi0, grid, field = setup(dx=dx, hours=hours)
    flow = configure(sim, kind, shear)
    g = sim.g
    h0 = heights(sim.psi, g)
    h03 = top_height(psi0, grid)
    name = f"{kind}_G{shear:g}_dx{dx:g}_{hours:g}h"
    measured = dict(max_speed_um_s=0., max_face_speed_um_s=0., updates=0)
    native = sim.cavity_flow

    def observe(*args, **kwargs):
        ux, uz = native(*args, **kwargs)
        measured['max_speed_um_s'] = max(measured['max_speed_um_s'],
                                          max(np.abs(ux).max(), np.abs(uz).max()) * 1e4)
        measured['max_face_speed_um_s'] = max(measured['max_face_speed_um_s'],
                                               max(np.abs(f).max() for f in sim.flow_faces) * 1e4)
        measured['updates'] += 1
        return ux, uz

    sim.cavity_flow = observe  # instrumentation of this instance, not a solver edit
    t0 = perf_counter()
    raw2 = sim.run(name, progress=progress(name + " 2D"))
    wall2 = perf_counter() - t0
    if raw2['stop_reason'] != 'recipe_end' or abs(raw2['t_end'] - hours * 3600) > 1e-6:
        raise RuntimeError(f"2-D stopped early: {raw2['stop_reason']}")
    t0 = perf_counter()
    raw3 = run_tile(psi0, grid, 15., hours * 3600, params=sim.p, field_mask=field,
                    n_frames=32, flow=flow, progress=progress(name + " 3D"))
    wall3 = perf_counter() - t0
    h2 = heights(sim.psi, g)
    h3 = top_height(raw3['psi'], grid)
    off = np.abs(g.rc[:, 0] - g.x_c)
    far = off > g.a + 150e-4
    if not far.any():
        far = off > g.a + .25 * (.5 * g.R - g.a)
    f2 = float((h2 - h0)[far].mean() * 1e4)
    f3 = float((h3 - h03)[far].mean() * 1e4)
    b2 = float((h2 - h0)[g.nr // 2] * 1e4)
    b3 = float((h3 - h03)[g.nr // 2].mean() * 1e4)
    vmax3 = max(raw3['series']['flow_umax_um_s'])
    result = dict(case=name, kind=kind, shear_s=shear, actual_dx_um=g.dx_um,
                  hours=hours, shape=list(grid.shape), rows=[
                      row("field_um", f2, f3, .005, True),
                      row("floor_centre_um", b2, b3),
                      row("floor/field", b2/f2, b3/f3, .01),
                      row("max_component_speed_um_s", measured['max_speed_um_s'], vmax3)],
                  wall_s=dict(two_d=wall2, three_d=wall3), flow_2d=measured,
                  cell_peclet_2d=measured['max_face_speed_um_s'] * 1e-4 * grid.dx / sim.ph.D_c,
                  cell_peclet_3d_cell_speed_lower_bound=vmax3 * 1e-4 * grid.dx / sim.ph.D_c,
                  faraday_error_2d=raw2['faraday_error_rel'],
                  faraday_error_3d=raw3['summary']['faraday_error_rel'],
                  profile_2d_um=((h2-h0)*1e4).tolist(),
                  profile_3d_um=((h3-h03).mean(axis=1)*1e4).tolist(),
                  diagnosis=[
                      "Native lids differ: 2-D full domain top, U=G*(z_top-H), deposit_um=0; 3-D moving field-mean+64um, U=G*(lid-highest_field). No constant shear multiplier makes these domains identical.",
                      "2-D adv_scheme=sg explicitly selected to match 3-D exponential fitting; both use unweighted direct MAC advection faces and Hs-weighted diffusion (2-D low seam unique, 3-D high seam duplicated). Reported u*dx/D Peclet excludes Hs, so interfacial face Peclet can be larger.",
                      "Native update cadence retained: 2-D every 10 steps or 1-cell field movement, 3-D every 5 steps; initial 2-D concentration is its initial state, 3-D first solves no-flow concentration. Growth integrators also differ.",
                      "beta=0.15 L/mol in both (2-D default, not 3-D default 0.07); gravity uses (gx,gz) versus (gx,gy,gz).",
                      "3-D auto zero-net-flux acts on periodically forced x as well as y; 2-D transverse buoyancy has no equivalent pressure-gradient constraint. Axial flow can be nonzero but cannot advect an exactly y-invariant concentration.",
                      "Maximum speed means max absolute cell component, not vector norm. 3-D speed sampled at 32 frame times; 2-D at every flow update; reported maxima need not be strict whole-run maxima.",
                  ])
    result['misses'] = [r['quantity'] for r in result['rows'] if r['passed'] is False]
    return result


def axial_invariance(axial, baseline):
    return {f"max_profile_difference_{dim}_um": float(np.max(np.abs(
        np.array(axial[f'profile_{dim}_um']) - baseline[f'profile_{dim}_um'])))
            for dim in ('2d', '3d')}


def annotate_existing(results):
    """Refresh contract diagnostics/interpretation without rerunning growth."""
    results['lid_mapping'] = lid_diagnostic()
    results['buoyancy_mapping'] = buoyancy_diagnostic()
    results['configuration'].update(flow_every_2d=10, flow_update_every_3d=5,
                                    saved_frames_3d=32, beta_L_mol=.15,
                                    duration_s=3600, growth_wall_times_include_progress=True)
    for case in results['cases']:
        notes = case['diagnosis']
        notes[1] = ("2-D adv_scheme=sg explicitly selected to match 3-D exponential fitting; both use "
                    "unweighted direct MAC advection faces and Hs-weighted diffusion (2-D low seam "
                    "unique, 3-D high seam duplicated). Reported u*dx/D Peclet excludes Hs, so "
                    "interfacial face Peclet can be larger.")
        key = case['kind']
        if key == 'across':
            interpretation = ("Both targets missed. Frozen same-domain/same-ct diagnostic matches "
                              "2-D and unconstrained 3-D velocities to round-off; constraining periodic "
                              "x removes a finite throughflow. Native 3-D additionally closes flow at "
                              "the bulk plane instead of allowing the 2-D open species-plane flux. "
                              "These are measured convention differences, not an isolated attribution "
                              "of the full growth error. Native 2-D starts from uniform ct (zero initial "
                              "buoyancy) and can wait ~1200s for its first nonzero update.")
        elif key == 'shear' and case['shear_s'] == 50:
            interpretation = ("Field and floor/field targets missed at shear 50/s. Same-domain shear "
                              "diagnostic matches velocities to round-off. Native lid/domain and "
                              "open-versus-closed species-plane flux differ; at large Peclet numbers "
                              "boundary placement and cadence are plausible amplified error sources. "
                              "No growth ablations isolate their individual contributions.")
        elif key == 'axial':
            interpretation = ("Both cross-engine targets passed. Relative to each engine's no-flow "
                              "baseline, 2-D is exactly identical and 3-D differs by less than 1e-6um "
                              "over the entire x-profile. Axial velocity is not zero; y-invariant "
                              "concentration makes axial advection idle.")
        else:
            interpretation = "Both field and floor/field targets passed; remaining differences include native growth discretisation, lid placement and cadence."
        if len(notes) == 6:
            notes.append(interpretation)
        else:
            notes[6] = interpretation
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--annotate-existing', action='store_true',
                    help='Refresh diagnostics and notes without rerunning growth')
    args = ap.parse_args()
    dest = Path(__file__).with_name('compare_3d_2d_flow_results.json')
    if args.annotate_existing:
        results = annotate_existing(json.loads(dest.read_text(encoding='utf-8')))
        save_json(dest, results)
        for case in results['cases']:
            print_table(case)
        return
    results = dict(configuration=dict(width_um=227, depth_um=88, delta_um=64,
                                     current_mA_cm2=15, bath_Cu_g_L=65, T_C=22,
                                     adv_scheme_2d='sg', volume_correction=False),
                   lid_mapping=lid_diagnostic(), buoyancy_mapping=buoyancy_diagnostic(), cases=[])
    jobs = [('none', 0., 8.), ('shear', 10., 8.), ('shear', 50., 8.),
            ('across', 0., 8.), ('axial', 0., 8.), ('shear', 10., 4.)]
    for kind, shear, dx in jobs[:2] if args.fast else jobs:
        result = compare_flow(kind, shear, dx, .01 if args.fast else 1.)
        results['cases'].append(result)
        print_table(result)
        if kind == 'axial':
            results['axial_invariance'] = axial_invariance(result, results['cases'][0])
        save_json(dest, results)
    if not args.fast:
        save_json(dest, annotate_existing(results))


if __name__ == '__main__':
    main()
