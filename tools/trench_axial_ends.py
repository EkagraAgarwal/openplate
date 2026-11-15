"""T2.4 finite trench, half-width mirror tile, axial gravity and closed ends.

Run PYTHONPATH=. python tools/trench_axial_ends.py. A child benchmarks the
requested 2000 um trench; a >300 s flow solve triggers a 1000 um retry.
Only measurements are written; no solver or volume correction is applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

import numpy as np

from platesim.feature.engine.params import Params
from platesim.flow3d import buoyancy_force, solve_stokes3d
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d import levelset3d as ls
from platesim.tile3d.sim import run_tile, top_height
from tools.compare_3d_2d_flow import progress, save_json


def setup_trench(length_um=2000., dx_um=8., width_um=200., depth_um=80.,
                 half_span_um=248., end_margin_um=128., top_um=240.):
    """x=0 mirror; y=0..length cavity with field beyond both solid end walls.

    External y sides are impermeable/free-slip, NOT the trench ends: the end
    walls are explicitly solid below z=0. The fluid lid closes at the bulk plane.
    """
    dx = dx_um * 1e-4
    bottom_um = -depth_um - 2 * dx_um
    shape = tuple(int(round(n / dx_um)) for n in
                  (half_span_um, length_um + 2 * end_margin_um, top_um - bottom_um))
    bc = dict(x_lo='symmetry', x_hi='symmetry', y_lo='symmetry',
              y_hi='symmetry', z_lo='bottom', z_hi='bulk')
    grid = Grid3D((shape[0], shape[1], shape[2]), dx,
                  (0., -end_margin_um * 1e-4, bottom_um * 1e-4), bc)
    x, y, z = grid.meshgrid()
    cavity = np.minimum.reduce([width_um * .5e-4 - x, y,
                               length_um * 1e-4 - y, z + depth_um * 1e-4])
    psi = ls.reinit(np.maximum(z, cavity), dx, periodic=grid.periodic)
    xc, yc, _ = grid.centers()
    field = ((xc[:, None] > width_um * .5e-4) |
             (yc[None, :] < 0) | (yc[None, :] > length_um * 1e-4))
    p = Params(suppressor=False, delta_um=64., efficiency=1., transport='enhanced',
               flow_shear_s=0., k_mix=0.)
    return psi, grid, field, p


def benchmark(length_um, dest):
    psi, grid, field, p = setup_trench(length_um)
    plane = float(top_height(psi, grid)[field].mean()) + 64e-4
    result: dict[str, Any] = dict(length_um=length_um, shape=list(grid.shape), periodic=list(grid.periodic))
    print(f"benchmark length={length_um:g}um shape={grid.shape}: FieldSolver.solve", flush=True)
    t0 = perf_counter()
    out = FieldSolver(p, 22.).solve(psi, grid.dx, grid.origin[2], plane, .015,
                                   periodic=grid.periodic)
    result['field_wall_s'] = perf_counter() - t0
    result['field_converged'] = bool(out['info']['converged'])
    if not result['field_converged']:
        raise RuntimeError('initial field benchmark did not converge')
    result['stage'] = 'stokes'
    save_json(dest, result)
    print(f"field {result['field_wall_s']:.2f}s; solve_stokes3d starting", flush=True)
    t0 = perf_counter()
    flow = solve_stokes3d(psi, grid.dx, grid.origin[2], plane, periodic=grid.periodic,
                          body_force=buoyancy_force(out['ct'], p.c_bulk * 1e3, .07, (0., -1., 0.)))
    result.update(flow_wall_s=perf_counter() - t0, stage='done',
                  flow_max_component_um_s=max(np.abs(flow[k]).max() for k in ('ux', 'uy', 'uz'))*1e4,
                  flow_info=flow['info'])
    save_json(dest, result)
    print(f"Stokes {result['flow_wall_s']:.2f}s", flush=True)


def timed_probe(length_um, budget_s=300.):
    """Separate process bounds a single flow solve without patching its solver.

    Field time is not charged to the flow budget: the probe writes a stage
    checkpoint and the parent uses process.wait for completion in bounded stages.
    """
    import time
    with tempfile.TemporaryDirectory(prefix='t24_probe_') as tmp:
        dest = Path(tmp) / 'probe.json'
        child = subprocess.Popen([sys.executable, '-u', __file__, '--probe', str(dest),
                                  '--length', str(length_um)])
        start = perf_counter()
        try:
            # Field solves are allowed their own 10-minute bound. Waiting here
            # is local process supervision, not an asynchronous tool job poll.
            while child.poll() is None and not dest.exists():
                if perf_counter() - start > 600:
                    raise subprocess.TimeoutExpired(child.args, 600)
                time.sleep(.5)
            result = json.loads(dest.read_text()) if dest.exists() else {}
            try:
                child.wait(timeout=budget_s)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                result.update(stage='flow_timeout', flow_wall_s_lower_bound=budget_s)
                return result
            if child.returncode:
                raise RuntimeError(f'benchmark child exited {child.returncode}: {result}')
            return json.loads(dest.read_text())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


def profiles(psi0, psi, grid, length_um) -> dict[str, Any]:
    h0 = top_height(psi0, grid) * 1e4
    h = top_height(psi, grid) * 1e4
    gain = h - h0
    x, y, _ = (v * 1e4 for v in grid.centers())
    inside = (y > 0) & (y < length_um)
    yy = y[inside]
    # Centre-line floor, far-side field mean, and full floor width average.
    floor = gain[0, inside]
    field = gain[x > 200., :][:, inside].mean(axis=0)
    floor_mean = gain[x < 84., :][:, inside].mean(axis=0)
    # A fixed column within the grown end wall no longer samples exposed floor.
    # Keep its envelope in the dense profile, but flag it and omit it in region
    # floor averages. Ten requested samples start >=3dx from either solid end.
    encroached = h[0, inside] > -40.
    eligible = np.flatnonzero((yy >= 3*grid.dx*1e4) & (yy <= length_um-3*grid.dx*1e4))
    idx = eligible[np.rint(np.linspace(0, len(eligible)-1, 10)).astype(int)]
    groups = dict(bottom=yy < .1 * length_um,
                  middle=np.abs(yy - .5*length_um) < .05*length_um,
                  top=yy > .9*length_um)
    return dict(y_um=yy.tolist(), floor_centre_um=floor.tolist(),
                end_wall_encroached=encroached.tolist(),
                floor_inner_width_mean_um=floor_mean.tolist(), field_um=field.tolist(),
                samples=[dict(y_um=float(yy[i]), floor_um=float(floor[i]),
                              field_um=float(field[i]), floor_field=float(floor[i]/field[i]),
                              end_wall_encroached=bool(encroached[i]))
                         for i in idx],
                regions={key: dict(floor_um=float(floor[mask & ~encroached].mean()),
                                   surface_envelope_um=float(floor[mask].mean()),
                                   encroached_columns=int(np.sum(mask & encroached)),
                                   field_um=float(field[mask].mean())) for key, mask in groups.items()})


def end_zones(profile, length_um, reference=None, threshold=.01):
    """First return within 1% of middle (absolute thickness / middle floor).

    With reference, use convection-minus-no-flow, subtract its middle value,
    and normalise by no-flow middle floor: separates geometric end effects.
    If never within tolerance by the midpoint, return half the trench length.
    """
    y = np.array(profile['y_um'])
    a = np.array(profile['floor_centre_um'])
    mid = np.abs(y - .5*length_um) < .05*length_um
    scale = np.mean(a[mid])
    if reference is not None:
        b = np.array(reference['floor_centre_um'])
        scale = np.mean(b[mid])
        a = a - b
    affected = np.abs(a - np.mean(a[mid])) > threshold * abs(scale)
    zones = {}
    for name, dist in [('bottom', y), ('top', length_um-y)]:
        order = np.argsort(dist)
        candidates = order[(dist[order] <= length_um/2) & ~affected[order]]
        zones[name + '_um'] = float(dist[candidates[0]]) if len(candidates) else length_um/2
    return dict(threshold_fraction=threshold, resolution_um=float(y[1]-y[0]), **zones)


def annotate_existing(results):
    """Re-sample saved dense profiles; no flow/field/growth solves are repeated.

    Initial zero heights are reproducible from setup_trench, so an end-wall
    encroachment flag can also be recovered from an older dense gain profile.
    """
    length = results['actual_length_um']
    psi0, grid, _, _ = setup_trench(length)
    y0 = grid.centers()[1] * 1e4
    h0 = top_height(psi0, grid)[0, (y0 > 0) & (y0 < length)] * 1e4
    for run in results['runs'].values():
        y = np.array(run['y_um'])
        floor = np.array(run['floor_centre_um'])
        field = np.array(run['field_um'])
        encroached = floor + h0 > -40.
        run['end_wall_encroached'] = encroached.tolist()
        eligible = np.flatnonzero((y >= 24.) & (y <= length-24.))
        idx = eligible[np.rint(np.linspace(0, len(eligible)-1, 10)).astype(int)]
        run['samples'] = [dict(y_um=float(y[i]), floor_um=float(floor[i]),
                               field_um=float(field[i]), floor_field=float(floor[i]/field[i]),
                               end_wall_encroached=bool(encroached[i])) for i in idx]
        groups = dict(bottom=y < .1*length, middle=np.abs(y-.5*length) < .05*length,
                      top=y > .9*length)
        run['regions'] = {
            key: dict(floor_um=float(floor[mask & ~encroached].mean()),
                      surface_envelope_um=float(floor[mask].mean()),
                      encroached_columns=int(np.sum(mask & encroached)),
                      field_um=float(field[mask].mean())) for key,mask in groups.items()}
    results['diagnosis'][2] = (
        'Profiles subtract actual initial highest psi=0 crossings. Floor is x=4um; field '
        'is the mean of x>200um columns. Ten samples start at least 3dx from each end. '
        'Dense floor profiles include column envelopes under grown end walls; final '
        'height above -40um flags end-wall encroachment, excluded from regional floor '
        'means. Top/bottom are 10%-length averages, middle is the central 10%.')
    no_error = results['runs']['no_flow']['summary']['faraday_error_rel'] * 100
    flow_error = results['runs']['axial_flow']['summary']['faraday_error_rel'] * 100
    note = (f'Faraday volume errors are {no_error:+.3f}% without flow and {flow_error:+.3f}% '
            'with axial flow on this grid, a numerical accuracy caveat; no volume '
            'correction was used. Large envelope gains in the first end columns are '
            'end-wall advance, not 80um of deposited floor copper.')
    if len(results['diagnosis']) == 5:
        results['diagnosis'].append(note)
    else:
        results['diagnosis'][5] = note
    results['postprocessing'] = 'Dense saved profiles re-sampled at >=3dx from ends; encroached columns excluded from regional floor means. No integration rerun.'
    return results


def run_experiment(length_um=2000., hours=1., do_probe=True):
    dest = Path(__file__).with_name('trench_axial_ends_results.json')
    results: dict[str, Any] = dict(requested_length_um=length_um, width_um=200., depth_um=80., dx_um=8.,
                   hours=hours, bath_Cu_g_L=65., current_mA_cm2=15., delta_um=64.,
                   gravity_dir=[0,-1,0], beta_L_mol=.07, benchmarks=[], runs={})
    if do_probe:
        probe = timed_probe(length_um)
        results['benchmarks'].append(probe)
        save_json(dest, results)
        if probe['stage'] == 'flow_timeout' or probe.get('flow_wall_s', 0) > 300:
            length_um = 1000.
            results['reduction_reason'] = '2000um single Stokes solve exceeded 300s'
            probe = timed_probe(length_um)
            results['benchmarks'].append(probe)
            if probe['stage'] != 'done':
                results['status'] = 'blocked: reduced-size flow benchmark also timed out'
                save_json(dest, results)
                return results
    results['actual_length_um'] = length_um
    psi0, grid, field, p = setup_trench(length_um)
    results['grid'] = dict(shape=list(grid.shape), periodic=list(grid.periodic),
                           half_span_um=248., end_margin_um=128., top_um=240.)
    for label, flow in [('no_flow', None), ('axial_flow', dict(gravity_dir=(0., -1., 0.)))]:
        print(f"{label} length={length_um:g}um duration={hours*3600:g}s", flush=True)
        t0 = perf_counter()
        out = run_tile(psi0, grid, 15., hours*3600, params=p, field_mask=field,
                        flow=flow, n_frames=10, progress=progress(label))
        assert isinstance(out, dict)  # no run_dir: run_tile returns in-memory data
        result = profiles(psi0, out['psi'], grid, length_um)
        result.update(wall_s=perf_counter()-t0, summary=out['summary'],
                      max_sampled_component_speed_um_s=max(out['series']['flow_umax_um_s']))
        result['end_affected_zone'] = end_zones(result, length_um)
        results['runs'][label] = result
        save_json(dest, results)
    results['convection_end_affected_zone'] = end_zones(results['runs']['axial_flow'], length_um,
                                                       results['runs']['no_flow'])
    results['diagnosis'] = [
        'Finite closed solid ends with 128um external field margins; all lateral external boundaries are impermeable/free-slip. x=0 is half-width mirror symmetry; x=248um is a second symmetry boundary (finite pitch). No periodic y.',
        'Gravity -y: bottom end y=0, top end y=L; lighter depleted electrolyte rises toward +y. beta=0.07 is the unmodified run_tile default, unlike the explicitly matched beta=0.15 in Part A.',
        'Profiles subtract actual initial highest psi=0 crossings. Floor is x=4um; field is the mean of x>200um columns. Ten samples start at least 3dx from each end. Dense floor profiles include column envelopes under grown end walls; final height above -40um flags end-wall encroachment, excluded from regional floor means. Top/bottom are 10%-length averages, middle is the central 10%.',
        'End zone is distance from each end to first floor thickness within 1% of middle; centre distances have dx uncertainty and this is a threshold diagnostic, not an exponential fit. Convection-only zone subtracts no-flow and its middle contrast.',
        'No volume correction or solver patches. Native flow update_every=5, moving field-mean bulk-plane lid. Maximum sampled speed is max absolute cell component at saved frame times.'
    ]
    results['status'] = 'complete'
    save_json(dest, annotate_existing(results))
    print(json.dumps({k: v['samples'] for k,v in results['runs'].items()}, indent=2), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--length', type=float, default=2000.)
    ap.add_argument('--hours', type=float, default=1.)
    ap.add_argument('--probe', type=Path)
    ap.add_argument('--annotate-existing', action='store_true')
    args = ap.parse_args()
    if args.annotate_existing:
        dest = Path(__file__).with_name('trench_axial_ends_results.json')
        results = annotate_existing(json.loads(dest.read_text(encoding='utf-8')))
        save_json(dest, results)
        print(json.dumps({k: v['samples'] for k,v in results['runs'].items()}, indent=2))
    elif args.probe:
        benchmark(args.length, args.probe)
    else:
        run_experiment(args.length, args.hours)


if __name__ == '__main__':
    main()
