"""T2.4 small-grid convection smoke tests; long measurements opt in.

T24_RUN_SLOW=1 enables the slow-marked one-hour integrations. Otherwise the
documented plain pytest command stays fast, even without -m 'not slow'.
"""
import os
import json
from pathlib import Path

import numpy as np
import pytest

from platesim.feature.engine.params import Params
from platesim.flow3d import buoyancy_force, solve_stokes3d
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.sim import run_tile, top_height
from tools.compare_3d_2d_flow import (axial_invariance, compare_flow, configure,
                                    lid_diagnostic, buoyancy_diagnostic)
from tools.compare_3d_2d import setup
from tools.trench_axial_ends import setup_trench, profiles, end_zones, run_experiment


def small_extrusion():
    grid = Grid3D((12, 3, 14), 8e-4, (0., 0., -32e-4))
    x, y, z = grid.meshgrid()
    psi = np.maximum(z, np.minimum(24e-4 - np.abs(x-48e-4), z+16e-4))
    field = np.abs(x[:, :, 0]-48e-4) > 32e-4
    p = Params(suppressor=False, flow_shear_s=0., k_mix=0., delta_um=32.)
    return psi, grid, field, p


def test_shear_lid_mapping_same_domain():
    result = lid_diagnostic(dx=24., shear=10.)
    assert result['matched_relative_error'] < 1e-9
    assert result['two_d_lid_um'] > result['native_three_d_initial_lid_um']


def test_beta_and_gravity_projection_match_2d():
    sim, _, _, _ = setup(dx=24., hours=.001)
    flow = configure(sim, 'across')
    assert sim.p.flow_solver == 'stokes'
    assert sim.p.adv_scheme == 'sg'
    assert sim.p.gravity_dir == (1., 0.)
    assert flow['gravity_dir'] == (1., 0., 0.)
    assert flow['beta_L_mol'] == sim.p.beta_cu_L_mol == .15
    flow = configure(sim, 'axial')
    assert sim.p.gravity_dir == (0., 0.)
    assert flow['gravity_dir'] == (0., -1., 0.)


def test_frozen_buoyancy_matches_only_with_same_flux_convention():
    result = buoyancy_diagnostic(dx=32.)
    assert result['matched_unconstrained_velocity_error_um_s'] < 1e-6
    assert abs(result['unconstrained_x_section_flux_cm3_s']) > 1e-10
    assert abs(result['constrained_x_section_flux_cm3_s']) < 1e-12


def test_axial_flow_changes_velocity_not_invariant_concentration():
    psi, grid, field, p = small_extrusion()
    plane = float(top_height(psi, grid)[field].mean()) + 32e-4
    solver = FieldSolver(p, 22.)
    base = solver.solve(psi, grid.dx, grid.origin[2], plane, .015)
    body = buoyancy_force(base['ct'], p.c_bulk*1e3, .15, (0., -1., 0.))
    flow = solve_stokes3d(psi, grid.dx, grid.origin[2], plane, body_force=body)
    assert np.abs(flow['uy']).max() > 0
    assert flow['info']['div_max_rel'] < 1e-9
    assert np.abs(flow['faces'][1][:, 0, :].sum()) < 1e-10
    conv = solver.solve(psi, grid.dx, grid.origin[2], plane, .015, flow_faces=flow['faces'])
    np.testing.assert_allclose(conv['ct'], base['ct'], atol=2e-4, rtol=0)
    np.testing.assert_allclose(conv['speed'], base['speed'], atol=2e-9, rtol=0)


def test_run_tile_with_shear_smoke():
    psi, grid, field, p = small_extrusion()
    out = run_tile(psi, grid, 15., 12., delta_um=32., params=p, field_mask=field,
                   n_frames=2, flow=dict(shear_rate_s=10.))
    assert np.isfinite(out['psi']).all()
    assert out['series']['t_s'][-1] == 12.
    assert max(out['series']['flow_umax_um_s']) > 0
    assert (top_height(out['psi'], grid)-top_height(psi, grid))[field].mean() > 0


def test_axial_no_flow_growth_invariance_small_grids():
    sims = [setup(dx=32., hours=.001)[0] for _ in range(2)]
    for sim, kind in zip(sims, ('none', 'axial')):
        configure(sim, kind)
        out = sim.run('axial smoke')
        assert out['stop_reason'] == 'recipe_end'
    np.testing.assert_array_equal(sims[0].psi, sims[1].psi)
    psi, grid, field, p = small_extrusion()
    base = run_tile(psi, grid, 15., 12., delta_um=32., params=p, field_mask=field)
    axial = run_tile(psi, grid, 15., 12., delta_um=32., params=p, field_mask=field,
                     flow=dict(gravity_dir=(0., -1., 0.), beta_L_mol=.15))
    np.testing.assert_allclose(top_height(axial['psi'], grid), top_height(base['psi'], grid),
                               atol=1e-8, rtol=0)


def test_finite_trench_closed_ends_and_profile_convention():
    psi, grid, field, p = setup_trench(length_um=160., dx_um=8.,
                                      half_span_um=128., end_margin_um=32., top_um=96.)
    x, y, z = (v*1e4 for v in grid.centers())
    assert grid.periodic == (False, False, False)
    assert grid.bc['x_lo'] == 'symmetry'
    k = np.argmin(abs(z+40))
    assert (psi[0, (y < -16) | (y > 176), k] < 0).all()
    assert (psi[0, (y > 16) & (y < 144), k] > 0).all()
    assert field.any() and not field.all()
    # Use a wider sampling grid so that the far field x>200um exists.
    psi, grid, _, _ = setup_trench(length_um=160., end_margin_um=32., top_um=96.)
    profile = profiles(psi, psi-1e-4, grid, 160.)
    assert len(profile['samples']) == 10
    assert np.isfinite(profile['floor_centre_um']).all()
    zones = end_zones(profile, 160.)
    assert 0 <= zones['top_um'] <= 80
    assert 0 <= zones['bottom_um'] <= 80


def test_axial_invariance_comparison_metric():
    a = dict(profile_2d_um=[1., 2.], profile_3d_um=[2., 3.])
    assert axial_invariance(a, a) == dict(max_profile_difference_2d_um=0.,
                                         max_profile_difference_3d_um=0.)


def test_saved_measurements_are_complete_and_do_not_hide_misses():
    root = Path(__file__).resolve().parents[1]
    a = json.loads((root/'tools/compare_3d_2d_flow_results.json').read_text())
    assert len(a['cases']) == 6
    for case in a['cases']:
        assert case['hours'] == 1.
        for r in case['rows']:
            if r['target'] is not None:
                assert r['passed'] == (abs(r['difference']) <= r['target'])
        assert case['misses'] == [r['quantity'] for r in case['rows'] if r['passed'] is False]
    assert a['axial_invariance']['max_profile_difference_2d_um'] == 0.
    assert a['axial_invariance']['max_profile_difference_3d_um'] < 1e-6
    b = json.loads((root/'tools/trench_axial_ends_results.json').read_text())
    assert b['status'] == 'complete'
    assert b['actual_length_um'] in (1000., 2000.)
    assert b['grid']['periodic'] == [False, False, False]
    for run in b['runs'].values():
        assert len(run['samples']) == 10
        assert len(run['end_wall_encroached']) == len(run['y_um'])
        for sample in run['samples']:
            assert 24 <= sample['y_um'] <= b['actual_length_um']-24
            assert not sample['end_wall_encroached']
            assert 0 < sample['floor_um'] < 40
            assert sample['field_um'] > 0
        for region in run['regions'].values():
            assert 0 < region['floor_um'] < 40
    nf = b['runs']['no_flow']['regions']
    assert abs(nf['bottom']['floor_um']-nf['top']['floor_um']) < 1e-4


@pytest.mark.slow
@pytest.mark.skipif(os.getenv('T24_RUN_SLOW') != '1', reason='set T24_RUN_SLOW=1 for 1h runs')
@pytest.mark.parametrize('kind,shear,dx', [('shear',10.,8.), ('shear',50.,8.),
                                         ('across',0.,8.), ('axial',0.,8.),
                                         ('shear',10.,4.)])
def test_hour_comparisons_report_targets_without_hiding_misses(kind, shear, dx):
    result = compare_flow(kind, shear, dx)
    for r in result['rows']:
        assert np.isfinite([r['two_d'], r['three_d'], r['difference']]).all()
    assert result['rows'][0]['two_d'] > 0
    assert result['rows'][0]['three_d'] > 0
    # Strict targets are measurements, not loosened assertions: preserve misses.
    assert result['misses'] == [r['quantity'] for r in result['rows'] if r['passed'] is False]


@pytest.mark.slow
@pytest.mark.skipif(os.getenv('T24_RUN_SLOW') != '1', reason='set T24_RUN_SLOW=1 for 1h runs')
def test_hour_axial_invariance_against_no_flow():
    base = compare_flow('none')
    axial = compare_flow('axial')
    differences = axial_invariance(axial, base)
    assert differences['max_profile_difference_2d_um'] < 1e-8
    assert differences['max_profile_difference_3d_um'] < .01


@pytest.mark.slow
@pytest.mark.skipif(os.getenv('T24_RUN_SLOW') != '1', reason='set T24_RUN_SLOW=1 for finite 1h run')
def test_hour_finite_trench_axial_ends():
    result = run_experiment()
    assert result['status'] == 'complete'
    assert result['grid']['periodic'] == [False, False, False]
    for run in result['runs'].values():
        assert len(run['samples']) == 10
        assert np.isfinite([s['floor_um'] for s in run['samples']]).all()
