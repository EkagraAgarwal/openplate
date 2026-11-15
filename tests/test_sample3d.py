"""Small preview tiles; run directories use pytest's external basetemp."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from platesim.sample3d import build_geometry, plan_tile, run_sample
from platesim.spec3d import Manifest

EXAMPLES = Path('platesim/spec3d/examples')


def example(name):
    data = json.loads((EXAMPLES / f'{name}.json').read_text())
    data['numerics'] = dict(quality='preview')
    data['recipe']['steps'][0]['duration_h'] = .1
    return data


def small(array=False, long=False):
    data = example('single_trench')
    data['layout'] = dict(die_um=[1000, 1000], features=[dict(
        kind='rect' if long else 'circle', x_um=400, y_um=400,
        w_um=32, h_um=320 if long else 32)])
    if array:
        data['layout']['features'][0]['array'] = dict(nx=3, ny=3, pitch_x_um=64, pitch_y_um=64)
    data['etch']['default'] = dict(depth_um=24)
    data['setup']['delta_um'] = 24
    return data


@pytest.mark.parametrize('name,kinds,weights', [
    ('single_trench', ['trench_mid', 'trench_end'], [1, 2]),
    ('tsv_array', ['array_quarter'], [25]),
    ('coupon_example', ['trench_mid', 'trench_end'] * 3 + ['compact_quarter'] * 5,
     [1, 2] * 3 + [1] * 5),
])
def test_shipped_plans(name, kinds, weights):
    data = example(name)
    data['recipe'] = json.loads((EXAMPLES / f'{name}.json').read_text())['recipe']
    plans = plan_tile(data)
    assert isinstance(plans, list)
    assert [p['kind'] for p in plans] == kinds
    assert [p['weight'] for p in plans] == weights
    assert all(p['n_cells'] <= 200_000 for p in plans)
    assert all(n % 8 == 0 for p in plans for n in p['shape'])
    assert all(p['feature_ids'] and p['assumptions'] for p in plans)
    assert all(set(p['side_bc'].values()) <= {'symmetry', 'periodic'} for p in plans)


def test_regular_explicit_array_and_neighbours():
    data = small()
    f = data['layout']['features'][0]
    data['layout']['features'] = [dict(f, x_um=x, y_um=y) for x in (200, 264) for y in (200, 264)]
    tile, = plan_tile(data)
    assert tile['kind'] == 'array_quarter'
    assert tile['weight'] == 4 and len(tile['feature_ids']) == 4
    data['layout']['features'].pop()
    plans = plan_tile(data)
    assert len(plans) == 3
    assert all(any('neighbour interaction neglected' in flag for flag in p['flags']) for p in plans)


def test_cap_quality_and_exact_pitch():
    data = example('tsv_array')
    tile, = plan_tile(data)
    assert tile['window_um'][2] == pytest.approx(59.5)
    assert tile['shape'][0] * tile['dx_um'] == pytest.approx(59.5)
    data['numerics'] = dict(quality='preview', dx_um=1)
    tile, = plan_tile(data)
    assert tile['n_cells'] <= 200_000 and 'preview' in tile['flags']
    data['numerics']['quality'] = 'fine'
    with pytest.raises(ValueError, match='fix:'):
        plan_tile(data)
    with pytest.raises(ValueError, match='field_margin_um'):
        plan_tile(data, field_margin_um=-1)


def solve(psi, grid, data):
    from platesim.tile3d.sim import run_tile
    return run_tile(psi, grid, 10, 360, delta_um=data['setup']['delta_um'],
                    n_frames=2, backend='numpy')['psi']


@pytest.mark.parametrize('long', [False, True])
def test_symmetry_matches_full_tile(long):
    data = small(array=not long, long=long)
    plan = plan_tile(data, field_margin_um=32)[0]
    bare, grid = build_geometry(data, plan)
    from platesim.sample3d.pipeline import _ReflectedResult
    class BareResult:
        psi_si = bare
        times_s = np.array([0.])
        def psi(self, index=-1):
            return bare
    source = BareResult()
    source.grid = grid
    reflected = _ReflectedResult(source, plan['mirror_axes'])
    assert reflected.grid.shape[0] == 2 * grid.shape[0]
    assert reflected.grid.shape[1] == grid.shape[1] * (1 if long else 2)
    assert reflected.psi().shape == reflected.grid.shape
    full = copy.deepcopy(plan)
    full['mirror_axes'] = []
    full['window_um'][0] = -plan['window_um'][2]
    full['shape'][0] *= 2
    if not long:
        full['window_um'][1] = -plan['window_um'][3]
        full['shape'][1] *= 2
        full['periodic'] = [True, True, False]
        full['side_bc'] = {f'{a}_{s}': 'periodic' for a in 'xy' for s in ('lo', 'hi')}
    full_bare, full_grid = build_geometry(data, full)
    psi = solve(bare, grid, data)
    full_psi = solve(full_bare, full_grid, data)
    crop = full_psi[grid.shape[0]:]
    crop_bare = full_bare[grid.shape[0]:]
    if not long:
        crop, crop_bare = crop[:, grid.shape[1]:], crop_bare[:, grid.shape[1]:]
    from platesim.tile3d.sim import top_height
    growth = top_height(psi, grid) - top_height(bare, grid)
    ref = top_height(crop, grid) - top_height(crop_bare, grid)
    opening = (bare[:, :, np.argmin(abs(grid.centers()[2] + .0012))] > 0)
    for mask in (opening, ~opening):
        assert np.median(growth[mask]) == pytest.approx(np.median(ref[mask]), rel=.02, abs=1e-8)
    if long:
        assert np.max(abs(growth - ref)) < .02 * np.max(abs(ref))


def test_end_to_end_single_trench(tmp_path):
    data = example('single_trench')
    data['numerics']['dx_um'] = 24  # <40k cells per tile
    data['recipe']['steps'].append(dict(data['recipe']['steps'][0]))
    out = tmp_path / 'run'
    manifest = run_sample(data, out)
    assert Manifest.model_validate_json((out / 'manifest.json').read_text()).model_dump(mode='json') == manifest
    assert all(q['feature_id'] == 'feature_0000' for q in manifest['quantities'])
    assert all(q['flag'] == 'preview' for q in manifest['quantities'])
    names = {q['name'] for q in manifest['quantities']}
    assert {'floor_over_field', 'trench_end.floor_over_field'} <= names
    plans = json.loads((out / 'tiles.json').read_text())
    assert [p['kind'] for p in plans] == ['trench_mid', 'trench_end']
    for p in plans:
        directory = out / 'tiles' / p['tile_id']
        assert all((directory / f'{name}.png').stat().st_size > 100
                   for name in ('surface', 'height_map', 'cross_section'))
        assert (directory / 'grid.json').exists()
        from platesim.tile3d.result import TileResult
        result = TileResult(directory)
        assert result.times_s.tolist() == pytest.approx([0, 360, 720])
        assert result.series['charge_C_cm2'][-1] == pytest.approx(
            720 * (20 * 10 - 20) / 11 * 1e-3)
    assert all(not Path(a['path']).is_absolute() for a in manifest['artifacts'])
    assert 'profile' in manifest['summary_text']
    with pytest.raises(FileExistsError):
        run_sample(data, out)


def test_cli_preview_plan():
    run = subprocess.run([sys.executable, '-m', 'platesim.sample3d', 'plan',
                          str(EXAMPLES / 'single_trench.json'), '--quality', 'preview'],
                         check=True, capture_output=True, text=True)
    plans = json.loads(run.stdout)
    assert [p['kind'] for p in plans] == ['trench_mid', 'trench_end']


def test_end_section_is_full_reflected_width(tmp_path):
    from types import SimpleNamespace
    from platesim.sample3d.pipeline import _ReflectedResult, _section_line
    from platesim.analysis3d import cross_section
    data = small(long=True)
    plan = plan_tile(data, field_margin_um=32)[1]
    bare, grid = build_geometry(data, plan)
    result = _ReflectedResult(SimpleNamespace(grid=grid, psi_si=bare,
                             times_s=[0], psi=lambda index: bare), plan['mirror_axes'])
    line = _section_line(plan, result.grid)
    assert line[0][0] == pytest.approx(-line[1][0])
    assert line[0][1] == line[1][1] == pytest.approx(144)
    section = cross_section(result.psi(), result.psi_si, result.grid, line)
    assert np.allclose(section['psi'], section['psi'][::-1], atol=1e-9)
    # At half depth, both walls and the central opening are present.
    k = np.argmin(abs(np.asarray(section['z_um']) + 12))
    row = section['psi'][:, k]
    assert row[0] < 0 and row[-1] < 0 and row[len(row)//2] > 0


def test_rounding_keeps_symmetry_and_extends_only_field():
    data = small(long=True)
    mid, end = plan_tile(data, field_margin_um=32)
    assert mid['window_um'][:2] == [0, 0]
    assert end['window_um'][:2] == [0, 128]
    for plan in (mid, end):
        dx = plan['dx_um']
        assert all(n % 8 == 0 for n in plan['shape'])
        assert plan['z_extent_um'][0] == -(np.ceil(24/dx)+2)*dx
        assert plan['geometry_feature']['x_um'] == 0


def test_default_does_not_pass_flow(tmp_path, monkeypatch):
    from platesim.tile3d import sim

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        assert 'flow' not in kwargs
        raise Captured

    monkeypatch.setattr(sim, 'run_tile', capture)
    data = small(long=True)
    assert data['setup'].get('stir_shear_1_s') is None
    plans = plan_tile(data)
    assert all(not any('Resolved Couette shear' in a for a in p['assumptions']) for p in plans)
    with pytest.raises(Captured):
        run_sample(data, tmp_path / 'default')


def test_rotated_parallel_shear_kwargs(tmp_path, monkeypatch):
    from platesim.sample3d.pipeline import _run_one, _spec
    from platesim.tile3d import sim

    data = small(long=True)
    data['layout']['features'][0]['rotation_deg'] = 90
    mid = plan_tile(data, field_margin_um=32)[0]
    # Only the middle strip has a single nonperiodic axis. The end diagnostic
    # also has a no-flux y cut and cannot support this axial shear.
    data['setup'].update(stir_shear_1_s=30, stir_angle_deg=180)

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        flow = kwargs['flow']
        assert flow['shear_rate_s'] == 30
        assert flow['shear_dir'] == pytest.approx((0, 1), abs=1e-12)
        assert flow['lid_um'] is None
        assert flow['update_every'] == 5
        assert flow['axial_net_flux_zero'] is None
        assert set(flow) == {'shear_rate_s', 'shear_dir', 'lid_um',
                             'update_every', 'axial_net_flux_zero'}
        raise Captured

    monkeypatch.setattr(sim, 'run_tile', capture)
    with pytest.raises(Captured):
        _run_one(_spec(data), mid, tmp_path / 'parallel')

    data['setup']['stir_angle_deg'] = 360
    def capture_reverse(*args, **kwargs):
        assert kwargs['flow']['shear_dir'] == pytest.approx((0, -1), abs=1e-12)
        raise Captured
    monkeypatch.setattr(sim, 'run_tile', capture_reverse)
    with pytest.raises(Captured):
        _run_one(_spec(data), mid, tmp_path / 'parallel_reverse')


@pytest.mark.parametrize('long,angle', [(False, 0), (False, 45), (True, 0), (True, 90)])
def test_shear_plans_full_periodic_tiles(long, angle):
    # mirrored tiles have no-flux sides that shear cannot cross: sheared runs plan full periodic tiles
    from platesim.sample3d.pipeline import _tile_flow, _spec
    data = small(long=long)
    data['setup'].update(stir_shear_1_s=30, stir_angle_deg=angle)
    plans = plan_tile(data)
    assert [p['kind'] for p in plans] == (['trench_mid_periodic'] if long else ['compact_periodic'])
    for p in plans:
        assert p['mirror_axes'] == [] and all(v == 'periodic' for v in p['side_bc'].values())
        assert p['window_um'][0] < 0 < p['window_um'][2]
        assert any('periodic image' in a for a in p['assumptions'])
        flow = _tile_flow(_spec(data), p)           # no symmetry error
        a = np.deg2rad(angle - p['rotation_from_die_deg'])
        assert flow['shear_dir'] == pytest.approx((np.cos(a), np.sin(a)))
    # without shear the mirrored tiles are unchanged
    for off in (None, 0):                            # 0 means "no shear", same as None
        data['setup']['stir_shear_1_s'] = off
        plans = plan_tile(data)
        assert all(p['mirror_axes'] for p in plans)
        assert not any('Resolved Couette shear' in a for p in plans for a in p['assumptions'])


def test_real_preview_resolved_shear(tmp_path, monkeypatch):
    from platesim.tile3d import sim

    data = example('quick_trench')
    data['layout']['die_um'] = [600, 600]
    feature = data['layout']['features'][0]
    # A rotated array uses the planner's existing full periodic cell. Quarter
    # compact tiles cannot support any nonzero in-plane shear honestly.
    feature.update(rotation_deg=30,
                   array=dict(nx=2, ny=2, pitch_x_um=160, pitch_y_um=160))
    data['numerics']['dx_um'] = 10
    data['recipe']['steps'][0]['duration_h'] = .1
    captured = []
    real_run_tile = sim.run_tile

    def capture(*args, **kwargs):
        captured.append(kwargs)
        return real_run_tile(*args, **kwargs)

    monkeypatch.setattr(sim, 'run_tile', capture)
    baseline = run_sample(data, tmp_path / 'still')
    assert all('flow' not in kw for kw in captured)
    captured.clear()
    data['setup'].update(stir_shear_1_s=30, stir_angle_deg=0)
    out = tmp_path / 'shear'
    flowing = run_sample(data, out)
    assert captured and all(kw['flow']['shear_rate_s'] == 30 for kw in captured)
    assert all(kw['flow']['shear_dir'] == pytest.approx((1, 0)) for kw in captured)
    faraday = [c for c in flowing['checks'] if c['name'].endswith('faraday_error_rel')]
    assert faraday and all(c['passed'] for c in faraday)
    def ratio(manifest):
        return next(q['value'] for q in manifest['quantities'] if q['name'] == 'floor_over_field')
    assert abs(ratio(flowing) - ratio(baseline)) > 1e-6
    assert 'Resolved Couette shear G=30 1/s at angle 0' in flowing['summary_text']
    assert 'no buoyancy' in flowing['summary_text']
    assert 'No additives, resolved flow' not in flowing['summary_text']
    plans = json.loads((out / 'tiles.json').read_text())
    assert all(any('Resolved Couette shear G=30' in a for a in p['assumptions']) for p in plans)
