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
