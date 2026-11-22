"""Contract 10h: opt-in solutal buoyancy in 3-D sample runs (synthetic specs only)."""
import copy
import json
from pathlib import Path

import pytest

from platesim.sample3d.pipeline import (
    _gravity_die, _gravity_tile, _spec, _tile_flow, plan_tile, run_sample,
)
from platesim.spec3d import SampleSpec, spec_hash

EXAMPLES = Path(__file__).parents[1] / 'platesim/spec3d/examples'
HASHES = {
    'coupon_example': 'c3128a4943c69c5b',
    'quick_trench': '3a33c86673777779',
    'single_trench': '7f1d0dd26430b349',
    'tsv_array': 'add7c7658ea90070',
}


def example(name='quick_trench'):
    return json.loads((EXAMPLES / f'{name}.json').read_text())


@pytest.mark.parametrize('name,prefix', sorted(HASHES.items()))
def test_example_hashes_unchanged(name, prefix):
    assert spec_hash(example(name)).startswith(prefix)


def test_explicit_false_hash_equals_absent():
    data = example()
    base = spec_hash(data)
    off = copy.deepcopy(data)
    off['setup']['buoyancy'] = False
    assert spec_hash(off) == base
    assert spec_hash(SampleSpec.model_validate(off)) == base
    on = copy.deepcopy(data)
    on['setup']['buoyancy'] = True
    assert spec_hash(on) != base


def _setup(**changes):
    data = example()
    data['setup'].update(changes)
    return _spec(data).setup


@pytest.mark.parametrize('up_edge,expected', [
    ('+y', (0., -1., 0.)), ('-y', (0., 1., 0.)),
    ('+x', (-1., 0., 0.)), ('-x', (1., 0., 0.)),
])
def test_gravity_die_vertical(up_edge, expected):
    assert _gravity_die(_setup(orientation='vertical', up_edge=up_edge)) == pytest.approx(expected)


@pytest.mark.parametrize('orientation,expected', [
    ('face_up', (0., 0., -1.)), ('face_down', (0., 0., 1.)),
])
def test_gravity_die_faces(orientation, expected):
    assert _gravity_die(_setup(orientation=orientation, up_edge='-x')) == pytest.approx(expected)


def _rotated_plan(degrees):
    return dict(tile_id='unit_tile', rotation_from_die_deg=degrees,
                side_bc={f'{a}_{s}': 'periodic' for a in 'xy' for s in ('lo', 'hi')})


def test_gravity_tile_rotation():
    setup = _setup(orientation='vertical', up_edge='+y')  # die gravity (0, -1, 0)
    assert _gravity_tile(setup, _rotated_plan(0)) == pytest.approx((0., -1., 0.))
    assert _gravity_tile(setup, _rotated_plan(90)) == pytest.approx((-1., 0., 0.))
    assert _gravity_tile(setup, _rotated_plan(-90)) == pytest.approx((1., 0., 0.))
    assert _gravity_tile(setup, _rotated_plan(180)) == pytest.approx((0., 1., 0.), abs=1e-12)
    face_up = _setup(orientation='face_up')
    assert _gravity_tile(face_up, _rotated_plan(90)) == pytest.approx((0., 0., -1.))


def test_buoyancy_plans_full_periodic_like_shear():
    base = plan_tile(example())
    assert [p['kind'] for p in base] == ['compact_quarter']
    buoyant, sheared = example(), example()
    buoyant['setup']['buoyancy'] = True
    sheared['setup'].update(stir_shear_1_s=30, stir_angle_deg=0)
    pb, ps = plan_tile(buoyant), plan_tile(sheared)
    assert [p['kind'] for p in pb] == [p['kind'] for p in ps] == ['compact_periodic']
    assert all('trench_end' not in p['kind'] for p in pb)
    structural = ('tile_id', 'kind', 'feature_ids', 'weight', 'window_um', 'dx_um',
                  'z_extent_um', 'side_bc', 'periodic', 'mirror_axes', 'shape',
                  'n_cells', 'rotation_from_die_deg', 'geometry_feature', 'source_center_um')
    for a, b in zip(pb, ps):
        assert all(a[key] == b[key] for key in structural)
        assert a['mirror_axes'] == [] and all(v == 'periodic' for v in a['side_bc'].values())
    assert any('Solutal buoyancy' in note for p in pb for note in p['assumptions'])
    assert not any('Resolved Couette shear' in note for p in pb for note in p['assumptions'])


def test_buoyancy_false_plan_unchanged():
    data = example()
    absent = plan_tile(data)
    off = copy.deepcopy(data)
    off['setup']['buoyancy'] = False
    assert plan_tile(off) == absent


def test_flow_dict_modes():
    data = example()
    data['setup'].update(buoyancy=True, up_edge='+y')
    plan = plan_tile(data)[0]
    flow = _tile_flow(_spec(data), plan)
    assert flow['shear_rate_s'] == 0
    assert flow['gravity_dir'] == pytest.approx((0., -1., 0.))
    assert flow['lid_um'] is None and flow['update_every'] == 5
    assert flow['axial_net_flux_zero'] is None

    data['setup'].update(stir_shear_1_s=30, stir_angle_deg=0)
    both = _tile_flow(_spec(data), plan)
    assert both['shear_rate_s'] == 30
    assert both['shear_dir'] == pytest.approx((1., 0.))
    assert both['gravity_dir'] == pytest.approx((0., -1., 0.))

    data['setup'].pop('buoyancy')
    shear_only = _tile_flow(_spec(data), plan)
    assert 'gravity_dir' not in shear_only
    assert set(shear_only) == {'shear_rate_s', 'shear_dir', 'lid_um',
                              'update_every', 'axial_net_flux_zero'}


def test_inplane_gravity_crossing_no_flux_side_rejected():
    data = example()
    data['setup']['buoyancy'] = True
    spec = _spec(data)
    plan = dict(plan_tile(data)[0])
    plan['side_bc'] = dict(plan['side_bc'], y_lo='symmetry')
    with pytest.raises(ValueError, match='setup.buoyancy=True') as exc:
        _tile_flow(spec, plan)
    assert plan['tile_id'] in str(exc.value) and 'fix:' in str(exc.value)
    for orientation in ('face_up', 'face_down'):
        data['setup']['orientation'] = orientation
        _tile_flow(_spec(data), plan)  # no in-plane component: no error


def test_end_to_end_buoyancy(tmp_path):
    data = example()
    data['numerics'] = dict(quality='preview', dx_um=8)
    data['recipe']['steps'][0]['duration_h'] = 0.05
    data['setup'].update(orientation='vertical', up_edge='+y', buoyancy=True)
    manifest = run_sample(data, tmp_path / 'buoyant')
    summary = manifest['summary_text']
    assert 'Solutal buoyancy' in summary and '0.15 L/mol' in summary
    faraday = [c for c in manifest['checks'] if c['name'].endswith('faraday_error_rel')]
    assert faraday and all(c['passed'] for c in faraday)
