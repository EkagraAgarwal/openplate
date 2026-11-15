"""Fast coupon-to-tile contract tests: no coupon FEM or 3-D solve."""
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from platesim.multiscale import bridge
from platesim.sample3d import pipeline
from platesim.spec3d import SampleSpec, Setup


def sample():
    return json.loads(Path('platesim/spec3d/examples/single_trench.json').read_text())


@pytest.mark.parametrize('local,mean', [(12., 12.), (18., 12.), (6., 12.)])
@pytest.mark.parametrize('stir', [None, 0., 45.])
def test_tile3d_conditions_normalises_current(monkeypatch, local, mean, stir):
    defaults = dict(bridge.DEFAULT_RUN)
    theta = {'test': 1.}
    site = dict(j_mA_cm2=local, j_mean_mA_cm2=mean, delta_um=37., efficiency=.97)

    def fake(run, x_cm, y_cm, **kwargs):
        assert run == {**defaults, 'j_mA_cm2': 23., 't_min': 2., 'T_C': 28.,
                       'stir_pct': defaults['stir_pct'] if stir is None else stir}
        assert (x_cm, y_cm) == (.2, -.3)
        assert kwargs == dict(theta=theta, resolution='fine')
        return site

    monkeypatch.setattr(bridge, 'site_conditions', fake)
    result = bridge.tile3d_conditions(23., 120., .2, -.3, T_C=28., stir_pct=stir,
                                      theta=theta, resolution='fine')
    assert result == dict(j_local_mA_cm2=23. * local / mean, scale=local / mean,
                         delta_um=37., efficiency=.97, site=site)
    assert result['site'] is site
    assert bridge.DEFAULT_RUN == defaults


@pytest.mark.parametrize('enabled', [False, True])
def test_pipeline_local_conditions(tmp_path, monkeypatch, enabled):
    from platesim import analysis3d, render3d
    from platesim.tile3d import sim
    from platesim.tile3d.grid import Grid3D
    from platesim.tile3d.result import TileResult

    data = sample()
    data['numerics']['quality'] = 'preview'
    data['recipe']['steps'] = [dict(t_forward_ms=1, j_forward_mA_cm2=j,
                                  duration_h=seconds / 3600)
                               for j, seconds in [(10., 60.), (20., 120.)]]
    data['setup']['stir_pct'] = 0.  # ignored unless a position is supplied
    if enabled:
        data['setup']['coupon_position_cm'] = [.2, -.3]
    # Plan without the position: plan_tile would otherwise run the real coupon model (covered below).
    plan = pipeline.plan_tile({**data, 'setup': {k: v for k, v in data['setup'].items()
                                                 if k != 'coupon_position_cm'}})[0]
    plan['mirror_axes'] = []
    monkeypatch.setattr(pipeline, 'plan_tile', lambda *a, **kw: [plan])
    grid = Grid3D((4, 4, 6), 1e-4)
    bare = np.broadcast_to((np.arange(6) - 2.5) * grid.dx, grid.shape).copy()
    monkeypatch.setattr(pipeline, 'build_geometry', lambda *a: (bare, grid))
    conditions_calls, solver_calls = [], []

    def conditions(j, duration, x, y, **kwargs):
        assert enabled, 'default path must not call the coupon model'
        assert (x, y) == (.2, -.3)
        assert kwargs == dict(T_C=25., stir_pct=0.)
        conditions_calls.append((j, duration))
        return dict(j_local_mA_cm2=j * 1.5, delta_um=37. + len(conditions_calls),
                    efficiency=.98 - .01 * len(conditions_calls), scale=1.5)

    def solve(psi, actual_grid, j, duration, **kwargs):
        assert actual_grid is grid
        solver_calls.append((j, duration, kwargs['delta_um'], kwargs['params']))
        return dict(psi=psi, series={'cu_balance_rel': [0.]})

    monkeypatch.setattr(bridge, 'tile3d_conditions', conditions)
    monkeypatch.setattr(sim, 'run_tile', solve)
    monkeypatch.setattr(analysis3d, 'height_map', lambda *a: dict(
        height_um=np.zeros((4, 4)), x_um=np.arange(4), y_um=np.arange(4), stats={}))
    monkeypatch.setattr(analysis3d, 'feature_report', lambda *a: {'fill_fraction': [0.]})
    monkeypatch.setattr(analysis3d, 'void_report', lambda *a: dict(count=0, volumes_um3=[]))
    monkeypatch.setattr(analysis3d, 'cross_section', lambda *a: {})
    def render(*args, **kwargs):
        return dict(path='preview.png', title='Stub preview')
    for name in ('render_surface', 'render_height_map', 'render_cross_section'):
        monkeypatch.setattr(render3d, name, render)

    manifest = pipeline.run_sample(data, tmp_path / 'run')
    expected_charge = expected_cu_charge = 0.
    for i, (j, duration, delta, params) in enumerate(solver_calls):
        net = (10., 20.)[i]
        eta = (.97, .96)[i] if enabled else 1.
        assert j == net * (1.5 if enabled else 1.)
        assert duration == (60., 120.)[i]
        assert delta == (38. + i if enabled else 64.)
        assert params.efficiency == pytest.approx(eta)
        expected_charge += j * 1e-3 * duration
        expected_cu_charge += j * 1e-3 * duration * eta
    assert len(solver_calls) == 2
    assert len(conditions_calls) == (2 if enabled else 0)
    if enabled:
        assert solver_calls[0][3] is not solver_calls[1][3]
    result = TileResult(tmp_path / 'run' / 'tiles' / plan['tile_id'])
    area = grid.shape[0] * grid.shape[1] * grid.dx**2
    assert result.series['charge_C_cm2'][-1] == expected_charge
    expected = expected_cu_charge * area * pipeline.MFAC
    assert result.series['faraday_volume_cm3'][-1] == pytest.approx(expected)
    quantities = {q['name']: q['value'] for q in manifest['quantities']}
    if enabled:
        assert quantities['step_1.j_local_mA_cm2'] == 15.
        assert quantities['step_2.delta_um'] == 39.
        assert quantities['step_2.efficiency'] == pytest.approx(.96)
        assert quantities['step_2.scale'] == 1.5
        assert 'Coupon position (0.2, -0.3) cm' in manifest['summary_text']
    else:
        assert not any(name.startswith('step_') for name in quantities)
        assert 'Coupon position' not in manifest['summary_text']
        assert result.series['faraday_volume_cm3'][-1] == expected_charge * area * pipeline.MFAC


@pytest.mark.parametrize('fields', [{}, {'coupon_position_cm': None, 'stir_pct': None},
                                 {'coupon_position_cm': [.25, -.5], 'stir_pct': 0.}])
def test_spec_round_trip(fields):
    data = sample()
    data['setup'].update(fields)
    spec = SampleSpec.model_validate(data)
    assert SampleSpec.model_validate_json(spec.model_dump_json()) == spec
    assert spec.setup.coupon_position_cm == (tuple(fields['coupon_position_cm'])
        if fields.get('coupon_position_cm') is not None else None)
    assert spec.setup.stir_pct == fields.get('stir_pct')
    assert {'coupon_position_cm', 'stir_pct'} <= Setup.model_json_schema()['properties'].keys()


@pytest.mark.parametrize('fields', [dict(coupon_position_cm=[0.]),
    dict(coupon_position_cm=[0., 0., 0.]), dict(coupon_position_cm=[float('inf'), 0.]),
    dict(stir_pct=-1.), dict(stir_pct=float('nan')), dict(unknown=1)])
def test_setup_remains_strict(fields):
    with pytest.raises(ValidationError):
        Setup(taped=True, area_cm2=1., **fields)


def test_plan_tile_sizes_height_for_local_current(monkeypatch):
    """At a coupon edge the local current exceeds the mean, so the grid must be sized for it."""
    from platesim.sample3d import planner
    data = sample()
    data['setup']['coupon_position_cm'] = [.8, .8]
    seen = []

    def plan_features(spec, faraday_um, duration, margin):
        seen.append(faraday_um)
        return []
    monkeypatch.setattr(planner, 'plan_features', plan_features)
    monkeypatch.setattr(bridge, 'tile3d_conditions',
                        lambda j, *a, **kw: dict(j_local_mA_cm2=1.5 * j))
    pipeline.plan_tile(data)
    del data['setup']['coupon_position_cm']
    pipeline.plan_tile(data)
    assert seen[0] == pytest.approx(1.5 * seen[1])
