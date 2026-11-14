"""Small CPU-only sample integration tests; no repository-local run directories."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from platesim.core.constants import F, M_CU, RHO_CU
from platesim.spec3d import Manifest, SampleSpec
from platesim.sample3d import build_geometry, plan_tile, run_sample
from platesim.tile3d.result import TileResult


def trench_spec():
    return SampleSpec.model_validate(dict(
        layout=dict(die_um=[96, 16], features=[dict(kind='rect', x_um=48,
                    y_um=8, w_um=48, h_um=16)]),
        etch=dict(default=dict(depth_um=32)),
        recipe=dict(steps=[dict(t_forward_ms=10, j_forward_mA_cm2=10, duration_h=.5)]),
        bath=dict(cu_g_L=40, h2so4_g_L=100, cl_ppm=50, temperature_C=25),
        setup=dict(taped=True, area_cm2=1, delta_um=32),
        numerics=dict(quality='preview')))


def test_plan_array_numbers():
    data = trench_spec().model_dump()
    data['layout'] = dict(die_um=[768, 128], features=[dict(kind='rect', x_um=192,
        y_um=32, w_um=48, h_um=16,
        array=dict(nx=3, ny=2, pitch_x_um=192, pitch_y_um=64))])
    plan = plan_tile(SampleSpec.model_validate(data))
    assert plan['window_um'] == [96, 0, 288, 64]
    assert plan['dx_um'] == 8
    assert plan['z_extent_um'] == [-48, 72]
    assert plan['periodic'] == [True, True, False]
    assert plan['shape'] == [24, 8, 15]
    assert plan['n_cells'] == 2880
    assert plan['est_seconds'] > 0


def test_geometry_volume_and_grid():
    spec = trench_spec()
    plan = plan_tile(spec)
    psi, grid = build_geometry(spec, plan)
    z = grid.centers()[2]
    etched = np.count_nonzero((psi > 0) & (z[None, None, :] < 0)) * grid.cell_volume * 1e12
    assert etched == pytest.approx(48 * 16 * 32, rel=.02)
    assert psi.dtype == np.float32
    assert grid.periodic == (False, False, False)
    # A point well above the field checks cm-valued distance, not um.
    assert psi[0, 0, -1] == pytest.approx(z[-1], rel=.01)
    data = spec.model_dump()
    data['layout'] = dict(die_um=[768, 128], features=[dict(kind='rect', x_um=192,
        y_um=32, w_um=48, h_um=16,
        array=dict(nx=3, ny=2, pitch_x_um=192, pitch_y_um=64))])
    array_spec = SampleSpec.model_validate(data)
    array_plan = plan_tile(array_spec)
    _, array_grid = build_geometry(array_spec, array_plan)
    assert array_grid.origin == pytest.approx((.0096, 0, -.0048))
    assert array_grid.periodic == (True, True, False)


def test_cell_limit_names_fix():
    data = trench_spec().model_dump()
    data['layout']['die_um'] = [8000, 8000]
    with pytest.raises(ValueError, match=r'(?i)(increase.*dx_um|preview|smaller.*tile)'):
        plan_tile(SampleSpec.model_validate(data))


@pytest.mark.parametrize('quality,dx', [('preview', 8), ('standard', 4), ('fine', 2)])
def test_quality_and_override(quality, dx):
    data = trench_spec().model_dump()
    data['numerics']['quality'] = quality
    assert plan_tile(data)['dx_um'] == dx
    data['numerics']['dx_um'] = 8
    data['etch']['default']['depth_vs_width'] = [[16, 32], [64, 96]]
    plan = plan_tile(data)
    assert plan['dx_um'] == 8
    assert plan['z_extent_um'][0] == -112


def test_circle_diameter_adapter_and_outward_rounding():
    data = trench_spec().model_dump()
    data['layout'] = dict(die_um=[99, 65], features=[dict(kind='circle', x_um=48,
                         y_um=32, w_um=48, h_um=48)])
    plan = plan_tile(data)
    assert plan['window_um'] == [0, 0, 104, 72]
    psi, grid = build_geometry(data, plan)
    assert psi[6, 4, 3] > 0  # a real etched cavity, despite omitted r_um
    assert grid.shape == tuple(plan['shape'])


def test_mixed_array_falls_back_to_die():
    data = trench_spec().model_dump()
    data['layout'] = dict(die_um=[192, 64], features=[
        dict(kind='rect', x_um=48, y_um=32, w_um=24, h_um=16,
             array=dict(nx=2, ny=1, pitch_x_um=96, pitch_y_um=64)),
        dict(kind='rect', x_um=96, y_um=32, w_um=16, h_um=16)])
    plan = plan_tile(data)
    assert plan['window_um'] == [0, 0, 192, 64]
    assert plan['periodic'] == [False, False, False]


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    out = tmp_path_factory.mktemp('sample3d') / 'run'
    started = time.perf_counter()
    manifest = run_sample(trench_spec(), out)
    return out, manifest, time.perf_counter() - started


def test_end_to_end(completed):
    out, manifest, wall = completed
    validated = Manifest.model_validate_json((out / 'manifest.json').read_text())
    assert validated.model_dump(mode='json') == manifest
    pngs = [out / a.path for a in validated.artifacts if a.kind == 'png']
    assert len(pngs) == 3
    assert all(p.is_file() and p.stat().st_size > 100 for p in pngs)
    assert all(q.flag == 'uncalibrated' for q in validated.quantities)
    checks = {c.name: c for c in validated.checks}
    assert checks['faraday_error_rel'].passed
    assert checks['cu_balance_rel'].passed
    q = {v.name: v.value for v in validated.quantities}
    print(f"E2E cells={plan_tile(trench_spec())['n_cells']} wall_s={wall:.3f} "
          f"field_um={q.get('field_um')} floor_um={q.get('floor_um')} "
          f"faraday_error_rel={checks['faraday_error_rel'].value}")


def test_two_steps_summed_faraday(tmp_path):
    data = trench_spec().model_dump()
    data['recipe']['steps'][0]['duration_h'] = .2
    data['recipe']['steps'].append(dict(t_forward_ms=10, t_reverse_ms=5,
        j_forward_mA_cm2=20, j_reverse_mA_cm2=10, charge_C_cm2=7.2))
    spec = SampleSpec.model_validate(data)
    out = tmp_path / 'two_steps'
    run_sample(spec, out)
    result = TileResult(out)
    total_charge = 10e-3 * .2 * 3600 + 7.2
    expected_um = total_charge * M_CU / (2 * F * RHO_CU) * 1e4
    area = result.grid.shape[0] * result.grid.shape[1] * result.grid.dx**2
    deposited_um = result.series['solid_volume_cm3'][-1] / area * 1e4
    assert deposited_um == pytest.approx(expected_um, rel=.05)
    assert result.times_s[-1] == pytest.approx(1440)
    assert result.times_s.tolist() == pytest.approx([0, 720, 1440])
    assert result.series['charge_C_cm2'][-1] == pytest.approx(total_charge)
    assert np.any(result.psi() != result.psi(1))


def test_cli_plan(tmp_path):
    path = tmp_path / 'spec.json'
    path.write_text(trench_spec().model_dump_json())
    run = subprocess.run([sys.executable, '-m', 'platesim.sample3d', 'plan', str(path)],
                         capture_output=True, text=True, check=True,
                         env={**os.environ, 'PYTHONPATH': str(Path.cwd())})
    assert json.loads(run.stdout)['n_cells'] == plan_tile(trench_spec())['n_cells']
