"""Public synthetic checks for cycle-resolved plating; no team fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest

from platesim.feature.engine.params import Params
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.pulse import capacitive_cycle, diffusion_response, surface_at, surface_concentration
from platesim.tile3d.sim import run_tile

PR = dict(t_f_ms=9., t_r_ms=.5, t_off_ms=.5,
          j_f_mA_cm2=18.75, j_r_mA_cm2=37.5, model='resolved')


def geometry(rough=False):
    grid = Grid3D.from_um((32, 8, 104), 4., origin_um=(0, 0, -16))
    x, _, z = grid.meshgrid()
    return grid, z - (1e-4 * np.cos(2 * np.pi * x / (32e-4)) if rough else 0.)


def solve(pulse=None, params=None, rough=False):
    grid, psi = geometry(rough)
    out = FieldSolver(params, pulse=pulse).solve(psi, grid.dx, grid.origin[2], 64e-4, .015)
    return out, grid


@pytest.mark.parametrize('pulse', [None, {'model': 'net'}, dict(PR, model='net'), dict(PR,
    t_r_ms=0., t_off_ms=0., j_f_mA_cm2=15.)])
def test_legacy_and_dc_limit_bit_exact(pulse):
    base, _ = solve(rough=True)
    out, _ = solve(pulse, rough=True)
    for key in ('ct', 'phi', 'i', 'speed', 'w', 'V'):
        np.testing.assert_array_equal(out[key], base[key])


def test_phase_charge_and_outer_cu_balance():
    out, grid = solve(PR, rough=True)
    pulse = out['info']['pulse']
    assert pulse['converged']
    assert pulse['phase_current_A_cm2'] == pytest.approx([.01875, -.0375, 0.], abs=1e-10)
    assert abs(pulse['faraday_error_rel']) < 1e-8
    assert abs(out['info']['cu_balance_rel']) < .005
    integral = np.sum(out['w'] * out['speed']) * grid.dx**3
    expected = .015 * grid.shape[0] * grid.shape[1] * grid.dx**2 * FieldSolver().ph.Mfac
    assert integral / expected == pytest.approx(1., abs=.005)
    assert all(t > 0 for t in pulse['tau_s'][:2])
    assert pulse['tau_s'][2] is None  # off has zero applied j: not a finite charging time


def test_large_exchange_flat_ohmic_limit():
    # A Tafel branch does NOT linearise by increasing i0: V merely shifts.
    # On a planar electrode the ohmic limit has an exact uniform lateral speed;
    # compare that observable, not the unresolved diffuse-band normal profile.
    params = Params(suppressor=False, i0_u=1e3)
    dc, grid = solve(params=params)
    pr, _ = solve(PR, params=params)
    column = lambda o: np.sum(o['w'] * o['speed'], axis=2) * grid.dx
    np.testing.assert_allclose(column(pr), column(dc), rtol=2e-6)


def test_time_loop_faraday():
    grid, psi = geometry()
    out = run_tile(psi, grid, 15., 180., delta_um=64., pulse=PR, n_frames=3, backend='numpy')
    assert abs(out['summary']['faraday_error_rel']) < .005


def test_periodic_diffusion_matches_abel_rectangular_pulses():
    # Abel solution for repeated rectangular on pulses, evaluated at end on:
    # c=-2 J/sqrt(pi D) sum(sqrt(age_start)-sqrt(age_end)).
    # Subtract DC ramp and its asymptotic offset to isolate the zero-mean
    # periodic solution. Euler-Maclaurin tail removes the O(N**-1/2) start-up term.
    T, duty, J, D = .01, .4, 2e-7, 4e-6
    N = 100_000
    age = np.arange(N, dtype=float)
    periodic_sum = (np.sum(np.sqrt(age + duty) - np.sqrt(age)) - duty * np.sqrt(N)
                    + duty * (1-duty) / (4 * np.sqrt(N)))
    exact = -2 * J * np.sqrt(T / (np.pi * D)) * periodic_sum
    actual = surface_at(duty * T, [J, 0., 0.], (duty*T, 0., (1-duty)*T), D)
    assert actual == pytest.approx(exact, rel=3e-4)


def test_diffusion_phase_mean_conservation_and_harmonic_convergence():
    ts, D = (.009, .0005, .0005), 4e-6
    a = diffusion_response(ts, D)
    b = diffusion_response(ts, D, 2048)
    np.testing.assert_allclose(a, b, atol=.005, rtol=1e-4)
    c = surface_concentration(np.ones(5), np.tile([.01875, -.0375, 0.], (5, 1)).T,
                              ts, D, .001)
    np.testing.assert_allclose(np.array(ts) / sum(ts) @ c, 1., atol=1e-14)


def test_double_layer_charge_is_cyclic():
    out, _ = solve(PR, Params(suppressor=False, C_dl_F_cm2=30e-6))
    report = out['info']['pulse']
    assert report['converged']
    assert sum(report['capacitive_charge_C_cm2']) == pytest.approx(0., abs=1e-16)
    assert abs(report['faraday_error_rel']) < 1e-8
    assert report['phase_current_A_cm2'][2] != 0.  # off-time relaxation is Faradaic


def test_periodic_rc_reverse_loss_and_time_constants():
    p = Params(suppressor=False)
    ph = FieldSolver(p).ph
    ts, applied = (.009, .0005, .0005), np.array([.01875, -.0375, 0.])
    effective, report = capacitive_cycle(ts, applied, 30e-6, ph.i0_u,
        p.alpha_u * ph.f_RT, p.pulse_alpha_a * ph.f_RT, [.9, .9, .9])
    assert report['periodic_eta_error_V'] < 1e-10
    assert abs(report['cycle_charge_error_C_cm2']) < 1e-12
    assert np.dot(ts, effective) == pytest.approx(np.dot(ts, applied), rel=1e-8)
    assert .15 < report['reverse_capacitive_fraction'] < .4
    assert abs(effective[1]) < abs(applied[1])
    assert effective[2] < 0.  # the charged anodic double layer dissolves Cu during off
    assert report['differential_tau_s'][0] == pytest.approx(30e-6/(p.alpha_u*ph.f_RT*.01875), rel=.02)
    assert report['differential_tau_s'][1] == pytest.approx(30e-6/(p.pulse_alpha_a*ph.f_RT*.0375), rel=.03)
    _, fine = capacitive_cycle(ts, applied, 30e-6, ph.i0_u,
        p.alpha_u*ph.f_RT, p.pulse_alpha_a*ph.f_RT, [.9, .9, .9], rtol=2e-9, method='Radau')
    np.testing.assert_allclose(report['eta_end_V'], fine['eta_end_V'], atol=2e-7)


def test_capacitance_off_reports_but_does_not_remove_charge():
    p = Params(suppressor=False)
    ph = FieldSolver(p).ph
    ts, applied = (.009, .0005, .0005), [.01875, -.0375, 0.]
    effective, report = capacitive_cycle(ts, applied, 0., ph.i0_u,
        p.alpha_u*ph.f_RT, p.pulse_alpha_a*ph.f_RT, [1., 1., 1.])
    np.testing.assert_array_equal(effective, applied)
    assert report['C_dl_report_F_cm2'] == 30e-6
    assert report['reverse_capacitive_fraction'] == 0.
    assert report['diagnostic_reverse_capacitive_fraction'] > .15


def test_rc_matches_exact_linear_circuit():
    p = Params(suppressor=False)
    ph = FieldSolver(p).ph
    C = 30e-6
    dt = np.array([.009, .0005, .0005])
    applied = np.array([1e-7, -2e-7, 0.])
    conductance = ph.i0_u * (p.alpha_u + p.pulse_alpha_a) * ph.f_RT
    decay = np.exp(-dt * conductance / C)
    equilibrium = -applied / conductance
    offset = 0.
    for a, e in zip(decay, equilibrium):
        offset = a*offset + (1-a)*e
    eta = offset / (1-decay.prod())
    exact = applied.copy()
    for k in range(3):
        end = decay[k]*eta + (1-decay[k])*equilibrium[k]
        exact[k] += C*(end-eta)/dt[k]
        eta = end
    actual, _ = capacitive_cycle(dt, applied, C, ph.i0_u, p.alpha_u*ph.f_RT,
        p.pulse_alpha_a*ph.f_RT, [1., 1., 1.], rtol=2e-10)
    np.testing.assert_allclose(actual, exact, rtol=2e-4, atol=1e-14)


def test_spec_roundtrip():
    from platesim.spec3d.models import SampleSpec, Numerics
    data = json.loads(Path('platesim/spec3d/examples/single_trench.json').read_text())
    data['numerics'].update(pulse_model='resolved', frames_per_step=5)
    spec = SampleSpec.model_validate(data)
    assert SampleSpec.model_validate_json(spec.model_dump_json()) == spec
    assert Numerics().pulse_model == 'net' and Numerics().frames_per_step == 2


def test_pipeline_passes_waveform_and_saves_intermediate_frames(tmp_path, monkeypatch):
    from platesim.sample3d import run_sample
    from platesim.tile3d import sim
    from platesim.tile3d.result import TileResult
    data = json.loads(Path('platesim/spec3d/examples/single_trench.json').read_text())
    data['layout'] = dict(die_um=[300., 300.], features=[dict(
        kind='circle', x_um=150., y_um=150., w_um=32., h_um=32.)])
    data['etch']['default'] = dict(depth_um=16.)
    data['setup']['delta_um'] = 24.
    data['numerics'] = dict(quality='preview', dx_um=4., pulse_model='resolved', frames_per_step=4)
    data['recipe']['steps'] = [dict(t_forward_ms=9., t_reverse_ms=.5, t_off_ms=.5,
        j_forward_mA_cm2=18.75, j_reverse_mA_cm2=37.5, duration_h=2./3600)]
    real = sim.run_tile
    captured = []
    def capture(*args, **kwargs):
        captured.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(sim, 'run_tile', capture)
    manifest = run_sample(data, tmp_path / 'recipe', field_margin_um=16.)
    assert captured and all(k['pulse'] == PR for k in captured)
    assert all(k['n_frames'] == 4 for k in captured)
    plans = json.loads((tmp_path / 'recipe' / 'tiles.json').read_text())
    for plan in plans:
        result = TileResult(tmp_path / 'recipe' / 'tiles' / plan['tile_id'])
        np.testing.assert_allclose(result.times_s, np.linspace(0., 2., 4))
        assert result.series['charge_C_cm2'][-1] == pytest.approx(.03)
    assert 'cycle-resolved' in manifest['waveform_sensitivity']
    assert any(q['name'] == 'step_1.tau_forward_s' for q in manifest['quantities'])


@pytest.mark.parametrize('change', [dict(t_f_ms=-1), dict(j_r_mA_cm2=-1), dict(model='bad'),
                                   dict(j_f_mA_cm2=20)])
def test_invalid_waveform(change):
    with pytest.raises(ValueError):
        solve(dict(PR, **change))


def test_resolved_pulse_with_shear_flow_runs_and_conserves_charge():
    """Resolved pulse-reverse combined with resolved shear flow (never exercised end to end before)."""
    grid, psi = geometry(True)
    lid = (grid.origin[2] + grid.shape[2] * grid.dx) * 1e4 - 8.0
    out = run_tile(psi, grid, 15., 120., delta_um=64., n_frames=2, dt_max=60., backend='numpy',
                   pulse=PR, flow=dict(shear_rate_s=20., lid_um=None, update_every=1))
    summary = out['summary']
    assert summary['pulse']['model'] == 'resolved' and summary['pulse']['converged']
    assert abs(summary['faraday_error_rel']) < 0.05
    assert np.isfinite(out['psi']).all()


def test_coupon_position_scales_the_resolved_waveform(tmp_path, monkeypatch):
    """With a coupon position the local current differs from the recipe net current; the resolved
    waveform must be scaled with it (Waveform.parse requires its average to equal the solved current)."""
    from platesim.multiscale import bridge
    from platesim.sample3d.pipeline import run_sample
    from platesim.tile3d import sim
    data = json.loads(Path('platesim/spec3d/examples/single_trench.json').read_text())
    data['layout'] = dict(die_um=[300., 300.], features=[dict(
        kind='circle', x_um=150., y_um=150., w_um=32., h_um=32.)])
    data['etch']['default'] = dict(depth_um=16.)
    data['setup']['delta_um'] = 24.
    data['setup']['coupon_position_cm'] = [.5, .5]
    data['numerics'] = dict(quality='preview', dx_um=4., pulse_model='resolved')
    data['recipe']['steps'] = [dict(t_forward_ms=9., t_reverse_ms=.5, t_off_ms=.5,
        j_forward_mA_cm2=18.75, j_reverse_mA_cm2=37.5, duration_h=2./3600)]
    monkeypatch.setattr(bridge, 'tile3d_conditions', lambda j, *a, **k: dict(
        j_local_mA_cm2=1.3 * j, delta_um=24., efficiency=.97, scale=1.3, site={}))
    real, captured = sim.run_tile, []

    def capture(*args, **kwargs):
        captured.append((args[2], kwargs['pulse']))
        return real(*args, **kwargs)
    monkeypatch.setattr(sim, 'run_tile', capture)
    run_sample(data, tmp_path / 'run', field_margin_um=16.)
    for j_local, pulse in captured:
        assert j_local == pytest.approx(1.3 * 15.)
        assert pulse['j_f_mA_cm2'] == pytest.approx(1.3 * 18.75)
        assert pulse['j_r_mA_cm2'] == pytest.approx(1.3 * 37.5)
