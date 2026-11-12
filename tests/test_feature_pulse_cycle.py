"""Tests for platesim.feature.pulse_cycle: the 1-D time-resolved PR cycle vs the duty-averaged answer.

Contract (plan item D3a):
  * DC limit (t_r = t_off = 0): the periodic solution is the steady analytic profile, linear in x,
    c_s = c_far - j delta/(2 F D), to 1e-4;
  * Faraday: the cycle-averaged deposition equals the net applied charge/(2F) to 1e-6, with and without
    the double layer (a periodic capacitor returns what it takes);
  * high-frequency limit (all phase times x 1e-3): time-resolved and duty-averaged surface concentration
    agree to 1e-3;
  * convergence: halving dt and doubling the x nodes changes the reported numbers by < 1 %.

Runs in ~30 s:  python -m pytest tests/test_feature_pulse_cycle.py -q -s
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from platesim.core.constants import F
from platesim.feature.engine.params import Params, derive
from platesim.feature.pulse_cycle import (C_DL_DEFAULT, TEAM_WAVEFORM, _steady_u_s, _geometry, _mesh,
                                          _eta_for, cycle_vs_duty_average)

KEYS = ('v_um_h', 'c_s_mean', 'c_s_min', 'c_s_max', 'eta_min_V', 'eta_max_V', 'dl_charge_frac')
DC_WF = dict(t_r_ms=0.0, t_off_ms=0.0)


def _analytic(delta_um, j_net_A, c_far_frac):
    """Steady 1-D profile of the same model: u = c_far - G (1 - X), G = delta j/(2 F D c_bulk)."""
    p, ph = Params(), derive(Params(), 22.0)
    xc, vol = _mesh(delta_um * 1e-4)
    g = _geometry(p, ph, delta_um * 1e-4, c_far_frac, 0.0, 0.0, xc, vol)
    return g, xc / (delta_um * 1e-4), _steady_u_s(g, j_net_A), g['q'] * j_net_A


# ------------------------------------------------------------------ 1. DC limit
def test_dc_limit_is_the_steady_analytic_profile():
    delta_um, c_far = 64.0, 0.8
    r = cycle_vs_duty_average(delta_um=delta_um, c_far_frac=c_far, waveform=DC_WF)
    assert r['converged'] and r['n_cycles'] <= 3
    g, X, u_s, G = _analytic(delta_um, r['j_net_mA_cm2'] * 1e-3, c_far)
    # the periodic (here: steady) profile is linear from c_far at delta to c_s at the electrode
    np.testing.assert_allclose(r['c_profile'], c_far - G * (1.0 - X), atol=1e-4)
    assert abs(r['c_s_mean'] - u_s) < 1e-4
    assert abs(r['c_s_max'] - r['c_s_min']) < 1e-4            # no pulsating layer in DC
    # Faraday at DC: the deposition is Mfac * j_net
    assert abs(r['v_um_h'] / r['da_v_um_h'] - 1.0) < 1e-6
    assert abs(r['q_faradaic_c_cm2'] / r['q_net_applied_c_cm2'] - 1.0) < 1e-6


# ------------------------------------------------------------------ 2. Faraday
@pytest.mark.parametrize('C_dl', [0.0, C_DL_DEFAULT])
def test_faraday_charge_conservation(C_dl):
    r = cycle_vs_duty_average(delta_um=64.0, c_far_frac=1.0, C_dl=C_dl)
    assert r['converged']
    # the net deposited charge over a cycle is the net applied charge (the capacitor returns its charge)
    assert abs(r['q_faradaic_c_cm2'] / r['q_net_applied_c_cm2'] - 1.0) < 1e-6
    # ... and the net rate is the Faraday rate of the NET current (the engine's duty-averaged rate)
    j_net, Mfac = r['j_net_mA_cm2'] * 1e-3, r['Mfac_cm_s']
    assert abs(r['v_um_h'] / (j_net * Mfac * 1e4 * 3600.0) - 1.0) < 1e-6
    assert abs(r['v_um_h'] / r['da_v_um_h'] - 1.0) < 1e-6
    if C_dl > 0:
        # the double layer does carry charge every cycle, it just integrates to zero
        assert 0.02 < r['dl_charge_frac'] < 0.15
        assert 0.1 < r['rev_dl_frac'] < 0.6
    else:
        assert r['dl_charge_frac'] == 0.0 and r['rev_dl_frac'] == 0.0


# ------------------------------------------------------------------ 3. high-frequency limit
def test_high_frequency_limit_matches_duty_average():
    hf = dict(TEAM_WAVEFORM)
    for k in ('t_f_ms', 't_r_ms', 't_off_ms'):
        hf[k] = TEAM_WAVEFORM[k] * 1e-3
    r = cycle_vs_duty_average(delta_um=64.0, c_far_frac=1.0, waveform=hf)
    assert r['converged']
    # no time for a pulsating layer: the surface concentration is the duty-averaged one
    assert abs(r['c_s_mean'] / r['da_c_s'] - 1.0) < 1e-3
    assert abs(r['c_s_max'] - r['c_s_min']) < 1e-3 * r['c_s_mean']
    assert abs(r['rel_diff_c_s_mean']) < 1e-3


# ------------------------------------------------------------------ 4. convergence
def test_convergence_dt_and_mesh():
    base = cycle_vs_duty_average(delta_um=64.0, c_far_frac=0.7)
    fine = cycle_vs_duty_average(delta_um=64.0, c_far_frac=0.7, dt_scale=0.5, n_x_factor=2.0)
    assert base['converged'] and fine['converged']
    assert fine['n_cells'] > 1.8 * base['n_cells']
    assert fine['n_steps'] > 1.5 * base['n_steps']
    for k in KEYS:
        assert abs(fine[k] / base[k] - 1.0) < 1e-2, (k, base[k], fine[k])


# ------------------------------------------------------------------ 5. the team waveform table
def test_print_team_waveform_table():
    print('\nteam waveform 9/0.5/0.5 ms, rr 2, 15 mA/cm2 net, delta 64 um, C_dl %g uF/cm2'
          % (C_DL_DEFAULT * 1e6))
    print(' cfar    v_res   v_duty  d_v%  cs_mean  cs_duty  d_cs%  cs_min  cs_max   eta_lo   eta_hi '
          '  dl%  revDL% cyc')
    for cf in (1.0, 0.4):
        r = cycle_vs_duty_average(delta_um=64.0, c_far_frac=cf)
        print('%5.2f %8.3f %8.3f %+6.2f %8.5f %8.5f %+6.3f %7.4f %7.4f %+7.4f %+7.4f %6.2f %6.1f %3d'
              % (cf, r['v_um_h'], r['da_v_um_h'], 100 * r['rel_diff_v'], r['c_s_mean'], r['da_c_s'],
                 100 * r['rel_diff_c_s_mean'], r['c_s_min'], r['c_s_max'], r['eta_min_V'],
                 r['eta_max_V'], 100 * r['dl_charge_frac'], 100 * r['rev_dl_frac'], r['n_cycles']))
        assert r['converged']
        assert abs(r['rel_diff_v']) < 1e-6                 # charge conservation: no rate error at all
        assert r['c_s_min'] > 0.2                          # still far from the transport limit
