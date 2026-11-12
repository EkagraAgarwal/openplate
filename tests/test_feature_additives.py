"""Tests for platesim.feature.engine.additives (plan v2 D2, additive surface chemistry).

Contract:
  1. each species reaches the analytic Langmuir steady state a/(a + k_des [+ k_inc v]);
  2. CEAC: a shrinking circle with no adsorption keeps theta*R constant to 1e-10 over many
     steps (enrichment ~ 1/R), both through the 1-D arclength step and the pure geometric
     area-conserving update;
  3. the 1-D area-conserving update conserves sum(theta*L) exactly and caps at 1 on growth;
  4. Cl-mediated suppressor adsorption is exactly zero at theta_cl = 0 and monotone in theta_cl;
  5. leveller coverage orders with the local supply k_l c_l (protrusions get more);
  6. all coverages stay in [0, 1] for absurd dt / concentration / velocity (exponential
     integrator, not explicit Euler), and reach the steady state in one stiff step.

Runs in well under a second.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine.additives import (
    AdditiveSet,
    accelerator_step,
    accelerator_step_arclength,
    area_conserving_update,
    displace_suppressor,
    langmuir_steady_state,
    leveller_i0_factor,
    leveller_sink,
    reverse_pulse_desorption,
    reverse_pulse_dissolution_factor,
    suppressor_cl_rate,
    suppressor_i0_factor,
    suppressor_sink,
)


# ------------------------------------------------------------------ 1. Langmuir steady states
def test_chloride_langmuir_steady_state():
    s = AdditiveSet()
    c, v = 2.0e-6, 1.0e-6                     # mol/cm3, cm/s
    a = s.k_cl * c / s.Gamma_cl
    th_ss = langmuir_steady_state(a, s.k_des_cl + s.k_inc_cl * v)
    assert th_ss == pytest.approx(a / (a + s.k_des_cl + s.k_inc_cl * v))
    th = 0.0
    for _ in range(600):
        th = s.chloride_step(th, 0.5, c_cl=c, v=v)
    assert abs(th - th_ss) < 1e-12
    # and the stiff shortcut: one huge dt lands on the same point
    assert abs(float(s.chloride_step(0.0, 1.0e9, c_cl=c, v=v)) - th_ss) < 1e-12


def test_suppressor_langmuir_steady_state():
    s = AdditiveSet()
    th_cl, v = 0.7, 1.0e-6
    a = suppressor_cl_rate(s.c_s, th_cl, s.k_plus, n_cl=s.n_cl)
    th_ss = langmuir_steady_state(a, s.k_minus * v + s.k_des_s)
    th = 0.0
    for _ in range(600):
        th = s.suppressor_step(th, 0.5, th_cl, v=v)
    assert abs(th - th_ss) < 1e-12
    assert 0.0 < th < 1.0
    # full Cl coverage reduces to the plain Langmuir form with a = k_plus c_s
    assert float(suppressor_cl_rate(s.c_s, 1.0, s.k_plus)) == pytest.approx(s.k_plus * s.c_s)


def test_accelerator_langmuir_steady_state():
    s = AdditiveSet()
    v = 1.0e-6
    a = s.k_a * s.c_a
    th_ss = langmuir_steady_state(a, s.k_inc_a * v)
    th = 0.0
    for _ in range(600):
        th = s.accelerator_step(th, 0.5, v=v, kappa=0.0)   # no area change: pure adsorption/burial
    assert abs(th - th_ss) < 1e-12
    # incorporation alone (no adsorption) empties the layer (exp(-R dt) underflows to the cut)
    assert float(accelerator_step(1.0, 1.0e9, 0.0, v=v, k_inc=s.k_inc_a)) < 1e-100


def test_leveller_langmuir_steady_state():
    s = AdditiveSet()
    v = 1.0e-6
    a = s.k_l * s.c_l / s.Gamma_l
    th_ss = langmuir_steady_state(a, s.k_des_l + s.k_inc_l * v)
    th = 0.0
    for _ in range(600):
        th = s.leveller_step(th, 0.5, v=v)
    assert abs(th - th_ss) < 1e-12
    # the sink and the coverage rate are the same mechanism: Gamma * dtheta/dt ~= sink at low theta
    dt = 1e-6
    th1 = float(s.leveller_step(0.0, dt, v=0.0))
    assert s.Gamma_l * th1 / dt == pytest.approx(float(leveller_sink(s.c_l, 0.0, s.k_l)), rel=1e-6)


# ------------------------------------------------------------------ 2. CEAC on a shrinking circle
@pytest.mark.parametrize("use_arclength", [True, False])
def test_ceac_shrinking_circle_conserves_theta_times_R(use_arclength):
    """No adsorption, no incorporation: theta*R is invariant on a shrinking circle (1/R enrichment)."""
    R0, v, dt, n = 1.0e-4, 2.0e-8, 1.0, 2000      # cm, cm/s, s, steps (R shrinks by 40 %)
    theta0 = 0.4
    theta, R = theta0, R0
    for _ in range(n):
        R_new = R - v * dt
        if use_arclength:
            theta = float(accelerator_step_arclength(theta, dt, 0.0, v, R, R_new, k_a=0.0))
        else:
            theta = float(area_conserving_update(theta, R, R_new))
        R = R_new
    assert theta * R == pytest.approx(theta0 * R0, rel=1e-10)
    assert theta == pytest.approx(theta0 * R0 / R, rel=1e-12)          # enrichment ~ 1/R
    assert theta > theta0                                              # strictly enriched


def test_ceac_arclength_matches_geometric_update_with_adsorption_off():
    L_old = np.array([1.0, 2.0, 5.0])
    L_new = np.array([0.5, 1.0, 10.0])
    th = np.array([0.4, 0.3, 0.2])
    np.testing.assert_allclose(accelerator_step_arclength(th, 1.0, 0.0, 0.0, L_old, L_new, k_a=0.0),
                               area_conserving_update(th, L_old, L_new), rtol=0, atol=1e-15)


def test_ceac_curvature_term_has_the_geometry_sign():
    """Continuum form: -theta kappa v, and dA/dt = kappa v A, so a growing (positive) area dilutes."""
    th_grow = float(accelerator_step(0.5, 1.0, 0.0, v=+1e-4, kappa=1e3, k_a=0.0))
    th_shrink = float(accelerator_step(0.5, 1.0, 0.0, v=-1e-4, kappa=1e3, k_a=0.0))
    assert abs(th_grow - 0.5 * np.exp(-1e-1)) < 1e-15
    assert abs(th_shrink - 0.5 * np.exp(+1e-1)) < 1e-15
    # a larger shrink caps at the monolayer
    assert float(accelerator_step(0.5, 1.0, 0.0, v=-1e-3, kappa=1e3, k_a=0.0)) == 1.0


def test_accelerator_displaces_suppressor():
    assert float(displace_suppressor(0.9, 0.6)) == pytest.approx(0.4)
    assert float(displace_suppressor(0.2, 0.6)) == pytest.approx(0.2)
    assert float(displace_suppressor(0.9, 1.0)) == 0.0


# ------------------------------------------------------------------ 3. 1-D area conservation
def test_area_conserving_update_conserves_sum_theta_L():
    rng = np.random.default_rng(7)
    n = 200
    th = rng.uniform(0.0, 0.4, n)
    L_old = rng.uniform(0.5, 2.0, n)
    L_new = L_old * rng.uniform(1.0, 3.0, n)              # growth only: no cap, conservation exact
    th_new = area_conserving_update(th, L_old, L_new)
    assert np.all(th_new <= 1.0 + 1e-15)
    assert float(np.sum(th_new * L_new)) == pytest.approx(float(np.sum(th * L_old)), rel=1e-13)


def test_area_conserving_update_caps_at_one():
    th = np.array([1.0, 0.5, 0.0])
    L_old = np.array([2.0, 2.0, 2.0])
    L_new = np.array([1.0, 1.0, 1.0])                      # halving the lengths doubles theta
    np.testing.assert_allclose(area_conserving_update(th, L_old, L_new), [1.0, 1.0, 0.0], atol=1e-15)


def test_area_conserving_update_degenerate_lengths():
    out = area_conserving_update(np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 2.0]))
    np.testing.assert_allclose(out, [0.0, 0.25])


# ------------------------------------------------------------------ 4. Cl-mediated S adsorption
def test_suppressor_adsorption_is_zero_without_chloride_and_monotone():
    s = AdditiveSet()
    th_cl = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    rate = suppressor_cl_rate(s.c_s, th_cl, s.k_plus)
    assert rate[0] == 0.0                                             # exactly zero, not ~1e-30
    assert np.all(np.diff(rate) > 0.0)                                # strictly monotone
    sink = suppressor_sink(s.c_s, 0.0, th_cl, s.Gamma_s, s.k_plus)
    assert sink[0] == 0.0
    assert np.all(np.diff(sink) > 0.0)
    # a surface with no Cl never grows suppressor coverage from zero
    th = 0.0
    for _ in range(100):
        th = s.suppressor_step(th, 10.0, 0.0, v=0.0)
    assert th == 0.0
    # and the Cl-gated rate is linear in theta_cl (n_cl = 1 default)
    np.testing.assert_allclose(rate, s.k_plus * s.c_s * th_cl, rtol=1e-15)


def test_suppressor_i0_factor_matches_the_engine_interpolation():
    th = np.array([0.0, 0.25, 0.5, 1.0])
    f = suppressor_i0_factor(th, 2.0e-3, 1.3e-5)
    np.testing.assert_allclose(f, (1 - th) * 2.0e-3 / 2.0e-3 + th * 1.3e-5 / 2.0e-3, rtol=1e-15)
    assert f[0] == 1.0
    assert f[-1] == pytest.approx(1.3e-5 / 2.0e-3)


# ------------------------------------------------------------------ 5. leveller supply ordering
def test_leveller_coverage_larger_where_flux_is_larger():
    s = AdditiveSet()
    c_l = np.array([0.0, 1.0e-9, 5.0e-9, 1.0e-8, 5.0e-8])            # local front concentration
    th = np.zeros_like(c_l)
    for _ in range(50):
        th = s.leveller_step(th, 1.0, c_l=c_l, v=0.0)
    assert np.all(np.diff(th) > 0.0)                                  # monotone in the supply
    assert np.all((th >= 0.0) & (th <= 1.0))
    # 50 steps of dt = 1 s reach the Langmuir point a/(a + k_des + k_inc v), a = k_l c_l/Gamma_l
    a = s.k_l * c_l / s.Gamma_l
    np.testing.assert_allclose(th, a / (a + s.k_des_l + s.k_inc_l * 0.0), rtol=1e-10)


def test_leveller_blocking_factor():
    assert float(leveller_i0_factor(0.0)) == 1.0
    assert float(leveller_i0_factor(1.0)) == 0.0
    assert leveller_i0_factor(np.linspace(0, 1, 11))[3] == pytest.approx(0.7)
    # exponent > 1 blocks less at low coverage, never leaves [0, 1]
    f = leveller_i0_factor(np.linspace(0, 1, 21), exponent=1.5)
    assert np.all((f >= 0.0) & (f <= 1.0))


# ------------------------------------------------------------------ 6. bounded for any dt
def test_coverages_bounded_under_large_dt():
    s = AdditiveSet()
    huge_dt = 1.0e12
    th0 = np.array([0.0, 1.0, 0.5])
    cases = [
        s.chloride_step(th0, huge_dt, c_cl=1.0e-2, v=+1e-4),
        s.chloride_step(th0, huge_dt, c_cl=1.0e-2, v=-1e-4),
        s.suppressor_step(th0, huge_dt, 0.5, c_s=1.0e-2, v=+1e-4),
        s.accelerator_step(th0, huge_dt, c_a=1.0e-2, v=+1e-4, kappa=+1e6),
        s.accelerator_step(th0, huge_dt, c_a=1.0e-2, v=-1e-4, kappa=-1e6),
        accelerator_step_arclength(th0, huge_dt, 1.0e-2, 0.0, np.array([1.0, 1.0, 1.0]),
                                   np.array([1e-9, 1.0, 10.0])),
        s.leveller_step(th0, huge_dt, c_l=1.0e-2, v=-1e-4),
        s.reverse_pulse_desorption(th0, huge_dt, i_a=1.0),
    ]
    for out in cases:
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))


def test_stiff_step_is_the_steady_state_not_an_explosion():
    """Explicit Euler would give theta ~ -1e11 at dt = 1e9; the exponential integrator saturates."""
    s = AdditiveSet()
    v = 1.0e-6
    a = s.k_l * s.c_l / s.Gamma_l
    th_ss = a / (a + s.k_des_l + s.k_inc_l * v)
    assert float(s.leveller_step(0.0, 1.0e9, v=v)) == pytest.approx(th_ss, rel=1e-12)


def test_steady_state_is_independent_of_step_size():
    s = AdditiveSet()
    v = 1.0e-6
    fine = 0.0
    for _ in range(10_000):
        fine = s.chloride_step(fine, 0.01, v=v)
    coarse = 0.0
    for _ in range(10):
        coarse = s.chloride_step(coarse, 10.0, v=v)
    assert abs(float(fine) - float(coarse)) < 1e-6


# ------------------------------------------------------------------ 7. reverse pulse
def test_reverse_pulse_desorption_decays_exponentially_and_is_null_at_zero_current():
    th0, dt, i_a, k_rd = 0.8, 5.0e-3, 2.0e-3, 20.0
    out = float(reverse_pulse_desorption(th0, dt, i_a, k_rd=k_rd))
    assert out == pytest.approx(th0 * np.exp(-k_rd * i_a / 1.0e-3 * dt), rel=1e-15)
    assert out < th0
    assert float(reverse_pulse_desorption(th0, dt, 0.0, k_rd=k_rd)) == th0     # nothing to dissolve
    assert float(reverse_pulse_desorption(th0, dt, i_a, k_rd=0.0)) == th0      # k_rd off = legacy
    # monotone in the anodic current
    ser = reverse_pulse_desorption(th0, dt, np.array([0.0, 1e-3, 5e-3, 1e-2]), k_rd=k_rd)
    assert np.all(np.diff(ser) < 0.0)


def test_reverse_pulse_dissolution_factor():
    assert float(reverse_pulse_dissolution_factor(0.0, 0.0, 0.0)) == 1.0
    # covered surface dissolves less
    f = reverse_pulse_dissolution_factor(np.linspace(0, 1, 5))
    assert np.all(np.diff(f) < 0.0)
    assert np.all((f >= 0.0) & (f <= 1.0))
    assert float(reverse_pulse_dissolution_factor(1.0, 1.0, 1.0)) == 0.0


# ------------------------------------------------------------------ 8. composition
def test_combined_i0_factor_composition():
    s = AdditiveSet()
    clean = float(s.combined_i0_factor(0.0, 0.0, 0.0))
    assert clean == 1.0
    th_a = np.linspace(0.0, 1.0, 6)
    f = s.combined_i0_factor(np.zeros_like(th_a), th_a, np.zeros_like(th_a))
    np.testing.assert_allclose(f, 1.0 + s.b_acc * th_a, rtol=1e-15)         # accelerator raises i0
    assert np.all(np.diff(f) > 0.0)
    th_s = np.array([0.0, 0.3, 0.7, 1.0])
    g = s.combined_i0_factor(th_s, 0.0, 0.0)
    assert np.all(np.diff(g) < 0.0)                                         # suppressor lowers i0
    assert g[-1] == pytest.approx(s.i0_s / s.i0_u)
    h = s.combined_i0_factor(0.0, 0.0, np.array([0.0, 0.5, 1.0]))
    assert h[-1] == pytest.approx(0.0)                                      # full leveller blocks
    # displacement: theta_s is limited to 1 - theta_a before the product
    assert float(s.combined_i0_factor(0.9, 0.6, 0.0)) == pytest.approx(float(s.combined_i0_factor(0.4, 0.6, 0.0)))
    # no Cl -> no suppressor coverage -> no suppression, whatever the bulk S concentration
    assert float(s.suppressor_step(0.0, 1.0e6, 0.0, c_s=s.c_s)) == 0.0
