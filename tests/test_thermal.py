"""Tests for platesim.cell.thermal (plan v2 D7: Joule heating and temperature rise).

Run: uv run --with pytest python -m pytest tests/test_thermal.py -q -p no:cacheprovider
(must finish in well under 60 s)
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from platesim.cell import thermal as th
from platesim.core.bath import Bath

# ---------------------------------------------------------------- lumped model

def test_lumped_matches_analytic_closed_form():
    """RK4 of C dT/dt = Q - UA (T - T_inf) vs T = T_inf + (Q/hA)(1 - exp(-t/tau))."""
    lt = th.lumped_parameters(0.5, 2.0, 1.0, T0_C=22.0, T_inf_C=22.0)
    t, T = lt.T_history(4.5 * 3600.0, n_steps=2000)

    # closed form written out independently of the module
    q_over_hA = lt.Q_W / lt.UA_W_K
    tau = lt.C_J_K / lt.UA_W_K
    analytic = lt.T_inf_C + q_over_hA * (1.0 - np.exp(-t / tau))

    dT_scale = q_over_hA
    assert np.max(np.abs(T - analytic)) / dT_scale < 1e-6
    # and the module's own closed form agrees with the written-out one
    assert np.max(np.abs(lt.T_analytic(t) - analytic)) / dT_scale < 1e-6
    # monotone approach to the steady rise from below
    assert np.all(np.diff(T) > 0.0)
    assert T[-1] < lt.T_steady_C


def test_lumped_steady_state_and_time_constant():
    lt = th.lumped_parameters(1.0, 1.0, 0.5)
    assert lt.dT_steady_K == pytest.approx(lt.Q_W / lt.UA_W_K)
    assert lt.tau_s == pytest.approx(lt.C_J_K / lt.UA_W_K)
    # T(t) at t >> tau is the steady value
    assert lt.dT_at(1e7) == pytest.approx(lt.dT_steady_K, rel=1e-9)
    # doubling the loss area (same h) halves the steady rise and the time constant
    lt2 = th.lumped_parameters(1.0, 1.0, 0.5,
                               area_cm2=2.0 * th.beaker_surface_area_cm2(0.5))
    assert lt2.dT_steady_K == pytest.approx(0.5 * lt.dT_steady_K, rel=1e-12)
    assert lt2.tau_s == pytest.approx(0.5 * lt.tau_s, rel=1e-12)
    # extra (beaker glass) heat capacity slows the response without changing dT_inf
    lt3 = th.lumped_parameters(1.0, 1.0, 0.5, extra_heat_capacity_J_K=500.0)
    assert lt3.dT_steady_K == pytest.approx(lt.dT_steady_K, rel=1e-12)
    assert lt3.tau_s > lt.tau_s


def test_ohmic_resistance_and_i2r_power():
    R = th.ohmic_resistance_ohm(0.065, 4.0, 4.0)          # gap 4 cm, 4 cm2 coupon
    assert R == pytest.approx(4.0 / (0.065 * 4.0), rel=1e-15)
    # 12 mA/cm2 on the 2x2 cm coupon = 0.048 A
    I = 0.048
    lt = th.lumped_parameters(I, None, 1.0, R_ohm=R)
    assert lt.Q_W == pytest.approx(I ** 2 * R, rel=1e-15)
    # ohmic drop is a fraction of the cell voltage, so I^2 R <= I V_cell
    assert I * R < 2.0
    assert lt.Q_W < I * 2.0
    with pytest.raises(ValueError):
        th.lumped_parameters(0.5, None, 1.0)          # neither V_cell nor R_ohm
    with pytest.raises(ValueError):
        th.lumped_parameters(0.5, 2.0, 0.0)           # zero volume


# ---------------------------------------------------------------- 2-D finite volume

def _parabola_case(nx=41, ny=9, Lx=4.0, Ly=2.0, q=2.0e-3, k=th.WATER_K_W_CM_K,
                   T_wall=25.0):
    """Uniform source, Dirichlet left/right, insulated top/bottom -> 1-D parabola."""
    hf = th.solve_heat_2d(q, Lx, Ly, nx=nx, ny=ny, k_W_cmK=k, T_wall_C=T_wall,
                          walls={"left": "dirichlet", "right": "dirichlet",
                                 "top": "insulated", "bottom": "insulated"})
    a = Lx / 2.0
    xc = hf.x_cm - a                        # x_cm is in [0, Lx]; centre the parabola
    exact = T_wall + q / (2.0 * k) * (a * a - xc * xc)
    return hf, exact


def test_2d_fv_matches_analytic_parabola():
    hf, exact = _parabola_case()
    # temperature field
    assert np.max(np.abs(hf.T_C - exact[None, :])) / hf.dT_max_K < 1e-2
    # the 1-D solution must be y-independent
    assert np.max(np.ptp(hf.T_C, axis=0)) < 1e-12
    # peak and mean rise of the parabola q/(2k)(a^2 - x^2): peak q a^2/(2k),
    # mean q a^2/(3k) (sampled at the 41 cell centres)
    a = hf.Lx_cm / 2.0
    q, k = 2.0e-3, hf.k_W_cmK
    assert hf.dT_max_K == pytest.approx(q * a * a / (2.0 * k), rel=1e-2)
    assert hf.dT_mean_K == pytest.approx(q * a * a / (3.0 * k), rel=1e-2)
    assert hf.dT_K.min() > 0.0


def test_2d_energy_balance_uniform_source():
    hf = th.solve_heat_2d(2.0e-3, 4.0, 2.0, nx=20, ny=12, T_wall_C=22.0)
    assert hf.source_W > 0.0
    assert hf.energy_balance_rel < 1e-6
    assert hf.discrete_residual_rel < 1e-6
    assert hf.boundary_flux_W == pytest.approx(hf.source_W, rel=1e-6)
    # total source equals q * volume of the section (per cm depth)
    assert hf.source_W == pytest.approx(2.0e-3 * 4.0 * 2.0, rel=1e-12)
    # heat must leave through the (Dirichlet) walls, i.e. net flux is positive
    assert hf.boundary_flux_W > 0.0


def test_2d_energy_balance_current_density_field():
    """Non-uniform |j| field: source = |j|^2/kappa, balance still closes."""
    Lx, Ly, nx, ny = 4.0, 2.0, 32, 24
    X, _ = np.meshgrid(np.linspace(0.0, Lx, nx), np.linspace(0.0, Ly, ny))
    j = 0.012 * (1.0 + 0.5 * np.sin(np.pi * X / Lx))          # A/cm2, edge peaked
    kappa = 0.065                                              # S/cm
    q = th.volumetric_source(j, kappa)
    assert np.asarray(q).shape == (ny, nx)
    hf = th.solve_heat_2d(q, Lx, Ly, nx=nx, ny=ny, T_wall_C=22.0)
    assert hf.energy_balance_rel < 1e-6
    assert hf.discrete_residual_rel < 1e-6
    assert hf.source_W == pytest.approx(float(np.sum(j * j / kappa)) * (Lx / nx) * (Ly / ny),
                                        rel=1e-9)
    assert 0.0 < hf.dT_max_K < 20.0
    assert hf.dT_max_K > hf.dT_mean_K > 0.0


def test_2d_robin_wall_matches_film_analytic():
    """Both y-walls Robin at h: peak rise = q b^2/(2k) + q b/h (b = Ly/2)."""
    q, k, Ly, h = 2.0e-3, th.WATER_K_W_CM_K, 2.0, 10.0
    hf = th.solve_heat_2d(q, 4.0, Ly, nx=9, ny=41, k_W_cmK=k, T_wall_C=22.0,
                          walls={"left": "insulated", "right": "insulated",
                                 "top": ("robin", h), "bottom": ("robin", h)})
    b = Ly / 2.0
    analytic = q * b * b / (2.0 * k) + q * b / (h * th.W_M2K_TO_W_CM2K)
    assert hf.dT_max_K == pytest.approx(analytic, rel=1e-2)
    assert hf.energy_balance_rel < 1e-6
    assert hf.dT_K.mean() > 0.0


def test_2d_robin_limits_and_validation():
    q, Lx, Ly = 2.0e-3, 4.0, 2.0
    # h -> infinity recovers the Dirichlet solution; h -> 0 recovers insulation
    h_big = th.solve_heat_2d(q, Lx, Ly, nx=15, ny=15,
                             walls={"top": 1e12, "bottom": 1e12})
    d_dir = th.solve_heat_2d(q, Lx, Ly, nx=15, ny=15, walls={"top": "dirichlet",
                                                            "bottom": "dirichlet"})
    assert h_big.dT_max_K == pytest.approx(d_dir.dT_max_K, rel=1e-6)
    h_zero = th.solve_heat_2d(q, Lx, Ly, nx=15, ny=15,
                              walls={"top": 0.0, "bottom": 0.0})
    d_ins = th.solve_heat_2d(q, Lx, Ly, nx=15, ny=15,
                             walls={"top": "insulated", "bottom": "insulated"})
    assert h_zero.dT_max_K == pytest.approx(d_ins.dT_max_K, rel=1e-12)
    # a fully insulated section has no steady solution
    with pytest.raises(ValueError):
        th.solve_heat_2d(q, Lx, Ly, walls={w: "insulated" for w in th._WALLS})
    with pytest.raises(ValueError):
        th.solve_heat_2d(q, Lx, Ly, nx=15, ny=15, walls={"middle": "dirichlet"})
    with pytest.raises(ValueError):
        th.solve_heat_2d(q, Lx, Ly, nx=15, ny=15, walls={"top": "reflective"})
    with pytest.raises(ValueError):        # wrong field shape
        th.solve_heat_2d(np.zeros((5, 5)), Lx, Ly, nx=15, ny=15)
    with pytest.raises(ValueError):
        th.solve_heat_2d(q, -1.0, Ly)


def test_volumetric_source_and_section_wrapper():
    assert th.volumetric_source(0.012, 0.065) == pytest.approx(0.012 ** 2 / 0.065)
    j = np.array([0.0, 0.01, 0.02])
    q = th.volumetric_source(j, 0.065)
    assert np.allclose(q, j * j / 0.065)
    with pytest.raises(ValueError):
        th.volumetric_source(0.01, 0.0)
    # 1-D j profile is tiled over y and equals the explicit 2-D field
    hf1 = th.cell_section_heat(0.012, 0.065, 4.0, 2.0, nx=12, ny=6, T_bulk_C=22.0)
    hf2 = th.solve_heat_2d(th.volumetric_source(0.012, 0.065), 4.0, 2.0,
                           nx=12, ny=6, T_wall_C=22.0)
    assert np.allclose(hf1.T_C, hf2.T_C)
    assert hf1.dT_max_K > 0.0


# ---------------------------------------------------------------- batch response

def test_bath_property_shift_reuses_bath_laws():
    sh = th.bath_property_shift(22.0, 2.0)
    b0, b1 = Bath().at(22.0), Bath().at(24.0)
    assert sh["D_cu_cm2_s"] == pytest.approx(b0.D_cu, rel=1e-15)
    assert sh["D_cu_cm2_s_new"] == pytest.approx(b1.D_cu, rel=1e-15)
    assert sh["kappa_S_cm_new"] == pytest.approx(b1.kappa, rel=1e-15)
    assert sh["i0_cu_A_cm2_new"] == pytest.approx(b1.i0_cu, rel=1e-15)
    # all three rise with T, and Arrhenius i0 rises fastest
    assert sh["kappa_rel_pct"] > 0.0
    assert sh["D_cu_rel_pct"] > 0.0
    assert sh["i0_cu_rel_pct"] > sh["kappa_rel_pct"]
    # zero rise -> zero change
    sh0 = th.bath_property_shift(22.0, 0.0)
    assert sh0["kappa_rel_pct"] == pytest.approx(0.0, abs=1e-12)
    assert sh0["i0_cu_rel_pct"] == pytest.approx(0.0, abs=1e-12)


def test_beaker_example_05A_2V_1L_45h():
    """The plan's beaker example: a few K at most, finite and positive."""
    rep = th.temperature_rise_report(0.5, 2.0, 1.0, 4.5)
    print("\nbeaker 0.5 A, 2 V, 1 L, 4.5 h ->", rep["summary_line"])
    assert rep["P_W"] == pytest.approx(1.0)
    assert math.isfinite(rep["dT_steady_K"]) and 0.0 < rep["dT_steady_K"] < 5.0
    assert math.isfinite(rep["dT_end_K"]) and 0.0 < rep["dT_end_K"] < 5.0
    assert 0.0 < rep["dT_end_K"] < rep["dT_steady_K"]
    assert rep["T_end_C"] == pytest.approx(rep["T0_C"] + rep["dT_end_K"], rel=1e-15)
    # RK4 cross-check of the closed form reported for the run
    assert rep["dT_end_rk4_K"] == pytest.approx(rep["dT_end_K"], rel=1e-6)
    # tau is hours, not seconds
    assert 0.5 < rep["tau_h"] < 10.0
    assert 0.0 < rep["kappa_rel_pct"] < 10.0
    assert 0.0 < rep["D_cu_rel_pct"] < 10.0
    assert 0.0 < rep["i0_cu_rel_pct"] < 20.0
    assert "kappa" in rep["summary_line"]
    json.dumps(rep)                       # JSON-serialisable for the run record


def test_report_i2r_and_2d_section():
    R = th.ohmic_resistance_ohm(0.065, 4.0, 4.0)
    rep = th.temperature_rise_report(0.5, None, 1.0, 4.5, R_ohm=R)
    assert rep["power_source"] == "I^2*R_ohm"
    assert rep["P_W"] == pytest.approx(0.25 * R, rel=1e-15)
    with_section = th.temperature_rise_report(0.5, 2.0, 1.0, 4.5,
                                              j_A_cm2=0.012, kappa_S_cm=0.065)
    sec = with_section["section"]
    assert sec["energy_balance_rel"] < 1e-6
    assert 0.0 < sec["dT_max_K"] < 20.0      # the cell section, not the whole beaker
    assert sec["dT_max_K"] > sec["dT_mean_K"] > 0.0
    assert sec["source_W_per_cm"] == pytest.approx(0.012 ** 2 / 0.065 * 4.0 * 2.0,
                                                   rel=1e-6)
    # longer runs heat up more but never past the steady rise
    short = th.temperature_rise_report(0.5, 2.0, 1.0, 0.5)
    long = th.temperature_rise_report(0.5, 2.0, 1.0, 24.0)
    assert short["dT_end_K"] < long["dT_end_K"] < long["dT_steady_K"]
    with pytest.raises(ValueError):
        th.temperature_rise_report(0.5, 2.0, 1.0, -1.0)
