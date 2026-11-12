"""Tests for platesim.efficiency.side_reactions (plan item D5b). No data files needed.

Run: python -m pytest tests/test_side_reactions.py -q -s   (< 60 s)
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from platesim.core.bath import Bath
from platesim.core.constants import F, R_GAS
from platesim.efficiency import side_reactions as sr

T_C = 22.0
DELTA_UM = 65.0
PH = 1.7


def _b_tafel(alpha=0.5, T_C=T_C):
    return 2.303 * R_GAS * (T_C + 273.15) / (alpha * F)


# --------------------------------------------------------------------------- 1. HER
def test_her_tafel_slope():
    """2.303 RT/(alpha F) recovered from two kinetic-regime points."""
    i1 = sr.her_current(-0.20, PH, T_C, use_transport=False)
    i2 = sr.her_current(-0.30, PH, T_C, use_transport=False)
    b_meas = 0.10 / math.log10(i2 / i1)
    assert i2 > i1
    assert b_meas == pytest.approx(_b_tafel(), rel=1e-9)
    assert sr.tafel_slope_V(0.5, T_C) == pytest.approx(_b_tafel(), rel=1e-12)
    print(f"HER Tafel slope: {b_meas:.6f} V/dec (expected {_b_tafel():.6f})")


def test_her_activity_and_temperature_scaling():
    """i0 scales with a_H+ = 10^-pH; the slope grows with T."""
    i_ph0 = sr.her_current(-0.25, 0.0, T_C, use_transport=False)
    i_ph2 = sr.her_current(-0.25, 2.0, T_C, use_transport=False)
    assert i_ph0 / i_ph2 == pytest.approx(100.0, rel=1e-9)
    assert sr.tafel_slope_V(0.5, 40.0) > sr.tafel_slope_V(0.5, 20.0)


def test_her_series_limits():
    """i -> i_lim at large eta, i -> i_k at small eta, exact series form in between."""
    i_lim = 1e-3
    assert sr.her_current(-2.0, PH, T_C, i_lim_A_cm2=i_lim) == pytest.approx(i_lim, rel=1e-9)
    i_k = sr.her_current(-0.05, PH, T_C, use_transport=False)
    i_series = sr.her_current(-0.05, PH, T_C, i_lim_A_cm2=i_lim)
    assert i_series == pytest.approx(i_k, rel=1e-4)
    assert i_series == pytest.approx(i_k * i_lim / (i_k + i_lim), rel=1e-12)


def test_her_transport_limit_formula():
    """i_lim,H = F D_H c_H / delta with D_H from feature.engine.params and c_H from pH."""
    from platesim.feature.engine.params import Params, stokes_einstein_factor
    delta_cm = DELTA_UM * 1e-4
    c_h = 10.0 ** (-PH) * 1e-3                                  # mol/cm3
    d_h = Params.D_H25 * stokes_einstein_factor(T_C)
    i_lim = F * d_h * c_h / delta_cm
    i = sr.her_current(-2.0, PH, T_C)                           # deep into the plateau
    assert i == pytest.approx(i_lim, rel=1e-2)
    print(f"HER i_lim (pH {PH}, delta {DELTA_UM:g} um) = {i_lim * 1e3:.2f} mA/cm2")


def test_her_overpotential_from_cu_potential():
    """eta_H = eta_Cu - (E_H2 - E_Cu) = eta_Cu + 0.0592 pH + 0.337."""
    eta_h = sr.her_overpotential(-0.36, PH, T_C=T_C)
    d_e = sr.her_equilibrium_V(PH, T_C) - sr.cu_equilibrium_V(sr.A_CU_DEFAULT, T_C)
    assert d_e == pytest.approx(-0.0592 * 1.7 * (295.15 / 298.15) - 0.3373, abs=2e-3)
    assert eta_h == pytest.approx(-0.36 - d_e, rel=1e-12)
    assert 0.05 < eta_h < 0.10          # anode-side of the H2 equilibrium -> HER suppressed
    # higher pH suppresses HER at fixed E (E_H2 moves negative)
    assert sr.her_overpotential(-0.36, 3.0, T_C=T_C) > eta_h
    print(f"eta_H at eta_Cu=-0.36 V, pH {PH}: {eta_h * 1e3:+.1f} mV")


def test_surface_ph_one_shot_mass_balance():
    assert sr.surface_ph(PH, 0.0, DELTA_UM, T_C=T_C) == pytest.approx(PH, rel=1e-12)
    assert sr.surface_ph(PH, 1e-3, DELTA_UM, T_C=T_C) > PH
    assert sr.surface_ph(PH, 0.05, DELTA_UM, T_C=T_C) == pytest.approx(14.0)   # H+ depleted


# --------------------------------------------------------------------------- 2. ORR
def test_orr_limiting_current_formula_and_share():
    """i_lim,O2 = 4 F D c / delta; share at 15 mA/cm2 and 65 um is O(1 %)."""
    bath = Bath().at(T_C)
    i_lim = sr.orr_current(DELTA_UM, T_C)
    assert i_lim == pytest.approx(4 * F * bath.D_o2 * bath.c_o2 / (DELTA_UM * 1e-4), rel=1e-12)
    # the Sc-scaled variant is exactly Bath.j_lim_o2 (plain delta overestimates by ~2x)
    i_sc = sr.orr_current(DELTA_UM, T_C, sc_delta_scaling=True)
    assert i_sc == pytest.approx(bath.j_lim_o2(DELTA_UM * 1e-4), rel=1e-12)
    assert i_lim / i_sc == pytest.approx((bath.D_o2 / bath.D_cu) ** (1 / 3), rel=1e-12)
    share = i_lim / 0.015
    assert 0.005 < share < 0.05
    # thinner layer -> proportionally larger flux
    assert sr.orr_current(20.0, T_C) / i_lim == pytest.approx(DELTA_UM / 20.0, rel=1e-12)
    print(f"ORR i_lim (delta {DELTA_UM:g} um) = {i_lim * 1e3:.3f} mA/cm2 "
          f"= {share * 100:.2f} % of 15 mA/cm2 "
          f"(Sc-scaled/Bath.j_lim_o2: {i_sc * 1e3:.3f} mA/cm2 = {i_sc / 0.015 * 100:.2f} %)")


def test_orr_sechenov_option():
    base = sr.orr_current(DELTA_UM, T_C)
    salted = sr.orr_current(DELTA_UM, T_C, sechenov_k_L_mol=0.05)
    assert salted < base
    assert salted / base == pytest.approx(10.0 ** (-0.05 * Bath().c_cu * 1e3), rel=1e-12)


# --------------------------------------------------------------------------- 3. split
def test_efficiency_is_one_without_side_reactions():
    r = sr.faradaic_efficiency(0.015, -0.36, DELTA_UM, PH, T_C=T_C,
                               use_her=False, use_orr=False)
    assert r["converged"]
    assert r["efficiency"] == pytest.approx(1.0, rel=1e-9)
    assert r["j_Cu_A_cm2"] == pytest.approx(0.015, rel=1e-9)
    assert r["eta_Cu_V"] == pytest.approx(-0.36, rel=1e-7)
    assert r["j_H2_A_cm2"] == 0.0 and r["j_ORR_A_cm2"] == 0.0
    assert r["eta_H_V"] == pytest.approx(sr.her_overpotential(-0.36, PH, T_C=T_C), rel=1e-6)


def test_orr_share_at_15mA_is_O1pct():
    r = sr.faradaic_efficiency(0.015, -0.36, DELTA_UM, PH, T_C=T_C, use_her=False)
    share = r["j_ORR_A_cm2"] / r["j_total_A_cm2"]
    assert 0.005 < share < 0.05
    assert r["efficiency"] == pytest.approx(1.0 - share, rel=1e-9)
    # side reactions shift the Cu potential towards E_Cu by a few mV
    assert -0.36 < r["eta_Cu_V"] < -0.35
    print(f"ORR share at 15 mA/cm2, delta {DELTA_UM:g} um: {share * 100:.2f} % "
          f"-> eff {r['efficiency']:.4f}, eta_Cu {r['eta_Cu_V'] * 1e3:.1f} mV")


def test_her_share_at_15mA_is_negligible():
    r = sr.faradaic_efficiency(0.015, -0.36, DELTA_UM, PH, T_C=T_C, use_orr=False)
    share = r["j_H2_A_cm2"] / r["j_total_A_cm2"]
    assert share < 1e-4
    assert r["eta_H_V"] > 0.0                     # cathode above the H2 equilibrium
    assert r["supersat_H2"] < 0.0                 # no H2 bubble growth at this flux
    print(f"HER share at 15 mA/cm2, pH {PH}: {share:.2e} "
          f"(i_H2 = {r['j_H2_A_cm2'] * 1e3:.2e} mA/cm2, eta_H {r['eta_H_V'] * 1e3:+.1f} mV)")


def test_h2_wall_concentration_feeds_bubbles():
    """Strong HER -> wall concentration above Henry saturation, positive EP growth rate."""
    r = sr.faradaic_efficiency(0.02, -0.60, DELTA_UM, 0.0, T_C=T_C, i0_H=1e-5, use_orr=False)
    assert r["converged"]
    assert r["j_H2_A_cm2"] > 1e-3
    assert r["c_H2_wall_mol_cm3"] > r["c_H2_sat_mol_cm3"]
    assert r["supersat_H2"] > 0.0
    assert r["bubble_growth_um_s"] > 0.0
    i_lim_h2 = 2 * F * (5e-9 * 1e4) * r["c_H2_sat_mol_cm3"] / (DELTA_UM * 1e-4)
    assert r["j_H2_A_cm2"] > i_lim_h2
    print(f"H2 wall {r['c_H2_wall_mol_cm3']:.3e} mol/cm3, S = {r['S_H2']:.3f}, "
          f"EP growth at 5 um: {r['bubble_growth_um_s']:.3f} um/s")


# --------------------------------------------------------------------------- 4. anode
def test_anode_salt_film_forms_above_i_lim():
    bath = Bath().at(T_C)
    i_lim = 2 * F * bath.D_cu * (sr.C_SAT_CUSO4_MOL_L - bath.c_cu * 1e3) * 1e-3 / (DELTA_UM * 1e-4)
    lo = sr.anode_potential(0.8 * i_lim, T_C=T_C, film=True)
    hi = sr.anode_potential(1.8 * i_lim, T_C=T_C, film=True)
    assert lo["i_lim_anode_A_cm2"] == pytest.approx(i_lim, rel=1e-12)
    assert not lo["film_forms"] and lo["eta_film_V"] == 0.0
    assert hi["film_forms"] and hi["film_applied"]
    assert hi["eta_V"] > lo["eta_V"]
    assert hi["c_surf_mol_L"] > sr.C_SAT_CUSO4_MOL_L     # supersaturated surface
    # film switch off -> same criterion reported, no extra overpotential
    off = sr.anode_potential(1.8 * i_lim, T_C=T_C, film=False)
    assert off["film_forms"] and not off["film_applied"] and off["eta_film_V"] == 0.0
    # Tafel slope of the anodic branch
    j1, j2 = 0.2 * i_lim, 0.4 * i_lim
    b_meas = (sr.anode_potential(j2, T_C=T_C)["eta_kin_V"]
              - sr.anode_potential(j1, T_C=T_C)["eta_kin_V"]) / math.log10(j2 / j1)
    assert b_meas == pytest.approx(sr.tafel_slope_V(Bath().alpha_a, T_C), rel=1e-6)
    print(f"anode i_lim = {i_lim * 1e3:.2f} mA/cm2; eta at 1.8 i_lim = "
          f"{hi['eta_V'] * 1e3:.2f} mV (kin {hi['eta_kin_V'] * 1e3:.2f}, "
          f"film {hi['eta_film_V'] * 1e3:.2f})")


def test_anode_eta_increases_with_current():
    etas = [sr.anode_potential(j, T_C=T_C, film=True)["eta_V"]
            for j in (1e-3, 5e-3, 0.01, 0.03, 0.05, 0.1)]
    assert all(b > a for a, b in zip(etas, etas[1:]))


# --------------------------------------------------------------------------- 5. table
def test_efficiency_table():
    print("\n  faradaic efficiency vs j (pH 1.7, 22 C)   [* = ORR flux exceeds j_total]")
    print("  delta_um |   1 mA/cm2 |   5 mA/cm2 |  15 mA/cm2 |  50 mA/cm2")
    for delta in (20.0, 65.0, 120.0):
        row, effs = [], []
        for j_ma in (1.0, 5.0, 15.0, 50.0):
            j = j_ma * 1e-3
            r = sr.faradaic_efficiency(j, sr.cu_overpotential_V(j, T_C=T_C), delta, PH, T_C=T_C)
            effs.append(r["efficiency"])
            row.append(f"{r['efficiency']:9.4f}{'' if r['converged'] else '*'}")
        print(f"  {delta:8.0f} | " + " | ".join(row))
        assert effs[-1] > 0.95                        # 50 mA/cm2 is essentially all Cu
        assert all(b >= a - 1e-12 for a, b in zip(effs, effs[1:]))
        # thinner diffusion layer -> more O2 -> lower efficiency (where a root exists)
        other = sr.faradaic_efficiency(
            0.015, sr.cu_overpotential_V(0.015, T_C=T_C), delta, PH, T_C=T_C)["efficiency"]
        assert 0.0 < other <= 1.0
    e20 = sr.faradaic_efficiency(0.015, sr.cu_overpotential_V(0.015, T_C=T_C), 20.0,
                                 PH, T_C=T_C)["efficiency"]
    e120 = sr.faradaic_efficiency(0.015, sr.cu_overpotential_V(0.015, T_C=T_C), 120.0,
                                  PH, T_C=T_C)["efficiency"]
    assert e20 < e120

