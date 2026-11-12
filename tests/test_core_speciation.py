"""Tests for platesim.core.speciation (plan item D1a).

Run: uv run --with pytest python -m pytest tests/test_core_speciation.py -q -s
Must finish well under 60 s (no PDE, no meshing).
"""
from __future__ import annotations

import math

import pytest

from platesim.core.speciation import (
    KAPPA_22,
    conductivity,
    conductivity_summary,
    dissociated_spec,
    fully_dissociated_enhancement,
    migration_enhancement_speciated,
    speciate,
)
from platesim.feature.engine.params import (
    Params,
    default_enhancement,
    derive,
    migration_enhancement,
)

M_CU = 63.546
M_H2SO4 = 98.079
C_CU = 65.0 / M_CU          # 1.0229 mol/L
C_ACID = 8.0 / M_H2SO4      # 0.08157 mol/L


def _balances(s):
    cu = s["c_cu"] + s["c_cuso4"] - s["c_cu_total"]
    so4 = s["c_so4"] + s["c_hso4"] + s["c_cuso4"] - s["c_so4_total"]
    chg = 2 * s["c_cu"] + s["c_h"] - 2 * s["c_so4"] - s["c_hso4"]
    return cu, so4, chg


@pytest.mark.parametrize("activity", ["davies", "ideal", "bdot"])
@pytest.mark.parametrize("T_C,acid", [(22.0, 8.0), (40.0, 8.0), (22.0, 0.0), (22.0, 180.0)])
def test_mass_balances(activity, T_C, acid):
    """Cu and sulfate mass balances and electroneutrality to 1e-10 relative."""
    s = speciate(C_CU, acid / M_H2SO4, T_C=T_C, activity=activity)
    assert s["converged"], s["residual"]
    cu, so4, chg = _balances(s)
    c_q = 2 * C_CU + 2 * acid / M_H2SO4 + 1e-12
    assert abs(cu) / s["c_cu_total"] < 1e-10
    assert abs(so4) / s["c_so4_total"] < 1e-10
    assert abs(chg) / c_q < 1e-10
    # every species is non-negative and I is consistent with the free ions
    for k in ("c_cu", "c_cuso4", "c_h", "c_hso4", "c_so4", "I"):
        assert s[k] >= 0.0
    i_chk = 0.5 * (4 * s["c_cu"] + s["c_h"] + s["c_hso4"] + 4 * s["c_so4"])
    assert abs(i_chk - s["I"]) / max(s["I"], 1e-12) < 1e-10


def test_full_dissociation_limit():
    """Ka2 -> inf and K_ip -> 0 reproduce full dissociation exactly."""
    s = speciate(C_CU, C_ACID, T_C=22.0, Ka2=1e30, K_ip=0.0, activity="ideal")
    assert s["converged"]
    assert s["c_cuso4"] / C_CU < 1e-20          # no ion pair
    assert s["c_hso4"] / C_ACID < 1e-20        # no bisulfate
    assert s["c_cu"] == pytest.approx(C_CU, rel=1e-12)
    assert s["c_so4"] == pytest.approx(C_CU + C_ACID, rel=1e-12)
    assert s["c_h"] == pytest.approx(2.0 * C_ACID, rel=1e-10)
    s0 = speciate(C_CU, 0.0, T_C=22.0, Ka2=1e30, K_ip=0.0, activity="ideal")
    assert s0["c_so4"] == pytest.approx(C_CU, rel=1e-12)
    # C_acid = 0: the proton balance is c_H + c_HS = 0, so both are zero; the log-space
    # solver leaves a ~1e-12 mol/L numerical floor.
    assert s0["c_h"] < 1e-9 and s0["c_hso4"] < 1e-9


def test_dilute_sulfuric_acid_quadratic():
    """0.01 mol/L pure H2SO4, ideal: HSO4- fraction ~ analytic quadratic."""
    c = 0.01
    s = speciate(0.0, c, T_C=22.0, activity="ideal")
    assert s["converged"]
    ka2 = s["Ka2"]
    # c_SO4 = x, c_HSO4 = c - x, c_H = c + x, Ka2 (c - x) = (c + x) x
    disc = (c + ka2) ** 2 + 4.0 * ka2 * c
    x = (-(c + ka2) + math.sqrt(disc)) / 2.0
    assert s["c_so4"] == pytest.approx(x, rel=1e-10)
    assert s["c_hso4"] == pytest.approx(c - x, rel=1e-10)
    assert s["c_h"] == pytest.approx(c + x, rel=1e-10)
    frac = s["c_hso4"] / c
    assert 0.2 < frac < 0.8                     # Ka2 ~ 1e-2 vs c = 1e-2: neither limit
    assert s["c_cu"] == 0.0 and s["c_cuso4"] == 0.0


def test_enhancement_reproduces_params():
    """No pair, full dissociation -> params.migration_enhancement for the default bath."""
    spec = dissociated_spec(C_CU, C_ACID)
    e = migration_enhancement_speciated(spec, include_pair=False)
    ref = migration_enhancement(C_CU, [(+1, 2 * C_ACID), (-2, C_CU + C_ACID)])
    assert ref == pytest.approx(1.6690357, abs=1e-6)
    assert e == pytest.approx(ref, abs=1e-4)
    assert e == pytest.approx(default_enhancement(), abs=1e-4)
    assert fully_dissociated_enhancement(C_CU, C_ACID) == pytest.approx(ref, abs=1e-4)


def test_conductivity_matches_phys_kappa_ne():
    """Fully dissociated NE with the engine's own diffusivities == Phys.kappa_ne (25 C)."""
    spec = dissociated_spec(C_CU, C_ACID)
    D = {"cu": 2.65e-6, "h": 9.31e-5, "so4": 1.07e-5, "hso4": 1.33e-5}
    k = conductivity(spec, D=D, T_C=25.0, model="nernst_einstein")
    ref = derive(Params(), 25.0).kappa_ne
    assert k == pytest.approx(ref, rel=1e-6)
    # the pair-restored model must equal the base model when there is no pair
    assert conductivity(spec, D=D, T_C=25.0, model="ne_with_pairs") == pytest.approx(k, rel=1e-12)


def test_speciation_lowers_conductivity():
    """Pairing / HSO4- remove carriers: speciated kappa << dissociated kappa."""
    diss = dissociated_spec(C_CU, C_ACID)
    spec = speciate(C_CU, C_ACID, T_C=22.0, activity="davies")
    k_diss = conductivity(diss, T_C=22.0)
    k_spec = conductivity(spec, T_C=22.0)
    assert k_spec < 0.5 * k_diss
    assert k_diss > 3.0 * KAPPA_22 > 0.3 * k_diss   # placeholder is far below free-ion NE
    summary = conductivity_summary(spec)
    assert summary["kappa_ne"] == pytest.approx(k_spec)
    assert summary["kappa_ne_with_pairs"] > k_spec
    assert summary["ratio_ne"] == pytest.approx(k_spec / KAPPA_22)


def test_enhancement_bounds():
    """Fully dissociated 1.669 <= speciated E, and E stays below the pure-salt limit 2."""
    spec = speciate(C_CU, C_ACID, T_C=22.0, activity="davies")
    e = migration_enhancement_speciated(spec)
    e_pair = migration_enhancement_speciated(spec, include_pair=True)
    assert 1.669 < e < 2.05
    assert 1.0 < e_pair < e


def test_default_bath_table(capsys):
    """Print the default-bath numbers (species, pH, kappa, E)."""
    spec = speciate(C_CU, C_ACID, T_C=22.0, activity="davies")
    diss = dissociated_spec(C_CU, C_ACID)
    ks = conductivity_summary(spec)
    print()
    print("default bath 65 g/L Cu + 8 g/L H2SO4 at 22 C (Davies)")
    print(f"  species [mol/L]: Cu2+ {spec['c_cu']:.4f}  CuSO4 {spec['c_cuso4']:.4f}  "
          f"H+ {spec['c_h']:.4f}  HSO4- {spec['c_hso4']:.4f}  SO4^2- {spec['c_so4']:.4f}")
    print(f"  I {spec['I']:.3f} mol/L   pH {spec['pH']:.2f}   "
          f"gamma_Cu {spec['gamma']['cu']:.3f}  gamma_H {spec['gamma']['h']:.3f}")
    print(f"  kappa_NE dissociated {conductivity(diss, T_C=22.0):.4f} S/cm   "
          f"speciated {ks['kappa_ne']:.4f} S/cm   placeholder {KAPPA_22}")
    print(f"  E: fully dissociated {fully_dissociated_enhancement(C_CU, C_ACID):.4f}   "
          f"speciated-frozen {migration_enhancement_speciated(spec):.4f}   "
          f"with pair diffusion {migration_enhancement_speciated(spec, include_pair=True):.4f}")
    assert spec["converged"]
    assert 1.4 < spec["pH"] < 2.0
    assert 1.6 < migration_enhancement_speciated(spec) < 2.05
