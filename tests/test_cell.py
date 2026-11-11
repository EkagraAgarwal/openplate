"""Validation tests for platesim.cell (3-D beaker current distribution).

Run: python -m pytest tests/test_cell.py -q   (must finish < 60 s)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from platesim.cell import MODEL_SPEC, run, solve_cell
from platesim.cell.solver import solve_duct
from platesim.core.bath import Bath
from platesim.core.cell import CellGeometry

I_TEST = 0.048  # A -> 12 mA/cm2 on the 2x2 cm coupon


def _fast_bath(**kw) -> Bath:
    """Bath with effectively-infinite exchange current (primary / ohmic limits)."""
    return replace(Bath(), i0_cu_22=1e2, i0_cu_supp_22=1e2, **kw)


@pytest.fixture(scope="module")
def beaker_default():
    """One gmsh mesh + solve at default (suppressed) kinetics; shared by several tests."""
    return solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse")


def test_charge_conservation(beaker_default):
    res = beaker_default
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert abs(res.I_anode_A + I_TEST) / I_TEST < 0.01
    # V_cell decomposition is exact by construction
    assert abs(res.V_ohm_V + res.eta_c_V + res.eta_a_V - res.V_cell_V) < 1e-9
    assert res.V_cell_V > 0.2
    # taped: everything lands on the front face
    assert res.front_fraction == pytest.approx(1.0, abs=1e-6)
    assert res.j_mA_cm2.shape == (41, 41)
    assert np.all(res.j_mA_cm2 > 0)
    assert np.all(res.thickness_um > 0)


def test_primary_distribution_edge_heavier_than_centre():
    """Huge i0, no mass-transfer limit -> primary distribution: edges/corners carry more."""
    res = solve_cell(CellGeometry(), _fast_bath(), I_TEST, delta_um=0.0,
                     resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.edge_center_ratio > 1.3, (
        f"primary distribution should be edge-heavy, got edge/centre = "
        f"{res.edge_center_ratio:.2f}")
    assert res.Wa < 0.05  # primary regime: Wagner number << 1


def test_uniform_distribution_at_large_kappa():
    """kappa -> large: Wa >> 1, activation-controlled -> nearly uniform j."""
    bath = replace(Bath(), kappa_22=65.0)  # 1000x default conductivity
    res = solve_cell(CellGeometry(), bath, I_TEST, resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.Wa > 10.0
    assert res.std_over_mean < 0.05, (
        f"distribution should be uniform, got std/mean = {res.std_over_mean:.3f}")


def test_untaped_back_edge_fraction():
    """Untaped: a sizeable share of I_total must NOT reach the front face."""
    res = solve_cell(CellGeometry(taped=False), _fast_bath(), I_TEST, delta_um=0.0,
                     resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.loss_fraction > 0.05
    assert res.front_fraction + res.loss_fraction == pytest.approx(1.0, abs=1e-6)


def test_stray_patch_steals_current():
    """The lumped wet-clip patch (stray_area) is extra active cathode area."""
    res = solve_cell(CellGeometry(taped=True, stray_area=0.3), _fast_bath(), I_TEST,
                     delta_um=0.0, resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.cathode_area_cm2 == pytest.approx(4.0 + 0.3, abs=0.02)
    assert res.loss_fraction > 0.01          # some current lands on the clip, not the front
    assert res.front_fraction + res.loss_fraction == pytest.approx(1.0, abs=1e-6)


def test_delta_map_suppression_asymmetry():
    """A spatially varying delta_map must skew j(x, y): thick layer -> less current."""
    dm = {"x": np.array([-1.0, 0.0, 1.0]),
          "y": np.array([3.0, 4.0, 5.0]),
          "delta_um": np.array([[60.0, 180.0, 300.0]] * 3)}   # rows = y, right side thick
    res = solve_cell(CellGeometry(), Bath(), I_TEST, delta_um=60.0, delta_map=dm,
                     resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.corner_LR_grad_pct > 5.0, (
        f"right half should be mass-transfer limited, L/R grad = "
        f"{res.corner_LR_grad_pct:.1f} %")
    assert any("delta_map" in w for w in res.warnings)


def test_duct_1d_ohmic():
    """Coupon + anode spanning a rectangular duct: V_ohm = I*gap/(kappa*A) within 3 %."""
    gap, w, h = 4.0, 2.0, 2.0
    bath = Bath()
    out = solve_duct(gap, w, h, I_TEST, _fast_bath(), suppressed=True, delta_um=0.0)
    v_ohm_theory = I_TEST * gap / (bath.kappa * w * h)
    assert abs(out["V_cell_V"] - v_ohm_theory) / v_ohm_theory < 0.03
    # and the duct solution really is 1-D: uniform j on the cathode face
    assert out["j_std_over_mean"] < 1e-6
    assert abs(out["I_cathode_A"] - I_TEST) / I_TEST < 1e-5


def test_run_plugin_contract():
    out = run({"resolution": "coarse", "t_min": 30.0})
    assert out["summary"]["I_total_mA"] == pytest.approx(48.0, rel=1e-6)
    kinds = [p["kind"] for p in out["plots"]]
    assert kinds == ["heatmap", "heatmap", "line"]
    assert out["summary"]["V_cell_V"] > 0.2
    assert MODEL_SPEC["id"] == "cell_current"
    assert {p["name"] for p in MODEL_SPEC["params"]} >= {
        "j_mA_cm2", "t_min", "T_C", "efficiency", "delta_um", "suppressed", "taped",
        "stray_area_cm2", "gap_cm", "anode_w_cm", "coupon_w_cm", "coupon_h_cm",
        "kappa_S_cm", "resolution"}
    assert isinstance(out["assumptions"], list) and out["assumptions"]
    # JSON-serialisable
    import json
    json.dumps(out)


def test_gmsh_fallback_box(monkeypatch):
    """If gmsh cannot be imported, solve_cell falls back to a structured box domain."""
    from platesim.cell import mesh as meshmod
    from platesim.cell.solver import _MESH_CACHE

    _MESH_CACHE.clear()                       # do not reuse the gmsh mesh for this geometry
    monkeypatch.setattr(meshmod, "_gmsh_available", lambda: False)
    try:
        res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse")
        assert res.domain == "structured_box"
        assert any("BOX approximation" in w for w in res.warnings)
        assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
        assert np.all(res.j_mA_cm2 > 0)
    finally:
        _MESH_CACHE.clear()                   # never leave a box mesh under the gmsh key
