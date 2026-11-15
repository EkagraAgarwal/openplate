"""Tests for platesim.flow (correlation mass-transfer model)."""
from __future__ import annotations

import math

from platesim.core.bath import Bath
from platesim.core.cell import CellGeometry
from platesim.flow import MODEL_SPEC, run
from platesim.flow.correlation import (
    MassTransferParams, TOUABI_RPM, TOUABI_KM, delta_mean_um, delta_map,
)


def test_delta_mean_monotonic_decreasing_in_rpm():
    rpms = [0.0, 50.0, 100.0, 300.0, 600.0, 1000.0, 1500.0]
    deltas = [delta_mean_um(r) for r in rpms]
    assert all(deltas[i] > deltas[i + 1] for i in range(len(deltas) - 1))


def test_delta_mean_rpm0_is_natural_floor():
    p = MassTransferParams()
    assert delta_mean_um(0.0, p) == p.delta_nat_um
    assert math.isfinite(delta_mean_um(0.0, p))


def test_baffles_thin_forced_layer_only():
    # baffles only scale the forced part -> smaller delta when stirring, none at rpm=0
    assert delta_mean_um(300.0, baffles=True) < delta_mean_um(300.0, baffles=False)
    assert delta_mean_um(0.0, baffles=True) == delta_mean_um(0.0, baffles=False)


def test_delta_map_shapes_and_mean():
    geom, bath = CellGeometry(), Bath()
    m = delta_map(geom, bath, 300.0)
    assert len(m["x"]) == 41 and len(m["y"]) == 41
    assert len(m["delta_um"]) == 41 and all(len(row) == 41 for row in m["delta_um"])
    grid_mean = sum(sum(row) for row in m["delta_um"]) / (41 * 41)
    assert abs(grid_mean - m["mean_um"]) / m["mean_um"] < 0.01
    assert abs(m["mean_um"] - delta_mean_um(300.0)) / m["mean_um"] < 0.01
    # k_m = D_cu_eff / delta_mean
    assert abs(m["k_m_cm_s"] - bath.D_cu_eff / (m["mean_um"] * 1e-4)) < 1e-15


def test_delta_map_leveque_profile_increases_along_flow():
    m = delta_map(CellGeometry(), Bath(), 300.0)
    row = m["delta_um"][0]
    assert row[-1] > row[0]                    # boundary layer grows along +x
    # uniform in y
    assert all(r == m["delta_um"][0] for r in m["delta_um"])


def test_run_contract_keys_and_summary():
    r = run({"rpm": 300.0})
    assert set(r) >= {"summary", "plots", "warnings", "assumptions"}
    s = r["summary"]
    for key in ("delta_mean_um", "k_m_cm_s", "j_lim_cu_mA_cm2", "j_lim_o2_mA_cm2"):
        assert key in s
    assert s["delta_mean_um"] > 0 and s["j_lim_cu_mA_cm2"] > 0
    kinds = [p["kind"] for p in r["plots"]]
    assert kinds.count("heatmap") >= 1 and kinds.count("line") >= 3


def test_run_lbm_fallback_when_missing_or_ok():
    # lbm.py is written by another worker: either importable or falls back with a warning
    r = run({"model": "lbm", "rpm": 300.0})
    assert set(r) >= {"summary", "plots", "warnings", "assumptions"}
    lbm_msgs = [w for w in r["warnings"] if "lbm" in w.lower()]
    if r["summary"].get("model") != "lbm":
        assert lbm_msgs, "fallback without an lbm warning"


def test_model_spec_shape():
    assert MODEL_SPEC["id"] == "flow_mass_transfer"
    names = [p["name"] for p in MODEL_SPEC["params"]]
    assert names == ["model", "rpm", "T_C", "baffles", "delta0_um", "n_exp",
                     "delta_nat_um", "stir_bar_len_cm", "beaker_id_cm", "resolution"]


def test_touabi_constants():
    assert len(TOUABI_RPM) == len(TOUABI_KM) == 4
    assert TOUABI_RPM[0] == 0.0 and TOUABI_RPM[-1] == 1500.0


# ---- D4a: natural convection (vertical coupon) --------------------------------
from platesim.flow.correlation import natural_convection, regime  # noqa: E402


def test_natural_convection_default_bath_band_and_laminar():
    bath = Bath()
    est = natural_convection(2.0, bath.c_cu * 1e3, bath.D_cu)
    print("natural convection 2 cm, 1 M CuSO4: Gr=%.4g Sc=%.4g Ra=%.4g "
          "delta_mean=%.2f um i_lim=%.2f mA/cm2 regime=%s"
          % (est["Gr"], est["Sc"], est["Ra"], est["delta_mean_um"],
             est["i_lim_mA_cm2"], est["regime"]))
    assert 30.0 < est["delta_mean_um"] < 300.0
    assert est["regime"] == "laminar"
    assert est["Ra"] == est["Gr"] * est["Sc"]
    assert est["delta_local_um"](1.0) > 0.0


def test_natural_convection_height_scaling():
    bath = Bath()
    a = natural_convection(2.0, bath.c_cu * 1e3, bath.D_cu)["delta_mean_um"]
    b = natural_convection(4.0, bath.c_cu * 1e3, bath.D_cu)["delta_mean_um"]
    assert abs(b / a - 2.0 ** 0.25) / 2.0 ** 0.25 < 1e-12     # delta ~ L^(1/4)


def test_natural_convection_limiting_current_scaling():
    bath = Bath()
    c = bath.c_cu * 1e3
    a = natural_convection(2.0, c, bath.D_cu)["i_lim_mA_cm2"]
    b = natural_convection(2.0, 2.0 * c, bath.D_cu)["i_lim_mA_cm2"]
    assert abs(b / a - 2.0 ** 1.25) / 2.0 ** 1.25 < 1e-12     # i_lim ~ c^(5/4)


def test_regime_thresholds_monotone():
    assert regime(1.0e6) == "laminar"
    assert regime(1.0e11) == "transitional"
    assert regime(1.0e13) == "turbulent"


def test_use_natural_false_is_bit_identical():
    p = MassTransferParams()
    # values from the constant delta_nat_um = 80 um path, before the D4 change
    expected = {0.0: 80.0, 100.0: 32.74158815808477, 300.0: 22.857142857142858}
    for rpm, want in expected.items():
        assert delta_mean_um(rpm, p) == want


def test_use_natural_true_uses_physics_floor():
    bath = Bath()
    p = MassTransferParams(use_natural=True)
    est = natural_convection(2.0, bath.c_cu * 1e3, bath.D_cu)
    got = delta_mean_um(0.0, p, bath=bath, height_cm=2.0)
    assert got == est["delta_mean_um"]
    assert got != MassTransferParams().delta_nat_um


def test_run_contract_includes_natural_convection_subdict():
    r = run({"rpm": 300.0})
    s = r["summary"]
    for key in ("delta_mean_um", "k_m_cm_s", "j_lim_cu_mA_cm2", "j_lim_o2_mA_cm2",
                "model", "rpm"):
        assert key in s
    nc = s["natural_convection"]
    for key in ("Gr", "Sc", "Ra", "regime", "delta_mean_um", "i_lim_mA_cm2"):
        assert key in nc
    assert 30.0 < nc["delta_mean_um"] < 300.0
    assert nc["Ra"] > 0.0 and nc["regime"] in ("laminar", "transitional", "turbulent")

