"""Tests for platesim.trenchflow.model (coupled trench growth model, galvanostatic k solve).

Small cases only (dx = 8 um, 2 stages) so the module runs in a few seconds.
"""
import time

import numpy as np
import pytest

from platesim.feature.engine import dc, run_case
from platesim.feature.engine.params import Params, derive
from platesim.trenchflow.model import (model_constants, predict_site, shear_from_delta, solve_k, stage,
                                       stage_fields)
from platesim.trenchflow.geometry import trench_grid
from platesim.trenchflow.transport import region_means

P = Params()
PH = derive(P, 22.0)
D = float(PH.D_c)

SMALL = dict(width_um=160.0, depth_um=80.0, delta_um=80.0, shear_rate_s=50.0, j_field_mA_cm2=15.0,
             deposit_um=0.0, dx_um=8.0)

# cheap synthetic site: 60 x 30 um trench, default trenchflow pitch (60 + 4 * 30 = 180 um)
SITE = {"id": "synth60", "width_um": 60.0, "depth_um": 30.0, "T_C": 22.0, "hours": 0.5,
        "field_um": {"lo": 6.0, "hi": 8.0, "best": 7.0},
        "floor_um": {"lo": 5.5, "hi": 6.5, "best": 6.0}}


def _delta_from_gamma(G, D_, L=2.0):
    """Inverse of shear_from_delta: delta_um for a given G (same Leveque mean relation)."""
    k_mean = 1.5 * 0.538 * D_ ** (2.0 / 3.0) * G ** (1.0 / 3.0) * L ** (-1.0 / 3.0)
    return D_ / k_mean * 1e4


def test_shear_from_delta_round_trip():
    """delta -> G -> delta is exact (same formula), and G decreases with delta."""
    for delta in (30.0, 45.0, 64.0, 90.0, 140.0):
        G = shear_from_delta(delta, D)
        back = _delta_from_gamma(G, D)
        assert abs(back - delta) / delta < 1e-9, (delta, G, back)
    g = [shear_from_delta(d, D) for d in (30.0, 64.0, 120.0)]
    assert g[0] > g[1] > g[2]


def test_solve_k_hits_target_field_flux():
    """solve_k must reproduce the requested field flux density within 0.5 %."""
    grid = trench_grid(160.0, 80.0, height_um=80.0, dx_um=8.0)
    target = 7.6e-5                                   # cm/s, ~15 mA/cm2 at c_bulk
    k, res = solve_k(grid, None, D, target)
    got = region_means(res, grid)["field"]
    assert abs(got - target) / target < 5e-3, (got, target)
    assert 1e-7 <= k <= 1e2


def test_stage_ratios_small_trench():
    """floor_over_field <= 1.02 in a small trench; flow must not lower it below the no-flow value."""
    flow = stage(**SMALL, use_flow=True)
    noflow = stage(**SMALL, use_flow=False)
    for out in (flow, noflow):
        assert 0.0 < out["floor_over_field"] <= 1.02, out["floor_over_field"]
        assert out["field_flux_cm_s"] > 0.0
        assert out["i_lim_field_mA_cm2"] > SMALL["j_field_mA_cm2"]
        assert len(out["grid_shape"]) == 2
    assert flow["floor_over_field"] >= noflow["floor_over_field"] - 0.02, (flow, noflow)
    assert flow["eddy"] is not None and "penetration_um" in flow["eddy"]
    assert noflow["eddy"] is None
    assert np.isfinite(flow["wall_up_over_field"]) and np.isfinite(flow["wall_down_over_field"])


def test_above_limit_raises():
    """j_field above 2 F c_bulk D / delta must raise with the limit in the message."""
    with pytest.raises(ValueError) as exc:
        stage(**{**SMALL, "j_field_mA_cm2": 1e4})
    assert "i_lim" in str(exc.value) or "limit" in str(exc.value)


def test_predict_site_keys_and_bound():
    """predict_site on a synthetic site returns the documented keys and floor <= field * 1.02."""
    site = {"id": "synth", "width_um": 160.0, "depth_um": 80.0, "T_C": 22.0, "hours": 0.5,
            "field_um": {"lo": 7.0, "hi": 9.0, "best": 8.0}, "floor_um": {"lo": 6.0, "hi": 7.0, "best": 6.5}}
    out = predict_site(site, 64.0, n_stages=2, dx_um=8.0)
    keys = {"id", "field_um", "j_field_mA_cm2", "delta_um", "shear_rate_s", "floor_um", "wall_up_um",
            "wall_down_um", "floor_over_field", "wall_asymmetry", "stages", "use_flow"}
    assert keys <= set(out)
    assert out["field_um"] == 8.0
    assert out["floor_um"] <= out["field_um"] * 1.02, (out["floor_um"], out["field_um"])
    assert len(out["stages"]) == 2
    for st in out["stages"]:
        assert "t_field_um" in st and "deposit_um" in st
    assert abs(out["floor_over_field"] - out["floor_um"] / out["field_um"]) < 1e-12
    # j_field from the measured field thickness: field_um*1e-4 / (Mfac * hours * 3600)
    _, _, mfac = model_constants(22.0)
    assert abs(out["j_field_mA_cm2"] - 8.0e-4 / (mfac * 0.5 * 3600.0) * 1e3) < 1e-9


def test_default_constants_equal_explicit_ph_exactly():
    """No-ph defaults are bit-identical to passing ph = derive(Params(), 22.0)."""
    ph = derive(Params(), 22.0)
    assert model_constants(22.0) == model_constants(22.0, p=Params(), ph=ph)
    a = stage(**SMALL, use_flow=True)
    b = stage(**SMALL, use_flow=True, p=Params(), ph=ph)
    assert a == b

    oa = predict_site(SITE, 20.0, n_stages=2, dx_um=8.0)
    ob = predict_site(SITE, 20.0, n_stages=2, dx_um=8.0, p=Params(), ph=ph)
    assert oa == ob


def test_efficiency_changes_mfac_and_j_field_by_inverse_factor():
    """Params(efficiency=0.8) -> Mfac * 0.8, so the field current from field_um scales by 1/0.8."""
    p08 = Params(efficiency=0.8)
    ph08 = derive(p08, 22.0)
    D0, cb0, m0 = model_constants(22.0)
    D1, cb1, m1 = model_constants(22.0, p=p08, ph=ph08)
    assert D1 == D0 and cb1 == cb0                     # efficiency only enters Mfac
    assert m1 == pytest.approx(0.8 * m0, rel=1e-15)

    base = predict_site(SITE, 20.0, n_stages=2, dx_um=8.0)
    eff8 = predict_site(SITE, 20.0, n_stages=2, dx_um=8.0, p=p08, ph=ph08)
    assert eff8["j_field_mA_cm2"] / base["j_field_mA_cm2"] == pytest.approx(1.0 / 0.8, rel=1e-12)


def test_d_source_reports_which_diffusivity_was_used():
    """D_source names the transport path; tertiary keeps the lumped migration on the 1-D model."""
    default = stage(**SMALL, use_flow=False)
    assert default["D_source"] == "enhanced:ph.D_c"
    assert "D_source" in stage_fields(**SMALL, use_flow=False)
    assert "D_source" in predict_site(SITE, 20.0, n_stages=2, dx_um=8.0)

    tp = Params(transport="tertiary")
    ph_t = derive(tp, 22.0)
    assert ph_t.D_c == ph_t.D_cu                      # the engine drops migration on this path
    tert = stage(**SMALL, use_flow=False, p=tp, ph=ph_t)
    assert "tertiary" in tert["D_source"]
    # no potential field in the 1-D quick model -> E_mig*D_cu, i.e. the same D as the default path
    assert model_constants(22.0, p=tp, ph=ph_t)[0] == model_constants(22.0)[0]
    assert {k: v for k, v in tert.items() if k != "D_source"} == \
        {k: v for k, v in default.items() if k != "D_source"}


def test_start_floor_over_field_matches_feature_engine():
    """At the start of plating the no-flow quick model agrees with trench_full within 0.05.

    Measured on this host: feature engine 0.96336, trenchflow 0.92950, difference 0.0339 (0.05 h of DC
    at 10 mA/cm2, dx 8, delta 20, 60 x 30 um trench, f_open 0.2).
    """
    delta, j = 20.0, 10.0
    p = Params(geometry="trench_full", H_um=30.0, dx_um=8.0, delta_um=delta, k_mix=0.0,
               suppressor=False, ohmic=False, flow_shear_s=0.0, max_overburden_um=0.0)
    res = run_case(60.0, 22.0, dc(j, 0.05), p, f_open=0.2, label="a7_xcheck")
    s = res["series"]
    fe = s["floor_height_um"][-1] / s["face_thickness_um"][-1]
    tf = stage_fields(60.0, 30.0, delta, 0.0, j, 0.0, T_C=22.0, dx_um=8.0, use_flow=False)
    assert abs(fe - tf["floor_over_field"]) <= 0.05, (fe, tf["floor_over_field"])


def test_flow_solver_default_lbm_is_unchanged():
    """The default is 'lbm' and passing flow_solver='lbm' reproduces it exactly."""
    a = stage_fields(**SMALL)
    b = stage_fields(**SMALL, flow_solver="lbm")
    assert a["flow_solver"] == b["flow_solver"] == "lbm"
    for key in ("floor_over_field", "wall_up_over_field", "wall_down_over_field", "field_flux_cm_s",
                "k_cm_s"):
        assert a[key] == b[key], key
    np.testing.assert_array_equal(a["flow"].ux, b["flow"].ux)
    np.testing.assert_array_equal(a["flow"].uz, b["flow"].uz)
    assert a["eddy"] == b["eddy"]
    assert stage(**SMALL)["flow_solver"] == "lbm"
    assert predict_site(SITE, 20.0, n_stages=1, dx_um=8.0)["flow_solver"] == "lbm"


def test_flow_solver_stokes_small_trench():
    """flow_solver='stokes' solves the same small trench directly and lands within 0.05 on the ratio.

    60 x 30 um trench, dx 6, delta 64 um, G from shear_from_delta. The two solvers disagree by a few
    percent on the floor flux (the lattice is the coarser approximation); both stay in [0, 1.02].
    """
    delta = 64.0
    G = shear_from_delta(delta, D)
    kw = {"width_um": 60.0, "depth_um": 30.0, "delta_um": delta, "shear_rate_s": G,
          "j_field_mA_cm2": 15.0, "deposit_um": 0.0, "dx_um": 6.0}
    t0 = time.perf_counter()
    lbm = stage_fields(**kw)
    t_lbm = time.perf_counter() - t0
    t0 = time.perf_counter()
    st = stage_fields(**kw, flow_solver="stokes")
    t_st = time.perf_counter() - t0
    print(f"\nstage_fields timing: lbm {t_lbm * 1e3:.0f} ms, stokes {t_st * 1e3:.0f} ms "
          f"(grid {st['grid_shape']})")
    print(f"floor_over_field: lbm {lbm['floor_over_field']:.6f}, stokes {st['floor_over_field']:.6f}")

    assert lbm["flow_solver"] == "lbm"
    assert st["flow_solver"] == "stokes"
    assert st["flow"].info["method"] == "stokes_mac"
    assert st["flow"].info["div_max_rel"] < 1e-10
    assert 0.0 < st["floor_over_field"] <= 1.02, st["floor_over_field"]
    assert abs(st["floor_over_field"] - lbm["floor_over_field"]) < 0.05
    assert st["eddy"] is not None
    assert set(st["eddy"]) == {"u_max_in_trench_cm_s", "has_recirculation", "penetration_um"}
    # the switch survives the wrappers: stage() forwards it and reports it
    s = stage(**kw, flow_solver="stokes")
    assert s["flow_solver"] == "stokes"
    assert s["floor_over_field"] == st["floor_over_field"]


def test_flow_solver_invalid_name_raises():
    """Any name other than 'lbm' / 'stokes' raises ValueError in every entry point."""
    with pytest.raises(ValueError, match="flow_solver"):
        stage_fields(**SMALL, flow_solver="fd")
    with pytest.raises(ValueError, match="flow_solver"):
        stage(**SMALL, flow_solver="navier")
    with pytest.raises(ValueError, match="flow_solver"):
        predict_site(SITE, 20.0, n_stages=1, dx_um=8.0, flow_solver="lattice")

