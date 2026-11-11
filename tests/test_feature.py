"""Tests for platesim.feature: feature-scale fill model (vias AND trenches).

Fast subset (< 60 s): spec/grid/geometry-unit tests, MODEL_SPEC run() end-to-end, trench Faraday
check, via regression against the research engine.  Slow (marked): trench seam pinch.
    python -m pytest tests/test_feature.py -q                 # all, target < 120 s
    python -m pytest tests/test_feature.py -q -m "not slow"    # fast subset
Validation numbers behind these tests: python -m platesim.feature.validate_trench
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time

import numpy as np
import pytest

from platesim.feature import MODEL_SPEC, run as model_run
from platesim.feature import validate_trench as vt
from platesim.feature.engine import Params, by_name, pr_forward, pulse_reverse, run_case
from platesim.feature.engine.levelset import Grid, curvature_div_n
from platesim.feature.engine.run_case import DEFAULT_OUT_DIR


# ------------------------------------------------------------------ MODEL_SPEC contract
def test_model_spec_contract():
    assert MODEL_SPEC["id"] == "feature_fill"
    params = {p["name"]: p for p in MODEL_SPEC["params"]}
    required = ["geometry", "d_um", "H_um", "T_C", "recipe", "j_mA_cm2", "tmax_h", "f_open",
                "k_minus", "delta_um", "k_mix", "efficiency", "D_cu25", "suppressor"]
    assert set(required) <= set(params)
    for p in MODEL_SPEC["params"]:
        assert p["type"] in ("float", "enum", "bool")
        assert "default" in p and "help" in p
    assert params["geometry"]["type"] == "enum" and params["geometry"]["choices"] == ["via", "trench", "trench_full"]
    assert params["geometry"]["default"] == "via"
    assert params["recipe"]["choices"] == ["dc", "pulse_reverse", "plan_ramp", "plan_ramp_pr", "team_staged",
                                           "fast_x1.4"]
    assert params["recipe"]["default"] == "dc"
    assert params["d_um"]["default"] == 30.0
    assert "30" in params["d_um"]["help"] and "1000" in params["d_um"]["help"]   # choices note
    assert params["H_um"]["default"] == 300.0
    assert params["T_C"]["default"] == 40.0
    assert params["j_mA_cm2"]["default"] == 2.5
    assert params["tmax_h"]["default"] == 4.0
    assert params["f_open"]["default"] == 0.05
    assert params["k_minus"]["default"] == 1.0e6
    assert params["delta_um"]["default"] == 20.0
    assert params["k_mix"]["default"] == 1.0
    assert params["efficiency"]["default"] == 1.0
    assert params["D_cu25"]["default"] == 2.65e-6
    assert params["suppressor"]["type"] == "bool" and params["suppressor"]["default"] is True


def test_run_rejects_unknown_geometry():
    with pytest.raises(ValueError):
        model_run({"geometry": "cylinder"})


# ------------------------------------------------------------------ grid: every r-weight location
def test_grid_weights_via_vs_trench():
    f_open, H, delta = 0.05, 300.0, 20.0
    gv = Grid(100.0, H, f_open, delta, geometry="via")
    gt = Grid(100.0, H, f_open, delta, geometry="trench")
    # R: via a/sqrt(f_open), trench a/f_open
    assert math.isclose(gv.R_um, 50.0 / math.sqrt(f_open))
    assert math.isclose(gt.R_um, 50.0 / f_open, rel_tol=1e-12)
    dx = gv.dx
    # via: cell volume ~ 2 pi r dx^2, east face 2 pi r_e dx, axial face 2 pi r dx
    r = gv.rc[:, 0]
    np.testing.assert_allclose(gv.vol[:, 0], 2 * np.pi * r * dx * dx)
    np.testing.assert_allclose(gv.AE[:, 0], 2 * np.pi * (np.arange(1, gv.nr + 1) * dx) * dx)
    np.testing.assert_allclose(gv.AZ[:, 0], 2 * np.pi * r * dx)
    assert math.isclose(gv.area_cell, math.pi * gv.R ** 2, rel_tol=1e-12)
    assert math.isclose(gv.via_volume, math.pi * gv.a ** 2 * gv.H, rel_tol=1e-12)
    # trench: weight 1 everywhere - vol = dx^2, AE = AZ = dx, set current = R per unit length
    dxt = gt.dx
    np.testing.assert_allclose(gt.vol[:, 0], dxt * dxt)
    np.testing.assert_allclose(gt.AE[:, 0], dxt)
    np.testing.assert_allclose(gt.AZ[:, 0], dxt)
    assert math.isclose(gt.area_cell, gt.R, rel_tol=1e-12)
    assert math.isclose(gt.via_volume, 2.0 * gt.a * gt.H, rel_tol=1e-12)
    # conductance factors used by the FV stencil: via geo_r = 2 pi r_face, trench geo_r = 1
    np.testing.assert_allclose(gv.AE[:, 0] / dx, 2 * np.pi * (np.arange(1, gv.nr + 1) * dx))
    np.testing.assert_allclose(gt.AE[:, 0] / dxt, 1.0)
    with pytest.raises(ValueError):
        Grid(30.0, H, f_open, delta, geometry="sphere")


def test_curvature_div_n_drops_1_over_r_for_trench():
    """pr/rc (axisymmetric) term present for the via, absent for the trench."""
    psi_setup = []
    for geometry in ("via", "trench"):
        g = Grid(30.0, 100.0, 0.3, 20.0, dx_um=2.0, geometry=geometry)
        psi = 0.5 * (g.rc ** 2 + g.zc ** 2)                 # prr = pzz = 1, pr = rc, gz = zc
        i, j = g.nr // 2, g.nz // 2
        psi_setup.append((g, psi, curvature_div_n(psi, g)[i, j]))
    (_, _, c_via), (_, _, c_trench) = psi_setup
    # via: prr + pzz + pr/rc = 1 + 1 + 1 = 3; trench: prr + pzz = 2
    assert math.isclose(c_trench, 2.0, abs_tol=1e-9)
    assert math.isclose(c_via, 3.0, abs_tol=1e-9)


# ------------------------------------------------------------------ (a) Faraday, trench
def test_trench_faraday_within_2pct():
    t0 = time.time()
    r = vt.v_faraday()
    assert r["pass"], r
    assert abs(r["rel_err"]) < 0.02, r["rel_err"]
    assert r["max_balance_error"] < 0.005
    assert time.time() - t0 < 30


# ------------------------------------------------------------------ (c) via regression vs research engine
@pytest.mark.skipif(not os.path.isdir(vt.RESEARCH_DIR), reason="research/via_fill_sim not present")
def test_via_regression_matches_research_engine():
    r = vt.v_via_regression()
    assert r["pass"], r
    assert abs(r["rel_err_final"]) < 0.005, r          # 0.5 % in fill fraction (bit-identical in practice)
    assert max(abs(x) for x in r["rel_err_series"]) < 0.005


# ------------------------------------------------------------------ MODEL_SPEC run() end-to-end
def test_model_run_end_to_end():
    calls = []

    def progress(frac, msg=""):
        calls.append((float(frac), msg))

    out = model_run({"geometry": "trench", "d_um": 30.0, "H_um": 300.0, "T_C": 25.0,
                     "recipe": "dc", "j_mA_cm2": 2.0, "tmax_h": 0.1, "f_open": 0.2,
                     "k_minus": 1.0e6, "delta_um": 20.0, "k_mix": 1.0, "efficiency": 1.0,
                     "D_cu25": 2.65e-6, "suppressor": True},
                    progress=progress)
    # --- contract shape
    assert set(out) >= {"summary", "plots", "warnings", "assumptions"}
    summary = out["summary"]
    assert set(summary) == {"outcome", "fill_time", "fill_fraction_final", "overburden",
                            "void_volume_frac"}
    assert summary["outcome"] in ("filled_void_free", "void", "seam", "keyhole", "incomplete")
    assert 0.0 <= summary["fill_fraction_final"] <= 1.0
    assert 0.0 <= summary["void_volume_frac"] <= 1.0
    # --- progress: starts at 0, ends at 1.0, monotonically increasing, fraction of recipe time
    fracs = [f for f, _ in calls]
    assert len(fracs) >= 3 and fracs[0] == 0.0 and fracs[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert all(b >= a - 1e-12 for a, b in zip(fracs, fracs[1:]))
    # --- plots: 4 line + contours + heatmap
    plots = out["plots"]
    assert [p["kind"] for p in plots] == ["line", "line", "line", "line", "contours", "heatmap"]
    for p in plots[:4]:
        assert p["x"] and len(p["x"]) == len(p["series"][0]["y"])
        assert p["xlabel"] and p["ylabel"]
    assert plots[2]["series"][0]["name"] == "theta_floor" and plots[2]["series"][1]["name"] == "theta_face"
    cont = plots[4]
    assert cont["frames"] and all("t" in f and "lines" in f for f in cont["frames"])
    assert any(f["lines"] for f in cont["frames"])
    line0 = next(ln for f in cont["frames"] for ln in f["lines"])
    assert len(line0) >= 2 and len(line0[0]) == 2        # polyline [[x, z], ...] in um
    assert cont["frames"][0]["lines"][0][0][0] >= 0.0    # half domain x/r >= 0, units um
    hm = plots[5]
    assert len(hm["x"]) == 48 and len(hm["y"]) == 64 and len(hm["z"]) == 64
    assert len(hm["z"][0]) == 48 and hm["zlabel"] == "c/c_bulk"
    # --- warnings: uncalibrated parameters + tuned k_minus
    assert any("Uncalibrated" in w for w in out["warnings"])
    assert any("k_minus" in w and "TUNED" in w for w in out["warnings"])
    assert out["assumptions"]
    # --- strictly JSON-serialisable (no NaN)
    json.dumps(out, allow_nan=False)


# ------------------------------------------------------------------ default output directory
def test_default_out_dir_is_temp_not_hardcoded():
    if "PLATESIM_FEATURE_OUT" not in os.environ:
        assert os.path.normpath(tempfile.gettempdir()) in os.path.normpath(DEFAULT_OUT_DIR)
    assert os.path.isabs(DEFAULT_OUT_DIR)
    assert "via_fill_sim" not in DEFAULT_OUT_DIR          # no research-era D:\ path baked in


# ------------------------------------------------------------------ (b) seam - slow
@pytest.mark.slow
def test_trench_seam_pinches_at_half_width():
    t0 = time.time()
    r = vt.v_seam()
    assert r["pass"], r
    assert r["outcome"] == "seam" and r["stop"] == "sealed"
    w2 = vt.SEAM_CASE["d_um"] / 2.0
    face = r["face_thickness_at_closure_um"]
    assert face is not None and 0.75 * w2 <= face <= 1.25 * w2      # walls grown ~ w/2 each
    void = max(r["voids"], key=lambda v: v["volume_frac"])
    assert void["cls"] == "seam" and void["rmax_um"] <= 0.4 * vt.SEAM_CASE["d_um"] / 2
    assert void["z1_um"] - void["z0_um"] >= 3.0 * max(2 * void["rmax_um"], 1.0)
    assert time.time() - t0 < 60


# ------------------------------------------------------------------ pulse reverse with OFF time
def test_pr_forward_team_waveform():
    """9 ms forward / 0.5 ms reverse at 2x / 0.5 ms off, 15 mA/cm2 net -> j_f = 18.75 mA/cm2."""
    pr = {"t_f_ms": 9.0, "t_r_ms": 0.5, "t_off_ms": 0.5, "rr": 2.0}
    j_f, duty_f, duty_r = pr_forward(15.0, pr)
    assert math.isclose(j_f, 18.75, rel_tol=1e-12)
    assert math.isclose(duty_f, 0.9, rel_tol=1e-12)
    assert math.isclose(duty_r, 0.05, rel_tol=1e-12)
    assert math.isclose(j_f * duty_f - 2.0 * j_f * duty_r, 15.0, rel_tol=1e-12)   # cycle average = net


def test_pr_off_time_absent_equals_zero():
    """No 't_off_ms' key == 't_off_ms': 0.0, and both reduce to the legacy (tf+tr) formulas."""
    pr0 = {"t_f_ms": 20.0, "t_r_ms": 1.0, "rr": 3.0}
    pr1 = {"t_f_ms": 20.0, "t_r_ms": 1.0, "t_off_ms": 0.0, "rr": 3.0}
    assert pr_forward(3.0, pr0) == pr_forward(3.0, pr1)
    j_f, duty_f, duty_r = pr_forward(3.0, pr0)
    assert math.isclose(j_f, 3.0 * 21.0 / 17.0, rel_tol=1e-12)
    assert math.isclose(duty_f, 20.0 / 21.0, rel_tol=1e-12)
    assert math.isclose(duty_r, 1.0 / 21.0, rel_tol=1e-12)


def test_pulse_reverse_rejects_non_cathodic_net():
    with pytest.raises(ValueError):
        pulse_reverse(15.0, 0.25, t_f_ms=1.0, t_r_ms=1.0, rr=2.0)


def test_pulse_reverse_recipe_shape_and_by_name():
    rec = pulse_reverse(15.0, 0.25)
    assert rec.name == "pr_15mA_0.25h" and len(rec.steps) == 1
    s = rec.steps[0]
    assert s.label == "pr" and s.j_mA_cm2 == 15.0 and math.isclose(s.duration_s, 900.0)
    assert s.pr == {"t_f_ms": 9.0, "t_r_ms": 0.5, "t_off_ms": 0.5, "rr": 2.0}
    rec2 = by_name("pulse_reverse", 22.0, j=15.0, t_max_h=0.25)          # default timings
    assert rec2.name == rec.name and rec2.steps[0].pr == s.pr


def test_pr_trench_run_finishes_and_balances_faraday():
    """End-to-end PR trench run: finishes, face grows, deposit matches the charge passed (3 %)."""
    rec = pulse_reverse(15.0, 0.25)
    eng = Params(geometry="trench", H_um=90.0, suppressor=False, dx_um=8.0)
    res = run_case(227.0, 22.0, rec, eng, f_open=0.2, label="pr_e2e")
    result = res["result"]
    assert result["t_end_s"] > 0.0
    assert result["overburden_um"] > 0.0                     # face (field) thickness
    fa = res["meta"]["faraday"]
    dep, chg = fa["deposited_volume_cm3"], fa["charge_volume_cm3"]
    assert chg > 0.0
    assert abs(dep - chg) / chg < 0.03, (dep, chg)
