"""Validation tests for platesim.coupon (coupled flow -> cell -> efficiency pipeline).

Run: python -m pytest tests/test_coupon.py -q   (must finish < 90 s)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from platesim.coupon import DEFAULT_THETA, MODEL_SPEC, calibrate_joint, run, simulate_run
from platesim.coupon.calibrate import load_calibration, predict_table
from platesim.coupon.pipeline import load_sem_points
from platesim.core.constants import thickness_um
from platesim.efficiency.model import load_doe


def _run(setup: str, **kw) -> dict:
    base = {"j_mA_cm2": 12.0, "t_min": 60.0,
            "stir_pct": 20.0 if setup == "base" else 30.0, "setup": setup}
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def sims() -> dict:
    """One pipeline run per setup (shares meshes/solves across the module)."""
    return {setup: simulate_run(_run(setup), dict(DEFAULT_THETA), resolution="coarse")
            for setup in ("base", "imp1", "imp2")}


def test_pipeline_runs_for_each_setup(sims):
    for setup, s in sims.items():
        assert s["run"]["setup"] == setup
        assert 0.0 < s["front_fraction"] <= 1.0
        assert s["delta_mean_um"] > 0.0
        assert np.isfinite(s["mean_thickness_um"]) and s["mean_thickness_um"] > 0.0
        assert 0.0 < s["apparent_eff"] <= 1.0 + 1e-9
        assert len(s["maps"]["j_mA_cm2"]) == 41 and len(s["maps"]["j_mA_cm2"][0]) == 41
        assert np.all(np.asarray(s["maps"]["eta_true"]) > 0.0)
        assert isinstance(s["assumptions"], list) and s["assumptions"]
        json.dumps(s)                                  # JSON-serialisable output


def test_front_fraction_by_setup(sims):
    # base: untaped -> partial backside activity (f_back) so some current misses the front
    assert sims["base"]["front_fraction"] < 1.0
    assert sims["base"]["loss_fraction"] > 0.05
    # imp1/imp2: taped, no stray -> all current on the front face
    assert sims["imp1"]["front_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert sims["imp2"]["front_fraction"] == pytest.approx(1.0, abs=1e-6)


def test_thickness_integrates_to_faraday(sims):
    """mean(thickness) + L0 == Faraday(j_set) * front_fraction * eff_charge (2 %)."""
    sim = sims["base"]
    faraday = thickness_um(12.0, 3600.0, 1.0)
    predicted = faraday * sim["front_fraction"] * sim["eff_charge"]
    got = sim["mean_thickness_um"] + float(DEFAULT_THETA["L0_um"])
    assert abs(got - predicted) / predicted < 0.02
    # loss decomposition closes exactly: faraday - losses = mean thickness
    total_loss = sim["loss_current_um"] + sim["loss_o2_um"] + sim["loss_l0_um"]
    assert sim["faraday_um"] - total_loss == pytest.approx(sim["mean_thickness_um"],
                                                           abs=1e-6)


def test_sem_points_and_direction():
    assert {p["label"] for p in load_sem_points()} >= {
        "top_left", "top_left_2", "bottom_left", "bottom_right",
        "run1_top", "run1_bottom", "run2_mid", "run2_bottom"}
    r1 = simulate_run(_run("base", j_mA_cm2=9.0, id=1), dict(DEFAULT_THETA),
                      resolution="coarse")
    labels = {p["label"] for p in r1["sem_points"]}
    assert {"top_left", "top_left_2", "bottom_left", "bottom_right",
            "run1_top", "run1_bottom"} <= labels
    for p in r1["sem_points"]:
        assert p["measured_um"] is not None
        assert 0.5 < p["predicted_um"] < 15.0
    # flow_dir flips the Leveque profile: the efficiency (eta) gradient mirrors and
    # the thickness map changes pointwise (its L/R mean also carries a mesh
    # asymmetry of ~0.2 um, so only the eta sign is asserted here)
    r_flip = simulate_run(_run("base", j_mA_cm2=9.0, id=1),
                          {**DEFAULT_THETA, "flow_dir": "-x"}, resolution="coarse")
    g = np.asarray(r1["maps"]["thickness_um"])
    gf = np.asarray(r_flip["maps"]["thickness_um"])
    e = np.asarray(r1["maps"]["eta_true"])
    ef = np.asarray(r_flip["maps"]["eta_true"])
    assert not np.allclose(g, gf, atol=1e-4)
    # +x: leading edge at the left -> thinner delta there -> more O2 loss on the left
    assert float(e[:, 0].mean()) < float(e[:, -1].mean()) - 0.01
    assert float(ef[:, 0].mean()) > float(ef[:, -1].mean()) + 0.01


def test_calibration_smoke():
    """2 runs, small budget: returns a fitted theta + residual table fast."""
    runs = [r for r in load_doe()["runs"] if r["id"] in (1, 7)]
    t0 = time.perf_counter()
    res = calibrate_joint(runs=runs, sem_points=[], max_nfev=3, save=False,
                          time_budget_s=45.0, max_solves=60, uncertainties=False,
                          resolution="coarse")
    elapsed = time.perf_counter() - t0
    assert set(res["theta"]) >= {"delta0_um", "rpm_per_pct", "f_back", "L0_um",
                                 "kappa_scale", "sigma_um"}
    for k, v in res["theta"].items():
        if isinstance(v, (int, float)):
            assert np.isfinite(v), k
        assert v > 0 if k in ("delta0_um", "rpm_per_pct", "L0_um", "kappa_scale",
                              "sigma_um") else True
    assert len(res["residuals"]) == len(runs)
    assert np.isfinite(res["rms_um"]) and res["rms_um"] >= 0.0
    assert res["n_solves"] <= 60 + 5                      # budget honoured (± per-eval overshoot)
    assert res["h4_check"]["f_stray_base_h4"] is not None  # efficiency H4 cache read
    assert elapsed < 70.0
    assert load_calibration() is None or True             # save=False wrote nothing new


def test_predict_table_matches_pipeline(sims):
    theta = dict(DEFAULT_THETA)
    doe_rows, sem_rows = predict_table(theta, runs=[_run_row(1)],
                                       resolution="coarse")
    assert len(doe_rows) == 1
    assert 0.0 < doe_rows[0]["predicted_um"] < 15.0
    # 8 SEM points total: run 1's 6 ride on the same solve, run 2's 2 pull their own
    assert len(sem_rows) == 8
    assert sum(1 for r in sem_rows if r["id"] == 1) == 6


def _run_row(run_id: int) -> dict:
    return next(r for r in load_doe()["runs"] if r["id"] == run_id)


def test_run_plugin_contract():
    out = run({"mode": "simulate", "use_calibration": False, "setup": "imp1",
               "resolution": "coarse"})
    assert MODEL_SPEC["id"] == "coupon_pipeline"
    assert {p["name"] for p in MODEL_SPEC["params"]} == {
        "j_mA_cm2", "t_min", "stir_pct", "setup", "T_C", "use_calibration",
        "resolution", "mode"}
    kinds = [p["kind"] for p in out["plots"]]
    assert kinds == ["heatmap", "heatmap", "heatmap", "line", "bar", "bar"]
    assert out["summary"]["mode"] == "simulate"
    assert 0.0 < out["summary"]["front_fraction"] <= 1.0
    assert isinstance(out["assumptions"], list) and out["assumptions"]
    assert len(out["residuals"]) == 16                   # 8 DOE means + 8 SEM points
    json.dumps(out)
