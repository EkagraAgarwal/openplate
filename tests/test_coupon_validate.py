"""Light tests for platesim.coupon.validate (LOO validation + trench blind prediction).

Run: python -m pytest tests/test_coupon_validate.py -q      (must finish < 60 s)
The real fits are slow; the smoke test only runs with PLATESIM_SLOW=1.
"""
from __future__ import annotations

import json
import os

import pytest

from platesim.core.constants import thickness_um
from platesim.coupon.validate import (
    TRENCH_MEASURED, TRENCH_RUN, VALIDATION_PATH, leave_one_out, leave_setup_out,
    main, make_row, max_abs, mean_bias, rms, theta_slim, trench_coupon_prediction,
)

SLOW = os.environ.get("PLATESIM_SLOW") == "1"


def _fake_run(run_id=3, setup="base", j=6.0, t=120.0, stir=20.0, meas=7.3):
    return {"id": run_id, "setup": setup, "j_mA_cm2": j, "t_min": t,
            "stir_pct": stir, "expected_um": 15.88, "actual_um": meas}


# ---------------------------------------------------------------------------
# imports / API surface
# ---------------------------------------------------------------------------
def test_module_api_and_paths():
    for fn in (leave_one_out, leave_setup_out, trench_coupon_prediction, main):
        assert callable(fn)
    assert VALIDATION_PATH.name == "validation.json"
    assert VALIDATION_PATH.parent.name == "coupon"
    # the script must never point at the calibration file
    assert VALIDATION_PATH != VALIDATION_PATH.parent / "calibration.json"


def test_trench_constants_and_faraday():
    assert TRENCH_RUN["setup"] == "imp1"
    assert TRENCH_RUN["j_mA_cm2"] == 15.0
    assert TRENCH_RUN["t_min"] == 270.0
    assert TRENCH_RUN["stir_pct"] == 30.0
    assert TRENCH_MEASURED["min_um"] == 39.5
    assert TRENCH_MEASURED["max_um"] == 91.0
    assert TRENCH_MEASURED["sites_um"] == [39.5, 69.0, 82.0]
    # [C] 100 % Faraday at 15 mA/cm2 for 270 min ~ 89.3 um
    faraday = thickness_um(TRENCH_RUN["j_mA_cm2"], TRENCH_RUN["t_min"] * 60.0, 1.0)
    assert faraday == pytest.approx(TRENCH_MEASURED["faraday_um"], abs=0.2)


# ---------------------------------------------------------------------------
# row format
# ---------------------------------------------------------------------------
def test_make_row_keys_and_residual_sign():
    run = _fake_run(meas=7.3)
    theta = {"delta0_um": 77.8, "rpm_per_pct": 15.4, "flow_dir": "+x"}
    row = make_row(run, 8.4, 8.9, theta, budget_hit=False)
    expected = {"id", "setup", "j_mA_cm2", "t_min", "stir_pct", "measured_um",
                "pred_loo_um", "pred_full_um", "resid_loo_um", "resid_full_um",
                "theta_loo", "budget_hit"}
    assert expected <= set(row)
    assert row["id"] == 3 and row["setup"] == "base"
    assert row["j_mA_cm2"] == 6.0 and row["t_min"] == 120.0
    assert row["stir_pct"] == 20.0
    assert row["measured_um"] == 7.3
    assert row["pred_loo_um"] == 8.4
    # resid = measured - predicted (calibrate.py convention)
    assert row["resid_loo_um"] == pytest.approx(7.3 - 8.4)
    assert row["resid_full_um"] == pytest.approx(7.3 - 8.9)
    assert row["theta_loo"]["flow_dir"] == "+x"
    assert row["theta_loo"]["delta0_um"] == pytest.approx(77.8)
    assert row["budget_hit"] is False
    json.dumps(row)                                   # JSON-serialisable


def test_make_row_without_full_prediction():
    row = make_row(_fake_run(), 7.0)
    assert row["pred_full_um"] is None
    assert row["resid_full_um"] is None
    assert row["theta_loo"] is None
    assert row["resid_loo_um"] == pytest.approx(0.3)


def test_theta_slim_keeps_physics_keys_and_types():
    th = theta_slim({"delta0_um": 80.0, "flow_dir": "+y", "L0_um": 0.5,
                     "f_back_imp": 0.0, "anode_base_cm": 2.0})
    assert th["flow_dir"] == "+y"
    assert isinstance(th["delta0_um"], float) and th["delta0_um"] == 80.0
    assert "f_back_imp" not in th and "anode_base_cm" not in th
    assert set(th) <= {"delta0_um", "n_exp", "rpm_per_pct", "f_back", "L0_um",
                       "kappa_scale", "sigma_um", "flow_dir"}
    # missing keys are filled from DEFAULT_THETA
    assert th["sigma_um"] > 0.0


# ---------------------------------------------------------------------------
# statistics helpers on synthetic numbers
# ---------------------------------------------------------------------------
def test_rms_on_synthetic_numbers():
    assert rms([3.0, 4.0]) == pytest.approx(12.5 ** 0.5)
    assert rms([-3.0, 3.0]) == pytest.approx(3.0)
    assert rms([1.9, -1.9, 1.9, -1.9]) == pytest.approx(1.9)
    assert rms([0.0, 0.0]) == 0.0
    assert rms([]) is None
    assert rms([None, float("nan")]) is None
    assert rms([2.0, None, float("nan"), -2.0]) == pytest.approx(2.0)


def test_max_abs_and_mean_bias():
    assert max_abs([-2.5, 1.0, 0.3]) == pytest.approx(2.5)
    assert max_abs([None, float("nan")]) is None
    assert mean_bias([1.0, -2.0, 4.0]) == pytest.approx(1.0)
    assert mean_bias([]) is None
    # bias of a residual list is measured - predicted
    resid = [-0.2, 1.5, -0.4]
    assert mean_bias(resid) == pytest.approx(0.3)
    assert rms(resid) == pytest.approx(((0.04 + 2.25 + 0.16) / 3) ** 0.5)


# ---------------------------------------------------------------------------
# slow smoke tests (opt-in: PLATESIM_SLOW=1)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SLOW, reason="slow: one coupon simulate_run (~10 s); "
                                     "set PLATESIM_SLOW=1")
def test_trench_prediction_smoke():
    res = trench_coupon_prediction(resolution="coarse")
    p = res["pred"]
    assert 0.0 < p["min_um"] <= p["p50_um"] <= p["max_um"]
    assert p["min_um"] <= p["mean_um"] <= p["max_um"]
    assert p["p5_um"] <= p["p95_um"]
    assert p["max_min_ratio"] == pytest.approx(p["max_um"] / p["min_um"])
    assert set(res["inside_range"]) >= {"no_trench", "narrow_trench", "wide_trench"}
    assert res["measured"]["min_um"] == 39.5
    assert any("pulse-reverse" in n for n in res["notes"])
    json.dumps(res)


@pytest.mark.skipif(not SLOW, reason="slow: one LOO refit + 8 full-fit predictions; "
                                     "set PLATESIM_SLOW=1")
def test_leave_one_out_smoke_single_run():
    res = leave_one_out(ids=[3], max_nfev=2, time_budget_s=90.0, resolution="coarse")
    assert len(res["rows"]) == 1
    row = res["rows"][0]
    assert row["id"] == 3
    assert row["pred_loo_um"] > 0.0
    assert res["faraday_rms_um"] is not None and res["faraday_rms_um"] > 0.0
    assert isinstance(res["notes"], list) and res["notes"]
    json.dumps(res)
