"""Tests for platesim.feature.calibrate (SEM trench calibration harness).

All runs use dx_um=8.0 and 0.5 h synthetic plating so the module stays well under 60 s.
"""
from __future__ import annotations

import pytest

from platesim.feature import calibrate as cal


@pytest.fixture(autouse=True)
def _fresh_cache():
    """predict() caches by (site id, theta, dx, use_pr); keep tests independent."""
    cal._CACHE.clear()
    yield
    cal._CACHE.clear()


def _fake_site(site_id: str, width_um: float) -> dict:
    return {"id": site_id, "sample": "synthetic", "width_um": width_um, "depth_um": 90.0,
            "T_C": 22.0, "j_mA_cm2": 15.0, "hours": 0.5,
            "field_um": {"lo": 0.0, "hi": 0.0, "best": 0.0},
            "floor_um": {"lo": 0.0, "hi": 0.0, "best": 0.0}}


def _measure(site: dict, theta: dict) -> dict:
    """Turn a predict() run into a fake measurement with lo/hi = best +/- 1 um."""
    pred = cal.predict(site, theta, dx_um=8.0)
    site = dict(site)
    for key in ("field_um", "floor_um"):
        best = float(pred[key])
        site[key] = {"lo": best - 1.0, "hi": best + 1.0, "best": best}
    return site


def test_synthetic_recovery_efficiency():
    sites = [_measure(_fake_site("synth_narrow", 227.0), {"efficiency": 0.8}),
             _measure(_fake_site("synth_wide", 500.0), {"efficiency": 0.8})]
    out = cal.fit(sites, ["efficiency"], dx_um=8.0)
    assert out["theta"]["efficiency"] == pytest.approx(0.8, abs=0.03)
    assert out["identifiable"] is True
    assert out["n_data"] == 4 and out["n_free"] == 1
    assert out["param_sd"]["efficiency"] is not None
    assert len(out["table"]) == 2
    for row in out["table"]:
        assert row["field_pred"] == pytest.approx(row["field_meas"], abs=0.5)


def test_too_many_parameters_raises():
    sites = [_fake_site("one", 227.0)]
    sites[0]["field_um"] = {"lo": 6.0, "hi": 8.0, "best": 7.0}
    sites[0]["floor_um"] = {"lo": 5.0, "hi": 7.0, "best": 6.0}
    with pytest.raises(ValueError, match="cannot fit 2 parameters to 2 numbers"):
        cal.fit(sites, ["efficiency", "delta_um"], dx_um=8.0)


def test_unknown_parameter_raises():
    sites = [_fake_site("one", 227.0)]
    with pytest.raises(ValueError, match="unknown calibration parameter"):
        cal.predict(sites[0], {"nonsense": 1.0}, dx_um=8.0)


def test_load_sites_trench1():
    sites = cal.load_sites()
    assert [s["id"] for s in sites] == ["trench1_wide", "trench1_narrow"]
    assert {s["sample"] for s in sites} == {"Trench 1"}
    for s in sites:
        for key in ("width_um", "depth_um", "T_C", "j_mA_cm2", "hours", "field_um", "floor_um"):
            assert key in s
    assert sites[1]["floor_um"]["lo"] == sites[1]["floor_um"]["hi"] == 53.0


def test_residual_sigma_floor_is_3um(monkeypatch):
    """lo == hi must not divide by zero: sigma is floored at 3 um (reading uncertainty)."""
    site = _fake_site("floor3", 227.0)
    site["field_um"] = {"lo": 82.0, "hi": 82.0, "best": 82.0}
    site["floor_um"] = {"lo": 75.0, "hi": 75.0, "best": 75.0}
    monkeypatch.setattr(cal, "predict", lambda *a, **k: {"field_um": 100.0, "floor_um": 50.0,
                                                         "ratio": 0.5, "outcome": "x", "warnings": []})
    r = cal.residuals({}, [site], dx_um=8.0)
    assert r[0] == pytest.approx((100.0 - 82.0) / 3.0)
    assert r[1] == pytest.approx((50.0 - 75.0) / 3.0)

    site["field_um"] = {"lo": 70.0, "hi": 90.0, "best": 80.0}
    r = cal.residuals({}, [site], dx_um=8.0)
    assert r[0] == pytest.approx((100.0 - 80.0) / 10.0)     # half-span 10 um > 3 um floor