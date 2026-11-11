"""Tests for platesim.feature.convergence + the efficiency-scaled Cu2+ sink and Cu mass balance.

Short DC/PR trench runs at dx = 8 um, plus one dx = 1.0 um filled-stop run; the whole file targets < 60 s.
"""
from __future__ import annotations

import json
import math
import time

import pytest

from platesim.feature import calibrate as cal
from platesim.feature import convergence as conv
from platesim.feature.engine import Params, dc, run_case

SITE = dict(id="test_narrow", sample="synthetic", width_um=227.0, depth_um=90.0, T_C=22.0,
            j_mA_cm2=15.0, hours=0.25,
            pr={"t_f_ms": 9.0, "t_r_ms": 0.5, "t_off_ms": 0.5, "rr": 2.0},
            field_um={"lo": 0.0, "hi": 0.0, "best": 0.0}, floor_um={"lo": 0.0, "hi": 0.0, "best": 0.0})


def _trench_run(efficiency: float, hours: float = 0.25, return_sim: bool = False):
    p = Params(geometry="trench", H_um=90.0, suppressor=False, dx_um=8.0, efficiency=efficiency)
    return run_case(227.0, 22.0, dc(15.0, hours), p, f_open=0.2, label=f"eff{efficiency:g}",
                    out=None, return_sim=return_sim)


def _last(series: dict, key: str) -> float:
    return next(float(v) for v in reversed(series[key]) if v is not None)


# ------------------------------------------------------------------ efficiency in the Cu2+ sink
def test_efficiency_halves_thickness_and_raises_floor_concentration():
    """Same current, half the efficiency -> half the growth and less Cu2+ depletion (higher c_floor)."""
    lo = _trench_run(0.5)
    hi = _trench_run(1.0)
    f_lo, f_hi = _last(lo["series"], "face_thickness_um"), _last(hi["series"], "face_thickness_um")
    assert f_hi > 0.0 and 0.45 < f_lo / f_hi < 0.55, (f_lo, f_hi)
    # the sink is now scaled by the efficiency, so at efficiency 0.5 the floor is less depleted
    c_lo, c_hi = _last(lo["series"], "c_floor"), _last(hi["series"], "c_floor")
    assert c_lo > c_hi, (c_lo, c_hi)


# ------------------------------------------------------------------ Cu2+ mass balance
def test_cu_balance_closes_on_short_dc_trench():
    """Moles of Cu2+ through the top Dirichlet plane vs moles deposited (no level set involved)."""
    for eff in (1.0, 0.82):
        res, sim = _trench_run(eff, hours=0.5, return_sim=True)
        rel = res["result"]["cu_balance_rel"]
        assert rel is not None, res["result"]
        assert abs(rel) < 0.05, (eff, rel)
        assert sim.cu_balance_rel is not None and sim.cu_balance_rel == pytest.approx(rel, abs=5e-7)
        assert sim.cu_influx > 0.0


def test_cu_balance_survives_a_filled_stop():
    """The 'filled' break fires after the diagnostics but before the front advance and q_total, and a
    stopped run must not keep that step's influx.  A low fill_stop override (0.05 instead of 0.995)
    reaches the 'filled' break in ~5 steps on a small trench (0.1 s), so a one-step mismatch between
    influx and charge would show up as a ~20 % error here."""
    p = Params(geometry="trench", H_um=20.0, suppressor=False, dx_um=1.0, fill_stop=0.05)
    res, sim = run_case(15.0, 22.0, dc(15.0, 0.5), p, f_open=0.2, label="filled_stop",
                        out=None, return_sim=True)
    assert res["result"]["stop_reason"] == "filled"
    assert res["meta"]["steps"] <= 8
    rel = res["result"]["cu_balance_rel"]
    assert rel is not None and abs(rel) < 0.05, (rel, res["meta"]["steps"])
    assert sim.q_total > 0.0 and sim.cu_influx > 0.0
    assert sim.cu_balance_rel == pytest.approx(rel, abs=5e-7)


# ------------------------------------------------------------------ convergence harness
def test_quick_run_returns_expected_keys():
    t0 = time.time()
    out = conv.run(quick=True)                       # no `out` -> the production JSON is not touched
    assert time.time() - t0 < 60.0, time.time() - t0
    assert out["quick"] is True
    assert set(out) >= {"conditions", "runs", "convergence", "timestep", "wall_s"}
    assert out["conditions"]["hours"] == 0.5
    assert set(out["runs"]) == {"trench1_wide", "trench1_narrow"}
    for case, rows in out["runs"].items():
        assert [r["dx_requested"] for r in rows] == [8.0, 6.0], case
        for r in rows:
            for key in ("dx_um", "field_um", "floor_um", "ratio", "faraday_um", "faraday_rel_err",
                        "field_vs_faraday_rel", "cu_balance_rel", "wall_s", "steps", "stop_reason"):
                assert key in r, (case, key)
            assert r["field_um"] > 0.0 and r["floor_um"] > 0.0
            assert r["ratio"] == pytest.approx(r["floor_um"] / r["field_um"], rel=1e-9)
            # faraday_rel_err = engine deposit-vs-charge check (not the physical field deficit)
            assert r["faraday_rel_err"] is not None and abs(r["faraday_rel_err"]) < 0.02, r
            assert r["field_vs_faraday_rel"] is not None
            assert math.isfinite(r["field_vs_faraday_rel"])
            assert r["field_vs_faraday_rel"] != pytest.approx(r["faraday_rel_err"], abs=1e-6)
            assert r["cu_balance_rel"] is not None and abs(r["cu_balance_rel"]) < 0.05
            assert r["wall_s"] >= 0.0
        assert set(out["convergence"][case]) >= {"field_um", "floor_um", "ratio", "dx_requested",
                                                 "dx_um_effective", "note"}
    assert out["timestep"].get("skipped") is True
    json.dumps(out, allow_nan=False)


def test_predict_full_reproduces_calibrate_predict():
    """_predict_full mirrors calibrate.predict's Params/recipe: same field and floor at the same dx."""
    cal._CACHE.clear()
    site = dict(SITE)
    a = conv._predict_full(site, conv.THETA, 8.0)
    b = cal.predict(site, conv.THETA, dx_um=8.0)
    cal._CACHE.clear()
    assert a["field_um"] == pytest.approx(b["field_um"], abs=1e-6)
    assert a["floor_um"] == pytest.approx(b["floor_um"], abs=1e-6)
    assert a["dx_requested"] == 8.0 and a["cfl"] == 0.5


def test_order_extrapolation_recovers_a_known_order():
    """f = f0 + C dx^2 on three grids: order 2 and the exact limit (guards _order_extrap)."""
    dx = [6.0, 4.0, 3.0]
    f0, c = 0.75, 0.004
    f = [f0 + c * d ** 2 for d in dx]
    p, lim = conv._order_extrap(dx, f)
    assert p == pytest.approx(2.0, abs=1e-6)
    assert lim == pytest.approx(f0, abs=1e-10)
