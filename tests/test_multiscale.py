"""Tests for the coupon -> trench multiscale bridge (quick settings only).

Runtime budget < 2 min: one coupon pipeline solve (memoised across the tests), a
handful of quick feature-scale predictions (dx 8 um, 2 flow stages), and two
solver-free checks (summarize() intervals, resolution threading under monkeypatch).
"""
import math

import pytest

from platesim.feature.calibrate import load_sites
from platesim.multiscale import cache_stats, clear_cache, predict_trench, site_conditions
from platesim.multiscale.validate import GRID_CM, TRENCH_RUN, summarize
from platesim.trenchflow.model import model_constants

SITE_KEYS = {"levelset": {"field_um", "floor_um", "floor_over_field", "wall_um"},
             "flow": {"field_um", "floor_um", "floor_over_field", "wall_um",
                      "wall_up_um", "wall_down_um"}}
COMMON_KEYS = {"model", "site_id", "x_cm", "y_cm", "j_mA_cm2", "efficiency",
               "delta_um", "quick", "field_um", "floor_um", "floor_over_field",
               "wall_um", "warnings", "error"}


@pytest.fixture(scope="module")
def centre():
    """Coupon conditions at the coupon centre (one pipeline solve for the module)."""
    clear_cache()
    return site_conditions(TRENCH_RUN, 0.0, 0.0)


def test_centre_conditions_physical(centre):
    """Centre conditions are finite and inside physical bounds."""
    for key in ("x_cm", "y_cm", "j_mA_cm2", "delta_um", "efficiency",
                "coupon_thickness_um", "front_fraction", "delta_mean_um", "rpm",
                "j_mean_mA_cm2"):
        assert key in centre, key
        assert math.isfinite(centre[key]), (key, centre[key])
    assert 0.0 < centre["j_mA_cm2"] < 60.0, centre["j_mA_cm2"]
    assert 10.0 < centre["delta_um"] < 300.0, centre["delta_um"]
    assert 0.0 < centre["efficiency"] <= 1.0, centre["efficiency"]
    # imp1 is TAPED with f_back_imp = 0, so the cell solve gives exactly 1.0 here:
    # the bound is 0 < front <= 1 and the equality is asserted so a change shows up.
    assert 0.0 < centre["front_fraction"] <= 1.0, centre["front_fraction"]
    assert centre["delta_mean_um"] > 0.0
    assert centre["rpm"] > 0.0
    assert centre["coupon_thickness_um"] > 0.0


def test_grid_conditions_shared_solve_and_mean_current(centre):
    """One coupon solve for the 5x5 grid; the mean local j matches j_set * front."""
    before = cache_stats()["misses"]
    conds = [site_conditions(TRENCH_RUN, x, y) for y in GRID_CM for x in GRID_CM]
    assert cache_stats()["misses"] == before, "positions must reuse the coupon solve"
    assert len(conds) == len(GRID_CM) ** 2
    for c in conds:
        assert math.isfinite(c["j_mA_cm2"]) and c["j_mA_cm2"] > 0.0
        assert 0.0 < c["efficiency"] <= 1.0

    j_set = float(TRENCH_RUN["j_mA_cm2"])
    target = j_set * centre["front_fraction"]
    # area-weighted mean from the simulation (not a plain average of grid samples)
    assert abs(centre["j_mean_mA_cm2"] / target - 1.0) < 0.05, (
        centre["j_mean_mA_cm2"], target)
    # the interior 5x5 nodes must bracket that face mean: they do not include the
    # outermost 0.2 cm ring, where the cell solve puts ~40 % extra current, so their
    # plain average is BELOW the face mean by construction (not a test of the mean)
    j_vals = [c["j_mA_cm2"] for c in conds]
    assert min(j_vals) < centre["j_mean_mA_cm2"] < max(j_vals)
    assert abs(sum(j_vals) / len(j_vals) / target - 1.0) < 0.15
    # the face is not uniform: edge/centre contrast must be positive
    assert max(j_vals) > min(j_vals)


@pytest.mark.parametrize("site_index", [0, 1])
@pytest.mark.parametrize("model", ["levelset", "flow"])
def test_predict_trench_quick(centre, site_index, model):
    """predict_trench runs end to end in quick mode for both models and both sites."""
    site = load_sites()[site_index]
    out = predict_trench(site, centre, model=model, quick=True)
    assert COMMON_KEYS <= set(out), COMMON_KEYS - set(out)
    assert SITE_KEYS[model] <= set(out), SITE_KEYS[model] - set(out)
    assert out["model"] == model
    assert out["quick"] is True
    assert out["site_id"].startswith(site["id"])
    assert out["site_id"] != site["id"]          # position-unique id
    assert "_j" in out["site_id"] and "_h" in out["site_id"]
    assert out["error"] is None
    assert out["field_um"] > 0.0
    assert 0.0 < out["floor_um"] <= out["field_um"] * 1.05
    assert 0.0 < out["floor_over_field"] <= 1.05
    if model == "levelset":
        assert out["wall_um"] is None
    else:
        assert out["wall_um"] == pytest.approx(
            0.5 * (out["wall_up_um"] + out["wall_down_um"]))
        assert out["j_field_mA_cm2"] > 0.0


def test_levelset_cache_uses_unique_ids(centre):
    """Two positions must not share the feature.calibrate cache entry (unique ids)."""
    site = load_sites()[0]
    a = predict_trench(site, {**centre, "x_cm": 0.0, "y_cm": 0.0}, "levelset", quick=True)
    b = predict_trench(site, {**centre, "x_cm": 0.8, "y_cm": 0.8}, "levelset", quick=True)
    assert a["site_id"] != b["site_id"]
    # same local conditions but a different position id -> same physics, new cache key
    c = predict_trench(site, {**centre, "x_cm": 0.8, "y_cm": 0.8}, "levelset", quick=True)
    assert c["field_um"] == pytest.approx(b["field_um"])


def test_flow_field_current_from_local_conditions(centre):
    """flow: field_um = j*1e-3*Mfac*eff*hours*3600/1e-4 and j_field = j*eff (narrow site)."""
    site = load_sites()[1]                       # narrow site: cheapest flow solve
    out = predict_trench(site, centre, model="flow", quick=True)
    mfac = float(model_constants(float(site["T_C"]))[2])
    j = centre["j_mA_cm2"]
    eff = centre["efficiency"]
    t_s = float(site["hours"]) * 3600.0
    expected = j * 1e-3 * mfac * eff * t_s / 1e-4
    assert out["field_um"] == pytest.approx(expected, rel=1e-12)
    # predict_site inverts field_um -> field current, so it runs at j*eff
    assert out["j_field_mA_cm2"] == pytest.approx(j * eff, rel=1e-9)
    assert out["site_id"].endswith(f"_j{j:.3f}_h{float(site['hours']):g}")


def test_summarize_measured_intervals():
    """summarize() uses the measured range [lo,hi]; only the degenerate floor is 53+-3."""
    sites = load_sites()
    narrow = [s for s in sites if str(s["id"]).endswith("narrow")][0]
    key = f"{narrow['id']}|levelset"
    assert narrow["floor_um"]["lo"] == narrow["floor_um"]["hi"] == 53.0

    def pred(x, field, floor):
        return {"pos": {"x_cm": x, "y_cm": 0.0, "coupon_thickness_um": 80.0},
                "preds": {key: {"model": "levelset", "x_cm": x, "y_cm": 0.0,
                                "field_um": field, "floor_um": floor,
                                "floor_over_field": (None if floor is None
                                                     else floor / field),
                                "error": None if floor is not None else "ValueError"}}}

    # narrow field = measured range [69, 80]; narrow floor is degenerate (lo == hi),
    # so it uses the repo 3 um reading floor -> [50, 56]
    results = [pred(0.0, 70.0, 53.0),      # field in, floor in  -> reproduces
               pred(0.4, 70.0, 57.0),      # field in, floor out
               pred(-0.4, 60.0, 52.0),     # field out, floor in
               pred(0.8, 70.0, None)]      # failed solve: skipped, counted separately
    out = summarize(results, sites, sigma_um=1.875)
    s = out["per_site_model"][key]
    assert s["measured_floor_um"] == [50.0, 56.0]
    assert s["measured_floor_sigma_um"] == 3.0
    assert s["measured_field_um"] == [69.0, 80.0]          # measured range, not best+-sigma
    assert s["n_positions"] == 3 and s["n_failed"] == 1
    assert s["frac_field_inside"] == pytest.approx(2.0 / 3.0)
    assert s["frac_floor_inside"] == pytest.approx(2.0 / 3.0)
    assert s["frac_both_inside"] == pytest.approx(1.0 / 3.0)
    assert [p["x_cm"] for p in s["reproducing_positions"]] == [0.0]
    assert out["flat_site"]["measured_um"] == pytest.approx(39.5)
    assert "best +/- 3" in out["measured_rule"]


def test_kmix_switch_changes_levelset_flow_ignores_it(centre):
    """k_mix is a level-set model switch: it lowers floor/ratio; flow ignores it."""
    site = load_sites()[1]                       # narrow site (cheapest solves)
    hi = predict_trench(site, centre, "levelset", quick=True, k_mix=1.0)
    lo = predict_trench(site, centre, "levelset", quick=True, k_mix=0.0)
    assert hi["k_mix"] == 1.0 and lo["k_mix"] == 0.0
    # eddy mixing at k_mix=1 suppresses the bottom-up signal: floor and ratio rise
    assert lo["floor_um"] < hi["floor_um"] - 1.0, (lo["floor_um"], hi["floor_um"])
    assert lo["floor_over_field"] < hi["floor_over_field"]
    f1 = predict_trench(site, centre, "flow", quick=True, k_mix=1.0)
    f0 = predict_trench(site, centre, "flow", quick=True, k_mix=0.0)
    assert f1["k_mix"] is None and f0["k_mix"] is None     # not a flow parameter
    assert f0["floor_um"] == pytest.approx(f1["floor_um"])


def test_evaluate_grid_threads_resolution(monkeypatch):
    """evaluate_grid(resolution=...) reaches site_conditions (recorded, no solve)."""
    import platesim.multiscale.validate as V

    seen: list[str] = []

    def fake_conditions(run, x_cm, y_cm, theta=None, resolution="coarse"):
        seen.append(str(resolution))
        return {"x_cm": float(x_cm), "y_cm": float(y_cm), "j_mA_cm2": 15.0,
                "delta_um": 30.0, "efficiency": 0.95, "coupon_thickness_um": 60.0,
                "front_fraction": 1.0, "delta_mean_um": 31.0, "rpm": 400.0,
                "j_mean_mA_cm2": 15.1}

    monkeypatch.setattr(V, "site_conditions", fake_conditions)
    monkeypatch.setattr(V, "_eval_position",
                        lambda task: {"pos": task[0], "preds": {}})
    out = V.evaluate_grid(quick=True, jobs=1, resolution="fine")
    assert len(out) == len(V.GRID_CM) ** 2
    assert seen == ["fine"] * len(out)
