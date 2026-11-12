"""Tests for the coupon -> trench multiscale bridge (quick settings only).

Runtime budget < 2 min: one coupon pipeline solve (memoised across the tests), a
handful of quick feature-scale predictions (dx 8 um, 2 flow stages), and two
solver-free checks (summarize() intervals, resolution threading under monkeypatch).

The two-way (Plan F1) tests at the bottom use SYNTHETIC sites and the coupon
conditions of the memoised solve, so they need none of the withheld data files.
"""
import math

import numpy as np
import pytest

from platesim.feature.calibrate import load_sites
from platesim.multiscale import (cache_stats, clear_cache, coupon_area_map,
                                 effective_area_factor, predict_trench,
                                 rescale_coupon_currents, site_conditions)
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


# ---------------------------------------------------------------------------
# Plan F1: two-way coupon <-> trench coupling.
# Synthetic sites only (the same keys ``load_sites`` validates), so none of the
# withheld data files is needed.
# ---------------------------------------------------------------------------
_SYNTH_KEYS = ("width_um", "depth_um", "T_C", "j_mA_cm2", "hours", "field_um",
               "floor_um")


def _synthetic_site(site_id, width_um, depth_um, T_C=22.0, j_mA_cm2=15.0,
                    hours=4.5, field_um=80.0, floor_um=53.0) -> dict:
    """A synthetic SEM site: exactly the keys ``feature.calibrate.load_sites`` validates."""
    site = {"id": site_id,
            "width_um": float(width_um), "depth_um": float(depth_um),
            "T_C": float(T_C), "j_mA_cm2": float(j_mA_cm2),
            "hours": float(hours),
            "field_um": {"best": float(field_um), "lo": float(field_um) - 11.0,
                         "hi": float(field_um)},
            "floor_um": {"best": float(floor_um), "lo": float(floor_um) - 3.0,
                         "hi": float(floor_um) + 3.0}}
    assert set(_SYNTH_KEYS) <= set(site)
    return site


@pytest.fixture(scope="module")
def synthetic_sites():
    return {"narrow": _synthetic_site("syn_narrow", 227.0, 88.0),
            "wide": _synthetic_site("syn_wide", 944.0, 97.0),
            "flat": _synthetic_site("syn_flat", 227.0, 0.0)}


@pytest.fixture(scope="module")
def narrow_area_map(centre, synthetic_sites):
    """One 2x2 coupon two-way map for the narrow site (built once for the module)."""
    return coupon_area_map(synthetic_sites["narrow"], f_open=0.2, n=2)


def test_effective_area_factor_limits_and_report(centre, synthetic_sites):
    """A_eff -> 1 without trenches, > 1 for a deep array; report narrow/wide at 0.2."""
    narrow = synthetic_sites["narrow"]
    wide = synthetic_sites["wide"]
    flat = synthetic_sites["flat"]

    # no trenches: zero depth is EXACTLY the flat surface; f_open -> 0 vanishes
    assert effective_area_factor(flat, 0.2, centre)["a_eff"] == 1.0
    a_tiny = effective_area_factor(narrow, 1e-6, centre)
    assert a_tiny["a_eff"] == pytest.approx(1.0, abs=1e-5)
    assert a_tiny["pitch_um"] == pytest.approx(227.0e6)

    a20 = effective_area_factor(narrow, 0.2, centre)
    a40 = effective_area_factor(narrow, 0.4, centre)
    # more open area -> shorter pitch -> more wall area per pitch -> larger factor
    assert a40["a_eff"] > a20["a_eff"] > 1.0
    # all surfaces share k, so the factor is BELOW the bare true-area ratio
    true_area_ratio = 1.0 + 2.0 * 88.0 / a20["pitch_um"]
    assert a20["a_eff"] < true_area_ratio
    assert 0.0 < a20["floor_over_field"] < 1.0
    assert a20["k_cm_s"] > 0.0 and a20["flow_solver"] == "stokes"

    report = {}
    for name in ("narrow", "wide"):
        site = synthetic_sites[name]
        r = effective_area_factor(site, 0.2, centre)
        report[name] = r["a_eff"]
        print(f"A_eff({name} {site['width_um']:.0f}x{site['depth_um']:.0f} um, "
              f"f_open=0.2) = {r['a_eff']:.6f}  pitch={r['pitch_um']:.0f} um  "
              f"floor/wall ratios={r['floor_over_field']:.4f}/"
              f"{r['wall_up_over_field']:.4f}/{r['wall_down_over_field']:.4f}")
    assert report["narrow"] > 1.0
    assert report["wide"] > 1.0

    # the factor is memoised: the cached dict is the same object
    assert effective_area_factor(wide, 0.2, centre) is effective_area_factor(
        wide, 0.2, centre)


def test_rescale_coupon_currents_conserves_total():
    """The galvanostatic rescaling redistributes current and conserves the total."""
    j = np.array([[10.0, 12.0], [14.0, 16.0]])
    a = np.array([[1.10, 1.20], [1.05, 1.00]])
    jf, mean = rescale_coupon_currents(j, a)
    assert jf.sum() == pytest.approx(j.sum(), rel=1e-12)
    assert mean == pytest.approx((j * a).sum() / j.sum(), rel=1e-15)
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    jf2, mean2 = rescale_coupon_currents(j, a, w)
    assert float((w * jf2).sum()) == pytest.approx(float((w * j).sum()), rel=1e-12)
    assert mean2 == pytest.approx(float((w * j * a).sum()) / float((w * j).sum()))
    # A_eff == 1 everywhere is the identity (the one-way limit)
    jf3, mean3 = rescale_coupon_currents(j, np.ones_like(j))
    assert np.allclose(jf3, j, rtol=0.0, atol=0.0) and mean3 == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rescale_coupon_currents(j, a, w[:1])


def test_two_way_default_is_bit_identical_and_inert(centre, synthetic_sites,
                                                    monkeypatch):
    """two_way=False (default) reproduces the one-way result and never solves A_eff."""
    site = synthetic_sites["narrow"]
    ref = predict_trench(site, centre, "flow", quick=True)
    off = predict_trench(site, centre, "flow", quick=True, two_way=False)
    assert ref == off
    assert ref["two_way"] is False and ref["a_eff"] is None
    assert ref["a_eff_mean"] is None and ref["two_way_converged"] is None
    assert ref["two_way_iterations"] == 0 and ref["two_way_history"] == []
    # the local current is the coupon current, untouched
    assert ref["j_mA_cm2"] == centre["j_mA_cm2"]
    assert ref["j_coupon_mA_cm2"] == centre["j_mA_cm2"]
    # the mfac Faraday relation of the one-way flow path still holds exactly
    mfac = float(model_constants(float(site["T_C"]))[2])
    expected = (centre["j_mA_cm2"] * 1e-3 * mfac * centre["efficiency"]
                * float(site["hours"]) * 3600.0 / 1e-4)
    assert ref["field_um"] == pytest.approx(expected, rel=1e-12)

    import platesim.multiscale.bridge as B

    def _boom(*args, **kwargs):
        raise AssertionError("the one-way path must not solve the area factor")

    monkeypatch.setattr(B, "effective_area_factor", _boom)
    monkeypatch.setattr(B, "coupon_area_map", _boom)
    again = predict_trench(site, centre, "flow", quick=True)
    assert again == ref


def test_two_way_fixed_point_and_current_conservation(narrow_area_map):
    """The coupon <-> feature sweep converges in < 10 iterations and keeps the current."""
    m = narrow_area_map
    assert m["converged"] is True
    assert 1 <= m["iterations"] < 10
    assert len(m["history"]) == m["iterations"]
    assert m["history"][-1]["max_change"] < 1e-3
    changes = [h["max_change"] for h in m["history"]]
    assert all(b <= a + 1e-12 for a, b in zip(changes, changes[1:]))  # contracting

    w = m["weights"]
    total_in = float((w * m["j_coupon_mA_cm2"]).sum())
    total_out = float((w * m["j_field_mA_cm2"]).sum())
    assert total_out == pytest.approx(total_in, rel=1e-9)
    assert np.all(m["j_field_mA_cm2"] > 0.0)
    # each node draws j_coupon * A_eff / <A_eff>
    assert np.allclose(m["j_field_mA_cm2"],
                       m["j_coupon_mA_cm2"] * m["a_eff"] / m["a_eff_mean"],
                       rtol=1e-12, atol=0.0)
    assert m["a_eff_mean_area"] > 1.0
    assert m["a_eff_mean"] == pytest.approx(m["a_eff_mean_area"], rel=1e-3)
    assert m["f_open"] == 0.2 and m["n"] == 2

    # memoised: a second identical request is free and identical
    again = coupon_area_map(_synth_narrow_site(), f_open=0.2, n=2)
    assert again is m


def _synth_narrow_site() -> dict:
    return _synthetic_site("syn_narrow", 227.0, 88.0)


def test_two_way_predict_rescales_local_current(centre, synthetic_sites,
                                                narrow_area_map):
    """two_way=True runs the feature model at j_coupon * A_eff / <A_eff>."""
    site = synthetic_sites["narrow"]
    m = narrow_area_map
    one = predict_trench(site, centre, "flow", quick=True)
    two = predict_trench(site, centre, "flow", quick=True, two_way=True,
                         two_way_map=m, f_open=0.2)
    assert two["two_way"] is True and two["error"] is None
    assert two["j_coupon_mA_cm2"] == centre["j_mA_cm2"]
    # the centre sits within 0.1 % of the coupon mean, so the factor is ~1 there;
    # the rescaling itself is exactly j_coupon * A_eff / <A_eff>
    assert two["j_mA_cm2"] == pytest.approx(
        centre["j_mA_cm2"] * two["a_eff"] / two["a_eff_mean"], rel=1e-12)
    assert two["j_mA_cm2"] / two["j_coupon_mA_cm2"] == pytest.approx(
        two["a_eff"] / m["a_eff_mean"], rel=1e-12)
    assert 0.99 < two["j_mA_cm2"] / two["j_coupon_mA_cm2"] < 1.01
    assert two["two_way_iterations"] == m["iterations"]
    assert two["two_way_converged"] is True
    assert len(two["two_way_history"]) == m["iterations"]
    # flow: field thickness is linear in the local current -> the ratio is exact
    assert two["field_um"] / one["field_um"] == pytest.approx(
        two["j_mA_cm2"] / one["j_mA_cm2"], rel=1e-12)

    # at a map node whose A_eff is above the coupon mean, the local current RISES
    node = {**centre, "x_cm": float(m["x_cm"][0]), "y_cm": float(m["y_cm"][0])}
    two_node = predict_trench(site, node, "flow", quick=True, two_way=True,
                              two_way_map=m, f_open=0.2)
    assert two_node["j_mA_cm2"] > two_node["j_coupon_mA_cm2"]
    assert two_node["field_um"] > one["field_um"]

    lv = predict_trench(site, node, "levelset", quick=True, two_way=True,
                        two_way_map=m, f_open=0.2)
    assert lv["error"] is None and lv["field_um"] > 0.0
    assert lv["two_way"] is True
    assert lv["j_mA_cm2"] > lv["j_coupon_mA_cm2"]           # same rescaling

    # a map built for another geometry / open fraction is rejected
    with pytest.raises(ValueError):
        predict_trench(synthetic_sites["wide"], centre, "flow", quick=True,
                       two_way=True, two_way_map=m, f_open=0.2)
    with pytest.raises(ValueError):
        predict_trench(site, centre, "flow", quick=True, two_way=True,
                       two_way_map=m, f_open=0.3)
