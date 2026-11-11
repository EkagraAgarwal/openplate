"""Tests for platesim.trenchflow.convergence (Task 2 of plan.md).

Covered:
1. ``run(quick=True)`` returns the documented keys and writes a quick JSON artifact - pytest always
   redirects it to a tmp directory, so neither ``convergence.json`` nor ``convergence_quick.json``
   in the package is touched by the test suite.
2. The Brownian tracers and the continuum solver agree on floor/field and wall/field in a real
   120 x 60 um trench with LBM flow and ~30 % surface depletion (within 3 standard errors or 8 %).
3. floor/field moves by less than 0.03 between dx 6 and dx 4 (Trench 1 narrow site, one stage).
4. The Richardson helper reports NaN (not an exception) for a deliberately non-monotone sequence,
   and recovers a known power law on non-uniform grids.
5. The written file is strict JSON: non-finite in-memory values appear as ``null``.

The solver-bearing tests are the two expensive ones (about 30 s and 15 s); the whole file stays
under a minute so the five ``test_trenchflow_*`` files pass in about 75 s.
"""
import json
import math

import numpy as np
import pytest

from platesim.trenchflow.convergence import (
    convergence_row,
    particle_cross_check,
    richardson,
    run,
    site_conditions,
)

NARROW = {"id": "test_narrow", "width_um": 227.0, "depth_um": 88.0, "T_C": 22.0, "hours": 4.5,
          "field_um": {"lo": 69.0, "hi": 80.0, "best": 69.0},
          "floor_um": {"lo": 53.0, "hi": 53.0, "best": 53.0}}


@pytest.fixture(scope="module")
def quick_path(tmp_path_factory):
    """Tmp destination for the quick artifact: pytest never writes into the package."""
    return tmp_path_factory.mktemp("trenchflow_convergence") / "convergence_quick.json"


@pytest.fixture(scope="module")
def quick_result(quick_path):
    """One quick run, shared by the key/JSON assertions (it costs ~30 s)."""
    return run(quick=True, out_path=quick_path)


def test_quick_run_keys_and_json(quick_result):
    """quick run: expected keys, one site per table, JSON on disk and JSON-serialisable."""
    res = quick_result
    assert {"quick", "constants", "sites", "march", "tau_clamp", "particle_cross_check",
            "notes", "runtime_s"} <= set(res)
    assert res["quick"] is True
    assert res["constants"]["dx_list_um"] == [8.0, 6.0, 4.0]
    assert len(res["sites"]) == 1
    site = res["sites"][0]
    assert {"id", "conditions", "rows", "richardson_floor_over_field"} <= set(site)
    cond = site["conditions"]
    for key in ("delta_um", "shear_rate_s", "j_field_mA_cm2", "D_cm2_s", "field_um"):
        assert np.isfinite(cond[key])


@pytest.mark.parametrize("key", ["floor_over_field", "wall_up_over_field", "wall_down_over_field",
                                 "wall_over_field", "k_cm_s", "lbm_converged", "lbm_steps",
                                 "penetration_um", "wall_s"])
def test_quick_row_keys(quick_result, key):
    """Every grid row carries the quantities the task asks to report."""
    for row in quick_result["sites"][0]["rows"]:
        assert key in row


def test_quick_json_round_trip(quick_result, quick_path):
    """The written JSON parses, is strict (no bare NaN) and matches what run() returned."""
    assert quick_path.exists()
    text = quick_path.read_text()

    def _reject(token):                                  # json calls this only for NaN/Infinity
        raise AssertionError(f"non-finite token {token!r} in the written JSON")

    on_disk = json.loads(text, parse_constant=_reject)
    assert on_disk["quick"] is True
    assert quick_result["output_json"] == str(quick_path)
    assert [s["id"] for s in on_disk["sites"]] == [s["id"] for s in quick_result["sites"]]
    dx_disk = [r["dx_um"] for r in on_disk["sites"][0]["rows"]]
    dx_res = [r["dx_um"] for r in quick_result["sites"][0]["rows"]]
    assert dx_disk == dx_res
    assert all(r["lbm_converged"] for r in on_disk["sites"][0]["rows"])
    rch = on_disk["sites"][0]["richardson_floor_over_field"]
    assert {"observed_order", "extrapolated", "dx_probe_minus_extrapolated"} <= set(rch)
    # one march row -> the spread is undefined in memory (NaN) and null on disk
    entry = on_disk["march"]["sites"][0]
    assert entry["spread_defined"] is False
    assert entry["spread_floor_over_field"] is None
    assert math.isnan(quick_result["march"]["sites"][0]["spread_floor_over_field"])


def test_quick_march_and_tau_sections(quick_result):
    """Quick mode still exercises the marching sweep (1 run) and the tau-clamp check."""
    march = quick_result["march"]
    assert march is not None and len(march["sites"]) == 1
    rows = march["sites"][0]["rows"]
    assert [(r["dx_um"], r["n_stages"]) for r in rows] == [(8.0, 2)]
    # one run cannot define a spread: NaN (in memory), spread_defined False, with a note
    entry = march["sites"][0]
    assert entry["spread_defined"] is False
    assert math.isnan(entry["spread_floor_over_field"])
    assert math.isnan(entry["min_floor_over_field"]) and math.isnan(entry["max_floor_over_field"])
    assert "spread undefined" in entry["note"]
    # the march summary must not claim an LBM convergence check it does not have
    assert "lbm_converged_all" not in rows[0]
    assert "eddy_summary_available_all" in rows[0]
    tau = quick_result["tau_clamp"]
    assert tau is not None and len(tau["pairs"]) >= 1
    # the requested lid pair is clamped to the same lattice lid in this cavity
    assert tau["degenerate_requested_pair"] is True
    assert tau["pairs"][0]["max_rel_diff"] == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(tau["pairs"][0]["tau"][0])
    # the finding is reported explicitly, not hidden behind the degenerate pair
    assert isinstance(tau["interpretation"], str) and "degenerate" in tau["interpretation"]
    for key in ("tau_clamp_benign", "max_rel_diff", "max_rel_diff_sig", "sig_threshold_frac",
                "benign_threshold"):
        assert key in tau
    assert tau["sig_threshold_frac"] == pytest.approx(0.05)
    assert tau["benign_threshold"] == pytest.approx(0.05)
    if len(tau["pairs"]) > 1:
        assert tau["max_rel_diff"] >= tau["pairs"][1]["max_rel_diff"] - 1e-12
        assert tau["tau_clamp_benign"] == bool(tau["max_rel_diff_sig"] < tau["benign_threshold"])
    # plan item 3: the tau-clamp site is the narrow one
    assert quick_result["constants"]["tau_site"] == quick_result["sites"][0]["id"]


def test_particles_versus_continuum_in_trench():
    """120 x 60 um trench, dx 6, LBM flow, ~30 % depletion: tracers vs continuum agree."""
    pc = particle_cross_check()
    st = pc["settings"]
    assert (st["width_um"], st["depth_um"], st["dx_um"]) == (120.0, 60.0, 6.0)
    assert st["use_flow"] is True and st["lbm_converged"] is True
    # 20-40 % depletion of the field surface (c_wall = 0.6-0.8)
    assert 0.6 <= pc["c_wall_field"] <= 0.8, pc["c_wall_field"]
    # the shear input is documented as such, not as the Nernst-consistent delta
    assert "SHEAR input" in st["delta_note"] and "delta_eff_um" in st["delta_note"]
    for name in ("field", "floor", "wall"):
        assert np.isfinite(pc["continuum"][name]) and pc["continuum"][name] > 0.0
        assert np.isfinite(pc["particles"][name]) and pc["particles"][name] > 0.0
    for name in ("field_counts", "floor_counts", "wall_counts"):
        assert pc["particles"][name] > 300.0, (name, pc["particles"][name])
    for key in ("floor_over_field", "wall_over_field"):
        cmp = pc["comparisons"][key]
        assert np.isfinite(cmp["difference"]), (key, cmp)
        assert abs(cmp["difference"]) <= cmp["tolerance"], (key, cmp)
        assert abs(cmp["difference"]) <= 0.08 * abs(cmp["continuum"]), (key, cmp)
    assert pc["agree"] is True


def test_floor_over_field_converged_between_dx6_and_dx4():
    """floor/field changes by < 0.03 between dx 6 and dx 4 (narrow site, one stage, with flow)."""
    cond = site_conditions(NARROW)
    r6 = convergence_row(NARROW, 6.0, cond)
    r4 = convergence_row(NARROW, 4.0, cond)
    for row in (r6, r4):
        assert row["lbm_converged"] is True
        assert 0.0 < row["floor_over_field"] < 1.05
        assert np.isfinite(row["wall_up_over_field"]) and np.isfinite(row["wall_down_over_field"])
    assert abs(r6["floor_over_field"] - r4["floor_over_field"]) < 0.03
    # the field flux is set by j_field, so k must be grid independent to a few percent
    assert abs(r6["k_cm_s"] - r4["k_cm_s"]) / r4["k_cm_s"] < 0.05


def _rows(values, dxs=(8.0, 6.0, 4.0)):
    return [{"dx_um": h, "floor_over_field": v} for h, v in zip(dxs, values)]


def test_richardson_power_law_nonuniform_grid():
    """f = f0 + C dx^2 on the non-uniform grid 8/6/4 recovers p = 2 and f0 (general formula)."""
    f0, c = 0.9000, 1.0e-4
    dxs = (8.0, 6.0, 4.0)
    vals = [f0 + c * h ** 2 for h in dxs]
    res = richardson(_rows(vals, dxs))
    assert res["observed_order"] == pytest.approx(2.0, rel=1e-6, abs=1e-9)
    assert res["extrapolated"] == pytest.approx(f0, abs=1e-12)
    # dx = 4 is the finest grid of that set: its error is +C * 16, ~1.6e-3
    assert abs(res["dx_probe_minus_extrapolated"] - c * 16.0) < 1e-9

    # a p = 1.5 law on 6/4/3 (also non-uniform) must also come back exactly
    dxs2 = (6.0, 4.0, 3.0)
    res2 = richardson(_rows([f0 + 2e-3 * h ** 1.5 for h in dxs2], dxs2), dx_probe=4.0)
    assert res2["observed_order"] == pytest.approx(1.5, rel=1e-6)
    assert res2["extrapolated"] == pytest.approx(f0, abs=1e-10)


def test_richardson_monotone_and_non_monotone():
    """Non-monotone or degenerate data give NaN instead of an exception."""
    mono = richardson(_rows((0.950, 0.940, 0.935)))
    assert np.isfinite(mono["observed_order"]) and mono["observed_order"] > 0.0
    assert np.isfinite(mono["extrapolated"])
    assert abs(mono["dx_probe_minus_extrapolated"]) < 0.01

    bad = richardson(_rows((0.930, 0.960, 0.935)))          # not monotone in dx
    assert np.isnan(bad["observed_order"])
    assert np.isnan(bad["extrapolated"])
    assert "NaN" in bad["note"]

    flat = richardson(_rows((0.940, 0.940, 0.940)))          # zero second difference
    assert np.isnan(flat["observed_order"])

    short = richardson(_rows((0.94, 0.935), dxs=(8.0, 4.0)))
    assert np.isnan(short["extrapolated"]) and "at least 3" in short["note"]
