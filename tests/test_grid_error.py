"""Tests for platesim.feature.grid_error (plan v2 item C6).

Manufactured sequences f = f0 + C dx^p must recover f0 and p to 1e-8 (uniform and non-uniform
refinement ratios); the two-grid case must use p_assumed and the conservative Fs = 3; oscillatory
data must be flagged and fall back to the spread; and the 1 h narrow-trench measurements from plan v2
wave-b must extrapolate to the same limit with overlapping error bars.  The whole file is pure
arithmetic (no engine run) and targets well under 60 s.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from platesim.feature.grid_error import grid_error_column, main, richardson

# 1 h, 15 mA/cm2 net, narrow Trench 1 (227 x 88 um), floor/field: plan v2 wave-b measurements.
UPWIND_1H = {8.0: 0.882, 4.0: 0.872, 2.0: 0.873, 1.0: 0.87089}
TVD_1H = {4.0: 0.860, 2.0: 0.867, 1.0: 0.86927}
LIMIT_1H = 0.870


def _seq(f0: float, c: float, p: float, dxs) -> dict[float, float]:
    return {float(dx): f0 + c * float(dx) ** p for dx in dxs}


# ------------------------------------------------------------------ manufactured power laws
@pytest.mark.parametrize("p", [1.0, 2.0])
@pytest.mark.parametrize("dxs", [(8, 4, 2), (8, 4, 2, 1)])
def test_uniform_ratios_recover_f0_and_order(p, dxs):
    f0, c = 0.85, 0.004
    est = richardson(_seq(f0, c, p, dxs))
    assert est["monotone"] is True
    assert est["n_grids"] == len(dxs)
    expected_used = [4.0, 2.0, 1.0] if len(dxs) == 4 else [8.0, 4.0, 2.0]
    assert est["dx_used"] == expected_used
    assert abs(est["extrapolated"] - f0) <= 1e-8, est
    assert abs(est["p_observed"] - p) <= 1e-8, est
    assert "uniform" not in est["note"]                    # uniform ratios are not labelled
    # the reported error is the distance of the finest value from the limit
    assert est["error_fine"] == pytest.approx(abs(est["values_used"][-1] - f0), abs=1e-12)


@pytest.mark.parametrize("dxs", [(8, 6, 4), (6, 4, 3), (8, 6, 4, 2)])
def test_non_uniform_ratios_recover_f0_and_order(dxs):
    """Celik et al. (2008) iterative order: r21 != r32 must still recover the exact power law."""
    f0, c, p = 0.88, 3.0e-4, 1.7
    est = richardson(_seq(f0, c, p, dxs))
    assert est["monotone"] is True
    assert abs(est["extrapolated"] - f0) <= 1e-8, est
    assert abs(est["p_observed"] - p) <= 1e-8, est
    assert "non-uniform" in est["note"], est["note"]


def test_more_than_three_grids_uses_the_three_finest():
    """A corrupt coarse point must not move the estimate: only dx = 8, 4, 2 enter."""
    f0, c, p = 0.9, 2.0e-3, 2.0
    vals = _seq(f0, c, p, (16, 8, 4, 2))
    vals[16.0] += 0.5
    est = richardson(vals)
    assert est["dx_used"] == [8.0, 4.0, 2.0]
    assert est["n_grids"] == 4
    assert abs(est["extrapolated"] - f0) <= 1e-8
    assert abs(est["p_observed"] - p) <= 1e-8
    assert "three finest of 4 grids" in est["note"]


# ------------------------------------------------------------------ two grids
def test_two_grids_use_p_assumed_and_fs_three():
    f0, c, p = 0.85, 0.004, 2.0
    vals = _seq(f0, c, p, (4, 2))
    est = richardson(vals, p_assumed=p)
    assert est["n_grids"] == 2
    assert est["p_observed"] is None                       # order not resolvable from two grids
    assert est["monotone"] is True
    assert est["fs"] == 3.0
    assert abs(est["extrapolated"] - f0) <= 1e-12, est
    assert est["error_fine"] == pytest.approx(abs(vals[2.0] - f0), abs=1e-12)
    assert est["gci_fine"] == pytest.approx(3.0 * est["error_fine"] / abs(vals[2.0]), rel=1e-12)
    assert "supplied" in est["note"]

    # default second order when the caller supplies none
    default = richardson(vals)
    assert default["extrapolated"] == pytest.approx(est["extrapolated"], abs=1e-12)
    assert "assumed" in default["note"]


def test_three_grids_use_fs_125():
    f0, c, p = 0.85, 0.004, 2.0
    est = richardson(_seq(f0, c, p, (8, 4, 2)))
    assert est["fs"] == 1.25
    assert est["gci_fine"] == pytest.approx(1.25 * est["error_fine"] / abs(est["values_used"][-1]),
                                            rel=1e-12)


def test_invalid_p_assumed_rejected():
    with pytest.raises(ValueError):
        richardson({4.0: 0.9, 2.0: 0.89}, p_assumed=0.0)
    with pytest.raises(ValueError):
        richardson({4.0: 0.9, 2.0: 0.89}, p_assumed=float("nan"))


# ------------------------------------------------------------------ non-monotone / degenerate
def test_non_monotone_flagged_and_falls_back_to_the_spread():
    """Upwind 1 h data: 0.872 (dx 4) < 0.873 (dx 2) > 0.87089 (dx 1) -- no real order."""
    est = richardson(UPWIND_1H)
    assert est["monotone"] is False
    assert est["p_observed"] is None
    assert est["dx_used"] == [4.0, 2.0, 1.0]
    assert est["extrapolated"] == pytest.approx(0.87089, abs=0.0)      # the finest value
    assert est["error_fine"] == pytest.approx(0.873 - 0.87089, abs=1e-12)
    assert est["gci_fine"] == pytest.approx(1.25 * est["error_fine"] / 0.87089, rel=1e-12)
    assert "non-monotone" in est["note"]
    assert "spread" in est["note"]


def test_single_grid_and_empty_input_do_not_raise():
    one = richardson({2.0: 0.87})
    assert one["extrapolated"] == 0.87 and one["n_grids"] == 1
    assert one["gci_fine"] == 0.0 and one["p_observed"] is None
    empty = richardson({})
    assert empty["n_grids"] == 0 and empty["extrapolated"] is None
    assert "no usable grids" in empty["note"]
    # non-finite / non-positive entries are ignored, not fatal
    mixed = richardson({4.0: 0.9, 2.0: 0.89, 0.0: 1.0, -1.0: 0.5, 1.0: float("nan")})
    assert mixed["n_grids"] == 2 and mixed["extrapolated"] is not None


# ------------------------------------------------------------------ real 1 h narrow-trench regression
def test_narrow_trench_1h_upwind_and_tvd_agree_with_overlapping_error_bars():
    """Both schemes extrapolate to 0.870 +- 0.006 and their error bars overlap (plan v2 wave-b)."""
    up = richardson(UPWIND_1H)
    tv = richardson(TVD_1H)

    assert abs(up["extrapolated"] - LIMIT_1H) <= 0.006, up
    assert abs(tv["extrapolated"] - LIMIT_1H) <= 0.006, tv

    up_lo, up_hi = up["extrapolated"] - up["error_fine"], up["extrapolated"] + up["error_fine"]
    tv_lo, tv_hi = tv["extrapolated"] - tv["error_fine"], tv["extrapolated"] + tv["error_fine"]
    assert max(up_lo, tv_lo) <= min(up_hi, tv_hi), (up_lo, up_hi, tv_lo, tv_hi)

    # tvd is a clean monotone sequence with a first-to-second-order observed rate
    assert tv["monotone"] is True and tv["p_observed"] is not None
    assert 1.0 < tv["p_observed"] < 3.0, tv
    assert up["monotone"] is False and up["p_observed"] is None
    # both carry a usable relative grid-error column
    assert 0.0 < up["gci_fine"] < 0.01 and 0.0 < tv["gci_fine"] < 0.01


# ------------------------------------------------------------------ table column
def _table() -> list[dict]:
    rows = []
    for scheme, vals in (("upwind", UPWIND_1H), ("tvd", TVD_1H)):
        for dx, ratio in vals.items():
            rows.append({"scheme": scheme, "dx": dx, "ratio": ratio})
    rows.append({"scheme": "tvd", "dx": 8.0, "ratio": None})       # unusable cell, same group
    return rows


def test_grid_error_column_adds_one_estimate_per_group():
    rows = _table()
    out = grid_error_column(rows, group_by=("scheme",))
    assert len(out) == len(rows)
    assert all("grid_err" in r and "extrapolated" in r for r in out)
    assert "grid_err" not in rows[0] and "extrapolated" not in rows[0]      # input untouched

    for scheme, vals in (("upwind", UPWIND_1H), ("tvd", TVD_1H)):
        group = [r for r in out if r["scheme"] == scheme]
        ref = richardson(vals)
        assert len({r["extrapolated"] for r in group}) == 1
        assert len({r["grid_err"] for r in group}) == 1
        assert group[0]["extrapolated"] == pytest.approx(ref["extrapolated"])
        assert group[0]["grid_err"] == pytest.approx(ref["gci_fine"])
    # the None-valued row stays in place but does not change the tvd estimate
    assert out[-1]["grid_err"] is not None


def test_grid_error_column_auto_groups_and_handles_small_groups():
    rows = [
        {"site": "narrow", "dx": 4.0, "ratio": 0.86},
        {"site": "narrow", "dx": 2.0, "ratio": 0.867},
        {"site": "narrow", "dx": 1.0, "ratio": 0.86927},
        {"site": "wide", "dx": 2.0, "ratio": 0.95},       # single grid: no estimate
        {"site": "wide", "dx": 1.0, "ratio": 0.951},
    ]
    out = grid_error_column(rows, key_dx="dx", key_val="ratio")     # group_by=() -> all other keys
    narrow = [r for r in out if r["site"] == "narrow"]
    wide = [r for r in out if r["site"] == "wide"]
    assert narrow[0]["extrapolated"] == pytest.approx(richardson(TVD_1H)["extrapolated"])
    assert all(r["grid_err"] is not None for r in narrow)
    assert all(r["grid_err"] is not None for r in wide)             # two grids: Fs = 3 estimate
    single = grid_error_column([{"site": "x", "dx": 1.0, "ratio": 0.9}])
    assert single[0]["grid_err"] is None and single[0]["extrapolated"] is None
    # re-applying the function to its own output reproduces the same columns (no self-grouping)
    again = grid_error_column(out)
    assert [(r["extrapolated"], r["grid_err"]) for r in again] == \
           [(r["extrapolated"], r["grid_err"]) for r in out]


def test_grid_error_column_rejects_conflicting_duplicate_dx():
    rows = [{"scheme": "a", "dx": 2.0, "ratio": 0.9},
            {"scheme": "a", "dx": 2.0, "ratio": 0.91},
            {"scheme": "a", "dx": 1.0, "ratio": 0.89}]
    with pytest.raises(ValueError, match="conflicting values"):
        grid_error_column(rows, group_by=("scheme",))


# ------------------------------------------------------------------ CLI
def test_cli_prints_estimate_and_json(monkeypatch, capsys):
    text_in = json.dumps({str(k): v for k, v in UPWIND_1H.items()})

    monkeypatch.setattr(sys, "stdin", io.StringIO(text_in))
    assert main(["-"]) == 0
    text = capsys.readouterr().out
    assert "extrapolated = 0.87089" in text
    assert "observed order = -" in text
    assert "GCI(fine)" in text and "non-monotone" in text

    monkeypatch.setattr(sys, "stdin", io.StringIO(text_in))
    assert main(["-", "--json"]) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["extrapolated"] == pytest.approx(0.87089)
    assert est["monotone"] is False

    monkeypatch.setattr(sys, "stdin", io.StringIO("[1, 2, 3]"))
    assert main(["-"]) == 2
    assert main(["does-not-exist.json"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_module_entry_point():
    cp = subprocess.run([sys.executable, "-m", "platesim.feature.grid_error", "-"],
                        input=json.dumps({str(k): v for k, v in TVD_1H.items()}),
                        capture_output=True, text=True, timeout=60)
    assert cp.returncode == 0, cp.stderr
    assert "extrapolated = 0.870359" in cp.stdout
    assert "observed order = 1.62466" in cp.stdout
