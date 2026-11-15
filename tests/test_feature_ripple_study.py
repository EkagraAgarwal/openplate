"""Tests for platesim.feature.ripple_study (contour metrics, case list, one 3-minute smoke run).

The whole file targets < 60 s: the only simulation is a 0.05 h narrow trench_full run at dx 8 with
the flow switched off (well under a second on this machine).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from platesim.feature import ripple_study as rs

R_UM, A_UM, H_UM, DX_UM = 1135.0, 113.5, 200.0, 8.0
FLOOR_UM, FIELD_UM, BUMP_UM, BUMP_X = 50.0, 80.0, 10.0, 84.0


def _synthetic_result() -> dict:
    """Hand-built trench_full contour: floor at 50 um, field at H + 80 um, one 10 um bump on the field.

    H = 200 um keeps z = H/2 above the floor, so the mid-depth wall cut is exercised as well."""
    xc = 0.5 * R_UM
    z_field = H_UM + FIELD_UM
    xs = np.arange(0.0, xc - A_UM + 1e-9, 1.0)
    up = z_field + BUMP_UM * np.exp(-0.5 * ((xs - BUMP_X) / 10.0) ** 2)
    contour = [
        [[xc - A_UM, FLOOR_UM], [xc + A_UM, FLOOR_UM]],          # trench floor
        [[xc - A_UM, FLOOR_UM], [xc - A_UM, H_UM]],              # left wall (vertical, no growth)
        [[xc + A_UM, FLOOR_UM], [xc + A_UM, H_UM]],              # right wall
        np.column_stack([xs, up]).tolist(),                      # left field, bump at x = 84 um
        [[xc + A_UM, z_field], [R_UM, z_field]],                 # right field, flat
    ]
    return dict(meta=dict(grid=dict(R_um=R_UM, a_um=A_UM, H_um=H_UM, dx_um=DX_UM), steps=123, wall_s=4.5),
                series=dict(face_thickness_um=[70.0, 80.0], floor_height_um=[48.0, 50.0]),
                frames=[dict(contour=contour, t_s=16200.0)], result=dict(cu_balance_rel=1e-3))


def test_surface_stats_of_a_known_trench():
    s = rs.surface_stats(_synthetic_result())
    assert s["field_mean_um"] == pytest.approx(FIELD_UM, abs=0.5)
    assert s["field_min_um"] == pytest.approx(FIELD_UM, abs=0.5)
    assert s["field_max_um"] == pytest.approx(FIELD_UM + BUMP_UM, abs=0.5)   # the bump is found
    assert s["field_std_um"] > 1.0
    assert s["shoulder_max_um"] == pytest.approx(FIELD_UM, abs=0.5)
    assert s["floor_mean_um"] == pytest.approx(FLOOR_UM, abs=0.5)
    assert s["floor_min_um"] == pytest.approx(FLOOR_UM, abs=0.5)
    assert s["floor_max_um"] == pytest.approx(FLOOR_UM, abs=0.5)
    assert s["wall_up_um"] == pytest.approx(0.0, abs=0.5)                    # wall has not moved
    assert s["wall_down_um"] == pytest.approx(0.0, abs=0.5)
    assert s["ratio_mean"] == pytest.approx(FLOOR_UM / s["field_mean_um"], rel=1e-6)
    # the copies of the engine series / result
    assert s["face_thickness_um"] == pytest.approx(80.0)
    assert s["floor_height_um"] == pytest.approx(50.0)
    assert s["ratio_legacy"] == pytest.approx(50.0 / 80.0)
    assert s["cu_balance_rel"] == pytest.approx(1e-3)
    assert s["cu_balance_adv_rel"] is None                                   # absent -> None
    assert s["wall_s"] == pytest.approx(4.5) and s["steps"] == 123
    assert math.isnan(s["ripple_wavelength_um"]) or s["ripple_wavelength_um"] > 0.0


def test_surface_profile_is_resampled_on_the_column_centres():
    stats, profile = rs._surface_and_stats(_synthetic_result())
    xs = np.asarray(profile["x_um"])
    assert xs.size == int(round(R_UM / DX_UM))
    assert xs[0] == pytest.approx(0.5 * DX_UM)
    assert np.allclose(np.diff(xs), DX_UM)
    assert max(v for v in profile["upper_um"] if v is not None) >= H_UM + FIELD_UM


# ------------------------------------------------------- synthetic contour metrics (no engine run)
def _contour_result(contour: list) -> dict:
    """A run_case-shaped dict carrying only what the contour metrics read (no engine, no data file)."""
    return dict(meta=dict(grid=dict(R_um=R_UM, a_um=A_UM, H_um=H_UM, dx_um=DX_UM)),
                series={}, frames=[dict(contour=contour, t_s=16200.0)], result={})


def _trench_contour(wall_um: float = 0.0, field=None) -> list:
    """Synthetic trench_full contour: floor at FLOOR_UM, vertical walls at x_c -+ (A_UM - wall_um).

    `field(x)` is the field thickness above H (default 80 um flat), so the field surface is at
    z = H_UM + field(x).  The walls are vertical polylines, exactly as the engine's psi = 0 lines
    resolve a wall that has grown inward."""
    xc = 0.5 * R_UM
    xl, xr = xc - A_UM + wall_um, xc + A_UM - wall_um

    def thick(v):
        v = np.asarray(v, float)
        return np.full_like(v, FIELD_UM) if field is None else np.asarray(field(v), float)

    fx = np.arange(0.0, xl + 1e-9, 1.0)
    rx = np.arange(xr, R_UM + 1e-9, 1.0)
    return [
        [[xl, FLOOR_UM], [xr, FLOOR_UM]],                        # trench floor
        [[xl, FLOOR_UM], [xl, H_UM]],                            # left wall (vertical: never flat)
        [[xr, FLOOR_UM], [xr, H_UM]],                            # right wall
        np.column_stack([fx, H_UM + thick(fx)]).tolist(),        # left field, wraps onto the next pitch
        np.column_stack([rx, H_UM + thick(rx)]).tolist(),        # right field
    ]


def test_wall_thickness_reads_walls_moved_inward_by_7_um():
    s = rs.surface_stats(_contour_result(_trench_contour(wall_um=7.0)))
    assert s["wall_up_um"] == pytest.approx(7.0, abs=0.5 * DX_UM)
    assert s["wall_down_um"] == pytest.approx(7.0, abs=0.5 * DX_UM)


def test_wall_thickness_of_an_unmoved_wall_is_zero_not_nan():
    s = rs.surface_stats(_contour_result(_trench_contour()))
    assert s["wall_up_um"] == pytest.approx(0.0, abs=1e-9)
    assert s["wall_down_um"] == pytest.approx(0.0, abs=1e-9)


def test_wall_thickness_ignores_a_contour_piece_lying_on_the_cut_line():
    """A horizontal psi = 0 piece at z = H/2 must not report its midpoint as a wall position."""
    xc = 0.5 * R_UM
    contour = _trench_contour()
    contour.append([[xc - A_UM, 0.5 * H_UM], [xc + A_UM, 0.5 * H_UM]])
    s = rs.surface_stats(_contour_result(contour))
    assert s["wall_up_um"] == pytest.approx(0.0, abs=1e-9)
    assert s["wall_down_um"] == pytest.approx(0.0, abs=1e-9)


def test_ripple_wavelength_of_a_sine_field_profile():
    lam = 50.0

    def field(x):
        return FIELD_UM + 3.0 * np.sin(2.0 * np.pi * x / lam)     # z = H + 80 + 3 sin(2 pi x / 50)

    s = rs.surface_stats(_contour_result(_trench_contour(field=field)))
    assert math.isfinite(s["ripple_wavelength_um"])
    assert s["ripple_wavelength_um"] == pytest.approx(lam, rel=0.10)


def test_ripple_of_a_flat_field_is_nan():
    s = rs.surface_stats(_contour_result(_trench_contour()))
    assert math.isnan(s["ripple_wavelength_um"])


# ------------------------------------------------------------------ case list and CLI
def test_build_cases_names_are_unique_and_cover_every_group():
    cases = rs.build_cases(False)
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names))
    assert {c["group"] for c in cases} == set(rs.GROUPS)
    assert {"base_narrow", "base_wide", "legacy_narrow", "reverse_narrow",
            "tertiary_narrow_noflow", "flat_dx4"} <= set(names)
    # equal charge for the current sweep and a filesystem-safe name for every case
    for c in cases:
        assert c["name"] == "".join(ch for ch in c["name"] if ch.isalnum() or ch in "_-")
        if c["group"] == "current":
            assert c["hours"] * c["j"] == pytest.approx(rs.HOURS * rs.J_MA)
        if c["group"] == "legacy":
            assert c["params"]["adv_form"] == "divergence"


def test_quick_cases_are_the_narrow_dx8_subset():
    full, quick = rs.build_cases(False), rs.build_cases(True)
    assert quick and {c["name"] for c in quick} < {c["name"] for c in full}
    for c in quick:
        assert c["site"] == "narrow"
        assert float(c["params"].get("dx_um", rs.DX_UM)) == 8.0
        if c["group"] == "current":                       # equal charge -> longer than QUICK_HOURS
            assert c["hours"] * c["j"] == pytest.approx(rs.QUICK_HOURS * rs.J_MA)
        else:
            assert c["hours"] == rs.QUICK_HOURS


def test_list_prints_the_case_names(capsys):
    cases = rs.main(["--list"])
    text = capsys.readouterr().out.strip().splitlines()
    assert len(text) == len(cases)
    assert "shear_narrow_G3" in " ".join(text)


def test_summary_table_has_one_row_per_case():
    res = {
        "a": dict(case=dict(name="a", group="g2"), error=None,
                  stats=dict(field_mean_um=80.0, field_min_um=79.0, field_max_um=81.0, floor_mean_um=50.0,
                             ratio_mean=0.625, shoulder_max_um=80.5, cu_balance_rel=1e-3,
                             cu_balance_adv_rel=None, wall_s=3.0, ripple_wavelength_um=float("nan"))),
        "b": dict(case=dict(name="b", group="g1"), stats=None, error="ValueError: boom"),
    }
    lines = rs.summary_table(res).splitlines()
    assert lines[0].startswith("| case | group |") and len(lines) == 4
    assert "b" in lines[2] and "ERROR" in lines[2]        # sorted by group: g1 (error) first
    assert "a" in lines[3] and "50.00" in lines[3]


# ------------------------------------------------------------------ one real short run
def test_smoke_run_one_returns_a_finite_field():
    case = dict(name="smoke_narrow", site="narrow", group="base", hours=0.05, j=15.0, f_open=0.2,
                params={"dx_um": 8.0, "flow_shear_s": 0.0})
    r = rs.run_one(case)
    assert r["error"] is None, r["error"]
    s = r["stats"]
    assert math.isfinite(s["field_mean_um"]) and 0.0 < s["field_mean_um"] < 20.0
    assert math.isfinite(s["floor_mean_um"])
    assert s["field_min_um"] <= s["field_mean_um"] <= s["field_max_um"]
    assert s["cu_balance_rel"] is not None and abs(s["cu_balance_rel"]) < 0.05
    assert len(r["profile"]["x_um"]) > 50 and r["profile"]["x_um"][0] > 0.0
