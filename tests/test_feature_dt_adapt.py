"""Adaptive dt_max controller (Params.dt_adapt / dt_max_adapt / dt_tol) of the feature engine time loop.

Check case: narrow trench half-domain, explicit geometry (the measured-site data file is not part of this
public repo).  Contract:
  * dt_adapt=False (the default) is BIT-IDENTICAL to the pre-controller engine: same step count, same
    final floor/field ratio and face thickness (compared against the golden numbers below, ==).
  * dt_adapt=True at the default dt_tol=2e-3 gives >= 30 % fewer steps, the final floor/field ratio within
    0.002 of the default and the final face thickness within 0.5 %.
  * dt_tol trades speed against accuracy; smaller values fall back to the fixed ceiling.
Runs in ~20 s:  python -m pytest tests/test_feature_dt_adapt.py -q
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from platesim.feature.engine.params import Params
from platesim.feature.engine.recipes import pulse_reverse
from platesim.feature.engine.run_case import run_case
from platesim.feature.engine.sim import ViaSim, dt_change_measure

# ---- the check case (see the module docstring) ------------------------------------------------------------
SIZE_UM, T_C, F_OPEN, LABEL = 227.0, 22.0, 0.2, "narrow_half"
# golden numbers of the DEFAULT run, measured on branch plan-v2/wave-A before the controller was added:
GOLDEN_STEPS = 141
GOLDEN_RATIO = 0.7449572510776955           # floor_height_um / face_thickness_um, last series value
GOLDEN_FACE = 85.57524924730443             # face_thickness_um, um, last series value
RATIO_TOL = 2e-3                            # accuracy gate of the check (absolute, on the ratio)
FACE_TOL = 5e-3                             # accuracy gate of the check (relative, on the face thickness)
STEP_FRACTION = 0.7                         # "at least 30 % fewer steps"

_RUNS: dict = {}                            # (dt_adapt, dt_tol) -> (raw, wall_s), one run per config


def case_params(**kw) -> Params:
    """Explicit narrow-trench geometry of the check case, plus the adaptive-dt overrides."""
    args = dict(geometry="trench", suppressor=False, ohmic=True, dx_um=8.0, delta_um=64.0, k_mix=0.0, H_um=88.0)
    args.update(kw)
    return Params(**args)


def case_recipe():
    return pulse_reverse(15.0, 4.5, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)


def run_raw(**kw):
    """(raw result, wall seconds) of one ViaSim run; cached per configuration.

    ViaSim is used directly (run_case() would round the series to 6 decimals before it is returned) because
    the default-path check compares floats bit for bit.  test_run_case_entry_point covers the public wrapper.
    """
    key = (kw.get("dt_adapt", False), kw.get("dt_tol", None))
    if key not in _RUNS:
        sim = ViaSim(SIZE_UM, T_C, case_params(**kw), F_OPEN, case_recipe())
        t0 = time.time()
        _RUNS[key] = (sim.run(LABEL), time.time() - t0)
    return _RUNS[key]


def final_numbers(raw):
    s = raw["series"]
    ratio = float(s["floor_height_um"][-1] / s["face_thickness_um"][-1])
    return float(s["floor_height_um"][-1]), float(s["face_thickness_um"][-1]), ratio


# ------------------------------------------------------------------ unit: the change measure
def test_dt_change_measure_is_relative_and_robust():
    v0 = np.linspace(0.0, 1.0, 101)                     # a flat profile with a known 92nd percentile
    assert dt_change_measure(v0, v0) == 0.0
    v1 = v0 + 0.01
    expected = np.percentile(np.abs(v1 - v0), 92.0) / np.abs(v1).max()
    assert dt_change_measure(v1, v0) == pytest.approx(expected)
    # scale free: doubling the speed level and the change leaves the relative measure alone
    assert dt_change_measure(2 * v1, 2 * v0) == pytest.approx(expected)
    assert dt_change_measure(np.zeros(5), np.zeros(5)) == 0.0          # zero velocity: guarded, not NaN


# ------------------------------------------------------------------ default path: bit identical
def test_default_run_is_bit_identical():
    raw, _ = run_raw()
    floor, face, ratio = final_numbers(raw)
    assert raw["steps"] == GOLDEN_STEPS
    assert ratio == GOLDEN_RATIO                       # exact: the controller must not touch this path
    assert face == GOLDEN_FACE
    assert "dt_max_eff" not in raw["series"]           # no extra key in the default series


def test_adaptive_params_default_off():
    p = Params()
    assert p.dt_adapt is False
    assert p.dt_max_adapt == 900.0 and p.dt_tol == 2e-3


def test_run_case_entry_point_unchanged():
    """The public wrapper still produces the default result and the default series keys."""
    res = run_case(SIZE_UM, T_C, case_recipe(), case_params(), f_open=F_OPEN, label=LABEL, out=None)
    s = res["series"]
    assert "dt_max_eff" not in s
    ratio = s["floor_height_um"][-1] / s["face_thickness_um"][-1]
    assert ratio == pytest.approx(GOLDEN_RATIO, abs=1e-6)     # run_case rounds the series to 6 decimals
    assert res["meta"]["steps"] == GOLDEN_STEPS


# ------------------------------------------------------------------ adaptive: speed vs accuracy
def test_dt_adapt_speedup_and_accuracy():
    base, wall0 = run_raw()
    adapt, wall1 = run_raw(dt_adapt=True, dt_tol=2e-3)
    _, _, ratio0 = final_numbers(base)
    floor1, face1, ratio1 = final_numbers(adapt)
    _, face0, _ = final_numbers(base)
    assert adapt["steps"] <= STEP_FRACTION * GOLDEN_STEPS, (
        f"only {100 * (1 - adapt['steps'] / GOLDEN_STEPS):.1f} % fewer steps "
        f"({adapt['steps']} vs {GOLDEN_STEPS})")
    assert abs(ratio1 - ratio0) <= RATIO_TOL, (ratio1, ratio0)
    assert abs(face1 / face0 - 1.0) <= FACE_TOL, (face1, face0)
    assert adapt["wall_s"] < wall0 and wall1 < wall0
    print(f"\n[dt_adapt] default {GOLDEN_STEPS} steps / {wall0:.2f} s vs adaptive {adapt['steps']} steps / "
          f"{wall1:.2f} s ({100 * (1 - adapt['steps'] / GOLDEN_STEPS):.1f} % fewer), "
          f"ratio {ratio0:.9f} -> {ratio1:.9f} ({ratio1 - ratio0:+.2e}), "
          f"face {face0:.4f} -> {face1:.4f} um ({100 * (face1 / face0 - 1):+.3f} %)")


def test_dt_adapt_records_dt_max_eff():
    adapt, _ = run_raw(dt_adapt=True, dt_tol=2e-3)
    s = adapt["series"]
    p = case_params(dt_adapt=True)
    assert "dt_max_eff" in s
    assert len(s["dt_max_eff"]) == len(s["t_s"])
    eff = np.asarray(s["dt_max_eff"], float)
    assert np.all(eff >= p.dt_max) and np.all(eff <= p.dt_max_adapt)
    assert eff.max() > p.dt_max                       # the ceiling really moved above the fixed value
    dt = np.asarray(s["dt_s"], float)
    assert np.all(dt <= eff * (1.0 + 1e-12) + 1e-12)  # the ceiling is respected by the steps


def test_dt_tol_tradeoff_table():
    """Smaller dt_tol -> more steps, smaller error; at some point the ceiling pins at the fixed dt_max."""
    base, _ = run_raw()
    _, face0, ratio0 = final_numbers(base)
    rows = []
    for tol in (2e-3, 1e-3, 5e-4):
        raw, wall = run_raw(dt_adapt=True, dt_tol=tol)
        _, face, ratio = final_numbers(raw)
        rows.append((tol, raw["steps"], ratio, face, wall))
        assert abs(ratio - ratio0) <= RATIO_TOL, (tol, ratio, ratio0)     # never looser than the gate
    steps = [r[1] for r in rows]
    assert steps == sorted(steps), steps                # a tighter tolerance never saves steps
    assert rows[-1][1] == GOLDEN_STEPS                  # the tightest one falls back to the fixed ceiling
    assert rows[-1][2] == ratio0 and rows[-1][3] == face0
    print("\n[dt_tol]   tol     steps   ratio          d(ratio)    d(face)    wall_s")
    for tol, n, ratio, face, wall in rows:
        print(f"           {tol:<7g} {n:5d}   {ratio:.9f}  {ratio - ratio0:+.2e}  "
              f"{100 * (face / face0 - 1):+.3f}%   {wall:.2f}")
