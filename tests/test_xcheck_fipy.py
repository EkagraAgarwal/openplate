"""Tests for the FiPy cross-check of the level-set trench model (plan.md Task 4).

Skipped when fipy (or scikit-fmm, which ``DistanceVariable.calcDistanceFunction`` needs) is not
installed.  The full comparison lives in ``platesim/xcheck/fipy_trench.py``; here a short trench is
run through both codes and the deposited thickness and charge balance are compared.
"""
import pytest

pytest.importorskip("fipy")
pytest.importorskip("skfmm")

import numpy as np  # noqa: E402

from platesim.xcheck import fipy_trench as ft  # noqa: E402

# small additive-free trench: 120 um wide, 60 um deep, f_open 0.2 -> R = 300 um
SMALL = dict(d_um=120.0, H_um=60.0, j_mA_cm2=15.0, hours=0.25, delta_um=64.0, dx_um=6.0,
             max_overburden_um=20.0, dt_max=30.0)


@pytest.fixture(scope="module")
def small_runs():
    """(fipy, platesim) results for the same short run; computed once."""
    assert ft._FIPY_OK, ft._FIPY_ERR
    f = ft.run_fipy(**SMALL)
    p = ft.run_platesim(**SMALL)
    return f, p


def test_fipy_inputs_and_units(small_runs):
    f, _ = small_runs
    inp = f["inputs"]
    # platesim's migration-enhanced D, not the bare Cu2+ diffusivity
    assert inp["D_m2_s"] == pytest.approx(4.0834e-10, rel=2e-3)
    assert inp["D_m2_s"] > 1.5 * inp["D_cu_m2_s"]
    assert inp["c_bulk_mol_m3"] == pytest.approx(1022.88, rel=1e-4)
    assert inp["i0_A_m2"] == pytest.approx(20.0, rel=1e-3)
    # I_t = j * R per unit length
    assert f["I_t_A_per_m"] == pytest.approx(150.0 * 300e-6, rel=1e-6)
    assert "moving Dirichlet" in inp["bc_cu2"]


def test_fipy_galvanostat_and_charge(small_runs):
    f, _ = small_runs
    # independent check that the overpotential solves sum(A_if i) = I_t: the residual is re-evaluated
    # here from (c, eta) with the Tafel law, not trusted from the root finder
    assert f["galvanostat_residual_rel"] < 1e-6
    # physical Tafel range at 15 mA/cm2, mildly transport limited
    assert -0.20 < f["eta_mV"] * 1e-3 < -0.06
    # level-set advance agrees with the Faraday charge (best test: < 3 %)
    assert abs(f["charge_rel"]) < 0.03


def test_platesim_charge(small_runs):
    _, p = small_runs
    assert p["stop_reason"] == "recipe_end"
    assert abs(p["charge_rel"]) < 0.03


def test_mean_thickness_agrees_within_5pc(small_runs):
    f, p = small_runs
    rel = abs(f["mean_thickness_um"] / p["mean_thickness_um"] - 1.0)
    assert rel < 0.05, (f["mean_thickness_um"], p["mean_thickness_um"], rel)


def test_field_and_floor_close(small_runs):
    f, p = small_runs
    assert abs(f["field_um"] - p["field_um"]) < 0.1 * p["field_um"]
    assert abs(f["floor_um"] - p["floor_um"]) < 0.1 * p["floor_um"]
    assert 0.0 < f["ratio"] < 1.5


def test_compare_pass_mark():
    f = dict(field_um=82.0, floor_um=75.0, ratio=75.0 / 82.0, mean_thickness_um=80.0)
    p = dict(field_um=82.0, floor_um=74.6, ratio=74.6 / 82.0, mean_thickness_um=80.0)
    out = ft.compare(f, p)
    assert out["passed"] and out["floor_over_field"]["abs_diff"] < 0.02
    p2 = dict(p, ratio=74.6 / 82.0 - 0.05)
    assert not ft.compare(f, p2)["passed"]


def test_clean_json_safe():
    assert ft._clean(dict(a=float("nan"), b=[1.0, float("inf")], c=np.float64(2.0))) == \
        dict(a=None, b=[1.0, None], c=2.0)
