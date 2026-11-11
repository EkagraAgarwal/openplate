"""Tests for platesim.trenchflow.model (coupled trench growth model, galvanostatic k solve).

Small cases only (dx = 8 um, 2 stages) so the module runs in a few seconds.
"""
import numpy as np
import pytest

from platesim.feature.engine.params import Params, derive
from platesim.trenchflow.model import (model_constants, predict_site, shear_from_delta, solve_k, stage)
from platesim.trenchflow.geometry import trench_grid
from platesim.trenchflow.transport import region_means

P = Params()
PH = derive(P, 22.0)
D = float(PH.D_c)

SMALL = dict(width_um=160.0, depth_um=80.0, delta_um=80.0, shear_rate_s=50.0, j_field_mA_cm2=15.0,
             deposit_um=0.0, dx_um=8.0)


def _delta_from_gamma(G, D_, L=2.0):
    """Inverse of shear_from_delta: delta_um for a given G (same Leveque mean relation)."""
    k_mean = 1.5 * 0.538 * D_ ** (2.0 / 3.0) * G ** (1.0 / 3.0) * L ** (-1.0 / 3.0)
    return D_ / k_mean * 1e4


def test_shear_from_delta_round_trip():
    """delta -> G -> delta is exact (same formula), and G decreases with delta."""
    for delta in (30.0, 45.0, 64.0, 90.0, 140.0):
        G = shear_from_delta(delta, D)
        back = _delta_from_gamma(G, D)
        assert abs(back - delta) / delta < 1e-9, (delta, G, back)
    g = [shear_from_delta(d, D) for d in (30.0, 64.0, 120.0)]
    assert g[0] > g[1] > g[2]


def test_solve_k_hits_target_field_flux():
    """solve_k must reproduce the requested field flux density within 0.5 %."""
    grid = trench_grid(160.0, 80.0, height_um=80.0, dx_um=8.0)
    target = 7.6e-5                                   # cm/s, ~15 mA/cm2 at c_bulk
    k, res = solve_k(grid, None, D, target)
    got = region_means(res, grid)["field"]
    assert abs(got - target) / target < 5e-3, (got, target)
    assert 1e-7 <= k <= 1e2


def test_stage_ratios_small_trench():
    """floor_over_field <= 1.02 in a small trench; flow must not lower it below the no-flow value."""
    flow = stage(**SMALL, use_flow=True)
    noflow = stage(**SMALL, use_flow=False)
    for out in (flow, noflow):
        assert 0.0 < out["floor_over_field"] <= 1.02, out["floor_over_field"]
        assert out["field_flux_cm_s"] > 0.0
        assert out["i_lim_field_mA_cm2"] > SMALL["j_field_mA_cm2"]
        assert len(out["grid_shape"]) == 2
    assert flow["floor_over_field"] >= noflow["floor_over_field"] - 0.02, (flow, noflow)
    assert flow["eddy"] is not None and "penetration_um" in flow["eddy"]
    assert noflow["eddy"] is None
    assert np.isfinite(flow["wall_up_over_field"]) and np.isfinite(flow["wall_down_over_field"])


def test_above_limit_raises():
    """j_field above 2 F c_bulk D / delta must raise with the limit in the message."""
    with pytest.raises(ValueError) as exc:
        stage(**{**SMALL, "j_field_mA_cm2": 1e4})
    assert "i_lim" in str(exc.value) or "limit" in str(exc.value)


def test_predict_site_keys_and_bound():
    """predict_site on a synthetic site returns the documented keys and floor <= field * 1.02."""
    site = {"id": "synth", "width_um": 160.0, "depth_um": 80.0, "T_C": 22.0, "hours": 0.5,
            "field_um": {"lo": 7.0, "hi": 9.0, "best": 8.0}, "floor_um": {"lo": 6.0, "hi": 7.0, "best": 6.5}}
    out = predict_site(site, 64.0, n_stages=2, dx_um=8.0)
    keys = {"id", "field_um", "j_field_mA_cm2", "delta_um", "shear_rate_s", "floor_um", "wall_up_um",
            "wall_down_um", "floor_over_field", "wall_asymmetry", "stages", "use_flow"}
    assert keys <= set(out)
    assert out["field_um"] == 8.0
    assert out["floor_um"] <= out["field_um"] * 1.02, (out["floor_um"], out["field_um"])
    assert len(out["stages"]) == 2
    for st in out["stages"]:
        assert "t_field_um" in st and "deposit_um" in st
    assert abs(out["floor_over_field"] - out["floor_um"] / out["field_um"]) < 1e-12
    # j_field from the measured field thickness: field_um*1e-4 / (Mfac * hours * 3600)
    _, _, mfac = model_constants(22.0)
    assert abs(out["j_field_mA_cm2"] - 8.0e-4 / (mfac * 0.5 * 3600.0) * 1e3) < 1e-9
