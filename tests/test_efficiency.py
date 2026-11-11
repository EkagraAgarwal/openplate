"""Tests for platesim.efficiency (planar Cu current-efficiency loss model + calibration).

Run: python -m pytest tests/test_efficiency.py -q   (< 60 s)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from platesim.core.constants import thickness_um
from platesim.efficiency import MODEL_SPEC, run as model_run
from platesim.efficiency.calibrate import (
    calibrate, fit_map, predict_pending_runs, sample_mcmc, samples_summary,
    true_efficiency_curve,
)
from platesim.efficiency.model import (
    HYPOTHESES, calibration_runs, default_theta, predict,
)

RUN12 = {"j_mA_cm2": 12.0, "t_min": 60.0, "stir_pct": 20.0, "setup": "base"}


def _no_loss_theta():
    """All loss switches off -> pure Faraday bookkeeping."""
    th = default_theta("H4")
    th.update({"use_o2": False, "use_h2": False, "use_stray": False,
               "use_cc": False, "use_area": False, "use_l0": False})
    return th


# ---------------------------------------------------------------------------
# 1. Faraday identity
# ---------------------------------------------------------------------------
def test_faraday_identity_no_losses():
    for j, t_min in [(6.0, 120.0), (12.0, 30.0), (24.0, 60.0)]:
        run = {"j_mA_cm2": j, "t_min": t_min, "stir_pct": 20.0, "setup": "base"}
        p = predict(run, _no_loss_theta())
        assert p["thickness_um"] == pytest.approx(thickness_um(j, t_min * 60.0), rel=1e-9)
        assert p["apparent_eff"] == pytest.approx(1.0, rel=1e-9)
        # loss bars must sum to zero
        b = p["breakdown"]
        total_loss = (b["loss_stray_um"] + b["loss_cc_um"] + b["loss_o2_um"]
                      + b["loss_h2_um"] + b["loss_l0_um"] + b["loss_area_um"])
        assert abs(total_loss) < 1e-9


# ---------------------------------------------------------------------------
# 2. Each loss term reduces thickness, monotonically in its magnitude
# ---------------------------------------------------------------------------
def _thickness(**overrides):
    th = _no_loss_theta()
    th.update(overrides)
    return predict(RUN12, th)["thickness_um"]


def test_faraday_reference_is_largest():
    base = _thickness()
    assert base == pytest.approx(thickness_um(12.0, 3600.0), rel=1e-9)


def test_loss_terms_reduce_thickness():
    cases = {
        "o2":     {"use_o2": True, "delta0_um": 80.0},                       # ~0.16 mA/cm2 parasitic
        "h2":     {"use_h2": True, "delta0_um": 5000.0},                   # huge delta -> j_lim < j -> H2 on
        "stray":  {"use_stray": True, "f_stray_base": 0.30},
        "cc":     {"use_cc": True, "V_max": 0.5},                            # supply sags hard
        "area":   {"use_area": True, "area_base": 1.30},                     # charge diluted over A_true
        "l0":     {"use_l0": True, "L0_um": 1.0},
    }
    ref = _thickness()
    for name, ov in cases.items():
        t = _thickness(**ov)
        assert t < ref - 1e-6, f"{name} did not reduce thickness ({t} vs {ref})"


def test_loss_monotone_in_magnitude():
    # stray fraction
    assert _thickness(use_stray=True, f_stray_base=0.30) < _thickness(use_stray=True, f_stray_base=0.10)
    # startup loss
    assert _thickness(use_l0=True, L0_um=1.5) < _thickness(use_l0=True, L0_um=0.5)
    # compliance voltage (lower V_max -> more CV sag)
    assert _thickness(use_cc=True, V_max=0.5) < _thickness(use_cc=True, V_max=2.0)
    # area ratio
    assert _thickness(use_area=True, area_base=1.4) < _thickness(use_area=True, area_base=1.1)
    # O2: thinner diffusion layer -> more parasitic flux -> less thickness
    assert _thickness(use_o2=True, delta0_um=30.0) < _thickness(use_o2=True, delta0_um=120.0)
    # breakdown losses sum to (faraday - thickness)
    th = _no_loss_theta()
    th.update({"use_o2": True, "use_stray": True, "use_cc": True, "use_area": True,
               "use_l0": True, "f_stray_base": 0.3, "area_base": 1.25, "V_max": 0.8,
               "L0_um": 0.8, "delta0_um": 80.0})
    p = predict(RUN12, th)
    b = p["breakdown"]
    total = (b["loss_stray_um"] + b["loss_cc_um"] + b["loss_o2_um"] + b["loss_h2_um"]
             + b["loss_l0_um"] + b["loss_area_um"])
    assert total == pytest.approx(b["faraday_um"] - p["thickness_um"], abs=1e-6)


# ---------------------------------------------------------------------------
# 3. MAP fit of H4 reproduces runs 1-8 within 1.0 um RMS
# ---------------------------------------------------------------------------
def test_map_h4_rms_under_1um():
    res = fit_map("H4", n_starts=3, max_nfev=300)
    runs = calibration_runs()
    resid = [r["actual_um"] - predict(r, res["theta_map"], hypothesis="H4")["thickness_um"]
             for r in runs]
    rms = math.sqrt(np.mean(np.square(resid)))
    assert rms < 1.0, f"H4 MAP rms {rms:.3f} um >= 1.0"
    # calibrate() wrapper also agrees and caches
    out = calibrate("H4", method="map", force=True, n_starts=3)
    assert out["rms_um"] == pytest.approx(rms, abs=0.05)
    assert out["n_params"] == len(HYPOTHESES["H4"])
    assert (Path(__file__).resolve().parents[1] / "platesim" / "efficiency"
            / "calibration_results.json").exists()


# ---------------------------------------------------------------------------
# 4. MCMC smoke test (few steps) - emcee if present, else Metropolis fallback
# ---------------------------------------------------------------------------
def test_mcmc_smoke():
    m = sample_mcmc("H1", n_steps=60, seed=1)
    assert m["backend"] in ("emcee", "metropolis")
    assert m["n_samples"] > 10
    s = samples_summary(m["samples"])
    for name in HYPOTHESES["H1"]:
        assert np.isfinite(s[name]["median"])
        assert s[name]["q16"] <= s[name]["median"] <= s[name]["q84"]
    out = calibrate("H1", method="mcmc", n_steps=60, force=True, n_starts=2)
    assert out["samples_summary"] is not None
    assert np.isfinite(out["loglik"]) and np.isfinite(out["bic"])


# ---------------------------------------------------------------------------
# 5. Module contract: MODEL_SPEC, run() predict + calibrate, pending predictions
# ---------------------------------------------------------------------------
def test_model_spec_shape():
    assert MODEL_SPEC["id"] == "planar_efficiency"
    names = {p["name"] for p in MODEL_SPEC["params"]}
    assert {"mode", "hypothesis", "j_mA_cm2", "t_min", "stir_pct", "setup", "T_C", "method"} <= names


def test_run_predict_mode():
    out = model_run({"mode": "predict", "hypothesis": "H1", "j_mA_cm2": 12.0,
                     "t_min": 60.0, "stir_pct": 30.0, "setup": "imp1", "T_C": 22.0})
    assert "thickness_um" in out["summary"]
    kinds = {p["kind"] for p in out["plots"]}
    assert kinds <= {"line", "bar", "heatmap", "contours"}
    bar = [p for p in out["plots"] if p["kind"] == "bar"][0]
    assert len(bar["categories"]) == len(bar["series"][0]["y"]) == 6


def test_run_calibrate_mode_and_pending_predictions():
    out = model_run({"mode": "calibrate", "hypothesis": "H4", "method": "map"})
    comp = out["summary"]["comparison_table"]
    assert {c["hypothesis"] for c in comp} == {"H1", "H2", "H3", "H4"}
    assert out["summary"]["run9_um"] is not None
    pending = predict_pending_runs(hypothesis="H4")
    assert len(pending) == 2 and all(p["lo_um"] <= p["predicted_um"] <= p["hi_um"] for p in pending)
    curves = true_efficiency_curve(hypothesis="H4")
    assert set(curves["curves"]) == {"22C", "40C"}
    for ys in curves["curves"].values():
        assert all(0.0 <= y <= 1.0 for y in ys)
    # taped true efficiency should exceed the team's apparent 46-69 % readings
    assert max(curves["curves"]["22C"]) > 0.7
