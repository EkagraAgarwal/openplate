"""Engine hooks batch 1: level-set scheme switches, the C5 bulk-plane rule and D6 Gibbs-Thomson.

Wiring tests for plan v2 items B4/C3 (WENO5 + narrow band + Faraday volume correction), C5 (bulk-plane
rule) and D6 (capillarity), all behind Params switches whose DEFAULTS are bit-identical to the pre-hook
engine.  The exact bit-identity is verified by the golden harness
(output/plan_v2_runs/scripts/golden.py + golden_diff.py), not by this file.

Every number below is a 0.5 h narrow trench_full run (d = 227 um, H = 88 um, f_open = 0.2, dx 8 um,
15 mA/cm2 pulse-reverse, no suppressor, no shear), so the module runs in ~10 s.  "floor/field" is the last
series pair floor_mean_um / field_mean_um: two FATIGUE-FREE front measurements of the level set (the
central 40 % of the trench floor and the field far from both walls), NOT the delta plane, so the
bulk_plane switch cannot flatter the comparison.

What this file does NOT show: that WENO5 is more accurate than Godunov1.  Both schemes are within
0.01 of each other here because a 0.5 h run and this smooth geometry do not discriminate them; the
accuracy claim belongs to tests/test_levelset_kernels.py (circle/square convergence).  Likewise the
volume-corrected Faraday error is ~0 BY CONSTRUCTION and is printed only as a wiring check.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine.fields import gibbs_thomson_eta
from platesim.feature.engine.levelset import Grid, column_crossing
from platesim.feature.engine.params import (BULK_PLANES, F, OMEGA_CU_M3_MOL, Params, derive)
from platesim.feature.engine.recipes import pulse_reverse
from platesim.feature.engine.sim import ViaSim

D_UM, H_UM, F_OPEN, DX_UM, DELTA_UM, J_NET = 227.0, 88.0, 0.2, 8.0, 64.0, 15.0
HOURS = 0.5

BASE = dict(geometry="trench_full", suppressor=False, ohmic=True, dx_um=DX_UM, delta_um=DELTA_UM,
            k_mix=0.0, max_overburden_um=140.0, H_um=H_UM, flow_shear_s=0.0)

# every switch of this batch, one short run each
SWITCHES = dict(
    weno5=dict(ls_scheme="weno5"),
    narrow_band=dict(ls_scheme="weno5", ls_narrow_band=True),
    volume_correct=dict(ls_scheme="weno5", ls_volume_correct=True),
    bulk_field_mean=dict(bulk_plane="field_mean"),
    bulk_fixed=dict(bulk_plane="fixed"),
    gibbs_thomson=dict(gibbs_thomson=True),
)

_RUNS: dict[str, dict] = {}


def _run(name, hours=HOURS, **kw):
    """One cached pulse-reverse trench_full run; floor/field are the last series samples."""
    if name in _RUNS:
        return _RUNS[name]
    p = Params(**{**BASE, **kw})
    rec = pulse_reverse(J_NET, hours, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)
    sim = ViaSim(D_UM, 22.0, p, F_OPEN, rec)
    raw = sim.run(name)
    s = raw["series"]
    floor, field = float(s["floor_mean_um"][-1]), float(s["field_mean_um"][-1])
    out = dict(raw=raw, params=p, sim=sim, floor=floor, field=field, ratio=floor / field,
               cu=raw["cu_balance_rel"], far=raw["faraday_error_rel"])
    _RUNS[name] = out
    return out


def _fmt(name, r):
    return (f"{name:16s} floor/field={r['ratio']:.6f} (floor {r['floor']:.4f} um, field {r['field']:.4f} um) "
            f"cu_balance_rel={r['cu']:+.2e} faraday_error_rel={r['far']:+.2e} "
            f"steps={r['raw']['steps']} stop={r['raw']['stop_reason']}")


# ------------------------------------------------------------------ 1. defaults and switch validation
def test_new_params_default_to_the_legacy_behaviour():
    p = Params()
    assert p.ls_scheme == "godunov1"
    assert p.ls_narrow_band is False and p.ls_volume_correct is False
    assert p.bulk_plane == "edge"
    assert p.gibbs_thomson is False and p.gamma_cu_J_m2 == pytest.approx(1.8)
    derive(p, 22.0)                                    # defaults validate


def test_bulk_plane_and_gamma_are_validated():
    with pytest.raises(ValueError, match="bulk_plane"):
        derive(Params(bulk_plane="mean"), 22.0)
    with pytest.raises(ValueError, match="gamma_cu_J_m2"):
        derive(Params(gamma_cu_J_m2=-1.0), 22.0)
    for opt in BULK_PLANES:
        derive(Params(bulk_plane=opt), 22.0)


# ------------------------------------------------------------------ 2. every switch runs and balances
def test_each_switch_runs_and_closes_the_cu_balance():
    """Every switch on, 0.5 h: the run completes and |cu_balance_rel| stays below the 0.5 % gate."""
    rows = [("godunov1", _run("godunov1"))]
    rows += [(name, _run(name, **kw)) for name, kw in SWITCHES.items()]
    for name, r in rows:
        print(_fmt(name, r))
        assert r["raw"]["stop_reason"] == "recipe_end"
        assert r["raw"]["steps"] > 5
        assert r["cu"] is not None and abs(r["cu"]) < 5e-3
        assert np.isfinite(r["floor"]) and np.isfinite(r["field"]) and r["field"] > 0.0


# ------------------------------------------------------------------ 3. level-set schemes (B4 / C3)
def test_weno5_and_narrow_band_match_godunov1_floor_over_field():
    """WENO5 (full grid and narrow band) must agree with the legacy Godunov1 run to 0.01 in floor/field,
    and the narrow band must reproduce the full-grid WENO5 result (same result, fewer updated cells)."""
    base = _run("godunov1")
    weno = _run("weno5", **SWITCHES["weno5"])
    band = _run("narrow_band", **SWITCHES["narrow_band"])
    for name, r in (("godunov1", base), ("weno5", weno), ("narrow_band", band)):
        print(_fmt(name, r))
    print(f"  |weno5 - godunov1|       = {abs(weno['ratio'] - base['ratio']):.2e}")
    print(f"  |narrow_band - godunov1| = {abs(band['ratio'] - base['ratio']):.2e}")
    assert abs(weno["ratio"] - base["ratio"]) < 1e-2
    assert abs(band["ratio"] - base["ratio"]) < 1e-2
    assert abs(band["ratio"] - weno["ratio"]) < 1e-2


def test_faraday_error_per_scheme():
    """Faraday check of the level set (deposit volume vs q*Mfac) for godunov1, weno5 and weno5 with
    ls_volume_correct.  The volume-corrected number is ~0 BY CONSTRUCTION (it is imposed every step), so it
    is printed as a wiring check only - it says nothing about which scheme is more accurate."""
    god = _run("godunov1")
    weno = _run("weno5", **SWITCHES["weno5"])
    vc = _run("volume_correct", **SWITCHES["volume_correct"])
    for name, r in (("godunov1", god), ("weno5", weno), ("weno5+volume_correct", vc)):
        print(f"  faraday_error_rel {name:22s} = {r['far']:+.3e}")
    # the raw dict reports the same convention run_case.meta['faraday'] uses (q_total * Mfac)
    ph = derive(weno["params"], 22.0)
    raw = weno["raw"]
    assert raw["faraday_volume"] == pytest.approx(raw["q_total"] * ph.Mfac)
    assert raw["faraday_error_rel"] == pytest.approx(raw["solid_gain"] / raw["faraday_volume"] - 1.0)
    # volume correction holds the deposit at the Faraday volume; uncorrected WENO5 drifts by 0.2-0.8 %
    assert abs(vc["far"]) < 1e-6
    assert abs(vc["far"]) < abs(weno["far"])
    assert abs(god["far"]) < 1e-2 and abs(weno["far"]) < 1e-2


# ------------------------------------------------------------------ 4. bulk-plane rule (C5)
def test_bulk_plane_rules_stay_within_2_percent_of_the_legacy_plane():
    """'field_mean' (all field columns) and 'fixed' (initial plane) must not move floor/field by more than
    0.02 relative to the legacy 'edge' rule (two edge columns) on a 0.5 h run."""
    edge = _run("godunov1")
    fm = _run("bulk_field_mean", **SWITCHES["bulk_field_mean"])
    fx = _run("bulk_fixed", **SWITCHES["bulk_fixed"])
    for name, r in (("edge (default)", edge), ("field_mean", fm), ("fixed", fx)):
        print(_fmt(name, r))
    print(f"  |field_mean - edge| = {abs(fm['ratio'] - edge['ratio']):.2e}")
    print(f"  |fixed - edge|      = {abs(fx['ratio'] - edge['ratio']):.2e}")
    assert abs(fm["ratio"] - edge["ratio"]) < 2e-2
    assert abs(fx["ratio"] - edge["ratio"]) < 2e-2
    # 'fixed' really freezes the plane at the initial height while the front moves on; 'edge' follows it
    assert fx["sim"].z_face0 is not None
    assert fx["sim"].face_height() == pytest.approx(fx["sim"].z_face0)
    assert edge["sim"].z_face0 is None
    assert fx["sim"].face_height() < edge["sim"].face_height()
    # and the legacy rule is still exactly the mean of the two edge columns
    g, psi = edge["sim"].g, edge["sim"].psi
    zs = [column_crossing(psi[i], g.zc[0], upper=True)[1] for i in (0, g.nr - 1)]
    assert edge["sim"].face_height() == pytest.approx(float(np.mean([z for z in zs if z is not None])))


# ------------------------------------------------------------------ 5. Gibbs-Thomson (D6)
def test_gibbs_thomson_shift_matches_the_analytic_circle():
    """eta_GT on a metal circle of radius R must be -gamma Omega / (2 F R) (kappa = +1/R for the level-set
    sign convention: psi > 0 in the electrolyte, so curvature_div_n is POSITIVE on convex metal)."""
    g = Grid(2.0, 10.0, 0.5, 1.0, 0.1, 1.0, geometry="trench_full")     # fine grid, 0.1 um cells
    for R_um in (0.6, 0.8, 1.0):                                        # 6-10 cells, no clipping
        R_cm = R_um * 1e-4
        psi = np.hypot(g.rc - g.x_c, g.zc - 5.0e-4) - R_cm        # centre at 5 um above the wafer
        eta = gibbs_thomson_eta(psi, g, 1.8)
        band = np.abs(psi) < g.dx
        analytic = -1.8 * OMEGA_CU_M3_MOL / (2.0 * F * R_cm * 1e-2)     # R in cm -> m
        assert band.sum() > 10
        print(f"  R = {R_um:.1f} um: eta_GT = {eta[band].mean():+.4e} V, analytic = {analytic:+.4e} V "
              f"(rel {eta[band].mean() / analytic - 1:+.3f})")
        assert eta[band].mean() == pytest.approx(analytic, rel=0.02)
        assert eta[band].mean() < 0.0          # convex metal -> eta_eff shifted toward zero (harder to plate)
    # the |kappa| <= 1/(2 dx) clip really limits the shift on a grid-scale feature
    psi = np.hypot(g.rc - g.x_c, g.zc - 5.0e-4) - 0.2 * g.dx
    cap = 1.8 * OMEGA_CU_M3_MOL * (1.0 / (2.0 * g.dx)) * 100.0 / (2.0 * F)
    assert np.abs(gibbs_thomson_eta(psi, g, 1.8)).max() == pytest.approx(cap)


def test_gibbs_thomson_is_negligible_at_um_scale():
    """On a um-scale trench the capillarity shift must not move floor/field by more than 1e-3 (gamma Omega /
    (2 F R) is tens of uV at R ~ um, i.e. far below the mV overpotentials of the run)."""
    base = _run("godunov1")
    gt = _run("gibbs_thomson", **SWITCHES["gibbs_thomson"])
    g = gt["sim"].g
    snap = gibbs_thomson_eta(gt["sim"].psi, g, 1.8)
    amp = float(np.abs(snap[np.abs(gt["sim"].psi) < g.dx]).max())
    print(_fmt("godunov1", base))
    print(_fmt("gibbs_thomson", gt))
    print(f"  max |eta_GT| on the interface = {amp:.3e} V;  |ratio diff| = "
          f"{abs(gt['ratio'] - base['ratio']):.2e}")
    assert abs(gt["ratio"] - base["ratio"]) < 1e-3
    assert amp < 1e-4                                     # sub-0.1 mV at dx 8 (clipped curvature included)
