"""Validation tests for platesim.cell (3-D beaker current distribution).

Run: python -m pytest tests/test_cell.py -q   (must finish < 60 s)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from platesim.cell import MODEL_SPEC, march_cell, run, solve_cell
from platesim.cell.solver import solve_duct
from platesim.core.bath import Bath
from platesim.core.cell import CellGeometry

I_TEST = 0.048  # A -> 12 mA/cm2 on the 2x2 cm coupon


def _fast_bath(**kw) -> Bath:
    """Bath with effectively-infinite exchange current (primary / ohmic limits)."""
    return replace(Bath(), i0_cu_22=1e2, i0_cu_supp_22=1e2, **kw)


@pytest.fixture(scope="module")
def beaker_default():
    """One gmsh mesh + solve at default (suppressed) kinetics; shared by several tests."""
    return solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse")


def test_charge_conservation(beaker_default):
    res = beaker_default
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert abs(res.I_anode_A + I_TEST) / I_TEST < 0.01
    # V_cell decomposition is exact by construction
    assert abs(res.V_ohm_V + res.eta_c_V + res.eta_a_V - res.V_cell_V) < 1e-9
    assert res.V_cell_V > 0.2
    # taped: everything lands on the front face
    assert res.front_fraction == pytest.approx(1.0, abs=1e-6)
    assert res.j_mA_cm2.shape == (41, 41)
    assert np.all(res.j_mA_cm2 > 0)
    assert np.all(res.thickness_um > 0)


def test_primary_distribution_edge_heavier_than_centre():
    """Huge i0, no mass-transfer limit -> primary distribution: edges/corners carry more."""
    res = solve_cell(CellGeometry(), _fast_bath(), I_TEST, delta_um=0.0,
                     resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.edge_center_ratio > 1.3, (
        f"primary distribution should be edge-heavy, got edge/centre = "
        f"{res.edge_center_ratio:.2f}")
    assert res.Wa < 0.05  # primary regime: Wagner number << 1


def test_uniform_distribution_at_large_kappa():
    """kappa -> large: Wa >> 1, activation-controlled -> nearly uniform j."""
    bath = replace(Bath(), kappa_22=65.0)  # 1000x default conductivity
    res = solve_cell(CellGeometry(), bath, I_TEST, resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.Wa > 10.0
    assert res.std_over_mean < 0.05, (
        f"distribution should be uniform, got std/mean = {res.std_over_mean:.3f}")


def test_untaped_back_edge_fraction():
    """Untaped: a sizeable share of I_total must NOT reach the front face."""
    res = solve_cell(CellGeometry(taped=False), _fast_bath(), I_TEST, delta_um=0.0,
                     resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.loss_fraction > 0.05
    assert res.front_fraction + res.loss_fraction == pytest.approx(1.0, abs=1e-6)


def test_stray_patch_steals_current():
    """The lumped wet-clip patch (stray_area) is extra active cathode area."""
    res = solve_cell(CellGeometry(taped=True, stray_area=0.3), _fast_bath(), I_TEST,
                     delta_um=0.0, resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.cathode_area_cm2 == pytest.approx(4.0 + 0.3, abs=0.02)
    assert res.loss_fraction > 0.01          # some current lands on the clip, not the front
    assert res.front_fraction + res.loss_fraction == pytest.approx(1.0, abs=1e-6)


def test_delta_map_suppression_asymmetry():
    """A spatially varying delta_map must skew j(x, y): thick layer -> less current."""
    dm = {"x": np.array([-1.0, 0.0, 1.0]),
          "y": np.array([3.0, 4.0, 5.0]),
          "delta_um": np.array([[60.0, 180.0, 300.0]] * 3)}   # rows = y, right side thick
    res = solve_cell(CellGeometry(), Bath(), I_TEST, delta_um=60.0, delta_map=dm,
                     resolution="coarse")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert res.corner_LR_grad_pct > 5.0, (
        f"right half should be mass-transfer limited, L/R grad = "
        f"{res.corner_LR_grad_pct:.1f} %")
    assert any("delta_map" in w for w in res.warnings)


def test_duct_1d_ohmic():
    """Coupon + anode spanning a rectangular duct: V_ohm = I*gap/(kappa*A) within 3 %."""
    gap, w, h = 4.0, 2.0, 2.0
    bath = Bath()
    out = solve_duct(gap, w, h, I_TEST, _fast_bath(), suppressed=True, delta_um=0.0)
    v_ohm_theory = I_TEST * gap / (bath.kappa * w * h)
    assert abs(out["V_cell_V"] - v_ohm_theory) / v_ohm_theory < 0.03
    # and the duct solution really is 1-D: uniform j on the cathode face
    assert out["j_std_over_mean"] < 1e-6
    assert abs(out["I_cathode_A"] - I_TEST) / I_TEST < 1e-5


def test_run_plugin_contract():
    out = run({"resolution": "coarse", "t_min": 30.0})
    assert out["summary"]["I_total_mA"] == pytest.approx(48.0, rel=1e-6)
    kinds = [p["kind"] for p in out["plots"]]
    assert kinds == ["heatmap", "heatmap", "line"]
    assert out["summary"]["V_cell_V"] > 0.2
    assert MODEL_SPEC["id"] == "cell_current"
    assert {p["name"] for p in MODEL_SPEC["params"]} >= {
        "j_mA_cm2", "t_min", "T_C", "efficiency", "delta_um", "suppressed", "taped",
        "stray_area_cm2", "gap_cm", "anode_w_cm", "coupon_w_cm", "coupon_h_cm",
        "kappa_S_cm", "resolution"}
    assert isinstance(out["assumptions"], list) and out["assumptions"]
    # JSON-serialisable
    import json
    json.dumps(out)


def test_gmsh_fallback_box(monkeypatch):
    """If gmsh cannot be imported, solve_cell falls back to a structured box domain."""
    from platesim.cell import mesh as meshmod
    from platesim.cell.solver import _MESH_CACHE

    _MESH_CACHE.clear()                       # do not reuse the gmsh mesh for this geometry
    monkeypatch.setattr(meshmod, "_gmsh_available", lambda: False)
    try:
        res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse")
        assert res.domain == "structured_box"
        assert any("BOX approximation" in w for w in res.warnings)
        assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
        assert np.all(res.j_mA_cm2 > 0)
    finally:
        _MESH_CACHE.clear()                   # never leave a box mesh under the gmsh key


# ------------------------------------------------ D5a: thin-seed sheet resistance (terminal effect)

def _third_means(res, geom=None):
    """(near, far) mean j in the thirds nearest/farthest from the top clip edge."""
    geom = geom or CellGeometry()
    j, z = res.j_mA_cm2, res.z_cm
    h = geom.coupon_h
    near = float(j[z >= z.max() - h / 3.0].mean())
    far = float(j[z <= z.min() + h / 3.0].mean())
    return near, far


def test_seed_off_reproduces_equipotential(beaker_default):
    """seed_R_sq=None (the default) is bit-identical to the old behaviour."""
    res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse")
    assert np.array_equal(res.j_mA_cm2, beaker_default.j_mA_cm2)
    assert res.seed_R_sq is None
    assert res.n_seed_iter == 0
    assert res.vm_max_mV == 0.0


def test_seed_negligible_sheet_resistance(beaker_default):
    """R_s -> 0 must collapse onto the equipotential solution."""
    res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse", seed_R_sq=1e-6)
    ref = beaker_default.j_mA_cm2
    rel = np.max(np.abs(res.j_mA_cm2 - ref)) / ref.mean()
    print(f"\nR_s=1e-6: max relative j change {rel:.2e}, vm_max {res.vm_max_mV:.2e} mV")
    assert rel < 1e-4
    assert res.I_cathode_A == pytest.approx(I_TEST, rel=1e-6)


def test_seed_terminal_effect_conserves_current():
    """R_s = 0.05 ohm/sq: clip side plates heavier, current still conserved."""
    res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse", seed_R_sq=0.05)
    near, far = _third_means(res)
    print(f"\nR_s=0.05: near/far={near / far:.5f}, vm_max={res.vm_max_mV:.3f} mV, "
          f"n_seed={res.n_seed_iter}")
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    assert abs(res.I_anode_A + I_TEST) / I_TEST < 0.01
    assert near > far, f"near clip third must plate heavier: {near:.3f} vs {far:.3f}"
    assert res.vm_max_mV > 0.0
    assert res.n_seed_iter < 30
    assert res.seed_R_sq == 0.05


def test_seed_effect_grows_with_sheet_resistance(beaker_default):
    """Larger R_s -> stronger terminal effect (near/far ratio increases)."""
    ratios, vmax = [], []
    for rs in (0.02, 0.05, 0.2):
        res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse", seed_R_sq=rs)
        near, far = _third_means(res)
        ratios.append(near / far)
        vmax.append(res.vm_max_mV)
        print(f"R_s={rs:4g}: near/far={near / far:.5f}, vm_max={res.vm_max_mV:.3f} mV, "
              f"n_seed={res.n_seed_iter}")
        assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01
    print(f"base (equipotential) near/far = {_third_means(beaker_default)[0] / _third_means(beaker_default)[1]:.5f}")
    assert ratios[0] < ratios[1] < ratios[2]
    assert vmax[0] < vmax[1] < vmax[2]
    assert ratios[2] > ratios[0] + 0.01


def test_seed_untaped_warns_front_face_only():
    res = solve_cell(CellGeometry(taped=False), _fast_bath(), I_TEST, delta_um=0.0,
                     resolution="coarse", seed_R_sq=0.05)
    assert any("front face only" in w for w in res.warnings)
    assert abs(res.I_cathode_A - I_TEST) / I_TEST < 0.01


def test_sheet_poisson_1d_analytic():
    """Uniform j0 strip fed from one end: V_m(L) = R_s j0 L^2 / 2."""
    from platesim.cell.solver import _sheet_poisson

    rs, j0, length = 0.05, 0.012, 2.0
    nx, nz = 41, 5                      # uniform in z -> effectively 1-D in x
    j = np.full((nz, nx), j0)
    v = _sheet_poisson(j, rs, length / (nx - 1), 2.0 / (nz - 1), "left")
    v_analytic = rs * j0 * length ** 2 / 2.0
    print(f"\n1-D sheet: V(L) = {v.max() * 1e3:.4f} mV, analytic {v_analytic * 1e3:.4f} mV")
    assert v.max() == pytest.approx(v_analytic, rel=0.02)
    assert np.ptp(v, axis=0).max() < 1e-9 * max(v_analytic, 1.0)  # z-independent



# ------------------------------------------------ F3: pulse-reverse duty average

PR_TEAM = {"t_f_ms": 9.0, "t_r_ms": 0.5, "t_off_ms": 0.5, "rr": 2.0}
PR_DC = {"t_f_ms": 9.0, "t_r_ms": 0.0, "t_off_ms": 0.0, "rr": 2.0}


@pytest.fixture(scope="module")
def pr_default():
    """One PR solve (forward + reverse phase) shared by the charge and map tests."""
    return solve_cell(CellGeometry(), Bath(), I_TEST, t_s=1200.0, resolution="coarse",
                      pr=PR_TEAM)


def _pr_ref(pr=PR_TEAM):
    """(I_f, I_r, duty_f, duty_r) implied by a NET current of I_TEST."""
    tf, tr, toff, rr = pr["t_f_ms"], pr["t_r_ms"], pr.get("t_off_ms", 0.0), pr["rr"]
    P = tf + tr + toff
    I_f = I_TEST * P / (tf - rr * tr)
    return I_f, -rr * I_f, tf / P, tr / P


def test_pr_off_is_bit_identical(beaker_default):
    """pr=None (default) keeps the DC path bit-for-bit."""
    res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse", pr=None)
    assert np.array_equal(res.j_mA_cm2, beaker_default.j_mA_cm2)
    assert np.array_equal(res.thickness_um, beaker_default.thickness_um)
    assert res.pr is None
    assert res.j_forward_mA_cm2 is None and res.j_reverse_mA_cm2 is None
    assert res.I_net_A is None and res.pr_dl_fraction == 0.0
    assert res.I_front_net_A == pytest.approx(beaker_default.I_cathode_A, rel=1e-12)


def test_pr_dc_limit_equals_dc(beaker_default):
    """t_r = t_off = 0 -> the reverse phase carries no charge: exactly the DC solve."""
    res = solve_cell(CellGeometry(), Bath(), I_TEST, resolution="coarse", pr=PR_DC)
    ref = beaker_default.j_mA_cm2
    rel = float(np.max(np.abs(res.j_mA_cm2 - ref)) / np.max(np.abs(ref)))
    print(f"\nPR DC limit (t_r = t_off = 0): max relative j change {rel:.2e}")
    assert rel < 1e-9
    assert res.I_net_A == pytest.approx(I_TEST, rel=1e-6)
    assert res.duty_r == 0.0


def test_pr_charge_balance(pr_default):
    """duty_f*I_f - duty_r*|I_r| == I_total (Faraday charge of the net map)."""
    t_s = 1200.0
    res = pr_default
    I_f, I_r, duty_f, duty_r = _pr_ref()
    assert res.I_forward_A == pytest.approx(I_f, rel=1e-9)
    assert res.I_reverse_A == pytest.approx(I_r, rel=1e-9)
    assert res.duty_f == pytest.approx(duty_f, rel=1e-12)
    assert res.duty_r == pytest.approx(duty_r, rel=1e-12)
    q_net = res.I_net_A * t_s
    print(f"\ncharge: net {q_net:.9f} C, I_total*t {I_TEST * t_s:.9f} C, "
          f"rel {abs(q_net - I_TEST * t_s) / (I_TEST * t_s):.2e}")
    assert q_net == pytest.approx(I_TEST * t_s, rel=1e-6)
    # double-layer share removes (1 - f_dl) of the reverse faradaic charge
    dl = solve_cell(CellGeometry(), Bath(), I_TEST, t_s=t_s, resolution="coarse",
                    pr=PR_TEAM, pr_dl_fraction=1.0 / 3.0)
    q_dl = dl.I_net_A * t_s
    q_expected = q_net + (1.0 / 3.0) * duty_r * abs(I_r) * t_s
    print(f"charge with 1/3 double-layer share: {q_dl:.9f} C "
          f"(DC-equivalent +{q_dl - q_net:.9f} C of reverse charge withheld)")
    assert q_dl == pytest.approx(q_expected, rel=1e-6)


def test_pr_maps_and_ratios_vs_dc(beaker_default, pr_default):
    """PR net / forward / reverse maps are consistent; ratios are printed vs DC."""
    from platesim.cell.solver import _uniformity

    res = pr_default
    assert res.j_mA_cm2.shape == (41, 41)
    assert res.j_forward_mA_cm2.shape == (41, 41)
    assert res.j_reverse_mA_cm2.shape == (41, 41)
    assert np.all(res.j_reverse_mA_cm2 >= 0.0)          # removal density, positive
    assert np.all(res.j_mA_cm2 > 0.0)                   # net deposition everywhere
    assert np.all(res.thickness_um > 0.0)
    assert np.allclose(res.j_mA_cm2, res.duty_f * res.j_forward_mA_cm2
                       - res.duty_r * res.j_reverse_mA_cm2, rtol=0, atol=1e-12)
    # the net map carries the net current (grid quadrature, ~1 % bias)
    w = np.ones(41)
    w[0] = w[-1] = 0.5
    A = float(np.outer(w, w).sum()) * (2.0 / 40.0) ** 2
    j_net_area = float((res.j_mA_cm2 * np.outer(w, w)).sum()) * (2.0 / 40.0) ** 2
    u_dc = _uniformity(beaker_default.j_mA_cm2, 1.0, 1.0)
    u_pr = _uniformity(res.j_mA_cm2, 1.0, 1.0)
    u_rv = _uniformity(res.j_reverse_mA_cm2, 1.0, 1.0)
    near, far = _third_means(res)
    n_dc, f_dc = _third_means(beaker_default)
    print(f"\nDC: near/far={n_dc / f_dc:.4f}, edge/centre={u_dc['edge_center_ratio']:.4f}")
    print(f"PR: near/far={near / far:.4f}, edge/centre={u_pr['edge_center_ratio']:.4f}, "
          f"j(area)/I_net={j_net_area * 1e-3 / res.I_net_A:.4f} (A = {A:.3f} cm2), "
          f"reverse edge/centre={u_rv['edge_center_ratio']:.4f}")
    # the anodic phase is far more edge-concentrated -> the net map is levelled
    assert u_rv["edge_center_ratio"] > 1.5 * u_dc["edge_center_ratio"]
    assert u_pr["edge_center_ratio"] < u_dc["edge_center_ratio"]
    assert abs(j_net_area * 1e-3 / res.I_net_A - 1.0) < 0.02


# ------------------------------------------------ F4: time-marched coupon cell

def test_march_one_stage_equals_solve_cell():
    """One stage reproduces solve_cell at the same R_s (same solve, same map)."""
    for rs in (1e-3, None):
        m = march_cell(CellGeometry(), Bath(), I_TEST, hours=0.5, n_stages=1, seed_R_sq0=rs)
        ref = solve_cell(CellGeometry(), Bath(), I_TEST, t_s=1800.0, resolution="coarse",
                         seed_R_sq=rs)
        assert np.array_equal(m.stages[0].j_mA_cm2, ref.j_mA_cm2)
        rel = abs(float(m.thickness_um.mean()) / float(ref.thickness_um.mean()) - 1.0)
        shape = float(np.max(np.abs(m.thickness_um / m.thickness_um.mean()
                                    - ref.thickness_um / ref.thickness_um.mean())))
        print(f"\nR_s0={rs}: 1-stage march vs solve_cell: mean rel {rel:.2e}, "
              f"shape residual {shape:.2e} (charge projection "
              f"{m.stages[0].charge_scale:.5f})")
        assert shape < 1e-9                    # same map, only the level differs
        assert rel < 1e-2                      # the grid charge projection is ~0.8 %
        assert m.stages[0].I_net_A == pytest.approx(I_TEST, rel=1e-6)


def test_march_terminal_effect_fades_and_mass_balance():
    """R_s0 = 0.2 ohm/sq: the deposit lands on the sheet and the terminal effect fades."""
    m = march_cell(CellGeometry(), Bath(), I_TEST, hours=1.0, n_stages=6, seed_R_sq0=0.2)
    ratios = [s.near_far_thickness_ratio for s in m.stages]
    R_mean = [s.R_s_mean_ohm_sq for s in m.stages]
    for s in m.stages:
        print(f"\nstage {s.index}: R_s {s.R_s_mean_ohm_sq:8.4f} ohm/sq, "
              f"near/far(th) {s.near_far_thickness_ratio:.5f}, "
              f"near/far(j) {s.near_far_j_ratio:.5f}, "
              f"edge/centre(th) {s.edge_center_thickness_ratio:.4f}, "
              f"dep {s.dep_um.mean():.3f} um, V {s.V_cell_V:.4f} V")
    assert R_mean[0] > R_mean[-1]
    assert all(R_mean[i] >= R_mean[i + 1] - 1e-12 for i in range(len(R_mean) - 1))
    assert all(ratios[i] >= ratios[i + 1] - 1e-9 for i in range(len(ratios) - 1))
    assert ratios[-1] < ratios[0], f"terminal effect must fade: {ratios}"
    print(f"mass balance {m.mass_balance:.8f}, V_cell {m.V_cell_first_V:.4f} -> "
          f"{m.V_cell_last_V:.4f} V")
    assert m.mass_balance == pytest.approx(1.0, abs=1e-4)
    assert m.thickness_um.shape == (41, 41)
    assert np.all(m.thickness_um > 0.0)
    assert m.thickness_um.mean() == pytest.approx(
        float(__import__("platesim.core.constants", fromlist=["thickness_um"])
              .thickness_um(12.0, 3600.0)), rel=0.05)
