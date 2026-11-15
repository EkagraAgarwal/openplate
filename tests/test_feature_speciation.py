"""Plan v2 D1b: bath speciation wired into the tertiary feature-fill path.

What is switched on by ``Params(speciation=True)`` (tertiary transport only, see params.derive):
  * the bulk HSO4-/CuSO4^0 equilibrium of ``platesim.core.speciation`` is solved at setup
    (``fields.bulk_speciation``, cached across the per-step StepSolver rebuilds);
  * the BV/kinetic reactive concentration becomes gamma_Cu * c_Cu_free instead of total Cu (the
    transported field stays TOTAL Cu and the free fraction is applied as the bulk ratio);
  * the local Nernst-Einstein kappa is rescaled by kappa_speciated / kappa_fully_dissociated, so the
    bath conductivity drops from the free-ion value to the speciated one.

All numbers here are trend-level [J]: Davies at I ~ 0.8 mol/L is an extrapolation and log10 K_ip = 2.3
is not verified.  The tests therefore assert DIRECTIONS and invariants, not absolute chemistry.

``speciation=False`` (the default) must be bit-identical: the reference numbers below were recorded
from the unmodified engine (unrounded ``ViaSim.run`` series, 0.25 h ohmic trench case, dx 8 um).
"""
from __future__ import annotations

import numpy as np

from platesim.core.speciation import conductivity, dissociated_spec, speciate
from platesim.feature.engine.fields import StepGeom, StepSolver
from platesim.feature.engine.levelset import Grid
from platesim.feature.engine.params import Params, derive
from platesim.feature.engine.recipes import dc
from platesim.feature.engine.run_case import run_case
from platesim.feature.engine.sim import ViaSim

# Short Trench 1 cell (d = 227 um, H = 88 um), dx 8 um, delta 64 um, additive-free, ohmic on.
TRENCH = dict(H_um=88.0, dx_um=8.0, delta_um=64.0)
D_UM = 227.0
F_OPEN = 0.2
J_MA = 5.0

# Recorded from the unmodified engine BEFORE the D1b edit (exact float64, 0.25 h run).
REF_025H = dict(
    V=-0.04409444399527637,
    eta_floor=-0.04300430419203643,
    eta_face=-0.04371446738765923,
    floor_height_um=1.265194601209093,
    face_thickness_um=1.3668500393282206,
    I_total=0.0002837500000000012,
    c_floor=0.9057769226563752,
)
REF_CU_BALANCE = -9.313578046565851e-07


def trench_params(speciation=False, **kw):
    return Params(transport="tertiary", geometry="trench", k_mix=0.0, suppressor=False, ohmic=True,
                  max_overburden_um=40.0, speciation=speciation, **TRENCH, **kw)


def run_trench(hours=0.25, speciation=False, j_ma=J_MA):
    """ViaSim.run (raw, unrounded series) plus the sim, for the short trench cell."""
    sim = ViaSim(D_UM, 22.0, trench_params(speciation), F_OPEN, dc(j_ma, hours))
    return sim.run("speciation"), sim


def slab(speciation=False):
    """1-D flat slab (f_open = 1, front at z = 0, Dirichlet plane at delta = 20 um), dx 0.5 um.

    Same construction as test_feature_tertiary.slab: every column is identical, nr is held at 6.
    """
    H_UM, DX_UM, DELTA_UM = 2.0, 0.5, 20.0
    p = Params(transport="tertiary", geometry="trench", H_um=H_UM, suppressor=False, dx_um=DX_UM,
               delta_um=DELTA_UM, max_overburden_um=2.0, speciation=speciation)
    ph = derive(p, 22.0)
    g = Grid(6.0 * DX_UM / 0.5, H_UM, 1.0, DELTA_UM, dx_um=DX_UM, max_overburden_um=2.0, geometry="trench")
    psi = np.broadcast_to(g.zc, (g.nr, g.nz)).copy()
    geo = StepGeom(g, psi, DELTA_UM * 1e-4, p)
    shp = (g.nr, g.nz)
    st = dict(theta=np.zeros(shp), theta_a=np.zeros(shp), s=np.ones(shp), ctil=np.ones(shp),
              phi=np.zeros(shp), phi_r=np.zeros(shp), V=0.0, V_r=0.0)
    solver = StepSolver(geo, p, ph, np.zeros(shp))
    ctrl = dict(mode="I", I_t=5.0e-3 * g.area_cell, duty_f=1.0, duty_r=0.0, rr=0.0, pr=False)
    out = solver.solve(st, ctrl, 120.0)
    return p, ph, g, geo, solver, out


def test_speciation_off_is_bit_identical():
    """speciation=False leaves the tertiary path untouched: the 0.25 h ohmic trench run reproduces the
    pre-D1b numbers to 1e-12 (V, floor/face overpotential and thickness, current, floor concentration,
    Cu balance)."""
    raw, sim = run_trench(hours=0.25, speciation=False)
    assert sim.spec is None
    s = raw["series"]
    for key, ref in REF_025H.items():
        got = float(s[key][-1])
        assert abs(got - ref) <= 1e-12 * max(1.0, abs(ref)), (key, got, ref)
    assert abs(raw["cu_balance_rel"] - REF_CU_BALANCE) <= 1e-12 * max(1.0, abs(REF_CU_BALANCE))


def test_speciation_bulk_state_and_kappa_ratio():
    """Setup solves the bulk speciation: less than half of the Cu is free, the activity-corrected
    reactive fraction is ~7 %, and the local kappa is exactly the fully-dissociated one times the
    speciated conductivity ratio computed independently from platesim.core.speciation."""
    p, ph, g, geo, solver, out = slab(speciation=True)
    assert solver.spec is not None and solver.spec["converged"]
    # ---- bulk composition: ion pairing + HSO4- formation remove most of the free Cu / sulfate
    assert 0.0 < solver.r_cu < 0.5, solver.r_cu
    assert 0.0 < solver.r_so4 < 0.5, solver.r_so4
    assert 0.0 < solver.r_react < 0.5, solver.r_react               # gamma_Cu * free fraction
    np.testing.assert_allclose(solver.r_react,
                               solver.spec["gamma"]["cu"] * solver.r_cu, rtol=1e-15)
    # ---- independent recomputation of the conductivity ratio with the ENGINE diffusivities
    D25 = {"cu": p.D_cu25, "h": p.D_H25, "so4": p.D_SO425}
    c_cu_L, c_acid_L = p.c_bulk * 1e3, 0.5 * ph.c_h_bulk * 1e3
    spec = speciate(c_cu_L, c_acid_L, T_C=22.0, Ka2=p.Ka2, K_ip=10.0 ** p.logK_ip, activity=p.activity)
    k_spec = conductivity(spec, D=D25, T_C=22.0)
    k_ref = conductivity(dissociated_spec(c_cu_L, c_acid_L, T_C=22.0), D=D25, T_C=22.0)
    assert abs(solver.kappa_ratio - k_spec / k_ref) < 1e-12
    assert abs(solver.kappa_bulk - k_spec) < 1e-12 * k_spec
    # ---- the placeholder (0.065) sits between the speciated and free-ion values (trend-level), and the
    # local kappa at the bulk point equals kappa_speciated exactly (fully dissociated sum x ratio)
    assert 0.0 < solver.kappa_bulk < p.kappa_22 < ph.kappa_ne
    act = geo.act
    c_bulk = np.full(act.sum(), p.c_bulk)
    c_h_bulk = np.full(act.sum(), ph.c_h_bulk)
    c_s_bulk = np.full(act.sum(), ph.c_so4_bulk)
    k_on = solver._kappa(c_bulk, c_h_bulk, c_s_bulk)[act]
    np.testing.assert_allclose(k_on, solver.kappa_bulk, rtol=1e-12)
    # ---- same state, speciation off: the ratio between the two is exactly kappa_ratio
    p0, ph0, g0, geo0, solver0, _ = slab(speciation=False)
    assert solver0.spec is None and solver0.kappa_ratio == 1.0
    k_off = solver0._kappa(c_bulk, c_h_bulk, c_s_bulk)[act]
    np.testing.assert_allclose(k_on, k_off * solver.kappa_ratio, rtol=1e-13)


def test_speciation_lowers_exchange_current_and_raises_overpotential():
    """Same galvanostatic slab solve (5 mA/cm2): the reactive Cu is ~7 % of the total, so the exchange
    current must drop and the overpotential must become more negative (Trend-level expectation of D1b;
    kappa also drops, both push the same way)."""
    _, _, _, _, solver_off, out_off = slab(speciation=False)
    _, _, _, _, solver_on, out_on = slab(speciation=True)
    for out in (out_off, out_on):
        assert out["info"]["converged"], out["info"]
        assert abs(out["info"]["bal"]) < solver_off.p.fix_tol_bal
    eta_off = float(np.min(out_off["eta"]))
    eta_on = float(np.min(out_on["eta"]))
    assert eta_off < 0.0 and eta_on < 0.0
    assert abs(eta_on) > abs(eta_off) + 0.02, (eta_on, eta_off)   # measured: -0.178 vs -0.048 V
    # reactive concentration < total, kappa < fully dissociated
    assert solver_on.r_react < 0.5
    assert solver_on.kappa_bulk < solver_off.ph.kappa_ne


def test_speciation_trench_run_completes_and_cu_balance_closes():
    """0.25 h, 5 mA/cm2 short trench with speciation on: the level-set run completes, the Cu2+ balance
    against the Faraday deposit still closes (|cu_balance_rel| < 0.005), and the species state is finite
    and non-negative."""
    raw, sim = run_trench(hours=0.25, speciation=True)
    assert sim.spec is not None
    assert 0.0 < sim.spec["r_react"] < 0.5
    assert raw["stop_reason"] == "recipe_end"
    assert raw["face_thickness_um"] > 0.0
    assert raw["cu_balance_rel"] is not None and np.isfinite(raw["cu_balance_rel"])
    assert abs(raw["cu_balance_rel"]) < 0.005, raw["cu_balance_rel"]
    for key in ("c_h", "c_so4", "ctil"):
        v = sim.st[key]
        assert np.all(np.isfinite(v)), key
        assert v.min() >= 0.0, (key, v.min())


def test_speciation_overpotential_direction_in_trench():
    """End-to-end direction check on the trench cell: at the same galvanostatic current the speciated
    bath runs at a more negative cell voltage / floor overpotential than the fully dissociated one."""
    ref, _ = run_trench(hours=0.02, speciation=False)
    spc, _ = run_trench(hours=0.02, speciation=True)
    v_off = float(np.min(ref["series"]["eta_floor"]))
    v_on = float(np.min(spc["series"]["eta_floor"]))
    assert v_on < v_off - 0.02, (v_on, v_off)
    assert spc["cu_balance_rel"] is not None and abs(spc["cu_balance_rel"]) < 0.005
