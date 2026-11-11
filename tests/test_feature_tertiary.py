"""Tertiary current distribution (Task 6): explicit Cu2+ / H+ / SO4^2- transport in the level-set engine.

1-D planar limiting current.  With every non-reacting ion blocked (N_i = 0) the steady 1-D
Nernst-Planck problem with electroneutrality has the analytic limiting current that
``params.migration_enhancement`` encodes:

    i_lim = 2 F D_Cu c_Cu,b E / delta,    E = 1 + (2/c_b) int_{phi_L}^{0} c_Cu d phi_hat = 1.6690

for this bath (65 g/L Cu + 8 g/L H2SO4, fully dissociated, 22 C), i.e.

    E_num = i_lim delta / (2 F D_Cu c_b) = 1.669 +/- 2 %   at delta = 20 um  (403 mA/cm2),

and E = 2.000 +/- 2 % for a CuSO4-only (2:2 binary) bath, where i_lim = 4 F D_Cu c_b / delta.
The model limiting current is measured on ``StepSolver`` itself (no sim.py): the galvanostatic
current is bisected until the Cu2+ concentration at the front reaches zero.

See research/tertiary_transport_notes.md sections 4-5 for the scheme and the target numbers.
"""
from __future__ import annotations

import numpy as np

from platesim.feature.engine.fields import StepGeom, StepSolver, bern
from platesim.feature.engine.levelset import Grid
from platesim.feature.engine.params import F, M_CU, Params, default_enhancement, derive
from platesim.feature.engine.recipes import dc
from platesim.feature.engine.run_case import run_case
from platesim.feature.engine.sim import ViaSim

DELTA_UM = 20.0
H_UM = 2.0            # front at z = 0 (psi = z), diffusion layer above
DX_UM = 0.125         # ~0.1 % on E_num; the smeared interface costs up to 5 % at dx = 0.5 um
E_BATH = 1.6690


def test_bernoulli_function_branches():
    """B(x) = x/(exp(x)-1): Taylor |x| < 1e-3, x/expm1 in the middle, x exp(-x) for x > 40, no overflow."""
    x = np.array([-45.0, -20.0, -2.0, -1e-4, 0.0, 1e-4, 0.9e-3, 0.5, 2.0, 20.0, 41.0, 45.0])
    ref = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    ref[nz] = x[nz] / np.expm1(np.clip(x[nz], -700.0, 700.0))
    np.testing.assert_allclose(bern(x), ref, rtol=1e-6, atol=0.0)
    # exact identity B(-x) = B(x) + x (one exponential per face)
    np.testing.assert_allclose(bern(-x), bern(x) + x, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(bern(np.array([-745.0, -50.0, 50.0, 745.0]))))



def slab(c_h_bulk=None, c_so4_bulk=None):
    """1-D flat slab: f_open = 1, psi = z (front at z = 0), Dirichlet plane at delta.  Every column of
    the (nr, nz) grid is identical, so nr is held at 6 and the geometry carries no trench effect."""
    kw = {}
    if c_h_bulk is not None:
        kw.update(c_h_bulk=c_h_bulk, c_so4_bulk=c_so4_bulk)
    p = Params(transport="tertiary", geometry="trench", H_um=H_UM, suppressor=False, dx_um=DX_UM,
               delta_um=DELTA_UM, max_overburden_um=2.0, **kw)
    ph = derive(p, 22.0)
    d_um = 6.0 * DX_UM / 0.5                  # R = d/2 = 6*dx -> nr = 6
    g = Grid(d_um, H_UM, 1.0, DELTA_UM, dx_um=DX_UM, max_overburden_um=2.0, geometry="trench")
    psi = np.broadcast_to(g.zc, (g.nr, g.nz)).copy()
    geo = StepGeom(g, psi, DELTA_UM * 1e-4, p)
    return p, ph, g, geo


def solve_slab(p, ph, g, geo, j_mA_cm2):
    """One frozen-geometry step at a fixed current density (mA/cm2); returns (out, st, solver, ctrl)."""
    shp = (g.nr, g.nz)
    st = dict(theta=np.zeros(shp), theta_a=np.zeros(shp), s=np.ones(shp), ctil=np.ones(shp),
              phi=np.zeros(shp), phi_r=np.zeros(shp), V=0.0, V_r=0.0)
    solver = StepSolver(geo, p, ph, np.zeros(shp))
    ctrl = dict(mode="I", I_t=j_mA_cm2 * 1e-3 * g.area_cell, duty_f=1.0, duty_r=0.0, rr=0.0, pr=False)
    out = solver.solve(st, ctrl, 120.0)
    return out, st, solver, ctrl


def front_value(v, geo, g):
    """Value at the front z = 0 by linear interpolation between the two straddling active cells."""
    z = g.zc[0]
    jb = int(np.searchsorted(z, 0.0) - 1)
    t = (0.0 - z[jb]) / g.dx
    full = np.zeros((g.nr, g.nz))
    full[geo.act] = v
    return float(((1.0 - t) * full[:, jb] + t * full[:, jb + 1]).mean())


def model_limiting_current(p, ph, g, geo, i_lim_mA, tol=1e-4, n=9):
    """Bisect the galvanostatic current for the largest value with a converged solve and c(front) > tol."""
    lo, hi = 0.85, 1.2
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        out, _, _, _ = solve_slab(p, ph, g, geo, mid * i_lim_mA)
        ok = out["info"]["converged"] and front_value(out["ct"], geo, g) > tol
        lo, hi = (mid, hi) if ok else (lo, mid)
    return 0.5 * (lo + hi)


def e_num(p, ph, i_lim_mA):
    """E_num = i_lim delta / (2 F D_Cu c_b) of the model limiting current (i_lim_mA in mA/cm2)."""
    return i_lim_mA * 1e-3 * DELTA_UM * 1e-4 / (2 * F * ph.D_cu * p.c_bulk)


def test_tertiary_1d_planar_limiting_current():
    """E_num = 1.6690 +/- 2 % (i_lim = 403 mA/cm2 at delta = 20 um) from the tertiary solver."""
    p, ph, g, geo = slab()
    i_lim = 2 * F * ph.D_cu * p.c_bulk * E_BATH / (DELTA_UM * 1e-4) * 1e3          # mA/cm2
    j_lim = model_limiting_current(p, ph, g, geo, i_lim)
    e = e_num(p, ph, j_lim * i_lim)
    assert abs(e / E_BATH - 1.0) < 0.02, f"E_num = {e:.4f} (analytic {E_BATH:.4f}), i_lim = {j_lim * i_lim:.1f} mA/cm2"
    assert abs(j_lim * i_lim - 403.0) < 0.02 * 403.0, j_lim * i_lim


def test_tertiary_1d_limiting_current_binary_cuso4():
    """CuSO4 only (c_H = 0, c_SO4 = c_Cu): the 2:2 binary salt gives E = 2 exactly."""
    p, ph, g, geo = slab(c_h_bulk=0.0, c_so4_bulk=65 / M_CU * 1e-3)
    i_lim = 4 * F * ph.D_cu * p.c_bulk / (DELTA_UM * 1e-4) * 1e3                    # mA/cm2, E = 2
    j_lim = model_limiting_current(p, ph, g, geo, i_lim)
    e = e_num(p, ph, j_lim * i_lim)
    assert abs(e / 2.0 - 1.0) < 0.02, f"E_num = {e:.4f} (analytic 2.0000), i_lim = {j_lim * i_lim:.1f} mA/cm2"


def test_tertiary_species_potential_and_charge_closure():
    """Direct tertiary solve (no sim): positive concentrations, finite local kappa, Cu2+ mass balance and
    local charge balance div i_l = 0 with the diffusion-potential source."""
    p, ph, g, geo = slab()
    i_lim = 2 * F * ph.D_cu * p.c_bulk * E_BATH / (DELTA_UM * 1e-4) * 1e3
    out, st, solver, ctrl = solve_slab(p, ph, g, geo, 0.6 * i_lim)
    info = out["info"]
    assert info["converged"] and abs(info["bal"]) < p.fix_tol_bal
    act = geo.act
    ct, c_h, c_s = out["ct"], out["c_h"], out["c_so4"]          # ct normalised, c_h/c_s absolute
    c_cu = ct * p.c_bulk
    # ---- positivity / finiteness of every species
    for name, v in (("cu", c_cu), ("h", c_h), ("so4", c_s)):
        assert np.all(np.isfinite(v)) and v.min() > 0.0, name
    assert 0.0 < ct.min() < 1.0 and ct.max() <= 3.0
    # ---- electroneutrality closure (sulfate is eliminated, so this is exact by construction)
    assert np.max(np.abs(2 * c_cu + c_h - 2 * c_s)) < 1e-9 * p.c_bulk
    # ---- local kappa: finite, inside the physical band of the notes (0.02 - 1.0 S/cm)
    kap = solver._kappa(c_cu, c_h, c_s)[act]
    assert np.all(np.isfinite(kap))
    assert 0.02 < kap.min() and kap.max() < 1.0, (kap.min(), kap.max())
    # ---- Cu2+ mass balance: what enters through the delta plane is the deposited charge
    inflow = np.sum(solver.bc_c[geo.ptop] * (p.c_bulk - c_cu[geo.ptop]))
    assert abs(inflow / (ctrl["I_t"] / (2 * F)) - 1.0) < 5e-3, inflow / (ctrl["I_t"] / (2 * F))
    # ---- charge closure of the potential equation: div i_l = 0 per cell (residual normalised by w|i|)
    kp, _ = geo.K_data(solver._kappa(c_cu, c_h, c_s))
    res = geo.matvec(kp, out["phi"]) + geo.w * out["i_f"] - solver._diff_src(c_cu, c_h, c_s)
    scale = float(np.sum(geo.w * np.abs(out["i_f"])) + np.sum(np.abs(solver._diff_src(c_cu, c_h, c_s))))
    assert np.max(np.abs(res)) / scale < 5e-2, (np.max(np.abs(res)), scale)
    # ---- H+ accumulates at the front (cation migration, no HER): the documented artifact
    h_front = front_value(c_h, geo, g) / ph.c_h_bulk
    assert 1.0 <= h_front < 3.0, h_front


# ---------------------------------------------------------------- Task 6B: sim.py integration
# Short Trench 1 cell (d = 227, H = 88 um), dx 8 um, delta 64 um, additive-free, ohmic potential on.
# The runs are cheap (0.1-0.9 s) and stay inside the tertiary regime: at 5 mA/cm2 the Cu2+ depletion is
# only a few percent of c_bulk (i_lim ~ 130 mA/cm2 at delta = 64 um), so the two transport models see
# almost the same concentration field and their floor/field ratios must agree.
TRENCH = dict(H_um=88.0, dx_um=8.0, delta_um=64.0)
D_UM = 227.0
PITCH = dict(f_open=0.2, T_C=22.0)


def short_trench(transport, j_mA_cm2=5.0, hours=0.02, ohmic=True):
    """run_case on the short trench cell; returns (result, sim)."""
    p = Params(transport=transport, geometry="trench", k_mix=0.0, suppressor=False, ohmic=ohmic,
               max_overburden_um=40.0, **TRENCH)
    res, sim = run_case(D_UM, PITCH["T_C"], dc(j_mA_cm2, hours), p, f_open=PITCH["f_open"],
                        label=f"tertiary_{transport}", return_sim=True)
    return res, sim


def floor_over_field(res):
    """Floor/field deposit ratio from the engine series (the definition of feature/convergence.py)."""
    s = res["series"]
    field, floor = s["face_thickness_um"][-1], s["floor_height_um"][-1]
    assert field is not None and floor is not None and field > 0.0 and floor >= 0.0, (field, floor)
    return floor / field


def test_tertiary_low_current_matches_enhanced_floor_over_field():
    """At 5 mA/cm2 the tertiary path (local kappa, explicit migration) and the enhanced path
    (D_c = E_mig D_Cu, kappa = kappa_22) give the same floor/field within 0.01.

    ohmic=True on purpose: the point of the tertiary path is the potential/migration coupling, and
    ohmic=False would set phi = 0, i.e. remove the tertiary migration term entirely while the enhanced
    path keeps its E_mig enhancement -- that comparison would differ by 0.04 (measured) for a reason
    that is not a model discrepancy.  With ohmic=True the only difference left is the conductivity:
    local kappa = F^2/(RT) sum z_i^2 D_i c_i vs the constant kappa_22, plus the diffusion-potential
    source of the Cu2+/H+/SO4 pool.  Measured here: enhanced 0.91736, tertiary 0.92583, |diff| 0.00847.
    """
    res_e, _ = short_trench("enhanced")
    res_t, _ = short_trench("tertiary")
    r_e, r_t = floor_over_field(res_e), floor_over_field(res_t)
    assert 0.8 < r_e < 1.0 and 0.8 < r_t < 1.0, (r_e, r_t)
    assert abs(r_e - r_t) < 0.01, f"enhanced {r_e:.5f} vs tertiary {r_t:.5f}"


def test_tertiary_charge_conservation():
    """Tertiary trench run (5 mA/cm2, 0.25 h): the level-set-free Cu2+ balance and the Faraday deposit
    both close within 1 %.  cu_balance_rel compares the Cu2+ flux through the moving Dirichlet plane with
    q_total*efficiency/(2F), so it isolates transport from the interface discretisation."""
    res, _ = short_trench("tertiary", hours=0.25)
    cubal = res["result"]["cu_balance_rel"]
    assert cubal is not None and np.isfinite(cubal), cubal
    assert abs(cubal) < 0.01, cubal
    fa = res["meta"]["faraday"]
    dep, chg = fa["deposited_volume_cm3"], fa["charge_volume_cm3"]
    assert chg > 0.0
    assert abs(dep - chg) / chg < 0.01, (dep, chg)


def test_tertiary_sim_state_and_low_current_smoke():
    """ViaSim seeds the explicit species state for transport='tertiary' only (bulk H+ / sulfate), and a
    low-current run_case completes with finite, non-negative species arrays.  The Cu snapshot/series
    stay on st['ctil']."""
    p = Params(transport="tertiary", geometry="trench", k_mix=0.0, suppressor=False, ohmic=True,
               max_overburden_um=40.0, **TRENCH)
    sim = ViaSim(D_UM, PITCH["T_C"], p, PITCH["f_open"], dc(5.0, 0.02))
    assert "c_h" in sim.st and "c_so4" in sim.st
    np.testing.assert_allclose(sim.st["c_h"], sim.ph.c_h_bulk)
    np.testing.assert_allclose(sim.st["c_so4"], sim.ph.c_so4_bulk)
    # the enhanced state is untouched (no extra keys, so the legacy backup/vector path is unchanged)
    pe = Params(transport="enhanced", geometry="trench", k_mix=0.0, suppressor=False, ohmic=True,
                max_overburden_um=40.0, **TRENCH)
    se = ViaSim(D_UM, PITCH["T_C"], pe, PITCH["f_open"], dc(5.0, 0.02))
    assert "c_h" not in se.st and "c_so4" not in se.st
    # ---- low-current end-to-end run
    res, sim = short_trench("tertiary", j_mA_cm2=2.0, hours=0.02)
    assert res["result"]["stop_reason"] == "recipe_end"
    assert res["result"]["overburden_um"] > 0.0
    assert res["result"]["cu_balance_rel"] is not None and np.isfinite(res["result"]["cu_balance_rel"])
    for key in ("c_h", "c_so4", "ctil"):
        v = sim.st[key]
        assert np.all(np.isfinite(v)), key
        assert v.min() >= 0.0, (key, v.min())
    for key in ("floor_height_um", "face_thickness_um", "c_floor"):
        assert np.all(np.isfinite(np.asarray(res["series"][key], float))), key
