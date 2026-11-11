"""Geometry + resolved-flow tests for the full-width periodic trench (plan.md Task 5).

`trench_full` = planar 2-D over the full pitch d / f_open, periodic in x, no mirror symmetry.
Fast geometry/operator tests plus short flow-coupled runs (zero shear, Couette, wall asymmetry +
Faraday, and the narrow Trench 1 start against platesim.trenchflow).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from platesim.feature.engine import Params, dc, run_case
from platesim.feature.engine.fields import StepGeom
from platesim.feature.engine.levelset import (Grid, advect, band_average, extend, grad_mag, initial_psi,
                                              reinit)
from platesim.feature.engine.params import derive as derive_phys
from platesim.trenchflow.model import shear_from_delta, stage_fields

D_UM, H_UM, F_OPEN, DELTA = 30.0, 100.0, 0.3, 20.0


def _grids(dx_um=2.0):
    g_t = Grid(D_UM, H_UM, F_OPEN, DELTA, dx_um=dx_um, geometry="trench")
    g_f = Grid(D_UM, H_UM, F_OPEN, DELTA, dx_um=dx_um, geometry="trench_full")
    return g_t, g_f


# ------------------------------------------------------------------ grid
def test_full_grid_is_two_half_domains():
    g_t, g_f = _grids()
    # pitch = d / f_open, twice the half-domain R
    assert math.isclose(g_f.R_um, D_UM / F_OPEN, rel_tol=1e-12)
    assert math.isclose(g_f.R_um, 2.0 * g_t.R_um, rel_tol=1e-12)
    # nr = 2 * nr_half and dx is identical to the half-domain grid
    assert g_f.nr_half == g_t.nr
    assert g_f.nr == 2 * g_f.nr_half
    assert g_f.dx == g_t.dx
    assert g_f.dx_um == g_t.dx_um
    # planar, periodic, no mirror: x centres cover 0..pitch, centre at pitch/2
    assert g_f.planar and g_f.periodic_x
    assert g_f.geometry == "trench_full"
    assert not g_t.periodic_x and g_t.planar
    assert math.isclose(g_f.x_c, 0.5 * g_f.R, rel_tol=1e-12)
    assert math.isclose(g_f.rc[0, 0], 0.5 * g_f.dx) and math.isclose(
        g_f.rc[-1, 0] + g_f.rc[0, 0], g_f.R, rel_tol=1e-12)
    # weight-1 planar quantities, projected width = pitch
    np.testing.assert_allclose(g_f.vol[:, 0], g_f.dx ** 2)
    np.testing.assert_allclose(g_f.AE[:, 0], g_f.dx)
    np.testing.assert_allclose(g_f.AZ[:, 0], g_f.dx)
    assert math.isclose(g_f.area_cell, g_f.R, rel_tol=1e-12)
    assert math.isclose(g_f.area_cell, D_UM * 1e-4 / F_OPEN, rel_tol=1e-12)
    assert math.isclose(g_f.via_volume, 2.0 * g_f.a * g_f.H, rel_tol=1e-12)
    # via_w is two-sided: both walls contribute, ~2a/dx cells wide in total
    j = int(np.argmin(np.abs(g_f.zc[0] - 0.5 * g_f.H)))
    col = g_f.via_w[:, j] > 0
    assert col[:g_f.nr // 2].any() and col[g_f.nr // 2:].any()
    assert abs(col.sum() - 2.0 * g_f.a / g_f.dx) <= 2


# ------------------------------------------------------------------ initial level set
def test_initial_psi_is_periodic_and_symmetric():
    _, g = _grids()
    psi = initial_psi(g)
    assert psi.shape == (g.nr, g.nz)
    # symmetric about the trench centre x_c (no mirror used: explicit abs distance); the FMM sweep is not
    # exactly order-independent, so allow a few percent of a cell
    np.testing.assert_allclose(psi, psi[::-1], rtol=0.0, atol=0.05 * g.dx)
    # continuous across the periodic seam
    assert np.abs(psi[0, :] - psi[-1, :]).max() <= 2.0 * g.dx
    # sign pattern: metal below the floor, electrolyte in the trench and on both fields
    j_floor = int(np.argmin(np.abs(g.zc[0] + 0.5 * g.dx)))
    assert psi[:, j_floor].max() < 0
    i_c = g.nr // 2
    j_mid = int(np.argmin(np.abs(g.zc[0] - 0.5 * g.H)))
    assert psi[i_c, j_mid] > 0            # trench mouth region
    assert psi[0, j_mid] < 0 and psi[-1, j_mid] < 0     # field metal on both sides


def test_periodic_reinit_keeps_linear_distance_and_extend_keeps_constant():
    _, g = _grids(dx_um=4.0)
    z0 = 0.25 * g.H
    psi = np.broadcast_to(g.zc - z0, (g.nr, g.nz)).copy()      # exact signed distance to z = z0
    d = reinit(psi, g.dx, narrow=False, periodic=True)
    np.testing.assert_allclose(d, psi, atol=1e-4 * g.dx)
    d2, f2 = extend(psi, np.full_like(psi, 2.5), g.dx, periodic=True)
    np.testing.assert_allclose(d2, psi, atol=1e-4 * g.dx)
    np.testing.assert_allclose(f2, 2.5, atol=1e-12)
    # narrow band: cells far from the front keep their old value (here: still the exact distance)
    d3 = reinit(psi, g.dx, narrow=True, periodic=True)
    np.testing.assert_allclose(d3, psi, atol=1e-4 * g.dx)


# ------------------------------------------------------------------ operators
def test_advect_zero_velocity_is_identity():
    _, g = _grids()
    psi = initial_psi(g)
    out = advect(psi, np.zeros_like(psi), 0.5 * g.dx, g.dx, periodic=g.periodic_x)
    np.testing.assert_array_equal(out, psi)


def test_operators_accept_the_grid_directly():
    """fields.py / sim.py can pass the Grid instead of dx (+ periodic flag)."""
    _, g = _grids()
    psi = initial_psi(g)
    np.testing.assert_array_equal(grad_mag(psi, g), grad_mag(psi, g.dx, periodic=g.periodic_x))
    v = np.ones_like(psi)
    np.testing.assert_array_equal(advect(psi, v, 0.1 * g.dx, g),
                                  advect(psi, v, 0.1 * g.dx, g.dx, periodic=g.periodic_x))


def test_band_average_wraps_a_constant_field():
    _, g = _grids(dx_um=4.0)
    # periodic interface through x = 0 and x = pitch/2: sin(2 pi x / pitch) changes sign at the seam
    psi = np.broadcast_to(-3.0 * g.dx * np.sin(2.0 * np.pi * g.rc / g.R), (g.nr, g.nz)).copy()
    psi = reinit(psi, g.dx, narrow=False, periodic=True)
    w = np.ones_like(psi)
    out = band_average(psi, w, np.full_like(psi, 3.0), g)
    near = np.abs(psi) < 2.5 * g.dx
    assert near.any()
    assert np.all(out[near & (out != 0.0)] > 0.0)
    sel = out != 0.0
    np.testing.assert_allclose(out[sel], 3.0, rtol=1e-12)
    # the seam columns take part in the wrap: some interface cells sit in the first/last 3 columns
    assert near[0, :].any() or near[1, :].any()


# ------------------------------------------------------------------ advection operator (signs + units + forms)
def test_face_fluxes_and_adv_matrix_signs():
    """Uniform flow over a flat periodic wall: face fluxes and BOTH upwind operator forms.

    A flat `f_open=1` level set makes every row of active cells a complete periodic chain, so a uniform
    x-velocity is discretely divergence-free there.  `adv_matrix(form='divergence')` is the legacy
    conservative form +div(Hs u c) (COLUMN sums zero, row sums = the discrete flux divergence);
    `adv_matrix(form='gradient')` is u.grad(c) = sum_faces F (c_f - c_P) (ROW sums zero, so a uniform field
    is preserved for ANY flux set, solenoidal or not).
    """
    g = Grid(D_UM, H_UM, 1.0, DELTA, dx_um=4.0, geometry="trench_full")
    psi = initial_psi(g)                         # z: the wall is at z = 0, no trench
    geo = StepGeom(g, psi, 20.0 * g.dx, Params(geometry="trench_full"))
    n = geo.n
    assert n > 0 and geo.jw.size > 0
    zero_r, zero_z, zero_w = np.zeros_like(geo.pr, float), np.zeros_like(geo.pz, float), np.zeros_like(geo.pw, float)

    ux, uz = np.ones((g.nr, g.nz)), np.zeros((g.nr, g.nz))
    Fr, Fz, Fw = geo.face_fluxes(ux, uz)
    assert np.all(Fz == 0.0)
    # p -> q on the wrap face is -x, so a +x flow gives a negative wrap flux
    assert (Fr > 0).any() and np.all(Fr >= 0.0)
    assert (Fw < 0).any() and np.all(Fw <= 0.0)
    pos = Fr > 0
    np.testing.assert_allclose(Fr[pos], (geo.Hf_r * geo.geo_r * g.dx)[pos])
    neg = Fw < 0
    np.testing.assert_allclose(Fw[neg], (-geo.Hf_w * geo.geo_w * g.dx)[neg])

    # ---- divergence form (legacy): row sums = discrete divergence, zero for this solenoidal flow
    Ad = geo.adv_matrix(Fr, Fz, Fw, form="divergence")
    np.testing.assert_allclose(Ad @ np.ones(n), 0.0, atol=1e-12)
    np.testing.assert_allclose(geo.flux_divergence(Fr, Fz, Fw), 0.0, atol=1e-12)

    # positive r-face fluxes: +F on (p,p), -F on (q,p); no entry on (p,q)
    Ar = geo.adv_matrix(Fr, zero_z, zero_w, form="divergence").toarray()
    np.testing.assert_allclose(Ar[geo.pr, geo.pr], Fr)
    np.testing.assert_allclose(Ar[geo.qr, geo.pr], -Fr)
    assert np.all(Ar[geo.pr, geo.qr] == 0.0)

    # reverse the flow: the wrap face has F > 0 (p = column 0, q = column nr-1)
    Fw_pos = -Fw
    Aw = geo.adv_matrix(zero_r, zero_z, Fw_pos, form="divergence").toarray()
    np.testing.assert_allclose(Aw[geo.pw, geo.pw], Fw_pos)
    np.testing.assert_allclose(Aw[geo.qw, geo.pw], -Fw_pos)
    assert np.all(Aw[geo.pw, geo.qw] == 0.0) and np.all(Aw[geo.qw, geo.qw] == 0.0)

    # F < 0 branch: A[p,q] = F, A[q,q] = -F
    An = geo.adv_matrix(-Fr, zero_z, zero_w, form="divergence").toarray()
    np.testing.assert_allclose(An[geo.pr, geo.qr], -Fr)
    np.testing.assert_allclose(An[geo.qr, geo.qr], Fr)

    # ---- gradient form: for a uniform ux (solenoidal here) it equals the divergence form everywhere
    Ag = geo.adv_matrix(Fr, Fz, Fw, form="gradient")
    solen = np.abs(geo.flux_divergence(Fr, Fz, Fw)) <= 1e-9 * np.abs(Fr).max()
    assert solen.all()                                       # the interior rows are discretely mass-consistent
    np.testing.assert_allclose((Ag - Ad).toarray()[solen], 0.0, atol=1e-12 * np.abs(Ad).max())
    np.testing.assert_allclose(Ag @ np.ones(n), 0.0, atol=1e-12)

    # ---- arbitrary NON-solenoidal flux set: ROW sums are exactly zero for 'gradient', not for 'divergence'
    rng = np.random.default_rng(7)
    sF = 1e-3
    Frn, Fzn, Fwn = (rng.normal(size=Fr.size) * sF, rng.normal(size=Fz.size) * sF, rng.normal(size=Fw.size) * sF)
    Agn = geo.adv_matrix(Frn, Fzn, Fwn, form="gradient")
    Adn = geo.adv_matrix(Frn, Fzn, Fwn, form="divergence")
    div = geo.flux_divergence(Frn, Fzn, Fwn)
    scale = np.abs(Agn).sum(axis=1).ravel()
    # |A_grad @ 1| < 1e-12 * row scale  <=>  every row sums to zero to machine precision
    assert np.all(np.abs(Agn @ np.ones(n)) <= 1e-12 * np.maximum(scale, 1e-300))
    np.testing.assert_allclose(div, Adn @ np.ones(n), atol=1e-18)
    assert np.abs(div).max() > 0.1 * sF                      # genuinely non-solenoidal
    # the two forms differ by exactly diag(div) = the spurious source -c div(Hs u) of the legacy form
    np.testing.assert_allclose((Adn - Agn).toarray(), np.diag(div), atol=1e-18)
    # upwind signs of the gradient form: F > 0 puts +F on (q,q) and -F on (q,p)
    Agr = geo.adv_matrix(Fr, zero_z, zero_w, form="gradient").toarray()
    np.testing.assert_allclose(Agr[geo.qr, geo.qr], Fr)
    np.testing.assert_allclose(Agr[geo.qr, geo.pr], -Fr)
    # ... and F < 0 puts -F on (p,p) and +F on (p,q)
    Agn2 = geo.adv_matrix(-Fr, zero_z, zero_w, form="gradient").toarray()
    np.testing.assert_allclose(Agn2[geo.pr, geo.pr], Fr)
    np.testing.assert_allclose(Agn2[geo.pr, geo.qr], -Fr)

    with pytest.raises(ValueError):
        geo.adv_matrix(Fr, Fz, Fw, form="flux")


# ------------------------------------------------------------------ flow: zero shear == mirrored half-domain
def test_zero_shear_matches_mirrored_trench_half_domain():
    """flow_shear_s=0: the full pitch reproduces the mirrored 'trench' half-domain deposit (1 %)."""
    def run(geometry):
        p = Params(geometry=geometry, H_um=H_UM, suppressor=False, ohmic=False, dx_um=8.0,
                   delta_um=20.0, k_mix=0.0, flow_shear_s=0.0)
        return run_case(D_UM, 22.0, dc(10.0, 0.2), p, f_open=F_OPEN, label=f"zero_{geometry}")

    rt, rf = run("trench"), run("trench_full")
    st, sf = rt["series"], rf["series"]
    # deposits, not absolute crossings: the floor starts at the (identical) rounded corner level
    floor_t = st["floor_height_um"][-1] - st["floor_height_um"][0]
    floor_f = sf["floor_height_um"][-1] - sf["floor_height_um"][0]
    face_t, face_f = st["face_thickness_um"][-1], sf["face_thickness_um"][-1]
    assert floor_t > 0.5 and face_t > 0.2, (floor_t, face_t)        # a measurable deposit
    assert abs(floor_f - floor_t) <= 0.01 * abs(floor_t), (floor_t, floor_f)
    assert abs(face_f - face_t) <= 0.01 * abs(face_t), (face_t, face_f)
    fill_t = rt["result"]["fill_fraction_final"]
    fill_f = rf["result"]["fill_fraction_final"]
    assert abs(fill_f - fill_t) <= 0.01 * fill_t, (fill_t, fill_f)


# ------------------------------------------------------------------ flow: Couette over a flat wall
def _couette_run(shear):
    p = Params(geometry="trench_full", H_um=10.0, suppressor=False, ohmic=False, dx_um=2.0,
               delta_um=64.0, k_mix=0.0, flow_shear_s=shear, max_overburden_um=20.0)
    return run_case(20.0, 22.0, dc(2.0, 0.05), p, f_open=1.0, label="couette")


def test_couette_flow_leaves_1d_diffusion_unchanged():
    """A shear flow parallel to a flat wall has no normal velocity: the 1-D solution is unchanged."""
    ph = derive_phys(Params(), 22.0)
    a, b = _couette_run(0.0), _couette_run(shear_from_delta(64.0, ph.D_c))
    sa, sb = a["series"], b["series"]
    for key in ("c_floor", "floor_height_um"):
        va, vb = sa[key][-1], sb[key][-1]
        assert abs(vb - va) <= 0.01 * abs(va), (key, va, vb)
    fa = a["result"]["fill_fraction_final"]
    fb = b["result"]["fill_fraction_final"]
    assert abs(fb - fa) <= 0.01 * fa, (fa, fb)


# ------------------------------------------------------------------ flow: wall asymmetry + Faraday
def _wall_thickness_um(psi, g, z_frac=0.5):
    """Upstream (left) and downstream (right) wall deposit thickness at z = z_frac * H, in um."""
    j = int(np.argmin(np.abs(g.zc[0] - z_frac * g.H)))
    row, x = psi[:, j], g.rc[:, 0]
    left = np.flatnonzero((row[:-1] <= 0) & (row[1:] > 0))       # metal -> electrolyte (left wall)
    right = np.flatnonzero((row[:-1] > 0) & (row[1:] <= 0))      # electrolyte -> metal (right wall)
    assert left.size and right.size

    def interp(i):
        t = row[i] / (row[i] - row[i + 1])
        return x[i] + t * (x[i + 1] - x[i])

    xl, xr = g.x_c - g.a, g.x_c + g.a
    return (interp(left[0]) - xl) / 1e-4, (xr - interp(right[-1])) / 1e-4


def test_flow_makes_walls_asymmetric_and_faraday_holds():
    ph = derive_phys(Params(), 22.0)
    G = shear_from_delta(20.0, ph.D_c)
    p = Params(geometry="trench_full", H_um=30.0, suppressor=False, ohmic=False, dx_um=4.0,
               delta_um=20.0, k_mix=0.0, flow_shear_s=G, max_overburden_um=20.0)
    res, sim = run_case(60.0, 22.0, dc(10.0, 1.0), p, f_open=0.5, label="wall_flow", return_sim=True)
    t_up, t_dn = _wall_thickness_um(sim.psi, sim.g)
    assert t_up > 0.0 and t_dn > 0.0, (t_up, t_dn)
    # Task 7A: the upwind 'gradient' operator (adv_form default) removes the spurious -c div(Hs u) source of
    # the legacy 'divergence' form, which also dominated this asymmetry: measured upstream/downstream
    # 7.44/6.71 um (10.2 %) with 'divergence' vs 10.72/10.77 um (0.50 %) with 'gradient'.  The physical
    # statement (the resolved flow breaks the wall symmetry) still holds, just an order of magnitude weaker.
    assert abs(t_up - t_dn) / (0.5 * (t_up + t_dn)) > 0.002, (t_up, t_dn)   # resolved flow breaks symmetry
    fa = res["meta"]["faraday"]
    dep, chg = fa["deposited_volume_cm3"], fa["charge_volume_cm3"]
    assert abs(dep - chg) / chg < 0.03, (dep, chg)


# ------------------------------------------------------------------ flow: narrow Trench 1 start vs trenchflow
def test_narrow_start_floor_over_field_matches_trenchflow():
    """Narrow Trench 1 at the start: floor/field growth agrees with trenchflow.model.stage_fields."""
    d, h, delta, dx, j = 227.0, 88.0, 64.0, 4.0, 11.589
    ph = derive_phys(Params(), 22.0)
    G = shear_from_delta(delta, ph.D_c)
    pitch = 580.0                                    # matches trenchflow.trench_grid's default pitch
    p = Params(geometry="trench_full", H_um=h, suppressor=False, ohmic=False, dx_um=dx, delta_um=delta,
               k_mix=0.0, flow_shear_s=G, max_overburden_um=0.0)
    res, sim = run_case(d, 22.0, dc(j, 0.02), p, f_open=d / pitch, label="narrow_start", return_sim=True)
    s = res["series"]
    ratio = s["floor_height_um"][-1] / s["face_thickness_um"][-1]
    ref = stage_fields(d, h, delta, G, j, 0.0, T_C=22.0, dx_um=dx, use_flow=True)
    assert abs(ratio - ref["floor_over_field"]) <= 0.03, (ratio, ref["floor_over_field"])

