"""MAC Stokes flow (flow_solver='stokes') + conservative advection with an open top (plan items B1c/C2).

The Stokes solver of `platesim.trenchflow.stokes` returns staggered MAC face velocities that are
discretely divergence free in every open cell.  This module checks that
  a. `StepGeom.mac_face_fluxes` maps them onto the existing stencil faces without interpolation, so the
     per-cell flux balance closes in every active fluid cell (the top plane flux is the only face the
     species stencil truncates);
  b. the conservative 'divergence' operator built from those fluxes, plus the open-top boundary terms
     (outflow -> diagonal, inflow -> rhs), preserves a uniform field exactly;
  c. a 0.5 h narrow-site run with the Stokes flow completes, its Cu2+ balance closes (|cu_balance_rel| <
     0.5 %) and the far field stays flat (std < 0.5 um, no ripple), and is much faster than the LBM flow;
  d. the default flow_solver is 'lbm' and is bit-identical to an explicit 'lbm'.

Check case: narrow Trench 1 cell (d = 227 um, H = 88 um, f_open = 0.2, dx = 8 um, delta = 64 um) built
explicitly -- the site data file is not part of this public repo.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from platesim.feature.engine.fields import StepGeom, sg_face_coeffs
from platesim.feature.engine.levelset import UM
from platesim.feature.engine.params import Params, derive
from platesim.feature.engine.recipes import pulse_reverse
from platesim.feature.engine.run_case import run_case
from platesim.feature.engine.sim import ViaSim

D_UM, H_UM, F_OPEN, DX_UM, DELTA_UM, J_NET = 227.0, 88.0, 0.2, 8.0, 64.0, 15.0

BASE = dict(geometry="trench_full", suppressor=False, ohmic=True, dx_um=DX_UM, delta_um=DELTA_UM,
            k_mix=0.0, max_overburden_um=140.0, H_um=H_UM, flow_shear_s=59.0)


def _params(**kw):
    args = dict(BASE)
    args.update(kw)
    return Params(**args)


def _recipe(hours):
    return pulse_reverse(J_NET, hours, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)


_SETUP: dict = {}
_RUNS: dict = {}


def _step0(fs="stokes"):
    """(sim, StepGeom, (ufx, ufz)) of the initial geometry with flow_solver=fs solved once."""
    if fs not in _SETUP:
        p = _params(flow_solver=fs)
        sim = ViaSim(D_UM, 22.0, p, F_OPEN, _recipe(0.5))
        sim.cavity_flow(p.flow_shear_s)
        z_plane = sim.face_height() + p.delta_um * UM
        _SETUP[fs] = (sim, StepGeom(sim.g, sim.psi, z_plane, p), sim.flow_faces)
    return _SETUP[fs]


def _run(fs, hours):
    """(run_case result, wall seconds) of the narrow case with flow_solver=fs, cached per config."""
    key = (fs, hours)
    if key not in _RUNS:
        p = _params(flow_solver=fs)
        t0 = time.time()
        res = run_case(D_UM, 22.0, _recipe(hours), p, f_open=F_OPEN, label=f"narrow_{fs}", out=None)
        _RUNS[key] = (res, time.time() - t0)
    return _RUNS[key]


# ------------------------------------------------------------------ params
def test_flow_solver_is_validated_and_defaults_to_lbm():
    assert Params().flow_solver == "lbm"
    assert Params(flow_solver="stokes").flow_solver == "stokes"
    with pytest.raises(ValueError, match="flow_solver"):
        derive(Params(flow_solver="navier"), 22.0)


# ------------------------------------------------------------------ a. face mapping + discrete divergence
def test_mac_fluxes_use_the_face_velocities_directly_and_are_divergence_free():
    """No interpolation: every stencil face flux is the MAC face velocity times dx * geo, and the per-cell
    sum of outgoing fluxes (plus the top-plane flux on the Dirichlet row) is zero in every active fluid
    cell to < 1e-10 of the largest face flux."""
    sim, geo, (ufx, ufz) = _step0()
    g = sim.g
    F_r, F_z, F_w, F_top = geo.mac_face_fluxes(ufx, ufz)
    assert ufx.shape == (g.nr, g.nz) and ufz.shape == (g.nr, g.nz + 1)

    # face mapping: r-face (ir, jr)-(ir+1, jr) -> ufx[ir+1, jr]; z-face (iz, jz)-(iz, jz+1) -> ufz[iz, jz+1];
    # seam face (p = column 0 -> q = column nr-1, oriented -x) -> -ufx[0, jw]
    np.testing.assert_array_equal(F_r, ufx[geo.ir + 1, geo.jr] * g.dx * geo.geo_r)
    np.testing.assert_array_equal(F_z, ufz[geo.iz, geo.jz + 1] * g.dx * geo.geo_z)
    np.testing.assert_array_equal(F_w, -ufx[0, geo.jw] * g.dx * geo.geo_w)
    np.testing.assert_array_equal(F_top, ufz[geo.itop, geo.jt + 1] * g.dx * (g.AZ[geo.itop, 0] / g.dx))

    # the flow is not trivial, and it does cross the Dirichlet plane in both directions
    fmax = max(np.abs(F_r).max(), np.abs(F_z).max(), np.abs(F_w).max())
    assert fmax > 0.0 and (F_top > 0).any() and (F_top < 0).any()

    top = np.zeros(geo.n)
    top[geo.ptop] = F_top
    div = geo.flux_divergence(F_r, F_z, F_w) + top          # net OUTGOING flux of the cell (top included)
    fluid = geo.psi[geo.act] > 0
    assert fluid.any()
    assert np.abs(div[fluid]).max() < 1e-10 * fmax, (np.abs(div[fluid]).max(), fmax)


# ------------------------------------------------------------------ b. uniform field preserved
def test_conservative_operator_with_open_top_preserves_a_uniform_field():
    """'divergence' form + top boundary terms: (adv @ 1 - adv_bc) == 0 to 1e-10 of the largest face flux.
    Outflow top cells get +F_top on the diagonal, inflow top cells get |F_top| on the rhs."""
    sim, geo, faces = _step0()
    F_r, F_z, F_w, F_top = geo.mac_face_fluxes(*faces)
    adv, adv_bc, top_ret = geo.mac_advection(*faces)
    np.testing.assert_array_equal(top_ret, F_top)
    fmax = max(np.abs(F_r).max(), np.abs(F_z).max(), np.abs(F_w).max())

    resid = adv @ np.ones(geo.n) - adv_bc
    assert np.abs(resid).max() < 1e-10 * fmax, (np.abs(resid).max(), fmax)
    # the open-top boundary is genuinely exercised: both an outflow (diagonal) and an inflow (rhs) cell
    assert (adv_bc > 0).any() and np.any(adv.diagonal()[geo.ptop] > 0)
    # outflow = interior deficit, inflow source = interior surplus: adv@1 - adv_bc == 0 term by term at the top
    np.testing.assert_allclose(resid[geo.ptop], 0.0, atol=1e-14 * fmax)


# ------------------------------------------------------------------ c. run: balance, flat field, speed
def test_stokes_run_closes_the_cu_balance_stays_flat_and_beats_lbm():
    """0.5 h at 15 mA/cm2 (pulse reverse, additive-free, ohmic, G = 59 1/s):
    the conservative MAC advection makes cu_balance_rel close and the field stays flat; the Stokes solve
    is much faster than the LBM one on the same case."""
    res, wall_s = _run("stokes", 0.5)
    s, r = res["series"], res["result"]
    assert r["stop_reason"] in ("recipe_end", "filled_void_free"), r["stop_reason"]
    assert abs(r["cu_balance_rel"]) < 0.005, r["cu_balance_rel"]
    # discretely divergence-free fluxes: the operator has no net internal source, so the "honest" advective
    # balance equals cu_balance_rel (which now contains the plane's advective influx)
    assert abs(r["cu_balance_adv_rel"] - r["cu_balance_rel"]) < 1e-5, (r["cu_balance_rel"],
                                                                       r["cu_balance_adv_rel"])
    std = s["field_std_um"][-1]
    assert std < 0.5, f"field is not flat: std = {std:.3f} um"
    assert np.isfinite(s["field_mean_um"][-1]) and np.isfinite(s["floor_mean_um"][-1])
    print(f"\nstokes 0.5 h: field {s['field_mean_um'][-1]:.3f} um (std {std:.3f}, range "
          f"{s['field_max_um'][-1] - s['field_min_um'][-1]:.3f}), floor {s['floor_mean_um'][-1]:.3f} um, "
          f"cu_balance_rel {r['cu_balance_rel']:+.2e}, steps {res['meta']['steps']}, wall {wall_s:.2f} s")

    res_l, wall_l = _run("lbm", 0.5)
    assert res_l["result"]["t_end_s"] > 0.0
    print(f"wall 0.5 h: stokes {wall_s:.2f} s vs lbm {wall_l:.2f} s (x{wall_l / wall_s:.1f} faster)")
    assert wall_s < 0.5 * wall_l, (wall_s, wall_l)


# ------------------------------------------------------------------ d. default == lbm, bit for bit
def test_default_flow_solver_is_bit_identical_to_lbm():
    """A 0.1 h run with no flow_solver set equals an explicit flow_solver='lbm' run exactly (raw series)."""
    rec = _recipe(0.1)
    raws = []
    for kw in ({}, {"flow_solver": "lbm"}):
        sim = ViaSim(D_UM, 22.0, _params(**kw), F_OPEN, rec)
        raws.append(sim.run("identical"))
    sa, sb = raws[0]["series"], raws[1]["series"]
    assert set(sa) == set(sb)
    assert len(sa["t_s"]) > 1
    for key in sa:
        np.testing.assert_array_equal(np.asarray(sa[key], float), np.asarray(sb[key], float),
                                      err_msg=f"series {key!r} differs")
    assert raws[0]["steps"] == raws[1]["steps"]
    assert raws[0]["cu_influx"] == raws[1]["cu_influx"]


# ================================================================== C1: exponentially fitted (SG) faces
def test_adv_scheme_is_validated_and_defaults_to_upwind():
    """adv_scheme defaults to the legacy 'upwind' path, is validated, and rejects the tertiary transport
    (that path already uses SG with the migration argument; folding advection in is not implemented)."""
    assert Params().adv_scheme == "upwind"
    assert Params(flow_solver="stokes", adv_scheme="sg").adv_scheme == "sg"
    with pytest.raises(ValueError, match="adv_scheme"):
        derive(Params(adv_scheme="central"), 22.0)
    with pytest.raises(NotImplementedError, match="tertiary"):
        derive(Params(transport="tertiary", adv_scheme="sg"), 22.0)
    # the flag is inert without the Stokes solver: the LBM path ignores it
    assert derive(Params(adv_scheme="sg"), 22.0).transport == "enhanced"


def test_sg_face_coeffs_reproduce_the_exact_steady_1d_profile():
    """B(-P) / B(P) is the exact ratio of the steady 1-D advection-diffusion solution, so the discrete
    chain reproduces c(x) = (exp(Pe x/L) - 1)/(exp(Pe) - 1) at the nodes for any Pe."""
    n, L, D = 10, 1.0, 1.0
    dx = L / n
    x = np.arange(n + 1) * dx
    for Pe in (0.1, 10.0, 100.0):
        g = np.full(n, D / dx)
        F = np.full(n, Pe * D / L)                     # P = F/g = Pe dx/L
        gp, gq = sg_face_coeffs(g, F)
        assert np.all(gp >= 0.0) and np.all(gq >= 0.0)
        np.testing.assert_allclose(gp - gq, F, rtol=1e-13, atol=0.0)   # row sums = flux = g P
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        A[0, 0] = A[n, n] = 1.0                        # Dirichlet c(0) = 0, c(L) = 1
        b[n] = 1.0
        for i in range(1, n):
            A[i, i - 1], A[i, i], A[i, i + 1] = -gp[i - 1], gq[i - 1] + gp[i], -gq[i]
        c = np.linalg.solve(A, b)
        exact = (np.exp(Pe * x / L) - 1.0) / (np.exp(Pe) - 1.0)
        np.testing.assert_allclose(c, exact, rtol=0.0, atol=1e-10,
                                   err_msg=f"Pe = {Pe}: SG chain is not exact")
    # P -> 0 is pure diffusion, P >> 1 the upwind flux
    gp0, gq0 = sg_face_coeffs(np.array([2.0]), np.array([0.0]))
    np.testing.assert_allclose([gp0[0], gq0[0]], [2.0, 2.0], rtol=1e-14)
    gpb, gqb = sg_face_coeffs(np.array([2.0]), np.array([200.0]))
    np.testing.assert_allclose([gpb[0], gqb[0]], [200.0, 0.0], rtol=0.0, atol=1e-12)
    # no diffusive conductance but a flux (impossible on the Stokes path): upwind fallback
    gpd, gqd = sg_face_coeffs(np.array([0.0, 0.0]), np.array([3.0, -3.0]))
    np.testing.assert_array_equal(gpd, [3.0, 0.0])
    np.testing.assert_array_equal(gqd, [0.0, 3.0])


def test_sg_operator_preserves_a_uniform_field_on_the_stokes_faces():
    """(A_sg + top outflow) @ 1 == bc + adv_bc to 1e-10 of the largest face flux: every interior row sums to
    the flux divergence (0 for the divergence-free MAC field) and the top row to -F_top + F_top."""
    sim, geo, faces = _step0()
    ph = derive(_params(flow_solver="stokes", adv_scheme="sg"), 22.0)
    F_r, F_z, F_w, F_top = geo.mac_face_fluxes(*faces)
    _, adv_bc, top_ret = geo.mac_advection(*faces)
    np.testing.assert_array_equal(top_ret, F_top)
    fmax = max(np.abs(F_r).max(), np.abs(F_z).max(), np.abs(F_w).max())
    Dt = np.zeros((sim.g.nr, sim.g.nz))
    for D in (ph.D_c + Dt, ph.D_s + Dt):
        A, bc = geo.sg_advdiff(D, F_r, F_z, F_w)
        top_diag, top_rhs = geo.top_advection(F_top)
        A = (A + sp.diags(top_diag)).tocsr()
        assert A.diagonal().min() > 0.0               # M-matrix, positive diagonal
        resid = A @ np.ones(geo.n) - (bc + adv_bc)
        assert np.abs(resid).max() < 1e-10 * fmax, (np.abs(resid).max(), fmax)
        np.testing.assert_array_equal(top_rhs[geo.ptop], np.maximum(-F_top, 0.0))
    # the cell Peclet number on the r-faces really is large (the reason for the exponential fit)
    D = ph.D_c + Dt
    g_r = geo.Hf_r * 0.5 * (D[geo.ir, geo.jr] + D[geo.ir + 1, geo.jr]) * geo.geo_r
    P = np.abs(F_r) / np.maximum(g_r, 1e-300)
    assert P.max() > 10.0, P.max()
    print(f"\nSG faces: n {geo.n}, max cell Peclet {P.max():.1f}, uniform-field residual "
          f"{np.abs(A @ np.ones(geo.n) - (bc + adv_bc)).max():.2e}")


# ------------------------------------------------------------------ grid convergence (the actual goal)
_GRID: dict = {}


def _run_grid(dx, scheme, hours=1.0):
    """Narrow-trench Stokes run at dx with the advective scheme, cached per (dx, scheme, hours)."""
    key = (float(dx), scheme, float(hours))
    if key not in _GRID:
        p = _params(flow_solver="stokes", adv_scheme=scheme, dx_um=dx)
        _GRID[key] = run_case(D_UM, 22.0, _recipe(hours), p, f_open=F_OPEN,
                              label=f"narrow_{scheme}_dx{dx:g}", out=None)
    return _GRID[key]


def _ratio(res):
    """floor_mean / field_mean of the final frame (series metrics of the run)."""
    s = res["series"]
    return float(s["floor_mean_um"][-1]) / float(s["field_mean_um"][-1])


def test_sg_reduces_the_grid_error_of_the_floor_to_field_ratio():
    """1 h at 15 mA/cm2, G = 59 1/s: the exponentially fitted faces remove the O(dx) numerical diffusion, so
    the floor/field ratio moves less between dx 8 and dx 4 than with the upwind operator.  Also checks that
    the Cu2+ balance still closes with 'sg' (the top plane is untouched)."""
    r = {}
    for scheme in ("upwind", "sg"):
        for dx in (8.0, 4.0):
            res = _run_grid(dx, scheme)
            r[(scheme, dx)] = _ratio(res)
            if scheme == "sg":
                assert abs(res["result"]["cu_balance_rel"]) < 0.005, res["result"]["cu_balance_rel"]
                assert abs(res["result"]["cu_balance_adv_rel"]
                           - res["result"]["cu_balance_rel"]) < 1e-5
    gap_u = abs(r[("upwind", 8.0)] - r[("upwind", 4.0)])
    gap_s = abs(r[("sg", 8.0)] - r[("sg", 4.0)])
    print(f"\ngrid convergence (floor/field, 1 h): upwind dx8 {r[('upwind', 8.0)]:.5f} "
          f"dx4 {r[('upwind', 4.0)]:.5f} (gap {gap_u:.5f}) | sg dx8 {r[('sg', 8.0)]:.5f} "
          f"dx4 {r[('sg', 4.0)]:.5f} (gap {gap_s:.5f})")
    assert gap_s < gap_u, (gap_s, gap_u)


def test_upwind_scheme_is_bit_identical_to_the_default_stokes_path():
    """adv_scheme='upwind' is the legacy Stokes operator: the raw series of a 0.1 h run equal those of the
    default (no adv_scheme given) and those of 'sg' differ, so the switch is actually wired."""
    rec = _recipe(0.1)
    raws = {}
    for kw in ({}, {"adv_scheme": "upwind"}, {"adv_scheme": "sg"}):
        sim = ViaSim(D_UM, 22.0, _params(flow_solver="stokes", **kw), F_OPEN, rec)
        raws[kw.get("adv_scheme", "default")] = sim.run("adv_scheme")
    base, expl, sg = raws["default"]["series"], raws["upwind"]["series"], raws["sg"]["series"]
    assert set(base) == set(expl) == set(sg)
    for key in base:
        np.testing.assert_array_equal(np.asarray(base[key], float), np.asarray(expl[key], float),
                                      err_msg=f"series {key!r} differs from the explicit 'upwind' run")
    assert raws["default"]["steps"] == raws["upwind"]["steps"]
    assert raws["default"]["cu_influx"] == raws["upwind"]["cu_influx"]
    assert any(not np.array_equal(np.asarray(base[k], float), np.asarray(sg[k], float)) for k in base), \
        "'sg' produced exactly the upwind series: the scheme is not wired in"



