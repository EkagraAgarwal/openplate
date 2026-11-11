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

from platesim.feature.engine.fields import StepGeom
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

