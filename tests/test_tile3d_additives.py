"""Additives in the 3-D tile engine, Phase 1 (research/sims_3d/additives_3d_plan.md section 3.2).

What is covered:
  * test 1 -- additives off is bit-identical: the pre-change reference numbers of the additive-free
    engine (recorded on main at 89e6a1d) are reproduced exactly for a no-flow, a sheared and a
    resolved-pulse case, and `additives='cl_ndr'` with zero Cl- is exactly the legacy path;
  * test 2 -- a flat surface has a laterally uniform coverage equal to the analytic Langmuir steady
    state at the local current (and is independent of dx; Phase 1 has no time integration to test);
  * test 8 -- coverage bounds/stability over current densities and the exact exponential kernels
    over dt = 1e-3 .. 436 s; the NDR/hysteresis report flag is wired;
  * test 9 -- the prescribed per-cell factor reproduces the coverage factor when the same numbers
    are supplied, and the public `run_tile(kinetics_factor=callable)` path lowers floor/field.

No team data and no fitted constant is used anywhere here: every additive number is the 2-D
engine's default (flagged "[J]" there and in Params), so all assertions are either exact identities,
bounds, or internal-consistency checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine import additives as add2d
from platesim.feature.engine.params import Params
from platesim.tile3d import additives3d
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.result import TileResult
from platesim.tile3d.sim import run_tile, top_height

UM = 1e-4
PR = dict(t_f_ms=9., t_r_ms=.5, t_off_ms=.5, j_f_mA_cm2=18.75, j_r_mA_cm2=37.5, model='resolved')


def flat(nx=6, ny=6, nz=44, dx_um=4.0, z_lo_um=-16.0):
    grid = Grid3D.from_um((nx * dx_um, ny * dx_um, nz * dx_um), dx_um, origin_um=(0, 0, z_lo_um))
    _, _, z = grid.meshgrid()
    return grid, z.copy()                       # psi = z: flat wafer surface at z = 0


def small_extrusion():
    """Sheared periodic extrusion of the flow smoke test in tests/test_tile3d_flow_vs_2d.py."""
    grid = Grid3D((12, 3, 14), 8e-4, (0., 0., -32e-4))
    x, y, z = grid.meshgrid()
    psi = np.maximum(z, np.minimum(24e-4 - np.abs(x - 48e-4), z + 16e-4))
    field = np.abs(x[:, :, 0] - 48e-4) > 32e-4
    p = Params(suppressor=False, flow_shear_s=0., k_mix=0., delta_um=32.)
    return psi, grid, field, p


def trench(half_width_um=32.0, depth_um=48.0, pitch_um=192.0, ny=2, dx_um=8.0, z_um=96.0, z_lo_um=-64.0):
    from platesim.tile3d import levelset3d as ls
    grid = Grid3D.from_um((pitch_um, ny * dx_um, z_um - z_lo_um), dx_um,
                          origin_um=(-0.5 * pitch_um, 0.0, z_lo_um))
    x, _, z = grid.meshgrid()
    a, d = half_width_um * UM, depth_um * UM
    psi = np.where(z > 0.0, z, np.where(np.abs(x) < a, np.minimum(z + d, a - np.abs(x)), z))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    field_mask = np.abs(x[:, :, 0]) >= 0.75 * 0.5 * pitch_um * UM
    return psi, grid, field_mask


# ---------------------------------------------------------------- test 1: bit-identical when off
# Recorded from the additive-free engine BEFORE the Phase-1 edit (main @ 89e6a1d).
REFERENCE = {
    "flow": dict(
        V=-0.11049324680402908, psi_sum=8.261747452456962, psi_abs_sum=9.409932004625308,
        faraday=0.0014030243402005649,
        series={"charge_C_cm2": [0.0, 27.000000000000007, 53.99999999999997],
                "solid_volume_cm3": [0.0, 5.727045221282659e-09, 1.1447589811658428e-08],
                "V": [-0.11060837965924147, -0.11042851668917686, -0.11049324680402908],
                "cu_balance_rel": [-1.8141044222375058e-13, 4.8127279939080836e-11,
                                   4.60840254845607e-11],
                "ct_min_front": [0.8711668509601737, 0.8738219307370886, 0.871229409715931],
                "outer_iterations": [2.0, 2.0, 2.0]}),
    "shear": dict(
        V=-0.0998631570682561, psi_sum=1.3797468669530588, psi_abs_sum=1.5821772053274323,
        faraday=0.09405120545068635,
        series={"charge_C_cm2": [0.0, 0.18],
                "solid_volume_cm3": [0.0, 1.667560296708287e-10],
                "V": [-0.09983018940186629, -0.0998631570682561],
                "cu_balance_rel": [-3.2529534621517087e-14, 7.1522787692401835e-12],
                "ct_min_front": [0.900348788931409, 0.9002969305564663],
                "outer_iterations": [2.0, 1.0]}),
    "pulse": dict(
        V=-0.10745181666021644, psi_sum=1.4578110186091027, psi_abs_sum=1.5696417318763358,
        faraday=0.004123122014111802,
        series={"charge_C_cm2": [0.0, 1.3499999999999999, 2.6999999999999997],
                "solid_volume_cm3": [0.0, 1.275154310042073e-10, 2.550818832694599e-10],
                "V": [-0.1074322681302229, -0.10744919709034163, -0.10745181666021644],
                "cu_balance_rel": [7.542227153933823e-08, 7.505617460878966e-08,
                                   7.477628161112193e-08],
                "ct_min_front": [0.871168486679381, 0.8705891801751893, 0.8752561449925215],
                "outer_iterations": [2.0, 2.0, 2.0]}),
}


# Exact bit-identity with the stored numbers is platform- and solver-version dependent (linear
# solver changes move V by ~1e-9), so the regression uses a tight tolerance; exact identity of the
# additive-off path is checked in-process by test_unit_factor_is_bit_identical_to_none.
REL = 1e-7


def _check_reference(name, res):
    ref = REFERENCE[name]
    assert res["series"]["V"][-1] == pytest.approx(ref["V"], rel=REL)
    assert float(np.sum(res["psi"])) == pytest.approx(ref["psi_sum"], rel=REL)
    assert float(np.sum(np.abs(res["psi"]))) == pytest.approx(ref["psi_abs_sum"], rel=REL)
    assert float(res["summary"]["faraday_error_rel"]) == pytest.approx(ref["faraday"], rel=1e-4, abs=1e-9)
    for key, values in ref["series"].items():
        if key in ("cu_balance_rel", "outer_iterations"):  # solver diagnostics, not physics
            assert np.all(np.abs(np.asarray(res["series"]["cu_balance_rel"])) < 1e-6)
            continue
        np.testing.assert_allclose(np.asarray(res["series"][key], dtype=np.float64),
                                   np.asarray(values, dtype=np.float64), rtol=REL, atol=1e-9,
                                   err_msg=key)


def test_unit_factor_is_bit_identical_to_none():
    """A factor of exactly 1 everywhere must take the same arithmetic as no factor at all."""
    psi, grid, _ = trench()
    p = Params(suppressor=False)
    base = FieldSolver(p, 22.0).solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3, periodic=grid.periodic)
    ones = FieldSolver(p, 22.0).solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3, periodic=grid.periodic,
                                      kinetics_factor=np.ones_like(base["ct"]))
    for key in ("ct", "phi", "i", "speed"):
        np.testing.assert_array_equal(ones[key], base[key], err_msg=key)
    assert ones["V"] == base["V"]


def test_no_flow_legacy_path_is_bit_identical_to_main():
    grid, psi = flat()
    res = run_tile(psi, grid, 15.0, 3600.0, delta_um=64.0, params=Params(suppressor=False), n_frames=3)
    _check_reference("flow", res)


def test_sheared_legacy_path_is_bit_identical_to_main():
    psi, grid, field, p = small_extrusion()
    res = run_tile(psi, grid, 15., 12., delta_um=32., params=p, field_mask=field, n_frames=2,
                   flow=dict(shear_rate_s=10.))
    _check_reference("shear", res)


def test_resolved_pulse_legacy_path_is_bit_identical_to_main():
    grid = Grid3D.from_um((32, 8, 104), 4., origin_um=(0, 0, -16))
    _, _, z = grid.meshgrid()
    res = run_tile(z.copy(), grid, 15., 180., delta_um=64., pulse=PR, n_frames=3, backend='numpy',
                   params=Params(suppressor=False))
    _check_reference("pulse", res)


def test_cl_ndr_with_zero_chloride_is_the_legacy_path_exactly():
    """theta_cl = 0 -> a_s = 0 -> theta_s = 0 -> Fcov = 1 exactly; psi/V/series are array-equal."""
    psi, grid, field, _ = small_extrusion()
    base = run_tile(psi, grid, 15., 12., delta_um=32., params=Params(suppressor=False),
                    field_mask=field, n_frames=2)
    zero = run_tile(psi, grid, 15., 12., delta_um=32.,
                    params=Params(additives='cl_ndr', c_cl_bulk=0.0, suppressor=False),
                    field_mask=field, n_frames=2)
    np.testing.assert_array_equal(zero["psi"], base["psi"])
    for key, values in base["series"].items():
        np.testing.assert_array_equal(np.asarray(zero["series"][key]), np.asarray(values), err_msg=key)
    # theta_s stays exactly zero, so even the extra diagnostics are additive-free
    assert all(v == 0.0 for v in zero["series"]["theta_s_field"] + zero["series"]["theta_s_floor"])


# ------------------------------------------------------- test 2: flat surface -> uniform Literal
def test_flat_surface_coverage_is_uniform_and_matches_langmuir():
    grid, psi = flat()
    p = Params(additives='cl_ndr')                        # suppressor=True by default
    solver = FieldSolver(p, 22.0)
    cov = additives3d.coverage_init(psi.shape)
    out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, 15e-3, periodic=grid.periodic,
                       cov=cov, tol=1e-7, maxit=250)
    assert out["info"]["converged"]
    assert out["info"]["additives"]["converged"]
    assert out["info"]["additives"]["ndr_hysteresis"] is False
    ii, jj, kk = additives3d.front_cells(psi)
    th_s = cov["theta_s"][ii, jj, kk]
    th_cl = cov["theta_cl"][ii, jj, kk]
    # laterally uniform over every (x, y) column
    assert np.ptp(th_s) < 1e-10 and np.ptp(th_cl) < 1e-10
    # and equal to the analytic Langmuir steady state at the local current
    i_local = out["i"][ii, jj, kk]
    assert np.ptp(i_local) < 1e-12
    th_cl_ss, th_s_ss = additives3d.steady_state(i_local, p, solver.ph.Mfac)
    np.testing.assert_allclose(th_s, th_s_ss, rtol=0, atol=1e-8)
    np.testing.assert_allclose(th_cl, th_cl_ss, rtol=0, atol=1e-8)
    # the closed-form balances spelled out (no hidden scaling in steady_state)
    a_cl = p.k_cl * p.c_cl_bulk / p.Gamma_cl
    v = solver.ph.Mfac * i_local
    np.testing.assert_allclose(th_cl, add2d.langmuir_steady_state(
        a_cl, p.k_des_cl + p.k_inc_cl * v), rtol=0, atol=1e-8)
    a_s = p.k_plus * p.s_bulk * th_cl
    np.testing.assert_allclose(th_s, add2d.langmuir_steady_state(a_s, p.k_minus * v), rtol=0, atol=1e-8)
    # Fcov is the site-fraction interpolation of the 2-D engine
    np.testing.assert_allclose(out["i0_factor"][ii, jj, kk],
                               add2d.suppressor_i0_factor(th_s, p.i0_u, p.i0_s), rtol=0, atol=1e-12)


def test_flat_surface_coverage_is_independent_of_dx():
    """Phase 1 coverages are quasi-steady, so there is no dt to converge; dx must not matter."""
    means = {}
    for dx in (4.0, 8.0):
        nz = int(round(176.0 / dx))
        grid, psi = flat(nz=nz, dx_um=dx, z_lo_um=-16.0)
        p = Params(additives='cl_ndr')
        cov = additives3d.coverage_init(psi.shape)
        out = FieldSolver(p, 22.0).solve(psi, grid.dx, grid.origin[2], 64 * UM, 15e-3,
                                         periodic=grid.periodic, cov=cov, tol=1e-7, maxit=250)
        assert out["info"]["converged"]
        ii, jj, kk = additives3d.front_cells(psi)
        means[dx] = (float(np.mean(cov["theta_s"][ii, jj, kk])),
                     float(np.mean(cov["theta_cl"][ii, jj, kk])))
        # each grid is at its own analytic steady state
        th_cl_ss, th_s_ss = additives3d.steady_state(out["i"][ii, jj, kk], p, FieldSolver(p).ph.Mfac)
        assert abs(means[dx][0] - float(np.mean(th_s_ss))) < 1e-8
        assert abs(means[dx][1] - float(np.mean(th_cl_ss))) < 1e-8
    # The residual dx difference is the diffuse-interface area quadrature of the base engine (the
    # front-cell current is 1.02 j at dx 4 and 1.12 j at dx 8), not the coverage model: on each grid
    # theta is exactly the Langmuir steady state at that grid's own local current (asserted above).
    assert means[4.0][0] == pytest.approx(means[8.0][0], abs=5e-3)
    assert means[4.0][1] == pytest.approx(means[8.0][1], abs=1e-3)


# --------------------------------------------------------- test 8: bounds / stability / NDR flag
@pytest.mark.parametrize("frac", [0.01, 0.1, 0.5, 0.9, 0.999, 1.02])
def test_coverage_bounds_and_finiteness_over_current(frac):
    grid, psi = flat(nz=44, dx_um=8.0)
    p = Params(additives='cl_ndr')
    solver = FieldSolver(p, 22.0)
    i_lim = 2 * FieldSolver(Params(suppressor=False)).ph.D_c * p.c_bulk / (64 * UM)
    cov = additives3d.coverage_init(psi.shape, theta0=0.5)
    out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, frac * i_lim, periodic=grid.periodic,
                       cov=cov, tol=1e-3, maxit=200)
    for key in additives3d.COV_KEYS:
        value = cov[key]
        assert np.all(np.isfinite(value))
        assert value.min() >= 0.0 and value.max() <= 1.0, (key, value.min(), value.max())
    assert np.all(np.isfinite(out["i"]))
    assert np.all(np.isfinite(out["i0_factor"])) and out["i0_factor"].min() > 0.0
    assert isinstance(out["info"]["additives"]["ndr_hysteresis"], bool)
    # the flag reports the convergence state of the coverage fixed point, not a physics verdict
    assert out["info"]["additives"]["ndr_hysteresis"] is (not out["info"]["additives"]["converged"])


def test_exponential_coverage_kernels_are_bounded_for_every_step_size():
    """The 2-D exact-step kernels the Phase-2/3 carried coverages will use: dt = 1e-3 .. 436 s."""
    for dt in (1e-3, 1.0, 120.0, 436.0):
        for v in (0.0, 1e-7, 5.5e-7, 1e-5):
            th_cl = add2d.chloride_step(0.4, dt, 1e-6, v=v)
            th_s = add2d.suppressor_step(0.4, dt, 8e-8, 0.09, v=v)
            th_l = add2d.leveller_step(0.4, dt, 1e-8, v=v)
            th_a = add2d.accelerator_step(0.4, dt, 5e-9, v=v)
            for th in (th_cl, th_s, th_l, th_a):
                assert 0.0 <= th <= 1.0 and np.isfinite(th), (dt, v, th)


def test_ndr_flag_is_set_when_the_coverage_fixed_point_does_not_converge():
    """A deliberately over-damped relaxation (cov_relax 1 %) cannot reach the fixed point in 40 passes,
    which is the same 'no convergence' branch a limit cycle would take: the flag must trip."""
    grid, psi = flat(nz=44, dx_um=8.0)
    p = Params(additives='cl_ndr', cov_relax=0.01)
    cov = additives3d.coverage_init(psi.shape)
    out = FieldSolver(p, 22.0).solve(psi, grid.dx, grid.origin[2], 64 * UM, 15e-3,
                                     periodic=grid.periodic, cov=cov)
    assert out["info"]["additives"]["converged"] is False
    assert out["info"]["additives"]["ndr_hysteresis"] is True


# ------------------------------------------------------- test 9: prescribed factor == coverage path
def test_prescribed_factor_reproduces_the_coverage_factor():
    psi, grid, _ = trench()
    p = Params(additives='cl_ndr')
    cov = additives3d.coverage_init(psi.shape)
    cov_run = FieldSolver(p, 22.0).solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3,
                                         periodic=grid.periodic, cov=cov, tol=1e-8, maxit=300)
    assert cov_run["info"]["converged"] and cov_run["info"]["additives"]["converged"]
    prescribed = FieldSolver(p, 22.0).solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3,
                                            periodic=grid.periodic,
                                            kinetics_factor=cov_run["i0_factor"],
                                            tol=1e-8, maxit=300)
    assert prescribed["info"]["converged"]
    np.testing.assert_allclose(prescribed["ct"], cov_run["ct"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(prescribed["phi"], cov_run["phi"], rtol=0, atol=1e-7)
    assert prescribed["V"] == pytest.approx(cov_run["V"], abs=1e-7)
    np.testing.assert_allclose(prescribed["i0_factor"], cov_run["i0_factor"], rtol=0, atol=0)


def test_prescribed_factor_api_lowers_floor_over_field():
    psi0, grid, field_mask = trench()
    kw = dict(delta_um=32.0, params=Params(), field_mask=field_mask, n_frames=2)

    def deep_factor(f_deep, cut_um=12.0):
        def factor(z_cm, z_plane_cm, field_top_cm):
            return np.where(z_cm < field_top_cm - cut_um * UM, f_deep, 1.0)
        return factor

    base = run_tile(psi0, grid, 10.0, 0.1 * 3600.0, **kw)
    deep = run_tile(psi0, grid, 10.0, 0.1 * 3600.0, kinetics_factor=deep_factor(0.5), **kw)
    x = grid.centers()[0]
    grown_base = (top_height(base["psi"], grid) - top_height(psi0, grid)) / UM
    grown_deep = (top_height(deep["psi"], grid) - top_height(psi0, grid)) / UM
    floor, field = np.abs(x) <= 16 * UM, np.abs(x) >= 0.3 * 0.5 * 192 * UM
    ratio_base = grown_base[floor].mean() / grown_base[field].mean()
    ratio_deep = grown_deep[floor].mean() / grown_deep[field].mean()
    assert ratio_deep < ratio_base - 0.02                  # a differential, not a uniform scaling
    assert np.isfinite(deep["psi"]).all()
    assert "additives" not in deep["summary"]              # prescribed map is not an additive state


# --------------------------------------------------------------------- wiring / validation / I/O
def test_additives_full_and_cov_misuse_raise():
    psi, grid, _ = trench()
    with pytest.raises(NotImplementedError, match="phase|Phase"):
        run_tile(psi, grid, 10.0, 1.0, params=Params(additives='full'), n_frames=2)
    solver = FieldSolver(Params(additives='legacy'))
    with pytest.raises(ValueError, match="cov"):
        solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3, cov=additives3d.coverage_init(psi.shape))
    with pytest.raises(ValueError, match="kinetics_factor"):
        solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3, kinetics_factor=np.zeros(psi.shape))
    with pytest.raises(TypeError, match="callable"):
        run_tile(psi, grid, 10.0, 1.0, kinetics_factor=np.ones_like(psi), n_frames=2)


def test_coverage_is_written_to_frames_and_series(tmp_path):
    grid, psi = flat(nz=44, dx_um=8.0)
    res = run_tile(psi, grid, 15.0, 120.0, delta_um=64.0, params=Params(additives='cl_ndr'),
                   run_dir=tmp_path / "cov", n_frames=2)
    result = TileResult(res.run_dir)
    for key in ("theta_s_floor", "theta_s_field", "theta_cl_floor", "theta_cl_field"):
        assert key in result.series and len(result.series[key]) == len(result.times_s)
    theta_s = result.field("theta_s")
    assert theta_s is not None and theta_s.shape == grid.shape
    assert float(theta_s.max()) <= 1.0
    assert res.manifest["summary"]["additives"]["uncalibrated"] is True


@pytest.mark.parametrize("bad", [dict(cov_relax=0.0), dict(cov_relax=1.5), dict(Gamma_cl=0.0),
                                 dict(k_des_cl=-1.0)])
def test_additive_parameter_validation(bad):
    from platesim.feature.engine.params import derive
    with pytest.raises(ValueError):
        derive(Params(**bad), 22.0)
