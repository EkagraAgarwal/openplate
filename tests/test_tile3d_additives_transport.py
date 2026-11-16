"""Additives in the 3-D tile engine, Phase 2: the transported accelerator (plan 3.2 tests 4-6).

What is covered (3-D additives port, Phase-2 scope):
  * test 1 -- additives off is unchanged: ``additives='cl_ndr'`` still reproduces the pre-change reference
    of the phase-1 suite (rel 1e-6, re-checked here on one case) and ``additives='acc'`` with an empty
    accelerator bath (``c_acc = 0``) is *array-equal* to ``cl_ndr`` -- the new mode contributes nothing
    when it is off;
  * test 2 -- a flat surface has the analytic 1-D steady profile of the accelerator field with a surface
    consumption sink: laterally uniform to 1e-9, linear above the interface band, and equal to
    ``alpha(z) = (D + q z)/(D + q delta)`` with ``q = Gamma_acc k_acc (1 - theta_a)`` (3 %);
  * test 6 -- transport closure: bulk-plane influx = surface consumption (|a_balance_rel| < 1e-3) on the
    flat slab and on a trench;
  * test 8 -- bounds: 0 <= theta_a <= 1 and 0 <= alpha <= 1 over current densities up to the mass-transfer
    limit, with the same exponential Langmuir kernel as the 2-D accelerator;
  * geometry trend -- a deep narrow trench depletes the floor accelerator more than a wide open one, and
    ``run_tile`` with ``additives='acc'`` lowers floor/field relative to the additive-free run (the sign
    of the mechanism the Step-0 probe asked for);
  * wiring -- series/frames/summary keys, the 2-D engine rejecting the 3-D-only mode, 'full' still
    raising, and the new Params validation.

No team data and no fitted constant is used anywhere: every additive number is an uncalibrated [J]
default from ``Params``/``AdditiveSet``, so the assertions are exact identities, analytic relations,
bounds or internal-consistency checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine import additives as add2d
from platesim.feature.engine.params import ADDITIVES_MODES, Params, derive
from platesim.tile3d import additives3d
from platesim.tile3d import levelset3d as ls
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.result import TileResult
from platesim.tile3d.sim import run_tile, top_height

UM = 1e-4


def flat(nx=6, ny=6, nz=44, dx_um=4.0, z_lo_um=-16.0):
    grid = Grid3D.from_um((nx * dx_um, ny * dx_um, nz * dx_um), dx_um, origin_um=(0, 0, z_lo_um))
    _, _, z = grid.meshgrid()
    return grid, z.copy()


def trench(half_width_um=32.0, depth_um=48.0, pitch_um=192.0, ny=2, dx_um=8.0, z_um=96.0, z_lo_um=-64.0):
    grid = Grid3D.from_um((pitch_um, ny * dx_um, z_um - z_lo_um), dx_um,
                          origin_um=(-0.5 * pitch_um, 0.0, z_lo_um))
    x, _, z = grid.meshgrid()
    a, d = half_width_um * UM, depth_um * UM
    psi = np.where(z > 0.0, z, np.where(np.abs(x) < a, np.minimum(z + d, a - np.abs(x)), z))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    field_mask = np.abs(x[:, :, 0]) >= 0.75 * 0.5 * pitch_um * UM
    return psi, grid, field_mask


def flat_solve(j_mA_cm2=15.0, p=None, tol=1e-7, maxit=300):
    grid, psi = flat()
    p = Params(additives="acc", suppressor=False) if p is None else p
    solver = FieldSolver(p, 22.0)
    cov = additives3d.coverage_init(psi.shape, with_acc=True)
    out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, j_mA_cm2 * 1e-3, periodic=grid.periodic,
                       cov=cov, tol=tol, maxit=maxit)
    return grid, psi, solver, cov, out


# ------------------------------------------------------------------ test 1: unchanged when off
# Phase-1 reference (main @ 89e6a1d, flat 6 x 6 x 44, dx 4, 1 h, 15 mA/cm2): re-checked here at rel 1e-6
# so this file fails if the Phase-2 edit moves the additives-off path.
REF_FLAT = dict(V=-0.11049324680402908, psi_sum=8.261747452456962, faraday=0.0014030243402005649)


def test_cl_ndr_still_reproduces_the_additive_free_reference():
    grid, psi = flat()
    res = run_tile(psi, grid, 15.0, 3600.0, delta_um=64.0, params=Params(suppressor=False), n_frames=3)
    assert res["series"]["V"][-1] == pytest.approx(REF_FLAT["V"], rel=1e-6)
    assert float(np.sum(res["psi"])) == pytest.approx(REF_FLAT["psi_sum"], rel=1e-6)
    assert float(res["summary"]["faraday_error_rel"]) == pytest.approx(REF_FLAT["faraday"], rel=1e-4, abs=1e-9)


def test_acc_with_an_empty_bath_is_array_equal_to_cl_ndr():
    """c_acc = 0 -> theta_a = 0 exactly -> Fcov = 1 exactly: the whole run is the 'cl_ndr' one."""
    psi, grid, field = trench()
    kw = dict(delta_um=32.0, field_mask=field, n_frames=2)
    base = run_tile(psi, grid, 15.0, 60.0, params=Params(additives="cl_ndr", suppressor=False), **kw)
    zero = run_tile(psi, grid, 15.0, 60.0, params=Params(additives="acc", suppressor=False, c_acc=0.0), **kw)
    np.testing.assert_array_equal(zero["psi"], base["psi"])
    for key, values in base["series"].items():
        np.testing.assert_array_equal(np.asarray(zero["series"][key], dtype=float),
                                      np.asarray(values, dtype=float), err_msg=key)
    assert float(np.max(zero["fields"]["theta_a"])) == 0.0
    # the transport solve still runs (the bath is empty, not the equation switched off), and the field
    # stays inside the maximum principle 0 <= alpha <= 1
    assert np.all(np.isfinite(zero["fields"]["a_acc"]))
    assert 0.0 <= float(np.min(zero["fields"]["a_acc"])) and float(np.max(zero["fields"]["a_acc"])) <= 1.0


def test_acc_off_kappa_is_exactly_zero_and_theta_a_equals_the_langmuir_steady_state():
    """The quasi-steady coverage is the exact d theta/dt = 0 root of the 2-D accelerator balance."""
    grid, psi, solver, cov, out = flat_solve()
    p, ph = solver.p, solver.ph
    ii, jj, kk = additives3d.front_cells(psi)
    th = cov["theta_a"][ii, jj, kk]
    i_loc = np.ravel(out["i"][ii, jj, kk])
    a_front = np.ravel(cov[additives3d.ACC_KEY][ii, jj, kk])
    np.testing.assert_allclose(np.ravel(th),
                               additives3d.accelerator_steady_state(i_loc, a_front, p, ph.Mfac),
                               rtol=0, atol=5e-7)   # stored value is the relaxed fixed point (tol 1e-7)
    # equals the 2-D exact-step fixed point of AdditiveSet.accelerator_step (no curvature, no desorption)
    s = add2d.AdditiveSet(c_a=p.c_acc, k_a=p.k_acc, k_inc_a=p.k_inc_a)
    for idx in range(0, i_loc.size, max(i_loc.size // 4, 1)):
        v = ph.Mfac * i_loc[idx]
        theta = 0.0
        for _ in range(2000):
            theta = float(s.accelerator_step(theta, 0.5, c_a=p.c_acc * a_front[idx], v=v, kappa=0.0))
        assert float(th.ravel()[idx]) == pytest.approx(theta, abs=1e-7)
    # i0 multiplier is exactly the accelerator factor with no suppressor coverage
    np.testing.assert_allclose(out["i0_factor"][ii, jj, kk], 1.0 + p.b_acc * th, rtol=1e-14)


# ------------------------------------------------------- test 2: flat surface -> analytic 1-D profile
def test_flat_surface_accelerator_field_is_the_analytic_1d_profile():
    grid, psi, solver, cov, out = flat_solve()
    p, ph = solver.p, solver.ph
    assert out["info"]["converged"] and out["info"]["additives"]["converged"]
    ii, jj, kk = additives3d.front_cells(psi)
    kw = int(kk[0, 0])
    assert np.ptp(kk) == 0
    alpha = cov[additives3d.ACC_KEY]
    # 1. laterally uniform over every (x, y) column
    assert np.ptp(alpha[:, :, kw]) < 1e-9
    # 2. the wall flux is the surface sink: J = q alpha_wall, q = Gamma_acc k_acc (1 - theta_a)
    th_wall = float(cov["theta_a"][ii[0, 0], jj[0, 0], kw])
    q = p.Gamma_acc * p.k_acc * (1.0 - th_wall)
    a_wall = float(alpha[0, 0, kw])
    kt, zc, z_plane = out["kt"], grid.centers()[2], 64 * UM
    d_top = z_plane - zc[kt]
    Hs_top = 0.5 * (1 + np.tanh(min(zc[kt] / (p.eps_fac * grid.dx), 30.0)))
    J_plane = ph.D_acc * Hs_top / (grid.dx * d_top) * (1.0 - float(alpha[0, 0, kt])) * grid.dx
    assert J_plane == pytest.approx(q * a_wall, rel=5e-2)
    # 3. the profile above the diffuse band is the closed-form steady solution
    #    alpha(z) = (D + q z)/(D + q delta)  (linear in z, zero at z = -D/q)
    z = zc[kw + 2:kt + 1]
    a = alpha[0, 0, kw + 2:kt + 1]
    slope, intercept = np.polyfit(z, a, 1)
    np.testing.assert_allclose(a, (ph.D_acc + q * z) / (ph.D_acc + q * z_plane), rtol=3e-2)
    assert slope > 0.0
    assert intercept / slope == pytest.approx(ph.D_acc / q, rel=3e-2)      # zero of the line at -D/q
    assert intercept == pytest.approx(ph.D_acc / (ph.D_acc + q * z_plane), rel=3e-2)
    # 4. and the flat surface keeps the accelerator field independent of x and y at every height
    for k in range(kw, kt + 1):
        assert np.ptp(alpha[:, :, k]) < 1e-9 * max(1.0, float(np.max(alpha[:, :, k])))


def test_flat_accelerator_field_is_independent_of_dx():
    """The 1-D steady solution is a property of (D, q, delta), not of the grid: on each grid the front
    cell matches the closed form evaluated at that cell's own height (the diffuse-interface sample point
    moves by half a cell between dx 4 and dx 8, which is the only residual difference)."""
    vals = {}
    for dx in (4.0, 8.0):
        nz = int(round(176.0 / dx))
        grid, psi = flat(nz=nz, dx_um=dx)
        p = Params(additives="acc", suppressor=False)
        solver = FieldSolver(p, 22.0)
        cov = additives3d.coverage_init(psi.shape, with_acc=True)
        out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, 15e-3, periodic=grid.periodic, cov=cov,
                           tol=1e-7, maxit=300)
        ii, jj, kk = additives3d.front_cells(psi)
        z_wall = grid.centers()[2][int(kk[0, 0])]
        vals[dx] = (float(np.mean(cov[additives3d.ACC_KEY][ii, jj, kk])),
                    float(np.mean(cov["theta_a"][ii, jj, kk])))
        q = p.Gamma_acc * p.k_acc * (1.0 - vals[dx][1])
        analytic = (solver.ph.D_acc + q * z_wall) / (solver.ph.D_acc + q * 64 * UM)
        assert vals[dx][0] == pytest.approx(analytic, rel=2.5e-2), (dx, vals[dx][0], analytic)
    assert vals[4.0][1] == pytest.approx(vals[8.0][1], abs=1e-2)
    assert vals[4.0][0] == pytest.approx(vals[8.0][0], abs=4e-2)


# ------------------------------------------------------------- test 6: transport closure
def test_transport_closure_on_the_flat_slab():
    grid, psi, solver, cov, out = flat_solve()
    info = out["info"]["additives"]
    assert abs(info["a_balance_rel"]) < 1e-3
    assert info["acc_sink_mol_cm2_s"] > 0.0 and info["acc_influx_mol_cm2_s"] > 0.0


def test_transport_closure_on_a_trench():
    psi, grid, field = trench()
    p = Params(additives="acc", suppressor=False)
    solver = FieldSolver(p, 22.0)
    cov = additives3d.coverage_init(psi.shape, with_acc=True)
    out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3, periodic=grid.periodic, cov=cov,
                       tol=1e-7, maxit=300)
    add = out["info"]["additives"]
    assert out["info"]["converged"] and add["converged"]
    assert abs(add["a_balance_rel"]) < 1e-3, add["a_balance_rel"]
    # the sink is what the coverage says it is: Gamma_acc k_inc_a v theta_a (net irreversible loss)
    assert add["acc_sink_mol_cm2_s"] > 0.0
    assert abs(out["info"]["cu_balance_rel"]) < 1e-5


# ------------------------------------------------------------------- test 8: bounds / maximum principle
@pytest.mark.parametrize("frac", [0.01, 0.1, 0.5, 0.9, 0.999, 1.02])
def test_accelerator_bounds_over_current(frac):
    grid, psi = flat(nz=44, dx_um=8.0)
    p = Params(additives="acc")
    solver = FieldSolver(p, 22.0)
    i_lim = 2 * solver.ph.D_c * p.c_bulk / (64 * UM)
    cov = additives3d.coverage_init(psi.shape, theta0=0.5, with_acc=True)
    out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, frac * i_lim, periodic=grid.periodic,
                       cov=cov, tol=1e-3, maxit=200)
    th_a = cov["theta_a"]
    alpha = cov[additives3d.ACC_KEY]
    assert np.all(np.isfinite(th_a)) and th_a.min() >= 0.0 and th_a.max() <= 1.0
    assert np.all(np.isfinite(alpha)) and alpha.min() >= 0.0 and alpha.max() <= 1.0 + 1e-12
    assert np.all(np.isfinite(out["i0_factor"])) and out["i0_factor"].min() > 0.0
    # an empty surface has no acceleration: 1 + b_acc * 0 == 1
    assert float(additives3d.coverage_factor(np.zeros(2), Params(suppressor=False)).max()) == 1.0


def test_accelerator_kernel_is_the_2d_langmuir_form_and_bounded_for_any_dt():
    p = Params(additives="acc")
    s = add2d.AdditiveSet(c_a=p.c_acc, k_a=p.k_acc, k_inc_a=p.k_inc_a)
    mfac = 3.7e-5
    for dt in (1e-3, 1.0, 120.0, 436.0):
        for v in (0.0, 1e-7, 5.5e-7, 1e-5):
            score = float(s.accelerator_step(0.4, dt, c_a=p.c_acc, v=v, kappa=0.0))
            assert 0.0 <= score <= 1.0
            th_ss = float(np.ravel(additives3d.accelerator_steady_state(
                np.array([v / mfac]), np.array([1.0]), p, mfac))[0])
            assert 0.0 <= th_ss <= 1.0
            # the quasi-steady value is the fixed point of the 2-D exact step to round-off
            th = 0.0
            for _ in range(4000):
                th = float(s.accelerator_step(th, 0.5, c_a=p.c_acc, v=v, kappa=0.0))
            assert abs(th - th_ss) < 1e-6      # exponential approach, dt = 0.5 s, 4000 steps
    # zero bath -> exactly zero coverage; zero current -> the pure Langmuir limit
    assert float(additives3d.accelerator_steady_state(0.01, 1.0, Params(additives="acc", c_acc=0.0),
                                                      3.7e-5)) == 0.0
    th0 = additives3d.accelerator_steady_state(0.0, 1.0, p, 3.7e-5)
    assert float(th0) == pytest.approx(p.k_acc * p.c_acc / (p.k_acc * p.c_acc + p.k_des_a), rel=1e-12)


# --------------------------------------------- geometry trend: narrow depletes more than wide
def scaled_trench(width_um, depth_um=48.0, dx_um=8.0, ny=2, delta_um=64.0):
    """Public synthetic trench, pitch = width / 0.2, no growth (as-etched), one field solve."""
    pitch_um = width_um / 0.2
    nx = int(round(pitch_um / dx_um))
    nx += nx % 2
    grid = Grid3D((nx, ny, int(round((depth_um + delta_um + 32.0) / dx_um))), dx_um * UM,
                  (-0.5 * nx * dx_um * UM, 0.0, -(depth_um + dx_um) * UM))
    x, _, z = grid.meshgrid()
    half = 0.5 * width_um * UM
    psi = np.where(z > 0.0, z, np.where(np.abs(x) < half,
                                       np.minimum(z + depth_um * UM, half - np.abs(x)), z))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    field_mask = np.abs(x[:, :, 0]) >= (half + 0.3 * pitch_um * UM)
    floor_mask = np.abs(x[:, :, 0]) <= 0.25 * width_um * UM
    return psi, grid, field_mask, floor_mask


def _front_alpha(width_um, depth_um=48.0, p=None):
    psi, grid, field_mask, floor_mask = scaled_trench(width_um, depth_um)
    p = Params(additives="acc", suppressor=False) if p is None else p
    solver = FieldSolver(p, 22.0)
    cov = additives3d.coverage_init(psi.shape, with_acc=True)
    out = solver.solve(psi, grid.dx, grid.origin[2], 64.0 * UM, 11.6e-3, periodic=grid.periodic,
                       cov=cov, tol=1e-8, maxit=300)
    ii, jj, kk = additives3d.front_cells(psi)
    alpha = cov[additives3d.ACC_KEY][ii, jj, kk]
    return (float(np.mean(alpha[~field_mask])), float(np.mean(alpha[field_mask])),
            float(np.mean(cov["theta_a"][ii, jj, kk][~field_mask])),
            float(np.mean(cov["theta_a"][ii, jj, kk][field_mask])), out)


def test_narrow_trench_depletes_the_floor_accelerator_more_than_a_wide_one():
    """The transport-limited differential: the floor of a deep narrow trench is more starved than the
    field, and a wide open trench shows much less depletion (plan 0.2's required width specificity)."""
    rows = {}
    for width in (96.0, 480.0):
        rows[width] = _front_alpha(width)
    for width, (a_floor, a_field, th_floor, th_field, out) in rows.items():
        assert a_floor < a_field, (width, a_floor, a_field)             # floor is depleted
        assert 0.0 <= a_floor and a_field <= 1.0
        assert 0.0 <= th_floor <= 1.0 and 0.0 <= th_field <= 1.0
        assert abs(out["info"]["additives"]["a_balance_rel"]) < 1e-3
    dep_narrow = 1.0 - rows[96.0][0] / rows[96.0][1]
    dep_wide = 1.0 - rows[480.0][0] / rows[480.0][1]
    assert dep_narrow > 2.0 * max(dep_wide, 1e-6), (dep_narrow, dep_wide)


def test_acc_lowers_floor_over_field_relative_to_the_additive_free_run():
    """Sign check of the mechanism: a floor-depleted accelerator reduces the floor/field ratio."""
    psi0, grid, field = trench()
    kw = dict(delta_um=32.0, params=Params(suppressor=False), field_mask=field, n_frames=2)
    base = run_tile(psi0, grid, 10.0, 0.15 * 3600.0, **kw)
    acc = run_tile(psi0, grid, 10.0, 0.15 * 3600.0,
                   params=Params(additives="acc", suppressor=False), field_mask=field, n_frames=2)
    x = grid.centers()[0]
    floor, fld = np.abs(x) <= 16 * UM, np.abs(x) >= 0.3 * 0.5 * 192 * UM

    def ratio(res):
        grown = (top_height(res["psi"], grid) - top_height(psi0, grid)) / UM
        return float(np.mean(grown[floor]) / np.mean(grown[fld]))

    assert ratio(acc) < ratio(base)
    assert acc["series"]["theta_a_floor"][-1] < acc["series"]["theta_a_field"][-1]
    assert acc["series"]["a_floor"][-1] < acc["series"]["a_field"][-1]
    assert acc["summary"]["additives"]["uncalibrated"] is True
    assert "additives" in acc["summary"] and "additives" not in base["summary"]


# -------------------------------------------------------------------------- wiring / I/O / validation
def test_acc_series_frames_and_summary(tmp_path):
    grid, psi = flat(nz=44, dx_um=8.0)
    res = run_tile(psi, grid, 15.0, 120.0, delta_um=64.0, params=Params(additives="acc"),
                   run_dir=tmp_path / "acc", n_frames=2)
    result = TileResult(res.run_dir)
    for key in ("theta_a_floor", "theta_a_field", "a_floor", "a_field", "theta_s_field"):
        assert key in result.series and len(result.series[key]) == len(result.times_s)
    for key in ("theta_s", "theta_cl", "theta_a", "a_acc"):
        field = result.field(key)
        assert field is not None and field.shape == grid.shape
    assert float(result.field("a_acc").max()) <= 1.0
    summary = res.manifest["summary"]["additives"]
    assert summary["uncalibrated"] is True
    for key in ("a_balance_rel", "a_min_front", "acc_change", "theta_a"):
        assert key in summary


def test_modes_and_parameter_validation():
    assert "acc" in ADDITIVES_MODES
    assert derive(Params(additives="acc"), 22.0).D_acc > 0.0
    assert derive(Params(additives="cl_ndr"), 22.0).D_acc == 0.0
    for bad in (dict(D_acc25=0.0), dict(D_acc25=-1.0), dict(Gamma_acc=0.0), dict(k_inc_a=-1.0),
                dict(k_des_a=-1.0), dict(c_acc=-1e-9)):
        with pytest.raises(ValueError):
            derive(Params(**bad), 22.0)
    # 'acc' is the 3-D mode only; the 2-D engine refuses it instead of silently running legacy
    from platesim.feature.engine.recipes import dc
    from platesim.feature.engine.sim import ViaSim
    with pytest.raises(NotImplementedError, match="acc"):
        ViaSim(227.0, 22.0, Params(additives="acc"), 0.2, dc(5.0, 0.1))
    # 'full' is still phases 3-4
    psi, grid, _ = trench()
    with pytest.raises(NotImplementedError, match="full"):
        run_tile(psi, grid, 10.0, 1.0, params=Params(additives="full"), n_frames=2)


def test_acc_requires_the_bulk_field_and_a_diffusivity():
    psi, grid, _ = trench()
    cov = additives3d.coverage_init(psi.shape)                     # no 'a'
    with pytest.raises(ValueError, match="a"):
        FieldSolver(Params(additives="acc")).solve(psi, grid.dx, grid.origin[2], 64 * UM, 10e-3,
                                                   periodic=grid.periodic, cov=cov)


def test_acc_runs_with_flow_advection():
    """Optional advection by the resolved flow: the neutral accelerator uses the same MAC faces."""
    psi, grid, field = trench()
    kw = dict(delta_um=32.0, params=Params(additives="acc", suppressor=False),
              field_mask=field, n_frames=2, flow=dict(shear_rate_s=10.0))
    res = run_tile(psi, grid, 10.0, 60.0, **kw)
    a = res["fields"]["a_acc"]
    assert np.all(np.isfinite(a)) and a.min() >= 0.0 and a.max() <= 1.0
    assert abs(res["summary"]["additives"]["a_balance_rel"]) < 1e-2
    assert res["summary"]["additives"]["converged"]
