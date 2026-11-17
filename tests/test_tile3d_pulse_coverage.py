"""Pulse-resolved additive coverages, Phase 4 Level A (brief `pulse_coverage_brief.md`).

No team data: generic geometry/numbers only. Each test uses small grids (<60 s).
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine.params import Params, derive
from platesim.tile3d import additives3d
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.sim import run_tile

UM = 1e-4
PR = dict(t_f_ms=9., t_r_ms=.5, t_off_ms=.5, j_f_mA_cm2=18.75, j_r_mA_cm2=37.5,
          model='resolved')


def flat(nx=6, ny=6, nz=44, dx_um=4.0, z_lo_um=-16.0):
    grid = Grid3D.from_um((nx * dx_um, ny * dx_um, nz * dx_um), dx_um,
                          origin_um=(0, 0, z_lo_um))
    _, _, z = grid.meshgrid()
    return grid, z.copy()


def test_a_periodic_phase_means_vs_brute_force():
    rng = np.random.default_rng(1)
    n = 10
    taus = np.array([9e-3, 5e-4, 5e-4])
    a = 10.0 ** rng.uniform(-1.0, 1.5, size=n)
    L = 10.0 ** rng.uniform(-1.0, 2.0, size=(3, n))
    a[0] = 1e-4
    L[:, 0] = np.array([0.0, 100.0, 100.0])
    a[1] = 100.0
    L[:, 1] = np.array([5000.0, 5000.0, 5000.0])
    R = a[None, :] + L
    rtau = R * taus[:, None]
    assert rtau.min() < 2e-6 and rtau.max() > 40.0
    x_start, xbar = additives3d.periodic_phase_means(a, L, taus)
    assert x_start.shape == (3, n) and xbar.shape == (3, n)
    E = np.exp(-rtau)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(R > 0, a[None, :] / np.where(R > 0, R, 1.0), 0.0)
    x = np.zeros(n)
    for _ in range(200):
        for k in range(3):
            x = w[k] + (x - w[k]) * E[k] if taus[k] > 0 else x
    for _ in range(20000):
        if np.max(np.abs(x - x_start[0])) < 1e-12:
            break
        for k in range(3):
            x = w[k] + (x - w[k]) * E[k] if taus[k] > 0 else x
    np.testing.assert_allclose(x, x_start[0], rtol=0, atol=1e-10)
    T = float(taus.sum())
    m_tot = 2000
    for j in range(n):
        xs0 = float(x_start[0, j])
        x_cur = xs0
        num = 0.0
        for k in range(3):
            if taus[k] == 0:
                continue
            m_k = max(1, int(round(m_tot * taus[k] / T)))
            dt = taus[k] / m_k
            Rjk = float(R[k, j])
            wjk = float(w[k, j])
            Esub = np.exp(-Rjk * dt)
            xs = [x_cur]
            for _ in range(m_k):
                x_cur = wjk + (x_cur - wjk) * Esub if Rjk > 0 else x_cur
                xs.append(x_cur)
            xs = np.array(xs)
            num += float(np.trapezoid(xs, dx=dt))
        brute_mean = num / T
        cyc = float(np.dot(taus / T, xbar[:, j]))
        assert brute_mean == pytest.approx(cyc, abs=1e-6)
    np.testing.assert_allclose((taus / T) @ xbar, (taus / T) @ x_start * 0 + (taus / T) @ xbar,
                               rtol=0, atol=1e-12)


def test_b_limits():
    n = 5
    taus = np.array([1.0, 1.0, 1.0])
    a = np.full(n, 1e5)
    L = np.full((3, n), 9e5)
    _, xbar = additives3d.periodic_phase_means(a, L, taus)
    w = 1e5 / 1e6
    np.testing.assert_allclose(xbar, w, rtol=0, atol=2e-6)
    x0, xb0 = additives3d.periodic_phase_means(np.zeros(3), np.zeros((3, 3)),
                                               np.array([0.01, 0.005, 0.005]))
    np.testing.assert_array_equal(x0, np.zeros((3, 3)))
    np.testing.assert_array_equal(xb0, np.zeros((3, 3)))
    x = 0.37
    for k in range(3):
        x = x  # R=0 -> identity
    assert x == pytest.approx(0.37)
    T = 0.01
    aa = np.array([2.0, 2.0])
    Lf = np.array([3.0, 0.5])
    L1 = np.array([[3.0, 0.5], [10.0, 7.0], [20.0, 1.0]])
    xs, xb = additives3d.periodic_phase_means(aa, L1, np.array([T, 0.0, 0.0]))
    np.testing.assert_allclose(xs[0], aa / (aa + Lf), rtol=0, atol=1e-12)
    np.testing.assert_allclose(xb[0], aa / (aa + Lf), rtol=0, atol=1e-12)


def test_c_mean_is_bit_identical_to_omitting_it():
    grid, psi = flat(nx=4, ny=4, nz=30, dx_um=8.0)
    kw = dict(delta_um=64.0, n_frames=2)
    base = run_tile(psi, grid, 15.0, 30.0, params=Params(additives="acc", suppressor=False),
                    pulse=PR, **kw)
    mean = run_tile(psi, grid, 15.0, 30.0,
                    params=Params(additives="acc", suppressor=False, pulse_coverage="mean"),
                    pulse=PR, **kw)
    np.testing.assert_array_equal(mean["psi"], base["psi"])
    assert set(mean["series"]) == set(base["series"])
    for key, values in base["series"].items():
        np.testing.assert_array_equal(np.asarray(mean["series"][key]), np.asarray(values),
                                      err_msg=key)


def _trench_small():
    from platesim.tile3d import levelset3d as ls
    grid = Grid3D.from_um((192.0, 16.0, 160.0), 8.0, origin_um=(-96.0, 0.0, -64.0))
    x, _, z = grid.meshgrid()
    a, d = 32.0 * UM, 48.0 * UM
    psi = np.where(z > 0.0, z, np.where(np.abs(x) < a, np.minimum(z + d, a - np.abs(x)), z))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    field_mask = np.abs(x[:, :, 0]) >= 0.75 * 0.5 * 192.0 * UM
    return psi, grid, field_mask


def test_d_forward_only_matches_mean():
    psi0, grid, field = _trench_small()
    dc = dict(t_f_ms=10., t_r_ms=0., t_off_ms=0., j_f_mA_cm2=15., j_r_mA_cm2=0.,
              model='resolved')
    kw = dict(delta_um=32.0, field_mask=field, n_frames=2)
    p_mean = Params(additives="acc", suppressor=False, pulse_coverage="mean")
    p_res = Params(additives="acc", suppressor=False, pulse_coverage="resolved",
                   k_strip_s=0.0, k_strip_a=0.0)
    base = run_tile(psi0, grid, 15.0, 20.0, params=p_mean, pulse=dc, **kw)
    res = run_tile(psi0, grid, 15.0, 20.0, params=p_res, pulse=dc, **kw)
    from platesim.tile3d.sim import top_height
    x = grid.centers()[0]
    floor = np.abs(x) <= 16 * UM
    fld = np.abs(x) >= 0.3 * 0.5 * 192 * UM
    g0 = (top_height(base["psi"], grid) - top_height(psi0, grid)) / UM
    g1 = (top_height(res["psi"], grid) - top_height(psi0, grid)) / UM
    for mask in (floor, fld):
        assert float(np.mean(g1[mask])) == pytest.approx(float(np.mean(g0[mask])), rel=1e-6)


def test_e_stripping_lowers_theta_and_return_raises_alpha():
    p0 = Params(additives="acc", suppressor=False, pulse_coverage="resolved",
                k_strip_a=0.0, strip_return_a=1.0)
    p1 = Params(additives="acc", suppressor=False, pulse_coverage="resolved",
                k_strip_a=5000.0, strip_return_a=1.0)
    p2 = Params(additives="acc", suppressor=False, pulse_coverage="resolved",
                k_strip_a=5000.0, strip_return_a=0.0)
    outs = []
    for p in (p0, p1, p2):
        grid, psi = flat(nx=4, ny=4, nz=30, dx_um=8.0)
        solver = FieldSolver(p, 22.0, pulse=PR)
        cov = additives3d.coverage_init(psi.shape, with_acc=True)
        out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, 15e-3,
                           periodic=grid.periodic, cov=cov, tol=2e-4, maxit=60)
        assert out["info"]["pulse"]["converged"]
        outs.append((grid, psi, cov, out))
    th0 = float(outs[0][3]["info"]["additives"]["theta_a"][2])
    th1 = float(outs[1][3]["info"]["additives"]["theta_a"][2])
    assert th1 < th0
    a1 = float(outs[1][3]["info"]["additives"]["a_front"][2])
    a2 = float(outs[2][3]["info"]["additives"]["a_front"][2])
    assert a2 < a1
    for _, _, _, out in outs[1:]:
        assert abs(out["info"]["additives"]["a_balance_rel"]) < 1e-3
    f0 = abs(outs[0][3]["info"]["pulse"]["faraday_error_rel"])
    f1 = abs(outs[1][3]["info"]["pulse"]["faraday_error_rel"])
    f2 = abs(outs[2][3]["info"]["pulse"]["faraday_error_rel"])
    assert f0 < 1e-2 and f1 < 1e-2 and f2 < 1e-2
    assert max(f0, f1, f2) / max(min(f0, f1, f2), 1e-12) < 1e3
    for _, _, cov, _ in outs:
        pc = cov["theta_s"], cov["theta_a"]
        assert all(np.all(np.isfinite(v)) for v in pc)


def test_f_bounds():
    p = Params(additives="acc", suppressor=False, pulse_coverage="resolved",
               k_strip_a=5000.0, strip_return_a=0.0)
    grid, psi = flat(nx=4, ny=4, nz=30, dx_um=8.0)
    solver = FieldSolver(p, 22.0, pulse=PR)
    cov = additives3d.coverage_init(psi.shape, with_acc=True)
    out = solver.solve(psi, grid.dx, grid.origin[2], 64 * UM, 15e-3,
                       periodic=grid.periodic, cov=cov, tol=2e-4, maxit=60)
    for key in ("theta_cl", "theta_s", "theta_a"):
        v = cov[key]
        assert np.all(np.isfinite(v)) and v.min() >= 0.0 and v.max() <= 1.0
    a = cov[additives3d.ACC_KEY]
    assert np.all(np.isfinite(a)) and a.min() >= 0.0
    assert out["info"]["additives"]["pulse_coverage"]["model"] == "resolved Level A"
    assert "note" in out["info"]["additives"]["pulse_coverage"]


def test_param_validation():
    for bad in (dict(pulse_coverage="bad"), dict(k_strip_s=-1.0), dict(k_strip_a=-1.0),
                dict(strip_return_a=-0.1), dict(strip_return_a=1.1)):
        with pytest.raises(ValueError):
            derive(Params(**bad), 22.0)
