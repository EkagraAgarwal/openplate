"""MAC regression tests; lengths are cm except the read-only 2-D reference."""
import os

import numpy as np
import pytest
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from platesim.flow3d import buoyancy_force, solve_stokes3d
from platesim.trenchflow.geometry import trench_grid
from platesim.trenchflow.stokes import solve_stokes


def extrude(grid, ny):
    return np.repeat(np.where(grid.solid.T[:, None, :], -0.5, 0.5), ny, axis=1) * grid.dx_um * 1e-4


def solve_extruded(grid, ny=2, **kw):
    dx = grid.dx_um * 1e-4
    return solve_stokes3d(extrude(grid, ny), dx, (grid.z_um[0] - grid.dx_um / 2) * 1e-4,
                          grid.height_um * 1e-4, **kw)


def mass_error(result, dx):
    div = sum(np.diff(f, axis=a) for a, f in enumerate(result['faces']))
    speed = max(np.max(np.abs(f)) for f in result['faces'])
    assert np.max(np.abs(div)) / max(speed, 1e-30) < 1e-8
    assert result['info']['divergence_max'] == pytest.approx(np.max(np.abs(div)) / dx)


@pytest.mark.parametrize('solver', ['direct', 'iterative'])
@pytest.mark.parametrize('periodic', [(True, True, False), (False, True, False)])
def test_plane_couette(solver, periodic):
    dx, shear = 4e-4, 17.0
    psi = np.ones((5, 3, 12)) * dx
    # Nonperiodic x has impermeable sides, so drive along y instead.
    direction = (1., 0.) if periodic[0] else (0., 1.)
    r = solve_stokes3d(psi, dx, 0., 12 * dx, shear_rate_s=shear,
                      shear_dir=direction, periodic=periodic, solver=solver)
    exact = np.broadcast_to(shear * (np.arange(12) + .5) * dx, psi.shape)
    for a, name in enumerate(('ux', 'uy', 'uz')):
        np.testing.assert_allclose(r[name], exact * (*direction, 0.)[a], atol=1e-10, rtol=0)
    mass_error(r, dx)


@pytest.mark.parametrize('ny', [1, 2, 4])
@pytest.mark.parametrize('deposit', [0., 4.])
def test_extruded_cross_trench(ny, deposit):
    g = trench_grid(24, 16, pitch_um=48, height_um=32, dx_um=4, deposit_um=deposit)
    ref = solve_stokes(g, 23.)
    r = solve_extruded(g, ny, shear_rate_s=23.)
    scale = np.max(np.abs(ref.ux))
    errors = []
    for name in ('ux', 'uz'):
        err = np.max(np.abs(r[name] - getattr(ref, name).T[:, None, :])) / scale
        errors.append(err)
        assert err < 1e-6
    assert np.max(np.abs(r['uy'])) < 1e-12
    mass_error(r, g.dx_um * 1e-4)
    print(f'cross trench ny={ny} deposit={deposit}: ux/U={errors[0]:.3e}, uz/U={errors[1]:.3e}')


def test_along_trench_poisson():
    g = trench_grid(24, 16, pitch_um=48, height_um=32, dx_um=4)
    r = solve_extruded(g, 3, shear_rate_s=23., shear_dir=(0., 1.))
    fluid = ~g.solid
    ids = np.full(g.shape, -1)
    ids[fluid] = np.arange(fluid.sum())
    a = lil_matrix((fluid.sum(), fluid.sum()))
    b = np.zeros(fluid.sum())
    lid = 23 * g.height_um * 1e-4
    for k, i in zip(*np.where(fluid)):
        row = ids[k, i]
        for dk, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            kk, ii = k + dk, (i + di) % g.shape[1]
            if 0 <= kk < g.shape[0] and fluid[kk, ii]:
                a[row, row] += 1
                a[row, ids[kk, ii]] -= 1
            else:
                a[row, row] += 2
                if kk == g.shape[0]:
                    b[row] += 2 * lid
    expected = np.zeros(g.shape)
    expected[fluid] = spsolve(a.tocsr(), b)
    np.testing.assert_allclose(r['uy'], np.broadcast_to(expected.T[:, None, :], r['uy'].shape), atol=1e-11)
    assert max(np.max(np.abs(r[n])) for n in ('ux', 'uz')) < 1e-12
    mass_error(r, g.dx_um * 1e-4)


def test_hydrostatic_buoyancy():
    shape, dx = (6, 4, 12), 4e-4
    ct = np.ones(shape)
    ct[:, :, :4] = .5
    f = buoyancy_force(ct, .8, .2, (0, 0, -1))
    assert f[2][0, 0, 0] == pytest.approx(981 * .2 * .8 * .5)
    r = solve_stokes3d(np.ones(shape), dx, 0., shape[2] * dx, body_force=f)
    assert max(np.max(np.abs(r[n])) for n in ('ux', 'uy', 'uz')) < 1e-11
    assert r['info']['divergence_max'] < 1e-9
    # Kinematic pressure has units cm²/s² and grad(p) = acceleration.
    expected_dp = .5 * (f[2][:, :, :-1] + f[2][:, :, 1:]) * dx
    np.testing.assert_allclose(np.diff(r['p'], axis=2), expected_dp, atol=1e-12)
    for pin in r['info']['pin_cells']:
        assert r['p'][pin] == 0.


def test_extruded_buoyant_circulation():
    g = trench_grid(24, 16, pitch_um=48, height_um=32, dx_um=4)
    ct = np.ones(g.shape)
    ct[(g.z_um[:, None] < 8) & (np.abs(g.x_um[None, :] - 24) < 12)] = .5
    fx, _, fz = buoyancy_force(ct, .8, .2, (1, 0, 0))
    ref = solve_stokes(g, 0., body_force=(fx, fz))
    f3 = tuple(np.repeat(f.T[:, None, :], 2, axis=1) for f in (fx, np.zeros_like(fx), fz))
    r = solve_extruded(g, 2, body_force=f3)
    scale = np.max(np.abs(ref.ux))
    assert scale > 1e-8
    for name in ('ux', 'uz'):
        error = np.max(np.abs(r[name] - getattr(ref, name).T[:, None, :])) / scale
        print(f'buoyant {name}/U difference={error:.3e}')
        assert error < 1e-6
    assert np.max(np.abs(r['uy'])) < 1e-12
    mass_error(r, g.dx_um * 1e-4)


def test_sealed_void_and_periodic_regions():
    psi = -np.ones((10, 6, 14))
    psi[:, :, 9:] = 1
    psi[3:6, 2:4, 2:5] = 1
    # A separate pocket straddles the periodic seam (one region, not two).
    psi[[0, -1], 2:4, 2:5] = 1
    r = solve_stokes3d(psi, 4e-4, 0, 14 * 4e-4, shear_rate_s=20)
    assert r['info']['n_fluid_regions'] == 3
    assert len(r['info']['pin_cells']) == 3
    for name in ('ux', 'uy', 'uz'):
        assert np.max(np.abs(r[name][:, :, :9])) < 1e-11
    mass_error(r, 4e-4)


def test_iterative_matches_direct_3d():
    psi = np.ones((8, 7, 10))
    psi[:, :, :4] = -1
    psi[2:6, 2:5, :4] = 1
    args = (psi, 4e-4, -4 * 4e-4, 6 * 4e-4)
    direct = solve_stokes3d(*args, shear_rate_s=20, solver='direct')
    iterative = solve_stokes3d(*args, shear_rate_s=20, solver='iterative')
    for name in ('ux', 'uy', 'uz'):
        np.testing.assert_allclose(iterative[name], direct[name], atol=1e-10, rtol=1e-7)
    mass_error(iterative, 4e-4)
    print('small via:', iterative['info'])


@pytest.mark.parametrize('height_cells', [10., 10.25, 10.75])
def test_lid_inside_grid(height_cells):
    dx = 4e-4
    n = np.count_nonzero(np.arange(12) + .5 < height_cells)
    r = solve_stokes3d(np.ones((3, 2, 12)), dx, 0., height_cells * dx, shear_rate_s=10)
    np.testing.assert_allclose(r['ux'][:, :, :n], np.broadcast_to(10 * (np.arange(n) + .5) * dx, (3, 2, n)), atol=1e-10)
    assert not r['ux'][:, :, n:].any()
    mass_error(r, dx)


def test_nearly_coincident_lid_rejected():
    dx = 4e-4
    with pytest.raises(ValueError, match='lid.*centre'):
        solve_stokes3d(np.ones((3, 2, 12)), dx, 0., (8.5 + 1e-7) * dx,
                       shear_rate_s=10)


@pytest.mark.parametrize('kw', [{'dx': 0}, {'nu_cm2_s': -1}, {'periodic': (True, True, True)},
                               {'shear_dir': (0, 0)}, {'solver': 'unknown'},
                               {'body_force': (np.zeros((2, 2)),) * 3}])
def test_invalid_inputs(kw):
    args = dict(psi=np.ones((3, 2, 4)), dx=4e-4, z0=0., z_lid=16e-4)
    args.update(kw)
    with pytest.raises(ValueError):
        solve_stokes3d(**args)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PLATESIM_RUN_SLOW') != '1', reason='set PLATESIM_RUN_SLOW=1 for 48x48x40 benchmark')
def test_square_via_48():
    dx = 4e-4
    psi = -np.ones((48, 48, 40))
    psi[:, :, 20:] = 1
    psi[16:32, 16:32, 2:20] = 1
    r = solve_stokes3d(psi, dx, -20 * dx, 20 * dx, shear_rate_s=20)
    assert r['info']['solver'] == 'iterative'
    assert r['info']['converged']
    mass_error(r, dx)
    print('48x48x40 via:', r['info'])
