"""L1 field solve and time loop of the 3-D tile engine: flat-electrode checks (plan checks V1, V4, V8)."""
import numpy as np
import pytest

from platesim.core.constants import F
from platesim.feature.engine.params import Params
from platesim.tile3d.fields import FieldSolver
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.sim import run_tile, top_height

UM = 1e-4


def flat(nx=6, ny=6, nz=44, dx_um=4.0, z_lo_um=-16.0):
    grid = Grid3D.from_um((nx * dx_um, ny * dx_um, nz * dx_um), dx_um, origin_um=(0, 0, z_lo_um))
    _, _, z = grid.meshgrid()
    return grid, z.copy()                       # psi = z: flat wafer surface at z = 0


def test_flat_fields_match_1d_solution():
    grid, psi = flat()
    p = Params(suppressor=False)
    s = FieldSolver(p, 22.0)
    j, delta = 15e-3, 64 * UM
    out = s.solve(psi, grid.dx, grid.origin[2], delta, j, periodic=grid.periodic)
    info = out["info"]
    assert info["converged"]
    assert abs(info["current_error"]) < 1e-8
    assert abs(info["cu_balance_rel"]) < 5e-3
    # surface concentration of the 1-D stagnant layer: ct = 1 - j eff delta / (2 F D c_bulk)
    expected = 1 - j * p.efficiency * delta / (2 * F * s.ph.D_c * p.c_bulk)
    k0 = int(np.argmin(np.abs(grid.centers()[2])))
    assert out["ct"][0, 0, k0] == pytest.approx(expected, abs=0.02)
    # laterally uniform, and the ohmic drop across the layer is j delta / kappa
    assert np.ptp(out["ct"], axis=(0, 1)).max() < 1e-9
    assert -out["phi"][0, 0, k0] == pytest.approx(j * delta / s.ph.kappa, rel=0.06)


def test_limiting_current_is_approached_not_exceeded():
    grid, psi = flat()
    p = Params(suppressor=False)
    s = FieldSolver(p, 22.0)
    delta = 64 * UM
    j_lim = 2 * F * s.ph.D_c * p.c_bulk / delta
    out = s.solve(psi, grid.dx, grid.origin[2], delta, 0.9 * j_lim, periodic=grid.periodic)
    assert out["info"]["converged"]
    assert 0.02 < out["info"]["ct_min_front"] < 0.25


@pytest.mark.parametrize("frac", [0.3, 0.9])
def test_multigrid_matches_direct_on_trench(frac):
    # matrix-free MGCG (forced) and sparse LU must give the same fields on a trench tile
    dx = 8.0
    grid = Grid3D.from_um((128, 32, 256), dx, origin_um=(-64, -16, -96))
    x, _, z = grid.meshgrid()
    from platesim.tile3d import levelset3d as ls
    psi = ls.reinit(np.where(np.abs(x) < 32 * UM, z + 64 * UM, z), grid.dx, periodic=grid.periodic)
    p = Params(suppressor=False)
    j = frac * 2 * F * FieldSolver(p).ph.D_c * p.c_bulk / (64 * UM)
    outs = [FieldSolver(p, linear=m).solve(psi, grid.dx, grid.origin[2], 64 * UM, j, periodic=grid.periodic)
            for m in ("direct", "mg")]
    assert outs[1]["info"]["linear"] == "mgcg" and outs[0]["info"]["linear"] == "direct"
    assert all(o["info"]["converged"] for o in outs)
    assert np.max(np.abs(outs[0]["ct"] - outs[1]["ct"])) < 1e-6
    assert outs[1]["V"] == pytest.approx(outs[0]["V"], abs=1e-6)          # 1 uV
    assert abs(outs[1]["info"]["cu_balance_rel"]) < 1e-5


def test_flat_growth_equals_faraday():
    grid, psi = flat()
    res = run_tile(psi, grid, j_net_mA_cm2=15.0, duration_s=3600.0, delta_um=64.0,
                   params=Params(suppressor=False), n_frames=3)
    h = top_height(res["psi"], grid)
    expected_um = 15e-3 * 3600 * FieldSolver(Params(suppressor=False)).ph.Mfac / UM   # 19.85 um
    assert expected_um == pytest.approx(19.85, abs=0.05)
    assert h.mean() / UM == pytest.approx(expected_um, rel=5e-3)
    assert h.std() / UM < 0.01
    assert abs(res["summary"]["faraday_error_rel"]) < 5e-3
    assert max(abs(v) for v in res["series"]["cu_balance_rel"]) < 5e-3


def test_trench_runs_and_floor_lags_field():
    # 96 um wide, 80 um deep trench extruded along y, periodic pitch 240 um; 0.5 h at 15 mA/cm2
    dx = 8.0
    grid = Grid3D.from_um((240, 16, 320), dx, origin_um=(-120, 0, -96))
    x, _, z = grid.meshgrid()
    a, depth = 48 * UM, 80 * UM
    inside = np.abs(x) < a
    psi = np.where(inside, np.minimum(z + depth, a - np.abs(x)),
                   np.where(z > 0, np.hypot(np.maximum(a - np.abs(x), 0) * 0, z), z))
    psi = np.where(inside & (z > 0), z + 0 * x, psi)
    from platesim.tile3d import levelset3d as ls
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    res = run_tile(psi, grid, 15.0, 1800.0, delta_um=64.0, params=Params(suppressor=False), n_frames=2)
    assert abs(res["summary"]["faraday_error_rel"]) < 0.05      # dx 8 um, sharp corners: coarse on purpose
    h0, h1 = top_height(psi, grid), top_height(res["psi"], grid)
    grown = (h1 - h0)[:, 0] / UM
    xc = grid.centers()[0] / UM
    field, floor = grown[np.abs(xc) > 80].mean(), grown[np.abs(xc) < 16].mean()
    # 9.9 um of Faraday thickness per projected area is spread over 400/240 of wetted area:
    # uniform (conformal) growth would be 5.96 um, a field-only deposit 9.9 um
    assert 5.96 < field < 9.9
    assert 0.5 < floor / field < 1.0             # diffusion-limited floor lags the field
