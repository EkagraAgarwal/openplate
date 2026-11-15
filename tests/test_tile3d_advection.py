"""Cu2+ advection by the 3-D Stokes field in the L1 field solve (operators.assemble_advdiff, FieldSolver flow_faces)."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from platesim.core.constants import F
from platesim.feature.engine.params import Params
from platesim.flow3d import buoyancy_force, solve_stokes3d
from platesim.tile3d import levelset3d as ls
from platesim.tile3d import operators
from platesim.tile3d.fields import FieldSolver, _faces
from platesim.tile3d.grid import Grid3D

UM = 1e-4


def trench(ny=4, dx=8.0, a_um=32.0, depth_um=64.0, nz_um=256.0):
    grid = Grid3D.from_um((128, ny * dx, nz_um), dx, origin_um=(-64, 0, -96))
    x, _, z = grid.meshgrid()
    psi = ls.reinit(np.where(np.abs(x) < a_um * UM, z + depth_um * UM, z), grid.dx, periodic=grid.periodic)
    return grid, psi


def flow(grid, psi, z_lid, **kw):
    return solve_stokes3d(psi, grid.dx, grid.origin[2], z_lid, periodic=grid.periodic, **kw)


@pytest.mark.parametrize("lid_above_plane_um", [0.0, 96.0])
def test_uniform_field_is_preserved(lid_above_plane_um):
    # lid above the bulk plane: flow crosses the Dirichlet plane (upwinded F_top path)
    grid, psi = trench()
    z_plane = 64 * UM
    fl = flow(grid, psi, z_plane + lid_above_plane_um * UM, shear_rate_s=20.0,
              body_force=(np.zeros(grid.shape), np.zeros(grid.shape), np.zeros(grid.shape)))
    u = fl["faces"]
    kt = int(np.floor((z_plane - 0.5 * grid.dx - grid.origin[2]) / grid.dx - 0.5))
    if lid_above_plane_um:
        assert np.max(np.abs(u[2][:, :, kt + 1])) > 0 or np.max(np.abs(u[0][:, :, kt])) > 0
    Hs = np.maximum(0.5 * (1 + np.tanh(np.clip(psi[:, :, :kt + 1] / (1.5 * grid.dx), -30, 30))), 1e-6)
    hx, hy, hz = _faces(Hs, grid.periodic, top_scale=Hs[:, :, -1])
    active = psi[:, :, :kt + 1] > -3 * grid.dx
    slab = (u[0][:, :, :kt + 1], u[1][:, :, :kt + 1], u[2][:, :, :kt + 2])
    A, b = operators.assemble_advdiff(5e-6 * hx, 5e-6 * hy, 5e-6 * hz, grid.dx, slab, active=active,
                                      dirichlet={"z_hi": 1.0}, periodic=grid.periodic)
    c = spsolve(A.tocsc(), b)
    assert np.max(np.abs(c[active.ravel()] - 1.0)) < 1e-9
    # and the operator is not symmetric (the flow is really in it)
    assert abs(A - A.T).max() > 0


def test_no_flow_equals_symmetric_operator():
    grid, psi = trench()
    shape = (grid.shape[0], grid.shape[1], 20)
    k = np.ones(shape)
    kf = _faces(k, grid.periodic, top_scale=k[:, :, -1])
    zero = tuple(np.zeros_like(f) for f in kf)
    A1, b1 = operators.assemble_advdiff(*kf, grid.dx, zero, dirichlet={"z_hi": 1.0}, periodic=grid.periodic)
    A0, b0 = operators.assemble(*kf, grid.dx, dirichlet={"z_hi": 1.0}, periodic=grid.periodic)
    assert abs(A1 - A0).max() < 1e-9 * abs(A0).max()
    assert np.allclose(b1, b0)


def solve(psi, grid, j_frac=0.5, flow_faces=None, linear="auto"):
    p = Params(suppressor=False)
    s = FieldSolver(p, linear=linear)
    j = j_frac * 2 * F * s.ph.D_c * p.c_bulk / (64 * UM)
    return s.solve(psi, grid.dx, grid.origin[2], grid.origin[2] + 0 + 160 * UM, j,
                   periodic=grid.periodic, flow_faces=flow_faces)


def test_axial_flow_in_uniform_section_changes_nothing():
    # extruded along y and periodic: dc/dy = 0, so flow along the trench cannot change ct
    grid, psi = trench()
    z_plane = grid.origin[2] + 160 * UM
    base = solve(psi, grid)
    ct = base["ct"]
    body = buoyancy_force(ct, 0.63, 0.07, (0.0, -1.0, 0.0))
    fl = flow(grid, psi, z_plane, body_force=body)
    assert np.max(np.abs(fl["uy"])) > 1e-4                       # > 1 um/s along the trench
    out = solve(psi, grid, flow_faces=fl["faces"])
    assert np.max(np.abs(out["ct"] - ct)) < 1e-6
    assert abs(out["info"]["cu_balance_rel"]) < 1e-4


def test_cross_flow_changes_floor_and_balance_closes():
    grid, psi = trench()
    z_plane = grid.origin[2] + 160 * UM
    base = solve(psi, grid)
    fl = flow(grid, psi, z_plane, shear_rate_s=30.0)
    out = solve(psi, grid, flow_faces=fl["faces"])
    assert out["info"]["converged"]
    assert abs(out["info"]["cu_balance_rel"]) < 1e-4
    assert abs(out["info"]["current_error"]) < 1e-8
    # shear across the trench renews the electrolyte: the front concentration rises somewhere
    assert out["info"]["ct_min_front"] > base["info"]["ct_min_front"] + 1e-4


def _floor_over_field(res, psi0, grid):
    from platesim.tile3d.sim import top_height
    grown = (top_height(res["psi"], grid) - top_height(psi0, grid))[:, 0] / UM
    xc = grid.centers()[0] / UM
    return grown[np.abs(xc) < 16].mean() / grown[np.abs(xc) > 48].mean(), grown


def test_run_tile_flow_axial_noop_and_shear_helps():
    from platesim.tile3d.sim import run_tile
    grid, psi = trench(nz_um=320)
    kw = dict(delta_um=64.0, params=Params(suppressor=False), n_frames=2)
    base = run_tile(psi, grid, 15.0, 900.0, **kw)
    axial = run_tile(psi, grid, 15.0, 900.0, flow=dict(gravity_dir=(0.0, -1.0, 0.0), update_every=2), **kw)
    shear = run_tile(psi, grid, 15.0, 900.0, flow=dict(shear_rate_s=30.0, update_every=2), **kw)
    r0, g0 = _floor_over_field(base, psi, grid)
    ra, ga = _floor_over_field(axial, psi, grid)
    rs, _ = _floor_over_field(shear, psi, grid)
    assert max(axial["series"]["flow_umax_um_s"]) > 0.1          # buoyant flow exists along the trench
    assert np.max(np.abs(ga - g0)) < 1e-3                         # ... and changes nothing (dc/dy = 0)
    assert rs > r0 + 0.005                                        # cross shear feeds the floor
    for r in (base, axial, shear):
        assert abs(r["summary"]["faraday_error_rel"]) < 0.05


@pytest.mark.parametrize("linear", ["mg", "amg"])
def test_iterative_matches_direct_with_flow(linear):
    grid, psi = trench(ny=8)
    z_plane = grid.origin[2] + 160 * UM
    fl = flow(grid, psi, z_plane, shear_rate_s=30.0, shear_dir=(0.6, 0.8))
    ref = solve(psi, grid, flow_faces=fl["faces"], linear="direct")
    out = solve(psi, grid, flow_faces=fl["faces"], linear=linear)
    assert out["info"]["linear_cu"].startswith("gmres")
    assert np.max(np.abs(out["ct"] - ref["ct"])) < 1e-6
    assert out["V"] == pytest.approx(ref["V"], abs=1e-6)
