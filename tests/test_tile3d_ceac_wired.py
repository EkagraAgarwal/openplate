"""Phase 3 step 2: the CEAC carried accelerator wired into run_tile behind Params.ceac.

The kernel (ceac_carry_step) is unit-tested in test_tile3d_ceac.py; this file checks the WIRING:
  (a) ceac=False leaves the cl_ndr path bit-identical to the same call on the default engine;
  (b) flat cl_ndr + ceac with a nearly inert accelerator follows the 0-D Langmuir + burial solution,
      and the carried theta_a is the one the kinetics (coverage_factor) see;
  (c) Params validation rejects ceac with a non-additive mode or a resolved-pulse coverage;
  (d) the curvature term enriches concave and dilutes convex front cells (burial off); (d2) with burial
      the depleted floor keeps more accelerator than the field;
  (e) acc + ceac runs to completion, converges and closes the accelerator transport balance.

No team data or fitted constants: every number is a Params default, so the assertions are identities,
analytic relations or internal-consistency checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine.params import Params, derive
from platesim.tile3d import additives3d
from platesim.tile3d import levelset3d as ls
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.result import TileResult
from platesim.tile3d.sim import run_tile

UM = 1e-4


def flat(nx=6, ny=6, nz=44, dx_um=4.0, z_lo_um=-16.0):
    grid = Grid3D.from_um((nx * dx_um, ny * dx_um, nz * dx_um), dx_um,
                          origin_um=(0.0, 0.0, z_lo_um))
    _, _, z = grid.meshgrid()
    return grid, z.copy()


def trench(width_um=20.0, depth_um=30.0, dx_um=2.0, ny=8, delta_um=64.0):
    """Generic narrow trench, pitch = 5 x width, periodic in x and y (Grid3D default)."""
    pitch_um = 5.0 * width_um
    z_extent_um = depth_um + delta_um + 32.0
    grid = Grid3D.from_um((pitch_um, ny * dx_um, z_extent_um), dx_um,
                          origin_um=(-0.5 * pitch_um, 0.0, -(depth_um + dx_um)))
    x, _, z = grid.meshgrid()
    half, d = 0.5 * width_um * UM, depth_um * UM
    psi = np.where(z > 0.0, z,
                   np.where(np.abs(x) < half, np.minimum(z + d, half - np.abs(x)), z))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    field_mask = np.abs(x[:, :, 0]) >= half + 0.3 * pitch_um * UM
    return psi, grid, field_mask


# ------------------------------------------------------------------ (a) ceac=False is untouched
def test_ceac_false_is_bit_identical_to_default_cl_ndr():
    psi, grid, field = trench()
    kw = dict(delta_um=64.0, field_mask=field, n_frames=3, dt_max=20.0)
    base = run_tile(psi, grid, 10.0, 60.0, params=Params(additives="cl_ndr", suppressor=False), **kw)
    off = run_tile(psi, grid, 10.0, 60.0,
                   params=Params(additives="cl_ndr", suppressor=False, ceac=False), **kw)
    np.testing.assert_array_equal(off["psi"], base["psi"])
    assert set(off["series"]) == set(base["series"])
    for key, values in base["series"].items():
        np.testing.assert_array_equal(np.asarray(off["series"][key], dtype=float),
                                      np.asarray(values, dtype=float), err_msg=key)
    assert set(off["summary"]) == set(base["summary"])


# ------------------------------------------- (b) flat cl_ndr + ceac: 0-D Langmuir with burial
def test_ceac_flat_coverage_follows_zerod_langmuir_and_drives_kinetics():
    grid, psi = flat()
    # cov_relax=1 snaps theta_s to exactly 0 (no suppressor), so coverage_factor is 1+b_acc*theta_a
    p = Params(additives="cl_ndr", ceac=True, suppressor=False, b_acc=1.0, theta0=0.5,
               cov_relax=1.0)
    res = run_tile(psi, grid, 10.0, 60.0, delta_um=64.0, params=p, n_frames=3, dt_max=20.0)

    v = derive(p, 22.0).Mfac * 10e-3                    # cm/s, uniform current on a flat front
    rate = p.k_acc * p.c_acc + p.k_inc_a * v            # adsorption + burial, kappa ~ 0 on a plane
    theta_inf = p.k_acc * p.c_acc / rate
    t = res["series"]["t_s"][-1]
    expect = theta_inf + (p.theta0 - theta_inf) * np.exp(-rate * t)
    th = res["series"]["theta_a_field"][-1]
    print(f"\nflat CEAC: t={t:.1f}s theta_a={th:.6f} analytic={expect:.6f} |err|={abs(th-expect):.2e}")
    assert abs(th - expect) < 1e-3
    assert res["series"]["theta_a_floor"][-1] == pytest.approx(th, abs=1e-12)  # flat: one region

    # coverage_factor sees the CARRIED theta_a: the last solve's kinetic factor is exactly 1+b_acc*th.
    ii, jj, kk = additives3d.front_cells(res["psi"])
    factor = res["fields"]["i0_factor"][ii, jj, kk]
    np.testing.assert_allclose(factor, 1.0 + p.b_acc * th, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(factor,
                               additives3d.coverage_factor(0.0, p, th), rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------ (c) validation
def test_ceac_validation():
    with pytest.raises(ValueError) as e:
        derive(Params(additives="legacy", ceac=True), 22.0)
    assert "additives" in str(e.value) and "pulse_coverage" in str(e.value)

    for mode in ("cl_ndr", "acc"):
        with pytest.raises(ValueError) as e:
            derive(Params(additives=mode, ceac=True, pulse_coverage="resolved"), 22.0)
        assert "pulse_coverage" in str(e.value) and "additives" in str(e.value)
        derive(Params(additives=mode, ceac=True), 22.0)     # valid combos do not raise
    derive(Params(additives="cl_ndr", ceac=False), 22.0)


# --------------------------------------------------- (d) CEAC curvature term in the wired loop
def test_ceac_curvature_enriches_concave_and_dilutes_convex_cells(tmp_path):
    """Isolate the area-change term: no burial (k_inc_a = 0), no suppressor, weak b_acc, so adsorption is
    the same everywhere and only -theta kappa v separates the cells.  The trench's floor corners are
    concave (kappa < 0: area shrinks, coverage enriched), its mouth corners convex (kappa > 0: diluted).
    The flat-front 0-D value 1 - exp(-k_acc c_acc t) must lie between the two."""
    psi, grid, field = trench()
    p = Params(additives="cl_ndr", ceac=True, suppressor=False, b_acc=1.0, k_inc_a=0.0, cov_relax=1.0)
    res = run_tile(psi, grid, 50.0, 120.0, delta_um=64.0, params=p, field_mask=field,
                   n_frames=2, dt_max=10.0, run_dir=tmp_path / "ceac_kappa")
    psi_end, theta = res.psi(-1), res.field("theta_a")
    t = float(res.times_s[-1])
    kappa = ls.front_curvature(psi_end, grid.dx, grid.periodic)
    front = np.abs(psi_end) < grid.dx
    kmax = 0.5 / grid.dx
    concave = front & (kappa < -0.5 * kmax)
    convex = front & (kappa > 0.5 * kmax)
    th0d = 1.0 - np.exp(-p.k_acc * p.c_acc * t)
    th_cc, th_cv = float(np.mean(theta[concave])), float(np.mean(theta[convex]))
    print(f"\nwired CEAC t={t:.0f}s: concave {th_cc:.4f} ({int(concave.sum())} cells) > 0-D {th0d:.4f} "
          f"> convex {th_cv:.4f} ({int(convex.sum())} cells)")
    assert concave.sum() > 0 and convex.sum() > 0
    assert th_cc > th0d > th_cv


# --------------------------------------------------- (d2) burial contrast: floor vs field
def test_ceac_narrow_trench_floor_above_field_with_burial():
    """With the default burial k_inc_a v, the depleted floor (lower v) keeps more accelerator than the
    field; this is the burial differential, not curvature (see the test above for curvature)."""
    psi, grid, field = trench()
    p = Params(additives="cl_ndr", ceac=True)
    res = run_tile(psi, grid, 10.0, 90.0, delta_um=64.0, params=p, field_mask=field,
                   n_frames=2, dt_max=10.0)
    th_floor = res["series"]["theta_a_floor"][-1]
    th_field = res["series"]["theta_a_field"][-1]
    print(f"\nnarrow-trench CEAC: theta_a floor={th_floor:.6f} field={th_field:.6f} "
          f"steps={res['summary']['steps']}")
    assert th_floor > th_field


# --------------------------------------------- (e) acc + ceac: completion, closure, I/O
def test_ceac_acc_mode_converges_and_closes_transport(tmp_path):
    grid, psi = flat()
    p = Params(additives="acc", ceac=True, suppressor=False)
    res = run_tile(psi, grid, 10.0, 60.0, delta_um=64.0, params=p, n_frames=2, dt_max=120.0,
                   run_dir=tmp_path / "acc_ceac")
    add = res.manifest["summary"]["additives"]
    closure = float(add["a_balance_rel"])
    print(f"\nacc+ceac: converged={add['converged']} steps={res.manifest['summary']['steps']} "
          f"a_balance_rel={closure:+.3e} theta_a={add['theta_a']}")
    assert add["converged"] is True
    assert abs(closure) < 1e-3
    result = TileResult(res.run_dir)
    for key in ("theta_a_floor", "theta_a_field", "a_floor", "a_field"):
        assert key in result.series and len(result.series[key]) == len(result.times_s)
    assert result.field("theta_a") is not None and result.field("theta_a").shape == grid.shape
