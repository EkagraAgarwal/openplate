"""Validation tests for the Brownian-tracer solver, platesim.trenchflow.particles.

The flat-wall cases have closed-form answers: a semi-infinite electrolyte column of height H above an
absorbing/partially absorbing wall, with c = 1 at z = H, has the steady flux density
J = 1 / (1/k + H/D)  (reaction in series with diffusion)   [C],

which is used as the reference for tests 1, 2 and 4.  Tolerances are 8-10 % because the statistical
error of a time-averaged absorption count (about 2.5 % Poisson here) is amplified by the slow
relaxation of the column (time H^2/D ~ 11 s, longer than the sampling window).

The solver offers ``interp="bilinear"`` (velocity interpolated over the four surrounding cell
centres, no-slip on solid neighbours) and ``wall="specular"`` (a non-absorbed encounter is mirrored
about the cell face it crossed) as options; the defaults stay ``nearest`` / ``return`` until a
higher-statistics comparison shows which is closer to the continuum.  Tests 7-10 pin the options: the interpolation helper itself, the uniform
flow drift (bilinear == nearest far from the wall, where the stencil is all fluid), the mirror
geometry of the wall rule, and the flat-wall 1-D flux with both wall rules (same seed, same particle
count) against ``1/(1/k + H/D)``.

E1b adds the compiled transport kernel and the migration drift.  The last tests pin the new
interfaces: ``backend="numpy"`` is bit-identical to the default call, ``backend="numba"`` reproduces
the numpy flat-wall flux within counting noise, a uniform potential on a flat wall reproduces the
analytic Nernst-Planck flux enhancement, ``grid_from_levelset`` round-trips an engine level set, and
the (slow) high-statistics decision run resolves the four interp x wall combinations against the
continuum.
"""
import time

import numpy as np
import pytest

from platesim.core.constants import F, R_GAS
from platesim.trenchflow.geometry import FlowField, couette_flow, flat_grid, trench_grid
from platesim.trenchflow.particles import (
    _bilinear,
    _specular_reflect,
    _velocity_at,
    grid_from_levelset,
    region_means,
    simulate,
)

D = 5.6e-6                    # Cu2+ diffusivity, cm2/s [J] value used across the project
H_UM = 80.0
H_CM = H_UM * 1e-4


def _exact_flux(k, h_cm=H_CM, d_cm2_s=D):
    """1/(1/k + H/D): reaction resistance in series with the diffusion resistance [C]."""
    return 1.0 / (h_cm / d_cm2_s + (0.0 if np.isinf(k) else 1.0 / k))


def _cell_of(grid, x, z):
    """Independent re-implementation of the nearest-cell lookup, to test the solver without it."""
    ze0 = float(grid.z_um[0]) - 0.5 * float(grid.dx_um)
    iz = np.floor((z - ze0) / grid.dx_um).astype(int)
    np.clip(iz, 0, grid.shape[0] - 1, out=iz)
    ix = np.floor(x / grid.dx_um).astype(int) % grid.shape[1]
    return iz, ix


def test_flat_wall_partially_absorbing():
    """k = 1e-3 cm/s: flux = 1/(1/k + H/D) within 8 %."""
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    res = simulate(g, None, D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=0.6, t_sample_s=8.0, seed=0)
    f = np.asarray(res["surf_flux_density"])
    exact = _exact_flux(1e-3)
    counts = float(res["surf_counts"].sum())
    assert f.size == g.shape[1] == 20
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)
    assert counts > 1000.0, f"too few absorption events ({counts:.0f}), Poisson error > 3 %"
    assert 0.0 < res["P_abs"] < 0.1
    ratio = f.mean() / exact
    assert abs(ratio - 1.0) < 0.08, (
        f"mean flux {f.mean():.3e} cm/s vs exact {exact:.3e} cm/s (ratio {ratio:.3f}, "
        f"counts {counts:.0f}, Poisson {100.0 / np.sqrt(counts):.2f} %)")


def test_flat_wall_absorbing_limit():
    """k = inf (P_abs = 1): flux density = D/H within 8 %."""
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    res = simulate(g, None, D_cm2_s=D, k_cm_s=np.inf, n_ref_per_um2=0.45, t_sample_s=7.5, seed=0)
    f = np.asarray(res["surf_flux_density"])
    exact = _exact_flux(np.inf)                      # = D/H
    counts = float(res["surf_counts"].sum())
    assert res["P_abs"] == 1.0
    assert counts > 1000.0
    ratio = f.mean() / exact
    assert abs(ratio - 1.0) < 0.08, (
        f"mean flux {f.mean():.3e} cm/s vs exact {exact:.3e} cm/s (ratio {ratio:.3f}, "
        f"counts {counts:.0f}, Poisson {100.0 / np.sqrt(counts):.2f} %)")


def test_reproducible_with_seed():
    """Identical seeds give identical counts; a different seed gives different counts."""
    g = flat_grid(pitch_um=40, height_um=40, dx_um=4)
    a = simulate(g, None, D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=0.5, t_burn_s=0.5,
                 t_sample_s=0.5, seed=7)
    b = simulate(g, None, D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=0.5, t_burn_s=0.5,
                 t_sample_s=0.5, seed=7)
    c = simulate(g, None, D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=0.5, t_burn_s=0.5,
                 t_sample_s=0.5, seed=8)
    assert np.array_equal(a["surf_counts"], b["surf_counts"])
    assert a["dt_s"] == b["dt_s"] and a["steps"] == b["steps"]
    assert a["surf_counts"].sum() > 0
    assert not np.array_equal(a["surf_counts"], c["surf_counts"])
    assert a["tracks"] is None


def test_couette_flow_parallel_to_wall_keeps_flux():
    """A shear flow parallel to a flat wall cannot change the 1-D flux (within 10 %)."""
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    flow = couette_flow(g, 80.0)
    # dt_s is passed explicitly: the default step would be CFL-limited by |u|max = 6.2e3 um/s (0.5 dx),
    # but the flow is parallel to a flat wall so a long advective step cannot enter a solid cell, and
    # nearest-cell sampling of ux = shear*z is exact for this field.  The z dynamics (and hence the
    # flux) are identical to the no-flow case; only the x coordinate is advected (periodic wrap).
    res = simulate(g, flow, D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=0.6, t_sample_s=8.0, seed=0,
                   dt_s=1.75e-3)
    f = np.asarray(res["surf_flux_density"])
    exact = _exact_flux(1e-3)
    ratio = f.mean() / exact
    counts = float(res["surf_counts"].sum())
    assert counts > 1000.0
    assert abs(ratio - 1.0) < 0.10, (
        f"couette flux {f.mean():.3e} vs no-flow exact {exact:.3e} (ratio {ratio:.3f})")


def test_advection_term_uses_the_flow_velocity():
    """One step of the random walk moves x by ux(z)*dt on average (paired check against the tracks)."""
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    shear = 80.0
    flow = couette_flow(g, shear)
    res = simulate(g, flow, D_cm2_s=D, k_cm_s=0.0, n_ref_per_um2=1.0, t_burn_s=0.0,
                   t_sample_s=1e-2, dt_s=1e-3, seed=0, n_tracks=1000)
    tr = np.asarray(res["tracks"])
    x0, z0 = tr[:, 0, 0], tr[:, 0, 1]
    dx = tr[:, 1, 0] - x0                       # one step of dt = 1e-3 s, unwrapped below
    dx = (dx + 0.5 * g.pitch_um) % g.pitch_um - 0.5 * g.pitch_um   # |true step| << pitch/2
    alive = np.isfinite(dx)
    assert alive.sum() > 800                    # k = 0: only the tagged particles that sit in the
    #                                             refreshed top layer at t = 0 die (about 10 %)
    iz, _ = _cell_of(g, x0[alive], z0[alive])   # nearest-cell ux uses the cell centre, not z itself
    ux_um_s = shear * np.asarray(g.z_um)[iz]    # ux = shear*z[cm]*1e-4 cm/s -> *1e4 = shear*z um/s
    pred = ux_um_s * 1e-3
    err = dx[alive] - pred
    # per-particle rms free step is 1.06 um, so the mean over ~1000 particles is ~0.03 um
    assert np.std(pred) > 0.2, "the tagged particles do not see a range of velocities"
    assert abs(err.mean()) < 0.15, (
        f"mean x displacement {dx[alive].mean():.3f} um vs ux*dt {pred.mean():.3f} um")


def test_small_trench_runs_and_stays_in_electrolyte():
    """A small trench: floor flux below the field flux, and no particle ever inside solid."""
    g = trench_grid(96, 48, height_um=96, dx_um=8)
    res = simulate(g, None, D_cm2_s=D, k_cm_s=1e-2, n_ref_per_um2=0.1, t_burn_s=4.0,
                   t_sample_s=4.0, seed=3)
    rm = region_means(res, g)
    assert res["steps"] > 0 and res["surf_counts"].sum() > 50
    assert rm["field_cells"] > 0 and rm["floor_cells"] > 0 and rm["wall_cells"] > 0
    assert rm["field_counts"] > 0 and rm["floor_counts"] > 0
    assert 0.0 < rm["floor_over_field"] < 1.0, (
        f"floor/field = {rm['floor_over_field']:.3f} "
        f"(floor {rm['floor']:.3e} +- {rm['floor_se']:.2e}, field {rm['field']:.3e} +- {rm['field_se']:.2e})")
    # invariant: every particle is in an electrolyte cell at the end of the run
    x = np.asarray(res["x_final_um"])
    z = np.asarray(res["z_final_um"])
    assert x.size > 0
    assert np.all(x >= 0.0) and np.all(x < g.pitch_um)
    assert np.all(z >= g.z_um[0] - 0.5 * g.dx_um) and np.all(z <= g.height_um)
    iz, ix = _cell_of(g, x, z)
    assert not np.any(g.solid[iz, ix]), "a particle ended a step inside a solid cell"
    # the concentration estimate is NaN in solid, finite in the electrolyte, 1 at the reservoir
    c = np.asarray(res["c"])
    assert np.all(np.isnan(c[g.solid])) and np.all(np.isfinite(c[~g.solid]))
    assert 0.5 < np.nanmax(c) <= 1.5


def test_bilinear_interpolation_helper():
    """_bilinear: exact at a cell centre, the mean at a face midpoint, periodic in x."""
    g = flat_grid(pitch_um=80, height_um=40, dx_um=4)
    dx = float(g.dx_um)
    ze0 = float(g.z_um[0]) - 0.5 * dx
    nz, nx = g.shape
    f = np.arange(nz * nx, dtype=float).reshape(nz, nx)      # one distinct value per cell

    def at(z, x):
        return float(_bilinear(f, np.array([float(z)]), np.array([float(x)]), ze0, dx, nz, nx)[0])

    for iz, ix in ((0, 0), (5, 7), (nz - 1, nx - 1)):
        assert at(g.z_um[iz], g.x_um[ix]) == f[iz, ix]       # exactly the cell value at the centre
    # a face midpoint is the linear average of the two cells it separates, in x and in z
    assert at(g.z_um[5], 4.0 * dx) == pytest.approx(0.5 * (f[5, 3] + f[5, 4]))
    assert at(ze0 + 6.0 * dx, g.x_um[2]) == pytest.approx(0.5 * (f[5, 2] + f[6, 2]))
    # x is periodic: a point just left of 0 interpolates exactly like the same point at pitch - eps
    assert at(g.z_um[5], -1.0) == pytest.approx(at(g.z_um[5], g.pitch_um - 1.0))
    # z is clamped: above the top row centre the top row's value is returned (no extrapolation)
    assert at(float(g.height_um), g.x_um[0]) == pytest.approx(f[nz - 1, 0])
    # no-slip: a solid stencil neighbour contributes zero velocity, so a uniform flow ramps linearly
    # from u/2 at the wall face (mirror of the zero) to u at the first fluid cell centre
    ux = np.where(g.solid, 0.0, 1.0)
    zq = np.array([0.0, 1.0, 2.0, 4.0])                      # flat_grid: solid is z < 0, centre at 2
    got = _velocity_at(ux, np.zeros(g.shape), zq, np.full(4, g.x_um[3]),
                       ze0, dx, nz, nx, "bilinear")[0]
    assert np.allclose(got, [0.5, 0.75, 1.0, 1.0])
    # the nearest rule ignores the wall entirely: the same points all return the cell value
    got_n = _velocity_at(ux, np.zeros(g.shape), zq, np.full(4, g.x_um[3]),
                         ze0, dx, nz, nx, "nearest")[0]
    assert np.allclose(got_n, 1.0)


def test_uniform_flow_bilinear_matches_nearest():
    """Uniform flow: bilinear and nearest drift identically away from the wall, where all four
    stencil cells are fluid (the no-slip part of the stencil is exercised in the helper test)."""
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    u_free = 0.5                                              # cm/s
    ux = np.where(g.solid, 0.0, u_free)
    flow = FlowField(grid=g, ux=ux, uz=np.zeros(g.shape), shear_rate_s=0.0)
    kw = {"D_cm2_s": D, "k_cm_s": 0.0, "n_ref_per_um2": 1.0, "t_burn_s": 0.0, "t_sample_s": 1e-2,
          "dt_s": 1e-3, "seed": 0, "n_tracks": 2000}
    near = simulate(g, flow, interp="nearest", **kw)
    bil = simulate(g, flow, interp="bilinear", **kw)
    assert near["interp"] == "nearest" and bil["interp"] == "bilinear"
    assert near["wall"] == "return" and bil["wall"] == "return"       # the default wall rule

    def step_x(res):
        tr = np.asarray(res["tracks"])
        d = tr[:, 1, 0] - tr[:, 0, 0]
        return (d + 0.5 * g.pitch_um) % g.pitch_um - 0.5 * g.pitch_um   # unwrapped one-step drift

    z0 = np.asarray(near["tracks"])[:, 0, 1]
    dxn, dxb = step_x(near), step_x(bil)
    free = u_free * 1e-3 * 1e4                                # ux * dt in um
    far = np.isfinite(dxn) & np.isfinite(dxb) & (z0 > 2.5 * g.dx_um)
    assert far.sum() > 500, "too few tagged particles away from the wall"
    # the Brownian kick is the same draw in both runs, so the difference is the advection only
    assert np.allclose(dxb[far], dxn[far], atol=1e-9)         # bilinear == nearest, all 4 cells fluid
    assert abs(dxn[far].mean() - free) < 0.05, (
        f"mean one-step drift {dxn[far].mean():.3f} um vs ux*dt {free:.3f} um")


def test_specular_reflection_geometry():
    """The wall rule mirrors the step about the face it crossed: floor, side wall, periodic seam."""
    ze0, dx, pitch = -8.0, 4.0, 40.0
    # flat wall: start at z = 1.0 (row 2), step to z = -0.5 (row 1, solid) -> mirror to +0.5
    xr, zr = _specular_reflect(np.array([13.0]), np.array([1.0]), np.array([13.0]),
                               np.array([-0.5]), np.array([2]), np.array([3]), np.array([1]),
                               np.array([3]), ze0, dx, pitch)
    assert zr[0] == pytest.approx(0.5) and xr[0] == pytest.approx(13.0)
    # side wall: start at x = 7.5 (ix 1), step to x = 8.5 (ix 2, solid) -> mirror about x = 8
    xr, zr = _specular_reflect(np.array([7.5]), np.array([5.0]), np.array([8.5]),
                               np.array([5.0]), np.array([3]), np.array([1]), np.array([3]),
                               np.array([2]), ze0, dx, pitch)
    assert xr[0] == pytest.approx(7.5) and zr[0] == pytest.approx(5.0)
    # periodic seam: start at x = 0.5 (ix 0), step to x = 39.5 (ix nx-1, solid) -> mirror about 0
    xr, zr = _specular_reflect(np.array([0.5]), np.array([5.0]), np.array([39.5]),
                               np.array([5.0]), np.array([3]), np.array([0]), np.array([3]),
                               np.array([9]), ze0, dx, pitch)
    assert xr[0] == pytest.approx(0.5) and zr[0] == pytest.approx(5.0)
    # a corner step (both indices change) is mirrored about the face crossed last, i.e. entered
    xr, zr = _specular_reflect(np.array([7.6]), np.array([1.0]), np.array([8.4]),
                               np.array([-0.4]), np.array([2]), np.array([1]), np.array([1]),
                               np.array([2]), ze0, dx, pitch)
    assert zr[0] == pytest.approx(0.4) and xr[0] == pytest.approx(8.4)


def test_flat_wall_specular_vs_return():
    """Both wall rules give the analytic 1-D flux; the specular rule is inside 5 %.

    Same seed and particle budget for both.  ``wall="return"`` (the default) undoes a non-absorbed
    encounter, ``wall="specular"`` mirrors it about the crossed face.  The rules can
    only differ inside one rms step of the wall, so the difference in the 1-D steady flux is a
    sub-percent effect at usable resolutions - measured (this column, six seeds, sigma/H = 1.8 %)
    ``specular - return = -0.15 % +- 0.73 %``, and at dx = 6 um / H = 90 um with k = 1.8e-4 cm/s
    (the particle cross-check settings) specular lands 1.6 % closer to the analytic value
    (-1.0 % vs -2.6 %); resolving either number at 3 sigma needs ~1e5 absorption events, i.e. about
    a minute of wall time per run, so the ordering is *not* asserted here and the residual against
    the analytic ``1/(1/k + H/D)`` is dominated by the cell-size boundary layer (the wall
    concentration is sampled at the first cell centre) and the finite burn-in, both common to the
    two rules.  What this test pins is that neither rule introduces a gross wall artefact: both are
    within 5 % and they agree with each other to 3 %.  The reflection geometry itself is checked
    exactly in :func:`test_specular_reflection_geometry`.
    """
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    kw = {"D_cm2_s": D, "k_cm_s": 1e-3, "n_ref_per_um2": 1.5, "t_sample_s": 6.0, "seed": 0}
    spec = simulate(g, None, wall="specular", **kw)
    ret = simulate(g, None, wall="return", **kw)
    exact = _exact_flux(1e-3)
    f_s = float(np.asarray(spec["surf_flux_density"]).mean())
    f_r = float(np.asarray(ret["surf_flux_density"]).mean())
    n_s = float(spec["surf_counts"].sum())
    n_r = float(ret["surf_counts"].sum())
    assert spec["wall"] == "specular" and ret["wall"] == "return"
    assert 0.0 < spec["P_abs"] < 0.1
    assert n_s > 2500 and n_r > 2500, (n_s, n_r)            # ~2 % Poisson per run
    assert abs(f_s / exact - 1.0) < 0.05, (
        f"specular flux {f_s:.3e} vs exact {exact:.3e} (ratio {f_s / exact:.3f}, n {n_s:.0f})")
    assert abs(f_r / exact - 1.0) < 0.05, (
        f"return flux {f_r:.3e} vs exact {exact:.3e} (ratio {f_r / exact:.3f}, n {n_r:.0f})")
    assert abs(f_s - f_r) / f_r < 0.03, (
        f"wall rules differ by {100.0 * abs(f_s - f_r) / f_r:.2f} % "
        f"(specular {f_s:.3e}, return {f_r:.3e})")


# ---------------------------------------------------------------------------------------------
# E1b: compiled backend, migration drift, engine geometry, high-statistics decision run
# ---------------------------------------------------------------------------------------------
def _np_flux(k_cm_s, h_cm=H_CM, d_cm2_s=D, drift_cm_s=0.0):
    """1-D steady flux with a uniform drift ``drift_cm_s`` towards the wall [C].

    Steady state of d/dz(-D dc/dz + w c) = 0 with c(H) = 1 and D dc/dz|_0 = k c(0), for a drift w
    pointing at the wall:  J = 1 / [ (H/D) (1 - e^-P)/P + e^-P / k ],  P = w H / D.  ``w = 0``
    reduces to ``1/(1/k + H/D)`` (:func:`_exact_flux`), the reaction resistance in series with the
    diffusion resistance.
    """
    if drift_cm_s == 0.0:
        return _exact_flux(k_cm_s, h_cm, d_cm2_s)
    P = float(drift_cm_s) * float(h_cm) / float(d_cm2_s)
    return 1.0 / (float(h_cm) / float(d_cm2_s) * (1.0 - np.exp(-P)) / P
                  + np.exp(-P) * (0.0 if np.isinf(k_cm_s) else 1.0 / k_cm_s))


def test_defaults_bit_identical_with_numpy_backend():
    """``backend="numpy"`` (the default) and the new keyword arguments change nothing.

    The pre-E1b call, the explicit numpy backend with the new defaults, and a run with a *uniform*
    potential (zero gradient -> zero drift) must give bit-identical particle counts, fluxes and
    concentration fields.
    """
    g = flat_grid(pitch_um=40, height_um=40, dx_um=4)
    kw = {"D_cm2_s": D, "k_cm_s": 1e-3, "n_ref_per_um2": 0.5, "t_burn_s": 0.5, "t_sample_s": 0.5,
          "seed": 11}
    a = simulate(g, None, **kw)
    b = simulate(g, None, backend="numpy", phi=None, z_ion=2, T_C=22.0, **kw)
    c = simulate(g, None, phi=np.zeros(g.shape), z_ion=2, T_C=22.0, **kw)
    assert a["backend"] == b["backend"] == "numpy"
    assert a["migration"] is False and c["migration"] is True
    for res in (b, c):
        assert np.array_equal(a["surf_counts"], res["surf_counts"])
        assert np.array_equal(a["surf_flux_density"], res["surf_flux_density"])
        assert np.array_equal(a["c"], res["c"], equal_nan=True)
        assert np.array_equal(a["x_final_um"], res["x_final_um"])
        assert (a["steps"], a["dt_s"], a["P_abs"]) == (res["steps"], res["dt_s"], res["P_abs"])
    assert a["surf_counts"].sum() > 0


def test_migration_drift_matches_nernst_planck():
    """Uniform potential gradient on a flat wall: flux = the analytic Nernst-Planck value.

    ``phi = a z`` gives the constant drift ``u = -z_ion F D/(R T) dphi/dz = 0.5 D/H`` towards the
    wall (P = uH/D = 0.5), i.e. a 40 % flux enhancement over ``1/(1/k + H/D)``.  The particle flux
    is compared with :func:`_np_flux` and must also be *enhanced* with respect to the no-drift value.
    """
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    T_C = 22.0
    mob = 2.0 * D * F / (R_GAS * (T_C + 273.15))          # cm2/(s V) for z_ion = 2
    u = 0.5 * D / H_CM                                     # drift towards the wall, cm/s
    a = u / (mob * 1e4)                                    # phi = a z, V per um
    phi = a * np.broadcast_to(np.asarray(g.z_um)[:, None], g.shape)
    res = simulate(g, None, D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=1.5, t_sample_s=4.0, seed=0,
                   phi=phi, T_C=T_C)
    f = float(np.asarray(res["surf_flux_density"]).mean())
    n = float(res["surf_counts"].sum())
    exact = _np_flux(1e-3, drift_cm_s=u)
    assert res["migration"] is True
    assert n > 2000.0, f"too few absorptions ({n:.0f}), Poisson error > 2 %"
    assert abs(f / exact - 1.0) < 0.06, (
        f"drift flux {f:.3e} vs Nernst-Planck {exact:.3e} (ratio {f / exact:.3f}, n {n:.0f}, "
        f"Poisson {100.0 / np.sqrt(n):.2f} %)")
    # the enhancement is real and of the analytic size (1.40 at P = 0.5)
    assert f > 1.25 * _exact_flux(1e-3), (f, _exact_flux(1e-3))
    assert 1.3 < f / _exact_flux(1e-3) < 1.5, f / _exact_flux(1e-3)


def test_numba_backend_matches_numpy_flat_wall():
    """``backend="numba"`` reproduces the numpy flat-wall flux within 3 standard errors.

    The two backends draw from different random streams (numpy's ``default_rng`` vs the kernel's
    counter-based per-particle streams), so they agree statistically, not bit for bit.  Both must
    also land on the analytic 1-D value; the residual is the cell-size boundary layer they share.
    """
    pytest.importorskip("numba")
    g = flat_grid(pitch_um=80, height_um=H_UM, dx_um=4)
    kw = {"D_cm2_s": D, "k_cm_s": 1e-3, "n_ref_per_um2": 1.5, "t_sample_s": 5.0, "seed": 0}
    r_np = simulate(g, None, backend="numpy", **kw)
    r_nb = simulate(g, None, backend="numba", **kw)
    f_np = float(np.asarray(r_np["surf_flux_density"]).mean())
    f_nb = float(np.asarray(r_nb["surf_flux_density"]).mean())
    n_np = float(r_np["surf_counts"].sum())
    n_nb = float(r_nb["surf_counts"].sum())
    assert r_nb["backend"] == "numba" and r_np["backend"] == "numpy"
    assert r_nb["tracks"] is None
    assert (r_np["steps"], r_np["dt_s"], r_np["P_abs"]) == (r_nb["steps"], r_nb["dt_s"], r_nb["P_abs"])
    assert r_nb["migration"] is False
    assert n_np > 2000 and n_nb > 2000, (n_np, n_nb)
    se = f_np * np.sqrt(1.0 / n_np + 1.0 / n_nb)          # standard error of the difference
    assert abs(f_nb - f_np) <= 3.0 * se, (
        f"numba flux {f_nb:.3e} vs numpy {f_np:.3e}: {abs(f_nb - f_np) / se:.2f} se "
        f"(se {se:.2e}, counts {n_np:.0f}/{n_nb:.0f})")
    exact = _exact_flux(1e-3)
    for name, f in (("numpy", f_np), ("numba", f_nb)):
        assert abs(f / exact - 1.0) < 0.05, (name, f, exact)
    # the concentration field is the same to counting noise
    assert np.allclose(np.asarray(r_nb["c"]), np.asarray(r_np["c"]), atol=0.1, equal_nan=True)
    # invariants of the compiled path: every particle ends in the electrolyte
    x = np.asarray(r_nb["x_final_um"])
    z = np.asarray(r_nb["z_final_um"])
    assert x.size > 0 and np.all(x >= 0.0) and np.all(x < g.pitch_um)
    assert np.all(z >= g.z_um[0] - 0.5 * g.dx_um) and np.all(z <= g.height_um)
    iz, ix = _cell_of(g, x, z)
    assert not np.any(np.asarray(g.solid)[iz, ix])


def test_grid_from_levelset_round_trip():
    """``grid_from_levelset`` maps an engine level set to the tracer's CavityGrid.

    Same conventions as :meth:`platesim.feature.engine.sim.ViaSim.cavity_flow`: ``solid = psi <= 0``
    transposed from (nr, nz) to (nz, nx), engine centimetres converted to um, ``z = 0`` the wafer
    plane and ``height_um`` the top Dirichlet plane.  The returned grid must be usable by
    :func:`simulate` as it is.
    """
    ls = pytest.importorskip("platesim.feature.engine.levelset")
    g = ls.Grid(120.0, 60.0, 0.5, 32.0, dx_um=6.0, geometry="trench_full")
    psi = ls.initial_psi(g, 2)
    grid = grid_from_levelset(psi, g)

    assert grid.shape == (g.nz, g.nr)
    assert np.array_equal(grid.solid, (psi <= 0.0).T)
    assert np.allclose(grid.x_um, g.rc[:, 0] / ls.UM)
    assert np.allclose(grid.z_um, (g.zc[0] - g.H) / ls.UM)
    assert grid.dx_um == pytest.approx(g.dx_um)
    assert grid.pitch_um == pytest.approx(g.R_um)
    assert (grid.width_um, grid.depth_um) == pytest.approx((g.d_um, g.H_um))
    assert grid.height_um == pytest.approx((g.z_top - g.H) / ls.UM)
    z_bottom = grid.z_um[0] - 0.5 * grid.dx_um          # particles.simulate's ze0
    assert z_bottom == pytest.approx((g.z_lo - g.H) / ls.UM)
    assert z_bottom + grid.shape[0] * grid.dx_um == pytest.approx(grid.height_um)
    # the level set's solid is the wafer + trench metal: bottom rows solid, top row electrolyte
    assert grid.solid[0].all() and not grid.solid[-1].any()
    mid = grid.shape[1] // 2
    assert grid.z_um[int(np.argmax(~grid.solid[:, mid]))] < -50.0        # trench floor at -60 um
    assert grid.z_um[int(np.argmax(~grid.solid[:, 0]))] > 0.0            # field plane at z = 0

    res = simulate(grid, None, D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=0.5, t_burn_s=1.0,
                   t_sample_s=1.0, seed=0)
    assert res["steps"] > 0 and res["surf_counts"].sum() > 0
    x = np.asarray(res["x_final_um"])
    z = np.asarray(res["z_final_um"])
    assert x.size > 0 and np.all(x >= 0.0) and np.all(x < grid.pitch_um)
    iz, ix = _cell_of(grid, x, z)
    assert not np.any(grid.solid[iz, ix])

    with pytest.raises(ValueError):
        grid_from_levelset(psi[:-1], g)
    with pytest.raises(ValueError):
        grid_from_levelset(np.where(np.eye(*psi.shape) > 0, np.nan, psi), g)


def _high_stat_cross_check(n_ref_per_um2=10.0, t_burn_s=30.0, t_sample_s=30.0, seed=0):
    """E1b decision run: ``convergence.particle_cross_check``'s recipe at high statistics.

    Same geometry, flow and rate constant as
    :func:`platesim.trenchflow.convergence.particle_cross_check` (120 x 60 um trench, dx 6 um, LBM
    shear flow from ``shear_from_delta(128, D)``, ``k`` from ``c_wall_target = 0.7``), but
    ``n_ref_per_um2 = 10`` instead of 0.25 and ``backend="numba"``.  The absorption counts grow by
    ~40x, so the Poisson standard error of floor/field drops to < 0.005 (wall/field to ~0.005) and
    the four interp x wall combinations can be ranked.  Nothing here changes a default.
    """
    from platesim.trenchflow.cavity_lbm import solve_flow
    from platesim.trenchflow.convergence import NU_CM2_S, _k_for_wall_concentration
    from platesim.trenchflow.model import model_constants, shear_from_delta
    from platesim.trenchflow.particles import region_means as particle_region_means
    from platesim.trenchflow.transport import region_means as continuum_region_means
    from platesim.trenchflow.transport import solve_concentration

    d_cm2_s = float(model_constants(22.0)[0])
    grid = trench_grid(120.0, 60.0, height_um=90.0, dx_um=6.0)
    flow = solve_flow(grid, float(shear_from_delta(128.0, d_cm2_s)), nu_cm2_s=NU_CM2_S)
    k_cm_s = _k_for_wall_concentration(grid, flow, d_cm2_s, 0.7)
    crm = continuum_region_means(solve_concentration(grid, flow, D_cm2_s=d_cm2_s, k_cm_s=k_cm_s),
                                 grid)
    out = {"k_cm_s": float(k_cm_s), "grid_shape": tuple(int(v) for v in grid.shape),
           "n_ref_per_um2": float(n_ref_per_um2), "t_burn_s": float(t_burn_s),
           "t_sample_s": float(t_sample_s),
           "continuum": {key: float(crm[key])
                         for key in ("floor_over_field", "wall_over_field")},
           "combinations": {}}
    for interp, wall in (("nearest", "return"), ("nearest", "specular"),
                         ("bilinear", "return"), ("bilinear", "specular")):
        t0 = time.perf_counter()
        res = simulate(grid, flow, D_cm2_s=d_cm2_s, k_cm_s=k_cm_s,
                       n_ref_per_um2=float(n_ref_per_um2), t_burn_s=float(t_burn_s),
                       t_sample_s=float(t_sample_s), seed=int(seed), interp=interp, wall=wall,
                       backend="numba")
        wall_s = time.perf_counter() - t0
        prm = particle_region_means(res, grid)
        row = {"wall_s": wall_s, "kernel_s": float(res.get("kernel_s", float("nan"))),
               "n_particles_mean": float(res["n_particles_mean"]), "interp": interp, "wall": wall}
        for key in ("floor_over_field", "wall_over_field"):
            a, b = key.split("_over_")
            n_a = float(prm.get(a + "_counts", 0.0))
            n_b = float(prm.get(b + "_counts", 0.0))
            p = float(prm[key])
            c = float(crm[key])
            se_rel = float(np.sqrt(1.0 / n_a + 1.0 / n_b)) if (n_a > 0 and n_b > 0) else float("nan")
            se_abs = abs(p) * se_rel if np.isfinite(se_rel) else float("nan")
            row[key] = {"particles": p, "continuum": c, "difference": p - c,
                        "se_rel": se_rel, "se_abs": se_abs,
                        "counts": (n_a, n_b),
                        "within_2se": bool(np.isfinite(se_abs) and abs(p - c) <= 2.0 * se_abs)}
        out["combinations"][f"{interp}+{wall}"] = row
    return out


@pytest.mark.slow
def test_high_statistics_wall_rule_decision():
    """E1b decision run: floor/field and wall/field to a standard error below 0.005.

    Prints the four interp x wall combinations against the continuum (run with ``-s``) and asserts
    the statistical requirement -- floor/field resolved to ``se_abs < 0.005`` and wall/field to
    ``se_abs < 0.006`` (its counts are 2.5x smaller) -- plus that the best combination lands within
    2 % of the continuum, the band set by the two discretisations themselves (the continuum's
    upwind numerical diffusion and the tracer's source layer; see ``simulate``'s docstring).  Which
    combination is closest is the lead's decision, not something this test should pin.
    """
    pytest.importorskip("numba")
    out = _high_stat_cross_check()
    print("\nE1b decision run: 120 x 60 um, dx 6, LBM flow, "
          f"k = {out['k_cm_s']:.4e} cm/s, n_ref {out['n_ref_per_um2']:g}/um2, "
          f"{out['t_burn_s']:g} s burn + {out['t_sample_s']:g} s sample, numba backend")
    print(f"continuum floor/field {out['continuum']['floor_over_field']:.4f}  "
          f"wall/field {out['continuum']['wall_over_field']:.4f}")
    print(f"{'interp+wall':21s} {'floor/field':>11s} {'diff':>8s} {'se':>7s} {'2se':>5s} "
          f"{'wall/field':>11s} {'diff':>8s} {'se':>7s} {'2se':>5s} {'counts':>13s} {'s':>6s}")
    for name, row in out["combinations"].items():
        ff, wf = row["floor_over_field"], row["wall_over_field"]
        print(f"{name:21s} {ff['particles']:11.4f} {ff['difference']:+8.4f} {ff['se_abs']:7.4f} "
              f"{str(ff['within_2se']):>5s} {wf['particles']:11.4f} {wf['difference']:+8.4f} "
              f"{wf['se_abs']:7.4f} {str(wf['within_2se']):>5s} "
              f"{ff['counts'][0]:6.0f}/{ff['counts'][1]:6.0f} {row['wall_s']:6.1f}")
    for name, row in out["combinations"].items():
        for key in ("floor_over_field", "wall_over_field"):
            cell = row[key]
            assert np.isfinite(cell["particles"]) and np.isfinite(cell["continuum"])
            limit = 0.005 if key == "floor_over_field" else 0.006
            assert cell["se_abs"] < limit, (name, key, cell)
    # the closest rule reproduces floor/field within the 2 % comparison band
    best = min(abs(row["floor_over_field"]["difference"]) for row in out["combinations"].values())
    assert best <= 0.02, best


# ---------------------------------------------------------------------------------------------
# E1b: Erban-Chapman second-order (Andrews-Bray) reactive boundary
# ---------------------------------------------------------------------------------------------
def test_wall_touch_prob_is_the_andrews_bray_term():
    """``_wall_touch_prob`` = 1 - prod(1 - exp(-d_old d_new /(D dt))) over the solid faces.

    Erban & Chapman (2007) Sec. 2.2.  Exact values first (flat wall: exp(-d_old d_new / D_dt)), then
    the geometry: a step that crossed a face does not double count, an interior particle can never
    touch a wall, and a trench side wall is found exactly.
    """
    from platesim.trenchflow.particles import _wall_touch_prob

    g = flat_grid(pitch_um=80, height_um=40, dx_um=4)
    nz, nx = g.shape
    ze0 = float(g.z_um[0]) - 0.5 * float(g.dx_um)
    solid = np.asarray(g.solid, dtype=bool)
    nb = np.zeros(g.shape, dtype=int)
    nb += np.roll(solid, 1, axis=1) + np.roll(solid, -1, axis=1)
    nb[1:] += solid[:-1]
    nb[:-1] += solid[1:]
    surf = (~solid) & (nb > 0)

    D_dt = 0.25 * 0.5                                   # um2, arbitrary but > 0
    z_row = int(np.argmax(g.z_um >= 0.0))               # first fluid row, z in [0, 4]
    assert g.z_um[z_row] == 2.0
    iz0 = np.full(3, z_row)
    ix0 = np.full(3, 3)
    z_old = np.array([0.5, 1.0, 3.0])                   # distances to the wall at z = 0
    z_new = np.array([0.2, 1.0, 3.0])                   # stays on the fluid side
    got = _wall_touch_prob(np.full(3, g.x_um[3]), z_old, np.zeros(3), z_new - z_old, iz0, ix0,
                           solid, surf, ze0, float(g.dx_um), nz, nx, D_dt)
    assert np.allclose(got, np.exp(-z_old * z_new / D_dt), rtol=1e-12), got
    # a step that ends inside the wall is the crossing branch's job: no double count
    got = _wall_touch_prob(np.full(1, g.x_um[3]), np.array([0.5]), np.zeros(1), np.array([-1.0]),
                           np.array([z_row]), np.array([3]), solid, surf, ze0,
                           float(g.dx_um), nz, nx, D_dt)
    assert got[0] == 0.0
    # an interior particle (no solid face neighbour) can never touch a wall
    iz_c = int(np.argmax(g.z_um >= 20.0))
    assert not surf[iz_c, 3]
    got = _wall_touch_prob(np.full(1, g.x_um[3]), np.array([20.0]), np.zeros(1), np.zeros(1),
                           np.array([iz_c]), np.array([3]), solid, surf, ze0,
                           float(g.dx_um), nz, nx, D_dt)
    assert got[0] == 0.0

    # trench side wall: the first fluid cell inside the trench has solid to its left
    t = trench_grid(64.0, 32.0, height_um=64.0, dx_um=4.0)
    nz2, nx2 = t.shape
    solid2 = np.asarray(t.solid, dtype=bool)
    ze2 = float(t.z_um[0]) - 0.5 * float(t.dx_um)
    nb2 = np.zeros(t.shape, dtype=int)
    nb2 += np.roll(solid2, 1, axis=1) + np.roll(solid2, -1, axis=1)
    nb2[1:] += solid2[:-1]
    nb2[:-1] += solid2[1:]
    sur2 = (~solid2) & (nb2 > 0)
    iz_w = int(np.argmax(t.z_um >= -15.0))
    ix_w = int(np.argmax(~solid2[iz_w]))
    assert solid2[iz_w, ix_w - 1], "expected a solid cell left of the first trench cell"
    d_old, d_new = 1.0, 0.5
    got = _wall_touch_prob(np.array([ix_w * float(t.dx_um) + d_old]), np.array([t.z_um[iz_w]]),
                           np.array([-(d_old - d_new)]), np.zeros(1),
                           np.array([iz_w]), np.array([ix_w]),
                           solid2, sur2, ze2, float(t.dx_um), nz2, nx2, D_dt)
    assert got[0] == pytest.approx(float(np.exp(-d_old * d_new / D_dt)), rel=1e-12)


def test_second_order_reactive_boundary_matches_robin_flux():
    """Erban-Chapman (a)+(b) boundary on a 1-D reactive slab: analytic Robin flux within 3 se.

    ``reactive="second"`` is their two-part rule: a step that ends in the wall reacts with
    ``p = k sqrt(pi dt/D)/2`` and, in addition, a step that *touched* the wall and came back reacts
    with ``exp(-d_old d_new/(D dt)) p`` (the Andrews-Bray term (b)).  Both parts together give the
    Robin constant ``K = 2 p sqrt(D/(pi dt)) = k``, so to first order the flux is unchanged but the
    O(sqrt(dt)) boundary-layer error drops out.

    The reference is ``1/(1/k + H_eff/D)`` with ``H_eff = H - l_src``: the top reservoir imposes
    c = 1 over the source layer (thickness ``max(3 sigma, dx)``), so the column the tracer really
    solves is shorter by that layer.  Without the correction the same run reads +2 to +5 % high for
    every wall rule at dx 4-8 (measured), i.e. the residual is the reservoir, not the reaction rule.
    """
    g = flat_grid(pitch_um=160, height_um=H_UM, dx_um=4)
    kw = dict(D_cm2_s=D, k_cm_s=1e-3, n_ref_per_um2=0.8, t_burn_s=6.0, t_sample_s=9.0, seed=0,
              wall="specular")
    r2 = simulate(g, None, reactive="second", **kw)
    r1 = simulate(g, None, reactive="first", **kw)
    f1 = float(np.asarray(r1["surf_flux_density"]).mean())
    f2 = float(np.asarray(r2["surf_flux_density"]).mean())
    n1 = float(r1["surf_counts"].sum())
    n2 = float(r2["surf_counts"].sum())
    assert r2["reactive"] == "second" and r1["reactive"] == "first"
    assert n1 > 3000 and n2 > 3000, (n1, n2)
    # the per-touch probability is halved and the two-part rule makes up for it
    assert r2["P_abs"] == pytest.approx(0.5 * r1["P_abs"], rel=1e-12)
    assert (r1["steps"], r1["dt_s"], r1["P_abs_first_order"]) == (r2["steps"], r2["dt_s"],
                                                                 r2["P_abs_first_order"])
    l_src = max(3.0 * float(r2["sigma_um"]), float(g.dx_um))
    h_cm = (float(g.height_um) - l_src) * 1e-4
    exact = 1.0 / (h_cm / D + 1.0 / 1e-3)
    se2 = 1.0 / np.sqrt(n2)
    assert abs(f2 / exact - 1.0) <= 3.0 * se2, (
        f"second-order flux {f2:.3e} vs analytic {exact:.3e} (ratio {f2 / exact:.4f}, "
        f"se {se2 * 100:.2f} %, H_eff {h_cm * 1e4:.1f} um)")
    # the two rules agree: same leading-order Robin constant
    se_diff = np.hypot(1.0 / np.sqrt(n1), 1.0 / np.sqrt(n2))
    assert abs(f2 - f1) <= 3.0 * se_diff * f1, (
        f"first {f1:.3e} vs second {f2:.3e} order differ by "
        f"{100.0 * abs(f2 - f1) / f1:.2f} % (se {100.0 * se_diff:.2f} %)")
    # k = 0 must not absorb through the Andrews-Bray term
    r0 = simulate(g, None, D_cm2_s=D, k_cm_s=0.0, n_ref_per_um2=0.5, t_burn_s=1.0,
                  t_sample_s=1.0, seed=0, reactive="second")
    assert r0["P_abs"] == 0.0 and float(r0["surf_counts"].sum()) == 0.0
    with pytest.raises(ValueError):
        simulate(g, None, reactive="third")


def test_second_order_boundary_numba_matches_numpy():
    """The compiled kernel implements the same (a)+(b) rule (statistical agreement, small trench)."""
    pytest.importorskip("numba")
    g = trench_grid(96, 48, height_um=96, dx_um=8)
    kw = dict(D_cm2_s=D, k_cm_s=1e-2, n_ref_per_um2=0.6, t_burn_s=3.0, t_sample_s=3.0, seed=5,
              reactive="second")
    a = simulate(g, None, backend="numpy", **kw)
    b = simulate(g, None, backend="numba", **kw)
    assert b["reactive"] == a["reactive"] == "second"
    assert b["backend"] == "numba"
    assert b["P_abs"] == pytest.approx(a["P_abs"], rel=1e-12)
    na, nb = float(a["surf_counts"].sum()), float(b["surf_counts"].sum())
    assert na > 500 and nb > 500, (na, nb)
    assert abs(na - nb) <= 4.0 * np.sqrt(na + nb), (na, nb)
    ma, mb = region_means(a, g), region_means(b, g)

    def _se(m, key):
        a_, b_ = key.split("_over_")
        return abs(m[key]) * np.sqrt(1.0 / m[a_ + "_counts"] + 1.0 / m[b_ + "_counts"])

    for key in ("floor_over_field", "wall_over_field"):
        assert abs(ma[key] - mb[key]) <= 3.0 * np.hypot(_se(ma, key), _se(mb, key)) + 0.02, (
            key, ma[key], mb[key])


def test_divergence_free_field_does_not_accumulate():
    """An incompressible flow keeps the tracer uniform; a compressive one piles it up.

    The stream function psi = A sin(2 pi x/P) sin^2(pi z/H) gives an exactly divergence-free field
    that vanishes on the wall and on the reservoir plane (ux = d psi/dz, uz = -d psi/dx).  With k = 0
    the steady state of the tracer is c = 1 everywhere, so any cell-to-cell structure beyond counting
    noise is spurious accumulation -- the failure mode of a non-solenoidal advection.  The control
    uses div u != 0 and must concentrate, which shows the metric has power.
    """
    g = flat_grid(pitch_um=80.0, height_um=40.0, dx_um=4.0)
    nz, nx = g.shape
    zz = np.asarray(g.z_um)[:, None]
    xx = np.asarray(g.x_um)[None, :]
    H, P = float(g.height_um), float(g.pitch_um)
    A = 400.0                                   # um2/s, so |u| ~ 30 um/s (advection ~ diffusion)
    ux = np.where(g.solid, 0.0, A * (np.pi / H) * np.sin(2 * np.pi * xx / P)
                  * np.sin(2 * np.pi * zz / H) * 1e-4)
    uz = np.where(g.solid, 0.0, -A * (2 * np.pi / P) * np.cos(2 * np.pi * xx / P)
                  * np.sin(np.pi * zz / H) ** 2 * 1e-4)
    # the input field really is divergence-free (central differences on the fluid cells)
    fluid = ~np.asarray(g.solid)
    div = np.zeros(g.shape)
    div[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / (2 * g.dx_um)
    div[1:-1, :] += (uz[2:, :] - uz[:-2, :]) / (2 * g.dx_um)
    scale = max(np.abs(ux).max(), np.abs(uz).max()) / g.dx_um
    inner = fluid.copy()
    inner[0] = inner[-1] = False
    inner[:, 0] = inner[:, -1] = False
    inner[1:-1] &= fluid[:-2] & fluid[2:]        # central differences need fluid neighbours
    assert np.abs(div[inner]).max() < 0.02 * scale, np.abs(div[inner]).max() / scale

    flow = FlowField(grid=g, ux=ux, uz=uz, shear_rate_s=0.0)
    kw = dict(D_cm2_s=D, k_cm_s=0.0, n_ref_per_um2=1.5, t_burn_s=3.0, t_sample_s=6.0, seed=5,
              interp="bilinear")

    def blocks(c, half=2):
        """Mean c over 2 x 2 cell blocks (suppresses the per-cell Poisson noise)."""
        out = []
        for j in range(half, nz - half, 2):
            for i in range(0, nx, 2):
                blk = c[j - half:j + half, i:i + 2]
                if np.all(np.isfinite(blk)):
                    out.append(blk.mean())
        return np.array(out)

    bl = blocks(np.asarray(simulate(g, flow, **kw)["c"]))
    assert bl.size > 20
    assert bl.max() < 1.10 and bl.min() > 0.90, (bl.min(), bl.max())
    # compressive control: ux = B sin(2 pi x/P) sin(pi z/H), uz = 0 -> div u != 0
    B = 30.0                                   # um/s
    uxc = np.where(g.solid, 0.0, B * np.sin(2 * np.pi * xx / P) * np.sin(np.pi * zz / H) * 1e-4)
    flowc = FlowField(grid=g, ux=uxc, uz=np.zeros(g.shape), shear_rate_s=0.0)
    blc = blocks(np.asarray(simulate(g, flowc, **kw)["c"]))
    assert blc.max() - blc.min() > 0.5 and blc.max() > 1.2, (blc.min(), blc.max())



