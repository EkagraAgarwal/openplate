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
"""
import numpy as np
import pytest

from platesim.trenchflow.geometry import FlowField, couette_flow, flat_grid, trench_grid
from platesim.trenchflow.particles import (
    _bilinear,
    _specular_reflect,
    _velocity_at,
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

