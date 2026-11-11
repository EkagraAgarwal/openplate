"""Validation tests for the Brownian-tracer solver, platesim.trenchflow.particles.

The flat-wall cases have closed-form answers: a semi-infinite electrolyte column of height H above an
absorbing/partially absorbing wall, with c = 1 at z = H, has the steady flux density
J = 1 / (1/k + H/D)  (reaction in series with diffusion)   [C],

which is used as the reference for tests 1, 2 and 4.  Tolerances are 8-10 % because the statistical
error of a time-averaged absorption count (about 2.5 % Poisson here) is amplified by the slow
relaxation of the column (time H^2/D ~ 11 s, longer than the sampling window).
"""
import numpy as np

from platesim.trenchflow.geometry import couette_flow, flat_grid, trench_grid
from platesim.trenchflow.particles import region_means, simulate

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
