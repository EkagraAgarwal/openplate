"""Tests for platesim.trenchflow.cavity_lbm (D2Q9 shear flow over a periodic trench).

Small grids only (dx = 8 um) so the whole module runs in a few seconds.
Analytic reference: Couette flow u_x = G z over a flat wall, with the no-slip plane at z = 0 and the
first fluid row centre at z = dx/2 (halfway bounce-back).
"""
import numpy as np

from platesim.trenchflow.cavity_lbm import eddy_summary, penetration_depth_um, solve_flow
from platesim.trenchflow.geometry import flat_grid, trench_grid


def test_couette_linear_profile():
    """Flat wall + moving lid: u_x(z) must be the analytic linear Couette profile, u_z ~ 0."""
    g = flat_grid(pitch_um=80, height_um=160, dx_um=8)
    shear = 50.0
    flow = solve_flow(g, shear)
    assert flow.converged
    u_top = shear * g.height_um * 1e-4                      # cm/s at the lid
    fluid = ~g.solid[:, 0]
    z_um = g.z_um[fluid]
    exact = shear * z_um * 1e-4                             # cm/s
    sim = flow.ux[fluid, 3]
    err = float(np.max(np.abs(sim - exact)) / u_top)
    assert err < 0.03, f"Couette profile error {err:.3%} of u_top"
    # halfway bounce-back puts the wall at z = 0 exactly, so the first fluid row is at z = dx/2
    assert abs(z_um[0] - 0.5 * g.dx_um) < 1e-9
    assert np.abs(flow.uz).max() / u_top < 1e-6, "uz must vanish in pure shear flow"


def test_shear_linearity():
    """Doubling the shear rate doubles the field (same grid, same matched Re and tau)."""
    g = trench_grid(width_um=160, depth_um=80, height_um=160, dx_um=8)
    slow = solve_flow(g, 25.0)
    fast = solve_flow(g, 50.0)
    assert slow.converged and fast.converged
    assert abs(slow.info["tau"] - fast.info["tau"]) < 1e-12
    scale = np.abs(fast.ux).max()
    for a, b in ((slow.ux, fast.ux), (slow.uz, fast.uz)):
        dev = float(np.abs(2.0 * a - b).max() / scale)
        assert dev < 0.02, f"nonlinear response {dev:.2%} of max|u|"


def test_trench_field_and_mass_conservation():
    """Trench cavity: converged, no velocity in solid, slower than the outer flow, zero net vertical flux."""
    g = trench_grid(width_um=160, depth_um=80, height_um=160, dx_um=8)
    shear = 50.0
    flow = solve_flow(g, shear)
    u_top = shear * g.height_um * 1e-4
    assert flow.converged
    assert flow.ux.shape == g.shape and flow.uz.shape == g.shape
    assert flow.ux.dtype == np.float64 and flow.uz.dtype == np.float64
    assert np.all(flow.ux[g.solid] == 0.0) and np.all(flow.uz[g.solid] == 0.0)
    for k in ("steps", "tau", "u_lid_lb", "re", "max_mach", "wall_s", "residual"):
        assert k in flow.info

    speed = np.hypot(flow.ux, flow.uz)
    below = (g.z_um[:, None] < 0.0) & (~g.solid)
    above = (g.z_um[:, None] > 0.0) & (~g.solid)
    assert speed[below].max() < speed[above].max()

    # steady incompressible flow in a periodic domain: no net vertical flux through a trench cross-section
    iz = int(np.argmin(np.abs(g.z_um + 0.5 * g.depth_um)))          # row mid-depth inside the trench
    trench = (g.x_um > g.x_left_um) & (g.x_um < g.x_left_um + g.width_um)
    cols = trench & ~g.solid[iz]
    assert cols.sum() > 3
    flux = abs(float(flow.uz[iz, cols].sum()))
    assert flux < 1e-3 * u_top * cols.sum(), f"net vertical flux {flux:.3e} cm/s"

    s = eddy_summary(flow)
    assert set(s) == {"u_max_in_trench_cm_s", "has_recirculation", "penetration_um"}
    assert isinstance(s["has_recirculation"], bool)
    assert 0.0 < s["penetration_um"] <= g.depth_um
    assert s["u_max_in_trench_cm_s"] > 0.0


def test_deep_narrow_cavity_recirculates():
    """Deep narrow cavity: an eddy forms and the outer flow does not reach the floor."""
    g = trench_grid(width_um=64, depth_um=128, height_um=128, dx_um=8)
    flow = solve_flow(g, 50.0)
    assert flow.converged
    s = eddy_summary(flow)
    assert s["has_recirculation"], "expected reverse flow inside the deep cavity"
    pen = penetration_depth_um(flow)
    assert pen == s["penetration_um"]
    assert 0.0 < pen < g.depth_um, f"penetration {pen} um vs depth {g.depth_um} um"
