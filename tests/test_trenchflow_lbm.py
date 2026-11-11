"""Tests for platesim.trenchflow.cavity_lbm (D2Q9 shear flow over a periodic trench).

Small grids only (dx = 4-8 um) so the whole module runs in a few seconds.
Analytic reference: Couette flow u_x = G z over a flat wall, with the no-slip plane at z = 0 and the
first fluid row centre at z = dx/2 (halfway bounce-back).
"""
import numpy as np

from platesim.trenchflow.cavity_lbm import eddy_summary, penetration_depth_um, solve_flow
from platesim.trenchflow.geometry import CavityGrid, flat_grid, trench_grid


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


# --- sign of the driving: negative shear reverses the cavity flow, it does not stop it -----------

def _symmetric_trench(dx_um: float = 6.0) -> CavityGrid:
    """Small trench symmetric about the periodic domain centre, so x -> -x is a pure index reversal."""
    g = trench_grid(width_um=48, depth_um=28, height_um=100, dx_um=dx_um)
    assert np.array_equal(g.solid[:, ::-1], g.solid), "test grid is not mirror symmetric"
    assert abs(0.5 * g.pitch_um - (g.x_left_um + 0.5 * g.width_um)) < 1e-12
    return g


def test_zero_shear_is_at_rest():
    """G = 0 is the only resting case."""
    g = _symmetric_trench()
    flow = solve_flow(g, 0.0)
    assert flow.converged
    assert np.all(flow.ux == 0.0) and np.all(flow.uz == 0.0)
    assert flow.shear_rate_s == 0.0
    assert flow.info["u_top_cm_s"] == 0.0
    assert flow.info["steps"] == 0


def test_negative_shear_reverses_the_flow_exactly():
    """-G is the mirror image of +G: ux -> -ux[:, ::-1], uz -> uz[:, ::-1], same Re/tau/u_lid."""
    g = _symmetric_trench()
    shear = 50.0
    pos = solve_flow(g, shear)
    neg = solve_flow(g, -shear)
    assert pos.converged and neg.converged
    scale = max(np.abs(pos.ux).max(), np.abs(pos.uz).max(), 1e-30)
    dev_x = float(np.abs(neg.ux + pos.ux[:, ::-1]).max() / scale)
    dev_z = float(np.abs(neg.uz - pos.uz[:, ::-1]).max() / scale)
    assert dev_x < 1e-10, f"ux mirror error {dev_x:.2e} of max|u|"
    assert dev_z < 1e-10, f"uz mirror error {dev_z:.2e} of max|u|"
    # Reynolds matching is done with |G|, so the lattice parameters are identical
    for k in ("tau", "u_lid_lb", "re", "re_sim", "nu_lb", "re_clamped", "n_fluid_rows"):
        assert neg.info[k] == pos.info[k], k
    # ... while the reported driving keeps the sign of G
    assert neg.shear_rate_s == -shear and pos.shear_rate_s == shear
    assert neg.info["u_top_cm_s"] == -pos.info["u_top_cm_s"] < 0.0


def test_negative_shear_is_not_at_rest_and_flows_in_minus_x():
    """A negative shear drives the lid in -x: the field is not at rest and the far field follows -G."""
    g = _symmetric_trench()
    shear = 50.0
    flow = solve_flow(g, -shear)
    assert flow.converged
    u_top = shear * (g.height_um - g.deposit_um) * 1e-4               # cm/s, magnitude
    assert np.abs(flow.ux).max() > 0.5 * u_top, "negative shear returned a nearly resting fluid"
    assert np.abs(flow.uz).max() > 1e-3 * u_top, "no reverse cell: the cavity flow did not turn over"

    # far field (topmost fluid row) is the analytic Couette value with the sign of G
    top = ~g.solid[-1]
    assert top.sum() > 3
    u_top_row = flow.ux[-1, top]
    assert np.all(u_top_row < 0.0)
    exact = -shear * g.z_um[-1] * 1e-4
    dev = abs(float(u_top_row.mean()) - exact) / abs(exact)
    assert dev < 0.02, f"far-field error {dev:.2%} of the analytic Couette value"
    assert flow.info["u_top_cm_s"] < 0.0


def test_negative_shear_recirculation_matches_positive():
    """eddy_summary detects recirculation as flow against the lid, so +G and -G agree."""
    g = _symmetric_trench()
    pos = eddy_summary(solve_flow(g, 50.0))
    neg = eddy_summary(solve_flow(g, -50.0))
    assert pos["has_recirculation"], "expected an eddy in the trench for +G"
    assert neg["has_recirculation"] == pos["has_recirculation"]
    assert neg["has_recirculation"] is True
    ref = max(pos["u_max_in_trench_cm_s"], 1e-30)
    assert abs(neg["u_max_in_trench_cm_s"] - pos["u_max_in_trench_cm_s"]) <= 1e-10 * ref
    assert abs(neg["penetration_um"] - pos["penetration_um"]) <= 1e-9 * max(g.depth_um, 1.0)
    assert 0.0 < neg["penetration_um"] <= g.depth_um

