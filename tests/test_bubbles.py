"""Tests for platesim.bubbles (plan item E2a): correlations and the Lagrangian bubble tracker.

Everything here is cheap (a few small Stokes solves, one 8 s tracking run); the whole module is
well under a minute. The printed numbers are the ones quoted in the E2a report.
"""
import math

import numpy as np

from platesim.bubbles import (
    P_ATM_PA,
    contact_disc_radius_um,
    diffusive_growth_rate_um_s,
    fritz_detachment_radius_um,
    henry_concentration_mol_m3,
    laplace_overpressure_Pa,
    rise_velocity,
    shading_fraction,
    track_bubbles,
)
from platesim.trenchflow.geometry import FlowField, flat_grid, trench_grid
from platesim.trenchflow.stokes import solve_stokes


def test_rise_velocity_ratio_and_r2_scaling():
    """Stokes/Hadamard ratio is exactly 2/3, both scale as R^2, and 10 um H2 rises at ~0.3 mm/s."""
    small, large = rise_velocity(5.0), rise_velocity(10.0)
    ratio = small["U_stokes_m_s"] / small["U_hr_m_s"]
    assert abs(ratio - 2.0 / 3.0) < 1e-15, ratio
    for key in ("U_stokes_m_s", "U_hr_m_s"):
        assert abs(large[key] / small[key] - 4.0) < 1e-12, (key, large[key] / small[key])

    rv = rise_velocity(10.0)
    print(f"\n10 um H2 bubble in water: Stokes {rv['U_stokes_mm_s']:.4f} mm/s, "
          f"Hadamard-Rybczynski {rv['U_hr_mm_s']:.4f} mm/s, Re = {rv['reynolds']:.3g}")
    assert 0.2e-3 <= rv["U_hr_m_s"] <= 0.4e-3, rv["U_hr_mm_s"]
    assert not rv["re_gt_1"] and rv["warning"] is None

    big = rise_velocity(1000.0)
    print(f"1 mm bubble: {big['U_hr_mm_s']:.1f} mm/s, Re = {big['reynolds']:.3g}, "
          f"flag={big['re_gt_1']}")
    assert big["re_gt_1"] and big["warning"] is not None


def test_fritz_detachment_radius_water():
    """Fritz radius for water at 45 deg is ~1.3 mm; linear in theta, sqrt in sigma."""
    r45 = fritz_detachment_radius_um(45.0)
    print(f"\nFritz radius, water 45 deg: {r45:.0f} um = {r45 / 1e3:.3f} mm")
    assert 500.0 <= r45 <= 2000.0, r45
    assert abs(fritz_detachment_radius_um(90.0) / r45 - 2.0) < 1e-12
    assert abs(fritz_detachment_radius_um(45.0, sigma=4 * 0.072) / r45 - 2.0) < 1e-12
    assert fritz_detachment_radius_um(0.0) == 0.0


def test_laplace_henry_and_shading():
    """Capillary pressure, Henry solubility and the geometric shading fraction."""
    p10 = laplace_overpressure_Pa(10.0)
    assert abs(p10 - 2.0 * 0.072 / 1e-5) < 1e-6
    assert abs(henry_concentration_mol_m3(P_ATM_PA) - 0.78) < 1e-9
    rc = contact_disc_radius_um(10.0, 45.0)
    assert abs(rc - 10.0 / math.sqrt(2.0)) < 1e-12
    f = shading_fraction(10.0, 45.0, 1e4)
    assert abs(f - 1e4 * math.pi * rc * rc / 1e8) < 1e-15
    assert shading_fraction(10.0, 0.0, 1e6) == 0.0
    assert shading_fraction(10.0, 45.0, 1e10) == 1.0
    print(f"\n10 um bubble, 45 deg: Laplace {p10 / 1e3:.1f} kPa, contact disc r = {rc:.2f} um, "
          f"1e4 bubbles/cm2 shade {100 * f:.2f} %")


def test_epstein_plesset_shrinks_and_grows():
    """Zero supersaturation dissolves (Laplace), supersaturation 1 grows; signs of the rate agree."""
    shrink = track_bubbles(None, 10.0, 0.0, 0.0, t_end_s=2.0, dt_s=0.05, gravity_dir=(0.0, 0.0),
                           supersat=0.0)
    grow = track_bubbles(None, 10.0, 0.0, 0.0, t_end_s=2.0, dt_s=0.05, gravity_dir=(0.0, 0.0),
                         supersat=1.0)
    r_shrink = shrink["R_um"][:, 0]
    r_grow = grow["R_um"][:, 0]
    print(f"\nR(10 um, supersat 0): {r_shrink[0]:.2f} -> {r_shrink[-1]:.2f} um; "
          f"supersat 1: {r_grow[0]:.2f} -> {r_grow[-1]:.2f} um")
    assert r_shrink[-1] < 0.8 * r_shrink[0]
    assert r_grow[-1] > 1.2 * r_grow[0]
    assert diffusive_growth_rate_um_s(10.0, supersat=0.0, t_s=0.1) < 0.0
    assert diffusive_growth_rate_um_s(10.0, supersat=1.0, t_s=0.1) > 0.0
    # the trajectory does not move in a quiescent liquid with out-of-plane gravity
    assert np.allclose(shrink["x_um"], 0.0) and np.allclose(shrink["z_um"], 0.0)


def test_epstein_plesset_sqrt_t_growth():
    """At long times the diffusive growth is R ~ sqrt(t): log-log slope within 0.1 of 0.5."""
    res = track_bubbles(None, 5.0, 0.0, 0.0, t_end_s=10.0, dt_s=0.02, gravity_dir=(0.0, 0.0),
                        supersat=1.0)
    t, R = res["t_s"], res["R_um"][:, 0]
    sel = t >= 4.0
    slope = float(np.polyfit(np.log(t[sel]), np.log(R[sel]), 1)[0])
    print(f"\nR(t) from {R[0]:.2f} to {R[-1]:.2f} um over {t[-1]:.1f} s; "
          f"log-log slope over t >= 4 s: {slope:.3f}")
    assert abs(slope - 0.5) < 0.1, slope


def test_tracker_quiescent_rise_at_terminal_velocity():
    """Quiescent liquid, gravity -z: the bubble rises at the Hadamard velocity to 1 %."""
    R0 = 10.0
    res = track_bubbles(None, R0, 0.0, 0.0, t_end_s=2.0, dt_s=0.05, gravity_dir=(0.0, -1.0),
                        sigma=0.0, supersat=0.0)
    t, z = res["t_s"], res["z_um"][:, 0]
    v_um_s = float((z[-1] - z[0]) / (t[-1] - t[0]))
    expected = rise_velocity(R0)["U_hr_cm_s"] * 1e4
    rel = abs(v_um_s - expected) / expected
    print(f"\nrise: measured {v_um_s:.3f} um/s, Hadamard {expected:.3f} um/s, "
          f"relative error {rel:.2e}; R {res['R_um'][0, 0]:.3f} -> {res['R_um'][-1, 0]:.3f} um")
    assert rel < 0.01, rel
    assert np.allclose(res["R_um"], R0), "sigma=0 must freeze the radius"
    assert np.allclose(res["x_um"], 0.0)
    assert not res["attached"].any() and res["events"] == []
    assert not res["re_gt_1"].any() and res["warnings"] == []

    big = track_bubbles(None, 1000.0, 0.0, 0.0, t_end_s=0.001, dt_s=0.001, gravity_dir=(0.0, 0.0),
                        sigma=0.0)
    assert big["re_gt_1"][0] and big["warnings"], "the tracker must surface the Re > 1 warning"

    # gravity along +x: the bubble must rise along -x and not move in z
    side = track_bubbles(None, R0, 0.0, 0.0, t_end_s=1.0, dt_s=0.05, gravity_dir=(1.0, 0.0),
                         sigma=0.0)
    vx = float(side["x_um"][-1, 0] / side["t_s"][-1])
    print(f"gravity +x: drift {vx:.3f} um/s in x, z stays {side['z_um'][-1, 0]:.1e} um")
    assert abs(vx + expected) < 0.01 * expected
    assert abs(side["z_um"][-1, 0]) < 1e-12


def test_tracker_follows_uniform_flow():
    """Uniform horizontal liquid velocity, gravity out of plane: two bubbles drift with the flow."""
    grid = flat_grid(pitch_um=200.0, height_um=200.0, dx_um=4.0)
    ux = 0.005                                          # cm/s = 50 um/s
    flow = FlowField(grid=grid, ux=np.full(grid.shape, ux), uz=np.zeros(grid.shape),
                     shear_rate_s=0.0, info={"kind": "uniform"})
    res = track_bubbles(flow, [5.0, 8.0], [50.0, 30.0], 60.0, t_end_s=2.0, dt_s=0.05,
                        gravity_dir=(0.0, 0.0), sigma=0.0)
    t = res["t_s"]
    v_um_s = (res["x_um"][-1] - res["x_um"][0]) / (t[-1] - t[0])
    z_drift = np.abs(res["z_um"][-1] - res["z_um"][0]).max()
    print(f"\nuniform flow: bubbles moved {res['x_um'][-1] - res['x_um'][0]} um in {t[-1]:.1f} s "
          f"({v_um_s} um/s vs {ux * 1e4:.2f} um/s imposed), max z drift {z_drift:.2e} um")
    assert np.allclose(v_um_s, ux * 1e4, rtol=0.01, atol=0.0)
    assert z_drift < 1e-9
    assert not res["attached"].any()


def test_tracker_attaches_and_detaches_at_fritz_radius():
    """A bubble on the wall sticks, grows in place and leaves once R exceeds the Fritz radius."""
    grid = flat_grid(pitch_um=200.0, height_um=200.0, dx_um=4.0)
    R0 = 5.0
    res = track_bubbles(FlowField(grid=grid, ux=np.zeros(grid.shape), uz=np.zeros(grid.shape),
                                  shear_rate_s=0.0), R0, 100.0, 4.0, t_end_s=6.0, dt_s=0.05,
                        gravity_dir=(0.0, -1.0), supersat=30.0, theta_deg=2.0)
    kinds = [e["kind"] for e in res["events"]]
    attach = next((e for e in res["events"] if e["kind"] == "attach"), None)
    detach = next((e for e in res["events"] if e["kind"] == "detach"), None)
    print(f"\nwall bubble (theta 2 deg): events "
          f"{[(e['kind'], round(e['t_s'], 2), round(e['R_um'], 1)) for e in res['events']]}, "
          f"fritz {res['fritz_radius_um']:.0f} um")
    assert attach is not None and "attach" in kinds
    assert detach is not None, "a growing bubble must pass the Fritz radius and leave the wall"
    assert detach["R_um"] > res["fritz_radius_um"]
    # the position is frozen while attached
    sel = (res["t_s"] >= attach["t_s"]) & (res["t_s"] < detach["t_s"])
    assert res["attached"][sel, 0].all()
    x_held = res["x_um"][sel, 0]
    z_held = res["z_um"][sel, 0]
    assert np.ptp(x_held) < 1e-9 and np.ptp(z_held) < 1e-9
    assert res["R_um"][-1, 0] > detach["R_um"]
    assert res["z_um"][-1, 0] > detach["z_um"], "after detachment the bubble rises"


def test_tracker_confined_by_trench_eddy():
    """A bubble released in a 1:1 trench under the closed Stokes eddy stays inside for the run.

    Gravity is out of plane, so there is no in-plane buoyancy: the closed recirculation is the only
    driver. The supersaturation is set to the Laplace equilibrium ``2 sigma / (R p_atm)`` so the
    radius is stationary and the test isolates the advection.
    """
    grid = trench_grid(width_um=80.0, depth_um=80.0, dx_um=4.0)
    flow = solve_stokes(grid, 80.0)
    assert flow.converged
    R0 = 10.0
    supersat = float(laplace_overpressure_Pa(R0) / P_ATM_PA)
    x0 = grid.x_left_um + 0.5 * grid.width_um
    res = track_bubbles(flow, R0, x0, -40.0, t_end_s=8.0, dt_s=0.05, gravity_dir=(0.0, 0.0),
                        supersat=supersat)
    x, z, R = res["x_um"][:, 0], res["z_um"][:, 0], res["R_um"][:, 0]
    print(f"\neddy: released at ({x0:.1f}, -40.0) um, ends at ({x[-1]:.1f}, {z[-1]:.1f}) um, "
          f"R {R[0]:.2f} -> {R[-1]:.2f} um, x in [{x.min():.1f}, {x.max():.1f}], "
          f"z in [{z.min():.1f}, {z.max():.1f}] um, "
          f"events {[(e['kind'], round(e['t_s'], 2)) for e in res['events']]}")
    assert x.min() > grid.x_left_um and x.max() < grid.x_left_um + grid.width_um
    assert z.min() > -grid.depth_um and z.max() < 0.0
    assert x.max() - x.min() > 20.0, "the eddy should carry the bubble around, not freeze it"
    assert abs(R[-1] - R0) < 0.01 * R0

