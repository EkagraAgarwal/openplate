"""Tests for platesim.bubbles.shading (plan item E2, coupling half).

Checks the geometry of the electrode shading mask, the seeded H2 nucleation criterion and the
one-step wrapper around the E2a physics.  Everything is a handful of small (<= 120 x 60) arrays, so
the module runs in a few seconds.
"""
import math

import numpy as np

from platesim.bubbles import fritz_detachment_radius_um
from platesim.bubbles.shading import (
    Bubble,
    attached_bubbles,
    h2_nucleation_sites,
    shading_mask,
    step_bubbles,
)

DX_CM = 1.0e-4                      # 1 um cells
DX_UM = 1.0e4 * DX_CM
ORIGIN = (0.5 * DX_CM, 0.5 * DX_CM)  # cell [0, 0] centred at (0.5, 0.5) um


def _flat_front(nr, nz, z_front_um):
    """psi > 0 above a flat wall at z_front_um (cell centres at (i + 0.5, j + 0.5) um)."""
    z = (np.arange(nz) + 0.5) * DX_UM
    return np.broadcast_to(z - z_front_um, (nr, nz)).copy()


def test_single_bubble_footprint_is_two_r_sin_theta():
    """One attached bubble shades exactly 2 R sin(theta) of a flat front, within one dx."""
    nr, nz = 120, 60
    z_f = 20.5
    R_um, theta = 10.0, 45.0
    psi = _flat_front(nr, nz, z_f)
    x_b = 50.5
    z_b = z_f + R_um * math.cos(math.radians(theta))          # sphere centre at contact angle theta
    b = Bubble(x_b, z_b, R_um, attached=True, theta_deg=theta)

    mask = shading_mask(psi, DX_CM, [b], periodic=True, origin_cm=ORIGIN)
    assert mask.shape == psi.shape
    assert mask.min() >= 0.0 and mask.max() <= 1.0

    row = 20                                                   # psi = 0 row
    assert abs(psi[0, row]) < 1e-12
    assert mask[50, row] == 0.0                                # dead centre of the contact disc
    assert mask[0, row] == 1.0                                 # far from the bubble

    r_c = R_um * math.sin(math.radians(theta))
    shaded_um = float((1.0 - mask[:, row]).sum()) * DX_UM      # anti-aliased coverage integral
    print(f"\nfootprint: r_c = {r_c:.3f} um, masked length {shaded_um:.3f} um "
          f"(expected {2 * r_c:.3f} um, dx = {DX_UM} um)")
    assert abs(shaded_um - 2.0 * r_c) <= 1.0 * DX_UM, shaded_um

    # the mask is a local footprint: cells two contact radii away are untouched
    assert mask[50 + int(math.ceil(2 * r_c)), row] == 1.0

    # the tracker freezes the centre at the sphere-touch distance R (not R cos(theta)); the
    # footprint must survive that placement too
    b_touch = Bubble(x_b, z_f + R_um, R_um, attached=True, theta_deg=theta)
    mask_touch = shading_mask(psi, DX_CM, [b_touch], periodic=True, origin_cm=ORIGIN)
    shaded_touch = float((1.0 - mask_touch[:, row]).sum()) * DX_UM
    assert abs(shaded_touch - 2.0 * r_c) <= 1.0 * DX_UM, shaded_touch


def test_no_bubbles_mask_is_exactly_one():
    """No (or no attached) bubbles -> the multiplicative factor is exactly 1 everywhere."""
    psi = _flat_front(40, 30, 10.5)
    for bubs in (None, [], [Bubble(5.0, 15.0, 4.0, attached=False, theta_deg=45.0)]):
        mask = shading_mask(psi, DX_CM, bubs, origin_cm=ORIGIN)
        assert np.all(mask == 1.0), np.unique(mask)


def test_periodic_wrap_is_symmetric_across_the_seam():
    """A bubble straddling the x seam shades both sides of the periodic boundary equally."""
    nr, nz = 64, 40
    z_f = 10.5
    psi = _flat_front(nr, nz, z_f)
    R_um, theta = 10.0, 45.0
    z_b = z_f + R_um * math.cos(math.radians(theta))
    b = Bubble(0.5, z_b, R_um, attached=True, theta_deg=theta)  # cell 0 centre

    mask = shading_mask(psi, DX_CM, [b], periodic=True, origin_cm=ORIGIN)
    row = 10
    assert mask[1, row] == mask[nr - 1, row]                   # distance 1 um on both sides
    assert mask[4, row] == mask[nr - 4, row]                   # distance 4 um on both sides
    assert mask[0, row] == 0.0
    print(f"\nperiodic: mask[1] = {mask[1, row]:.6f}, mask[nr-1] = {mask[nr - 1, row]:.6f}; "
          f"no-wrap would leave mask[{nr - 1}] = 1")

    # without wrapping the far side of the seam is unshaded
    nowrap = shading_mask(psi, DX_CM, [b], periodic=False, origin_cm=ORIGIN)
    assert nowrap[nr - 1, row] == 1.0


def test_detached_bubbles_do_not_shade():
    """Only attached bubbles shade: a detached bubble leaves the mask untouched."""
    psi = _flat_front(60, 40, 10.5)
    z_b = 10.5 + 10.0 * math.cos(math.radians(45.0))            # sphere centre at contact angle 45
    attached = Bubble(30.5, z_b, 10.0, attached=True, theta_deg=45.0)
    detached = Bubble(30.5, z_b, 10.0, attached=False, theta_deg=45.0)

    m_att = shading_mask(psi, DX_CM, [attached], origin_cm=ORIGIN)
    m_det = shading_mask(psi, DX_CM, [detached], origin_cm=ORIGIN)
    assert m_att.min() == 0.0
    assert np.all(m_det == 1.0)

    # attached_bubbles keeps the attached one only, from tuples and from Bubble objects
    rows = [(30.5, 15.5, 10.0, True, 45.0), (10.0, 15.5, 8.0, False, 45.0)]
    kept = attached_bubbles(rows)
    assert len(kept) == 1 and kept[0].attached and kept[0].x_um == 30.5
    assert attached_bubbles([detached]) == []
    assert abs(kept[0].contact_radius_um - 10.0 * math.sin(math.radians(45.0))) < 1e-12


def test_h2_nucleation_threshold_and_reproducibility():
    """Sites appear only above the supersaturation threshold and are reproducible with a seed."""
    nr, nz = 8, 8
    j_field = np.zeros((nr, nz))
    j_field[:4, :] = 0.01                        # 10 mA/cm2 of HER on the top half only
    j_field[4:, :] = 0.0

    res = h2_nucleation_sites(j_field, supersat_threshold=1.0, rng=np.random.default_rng(5),
                              rate_per_cell=10.0)
    sup = res["supersat_field"]
    print(f"\nnucleation: wall supersaturation {sup.max():.2f} above the heated cells, "
          f"{sup.min():.2f} below; {res['i'].size} seeds of {res['n_candidates']} candidates")
    assert res["n_candidates"] == int(np.count_nonzero(j_field > 0.0))
    assert res["i"].size > 0
    assert np.all(res["i"] < 4), "a cold cell must never seed"
    assert np.all(sup[res["i"], res["j"]] > 1.0)

    # exact same seed -> bit-identical sites and radii
    again = h2_nucleation_sites(j_field, supersat_threshold=1.0, rng=np.random.default_rng(5),
                                rate_per_cell=10.0)
    assert np.array_equal(res["i"], again["i"]) and np.array_equal(res["j"], again["j"])
    assert np.array_equal(res["R0_um"], again["R0_um"])

    # nothing crosses a threshold above the achievable supersaturation
    none = h2_nucleation_sites(j_field, supersat_threshold=100.0, rng=np.random.default_rng(5))
    assert none["n_candidates"] == 0 and none["i"].size == 0
    assert none["x_um"] is None

    # geometry handed back in um when the grid is given
    located = h2_nucleation_sites(j_field, supersat_threshold=1.0, rng=5, rate_per_cell=10.0,
                                  dx_cm=DX_CM)
    assert located["x_um"] is not None and located["z_um"] is not None
    assert np.allclose(located["x_um"], (0.5 + located["i"]) * DX_UM)
    assert np.allclose(located["z_um"], (0.5 + located["j"]) * DX_UM)


def test_step_bubbles_detaches_beyond_fritz_radius():
    """A bubble larger than the Fritz radius leaves the wall in one step_bubbles call."""
    theta = 2.0
    r_fritz = fritz_detachment_radius_um(theta)
    b = Bubble(50.0, 20.0, 1.2 * r_fritz, attached=True, theta_deg=theta)
    assert b.R_um > r_fritz

    out = step_bubbles([b], dt=0.01, flow=None, supersat_field=0.0)
    kinds = [e["kind"] for e in out["events"]]
    print(f"\nFritz radius {r_fritz:.1f} um, bubble {1.2 * r_fritz:.1f} um -> events {kinds}")
    assert "detach" in kinds
    assert b.attached is False
    assert out["bubbles"] is out["bubbles"] and out["bubbles"][0] is b


def test_step_bubbles_moves_a_free_bubble_with_the_flow():
    """A free bubble in a uniform flow drifts at the imposed velocity (gravity out of plane)."""
    from platesim.trenchflow.geometry import FlowField, flat_grid

    grid = flat_grid(pitch_um=200.0, height_um=100.0, dx_um=4.0)
    ux = 0.005                                                 # cm/s = 50 um/s
    flow = FlowField(grid=grid, ux=np.full(grid.shape, ux), uz=np.zeros(grid.shape),
                     shear_rate_s=0.0, info={"kind": "uniform"})
    b = Bubble(50.0, 20.0, 5.0, attached=False, theta_deg=45.0)
    dt = 0.1
    step_bubbles([b], dt=dt, flow=flow, supersat_field=0.0, gravity_dir=(0.0, 0.0))
    print(f"\nuniform flow: dx = {b.x_um - 50.0:.3f} um in {dt} s, z = {b.z_um:.3f} um")
    assert abs((b.x_um - 50.0) - ux * dt * 1e4) < 1e-6
    assert abs(b.z_um - 20.0) < 1e-9
    assert b.attached is False


def test_step_bubbles_grows_above_saturation():
    """A free bubble grows when the local field is supersaturated and shrinks at zero supersat."""
    grow = Bubble(0.0, 0.0, 10.0, attached=False, theta_deg=45.0)
    shrink = Bubble(0.0, 0.0, 10.0, attached=False, theta_deg=45.0)
    step_bubbles([grow], dt=0.01, flow=None, supersat_field=5.0)
    step_bubbles([shrink], dt=0.01, flow=None, supersat_field=0.0)
    print(f"\ngrowth: supersat 5 -> {grow.R_um:.4f} um, supersat 0 -> {shrink.R_um:.4f} um")
    assert grow.R_um > 10.0
    assert shrink.R_um < 10.0

    # an (nr, nz) supersaturation field is sampled in um coordinates
    field = np.full((10, 10), 5.0)
    b = Bubble((0.5 + 5) * DX_UM, (0.5 + 5) * DX_UM, 10.0, attached=False, theta_deg=45.0)
    out = step_bubbles([b], dt=0.01, flow=None, supersat_field=field, dx_cm=DX_CM,
                       origin_cm=ORIGIN)
    assert abs(out["supersat"][0] - 5.0) < 1e-12 and b.R_um > 10.0
