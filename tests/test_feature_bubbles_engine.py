"""Plan v2 E2, engine half: Lagrangian H2 bubbles shading the plating front.

``Params(bubbles=True)`` turns on, per engine step:

  * the H2 partial current of the front: ONE ``platesim.efficiency.side_reactions.faradaic_efficiency``
    solve at the interface-mean current density and overpotential, with its ``j_H2_A_cm2`` distributed over
    the front cells in proportion to the local cathodic current (the module's API is scalar per operating
    point; a per-cell brentq on the shared potential would be ~1e3 solves per step).  ``bubble_pH`` is the
    bath pH that call needs;
  * the wall supersaturation and the nucleation draw (``platesim.bubbles.shading.h2_nucleation_sites``,
    seeded RNG, threshold ``bubble_supersat_threshold``);
  * growth / rise / adhesion / detachment by ``shading.step_bubbles`` on the advanced level set, with the
    cell-centred flow when the cavity flow is available;
  * the interface weights w (the same w that carries the reaction current) multiplied by
    ``shading.shading_mask``: the unshaded area then carries the galvanostatic current, and the front speed
    is masked too, so a shaded patch does not grow.

``bubble_init`` is the injection hook: a list of (x_um, z_um, R_um) attached bubbles at t = 0.
``bubble_supersat`` is an imposed far-field H2 supersaturation (S-1) entering the same Henry balance as the
HER estimate; a bubble is only stable at S-1 = 2 sigma / (R p_atm), which is what the pit test sets.

bubbles=False is the default and leaves every legacy path untouched (no import of platesim.bubbles, no
extra series key); the golden harness verifies the bit-identity to the pre-E2 engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.bubbles.physics import P_ATM_PA, SIGMA_WATER_N_M
from platesim.feature.engine.levelset import column_crossing
from platesim.feature.engine.params import Params, derive
from platesim.feature.engine.recipes import dc
from platesim.feature.engine.sim import ViaSim

D_UM, H_UM, F_OPEN, DX_UM, DELTA_UM, J_MA = 227.0, 88.0, 0.2, 8.0, 64.0, 15.0
HOURS = 0.25
R_UM = 30.0                                     # seeded bubble radius, um (3.75 cells)
X_BUB_UM = 0.5 * D_UM / F_OPEN                  # pitch centre = trench centre (x = x_c)
S_EQ = 2.0 * SIGMA_WATER_N_M / (R_UM * 1e-6 * P_ATM_PA)   # 0.047: S-1 the bubble is stable at

BASE = dict(geometry="trench_full", suppressor=True, ohmic=True, dx_um=DX_UM, delta_um=DELTA_UM,
            k_mix=0.0, max_overburden_um=140.0, H_um=H_UM, flow_shear_s=0.0)

_RUNS: dict[str, dict] = {}


def _run(name, hours=HOURS, **kw):
    if name in _RUNS:
        return _RUNS[name]
    p = Params(**{**BASE, **kw})
    sim = ViaSim(D_UM, 22.0, p, F_OPEN, dc(J_MA, hours))
    raw = sim.run(name)
    s = raw["series"]
    out = dict(raw=raw, sim=sim, params=p, series=s, cu=raw["cu_balance_rel"],
               field=float(s["field_mean_um"][-1]), floor=float(s["floor_mean_um"][-1]),
               floor_min=float(s["floor_min_um"][-1]), floor_max=float(s["floor_max_um"][-1]),
               n_bub=float(s["n_bubbles"][-1]) if "n_bubbles" in s else None,
               shade=float(s["bubble_shade_frac"][-1]) if "bubble_shade_frac" in s else None)
    _RUNS[name] = out
    return out


def _heights(sim):
    """Interface height z [um above the original floor] of every column (upper=... lowest crossing)."""
    zc, h = sim.g.zc[0], []
    for i in range(sim.g.nr):
        _, z = column_crossing(sim.psi[i], zc, upper=False)
        h.append(np.nan if z is None else z / 1e-4)
    return np.asarray(h)


def _fmt(name, r):
    return (f"{name:10s} steps={r['raw']['steps']:3d} cu={r['cu']:+.2e} field={r['field']:.4f} "
            f"floor={r['floor']:.4f} floor_min={r['floor_min']:.4f} floor_max={r['floor_max']:.4f} "
            f"n_bubbles={r['n_bub']} shade_frac={r['shade']}")


# ------------------------------------------------------------------ 1. switch, defaults, validation
def test_defaults_and_validation():
    p = Params()
    assert p.bubbles is False
    assert p.bubble_contact_deg == pytest.approx(45.0)
    assert p.bubble_supersat_threshold == pytest.approx(0.0)
    assert tuple(p.bubble_init) == ()
    derive(p, 22.0)
    with pytest.raises(ValueError, match="bubble_contact_deg"):
        derive(Params(bubble_contact_deg=200.0), 22.0)
    with pytest.raises(ValueError, match="bubble_max"):
        derive(Params(bubble_max=-1), 22.0)


def test_bubbles_off_is_bit_identical():
    """bubbles=False adds no state and no series key, and two identical runs agree value for value (the
    golden harness pins the default to the pre-E2 numbers, bit for bit)."""
    a, b = _run("off_a", hours=0.02), _run("off_b", hours=0.02, bubbles=False)
    assert a["sim"].bubbles == [] and b["sim"].bubbles == []
    assert "n_bubbles" not in a["series"] and "bubble_shade_frac" not in a["series"]
    assert sorted(a["series"]) == sorted(b["series"])
    for key in a["series"]:
        np.testing.assert_array_equal(np.asarray(a["series"][key], dtype=float),
                                      np.asarray(b["series"][key], dtype=float))


# ------------------------------------------------------------------ 2. a seeded bubble pits the floor
def test_seeded_bubble_pits_the_floor_and_the_balance_closes():
    """One attached bubble on the trench floor (R = 30 um, contact disc radius R sin 45 = 21 um ~ 2.6 cells)
    held at its equilibrium supersaturation S-1 = 2 sigma/(R p_atm): after 0.25 h the columns under the
    footprint are still at the original floor level while the rest of the trench has grown ~3-4 um - a pit -
    and the Cu balance still closes (the galvanostatic current is carried by the unshaded area)."""
    base = _run("nobub")
    bub = _run("bub", bubbles=True, bubble_init=((X_BUB_UM, R_UM, R_UM),), bubble_supersat=S_EQ,
               bubble_supersat_threshold=0.5)
    print(_fmt("nobub", base))
    print(_fmt("bub", bub))
    assert bub["n_bub"] == pytest.approx(1.0)
    assert bub["shade"] > 0.005
    assert bub["raw"]["stop_reason"] == "recipe_end"
    assert abs(bub["cu"]) < 5e-3
    h0, h1 = _heights(base["sim"]), _heights(bub["sim"])
    g = bub["sim"].g
    x = g.rc[:, 0] / 1e-4 - g.x_c / 1e-4
    foot = np.abs(x) <= 0.7 * R_UM * np.sin(np.radians(45.0))    # inside the contact disc
    outer = (np.abs(x) > 1.6 * R_UM) & (np.abs(x) < 3.0 * R_UM)  # the floor just outside it
    print("   column heights (um): " + " ".join(f"{xx:+.0f}:{a:.2f}/{b:.2f}"
                                                for xx, a, b in zip(x[foot], h0[foot], h1[foot])))
    depth = float(np.mean(h0[foot] - h1[foot]))
    print(f"   mean deficit under the footprint = {depth:.3f} um "
          f"(bubble {np.mean(h1[foot]):.3f} vs no bubble {np.mean(h0[foot]):.3f})")
    print(f"   floor just outside the footprint (no bubble / bubble): "
          f"{np.mean(h0[outer]):.3f} / {np.mean(h1[outer]):.3f}")
    assert depth > 1.0
    assert np.nanmax(h1[foot]) < 0.6
    # the seeded bubble survived the 0.25 h at its equilibrium supersaturation
    assert len(bub["sim"].bubbles) == 1
    assert 0.6 * R_UM < bub["sim"].bubbles[0].R_um < 1.6 * R_UM


# ------------------------------------------------------------------ 3. no supersaturation -> no bubbles
def test_no_supersaturation_nucleates_nothing():
    """With bubbles=True but no dissolved H2 (bubble_supersat = 0, the HER share of an acid Cu bath is
    negligible at pH 1.7) no cell reaches the threshold, so nothing nucleates and the run is the bubble-free
    one; the series still records zero bubbles."""
    r = _run("nosup", bubbles=True, bubble_supersat=0.0)
    print(_fmt("nosup", r))
    assert r["n_bub"] == pytest.approx(0.0)
    assert r["shade"] == pytest.approx(0.0)
    assert r["sim"].bubbles == []
    assert abs(r["cu"]) < 5e-3
    base = _run("nobub")
    assert abs(r["field"] - base["field"]) < 1e-2
    assert abs(r["floor"] - base["floor"]) < 1e-2
