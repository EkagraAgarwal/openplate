"""Solutal (Boussinesq) buoyancy in the feature engine's Stokes cavity flow (plan v2 D4b).

The coupon hangs vertically, so gravity lies in the wafer plane: inside a trench the electrolyte is
depleted near the walls and floor (ctil = 0.6-0.9) and depleted CuSO4 is lighter (beta ~ 0.15 L/mol),
so the depleted fluid rises against gravity and drives a flow that superposes on the shear flow.

This module checks the engine chain Params.buoyancy -> f = g beta c_b (ctil - 1) gravity_dir ->
solve_stokes(body_force=...) on the narrow Trench 1 cell (d = 227 um, H = 88 um, f_open = 0.2, dx = 8 um,
delta = 64 um) and reports the max buoyant speed and the floor/field ratio for the three orientations:

  (1, 0)  across the trench: horizontal force in the cross-section (strongest drive)
  (0, -1) wafer face-up: vertical force, gravity into the wafer
  (0, 0)  along the trench: no in-plane force -> must reproduce buoyancy=False exactly

The sign is tested directly: a depleted patch in the trench moves AGAINST gravity (up for (0, -1)) and
the field is exactly mirrored when gravity is reversed.  cu_balance_rel closes to |.| < 0.005 in every
run, i.e. the buoyancy-driven flow conserves Cu2+ through the conservative MAC advection operator.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine.params import Params
from platesim.feature.engine.recipes import pulse_reverse
from platesim.feature.engine.run_case import run_case
from platesim.feature.engine.sim import ViaSim
from platesim.feature.ripple_study import surface_stats

D_UM, H_UM, F_OPEN, DX_UM, DELTA_UM, J_NET = 227.0, 88.0, 0.2, 8.0, 64.0, 15.0
SHEARS = (0.0, 59.0)                       # no shear (buoyancy only) and the narrow-site cross-flow
GDIRS = ((1.0, 0.0), (0.0, -1.0), (0.0, 0.0))


def _params(**kw):
    base = dict(geometry="trench_full", suppressor=False, ohmic=True, dx_um=DX_UM, delta_um=DELTA_UM,
                k_mix=0.0, max_overburden_um=140.0, H_um=H_UM, flow_solver="stokes")
    base.update(kw)
    return Params(**base)


def _recipe(hours=0.5):
    return pulse_reverse(J_NET, hours, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)


_RUNS: dict = {}


def _run(shear, gdir, buoyancy=True):
    """(run_case result, ViaSim) of the narrow case for one (shear, gravity) setting, cached."""
    key = (float(shear), tuple(gdir), bool(buoyancy))
    if key not in _RUNS:
        p = _params(flow_shear_s=float(shear), buoyancy=bool(buoyancy), gravity_dir=tuple(gdir))
        res, sim = run_case(D_UM, 22.0, _recipe(), p, f_open=F_OPEN,
                            label=f"buoy_s{shear:g}_g{gdir[0]:+g}{gdir[1]:+g}", out=None,
                            return_sim=True)
        _RUNS[key] = (res, sim)
    return _RUNS[key]


def _depleted_sim(gdir):
    """ViaSim of the initial narrow-trench geometry with a depleted patch (ctil = 0.6) in the trench."""
    p = _params(flow_shear_s=0.0, buoyancy=True, gravity_dir=tuple(gdir))
    sim = ViaSim(D_UM, 22.0, p, F_OPEN, _recipe())
    g = sim.g
    z = g.zc - g.H
    patch = (z < -20e-4) & (z > -60e-4) & (np.abs(g.rc - g.x_c) < 0.5 * g.a)
    assert patch.sum() > 20, "the depleted patch must cover fluid cells"
    sim.st['ctil'][patch] = 0.6
    return sim, patch


# ------------------------------------------------------------------ error handling
def test_buoyancy_requires_the_stokes_solver():
    """buoyancy=True with flow_solver='lbm' is rejected with a clear message, not silently ignored."""
    p = _params(flow_solver="lbm", buoyancy=True)
    sim = ViaSim(D_UM, 22.0, p, F_OPEN, _recipe())
    with pytest.raises(ValueError, match="buoyancy"):
        sim.cavity_flow(59.0)
    assert sim.flow_info == {}


# ------------------------------------------------------------------ sign of the body force
def test_depleted_electrolyte_rises_against_gravity():
    """A depleted patch is lighter: it moves against gravity, and reversing gravity reverses the field."""
    up, patch = _depleted_sim((0.0, -1.0))          # gravity into the wafer (-z): depleted fluid rises
    ux, uz = up.cavity_flow(0.0)
    mean_uz = float(uz[patch].mean())
    assert mean_uz > 0.0, f"depleted fluid does not rise against gravity: mean uz = {mean_uz:.3e}"
    down, _ = _depleted_sim((0.0, 1.0))             # gravity reversed: u(-g) = -u(g) (linear in the force)
    ux2, uz2 = down.cavity_flow(0.0)
    assert float(uz2[patch].mean()) < 0.0
    scale = max(np.abs(ux).max(), np.abs(uz).max())
    e = max(np.abs(ux2 + ux).max(), np.abs(uz2 + uz).max()) / scale
    assert e < 1e-10, f"reversing gravity did not reverse the flow: {e:.2e}"
    buoy = up.flow_info["buoyant_speed_cm_s"]
    assert buoy > 0.0 and abs(buoy - scale) < 1e-12 * scale
    print(f"\ndepleted patch ({patch.sum()} cells, ctil 0.6): mean uz = {mean_uz:+.3e} cm/s against "
          f"gravity (0, -1), max buoyant speed {buoy:.3e} cm/s, reversal error {e:.2e}")


# ------------------------------------------------------------------ the three orientations
@pytest.mark.parametrize("shear", SHEARS)
def test_narrow_trench_buoyancy_orientations(shear):
    """Across / face-up / along the trench at each shear: Cu balance closes and the response is reported."""
    rows = []
    for gdir in GDIRS:
        res, sim = _run(shear, gdir)
        r = res["result"]
        bal = r["cu_balance_rel"]
        assert bal is not None and abs(bal) < 0.005, f"gravity {gdir}: cu_balance_rel = {bal}"
        buoy = sim.flow_info.get("buoyant_speed_cm_s", 0.0)
        rows.append((gdir, buoy, float(surface_stats(res)["ratio_mean"])))
        print(f"\nshear {shear:5.1f} 1/s, gravity {gdir}: max buoyant speed {buoy:.3e} cm/s, "
              f"floor/field {rows[-1][2]:.4f}, cu_balance_rel {bal:+.2e}, fill "
              f"{r['fill_fraction_final']:.4f}")
    across, face_up, along = (row[1] for row in rows)
    assert across > 0.0 and face_up > 0.0
    assert along == 0.0, "gravity along the trench must not drive anything"


def test_buoyancy_along_the_trench_is_exactly_idle():
    """gravity_dir = (0, 0) (along the trench) is bit-identical to buoyancy=False, at both shears."""
    for shear in SHEARS:
        off, _ = _run(shear, (0.0, 0.0), buoyancy=False)
        on, _ = _run(shear, (0.0, 0.0))
        assert on["result"] == off["result"], f"shear {shear}: result differs from buoyancy=False"
        assert on["series"] == off["series"], f"shear {shear}: series differ from buoyancy=False"
        print(f"\nshear {shear:5.1f} 1/s: gravity (0, 0) == buoyancy off (result and series identical), "
              f"floor/field {surface_stats(on)['ratio_mean']:.6f}")
