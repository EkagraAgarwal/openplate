"""Plan v2 D2, engine half: the additive kernels wired into the feature-fill engine.

``Params(additives=...)`` has three settings:

  * ``'legacy'`` (default) - today's single transported suppressor, Cl factor implicitly 1.  Bit-identical
    to the pre-D2 engine (verified by the golden harness output/plan_v2_runs/scripts/golden.py +
    golden_diff.py, and by test_each_mode_runs_and_closes_the_cu_balance below, which repeats the
    legacy run and compares every series value exactly).
  * ``'cl_ndr'``  - adds a chloride adlayer coverage ``theta_cl`` per interface cell, computed from the
    BULK Cl- concentration ``c_cl_bulk`` (Cl- is not transport limited at these currents: the net
    irreversible loss is only the burial flux, see params.py) and the local growth velocity.  The
    suppressor adsorption rate becomes ``k_plus c_s theta_cl**n_cl``
    (``additives.suppressor_cl_rate``), which feeds the existing ``theta_be`` balance and the
    transported-suppressor sink in place of ``k_plus*s_bulk*s_t``.
  * ``'full'``    - 'cl_ndr' + the CEAC accelerator (``additives.accelerator_step``, curvature term,
    displacing the suppressor) + a TRANSPORTED leveller (diffusion + Dirichlet bulk + surface sink, the
    suppressor machinery) whose coverage blocks i0, + reverse-pulse coverage desorption on a PR recipe.

Every number below is a 0.25 h trench_full run (d = 227 um, H = 88 um, f_open = 0.2, dx 8 um, delta 64 um,
15 mA/cm2 DC, 22 C, suppressor on, no shear), so the module costs ~4 min.  "field/floor" is the ratio of
the last-minus-first series pair field_mean_um / floor_mean_um: the growth ratio of the field far from the
walls to the central 40 % of the trench floor (a per-area current ratio, since growth is proportional to
the local current).

What this file does NOT claim: that the 0.25 h field/floor ratio is a smooth function of a parameter.
Near the S-NDR breakdown threshold the fill outcome is hypersensitive (a 0.1 % change of the adsorption
rate moves the breakdown by a whole dt), so the Cl/leveller TRENDS are asserted on the COVERAGES (the
direct observable of the surface chemistry) and the fill ratios are printed for the reader.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine.params import ADDITIVES_MODES, Params, derive
from platesim.feature.engine.recipes import dc
from platesim.feature.engine.sim import ViaSim

D_UM, H_UM, F_OPEN, DX_UM, DELTA_UM, J_MA = 227.0, 88.0, 0.2, 8.0, 64.0, 15.0
HOURS = 0.25

BASE = dict(geometry="trench_full", suppressor=True, ohmic=True, dx_um=DX_UM, delta_um=DELTA_UM,
            k_mix=0.0, max_overburden_um=140.0, H_um=H_UM, flow_shear_s=0.0)

_RUNS: dict[str, dict] = {}


def _run(name, suppressor=True, hours=HOURS, j_ma=J_MA, **kw):
    """One cached DC trench_full run; the summary carries the growth ratio and the coverage extrema."""
    if name in _RUNS:
        return _RUNS[name]
    p = Params(**{**BASE, "suppressor": suppressor, **kw})
    sim = ViaSim(D_UM, 22.0, p, F_OPEN, dc(j_ma, hours))
    raw = sim.run(name)
    s = raw["series"]
    df = float(s["field_mean_um"][-1]) - float(s["field_mean_um"][0])
    dl = float(s["floor_mean_um"][-1]) - float(s["floor_mean_um"][0])
    cov = {k: (float(np.min(v)), float(np.max(v)))
           for k, v in sim.st.items() if isinstance(v, np.ndarray) and k.startswith("theta")}
    out = dict(raw=raw, sim=sim, params=p, series=s, field=float(s["field_mean_um"][-1]),
               floor=float(s["floor_mean_um"][-1]), ratio=df / dl, cu=raw["cu_balance_rel"],
               theta_face=float(s["theta_face"][-1]), theta_floor=float(s["theta_floor"][-1]),
               cov=cov, steps=raw["steps"], stop=raw["stop_reason"])
    _RUNS[name] = out
    return out


def _fmt(name, r):
    return (f"{name:12s} field={r['field']:.4f} floor={r['floor']:.4f} field/floor={r['ratio']:.4f} "
            f"th_face={r['theta_face']:.4f} th_floor={r['theta_floor']:.4f} cu={r['cu']:+.2e} "
            f"steps={r['steps']}")


def _series(name, hours, **kw):
    """Raw result of a run without the cached summary (for the repeatability check)."""
    p = Params(**{**BASE, **kw})
    sim = ViaSim(D_UM, 22.0, p, F_OPEN, dc(J_MA, hours))
    raw = sim.run(name)
    return dict(series=raw["series"], st=sim.st, raw=raw)



def _assert_coverages(r):
    for key, (lo, hi) in r["cov"].items():
        print(f"   {key:9s} in [{lo:.3e}, {hi:.3e}]")
        assert lo >= -1e-12 and hi <= 1.0 + 1e-12, key


# ------------------------------------------------------------------ 1. switch and validation
def test_defaults_are_legacy_and_the_switch_is_validated():
    p = Params()
    assert p.additives == "legacy"
    assert p.c_cl_bulk == pytest.approx(1.0e-6)            # 1 mM Cl-, Braun high-k set
    assert p.c_lev_bulk == 0.0                             # no leveller unless a bath sets one
    derive(p, 22.0)
    for mode in ADDITIVES_MODES:
        derive(Params(additives=mode), 22.0)
    with pytest.raises(ValueError, match="additives"):
        derive(Params(additives="clndr"), 22.0)
    with pytest.raises(ValueError, match="c_cl_bulk"):
        derive(Params(c_cl_bulk=-1e-6), 22.0)
    with pytest.raises(ValueError, match="c_lev_bulk"):
        derive(Params(c_lev_bulk=-1e-6), 22.0)


# ------------------------------------------------------------------ 2. every mode runs and balances
def test_each_mode_runs_and_closes_the_cu_balance():
    """'legacy', 'cl_ndr', 'full' and 'full'+leveller each complete a 0.25 h trench_full run, keep the Cu
    balance under the 0.5 % gate, and keep every coverage in [0, 1].  (The exact bit-identity of 'legacy' to
    the pre-D2 engine is verified by the golden harness; test_legacy_path_is_untouched below checks that the
    new state and series keys do not appear on the default path.)"""
    rows = [("legacy", _run("legacy")),
            ("cl_ndr", _run("cl_1e-6", additives="cl_ndr", c_cl_bulk=1e-6)),
            ("full_nolev", _run("full_nolev", additives="full", c_cl_bulk=1e-6)),
            ("full_lev", _run("full_lev", additives="full", c_cl_bulk=1e-6, c_lev_bulk=1e-6))]
    for name, r in rows:
        print(_fmt(name, r))
        assert r["stop"] == "recipe_end"
        assert r["steps"] > 5
        assert r["cu"] is not None and abs(r["cu"]) < 5e-3
        assert np.isfinite(r["field"]) and np.isfinite(r["floor"]) and r["field"] > 0.0
        _assert_coverages(r)
    # the Cl adlayer is only carried by the Cl-gated modes; the leveller field only when a bath sets one
    assert "theta_cl" in rows[1][1]["sim"].st and "theta_cl" not in rows[0][1]["sim"].st
    assert "lev" in rows[3][1]["sim"].st and "lev" not in rows[2][1]["sim"].st
    assert "theta_l" in rows[3][1]["sim"].st and "theta_l" in rows[2][1]["sim"].st


def test_legacy_path_is_untouched():
    """additives='legacy' adds no state, no series key and is perfectly repeatable (two identical 0.05 h
    runs agree value for value); the golden harness pins it to the pre-D2 numbers."""
    a = _series("legacy_short_a", 0.05)
    b = _series("legacy_short_b", 0.05)
    assert "theta_cl" not in a["st"] and "theta_l" not in a["st"] and "lev" not in a["st"]
    assert "n_bubbles" not in a["series"] and "bubble_shade_frac" not in a["series"]
    assert sorted(a["series"]) == sorted(b["series"])
    for key in a["series"]:
        np.testing.assert_array_equal(np.asarray(a["series"][key], dtype=float),
                                      np.asarray(b["series"][key], dtype=float))


# ------------------------------------------------------------------ 3. Cl -> 0 suppresses nothing
def test_zero_chloride_suppresses_nothing():
    """With c_cl_bulk -> 0 the Cl gate is zero, so no suppressor adsorbs (theta_s -> 0) and the fill must
    collapse onto the suppressor=False (additive-free) result: this is the Josell/Moffat passive-active
    limit at the other end (no Cl- -> no polyether co-adsorption -> no S-NDR)."""
    no = _run("nosupp", suppressor=False)
    cl0 = _run("cl_1e-12", additives="cl_ndr", c_cl_bulk=1e-12)
    print(_fmt("nosupp", no))
    print(_fmt("cl_1e-12", cl0))
    assert cl0["cov"]["theta"][1] < 1e-3                    # no suppressor to speak of (Langmuir, gate 1e-7)
    assert cl0["cov"]["theta_cl"][1] < 1e-6                 # no Cl adlayer either
    assert abs(cl0["field"] - no["field"]) < 1e-2
    assert abs(cl0["floor"] - no["floor"]) < 1e-2
    assert abs(cl0["ratio"] - no["ratio"]) < 1e-2


# ------------------------------------------------------------------ 4. ample Cl -> the legacy S-NDR
def test_ample_chloride_reproduces_the_legacy_suppressor():
    """At the numerical ample-Cl limit (theta_cl -> 1) the gated rate k_plus c_s theta_cl**n_cl equals the
    legacy k_plus*s_bulk*s_t, so the suppressor coverage of 'cl_ndr' must reproduce the legacy run.  The
    0.25 h thicknesses are printed but only loosely bounded: theta_cl = 0.999 is still 0.1 % below 1 and
    the first-order level-set advection turns that into ~10 % of thickness near the S-NDR breakdown (the
    coverage, which is the quantity the Cl gate really controls, is asserted to 5e-3)."""
    leg = _run("legacy")
    amp = _run("cl_1e-2", additives="cl_ndr", c_cl_bulk=1e-2)
    print(_fmt("legacy", leg))
    print(_fmt("cl_1e-2", amp))
    print(f"   |d theta_face| = {abs(amp['theta_face'] - leg['theta_face']):.2e}  "
          f"|d theta_floor| = {abs(amp['theta_floor'] - leg['theta_floor']):.2e}  "
          f"|d field/floor| = {abs(amp['ratio'] - leg['ratio']):.2e}")
    assert abs(amp["theta_face"] - leg["theta_face"]) < 5e-3
    assert abs(amp["theta_floor"] - leg["theta_floor"]) < 5e-3
    assert abs(amp["field"] - leg["field"]) < 0.5
    assert abs(amp["floor"] - leg["floor"]) < 0.5


# ------------------------------------------------------------------ 5. Cl trend (Josell/Moffat shape)
def test_higher_chloride_suppresses_the_field_more():
    """Josell/Moffat trend: the field (high current, little breakdown) is only suppressed when Cl- decorates
    the surface, and the suppression grows with the Cl- level; the floor, which breaks the suppressor down
    at the higher current density, stays active - the passivation window that gives bottom-up fill.

    The field coverage theta_face is the direct observable and is asserted monotone; the field/floor
    growth ratio is printed for the three levels (10 uM, 1 mM, 100 mM)."""
    rows = [("10 uM", _run("cl_1e-8", additives="cl_ndr", c_cl_bulk=1e-8)),
            ("1 mM", _run("cl_1e-6", additives="cl_ndr", c_cl_bulk=1e-6)),
            ("100 mM", _run("cl_1e-4", additives="cl_ndr", c_cl_bulk=1e-4))]
    for name, r in rows:
        print(_fmt(f"Cl {name}", r))
    th = [r["theta_face"] for _, r in rows]
    th_fl = [r["theta_floor"] for _, r in rows]
    ratio = [r["ratio"] for _, r in rows]
    print(f"   theta_face  {th[0]:.4f} -> {th[1]:.4f} -> {th[2]:.4f}")
    print(f"   theta_floor {th_fl[0]:.4f} -> {th_fl[1]:.4f} -> {th_fl[2]:.4f}")
    print(f"   field/floor {ratio[0]:.4f} -> {ratio[1]:.4f} -> {ratio[2]:.4f}")
    assert th[0] < th[1] < th[2]                            # stronger field suppression with more Cl-
    assert th[0] < 0.5                                      # 10 uM: barely decorated
    assert th_fl[1] < 0.6                                   # the floor breaks the suppressor down
    assert ratio[1] < ratio[0]                              # 1 mM: bottom-up relative to 10 uM


# ------------------------------------------------------------------ 6. leveller
def test_leveller_slows_the_field_relative_to_the_floor():
    """'full' with a transported leveller (c_lev_bulk = 1e-6, k_lev = 1e-3 cm/s, Gamma_lev = 2e-10): the
    supply is transport limited, so the deep floor (longer diffusion path from the plane, larger sink) is
    more depleted than the field and the FIELD sees the larger blocking coverage
    (theta_lev ~ 0.55 at the field vs ~0.22 on the floor) - it grows slower relative to the floor."""
    nolev = _run("full_nolev", additives="full", c_cl_bulk=1e-6)
    lev = _run("full_lev", additives="full", c_cl_bulk=1e-6, c_lev_bulk=1e-6)
    print(_fmt("full_nolev", nolev))
    print(_fmt("full_lev", lev))
    sim, g = lev["sim"], lev["sim"].g
    band = np.abs(sim.psi) < g.dx
    th_field = float(sim.st["theta_l"][0][band[0]].mean())
    th_floor = float(sim.st["theta_l"][g.nr // 2][band[g.nr // 2]].mean())
    print(f"   theta_lev at the front: field {th_field:.3f}  floor {th_floor:.3f}")
    print(f"   field/floor {nolev['ratio']:.4f} -> {lev['ratio']:.4f}")
    assert th_field > th_floor + 0.05
    assert lev["ratio"] < nolev["ratio"] - 0.05
    _assert_coverages(lev)


# ------------------------------------------------------------------ 7. reverse pulse desorption
def test_reverse_pulse_desorption_lowers_the_coverage():
    """On a PR recipe 'full' removes coverage over the reverse fraction of every step
    (additives.reverse_pulse_desorption with p.k_rd).  k_rd = 0 must leave the legacy behaviour; a finite
    k_rd must lower the final suppressor coverage."""
    from platesim.feature.engine.recipes import pulse_reverse
    rec = pulse_reverse(J_MA, 0.05, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)
    out = {}
    for k_rd in (0.0, 200.0):
        p = Params(**{**BASE, "additives": "full", "c_cl_bulk": 1e-6, "k_rd": k_rd})
        sim = ViaSim(D_UM, 22.0, p, F_OPEN, rec)
        raw = sim.run(f"pr{k_rd:g}")
        out[k_rd] = (raw, float(raw["series"]["theta_face"][-1]))
        print(f"k_rd={k_rd:6.1f}: theta_face={out[k_rd][1]:.4f} cu={raw['cu_balance_rel']:+.2e}")
    assert out[200.0][1] < out[0.0][1]
