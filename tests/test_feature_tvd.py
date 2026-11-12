"""Second-order TVD (MUSCL / van Leer) advection on the Stokes MAC path (plan v2 C1b).

`adv_scheme='tvd'` keeps the existing first-order upwind operator as the implicit linear part and adds
the difference between the limited (MUSCL / van Leer) and the upwind face fluxes to the rhs of every
Picard sweep (deferred correction).  This module checks
  a. the reconstruction helper itself: van Leer limiter properties, second-order accuracy on a smooth
     profile (error ratio ~4 under h -> h/2), TVD (no new extrema) across a step;
  b. the deferred correction of the geometry: zero for a uniform field (exactly) and conservative
     (sum over all active cells = 0 to 1e-12) for a random field;
  c. `adv_scheme='upwind'` is bit-identical to the default Stokes path (so the default behaviour is
     unchanged) and 'tvd' really produces a different solution;
  d. grid convergence (the goal): narrow Trench 1 (d = 227 um, H = 88 um, f_open = 0.2), 1 h at
     15 mA/cm2, G = 59 1/s.  The floor/field ratio of the final frame is compared at dx 8, 4, 2 um for
     upwind / sg / tvd; the tvd dx8->dx4 and dx4->dx2 gaps must be clearly smaller than upwind's and
     the dx4 grid error (|ratio(dx4) - ratio(dx2)|) below the 0.005 plan gate, with |cu_balance_rel|
     < 0.005 for every scheme.  Marked slow (9 full runs).

The site data file is not part of this public repo; the geometry is built explicitly in the tests.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from platesim.feature.engine.fields import StepGeom, muscl_face, vanleer
from platesim.feature.engine.levelset import UM
from platesim.feature.engine.params import Params
from platesim.feature.engine.recipes import pulse_reverse
from platesim.feature.engine.run_case import run_case
from platesim.feature.engine.sim import ViaSim
from platesim.feature.ripple_study import surface_stats

D_UM, H_UM, F_OPEN, DELTA_UM, J_NET = 227.0, 88.0, 0.2, 64.0, 15.0

BASE = dict(geometry="trench_full", suppressor=False, ohmic=True, delta_um=DELTA_UM,
            k_mix=0.0, max_overburden_um=140.0, H_um=H_UM, flow_solver="stokes",
            flow_shear_s=59.0)


def _params(**kw):
    args = dict(BASE)
    args.update(kw)
    return Params(**args)


def _recipe(hours):
    return pulse_reverse(J_NET, hours, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)


_SETUP: dict = {}


def _step0(dx=8.0):
    """(sim, StepGeom, (ufx, ufz), (F_r, F_z, F_w)) of the initial geometry after one Stokes solve."""
    if dx not in _SETUP:
        p = _params(dx_um=dx)
        sim = ViaSim(D_UM, 22.0, p, F_OPEN, _recipe(0.1))
        sim.cavity_flow(p.flow_shear_s)
        geo = StepGeom(sim.g, sim.psi, sim.face_height() + p.delta_um * UM, p)
        F_r, F_z, F_w, _ = geo.mac_face_fluxes(*sim.flow_faces)
        _SETUP[dx] = (sim, geo, sim.flow_faces, (F_r, F_z, F_w))
    return _SETUP[dx]


# ------------------------------------------------------------------ a. the reconstruction helper
def test_vanleer_limiter_properties():
    """psi(0) = 0, psi(1) = 1 (full second order on a smooth field), psi(2) = 4/3, psi <= 2 and
    psi = 0 on every anti-diffusive face (r <= 0), so the reconstruction is TVD."""
    assert vanleer(0.0) == 0.0
    assert vanleer(1.0) == pytest.approx(1.0)
    assert vanleer(2.0) == pytest.approx(4.0 / 3.0)
    r = np.linspace(-5.0, 50.0, 501)
    psi = vanleer(r)
    assert psi.max() <= 2.0
    assert np.all(psi[r <= 0.0] == 0.0)
    assert np.all(np.diff(psi[r > 0.0]) >= 0.0)             # monotone in the admissible region


def _smooth_error(m):
    """Max reconstruction error of exp(x) at the interior cell faces of a uniform grid with m cells."""
    x = np.linspace(0.0, 1.0, m + 1)
    c = np.exp(x)
    ho = muscl_face(c[1:-1], c[2:], c[:-2])                 # upwind i, downwind i+1, far-upwind i-1
    exact = np.exp(0.5 * (x[1:-1] + x[2:]))                 # the value at the face midpoint
    return float(np.abs(ho - exact).max())


def test_muscl_face_is_second_order_on_a_smooth_profile():
    """On a smooth monotone field r ~ 1 (psi ~ 1) and the face value is centred: the error drops by
    ~4 when h is halved (second order), unlike the first-order upwind value (ratio ~2)."""
    e_coarse, e_fine = _smooth_error(200), _smooth_error(400)
    ratio = e_coarse / e_fine
    assert 3.0 < ratio < 5.0, (e_coarse, e_fine, ratio)
    assert e_fine < 1e-5, e_fine
    # the same profile with the upwind (first-order) face value: ratio ~2, so 'tvd' is genuinely
    # second order and not just a rescaled upwind reconstruction
    def upwind_error(m):
        x = np.linspace(0.0, 1.0, m + 1)
        c = np.exp(x)
        exact = np.exp(0.5 * (x[1:-1] + x[2:]))
        return float(np.abs(c[1:-1] - exact).max())
    assert upwind_error(200) / upwind_error(400) < 2.5


def test_muscl_face_is_tvd_across_a_step():
    """A step 0 -> 1: the limited face values never leave [0, 1] (no new extrema) and stay monotone."""
    c = np.r_[np.zeros(50), np.ones(50)]
    ho = muscl_face(c[1:-1], c[2:], c[:-2])
    assert ho.min() >= 0.0 and ho.max() <= 1.0
    assert np.all(np.diff(ho) >= -1e-15)


# ------------------------------------------------------------------ b. the deferred correction
def test_uniform_field_correction_is_exactly_zero():
    """c = const -> the limited face value equals the upwind one on every face, so the deferred
    correction vanishes exactly (a uniform field therefore stays uniform, as with the upwind scheme)."""
    _, geo, _, faces = _step0()
    corr = geo.tvd_correction(np.ones(geo.n), *faces)
    np.testing.assert_array_equal(corr, np.zeros(geo.n))


def test_face_corrections_are_conservative():
    """Every face enters p and q with opposite signs, so the correction over all active cells sums to 0
    (the MUSCL scheme is conservative; the Cu balance is not altered by the deferred correction)."""
    _, geo, _, faces = _step0()
    c = np.random.default_rng(11).random(geo.n)
    corr = geo.tvd_correction(c, *faces)
    assert abs(float(corr.sum())) < 1e-12, corr.sum()
    # ... and it is not identically zero on a random field: the scheme really is active
    assert np.abs(corr).max() > 1e-12 * max(np.abs(F).max() for F in faces)


# ------------------------------------------------------------------ c. upwind unchanged, tvd wired
def test_upwind_is_bit_identical_to_the_default_and_tvd_differs():
    """adv_scheme='upwind' reproduces the legacy Stokes path bit for bit; 'tvd' produces a different
    series with the same step count and Cu2+ influx (so the switch is wired without altering anything
    else)."""
    rec = _recipe(0.1)
    out = {}
    for kw in ({}, {"adv_scheme": "upwind"}, {"adv_scheme": "tvd"}):
        sim = ViaSim(D_UM, 22.0, _params(**kw), F_OPEN, rec)
        out[kw.get("adv_scheme", "default")] = sim.run("tvd")
    base, expl, tvd = out["default"], out["upwind"], out["tvd"]
    assert set(base["series"]) == set(expl["series"]) == set(tvd["series"])
    for key in base["series"]:
        np.testing.assert_array_equal(np.asarray(base["series"][key], float),
                                      np.asarray(expl["series"][key], float),
                                      err_msg=f"default differs from explicit 'upwind' in {key!r}")
    assert base["steps"] == expl["steps"]
    assert base["cu_influx"] == expl["cu_influx"]
    assert tvd["steps"] == expl["steps"]
    assert any(not np.array_equal(np.asarray(base["series"][k], float),
                                  np.asarray(tvd["series"][k], float)) for k in base["series"]), \
        "'tvd' produced exactly the upwind series: the deferred correction is not wired in"


# ------------------------------------------------------------------ d. grid convergence (the goal)
_GRID: dict = {}


def _run_grid(dx, scheme, hours=1.0):
    key = (float(dx), scheme, float(hours))
    if key not in _GRID:
        p = _params(dx_um=dx, adv_scheme=scheme)
        t0 = time.time()
        _GRID[key] = (run_case(D_UM, 22.0, _recipe(hours), p, f_open=F_OPEN,
                               label=f"tvd_{scheme}_dx{dx:g}", out=None), time.time() - t0)
    return _GRID[key]


@pytest.mark.slow
def test_tvd_reduces_the_grid_error_of_the_floor_to_field_ratio():
    """1 h at 15 mA/cm2, G = 59 1/s, dx = 8, 4, 2 um.  The second-order limited faces remove the O(dx)
    numerical diffusion of the upwind flux, so the floor/field ratio of the final frame converges:
    the dx8->dx4 and dx4->dx2 gaps of 'tvd' are clearly smaller than upwind's and the dx4 grid error is
    below the 0.005 plan gate.  Every scheme keeps the Cu2+ balance closed (|cu_balance_rel| < 0.005).
    (The numbers quoted in the plan use dx 8/4/3/2; here the coarsest triplet 8/4/2 is run.)"""
    r, bal = {}, {}
    for scheme in ("upwind", "sg", "tvd"):
        for dx in (8.0, 4.0, 2.0):
            res, wall = _run_grid(dx, scheme)
            r[(scheme, dx)] = float(surface_stats(res)["ratio_mean"])
            bal[(scheme, dx)] = res["result"]["cu_balance_rel"]
            assert abs(bal[(scheme, dx)]) < 0.005, (scheme, dx, bal[(scheme, dx)])
    gaps = {}
    for scheme in ("upwind", "sg", "tvd"):
        gaps[(scheme, "8-4")] = abs(r[(scheme, 8.0)] - r[(scheme, 4.0)])
        gaps[(scheme, "4-2")] = abs(r[(scheme, 4.0)] - r[(scheme, 2.0)])
    print("\ngrid convergence (floor/field, 1 h):")
    for scheme in ("upwind", "sg", "tvd"):
        print(f"  {scheme:6s} dx8 {r[(scheme, 8.0)]:.5f}  dx4 {r[(scheme, 4.0)]:.5f}  "
              f"dx2 {r[(scheme, 2.0)]:.5f} | gap 8-4 {gaps[(scheme, '8-4')]:.5f}  "
              f"gap 4-2 {gaps[(scheme, '4-2')]:.5f} | cu_bal "
              f"{max(abs(bal[(scheme, dx)]) for dx in (8.0, 4.0, 2.0)):.2e}")
    # tvd is closer to grid independence in both refinement steps
    assert gaps[("tvd", "8-4")] < gaps[("upwind", "8-4")], gaps
    assert gaps[("tvd", "4-2")] < gaps[("upwind", "4-2")], gaps
    # plan gate: dx4 grid error of the ratio < 0.005
    assert gaps[("tvd", "4-2")] < 0.005, gaps[("tvd", "4-2")]
    # second order of the limiter is actually visible: the finest gap shrinks by ~half or more
    assert gaps[("tvd", "4-2")] < 0.75 * gaps[("tvd", "8-4")], gaps
