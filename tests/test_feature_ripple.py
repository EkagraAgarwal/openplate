"""Ripple regression for the `trench_full` advection operator (plan.md Task 7A).

Background: `face_fluxes` derives face fluxes from the LBM cell-centred velocities (arithmetic face mean,
masked by Hf and psi <= 0, truncated at the Dirichlet top plane where the LBM flow does have a vertical
component).  Those fluxes are NOT discretely divergence free, so the legacy conservative operator
+div(Hs u c) (Params.adv_form='divergence') injects a spurious source -c div(Hs u) into every cell and
ripples the field (4.5 h narrow site: 74-111 um with bumps on the trench shoulders).
Params.adv_form='gradient' (the default) assembles the row-sum-zero upwind form  sum_faces F (c_f - c_P),
which preserves a uniform field exactly (platesim/trenchflow/transport.py uses the same trick).

a. operator level: uniform-field preservation under a deliberately non-solenoidal velocity;
b. run level: short narrow-site trench_full runs, gradient (smooth) vs divergence (rippled);
c. metric level: with no flow the whole-pitch field/floor metrics reproduce the legacy single-column ones.
"""
from __future__ import annotations

import numpy as np

from platesim.feature.calibrate import load_sites
from platesim.feature.engine import Params, pulse_reverse, run_case
from platesim.feature.engine.fields import StepGeom
from platesim.feature.engine.levelset import Grid, initial_psi

# narrow Trench 1 cell: d = 227 um, H = 88 um, f_open = 0.2, dx 8 um, delta 64 um (see tmp/ripple_probe.py)
D_UM = 227.0
F_OPEN = 0.2
BASE = dict(geometry="trench_full", H_um=88.0, suppressor=False, ohmic=True, dx_um=8.0, delta_um=64.0,
            k_mix=0.0, max_overburden_um=140.0)


# ------------------------------------------------------------------ a. operator: uniform field preserved
def test_gradient_advection_preserves_uniform_field_for_non_solenoidal_flow():
    """Flat wall (f_open=1), ux = 1e-2 (1 + 0.5 sin(2 pi x / pitch)) cm/s, uz = 0: the x-flux varies with x,
    so the discrete divergence is non-zero and the legacy form turns it into a spurious cell source."""
    g = Grid(D_UM, 88.0, 1.0, 64.0, dx_um=8.0, geometry="trench_full")
    psi = initial_psi(g)                                   # flat: the wall is the field plane, no trench
    geo = StepGeom(g, psi, 20.0 * g.dx, Params(geometry="trench_full"))
    n = geo.n
    assert n > 0 and geo.jw.size > 0
    ux = 1e-2 * (1.0 + 0.5 * np.sin(2.0 * np.pi * g.rc / g.R)) * np.ones((1, g.nz))
    uz = np.zeros((g.nr, g.nz))
    F_r, F_z, F_w = geo.face_fluxes(ux, uz)

    A_grad = geo.adv_matrix(F_r, F_z, F_w, form="gradient")
    A_div = geo.adv_matrix(F_r, F_z, F_w, form="divergence")
    ones = np.ones(n)
    row_scale = np.abs(A_grad).sum(axis=1).ravel()
    assert np.all(np.abs(A_grad @ ones) <= 1e-12 * np.maximum(row_scale, 1e-300))     # rows sum to zero
    # the legacy form does NOT: its row sums are the discrete divergence of the same face fluxes
    div = geo.flux_divergence(F_r, F_z, F_w)
    np.testing.assert_allclose(A_div @ ones, div, atol=1e-18)
    assert np.abs(div).max() > 5e-3 * np.abs(F_r).max(), (np.abs(div).max(), np.abs(F_r).max())
    # ... and the difference of the two forms is exactly that spurious -c div(Hs u) diagonal
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    np.testing.assert_allclose((A_div - A_grad) @ x, div * x,
                               atol=1e-15 * np.abs(div).max() * np.abs(x).max())


# ------------------------------------------------------------------ b. run level: ripple vs smooth
def _narrow_run(form, hours):
    p = Params(adv_form=form, **BASE)
    return run_case(D_UM, 22.0, pulse_reverse(15.0, hours, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0),
                    p, f_open=F_OPEN, label=f"ripple_{form}")


def test_gradient_removes_the_trench_full_field_ripple():
    """2 h at 15 mA/cm2 (pulse reverse, additive-free, ohmic): the row-sum-zero operator leaves the field
    statistically flat, the legacy conservative form ripples it.

    1 h is too short for the 3x factor (gradient std 0.060 um vs divergence 0.066 um: the spurious mode
    grows superlinearly and only dominates once the deposit is thick), hence 2 h here.  Measured at 2 h:
    gradient std 0.121 um / (max-min) 0.43 um, divergence std 0.856 um / (max-min) 4.3 um.
    """
    res_g, res_d = _narrow_run("gradient", 2.0), _narrow_run("divergence", 2.0)
    sg, sd = res_g["series"], res_d["series"]
    std_g, std_d = sg["field_std_um"][-1], sd["field_std_um"][-1]
    rng_g = sg["field_max_um"][-1] - sg["field_min_um"][-1]
    rng_d = sd["field_max_um"][-1] - sd["field_min_um"][-1]
    # the gradient form keeps the far field flat
    assert std_g < 0.5 and rng_g < 1.5, (std_g, rng_g, sg["field_mean_um"][-1])
    # the divergence form reproduces the documented bug: at least 3x the ripple
    assert std_d >= 3.0 * std_g, (std_g, std_d, rng_g, rng_d)


# ------------------------------------------------------------------ c. no flow: new metrics == legacy numbers
def test_zero_shear_metrics_match_the_legacy_single_column_values():
    """flow_shear_s = 0 (0.3 h): the whole-pitch field mean must equal the legacy single-column
    face_thickness_um and the central-40 % floor mean the legacy floor_height_um."""
    p = Params(adv_form="gradient", flow_shear_s=0.0, **BASE)
    res = run_case(D_UM, 22.0, pulse_reverse(15.0, 0.3, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0),
                   p, f_open=F_OPEN, label="ripple_noflow")
    s = res["series"]
    for key in ("field_mean_um", "field_min_um", "field_max_um", "field_std_um",
                "floor_mean_um", "floor_min_um", "floor_max_um", "plane_gap_min_um"):
        assert key in s and s[key][-1] is not None, key
    assert abs(s["field_mean_um"][-1] - s["face_thickness_um"][-1]) < 0.1, (
        s["field_mean_um"][-1], s["face_thickness_um"][-1])
    assert abs(s["floor_mean_um"][-1] - s["floor_height_um"][-1]) < 0.3, (
        s["floor_mean_um"][-1], s["floor_height_um"][-1])
    # no flow -> no ripple, and the metrics stay finite over the whole series
    assert s["field_std_um"][-1] < 0.05, s["field_std_um"][-1]
    for key in ("field_std_um", "plane_gap_min_um", "floor_min_um"):
        assert np.all(np.isfinite(np.asarray(s[key], float))), key
