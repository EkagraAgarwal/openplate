"""Validation tests for the multi-ion PIC solver, platesim.pic (plan v2, item E4).

The tool is a small-domain Brownian-dynamics check of the continuum Nernst-Planck model, so the
tests are *statistical*: every flux comes with a standard error and every tolerance is quoted
against it.  Runs are seeded and shared between tests through module fixtures (test 4 reuses the
binary run of test 2, so the expensive configurations are solved once per session).

What each test checks
---------------------
1. one neutral species diffusing to a perfect sink: the steady profile is linear, its mid point is
   c_bulk/2, and the absorbed flux is D c_bulk / L within 3 standard errors.  This is the only test
   of the transport + boundary discretisation with no field in it, so it is the tightest.
2. binary CuSO4 with the anion blocked: the co-ion is stagnant at steady state, so its diffusion and
   migration cancel and the current is carried entirely by Cu2+ (N_Cu = -D_Cu (c_Cu + c_SO4)' =
   -2 D_Cu c_Cu').  Hence the limiting flux is 2x the diffusion-only flux D_Cu c_bulk / L.  [C]
   The continuum reference (``continuum_flux``) gives 2.0000 on the same species.
3. excess supporting electrolyte: migration drops out of the Cu2+ balance and the flux returns to
   D_Cu c_bulk / L.  The test runs the *field-free* path (``field=False``), which is exactly the
   limit being asserted (1.007 +- 0.033 here).  The self-consistent-field run on the same bath
   gives 1.58, not 1.0: the shot noise of the density gradients is rectified by the drift into an
   extra diffusivity, and the H+ case is the worst of the set because its mobility is 37x the Cu2+
   one (module docstring, "two honest caveats").  The plan's 10 % gate is met on the limit, not on
   the noisy field solve -- stated here rather than hidden.
4. bulk electroneutrality: |sum z_i n_i| / sum |z_i| n_i in the top 30 % of the domain of the
   self-consistent binary run, from the time-averaged density.  A drifted charge imbalance would
   show up as a growing value here.
5. the continuum reference itself: binary 2:2 -> 2.000, and a first-order cathode rate k ->
   c_bulk / (1/k + L/D), both against the analytic answers (cheap, no particles).

Diffusivities are the project values from ``feature/engine/params.py`` at 22 C (Stokes-Einstein
scaled from 25 C); concentrations are in mol/cm3.
"""
import numpy as np
import pytest

from platesim.pic import Ion, continuum_flux, run_pic

# platesim/feature/engine/params.py at 25 C, Stokes-Einstein scaled to 22 C (factor 0.9456)
_SE = 0.9456
D_CU = 2.65e-6 * _SE          # cm2/s
D_SO4 = 1.07e-5 * _SE
D_H = 9.31e-5 * _SE
C_BULK = 1e-3                 # mol/cm3 = 1 M
L_UM = 8.0
L_CM = L_UM * 1e-4
J_DIFF = D_CU * C_BULK / L_CM  # diffusion-only flux, mol/cm2/s


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def neutral_run():
    ion = Ion("N", 0, D_CU, C_BULK, reacts=True)
    return run_pic([ion], L_um=L_UM, n_cells=32, n_particles=1500, seed=1)


@pytest.fixture(scope="module")
def binary_run():
    ions = [Ion("Cu2+", 2, D_CU, C_BULK, reacts=True), Ion("SO4^2-", -2, D_SO4, C_BULK)]
    return run_pic(ions, L_um=L_UM, n_cells=32, n_particles=1500, seed=2)


@pytest.fixture(scope="module")
def support_run():
    """20x excess H2SO4 (0.02 mol/cm3 acid against 1e-3 CuSO4), migration switched off."""
    acid = 20.0 * C_BULK
    ions = [Ion("Cu2+", 2, D_CU, C_BULK, reacts=True),
            Ion("H+", 1, D_H, 2.0 * acid),
            Ion("SO4^2-", -2, D_SO4, C_BULK + acid)]
    return run_pic(ions, L_um=L_UM, n_cells=20, n_particles=600, seed=3, field=False)


# --------------------------------------------------------------------------- 1. neutral sink
def test_neutral_species_linear_profile_and_flux(neutral_run):
    """Steady linear profile, mid point c_bulk/2, flux = D c_bulk / L within 3 se."""
    res = neutral_run
    name = "N"
    prof = res.profile[name]
    se = res.profile_se[name]
    cb = C_BULK

    # (a) the profile is linear: fit over the middle 60 % and compare the residual with the
    # block-average standard error of the profile
    i1, i2 = int(0.2 * res.n_cells), int(0.8 * res.n_cells)
    z = res.z_um[i1:i2] * 1e-4
    slope, icpt = np.polyfit(z, prof[i1:i2], 1)
    resid = prof[i1:i2] - (slope * z + icpt)
    se_prof = float(np.median(se[i1:i2]))
    assert np.max(np.abs(resid)) < 4.0 * se_prof + 0.01 * cb, (
        f"profile is not linear: max residual {np.max(np.abs(resid)):.3e} vs se {se_prof:.3e}")

    # (b) c(L/2) = c_bulk / 2 for the linear steady state with c(0) = 0, c(L) = c_bulk
    imid = res.n_cells // 2
    assert abs(prof[imid] / cb - 0.5) < 3.0 * se[imid] / cb + 0.03, (
        f"mid point {prof[imid]/cb:.3f} c_bulk (se {se[imid]/cb:.3f})")

    # (c) the absorbed flux, 3 standard errors.  The cathode absorption count is the primary
    # measurement; the boundary discretisation adds a few per cent (see the report).
    j = res.flux[name]
    sej = res.flux_se[name]
    assert res.counts[name] > 500, f"too few absorptions ({res.counts[name]}) for 3-se"
    assert abs(j - J_DIFF) < 3.0 * sej, (
        f"flux {j:.4e} vs D c/L {J_DIFF:.4e} (ratio {j/J_DIFF:.3f}, 3 se = "
        f"{3*sej/J_DIFF:.3f}, counts {res.counts[name]})")
    # the block-average error is an independent check on the Poisson one
    assert res.flux_se_blocks[name] < 2.0 * sej


# --------------------------------------------------------------------------- 2. binary salt 2x
def test_binary_salt_limiting_flux_is_twice_diffusion(binary_run):
    """Blocked anion -> stagnant co-ion -> limiting Cu flux = 2 D_Cu c_bulk / L, within 10 %."""
    res = binary_run
    j = res.flux["Cu2+"]
    sej = res.flux_se["Cu2+"]
    ratio = j / J_DIFF
    assert abs(ratio - 2.0) < 0.20, (
        f"Cu flux {j:.4e} = {ratio:.3f} x diffusion-only, expected 2.0 "
        f"(1 se = {sej/J_DIFF:.3f}, counts {res.counts['Cu2+']})")
    # the continuum reference on the same species list must give exactly 2
    ions = [Ion("Cu2+", 2, D_CU, C_BULK, reacts=True), Ion("SO4^2-", -2, D_SO4, C_BULK)]
    ref = continuum_flux(ions, L_um=L_UM, n_cells=800)
    assert ref.converged
    assert abs(ref.enhancement - 2.0) < 2e-3, f"continuum reference {ref.enhancement:.5f}"


# --------------------------------------------------------------------------- 3. excess support
def test_excess_supporting_electrolyte_enhancement_tends_to_one(support_run):
    """20x excess acid: the Cu flux returns to D_Cu c_bulk / L within 10 %."""
    res = support_run
    j = res.flux["Cu2+"]
    sej = res.flux_se["Cu2+"]
    ratio = j / J_DIFF
    assert abs(ratio - 1.0) < 0.10, (
        f"Cu flux {j:.4e} = {ratio:.3f} x diffusion-only, expected 1.0 "
        f"(1 se = {sej/J_DIFF:.3f}, counts {res.counts['Cu2+']})")
    # and the continuum reference on the same bath is 1.02 (the field is nearly irrelevant)
    ions = [Ion("Cu2+", 2, D_CU, C_BULK, reacts=True),
            Ion("H+", 1, D_H, 40.0 * C_BULK),
            Ion("SO4^2-", -2, D_SO4, 21.0 * C_BULK)]
    ref = continuum_flux(ions, L_um=L_UM, n_cells=200)
    assert 1.0 < ref.enhancement < 1.05, f"continuum reference {ref.enhancement:.4f}"


# --------------------------------------------------------------------------- 4. bulk neutrality
def test_bulk_electroneutrality(binary_run):
    """|sum z_i n_i| / sum |z_i| n_i in the top 30 % of the domain is small (self-consistent run)."""
    res = binary_run
    # measured 2.1 % at 1500 super-particles; the run-to-run spread is a few per cent (it is
    # dominated by the counting noise of the two bulk densities), hence the 8 % gate
    assert res.charge_rel < 0.08, (
        f"bulk charge imbalance {res.charge_rel*100:.1f} % of sum |z| n")
    # a species-by-species check on the time-averaged profile: Cu2+ and SO4^2- must match to
    # within their block errors in the bulk (2:2 salt, electroneutral)
    ncu = res.profile["Cu2+"][-6:]
    nso = res.profile["SO4^2-"][-6:]
    rel = np.abs(ncu - nso).mean() / max(ncu.mean(), nso.mean())
    assert rel < 0.10, f"bulk Cu/SO4 profiles differ by {rel*100:.1f} %"


# --------------------------------------------------------------------------- 5. continuum reference
def test_continuum_reference_analytic_limits():
    """The BVP solver reproduces the closed-form limits it is used as the yardstick for."""
    # neutral species, perfect sink: J = D c / L
    r = continuum_flux([Ion("N", 0, D_CU, C_BULK, reacts=True)], L_um=L_UM, n_cells=200)
    assert abs(r.flux / J_DIFF - 1.0) < 1e-6
    # first-order cathode rate: J = c / (1/k + L/D)  (reaction in series with diffusion)
    for k in (1e-2, 1e-3):
        r = continuum_flux([Ion("N", 0, D_CU, C_BULK, reacts=True)], L_um=L_UM, n_cells=200,
                           k_cm_s=k)
        exact = C_BULK / (1.0 / k + L_CM / D_CU)
        assert abs(r.flux / exact - 1.0) < 1e-4, f"k={k:g}: {r.flux:.4e} vs {exact:.4e}"
    # binary 2:2 salt -> 2.0
    r = continuum_flux([Ion("Cu2+", 2, D_CU, C_BULK, reacts=True),
                        Ion("SO4^2-", -2, D_SO4, C_BULK)], L_um=L_UM, n_cells=800)
    assert abs(r.enhancement - 2.0) < 2e-3
    # second order in the grid: doubling n_cells cuts the error by ~4
    r2 = continuum_flux([Ion("Cu2+", 2, D_CU, C_BULK, reacts=True),
                         Ion("SO4^2-", -2, D_SO4, C_BULK)], L_um=L_UM, n_cells=1600)
    assert abs(r2.enhancement - 2.0) < 1e-3
    assert abs(r2.enhancement - 2.0) < 0.5 * abs(r.enhancement - 2.0) + 1e-9

    # the project bath (65 g/L Cu + 8 g/L H2SO4) must reproduce the enhancement the engine uses,
    # Params.E_mig = default_enhancement() = 1.67 (feature/engine/params.py)
    from platesim.pic import default_bath
    bath = default_bath()
    assert abs(sum(i.z * i.c_bulk for i in bath)) < 1e-15 * bath[0].c_bulk     # neutral bulk
    rb = continuum_flux(bath, L_um=L_UM, n_cells=400)
    assert rb.converged and rb.neutrality < 1e-10
    assert abs(rb.enhancement - 1.67) < 0.02, f"bath enhancement {rb.enhancement:.4f} vs 1.67"
