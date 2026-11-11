"""Validation of the 1+1-D SOS lattice kMC (plan item E3, part a).

Checks (runtime budget < 60 s):
* pure random deposition reproduces the Poisson-column law W^2 = <h> (5 %);
* mass conservation: sum(h) == number of deposition events, event bookkeeping;
* deposition off + rough start: height conserved, W decays monotonically in the mean;
* growth exponent beta = 0.25 +- 0.08 (Edwards-Wilkinson, 1+1 D) for the
  ``model="family"`` deposition rule; the Arrhenius variant is printed alongside;
* determinism for a fixed seed;
* the suppressor mask (theta_block) blocks deposition on the masked columns;
* the rate conversions F = j/(2 e N_s) and the bond-counted Arrhenius rate.

The Family rule is used for the exponent test because the default Arrhenius
parameters (E_d = 0.40 eV at 300 K -> 1.9e6 hops/s against F = 31 ML/s) put the
physical model deep in the relaxation-limited regime: no growth regime is
reachable in a test-sized run, so its beta is reported, not asserted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from platesim.kmc import (
    deposition_rate, deposition_rate_mA, hop_rate, interface_width, run_sos,
)

F15 = deposition_rate_mA(15.0)          # 31.2075 /s at N_s = 1.5e15 cm^-2 [J]
TARGETS_EW = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def _width_at(runs, targets):
    """W(t) averaged as <W^2> over runs, evaluated at the target mean heights."""
    w2 = np.array([np.interp(targets, r["h_mean"][1:], r["W"][1:] ** 2) for r in runs])
    return np.sqrt(w2.mean(axis=0))


def _fit_beta(runs, targets):
    """Least-squares slope of log W vs log <h> (W^2 averaged over the runs)."""
    w = _width_at(runs, targets)
    slope = np.polyfit(np.log(np.asarray(targets, float)), np.log(w), 1)[0]
    return float(slope), w


def test_units_and_rates():
    """F = j / (2 e N_s) and the bond-counted Arrhenius rate."""
    assert F15 == pytest.approx(31.2075, rel=1e-5)
    assert deposition_rate(0.015) == pytest.approx(F15)
    assert deposition_rate_mA(30.0) == pytest.approx(2.0 * F15)
    with pytest.raises(ValueError):
        deposition_rate(-1.0)
    # more bonds -> slower hop, monotone in n
    rates = [hop_rate(n, 300.0) for n in (0, 1, 2)]
    assert rates[0] > rates[1] > rates[2] > 0.0
    assert rates[0] == pytest.approx(1.9068e6, rel=1e-3)
    assert interface_width([0, 0, 0, 0]) == 0.0


def test_random_deposition_poisson_width():
    """Diffusion off: independent Poisson columns, so W^2 / <h> -> 1."""
    w2, hm, hops = [], [], 0
    for seed in range(6):
        res = run_sos(L=512, F=F15, E_d_eV=10.0, coverage_ML=50.0, seed=seed)
        w2.append(res["W_final"] ** 2)
        hm.append(res["h_final"])
        hops += res["events"]["hop"]
    assert hops == 0, "E_d = 10 eV must freeze surface diffusion"
    ratio = float(np.mean(w2) / np.mean(hm))
    print(f"\nrandom deposition: <h> = {np.mean(hm):.2f} ML, W^2/<h> = {ratio:.4f}")
    assert 0.95 <= ratio <= 1.05, ratio


def test_mass_conservation_and_event_bookkeeping():
    """sum(h) counts exactly the deposition events; hop counts add up."""
    res = run_sos(L=128, F=F15, T_K=400.0, coverage_ML=10.0, seed=1,
                  max_events=300_000)
    assert res["heights"].sum() == res["depositions"]
    assert res["depositions"] == res["events"]["deposition"]
    assert res["n_events"] == res["events"]["deposition"] + res["events"]["hop"]
    assert res["hop_by_class"].sum() == res["events"]["hop"]
    assert res["events"]["hop"] > 0
    assert res["h_final"] == pytest.approx(res["heights"].mean())


def test_diffusion_smooths_rough_surface():
    """Deposition off: height is conserved and W decays monotonically in the mean."""
    rng = np.random.default_rng(11)
    curves = []
    for seed in range(5):
        h0 = rng.integers(0, 9, 256)
        res = run_sos(L=256, F=0.0, T_K=300.0, coverage_ML=None, t_max_s=4.0e-6,
                      snapshots=6, seed=seed, heights0=h0)
        assert res["heights"].sum() == h0.sum(), "diffusion must conserve mass"
        assert res["events"]["deposition"] == 0
        curves.append(res["W"])
    w = np.mean(curves, axis=0)
    print(f"\nsmoothing (5 seeds): W = {np.round(w, 3)}")
    assert np.all(np.diff(w) < 0.0), w
    assert w[-1] < 0.9 * w[0]


def test_growth_exponent_family_ew():
    """Family rule: W ~ <h>^beta with beta = 0.25 (EW, 1+1 D) in the growth regime."""
    runs = [run_sos(L=256, F=F15, T_K=300.0, coverage_ML=32.0, seed=s,
                    snapshots=TARGETS_EW, model="family") for s in range(12)]
    beta, w = _fit_beta(runs, TARGETS_EW)
    print(f"\nfamily: beta = {beta:.3f} (target 0.25), W = {np.round(w, 3)}")
    assert abs(beta - 0.25) <= 0.08, beta


def test_determinism():
    """Same seed -> bit-identical trajectory; a different seed differs."""
    kw: dict[str, Any] = dict(L=64, F=F15, T_K=350.0, coverage_ML=3.0, snapshots=4,
                              seed=7)
    a = run_sos(**kw)
    b = run_sos(**kw)
    for key in ("t", "W", "h_mean", "heights"):
        assert np.array_equal(a[key], b[key]), key
    assert a["n_events"] == b["n_events"]
    c = run_sos(**{**kw, "seed": 8})
    assert not np.array_equal(a["heights"], c["heights"])


def test_suppressor_mask_blocks_deposition():
    """theta_block: masked columns receive no direct deposit (diffusion off)."""
    res = run_sos(L=256, F=F15, E_d_eV=10.0, coverage_ML=1.0, theta_block=0.5,
                  seed=2, max_events=200_000)
    blocked = res["blocked"]
    assert 0.35 < blocked.mean() < 0.65, blocked.mean()
    assert np.all(res["heights"][blocked] == 0)
    n_free = int((~blocked).sum())
    assert res["depositions"] == pytest.approx(F15 * n_free * res["time_s"], rel=0.02)
    assert res["h_final"] == pytest.approx(F15 * (n_free / 256) * res["time_s"], rel=0.02)
    print(f"\ntheta_block: {blocked.mean():.2f} of columns masked, <h> = "
          f"{res['h_final']:.3f} ML at t = {res['time_s']:.4f} s")


def test_default_parameters_report():
    """Print W(t) and beta at 300 K with F from 15 mA/cm2 (default parameters)."""
    print(f"\ndefault: F = {F15:.4f} /s (15 mA/cm2), T = 300 K, E_d = 0.40 eV, "
          f"E_b = 0.20 eV, nu0 = 1e13 /s")
    print("hop rates /s: " + ", ".join(f"n={n}: {hop_rate(n, 300.0):.3e}"
                                       for n in (0, 1, 2)))

    ts = [1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
    res = run_sos(L=256, F=F15, T_K=300.0, coverage_ML=50.0, snapshots=ts, seed=0)
    print(f"arrhenius 300 K, 50 ML ({res['n_events']} events, "
          f"{res['events']['hop'] / max(res['events']['deposition'], 1):.0f} hops per "
          f"deposited atom):")
    for t_, w_, hm_ in zip(res["t"][1:], res["W"][1:], res["h_mean"][1:]):
        print(f"  t = {t_:9.4f} s   <h> = {hm_:7.2f} ML   W = {w_:6.3f} ML")
    print(f"  events: deposition {res['events']['deposition']}, hop "
          f"{res['events']['hop']}, hop by class {res['hop_by_class'].tolist()}")
    beta_def = np.polyfit(np.log(res["h_mean"][3:]), np.log(res["W"][3:]), 1)[0]
    print(f"  beta (10-50 ML) = {beta_def:.3f} "
          f"(slow roughening from a near-equilibrium start; family below is the "
          f"assigned EW check)")

    runs = [run_sos(L=256, F=F15, T_K=300.0, coverage_ML=32.0, seed=s,
                    snapshots=TARGETS_EW, model="family") for s in range(12)]
    beta_f, w_f = _fit_beta(runs, TARGETS_EW)
    print(f"family rule, 300 K (EW reference): beta = {beta_f:.3f}, "
          f"W = {np.round(w_f, 3)}")

    kw: dict[str, Any] = dict(T_K=300.0, E_d_eV=0.60, E_b_eV=0.05)  # growth-regime setting [J]
    runs = [run_sos(L=256, F=F15, coverage_ML=32.0, seed=s, snapshots=TARGETS_EW,
                    **kw) for s in range(8)]
    beta_a, w_a = _fit_beta(runs, TARGETS_EW)
    print(f"arrhenius, E_d = 0.60 eV, E_b = 0.05 eV: beta = {beta_a:.3f}, "
          f"W = {np.round(w_a, 3)}")

    assert abs(beta_f - 0.25) <= 0.08
    assert res["W_final"] < 1.5, "300 K default must stay smooth (sub-2 ML width)"

