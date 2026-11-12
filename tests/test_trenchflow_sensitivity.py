"""Tests for platesim.trenchflow.sensitivity (Saltelli/Jansen Sobol indices, F6).

Three checks: the analytic Ishigami benchmark, an additive linear model where S1 == ST == the
squared-coefficient shares, and a tiny plumbing run of the trench model itself.
"""
import numpy as np

from platesim.trenchflow.sensitivity import (BOUNDS, INPUT_NAMES, sobol_design, sobol_indices,
                                             trench_ratio_sensitivity, unit_to_physical)


def ishigami(x, a=7.0, b=0.1):
    """Ishigami function, x_i ~ U(-pi, pi): sin(x1) + a sin^2(x2) + b x3^4 sin(x1)."""
    return float(np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * x[2] ** 4 * np.sin(x[0]))


def ishigami_analytic(a=7.0, b=0.1):
    """Closed-form Sobol indices of the Ishigami function.

    With g(x1, x3) = sin(x1)(1 + b x3^4):  Var(g) = 0.5 + b pi^4/5 + b^2 pi^8/18
    (E[x^4] = pi^4/5, E[x^8] = pi^8/9 over U(-pi, pi)), V1 = Var(E[f|x1]) = 0.5 (1 + b pi^4/5)^2,
    V2 = a^2/8, V3 = 0 and V13 = Var(g) - V1, so V = V1 + V2 + V13.
    """
    pi = np.pi
    var_g = 0.5 + b * pi ** 4 / 5.0 + b ** 2 * pi ** 8 / 18.0
    v1 = 0.5 * (1.0 + b * pi ** 4 / 5.0) ** 2
    v2 = a ** 2 / 8.0
    v13 = var_g - v1
    v = v1 + v2 + v13
    return {"S1": np.array([v1, v2, 0.0]) / v, "ST": np.array([v1 + v13, v2, v13]) / v, "V": v}


def test_ishigami_indices_match_analytic():
    """S1 = (0.314, 0.442, 0.0) and ST = (0.558, 0.442, 0.244) within 0.05 at n_base 2048."""
    pi = np.pi
    ref = ishigami_analytic()
    # sanity: the analytic identities the published numbers come from
    assert abs(ref["V"] - 13.8446) < 1e-3
    assert np.allclose(ref["S1"], (0.3139, 0.4424, 0.0), atol=2e-4)
    assert np.allclose(ref["ST"], (0.5576, 0.4424, 0.2437), atol=2e-4)

    res = sobol_indices(ishigami, [[-pi, pi]] * 3, n_base=2048, seed=0, n_boot=300)
    s1, st = np.asarray(res["S1"]), np.asarray(res["ST"])
    assert np.all(np.abs(s1 - ref["S1"]) <= 0.05), (s1, ref["S1"])
    assert np.all(np.abs(st - ref["ST"]) <= 0.05), (st, ref["ST"])
    # the CIs must bracket their own point estimate
    assert np.all(res["S1_lo"] <= s1 + 1e-12) and np.all(s1 <= res["S1_hi"] + 1e-12)
    assert np.all(res["ST_lo"] <= st + 1e-12) and np.all(st <= res["ST_hi"] + 1e-12)
    assert res["n_eval"] == 2048 * (3 + 2)


def test_linear_additive_s1_equals_st_and_shares():
    """f = c.x over the unit cube: S1 == ST == c_i^2 / sum(c_j^2) within 0.03."""
    c = np.array([1.0, 2.0, 0.5])

    def linear(row):
        return float(np.dot(c, row))

    shares = c ** 2 / float(np.sum(c ** 2))
    res = sobol_indices(linear, [[0.0, 1.0]] * 3, n_base=1024, seed=1, n_boot=300)
    s1, st = np.asarray(res["S1"]), np.asarray(res["ST"])
    assert np.all(np.abs(s1 - shares) <= 0.03), (s1, shares)
    assert np.all(np.abs(st - shares) <= 0.03), (st, shares)
    assert np.all(np.abs(st - s1) <= 0.03), (s1, st)


def test_design_is_not_degenerate_and_log_mapping_is_geometric():
    """A/B must not share columns; log-uniform dims map 0 -> lo, 0.5 -> geometric mean, 1 -> hi."""
    d = len(INPUT_NAMES)
    a, b, ab = sobol_design(256, d, seed=0)
    assert a.shape == (256, d) and b.shape == (256, d) and ab.shape == (d, 256, d)
    for i in range(d):
        assert abs(np.corrcoef(a[:, i], b[:, i])[0, 1]) < 0.2, i          # independent halves
        assert np.allclose(ab[i][:, i], b[:, i])                          # swapped column
        keep = [j for j in range(d) if j != i]
        assert np.allclose(ab[i][:, keep], a[:, keep])                     # all other columns kept
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    assert np.allclose(unit_to_physical(np.zeros(d)), lo)
    assert np.allclose(unit_to_physical(np.ones(d)), hi)
    half = unit_to_physical(np.full(d, 0.5))
    log_dims = np.array([name == "delta_um" or name == "shear_factor" for name in INPUT_NAMES])
    assert np.allclose(half[log_dims], np.sqrt(lo[log_dims] * hi[log_dims]))
    assert np.allclose(half[~log_dims], 0.5 * (lo[~log_dims] + hi[~log_dims]))


def test_trench_ratio_sensitivity_smoke():
    """n_base = 4 on the narrow site returns finite S1/ST for every input (plumbing only)."""
    res = trench_ratio_sensitivity(site="narrow", n_base=4, seed=0, workers=1, n_boot=100)
    assert res["names"] == list(INPUT_NAMES)
    assert res["n_eval"] == 4 * (len(INPUT_NAMES) + 2)
    assert res["site"] == "narrow" and res["flow_solver"] == "stokes" and res["use_flow"] is True
    assert len(res["grid_shape"]) == 2 and min(res["grid_shape"]) >= 8
    assert 0.0 < res["mean_y"] < 1.05, res["mean_y"]                     # floor <= field * 1.02
    for key in ("S1", "ST", "S1_lo", "S1_hi", "ST_lo", "ST_hi"):
        vals = np.asarray(res[key], dtype=float)
        assert vals.shape == (len(INPUT_NAMES),), (key, vals.shape)
        assert np.all(np.isfinite(vals)), (key, vals)
    assert res["dominant_input"] in INPUT_NAMES
    # efficiency only enters ph.Mfac, which stage() does not use when j_field is prescribed
    eff = list(INPUT_NAMES).index("efficiency")
    assert abs(res["ST"][eff]) < 1e-12


def test_trench_no_flow_reference_pins_the_shear_factor():
    """``use_flow=False`` pins shear_factor at 0 (no flow solve) and keeps every other index finite."""
    res = trench_ratio_sensitivity(site="narrow", n_base=4, seed=0, workers=1, n_boot=50,
                                   use_flow=False)
    assert res["use_flow"] is False
    shear = list(INPUT_NAMES).index("shear_factor")
    assert res["S1"][shear] == 0.0 and res["ST"][shear] == 0.0
    assert np.all(np.isfinite(res["S1"])) and np.all(np.isfinite(res["ST"]))
    assert 0.0 < res["mean_y"] < 1.05
