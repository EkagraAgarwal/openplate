"""Tests for the GP surrogate (F5) and the surrogate-posterior calibration / propagation (F7).

Every test uses a cheap analytic model, so the suite needs neither the level-set engine nor the
withheld SEM data.  Fixed seeds everywhere; total runtime a few seconds.

The analytic model is linear in ``(a, b)`` and has 6 observables::

    y_j = a * x_j + b * x_j**3 / 3,    x_j = 0.5 ... 3.0

so the posterior of ``(a, b)`` is well determined by 6 noisy numbers while a third parameter the
model does not use must come back as "not identified".
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.multiscale.posterior import (
    ParamPrior,
    default_priors,
    identifiability,
    log_prior_u,
    prior_samples,
    propagate,
    sample_posterior,
)
from platesim.multiscale.surrogate import (
    MultiOutputGP,
    ParamSpec,
    evaluate_model,
    fit_surrogate,
    holdout_error,
    lhs_design,
    sobol_design,
    to_physical,
    to_unit,
)

XS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
TRUTH = np.array([1.5, 0.8])          # (a, b) of the synthetic experiment
MEAS_SIGMA = 0.05                     # um, one fifth of the observable spread


def forward(theta) -> np.ndarray:
    """The synthetic "expensive model": 6 observables from (a, b[, c]); c is ignored."""
    a, b = float(theta[0]), float(theta[1])
    return a * XS + b * XS ** 3 / 3.0


def forward_named(theta) -> dict:
    out = forward(theta)
    return {f"y{j}": float(v) for j, v in enumerate(out)}


TWO = [ParamSpec("a", 0.5, 3.0), ParamSpec("b", 0.2, 1.5)]
THREE = TWO + [ParamSpec("c", 0.1, 10.0, log=True)]


# --------------------------------------------------------------------------------- design + GP
def test_design_and_transforms_roundtrip():
    """Sobol/LHS designs stay in the box, and the unit-cube transform is invertible."""
    specs = [ParamSpec("a", 0.5, 3.0), ParamSpec("d_um", 10.0, 300.0, log=True)]
    for X in (sobol_design(specs, 17, seed=0), lhs_design(specs, 17, seed=0)):
        assert X.shape == (17, 2)
        assert np.all(X[:, 0] >= 0.5) and np.all(X[:, 0] <= 3.0)
        assert np.all(X[:, 1] >= 10.0) and np.all(X[:, 1] <= 300.0)
        U = to_unit(X, specs)
        assert np.all(U >= 0.0) and np.all(U <= 1.0)
        assert np.allclose(to_physical(U, specs), X, rtol=1e-12, atol=0.0)
    # the log parameter spans its range in log space: the median design value is geometric
    assert np.median(sobol_design(specs, 1024, seed=1)[:, 1]) == pytest.approx(
        np.sqrt(10.0 * 300.0), rel=0.08)
    with pytest.raises(ValueError):
        ParamSpec("bad", 0.0, 1.0, log=True)


def test_gp_loo_error_small_on_smooth_function():
    """The emulator interpolates a smooth 2-parameter function (LOO and holdout errors)."""
    specs = [ParamSpec("u", 0.0, 1.0), ParamSpec("v", 0.0, 1.0)]

    def smooth(row):
        u, v = float(row[0]), float(row[1])
        return float(np.sin(2.5 * u) * np.cos(1.7 * v) + 0.3 * u * v)

    surr = fit_surrogate(smooth, specs, n=48, seed=0)
    assert isinstance(surr, MultiOutputGP)
    assert surr.names == ("y",)
    y_std = float(np.std(surr.Y))

    loo = surr.loo_error()["y"]
    assert loo["n"] == 48
    assert loo["rmse"] < 0.02 * y_std
    assert loo["r2"] > 0.98

    held = holdout_error(surr, smooth, n=32, seed=7)["y"]
    assert held["rmse"] < 0.05 * y_std
    assert held["r2"] > 0.95

    # predict -> (mean, std), and the std is smaller at a design point than far outside the box
    x_design = surr.X[0]
    mean_pt, std_pt = surr.models[0].predict(x_design)
    assert np.isfinite(mean_pt) and std_pt > 0.0
    _, std_far = surr.models[0].predict(np.array([1.6, 1.6]))
    assert std_far > std_pt

    # a vector-valued model gives one GP per output
    multi = fit_surrogate(forward, TWO, n=24, seed=1)
    assert multi.Y.shape == (24, 6)
    m, s = multi.predict(np.array([1.4, 0.9]))
    assert m.shape == (6,) and s.shape == (6,)
    assert np.all(np.isfinite(m)) and np.all(s >= 0.0)


def test_evaluate_model_accepts_dict_and_array():
    X = np.array([[1.0, 0.5], [2.0, 1.0]])
    vals, names = evaluate_model(forward_named, X)
    assert names == tuple(f"y{j}" for j in range(6))
    assert vals.shape == (2, 6)
    assert np.allclose(vals[0], forward(X[0]))
    vals_s, names_s = evaluate_model(forward, X)
    assert names_s == tuple(f"y{j}" for j in range(6)) and np.allclose(vals_s, vals)
    vals_q, names_q = evaluate_model(lambda row: float(row[0] + row[1]), X)
    assert names_q == ("y",) and np.allclose(vals_q[:, 0], [1.5, 3.0])


# ------------------------------------------------------------------------------------ posterior
def _posterior(seed: int = 0, noise: bool = True, **kw):
    surr = fit_surrogate(forward, TWO, n=36, seed=seed)
    obs = forward(TRUTH).copy()
    if noise:
        rng = np.random.default_rng(seed + 11)
        obs = obs + rng.normal(0.0, MEAS_SIGMA, size=XS.size)
    post = sample_posterior(surr, obs, MEAS_SIGMA, seed=seed,
                            nwalkers=kw.pop("nwalkers", 20),
                            nsteps=kw.pop("nsteps", 600),
                            burn=kw.pop("burn", 200), **kw)
    return surr, post


def test_posterior_90pct_interval_contains_truth():
    """Gaussian likelihood with sigma_meas + sigma_surrogate: the 90 % interval brackets truth."""
    _, post = _posterior(seed=0)
    assert post["method"] == "emcee"
    assert post["samples"].shape[1] == 2 and post["n_samples"] > 100
    for j, name in enumerate(post["names"]):
        q = post["quantiles"][name]
        assert q["q05"] < q["q50"] < q["q95"]
        assert q["q05"] <= TRUTH[j] <= q["q95"], (name, q, TRUTH[j])
        # the data do narrow both parameters
        assert post["identifiability"]["identified"][name]
    # the emulator is good enough that the posterior width is driven by the data, not the GP
    assert post["sd"]["a"] < 0.15 and post["sd"]["b"] < 0.15
    assert 0.1 < post["acceptance"] <= 1.0


def test_propagate_returns_band_containing_truth():
    """F7: posterior samples -> median and 5/95 % band of a prediction.

    The observation is noiseless here on purpose: this test is about the propagation mechanics
    (band ordering, truth inside the band), not about one particular noise realisation.
    """
    _, post = _posterior(seed=1, noise=False)
    x_new = 4.0

    def predict_fn(theta):
        return {"thickness_um": float(theta[0] * x_new + theta[1] * x_new ** 3 / 3.0)}

    truth = float(TRUTH[0] * x_new + TRUTH[1] * x_new ** 3 / 3.0)
    bands = propagate(post, predict_fn, n=200, seed=0)
    assert bands["n"] == 200
    b = bands["bands"]["thickness_um"]
    assert b["lo"] < b["median"] < b["hi"]
    assert b["lo"] <= truth <= b["hi"], (b, truth)
    assert bands["lo"]["thickness_um"] == pytest.approx(b["lo"])
    assert bands["median"]["thickness_um"] == pytest.approx(b["median"])

    # a scalar prediction function is named "y" and a smaller draw budget is honoured
    small = propagate(post, lambda th: float(th[0]), n=20, seed=0)
    assert small["names"] == ["y"] and small["n"] == 20
    assert small["lo"]["y"] <= TRUTH[0] <= small["hi"]["y"]


def test_unused_parameter_flagged_not_identified():
    """A parameter the model ignores keeps (almost) its prior width and must be flagged."""
    surr = fit_surrogate(forward, THREE, n=36, seed=0)
    rng = np.random.default_rng(2)
    obs = forward(TRUTH) + rng.normal(0.0, MEAS_SIGMA, size=XS.size)
    post = sample_posterior(surr, obs, MEAS_SIGMA, seed=0, nwalkers=24, nsteps=600, burn=200)
    rep = post["identifiability"]
    assert rep["not_identified"] == ["c"]
    assert rep["identified"]["a"] and rep["identified"]["b"]
    assert rep["ratio"]["c"] > 0.9                       # essentially the prior
    assert rep["ratio"]["a"] < 0.2 and rep["ratio"]["b"] < 0.2
    # prior width of c: the 5-95 % span of a log-uniform 0.1 .. 10 is 0.1*(100**0.95 - 100**0.05)
    expected = 0.1 * (100.0 ** 0.95 - 100.0 ** 0.05)
    assert rep["prior_width"]["c"] == pytest.approx(expected, rel=0.05)
    # pairwise correlations: a and b trade off, c is uncorrelated with both
    corr = {f"{p['a']}{p['b']}": p["r"] for p in rep["corr_pairs"]}
    assert corr["ab"] < -0.5
    assert abs(corr["ac"]) < 0.5 and abs(corr["bc"]) < 0.5
    assert post["quantiles"]["c"]["q05"] < 1.0 < post["quantiles"]["c"]["q95"]


def test_adaptive_metropolis_fallback_recovers_truth():
    """The emcee-free sampler must give the same qualitative answer (same data as the emcee test)."""
    _, post = _posterior(seed=0, method="metropolis", nsteps=1500, burn=500)
    assert post["method"] == "metropolis"
    assert post["samples"].shape[1] == 2 and post["n_samples"] > 100
    for j, name in enumerate(post["names"]):
        q = post["quantiles"][name]
        assert q["q05"] < q["q95"]
        assert q["q05"] <= TRUTH[j] <= q["q95"], (name, q)
    assert 0.02 < post["acceptance"] <= 1.0


def test_prior_helpers_and_lognormal_density():
    """default_priors, prior draws and the lognormal log-prior (peak at mu, support = box)."""
    priors = default_priors([ParamSpec("a", 0.5, 3.0),
                             ParamSpec("d_um", 10.0, 300.0, log=True)])
    assert [p.name for p in priors] == ["a", "d_um"]
    assert priors[1].is_log and priors[0].kind == "uniform"
    draws = prior_samples(priors, 64, seed=0)
    assert draws.shape == (64, 2)
    assert np.all((draws[:, 0] >= 0.5) & (draws[:, 0] <= 3.0))
    assert np.all((draws[:, 1] >= 10.0) & (draws[:, 1] <= 300.0))

    ln = [ParamPrior("a", 0.5, 3.0, kind="lognormal", mu=np.log(1.0), sigma=0.5)]
    u_peak = float(ParamSpec("a", 0.5, 3.0, log=True).physical_to_unit(1.0))
    U = np.array([[1e-4], [u_peak], [1.0 - 1e-4]])
    lp = log_prior_u(U, ln)
    assert lp[1] > lp[0] and lp[1] > lp[2]
    assert np.isfinite(lp).all()
    assert log_prior_u(np.array([[1.2]]), ln)[0] == -np.inf


def test_identifiability_report_directly():
    """The report is usable on raw samples (no sampler) and handles a single parameter."""
    rng = np.random.default_rng(5)
    priors = [ParamPrior("a", 0.0, 1.0), ParamPrior("b", 0.0, 1.0)]
    narrow = np.column_stack([rng.normal(0.5, 0.01, 400), rng.random(400)])
    rep = identifiability(narrow, ["a", "b"], priors, seed=1)
    assert rep["identified"] == {"a": True, "b": False}
    assert rep["not_identified"] == ["b"]
    assert rep["ratio"]["a"] < 0.1 and rep["ratio"]["b"] > 0.8
    rep1 = identifiability(narrow[:, :1], ["a"], priors[:1], seed=1)
    assert rep1["correlation"] == [[1.0]] and rep1["corr_pairs"] == []
