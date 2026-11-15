"""Posterior sampling on the GP surrogate and propagation into prediction ranges (plan F5/F7).

F5 (calibration): :func:`sample_posterior` replaces the MAP + finite-difference Jacobian of
``feature.calibrate.fit`` by a real posterior.  The expensive model is emulated by
:class:`platesim.multiscale.surrogate.MultiOutputGP`; the likelihood is Gaussian with the
measurement sigma and the surrogate std added **in quadrature**::

    var_k = sigma_k^2 + std_k(theta)^2
    log L(theta) = -0.5 * sum_k [ (obs_k - mean_k(theta))^2 / var_k + log(2 pi var_k) ]

so a parameter region where the emulator is unsure is automatically down-weighted.  Priors are
uniform or log-uniform (flat in the sampling variable) or lognormal (truncated to ``[lo, hi]``);
the sampler works on the unit cube, where every prior is a truncated density and the transform
Jacobian is already accounted for (see :func:`log_prior_u`).  The sampler is
``emcee.EnsembleSampler`` with ``vectorize=True`` (emcee IS a project dependency); when emcee is
missing or ``method="metropolis"`` is requested, a simple adaptive Metropolis in the unit cube
is used instead.

An identifiability report (:func:`identifiability`) compares the posterior 90 % width of every
parameter with its prior 90 % width and lists the pairwise posterior correlations.  A ratio
above ``0.7`` means the data barely narrowed that parameter: it is flagged ``not identified``.
That is the honest answer this project needs (one trench sample cannot determine four
parameters, see ``feature/calibrate.py``).

F7 (propagation): :func:`propagate` pushes posterior samples through any prediction function
and returns the median plus the 5/95 % band of every output, i.e. a range instead of a single
number.

CLI (never run by the tests)::

    uv run python -m platesim.multiscale.posterior --trench [--n 24] [--dx 8] [--workers 4]

builds the design on ``feature.calibrate.predict`` for the Trench-1 SEM sites with the
parameters ``(delta_um, flow_shear_s)``, samples the posterior and propagates it into field /
floor thickness ranges.  It needs the measured data file and the full level-set engine; the
physics of the engine is untouched (no parameter is fitted here beyond the posterior).

Honest caveat about ``flow_shear_s``: the engine only consumes ``Params.flow_shear_s`` on its
resolved-flow path (``geometry="trench_full"``); the half-domain ``geometry="trench"`` used by
``calibrate.predict`` is driven by ``delta_um`` alone.  The CLI therefore *reports* that
parameter as not identified instead of pretending the shear was measured.  Tests never run the
CLI (``--trench`` needs the withheld SEM data file).

Units follow the names (``_um``, ``_s``).  ``sigma`` is a measurement std in the units of the
corresponding observation.
"""
from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .surrogate import ParamSpec, evaluate_model, fit_surrogate, to_physical

try:                                             # emcee is a project dependency; keep a fallback
    import emcee
    _HAS_EMCEE = True
except Exception:                                # pragma: no cover - depends on the environment
    emcee = None
    _HAS_EMCEE = False

IDENTIFIED_RATIO = 0.7      # [J] posterior/prior 90 % width above which a parameter is
                            # "not identified" (the data narrowed it by less than ~30 %)

__all__ = ["IDENTIFIED_RATIO", "ParamPrior", "default_priors", "format_bands",
           "format_identifiability", "identifiability", "log_likelihood_u", "log_prior_u",
           "main", "prior_samples", "propagate", "sample_posterior"]


# ---------------------------------------------------------------------------------------- priors
@dataclass(frozen=True)
class ParamPrior:
    """Uniform, log-uniform or lognormal prior on one bounded parameter.

    * ``kind="uniform"``, ``log=False`` - constant density in ``x`` over ``[lo, hi]``.
    * ``kind="uniform"``, ``log=True``  - log-uniform over ``[lo, hi]`` (constant in ``log x``).
    * ``kind="lognormal"``              - lognormal with log-space mean ``mu`` and sd ``sigma``,
      truncated to ``[lo, hi]`` (``mu`` defaults to the midpoint of ``log(lo)`` and ``log(hi)``).
    """

    name: str
    lo: float
    hi: float
    log: bool = False
    kind: str = "uniform"
    mu: float | None = None
    sigma: float | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).lower()
        if kind not in ("uniform", "lognormal"):
            raise ValueError(f"{self.name}: unknown prior kind {self.kind!r}")
        lo, hi = float(self.lo), float(self.hi)
        if not (np.isfinite(lo) and np.isfinite(hi)) or not lo < hi:
            raise ValueError(f"{self.name}: need finite lo < hi, got ({self.lo}, {self.hi})")
        if kind == "lognormal":
            if lo <= 0.0:
                raise ValueError(f"{self.name}: a lognormal prior needs lo > 0, got {lo}")
            if self.sigma is not None and not float(self.sigma) > 0.0:
                raise ValueError(f"{self.name}: lognormal sigma must be > 0")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def is_log(self) -> bool:
        return bool(self.log) or self.kind == "lognormal"

    def spec(self) -> ParamSpec:
        return ParamSpec(self.name, self.lo, self.hi, log=self.is_log)

    def log_mu(self) -> float:
        if self.mu is not None:
            return float(self.mu)
        return 0.5 * (np.log(self.lo) + np.log(self.hi))

    def log_sigma(self) -> float:
        if self.sigma is not None:
            return float(self.sigma)
        return 0.25 * (np.log(self.hi) - np.log(self.lo))     # [J] quarter of the log span


def _as_priors(priors) -> list[ParamPrior]:
    out: list[ParamPrior] = []
    for p in priors:
        if isinstance(p, ParamPrior):
            out.append(p)
        elif isinstance(p, ParamSpec):
            out.append(ParamPrior(p.name, p.lo, p.hi, log=p.log))
        elif isinstance(p, dict):
            out.append(ParamPrior(**p))
        else:
            raise TypeError(f"cannot read a prior from {p!r}")
    names = [p.name for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate parameter names in the priors: {names}")
    return out


def default_priors(specs) -> list[ParamPrior]:
    """Uniform (or log-uniform when the spec is a log parameter) priors over the given box."""
    out = []
    for s in specs:
        if isinstance(s, ParamPrior):
            out.append(s)
        elif isinstance(s, ParamSpec):
            out.append(ParamPrior(s.name, s.lo, s.hi, log=s.log))
        else:
            sp = ParamSpec(str(s[0]), float(s[1]), float(s[2]), bool(s[3]) if len(s) > 3 else False)
            out.append(ParamPrior(sp.name, sp.lo, sp.hi, log=sp.log))
    return out


def prior_samples(priors, n: int, seed: int = 0) -> np.ndarray:
    """Draw ``n`` physical parameter rows from the priors (uniform in the unit cube)."""
    priors = _as_priors(priors)
    rng = np.random.default_rng(int(seed))
    u = rng.random((int(n), len(priors)))
    return to_physical(u, [p.spec() for p in priors])


def log_prior_u(U, priors) -> np.ndarray:
    """Log prior density of unit-cube coordinates, up to an additive constant.

    Uniform and log-uniform priors are flat in ``u``: for a uniform prior the Jacobian
    ``dx/du`` is a constant, for a log-uniform prior ``p(x) ~ 1/x`` cancels ``dx/du ~ x``.
    A truncated lognormal leaves ``-0.5 * ((log x - mu)/sigma)^2``.  Points outside the open
    cube are rejected with ``-inf`` (the box is the prior support).
    """
    priors = _as_priors(priors)
    U = np.atleast_2d(np.asarray(U, dtype=float))
    d = len(priors)
    if U.shape[1] != d:
        raise ValueError(f"expected {d} unit coordinates, got shape {U.shape}")
    lp = np.zeros(U.shape[0], dtype=float)
    inside = np.all((U > 0.0) & (U < 1.0), axis=1)
    lp[~inside] = -np.inf
    for i, p in enumerate(priors):
        if p.kind != "lognormal" or not np.any(inside):
            continue
        x = p.spec().unit_to_physical(np.clip(U[inside, i], 1e-15, 1.0 - 1e-15))
        z = (np.log(x) - p.log_mu()) / p.log_sigma()
        lp[inside] += -0.5 * z * z
    return lp


# ------------------------------------------------------------------------------------ likelihood
def _sigma_vector(sigma, k: int) -> np.ndarray:
    arr = np.asarray(sigma, dtype=float).ravel()
    if arr.size == 1:
        arr = np.full(k, float(arr[0]))
    if arr.size != k:
        raise ValueError(f"sigma has {arr.size} values for {k} observations")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"sigma must be finite and > 0, got {arr}")
    return arr


def log_likelihood_u(U, surrogate, observations, sigma) -> np.ndarray:
    """Gaussian log likelihood of ``observations`` at unit-cube parameters ``U``.

    Measurement sigma and surrogate std enter ``var`` in quadrature (F5).
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    specs = getattr(surrogate, "specs", None)
    if specs is None:
        raise TypeError("surrogate must expose .specs (a MultiOutputGP does)")
    X = to_physical(U, specs)
    mean, std = surrogate.predict(X, return_std=True)
    mean = np.atleast_2d(mean)
    std = np.atleast_2d(std)
    obs = np.atleast_1d(np.asarray(observations, dtype=float))
    if obs.size != mean.shape[1]:
        raise ValueError(f"{obs.size} observations but the surrogate predicts {mean.shape[1]} outputs")
    sig = _sigma_vector(sigma, obs.size)
    var = np.clip(sig[None, :] ** 2 + std ** 2, 1e-300, None)
    resid = obs[None, :] - mean
    return -0.5 * np.sum(resid * resid / var + np.log(2.0 * np.pi * var), axis=1)


# ----------------------------------------------------------------------------------- sampling
def _neg_log_prob(log_prob: Callable[[np.ndarray], float], u: np.ndarray) -> float:
    val = float(log_prob(u[None, :])[0])
    return 1e12 if not np.isfinite(val) else -val


def _map_u(log_prob: Callable[[np.ndarray], float], d: int, seed: int):
    """Unit-cube MAP (multi-start L-BFGS-B) and a Laplace proposal covariance.

    The covariance is ``2.38^2/d * (-H)^-1`` from a finite-difference Hessian of the log
    posterior at the MAP.  It is only used to initialise the chains (emcee walkers and the
    Metropolis proposal) and as the fallback sampler's baseline covariance.
    """
    rng = np.random.default_rng(int(seed) + 17)

    def neg(u: np.ndarray) -> float:
        return _neg_log_prob(log_prob, u)

    best_u = np.full(d, 0.5)
    best = neg(best_u)
    starts = [best_u] + [rng.uniform(0.25, 0.75, size=d) for _ in range(2)]
    for u0 in starts:
        res = minimize(neg, u0, method="L-BFGS-B", bounds=[(1e-6, 1.0 - 1e-6)] * d)
        if np.isfinite(res.fun) and res.fun < best:
            best_u, best = np.clip(np.asarray(res.x, float), 1e-5, 1.0 - 1e-5), float(res.fun)

    eps = 1e-3
    hess = np.zeros((d, d), dtype=float)
    eye = np.eye(d)
    try:
        for i in range(d):
            for j in range(i, d):
                hi, hj = eps * eye[i], eps * eye[j]
                val = (neg(best_u + hi + hj) - neg(best_u + hi - hj)
                       - neg(best_u - hi + hj) + neg(best_u - hi - hj)) / (4.0 * eps * eps)
                hess[i, j] = hess[j, i] = float(val)
    except Exception:                                    # pragma: no cover - defensive
        hess[:] = 0.0
    if not np.all(np.isfinite(hess)):
        hess[:] = 0.0
    w, vec = np.linalg.eigh(hess)
    wmax = float(np.max(w)) if w.size else 0.0
    if wmax > 0.0 and np.sum(w > 1e-6 * wmax) == d:      # positive definite => Laplace is usable
        w = np.clip(w, 1e-6 * wmax, None)
        cov = (vec / w[None, :]) @ vec.T
        cov = 0.5 * (cov + cov.T)
        cov = (2.38 ** 2 / d) * cov
    else:
        cov = np.eye(d) * (0.01 ** 2)
    return best_u, np.clip(cov, -1.0, 1.0) + 1e-12 * np.eye(d)


def _adaptive_metropolis(log_prob, ndim: int, nsteps: int, seed: int, init: np.ndarray,
                         cov0: np.ndarray | None = None):
    """Adaptive Metropolis in the unit cube (fallback when emcee is unavailable).

    Random-walk Gaussian proposal started from the Laplace covariance at the MAP and refreshed
    from the chain history every 100 steps as ``2.38^2/d * cov(history)``, with a mild global
    scale controller (target acceptance 0.15-0.40; Haario et al., 2001).  Proposals leaving the
    cube are rejected, which is exact for the truncated prior.  Adequate for the smooth 2-6
    dimensional posteriors of this project; emcee is preferred whenever it is importable.
    """
    rng = np.random.default_rng(int(seed))
    d = int(ndim)
    x = np.clip(np.asarray(init, dtype=float).copy(), 1e-5, 1.0 - 1e-5)
    lp = float(log_prob(x[None, :])[0])
    cov = np.eye(d) * (0.01 ** 2) if cov0 is None else np.array(cov0, dtype=float)
    scale = 1.0
    chain = np.empty((int(nsteps), d), dtype=float)
    n_acc = 0
    window_acc = 0
    window_n = 0
    adapt_from = max(100, int(nsteps) // 5)
    for it in range(int(nsteps)):
        prop = x + rng.multivariate_normal(np.zeros(d), cov)
        if np.any(prop <= 0.0) or np.any(prop >= 1.0):
            lpp = -np.inf
        else:
            lpp = float(log_prob(prop[None, :])[0])
        window_n += 1
        if np.isfinite(lpp) and np.log(rng.random()) < (lpp - lp):
            x, lp = prop, lpp
            n_acc += 1
            window_acc += 1
        chain[it] = x
        if it >= adapt_from and (it + 1) % 100 == 0:
            rate = window_acc / max(window_n, 1)
            scale = float(np.clip(scale * (0.9 if rate < 0.15 else 1.1 if rate > 0.40 else 1.0),
                                  0.3, 3.0))
            hist = chain[max(0, it - 1000):it + 1]
            if hist.shape[0] > 4 * d:
                base = np.atleast_2d(np.cov(hist, rowvar=False))
                cov = (2.38 ** 2 / d) * (scale ** 2) * base + 1e-10 * np.eye(d)
            window_acc = window_n = 0
    return chain, float(n_acc) / max(int(nsteps), 1)


def sample_posterior(surrogate, observations, sigma, priors=None, *, seed: int = 0,
                     nwalkers: int = 32, nsteps: int = 1200, burn: int = 400, thin: int = 1,
                     method: str = "auto", progress: bool = False) -> dict:
    """Sample the posterior of the surrogate's parameters (plan F5).

    Parameters
    ----------
    surrogate : MultiOutputGP
        Emulator whose inputs are the calibration parameters and whose outputs are the
        observables, fitted on a design that spans the prior box.
    observations, sigma : array
        Measured values and their std, in the surrogate's output order.
    priors : sequence of ParamPrior, optional
        Defaults to a uniform/log-uniform prior on the surrogate's parameter box.
    method : ``"auto"`` | ``"emcee"`` | ``"metropolis"``
        ``"auto"`` uses emcee when importable.

    Returns
    -------
    dict with ``samples`` (physical, shape ``(n, d)``), ``samples_u``, ``names``,
    ``quantiles`` (per parameter 5/50/95 %), ``mean``, ``sd``, ``map``, ``identifiability``,
    ``method``, ``acceptance`` and the sampler settings.
    """
    if not hasattr(surrogate, "predict"):
        raise TypeError("surrogate must provide predict(X) -> (mean, std)")
    obs = np.atleast_1d(np.asarray(observations, dtype=float))
    if obs.size == 0:
        raise ValueError("observations must not be empty")
    specs = getattr(surrogate, "specs", None)
    if specs is not None:
        if priors is None:
            priors = default_priors(specs)
        else:
            priors = _as_priors(priors)
            spec_names = [s.name for s in specs]
            if [p.name for p in priors] != spec_names:
                raise ValueError(f"prior names {[p.name for p in priors]} do not match the "
                                 f"surrogate's parameters {spec_names} (order matters)")
    else:
        if priors is None:
            raise ValueError("priors are required when the surrogate has no .specs")
        priors = _as_priors(priors)
    priors = _as_priors(priors)
    d = len(priors)
    sig = _sigma_vector(sigma, obs.size)

    def log_prob(U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        lp = log_prior_u(U, priors)
        good = np.isfinite(lp)
        out = np.full(U.shape[0], -np.inf)
        if np.any(good):
            out[good] = lp[good] + log_likelihood_u(U[good], surrogate, obs, sig)
        return out

    u_map, cov0 = _map_u(log_prob, d, seed)
    spec_list = [p.spec() for p in priors]

    method = str(method).lower()
    use_emcee = (method in ("auto", "emcee")) and _HAS_EMCEE
    if method == "emcee" and not _HAS_EMCEE:
        raise RuntimeError("method='emcee' requested but emcee is not importable")

    if use_emcee:
        nwalkers = max(int(nwalkers), 2 * d, 2)
        rng = np.random.default_rng(int(seed) + 1)
        # walkers spread over ~10 % of the Laplace proposal width: close enough to the posterior
        # for a fast burn-in, spread enough that emcee's stretch move can operate
        spread = rng.multivariate_normal(np.zeros(d), 0.01 * np.atleast_2d(cov0),
                                         size=nwalkers)
        p0 = np.clip(u_map[None, :] + spread, 1e-4, 1.0 - 1e-4)
        # deterministic chains without touching the caller's global numpy RNG
        state = np.random.get_state()
        try:
            np.random.seed(int(seed) & 0xFFFF_FFFF)
            sampler = emcee.EnsembleSampler(nwalkers, d, log_prob, vectorize=True)  # type: ignore[union-attr]
        finally:
            np.random.set_state(state)
        sampler.run_mcmc(p0, int(nsteps), progress=bool(progress))
        chain = sampler.get_chain(discard=int(burn), thin=int(thin), flat=True)
        acceptance = float(np.mean(sampler.acceptance_fraction))
        used = "emcee"
    else:
        chain_u, acceptance = _adaptive_metropolis(log_prob, d, int(nsteps),
                                                  int(seed) + 1, u_map, cov0)
        chain = chain_u[int(burn)::max(int(thin), 1)]
        used = "metropolis"

    chain_u = np.asarray(chain, dtype=float)
    if chain_u.ndim != 2 or chain_u.shape[1] != d or chain_u.shape[0] < 2:
        raise RuntimeError(f"sampler returned an unusable chain of shape {chain_u.shape}")
    samples = to_physical(chain_u, spec_list)

    q = np.percentile(samples, [5.0, 50.0, 95.0], axis=0)
    names = [p.name for p in priors]
    quantiles = {n: {"q05": float(q[0, i]), "q50": float(q[1, i]), "q95": float(q[2, i])}
                 for i, n in enumerate(names)}
    map_phys = to_physical(u_map[None, :], spec_list)[0]
    ident = identifiability(samples, names, priors, seed=seed)

    return {
        "samples": samples,
        "samples_u": chain_u,
        "names": names,
        "quantiles": quantiles,
        "mean": {n: float(np.mean(samples[:, i])) for i, n in enumerate(names)},
        "sd": {n: float(np.std(samples[:, i])) for i, n in enumerate(names)},
        "map": {n: float(map_phys[i]) for i, n in enumerate(names)},
        "identifiability": ident,
        "method": used,
        "acceptance": acceptance,
        "n_samples": int(samples.shape[0]),
        "nwalkers": int(nwalkers) if used == "emcee" else None,
        "nsteps": int(nsteps),
        "burn": int(burn),
        "thin": int(thin),
        "seed": int(seed),
        "observations": obs.tolist(),
        "sigma": sig.tolist(),
    }


# ------------------------------------------------------------------------------- identifiability
def identifiability(samples, names: Sequence[str], priors=None, *, threshold: float = IDENTIFIED_RATIO,
                    n_prior: int = 20000, seed: int = 0) -> dict:
    """Posterior-vs-prior 90 % width per parameter plus the pairwise posterior correlations.

    ``ratio = posterior_90_width / prior_90_width``.  ``ratio > threshold`` (default 0.7) means
    the data did not narrow the parameter, i.e. it is **not identified** by the observations.
    The prior widths are measured on prior draws (so a truncated lognormal is handled correctly).
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    names = [str(n) for n in names]
    d = samples.shape[1]
    if len(names) != d:
        raise ValueError(f"{len(names)} names for {d} sample columns")

    post_q = np.percentile(samples, [5.0, 95.0], axis=0)
    post_w = post_q[1] - post_q[0]
    if priors is not None:
        pri = prior_samples(priors, n_prior, seed=int(seed) + 999)
        pri_q = np.percentile(pri, [5.0, 95.0], axis=0)
        pri_w = pri_q[1] - pri_q[0]
    else:
        pri_w = np.full(d, np.nan)

    ratio = np.full(d, np.nan)
    ok = np.isfinite(pri_w) & (pri_w > 0.0)
    ratio[ok] = post_w[ok] / pri_w[ok]

    corr = np.eye(d)
    if d > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(samples, rowvar=False)
        corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
    pairs = [{"a": names[i], "b": names[j], "r": float(corr[i, j])}
             for i in range(d) for j in range(i + 1, d)]

    identified = {n: bool(np.isfinite(ratio[i]) and ratio[i] <= threshold)
                  for i, n in enumerate(names)}
    return {
        "names": names,
        "threshold": float(threshold),
        "posterior_width": {n: float(post_w[i]) for i, n in enumerate(names)},
        "prior_width": {n: (float(pri_w[i]) if np.isfinite(pri_w[i]) else None)
                        for i, n in enumerate(names)},
        "ratio": {n: (float(ratio[i]) if np.isfinite(ratio[i]) else None)
                  for i, n in enumerate(names)},
        "identified": identified,
        "not_identified": [n for n in names if not identified[n]],
        "correlation": corr.tolist(),
        "corr_pairs": sorted(pairs, key=lambda p: -abs(p["r"])),
    }


# ---------------------------------------------------------------------------------- propagation
def propagate(samples, predict_fn: Callable, n: int = 500, seed: int = 0,
              quantiles=(5.0, 50.0, 95.0), workers: int = 1) -> dict:
    """Push posterior samples through ``predict_fn`` and return per-output bands (plan F7).

    ``samples`` is the physical ``(m, d)`` array of :func:`sample_posterior` (or its result
    dict).  ``predict_fn(theta_row)`` may return a float, an array or a dict of scalars; the
    output names follow :func:`platesim.multiscale.surrogate.as_output_dict`.  At most ``n``
    samples are used (seeded, without replacement).  Returns the median and the 5/95 % band of
    every output, i.e. a range instead of a single number.
    """
    S = samples["samples"] if isinstance(samples, dict) else samples
    S = np.atleast_2d(np.asarray(S, dtype=float))
    if S.shape[0] < 2:
        raise ValueError(f"need at least 2 posterior samples, got {S.shape[0]}")
    rng = np.random.default_rng(int(seed) + 4241)
    take = min(int(n), S.shape[0])
    idx = np.sort(rng.choice(S.shape[0], size=take, replace=False))
    S_use = S[idx]
    values, names = evaluate_model(predict_fn, S_use, workers=workers)
    levels = [float(x) for x in quantiles]
    if len(levels) != 3:
        raise ValueError(f"quantiles must be (low, centre, high), got {quantiles}")
    q = np.percentile(values, levels, axis=0)
    out_names = [str(nm) for nm in names]
    bands = {nm: {"lo": float(q[0, j]), "median": float(q[1, j]), "hi": float(q[2, j])}
             for j, nm in enumerate(out_names)}
    return {
        "n": int(S_use.shape[0]),
        "names": out_names,
        "quantile_levels": levels,
        "median": {nm: bands[nm]["median"] for nm in out_names},
        "lo": {nm: bands[nm]["lo"] for nm in out_names},
        "hi": {nm: bands[nm]["hi"] for nm in out_names},
        "bands": bands,
        "values": values,
        "samples": S_use,
    }


# ------------------------------------------------------------------------------------- reporting
def format_identifiability(report: dict) -> str:
    """Plain-text table of the identifiability report (used by the CLI)."""
    lines = [f"{'param':<18s}{'prior 90%':>12s}{'post 90%':>12s}{'ratio':>8s}  flag"]
    for name in report["names"]:
        pw = report["prior_width"][name]
        ow = report["posterior_width"][name]
        rr = report["ratio"][name]
        flag = "identified" if report["identified"][name] else "NOT identified"
        lines.append(f"{name:<18s}{_fmt(pw):>12s}{_fmt(ow):>12s}{_fmt(rr):>8s}  {flag}")
    if report["corr_pairs"]:
        top = report["corr_pairs"][0]
        lines.append(f"strongest correlation: {top['a']} / {top['b']} r = {top['r']:+.2f}")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def format_bands(bands: dict) -> str:
    """Plain-text table of a :func:`propagate` result."""
    q = bands["quantile_levels"]
    head = f"{'output':<28s}{f'q{q[0]:g}':>10s}{'median':>10s}{f'q{q[2]:g}':>10s}"
    lines = [head]
    for name in bands["names"]:
        b = bands["bands"][name]
        lines.append(f"{name:<28s}{b['lo']:>10.3g}{b['median']:>10.3g}{b['hi']:>10.3g}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ CLI
# The two calibration parameters of the Trench-1 experiment.  delta_um is the Nernst layer of
# the level-set model (bounds and defaults as in feature.calibrate.FREE_PARAMS); flow_shear_s is
# the wall shear rate the model's resolved flow uses (Params.flow_shear_s, 1/s).  Both are
# sampled log-uniformly: neither is known to a factor of two and both must stay positive.
# Shear bounds are [J]: 1 /s is nearly stagnant, 3000 /s is strong cross-flow at a coupon face.
TRENCH_PARAMS = (ParamSpec("delta_um", 10.0, 300.0, log=True),
                 ParamSpec("flow_shear_s", 1.0, 3000.0, log=True))

_TRENCH_CACHE: dict = {}


def _last(series: dict, key: str):
    for v in reversed(series.get(key) or []):
        if v is not None:
            return float(v)
    return None


def _trench_predict(site: dict, theta, dx_um: float = 8.0):
    """``(field_um, floor_um)`` of one SEM site at ``theta = (delta_um, flow_shear_s)``.

    This mirrors ``feature.calibrate.predict`` (same ``Params``, same ``f_open``, same DC recipe
    and the same 21 um/h + 40 um overburden headroom) and additionally sets
    ``Params.flow_shear_s``; ``feature.calibrate.predict`` does not expose the shear, which is
    the only reason the engine is called directly here.  CLI-only: the tests never run it.

    Structural caveat, reported and not hidden: the engine consumes ``Params.flow_shear_s`` only
    on the resolved-flow path, which it enables for ``geometry="trench_full"``
    (``feature/engine/sim.py``).  With the half-domain ``geometry="trench"`` that
    ``calibrate.predict`` uses, the shear is inert, so the identifiability report of the CLI will
    flag ``flow_shear_s`` as not identified (ratio ~1).  That is the correct answer for this model
    configuration: the half-domain transport is driven by ``delta_um`` and ``k_mix`` only.
    """
    from ..feature import calibrate as cal
    from ..feature.engine import Params, dc, run_case

    delta_um, shear_s = float(theta[0]), float(theta[1])
    hours = float(site["hours"])
    key = (str(site.get("id") or f"{site['width_um']}x{site['depth_um']}"),
           round(delta_um, 9), round(shear_s, 6), round(float(dx_um), 6))
    hit = _TRENCH_CACHE.get(key)
    if hit is not None:
        return hit
    p = Params(geometry="trench", H_um=float(site["depth_um"]), suppressor=False,
               efficiency=float(cal.FREE_PARAMS["efficiency"][3]),
               delta_um=delta_um,
               i0_u=float(cal.FREE_PARAMS["i0_u"][3]),
               k_mix=float(cal.FREE_PARAMS["k_mix"][3]),
               flow_shear_s=shear_s, dx_um=float(dx_um),
               max_overburden_um=40.0 + 21.0 * hours)
    res = run_case(float(site["width_um"]), float(site["T_C"]),
                   dc(float(site["j_mA_cm2"]), hours), p,
                   f_open=float(cal.F_OPEN), label=str(key[0]), out=None)
    field = _last(res["series"], "face_thickness_um")
    floor = _last(res["series"], "floor_height_um")
    out = (float(field if field is not None else res["result"].get("overburden_um") or 0.0),
           float(floor if floor is not None else 0.0))
    _TRENCH_CACHE[key] = out
    return out


def _trench_model(row, sites, dx_um: float = 8.0) -> dict:
    """Design model: all site thicknesses at one parameter row (picklable via functools.partial)."""
    theta = np.asarray(row, dtype=float)
    out: dict[str, float] = {}
    for s in sites:
        field_um, floor_um = _trench_predict(s, theta, dx_um)
        sid = str(s.get("id") or f"{s['width_um']}x{s['depth_um']}")
        out[f"{sid}_field_um"] = field_um
        out[f"{sid}_floor_um"] = floor_um
    return out


def _trench_observations(sites) -> dict[str, tuple[float, float]]:
    """``{output_name: (measured_um, reading_sigma_um)}`` in the ``_trench_model`` naming."""
    from ..feature import calibrate as cal

    out: dict[str, tuple[float, float]] = {}
    for s in sites:
        sid = str(s.get("id") or f"{s['width_um']}x{s['depth_um']}")
        for which in ("field", "floor"):
            m = s[f"{which}_um"]
            sigma = max((float(m["hi"]) - float(m["lo"])) * 0.5, float(cal.SEM_SIGMA_FLOOR_UM))
            out[f"{sid}_{which}_um"] = (float(m["best"]), sigma)
    return out


def main(argv=None) -> dict:
    """CLI: GP-surrogate Bayesian calibration of the Trench-1 SEM sites (F5) + ranges (F7)."""
    ap = argparse.ArgumentParser(
        prog="python -m platesim.multiscale.posterior",
        description="GP-surrogate posterior calibration of the trench level-set model "
                    "(delta_um, flow_shear_s) and propagation into prediction ranges.")
    ap.add_argument("--trench", action="store_true",
                    help="run the (expensive) Trench-1 SEM calibration; without it nothing runs")
    ap.add_argument("--n", type=int, default=24, help="surrogate design size (default 24)")
    ap.add_argument("--dx", type=float, default=8.0, help="feature grid spacing, um (default 8)")
    ap.add_argument("--workers", type=int, default=1, help="processes for the design (default 1)")
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=1200)
    ap.add_argument("--burn", type=int, default=400)
    ap.add_argument("--propagate", type=int, default=400,
                    help="posterior draws propagated into prediction ranges (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the result as JSON to this path")
    a = ap.parse_args(argv)

    if not a.trench:
        ap.print_help()
        return {}

    from ..feature import calibrate as cal

    try:
        sites = cal.load_sites()
    except FileNotFoundError as exc:
        raise SystemExit(f"calibration data unavailable: {exc}") from exc

    model = functools.partial(_trench_model, sites=sites, dx_um=float(a.dx))
    print(f"design: {a.n} Sobol points x {2 * len(sites)} observables "
          f"(delta_um, flow_shear_s) at dx = {a.dx:g} um", flush=True)
    surr = fit_surrogate(model, TRENCH_PARAMS, n=int(a.n), seed=int(a.seed),
                         workers=int(a.workers))
    loo = surr.loo_error()
    for name, rep in loo.items():
        print(f"  surrogate LOO {name:<26s} rmse {rep['rmse']:.3f}  r2 {rep['r2']:+.3f}")

    obs_map = _trench_observations(sites)
    missing = [n for n in surr.names if n not in obs_map]
    if missing:
        raise SystemExit(f"no measurement for surrogate outputs {missing}")
    obs = [obs_map[n][0] for n in surr.names]
    sig = [obs_map[n][1] for n in surr.names]

    post = sample_posterior(surr, obs, sig, default_priors(TRENCH_PARAMS), seed=int(a.seed),
                            nwalkers=int(a.nwalkers), nsteps=int(a.nsteps), burn=int(a.burn))
    print(f"\nposterior: {post['method']}, {post['n_samples']} draws, "
          f"acceptance {post['acceptance']:.2f}")
    for name in post["names"]:
        q = post["quantiles"][name]
        print(f"  {name:<16s} {q['q50']:.4g}   [5%: {q['q05']:.4g}, 95%: {q['q95']:.4g}]")
    print("\n" + format_identifiability(post["identifiability"]))

    result: dict = {"surrogate": surr.summary(), "posterior": {
        k: v for k, v in post.items() if k not in ("samples", "samples_u")}}
    if int(a.propagate) > 0:
        site = sites[0]
        bands = propagate(post, lambda th: dict(zip(
            ("field_um", "floor_um"), _trench_predict(site, th, a.dx))), n=int(a.propagate),
            seed=int(a.seed))
        print(f"\npropagated ranges at {site.get('id', '?')} ({a.propagate} draws):")
        print(format_bands(bands))
        result["propagation"] = {k: v for k, v in bands.items() if k != "values"}
        result["propagation"]["reference_theta"] = [float(v) for v in np.median(post["samples"], axis=0)]

    if a.out:
        payload = {"fitted_with": "python -m platesim.multiscale.posterior",
                   "dx_um": float(a.dx), "seed": int(a.seed), **result}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"\nwrote {a.out}")
    return result


if __name__ == "__main__":
    main()
