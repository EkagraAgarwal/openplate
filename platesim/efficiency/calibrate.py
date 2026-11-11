"""Bayesian calibration of the planar efficiency model to the team DOE (runs 1-8).

- MAP: penalized least squares (scipy.optimize.least_squares) with weakly informative priors
  as residual terms; multi-start from prior draws.
- MCMC: emcee AffineInvariantSampler if importable, else a simple adaptive Metropolis sampler
  in the unconstrained space (see model.to_unconstrained).
- Model comparison: H1 (O2 only) / H2 (bookkeeping) / H3 (CV fallback) / H4 (combination)
  via BIC at the MAP and WAIC over MCMC samples; with only 8 points the fine distinctions
  (H2 vs H4 vs H3) are weak - see compare_hypotheses() notes.
- Results cached to platesim/efficiency/calibration_results.json.

Units: cm, s, A, V; param suffixes as in model.py.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..core.bath import Bath
from .model import (
    HYPOTHESES, HYPOTHESIS_TITLES, PARAM_SPECS,
    calibration_runs, default_theta, from_unconstrained, log_likelihood, log_posterior,
    predict, prior_median, to_unconstrained,
)

CACHE_PATH = Path(__file__).resolve().parent / "calibration_results.json"
CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# transforms helpers
# ---------------------------------------------------------------------------
def theta_from_u(u: np.ndarray | list[float], hypothesis: str) -> dict[str, float]:
    return {name: from_unconstrained(name, float(ui))
            for name, ui in zip(HYPOTHESES[hypothesis], u)}


def u_from_theta(theta: dict, hypothesis: str) -> list[float]:
    return [to_unconstrained(n, float(theta.get(n, prior_median(n)))) for n in HYPOTHESES[hypothesis]]


def _residuals(u: np.ndarray, hypothesis: str, runs: list[dict], bath: Bath) -> np.ndarray:
    """Penalized-LS residual stack: (data - pred)/sigma for each run + prior residuals in
    physical space (Gaussian core of each prior; bounds enforced by the transform)."""
    theta = theta_from_u(u, hypothesis)
    sigma = max(theta.get("sigma_um", 0.5), 0.05)
    resid = []
    for r in runs:
        pred = predict(r, theta, bath=bath, hypothesis=hypothesis)["thickness_um"]
        resid.append((r["actual_um"] - pred) / sigma)
    for name in HYPOTHESES[hypothesis]:
        x = float(theta.get(name, prior_median(name)))
        p = PARAM_SPECS[name]["prior"]
        if p[0] == "lognormal":
            resid.append((math.log(max(x, 1e-12)) - p[1]) / p[2])
        elif p[0] in ("norm", "truncnorm"):
            resid.append((x - p[1]) / p[2])
        elif p[0] == "halfnormal":
            resid.append(x / p[1])
    return np.asarray(resid, dtype=float)


# ---------------------------------------------------------------------------
# MAP fit
# ---------------------------------------------------------------------------
def fit_map(hypothesis: str, n_starts: int = 4, seed: int = 0, runs: list[dict] | None = None,
            bath: Bath | None = None, max_nfev: int = 400) -> dict:
    """MAP via multi-start penalized least squares. Returns theta_map, loglik, logpost."""
    runs = runs if runs is not None else calibration_runs()
    bath = bath or Bath()
    names = HYPOTHESES[hypothesis]
    rng = np.random.default_rng(seed)

    starts = [np.array(u_from_theta(default_theta(hypothesis), hypothesis))]
    for _ in range(max(n_starts - 1, 0)):
        # jitter each param by ~1 prior sd in unconstrained space
        u0 = starts[0] + rng.normal(0.0, 1.0, size=len(names))
        starts.append(u0)

    best = None
    for u0 in starts:
        try:
            sol = least_squares(_residuals, u0, args=(hypothesis, runs, bath),
                                method="lm" if len(runs) * 2 >= len(names) else "trf",
                                max_nfev=max_nfev)
        except Exception:
            continue
        theta = theta_from_u(sol.x, hypothesis)
        lp = log_posterior(theta, hypothesis, runs, bath)
        if best is None or lp > best["logpost"]:
            best = {"theta_map": theta, "logpost": lp,
                    "loglik": log_likelihood(theta, hypothesis, runs, bath)}
    if best is None:  # pragma: no cover - all starts failed
        theta = default_theta(hypothesis)
        best = {"theta_map": theta, "logpost": log_posterior(theta, hypothesis, runs, bath),
                "loglik": log_likelihood(theta, hypothesis, runs, bath)}
    return best


# ---------------------------------------------------------------------------
# MCMC (emcee, Metropolis fallback)
# ---------------------------------------------------------------------------
def _logprob_u(u: np.ndarray, hypothesis: str, runs: list[dict], bath: Bath) -> float:
    theta = theta_from_u(u, hypothesis)
    return log_posterior(theta, hypothesis, runs, bath)


def sample_mcmc(hypothesis: str, theta0: dict | None = None, n_steps: int = 1500,
                seed: int = 0, runs: list[dict] | None = None, bath: Bath | None = None,
                n_walkers: int | None = None, burn_frac: float = 0.3) -> dict:
    """Posterior samples in physical space. emcee if available, else adaptive Metropolis."""
    runs = runs if runs is not None else calibration_runs()
    bath = bath or Bath()
    names = HYPOTHESES[hypothesis]
    ndim = len(names)
    rng = np.random.default_rng(seed)
    theta0 = theta0 or default_theta(hypothesis)
    u0 = np.array(u_from_theta(theta0, hypothesis))

    try:
        import emcee  # optional dep (listed in report); fallback below if missing

        nw = n_walkers or max(2 * ndim, 26)
        pos = u0 + 1e-3 * rng.normal(size=(nw, ndim))
        sampler = emcee.EnsembleSampler(nw, ndim, _logprob_u, args=(hypothesis, runs, bath))
        sampler.run_mcmc(pos, n_steps, progress=False)
        raw = sampler.get_chain(flat=False)               # (steps, nw, ndim)
        chain: np.ndarray = np.asarray(raw, dtype=float)
        n_burn = int(n_steps * burn_frac)
        flat = chain[n_burn:].reshape(-1, ndim)
        backend = "emcee"
    except ImportError:
        # simple adaptive random-walk Metropolis in unconstrained space
        n_steps = max(int(n_steps), 50)
        n_burn = int(n_steps * burn_frac)
        sigma = np.ones(ndim) * 0.1
        accept = 0
        chain = np.empty((n_steps, ndim))
        chain[0] = u0
        lp = _logprob_u(u0, hypothesis, runs, bath)
        for i in range(1, n_steps):
            prop = chain[i - 1] + rng.normal(0.0, sigma)
            lp_p = _logprob_u(prop, hypothesis, runs, bath)
            if math.isfinite(lp_p) and math.log(max(rng.random(), 1e-300)) < lp_p - lp:
                chain[i], lp, accept = prop, lp_p, accept + 1
            else:
                chain[i] = chain[i - 1]
            if i % 50 == 0 and i <= n_burn:  # crude adaptation during burn-in
                rate = accept / i
                sigma *= 1.5 if rate < 0.2 else (0.7 if rate > 0.5 else 1.0)
        flat = chain[n_burn:]
        backend = "metropolis"

    samples = {name: np.array([from_unconstrained(name, u) for u in flat[:, k]])
               for k, name in enumerate(names)}
    return {"samples": samples, "backend": backend, "n_samples": int(flat.shape[0])}


def samples_summary(samples: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out = {}
    for name, xs in samples.items():
        q = np.percentile(xs, [16, 50, 84])
        out[name] = {"q16": float(q[0]), "median": float(q[1]), "q84": float(q[2]),
                     "mean": float(np.mean(xs))}
    return out


def waic(hypothesis: str, samples: dict[str, np.ndarray], runs: list[dict] | None = None,
         bath: Bath | None = None, max_samples: int = 400) -> tuple[float, float]:
    """WAIC = -2*(lppd - p_waic) from pointwise log-likelihoods at posterior samples."""
    runs = runs if runs is not None else calibration_runs()
    bath = bath or Bath()
    n_s = len(next(iter(samples.values())))
    idx = np.linspace(0, n_s - 1, min(n_s, max_samples)).astype(int)
    ll = np.empty((len(idx), len(runs)))
    for a, i in enumerate(idx):
        theta = {name: float(samples[name][i]) for name in samples}
        theta.update(default_theta(hypothesis))
        theta.update({name: float(samples[name][i]) for name in samples})
        sigma = max(theta.get("sigma_um", 0.5), 0.05)
        for b, r in enumerate(runs):
            pred = predict(r, theta, bath=bath, hypothesis=hypothesis)["thickness_um"]
            resid = r["actual_um"] - pred
            ll[a, b] = -0.5 * (resid / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    from scipy.special import logsumexp
    lppd = float(np.sum(logsumexp(ll, axis=0) - math.log(ll.shape[0])))
    p_waic = float(np.sum(np.var(ll, axis=0)))
    return -2.0 * (lppd - p_waic), p_waic


# ---------------------------------------------------------------------------
# calibration entry point + cache
# ---------------------------------------------------------------------------
def _rms(theta: dict, hypothesis: str, runs: list[dict], bath: Bath) -> float:
    resid = [r["actual_um"] - predict(r, theta, bath=bath, hypothesis=hypothesis)["thickness_um"]
             for r in runs]
    return float(math.sqrt(np.mean(np.square(resid))))


def _predict_all(theta: dict, hypothesis: str, runs: list[dict], bath: Bath) -> list[dict]:
    out = []
    for r in runs:
        p = predict(r, theta, bath=bath, hypothesis=hypothesis)
        out.append({"id": r.get("id"), "setup": r.get("setup"), "j_mA_cm2": r["j_mA_cm2"],
                    "t_min": r["t_min"], "stir_pct": r.get("stir_pct"),
                    "measured_um": r["actual_um"], "predicted_um": round(p["thickness_um"], 3),
                    "apparent_eff": round(p["apparent_eff"], 3)})
    return out


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if data.get("version") == CACHE_VERSION:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"version": CACHE_VERSION, "hypotheses": {}}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def calibrate(hypothesis: str = "H4", method: str = "map", n_steps: int = 1500,
              force: bool = False, seed: int = 0, n_starts: int = 4,
              bath: Bath | None = None) -> dict:
    """Fit one hypothesis to planar_doe runs 1-8.

    Returns {theta_map, samples_summary, loglik, bic, waic?, predictions, rms_um, ...}.
    Cached in calibration_results.json keyed by (hypothesis, method, n_steps).
    """
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"unknown hypothesis {hypothesis}; choose from {list(HYPOTHESES)}")
    if method not in ("map", "mcmc"):
        raise ValueError("method must be 'map' or 'mcmc'")
    bath = bath or Bath()
    runs = calibration_runs()
    n = len(runs)
    k = len(HYPOTHESES[hypothesis])

    cache = load_cache()
    key = f"{hypothesis}:{method}:{n_steps if method == 'mcmc' else 'map'}"
    if not force and key in cache["hypotheses"]:
        return cache["hypotheses"][key]

    t0 = time.time()
    map_fit = fit_map(hypothesis, n_starts=n_starts, seed=seed, runs=runs, bath=bath)
    theta_map = map_fit["theta_map"]
    loglik = map_fit["loglik"]
    bic = k * math.log(n) - 2.0 * loglik

    result = {
        "hypothesis": hypothesis,
        "title": HYPOTHESIS_TITLES[hypothesis],
        "method": method,
        "n_params": k,
        "n_data": n,
        "theta_map": {name: round(float(theta_map[name]), 5) for name in HYPOTHESES[hypothesis]},
        "samples_summary": None,   # filled for method="mcmc"
        "loglik": round(float(loglik), 3),
        "logpost": round(float(map_fit["logpost"]), 3),
        "bic": round(float(bic), 2),
        "rms_um": round(_rms(theta_map, hypothesis, runs, bath), 3),
        "predictions": _predict_all(theta_map, hypothesis, runs, bath),
        "runtime_s": None,
    }

    if method == "mcmc":
        mcmc = sample_mcmc(hypothesis, theta0=theta_map, n_steps=n_steps, seed=seed,
                           runs=runs, bath=bath)
        summary = samples_summary(mcmc["samples"])
        w, p_waic = waic(hypothesis, mcmc["samples"], runs=runs, bath=bath)
        # posterior-median theta as the point estimate for predictions
        theta_med = {name: summary[name]["median"] for name in HYPOTHESES[hypothesis]}
        result.update({
            "samples_summary": {name: {k2: round(v2, 5) for k2, v2 in s.items()}
                                for name, s in summary.items()},
            "mcmc_backend": mcmc["backend"],
            "n_samples": mcmc["n_samples"],
            "waic": round(float(w), 2),
            "p_waic": round(float(p_waic), 2),
            "theta_median": result["theta_map"],   # keep MAP for predictions; median in samples_summary
            "rms_um_median": round(_rms(theta_med, hypothesis, runs, bath), 3),
        })
    result["runtime_s"] = round(time.time() - t0, 1)

    cache["hypotheses"][key] = result
    save_cache(cache)
    return result


def compare_hypotheses(method: str = "map", n_steps: int = 800, force: bool = False) -> list[dict]:
    """Fit H1-H4 and return a comparison table (sorted by BIC)."""
    cache = load_cache()
    rows = []
    for h in ("H1", "H2", "H3", "H4"):
        r = calibrate(h, method=method, n_steps=n_steps, force=force)
        waic_v = r.get("waic")
        if waic_v is None:   # surface cached MCMC WAIC (highest n_steps wins) for this hypothesis
            best_key, best_n = None, -1
            for key, rr in cache.get("hypotheses", {}).items():
                if key.startswith(f"{h}:mcmc") and rr.get("waic") is not None:
                    try:
                        n_c = int(key.rsplit(":", 1)[-1])
                    except ValueError:
                        n_c = 0
                    if n_c > best_n:
                        best_key, best_n = key, n_c
            if best_key:
                waic_v = cache["hypotheses"][best_key]["waic"]
        rows.append({"hypothesis": r["hypothesis"], "title": r["title"], "n_params": r["n_params"],
                     "loglik": r["loglik"], "bic": r["bic"], "waic": waic_v, "rms_um": r["rms_um"]})
    rows.sort(key=lambda x: (x["bic"] if x["bic"] is not None else math.inf))
    return rows


# ---------------------------------------------------------------------------
# predictions for pending runs and true-efficiency curves
# ---------------------------------------------------------------------------
def predict_pending_runs(hypothesis: str = "H4", method: str = "map",
                         theta: dict | None = None, bath: Bath | None = None,
                         n_draw: int = 300, seed: int = 3) -> list[dict]:
    """Runs 9-10 (40 % stir, imp2): point prediction + credible interval.

    Parameter uncertainty is propagated from the cached MCMC samples_summary (split-normal
    marginals, q16/median/q84; independent marginals, so correlations are ignored - the
    interval is approximate). SEM noise (sigma_um) is added for the predictive interval.
    Falls back to a +/-1.96*sigma noise band when no MCMC summary is cached.
    """
    bath = bath or Bath()
    cache = load_cache()
    res = None
    for key, r in cache.get("hypotheses", {}).items():
        if key.startswith(f"{hypothesis}:{method}:"):
            res = r
            break
    if res is None:
        for key, r in cache.get("hypotheses", {}).items():
            if key.startswith(f"{hypothesis}:"):
                res = r
                break
    # borrow the richest MCMC samples_summary (highest n_steps) from cached entries
    summary = (res or {}).get("samples_summary")
    if not summary:
        best_key, best_n = None, -1
        for key, r in cache.get("hypotheses", {}).items():
            if key.startswith(f"{hypothesis}:mcmc") and r.get("samples_summary"):
                try:
                    n_c = int(key.rsplit(":", 1)[-1])
                except ValueError:
                    n_c = 0
                if n_c > best_n:
                    best_key, best_n = key, n_c
        if best_key:
            summary = cache["hypotheses"][best_key]["samples_summary"]
    if theta is None:
        theta = (res or {}).get("theta_map") or default_theta(hypothesis)
    sigma = max(theta.get("sigma_um", 0.6), 0.05)
    rng = np.random.default_rng(seed)

    def _draw_theta() -> dict:
        th = dict(theta)
        if not summary:
            return th
        for name, s in summary.items():
            med, s_lo, s_hi = s["median"], max(s["median"] - s["q16"], 1e-6), max(s["q84"] - s["median"], 1e-6)
            u = rng.normal()
            x = med + u * (s_lo if u < 0 else s_hi)
            spec = PARAM_SPECS.get(name)
            if spec and spec["transform"] == "log":
                x = max(x, 1e-9)
            if spec and spec.get("bounds"):
                x = min(max(x, spec["bounds"][0] + 1e-9), spec["bounds"][1] - 1e-9)
            th[name] = float(x)
        return th

    out = []
    for run in load_pending():
        p = predict(run, theta, bath=bath, hypothesis=hypothesis)
        draws, epi = [], []
        for _ in range(int(n_draw)):
            th_d = _draw_theta()
            t_d = predict(run, th_d, bath=bath, hypothesis=hypothesis)["thickness_um"]
            epi.append(t_d)
            draws.append(max(0.0, t_d + rng.normal(0.0, max(th_d.get("sigma_um", sigma), 0.05))))
        if summary:
            epi_q = np.percentile(epi, [2.5, 97.5])
            pred_q = np.percentile(draws, [2.5, 97.5])
            lo, hi = float(pred_q[0]), float(pred_q[1])
            lo_epi, hi_epi = float(epi_q[0]), float(epi_q[1])
            ci_basis = "MCMC param marginals + SEM noise"
        else:
            lo, hi = max(0.0, p["thickness_um"] - 1.96 * sigma), p["thickness_um"] + 1.96 * sigma
            lo_epi = hi_epi = p["thickness_um"]
            ci_basis = "SEM noise only (no MCMC summary cached)"
        out.append({
            "id": run.get("id"), "setup": run.get("setup"), "j_mA_cm2": run["j_mA_cm2"],
            "t_min": run["t_min"], "stir_pct": run.get("stir_pct"),
            "predicted_um": round(p["thickness_um"], 3),
            "lo_um": round(lo, 3), "hi_um": round(hi, 3),
            "lo_param_um": round(lo_epi, 3), "hi_param_um": round(hi_epi, 3),
            "ci_basis": ci_basis,
            "apparent_eff": round(p["apparent_eff"], 3),
            "true_grav_eff": round(p["true_grav_eff"], 3),
        })
    return out


def load_pending(cache: dict | None = None) -> list[dict]:
    from .model import load_doe
    doe = load_doe()
    return [r for r in doe["runs"] if r.get("actual_um") is None]


def true_efficiency_curve(theta: dict | None = None, hypothesis: str = "H4",
                          j_range: tuple[float, float, int] = (2.0, 40.0, 12),
                          t_min: float = 60.0, temps_C: tuple[float, ...] = (22.0, 40.0),
                          bath: Bath | None = None) -> dict:
    """Gravimetric TRUE efficiency vs j for a taped coupon (imp2, f/area at taped values),
    at each temperature. Gravimetric counts all Cu on the coupon: no area error, no stray;
    L0 (startup etch) still removes mass unless blank-corrected."""
    bath = bath or Bath()
    cache = load_cache()
    res = None
    for key, r in cache.get("hypotheses", {}).items():
        if key.startswith(f"{hypothesis}:"):
            res = r
            break
    th = dict(theta or (res or {}).get("theta_map") or default_theta(hypothesis))
    th["f_stray_imp2"] = 0.0
    th["area_imp2"] = 1.0

    js = np.linspace(j_range[0], j_range[1], int(j_range[2]))
    curves = {}
    for T in temps_C:
        b = bath.at(T)
        effs = []
        for j in js:
            run = {"j_mA_cm2": float(j), "t_min": t_min, "stir_pct": 40.0, "setup": "imp2", "T_C": T}
            effs.append(predict(run, th, bath=b, hypothesis=hypothesis)["true_grav_eff"])
        curves[f"{T:g}C"] = [round(e, 4) for e in effs]
    return {"j_mA_cm2": [round(float(j), 2) for j in js], "t_min": t_min,
            "stir_pct": 40.0, "setup": "imp2 (taped, gravimetric true eta)",
            "curves": curves}


if __name__ == "__main__":  # pragma: no cover
    for h in ("H1", "H2", "H3", "H4"):
        r = calibrate(h, method="map", force=True)
        print(h, "BIC", r["bic"], "rms", r["rms_um"], "loglik", r["loglik"])
