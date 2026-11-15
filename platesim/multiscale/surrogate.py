"""Gaussian-process surrogate (emulator) of an expensive model (improvement plan F5).

Why this module exists
----------------------
``feature.calibrate.fit`` fits a few parameters by MAP (a local ``least_squares`` run) and
reports a finite-difference Jacobian.  Each objective evaluation there is a full level-set
simulation, so an MCMC posterior is out of reach.  This module replaces the expensive model by
a cheap emulator:

1. :func:`make_design` draws a space-filling design (scrambled Sobol by default, Latin
   hypercube on request) over named, bounded parameters.  Parameters flagged ``log`` are
   sampled and interpolated in log space (a log-uniform design over the physical box).
2. :func:`evaluate_model` runs the user's model on the design, optionally in a process pool
   (same convention as :mod:`platesim.trenchflow.sensitivity`).
3. :class:`MultiOutputGP` fits one GP per model output and predicts ``(mean, std)``.  The
   backend is ``sklearn.gaussian_process.GaussianProcessRegressor`` (Matern-5/2 x constant +
   white) when sklearn is importable and a small numpy ARD-RBF GP whose hyperparameters are
   fitted by maximising the marginal likelihood with ``scipy.optimize`` otherwise.  sklearn is
   NOT a project dependency, so in this checkout the numpy path is the one that runs.
4. :meth:`MultiOutputGP.loo_error` reports a leave-one-out error (analytic GP LOO formula, so
   it costs no extra model evaluations) and :func:`holdout_error` evaluates on fresh points.

The model is any callable ``model(x) -> float | array | dict``.  A dict return names the
outputs; that is the natural shape for a multi-site calibration (``{"field_um": ..,
"floor_um": ..}``).  Nothing here knows about plating physics: the trench engine is wired in
by :mod:`platesim.multiscale.posterior` only, behind a CLI, and the tests use cheap analytic
models.

Assumptions / caveats
---------------------
* The design is deterministic (no model noise).  The GP nugget is therefore fitted to a tiny
  value; ``predict`` still adds it back to the latent variance (``include_noise=True``), so a
  prediction at a design point has a small but non-zero std.
* Hyperparameters are fitted on all points, so the analytic LOO error is mildly optimistic.
  It is a diagnostic of interpolation quality, not a calibrated error bar.  The surrogate
  std used in the Bayesian likelihood is the GP posterior std, which does grow between design
  points.
* Extrapolation outside the parameter box is allowed but meaningless: the RBF kernel reverts
  to the output mean with std -> the signal std.

Units follow the bound names (``_um``, ``_s`` ...).  Every number is a model output or a
kernel hyperparameter; no physical constant is introduced here.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import qmc

try:                                            # optional: sklearn is not a project dependency
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    _HAS_SKLEARN = True
except Exception:                               # pragma: no cover - depends on the environment
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------------------- params
@dataclass(frozen=True)
class ParamSpec:
    """One bounded model parameter.

    ``log`` samples/interpolates the parameter in log space, which needs ``0 < lo < hi``.
    The physical value is ``lo * (hi/lo)**u`` for ``u`` in the unit cube instead of
    ``lo + u*(hi-lo)``.
    """

    name: str
    lo: float
    hi: float
    log: bool = False

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("ParamSpec needs a non-empty name")
        lo, hi = float(self.lo), float(self.hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"{self.name}: bounds must be finite, got ({self.lo}, {self.hi})")
        if not lo < hi:
            raise ValueError(f"{self.name}: need lo < hi, got ({lo}, {hi})")
        if self.log and lo <= 0.0:
            raise ValueError(f"{self.name}: a log parameter needs lo > 0, got {lo}")

    def unit_to_physical(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if self.log:
            return float(self.lo) * (float(self.hi) / float(self.lo)) ** u
        return float(self.lo) + u * (float(self.hi) - float(self.lo))

    def physical_to_unit(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.log:
            return np.log(x / float(self.lo)) / np.log(float(self.hi) / float(self.lo))
        return (x - float(self.lo)) / (float(self.hi) - float(self.lo))


def _as_specs(specs: Sequence[ParamSpec] | Sequence[tuple] | Sequence[dict]) -> list[ParamSpec]:
    """Accept ParamSpec objects, ``(name, lo, hi)``/``(name, lo, hi, log)`` tuples, or dicts."""
    out: list[ParamSpec] = []
    for s in specs:
        if isinstance(s, ParamSpec):
            out.append(s)
        elif isinstance(s, dict):
            out.append(ParamSpec(**s))
        else:
            out.append(ParamSpec(str(s[0]), float(s[1]), float(s[2]),
                                 bool(s[3]) if len(s) > 3 else False))
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate parameter names: {names}")
    return out


def bounds_of(specs: Sequence[ParamSpec]) -> np.ndarray:
    """``(d, 2)`` physical bounds."""
    return np.array([[float(s.lo), float(s.hi)] for s in _as_specs(specs)], dtype=float)


def to_physical(u, specs) -> np.ndarray:
    """Map unit-cube coordinates (``..., d``) to physical values."""
    specs = _as_specs(specs)
    u = np.asarray(u, dtype=float)
    if u.shape[-1] != len(specs):
        raise ValueError(f"expected {len(specs)} unit coordinates, got shape {u.shape}")
    cols = [s.unit_to_physical(u[..., i]) for i, s in enumerate(specs)]
    return np.stack(cols, axis=-1)


def to_unit(x, specs) -> np.ndarray:
    """Map physical values (``..., d``) to unit-cube coordinates (values outside [0, 1] allowed)."""
    specs = _as_specs(specs)
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != len(specs):
        raise ValueError(f"expected {len(specs)} physical coordinates, got shape {x.shape}")
    cols = [s.physical_to_unit(x[..., i]) for i, s in enumerate(specs)]
    return np.stack(cols, axis=-1)


# ---------------------------------------------------------------------------------------- design
def _qmc(d: int, n: int, seed: int, method: str) -> np.ndarray:
    method = str(method).lower()
    with warnings.catch_warnings():
        # scipy warns that n is not a power of two; the extra points still fill the box
        warnings.simplefilter("ignore")
        if method in ("lhs", "latin", "latinhypercube"):
            return qmc.LatinHypercube(d=d, scramble=True, seed=int(seed)).random(int(n))
        if method in ("sobol", "qmc"):
            return qmc.Sobol(d=d, scramble=True, seed=int(seed)).random(int(n))
    raise ValueError(f"unknown design method {method!r}; use 'sobol' or 'lhs'")


def make_design(specs, n: int, seed: int = 0, method: str = "sobol", unit: bool = False):
    """``n`` space-filling points over the parameter box (physical values, or unit cube).

    ``unit=True`` returns the raw ``(n, d)`` unit-cube design, which is what the samplers in
    :mod:`platesim.multiscale.posterior` use.
    """
    specs = _as_specs(specs)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    u = _qmc(len(specs), n, seed, method)
    return u if unit else to_physical(u, specs)


def sobol_design(specs, n: int, seed: int = 0, unit: bool = False):
    """Scrambled Sobol design (see :func:`make_design`)."""
    return make_design(specs, n, seed=seed, method="sobol", unit=unit)


def lhs_design(specs, n: int, seed: int = 0, unit: bool = False):
    """Scrambled Latin-hypercube design (see :func:`make_design`)."""
    return make_design(specs, n, seed=seed, method="lhs", unit=unit)


# ---------------------------------------------------------------------------------- model calls
_WORKER_FN: Callable | None = None


def _worker_init(fn: Callable) -> None:
    global _WORKER_FN
    _WORKER_FN = fn
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        import os
        os.environ.setdefault(var, "1")


def _worker_call(row: np.ndarray):
    fn = _WORKER_FN
    if fn is None:
        raise RuntimeError("worker initializer did not install a model function")
    return fn(row)


def as_output_dict(res: Any) -> dict[str, float]:
    """Normalise one model result to ``{name: float}``.

    A dict keeps its keys, a scalar becomes ``{"y": value}``, and an array becomes ``y0..y{k-1}``
    (``{"y": value}`` when it has one element).
    """
    if isinstance(res, dict):
        out: dict[str, float] = {}
        for k, v in res.items():
            arr = np.asarray(v, dtype=float)
            if arr.size != 1:
                raise ValueError(f"model output {k!r} is not scalar (shape {arr.shape})")
            out[str(k)] = float(arr.ravel()[0])
        if not out:
            raise ValueError("model returned an empty dict")
        return out
    arr = np.asarray(res, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("model returned no values")
    if arr.size == 1:
        return {"y": float(arr[0])}
    return {f"y{i}": float(v) for i, v in enumerate(arr)}


def evaluate_model(model: Callable, X, workers: int = 1, progress: Callable | None = None):
    """Evaluate ``model`` on every row of ``X`` -> ``(values (n, k), names tuple)``.

    ``workers > 1`` uses a process pool and falls back to serial evaluation (with a
    ``RuntimeWarning``) when the model is not picklable, e.g. a closure or a lambda.
    ``progress(i, n)`` is called only on the serial path.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    rows = [X[i] for i in range(n)]
    results: list[Any] = []
    workers = int(workers or 1)
    if workers > 1 and n > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                     initargs=(model,)) as ex:
                results = list(ex.map(_worker_call, rows, chunksize=max(1, n // (4 * workers))))
        except Exception as exc:                # unpicklable model, restricted platform, ...
            warnings.warn(f"process pool unavailable ({type(exc).__name__}: {exc}); "
                          "evaluating serially", RuntimeWarning, stacklevel=2)
            results = []
    if not results:
        results = []
        for i, row in enumerate(rows, 1):
            results.append(model(row))
            if progress is not None:
                progress(i, n)
    names: tuple[str, ...] | None = None
    out: list[list[float]] = []
    for res in results:
        d = as_output_dict(res)
        if names is None:
            names = tuple(d)
        elif tuple(d) != names:
            raise ValueError(f"model outputs changed between rows: {tuple(d)} vs {names}")
        out.append([d[k] for k in names])
    return np.asarray(out, dtype=float), tuple(names or ())


# ------------------------------------------------------------------------------------- GP core
def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared distances, clipped at 0 (unit-cube coordinates, well conditioned)."""
    d = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.clip(d, 0.0, None)


class GPSurrogate:
    """Single-output GP regressor with ``fit`` / ``predict`` -> ``(mean, std)`` / ``loo_error``.

    Parameters
    ----------
    bounds : ``(d, 2)`` array, optional
        Physical coordinate box used to normalise the inputs to the unit cube.  Supplying it
        (as :class:`MultiOutputGP` does from the specs) makes the length scales meaningful
        relative to the whole parameter box even when the design does not span it.  Omitted:
        the box is taken from the training data.
    log_mask : ``(d,)`` bool array, optional
        Dimensions normalised in log space (the ``ParamSpec.log`` parameters), matching
        :func:`to_unit`; otherwise the normalisation is linear.
    backend : ``"auto"``, ``"numpy"``, ``"sklearn"``
        ``"auto"`` uses sklearn when importable, else the numpy ARD-RBF GP.
    include_noise : bool
        Add the fitted nugget to the predictive variance (the default, matching
        ``sklearn.predict(return_std=True)``).  Only the numpy backend reads it; sklearn's
        predictive std already contains its WhiteKernel term.
    """

    def __init__(self, *, bounds=None, log_mask=None, backend: str = "auto",
                 kernel: str = "matern52", n_restarts: int = 2, seed: int = 0,
                 include_noise: bool = True, jitter: float = 1e-10):
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.log_mask = None if log_mask is None else np.asarray(log_mask, dtype=bool)
        self._backend_req = str(backend).lower()
        self.kernel_name = str(kernel).lower()
        self.n_restarts = int(max(0, n_restarts))
        self.seed = int(seed)
        self.include_noise = bool(include_noise)
        self.jitter = float(jitter)

        self.backend: str | None = None
        self._xlo: np.ndarray | None = None
        self._xhi: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._ymu = 0.0
        self._ysd = 1.0
        self._gpr = None
        self._theta: np.ndarray | None = None
        self._cho = None
        self._alpha: np.ndarray | None = None
        self._Kinv_diag: np.ndarray | None = None

    # -- helpers ------------------------------------------------------------------------------
    def __repr__(self) -> str:
        d = 0 if self._x is None else self._x.shape[1]
        return (f"GPSurrogate(d={d}, backend={self.backend!r}, kernel={self.kernel_name!r}, "
                f"n={0 if self._x is None else self._x.shape[0]})")

    def _unit(self, X) -> np.ndarray:
        """Physical -> unit-cube inputs; log dimensions use ``log x`` (same as :func:`to_unit`)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._xlo is None or self._xhi is None:
            raise RuntimeError("GPSurrogate is not fitted")
        span = np.where(self._xhi > self._xlo, self._xhi - self._xlo, 1.0)
        U = (X - self._xlo) / span
        if self.log_mask is not None and U.shape[1] == self.log_mask.size:
            lm = self.log_mask
            if np.any(lm) and np.all(self._xlo[lm] > 0.0) and np.all(self._xhi[lm] > self._xlo[lm]):
                lo_log = np.log(self._xlo[lm])
                span_log = np.log(self._xhi[lm]) - lo_log
                U[:, lm] = (np.log(np.clip(X[:, lm], 1e-300, None)) - lo_log) / span_log
        return U

    def _ell(self) -> np.ndarray:
        assert self._theta is not None
        return np.exp(self._theta[:-1])

    def _sf2(self) -> float:
        # Signal variance fixed at the (unit) variance of the standardised output.  With y
        # standardised, a free amplitude is not identifiable against the nugget: the marginal
        # likelihood then drives it to a near-constant kernel with an inflated predictive std.
        return 1.0

    def _sn2(self) -> float:
        assert self._theta is not None
        return float(np.exp(2.0 * self._theta[-1]))

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        ell = self._ell()
        d2 = _sqdist(A, A) if B is None else _sqdist(A, B)
        return self._sf2() * np.exp(-0.5 * np.sum(d2[..., None] / (ell * ell)[None, None, :], axis=-1))

    # -- fitting ------------------------------------------------------------------------------
    def fit(self, X, y) -> GPSurrogate:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if y.size != n:
            raise ValueError(f"X has {n} rows but y has {y.size} values")
        if n < 3:
            raise ValueError(f"a GP needs at least 3 points, got {n}")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must be finite")

        if self.bounds is None:
            lo = X.min(axis=0)
            hi = X.max(axis=0)
            # a constant column carries no information: give it a unit span so the GP is flat
            span = np.where(hi > lo, hi - lo, 1.0)
            self._xlo, self._xhi = lo - 1e-9 * span, hi + 1e-9 * span
        else:
            if self.bounds.shape != (d, 2):
                raise ValueError(f"bounds must be (d, 2) = ({d}, 2), got {self.bounds.shape}")
            self._xlo, self._xhi = self.bounds[:, 0].copy(), self.bounds[:, 1].copy()

        self._x = self._unit(X)
        self._ymu = float(np.mean(y))
        self._ysd = float(np.std(y))
        if not np.isfinite(self._ysd) or self._ysd <= 0.0:
            self._ysd = 1.0
        self._y = (y - self._ymu) / self._ysd

        want_sklearn = self._backend_req in ("auto", "sklearn")
        if want_sklearn and _HAS_SKLEARN:
            self.backend = "sklearn"
            self._fit_sklearn()
        else:
            if self._backend_req == "sklearn" and not _HAS_SKLEARN:
                warnings.warn("sklearn is not importable; falling back to the numpy GP",
                              RuntimeWarning, stacklevel=2)
            self.backend = "numpy"
            self._fit_numpy()
        return self

    def _fit_sklearn(self) -> None:
        assert self._x is not None and self._y is not None
        d = self._x.shape[1]
        nu = {"matern52": 2.5, "matern32": 1.5, "matern12": 0.5}.get(self.kernel_name, 2.5)
        kernel = (ConstantKernel(1.0, (0.5, 2.0))     # bounded: a free amplitude is degenerate here
                  * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=nu)
                  + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-14, 1e-2)))
        self._gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                             n_restarts_optimizer=self.n_restarts,
                                             random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gpr.fit(self._x, self._y)
        K = np.asarray(self._gpr.kernel_(self._x, self._x), dtype=float)
        self._cho = cho_factor(K + self.jitter * np.eye(len(self._y)), lower=True)
        self._alpha = cho_solve(self._cho, self._y)
        self._Kinv_diag = np.diag(cho_solve(self._cho, np.eye(len(self._y))))

    def _fit_numpy(self) -> None:
        assert self._x is not None and self._y is not None
        d = self._x.shape[1]
        n = self._x.shape[0]
        d2 = _sqdist(self._x, self._x)
        jitter = self.jitter
        y = self._y

        def nll(theta: np.ndarray) -> float:
            ell = np.exp(theta[:d])
            sn2 = float(np.exp(2.0 * theta[d]))
            K = np.exp(-0.5 * np.sum(d2[..., None] / (ell * ell)[None, None, :], axis=-1))
            K = K + (sn2 + jitter) * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = cho_solve((L, True), y)
            return float(0.5 * (y @ alpha) + np.sum(np.log(np.diag(L)))
                         + 0.5 * n * np.log(2.0 * np.pi))

        lo = np.array([-8.0] * d + [-14.0])
        hi = np.array([2.0] * d + [-1.0])
        rng = np.random.default_rng(self.seed)
        base = np.concatenate([np.zeros(d), [-8.0]])          # l = 1 in box units, tiny noise
        starts = [base]
        for _ in range(self.n_restarts):
            starts.append(np.clip(base + rng.normal(0.0, 0.7, size=base.size), lo + 1e-6, hi - 1e-6))

        best_theta, best_val = base, nll(base)
        for x0 in starts:
            res = minimize(nll, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
            if np.isfinite(res.fun) and res.fun < best_val:
                best_theta, best_val = np.asarray(res.x, dtype=float), float(res.fun)
        self._theta = best_theta

        K = self._kernel_matrix(self._x) + (self._sn2() + jitter) * np.eye(n)
        self._cho = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._cho, y)
        self._Kinv_diag = np.diag(cho_solve(self._cho, np.eye(n)))

    # -- prediction --------------------------------------------------------------------------
    def predict(self, X, return_std: bool = True):
        """Posterior mean (and std) at physical ``X``.

        A 1-D ``X`` returns scalars; a 2-D ``X`` returns arrays of length ``len(X)``.
        """
        if self.backend is None:
            raise RuntimeError("GPSurrogate is not fitted")
        X_arr = np.asarray(X, dtype=float)
        single = X_arr.ndim == 1
        U = self._unit(X_arr)
        if self.backend == "sklearn":
            mean_u, std_u = self._gpr.predict(U, return_std=True)     # type: ignore[union-attr]
            mean = self._ymu + self._ysd * np.asarray(mean_u, dtype=float)
            std = self._ysd * np.asarray(std_u, dtype=float)
        else:
            assert self._x is not None and self._alpha is not None and self._cho is not None
            ks = self._kernel_matrix(U, self._x)                      # (m, n)
            mean = self._ymu + self._ysd * (ks @ self._alpha)
            V = cho_solve(self._cho, ks.T)                            # (n, m)
            var = self._sf2() - np.sum(ks.T * V, axis=0)              # sum over training points
            var = np.clip(var, 0.0, None)
            if self.include_noise:
                var = var + self._sn2()
            std = self._ysd * np.sqrt(var)
        if single:
            return float(np.asarray(mean).ravel()[0]), float(np.asarray(std).ravel()[0])
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)

    # -- diagnostics -------------------------------------------------------------------------
    def loo_error(self) -> dict:
        """Leave-one-out error from the analytic GP LOO formula (no refits).

        ``mu_i^{-i} = y_i - alpha_i / (K^-1)_ii`` and ``var_i^{-i} = 1 / (K^-1)_ii``
        (Rasmussen & Williams, GPML sec. 5.4.2; exact for a noiseless GP and approximate with
        the fitted nugget).  The hyperparameters saw all points, so the error is mildly
        optimistic.  Residuals are in the original ``y`` units.
        """
        if self._alpha is None or self._Kinv_diag is None:
            raise RuntimeError("GPSurrogate is not fitted")
        alpha = self._alpha
        diag = np.clip(self._Kinv_diag, 1e-300, None)
        resid = (alpha / diag) * self._ysd                # y_i - mu_i^{-i}, original units
        y_true = self._y * self._ysd + self._ymu
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return {
            "n": int(resid.size),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(np.mean(np.abs(resid))),
            "max_abs": float(np.max(np.abs(resid))),
            "r2": float(1.0 - np.sum(resid ** 2) / denom) if denom > 0.0 else float("nan"),
            "residuals": resid,
        }

    def hyperparameters(self) -> dict:
        """Fitted kernel hyperparameters (length scales in units of the parameter box)."""
        if self.backend == "sklearn":
            kernel = self._gpr.kernel_                       # type: ignore[union-attr]
            out = {"kernel": str(kernel), "backend": "sklearn"}
            for name, value in kernel.get_params().items():
                if name.endswith("__length_scale"):
                    out["length_scale"] = np.asarray(value, dtype=float).tolist()
                elif name.endswith("__noise_level"):
                    out["noise_level"] = float(value)
            return out
        if self._theta is None:
            raise RuntimeError("GPSurrogate is not fitted")
        return {"kernel": "ard_rbf", "backend": "numpy",
                "length_scale": np.exp(self._theta[:-1]).tolist(),
                "signal_std": float(np.sqrt(self._sf2())),      # fixed at 1 (standardised y)
                "noise_std": float(np.sqrt(self._sn2()))}


class MultiOutputGP:
    """One :class:`GPSurrogate` per output of a vector-valued model.

    ``predict(X) -> (mean (m, k), std (m, k))``; a 1-D ``X`` returns ``(k,), (k,)``.
    """

    def __init__(self, specs, output_names: Iterable[str] | None = None,
                 backend: str = "auto", kernel: str = "matern52", n_restarts: int = 2,
                 seed: int = 0, include_noise: bool = True):
        self.specs = _as_specs(specs)
        self._bounds = bounds_of(self.specs)
        self.output_names: tuple[str, ...] | None = (
            None if output_names is None else tuple(str(n) for n in output_names))
        self.backend_req = backend
        self.kernel = kernel
        self.n_restarts = n_restarts
        self.seed = seed
        self.include_noise = include_noise
        self.models: list[GPSurrogate] = []
        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None

    def __repr__(self) -> str:
        n = 0 if self.X is None else self.X.shape[0]
        return f"MultiOutputGP(n={n}, outputs={self.output_names}, params={[s.name for s in self.specs]})"

    @property
    def names(self) -> tuple[str, ...]:
        if self.output_names is None:
            raise RuntimeError("MultiOutputGP is not fitted")
        return self.output_names

    @property
    def design_unit(self) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("MultiOutputGP is not fitted")
        return to_unit(self.X, self.specs)

    def fit(self, X, Y) -> MultiOutputGP:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2 or Y.shape[0] != X.shape[0]:
            raise ValueError(f"Y must be (n, k) matching X (n={X.shape[0]}), got {Y.shape}")
        k = Y.shape[1]
        if self.output_names is None:
            self.output_names = ("y",) if k == 1 else tuple(f"y{i}" for i in range(k))
        if len(self.output_names) != k:
            raise ValueError(f"{len(self.output_names)} output names for {k} columns")
        self.X, self.Y = X, Y
        self.models = [GPSurrogate(bounds=self._bounds,
                                   log_mask=[s.log for s in self.specs],
                                   backend=self.backend_req,
                                   kernel=self.kernel, n_restarts=self.n_restarts,
                                   seed=self.seed + j, include_noise=self.include_noise).fit(X, Y[:, j])
                       for j in range(k)]
        return self

    def predict(self, X, return_std: bool = True):
        if not self.models:
            raise RuntimeError("MultiOutputGP is not fitted")
        X_arr = np.asarray(X, dtype=float)
        single = X_arr.ndim == 1
        preds = [m.predict(X_arr, return_std=True) for m in self.models]
        mean = np.column_stack([np.atleast_1d(p[0]) for p in preds])
        std = np.column_stack([np.atleast_1d(p[1]) for p in preds])
        if single:
            return mean.ravel(), std.ravel()
        return mean, std

    def loo_error(self) -> dict[str, dict]:
        """Per-output leave-one-out report (see :meth:`GPSurrogate.loo_error`)."""
        return {name: m.loo_error() for name, m in zip(self.names, self.models)}

    def summary(self) -> dict:
        """Design size, parameters, output names, LOO error and hyperparameters."""
        if self.X is None:
            raise RuntimeError("MultiOutputGP is not fitted")
        return {
            "n_design": int(self.X.shape[0]),
            "params": [s.name for s in self.specs],
            "bounds": {s.name: [float(s.lo), float(s.hi)] for s in self.specs},
            "log_params": [s.name for s in self.specs if s.log],
            "outputs": list(self.names),
            "backend": self.models[0].backend if self.models else None,
            "loo": {name: {k: v for k, v in rep.items() if k != "residuals"}
                    for name, rep in self.loo_error().items()},
            "hyperparameters": {name: m.hyperparameters()
                                for name, m in zip(self.names, self.models)},
        }


def fit_surrogate(model: Callable, specs, n: int = 32, seed: int = 0, method: str = "sobol",
                  workers: int = 1, backend: str = "auto", kernel: str = "matern52",
                  n_restarts: int = 2, progress: Callable | None = None) -> MultiOutputGP:
    """Design -> evaluate -> fit.  The one-call entry point.

    ``model`` takes one physical parameter row and returns a float, an array or a dict of
    scalars.  The returned surrogate keeps ``.X``, ``.Y`` and (via ``summary()``) the LOO error.
    """
    specs = _as_specs(specs)
    X = make_design(specs, n, seed=seed, method=method)
    Y, names = evaluate_model(model, X, workers=workers, progress=progress)
    return MultiOutputGP(specs, names, backend=backend, kernel=kernel,
                         n_restarts=n_restarts, seed=seed).fit(X, Y)


def holdout_error(surrogate: MultiOutputGP, model: Callable, n: int = 32, seed: int = 0,
                  method: str = "sobol", workers: int = 1) -> dict[str, dict]:
    """RMS/MAE/max/R2 of ``surrogate`` against fresh model evaluations (independent design)."""
    X = make_design(surrogate.specs, n, seed=int(seed) + 104729, method=method)
    Y, names = evaluate_model(model, X, workers=workers)
    if tuple(names) != tuple(surrogate.names):
        raise ValueError(f"model outputs {names} != surrogate outputs {surrogate.names}")
    mean, _ = surrogate.predict(X, return_std=False)
    out: dict[str, dict] = {}
    for j, name in enumerate(names):
        resid = mean[:, j] - Y[:, j]
        denom = float(np.sum((Y[:, j] - np.mean(Y[:, j])) ** 2))
        out[name] = {
            "n": int(resid.size),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(np.mean(np.abs(resid))),
            "max_abs": float(np.max(np.abs(resid))),
            "r2": float(1.0 - np.sum(resid ** 2) / denom) if denom > 0.0 else float("nan"),
        }
    return out


__all__ = ["GPSurrogate", "MultiOutputGP", "ParamSpec", "as_output_dict", "bounds_of",
           "evaluate_model", "fit_surrogate", "holdout_error", "lhs_design", "make_design",
           "sobol_design", "to_physical", "to_unit"]
