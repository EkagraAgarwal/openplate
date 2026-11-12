"""Dynamic additive (suppressor / accelerator / chloride) layer for the SOS kMC.

Plan item E3, part b.  Each column carries one additive state on its top site:
``EMPTY``, ``S`` (suppressor), ``A`` (accelerator), ``CL`` (chloride) or
``BLOCKED`` (the static ``theta_block`` mask, inert).  The layer shares the exact
rejection-free BKL clock of deposition and hops:

* S adsorption ``k_ads_S * c_S`` per eligible empty column [1/s];
* S desorption ``k_des_S`` per S column;
* A adsorption ``k_ads_A * c_A`` per empty column, desorption ``k_des_A``;
* Cl- adsorption ``k_ads_Cl * c_Cl`` per empty column, desorption ``k_des_Cl``;
* ``cl_mediated``: S may adsorb only on an empty column next to a Cl- column
  (chloride-mediated co-adsorption); with no Cl- neighbour the S adsorption rate
  is exactly zero;
* burial: a column that receives a Cu atom loses S/A with probability
  ``p_incorp`` (the ``k_minus`` consumption-by-growth term);
* deposition is multiplied by ``1 - f_block`` on S columns (``f_block = 1``,
  the default, is a full block) and by ``f_accel`` on A columns; empty and Cl-
  columns deposit at ``F``.

Mean-field steady state (no Cl, no accelerator) with ``f_block = 0``::

    dtheta/dt = k_ads c (1 - theta) - k_des theta - p_incorp F theta
    theta     = a / (a + k_des + p_incorp F),      a = k_ads c  [1/s]

With ``k_des = p_incorp = 0`` and no deposition this is the Langmuir isotherm
``theta = K c / (1 + K c)``, ``K = k_ads / k_des``.

``effective_rates`` maps a finished run back onto the continuum coverage equation
``dtheta/dt = k_plus c (1 - theta) - k_minus v theta`` (``k_plus`` in
cm3/(mol s), ``k_minus`` in 1/cm, ``v`` in cm/s) with block-averaged errors.

Assumptions [J]: one additive slot per column; additive is bound to the column
top and is not carried by a hopping atom; chloride neither blocks nor enhances
deposition; statically blocked columns accept no additive event.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EMPTY, S, A, CL, BLOCKED = 0, 1, 2, 3, 4
_NS = 5


@dataclass(frozen=True)
class Additive:
    """Rates and switches of the dynamic additive layer (SI-cm bath units).

    k_ads_*  cm3/(mol s)   adsorption rate constant
    c_*      mol/cm3       bulk concentration driving the adsorption
    k_des_*  1/s           desorption rate
    """

    k_ads_S: float = 0.0
    k_des_S: float = 0.0
    c_S: float = 0.0
    k_ads_A: float = 0.0
    k_des_A: float = 0.0
    c_A: float = 0.0
    k_ads_Cl: float = 0.0
    k_des_Cl: float = 0.0
    c_Cl: float = 0.0
    cl_mediated: bool = False
    p_incorp: float = 0.0          # burial probability per deposition on the site
    f_block: float = 1.0           # deposition factor on S sites is 1 - f_block
    f_accel: float = 1.0           # deposition factor on A sites
    theta0_S: float = 0.0          # random initial S coverage

    def __post_init__(self) -> None:
        for name in ("k_ads_S", "k_des_S", "c_S", "k_ads_A", "k_des_A", "c_A",
                     "k_ads_Cl", "k_des_Cl", "c_Cl", "f_accel"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"additive {name} must be finite and >= 0, got {v!r}")
        for name in ("p_incorp", "f_block", "theta0_S"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or not (0.0 <= v <= 1.0):
                raise ValueError(f"additive {name} must be in [0, 1], got {v!r}")


class AdditiveLayer:
    """Per-column occupancy with O(1) class-list bookkeeping (5 state classes)."""

    __slots__ = ("L", "LEFT", "RIGHT", "elig", "epos", "is_elig", "members",
                 "p", "pos", "spec")

    def __init__(self, LEFT: list[int], RIGHT: list[int], p: Additive,
                 spec: list[int]) -> None:
        self.LEFT = LEFT
        self.RIGHT = RIGHT
        self.p = p
        self.spec = spec
        n = self.L = len(spec)
        self.members: list[list[int]] = [[] for _ in range(_NS)]
        self.pos = [[0] * n for _ in range(_NS)]
        for i in range(n):
            s = spec[i]
            self.pos[s][i] = len(self.members[s])
            self.members[s].append(i)
        self.elig: list[int] = []
        self.is_elig = [False] * n
        self.epos = [0] * n
        if p.cl_mediated:
            for i in range(n):
                self._refresh_elig(i)

    # ---- class-list bookkeeping ----------------------------------------
    def set_spec(self, i: int, new: int) -> None:
        """Move column ``i`` to state ``new`` and repair the affected lists."""
        old = self.spec[i]
        if old == new:
            return
        lst = self.members[old]
        pos = self.pos[old]
        last = lst[-1]
        lst[pos[i]] = last
        pos[last] = pos[i]
        lst.pop()
        lst = self.members[new]
        pos = self.pos[new]
        pos[i] = len(lst)
        lst.append(i)
        self.spec[i] = new
        if self.p.cl_mediated:  # eligibility of i and its neighbours can change
            self._refresh_elig(i)
            self._refresh_elig(self.LEFT[i])
            self._refresh_elig(self.RIGHT[i])

    def _refresh_elig(self, i: int) -> None:
        sp = self.spec
        want = (sp[i] == EMPTY
                and (sp[self.LEFT[i]] == CL or sp[self.RIGHT[i]] == CL))
        if want == self.is_elig[i]:
            return
        if want:
            self.is_elig[i] = True
            self.epos[i] = len(self.elig)
            self.elig.append(i)
        else:
            self.is_elig[i] = False
            lst = self.elig
            k = self.epos[i]
            last = lst[-1]
            lst[k] = last
            self.epos[last] = k
            lst.pop()

    # ---- rate helpers (O(1) in the number of classes) ------------------
    def dep_weights(self, F: float, fb: float, fa: float) -> tuple[float, ...]:
        """Deposition-rate weights (empty, accelerator, suppressor, chloride)."""
        m = self.members
        return F * len(m[0]), F * fa * len(m[2]), F * fb * len(m[1]), F * len(m[3])

    def additive_rates(self) -> tuple[float, ...]:
        """Rates (ads_S, des_S, ads_A, des_A, ads_Cl, des_Cl) [1/s]."""
        p = self.p
        m = self.members
        n_free = len(self.elig) if p.cl_mediated else len(m[0])
        n_empty = len(m[0])
        return (p.k_ads_S * p.c_S * n_free, p.k_des_S * len(m[1]),
                p.k_ads_A * p.c_A * n_empty, p.k_des_A * len(m[2]),
                p.k_ads_Cl * p.c_Cl * n_empty, p.k_des_Cl * len(m[3]))

    def theta(self, state: int) -> float:
        return len(self.members[state]) / self.L

    # ---- selection ------------------------------------------------------
    def pick(self, state: int, rr) -> int:
        lst = self.members[state]
        return lst[rr(len(lst))] if lst else -1

    def pick_ads_S(self, rr) -> int:
        lst = self.elig if self.p.cl_mediated else self.members[EMPTY]
        return lst[rr(len(lst))] if lst else -1

    def pick_deposition(self, rnd, rr, w: tuple[float, ...]) -> int:
        """Weighted pick over the deposition classes (empty, A, S, Cl)."""
        u = rnd() * (w[0] + w[1] + w[2] + w[3])
        if u < w[0]:
            return self.pick(EMPTY, rr)
        u -= w[0]
        if u < w[1]:
            return self.pick(A, rr)
        u -= w[1]
        if u < w[2]:
            return self.pick(S, rr)
        return self.pick(CL, rr)


def _block_sigma(x: np.ndarray, i0: int, n_blocks: int) -> tuple[float, float]:
    """Mean and standard error of the mean by block averaging over x[i0:]."""
    seg = x[i0:]
    edges = np.linspace(0, seg.size, min(n_blocks, seg.size) + 1).astype(int)
    means = [float(seg[edges[k]:edges[k + 1]].mean())
             for k in range(edges.size - 1) if edges[k + 1] > edges[k]]
    if len(means) < 2:
        return float(seg.mean()), 0.0
    return float(np.mean(means)), float(np.std(means, ddof=1) / math.sqrt(len(means)))


def effective_rates(run_result: dict[str, Any], *, n_blocks: int = 8,
                    steady_fraction: float = 0.5) -> dict[str, Any]:
    """Effective continuum coverage parameters from a finished additive kMC run.

    The last ``1 - steady_fraction`` of the run (in time) is treated as the
    steady state.  There::

        theta   = <theta_S>                                   (block averaged)
        k_plus  = adsorption flux / (c_S (1 - theta))          cm3/(mol s)
        k_minus = incorporation flux / (v theta)              1/cm

    with ``v = d<h>/dt * d_ML`` the growth velocity [cm/s].  Errors are
    block-averaged (``n_blocks`` equal-sample blocks) combined in quadrature
    with the Poisson error of the event counts.  Returns ``nan`` for a
    parameter whose denominator vanishes (``c_S = 0``, ``theta = 0``, ``v = 0``).
    """
    ap = run_result.get("additive_params")
    if ap is None:
        raise ValueError("effective_rates needs a run with additive=Additive(...)")
    t = np.asarray(run_result["cov_t"], dtype=float)
    if t.size < 4 or t[-1] <= t[0]:
        raise ValueError("too few coverage samples for block averaging")
    th = np.asarray(run_result["cov_theta_S"], dtype=float)
    cum_ads = np.asarray(run_result["cum_ads_S"], dtype=float)
    cum_inc = np.asarray(run_result["cum_inc_S"], dtype=float)
    cum_h = np.asarray(run_result["cum_h"], dtype=float)

    L = float(run_result["L"])
    c = float(ap["c_S"])
    F = float(run_result["params"]["F"])
    d_ml = float(run_result["params"]["d_ML_cm"])

    t_split = t[0] + steady_fraction * (t[-1] - t[0])
    i0 = int(min(np.searchsorted(t, t_split, side="left"), t.size - 2))
    T = float(t[-1] - t[i0])
    if T <= 0.0:
        raise ValueError("empty steady-state window")

    theta, sig_theta = _block_sigma(th, i0, n_blocks)
    n_ads = float(cum_ads[-1] - cum_ads[i0])
    n_inc = float(cum_inc[-1] - cum_inc[i0])
    dh = float(cum_h[-1] - cum_h[i0])

    flux_ads = n_ads / (L * T)              # 1/s per site
    flux_inc = n_inc / (L * T)              # 1/s per site
    v_ml = dh / (L * T)                     # ML/s
    v_cm = v_ml * d_ml                      # cm/s

    free = 1.0 - theta
    k_plus = k_minus = math.nan
    sig_plus = sig_minus = math.nan
    if c > 0.0 and free > 1e-12:
        k_plus = flux_ads / (c * free)
        sig_flux = math.sqrt(max(n_ads, 1.0)) / (L * T)
        rel = math.sqrt((sig_flux / flux_ads) ** 2 + (sig_theta / free) ** 2)
        sig_plus = abs(k_plus) * rel
    if theta > 1e-12 and v_cm > 0.0:
        k_minus = flux_inc / (v_cm * theta)
        sig_flux = math.sqrt(max(n_inc, 1.0)) / (L * T)
        rel = math.sqrt((sig_flux / flux_inc) ** 2 + (sig_theta / theta) ** 2
                        + (1.0 / max(dh, 1.0)))
        sig_minus = abs(k_minus) * rel

    return {
        "theta_S": theta,
        "theta_S_err": sig_theta,
        "k_plus": k_plus,               # cm3/(mol s)
        "k_plus_err": sig_plus,
        "k_minus": k_minus,             # 1/cm
        "k_minus_err": sig_minus,
        "v_cm_s": v_cm,
        "F_ml_s": F,
        "c_S_mol_cm3": c,
        "d_ML_cm": d_ml,
        "window_s": T,
        "n_blocks": n_blocks,
        "n_ads_S": n_ads,
        "n_incorp_S": n_inc,
    }


def additive_params_dict(p: Additive) -> dict[str, Any]:
    """Plain-dict echo of an :class:`Additive` for the run-result payload."""
    return asdict(p)
