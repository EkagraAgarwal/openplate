"""1+1-D solid-on-solid lattice kinetic Monte Carlo (rejection-free BKL / n-fold way).

The interface is a column-height vector ``h[i]`` (periodic in x, SOS = no overhangs
or vacancies).  Two event families share one exact KMC clock:

deposition   ``F`` per free site (``F = j / 2 e N_s``; see ``platesim.kmc.units``)
diffusion    ``nu0 exp(-(E_d + n E_b)/kT)`` per site, ``n`` in {0, 1, 2} in-plane
             bonds of the moving atom at its current height, split 1/2 per direction

Site classes (the bond count ``n``) are stored in per-class member lists, so the
event-type choice is O(1) in the number of classes and only the 3-4 sites whose
bond count can change after an event are re-classified.  Time advances by the
exact exponential increment ``dt = -ln(u) / R_tot`` of the BKL algorithm, so the
trajectory is a statistically exact sample of the master equation (no null
events, no time discretisation).

``model="family"`` is a non-physical test variant used for the growth-exponent
validation: deposit on a random free column, then relax the adatom to the lowest
of {i-1, i, i+1} (the Family deposition rule, Edwards-Wilkinson universality
class in 1+1 D).

``additive=Additive(...)`` (plan item E3b, see :mod:`platesim.kmc.additive`) adds
a dynamic per-column additive layer (suppressor S, accelerator A, chloride) with
adsorption / desorption / burial events in the same BKL bookkeeping; deposition
is blocked on S columns by ``f_block`` and enhanced on A columns by ``f_accel``.
With ``additive=None`` (the default) the trajectory is bit-identical to E3a.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np

from .additive import (
    A, BLOCKED, CL, EMPTY, S, Additive, AdditiveLayer, additive_params_dict,
)
from .units import NU0_HZ, deposition_rate_mA, hop_rate, monolayer_thickness_cm

ASSUMPTIONS = [
    "1+1 D SOS: the interface is a column-height vector h[i] with periodic x; no "
    "overhangs, vacancies or desorption.",
    "Deposition is Poisson (rate F per free column), two electrons per Cu atom, "
    "F = j / (2 e N_s) with N_s = 1.5e15 cm^-2 [J].",
    "Surface diffusion is the bond-counting Arrhenius rate nu0 exp(-(E_d + n E_b)/kT), "
    "n = number of in-plane neighbours at the moving atom's own height (0, 1 or 2) "
    "in 1+1 D; nu0 = 1e13 1/s, E_d = 0.40 eV, E_b = 0.20 eV [J].",
    "A hop moves the top atom of column i onto the top of column i +- 1 (both "
    "directions equiprobable, half the site rate each).",
    "theta_block: static random mask of columns that never accept deposit (additive "
    "suppressor hook); it is not desorbed or re-randomised during the run.",
    "additive layer (Additive): one additive slot per column (S / A / Cl-), bound to "
    "the column top; hops do not carry it.  Deposition factor is 1 - f_block on S "
    "columns, f_accel on A columns, 1 on empty and Cl- columns [J].",
    "Additive burial: a deposition on an S/A column removes the additive with "
    "probability p_incorp (the continuum k_minus consumption by growth) [J].",
    "cl_mediated: suppressor adsorption requires an adjacent adsorbed Cl- column; "
    "Cl- has its own Langmuir adsorption/desorption and does not block deposition [J].",
    "W(t) is the population standard deviation (ddof = 0) of the column heights.",
    "'family' model: not a rate model, only for the EW growth-exponent check.",
]


def interface_width(heights: Sequence[float] | np.ndarray) -> float:
    """Interface width W = std(h) (population, ddof = 0)."""
    h = np.asarray(heights, dtype=float)
    if h.size == 0:
        return 0.0
    return float(np.std(h))


def _check_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def run_sos(
    L: int = 256,
    F: float | None = None,
    T_K: float = 300.0,
    E_d_eV: float = 0.40,
    E_b_eV: float = 0.20,
    coverage_ML: float | None = 50.0,
    seed: int = 0,
    theta_block: float = 0.0,
    snapshots: int | Sequence[float] | None = None,
    *,
    model: str = "arrhenius",
    nu0_Hz: float = NU0_HZ,
    t_max_s: float | None = None,
    max_events: int | None = None,
    heights0: Sequence[int] | None = None,
    additive: Additive | None = None,
) -> dict[str, Any]:
    """Run one SOS kMC trajectory.

    Parameters
    ----------
    L          number of columns (periodic), >= 8.
    F          deposition rate per free site [1/s]; ``None`` -> 15 mA/cm2 at
               300 K (``deposition_rate_mA(15.0)`` = 31.2 /s).
    T_K        temperature [K].
    E_d_eV     terrace diffusion barrier [eV], default 0.40 [J].
    E_b_eV     lateral bond energy [eV], default 0.20 [J].
    coverage_ML stop when the mean height reaches this many monolayers.
                At the default T_K = 300 K, E_d = 0.40 eV the hop rate (1.9e6 /s)
                exceeds F (31 /s) by 6e4, so 50 ML would need ~8e8 events; pass
                max_events (or a higher T) for a bounded, diffusion-limited run.
    seed       RNG seed (``random.Random``); same seed -> identical output.
    theta_block fraction of columns blocked for deposition, 0..1.
    snapshots  ``None``/0: final state only.  ``int N``: N checkpoints geometric
               between ``max(0.5, target / (4 N))`` and the target (mean height for
               growth runs, time for ``F = 0``/``coverage_ML=None`` runs).
               A sequence is read as target mean heights in a growth run and as
               target times otherwise.  An initial snapshot at t = 0 is always
               recorded when snapshots are requested.
    model      ``"arrhenius"`` (default) or ``"family"`` (test rule, see module doc).
    t_max_s    extra stop condition on the KMC time [s].
    max_events hard cap on the total number of events.
    heights0   optional initial heights (list of L non-negative ints).
    additive   optional :class:`platesim.kmc.additive.Additive` dynamic layer; the
               default ``None`` reproduces the E3a trajectory bit for bit.

    Returns
    -------
    dict with ``time_s``, ``t``/``W``/``h_mean`` (snapshot arrays), ``heights``
    (final), ``W_final``, ``h_final``, ``n_events``, ``depositions``, ``events``
    ({"deposition", "hop"} plus the additive channels when enabled),
    ``hop_by_class`` (counts for n = 0, 1, 2), ``blocked`` (bool mask), and the
    echoed parameters under ``params``.  With ``additive`` set it also returns
    ``additive`` (final coverages and event tallies), ``additive_params``,
    ``cov_t``/``cov_theta_S``/``cov_theta_A``/``cov_theta_Cl`` (coverage time
    series) and the cumulative tallies ``cum_ads_S``/``cum_inc_S``/``cum_inc_A``/
    ``cum_h`` aligned with ``cov_t`` (input for :func:`effective_rates`).
    """
    L = int(L)
    if L < 8:
        raise ValueError(f"L must be >= 8, got {L}")
    if F is None:
        F = deposition_rate_mA(15.0)
    F = float(F)
    if F < 0.0 or not math.isfinite(F):
        raise ValueError(f"F must be finite and >= 0, got {F!r}")
    T_K = _check_positive("T_K", T_K)
    if E_d_eV < 0.0 or E_b_eV < 0.0:
        raise ValueError("E_d_eV and E_b_eV must be >= 0")
    if not (0.0 <= theta_block <= 1.0):
        raise ValueError(f"theta_block must be in [0, 1], got {theta_block!r}")
    if model not in ("arrhenius", "family"):
        raise ValueError(f"model must be 'arrhenius' or 'family', got {model!r}")
    if coverage_ML is None and t_max_s is None and max_events is None:
        raise ValueError("need a stop condition: coverage_ML, t_max_s or max_events")
    if coverage_ML is not None and coverage_ML <= 0.0:
        raise ValueError(f"coverage_ML must be > 0, got {coverage_ML!r}")

    rng = random.Random(seed)
    if heights0 is None:
        h = [0] * L
    else:
        h = [int(x) for x in heights0]
        if len(h) != L or any(x < 0 for x in h):
            raise ValueError("heights0 must be a length-L sequence of non-negative ints")

    # static suppressor mask (drawn before any dynamics: seed-stable)
    blocked = [rng.random() < theta_block for _ in range(L)]
    free_idx = [i for i in range(L) if not blocked[i]]
    n_free = len(free_idx)
    if n_free == 0 and coverage_ML is not None and t_max_s is None and max_events is None:
        raise ValueError("theta_block = 1.0 leaves no free column; nothing can deposit")

    LEFT = [i - 1 if i > 0 else L - 1 for i in range(L)]
    RIGHT = [i + 1 if i + 1 < L else 0 for i in range(L)]

    # dynamic additive layer (spec[] per column; BLOCKED = the static mask)
    layer = None
    if additive is not None:
        spec = [BLOCKED if blocked[i] else EMPTY for i in range(L)]
        if additive.theta0_S > 0.0:
            p0 = additive.theta0_S
            for i in range(L):
                if spec[i] == EMPTY and rng.random() < p0:
                    spec[i] = S
        layer = AdditiveLayer(LEFT, RIGHT, additive, spec)

    # ---- per-class rate bookkeeping -------------------------------------
    rates = [0.0, 0.0, 0.0] if model == "family" else [
        hop_rate(n, T_K, E_d_eV, E_b_eV, nu0_Hz) for n in (0, 1, 2)
    ]
    cls = [-1] * L
    members: list[list[int]] = [[], [], []]
    pos = [0] * L
    for i in range(L):
        hi = h[i]
        if hi > 0:
            n = (h[LEFT[i]] >= hi) + (h[RIGHT[i]] >= hi)
            cls[i] = n
            pos[i] = len(members[n])
            members[n].append(i)

    def _setc(i: int) -> None:
        """Recompute the bond class of site i and repair the class lists."""
        hi = h[i]
        c = -1 if hi == 0 else (h[LEFT[i]] >= hi) + (h[RIGHT[i]] >= hi)
        old = cls[i]
        if c == old:
            return
        if old >= 0:
            lst = members[old]
            p = pos[i]
            last = lst[-1]
            lst[p] = last
            pos[last] = p
            lst.pop()
        cls[i] = c
        if c >= 0:
            lst = members[c]
            pos[i] = len(lst)
            lst.append(i)

    return _run_loop(
        L=L, F=F, T_K=T_K, E_d_eV=E_d_eV, E_b_eV=E_b_eV, nu0_Hz=nu0_Hz,
        coverage_ML=coverage_ML, theta_block=theta_block, snapshots=snapshots,
        model=model, t_max_s=t_max_s, max_events=max_events, h=h,
        blocked=blocked, free_idx=free_idx, n_free=n_free, LEFT=LEFT, RIGHT=RIGHT,
        rates=rates, members=members, _setc=_setc, rng=rng, seed=seed, layer=layer,
    )


def _snapshot_targets(snapshots: int | Sequence[float] | None, height_mode: bool,
                      target: float | None, horizon: float | None) -> list[float]:
    """Checkpoint positions: geometric for ``int``, verbatim for a sequence."""
    if snapshots is None:
        return []
    if isinstance(snapshots, (int, np.integer)):
        n = int(snapshots)
        if n <= 0:
            return []
        hi = float(target) if height_mode else float(horizon or 0.0)
        if hi <= 0.0 or not math.isfinite(hi):
            return []
        lo = max(hi / (4.0 * n), 1e-12)
        if height_mode:
            lo = min(hi, max(0.5, lo))
        if n == 1:
            return [hi]
        step = (hi / lo) ** (1.0 / (n - 1))
        return [lo * step ** k for k in range(n)]
    return [x for x in (float(v) for v in snapshots) if x > 0.0]


def _run_loop(*, L, F, T_K, E_d_eV, E_b_eV, nu0_Hz, coverage_ML, theta_block,
              snapshots, model, t_max_s, max_events, h, blocked, free_idx, n_free,
              LEFT, RIGHT, rates, members, _setc, rng, seed, layer=None):
    """BKL event loop (split out of :func:`run_sos` for readability)."""
    family = model == "family"
    height_mode = coverage_ML is not None and F > 0.0 and n_free > 0
    r0c, r1c, r2c = rates
    dep_rate = F * n_free

    add = layer is not None
    if add:
        p_add = layer.p
        fb = 1.0 - p_add.f_block          # deposition factor on S columns
        fa = p_add.f_accel                # deposition factor on A columns
        p_inc = p_add.p_incorp
    else:
        fb = fa = p_inc = 0.0

    r_tot0 = dep_rate
    if add:
        w0 = layer.dep_weights(F, fb, fa)
        r_tot0 = w0[0] + w0[1] + w0[2] + w0[3] + sum(layer.additive_rates())
    for c in (0, 1, 2):
        r_tot0 += rates[c] * len(members[c])
    horizon = t_max_s
    if not height_mode and horizon is None and max_events is not None and r_tot0 > 0.0:
        horizon = max_events / r_tot0
    targets = _snapshot_targets(snapshots, height_mode, coverage_ML, horizon)
    n_targets = len(targets)

    t = 0.0
    total_h = sum(h)
    n_events = 0
    n_dep = 0
    n_hop = 0
    hop_c = [0, 0, 0]
    n_ads_S = n_des_S = n_ads_A = n_des_A = 0
    n_ads_Cl = n_des_Cl = n_inc_S = n_inc_A = 0
    ts: list[float] = []
    ws: list[float] = []
    hms: list[float] = []
    # coverage samples (thinned to <= 512 points) + cumulative tallies
    cov_t: list[float] = []
    cov_S: list[float] = []
    cov_A: list[float] = []
    cov_Cl: list[float] = []
    cum_ads: list[int] = []
    cum_inc: list[int] = []
    cum_incA: list[int] = []
    cum_h: list[int] = []
    sample_step = 64

    def _record() -> None:
        ts.append(t)
        ws.append(float(np.std(h)))
        hms.append(total_h / L)

    def _sample() -> None:
        assert layer is not None
        cov_t.append(t)
        cov_S.append(layer.theta(S))
        cov_A.append(layer.theta(A))
        cov_Cl.append(layer.theta(CL))
        cum_ads.append(n_ads_S)
        cum_inc.append(n_inc_S)
        cum_incA.append(n_inc_A)
        cum_h.append(total_h)


    if n_targets:
        _record()

    rnd = rng.random
    rr = rng.randrange
    log = math.log
    wd: tuple[float, ...] = ()
    ras = rds = raa = rda = rac = rdc = 0.0
    h_stop = None if coverage_ML is None else int(math.ceil(coverage_ML * L))
    limit = -1 if max_events is None else int(max_events)
    k = 0
    if n_targets:
        cur = total_h / L if height_mode else t
        while k < n_targets and cur >= targets[k]:
            k += 1

    while True:
        if limit >= 0 and n_events >= limit:
            break
        if h_stop is not None and total_h >= h_stop:
            break
        if t_max_s is not None and t >= t_max_s:
            break

        m0 = len(members[0])
        m1 = len(members[1])
        m2 = len(members[2])
        r0 = r0c * m0
        r1 = r1c * m1
        r2 = r2c * m2
        if add:
            wd = layer.dep_weights(F, fb, fa)
            dep_rate = wd[0] + wd[1] + wd[2] + wd[3]
            ras, rds, raa, rda, rac, rdc = layer.additive_rates()
            r_add = ras + rds + raa + rda + rac + rdc
        else:
            r_add = 0.0
        r_tot = dep_rate + r0 + r1 + r2
        if add:
            r_tot += r_add
        if r_tot <= 0.0:
            break

        u = rnd()
        t += -log(u if u > 0.0 else 1e-300) / r_tot
        r = rnd() * r_tot

        if r < dep_rate:
            if add:
                i = layer.pick_deposition(rnd, rr, wd)
                if (p_inc > 0.0 and layer.spec[i] in (S, A) and rnd() < p_inc):
                    # incorporation / burial by growth
                    if layer.spec[i] == S:
                        n_inc_S += 1
                    else:
                        n_inc_A += 1
                    layer.set_spec(i, EMPTY)
            else:
                i = free_idx[rr(n_free)]
            h[i] += 1
            if family:
                li = LEFT[i]
                ri = RIGHT[i]
                hi = h[i]
                best = h[li] if h[li] < h[ri] else h[ri]
                if best < hi:  # Family relaxation: move to the lowest neighbour
                    if h[li] == best and h[ri] == best:
                        j = li if rnd() < 0.5 else ri
                    else:
                        j = li if h[li] == best else ri
                    h[i] -= 1
                    h[j] += 1
                    if j == ri:
                        _setc(li)
                        _setc(i)
                        _setc(j)
                        _setc(RIGHT[j])
                    else:
                        _setc(LEFT[j])
                        _setc(j)
                        _setc(i)
                        _setc(ri)
                else:
                    _setc(li)
                    _setc(i)
                    _setc(ri)
            else:
                _setc(LEFT[i])
                _setc(i)
                _setc(RIGHT[i])
            total_h += 1
            n_dep += 1
        elif add and r < dep_rate + r_add:
            v = r - dep_rate
            if v < ras:
                i = layer.pick_ads_S(rr)
                if i >= 0:
                    layer.set_spec(i, S)
                    n_ads_S += 1
            elif v < ras + rds:
                i = layer.pick(S, rr)
                if i >= 0:
                    layer.set_spec(i, EMPTY)
                    n_des_S += 1
            elif v < ras + rds + raa:
                i = layer.pick(EMPTY, rr)
                if i >= 0:
                    layer.set_spec(i, A)
                    n_ads_A += 1
            elif v < ras + rds + raa + rda:
                i = layer.pick(A, rr)
                if i >= 0:
                    layer.set_spec(i, EMPTY)
                    n_des_A += 1
            elif v < ras + rds + raa + rda + rac:
                i = layer.pick(EMPTY, rr)
                if i >= 0:
                    layer.set_spec(i, CL)
                    n_ads_Cl += 1
            else:
                i = layer.pick(CL, rr)
                if i >= 0:
                    layer.set_spec(i, EMPTY)
                    n_des_Cl += 1
        else:
            r -= dep_rate
            if r < r0:
                c = 0
            elif r < r0 + r1:
                c = 1
            else:
                c = 2
            lst = members[c]
            if not lst:  # floating-point edge only
                c = 0 if members[0] else (1 if members[1] else 2)
                lst = members[c]
            i = lst[rr(len(lst))]
            j = LEFT[i] if rnd() < 0.5 else RIGHT[i]
            h[i] -= 1
            h[j] += 1
            if j == RIGHT[i]:
                _setc(LEFT[i])
                _setc(i)
                _setc(j)
                _setc(RIGHT[j])
            else:
                _setc(LEFT[j])
                _setc(j)
                _setc(i)
                _setc(RIGHT[i])
            n_hop += 1
            hop_c[c] += 1

        n_events += 1

        if n_targets:
            cur = total_h / L if height_mode else t
            while k < n_targets and cur >= targets[k]:
                _record()
                k += 1

        if add and n_events % sample_step == 0:
            _sample()
            if len(cov_t) > 512:       # thin the series, keep memory O(1)
                for lst in (cov_t, cov_S, cov_A, cov_Cl, cum_ads, cum_inc,
                            cum_incA, cum_h):
                    del lst[1::2]
                sample_step *= 2

    if not ts or ts[-1] != t:
        _record()
    if add and (not cov_t or cov_t[-1] != t):
        _sample()

    W_final = ws[-1] if ws else float(np.std(h))
    out: dict[str, Any] = {
        "model": model,
        "L": L,
        "time_s": t,
        "t": np.asarray(ts, dtype=float),
        "W": np.asarray(ws, dtype=float),
        "h_mean": np.asarray(hms, dtype=float),
        "heights": np.asarray(h, dtype=int),
        "W_final": W_final,
        "h_final": total_h / L,
        "n_events": n_events,
        "depositions": n_dep,
        "events": {"deposition": n_dep, "hop": n_hop},
        "hop_by_class": np.asarray(hop_c, dtype=int),
        "blocked": np.asarray(blocked, dtype=bool),
        "params": {
            "L": L, "F": F, "T_K": T_K, "E_d_eV": E_d_eV, "E_b_eV": E_b_eV,
            "nu0_Hz": nu0_Hz, "coverage_ML": coverage_ML,
            "theta_block": theta_block, "seed": seed, "model": model,
        },
    }
    if add:
        out["events"].update({
            "ads_S": n_ads_S, "des_S": n_des_S, "ads_A": n_ads_A,
            "des_A": n_des_A, "ads_Cl": n_ads_Cl, "des_Cl": n_des_Cl,
        })   # every entry is one BKL event, so the values sum to n_events
        out["additive"] = {
            "theta_S": layer.theta(S), "theta_A": layer.theta(A),
            "theta_Cl": layer.theta(CL), "spec": np.asarray(layer.spec, dtype=int),
            "n_ads_S": n_ads_S, "n_des_S": n_des_S, "n_ads_A": n_ads_A,
            "n_des_A": n_des_A, "n_ads_Cl": n_ads_Cl, "n_des_Cl": n_des_Cl,
            "n_incorp_S": n_inc_S, "n_incorp_A": n_inc_A,
        }
        out["additive_params"] = additive_params_dict(p_add)  # type: ignore[arg-type]
        out["params"]["d_ML_cm"] = monolayer_thickness_cm()
        out["cov_t"] = np.asarray(cov_t, dtype=float)
        out["cov_theta_S"] = np.asarray(cov_S, dtype=float)
        out["cov_theta_A"] = np.asarray(cov_A, dtype=float)
        out["cov_theta_Cl"] = np.asarray(cov_Cl, dtype=float)
        out["cum_ads_S"] = np.asarray(cum_ads, dtype=int)
        out["cum_inc_S"] = np.asarray(cum_inc, dtype=int)
        out["cum_inc_A"] = np.asarray(cum_incA, dtype=int)
        out["cum_h"] = np.asarray(cum_h, dtype=int)
    return out


