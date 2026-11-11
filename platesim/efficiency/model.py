"""Planar Cu current-efficiency loss model (lumped, every term switchable and tagged).

apparent_thickness = (1/A_true) * M/(2 F rho) * integral[ I_front(t) * eta_true(j_front) ] dt - L0
with
  I_front(t)  = I_set * min(1, V_max/V_req(t)) * (1 - f_stray)      (CC->CV fallback, stray conductors)
  V_req(t)    = I*R_ohm + eta_c(j) + eta_a(j_a) + j_a*R_film(t)     (anode film grows past j_a_th)
  eta_true(j) = 1 - j_O2/j - j_H2/j                                 (O2 parasitic; H2 above j_lim_Cu)
  j_O2        = bath.j_lim_o2(delta), delta = delta0*(stir/20)^(-0.5)
  L0          = fixed startup thickness loss (seed etch / induction period)

Context (research/TSV_KNOWLEDGE_BASE.md sec. 2, team W9 deck sec. 14c): true cathodic efficiency
of acid Cu is usually 95-100 % [FT/AB report 06]; the team's 46-55 % (runs 1-6, untaped, 1:1
anode) jumping to 63-69 % (runs 7-8, taped, baffled, larger anode) is mostly bookkeeping [C, strong].

Units inside code: cm, s, A, V, mol; param names carry unit suffixes.
Tags: [FT] full text, [AB] abstract, [SH] secondhand, [C] calc, [J] judgement, [M] measured.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..core.bath import Bath
from ..core.constants import F, M_CU, RHO_CU, N_CU

N_DELTA = 0.5               # delta ~ stir^(-n); magnetic stirring n = 0.505 [FT Avdeef 2022, DOI 10.5599/admet.1568]
STIR_REF_PCT = 20.0         # delta0_um is defined at 20 % stir
A_MEAS_CM2 = 4.0            # coupon ~4 cm2 [M user 2026-10-28]
GAP_CM = 4.0                # anode-cathode distance [J KB sec 7 "3-5 cm"]
UM_PER_A_S_CM2 = M_CU / (N_CU * F * RHO_CU) * 1e4   # (A*s/cm2) -> um at 100 % efficiency

DOE_PATH = Path(__file__).resolve().parents[1] / "data" / "planar_doe.json"

# anode area / measured coupon area, per setup group.
# base: untaped coupon, ~1:1 anode area [M user]; imp1/imp2: "larger anode:cathode ratio" ~2:1
# [J] exact ratio not recorded; Enthone MICROFAB SC specifies 2:1-3:1 [FT report 02 sec 1A].
ANODE_RATIO = {"base": 1.0, "imp1": 2.0, "imp2": 2.0}

# ---------------------------------------------------------------------------
# Parameter registry: weakly informative priors (sources tagged) + transforms
# ---------------------------------------------------------------------------
# transform: "log" (positive), "logit" (bounded), "none".
# prior: ("lognormal", mu, sigma) on ln x; ("norm", mu, sd); ("truncnorm", mu, sd, lo, hi);
#        ("halfnormal", sd).
PARAM_SPECS: dict[str, dict] = {
    # --- common to all hypotheses (H1 core: constant parasitic flux + startup) ---
    "delta0_um":  {"transform": "none", "prior": ("norm", 80.0, 20.0),
                   "help": "Nernst-layer thickness at 20% stir, um [C: Fenech & Tobias 1959 unstirred "
                           "70-100 um in CuSO4/H2SO4; FT Avdeef 2022: 33-47 um at 300 rpm in wells]"},
    "L0_um":      {"transform": "log",  "prior": ("halfnormal", 0.8),
                   "help": "startup thickness loss, um [C: KB sec 2 two-point fit ~1 um; seed etch alone "
                           "3-5% of 300 nm ~0.01 um FT Martyak 2003]"},
    "sigma_um":   {"transform": "log",  "prior": ("halfnormal", 0.6),
                   "help": "SEM thickness noise floor, um [M: deck +-16% at ~7 um; ASTM B487 ~0.8 um abs "
                           "report 06; task budget 0.3-0.5 um + 5%]"},
    # --- H2: stray current / area bookkeeping, per setup group ---
    "f_stray_base": {"transform": "logit", "bounds": (0.0, 0.9), "prior": ("truncnorm", 0.30, 0.15, 0.0, 0.9),
                     "help": "stray conductor fraction, untaped group [J KB sec 2: 0-50% prime suspect]"},
    "f_stray_imp1": {"transform": "logit", "bounds": (0.0, 0.9), "prior": ("truncnorm", 0.05, 0.10, 0.0, 0.9),
                     "help": "stray fraction, taped group; small but not fixed at 0 [J taped covers "
                             "back/edges; clip wetting remains; imp1 still reads 63-69% M]"},
    "f_stray_imp2": {"transform": "logit", "bounds": (0.0, 0.9), "prior": ("truncnorm", 0.05, 0.10, 0.0, 0.9),
                     "help": "stray fraction, taped group (imp2) [J]"},
    "area_base":    {"transform": "none", "prior": ("norm", 1.25, 0.20),
                     "help": "A_true/A_meas, untaped [M report 06: 0.5-0.7 mm edges add 21-29%; "
                             "1.36-1.43 would turn a 70% reading into true 100%]"},
    "area_imp1":    {"transform": "none", "prior": ("norm", 1.00, 0.05),
                     "help": "A_true/A_meas, taped [J taped area well defined, residual SEM/edge spread]"},
    "area_imp2":    {"transform": "none", "prior": ("norm", 1.00, 0.05),
                     "help": "A_true/A_meas, taped (imp2) [J]"},
    # --- H3: CC->CV fallback / anode passivation ---
    "V_max":  {"transform": "log", "prior": ("truncnorm", 3.0, 1.0, 0.5, 12.0),
               "help": "supply compliance voltage, V [J bench supplies commonly set a few V for a beaker "
                       "cell; CC->CV fallback FT B&K Precision note 2025]"},
    "r_ohm":  {"transform": "log", "prior": ("lognormal", 0.0, 0.5),
               "help": "multiplier on geometric cell resistance gap/(kappa*A) [C kappa +-20% KB; geometry ASSUMED]"},
    "j_a_th": {"transform": "log", "prior": ("truncnorm", 21.5, 5.0, 1.0, 100.0),
               "help": "anode current-density threshold for film growth, mA/cm2 [FT Dow CuPulse: anode c.d. "
                       "'not to exceed 20 ASF' = 21.5 mA/cm2]"},
    "k_film": {"transform": "log", "prior": ("lognormal", math.log(0.1), 1.5),
               "help": "film resistance ramp, R_film = k_film*max(0,j_a-j_a_th)*t in Ohm*cm2 [J; mechanism "
                       "FT/AB Tromans & Silva 1997 DOI 10.5006/1.3280457: salt-film i_L transport-limited]"},
}

HYPOTHESES: dict[str, list[str]] = {
    "H1": ["delta0_um", "L0_um", "sigma_um"],
    "H2": ["delta0_um", "L0_um", "sigma_um",
           "f_stray_base", "f_stray_imp1",
           "area_base", "area_imp1"],
    "H3": ["delta0_um", "L0_um", "sigma_um",
           "V_max", "r_ohm", "j_a_th", "k_film"],
    "H4": ["delta0_um", "L0_um", "sigma_um",
           "f_stray_base", "f_stray_imp1",
           "area_base", "area_imp1",
           "V_max", "r_ohm", "j_a_th", "k_film"],
}
HYPOTHESIS_TITLES = {
    "H1": "constant parasitic flux only (O2) + startup loss",
    "H2": "H1 + stray current / area bookkeeping by setup group",
    "H3": "H1 + CC->CV fallback with anode passivation",
    "H4": "combination (H2 + H3)",
}

TINY = 1e-12


# ---------------------------------------------------------------------------
# prior / transform helpers
# ---------------------------------------------------------------------------
def prior_median(name: str) -> float:
    """Prior median (or mean for symmetric priors) - default value for predict mode."""
    kind = PARAM_SPECS[name]["prior"]
    if kind[0] == "lognormal":
        return math.exp(kind[1])
    if kind[0] == "norm":
        return kind[1]
    if kind[0] == "truncnorm":
        return min(max(kind[1], kind[3] + 1e-6), kind[4] - 1e-6)
    if kind[0] == "halfnormal":
        return 0.6745 * kind[1]   # median of half-normal(0,sd)
    raise ValueError(kind)


def default_theta(hypothesis: str = "H4") -> dict[str, float]:
    return {name: prior_median(name) for name in HYPOTHESES[hypothesis]}


def log_prior(theta: dict, names: list[str]) -> float:
    """ln p(theta) for the listed params (physical space). Missing keys -> prior median."""
    lp = 0.0
    for name in names:
        x = float(theta.get(name, prior_median(name)))
        spec = PARAM_SPECS[name]
        bounds = spec.get("bounds")
        if bounds is not None and not (bounds[0] < x < bounds[1]):
            return -math.inf
        if spec["transform"] == "log" and x <= 0:
            return -math.inf
        p = spec["prior"]
        if p[0] == "lognormal":
            mu, sig = p[1], p[2]
            lx = math.log(max(x, TINY))
            lp += -math.log(x) - 0.5 * ((lx - mu) / sig) ** 2 - 0.5 * math.log(2 * math.pi * sig**2)
        elif p[0] in ("norm", "truncnorm"):
            mu, sig = p[1], p[2]
            if p[0] == "truncnorm" and not (p[3] < x < p[4]):
                return -math.inf
            lp += -0.5 * ((x - mu) / sig) ** 2 - 0.5 * math.log(2 * math.pi * sig**2)
        elif p[0] == "halfnormal":
            sig = p[1]
            if x < 0:
                return -math.inf
            lp += -0.5 * (x / sig) ** 2 - 0.5 * math.log(2 * math.pi * sig**2)
        else:
            raise ValueError(p[0])
    return lp


# unconstrained-space transforms (for optimizers / MCMC)
def to_unconstrained(name: str, x: float) -> float:
    tr = PARAM_SPECS[name]["transform"]
    if tr == "log":
        return math.log(max(x, 1e-12))
    if tr == "logit":
        lo, hi = PARAM_SPECS[name]["bounds"]
        u = min(max((x - lo) / (hi - lo), 1e-9), 1 - 1e-9)
        return math.log(u / (1 - u))
    return x


def from_unconstrained(name: str, u: float) -> float:
    tr = PARAM_SPECS[name]["transform"]
    if tr == "log":
        return math.exp(min(max(u, -30.0), 30.0))
    if tr == "logit":
        lo, hi = PARAM_SPECS[name]["bounds"]
        u = min(max(u, -30.0), 30.0)
        return lo + (hi - lo) / (1 + math.exp(-u))
    return u


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def nernst_delta_cm(stir_pct: float, delta0_um: float) -> float:
    """delta = delta0 * (stir/20)^(-0.5) [FT Avdeef 2022 n=0.505]."""
    s = max(float(stir_pct), 1e-3)
    return delta0_um * 1e-4 * (s / STIR_REF_PCT) ** (-N_DELTA)


def _eta_c_V(j_cm2_A: float, bath: Bath) -> float:
    """Cathodic Tafel overpotential, suppressed face [FT Braun Table I i0; alpha_c=0.5 from Bath]."""
    j = max(j_cm2_A, 1e-9)
    b = 2.303 * 8.314462618 * bath.T_K / (bath.alpha_c * F)
    return b * math.log10(j / max(bath.i0_cu_supp, 1e-12))


def _eta_a_V(j_a_cm2_A: float, bath: Bath) -> float:
    """Anodic Tafel overpotential on a filmed P-Cu anode [J alpha_a=1.5 from Bath]."""
    j = max(j_a_cm2_A, 1e-9)
    b = 2.303 * 8.314462618 * bath.T_K / (bath.alpha_a * F)
    return b * math.log10(j / max(bath.i0_cu, 1e-12))


def predict(run: dict, theta: dict | None = None, bath: Bath | None = None,
            hypothesis: str = "H4", n_time: int = 48) -> dict:
    """Forward model for one planar run.

    run: {j_mA_cm2, t_min, stir_pct, setup} (+ optional T_C, expected_um, id).
    theta: physical parameter values; missing keys fall back to prior medians of `hypothesis`.
    Returns {thickness_um, apparent_eff, delivered_charge_C, breakdown{...}}.
    """
    if bath is None:
        bath = Bath()
    T_C = run.get("T_C")
    if T_C is not None:
        bath = bath.at(float(T_C))
    # always seed with the full H4 registry so every term has a value even under H1-H3;
    # `hypothesis` only declares which params are FITTED (see HYPOTHESES), the forward
    # model is shared. Unused terms sit at prior medians (or are switched off in tests).
    th = default_theta("H4")
    if theta:
        th.update(theta)
    # per-term switches: theta["use_o2"]=False etc. disable a loss term (tests, ablation) [tagged terms]
    use = {k: bool(th.get(f"use_{k}", True)) for k in ("o2", "h2", "stray", "cc", "area", "l0")}

    j_mA = float(run["j_mA_cm2"])
    t_s = float(run["t_min"]) * 60.0
    setup = str(run.get("setup", "base"))
    # imp2 = imp1 + smaller baffles and 40 % stir: same taping/anode/clip bookkeeping, so it shares
    # imp1's stray/area parameters (no imp2 data yet; separate params would be prior-only) [J]
    book = "imp1" if setup == "imp2" else setup
    area_ratio = max(float(th.get(f"area_{book}", th.get("area_base", 1.0))), 0.3)
    if not use["area"]:
        area_ratio = 1.0
    f_stray = float(th.get(f"f_stray_{book}", th.get("f_stray_base", 0.0)))
    if not use["stray"]:
        f_stray = 0.0

    A_meas = A_MEAS_CM2
    A_true = A_meas * area_ratio
    A_anode = A_meas * ANODE_RATIO.get(setup, 1.0)
    I_set = j_mA * 1e-3 * A_meas                       # A

    delta_cm = nernst_delta_cm(float(run.get("stir_pct", 20.0)), th["delta0_um"])
    j_o2_A = bath.j_lim_o2(delta_cm)                   # A/cm2 parasitic O2 flux [C via Bath.j_lim_o2]
    j_lim_cu_A = bath.j_lim_cu(delta_cm)               # A/cm2; H2 only above this [C]
    r_geo = GAP_CM / (bath.kappa * math.sqrt(A_anode * A_true))   # Ohm [J geometry assumed]

    n = max(int(n_time), 8)
    dt = t_s / n
    times = dt * (np.arange(n) + 0.5)                  # midpoint rule

    j_a_mA = (I_set / A_anode) * 1e3                   # A held constant in CC (pre-fallback)
    R_film = th["k_film"] * np.maximum(0.0, j_a_mA - th["j_a_th"]) * times   # Ohm*cm2, grows in time
    eta_a = _eta_a_V(j_a_mA * 1e-3, bath) + (j_a_mA * 1e-3) * R_film         # V
    V_req = I_set * r_geo * th["r_ohm"] + _eta_c_V(j_mA * 1e-3, bath) + eta_a
    V_req = np.maximum(V_req, 1e-4)
    # CC->CV fallback: past V_max the supply sits at V_max and delivered current sags [FT B&K note]
    V_max = th["V_max"] if use["cc"] else 1e9
    I_del = I_set * np.minimum(1.0, V_max / V_req)

    j_front = I_del * (1.0 - f_stray) / A_true         # true local cathode current density, A/cm2
    frac_o2 = np.minimum(1.0, j_o2_A / np.maximum(j_front, TINY)) if use["o2"] else np.zeros(n)
    frac_h2 = np.clip((j_front - j_lim_cu_A) / np.maximum(j_front, TINY), 0.0, 1.0) if use["h2"] else np.zeros(n)
    eta_true = np.clip(1.0 - frac_o2 - frac_h2, 0.0, 1.0)

    Q_del = float(I_del.sum() * dt)                                # C drawn from supply
    Q_front = float((I_del * (1.0 - f_stray)).sum() * dt)          # C landing on the coupon
    Q_eff = float((I_del * (1.0 - f_stray) * eta_true).sum() * dt)  # C that becomes Cu on the coupon

    # staged thickness decomposition (um): ideal -> area -> CC/CV -> stray -> eta -> L0
    t_ideal = UM_PER_A_S_CM2 * (I_set / A_meas) * t_s      # Faraday at SET current (the "expected")
    t_area = UM_PER_A_S_CM2 * (I_set / A_true) * t_s       # spread over true area, 100 % eff
    t_cc = UM_PER_A_S_CM2 * float(I_del.mean()) / A_true * t_s
    t_front = UM_PER_A_S_CM2 * Q_front / A_true
    t_eta = UM_PER_A_S_CM2 * Q_eff / A_true
    L0 = th["L0_um"] if use["l0"] else 0.0
    thickness = max(0.0, t_eta - L0)

    loss_area = t_ideal - t_area
    loss_cc = t_area - t_cc
    loss_stray = t_cc - t_front
    loss_o2 = UM_PER_A_S_CM2 * float((I_del * (1 - f_stray) * frac_o2).sum() * dt) / A_true
    loss_h2 = UM_PER_A_S_CM2 * float((I_del * (1 - f_stray) * frac_h2).sum() * dt) / A_true
    loss_l0 = min(L0, t_eta)

    apparent_eff = thickness / max(t_ideal, TINY)
    # gravimetric true efficiency of a TAPED coupon (no area bookkeeping): all Cu on coupon counts,
    # startup seed etch is a real mass loss unless blank-corrected
    grav_num = UM_PER_A_S_CM2 * Q_eff / A_meas - L0
    true_grav_eff = max(0.0, grav_num / max(t_ideal, TINY))

    return {
        "thickness_um": thickness,
        "apparent_eff": apparent_eff,
        "true_grav_eff": true_grav_eff,
        "delivered_charge_C": Q_del,
        "front_charge_C": Q_front,
        "effective_charge_C": Q_eff,
        "breakdown": {
            "faraday_um": t_ideal,
            "loss_area_um": loss_area,
            "loss_cc_um": loss_cc,
            "loss_stray_um": loss_stray,
            "loss_o2_um": loss_o2,
            "loss_h2_um": loss_h2,
            "loss_l0_um": loss_l0,
            "delta_um": delta_cm * 1e4,
            "j_o2_mA_cm2": j_o2_A * 1e3,
            "j_lim_cu_mA_cm2": j_lim_cu_A * 1e3,
            "V_req_mean_V": float(V_req.mean()),
            "V_req_max_V": float(V_req.max()),
            "V_max_V": th["V_max"],
            "I_set_A": I_set,
            "I_mean_A": float(I_del.mean()),
            "eta_true_mean": float(eta_true.mean()),
            "A_true_cm2": A_true,
        },
    }


# ---------------------------------------------------------------------------
# likelihood against the DOE
# ---------------------------------------------------------------------------
def load_doe(path: str | Path = DOE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calibration_runs(doe: dict | None = None) -> list[dict]:
    """Runs 1-8 (measured actual_um present)."""
    doe = doe or load_doe()
    return [r for r in doe["runs"] if r.get("actual_um") is not None and r["id"] <= 8]


def log_likelihood(theta: dict, hypothesis: str, runs: list[dict] | None = None,
                   bath: Bath | None = None) -> float:
    """Gaussian log-likelihood on measured SEM thickness, runs 1-8 by default."""
    runs = runs if runs is not None else calibration_runs()
    sigma = max(float(theta.get("sigma_um", prior_median("sigma_um"))), 0.05)
    ll = 0.0
    for r in runs:
        pred = predict(r, theta, bath=bath, hypothesis=hypothesis)
        resid = r["actual_um"] - pred["thickness_um"]
        ll += -0.5 * (resid / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    return ll


def log_posterior(theta: dict, hypothesis: str, runs: list[dict] | None = None,
                  bath: Bath | None = None) -> float:
    names = HYPOTHESES[hypothesis]
    lp = log_prior(theta, names)
    if not math.isfinite(lp):
        return -math.inf
    return lp + log_likelihood(theta, hypothesis, runs, bath)
