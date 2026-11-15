"""Bath speciation and conductivity for the acid copper sulfate bath (plan item D1a).

Standalone chemistry for CuSO4 + H2SO4 + H2O at the Nano3D operating point
(65 g/L Cu as CuSO4 -> c_Cu = 1.0229 mol/L, 8 g/L H2SO4 -> c_acid = 0.0816 mol/L).
Nothing here is wired into the engine yet: platesim.feature.engine still treats the
bath as fully dissociated (Cu2+, H+, SO4^2- only) with a constant kappa_22.

Species
-------
Cu2+, CuSO4^0 (contact ion pair), H+, HSO4-, SO4^2-.

    HSO4- <-> H+ + SO4^2-        Ka2  = a_H a_SO4 / a_HSO4        [J]
    Cu2+ + SO4^2- <-> CuSO4^0    K_ip = a_p / (a_Cu a_SO4)        [J]

Mass balances (analytical concentrations in mol/L of the bath):

    C_Cu = c_Cu + c_p                     copper
    C_S  = c_S + c_HS + c_p               total sulfate, C_S = C_Cu + c_acid
    C_H  = 2 c_acid                       protons (water autoprotolysis neglected)
    2 c_Cu + c_H = 2 c_S + c_HS           electroneutrality

Four relations for five species; the two equilibria close the system.  It is solved
in log10 space for (c_S, c_H, I) with a damped Newton iteration and analytic (no
inner iteration) activity coefficients, multi-start for robustness.

Honesty tags
------------
* Ka2 = 1.0e-2 at 25 C  [J]  (params.py records 1.05e-2 at 22 C; the van't Hoff
  correction below with dH = -22 kJ/mol returns 1.094e-2 at 22 C, consistent).
* log10 K_ip = 2.3       [J]  ion pairing of CuSO4 is literature-order, NOT verified.
* Davies activity coefficients are OUTSIDE their validity range at I ~ 1-3 mol/L.
  Do not read absolute numbers as predictions: they are trend-level [J] only.
  The default bath lands at I ~ 0.4-1.1 mol/L depending on the activity model
  (Davies: 0.84).  Davies already turns over near I ~ 1.9 mol/L, above which it
  predicts gamma > 1 for divalent ions; that is not physics, it is the formula's
  -0.3 I term.  Use `activity="ideal"` / `"bdot"` to bracket the model error.
* The pairing fraction (~79 % of Cu as CuSO4^0 for the default bath) is driven by
  log10 K_ip = 2.3 and is the least verified number here; treat it as an order.
* Activity coefficients use one ion-size parameter at most (B-dot); no Pitzer terms.
* `I` is computed as 0.5 sum z^2 c in mol/L.  It is NOT converted to molality
  (bath density ~1.1 kg/L would lower it by ~10 %); fine for trend-level work.

Equations used by conductivity / migration (see the function docstrings):
    kappa = F^2/(R T) sum_i z_i^2 D_i c_i        (Nernst-Einstein, S/cm)
    E     = 1 + 2/c_Cu,b * int_{phi_L}^{0} c_Cu(phi) dphi
with c_i(phi) = c_i,b exp(-z_i phi) for the inert ions and c_Cu(phi) fixed by
electroneutrality; the neutral ion pair carries no current.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .constants import F
# R_GAS is imported from params (8.314) rather than constants (8.314462618) so that a
# Nernst-Einstein kappa computed here is bit-comparable with Phys.kappa_ne.
from ..feature.engine.params import R_GAS, stokes_einstein_factor

# ---------------------------------------------------------------- literature constants
KA2_25 = 1.0e-2          # mol/L, HSO4- dissociation at 25 C [J]; params.py has 1.05e-2 @22 C
D_KA2 = -22.0e3          # J/mol, dH of HSO4- -> H+ + SO4 (NBS formation enthalpies) [FT, secondhand]
LOG10_K_IP = 2.3         # Cu2+ + SO4^2- -> CuSO4^0 at infinite dilution [J, NOT verified]
KAPPA_22 = 0.065         # S/cm, the engine placeholder kappa_22 - only for ratio reporting
A_DH25 = 0.5108          # Debye-Huckel A at 25 C, (mol/L)^-1/2, log10 basis [FT]

# Infinite-dilution ionic diffusivities at 25 C, cm2/s [FT, CRC-type tabulated values;
# transcribed from the task sheet, NOT re-verified against a primary source].
D_ION_25 = {
    "cu": 7.14e-6,       # Cu2+
    "h": 9.31e-5,        # H+
    "so4": 1.065e-5,     # SO4^2-
    "hso4": 1.33e-5,     # HSO4-
}

ACTIVITY_MODELS = ("davies", "ideal", "bdot")

_CHARGE = {"cu": 2, "h": 1, "so4": -2, "hso4": -1, "p": 0}


def ka2_at(T_C: float) -> float:
    """Ka2(T) by van't Hoff from Ka2_25 with dH = -22 kJ/mol [J]."""
    T_K = float(T_C) + 273.15
    return KA2_25 * math.exp(-D_KA2 / R_GAS * (1.0 / T_K - 1.0 / 298.15))


def _dh_A(T_C: float) -> float:
    """Debye-Huckel A(T): T^-3/2 scaling with rho*eps^1.5 held at its 25 C value [J]."""
    return A_DH25 * (298.15 / (float(T_C) + 273.15)) ** 1.5


def activity_coefficients(I, activity="davies", T_C=25.0, Ba=1.0, b_dot=0.1):
    """Mean-species activity coefficients on the log10 basis.

    davies : log10 g = -A z^2 ( sqrt(I)/(1+sqrt(I)) - 0.3 I )
    bdot   : log10 g = -A z^2 sqrt(I)/(1 + Ba sqrt(I)) + b_dot I
    ideal  : g = 1

    Ba = 1.0 and b_dot = 0.1 are BOTH placeholders [J] (Ba ~ 1 corresponds to an ion
    size ~3 A).  Neutral species (the CuSO4^0 pair) always get g = 1.
    Davies is invalid above I ~ 0.5 mol/L; see the module docstring.
    """
    if activity not in ACTIVITY_MODELS:
        raise ValueError(f"activity must be one of {ACTIVITY_MODELS}, got {activity!r}")
    I = max(float(I), 0.0)
    if activity == "ideal" or I == 0.0:
        return {k: 1.0 for k in _CHARGE}
    A = _dh_A(T_C)
    sq = math.sqrt(I)
    if activity == "davies":
        base = -A * (sq / (1.0 + sq) - 0.3 * I)
    else:  # bdot
        base = -A * sq / (1.0 + Ba * sq) + b_dot * I
    g = {}
    for k, z in _CHARGE.items():
        g[k] = 1.0 if z == 0 else 10.0 ** (base * z * z)
    return g


def _residuals(x, C_Cu, C_S, C_Q, Ka2, K_ip, activity, T_C, act_kw):
    """Residual vector for u = log10 c_S, v = log10 c_H, w = log10 I."""
    cs = 10.0 ** x[0]
    ch = 10.0 ** x[1]
    I = 10.0 ** x[2]
    g = activity_coefficients(I, activity, T_C, **act_kw)
    Kp = K_ip * g["cu"] * g["so4"] / g["p"]          # L/mol, effective pairing constant
    ccu = C_Cu / (1.0 + Kp * cs)                      # copper balance + pairing
    cp = C_Cu - ccu
    chs = g["h"] * g["so4"] * ch * cs / (g["hso4"] * Ka2)
    r1 = (cs + chs + cp - C_S) / C_S
    r2 = (2.0 * ccu + ch - 2.0 * cs - chs) / C_Q
    I_calc = 0.5 * (4.0 * ccu + ch + chs + 4.0 * cs)  # neutral pair excluded
    r3 = (I_calc - I) / max(I, 1e-12)
    return np.array([r1, r2, r3]), (ccu, cp, chs, cs, ch, I)


_LO = (-40.0, -40.0, -12.0)
_HI = (3.0, 3.0, 3.0)


def _newton(fun, x0, maxit=80, tol=1e-14):
    """Damped Newton on a 3-vector residual with a finite-difference Jacobian.

    Returns (max |residual|, x).  Deterministic and cheap (3x3); no scipy solver
    settings to tune, and it reports the residual actually achieved.
    """
    x = np.clip(np.asarray(x0, float), _LO, _HI)
    f, _ = fun(x)
    best_n = float(np.max(np.abs(f)))
    best_x = x.copy()
    for _ in range(maxit):
        f, _ = fun(x)
        n = float(np.max(np.abs(f)))
        if n < best_n:
            best_n, best_x = n, x.copy()
        if n < tol:
            break
        J = np.empty((3, 3))
        for j in range(3):
            h = 1e-6 * max(1.0, abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (fun(xp)[0] - fun(xm)[0]) / (2.0 * h)
        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -f, rcond=None)[0]
        lam = 1.0
        x_new = x
        for _ in range(24):
            xt = np.clip(x + lam * dx, _LO, _HI)
            ft, _ = fun(xt)
            if float(np.max(np.abs(ft))) < n:
                x_new = xt
                break
            lam *= 0.5
        if np.array_equal(x_new, x):
            break                      # no descent direction left at this tolerance
        x = x_new
    f, _ = fun(x)
    n = float(np.max(np.abs(f)))
    if n < best_n:
        best_n, best_x = n, x.copy()
    return best_n, best_x


def speciate(c_cu_mol_L, c_h2so4_mol_L, T_C=22.0, Ka2=None, K_ip=None,
             activity="davies"):
    """Equilibrium speciation of the CuSO4 + H2SO4 bath.

    Parameters
    ----------
    c_cu_mol_L  : analytical copper concentration (mol/L), e.g. 65 g/L / 63.546 = 1.0229
    c_h2so4_mol_L : analytical sulfuric acid concentration (mol/L), e.g. 8 g/L / 98.079
    T_C         : temperature, C
    Ka2         : HSO4- dissociation constant (mol/L).  None -> ka2_at(T_C) [J].
                  Use a very large value (1e30) for the fully-dissociated limit.
    K_ip        : CuSO4^0 association constant (L/mol).  None -> 10**2.3 [J].
                  Use 0.0 for no ion pairing.
    activity    : "davies" (default), "ideal", "bdot"

    Returns
    -------
    dict with free concentrations in mol/L: c_cu, c_cuso4, c_h, c_hso4, c_so4;
    totals c_cu_total, c_so4_total; I (mol/L); pH = -log10(a_H); gamma (dict);
    converged (bool); residual (max scaled residual); n_starts (starts tried);
    and the model constants actually used (T_C, Ka2, K_ip, activity).

    Mass balances and electroneutrality are satisfied to the reported (scaled)
    residual, < 1e-13 for almost every converged case.  See the module docstring for
    the validity caveat: Davies at I ~ 1-3 mol/L is an extrapolation [J].
    """
    C_Cu = float(c_cu_mol_L)
    C_A = float(c_h2so4_mol_L)
    if C_Cu < 0 or C_A < 0:
        raise ValueError("concentrations must be non-negative")
    C_S = C_Cu + C_A
    if C_S <= 0:
        raise ValueError("bath must contain sulfate")
    C_H = 2.0 * C_A
    C_Q = max(2.0 * C_Cu + C_H, 1e-6)
    Ka2 = ka2_at(T_C) if Ka2 is None else float(Ka2)
    K_ip = 10.0 ** LOG10_K_IP if K_ip is None else float(K_ip)
    if Ka2 <= 0:
        raise ValueError("Ka2 must be positive")
    if K_ip < 0:
        raise ValueError("K_ip must be non-negative")
    act_kw = {}

    fun = lambda x: _residuals(x, C_Cu, C_S, C_Q, Ka2, K_ip, activity, T_C, act_kw)

    # multi-start: sulfate may be almost fully free (high acid) or almost fully paired
    starts = []
    for fs in (0.9, 0.3, 1e-2, 1e-4):
        for ch0 in (max(C_H, 1e-9), max(1e-4, 0.3 * C_H), math.sqrt(Ka2) if Ka2 < 1e6 else 1e-7):
            I0 = max(0.5 * (4.0 * C_Cu * fs + C_H + 4.0 * C_S * fs), 1e-6)
            starts.append([math.log10(max(C_S * fs, 1e-12)),
                           math.log10(max(ch0, 1e-12)),
                           math.log10(I0)])
    best = (np.inf, None)
    for x0 in starts:
        n, x = _newton(fun, x0)
        if n < best[0]:
            best = (n, x)
        if n < 1e-15:
            break
    res, x = best
    _, aux = fun(x)
    ccu, cp, chs, cs, ch, I = (float(v) for v in aux)
    g = activity_coefficients(I, activity, T_C, **act_kw)
    # No proton source in this model (autoprotolysis neglected): pH is undefined at C_acid = 0.
    pH = float("nan") if C_A == 0.0 else -math.log10(max(g["h"] * ch, 1e-300))
    return {
        "c_cu": ccu,
        "c_cuso4": cp,
        "c_h": ch,
        "c_hso4": chs,
        "c_so4": cs,
        "c_cu_total": C_Cu,
        "c_so4_total": C_S,
        "I": I,
        "pH": pH,
        "gamma": g,
        "converged": bool(res < 1e-12),
        "residual": res,
        "n_starts": len(starts),
        "T_C": float(T_C),
        "Ka2": Ka2,
        "K_ip": K_ip,
        "activity": activity,
    }


def dissociated_spec(c_cu_mol_L, c_h2so4_mol_L, T_C=22.0):
    """Fully-dissociated reference state (Ka2 -> inf, K_ip -> 0, ideal activities).

    Exactly c_Cu = c_cu_total, c_SO4 = c_cu_total + c_h2so4, c_H = 2 c_h2so4.
    Used to reproduce the engine's legacy E = 1.669 and Phys.kappa_ne.
    """
    return speciate(c_cu_mol_L, c_h2so4_mol_L, T_C=T_C, Ka2=1e30, K_ip=0.0,
                    activity="ideal")


def conductivity(spec, D=None, T_C=22.0, model="nernst_einstein"):
    """Nernst-Einstein bath conductivity, S/cm.

        kappa = F^2/(R T) * sum_i z_i^2 D_i c_i

    Summed over the CHARGED species present in `spec`.  The CuSO4^0 ion pair is
    neutral and carries NO current, so it never appears in the base sum.

    model="nernst_einstein"  : free charged species only (physical baseline).
    model="ne_with_pairs"    : the same sum PLUS the pair restored to its free ions
                               (Cu2+ + SO4^2- with their own D).  Ion pairing removes
                               carriers; this counterfactual restores them, so it is an
                               UPPER BOUND, not a physical conductivity.  The difference
                               between the two isolates how much kappa is lost to pairing.

    D : optional dict overriding the 25 C ionic diffusivities, keys
        "cu", "h", "so4", "hso4" [cm2/s].  All are Stokes-Einstein scaled to T_C with
        platesim.feature.engine.params.stokes_einstein_factor (the same T scaling the
        engine uses), so T_C=25 reproduces the tabulated values exactly.

    The default D table is [FT, CRC-type, not re-verified]; D_Cu2+ = 7.14e-6 is the
    INFINITE-DILUTION value, not the engine's bath-effective D_cu25 = 2.65e-6.
    """
    if model not in ("nernst_einstein", "ne_with_pairs"):
        raise ValueError("model must be 'nernst_einstein' or 'ne_with_pairs'")
    Dm = dict(D_ION_25)
    if D:
        unknown = set(D) - set(Dm)
        if unknown:
            raise ValueError(f"unknown diffusivity keys {sorted(unknown)}")
        Dm.update({k: float(v) for k, v in D.items()})
    fD = stokes_einstein_factor(T_C)          # D(T)/D(25 C), same helper as the engine
    T_K = float(T_C) + 273.15
    f_RT = F / (R_GAS * T_K)
    c_cm3 = {k: spec[f"c_{k}"] * 1e-3 for k in ("cu", "h", "so4", "hso4")}
    s = (4.0 * Dm["cu"] * c_cm3["cu"]
         + Dm["h"] * c_cm3["h"]
         + 4.0 * Dm["so4"] * c_cm3["so4"]
         + Dm["hso4"] * c_cm3["hso4"])
    if model == "ne_with_pairs":
        cp = spec["c_cuso4"] * 1e-3
        s += 4.0 * Dm["cu"] * cp + 4.0 * Dm["so4"] * cp
    return float(F * f_RT * fD * s)


def conductivity_summary(spec, D=None, T_C=None, kappa_placeholder=KAPPA_22):
    """Both conductivity models plus their ratio to the engine placeholder 0.065 S/cm."""
    T = spec["T_C"] if T_C is None else T_C
    k_free = conductivity(spec, D=D, T_C=T, model="nernst_einstein")
    k_rest = conductivity(spec, D=D, T_C=T, model="ne_with_pairs")
    return {
        "T_C": T,
        "kappa_ne": k_free,
        "kappa_ne_with_pairs": k_rest,
        "kappa_placeholder": kappa_placeholder,
        "ratio_ne": k_free / kappa_placeholder,
        "ratio_ne_with_pairs": k_rest / kappa_placeholder,
    }


def migration_enhancement_speciated(spec, include_pair=False, D_pair=None):
    """1-D limiting-current migration enhancement E for the speciated bath.

    Generalises platesim.feature.engine.params.migration_enhancement: the inert ions
    are now every species except Cu2+, i.e. H+, HSO4- and SO4^2- with their FREE bulk
    concentrations, and the speciation is FROZEN through the diffusion layer.

    Inert ions (zero flux) obey c_i(phi) = c_i,b exp(-z_i phi); electroneutrality then
    fixes the free copper profile

        c_Cu(phi) = [ c_HS,b exp(phi) + 2 c_S,b exp(2 phi) - c_H,b exp(-phi) ] / 2 .

    phi_L is the root c_Cu(phi_L) = 0 (copper fully depleted).  With
    dphi = -F/(RT) dPhi, the standard 1-D balance gives

        E = 1 + (2/c_Cu,b) int_{phi_L}^{0} c_Cu(phi) dphi
          = 1 + [ c_HS (1 - e^{phi_L}) + c_S (1 - e^{2 phi_L}) + c_H (1 - e^{-phi_L}) ] / c_Cu,b

    which is exact (no quadrature).  Fully dissociated, no pairing: E = 1.6690 for the
    default bath, identical to params.migration_enhancement.

    include_pair : the neutral CuSO4^0 pair carries no current, but under frozen
        speciation it tracks the free copper (c_p = r c_Cu with r = c_p,b/c_Cu,b) and
        adds a diffusion-only stream.  The flux ratio relative to the pair-free
        diffusion reference is then
            E_pair = 1 + 2 int c_Cu dphi / ((1 + r*delta) c_Cu,b),  delta = D_pair/D_Cu.
        D_pair defaults to D_Cu2+ (delta = 1) [J, unmeasured].
    """
    c_cu = float(spec["c_cu"])
    if c_cu <= 0.0:
        raise ValueError("no free Cu2+ to deplete")
    c_p = float(spec["c_cuso4"])
    c_h = float(spec["c_h"])
    c_hs = float(spec["c_hso4"])
    c_s = float(spec["c_so4"])
    q0 = 2.0 * c_cu + c_h - 2.0 * c_s - c_hs
    if abs(q0) > 1e-6 * max(1.0, c_cu):
        raise ValueError(f"bulk not electroneutral: {q0:.3e} mol/L")

    def ccu(phi):
        return 0.5 * (c_hs * math.exp(phi) + 2.0 * c_s * math.exp(2.0 * phi)
                      - c_h * math.exp(-phi))

    phi_L = brentq(ccu, -60.0, 0.0, xtol=1e-15, rtol=1e-15, maxiter=200)
    # 1 - e^{a phi_L} written as -expm1(a phi_L) (stable for large negative phi_L)
    integral = 0.5 * (-c_hs * math.expm1(phi_L)
                      - c_s * math.expm1(2.0 * phi_L)
                      - c_h * math.expm1(-phi_L))
    if include_pair:
        delta = 1.0 if D_pair is None else float(D_pair) / D_ION_25["cu"]
        denom = c_cu * (1.0 + c_p / c_cu * delta)
    else:
        denom = c_cu
    return float(1.0 + 2.0 * integral / denom)


def fully_dissociated_enhancement(c_cu_mol_L, c_h2so4_mol_L):
    """E from params.migration_enhancement' equivalent, built from speciation terms."""
    return migration_enhancement_speciated(
        dissociated_spec(c_cu_mol_L, c_h2so4_mol_L), include_pair=False)


# ---------------------------------------------------------------- CLI
_M_CU = 63.546
_M_H2SO4 = 98.079


def _row(T_C, acid_g_L, c_cu_mol_L, activity="davies"):
    """One table row: speciation + both kappas + both E values."""
    spec = speciate(c_cu_mol_L, acid_g_L / _M_H2SO4, T_C=T_C, activity=activity)
    ks = conductivity_summary(spec)
    return {
        "T_C": T_C,
        "acid_g_L": acid_g_L,
        "c_cu": spec["c_cu"],
        "c_p": spec["c_cuso4"],
        "c_h": spec["c_h"],
        "c_hs": spec["c_hso4"],
        "c_s": spec["c_so4"],
        "I": spec["I"],
        "pH": spec["pH"],
        "kappa_ne": ks["kappa_ne"],
        "kappa_restored": ks["kappa_ne_with_pairs"],
        "ratio": ks["ratio_ne"],
        "E": migration_enhancement_speciated(spec),
        "E_pair": migration_enhancement_speciated(spec, include_pair=True),
        "E_diss": fully_dissociated_enhancement(c_cu_mol_L, acid_g_L / _M_H2SO4),
        "converged": spec["converged"],
        "residual": spec["residual"],
    }


_HDR = ("  T[C] H2SO4  Cu2+    CuSO4   H+      HSO4-   SO4^2-   I      pH    "
        "kappa_NE  kappa_pairs  k/0.065  E_spec  E_pair  E_diss")


def _fmt(r):
    ph = " n/a " if math.isnan(r["pH"]) else f"{r['pH']:5.2f}"
    return (f"{r['T_C']:5.0f} {r['acid_g_L']:5.0f} "
            f"{r['c_cu']:7.4f} {r['c_p']:7.4f} {r['c_h']:7.4f} {r['c_hs']:7.4f} "
            f"{r['c_s']:7.4f} {r['I']:6.3f} {ph} "
            f"{r['kappa_ne']:8.4f} {r['kappa_restored']:11.4f} "
            f"{r['ratio']:8.2f} {r['E']:7.3f} {r['E_pair']:7.3f} {r['E_diss']:7.3f}"
            f"{'' if r['converged'] else '  NOT CONVERGED'}")


def main():
    """python -m platesim.core.speciation - bath speciation / conductivity table."""
    c_cu = 65.0 / _M_CU
    print("platesim.core.speciation - acid copper sulfate bath, trend-level model")
    print(f"  c_Cu = 65 g/L = {c_cu:.4f} mol/L;  Ka2(25 C) = {KA2_25:.1e} [J], "
          f"log10 K_ip = {LOG10_K_IP} [J];  kappa placeholder = {KAPPA_22} S/cm")
    print("  concentrations mol/L; kappa S/cm; Davies above I ~ 0.5 mol/L is an "
          "EXTRAPOLATION [J]")
    print()
    print("default bath (65 g/L Cu), 22 and 40 C:")
    print(_HDR)
    for T in (22.0, 40.0):
        print(_fmt(_row(T, 8.0, c_cu)))
    print()
    print("H2SO4 series at 22 C:")
    print(_HDR)
    for acid in (0.0, 8.0, 50.0, 180.0):
        print(_fmt(_row(22.0, acid, c_cu)))
    print()
    r = _row(22.0, 8.0, c_cu)
    d = dissociated_spec(c_cu, 8.0 / _M_H2SO4)
    k_diss = conductivity(d, T_C=22.0)
    print(f"22 C, default bath: fully dissociated Nernst-Einstein kappa = {k_diss:.4f} S/cm "
          f"({k_diss / KAPPA_22:.1f}x the {KAPPA_22} placeholder)")
    print(f"                    speciated (Davies)        kappa = {r['kappa_ne']:.4f} S/cm "
          f"({r['ratio']:.2f}x the placeholder)")
    print("  -> the placeholder sits between the two; ion pairing + HSO4- formation remove "
          "roughly the whole 4-5x free-ion gap (trend-level, not a prediction).")


if __name__ == "__main__":
    main()
