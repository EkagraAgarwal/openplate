"""Side reactions on the Cu electrodes (plan item D5b): HER, O2 reduction, Cu anode.

Standalone companion to :mod:`platesim.efficiency.model`.  model.py lumps the parasitic
currents into ``eta_true = 1 - j_O2/j - j_H2/j``: O2 is a fixed transport-limited flux and
H2 appears only above the Cu limiting current.  Here the same two reactions are written as
electrode kinetics sharing ONE electrode potential with the Cu deposition, plus the anodic
half-cell, so the split can be computed instead of assumed.

Nothing here is wired into model.py.  Importing this module changes no existing behaviour:
the calibration path stays bit-identical (the module is never imported by model.py).

Cathode: three reductions share one potential E (V vs SHE)
----------------------------------------------------------
    Cu2+ + 2e-  -> Cu           i_Cu = i0_Cu 10^(-(E - E_Cu )/b_Cu),  b_Cu = 2.303 RT/(alpha_Cu F)
    H+   +  e-  -> 1/2 H2       i_H  = i0_H  10^(-(E - E_H2 )/b_H ),  b_H  = 2.303 RT/(alpha_H  F)
    1/2 O2 + 2H+ + 2e- -> H2O   i_O2 = i_lim,O2                      (transport limited)

    i_lim,H  = F   D_H  c_H  / delta                       [FT] Nernst layer, pure diffusion
    i_lim,O2 = 4 F D_O2 c_O2 / delta                       [FT] 4-electron reduction
    i_H,net  = i_k i_lim / (i_k + i_lim)                   [FT] kinetic + transport in series
    E_H2 - E_Cu = (0 - 0.0592 pH) - (0.337 + 0.0296 log10 a_Cu)   at 25 C [FT],
                  both Nernst slopes T-scaled by T/298.15.

The HER overpotential at a given Cu cathode overpotential eta_Cu follows from that offset:

    eta_H = E - E_H2 = eta_Cu - (E_H2 - E_Cu) = eta_Cu + 0.0592 pH + 0.337

so at a typical acid-Cu deposition potential (eta_Cu ~ -0.36 V at 15 mA/cm2, pH 1.7) the
cathode sits ~+0.08 V ABOVE the H+/H2 equilibrium: HER is essentially switched off, which
is why the measured acid-Cu efficiency is 95-100 % [FT/AB report 06].

Anode: Cu dissolution + optional salt film
------------------------------------------
    Cu -> Cu2+ + 2e-            i_a = i0_a 10^(eta_a/b_a),  b_a = 2.303 RT/(alpha_a F)
    i_lim,anode = 2 F D_Cu (c_sat - c_bulk) / delta        [FT] CuSO4 saturation limit
    c_surf = c_bulk + j delta/(2 F D_Cu)                   [FT] steady 1-D layer
    eta_conc = (RT/(2F)) ln(c_surf/c_bulk), pinned at c_sat
    film forms when j > i_lim,anode; above it the surface cannot exceed c_sat and the
    excess potential is carried by the film: eta_film = R_film (j - i_lim)   [J form].

Units: cm, s, A, V, mol; concentrations mol/cm3 unless a name ends in _mol_L.
Tags: [FT] textbook formula, [J] judgement (order of magnitude, NOT verified here), [C] calc.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

from ..bubbles.physics import D_H2_WATER_M2_S, K_H_H2_MOL_L_ATM, diffusive_growth_rate_um_s
from ..core.bath import Bath
from ..core.constants import F, R_GAS
from ..feature.engine.params import Params, stokes_einstein_factor

# --------------------------------------------------------------------------- constants
T_REF_K = 298.15
E0_CU_CU2_V = 0.337             # [FT] standard potential Cu2+/Cu vs SHE at 25 C
NERNST_CU_V_DEC = 0.0296        # [FT] 2.303 RT/(2F) at 25 C, per decade of a_Cu2+
NERNST_H_V_PH = 0.0592          # [FT] 2.303 RT/F at 25 C, per pH unit
I0_H_REF_A_CM2 = 1.0e-7         # [J] HER exchange current on Cu at pH 0, 22 C: literature order,
                                #     NOT verified (published Cu values span 1e-7 .. 1e-3 A/cm2)
ALPHA_H = 0.5                   # [J] HER transfer coefficient
ALPHA_CU = 0.5                  # [J] matches Bath.alpha_c
N_E_O2 = 4                      # [FT] O2 + 4H+ + 4e- -> 2 H2O in acid
C_SAT_CUSO4_MOL_L = 1.4         # [J] CuSO4 saturation at the anode surface, 22 C, NOT verified
R_FILM_DEFAULT_OHM_CM2 = 0.1    # [J] salt-film areal resistance, order of model.prior("k_film")
DELTA_DEFAULT_UM = 65.0         # [J] Nernst layer default (model delta0 prior 80 um at 20 % stir)
EA_I0_H_J_MOL = 35e3            # [J] same activation energy as Bath's Cu i0
D_H25_CM2_S = Params.D_H25      # cm2/s, H+ at 25 C (feature.engine.params)
A_CU_DEFAULT = 65.0 / 63.546    # [M] bath Cu2+ activity, mol/L with gamma = 1 (65 g/L bottle)
C_H2_BULK_DEFAULT = 0.0         # mol/cm3, dissolved H2 far from the cathode (fresh bath)


def _T_K(T_C: float) -> float:
    return float(T_C) + 273.15


def _nernst_T(T_C: float) -> float:
    """Nernst-slope T factor T/298.15 [FT; E0 kept at 25 C, dE0/dT dropped - J]."""
    return _T_K(T_C) / T_REF_K


def _delta_cm(delta_um: float) -> float:
    d = float(delta_um) * 1e-4
    if d <= 0.0:
        raise ValueError("delta_um must be positive")
    return d


def tafel_slope_V(alpha: float, T_C: float = 22.0) -> float:
    """Cathodic/anodic Tafel slope 2.303 RT/(alpha F), V per decade [FT]."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return 2.303 * R_GAS * _T_K(T_C) / (float(alpha) * F)


def her_equilibrium_V(pH: float, T_C: float = 22.0) -> float:
    """H+/H2 equilibrium potential vs SHE, -0.0592 pH at 25 C, T-scaled [FT]."""
    return -NERNST_H_V_PH * _nernst_T(T_C) * float(pH)


def cu_equilibrium_V(a_Cu: float = A_CU_DEFAULT, T_C: float = 22.0) -> float:
    """Cu2+/Cu equilibrium potential vs SHE, 0.337 + 0.0296 log10 a_Cu at 25 C [FT]."""
    a = max(float(a_Cu), 1e-30)
    return E0_CU_CU2_V + NERNST_CU_V_DEC * _nernst_T(T_C) * math.log10(a)


def her_overpotential(eta_Cu_V: float, pH: float, a_Cu: float = A_CU_DEFAULT,
                      T_C: float = 22.0) -> float:
    """HER overpotential eta_H = E - E_H2 at a Cu cathode potential E = E_Cu + eta_Cu.

    Negative = cathodic H2 evolution.  At pH 1.7 and eta_Cu = -0.36 V this returns about
    +0.08 V (HER strongly suppressed), because E_H2 sits ~0.44 V below E_Cu [C].
    """
    return float(eta_Cu_V) - (her_equilibrium_V(pH, T_C) - cu_equilibrium_V(a_Cu, T_C))


def cu_overpotential_V(j_A_cm2: float, i0_A_cm2: float | None = None, alpha: float | None = None,
                       T_C: float = 22.0, bath: Bath | None = None) -> float:
    """Cathodic Cu Tafel overpotential (V, negative), same convention as model._eta_c_V.

    Default kinetics are the suppressed-face set of :class:`platesim.core.bath.Bath`
    (i0 = 1.3e-5 A/cm2 at 22 C, alpha_c = 0.5) [FT Braun Table I].
    """
    b = Bath() if bath is None else bath
    if b.T_C != float(T_C):
        b = b.at(float(T_C))
    i0 = b.i0_cu_supp if i0_A_cm2 is None else float(i0_A_cm2)
    a = b.alpha_c if alpha is None else float(alpha)
    return -tafel_slope_V(a, T_C) * math.log10(max(float(j_A_cm2), 1e-15) / max(i0, 1e-15))


def her_current(eta_V: float, pH: float, T_C: float = 22.0, i0_H: float = I0_H_REF_A_CM2,
                alpha: float = ALPHA_H, c_H_mol_L: float | None = None,
                D_H: float | None = None, delta_um: float = DELTA_DEFAULT_UM,
                i_lim_A_cm2: float | None = None, use_transport: bool = True,
                Ea_J_mol: float = EA_I0_H_J_MOL) -> float:
    """Cathodic H+ reduction current on Cu, A/cm2, kinetic + transport in series.

    Tafel branch
        i_k = i0_H(T) 10^(-pH) 10^(-eta/b),   b = 2.303 RT/(alpha F)
    with i0_H(T) = i0_H exp[-Ea/R (1/T - 1/295.15)] [J, same Ea as the Cu i0 in Bath] and the
    a_H+ factor taken as 10^(-pH) (pH is -log10 a_H by definition).
    Transport limit
        i_lim = F D_H c_H / delta   [FT] (pure diffusion; H+ migration would raise it [J])
    Series (mixed kinetic/transport control)
        i = i_k i_lim / (i_k + i_lim)   [FT]

    eta_V is measured against the H+/H2 equilibrium; use :func:`her_overpotential` to get it
    from a Cu cathode overpotential.  `c_H_mol_L` overrides the free H+ concentration
    (e.g. from :func:`platesim.core.speciation.speciate`) while pH still sets the activity
    in the exchange current.  `i_lim_A_cm2` overrides the transport limit directly, and
    `use_transport=False` returns the pure Tafel branch.
    """
    T_K = _T_K(T_C)
    b = tafel_slope_V(alpha, T_C)
    i0 = (float(i0_H) * math.exp(-float(Ea_J_mol) / R_GAS * (1.0 / T_K - 1.0 / 295.15))
          * 10.0 ** (-float(pH)))
    # 10^(-eta/b) with the exponent clamped so that extreme overpotentials stay finite
    i_k = i0 * math.exp(min(-float(eta_V) / b * math.log(10.0), 700.0))
    if not use_transport:
        return i_k
    if i_lim_A_cm2 is None:
        c_h = 10.0 ** (-float(pH)) if c_H_mol_L is None else float(c_H_mol_L)
        D = (D_H25_CM2_S * stokes_einstein_factor(T_C)) if D_H is None else float(D_H)
        i_lim = F * D * c_h * 1e-3 / _delta_cm(delta_um)
    else:
        i_lim = float(i_lim_A_cm2)
    if i_lim <= 0.0:
        return 0.0
    if math.isinf(i_lim):
        return i_k
    return i_k * i_lim / (i_k + i_lim)


def surface_ph(pH_bulk: float, j_H_A_cm2: float, delta_um: float = DELTA_DEFAULT_UM,
               D_H: float | None = None, T_C: float = 22.0) -> float:
    """Local cathode pH from a steady 1-D H+ mass balance [FT, one shot].

    c_H,surf = c_H,bulk - j_H delta/(F D_H)  (H+ consumed at the wall), pH = -log10 c_H.
    Water autoprotolysis is not modelled, so the result is clipped at pH 14 [J].  This is
    the value to feed back into :func:`her_current`; the two should be iterated for a
    self-consistent kinetic/transport solution (one pass is enough at the current densities
    of the planar DOE, where < 5 % of the bulk H+ is consumed).
    """
    D = (D_H25_CM2_S * stokes_einstein_factor(T_C)) if D_H is None else float(D_H)
    c_b = 10.0 ** (-float(pH_bulk)) * 1e-3                      # mol/cm3
    c_s = c_b - max(float(j_H_A_cm2), 0.0) * _delta_cm(delta_um) / (F * D)
    return min(-math.log10(max(c_s, 1e-30) / 1e-3), 14.0)


def orr_current(delta_um: float, T_C: float = 22.0, c_O2_mol_cm3: float | None = None,
                D_O2: float | None = None, n_e: int = N_E_O2, bath: Bath | None = None,
                sechenov_k_L_mol: float = 0.0, c_salt_mol_L: float | None = None,
                sc_delta_scaling: bool = False) -> float:
    """Transport-limited O2 reduction current, A/cm2: i_lim = n F D c / delta  [FT].

    Defaults are the air-saturated bath of :class:`platesim.core.bath.Bath`: c_O2 = 2.6e-7
    mol/cm3 at 25 C (T-corrected) and D_O2 = 2.1e-5 cm2/s, so air saturation gives
    ~0.32 mA/cm2 at delta = 65 um.  This is the plain diffusion formula, i.e. the SAME
    diffusion-layer thickness for O2 and for Cu2+.

    `sc_delta_scaling=True` instead scales the layer by (D_O2/D_Cu)^(1/3) (flux-similarity
    form used by Bath.j_lim_o2, [J] for this bath); that is ~2x SMALLER here because
    D_O2 >> D_Cu, and reproduces Bath.j_lim_o2 exactly.

    Sechenov salting-out is optional and OFF by default (sechenov_k_L_mol = 0):
    c = c_water 10^(-k_s c_salt) [FT form, k_s NOT verified].  c_salt defaults to the
    bath copper concentration.
    """
    b = Bath().at(float(T_C)) if bath is None else bath
    c = b.c_o2 if c_O2_mol_cm3 is None else float(c_O2_mol_cm3)
    D = b.D_o2 if D_O2 is None else float(D_O2)
    if sechenov_k_L_mol:
        c_salt = b.c_cu * 1e3 if c_salt_mol_L is None else float(c_salt_mol_L)
        c *= 10.0 ** (-float(sechenov_k_L_mol) * c_salt)
    d = _delta_cm(delta_um)
    if sc_delta_scaling:
        d *= (D / b.D_cu) ** (1.0 / 3.0)
    return float(n_e) * F * D * c / d


def faradaic_efficiency(j_total: float, eta_Cu: float, delta_um: float, pH: float,
                        T_C: float = 22.0, a_Cu: float = A_CU_DEFAULT,
                        alpha_Cu: float = ALPHA_CU, alpha_H: float = ALPHA_H,
                        i0_H: float = I0_H_REF_A_CM2, c_H_mol_L: float | None = None,
                        D_H: float | None = None, c_O2_mol_cm3: float | None = None,
                        D_O2: float | None = None, use_her: bool = True,
                        use_orr: bool = True, bath: Bath | None = None,
                        orr_sc_delta_scaling: bool = False,
                        c_H2_bulk_mol_cm3: float = C_H2_BULK_DEFAULT,
                        k_H_H2_mol_L_atm: float = K_H_H2_MOL_L_ATM,
                        bubble_R_um: float = 5.0) -> dict:
    """Cu share of a fixed total current with HER and O2 reduction on the same potential.

    All three reactions sit at one potential E = E_Cu + eta.  The Cu branch is pinned by the
    caller's (j_total, eta_Cu) pair, i.e. i0_Cu = j_total 10^(eta_Cu/b_Cu), so eta_Cu is the
    Cu overpotential this current would need with NO side reactions (model._eta_c_V or
    :func:`cu_overpotential_V` give it).  Side reactions then take their share and the
    potential relaxes towards E_Cu; the solution is the root of

        i_Cu(eta) + i_H(eta) + i_ORR - j_total = 0        (brentq, J)

    with i_H from :func:`her_current` at eta_H = eta - (E_H2 - E_Cu) and i_ORR from
    :func:`orr_current` (transport limited, potential independent; plain delta by default,
    `orr_sc_delta_scaling=True` reproduces the Bath.j_lim_o2 thickness scaling).

    H2 at the wall (feeds :mod:`platesim.bubbles`): the steady 1-D layer balances the HER
    flux against diffusion away from the electrode,

        c_H2,wall = c_bulk + j_H2 delta / (2 F D_H2)      [FT],
        S = c_H2,wall / c_H2,sat,  supersat = S - 1,

    with D_H2 = 5e-9 m2/s [J] and c_H2,sat from Henry's law (7.8e-4 mol/L/atm [J]).  The
    Epstein-Plesset growth rate at `bubble_R_um` is returned as a convenience.

    Returns a dict; `efficiency` is j_Cu/j_total in [0, 1].  When the ORR flux alone exceeds
    j_total (thin delta, low j) no root exists in the model: efficiency is returned as 0.0
    with converged=False and status="side_currents_exceed_total".
    """
    j_t = float(j_total)
    if j_t <= 0.0:
        raise ValueError("j_total must be positive")
    b_cu = tafel_slope_V(alpha_Cu, T_C)
    i0_cu = j_t * 10.0 ** (float(eta_Cu) / b_cu)          # i_Cu(eta_Cu) = j_total
    j_orr = (orr_current(delta_um, T_C, c_O2_mol_cm3, D_O2, bath=bath,
                         sc_delta_scaling=orr_sc_delta_scaling)
             if use_orr else 0.0)

    def i_her(eta: float) -> float:
        if not use_her:
            return 0.0
        return her_current(her_overpotential(eta, pH, a_Cu, T_C), pH, T_C, i0_H, alpha_H,
                           c_H_mol_L, D_H, delta_um)

    def excess(eta: float) -> float:
        return i0_cu * 10.0 ** (-float(eta) / b_cu) + i_her(eta) + j_orr - j_t

    # bracket: normally the root is at a LESS negative potential than eta_Cu (side reactions
    # carry part of the current); only negative side currents would push it the other way.
    a = b = float(eta_Cu)
    converged = True
    status = "solved"
    if excess(a) >= 0.0:
        while excess(b) >= 0.0 and b < 3.0:
            b += 0.05
        if excess(b) >= 0.0:
            converged, status = False, "side_currents_exceed_total"
    else:
        while excess(a) <= 0.0 and a > -3.0:
            a -= 0.05
        if excess(a) <= 0.0:
            converged, status = False, "no_root"

    if converged:
        eta_sol = float(brentq(excess, a, b, xtol=1e-12, rtol=1e-14, maxiter=200))
        i_cu = i0_cu * 10.0 ** (-eta_sol / b_cu)
        i_h = i_her(eta_sol)
    else:
        eta_sol = float("nan")
        i_cu = 0.0
        i_h = i_her(float(eta_Cu))

    d_cm = _delta_cm(delta_um)
    rate = i_h / (2.0 * F)                                 # mol/cm2/s of H2
    d_h2 = D_H2_WATER_M2_S * 1e4                           # m2/s -> cm2/s
    c_wall = float(c_H2_bulk_mol_cm3) + rate * d_cm / d_h2
    c_sat = float(k_H_H2_mol_L_atm) * 1e-3                 # mol/L/atm -> mol/cm3 at 1 atm
    S = c_wall / c_sat if c_sat > 0.0 else float("inf")

    return {
        "j_total_A_cm2": j_t,
        "j_Cu_A_cm2": i_cu if converged else 0.0,
        "j_H2_A_cm2": i_h,
        "j_ORR_A_cm2": j_orr,
        "efficiency": max(0.0, min(1.0, (i_cu / j_t) if converged else 0.0)),
        "converged": converged,
        "status": status,
        "eta_Cu_V": eta_sol,
        "eta_Cu_given_V": float(eta_Cu),
        "eta_H_V": her_overpotential(eta_sol, pH, a_Cu, T_C) if converged else float("nan"),
        "E_V": cu_equilibrium_V(a_Cu, T_C) + eta_sol,
        "b_Cu_V_dec": b_cu,
        "b_H_V_dec": tafel_slope_V(alpha_H, T_C),
        "h2_rate_mol_cm2_s": rate,
        "c_H2_wall_mol_cm3": c_wall,
        "c_H2_sat_mol_cm3": c_sat,
        "S_H2": S,
        "supersat_H2": S - 1.0,
        "bubble_growth_um_s": diffusive_growth_rate_um_s(
            bubble_R_um, supersat=S - 1.0, D_m2_s=D_H2_WATER_M2_S, T_C=T_C),
        "delta_um": float(delta_um),
        "pH": float(pH),
        "T_C": float(T_C),
    }


def anode_potential(j_anode: float, T_C: float = 22.0, film: bool = False,
                    delta_um: float = DELTA_DEFAULT_UM,
                    c_sat_mol_L: float = C_SAT_CUSO4_MOL_L,
                    c_bulk_mol_L: float | None = None, D_Cu: float | None = None,
                    i0_a: float | None = None, alpha_a: float | None = None,
                    r_film_ohm_cm2: float = R_FILM_DEFAULT_OHM_CM2,
                    bath: Bath | None = None,
                    use_migration_enhancement: bool = False) -> dict:
    """Cu dissolution at the anode: Tafel kinetics + concentration/salt-film overpotential.

        eta_kin  = b_a log10(j / i0_a),  b_a = 2.303 RT/(alpha_a F)
        i_lim    = 2 F D_Cu (c_sat - c_bulk) / delta          [FT, CuSO4 saturation limit]
        c_surf   = c_bulk + j delta / (2 F D_Cu)              [FT, steady 1-D layer]
        eta_conc = (RT/2F) ln(c_surf/c_bulk), pinned at c_sat
        eta_film = r_film (j - i_lim)  only when film=True and j > i_lim   [J form]

    c_sat = 1.4 mol/L at 22 C [J, NOT verified] and c_bulk is the bath copper concentration
    (1.0229 mol/L), so at delta = 65 um i_lim ~ 27 mA/cm2 (same order as the 21.5 mA/cm2
    film threshold of model.theta["j_a_th"] [FT Dow CuPulse]).  D_Cu is the plain bath
    diffusivity by default; `use_migration_enhancement=True` multiplies it by Bath.E_mig
    (1.67), which raises i_lim to ~46 mA/cm2 - a modelling choice, not a measurement [J].

    Returns eta_V (total), the eta_kin/eta_conc/eta_film parts, i_lim_anode_A_cm2,
    film_forms (j > i_lim, reported whether or not `film` is enabled) and c_surf_mol_L.
    """
    b = Bath().at(float(T_C)) if bath is None else bath
    j = max(float(j_anode), 0.0)
    alpha = b.alpha_a if alpha_a is None else float(alpha_a)
    b_a = tafel_slope_V(alpha, T_C)
    i0 = b.i0_cu if i0_a is None else float(i0_a)
    eta_kin = b_a * math.log10(max(j, 1e-15) / max(i0, 1e-15))

    c_bulk = b.c_cu * 1e3 if c_bulk_mol_L is None else float(c_bulk_mol_L)   # mol/L
    D = b.D_cu_eff if use_migration_enhancement else b.D_cu
    if D_Cu is not None:
        D = float(D_Cu)
    d_cm = _delta_cm(delta_um)
    i_lim = 2.0 * F * D * (float(c_sat_mol_L) - c_bulk) * 1e-3 / d_cm
    film_forms = bool(j > i_lim) if i_lim > 0.0 else True

    c_surf = c_bulk + j * d_cm / (2.0 * F * D) * 1e3                         # mol/L
    c_surf_eff = min(c_surf, float(c_sat_mol_L)) if film_forms else c_surf
    eta_conc = (R_GAS * _T_K(T_C) / (2.0 * F)) * math.log(
        max(c_surf_eff, 1e-30) / max(c_bulk, 1e-30))
    eta_film = (max(0.0, j - i_lim) * float(r_film_ohm_cm2)
                if (film and film_forms) else 0.0)

    return {
        "eta_V": eta_kin + eta_conc + eta_film,
        "eta_kin_V": eta_kin,
        "eta_conc_V": eta_conc,
        "eta_film_V": eta_film,
        "film_forms": film_forms,
        "film_applied": bool(film and film_forms),
        "i_lim_anode_A_cm2": i_lim,
        "j_anode_A_cm2": j,
        "c_surf_mol_L": c_surf,
        "c_sat_mol_L": float(c_sat_mol_L),
        "c_bulk_mol_L": c_bulk,
        "b_a_V_dec": b_a,
        "i0_a_A_cm2": i0,
        "D_Cu_cm2_s": D,
        "delta_um": float(delta_um),
        "T_C": float(T_C),
    }


__all__ = [
    "ALPHA_H", "C_SAT_CUSO4_MOL_L", "E0_CU_CU2_V", "I0_H_REF_A_CM2", "NERNST_CU_V_DEC",
    "NERNST_H_V_PH", "anode_potential", "cu_equilibrium_V", "cu_overpotential_V",
    "faradaic_efficiency", "her_current", "her_equilibrium_V", "her_overpotential",
    "orr_current", "surface_ph", "tafel_slope_V",
]
