"""Bubble physics correlations (plan item E2a): rise, detachment, growth, shading.

Scope: the building blocks only. Nothing here is coupled to the plating engine; the tracker in
:mod:`platesim.bubbles.tracker` consumes these functions.

Why it matters: H2 from the side reaction (or air left in a trench after pre-wet) attaches to the
electrode, shades it and leaves pits. Everything below is an order-of-magnitude model for that.

Conventions
- SI inside: radius in m (arguments ending ``_um`` are um, as in the rest of platesim), velocity in
  m/s unless the name says ``_mm_s`` or ``_cm_s``, pressure in Pa, concentrations in mol/m3,
  diffusivity in m2/s, temperature in C, contact angle in degrees.
- Tagged constants: ``[FT]`` textbook formula, ``[J]`` judgement (order of magnitude, not verified).
  Only author/year combinations that are certain are cited.

Reference frame: a gas bubble in a liquid of density rho_l under gravity g, density difference
drho = rho_l - rho_gas (the gas density is neglected in the buoyancy term, it is ~1e-4 of the liquid).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.constants import R_GAS, celsius_to_K

# ----------------------------------------------------------------------------- constants
G_ACCEL_M_S2 = 9.81                 # [FT] standard gravity, m/s2

# Default liquid/gas pair: H2 bubble in water at 25 C.
RHO_LIQUID_KG_M3 = 997.0            # [FT] water at 25 C, kg/m3
MU_LIQUID_PA_S = 1.0e-3             # [FT] water at 25 C, Pa s
DRHO_KG_M3 = 997.0                  # [C] rho_water - rho_H2(1 atm, 25 C) = 997 - 0.089
SIGMA_WATER_N_M = 0.072             # [FT] water surface tension at 25 C, N/m
D_H2_WATER_M2_S = 5.0e-9            # [J] H2 in water, 25 C: 4-6e-9 in the literature; order of magnitude
K_H_H2_MOL_L_ATM = 7.8e-4           # [J] H2 in water at 25 C, mol/(L atm); ~7.7e-4 from tabulated
                                    #     Henry data, not verified here
FRITZ_PREFACTOR_DEG = 0.0104        # [FT] Fritz (1935) prefactor, valid with theta in DEGREES
P_ATM_PA = 101325.0                 # [FT] standard atmosphere


def rise_velocity(R_um, drho=DRHO_KG_M3, mu=MU_LIQUID_PA_S, rho_l=RHO_LIQUID_KG_M3,
                  g=G_ACCEL_M_S2):
    """Terminal rise velocity of a single small sphere: rigid (Stokes) and mobile (HR) limits.

    Stokes rigid sphere [FT]::

        U = 2/9 * drho * g * R**2 / mu

    Hadamard-Rybczynski mobile interface [FT], gas-in-liquid limit (mu_gas -> 0)::

        U = 1/3 * drho * g * R**2 / mu

    Both scale as R**2 and their ratio is exactly 2/3 (the mobile interface sheds momentum into the
    internal circulation, so it rises 1.5x faster). A real H2 bubble in water is close to the mobile
    limit as long as it is not loaded with surfactant.

    The return dict carries the bubble Reynolds number ``Re = rho_l * U * (2 R) / mu`` evaluated with
    the Hadamard velocity (the physical limit for a gas bubble) and ``re_gt_1``, the flag the caller
    must check: both formulas are creeping-flow results and say nothing above Re ~ 1.

    Returns keys: ``R_um``, ``U_stokes_m_s``, ``U_hr_m_s``, ``U_stokes_mm_s``, ``U_hr_mm_s``,
    ``U_hr_cm_s``, ``reynolds``, ``re_gt_1``, ``warning``.
    """
    R = float(R_um) * 1e-6
    if R <= 0.0:
        raise ValueError("R_um must be positive")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    k = drho * g * R * R / mu
    u_stokes = (2.0 / 9.0) * k
    u_hr = (1.0 / 3.0) * k
    re = rho_l * u_hr * (2.0 * R) / mu
    warn = None
    if re > 1.0:
        warn = (f"bubble Reynolds number {re:.3g} > 1: Stokes/Hadamard-Rybczynski are not valid, "
                "the drag is no longer linear in U (use a Schiller-Naumann/Clift-Grace-Weber "
                "correction or an empirical terminal velocity)")
    return {
        "R_um": float(R_um),
        "U_stokes_m_s": u_stokes,
        "U_hr_m_s": u_hr,
        "U_stokes_mm_s": u_stokes * 1e3,
        "U_hr_mm_s": u_hr * 1e3,
        "U_hr_cm_s": u_hr * 1e2,
        "reynolds": re,
        "re_gt_1": bool(re > 1.0),
        "warning": warn,
    }


def fritz_detachment_radius_um(theta_deg, sigma=SIGMA_WATER_N_M, drho=DRHO_KG_M3,
                               g=G_ACCEL_M_S2):
    """Fritz (1935) bubble detachment radius at a horizontal surface, in um [FT].

    ::

        R_d = 0.0104 * theta_deg * sqrt(sigma / (g * drho))

    ``theta_deg`` is the contact angle in DEGREES; the 0.0104 prefactor absorbs the unit conversion
    and is only valid for degrees (with radians the same correlation needs 0.0104 * 180/pi). The
    balance behind it is buoyancy of the grown bubble against the vertical component of the
    capillary force at the contact line.

    Order of magnitude: water (sigma = 0.072 N/m, drho = 997) at 45 deg gives ~1.4 mm. Real
    electrolytic H2 bubbles detach far smaller (10-100 um) because the correlation ignores the
    contact-angle hysteresis, the waiting time and the current density; treat it as an upper bound
    for a quasi-static, surfactant-free bubble.
    """
    if theta_deg < 0.0:
        raise ValueError("theta_deg must be non-negative")
    return FRITZ_PREFACTOR_DEG * float(theta_deg) * math.sqrt(sigma / (g * drho)) * 1e6


def laplace_overpressure_Pa(R_um, sigma=SIGMA_WATER_N_M):
    """Capillary overpressure inside a sphere, ``2 sigma / R`` [FT] (Pa). Accepts scalars/arrays."""
    R_m = np.asarray(R_um, dtype=float) * 1e-6
    return 2.0 * sigma / R_m


def ideal_gas_density_mol_m3(p_Pa, T_C=25.0):
    """Molar density of an ideal gas, ``p / (R T)`` [FT] (mol/m3)."""
    return float(p_Pa) / (R_GAS * celsius_to_K(float(T_C)))


def henry_concentration_mol_m3(p_Pa, k_H=K_H_H2_MOL_L_ATM):
    """Dissolved gas concentration in equilibrium with a partial pressure, ``c = k_H * p`` [FT].

    ``k_H`` is the Henry solubility in mol/(L atm) at the bath temperature; ``p_Pa`` is converted to
    atm. The default is H2 in water at 25 C, 7.8e-4 mol/(L atm) [J]: pure H2 at 1 atm dissolves to
    ~0.78 mol/m3 (0.78 mmol/L), which is the scale the diffusive growth rate is built on.
    """
    return float(k_H) * (float(p_Pa) / P_ATM_PA) * 1e3


def epstein_plesset_rate_m_s(R_m, c_inf_mol_m3, c_s_mol_m3, D_m2_s, t_s=None,
                             rho_g_mol_m3=None):
    """Quasi-steady (Epstein-Plesset) diffusive growth rate of a stationary bubble, m/s [FT].

    ::

        dR/dt = D (c_inf - c_s) / (rho_g R) * (1 + R / sqrt(pi D t))

    ``c_inf`` is the far-field dissolved-gas concentration, ``c_s`` the concentration at the bubble
    wall (in equilibrium with the gas pressure inside, so it includes the Laplace overpressure),
    ``rho_g`` the molar density of the gas in the bubble and ``t`` the time since the (instantaneous)
    start of the growth. The bracket is the Epstein-Plesset correction for the finite diffusion
    length sqrt(pi D t): it is 1 at t = 0+ and decays to 1 as the boundary layer grows, so passing
    ``t_s=None`` or a non-positive ``t_s`` returns the plain quasi-steady coefficient ``D dc/(rho R)``.

    Assumptions: spherical bubble, no convection and no Marangoni flow, quasi-steady diffusion
    (dR/dt << D/R), gas is ideal, no gas leaves through the substrate, negligible dissolved-gas
    depletion of the far field. ``R`` grows as sqrt(t) at long times because R dR/dt is constant.
    """
    if rho_g_mol_m3 is None:
        rho_g_mol_m3 = 40.4                     # [C] ideal gas at 1 atm, 25 C
    if rho_g_mol_m3 <= 0.0:
        raise ValueError("rho_g_mol_m3 must be positive")
    R = max(float(R_m), 1e-12)
    factor = 1.0
    if t_s is not None and float(t_s) > 0.0:
        factor = 1.0 + R / math.sqrt(math.pi * float(D_m2_s) * float(t_s))
    return float(D_m2_s) * (float(c_inf_mol_m3) - float(c_s_mol_m3)) / (rho_g_mol_m3 * R) * factor


def diffusive_growth_rate_um_s(R_um, supersat=0.0, D_m2_s=D_H2_WATER_M2_S,
                               k_H=K_H_H2_MOL_L_ATM, p_atm_Pa=P_ATM_PA, sigma=SIGMA_WATER_N_M,
                               T_C=25.0, t_s=None):
    """Epstein-Plesset growth rate of an H2 bubble, in um/s, with the interface state filled in.

    * Interface pressure: ``p = p_atm + 2 sigma / R`` (Laplace) [FT].
    * ``c_s``: Henry equilibrium with that pressure [FT]; ``c_inf = (1 + supersat) * c_sat`` with
      ``c_sat`` the Henry concentration at ``p_atm_Pa``. ``supersat = 0`` therefore *still* shrinks a
      bubble, because the Laplace term raises the concentration the wall needs (the classical
      "a small bubble dissolves" result); ``supersat = 1`` doubles the far field and it grows.
    * ``rho_g``: ideal gas at the interface pressure [FT].

    The sign convention is the physically important part: negative dR/dt = dissolution. Scalar
    arguments only (the tracker calls it per bubble).
    """
    p_int = float(p_atm_Pa) + float(laplace_overpressure_Pa(R_um, sigma))
    c_s = henry_concentration_mol_m3(p_int, k_H)
    c_inf = (1.0 + float(supersat)) * henry_concentration_mol_m3(p_atm_Pa, k_H)
    rho_g = ideal_gas_density_mol_m3(p_int, T_C)
    return epstein_plesset_rate_m_s(float(R_um) * 1e-6, c_inf, c_s, D_m2_s, t_s,
                                    rho_g) * 1e6


def contact_disc_radius_um(R_um, theta_deg):
    """Radius of the wetted contact disc of a sphere of radius R sitting at contact angle theta [FT].

    ``r_c = R sin(theta)``: the gas occupies the spherical cap below the contact plane, and the disc
    it hides on the electrode is what causes the shading.
    """
    if not 0.0 <= float(theta_deg) <= 180.0:
        raise ValueError("theta_deg must be in [0, 180]")
    return float(R_um) * math.sin(math.radians(float(theta_deg)))


def shading_fraction(R_um, theta_deg, number_density_per_cm2):
    """Fraction of a flat electrode strip hidden by attached bubbles [J] (geometric, no overlap).

    Each attached bubble covers a disc of radius ``R sin(theta)``, so with ``N`` bubbles per cm2
    (1 cm2 = 1e8 um2)::

        f = N * pi * (R sin theta)**2 / 1e8,   clipped to [0, 1]

    The clip is a placeholder for disc overlap; real coverage saturates as a random-sequential
    adsorption ``1 - exp(-N pi r_c**2)``, which agrees with the linear form while f << 1. Use the
    exponential form if coverages above ~30 % matter.
    """
    r_c = contact_disc_radius_um(R_um, theta_deg)
    area_um2 = math.pi * r_c * r_c
    f = float(number_density_per_cm2) * area_um2 / 1e8
    return float(min(1.0, max(0.0, f)))
