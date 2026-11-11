"""Bath properties for the Nano3D/Transene EP-EC1-65L-01-SEMI acid copper bath (KB section 1).

Every number is tagged as in research/TSV_KNOWLEDGE_BASE.md. Placeholders are marked [J] and are calibration targets.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict, replace

from .constants import F, R_GAS, M_CU


def _eta_water(T_C: float) -> float:
    """Water viscosity (Pa s, Vogel form), used for Stokes-Einstein scaling."""
    return 2.414e-5 * 10 ** (247.8 / (T_C + 273.15 - 140))


def stokes_einstein_factor(T_C: float, T_ref_C: float = 25.0) -> float:
    return ((T_C + 273.15) / (T_ref_C + 273.15)) * (_eta_water(T_ref_C) / _eta_water(T_C))


@dataclass
class Bath:
    cu_g_L: float = 65.0            # [M] bottle, 65 +- 2
    h2so4_g_L: float = 8.0          # [M] bottle
    cl_ppm: float = 40.0            # [M] bottle
    organics_ppm: float = 600.0     # [M] bottle, proprietary
    T_C: float = 22.0
    D_cu25: float = 2.65e-6         # cm2/s [FT] NIST/Braun
    E_mig: float = 1.67             # migration enhancement [C] fill_model_v2
    kappa_22: float = 0.065         # S/cm [C, +-20 %] not measured
    kappa_per_K: float = 0.018      # 1/K [J]
    D_o2_25: float = 2.1e-5         # cm2/s [FT]
    c_o2_sat_25: float = 2.6e-7     # mol/cm3 air-saturated water 25 C, salting-out ignored [FT]
    i0_cu_22: float = 2.0e-3        # A/cm2 unsuppressed [FT] Braun Table I
    i0_cu_supp_22: float = 1.3e-5   # A/cm2 suppressed face [FT] Braun Table I
    alpha_c: float = 0.5
    alpha_a: float = 1.5
    Ea_i0: float = 35e3             # J/mol [J]

    # ---- derived, temperature dependent
    @property
    def T_K(self) -> float:
        return self.T_C + 273.15

    @property
    def f_RT(self) -> float:
        return F / (R_GAS * self.T_K)

    @property
    def c_cu(self) -> float:
        """Cu2+ bulk concentration, mol/cm3."""
        return self.cu_g_L / M_CU * 1e-3

    @property
    def D_cu(self) -> float:
        return self.D_cu25 * stokes_einstein_factor(self.T_C)

    @property
    def D_cu_eff(self) -> float:
        return self.E_mig * self.D_cu

    @property
    def kappa(self) -> float:
        return self.kappa_22 * (1 + self.kappa_per_K * (self.T_C - 22.0))

    @property
    def D_o2(self) -> float:
        return self.D_o2_25 * stokes_einstein_factor(self.T_C)

    @property
    def c_o2(self) -> float:
        """Dissolved O2 (air-saturated), mol/cm3; ~2 %/K decrease from 25 C."""
        return self.c_o2_sat_25 * math.exp(-0.019 * (self.T_C - 25.0))

    def arrhenius(self) -> float:
        return math.exp(-self.Ea_i0 / R_GAS * (1 / self.T_K - 1 / 295.15))

    @property
    def i0_cu(self) -> float:
        return self.i0_cu_22 * self.arrhenius()

    @property
    def i0_cu_supp(self) -> float:
        return self.i0_cu_supp_22 * self.arrhenius()

    def j_lim_cu(self, delta_cm: float) -> float:
        """Cu2+ limiting current density A/cm2 for a Nernst layer delta."""
        return 2 * F * self.D_cu_eff * self.c_cu / delta_cm

    def j_lim_o2(self, delta_cm: float) -> float:
        """O2 reduction limiting current A/cm2 (4 e-), delta scaled by (D_o2/D_cu)^(1/3) (Sc^-1/3 rule)."""
        d = delta_cm * (self.D_o2 / self.D_cu) ** (1 / 3)
        return 4 * F * self.D_o2 * self.c_o2 / d

    def at(self, T_C: float) -> "Bath":
        return replace(self, T_C=T_C)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(c_cu=self.c_cu, D_cu=self.D_cu, kappa=self.kappa, i0_cu=self.i0_cu)
        return d
