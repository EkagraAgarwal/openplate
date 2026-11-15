"""One shared bath + kinetics parameter object with LEGACY constant sets and adapters.

Plan v2 items A4 (two conductivities that disagree by 3-4x) and F2 (one bath object across
``cell``, ``coupon``, ``efficiency``, ``feature``, ``trenchflow``).

Why this module exists
----------------------
There are two parameter sources today:

* :class:`platesim.core.bath.Bath` -- read by ``cell``, ``coupon`` and ``efficiency``.
  ``kappa_22 = 0.065 S/cm`` [C, +-20 %, not measured], ``E_mig = 1.67`` (literal),
  ``R_GAS = 8.314462618`` (``core.constants``).
* :class:`platesim.feature.engine.params.Params` / ``derive`` -- read by ``feature``,
  ``trenchflow`` and ``xcheck``.  Same ``kappa_22 = 0.065``, but ``E_mig`` is *computed*
  (``default_enhancement() = 1.6690357373508484``) and ``R_GAS = 8.314`` (module-local,
  deliberately not the core value).

The via regression is bit-identical only with each path's own constants, so this object
carries them as **legacy constant sets** instead of "fixing" the numbers:

===================  =================  ===========  ========================
set                  R [J/(mol K)]      F [C/mol]    E_mig rule
===================  =================  ===========  ========================
``feature_legacy``   8.314              96485.332    computed (1.6690357...)
``core_legacy``      8.314462618        96485.332    fixed 1.67
``codata``           8.314462618        96485.33212  computed
===================  =================  ===========  ========================

Conductivity is an explicit **choice** (``kappa_source``) and every consumer can ask
:meth:`SharedBath.kappa` for ``(value, source_label)`` so an output can say which one it used:

* ``placeholder_0.065`` -- ``0.065 * (1 + kappa_per_K*(T-22))``, the engine placeholder that
  both legacy paths use today (identical formula in ``Bath.kappa`` and ``Phys.kappa``).
* ``nernst_einstein``   -- ``F^2/(R T) sum z^2 D c`` with the *bath-effective* diffusivities
  (``D_cu25 = 2.65e-6``), i.e. bit-identical to ``Phys.kappa_ne``; ~0.257 S/cm at 22 C.
  (The infinite-dilution D table in ``core.speciation`` gives ~0.320 S/cm.)
* ``speciated``         -- ``core.speciation`` equilibrium + ``conductivity`` (pairs and
  HSO4- remove carriers); ~0.061 S/cm at 22 C.
* ``measured``          -- ``kappa_measured_S_cm``, taken at 22 C and scaled with the same
  linear ``kappa_per_K`` rule [J, the slope is the placeholder slope].

Units: cm, s, mol, A, V, K, g.  Concentrations are stored as grams per litre because that is
the ``Bath`` native form; the molar views are derived with exactly the expressions the two
legacy modules use, so the adapters are bit-identical with their defaults (see tests).

Honesty notes
-------------
* ``to_core_bath()`` cannot carry R or F: ``Bath`` has no such fields and always reads
  ``core.constants``.  The ``constants`` set therefore governs only this object's own
  derived quantities and ``E_mig`` for the ``Bath`` adapter.  ``diff_report`` states it.
* ``from_feature_params`` has to invert ``c_bulk``/``c_h_bulk`` back to g/L.  A plain
  ``x/M*M`` round trip is not exact in binary floating point, so the inverse scans a few
  ulps and keeps the value whose forward expression reproduces the input bit for bit.
* The feature engine has no Cl- concentration and no dissolved O2; the core Bath has no
  suppressor/accelerator set and no efficiency.  Those asymmetries are reported, not papered
  over (``diff_report`` marks them ``core-only`` / ``feature-only``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .bath import Bath, stokes_einstein_factor
from .constants import M_CU

M_H2SO4 = 98.079          # g/mol, the value both params.py and core.speciation use
KAPPA_REF_C = 22.0        # the temperature kappa_22 is defined at, in both legacy paths

KAPPA_SOURCES = ("placeholder_0.065", "nernst_einstein", "speciated", "measured")
CONSTANT_CHOICES = ("feature_legacy", "core_legacy", "codata")


@dataclass(frozen=True)
class _ConstSet:
    """R, F and the E_mig rule of one legacy/codata constant set."""

    r_gas: float
    faraday: float
    e_mig_rule: str                # "computed" | "fixed"
    e_mig_fixed: float | None
    note: str


CONSTANT_SETS: dict[str, _ConstSet] = {
    "feature_legacy": _ConstSet(
        8.314, 96485.332, "computed", None,
        "params.py R_GAS = 8.314 (module-local, keeps the via regression bit-identical); "
        "E_mig from default_enhancement()"),
    "core_legacy": _ConstSet(
        8.314462618, 96485.332, "fixed", 1.67,
        "core/bath.py E_mig = 1.67 (fill_model_v2 literal) with the full gas constant"),
    "codata": _ConstSet(
        8.314462618, 96485.33212, "computed", None,
        "CODATA/SI-2019 exact values (F = 96485.33212 C/mol); E_mig computed as in params.py"),
}


@dataclass(frozen=True)
class SharedBath:
    """Bath composition, transport, kinetics, additive set and the constant/conductivity choice.

    A frozen dataclass: use :meth:`at` for another temperature and :meth:`with_` for any
    other replacement.  Nothing here is wired into the engines yet; the adapters
    :meth:`to_core_bath` / :meth:`to_feature_kwargs` are the intended seam.
    """

    # ---- composition (Bath-native units, tags as in core/bath.py KB section 1)
    cu_g_L: float = 65.0                  # [M] bottle, 65 +- 2
    h2so4_g_L: float = 8.0                # [M] bottle
    cl_ppm: float = 40.0                  # [M] bottle
    organics_ppm: float = 600.0           # [M] bottle, proprietary
    T_C: float = 22.0

    # ---- transport
    D_cu25: float = 2.65e-6               # cm2/s [FT] NIST/Braun, bath-effective
    D_s25: float = 1.0e-6                 # cm2/s, suppressor (feature-only)
    D_H25: float = 9.31e-5                # cm2/s, H+ (feature tertiary path)
    D_SO425: float = 1.07e-5              # cm2/s, SO4^2- (feature tertiary path)
    D_o2_25: float = 2.1e-5               # cm2/s [FT] (core-only, O2 parasite)
    c_o2_sat_25: float = 2.6e-7           # mol/cm3 air-saturated water 25 C [FT] (core-only)

    # ---- kinetics (22 C reference, Arrhenius on every i0)
    i0_cu_22: float = 2.0e-3              # A/cm2 unsuppressed [FT] Braun Table I
    i0_cu_supp_22: float = 1.3e-5         # A/cm2 suppressed face [FT] Braun Table I
    i0_a_22: float = 2.0e-3               # A/cm2 anodic exchange (feature-only field)
    alpha_c: float = 0.5                  # cathodic, unsuppressed (Bath.alpha_c / Params.alpha_u)
    alpha_s: float = 0.5                  # cathodic, suppressed (feature-only; Bath reuses alpha_c)
    alpha_a: float = 1.5
    Ea_i0: float = 35e3                   # J/mol [J]
    T_ref_C: float = 22.0                 # Arrhenius reference (Bath hardcodes 295.15 K)

    # ---- additive set: suppressor S-NDR (Braun high-k) + optional accelerator
    suppressor: bool = True
    s_bulk: float = 8e-8                  # mol/cm3 (80 umol/L)
    Gamma: float = 9.2e-12                # mol/cm2 saturation coverage
    k_plus: float = 2.5e9                 # cm3/(mol s)
    k_minus: float = 1.0e6                # 1/cm of growth (TUNED, V5)
    theta0: float = 0.0
    s_transient: bool = True
    k_rd: float = 0.0                     # reverse-pulse suppressor desorption
    use_accel: bool = False
    c_acc: float = 5e-9                   # mol/cm3
    k_acc: float = 1.76e6                 # cm3/(mol s)
    b_acc: float = 4.5e-3 / 2.6e-5        # i0 multiplier slope

    # ---- current efficiency used by the growth velocity (Params.efficiency)
    efficiency: float = 1.0

    # ---- conductivity choice (A4) and speciation settings for the speciated option
    kappa_source: str = "placeholder_0.065"
    kappa_placeholder_S_cm: float = 0.065  # S/cm at 22 C [C, +-20 %, not measured]
    kappa_per_K: float = 0.018             # 1/K linear rule, identical in both paths
    kappa_measured_S_cm: float | None = None
    activity: str = "davies"               # 'ideal' | 'davies' | 'bdot' (core.speciation)
    logK_ip: float | None = 2.3            # log10 CuSO4^0 ion-pair constant [J, not verified]
    Ka2: float | None = None               # None -> core.speciation.ka2_at(T_C)

    # ---- constants set and an optional measured/fitted E_mig override
    constants: str = "feature_legacy"
    E_mig_measured: float | None = None
    # ---- provenance: which legacy source built this object (used only by diff_report to
    # blank out fields that source does not define).  'standalone' = all fields meaningful.
    source: str = "standalone"

    # ------------------------------------------------------------------ validation
    def __post_init__(self) -> None:
        if self.constants not in CONSTANT_SETS:
            raise ValueError(f"constants must be one of {CONSTANT_CHOICES}, got {self.constants!r}")
        if self.kappa_source not in KAPPA_SOURCES:
            raise ValueError(f"kappa_source must be one of {KAPPA_SOURCES}, got {self.kappa_source!r}")
        if self.source not in ("standalone", "core_bath", "feature_params"):
            raise ValueError("source must be 'standalone', 'core_bath' or 'feature_params'")
        if self.kappa_source == "measured" and self.kappa_measured_S_cm is None:
            raise ValueError("kappa_source='measured' requires kappa_measured_S_cm")
        if self.kappa_placeholder_S_cm <= 0.0:
            raise ValueError("kappa_placeholder_S_cm must be positive")
        if self.activity not in ("ideal", "davies", "bdot"):
            raise ValueError("activity must be 'ideal', 'davies' or 'bdot'")
        if self.cu_g_L < 0.0 or self.h2so4_g_L < 0.0:
            raise ValueError("concentrations must be non-negative")

    # ------------------------------------------------------------------ small helpers
    @property
    def consts(self) -> _ConstSet:
        return CONSTANT_SETS[self.constants]

    @property
    def r_gas(self) -> float:
        """Gas constant of the selected set, J/(mol K)."""
        return self.consts.r_gas

    @property
    def faraday(self) -> float:
        """Faraday constant of the selected set, C/mol."""
        return self.consts.faraday

    @property
    def T_K(self) -> float:
        return self.T_C + 273.15

    def at(self, T_C: float) -> "SharedBath":
        return replace(self, T_C=float(T_C))

    def with_(self, **changes: Any) -> "SharedBath":
        return replace(self, **changes)

    def __str__(self) -> str:  # short, stable one-liner for logs
        return (f"SharedBath({self.cu_g_L:g} g/L Cu, {self.h2so4_g_L:g} g/L H2SO4, "
                f"{self.T_C:g} C, kappa={self.kappa()[1]}, constants={self.constants})")

    # ------------------------------------------------------------------ composition views
    @property
    def c_Cu_mol_L(self) -> float:
        """Analytical Cu2+ concentration, mol/L."""
        return self.cu_g_L / M_CU

    @property
    def c_acid_mol_L(self) -> float:
        """Analytical H2SO4 concentration, mol/L."""
        return self.h2so4_g_L / M_H2SO4

    @property
    def c_cu(self) -> float:
        """Cu2+ bulk concentration, mol/cm3 (== Params.c_bulk expression)."""
        return self.cu_g_L / M_CU * 1e-3

    @property
    def c_h_bulk(self) -> float:
        """H+ bulk concentration, mol/cm3 (== Params.c_h_bulk expression)."""
        return 2.0 * self.c_acid_mol_L * 1e-3

    @property
    def c_so4_bulk(self) -> float:
        """Total sulfate, mol/cm3 (== Params.c_so4_bulk expression)."""
        return (self.cu_g_L / M_CU + self.h2so4_g_L / M_H2SO4) * 1e-3

    # ------------------------------------------------------------------ transport / kinetics
    @property
    def D_cu(self) -> float:
        return self.D_cu25 * stokes_einstein_factor(self.T_C)

    @property
    def D_s(self) -> float:
        return self.D_s25 * stokes_einstein_factor(self.T_C)

    @property
    def D_H(self) -> float:
        return self.D_H25 * stokes_einstein_factor(self.T_C)

    @property
    def D_so4(self) -> float:
        return self.D_SO425 * stokes_einstein_factor(self.T_C)

    @property
    def D_o2(self) -> float:
        return self.D_o2_25 * stokes_einstein_factor(self.T_C)

    @property
    def c_o2(self) -> float:
        """Dissolved O2 (air-saturated), mol/cm3; ~2 %/K decrease from 25 C."""
        return self.c_o2_sat_25 * math.exp(-0.019 * (self.T_C - 25.0))

    @property
    def E_mig(self) -> float:
        """Migration enhancement D_eff/D.  Measured override > constant-set rule."""
        if self.E_mig_measured is not None:
            return float(self.E_mig_measured)
        if self.consts.e_mig_rule == "fixed":
            fixed = self.consts.e_mig_fixed
            assert fixed is not None, "a 'fixed' E_mig rule must carry a value"
            return float(fixed)
        # Bit-identical to Params: the same function, the same quadrature.
        from ..feature.engine.params import default_enhancement
        return default_enhancement()

    @property
    def D_cu_eff(self) -> float:
        return self.E_mig * self.D_cu

    @property
    def f_RT(self) -> float:
        return self.faraday / (self.r_gas * self.T_K)

    def arrhenius(self, T_C: float | None = None) -> float:
        T = self.T_C if T_C is None else float(T_C)
        return math.exp(-self.Ea_i0 / self.r_gas * (1.0 / (T + 273.15) - 1.0 / (self.T_ref_C + 273.15)))

    @property
    def i0_cu(self) -> float:
        return self.i0_cu_22 * self.arrhenius()

    @property
    def i0_cu_supp(self) -> float:
        return self.i0_cu_supp_22 * self.arrhenius()

    def j_lim_cu(self, delta_cm: float) -> float:
        """Cu2+ limiting current density A/cm2 for a Nernst layer delta."""
        return 2 * self.faraday * self.D_cu_eff * self.c_cu / delta_cm

    def j_lim_o2(self, delta_cm: float) -> float:
        """O2 reduction limiting current A/cm2 (4 e-), Sc^-1/3 delta scaling."""
        d = delta_cm * (self.D_o2 / self.D_cu) ** (1 / 3)
        return 4 * self.faraday * self.D_o2 * self.c_o2 / d

    # ------------------------------------------------------------------ conductivity (A4)
    def kappa_placeholder(self, T_C: float | None = None) -> float:
        """Legacy placeholder: kappa_22*(1 + kappa_per_K*(T-22)); exact 0.065 at 22 C."""
        T = self.T_C if T_C is None else float(T_C)
        return self.kappa_placeholder_S_cm * (1.0 + self.kappa_per_K * (T - KAPPA_REF_C))

    def kappa_nernst_einstein(self, T_C: float | None = None) -> float:
        """Free-ion NE conductivity with the bath-effective D set, S/cm.

        Identical expression and operation order to ``params.derive(...).kappa_ne``
        (``F * f_RT * (4 D_cu c_bulk + D_H c_h + 4 D_so4 c_so4)``), so with the
        ``feature_legacy`` set it reproduces ``Phys.kappa_ne`` bit for bit.
        Using ``core.speciation.D_ION_25`` instead (D_cu 7.14e-6) gives ~0.320 S/cm.
        """
        T = self.T_C if T_C is None else float(T_C)
        fD = stokes_einstein_factor(T)
        D_cu = self.D_cu25 * fD
        D_H = self.D_H25 * fD
        D_so4 = self.D_SO425 * fD
        f_RT = self.faraday / (self.r_gas * (T + 273.15))
        return self.faraday * f_RT * (4 * D_cu * self.c_cu + D_H * self.c_h_bulk + 4 * D_so4 * self.c_so4_bulk)

    def kappa_speciated(self, T_C: float | None = None) -> float:
        """Senior-moment conductivity: core.speciation equilibrium + NE, S/cm."""
        from .speciation import conductivity, speciate
        T = self.T_C if T_C is None else float(T_C)
        K_ip = None if self.logK_ip is None else 10.0 ** float(self.logK_ip)
        spec = speciate(self.c_Cu_mol_L, self.c_acid_mol_L, T_C=T,
                        Ka2=self.Ka2, K_ip=K_ip, activity=self.activity)
        return conductivity(spec, T_C=T)

    def kappa(self, T_C: float | None = None) -> tuple[float, str]:
        """(conductivity S/cm, source label) for the selected ``kappa_source``."""
        src = self.kappa_source
        if src == "placeholder_0.065":
            return self.kappa_placeholder(T_C), f"placeholder_{self.kappa_placeholder_S_cm:g}"
        if src == "nernst_einstein":
            return self.kappa_nernst_einstein(T_C), "nernst_einstein"
        if src == "speciated":
            return self.kappa_speciated(T_C), "speciated"
        if src == "measured":
            measured = self.kappa_measured_S_cm
            assert measured is not None, "kappa_source='measured' requires kappa_measured_S_cm"
            T = self.T_C if T_C is None else float(T_C)
            return (float(measured) * (1.0 + self.kappa_per_K * (T - KAPPA_REF_C)),
                    "measured")
        raise ValueError(f"unknown kappa_source {src!r}")  # pragma: no cover - validated above

    # ------------------------------------------------------------------ adapters
    def to_core_bath(self) -> Bath:
        """``Bath`` carrying this object's values.

        With ``constants='core_legacy'`` the result equals the default ``Bath()`` field by
        field (exact).  Caveat: ``Bath`` reads ``core.constants`` for R and F, so those are
        never this object's; only ``E_mig``, the composition, the D values, the kinetics and
        ``kappa_22`` (evaluated at 22 C from the selected ``kappa_source``) cross over.
        """
        return Bath(
            cu_g_L=self.cu_g_L,
            h2so4_g_L=self.h2so4_g_L,
            cl_ppm=self.cl_ppm,
            organics_ppm=self.organics_ppm,
            T_C=self.T_C,
            D_cu25=self.D_cu25,
            E_mig=self.E_mig,
            kappa_22=self.kappa(KAPPA_REF_C)[0],
            kappa_per_K=self.kappa_per_K,
            D_o2_25=self.D_o2_25,
            c_o2_sat_25=self.c_o2_sat_25,
            i0_cu_22=self.i0_cu_22,
            i0_cu_supp_22=self.i0_cu_supp_22,
            alpha_c=self.alpha_c,
            alpha_a=self.alpha_a,
            Ea_i0=self.Ea_i0,
        )

    def to_feature_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``Params(**...)`` (imported read-only by the caller).

        With ``constants='feature_legacy'`` and the default ``kappa_source`` the result
        reproduces the default ``Params()`` field by field (exact).  ``kappa_22`` is always
        taken at 22 C from the selected ``kappa_source``, so selecting a different
        conductivity changes exactly one Params field.
        """
        return {
            # bath / electrolyte
            "c_bulk": self.c_cu,
            "D_cu25": self.D_cu25,
            "E_mig": self.E_mig,
            "kappa_22": self.kappa(KAPPA_REF_C)[0],
            "kappa_per_K": self.kappa_per_K,
            # tertiary transport
            "D_H25": self.D_H25,
            "D_SO425": self.D_SO425,
            "c_h_bulk": self.c_h_bulk,
            "c_so4_bulk": self.c_so4_bulk,
            # kinetics
            "i0_u": self.i0_cu_22,
            "alpha_u": self.alpha_c,
            "i0_s": self.i0_cu_supp_22,
            "alpha_s": self.alpha_s,
            "i0_a": self.i0_a_22,
            "alpha_a": self.alpha_a,
            "Ea_i0": self.Ea_i0,
            "T_ref_C": self.T_ref_C,
            # suppressor / additive set
            "suppressor": self.suppressor,
            "s_bulk": self.s_bulk,
            "D_s25": self.D_s25,
            "Gamma": self.Gamma,
            "k_plus": self.k_plus,
            "k_minus": self.k_minus,
            "theta0": self.theta0,
            "s_transient": self.s_transient,
            "k_rd": self.k_rd,
            "use_accel": self.use_accel,
            "c_acc": self.c_acc,
            "k_acc": self.k_acc,
            "b_acc": self.b_acc,
            # speciation chemistry used by the tertiary path
            "logK_ip": self.logK_ip,
            "activity": self.activity,
            # growth
            "efficiency": self.efficiency,
        }

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_core_bath(cls, b: Bath) -> "SharedBath":
        """SharedBath from a core ``Bath``; carries ``constants='core_legacy'``.

        ``E_mig`` is preserved verbatim (as a measured override) so the round trip
        ``to_core_bath()`` is bit-identical even if the caller changed ``Bath.E_mig``.
        Fields Bath does not have (suppressor set, efficiency) take their defaults.
        """
        return cls(
            cu_g_L=b.cu_g_L,
            h2so4_g_L=b.h2so4_g_L,
            cl_ppm=b.cl_ppm,
            organics_ppm=b.organics_ppm,
            T_C=b.T_C,
            D_cu25=b.D_cu25,
            D_o2_25=b.D_o2_25,
            c_o2_sat_25=b.c_o2_sat_25,
            i0_cu_22=b.i0_cu_22,
            i0_cu_supp_22=b.i0_cu_supp_22,
            alpha_c=b.alpha_c,
            alpha_a=b.alpha_a,
            Ea_i0=b.Ea_i0,
            T_ref_C=KAPPA_REF_C,          # Bath hardcodes 295.15 K = 22 C
            kappa_placeholder_S_cm=b.kappa_22,
            kappa_per_K=b.kappa_per_K,
            E_mig_measured=b.E_mig,
            constants="core_legacy",
            source="core_bath",
        )

    @classmethod
    def from_feature_params(cls, p: Any) -> "SharedBath":
        """SharedBath from feature ``Params`` (read-only import, duck-typed).

        ``constants='feature_legacy'``; ``E_mig`` is preserved as a measured override so the
        round trip is bit-identical.  Cl-/organics/O2, which the feature engine does not
        carry, keep their core defaults.  Temperature is 22 C (``Params`` has no bath T).
        """
        return cls(
            cu_g_L=_g_per_L(p.c_bulk * 1e3, M_CU),
            h2so4_g_L=_g_per_L(p.c_h_bulk / 2.0 * 1e3, M_H2SO4),
            T_C=KAPPA_REF_C,
            D_cu25=p.D_cu25,
            D_s25=p.D_s25,
            D_H25=p.D_H25,
            D_SO425=p.D_SO425,
            i0_cu_22=p.i0_u,
            i0_cu_supp_22=p.i0_s,
            i0_a_22=p.i0_a,
            alpha_c=p.alpha_u,
            alpha_s=p.alpha_s,
            alpha_a=p.alpha_a,
            Ea_i0=p.Ea_i0,
            T_ref_C=p.T_ref_C,
            suppressor=p.suppressor,
            s_bulk=p.s_bulk,
            Gamma=p.Gamma,
            k_plus=p.k_plus,
            k_minus=p.k_minus,
            theta0=p.theta0,
            s_transient=p.s_transient,
            k_rd=p.k_rd,
            use_accel=p.use_accel,
            c_acc=p.c_acc,
            k_acc=p.k_acc,
            b_acc=p.b_acc,
            efficiency=p.efficiency,
            kappa_placeholder_S_cm=p.kappa_22,
            kappa_per_K=p.kappa_per_K,
            logK_ip=p.logK_ip,
            activity=p.activity,
            E_mig_measured=p.E_mig,
            constants="feature_legacy",
            source="feature_params",
        )


def _g_per_L(target_mol_L: float, M: float, span: int = 4) -> float:
    """g/L whose ``g/M`` reproduces ``target_mol_L`` bit for bit (ulp scan).

    ``(g/M)*M == g`` is not exact in binary floating point, so ``from_feature_params``
    cannot invert ``c_bulk = g/M*1e-3`` algebraically without losing the last bit.  Pick
    the neighbouring float whose forward expression matches instead.
    """
    est = float(target_mol_L) * M
    for step in range(span + 1):
        for cand in ((est,) if step == 0 else
                     (est, _nudge(est, +step), _nudge(est, -step))):
            if cand / M == target_mol_L:
                return cand
    return est  # pragma: no cover - no exact pre-image; best estimate


def _nudge(x: float, steps: int) -> float:
    for _ in range(abs(steps)):
        x = math.nextafter(x, math.inf if steps > 0 else -math.inf)
    return x


# ---------------------------------------------------------------------- F2 audit
def diff_report(a: SharedBath, b: SharedBath, T_C: float = KAPPA_REF_C, *,
                print_table: bool = False,
                labels: tuple[str, str] = ("path A", "path B")) -> list[dict[str, Any]]:
    """Every audited quantity for two paths, with a bitwise ``differ`` flag.

    Typical use: ``diff_report(SharedBath.from_core_bath(Bath()),
    SharedBath.from_feature_params(Params()))``.  All temperature-dependent rows are
    evaluated at the same ``T_C`` so the comparison cannot hide behind a temperature.
    ``print_table=True`` also prints :func:`format_diff_report` (the A4/F2 audit table).

    Row keys: ``quantity``, ``path_a``, ``path_b``, ``differ``, ``note``.
    """
    A = a.at(T_C)
    B = b.at(T_C)
    hot = float(T_C) + 18.0
    A_hot = a.at(hot)
    B_hot = b.at(hot)
    rows: list[dict[str, Any]] = []

    def add(quantity: str, va: Any, vb: Any, note: str = "") -> None:
        differ = not _same(va, vb)
        rows.append({"quantity": quantity, "path_a": va, "path_b": vb,
                     "differ": differ, "note": note})

    # A quantity that one legacy source does not define at all is reported as "--", not as a
    # silent default: that is the F2 finding (Cl-/O2 core-only, suppressor/H+/SO4 feature-only).
    def core_only(obj: SharedBath, value: Any) -> Any:
        return None if obj.source == "feature_params" else value

    def feature_only(obj: SharedBath, value: Any) -> Any:
        return None if obj.source == "core_bath" else value

    # ---- constants
    add("R", A.r_gas, B.r_gas,
        "J/(mol K): core.constants 8.314462618 vs params.R_GAS 8.314; Bath reads core.constants")
    add("F", A.faraday, B.faraday,
        "C/mol: both legacy sets use 96485.332 (codata set: 96485.33212)")
    add("E_mig", A.E_mig, B.E_mig,
        "bath.py literal 1.67 vs params.default_enhancement() = 1.6690357373508484")

    # ---- conductivity (A4)
    add("kappa", A.kappa()[0], B.kappa()[0],
        f"selected source: {A.kappa()[1]} / {B.kappa()[1]} (both paths use the placeholder today)")
    add("kappa_nernst_einstein", A.kappa_nernst_einstein(), B.kappa_nernst_einstein(),
        "free-ion NE with each path's own R, F and D; ~0.257 S/cm vs the 0.065 placeholder")
    add("kappa_speciated", A.kappa_speciated(), B.kappa_speciated(),
        "core.speciation pairs/HSO4- removed; ~0.061 S/cm; differs with activity/K_ip")
    kA = A.kappa()[0]
    kB = B.kappa()[0]
    add("kappa_ne_over_placeholder",
        A.kappa_nernst_einstein() / kA if kA else None,
        B.kappa_nernst_einstein() / kB if kB else None,
        "A4: how far the placeholder sits below free-ion NE (3-4x)")
    add("kappa_per_K", A.kappa_per_K, B.kappa_per_K,
        "1/K: identical linear kappa(T) rule in both paths")

    # ---- transport
    add("D_cu25", A.D_cu25, B.D_cu25, "cm2/s, NIST/Braun, identical in both paths")
    add("D_cu", A.D_cu, B.D_cu, "cm2/s at T (Stokes-Einstein, identical helper)")
    add("D_cu_eff", A.D_cu_eff, B.D_cu_eff, "E_mig*D_cu; differs only through E_mig")
    add("D_s25", feature_only(A, A.D_s25), feature_only(B, B.D_s25),
        "feature-only suppressor diffusivity")
    add("D_H25", feature_only(A, A.D_H25), feature_only(B, B.D_H25),
        "feature-only (tertiary path)")
    add("D_SO425", feature_only(A, A.D_SO425), feature_only(B, B.D_SO425),
        "feature-only 1.07e-5; core.speciation.D_ION_25 uses 1.065e-5 for SO4")
    add("D_o2_25", core_only(A, A.D_o2_25), core_only(B, B.D_o2_25),
        "core-only (O2 parasitic term)")
    add("c_o2_sat_25", core_only(A, A.c_o2_sat_25), core_only(B, B.c_o2_sat_25),
        "core-only (O2 parasitic term)")

    # ---- kinetics
    add("i0_u", A.i0_cu, B.i0_cu, "A/cm2 at T; 2.0e-3 at 22 C, Arrhenius differs through R")
    add("i0_s", A.i0_cu_supp, B.i0_cu_supp, "A/cm2 at T; 1.3e-5 at 22 C")
    add("i0_u_hot", A_hot.i0_cu, B_hot.i0_cu,
        f"A/cm2 at {hot:g} C; the R difference shows up at temperature")
    add("i0_a", feature_only(A, A.i0_a_22 * A.arrhenius()), feature_only(B, B.i0_a_22 * B.arrhenius()),
        "feature-only anodic i0; Bath reuses i0_cu for the anode branch")
    add("alpha_u", A.alpha_c, B.alpha_c, "cathodic unsuppressed (Bath.alpha_c)")
    add("alpha_s", feature_only(A, A.alpha_s), feature_only(B, B.alpha_s),
        "feature-only separate suppressed alpha (Bath reuses alpha_c)")
    add("alpha_a", A.alpha_a, B.alpha_a, "anodic transfer coefficient")
    add("Ea_i0", A.Ea_i0, B.Ea_i0, "J/mol, identical")
    add("T_ref_arrhenius_K", 295.15, B.T_ref_C + 273.15,
        "Bath hardcodes 295.15 K inside arrhenius(); Params exposes T_ref_C=22")

    # ---- composition
    add("c_cu", A.c_cu, B.c_cu, "mol/cm3, cu_g_L/M_CU")
    add("c_h_bulk", A.c_h_bulk, B.c_h_bulk, "mol/cm3, 2*acid")
    add("c_so4_bulk", A.c_so4_bulk, B.c_so4_bulk, "mol/cm3, Cu + acid")
    add("cl_ppm", core_only(A, A.cl_ppm), core_only(B, B.cl_ppm),
        "core-only: the feature Params carries no Cl- concentration (enters via k_plus/k_minus)")
    add("organics_ppm", core_only(A, A.organics_ppm), core_only(B, B.organics_ppm),
        "core-only lumped organics; the feature engine uses the explicit S-NDR additive set")
    add("s_bulk", feature_only(A, A.s_bulk), feature_only(B, B.s_bulk),
        "feature-only suppressor concentration, mol/cm3")

    # ---- efficiency
    add("efficiency", A.efficiency, B.efficiency,
        "Bath has no efficiency (core thickness assumes 1.0); Params.efficiency = 1.0")
    if print_table:
        print(format_diff_report(rows, labels=labels))
    return rows


def _same(va: Any, vb: Any) -> bool:
    if va is None or vb is None:
        return va is None and vb is None
    if isinstance(va, bool) or isinstance(vb, bool):
        return bool(va) is bool(vb)
    return float(va) == float(vb)


def _fmt(v: Any) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(v)
    return f"{float(v):.10g}"


def format_diff_report(rows: list[dict[str, Any]], labels: tuple[str, str] = ("path A", "path B")) -> str:
    """Plain-text table of a :func:`diff_report` (one row per audited quantity)."""
    head = ("quantity", labels[0], labels[1], "differ", "note")
    body = [(r["quantity"], _fmt(r["path_a"]), _fmt(r["path_b"]),
             "YES" if r["differ"] else "no", r["note"]) for r in rows]
    widths = [max(len(row[i]) for row in (head,) + tuple(body)) for i in range(5)]
    lines = []
    for i, row in enumerate((head,) + tuple(body)):
        if i == 1:
            lines.append("  ".join("-" * w for w in widths))
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    n_diff = sum(1 for r in rows if r["differ"])
    lines.append(f"[{n_diff}/{len(rows)} quantified quantities differ; "
                 f"path A = {labels[0]}, path B = {labels[1]}]")
    return "\n".join(lines)
