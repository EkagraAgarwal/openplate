"""Nernst-layer (mass-transfer) correlation for the stirred beaker cell.

delta(rpm) combines a forced-convection term (stirrer) with the unstirred
natural-convection floor, then maps the mean onto the coupon as a Lévéque
boundary-layer profile along the flow direction.

Literature:
- natural convection floor 70-100 um (unstirred Cu sulfate) [SH] Fenech 1959
  (plating book chapter; via secondhand citation in the KB).
- forced term delta0 = 33-47 um at 300 rpm in stirred wells, exponent 0.505
  [FT] Avdeef, ADMET&DMPK 10:249 (2022), doi 10.5599/admet.1568.
- k_m reference points (dilute 0.05 M Cu, different cell) [FT] Touabi et al.,
  Int. J. Electrochem. Sci. 10:7227 (2015); plotting only.

Natural convection (vertical coupon, unstirred): laminar mass-transfer
correlation Sh_mean = 0.67 (Gr Sc)^(1/4) (Eisenberg/Tobias/Wilke form;
the 0.67 prefactor is a literature value [not verified] against the
paper).  Local Sh_x = 0.50 (Gr_x Sc)^(1/4), so delta_local ~ x^(1/4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.bath import Bath
from ..core.cell import CellGeometry
from ..core.constants import F

# ---- natural convection on a vertical plate (mass transfer)
G_ACCEL_CM_S2 = 981.0        # cm/s2
NU_WATER_CM2_S = 0.0114      # cm2/s, water kinematic viscosity, order of magnitude [J]
BETA_CUSO4_L_MOL = 0.15      # L/mol, solutal expansion coefficient, literature order [J][not verified]
# Laminar/turbulent limits for vertical-plate natural convection, Ra = Gr*Sc.
# Textbook heat-transfer transition is ~1e9 [J], but with Sc >> 1 electrochemical
# mass-transfer data stay laminar well above that; 1e11 / 1e12 are placeholders [J].
RA_NAT_LAM_MAX = 1.0e11
RA_NAT_TURB_MIN = 1.0e12


@dataclass
class MassTransferParams:
    delta_nat_um: float = 80.0   # unstirred natural-convection floor [SH] Fenech 1959 (70-100 um)
    delta0_um: float = 40.0      # forced layer at rpm_ref [FT] Avdeef 2022 (33-47 um @ 300 rpm)
    rpm_ref: float = 300.0       # rpm at which delta0 applies
    n: float = 0.5               # Avdeef exponent 0.505 ~ 0.5 [FT]
    baffle_factor: float = 0.8   # multiplies the forced part when baffles present [J]
    leveque: bool = True         # x-dependent (x - x0 + l0)^(1/3) profile on the coupon
    # ---- opt-in physics-based natural-convection floor (D4) --------------------
    use_natural: bool = False    # True: derive the unstirred floor from natural_convection()
    natural_height_cm: float | None = None      # None -> coupon height supplied by caller
    natural_c_bulk_mol_L: float | None = None   # None -> bath Cu bulk concentration
    natural_D_cm2_s: float | None = None        # None -> bath D_cu (no migration factor)
    natural_nu_cm2_s: float = NU_WATER_CM2_S
    natural_beta_L_mol: float = BETA_CUSO4_L_MOL


# Touabi et al. 2015, IJES 10:7227 [FT]: k_m at 0/500/1000/1500 rpm, dilute
# 0.05 M Cu, different cell geometry -> reference points for plotting only.
TOUABI_RPM = [0.0, 500.0, 1000.0, 1500.0]
TOUABI_KM = [1.42e-3, 2.52e-3, 3.18e-3, 7.99e-3]   # cm/s


def regime(Ra: float) -> str:
    """Natural-convection regime from the Rayleigh number Ra = Gr*Sc.

    laminar: Ra < 1e11, transitional: 1e11 <= Ra < 1e12, turbulent: Ra >= 1e12.
    Both thresholds are placeholders [J]: the usual vertical-plate heat-transfer
    value is ~1e9, but mass transfer with Sc >> 1 stays laminar to higher Ra.
    """
    if Ra < RA_NAT_LAM_MAX:
        return "laminar"
    if Ra < RA_NAT_TURB_MIN:
        return "transitional"
    return "turbulent"


def natural_convection(height_cm: float, c_bulk_mol_L: float, D_cm2_s: float,
                       nu_cm2_s: float = NU_WATER_CM2_S,
                       beta_L_mol: float = BETA_CUSO4_L_MOL,
                       i_frac: float = 1.0) -> dict:
    """Solutal natural convection on a vertical electrode at current fraction i_frac.

    Limiting current means i_frac = 1, interfacial Cu2+ concentration 0, so the
    density difference driving the boundary layer is
        drho/rho = beta * c_bulk * i_frac        (beta in L/mol, c_bulk in mol/L).
    Grashof / Rayleigh numbers use the electrode height L as the length scale:
        Gr = g L^3 (drho/rho) / nu^2,  Sc = nu / D,  Ra = Gr Sc.
    Laminar vertical-plate mass-transfer correlation (mean and local):
        Sh_mean = 0.67 (Gr Sc)^(1/4)         [not verified] 0.67 prefactor
        Sh_x    = 0.50 (Gr_x Sc)^(1/4)       -> delta_local ~ x^(1/4)
    Valid for laminar flow, about 1e4 < Ra < 1e12.  The 0.67 mean prefactor is
    the commonly quoted Eisenberg/Tobias/Wilke value; it is a literature number
    that was NOT verified against the paper here.

    Returns a dict with Gr, Sc, Ra, drho_over_rho, regime, Sh_mean, k_mean_cm_s,
    delta_mean_um (= D / k_mean), delta_local_um (callable, cm -> um) and
    i_lim_mA_cm2 (n = 2).  Diffusion is the plain D given, no migration factor.
    """
    L = float(height_cm)
    if L <= 0.0:
        raise ValueError("height_cm must be positive")
    if c_bulk_mol_L < 0.0 or D_cm2_s <= 0.0 or nu_cm2_s <= 0.0:
        raise ValueError("need c_bulk >= 0, D > 0, nu > 0")

    drho = beta_L_mol * float(c_bulk_mol_L) * float(i_frac)   # dimensionless
    Gr = G_ACCEL_CM_S2 * L ** 3 * drho / nu_cm2_s ** 2
    Sc = nu_cm2_s / D_cm2_s
    Ra = Gr * Sc

    Sh_mean = 0.67 * Ra ** 0.25
    k_mean = Sh_mean * D_cm2_s / L                            # cm/s
    delta_mean_um = D_cm2_s / k_mean * 1e4                    # cm -> um

    def delta_local_um(x_cm):
        """Local boundary-layer thickness (um) at distance x_cm from the leading edge."""
        def one(x):
            if x <= 0.0:
                raise ValueError("x must be positive")
            Gr_x = G_ACCEL_CM_S2 * x ** 3 * drho / nu_cm2_s ** 2
            Sh_x = 0.50 * (Gr_x * Sc) ** 0.25
            return x / Sh_x * 1e4
        if isinstance(x_cm, (list, tuple)):
            return [one(float(x)) for x in x_cm]
        return one(float(x_cm))

    c_mol_cm3 = 1e-3 * float(c_bulk_mol_L)
    i_lim = 2 * F * c_mol_cm3 * k_mean                      # A/cm2, n = 2
    return {
        "Gr": Gr,
        "Sc": Sc,
        "Ra": Ra,
        "drho_over_rho": drho,
        "regime": regime(Ra),
        "Sh_mean": Sh_mean,
        "k_mean_cm_s": k_mean,
        "delta_mean_um": delta_mean_um,
        "delta_local_um": delta_local_um,
        "i_lim_mA_cm2": i_lim * 1e3,
        "height_cm": L,
        "nu_cm2_s": nu_cm2_s,
        "beta_L_mol": beta_L_mol,
        "i_frac": i_frac,
    }


def natural_floor_um(p: MassTransferParams, bath: Bath | None = None,
                     height_cm: float | None = None) -> float:
    """Unstirred natural-convection floor (um) for the params.

    With p.use_natural, the floor is natural_convection()'s delta_mean_um using
    the height (p.natural_height_cm, else height_cm), c_bulk and D from the bath
    (plain D_cu, migration enhancement NOT included).  Falls back to
    p.delta_nat_um when the inputs are unavailable.
    """
    h = p.natural_height_cm if p.natural_height_cm is not None else height_cm
    c = p.natural_c_bulk_mol_L
    if c is None and bath is not None:
        c = bath.c_cu * 1e3                      # mol/cm3 -> mol/L
    D = p.natural_D_cm2_s
    if D is None and bath is not None:
        D = bath.D_cu                            # plain D, no E_mig factor
    if h is None or c is None or D is None:
        return p.delta_nat_um
    est = natural_convection(h, c, D, nu_cm2_s=p.natural_nu_cm2_s,
                             beta_L_mol=p.natural_beta_L_mol)
    return est["delta_mean_um"]


def delta_mean_um(rpm: float, p: MassTransferParams | None = None, baffles: bool = True,
                  bath: Bath | None = None, height_cm: float | None = None) -> float:
    """Area-mean Nernst-layer thickness (um) at the given stirrer speed.

    forced = delta0 * (rpm/rpm_ref)^(-n), infinite at rpm = 0;
    baffles multiply the forced part only (baffle_factor < 1 thins it);
    1/delta = 1/forced + 1/delta_nat (harmonic combine, forced dominates when fast).
    With p.use_natural the floor delta_nat comes from natural_convection()
    (see natural_floor_um); otherwise the constant p.delta_nat_um is used.
    """
    p = p or MassTransferParams()
    nat = natural_floor_um(p, bath, height_cm) if p.use_natural else p.delta_nat_um
    if rpm <= 0.0:
        forced = math.inf
    else:
        forced = p.delta0_um * (rpm / p.rpm_ref) ** (-p.n)
    if baffles:
        forced *= p.baffle_factor
    if math.isinf(forced):
        return nat
    return forced * nat / (forced + nat)


def delta_map(geom: CellGeometry, bath: Bath, rpm: float, p: MassTransferParams | None = None,
              baffles: bool = True, nx: int = 41, ny: int = 41) -> dict:
    """delta(x, y) on the coupon: x across width (cm, centred at 0), y along height (cm).

    If p.leveque, delta(x) ~ (x - x0 + l0)^(1/3) along +x with leading edge
    x0 = -w/2 and l0 = 0.05*w regularisation, normalised so its area mean
    equals delta_mean_um; uniform in y.
    """
    p = p or MassTransferParams()
    w, h = geom.coupon_w, geom.coupon_h
    mean_um = delta_mean_um(rpm, p, baffles=baffles, bath=bath, height_cm=h)
    x = [-w / 2 + w * i / (nx - 1) for i in range(nx)]     # cm, across width, centred 0
    y = [h * j / (ny - 1) for j in range(ny)]              # cm, along height from bottom

    if p.leveque:
        x0 = -w / 2.0
        l0 = 0.05 * w
        raw = [(xi - x0 + l0) ** (1.0 / 3.0) for xi in x]
        s = mean_um / (sum(raw) / len(raw))                # grid area-mean normalisation
        row = [s * r for r in raw]
    else:
        row = [mean_um] * nx
    delta_um = [list(row) for _ in range(ny)]              # uniform in y

    k_m = bath.D_cu_eff / (mean_um * 1e-4)                 # cm/s from mean layer (um -> cm)
    return {
        "x": x, "y": y, "delta_um": delta_um,
        "mean_um": mean_um, "k_m_cm_s": k_m,
    }


__all__ = ["MassTransferParams", "TOUABI_RPM", "TOUABI_KM", "delta_mean_um", "delta_map",
           "natural_convection", "natural_floor_um", "regime",
           "NU_WATER_CM2_S", "BETA_CUSO4_L_MOL", "RA_NAT_LAM_MAX", "RA_NAT_TURB_MIN"]
