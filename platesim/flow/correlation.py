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
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.bath import Bath
from ..core.cell import CellGeometry


@dataclass
class MassTransferParams:
    delta_nat_um: float = 80.0   # unstirred natural-convection floor [SH] Fenech 1959 (70-100 um)
    delta0_um: float = 40.0      # forced layer at rpm_ref [FT] Avdeef 2022 (33-47 um @ 300 rpm)
    rpm_ref: float = 300.0       # rpm at which delta0 applies
    n: float = 0.5               # Avdeef exponent 0.505 ~ 0.5 [FT]
    baffle_factor: float = 0.8   # multiplies the forced part when baffles present [J]
    leveque: bool = True         # x-dependent (x - x0 + l0)^(1/3) profile on the coupon


# Touabi et al. 2015, IJES 10:7227 [FT]: k_m at 0/500/1000/1500 rpm, dilute
# 0.05 M Cu, different cell geometry -> reference points for plotting only.
TOUABI_RPM = [0.0, 500.0, 1000.0, 1500.0]
TOUABI_KM = [1.42e-3, 2.52e-3, 3.18e-3, 7.99e-3]   # cm/s


def delta_mean_um(rpm: float, p: MassTransferParams | None = None, baffles: bool = True) -> float:
    """Area-mean Nernst-layer thickness (um) at the given stirrer speed.

    forced = delta0 * (rpm/rpm_ref)^(-n), infinite at rpm = 0;
    baffles multiply the forced part only (baffle_factor < 1 thins it);
    1/delta = 1/forced + 1/delta_nat (harmonic combine, forced dominates when fast).
    """
    p = p or MassTransferParams()
    if rpm <= 0.0:
        forced = math.inf
    else:
        forced = p.delta0_um * (rpm / p.rpm_ref) ** (-p.n)
    if baffles:
        forced *= p.baffle_factor
    if math.isinf(forced):
        return p.delta_nat_um
    return forced * p.delta_nat_um / (forced + p.delta_nat_um)


def delta_map(geom: CellGeometry, bath: Bath, rpm: float, p: MassTransferParams | None = None,
              baffles: bool = True, nx: int = 41, ny: int = 41) -> dict:
    """delta(x, y) on the coupon: x across width (cm, centred at 0), y along height (cm).

    If p.leveque, delta(x) ~ (x - x0 + l0)^(1/3) along +x with leading edge
    x0 = -w/2 and l0 = 0.05*w regularisation, normalised so its area mean
    equals delta_mean_um; uniform in y.
    """
    p = p or MassTransferParams()
    mean_um = delta_mean_um(rpm, p, baffles=baffles)
    w, h = geom.coupon_w, geom.coupon_h
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


__all__ = ["MassTransferParams", "TOUABI_RPM", "TOUABI_KM", "delta_mean_um", "delta_map"]
