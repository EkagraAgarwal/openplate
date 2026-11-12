"""platesim.kmc: minimal lattice kinetic Monte Carlo for interface roughness (E3a).

A 1+1-D solid-on-solid (SOS) lattice with periodic boundary conditions: the
interface is a column-height vector ``h[i]``.  Rejection-free BKL / n-fold-way
dynamics with exact exponential waiting times:

* deposition, rate ``F`` per free site, ``F = j / (2 e N_s)`` from the current
  density (``units.deposition_rate``);
* surface diffusion, rate ``nu0 * exp(-(E_d + n E_b) / kT)``, where ``n`` in
  {0, 1, 2} counts the in-plane nearest neighbours of the moving atom at its
  current height (Clarke-Vvedensky-type bond counting, 1+1 D);
* optional static suppressor mask (coverage ``theta_block``) that blocks
  deposition on a random subset of columns;
* optional dynamic additive layer (``additive=Additive(...)``, E3b): suppressor /
  accelerator / chloride adsorption, desorption, chloride-mediated co-adsorption
  and burial by growth, all in the same BKL bookkeeping (see
  :mod:`platesim.kmc.additive`).  ``effective_rates`` maps a run back onto the
  continuum coverage equation (``k_plus``, ``k_minus``).

``run_sos`` returns the final heights, W(t) = std(h), the mean height, the
event counts and the KMC time; with the additive layer it also returns the
coverage time series and event tallies.

Entry point::

    from platesim.kmc import deposition_rate, run_sos
    F = deposition_rate(0.015)          # 15 mA/cm2 -> 1/s per surface site
    res = run_sos(L=256, F=F, coverage_ML=50.0, snapshots=8, seed=0)
"""
from __future__ import annotations

from .additive import Additive, effective_rates
from .sos import ASSUMPTIONS, interface_width, run_sos
from .units import (
    E_CHARGE_C,
    K_B_EV,
    MONOLAYER_CM,
    N_A,
    N_SITES_CM2,
    deposition_rate,
    deposition_rate_mA,
    hop_rate,
    monolayer_thickness_cm,
)

__all__ = [
    "ASSUMPTIONS",
    "E_CHARGE_C",
    "K_B_EV",
    "MONOLAYER_CM",
    "N_A",
    "N_SITES_CM2",
    "Additive",
    "deposition_rate",
    "deposition_rate_mA",
    "effective_rates",
    "hop_rate",
    "interface_width",
    "monolayer_thickness_cm",
    "run_sos",
]
