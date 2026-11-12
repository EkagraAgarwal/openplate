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
  deposition on a random subset of columns - the hook for additive inhibition.

This is the engine + validation only; the continuum model is not coupled here.
``run_sos`` returns the final heights, W(t) = std(h), the mean height, the
event counts and the KMC time.

Entry point::

    from platesim.kmc import deposition_rate, run_sos
    F = deposition_rate(0.015)          # 15 mA/cm2 -> 1/s per surface site
    res = run_sos(L=256, F=F, coverage_ML=50.0, snapshots=8, seed=0)
"""
from __future__ import annotations

from .sos import ASSUMPTIONS, interface_width, run_sos
from .units import (
    E_CHARGE_C, K_B_EV, N_SITES_CM2, deposition_rate, deposition_rate_mA, hop_rate,
)

__all__ = [
    "ASSUMPTIONS",
    "E_CHARGE_C",
    "K_B_EV",
    "N_SITES_CM2",
    "deposition_rate",
    "deposition_rate_mA",
    "hop_rate",
    "interface_width",
    "run_sos",
]
