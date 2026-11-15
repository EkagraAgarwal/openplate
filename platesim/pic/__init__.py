"""platesim.pic: multi-ion particle-in-cell Brownian dynamics and its 1-D continuum reference.

Research track (plan v2 item E4).  This is a *validation* tool for the continuum Nernst-Planck
transport model, not a production solver: every particle stands for 10^6-10^8 ions and the noise
falls only as 1/sqrt(N).

Entry points::

    from platesim.pic import Ion, default_bath, run_pic, continuum_flux

    ions = default_bath()                       # Cu2+/H+/SO4^2- at 22 C, Cu2+ reacting
    ref  = continuum_flux(ions, L_um=8)         # steady 1-D NP reference: flux + profiles
    pic  = run_pic(ions, L_um=8, seed=0)        # super-particle dynamics with a self-consistent
                                                # quasi-neutral potential
    pic.flux["Cu2+"], pic.flux_se["Cu2+"]       # mol/cm2/s and its Poisson standard error
    pic.charge_rel                              # bulk electroneutrality residual

See :mod:`platesim.pic.ions` for the model, the potential closure and the analytic limits.
"""
from __future__ import annotations

from .ions import (
    ContinuumResult,
    Ion,
    PICResult,
    T_DEFAULT_C,
    continuum_flux,
    default_bath,
    run_pic,
)

__all__ = [
    "Ion",
    "PICResult",
    "ContinuumResult",
    "default_bath",
    "run_pic",
    "continuum_flux",
    "T_DEFAULT_C",
]
