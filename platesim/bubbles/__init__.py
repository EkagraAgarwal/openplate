"""platesim.bubbles: gas-bubble physics and a one-way Lagrangian bubble tracker (plan item E2a).

Gas bubbles -- H2 from the side reaction, or air left in a trench after pre-wet -- stick to the
electrode, shade it and cause pits. This package is the physics half of that story only; nothing is
coupled back into the plating engine yet.

``platesim.bubbles.physics`` (correlations, each a small documented function):
    rise_velocity               Stokes rigid-sphere and Hadamard-Rybczynski rise velocities + Re flag
    fritz_detachment_radius_um  Fritz detachment radius from the contact angle
    laplace_overpressure_Pa     capillary overpressure 2 sigma / R
    henry_concentration_mol_m3  Henry solubility (H2 in water default)
    epstein_plesset_rate_m_s    quasi-steady diffusive growth of a stationary bubble
    diffusive_growth_rate_um_s  the same with the Laplace pressure and Henry's law filled in
    contact_disc_radius_um      R sin(theta), the footprint of an attached bubble
    shading_fraction            fraction of an electrode strip hidden by attached bubbles

``platesim.bubbles.tracker``:
    track_bubbles               point bubbles in a frozen FlowField: advection + slip + growth +
                                adhesion/detachment, x periodic

Typical use::

    from platesim.bubbles import track_bubbles
    from platesim.trenchflow.geometry import trench_grid
    from platesim.trenchflow.stokes import solve_stokes

    grid = trench_grid(width_um=80, depth_um=80, dx_um=4.0)
    flow = solve_stokes(grid, 80.0)
    res = track_bubbles(flow, R0_um=5.0, x0_um=200.0, z0_um=-40.0, t_end_s=2.0, dt_s=0.02,
                        gravity_dir=(0.0, 0.0))

Units: um and s for geometry/time, cm/s for the flow field (the platesim.trenchflow convention),
SI inside the correlations (m, m/s, Pa, mol/m3).
"""
from __future__ import annotations

from .physics import (
    D_H2_WATER_M2_S,
    DRHO_KG_M3,
    FRITZ_PREFACTOR_DEG,
    G_ACCEL_M_S2,
    K_H_H2_MOL_L_ATM,
    MU_LIQUID_PA_S,
    P_ATM_PA,
    RHO_LIQUID_KG_M3,
    SIGMA_WATER_N_M,
    contact_disc_radius_um,
    diffusive_growth_rate_um_s,
    epstein_plesset_rate_m_s,
    fritz_detachment_radius_um,
    henry_concentration_mol_m3,
    ideal_gas_density_mol_m3,
    laplace_overpressure_Pa,
    rise_velocity,
    shading_fraction,
)
from .tracker import track_bubbles

__all__ = [
    "D_H2_WATER_M2_S",
    "DRHO_KG_M3",
    "FRITZ_PREFACTOR_DEG",
    "G_ACCEL_M_S2",
    "K_H_H2_MOL_L_ATM",
    "MU_LIQUID_PA_S",
    "P_ATM_PA",
    "RHO_LIQUID_KG_M3",
    "SIGMA_WATER_N_M",
    "contact_disc_radius_um",
    "diffusive_growth_rate_um_s",
    "epstein_plesset_rate_m_s",
    "fritz_detachment_radius_um",
    "henry_concentration_mol_m3",
    "ideal_gas_density_mol_m3",
    "laplace_overpressure_Pa",
    "rise_velocity",
    "shading_fraction",
    "track_bubbles",
]
