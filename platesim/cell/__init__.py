"""platesim.cell: 3-D coupon-scale current distribution in the beaker plating cell
(Cu electroplating, secondary + mass-transfer correction = pseudo-tertiary).

Model id "cell_current". solver.solve_cell() does the FEM work (gmsh mesh + scikit-fem
P1 Newton); this file exposes the server plug-in contract (MODEL_SPEC + run).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.bath import Bath
from ..core.cell import CellGeometry
from .solver import (ASSUMPTIONS, CellResult, MarchResult, MarchStage, march_cell,
                     solve_cell, solve_duct)

MODEL_SPEC = {
    "id": "cell_current",
    "title": "Coupon current distribution (3-D beaker cell)",
    "description": (
        "Secondary current distribution with a Nernst-layer mass-transfer correction "
        "in a full 3-D beaker geometry (gmsh mesh, scikit-fem P1, Newton on "
        "Butler-Volmer boundaries). Returns j(x,y) and thickness maps on the coupon, "
        "uniformity metrics, cell voltage decomposition and the untaped back/edge "
        "current fraction."
    ),
    "params": [
        {"name": "j_mA_cm2", "type": "float", "default": 12.0, "min": 0.1, "max": 100.0,
         "unit": "mA/cm2", "help": "target mean current density on the coupon front face "
                                   "(I_total = j * coupon_area)."},
        {"name": "t_min", "type": "float", "default": 60.0, "min": 1.0, "max": 1440.0,
         "unit": "min", "help": "plating time for the thickness map."},
        {"name": "T_C", "type": "float", "default": 22.0, "min": 10.0, "max": 45.0,
         "unit": "degC", "help": "bath temperature (kappa and i0 are rescaled)."},
        {"name": "efficiency", "type": "float", "default": 1.0, "min": 0.1, "max": 1.5,
         "unit": "-", "help": "current efficiency applied to the Faraday thickness map."},
        {"name": "delta_um", "type": "float", "default": 60.0, "min": 0.0, "max": 500.0,
         "unit": "um", "help": "constant Nernst-layer thickness (0 disables the "
                               "mass-transfer correction)."},
        {"name": "suppressed", "type": "bool", "default": True,
         "help": "use the additive-suppressed exchange current (Bath.i0_cu_supp); "
                 "False uses the unsuppressed Braun value."},
        {"name": "taped", "type": "bool", "default": True,
         "help": "True: only the coupon front face plates. False: back + 4 sidewalls "
                 "(0.05 cm Si) also carry current."},
        {"name": "stray_area_cm2", "type": "float", "default": 0.0, "min": 0.0, "max": 5.0,
         "unit": "cm2", "help": "lumped wet-clip conductor area added as a patch above the "
                                "coupon top edge (front face only)."},
        {"name": "gap_cm", "type": "float", "default": 4.0, "min": 0.5, "max": 8.0,
         "unit": "cm", "help": "anode-cathode face distance."},
        {"name": "anode_w_cm", "type": "float", "default": 5.0, "min": 1.0, "max": 9.0,
         "unit": "cm", "help": "anode width (clipped by the beaker wall if it does not fit); "
                               "anode height = width."},
        {"name": "coupon_w_cm", "type": "float", "default": 2.0, "min": 0.5, "max": 6.0,
         "unit": "cm", "help": "coupon plated width."},
        {"name": "coupon_h_cm", "type": "float", "default": 2.0, "min": 0.5, "max": 6.0,
         "unit": "cm", "help": "coupon plated height."},
        {"name": "kappa_S_cm", "type": "float", "default": 0.065, "min": 0.001, "max": 1.0,
         "unit": "S/cm", "help": "electrolyte conductivity at the run temperature."},
        {"name": "resolution", "type": "enum", "choices": ["coarse", "medium", "fine"],
         "default": "coarse", "help": "FEM mesh resolution (coarse ~ few seconds, "
                                      "fine can take tens of seconds)."},
    ],
}


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    p = params or {}
    j_mA_cm2 = float(p.get("j_mA_cm2", 12.0))
    t_min = float(p.get("t_min", 60.0))
    T_C = float(p.get("T_C", 22.0))
    efficiency = float(p.get("efficiency", 1.0))
    delta_um = float(p.get("delta_um", 60.0))
    suppressed = bool(p.get("suppressed", True))
    taped = bool(p.get("taped", True))
    stray_area = float(p.get("stray_area_cm2", 0.0))
    resolution = str(p.get("resolution", "coarse"))
    kappa_S_cm = float(p.get("kappa_S_cm", 0.065))

    geom = CellGeometry(
        gap=float(p.get("gap_cm", 4.0)),
        anode_w=float(p.get("anode_w_cm", 5.0)),
        anode_h=float(p.get("anode_w_cm", 5.0)),
        coupon_w=float(p.get("coupon_w_cm", 2.0)),
        coupon_h=float(p.get("coupon_h_cm", 2.0)),
        taped=taped,
        stray_area=stray_area,
    )
    bath = Bath().at(T_C)
    # kappa_S_cm is the conductivity AT the run temperature -> back out kappa_22 [C]
    bath = replace(bath, kappa_22=kappa_S_cm / (1.0 + bath.kappa_per_K * (T_C - 22.0)))

    I_total_A = j_mA_cm2 * geom.coupon_area / 1e3
    res = solve_cell(geom, bath, I_total_A, t_s=t_min * 60.0, efficiency=efficiency,
                     delta_um=delta_um, suppressed=suppressed, resolution=resolution,
                     progress=progress)

    # ---- plots
    x = [round(float(v), 4) for v in res.x_cm]
    z = [round(float(v), 4) for v in res.z_cm]
    heat_j = {"kind": "heatmap", "title": "Current density on coupon front face",
              "x": x, "y": z,
              "z": [[round(float(v), 4) for v in row] for row in res.j_mA_cm2],
              "zlabel": "j (mA/cm2)"}
    heat_th = {"kind": "heatmap", "title": f"Deposit thickness after {t_min:g} min "
                                           f"({100 * efficiency:g} % efficiency)",
               "x": x, "y": z,
               "z": [[round(float(v), 4) for v in row] for row in res.thickness_um],
               "zlabel": "thickness (um)"}
    iz = len(z) // 2
    ix = len(x) // 2
    # both centrelines share a normalised x in [-1, 1] (fraction of half-width/height),
    # so horizontal and vertical profiles overlay even when coupon_w != coupon_h
    xn = [round(float(v), 4) for v in np.linspace(-1.0, 1.0, len(z))]
    horiz = [round(float(v), 4) for v in res.j_mA_cm2[iz, :]]
    vert = [round(float(v), 4) for v in res.j_mA_cm2[:, ix]]
    line = {
        "kind": "line", "title": "j along the coupon centrelines",
        "x": xn,
        "series": [
            {"name": "horizontal (z = coupon centre)", "y": horiz},
            {"name": "vertical (y = 0)", "y": vert},
        ],
        "xlabel": "position from centre (fraction of half-size)", "ylabel": "j (mA/cm2)",
    }

    warnings = list(res.warnings)
    assumptions = list(res.assumptions)

    def r(v, n=4):
        return None if isinstance(v, float) and v != v else round(float(v), n)

    summary = {
        "I_total_mA": round(I_total_A * 1e3, 3),
        "V_cell_V": r(res.V_cell_V),
        "V_ohm_V": r(res.V_ohm_V),
        "eta_c_V": r(res.eta_c_V),
        "eta_a_V": r(res.eta_a_V),
        "Wa": r(res.Wa, 3),
        "j_mean_mA_cm2": r(res.j_mean_mA_cm2),
        "j_max_mA_cm2": r(res.j_max_mA_cm2),
        "j_min_mA_cm2": r(res.j_min_mA_cm2),
        "std_over_mean_pct": r(100 * res.std_over_mean, 2),
        "edge_center_ratio": r(res.edge_center_ratio, 3),
        "corner_LR_ratio": r(res.corner_LR_ratio, 3),
        "corner_LR_grad_pct": r(res.corner_LR_grad_pct, 2),
        "front_current_fraction": r(res.front_fraction, 4),
        "back_edge_stray_fraction": r(res.loss_fraction, 4),
        "anode_j_mA_cm2": r(res.anode_j_mA_cm2),
        "thickness_mean_um": r(float(res.thickness_um.mean()), 3),
        "thickness_max_um": r(float(res.thickness_um.max()), 3),
        "thickness_min_um": r(float(res.thickness_um.min()), 3),
        "thickness_nonuniformity_pct": r(
            100 * float(res.thickness_um.std() / max(res.thickness_um.mean(), 1e-30)), 2),
        "anode_area_cm2": r(res.anode_area_cm2, 2),
        "cathode_area_cm2": r(res.cathode_area_cm2, 2),
        "n_nodes": res.n_nodes,
        "n_elems": res.n_elems,
        "n_newton": res.n_newton,
        "mesh_time_s": r(res.mesh_time_s, 2),
        "solve_time_s": r(res.solve_time_s, 2),
        "domain": res.domain,
    }
    return {"summary": summary, "plots": [heat_j, heat_th, line],
            "warnings": warnings, "assumptions": assumptions}


__all__ = ["MODEL_SPEC", "run", "solve_cell", "solve_duct", "march_cell", "CellResult",
           "MarchResult", "MarchStage", "ASSUMPTIONS"]
