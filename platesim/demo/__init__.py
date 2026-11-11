"""platesim.demo: tiny end-to-end test plug-in for the server.

Model id: "demo_faraday". Computes the Faraday planar deposit thickness via
platesim.core.constants.thickness_um, sleeps ~2 s in 10 steps while reporting
progress (so the desktop app can exercise progress bars), and returns one of
each plot kind (line / heatmap / contours) to exercise the plot renderers.
"""
from __future__ import annotations

import math
import time

from ..core.constants import thickness_um

MODEL_SPEC = {
    "id": "demo_faraday",
    "title": "Demo: Faraday planar thickness",
    "description": (
        "Toy model for end-to-end server/app tests: Faraday thickness of a planar "
        "Cu deposit with a progress report, a thickness-vs-time line plot, a toy "
        "uniformity heatmap, and a 3-frame contour plot. Runs for ~2 s."
    ),
    "params": [
        {"name": "j_mA_cm2", "type": "float", "default": 20.0, "min": 0.1, "max": 100.0,
         "unit": "mA/cm2", "help": "Applied current density"},
        {"name": "t_min", "type": "float", "default": 10.0, "min": 0.1, "max": 480.0,
         "unit": "min", "help": "Plating time"},
        {"name": "efficiency", "type": "float", "default": 0.95, "min": 0.0, "max": 1.0,
         "unit": "-", "help": "Current efficiency (fraction of current that plates Cu)"},
    ],
}

_STEPS = 10
_TOTAL_SLEEP_S = 2.0


def run(params: dict, progress=lambda frac, msg="": None) -> dict:
    j = float(params["j_mA_cm2"])
    t_min = float(params["t_min"])
    eff = float(params["efficiency"])
    t_s = t_min * 60.0

    # --- progress-reporting "simulation" -----------------------------------
    times: list[float] = []
    thickness: list[float] = []
    for i in range(_STEPS):
        time.sleep(_TOTAL_SLEEP_S / _STEPS)
        frac = (i + 1) / _STEPS
        times.append(t_s * frac)
        thickness.append(thickness_um(j, t_s * frac, eff))
        progress(frac, f"plating {frac * 100:.0f}%")

    final_um = thickness_um(j, t_s, eff)
    charge_mC_cm2 = j * t_s  # mA/cm2 * s = mC/cm2

    # --- plots --------------------------------------------------------------
    line_plot = {
        "kind": "line",
        "title": "Faraday thickness vs time",
        "x": times,
        "series": [{"name": "thickness_um", "y": thickness}],
        "xlabel": "t (s)",
        "ylabel": "thickness (um)",
    }

    # Toy uniformity map: radial falloff of thickness around the wafer centre.
    n = 9
    span_um = 400.0
    xs = [-span_um + 2 * span_um * i / (n - 1) for i in range(n)]
    ys = list(xs)
    z = [
        [final_um * (1.0 - 0.04 * ((x / span_um) ** 2 + (y / span_um) ** 2)) for x in xs]
        for y in ys
    ]
    mean_um = sum(sum(row) for row in z) / (n * n)
    heatmap = {
        "kind": "heatmap",
        "title": "Toy uniformity map",
        "x": xs,
        "y": ys,
        "z": z,
        "zlabel": "thickness (um)",
    }

    # Contours: three frames of expanding rings of constant thickness.
    frames = []
    for frac in (1 / 3, 2 / 3, 1.0):
        lines = []
        for r0 in (80.0, 180.0, 280.0):
            r = r0 * (0.85 + 0.15 * frac)
            ring = [
                [r * math.cos(2 * math.pi * k / 32), r * math.sin(2 * math.pi * k / 32)]
                for k in range(32)
            ]
            lines.append(ring)
        frames.append({"t": t_s * frac, "lines": lines})
    contours = {
        "kind": "contours",
        "title": "Toy iso-thickness contours",
        "frames": frames,
    }

    return {
        "summary": {
            "thickness_um": final_um,
            "mean_thickness_um": mean_um,
            "j_mA_cm2": j,
            "t_min": t_min,
            "t_s": t_s,
            "efficiency": eff,
            "charge_mC_cm2": charge_mC_cm2,
        },
        "plots": [line_plot, heatmap, contours],
        "warnings": ["demo model: Faraday law only, no transport or additive effects"],
        "assumptions": [
            "uniform current density over the mapped area",
            "constant current efficiency over time",
            "no bath depletion or heating",
        ],
    }
