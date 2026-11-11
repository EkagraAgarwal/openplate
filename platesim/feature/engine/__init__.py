"""Level-set feature-fill engine, ported from research/via_fill_sim (read-only reference).

Two geometries via Params.geometry:
  'via'     - 2-D axisymmetric (r, z), numerics identical to the research engine (regression-tested);
  'trench'  - planar 2-D (x, z) per unit length, long trench of width w = d, half-domain x in [0, R],
              every r-weighting replaced by weight 1 (see levelset.Grid).
"""
from .params import Params, Phys, derive
from .recipes import (Recipe, Step, by_name, dc, plan_ramp, plan_ramp_pr, potentiostatic, pulse_reverse,
                      pr_forward, scaled, team_staged)
from .run_case import DEFAULT_OUT_DIR, decode_cfield, run_case
from .sim import ViaSim

__all__ = [
    "Params", "Phys", "derive",
    "Recipe", "Step", "by_name", "dc", "plan_ramp", "plan_ramp_pr", "potentiostatic", "pulse_reverse",
    "pr_forward", "scaled", "team_staged",
    "run_case", "decode_cfield", "DEFAULT_OUT_DIR", "ViaSim",
]
