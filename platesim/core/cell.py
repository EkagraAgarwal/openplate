"""Beaker-cell geometry. Defaults are ASSUMPTIONS (marked) until the team measures the cell; only the ~4 cm2
coupon is user-confirmed (2026-10-28). Lengths in cm."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class CellGeometry:
    beaker_id: float = 9.0            # cm inner diameter [J] ~600 mL beaker
    fill_height: float = 8.0          # cm liquid height [J]
    coupon_w: float = 2.0             # cm, plated (untaped) width; 2 x 2 = 4 cm2 [M, user]
    coupon_h: float = 2.0
    coupon_center_z: float = 4.0      # cm above beaker floor [J]
    anode_w: float = 5.0              # cm [J] KB section 7 proposal
    anode_h: float = 5.0
    gap: float = 4.0                  # cm anode-cathode face distance [J] KB "3-5 cm"
    taped: bool = True                # back and edges taped (W9 improvement #1)
    stray_area: float = 0.0           # cm2 of un-taped wetted conductor (clip, edges, back) [J]
    stir_bar_len: float = 3.8         # cm [J]
    stir_rpm: float = 300.0           # rpm [J]; W9 stir % -> rpm mapping unknown
    n_baffles: int = 4                # [M] baffles present; count/size [J]
    baffle_width: float = 0.8         # cm [J] ~ T/10-T/12 (IS 9522)

    @property
    def coupon_area(self) -> float:
        return self.coupon_w * self.coupon_h

    @property
    def anode_area(self) -> float:
        return self.anode_w * self.anode_h

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "CellGeometry":
        return cls(**json.loads(Path(path).read_text()))


# W9 stir setting (% of the plate dial) -> rpm: UNKNOWN; linear guess, calibration target
STIR_PCT_TO_RPM = {20: 300.0, 30: 450.0, 40: 600.0}
