"""Strict, engine-independent 3-D input and provenance models.

Feature coordinates are centres in die coordinates (origin at the lower left).
Unspecified physical inputs are required, not inferred from a bath name.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


def invalid(model, path, value, message):
    """Keep cross-field failures attached to an actionable field, not the root."""
    raise ValidationError.from_exception_data(type(model).__name__, [{
        "type": "value_error", "loc": tuple(path), "input": value,
        "ctx": {"error": ValueError(message)},
    }])


class FeatureArray(StrictModel):
    nx: Annotated[int, Field(gt=0, strict=True)]
    ny: Annotated[int, Field(gt=0, strict=True)]
    pitch_x_um: Positive
    pitch_y_um: Positive


class Feature(StrictModel):
    kind: Literal["rect", "rounded_rect", "circle"]
    x_um: float
    y_um: float
    w_um: Positive
    h_um: Positive
    r_um: Nonnegative = 0
    rotation_deg: float = 0
    layer: str = "etch"
    array: FeatureArray | None = None

    @model_validator(mode="after")
    def geometry(self):
        if self.r_um > min(self.w_um, self.h_um) / 2:
            invalid(self, ["r_um"], self.r_um, "radius must be at most half the smaller size")
        if self.kind == "circle" and self.w_um != self.h_um:
            invalid(self, ["h_um"], self.h_um, "circle height must equal its diameter w_um")
        return self


class Step(StrictModel):
    model_config = ConfigDict(json_schema_extra={"oneOf": [
        {"required": ["duration_h"], "properties": {
            "duration_h": {"type": "number"}, "charge_C_cm2": {"type": "null"}}},
        {"required": ["charge_C_cm2"], "properties": {
            "duration_h": {"type": "null"}, "charge_C_cm2": {"type": "number"}}},
    ]})
    t_forward_ms: Positive
    t_reverse_ms: Nonnegative = 0
    t_off_ms: Nonnegative = 0
    j_forward_mA_cm2: Positive
    j_reverse_mA_cm2: Nonnegative = 0
    duration_h: Positive | None = None
    charge_C_cm2: Positive | None = None

    @property
    def j_net_mA_cm2(self) -> float:
        """Cathodic-positive cycle average; reverse current is a positive magnitude."""
        period = self.t_forward_ms + self.t_reverse_ms + self.t_off_ms
        return (self.j_forward_mA_cm2 * self.t_forward_ms
                - self.j_reverse_mA_cm2 * self.t_reverse_ms) / period

    @property
    def duty(self) -> float:
        """Forward duty fraction, consistent with engine.recipes.pr_forward."""
        return self.t_forward_ms / (self.t_forward_ms + self.t_reverse_ms + self.t_off_ms)

    @model_validator(mode="after")
    def waveform(self):
        if (self.duration_h is None) == (self.charge_C_cm2 is None):
            invalid(self, ["charge_C_cm2"], self.charge_C_cm2,
                    "provide exactly one of duration_h or charge_C_cm2")
        if not math.isfinite(self.j_net_mA_cm2) or self.j_net_mA_cm2 <= 0:
            invalid(self, ["j_reverse_mA_cm2"], self.j_reverse_mA_cm2,
                    "net current must be positive; reduce reverse charge or increase forward charge")
        return self


class Recipe(StrictModel):
    steps: Annotated[list[Step], Field(min_length=1)]


class Bath(StrictModel):
    name: str = "EP-EC1-65L"
    cu_g_L: Positive
    h2so4_g_L: Nonnegative
    cl_ppm: Nonnegative
    temperature_C: Annotated[float, Field(ge=15, le=60)]


class Setup(StrictModel):
    orientation: Literal["vertical", "face_up", "face_down"] = "vertical"
    up_edge: Literal["+x", "-x", "+y", "-y"] = "+y"
    stir_shear_1_s: Nonnegative | None = None
    stir_angle_deg: float = 0
    coupon_position_cm: tuple[float, float] | None = None  # centred coupon: x across, y up
    stir_pct: Nonnegative | None = None  # coupon model only; None uses bridge default
    delta_um: Positive = 64
    taped: bool
    area_cm2: Positive


class Numerics(StrictModel):
    quality: Literal["preview", "standard", "fine"] = "standard"
    physics: Literal["L1", "L2", "L3"] = "L1"
    dx_um: Positive | None = None
    device: Literal["auto", "cpu"] | Annotated[int, Field(ge=0, strict=True)] = "auto"
    use_surrogate: bool = False


class Layout(StrictModel):
    file: Annotated[str, Field(min_length=1)] | None = None
    features: Annotated[list[Feature], Field(min_length=1)] | None = None
    die_um: tuple[Positive, Positive]
    pixel_um: Positive | None = None
    layer_map: dict[str, int | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def geometry(self):
        if (self.file is None) == (self.features is None):
            invalid(self, ["features"], self.features, "provide exactly one of file or features")
        for i, feature in enumerate(self.features or []):
            angle = math.radians(feature.rotation_deg % 360)
            c, s = abs(math.cos(angle)), abs(math.sin(angle))
            if feature.kind == "circle":
                rx = ry = feature.w_um / 2
            else:
                r = feature.r_um if feature.kind == "rounded_rect" else 0
                rx = c * (feature.w_um / 2 - r) + s * (feature.h_um / 2 - r) + r
                ry = s * (feature.w_um / 2 - r) + c * (feature.h_um / 2 - r) + r
            a = feature.array
            for axis, center, radius, extent, count, pitch in [
                ("x_um", feature.x_um, rx, self.die_um[0], a.nx if a else 1, a.pitch_x_um if a else 0),
                ("y_um", feature.y_um, ry, self.die_um[1], a.ny if a else 1, a.pitch_y_um if a else 0),
            ]:
                if center - radius < -1e-9 or center + (count - 1) * pitch + radius > extent + 1e-9:
                    invalid(self, ["features", i, axis], center,
                            "all feature and array bounds must lie inside die_um; move or shrink the feature")
        return self


class EtchLayer(StrictModel):
    depth_um: Positive
    sidewall_deg: Annotated[float, Field(ge=60, le=90)] = 90
    bottom_radius_um: Nonnegative = 0
    depth_vs_width: list[tuple[Positive, Positive]] | None = None
    seed: Literal["full", "thinning"] = "full"

    @model_validator(mode="after")
    def profile(self):
        if self.bottom_radius_um > self.depth_um:
            invalid(self, ["bottom_radius_um"], self.bottom_radius_um, "radius must not exceed depth_um")
        if self.depth_vs_width is not None:
            widths = [p[0] for p in self.depth_vs_width]
            if not widths or any(b <= a for a, b in zip(widths, widths[1:])):
                invalid(self, ["depth_vs_width"], self.depth_vs_width,
                        "provide nonempty width/depth pairs with strictly increasing widths")
        return self


class Etch(StrictModel):
    default: EtchLayer
    layers: dict[str, EtchLayer] = Field(default_factory=dict)


class SampleSpec(StrictModel):
    layout: Layout
    etch: Etch
    recipe: Recipe
    bath: Bath
    setup: Setup
    numerics: Numerics = Field(default_factory=Numerics)


class Quantity(StrictModel):
    name: str
    value: float
    unit: str
    uncertainty: Nonnegative | None = None
    flag: Literal["calibrated", "uncalibrated", "extrapolated", "interpolated", "surrogate", "preview"]
    feature_id: str | None = None


class Check(StrictModel):
    name: str
    value: float
    limit: float
    passed: bool


class Artifact(StrictModel):
    kind: Literal["png", "gif", "mp4", "svg", "vti", "vtp", "stl", "gltf", "csv", "json", "npz"]
    path: str
    title: str


class Manifest(StrictModel):
    spec_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    code_version: str
    backend: str
    device: Literal["auto", "cpu"] | Annotated[int, Field(ge=0, strict=True)]
    dx_um: Positive
    physics: Literal["L1", "L2", "L3"]
    created_utc: datetime
    quantities: list[Quantity] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    waveform_sensitivity: str
    artifacts: list[Artifact] = Field(default_factory=list)
    summary_text: str

    @model_validator(mode="after")
    def utc(self):
        offset = self.created_utc.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            invalid(self, ["created_utc"], self.created_utc, "use a timezone-aware UTC timestamp (Z or +00:00)")
        return self
