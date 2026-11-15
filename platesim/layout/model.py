"""Geometry containers. Coordinates and bounds are in micrometres."""
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy


@dataclass(frozen=True)
class Feature:
    id: str
    layer: str
    polygon: Any

    @property
    def bounds_um(self):
        return self.polygon.bounds


@dataclass
class Layout:
    features: list[Feature]
    die_outline: Any
    skipped_entities: list[dict] = field(default_factory=list)
    _source: dict | None = field(default=None, repr=False)

    @property
    def layers(self):
        result = {}
        for feature in self.features:
            result.setdefault(feature.layer, []).append(feature.polygon)
        return result

    @property
    def bounds_um(self):
        return self.die_outline.bounds

    def geometry(self, layer):
        from shapely.ops import unary_union
        return unary_union(self.layers.get(layer, []))

    def to_spec(self):
        """Return an independent copy of the original parametric description."""
        if self._source is None:
            raise ValueError("Imported polygons have no parametric spec; retain the source file")
        return deepcopy(self._source)


def make_layout(records, die_outline=None, skipped=None, source=None):
    """Expand polygon collections into independently identified instances."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    features = []
    for layer, geometry in records:
        if geometry.is_empty:
            continue
        parts = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            if not part.is_valid:
                raise ValueError(f"Invalid polygon on layer {layer!r}")
            features.append(Feature(f"{layer}:{len(features)}", str(layer), part))
    if die_outline is None:
        union = unary_union([f.polygon for f in features])
        die_outline = box(*union.bounds) if not union.is_empty else box(0, 0, 0, 0)
    return Layout(features, die_outline, skipped or [], deepcopy(source))
