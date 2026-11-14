"""DXF model-space coordinates are assumed to be micrometres (no implicit INSUNITS)."""
from .model import make_layout


def import_dxf(path, layer_map):
    import ezdxf
    from ezdxf.path import make_path
    from shapely.geometry import Polygon, Point
    doc = ezdxf.readfile(str(path))
    records, skipped = [], []
    for entity in doc.modelspace():
        layers = [name for name, source in layer_map.items() if source == entity.dxf.layer]
        if not layers:
            continue
        kind = entity.dxftype()
        reason = None
        polygon = None
        if kind == "CIRCLE":
            center = entity.ocs().to_wcs(entity.dxf.center)
            polygon = Point(center.x, center.y).buffer(entity.dxf.radius, quad_segs=256)
        elif kind == "LWPOLYLINE" and entity.closed:
            # ezdxf's path conversion preserves bulge arcs rather than dropping them.
            points = list(make_path(entity).flattening(distance=1e-4, segments=16))
            polygon = Polygon([(p.x, p.y) for p in points])
        else:
            reason = "open polyline" if kind == "LWPOLYLINE" else "unsupported entity"
        if reason is None and (polygon is None or polygon.is_empty or not polygon.is_valid):
            reason = "invalid or empty polygon"
        if reason:
            skipped.append(dict(type=kind, handle=entity.dxf.handle,
                                layer=entity.dxf.layer, reason=reason))
        else:
            records.extend((layer, polygon) for layer in layers)
    return make_layout(records, skipped=skipped)
