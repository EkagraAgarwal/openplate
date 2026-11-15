"""Parametric features use centre coordinates; arrays translate after rotation."""
from pathlib import Path
import math

from .model import make_layout


def load_layout(spec_layout):
    from shapely.geometry import Point, box
    from shapely.affinity import rotate, translate
    spec = dict(spec_layout)
    if spec.get("file"):
        path = Path(spec["file"])
        suffix = path.suffix.lower()
        if suffix in (".gds", ".gdsii"):
            from .import_gds import import_gds
            result = import_gds(path, spec.get("layer_map", {}))
        elif suffix == ".dxf":
            from .import_dxf import import_dxf
            result = import_dxf(path, spec.get("layer_map", {}))
        elif suffix in (".png", ".tif", ".tiff", ".webp"):
            from .import_image import import_image
            result = import_image(path, spec.get("pixel_um"))
        else:
            raise ValueError(f"layout.file={str(path)!r}: use PNG/TIFF/WEBP, GDS or DXF")
        if spec.get("die_um"):
            result.die_outline = box(0, 0, *spec["die_um"])
        return result
    wdie, hdie = spec["die_um"]
    if not all(math.isfinite(v) and v > 0 for v in (wdie, hdie)):
        raise ValueError("layout.die_um: require finite positive width and height")
    records = []
    for index, f in enumerate(spec.get("features") or []):
        kind = f["kind"]
        radius = f.get("r_um") or 0
        if kind == "circle":
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError(f"layout.features[{index}].r_um={radius}: use > 0")
            shape = Point(0, 0).buffer(radius, quad_segs=128)
        elif kind in ("rect", "rounded_rect"):
            w, h = f["w_um"], f["h_um"]
            if not all(math.isfinite(v) and v > 0 for v in (w, h)):
                raise ValueError(f"layout.features[{index}]: w_um and h_um must be > 0")
            if kind == "rect":
                shape = box(-w/2, -h/2, w/2, h/2)
            else:
                if not 0 <= radius <= min(w, h)/2:
                    raise ValueError(f"layout.features[{index}].r_um: use 0..min(w,h)/2")
                # Buffer the inset polygon; degenerate cores need a line/point.
                from shapely.geometry import LineString
                x, y = w/2-radius, h/2-radius
                core = (box(-x, -y, x, y) if x and y else
                        LineString([(-x, -y), (x, y)]) if x or y else Point(0, 0))
                shape = core.buffer(radius, quad_segs=128) if radius else core
        else:
            raise ValueError(f"layout.features[{index}].kind={kind!r}: use rect/rounded_rect/circle")
        shape = rotate(shape, f.get("rotation_deg") or 0, origin=(0, 0))
        array = f.get("array") or {}
        nx, ny = array.get("nx", 1), array.get("ny", 1)
        if any(not isinstance(v, int) or v <= 0 for v in (nx, ny)):
            raise ValueError("layout.features.array: nx and ny must be positive integers")
        for i in range(nx):
            for j in range(ny):
                x = f.get("x_um", 0)+i*array.get("pitch_x_um", 0)
                y = f.get("y_um", 0)+j*array.get("pitch_y_um", 0)
                records.append((f.get("layer", "etch"), translate(shape, x, y)))
    return make_layout(records, box(0, 0, wdie, hdie), source=spec)
