"""GDS coordinates are converted from library user units to micrometres."""
from .model import make_layout


def import_gds(path, layer_map, unit_um=None):
    import gdstk
    from shapely.geometry import Polygon
    import math
    library = gdstk.read_gds(str(path))
    scale = library.unit*1e6 if unit_um is None else float(unit_um)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"unit_um={unit_um!r}: use finite micrometres per GDS unit > 0")
    records = []
    # get_polygons recursively expands references and repetitions without mutating files.
    for cell in library.top_level():
        for polygon in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            for name, selector in layer_map.items():
                match = ((polygon.layer, polygon.datatype) == tuple(selector)
                         if isinstance(selector, (list, tuple)) else polygon.layer == selector)
                if match:
                    geometry = Polygon(polygon.points*scale)
                    # GDS holes use zero-width cut lines (weakly simple rings).
                    if not geometry.is_valid:
                        geometry = geometry.buffer(0)
                    records.append((name, geometry))
    return make_layout(records)
