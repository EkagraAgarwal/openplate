"""Conservative polygon fitting for the editable spec3d feature vocabulary."""
import math


def _fit(polygon, layer):
    from shapely.geometry import Point, box, LineString
    from shapely.affinity import rotate, translate

    rectangle = polygon.minimum_rotated_rectangle
    vertices = list(rectangle.exterior.coords)
    edges = [(math.dist(a, b), math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])) % 180)
             for a, b in zip(vertices, vertices[1:])]
    w, angle = max(edges, key=lambda e: (round(e[0], 8), -e[1]))
    h = min(e[0] for e in edges)
    if angle > 180 - 1e-8:
        angle = 0.
    x, y = rectangle.centroid.coords[0]
    radius = min(min(w, h)/2, math.sqrt(max(0., w*h-polygon.area)/(4-math.pi)))
    candidates = []
    for kind, r in (("rect", 0.), ("rounded_rect", radius), ("circle", math.sqrt(polygon.area/math.pi))):
        cw, ch, rotation = (2*r, 2*r, 0.) if kind == "circle" else (w, h, angle)
        if kind == "circle":
            shape = Point(0, 0).buffer(r, quad_segs=128)
        else:
            a, b = cw/2-r, ch/2-r
            core = (box(-a, -b, a, b) if a and b else
                    LineString([(-a, -b), (a, b)]) if a or b else Point(0, 0))
            shape = core.buffer(r, quad_segs=128) if r else core
        shape = translate(rotate(shape, rotation, origin=(0, 0)), x, y)
        # Area alone would accept arbitrary concave polygons as equal-area circles.
        # Symmetric difference also checks where that area lies (including holes).
        error = polygon.symmetric_difference(shape).area / polygon.area
        feature = dict(kind=kind, x_um=x, y_um=y, w_um=cw, h_um=ch,
                       r_um=r, rotation_deg=rotation, layer=layer)
        candidates.append((error, feature))
    return min(candidates, key=lambda item: item[0])


def _arrays(features):
    groups = []
    for feature in features:
        signature = {k: v for k, v in feature.items() if k not in ("x_um", "y_um")}
        for key, members in groups:
            if all(math.isclose(v, key[k], rel_tol=1e-5, abs_tol=1e-6)
                   if isinstance(v, (int, float)) else v == key[k]
                   for k, v in signature.items()):
                members.append(feature)
                break
        else:
            groups.append((signature, [feature]))
    output = []
    for _, members in groups:
        axes = []
        for axis in ("x_um", "y_um"):
            values = []
            for value in sorted(f[axis] for f in members):
                if not values or not math.isclose(value, values[-1], abs_tol=1e-5):
                    values.append(value)
            axes.append(values)
        xs, ys = axes
        pitches = [sum(b-a for a, b in zip(v, v[1:]))/(len(v)-1) if len(v) > 1 else 1.
                   for v in axes]
        regular = all(all(math.isclose(b-a, p, rel_tol=.01) for a, b in zip(v, v[1:]))
                      for v, p in zip(axes, pitches))
        cells = {(min(range(len(xs)), key=lambda i: abs(xs[i]-f["x_um"])),
                  min(range(len(ys)), key=lambda i: abs(ys[i]-f["y_um"]))) for f in members}
        if len(members) >= 2 and len(xs)*len(ys) == len(members) == len(cells) and regular:
            feature = dict(members[0], x_um=xs[0], y_um=ys[0],
                           array=dict(nx=len(xs), ny=len(ys), pitch_x_um=pitches[0], pitch_y_um=pitches[1]))
            output.append(feature)
        else:
            output.extend(members)
    return output


def fit_features(layout, layer="etch", tol_rel=0.03) -> dict:
    """Fit a named layer, preserving rejected polygons and importer diagnostics.

    Imported die origins are translated to (0, 0), as required by SampleSpec.
    Preview rings use that same coordinate system, including other named layers.
    """
    from shapely.affinity import translate
    if not math.isfinite(tol_rel) or tol_rel < 0:
        raise ValueError("tol_rel must be finite and nonnegative")
    x0, y0, x1, y1 = layout.bounds_um
    fitted, unfit, preview = [], [], []
    for index, item in enumerate(layout.features):
        polygon = translate(item.polygon, -x0, -y0)
        if len(preview) < 2000:
            preview.append(dict(layer=item.layer, rings_um=[list(map(list, polygon.exterior.coords))]))
        if item.layer != layer:
            continue
        if polygon.area <= 0 or polygon.minimum_rotated_rectangle.geom_type != "Polygon":
            error, feature = 1., None
        else:
            error, feature = _fit(polygon, layer)
        if feature is None or error > tol_rel:
            unfit.append(dict(index=index, area_um2=polygon.area, error_rel=error,
                              bounds_um=list(polygon.bounds)))
        else:
            fitted.append(feature)
    return dict(die_um=[x1-x0, y1-y0], features=_arrays(fitted), unfit=unfit,
                skipped=list(layout.skipped_entities), preview=preview)
