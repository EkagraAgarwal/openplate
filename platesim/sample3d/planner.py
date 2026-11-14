"""Representative, feature-local L1 domains (no solver calls)."""
from fractions import Fraction
from functools import reduce
import math

CELL_LIMIT = 200_000


def _regular(features):
    """Recognise complete Cartesian lattices, never infer a missing array member."""
    if len(features) < 2 or any(f.array for f in features):
        return None
    signature = lambda f: f.model_dump(exclude={'x_um', 'y_um'})
    if any(signature(f) != signature(features[0]) for f in features[1:]):
        return None
    axes = [sorted({getattr(f, a) for f in features}) for a in ('x_um', 'y_um')]
    if math.prod(map(len, axes)) != len(features):
        return None
    if len({(f.x_um, f.y_um) for f in features}) != len(features):
        return None
    pitches = []
    for values in axes:
        # A single row does not establish the transverse pitch.
        if len(values) < 2:
            return None
        pitch = values[1] - values[0]
        if any(not math.isclose(b-a, pitch, abs_tol=1e-7) for a, b in zip(values, values[1:])):
            return None
        pitches.append(pitch)
    return pitches


def _pitch_dx(spans, requested):
    # Exact decimal pitches, uniform cubic cells. Do not stretch the unit cell.
    fractions = [Fraction(str(v)).limit_denominator(10000) for v in spans]
    denominator = math.lcm(*(f.denominator for f in fractions))
    common = reduce(math.gcd, (f.numerator * (denominator // f.denominator) for f in fractions)) / denominator
    return common / max(1, math.ceil(common / requested - 1e-10))


def plan_features(spec, faraday_um, duration, field_margin_um=None):
    features = spec.layout.features or []
    if not features:
        raise ValueError('layout.file: representative tiling requires parametric features; fix: supply layout.features.')
    if field_margin_um is not None and (not math.isfinite(field_margin_um) or field_margin_um < 0):
        raise ValueError('field_margin_um must be finite and >= 0; fix: omit it for the depth-based default.')
    quality = spec.numerics.quality
    requested = spec.numerics.dx_um or {'preview': 8., 'standard': 4., 'fine': 2.}[quality]
    groups = {}
    for i, f in enumerate(features):
        signature = repr(f.model_dump(exclude={'x_um', 'y_um'}))
        groups.setdefault(signature, []).append(i)
    regular = {}
    for indices in groups.values():
        pitch = _regular([features[i] for i in indices])
        if pitch:
            regular[indices[0]] = (indices, pitch)
    skipped = {i for indices, _ in regular.values() for i in indices[1:]}
    plans = []
    for i, feature in enumerate(features):
        if i in skipped:
            continue
        etch = spec.etch.layers.get(feature.layer, spec.etch.default)
        depth = max(etch.depth_um, *(d for _, d in (etch.depth_vs_width or [])), 0.)
        margin = field_margin_um if field_margin_um is not None else max(3*depth, 2*spec.setup.delta_um, 100.)
        ids = [f'feature_{j:04d}' for j in regular.get(i, ([i], None))[0]]
        f = feature.model_dump(mode='json')
        f.update(x_um=0., y_um=0., array=None)
        pitch = regular[i][1] if i in regular else None
        weight = len(ids)
        if feature.array:
            pitch = [feature.array.pitch_x_um, feature.array.pitch_y_um]
            weight = feature.array.nx * feature.array.ny
        assumptions, flags = [], ['preview'] if quality == 'preview' else []
        symmetric = f['kind'] == 'circle' or math.isclose(f['rotation_deg'] % 90, 0., abs_tol=1e-9)
        if pitch:
            angle = math.radians(f['rotation_deg'])
            bounds = ([f['w_um'], f['h_um']] if f['kind'] == 'circle' else
                      [abs(math.cos(angle))*f['w_um']+abs(math.sin(angle))*f['h_um'],
                       abs(math.sin(angle))*f['w_um']+abs(math.cos(angle))*f['h_um']])
            if any(b >= p for b, p in zip(bounds, pitch)):
                raise ValueError(f'{ids}: overlapping/touching array; fix: increase pitch above feature bounds.')
            assumptions += ['Infinite regular array; edge-of-array effects are not modelled.']
            flags += ['edge-of-array effects not modelled']
            descriptors = [('array_quarter' if symmetric else 'array_periodic',
                            [0., 0., pitch[0]/2, pitch[1]/2] if symmetric else
                            [-pitch[0]/2, -pitch[1]/2, pitch[0]/2, pitch[1]/2],
                            ['x', 'y'] if symmetric else [], weight, False)]
        else:
            # Isotropic L1 is invariant under XY rotations: solve in feature axes.
            rotation = f['rotation_deg']
            f['rotation_deg'] = 0.
            if f['w_um'] > f['h_um']:
                f['w_um'], f['h_um'] = f['h_um'], f['w_um']
                rotation += 90
            assumptions += [f'Feature-local XY axes, rotated {rotation:g} degrees from die axes; infinite field, no die edge.']
            w, length = f['w_um'], f['h_um']
            if length / w > 4 and f['kind'] != 'circle':
                assumptions += ['Mid-length tile is the profile result; ends are independent local diagnostics, not additive volumes.']
                descriptors = [('trench_mid', [0., 0., w/2+margin, 0.], ['x'], 1, True),
                               ('trench_end', [0., length/2-w, w/2+margin, length/2+margin], ['x'], 2, False)]
            else:
                descriptors = [('compact_quarter', [0., 0., w/2+margin, length/2+margin], ['x', 'y'], 1, False)]
            # Polygon distances include all physical instances of other families.
            from .pipeline import _layout
            layout = _layout(spec)
            polygon = layout.features[sum((g.array.nx*g.array.ny if g.array else 1) for g in features[:i])].polygon
            others = [g.polygon for g in layout.features if not g.polygon.equals(polygon)]
            if any(polygon.distance(g) < 2*margin for g in others):
                flags += ['isolated feature in an infinite field: neighbour interaction neglected']
        for kind, window, mirrors, copies, thin in descriptors:
            target_dx = requested
            while True:
                dx = _pitch_dx([window[2]-window[0], window[3]-window[1]], target_dx) if pitch else target_dx
                actual = window.copy()
                if not pitch:
                    actual[2] = actual[0] + max(2, math.ceil((window[2]-window[0])/dx-1e-10))*dx
                    actual[3] = actual[1] + (4 if thin else max(2, math.ceil((window[3]-window[1])/dx-1e-10)))*dx
                zlo = -(math.ceil(depth/dx-1e-10)+2)*dx
                zhi = math.ceil((faraday_um+spec.setup.delta_um+4*dx)/dx-1e-10)*dx
                shape = [round((actual[2]-actual[0])/dx), round((actual[3]-actual[1])/dx), round((zhi-zlo)/dx)]
                cells = math.prod(shape)
                if min(shape) < 2:
                    raise ValueError(f'{ids}: pitch is underresolved; fix: decrease numerics.dx_um.')
                if cells <= CELL_LIMIT:
                    break
                if quality != 'preview':
                    raise ValueError(f'{ids} {kind}: n_cells={cells} exceeds {CELL_LIMIT}; fix: use preview '
                                     'for automatic coarsening, increase numerics.dx_um, or reduce field_margin_um.')
                target_dx *= 1.25
                if target_dx > max(window[2]-window[0], window[3]-window[1], depth)*4:
                    raise ValueError('Incommensurate pitches cannot fit the cell cap; fix: use commensurate pitches.')
            bc = {f'{a}_{s}': ('periodic' if (pitch and not symmetric) or (thin and a == 'y') else 'symmetry')
                  for a in 'xy' for s in ('lo', 'hi')}
            tile_flags = flags.copy()
            if target_dx > requested:
                tile_flags += [f'preview coarsened from requested dx={requested:g} um']
            tile_f = f.copy()
            if thin:
                tile_f.update(kind='rect', r_um=0., h_um=actual[3]*4, y_um=actual[3]/2)
            plans.append(dict(tile_id=f'{ids[0]}_{kind}', kind=kind, feature_ids=ids,
                weight=copies, window_um=actual, dx_um=dx, z_extent_um=[zlo, zhi],
                side_bc=bc, periodic=[bc['x_lo']=='periodic', bc['y_lo']=='periodic', False],
                mirror_axes=mirrors, shape=shape, n_cells=cells, field_margin_um=margin,
                assumptions=assumptions + (['End cut is a no-flux symmetry approximation one width inside the end.'] if kind=='trench_end' else []),
                flags=tile_flags, geometry_feature=tile_f, source_center_um=[feature.x_um, feature.y_um],
                faraday_thickness_um=faraday_um,
                est_seconds=max(.01, cells/1000*(duration/120+len(spec.recipe.steps))*.2)))
    return plans
