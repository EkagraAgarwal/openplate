"""CPU L1 pipeline; lengths at public geometry boundaries are explicitly converted."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from platesim import __version__
from platesim.core.constants import F, M_CU, RHO_CU
from platesim.spec3d import Artifact, Check, Manifest, Quantity, SampleSpec, spec_hash, validate_spec
from platesim.tile3d.grid import Grid3D

MFAC = M_CU / (2 * F * RHO_CU)  # cm3/C, divalent copper, unity current efficiency
CELL_LIMIT = 200_000


def _spec(value):
    if isinstance(value, SampleSpec):
        # Revalidate even model_copy(update=...) inputs.
        value = value.model_dump(mode='json')
    model, errors, _ = validate_spec(value)
    if errors:
        raise ValueError(json.dumps(errors))
    return model


def _duration(step):
    return (step.duration_h * 3600 if step.duration_h is not None
            else step.charge_C_cm2 * 1000 / step.j_net_mA_cm2)


def plan_tile(spec: SampleSpec) -> dict:
    """Plan one pitch for a single parametric array, otherwise the complete die.

    Round outward to full cells; never stretch a periodic pitch. Nonintegral
    pitches fall back to a whole-die symmetry tile. est_seconds is a rough CPU
    budgeting heuristic, not a performance guarantee (no solver is run).
    """
    spec = _spec(spec)
    dx = spec.numerics.dx_um or {'preview': 8., 'standard': 4., 'fine': 2.}[spec.numerics.quality]
    die = spec.layout.die_um
    window = [0., 0., *(max(2, math.ceil(v / dx - 1e-10)) * dx for v in die)]
    periodic = [False, False, False]
    features = spec.layout.features or []
    if len(features) == 1 and features[0].array is not None:
        f, a = features[0], features[0].array
        pitch = [a.pitch_x_um, a.pitch_y_um]
        low = []
        angle = math.radians(f.rotation_deg)
        bounds = [abs(math.cos(angle)) * f.w_um + abs(math.sin(angle)) * f.h_um,
                  abs(math.sin(angle)) * f.w_um + abs(math.cos(angle)) * f.h_um]
        if f.kind == 'circle':
            bounds = [f.w_um, f.h_um]
        for center, p, n, size in zip([f.x_um, f.y_um], pitch, [a.nx, a.ny], die):
            index = max(0, math.ceil((p / 2 - center) / p - 1e-10))
            start = center + index * p - p / 2
            low.append(start if index < n and start >= -1e-9 and start + p <= size + 1e-9 else None)
        if (all(v is not None for v in low)
                and all(b <= p + 1e-9 for b, p in zip(bounds, pitch))
                and all(p / dx >= 2 and math.isclose(p / dx, round(p / dx), abs_tol=1e-9, rel_tol=0)
                        for p in pitch)):
            window = [*low, low[0] + pitch[0], low[1] + pitch[1]]
            periodic = [True, True, False]
    etches = [spec.etch.default, *spec.etch.layers.values()]
    depth = max(max([e.depth_um, *(d for _, d in (e.depth_vs_width or []))]) for e in etches)
    charge = sum(s.j_net_mA_cm2 * 1e-3 * _duration(s) for s in spec.recipe.steps)
    faraday_um = charge * MFAC * 1e4
    zlo = -(math.ceil(depth / dx - 1e-10) + 2) * dx
    zhi = math.ceil((faraday_um + spec.setup.delta_um + 4 * dx) / dx - 1e-10) * dx
    shape = [round((window[2] - window[0]) / dx), round((window[3] - window[1]) / dx),
             round((zhi - zlo) / dx)]
    cells = math.prod(shape)
    if cells > CELL_LIMIT:
        suggested = dx * (cells / CELL_LIMIT) ** (1 / 3) * 1.05
        raise ValueError(f'n_cells={cells} exceeds allowed {CELL_LIMIT}; fix: increase numerics.dx_um '
                         f'(try >= {suggested:.2f} um), use preview quality, or use a smaller tile/die.')
    duration = sum(_duration(s) for s in spec.recipe.steps)
    return dict(window_um=window, dx_um=dx, z_extent_um=[zlo, zhi], periodic=periodic,
                shape=shape, n_cells=cells, faraday_thickness_um=faraday_um,
                est_seconds=max(.01, cells / 1000 * (duration / 120 + len(spec.recipe.steps)) * .2))


def _layout(spec):
    from platesim.layout import load_layout
    data = spec.layout.model_dump(mode='json')
    # spec3d defines circles by diameter; layout currently requires explicit radius.
    for feature in data.get('features') or []:
        if feature['kind'] == 'circle':
            feature['r_um'] = feature['w_um'] / 2
    return load_layout(data)


def build_geometry(spec, plan) -> tuple[np.ndarray, Grid3D]:
    """Union per-layer etched cavities; preserve plan origin and boundary kinds.

    etch_to_psi already returns cm. Its grid has XY origin zero and symmetry
    sides, so it is deliberately not used. Re-distance with the actual periodic
    flags after the layer union (the layout etcher uses nonperiodic marching).
    """
    from platesim.layout import etch_to_psi, rasterize
    import skfmm

    spec = _spec(spec)
    layout = _layout(spec)
    x0, y0, _, _ = plan['window_um']
    zlo, zhi = plan['z_extent_um']
    dx = plan['dx_um']
    bc = {f'{a}_{side}': 'periodic' if plan['periodic'][i] else 'symmetry'
          for i, a in enumerate('xy') for side in ('lo', 'hi')}
    bc.update(z_lo='bottom', z_hi='bulk')
    grid = Grid3D(tuple(plan['shape']), dx * 1e-4, (x0 * 1e-4, y0 * 1e-4, zlo * 1e-4), bc)
    psi = np.broadcast_to(grid.centers()[2], grid.shape).copy()
    for layer in layout.layers:
        mask = rasterize(layout, layer, plan['window_um'], dx)
        etch = spec.etch.layers.get(layer, spec.etch.default)
        cavity, _ = etch_to_psi(mask, dx, etch.model_dump(), (zlo, zhi))
        psi = np.maximum(psi, cavity)  # union of electrolyte cavities
    if psi.min() < 0 < psi.max():
        psi = np.asarray(skfmm.distance(psi, dx=grid.dx, periodic=grid.periodic))
    return np.ascontiguousarray(psi, dtype=np.float32), grid


def run_sample(spec: SampleSpec | dict | str, out_dir, progress=None) -> dict:
    """Run all steps into one result, saving the bare frame and each step end.

    Solver calls have no run_dir: TileResultWriter retains the original bare
    silicon, cumulative times/charge/volumes, and the carried final front. This
    avoids treating earlier Cu as bare silicon when a later step starts.
    """
    from platesim.analysis3d import height_map, feature_report, void_report, cross_section
    from platesim.render3d import render_surface, render_height_map, render_cross_section
    from platesim.feature.engine.params import Params, migration_enhancement
    from platesim.tile3d import levelset3d as ls
    from platesim.tile3d.result import TileResult, TileResultWriter
    from platesim.tile3d.sim import run_tile, top_height

    started = time.perf_counter()
    spec = _spec(spec)
    if spec.numerics.physics != 'L1' or spec.numerics.use_surrogate:
        raise ValueError('numerics: only physics="L1", use_surrogate=False supported; fix these fields.')
    if isinstance(spec.numerics.device, int):
        raise ValueError('numerics.device: L1 sample pipeline is CPU-only; fix: use "cpu" or "auto".')
    plan = plan_tile(spec)
    bare, grid = build_geometry(spec, plan)
    psi = bare.copy()
    out = Path(out_dir)
    writer = TileResultWriter(out, grid, bare)
    initial = ls.solid_volume(bare, grid.dx)
    area = grid.shape[0] * grid.shape[1] * grid.dx**2
    h0 = top_height(bare, grid)
    field_mask = h0 > h0.max() - grid.dx
    cu = spec.bath.cu_g_L / M_CU  # mol/L
    acid = spec.bath.h2so4_g_L / 98.079
    params = Params(suppressor=False, c_bulk=cu * 1e-3,
                    E_mig=migration_enhancement(cu, [(1, 2 * acid), (-2, cu + acid)]),
                    c_h_bulk=2 * acid * 1e-3, c_so4_bulk=(cu + acid) * 1e-3)
    elapsed = charge = deposited = expected = 0.
    worst_balance = 0.
    writer.add_frame(0., psi, scalars=dict(charge_C_cm2=0., solid_volume_cm3=0.,
                                          faraday_volume_cm3=0., cu_balance_rel=0.))
    durations = [_duration(s) for s in spec.recipe.steps]
    total_duration = sum(durations)
    for index, (step, duration) in enumerate(zip(spec.recipe.steps, durations)):
        def report(fraction, message):
            if progress is not None:
                progress((elapsed + fraction * duration) / total_duration,
                         f'recipe {index + 1}/{len(durations)}: {message}')
        result = run_tile(psi, grid, step.j_net_mA_cm2, duration,
                          delta_um=spec.setup.delta_um, params=params, T_C=spec.bath.temperature_C,
                          n_frames=2, backend='numpy', field_mask=field_mask, progress=report)
        psi = result['psi']
        elapsed += duration
        charge += step.j_net_mA_cm2 * 1e-3 * duration
        expected = charge * area * MFAC
        deposited = ls.solid_volume(psi, grid.dx) - initial
        worst_balance = max(worst_balance, *(abs(v) for v in result['series']['cu_balance_rel']))
        writer.add_frame(elapsed, psi, scalars=dict(charge_C_cm2=charge,
                         solid_volume_cm3=deposited, faraday_volume_cm3=expected,
                         cu_balance_rel=worst_balance))
    writer.close()
    result = TileResult(out)
    heights = height_map(psi, grid)
    features = feature_report(result)
    voids = void_report(psi, grid)
    # Section along x through the first feature centre present in this tile.
    layout = _layout(spec)
    xmin, ymin, xmax, ymax = plan['window_um']
    y = (ymin + ymax) / 2
    for feature in layout.features:
        cx, cy = feature.polygon.centroid.coords[0]
        if xmin <= cx <= xmax and ymin <= cy <= ymax:
            y = cy
            break
    section = cross_section(psi, bare, grid, [(xmin, y), (xmax, y)])
    renders = [render_surface(result, out_png=out / 'surface.png', camera='iso'),
               render_height_map(heights['height_um'], heights['x_um'], heights['y_um'],
                                 out / 'height_map.png'),
               render_cross_section(section, out / 'cross_section.png')]
    quantities = []
    def quantity(name, value, unit):
        if value is not None and np.isfinite(value):
            quantities.append(Quantity(name=name, value=float(value), unit=unit, flag='uncalibrated'))
    for name in ('field_um', 'floor_um', 'wall_um', 'overburden_um'):
        quantity(name, features.get(name), 'um')
    quantity('floor_over_field', features.get('floor_over_field'), '1')
    quantity('fill_fraction', features['fill_fraction'][-1], '1')
    quantity('closure_time_s', features.get('closure_time_s'), 's')
    quantity('void_count', voids['count'], '1')
    quantity('void_volume_um3', np.sum(voids['volumes_um3']), 'um3')
    quantity('deposited_cu_volume_um3', deposited * 1e12, 'um3')
    quantity('faraday_thickness_um', expected / area * 1e4, 'um')
    for name, value in heights['stats'].items():
        quantity(name, value, 'um')
    quantity('n_cells', plan['n_cells'], '1')
    quantity('wall_s', time.perf_counter() - started, 's')
    error = abs(deposited / expected - 1.) if expected else 0.
    artifacts = [Artifact(kind='png', path=Path(r['path']).name, title=r['title']) for r in renders]
    artifacts.extend([Artifact(kind='json', path='grid.json', title='Tile grid (cm)'),
                      Artifact(kind='json', path='series.json', title='Cumulative recipe series'),
                      Artifact(kind='npz', path='static.npz', title='Original bare silicon (cm)')])
    manifest = Manifest(spec_hash=spec_hash(spec), code_version=__version__, backend='numpy',
        device='cpu', dx_um=grid.dx * 1e4, physics='L1', created_utc=datetime.now(timezone.utc),
        quantities=quantities, checks=[Check(name='faraday_error_rel', value=error, limit=.05, passed=error <= .05),
        Check(name='cu_balance_rel', value=worst_balance, limit=.01, passed=worst_balance <= .01)],
        waveform_sensitivity='L1 uses only the net current; pulse timing and reverse-current effects '
                             'beyond the cycle average are not resolved.',
        artifacts=artifacts, summary_text='Uncalibrated L1, CPU, no additives or resolved flow. '
        'One result retains original silicon and cumulative step-end frames. '
        'Seed thinning is not modelled; conductivity and kinetics use uncalibrated engine defaults. '
        'Faraday volume correction is disabled; checks report actual numerical drift.')
    payload = manifest.model_dump(mode='json')
    temporary = out / 'manifest.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(out / 'manifest.json')
    if progress is not None:
        progress(1., 'manifest complete')
    return payload
