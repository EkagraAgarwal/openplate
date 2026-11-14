"""CPU L1 pipeline; lengths at public geometry boundaries are explicitly converted."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from platesim import __version__
from platesim.core.constants import F, M_CU, RHO_CU
from platesim.spec3d import Artifact, Check, Manifest, Quantity, SampleSpec, spec_hash, validate_spec
from platesim.tile3d.grid import Grid3D

MFAC = M_CU / (2 * F * RHO_CU)  # cm3/C, divalent copper, unity current efficiency


def _spec(value):
    if isinstance(value, SampleSpec):
        # Revalidate even model_copy(update=...) inputs.
        value = value.model_dump(mode='json')
    model, errors, _ = validate_spec(value)
    if errors:
        raise ValueError(json.dumps(errors))
    if model is None:
        raise ValueError('spec validation returned no model; fix: supply a valid SampleSpec.')
    return model


def _duration(step):
    return (step.duration_h * 3600 if step.duration_h is not None
            else step.charge_C_cm2 * 1000 / step.j_net_mA_cm2)


def plan_tile(spec: SampleSpec, *, field_margin_um=None) -> list[dict]:
    """Return representative feature-local tiles; see README for assumptions."""
    from .planner import plan_features
    spec = _spec(spec)
    charge = sum(s.j_net_mA_cm2 * 1e-3 * _duration(s) for s in spec.recipe.steps)
    return plan_features(spec, charge * MFAC * 1e4,
                         sum(_duration(s) for s in spec.recipe.steps), field_margin_um)


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
    from platesim.layout import load_layout
    feature = plan.get('geometry_feature')
    if feature:
        feature = feature.copy()
        if feature['kind'] == 'circle':
            feature['r_um'] = feature['w_um'] / 2
        layout = load_layout(dict(features=[feature], die_um=spec.layout.die_um))
    else:
        layout = _layout(spec)
    x0, y0, _, _ = plan['window_um']
    zlo, zhi = plan['z_extent_um']
    dx = plan['dx_um']
    bc = dict(plan['side_bc'])
    bc.update(z_lo='bottom', z_hi='bulk')
    grid = Grid3D(tuple(plan['shape']), dx * 1e-4, (x0 * 1e-4, y0 * 1e-4, zlo * 1e-4), bc)
    psi = np.broadcast_to(grid.centers()[2], grid.shape).copy()
    for layer in layout.layers:
        mask = rasterize(layout, layer, plan['window_um'], dx)
        etch = spec.etch.layers.get(layer, spec.etch.default)
        # Reflect centre cuts before etching, so fillets and width-dependent
        # depths see the complete feature. Periodic y strips are also extended.
        slices = [slice(None), slice(None)]
        for axis, a in enumerate('xy'):
            if a in plan.get('mirror_axes', []):
                n = mask.shape[axis]
                mask = np.concatenate([np.flip(mask, axis=axis), mask], axis=axis)
                slices[axis] = slice(n, 2*n)
            elif grid.periodic[axis]:
                n = mask.shape[axis]
                mask = np.concatenate([mask]*3, axis=axis)
                slices[axis] = slice(n, 2*n)
        cavity, _ = etch_to_psi(mask, dx, etch.model_dump(), (zlo, zhi))
        cavity = cavity[tuple(slices)]
        psi = np.maximum(psi, cavity)  # union of electrolyte cavities
    if psi.min() < 0 < psi.max():
        psi = np.asarray(skfmm.distance(psi, dx=grid.dx, periodic=grid.periodic))
    return np.ascontiguousarray(psi, dtype=np.float32), grid


class _ReflectedResult:
    """Lazy view: reflection is for geometry analysis/rendering, not the solve."""
    def __init__(self, result, axes):
        self.result, self.axes = result, axes
        shape, origin, bc = list(result.grid.shape), list(result.grid.origin), dict(result.grid.bc)
        for a in axes:
            i = 'xy'.index(a)
            origin[i] -= shape[i] * result.grid.dx
            shape[i] *= 2
        self.grid = Grid3D(tuple(shape), result.grid.dx, tuple(origin), bc)
        self.times_s = result.times_s
        self.psi_si = self.reflect(result.psi_si)

    def reflect(self, array):
        for a in self.axes:
            i = 'xy'.index(a)
            array = np.concatenate([np.flip(array, axis=i), array], axis=i)
        return array

    def psi(self, index=-1):
        return self.reflect(self.result.psi(index))


def _run_one(spec, plan, out_dir, progress=None) -> dict:
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
        step_result = run_tile(psi, grid, step.j_net_mA_cm2, duration,
                          delta_um=spec.setup.delta_um, params=params, T_C=spec.bath.temperature_C,
                          n_frames=2, backend='numpy', field_mask=field_mask, progress=report)
        psi = step_result['psi']
        elapsed += duration
        charge += step.j_net_mA_cm2 * 1e-3 * duration
        expected = charge * area * MFAC
        deposited = ls.solid_volume(psi, grid.dx) - initial
        worst_balance = max(worst_balance, *(abs(v) for v in step_result['series']['cu_balance_rel']))
        writer.add_frame(elapsed, psi, scalars=dict(charge_C_cm2=charge,
                         solid_volume_cm3=deposited, faraday_volume_cm3=expected,
                         cu_balance_rel=worst_balance))
    writer.close()
    result = _ReflectedResult(TileResult(out), plan.get('mirror_axes', []))
    psi, bare, analysis_grid = result.psi(), result.psi_si, result.grid
    heights = height_map(psi, analysis_grid)
    features = feature_report(result)
    voids = void_report(psi, analysis_grid)
    # Transverse profile, or a longitudinal section exposing the end cap.
    ymin = analysis_grid.origin[1]*1e4
    ymax = ymin + analysis_grid.shape[1]*analysis_grid.dx*1e4
    y = np.clip(plan['geometry_feature']['y_um'], ymin + plan['dx_um']/2, ymax - plan['dx_um']/2)
    # Cell-centre endpoints avoid roundoff outside the sampler's strict bounds.
    x_centres, y_centres, _ = analysis_grid.centers()
    line = ([(0., y_centres[0]*1e4), (0., y_centres[-1]*1e4)] if plan['kind'] == 'trench_end'
            else [(x_centres[0]*1e4, y), (x_centres[-1]*1e4, y)])
    section = cross_section(psi, bare, analysis_grid, line)
    renders = [render_surface(result, out_png=out / 'surface.png', camera='iso'),
               render_height_map(heights['height_um'], heights['x_um'], heights['y_um'],
                                 out / 'height_map.png'),
               render_cross_section(section, out / 'cross_section.png')]
    quantities = []
    def quantity(name, value, unit):
        if value is not None and np.isfinite(value):
            quantities.append(Quantity(name=name, value=float(value), unit=unit,
                flag='preview' if 'preview' in plan['flags'] else 'uncalibrated',
                feature_id=plan['feature_ids'][0]))
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
    quantity('tile_weight', plan['weight'], '1')
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
    if progress is not None:
        progress(1., 'manifest complete')
    return payload


def run_sample(spec: SampleSpec | dict | str, out_dir, progress=None, *, field_margin_um=None) -> dict:
    """Run representative tiles sequentially; write one strict aggregate Manifest.

    Machine-readable assumptions/weights are linked via the tiles.json artifact;
    they are also present in summary_text because Manifest forbids extra fields.
    """
    spec = _spec(spec)
    plans = plan_tile(spec, field_margin_um=field_margin_um)
    if spec.numerics.physics != 'L1' or spec.numerics.use_surrogate:
        raise ValueError('numerics: only physics="L1", use_surrogate=False supported; fix these fields.')
    if isinstance(spec.numerics.device, int):
        raise ValueError('numerics.device: CPU only; fix: use "cpu" or "auto".')
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; fix: pass a new or empty output directory.')
    out.mkdir(parents=True, exist_ok=True)              # an empty directory (e.g. made by mcp3d) is fine
    quantities, checks, artifacts, descriptions = [], [], [], []
    template = None
    for index, plan in enumerate(plans):
        prefix = f'tiles/{plan["tile_id"]}'
        def report(fraction, message):
            if progress:
                progress((index + fraction)/len(plans), f'{plan["tile_id"]}: {message}')
        tile = _run_one(spec, plan, out / prefix, report)
        template = tile
        # Only the mid-length/compact/array tile is the feature's profile result.
        # End diagnostics have qualified names and must not be averaged with it.
        for feature_id in plan['feature_ids']:
            for q in tile['quantities']:
                q = dict(q, feature_id=feature_id)
                if plan['kind'] == 'trench_end':
                    q['name'] = f'{plan["kind"]}.{q["name"]}'
                quantities.append(q)
        checks += [dict(c, name=f'{plan["tile_id"]}.{c["name"]}') for c in tile['checks']]
        artifacts += [dict(a, path=f'{prefix}/{a["path"]}') for a in tile['artifacts']]
        descriptions.append(f'{plan["tile_id"]}: feature_ids={plan["feature_ids"]}, weight={plan["weight"]}; '
                            + '; '.join(plan['assumptions'] + plan['flags']))
        # Writer's empty metadata manifest is retained for TileResult readers;
        # the only validated sample Manifest is at the sample root.
    (out / 'tiles.json').write_text(json.dumps(plans, indent=2, allow_nan=False), encoding='utf-8')
    artifacts.append(dict(kind='json', path='tiles.json', title='Tile plans, weights, flags and assumptions'))
    payload = dict(template, dx_um=max(p['dx_um'] for p in plans), quantities=quantities,
                   checks=checks, artifacts=artifacts,
                   summary_text='Uncalibrated CPU L1. Mid-length tiles are the profile result. '
                   'Mirrored renders/analysis reconstruct centre cuts; end cuts are not reflected. '
                   'Weights count physical copies, not symmetry sectors; do not sum overlapping diagnostic volumes. '
                   'Deposited volumes are simulated-sector volumes, while void analysis uses the reflected view. '
                   'No additives, resolved flow or seed thinning; conductivity/kinetics are uncalibrated defaults. '
                   + ' | '.join(descriptions))
    payload = Manifest.model_validate(payload).model_dump(mode='json')
    temporary = out / 'manifest.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(out / 'manifest.json')
    if progress:
        progress(1., 'manifest complete')
    return payload
