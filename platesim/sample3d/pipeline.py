"""CPU L1 pipeline; lengths at public geometry boundaries are explicitly converted."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from platesim import __version__
from platesim.core.constants import F, M_CU, RHO_CU
from platesim.spec3d import Artifact, Check, Manifest, Quantity, SampleSpec, spec_hash, validate_spec
from platesim.tile3d.grid import Grid3D

MFAC = M_CU / (2 * F * RHO_CU)  # cm3 per C of Cu charge (applied charge times efficiency)

_ADDITIVE_TRIPLET = ('_min', '_max', '_mean')


def _spec_params(spec: SampleSpec):
    """Legacy base Params plus the L3 additive overrides; nothing else changes."""
    from platesim.feature.engine.params import Params, migration_enhancement

    cu = spec.bath.cu_g_L / M_CU  # mol/L
    acid = spec.bath.h2so4_g_L / 98.079
    params = Params(suppressor=False, c_bulk=cu * 1e-3,
                    E_mig=migration_enhancement(cu, [(1, 2 * acid), (-2, cu + acid)]),
                    c_h_bulk=2 * acid * 1e-3, c_so4_bulk=(cu + acid) * 1e-3)
    additives = spec.additives
    if additives is None:
        return params
    overrides = dict(additives=additives.mode, suppressor=additives.suppressor,
                     c_cl_bulk=spec.bath.cl_ppm / 35.453 * 1e-6)
    mapping = (('s_bulk', additives.s_bulk_mol_cm3),
               ('c_acc', additives.c_acc_mol_cm3),
               ('Gamma_acc', additives.Gamma_acc_mol_cm2),
               ('k_acc', additives.k_acc_cm3_mol_s),
               ('b_acc', additives.b_acc),
               ('k_inc_a', additives.k_inc_a_1_cm),
               ('D_acc25', additives.D_acc_cm2_s))
    overrides.update({name: value for name, value in mapping if value is not None})
    overrides.update(pulse_coverage=additives.pulse_coverage, k_strip_s=additives.k_strip_s,
                     k_strip_a=additives.k_strip_a, strip_return_a=additives.strip_return_a)
    # Contract 10i: carried CEAC accelerator coverage (both default to False).
    overrides.update(ceac=additives.ceac, ceac_smooth=additives.ceac_smooth)
    return replace(params, **overrides)


def _additive_unit(name):
    if name.endswith('_mol_cm2_s'):
        return 'mol/cm2/s'
    if name.endswith('_cm2_s'):
        return 'cm2/s'
    if name.endswith('_mol_cm2'):
        return 'mol/cm2'
    if name.endswith('_per_cm'):
        return '1/cm'
    return '1'


def _flatten_report(path, value, out):
    """Emit JSON-safe scalar quantities from one run_tile summary entry."""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and np.isfinite(value):
        out.append((path, float(value)))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            suffix = _ADDITIVE_TRIPLET[i] if len(value) == 3 and i < 3 else f'_{i}'
            _flatten_report(path + suffix, item, out)


def _additive_quantities(index, report):
    out = []
    for key, value in report.items():
        path = f'step_{index}.additives.{key}'
        if key == 'pulse_coverage' and isinstance(value, dict):
            for subkey, subvalue in value.items():
                _flatten_report(path + '.' + subkey, subvalue, out)
        elif key not in ('model', 'note', 'note_acc', 'relax', 'cov_change'):  # solver diagnostics
            _flatten_report(path, value, out)
    return out


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


_BUOYANCY_NOTE = ('Solutal buoyancy: expansion 0.15 L/mol [J], uncalibrated; modelled across-trench '
                  'buoyancy overshot narrow-trench floor/field in validation.')

_CEAC_NOTE = ('CEAC carried accelerator coverage (uncalibrated); curvature resolved only for '
              'radii >= 3-4 dx.')


def _buoyancy_assumption():
    return _BUOYANCY_NOTE


def _shear_assumption(spec):
    # The shear-only text stays byte-identical; with buoyancy on, the (new) trailing
    # clause is replaced by the separate buoyancy sentence recorded in the manifest.
    tail = ('no buoyancy.' if not spec.setup.buoyancy
            else 'combined with the solutal buoyancy noted below.')
    return (f'Resolved Couette shear G={spec.setup.stir_shear_1_s:g} 1/s at angle '
            f'{spec.setup.stir_angle_deg:g} deg (die axes); EXPERIMENTAL: lid at the bulk plane '
            f'(vs 2-D domain-top lid: +0.03 floor/field at 50/s in T2.5); {tail}')


def _coupon_assumption(spec):
    if spec.setup.coupon_position_cm is None:
        return ''
    return (f'Coupon position {spec.setup.coupon_position_cm} cm (centred, x across, y up): '
            'local current, Nernst layer and efficiency from the DC coupon model '
            'calibrated to the team planar DOE. ')


def _tile_rotation(plan):
    """Degrees from die x to the tile x axis (planner: feature-local tiles; arrays keep die axes)."""
    return float(plan.get('rotation_from_die_deg', 0.0))


def _gravity_die(setup) -> tuple[float, float, float]:
    """Gravity unit vector in die axes: x,y in the die plane, z out of the die face."""
    if setup.orientation == 'face_up':
        return (0., 0., -1.)
    if setup.orientation == 'face_down':
        return (0., 0., 1.)
    return {'+y': (0., -1., 0.), '-y': (0., 1., 0.),
            '+x': (-1., 0., 0.), '-x': (1., 0., 0.)}[setup.up_edge]


def _gravity_tile(setup, plan) -> tuple[float, float, float]:
    """Rotate the in-plane gravity vector into tile axes (same rule as shear); z stays."""
    gx, gy, gz = _gravity_die(setup)
    angle = np.deg2rad(-_tile_rotation(plan))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return (gx * c - gy * s, gx * s + gy * c, gz)


def _check_inplane_flow(plan, axis, component, message, fix):
    """Reject a nonzero in-plane flow component crossing a remaining no-flux side."""
    boundaries = [f'{axis}_{side}' for side in ('lo', 'hi')
                  if plan['side_bc'].get(f'{axis}_{side}') == 'symmetry']
    if boundaries and abs(component) > 1e-12:
        raise ValueError(f'{message} crosses symmetry/no-flux boundary {boundaries} '
                         f'of tile {plan["tile_id"]}; fix: {fix}')


def _tile_flow(spec, plan):
    """Rotate die-frame shear/gravity into tile axes and reject flow across no-flux sides."""
    angle = np.deg2rad(spec.setup.stir_angle_deg - _tile_rotation(plan))
    direction = (float(np.cos(angle)), float(np.sin(angle)))
    if (spec.setup.stir_shear_1_s or 0) > 0:
        for axis, component in zip('xy', direction):
            _check_inplane_flow(
                plan, axis, component,
                f'setup.stir_shear_1_s={spec.setup.stir_shear_1_s:g}: resolved shear',
                'use stir_angle_deg along the symmetry axis, or a full tile '
                '(non-mirrored periodic array cell). Quarter tiles and trench-end '
                'diagnostics have no compatible nonzero in-plane shear direction.')
    flow = dict(shear_rate_s=spec.setup.stir_shear_1_s or 0, shear_dir=direction,
                lid_um=None, update_every=5, axial_net_flux_zero=None)
    if spec.setup.buoyancy:
        gravity = _gravity_tile(spec.setup, plan)
        for axis, component in zip('xy', gravity[:2]):
            _check_inplane_flow(
                plan, axis, component,
                f'setup.buoyancy=True: gravity component {component:g} along {axis}',
                'use a full tile periodic in x and y (non-mirrored periodic array '
                'cell) or set setup.buoyancy=False; face_up/face_down have no '
                'in-plane gravity component.')
        flow['gravity_dir'] = gravity
    return flow


def plan_tile(spec: SampleSpec, *, field_margin_um=None) -> list[dict]:
    """Return representative feature-local tiles; see README for assumptions."""
    from .planner import plan_features
    spec = _spec(spec)
    charge = sum(s.j_net_mA_cm2 * 1e-3 * _duration(s) for s in spec.recipe.steps)
    if spec.setup.coupon_position_cm is not None:
        # Size the grid for the LOCAL current (coupon edges carry up to ~1.4x the mean); same cached
        # coupon solves the run uses. Efficiency <= 1 is ignored here, so the height stays conservative.
        from platesim.multiscale.bridge import tile3d_conditions
        charge = sum(tile3d_conditions(s.j_net_mA_cm2, _duration(s), *spec.setup.coupon_position_cm,
                                       T_C=spec.bath.temperature_C, stir_pct=spec.setup.stir_pct)
                     ['j_local_mA_cm2'] * 1e-3 * _duration(s) for s in spec.recipe.steps)
    plans = plan_features(spec, charge * MFAC * 1e4,
                          sum(_duration(s) for s in spec.recipe.steps), field_margin_um)
    if spec.setup.stir_shear_1_s or spec.setup.buoyancy:  # 0/None mean no resolved shear
        for plan in plans:
            _tile_flow(spec, plan)  # validate every tile before creating output or solving
            if spec.setup.stir_shear_1_s:
                plan['assumptions'].append(_shear_assumption(spec))
            if spec.setup.buoyancy:
                plan['assumptions'].append(_buoyancy_assumption())
    return plans


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


def _section_line(plan, grid):
    """Transverse section across the reflected width, including end tiles.

    End diagnostics sample half a width inside the physical end cap. The
    artificial interior y cut is not a symmetry plane for reconstruction.
    """
    x, y, _ = (c*1e4 for c in grid.centers())
    feature = plan['geometry_feature']
    position = (feature['h_um']/2 - feature['w_um']/2
                if plan['kind'] == 'trench_end' else feature['y_um'])
    position = float(np.clip(position, y[0], y[-1]))
    return [(x[0], position), (x[-1], position)]


def _run_one(spec, plan, out_dir, progress=None) -> dict:
    """Run all steps into one result, saving the bare frame and each step end.

    Solver calls have no run_dir: TileResultWriter retains the original bare
    silicon, cumulative times/charge/volumes, and the carried final front. This
    avoids treating earlier Cu as bare silicon when a later step starts.
    """
    from platesim.analysis3d import height_map, feature_report, void_report, cross_section
    from platesim.render3d import render_surface, render_height_map, render_cross_section
    from platesim.tile3d import levelset3d as ls
    from platesim.tile3d.result import TileResult, TileResultWriter
    from platesim.tile3d.sim import run_tile, top_height

    started = time.perf_counter()
    spec = _spec(spec)
    if spec.numerics.physics not in ('L1', 'L3') or spec.numerics.use_surrogate:
        raise ValueError('numerics: only physics="L1" or "L3", use_surrogate=False supported; fix these fields.')
    if isinstance(spec.numerics.device, int):
        raise ValueError('numerics.device: L1 sample pipeline is CPU-only; fix: use "cpu" or "auto".')
    flowing = bool(spec.setup.stir_shear_1_s) or spec.setup.buoyancy
    flow_kwargs = dict(flow=_tile_flow(spec, plan)) if flowing else {}
    bare, grid = build_geometry(spec, plan)
    psi = bare.copy()
    out = Path(out_dir)
    writer = TileResultWriter(out, grid, bare)
    initial = ls.solid_volume(bare, grid.dx)
    area = grid.shape[0] * grid.shape[1] * grid.dx**2
    h0 = top_height(bare, grid)
    field_mask = h0 > h0.max() - grid.dx
    params = _spec_params(spec)
    elapsed = charge = deposited = expected = 0.
    worst_balance = 0.
    writer.add_frame(0., psi, scalars=dict(charge_C_cm2=0., solid_volume_cm3=0.,
                                          faraday_volume_cm3=0., cu_balance_rel=0.))
    durations = [_duration(s) for s in spec.recipe.steps]
    total_duration = sum(durations)
    local_conditions = []
    pulse_reports = []
    additives_reports = []
    theta_a_reports = []
    for index, (step, duration) in enumerate(zip(spec.recipe.steps, durations)):
        def report(fraction, message):
            if progress is not None:
                progress((elapsed + fraction * duration) / total_duration,
                         f'recipe {index + 1}/{len(durations)}: {message}')
        j_local, delta_um, step_params = step.j_net_mA_cm2, spec.setup.delta_um, params
        if spec.setup.coupon_position_cm is not None:
            from platesim.multiscale.bridge import tile3d_conditions
            conditions = tile3d_conditions(step.j_net_mA_cm2, duration,
                *spec.setup.coupon_position_cm, T_C=spec.bath.temperature_C,
                stir_pct=spec.setup.stir_pct)
            j_local, delta_um = conditions['j_local_mA_cm2'], conditions['delta_um']
            step_params = replace(params, efficiency=conditions['efficiency'])
            local_conditions.append(conditions)
        pulse_kwargs = {}
        if spec.numerics.pulse_model == 'resolved':
            scale = j_local / step.j_net_mA_cm2
            pulse_kwargs['pulse'] = dict(t_f_ms=step.t_forward_ms, t_r_ms=step.t_reverse_ms,
                t_off_ms=step.t_off_ms, j_f_mA_cm2=step.j_forward_mA_cm2 * scale,
                j_r_mA_cm2=step.j_reverse_mA_cm2 * scale, model='resolved')
            step_params = replace(step_params, C_dl_F_cm2=spec.numerics.pulse_C_dl_uF_cm2 * 1e-6)
        def save_intermediate(t_step, front, row):
            if 0 < t_step < duration - 1e-9:
                writer.add_frame(elapsed + t_step, front, scalars=dict(
                    charge_C_cm2=charge + row['charge_C_cm2'],
                    solid_volume_cm3=ls.solid_volume(front, grid.dx) - initial,
                    faraday_volume_cm3=expected + row['faraday_volume_cm3'],
                    cu_balance_rel=row['cu_balance_rel']))
        if spec.numerics.frames_per_step > 2:
            pulse_kwargs['frame_callback'] = save_intermediate
        step_result = run_tile(psi, grid, j_local, duration,
                          delta_um=delta_um, params=step_params, T_C=spec.bath.temperature_C,
                            n_frames=spec.numerics.frames_per_step, backend='numpy', field_mask=field_mask, progress=report,
                            **flow_kwargs, **pulse_kwargs)
        psi = step_result['psi']
        # Contract 10i: CEAC carries theta_a_floor/field in the series (or the summary).
        # The quantity names below are emitted only if not already present.
        summary_add = step_result.get('summary', {}).get('additives', {})
        last_theta_a = {}
        for name in ('theta_a_floor', 'theta_a_field'):
            values = step_result.get('series', {}).get(name)
            if values:
                last_theta_a[name] = float(values[-1])
            elif isinstance(summary_add, dict) and isinstance(summary_add.get(name), (int, float)):
                last_theta_a[name] = float(summary_add[name])
        if last_theta_a:
            theta_a_reports.append((index + 1, last_theta_a))
        if 'pulse' in step_result.get('summary', {}):
            pulse_reports.append((index + 1, step_result['summary']['pulse']))
        if 'additives' in step_result.get('summary', {}):
            additives_reports.append((index + 1, step_result['summary']['additives']))
        elapsed += duration
        step_charge = j_local * 1e-3 * duration
        charge += step_charge  # applied tile charge, before efficiency
        if spec.setup.coupon_position_cm is None:
            expected = charge * area * MFAC  # retain legacy floating-point ordering
        else:
            expected += step_charge * step_params.efficiency * area * MFAC
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
    line = _section_line(plan, analysis_grid)
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
    for index, conditions in enumerate(local_conditions, 1):
        for name, unit in [('j_local_mA_cm2', 'mA/cm2'), ('delta_um', 'um'),
                           ('efficiency', '1'), ('scale', '1')]:
            quantity(f'step_{index}.{name}', conditions[name], unit)
    for index, pulse_report in pulse_reports:
        quantity(f'step_{index}.pulse_faraday_error_rel', pulse_report['faraday_error_rel'], '1')
        quantity(f'step_{index}.pulse_alpha_a', pulse_report['alpha_a'], '1')
        for phase, tau in zip(('forward', 'reverse', 'off'), pulse_report['tau_s']):
            quantity(f'step_{index}.tau_{phase}_s', tau, 's')
        for phase, tau in zip(('forward', 'reverse', 'off'), pulse_report.get('differential_tau_s', [])):
            quantity(f'step_{index}.differential_tau_{phase}_s', tau, 's')
        for name in ('C_dl_F_cm2', 'reverse_capacitive_fraction', 'diagnostic_reverse_capacitive_fraction'):
            if name in pulse_report:
                quantity(f'step_{index}.{name}', pulse_report[name], 'F/cm2' if name == 'C_dl_F_cm2' else '1')
    for index, additive_report in additives_reports:
        for name, value in _additive_quantities(index, additive_report):
            quantity(name, value, _additive_unit(name))
    reported_names = {q.name for q in quantities}
    for index, last_theta_a in theta_a_reports:
        for name, value in last_theta_a.items():
            qname = f'step_{index}.{name}'
            if qname not in reported_names:
                quantity(qname, value, '1')
                reported_names.add(qname)
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
    if spec.numerics.pulse_model == 'resolved':
        sensitivity = ('Frozen-front cycle-resolved Tafel kinetics and periodic sub-grid diffusion; '
                       'uncalibrated anodic kinetics; grain refinement/nucleation not modelled.')
    elif spec.additives is not None:
        sensitivity = ('Uncalibrated L3 additive coverages use the cycle-mean current; pulse timing '
                       'and reverse-current effects beyond the cycle average are not resolved.')
    else:
        sensitivity = ('L1 uses only the net current; pulse timing and reverse-current effects '
                       'beyond the cycle average are not resolved.')
    if spec.additives is None:
        lead = ('Uncalibrated L1, CPU, no additives or resolved flow. '
                if not spec.setup.stir_shear_1_s else 'Uncalibrated L1, CPU, no additives. ')
    else:
        lead = (f'Uncalibrated L3 (additives mode "{spec.additives.mode}"), CPU; '
                'all additive quantities are uncalibrated engine defaults. ')
        if spec.additives.ceac:
            lead += _CEAC_NOTE + ' '
    if spec.setup.stir_shear_1_s:
        lead += _shear_assumption(spec) + ' '
    if spec.setup.buoyancy:
        lead += _buoyancy_assumption() + ' '
    manifest = Manifest(spec_hash=spec_hash(spec), code_version=__version__, backend='numpy',
        device='cpu', dx_um=grid.dx * 1e4, physics=spec.numerics.physics, created_utc=datetime.now(timezone.utc),
        quantities=quantities, checks=[Check(name='faraday_error_rel', value=error, limit=.05, passed=error <= .05),
        Check(name='cu_balance_rel', value=worst_balance, limit=.01, passed=worst_balance <= .01)],
        waveform_sensitivity=sensitivity,
        artifacts=artifacts, summary_text=lead +
        'One result retains original silicon and cumulative step-end frames. '
        'Seed thinning is not modelled; conductivity and kinetics use uncalibrated engine defaults. '
         'Faraday volume correction is disabled; checks report actual numerical drift.'
         + (' ' + _coupon_assumption(spec) if local_conditions else ''))
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
    if spec.numerics.physics not in ('L1', 'L3') or spec.numerics.use_surrogate:
        raise ValueError('numerics: only physics="L1" or "L3", use_surrogate=False supported; fix these fields.')
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
    if spec.additives is None:
        head = 'Uncalibrated CPU L1.'
        no_add = ('No additives, resolved flow or seed thinning; conductivity/kinetics are uncalibrated defaults. '
                  if not spec.setup.stir_shear_1_s else
                  'No additives or seed thinning; conductivity/kinetics are uncalibrated defaults. '
                  + _shear_assumption(spec) + ' ')
    else:
        head = f'Uncalibrated CPU L3 (additives mode "{spec.additives.mode}").'
        if spec.additives.ceac:
            head += ' ' + _CEAC_NOTE
        no_add = ('All additive quantities are uncalibrated engine defaults. '
                  + (_shear_assumption(spec) + ' ' if spec.setup.stir_shear_1_s else ''))
    buoy_note = (_buoyancy_assumption() + ' ') if spec.setup.buoyancy else ''
    payload = dict(template, dx_um=max(p['dx_um'] for p in plans), quantities=quantities,
                   checks=checks, artifacts=artifacts,
                   summary_text=head + ' Mid-length tiles are the profile result. '
                   'Mirrored renders/analysis reconstruct centre cuts; end cuts are not reflected. '
                   'Weights count physical copies, not symmetry sectors; do not sum overlapping diagnostic volumes. '
                   'Deposited volumes are simulated-sector volumes, while void analysis uses the reflected view. '
                   + no_add + buoy_note + _coupon_assumption(spec) + ' | '.join(descriptions))
    payload = Manifest.model_validate(payload).model_dump(mode='json')
    temporary = out / 'manifest.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(out / 'manifest.json')
    if progress:
        progress(1., 'manifest complete')
    return payload
