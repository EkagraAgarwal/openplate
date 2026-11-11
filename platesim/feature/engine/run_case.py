"""run_case(): run one feature-fill simulation (via or trench) and return / write a compact JSON-able dict.

CLI:  python -m platesim.feature.engine.run_case --d 100 --T 40 --recipe plan_ramp [--geometry trench]
      [--f_open 0.05] [--out path.json] [--set k_minus=3e5 ...]
Default output directory: $PLATESIM_FEATURE_OUT or <temp>/platesim_feature_runs/<label>.json
(pass out=... explicitly to place the JSON anywhere; out=None writes nothing).

JSON layout
  meta   : inputs, parameters, grid, wall-clock, run diagnostics
  series : <= 400 points per key (t_s, V [V], eta_floor/eta_face [V], I_total [A], fill_fraction, face_thickness_um,
           floor_height_um, void_volume_frac, theta_floor, theta_face, c_floor = c/c_bulk at the floor interface)
  frames : ~60 evenly spaced in time + final. contour = list of polylines [[r_um/x_um, z_um], ...] of psi = 0
           (half plane >= 0), voids = polylines of trapped pockets,
           cfield_b64 = base64(zlib(uint8 [64 rows (z, bottom first)][48 cols (r/x)]))
           with 0 = metal, 1..255 -> c/c_bulk = (v-1)/254, extent = [0, R, 0, z_view] um
  result : outcome in {filled_void_free, void, seam, keyhole, incomplete}, fill_time_s, final overburden, voids
Use decode_cfield(frame) to get the 64x48 float array (NaN in metal)."""
from __future__ import annotations
import argparse
import base64
import dataclasses
import json
import os
import tempfile
import time
import zlib

import numpy as np

from .params import Params, derive
from .recipes import Recipe, by_name, potentiostatic
from .sim import ViaSim

DEFAULT_OUT_DIR = os.environ.get("PLATESIM_FEATURE_OUT",
                                 os.path.join(tempfile.gettempdir(), "platesim_feature_runs"))


def decode_cfield(frame):
    """c/c_bulk on the coarse grid (64 z rows x 48 r cols), NaN in metal."""
    raw = np.frombuffer(zlib.decompress(base64.b64decode(frame['cfield_b64'])), np.uint8)
    a = raw.reshape(frame['cfield_shape']).astype(float)
    out = (a - 1) / 254.0
    out[a == 0] = np.nan
    return out


def _clean(x):
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    return x


def _decimate(arr, n=400):
    arr = list(arr)
    if len(arr) <= n:
        return arr
    idx = np.unique(np.linspace(0, len(arr) - 1, n).round().astype(int))
    return [arr[i] for i in idx]


def _series_json(series, nmax=400):
    L = len(series['t_s'])
    idx = np.arange(L) if L <= nmax else np.unique(np.linspace(0, L - 1, nmax).round().astype(int))
    out = {}
    for k, v in series.items():
        v = np.asarray(v, float)
        out[k] = [None if not np.isfinite(v[i]) else round(float(v[i]), 6 if k != 't_s' else 2) for i in idx]
    return out


def _frames_json(frames, nmax=61):
    if len(frames) > nmax:
        idx = np.unique(np.concatenate([np.linspace(0, len(frames) - 2, nmax - 1).round().astype(int),
                                        [len(frames) - 1]]))
        frames = [frames[i] for i in idx]
    out = []
    for f in frames:
        c = f['cfield']
        out.append(dict(
            t_s=round(f['t_s'], 1),
            contour=[ln.tolist() for ln in f['contour']],
            voids=[ln.tolist() for ln in f['voids']],
            cfield_b64=base64.b64encode(zlib.compress(c.tobytes(), 9)).decode('ascii'),
            cfield_shape=list(c.shape),
            extent=[round(e, 1) for e in f['extent']]))
    return out


def run_case(size_um, T_C, recipe: Recipe, params: Params | None = None, f_open=0.05, label='', out=None,
             return_sim=False, progress=None):
    """Run one simulation. size_um = via diameter / trench full width (w = d). Returns the JSON-able result
    dict (and writes it if out given). progress(frac, msg) is forwarded to the time loop."""
    p = params if params is not None else Params()
    t0 = time.time()
    sim = ViaSim(size_um, T_C, p, f_open, recipe)
    g = sim.g
    raw = sim.run(label, progress=progress)
    # ---- result classification
    voids = raw['voids']
    cav0 = sim.cav0
    if voids:
        big = max(voids, key=lambda v: v['vol'])
        outcome = {'seam': 'seam', 'keyhole': 'keyhole'}.get(big['cls'], 'void')
    elif raw['fill_final'] >= p.fill_stop:
        outcome = 'filled_void_free'
    else:
        outcome = 'incomplete'
    result = dict(
        outcome=outcome,
        stop_reason=raw['stop_reason'],
        fill_time_s=None if raw['fill_time'] is None else round(raw['fill_time'], 1),
        t_end_s=round(raw['t_end'], 1),
        fill_fraction_final=round(raw['fill_final'], 5),
        cu_balance_rel=raw['cu_balance_rel'] if raw['cu_balance_rel'] is None else round(raw['cu_balance_rel'], 6),
        cu_balance_adv_rel=(raw['cu_balance_adv_rel'] if raw['cu_balance_adv_rel'] is None
                            else round(raw['cu_balance_adv_rel'], 6)),
        overburden_um=round(raw['face_thickness_um'], 2),
        void_volume_frac=round(sum(v['vol'] for v in voids) / cav0, 5),
        voids=[dict(cls=v['cls'], t_formed_s=round(v['t_s'], 1), z0_um=round(v['z0'] / 1e-4, 1),
                    z1_um=round(v['z1'] / 1e-4, 1), rmax_um=round(v['rmax'] / 1e-4, 1),
                    volume_frac=round(v['vol'] / cav0, 5)) for v in voids])
    meta = dict(
        label=label, size_um=size_um, T_C=T_C, f_open=f_open, recipe=recipe.to_dict(), params=p.to_dict(),
        phys={k: float(v) for k, v in dataclasses.asdict(derive(p, T_C)).items()
              if not isinstance(v, str)},
        grid=dict(nr=g.nr, nz=g.nz, dx_um=round(g.dx_um, 4), R_um=round(g.R_um, 3), a_um=g.a_um, H_um=g.H_um,
                  geometry=g.geometry,
                  z_lo_um=round(g.z_lo / 1e-4, 3), z_top_um=round(g.z_top / 1e-4, 2),
                  eps_um=round(p.eps_fac * g.dx_um, 4)),
        wall_s=round(time.time() - t0, 2), steps=raw['steps'], max_balance_error=raw['max_bal'],
        mean_fixed_point_iterations=round(raw['mean_it'], 2),
        faraday=dict(deposited_volume_cm3=raw['solid_gain'],
                     charge_volume_cm3=raw['q_total'] * derive(p, T_C).Mfac),
        cfield_encoding='base64(zlib(uint8)); 0=metal, 1..255 -> c/c_bulk=(v-1)/254', numpy=np.__version__)
    res = dict(meta=meta, series=_series_json(raw['series']), frames=_frames_json(raw['frames']), result=result)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, 'w') as fh:
            json.dump(res, fh, separators=(',', ':'), default=_clean)
    return (res, sim) if return_sim else res


# ====================================================================================== CLI
def _parse_set(items):
    kv = {}
    for it in items or []:
        k, v = it.split('=', 1)
        if v.lower() in ('true', 'false'):
            kv[k] = v.lower() == 'true'
        elif v.lower() == 'none':
            kv[k] = None
        else:
            kv[k] = float(v)
    return kv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--d', type=float, required=True, help='via diameter / trench full width, um')
    ap.add_argument('--geometry', default='via', choices=['via', 'trench', 'trench_full'],
                    help='via = axisymmetric (r, z); trench = planar 2-D (x, z) half-domain; '
                         'trench_full = planar 2-D full pitch, periodic in x')
    ap.add_argument('--T', type=float, default=40.0, help='temperature, C')
    ap.add_argument('--recipe', default='plan_ramp',
                    help='dc | team_staged | plan_ramp | plan_ramp_pr | potentiostatic')
    ap.add_argument('--j', type=float, default=3.0, help='dc: mA/cm2')
    ap.add_argument('--tmax_h', type=float, default=6.0, help='dc: max duration, h')
    ap.add_argument('--pot', default='', help='potentiostatic: "dur_s:V_mV,dur_s:V_mV"')
    ap.add_argument('--f_open', type=float, default=0.05)
    ap.add_argument('--H', type=float, default=300.0)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--set', nargs='*', default=[], help='parameter overrides, e.g. k_minus=3e5 ohmic=false')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args(argv)
    kv = _parse_set(a.set)
    kv['H_um'] = a.H
    kv['geometry'] = a.geometry
    if a.verbose:
        kv['verbose'] = True
    p = Params(**kv)
    if a.recipe == 'potentiostatic':
        rec = potentiostatic([(float(x.split(':')[0]), float(x.split(':')[1])) for x in a.pot.split(',')])
    else:
        rec = by_name(a.recipe, a.T, j=a.j, t_max_h=a.tmax_h)
    label = a.label or f'{a.geometry[:2]}{a.d:g}_T{a.T:g}_{rec.name}'
    out = a.out or os.path.join(DEFAULT_OUT_DIR, label + '.json')
    res = run_case(a.d, a.T, rec, p, a.f_open, label, out)
    r = res['result']
    print(f"{label}: outcome={r['outcome']} stop={r['stop_reason']} fill_time_s={r['fill_time_s']} "
          f"fill={r['fill_fraction_final']} overburden_um={r['overburden_um']} wall={res['meta']['wall_s']}s "
          f"steps={res['meta']['steps']} max_bal={res['meta']['max_balance_error']:.2e} -> {out}")
    return res


if __name__ == '__main__':
    main()
