"""PUBLIC synthetic continuum pulse study; no team geometry or calibration data.

Run with PYTHONPATH=. and NUMBA_NUM_THREADS=4. Example:
  python tools/pulse_roughness.py --out C:/path/to/pulse-study.json
Reports actual level-set amplitude evolution, not an assumed PR smoothing factor.
The 3 um case has only six cells/wavelength at default dx: interpret it as a
resolution diagnostic, not a converged grain-scale prediction. Grain refinement,
nucleation, texture, and additives are outside this continuum model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from platesim.feature.engine.params import Params, derive
from platesim.tile3d.grid import Grid3D
from platesim.tile3d import levelset3d as ls
from platesim.tile3d.sim import run_tile, top_height

PR = dict(t_f_ms=9., t_r_ms=.5, t_off_ms=.5,
          j_f_mA_cm2=18.75, j_r_mA_cm2=37.5, model='resolved')
UM = 1e-4


def amplitude(front, grid, wavelength):
    x = grid.centers()[0] / UM
    h = np.mean(top_height(front, grid), axis=1) / UM
    harmonic = np.exp(-2j * np.pi * x / wavelength)
    return float(2 * abs(np.mean((h - h.mean()) * harmonic)))


def flat_case(wavelength, dx, duration, model, capacitance=0., amp=.25):
    grid = Grid3D.from_um((2*wavelength, 2*dx, 44.), dx, origin_um=(0., 0., -4.))
    x, _, z = grid.meshgrid()
    # Re-distance the initial perturbation to avoid an arbitrary |grad psi| bias.
    psi = ls.reinit(z - amp*UM*np.cos(2*np.pi*x/(wavelength*UM)),
                    grid.dx, periodic=grid.periodic)
    amps, times = [], []
    def save(t, front, row):
        times.append(t)
        amps.append(amplitude(front, grid, wavelength))
    started = time.perf_counter()
    result = run_tile(psi, grid, 15., duration, delta_um=30.,
        params=Params(suppressor=False, C_dl_F_cm2=capacitance),
        pulse=PR if model == 'resolved' else None, backend='numpy', n_frames=5,
        dt_max=30., frame_callback=save)
    rate = float(np.polyfit(times, np.log(np.maximum(amps, 1e-30)), 1)[0])
    return dict(case='sinusoid', wavelength_um=wavelength, dx_um=dx, model=model, C_dl_F_cm2=capacitance,
        cells=int(np.prod(grid.shape)), duration_s=duration, amplitude_initial_um=amps[0],
        amplitude_final_um=amps[-1], log_amplitude_rate_s=rate,
        amplitude_rate_um_min=(amps[-1]-amps[0])/duration*60,
        wall_s=time.perf_counter()-started, faraday_error_rel=result['summary']['faraday_error_rel'],
        times_s=times, amplitudes_um=amps, pulse=result['summary'].get('pulse'))


def trench_case(duration, model, capacitance=0.):
    grid = Grid3D.from_um((240., 16., 320.), 8., origin_um=(-120., 0., -96.))
    x, _, z = grid.meshgrid()
    # Signed union of the upper half-space and a rectangular electrolyte cavity.
    cavity = np.minimum(48*UM-np.abs(x), z+80*UM)
    psi = ls.reinit(np.maximum(z, cavity), grid.dx, periodic=grid.periodic)
    h0 = top_height(psi, grid)
    started = time.perf_counter()
    result = run_tile(psi, grid, 15., duration, delta_um=64.,
        params=Params(suppressor=False, C_dl_F_cm2=capacitance),
        pulse=PR if model == 'resolved' else None, backend='numpy', n_frames=2, dt_max=60.)
    growth = np.mean(top_height(result['psi'], grid)-h0, axis=1)/UM
    xx = grid.centers()[0]/UM
    field, floor = growth[np.abs(xx)>80].mean(), growth[np.abs(xx)<16].mean()
    corner = growth[(np.abs(xx)>48) & (np.abs(xx)<72)].mean()
    return dict(case='trench', model=model, C_dl_F_cm2=capacitance, duration_s=duration, dx_um=8.,
        field_growth_um=float(field), floor_growth_um=float(floor), corner_growth_um=float(corner),
        floor_over_field=float(floor/field), corner_over_field=float(corner/field),
        wall_s=time.perf_counter()-started, faraday_error_rel=result['summary']['faraday_error_rel'],
        pulse=result['summary'].get('pulse'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--duration', type=float, default=180.)
    parser.add_argument('--dx', type=float, default=.5)
    parser.add_argument('--wavelengths', type=float, nargs='+', default=[3., 6., 12., 24.])
    parser.add_argument('--amp-um', type=float, default=.25)
    parser.add_argument('--skip-trench', action='store_true')
    parser.add_argument('--capacitances-uF', type=float, nargs='+', default=[0., 30.])
    args = parser.parse_args()
    if args.duration <= 0 or not 0 < args.dx <= .5:
        parser.error('duration must be positive and 0 < dx <= 0.5 um')
    rows = []
    cases = [('net', 0.)] + [('resolved', c*1e-6) for c in args.capacitances_uF]
    if any(not np.isfinite(c) or c < 0 for c in args.capacitances_uF):
        parser.error('capacitances must be finite and nonnegative')
    for wavelength in args.wavelengths:
        for model, capacitance in cases:
            row = flat_case(wavelength, args.dx, args.duration, model, capacitance, args.amp_um)
            rows.append(row)
            print(json.dumps(row), flush=True)
    for model, capacitance in ([] if args.skip_trench else cases):
        row = trench_case(args.duration, model, capacitance)
        rows.append(row)
        print(json.dumps(row), flush=True)
    p = Params(suppressor=False)
    ph = derive(p, 22.)
    payload = dict(waveform=PR, rows=rows, bath=dict(c_bulk_mol_L=p.c_bulk*1e3,
        kappa_S_cm=ph.kappa, i0_A_cm2=ph.i0_u, alpha_c=p.alpha_u, alpha_a=p.pulse_alpha_a,
        Wa_forward_100um=ph.kappa/(p.alpha_u*ph.f_RT*.01875*.01),
        Wa_reverse_100um=ph.kappa/(p.pulse_alpha_a*ph.f_RT*.0375*.01)), assumptions=[
        'Synthetic public geometry; no team data.',
        'Frozen geometry per cycle; normal-column semi-infinite oscillatory diffusion.',
        'sqrt(D*T) may not be small versus 3 um wavelength: short-wave results are exploratory.',
        'No grain refinement, nucleation, texture, additives or seed depletion.',
        'RC model is periodic and lumped over interface area, not a spatial transient RC solve.',
        'Reference: pulse_reverse_literature.md sections 5-7; alpha_a=1.5 and i0=2 mA/cm2 are uncalibrated.',
        'Review Wa~150 uses kappa=0.5 S/cm; engine default kappa=0.065 S/cm is reported separately.',
        'No Faraday volume correction: actual discretisation drift is reported.',
        'Rates include numerical level-set smoothing; perform dx/dt convergence before physical inference.'])
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
