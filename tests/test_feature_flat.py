"""Flat electrode (trench_full, f_open = 1): the delta plane must follow the flat front.

Regression for the empty-active-set crash.  `ViaSim.face_height()` used to pin the interface at
z = 0 for f_open >= 0.999, so `z_plane` stayed at `delta_um` above the electrode while the flat
front grew past it.  Once the front was more than a few cells above the plane no cell in rows
<= jt held electrolyte, the active set of `StepGeom` was empty, every `np.bincount` in
`StepGeom.K_data` fell back to int64 and `diag += bc` raised
`UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64')`.
"""
import numpy as np

from platesim.feature.engine.fields import StepGeom
from platesim.feature.engine.levelset import UM
from platesim.feature.engine.params import Params, derive
from platesim.feature.engine.recipes import pulse_reverse
from platesim.feature.engine.run_case import run_case
from platesim.feature.engine.sim import ViaSim

J_NET_MA_CM2 = 15.0


def _flat_params(**kw):
    base = dict(geometry="trench_full", suppressor=False, ohmic=True, dx_um=8.0, delta_um=64.0,
                k_mix=0.0, max_overburden_um=140.0, H_um=88.0, flow_shear_s=0.0)
    base.update(kw)
    return Params(**base)


def _recipe(t_max_h=0.25):
    return pulse_reverse(J_NET_MA_CM2, t_max_h, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)


def _front_um(res):
    """z [um] of the psi = 0 line interpolated onto a uniform x grid from the final frame contour."""
    frame = res['frames'][-1]
    lines = [np.asarray(ln, float) for ln in frame['contour'] if len(ln) >= 2]
    assert lines, 'final frame has no psi = 0 contour'
    pitch = float(res['meta']['grid']['R_um'])
    x = np.linspace(1.0, pitch - 1.0, 64)
    z = np.full(x.size, np.nan)
    for ln in lines:                                   # polylines of [x_um, z_um]
        o = np.argsort(ln[:, 0])
        lx, lz = ln[o, 0], ln[o, 1]
        sel = (x >= lx[0]) & (x <= lx[-1])
        z[sel] = np.interp(x[sel], lx, lz)
    return x, z


def _faraday_um(p, t_s):
    """Deposit thickness for the net charge of one flat pulse-reverse run (efficiency 1)."""
    return derive(p, 22.0).Mfac * J_NET_MA_CM2 * 1e-3 * t_s / UM


def test_flat_run_is_flat_and_faradaic():
    """The flat reproducer runs, stays flat across x and deposits the Faraday thickness."""
    p = _flat_params()
    res = run_case(227.0, 22.0, _recipe(), p, f_open=1.0, label="flat", out=None)
    assert res['result']['stop_reason'] in ('recipe_end', 'filled_void_free')
    x, z = _front_um(res)
    assert np.isfinite(z).all(), 'psi = 0 line does not span the whole pitch'
    assert z.std() < 0.5, f'field is not flat: std = {z.std():.3f} um over x'
    thick = float(z.mean())                            # flat electrode surface sits at z = 0
    far = _faraday_um(p, res['result']['t_end_s'])
    assert abs(thick / far - 1.0) < 0.10, f'mean thickness {thick:.2f} um vs Faraday {far:.2f} um'


def test_flat_delta_plane_follows_the_front():
    """face_height() tracks the flat front, so the active set never collapses behind the plane."""
    p = _flat_params()
    sim = ViaSim(227.0, 22.0, p, 1.0, _recipe())
    g = sim.g
    assert abs(sim.face_height()) < 1e-9              # initial_psi puts the wall at z = 0
    for grown_um in (0.0, 70.0, 150.0):               # front well above the initial 64 um plane
        psi = sim.psi - grown_um * UM
        z_face = sim.face_height(psi)
        assert abs(z_face / UM - grown_um) < 1e-6
        geo = StepGeom(g, psi, z_face + p.delta_um * UM, p)
        assert geo.n > 0, f'active set empty at front = {grown_um} um'
        assert geo.pr.size > 0 and geo.pz.size > 0
        assert geo.ptop.size == g.nr                  # the whole plane is Dirichlet
        geo.K_data(np.full((g.nr, g.nz), 1e-5))


def test_flat_run_with_flow():
    """flow_shear_s = None (resolved LBM shear) is the default and must work on the flat cell too."""
    p = _flat_params(flow_shear_s=None)
    res = run_case(227.0, 22.0, _recipe(0.1), p, f_open=1.0, label="flat_flow", out=None)
    assert res['result']['t_end_s'] > 0.0
    _, z = _front_um(res)
    assert np.isfinite(z).all() and z.std() < 0.5


def test_k_data_tolerates_an_empty_active_set():
    """Robustness: an empty active set returns float data instead of int64 bincount sums."""
    p = _flat_params()
    g = ViaSim(227.0, 22.0, p, 1.0, _recipe()).g
    geo = StepGeom(g, np.full((g.nr, g.nz), -1.0), p.delta_um * UM, p)
    assert geo.n == 0
    data, bc = geo.K_data(np.full((g.nr, g.nz), 1e-5))
    assert data.size == 0 and bc.size == 0
    assert data.dtype == np.float64 and bc.dtype == np.float64
