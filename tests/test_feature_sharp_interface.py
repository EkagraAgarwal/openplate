"""Plan v2 C4: sharp-interface (cut-cell / ghost-fluid) interface fluxes -- ``Params.interface='sharp'``.

The legacy transport geometry represents the front by the smoothed band
``Hs = 0.5(1+tanh(psi/eps))``: the interface sink is spread over ~1.2 dx, every face conductance is
weighted by ``Hs`` and the interface weights are ``w_p = V_p |grad Hs|_p``.  ``interface='sharp'``
replaces that with a cut-cell description built from the same level set (fields.StepGeom):

  * cell electrolyte fractions ``ef`` (levelset.electrolyte_fraction) carry the storage term,
  * face apertures from a linear interpolation of psi along every face carry the conductances
    (exactly 1 for a face wholly on the electrolyte side, 0 for one wholly in the metal),
  * the interface weights ``w_p`` are the chord of the linear level-set reconstruction inside a cut
    cell (exact for a straight interface), placed on the sink node -- the cell itself when its centre
    is in the electrolyte, otherwise the neighbour along the normal (the small-cell rule),
  * the sink is rejected to the ghost-fluid distance ``|psi|`` of that node
    (``k_eff = k/(1+k|psi|/D)``), which makes the 1-D planar steady limiting current exact.

What the tests pin down:
  1. the switch and its validation;
  2. geometry: apertures for straight interfaces, the interface area (== the front length), the
     sink-node rule, and the M-matrix property of the assembled operator;
  3. 'diffuse' is bit-identical to the legacy geometry arrays;
  4. the goal: the 1-D flat-electrode limiting current at a NON-grid-aligned front height, dx 8 and 4,
     against the analytic ``D c_bulk / L`` -- sharp within 1 % at dx 8, diffuse far outside it (the
     numbers are printed; they are ~1 % vs ~100 % at dx 8 in this configuration);
  5. a short trench_full run: completes, the Cu2+ balance closes, floor/field printed for both
     representations;
  6. grid comparison of floor/field at dx 8 and 4 for a 0.5 h Stokes run (G = 59 1/s) for both
     representations.
"""
from __future__ import annotations

import numpy as np
import pytest

from platesim.feature.engine.fields import EF_FLOOR, StepGeom, StepSolver, _face_apertures
from platesim.feature.engine.levelset import UM, Grid
from platesim.feature.engine.params import Params, derive
from platesim.feature.engine.recipes import pulse_reverse
from platesim.feature.engine.run_case import run_case
from platesim.feature.ripple_study import surface_stats

F_C = 96485.0
D_UM, H_UM, F_OPEN, DELTA_UM = 227.0, 88.0, 0.2, 64.0
J_NET = 15.0


# ---------------------------------------------------------------------------------------- helpers
def _params(interface, **kw):
    base = dict(geometry="trench_full", suppressor=False, ohmic=True, dx_um=8.0, delta_um=DELTA_UM,
                k_mix=0.0, max_overburden_um=140.0, H_um=H_UM, flow_shear_s=0.0)
    base.update(kw)
    return Params(interface=interface, **base)


def _recipe(hours):
    return pulse_reverse(J_NET, hours, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)


def _slab(dx_um, z_off_cells, L_um, interface):
    """1-D flat slab (f_open = 1) with the front at (m + z_off_cells) dx and the Dirichlet plane L above.

    psi = z - z_front is the exact signed distance of a flat front (this is what initial_psi returns for
    f_open >= 0.999 once the front has grown to z_front), every column is identical."""
    p = Params(geometry="trench_full", suppressor=False, ohmic=False, dx_um=dx_um, delta_um=L_um,
               k_mix=0.0, max_overburden_um=L_um + 30.0, H_um=22.0, flow_shear_s=0.0,
               interface=interface)
    ph = derive(p, 22.0)
    g = Grid(227.0, 22.0, 1.0, L_um, dx_um=dx_um, max_overburden_um=L_um + 30.0,
             geometry="trench_full")
    z_f = (1.0 + z_off_cells) * g.dx
    psi = np.broadcast_to(g.zc - z_f, (g.nr, g.nz)).copy()
    geo = StepGeom(g, psi, z_f + L_um * UM, p)
    return p, ph, g, geo, z_f


def _solve_slab(p, ph, g, geo, V_set=-0.63):
    """Potentiostatic StepSolver solve; returns (interface current density, micrometer L of the plane)."""
    shp = (g.nr, g.nz)
    st = dict(theta=np.zeros(shp), theta_a=np.zeros(shp), s=np.ones(shp), ctil=np.ones(shp),
              phi=np.zeros(shp), phi_r=np.zeros(shp), V=0.0, V_r=0.0)
    solver = StepSolver(geo, p, ph, np.zeros(shp))
    ctrl = dict(mode="V", V_set=V_set, I_t=1.0, duty_f=1.0, duty_r=0.0, rr=0.0, pr=False)
    out = solver.solve(st, ctrl, 120.0)
    return float(np.dot(geo.w, out["i_f"]) / geo.w.sum()), out["info"]


# ------------------------------------------------------------------------------- 1. the switch
def test_interface_switch_defaults_to_diffuse_and_is_validated():
    assert Params().interface == "diffuse"
    derive(Params(interface="diffuse"), 22.0)            # explicit legacy value is accepted
    ph = derive(Params(interface="sharp"), 22.0)         # sharp is accepted ...
    assert ph.D_c == derive(Params(), 22.0).D_c          # ... and changes no physical parameter
    with pytest.raises(ValueError, match="interface"):
        derive(Params(interface="banded"), 22.0)


# ---------------------------------------------------------------------------- 2. cut-cell geometry
def test_face_apertures_are_exact_for_straight_interfaces():
    """A flat front: every z-face is wholly on one side (1 above the front, 0 below) and the x-faces of
    the row that contains the front carry the open fraction of their height.  A 45 deg front: the
    apertures are 1/2 on the faces it crosses."""
    _, _, g, geo, z_f = _slab(8.0, 0.37, DELTA_UM, "sharp")
    ax, az = _face_apertures(geo.psi, g)
    zface = g.zc[0] + 0.5 * g.dx
    assert np.all(az[:, zface > z_f + 1e-12] == 1.0)
    assert np.all(az[:, zface < z_f - 1e-12] == 0.0)
    j0 = int(np.nonzero(zface >= z_f)[0][0])             # first face above the front = cut cell
    assert az[0, j0] == pytest.approx(1.0)
    assert az[0, j0 - 1] == pytest.approx(0.0)
    # the x-faces of the cut cell carry the open length fraction of their height
    frac = (g.zc[0, j0] + 0.5 * g.dx - z_f) / g.dx       # upper end of cell j0 is zc[j0] + dx/2
    assert frac == pytest.approx(0.63, abs=1e-12)
    assert np.allclose(ax[:, j0], frac)
    assert np.all(ax[:, j0 - 1] == 0.0) and np.all(ax[:, j0 + 1] == 1.0)

    # 45 deg front (psi = x + z - c): the aperture of every face is the exact fraction of its length
    # above the line,  clip(0.5 + psi_face / dx, 0, 1)  with psi_face the value at the face centre
    c = g.zc[0, g.nz // 2]
    psi = np.broadcast_to(g.rc, (g.nr, g.nz)) + g.zc - c
    ax2, az2 = _face_apertures(psi.copy(), g)
    assert ax2.min() >= 0.0 and ax2.max() <= 1.0 and az2.min() >= 0.0 and az2.max() <= 1.0
    zf_ = g.zc[0][None, :g.nz - 1] + 0.5 * g.dx                     # z-face centres
    exp_az = np.clip(0.5 + (g.rc + zf_ - c) / g.dx, 0.0, 1.0)
    # columns 0 / nr-1 are skipped: their corner average wraps across the periodic seam (psi is not
    # periodic in this artificial 45 deg field), which is a property of the test field, not of the scheme
    np.testing.assert_allclose(az2[1:g.nr - 1, :g.nz - 1], exp_az[1:g.nr - 1], atol=1e-12)
    xf_ = (np.arange(g.nr - 1) + 1.0)[:, None] * g.dx               # x-face centres
    exp_ax = np.clip(0.5 + (xf_ + g.zc[0][None, :] - c) / g.dx, 0.0, 1.0)
    # the bottom / top rows are skipped: their corner average uses the edge-padded psi of the domain
    # boundary (a property of the padding, not of the aperture rule)
    np.testing.assert_allclose(ax2[:g.nr - 1, 1:g.nz - 1], exp_ax[:, 1:g.nz - 1], atol=1e-12)


def test_interface_area_is_the_front_length_and_lies_on_the_sink_node():
    """sum(w) is the front length (the pitch, exactly, for a flat front) for sharp and for the legacy
    band, independent of where the front sits inside a cell; the sink node of a cut cell whose centre is
    in the metal is the neighbour on the electrolyte side, and every sink node carries a positive
    ghost-fluid distance."""
    _, _, g, geo0, _ = _slab(8.0, 0.37, DELTA_UM, "sharp")
    for z_off in (0.0, 0.02, 0.25, 0.5, 0.75, 0.98, 1.0):
        _, _, g, geo, z_f = _slab(8.0, z_off, DELTA_UM, "sharp")
        # 0.0 / 1.0 put the front exactly on a cell face (the initial condition of every run): the
        # chord must then be counted once, in the electrolyte-side cell
        assert abs(geo.w.sum() / g.R - 1.0) < 1e-12, (z_off, geo.w.sum() / g.R)
        assert np.all(geo.w >= 0.0) and np.all(geo.vh > 0.0)
        # interface weights only in the row the front cuts (and the one it was merged into)
        rows = np.nonzero(geo.w_full.any(axis=0))[0]
        assert rows.size <= 2 and geo.psi[0, rows[0]] > -1.5 * g.dx
        # sink nodes are on the electrolyte side, at a non-negative distance from the interface
        act = geo.act
        assert np.all(geo.psi[act & (geo.w_full > 0)] >= 0.0)
        assert np.all(geo.d_if[act & (geo.w_full > 0)] >= 0.0)
    # the legacy band spreads the same total area over several cells
    _, _, g, geod, _ = _slab(8.0, 0.37, DELTA_UM, "diffuse")
    assert abs(geod.w.sum() / g.R - 1.0) < 0.01
    assert np.nonzero(geod.w_full.any(axis=0))[0].size > 3


def test_small_cut_cells_are_merged_into_the_majority_neighbour():
    """A front just below a cell face leaves a 2 %-electrolyte sliver: the sliver keeps no sink (it moves
    to the majority neighbour above), the storage is floored at EF_FLOOR and the interface area is
    conserved by the merge."""
    _, _, g, geo, z_f = _slab(8.0, 0.98, DELTA_UM, "sharp")   # front 2 % below the face at 2 dx
    j0 = int(np.nonzero(g.zc[0] + 0.5 * g.dx > z_f)[0][0])   # the cell that contains the front
    assert geo.ef[0, j0] == pytest.approx(0.02, abs=1e-12)
    assert geo.w_full[0, j0] == 0.0                          # no sink in the sliver ...
    assert geo.w_full[0, j0 + 1] > 0.0                       # ... it sits in the neighbour above
    assert geo.psi[0, j0 + 1] > 0.0
    assert geo.vh[geo.idx[0, j0]] == pytest.approx(g.vol[0, 0] * EF_FLOOR)
    assert abs(geo.w.sum() / g.R - 1.0) < 1e-12              # area is conserved by the merge


def test_sharp_operator_is_an_m_matrix():
    """Every face conductance of the assembled operator is non-negative and the row sums are non-negative
    (the diagonal dominates), so the matrix stays an M-matrix / solvable -- for the cut cells in
    particular, where the apertures are < 1."""
    for interface in ("sharp", "diffuse"):
        _, ph, g, geo, _ = _slab(8.0, 0.37, DELTA_UM, interface)
        D = np.full((g.nr, g.nz), ph.D_c)
        data, bc = geo.K_data(D)
        diag = data[geo.diag_pos]
        assert np.all(diag > 0.0), interface
        off = data.copy()
        off[geo.diag_pos] = 0.0
        assert np.all(off <= 0.0), interface
        rowsum = np.asarray(_row_sums(geo, data), float).ravel()
        assert np.all(rowsum >= -1e-12 * diag.max()), (interface, rowsum.min())
        assert np.all(geo.w >= 0.0) and np.all(geo.vh >= 0.0)


def _row_sums(geo, data):
    import scipy.sparse as sp
    A = sp.csr_matrix((data, geo.indices, geo.indptr), shape=(geo.n, geo.n))
    return A.sum(axis=1)


def test_diffuse_geometry_arrays_are_bit_identical_to_the_legacy_default():
    """interface='diffuse' must reproduce the legacy geometry arrays exactly (the C4 switch is inert)."""
    p0 = _params("diffuse")
    p0.interface = "diffuse"
    _, _, g, geo_d, _ = _slab(8.0, 0.37, DELTA_UM, "diffuse")
    geo_l = StepGeom(g, geo_d.psi, geo_d.z_plane, p0)
    for name in ("Hs", "gHs", "w", "w_full", "vh", "Hf_r", "Hf_z", "Hf_w", "Hs_top", "n",
                 "ir", "jr", "iz", "jz", "ptop"):
        np.testing.assert_array_equal(np.asarray(getattr(geo_l, name)),
                                      np.asarray(getattr(geo_d, name)), err_msg=name)
    assert geo_d.sharp is False and geo_d.d_if is None and geo_d.ef is None


# --------------------------------------------------------------------- 3. the 1-D limiting current
def test_1d_flat_limiting_current_is_exact_for_the_sharp_interface():
    """Flat electrode, front at a NON-grid-aligned height (0.37 of a cell), Dirichlet plane L = 64 um
    above it, strong cathodic overpotential (D/k = 0.02 um << L, so the interface concentration is ~0):
    compare the interface flux with the analytic  D c_bulk / L  at dx 8 and dx 4, for both
    representations.  printed errors (dx 8 / dx 4): sharp ~0.03 %, diffuse ~100 % / ~30 %."""
    L_UM = 64.0
    err = {}
    for interface in ("diffuse", "sharp"):
        for dx in (8.0, 4.0):
            p, ph, g, geo, z_f = _slab(dx, 0.37, L_UM, interface)
            i_ana, info = _solve_slab(p, ph, g, geo)
            assert info["converged"], (interface, dx, info)
            i_lim = 2.0 * F_C * ph.D_c * p.c_bulk / (L_UM * UM)
            err[(interface, dx)] = i_ana / i_lim - 1.0
            print(f"1-D limiting current dx {dx:g} um {interface:8s}: "
                  f"i = {i_ana:.6f} A/cm2  analytic {i_lim:.6f}  error {err[(interface, dx)]:+.4%}")
    # the interface area is the front length, so the mean current density is the interface current density
    assert abs(err[("sharp", 8.0)]) < 0.01, err                    # plan v2 C4 target at dx 8
    assert abs(err[("sharp", 4.0)]) < 0.01, err
    assert abs(err[("sharp", 8.0)]) < 0.1 * abs(err[("diffuse", 8.0)]), err
    assert abs(err[("diffuse", 8.0)]) > 0.05, err                  # the band really is biased here


# ------------------------------------------------------------------------- 4. short trench_full run
def test_short_trench_run_with_sharp_closes_the_cu_balance():
    """0.25 h at 15 mA/cm2, dx 8, enhanced transport: both representations complete, the Cu2+ balance
    closes for sharp and the floor/field ratio is printed for both."""
    out = {}
    for interface in ("diffuse", "sharp"):
        res = run_case(D_UM, 22.0, _recipe(0.25), _params(interface), f_open=F_OPEN,
                       label=f"c4_{interface}", out=None)
        r, st = res["result"], surface_stats(res)
        assert r["stop_reason"] in ("recipe_end", "filled_void_free"), (interface, r)
        out[interface] = (float(r["cu_balance_rel"]), float(st["ratio_mean"]),
                          float(st["floor_mean_um"]), float(st["field_mean_um"]))
    for interface, (bal, ratio, floor, field) in out.items():
        print(f"0.25 h dx 8 {interface:8s}: cu_balance_rel={bal:+.5f} floor/field={ratio:.5f} "
              f"floor={floor:.3f} um field={field:.3f} um")
        assert abs(bal) < 0.005, (interface, bal)


def test_flat_slab_run_with_sharp_starts_from_a_grid_aligned_front():
    """f_open = 1 flat slab, 0.25 h: the initial level set puts the front exactly ON a cell face, where
    the chord must be counted once.  Both representations complete, the balance closes and they deposit
    the same volume within 2 %."""
    fill = {}
    for interface in ("diffuse", "sharp"):
        p = _params(interface, max_overburden_um=140.0, H_um=88.0)
        res = run_case(227.0, 22.0, _recipe(0.25), p, f_open=1.0, label=f"c4_flat_{interface}", out=None)
        r = res["result"]
        assert r["stop_reason"] in ("recipe_end", "filled_void_free"), (interface, r)
        assert abs(float(r["cu_balance_rel"])) < 0.005, (interface, r["cu_balance_rel"])
        fill[interface] = float(r["fill_fraction_final"])
    print("flat slab 0.25 h fill fraction:", {k: round(v, 5) for k, v in fill.items()})
    assert abs(fill["sharp"] / fill["diffuse"] - 1.0) < 0.02, fill


# --------------------------------------------------------------- 5. grid comparison (0.5 h, Stokes)
def test_floor_to_field_ratio_is_grid_independent_for_sharp():
    """0.5 h at 15 mA/cm2, narrow trench (227 x 88 um, f_open 0.2, delta 64 um), Stokes flow G = 59 1/s:
    floor/field at dx 8 and dx 4 for both representations.  Printed: diffuse 0.892/0.873,
    sharp 0.870/0.870 -- the sharp front is grid independent to < 0.001, the band is not."""
    r = {}
    for interface in ("diffuse", "sharp"):
        for dx in (8.0, 4.0):
            res = run_case(D_UM, 22.0, _recipe(0.5),
                           _params(interface, dx_um=dx, flow_solver="stokes", flow_shear_s=59.0),
                           f_open=F_OPEN, label=f"c4_{interface}_dx{dx:g}", out=None)
            bal = float(res["result"]["cu_balance_rel"])
            assert abs(bal) < 0.005, (interface, dx, bal)
            r[(interface, float(dx))] = float(surface_stats(res)["ratio_mean"])
    print("0.5 h floor/field (Stokes G = 59):",
          {f"{k[0]} dx{k[1]:g}": round(v, 5) for k, v in r.items()})
    d_sharp = abs(r[("sharp", 8.0)] - r[("sharp", 4.0)])
    d_diffuse = abs(r[("diffuse", 8.0)] - r[("diffuse", 4.0)])
    assert d_sharp < 0.005, (d_sharp, r)
    assert d_sharp < d_diffuse, (d_sharp, d_diffuse, r)


# --------------------------------------------------------------- 6. the switch is inert by default
def test_sharp_and_diffuse_agree_when_the_front_is_grid_aligned_and_activation_controlled():
    """A weak overpotential (D/k >> L, i.e. kinetic control) makes the ghost-fluid factor ~1: both
    representations then describe the same physics and agree within a few percent."""
    L_UM = 64.0
    got = {}
    for interface in ("diffuse", "sharp"):
        p, ph, g, geo, _ = _slab(8.0, 0.37, L_UM, interface)
        got[interface], _ = _solve_slab(p, ph, g, geo, V_set=-0.10)
    assert abs(got["sharp"] / got["diffuse"] - 1.0) < 0.05, got
    print("activation-controlled flat slab, dx 8:", {k: round(v, 6) for k, v in got.items()})
