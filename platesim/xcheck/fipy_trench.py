"""Independent FiPy cross-check of platesim's additive-free trench deposition (plan.md Task 4).

The model is a plain ``Grid2D`` re-implementation of the structure of FiPy's
``examples/levelSet/electroChem/simpleTrenchSystem.py`` (see
``research/fipy_trench_notes.md``): a ``DistanceVariable`` front, an implicit
first-order-in-c Cu2+ sink (``DiffusionTermNoCorrection`` + ``ImplicitSourceTerm``),
``extendVariable(order=2)`` and ``AdvectionTerm`` front transport, with the
time step taken from the interface CFL.

What is *not* copied from platesim: the field solvers.  The two codes share only
the grid arithmetic (``platesim.feature.engine.levelset.Grid`` is used to fix the
cell layout so the comparison is cell for cell) and the physical parameter values
(``Params``/``derive``).

Units are SI inside this module (m, s, mol/m3, A/m2); the platesim parameter set
is cm-based and converted once, at the top of :func:`run_fipy`.

Boundary condition for Cu2+: platesim applies ``c = c_bulk`` on the plane
``z_plane = z_face + delta`` that follows the field front (sim.py:178-181).  A
fixed FiPy ``mesh.facesTop`` is overtaken by the front after ~3.2 h at
15 mA/cm2, so this module moves the Dirichlet plane with the front instead:
every step the cells with ``z >= z_plane`` are pinned to ``c_bulk`` by explicit
row elimination on the matrix FiPy assembles (option (b) of the task, a
genuinely moving top boundary).  Row elimination is used because FiPy 4.0.3
never feeds ``CellVariable.constrain(..., where=cell_mask)`` into the linear
system - only into ``.value`` reads (fipy/variables/variable.py) - so a
cell-level Dirichlet written the obvious way leaves the operator without any
Dirichlet row, i.e. singular.  The pin sits on the first cell row at or above
``z_plane``, so the effective plane is within one cell (0-1 dx, mean 0.5 dx = 2 um
at dx 4) of platesim's plane: 0-6.25 % (mean 3.1 %) of delta = 64 um higher, which
is the leading known boundary-condition discretisation term of this comparison
(a few 1e-3 on floor/field, see ``summary.bc_choice`` in the JSON).

One ingredient is deliberately shared with platesim rather than being FiPy's own:
the Cu2+ face diffusivity is smoothed with ``Hs = 0.5(1+tanh(phi/1.2dx))``,
platesim's ``StepGeom.Hs`` (fields.py:40).  Everything else in the ion equation
(``TransientTerm``, ``DiffusionTermNoCorrection``, the implicit linear sink) is
FiPy's own assembly.

CLI::

    python -m platesim.xcheck.fipy_trench --sites narrow      # both codes, writes the JSON
    python -m platesim.xcheck.fipy_trench --sites all --fipy-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from ..core.constants import F, M_CU, RHO_CU

try:                                                    # fipy is an optional extra
    from fipy import (AdvectionTerm, CellVariable, DistanceVariable, Grid2D,
                      ImplicitSourceTerm, TransientTerm)
    from fipy.terms.diffusionTerm import DiffusionTermNoCorrection
    from fipy.tools import serialComm
    _FIPY_OK = True
    _FIPY_ERR = ""
    try:                                                # one cheap Krylov sweep: eq.solve below only has
        from fipy.solvers import LinearGMRESSolver       # to assemble A and b; the solve is scipy's
        _ASSEMBLE_SOLVER = LinearGMRESSolver(iterations=1, tolerance=1e9)
    except Exception:                                   # pragma: no cover
        _ASSEMBLE_SOLVER = None
except Exception as _exc:                               # pragma: no cover - import guard
    _FIPY_OK = False
    _FIPY_ERR = repr(_exc)

UM = 1e-6                       # um -> m
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fipy_trench.json")

SITES = {                       # data/trench_sem.json, Trench 1, additive-free DC cross-check
    "narrow": dict(d_um=227.0, H_um=88.0),
    "wide": dict(d_um=944.0, H_um=97.0),
}


# --------------------------------------------------------------------------------------- units / params
def _si_inputs(d_um, H_um, j_mA_cm2, T_C, f_open, delta_um, dx_um, max_overburden_um):
    """platesim cm-based parameter set -> SI constants used by the FiPy model."""
    from ..feature.engine.params import Params, derive
    from ..feature.engine.levelset import Grid

    p = Params(geometry="trench", H_um=H_um, ohmic=False, suppressor=False, k_mix=0.0,
               efficiency=1.0, delta_um=delta_um, dx_um=dx_um, max_overburden_um=max_overburden_um)
    ph = derive(p, T_C)
    g = Grid(d_um, H_um, f_open, delta_um, dx_um, max_overburden_um, geometry="trench")
    omega = M_CU / RHO_CU * 1e-6                    # cm3/mol -> m3/mol
    si = dict(
        p=p, ph=ph, g=g,
        T_K=ph.T_K,
        f_RT=ph.f_RT,                               # 1/V
        D=ph.D_c * 1e-4,                            # E_mig * D_cu, cm2/s -> m2/s
        D_cu=ph.D_cu * 1e-4,
        c_bulk=p.c_bulk * 1e6,                      # mol/cm3 -> mol/m3
        i0=ph.i0_u * 1e4,                           # A/cm2 -> A/m2
        alpha=p.alpha_u,
        Mfac=ph.Mfac * 1e-6,                        # cm3/C -> m3/C
        omega=omega,
        j=j_mA_cm2 * 10.0,                          # mA/cm2 -> A/m2 (=1e-3 A/mA * 1e4 cm2/m2)
        delta=delta_um * UM,
    )
    si["I_t"] = si["j"] * g.R * 1e-2               # A per unit length in y (platesim: j*R)
    # ---- Grid works in cm; the SI model needs m
    si.update(dx=g.dx * 1e-2, a=g.a * 1e-2, H=g.H * 1e-2, R=g.R * 1e-2,
              z_lo=g.z_lo * 1e-2, z_top=g.z_top * 1e-2)
    return si


# --------------------------------------------------------------------------------------- level-set helpers
def _calc_distance(phi, order=2):
    """calcDistanceFunction with an skfmm fallback (order 2 -> 1 -> leave the field alone)."""
    for o, tag in ((order, "order=2"), (1, "order=1"), (None, "no-op")):
        if o is None:
            return tag
        try:
            phi.calcDistanceFunction(order=o)
            return tag
        except Exception:
            continue
    return "no-op"


def _top_surface(phi, dx, z_lo):
    """Topmost metal->electrolyte crossing (absolute z, m) of every x column.

    Same definition as platesim's ``levelset.column_crossing(upper=True)`` applied
    to the last column (field, x = R) or the first column (trench centre, x = 0)."""
    _, shape = phi.getLSMshape()                     # (ny, nx), FiPy's own internal reshape
    ny, nx = shape
    d = np.asarray(phi.value).reshape(ny, nx)
    out = np.full(nx, np.nan)
    neg, pos = d[:-1, :] <= 0.0, d[1:, :] > 0.0
    for i in range(nx):
        idx = np.flatnonzero(neg[:, i] & pos[:, i])
        if idx.size:
            j = idx[-1]
            out[i] = j + d[j, i] / (d[j, i] - d[j + 1, i])
    return z_lo + (out + 0.5) * dx                   # (cell index + fraction) -> z in m


def _solid_area(phi, dx):
    """Sub-cell solid cross-section per unit length (m2) from the signed distance."""
    f = np.clip(0.5 - np.asarray(phi.value) / dx, 0.0, 1.0)
    return float(f.sum() * dx * dx)


def _decimate(series, n=60):
    if len(series) <= n:
        return series
    idx = np.unique(np.linspace(0, len(series) - 1, n).round().astype(int))
    return [series[i] for i in idx]


def _clean(x):
    """JSON-safe: no bare NaN/Inf (strict JSON)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


# --------------------------------------------------------------------------------------- FiPy model
def run_fipy(d_um, H_um, j_mA_cm2=15.0, hours=4.5, T_C=22.0, f_open=0.2, delta_um=64.0,
             dx_um=4.0, max_overburden_um=140.0, cfl=0.5, dt_max=120.0, reinit_every=8,
             eta_tol=1e-7, verbose=False):
    """Galvanostatic additive-free trench deposition in FiPy.  Returns a result dict."""
    if not _FIPY_OK:
        raise RuntimeError("fipy is not importable: " + _FIPY_ERR)
    t_wall = time.time()
    si = _si_inputs(d_um, H_um, j_mA_cm2, T_C, f_open, delta_um, dx_um, max_overburden_um)
    g, D, c_bulk, i0, alpha = si["g"], si["D"], si["c_bulk"], si["i0"], si["alpha"]
    f_RT, Mfac, omega, I_t = si["f_RT"], si["Mfac"], si["omega"], si["I_t"]
    a, H, dx, delta = si["a"], si["H"], si["dx"], si["delta"]
    warnings = []

    mesh = Grid2D(dx=dx, dy=dx, nx=g.nr, ny=g.nz, communicator=serialComm)
    x = np.asarray(mesh.cellCenters[0]).reshape(-1)
    z = np.asarray(mesh.cellCenters[1]).reshape(-1) + si["z_lo"]   # platesim cell centres

    # ---- solid seed: the same metal mask as platesim levelset.initial_psi (metal = z<0 or the wall
    #      x>a below z=H), but without platesim's corner_cells=1.0 dx reinit rounding: this is the
    #      plain signed distance function of the sharp mask, so the two seeds differ by up to that
    #      rounding at the two trench-mouth/floor corners only
    phi = DistanceVariable(mesh=mesh, value=1.0, hasOld=True)
    phi.setValue(-1.0, where=(z < 0.0) | ((x > a) & (z < H)))
    seed_tag = _calc_distance(phi, 2)
    if seed_tag != "order=2":
        warnings.append(f"initial calcDistanceFunction fell back to {seed_tag}")

    # ---- Cu2+ with an implicit, exactly linear-in-c sink at the interface
    c = CellVariable(mesh=mesh, name="c_Cu", value=c_bulk, hasOld=True)

    def _areas():
        return np.asarray(phi.cellInterfaceAreas.value).reshape(-1)

    def sink_coeff(gamma):
        """Cell coefficient [1/s] of -gamma*A_if/(V*omega)*c, gamma = v/c  (m/s per mol/m3)."""
        return gamma * _areas() / (dx * dx * omega)

    def current(eta, cvals, areas):
        return float(np.sum(areas * i0 * (cvals / c_bulk)
                            * np.exp(np.clip(-alpha * f_RT * eta, -60.0, 60.0))))

    def solve_eta(cvals, areas):
        """Scalar overpotential such that sum(A_if i) = I_t  (I is monotone in |eta|)."""
        def f(e):
            return current(e, cvals, areas) - I_t
        try:
            return brentq(f, -2.0, 0.0, xtol=1e-13, maxiter=200), 0.0
        except ValueError:                            # not bracketed (empty/near-empty interface)
            e = -2.0 if f(-2.0) < 0.0 else 0.0
            return e, f(e)

    def solve_c(eta, z_plane, dt):
        """One solve of the Cu2+ equation with the Dirichlet plane at z >= z_plane.

        D_face is smoothed with platesim's Hs = 0.5(1+tanh(phi/eps)), eps = 1.2 dx (fields.py
        StepGeom.Hs): upstream's sharp _LevelSetDiffusionVariable zeroes every face touching a solid
        cell, which isolates the solid-side interface cells and lets their sink drain them to c -> 0.

        The Dirichlet plane is imposed by row elimination on the matrix FiPy assembles.  FiPy 4.0.3
        does not apply `CellVariable.constrain(..., where=cell_mask)` to the linear system (only to
        `.value` reads; see fipy/variables/variable.py), so a cell-level Dirichlet has to be
        eliminated here."""
        gamma = Mfac * i0 * math.exp(min(-alpha * f_RT * eta, 60.0)) / c_bulk
        hs = CellVariable(mesh=mesh, value=0.5 * (1.0 + np.tanh(np.clip(np.asarray(phi.value) / (1.2 * dx),
                                                                      -30.0, 30.0))))
        eq = (TransientTerm() - DiffusionTermNoCorrection((hs * D).arithmeticFaceValue)
              + ImplicitSourceTerm(CellVariable(mesh=mesh, value=sink_coeff(gamma))))
        eq.cacheMatrix()
        eq.cacheRHSvector()
        if _ASSEMBLE_SOLVER is None:
            eq.solve(c, dt=dt)                        # assembles A and b (its own solution is replaced)
        else:
            eq.solve(c, dt=dt, solver=_ASSEMBLE_SOLVER)
        A = sp.csr_matrix(eq.matrix.matrix)
        b = np.asarray(eq.RHSvector).reshape(-1)
        fixed = z >= z_plane
        if fixed.any():
            keep = (~fixed).astype(float)
            A = sp.diags(keep) @ A + sp.diags(1.0 - keep)
            b = keep * b + (1.0 - keep) * c_bulk
        c.setValue(spla.spsolve(A.tocsc(), b).reshape(np.shape(c.value)))

    # ---- initial state: eta from the geometric interface, then relax c (finite dt: the solid rows have no
    #      diffusion term, so a dt -> infinity "steady" solve would be singular)
    areas = _areas()
    c.updateOld()
    eta, _ = solve_eta(np.asarray(c.value).reshape(-1), areas)
    for _ in range(2):
        solve_c(eta, H + delta, dt_max)
        eta, _ = solve_eta(np.asarray(c.value).reshape(-1), areas)

    # ---- time loop
    t, total, dt = 0.0, hours * 3600.0, 0.0
    series = []
    q = 0.0                                           # integral I_t dt, C per unit length
    n_steps = 0
    max_cur_res = max_eta_jump = max_galv_res = 0.0
    area0 = _solid_area(phi, dx)
    surf0 = _top_surface(phi, dx, si["z_lo"])
    eta_prev = eta
    while t < total - 1e-9:
        surf = _top_surface(phi, dx, si["z_lo"])
        z_face = float(surf[-1]) if np.isfinite(surf[-1]) else H
        z_plane = z_face + delta
        if z_plane >= si["z_top"] - 2.0 * dx:
            warnings.append(f"Dirichlet plane reached the domain top at t={t / 3600:.2f} h")
            break
        areas = _areas()
        c.updateOld()
        cv = np.asarray(c.value).reshape(-1)

        # ---- CFL from the interface velocity at the current c and eta
        i_if = np.where(areas > 0.0, i0 * (cv / c_bulk) * math.exp(min(-alpha * f_RT * eta_prev, 60.0)), 0.0)
        v_if = float(i_if.max() * Mfac) if (areas > 0.0).any() else 0.0
        dt = min(dt_max, cfl * dx / v_if) if v_if > 0.0 else dt_max
        dt = max(min(dt, total - t), 1e-3)

        # ---- Cu2+: c and eta to mutual consistency (c is quasi-steady; 2 sweeps are enough)
        eta, ok = eta_prev, False
        for _ in range(6):
            solve_c(eta, z_plane, dt)
            cv = np.asarray(c.value).reshape(-1)
            eta_new, res = solve_eta(cv, areas)
            max_cur_res = max(max_cur_res, abs(res))
            ok = abs(eta_new - eta) < eta_tol
            eta = eta_new
            if ok:
                break
        if not ok:
            warnings.append(f"eta fixed point not converged at t={t / 3600:.2f} h")
        max_eta_jump = max(max_eta_jump, abs(eta - eta_prev))
        # independent galvanostat check: re-evaluate the Tafel current from (c, eta) on the geometry
        # the constraint was solved on, instead of trusting solve_eta's return value
        i_act = i0 * (cv / c_bulk) * np.exp(np.clip(-alpha * f_RT * eta, -60.0, 60.0))
        max_galv_res = max(max_galv_res, abs(float(np.sum(areas * i_act)) - I_t) / I_t)

        # ---- front velocity, constant-extended along the normals, then AdvectionTerm
        i_cell = i0 * (cv / c_bulk) * np.exp(np.clip(-alpha * f_RT * eta, -60.0, 60.0))
        v_cell = np.where(areas > 0.0, Mfac * i_cell, 0.0)
        ext = CellVariable(mesh=mesh, value=np.where(np.asarray(phi.value) > 0.0, v_cell, 0.0))
        phi.updateOld()
        try:
            phi.extendVariable(ext, order=2)
        except Exception as exc:                      # skfmm can raise on degenerate input
            warnings.append(f"extendVariable failed at t={t / 3600:.2f} h: {exc!r}")
        (TransientTerm() + AdvectionTerm(ext)).solve(phi, dt=dt)
        n_steps += 1
        t += dt
        q += I_t * dt
        if n_steps % max(int(reinit_every), 1) == 0:
            tag = _calc_distance(phi, 2)
            if tag != "order=2":
                warnings.append(f"reinit fell back to {tag} at t={t / 3600:.2f} h")

        surf = _top_surface(phi, dx, si["z_lo"])
        # NB: series field_um is the ABSOLUTE z of the field surface (as the summary field_um was
        # before the -H conversion); the summary field_um is the overburden above H, like platesim's
        series.append(dict(t_h=round(t / 3600.0, 4), dt_s=round(dt, 2), eta_mV=round(eta * 1e3, 3),
                           field_um=round(float(surf[-1]) * 1e6, 3) if np.isfinite(surf[-1]) else None,
                           floor_um=round(float(surf[0]) * 1e6, 3) if np.isfinite(surf[0]) else None,
                           c_if_min=round(float(cv[areas > 0.0].min()) / c_bulk, 5) if (areas > 0).any() else None))
        eta_prev = eta
        if verbose and n_steps % 10 == 0:
            print(f"  t={t / 3600:6.3f} h dt={dt:6.1f} eta={eta * 1e3:8.2f} mV "
                  f"field={surf[-1] * 1e6:7.3f} floor={surf[0] * 1e6:7.3f} um", flush=True)

    surf = _top_surface(phi, dx, si["z_lo"])
    area_dep = _solid_area(phi, dx) - area0
    faraday_area = Mfac * q
    return dict(
        code="fipy-" + __import__("fipy").__version__,
        field_um=float(surf[-1] * 1e6) - H_um, floor_um=float(surf[0] * 1e6),
        field_mean_um=float(np.nanmean(surf) * 1e6) - H_um,
        mean_thickness_um=float(np.nanmean(surf - surf0) * 1e6),
        ratio=float(surf[0] / (surf[-1] - H)) if (surf[-1] - H) > 0 else None,
        eta_mV=float(eta * 1e3), t_end_h=t / 3600.0, steps=n_steps, dt_last_s=dt,
        charge_rel=float(area_dep / faraday_area - 1.0) if faraday_area > 0 else None,
        area_dep_m2=area_dep, faraday_area_m2=faraday_area, I_t_A_per_m=I_t,
        max_current_residual_A_per_m=max_cur_res, max_eta_step_mV=max_eta_jump * 1e3,
        galvanostat_residual_rel=max_galv_res,
        wall_s=time.time() - t_wall, warnings=warnings, series=_decimate(series),
        inputs=dict(d_um=d_um, H_um=H_um, j_mA_cm2=j_mA_cm2, hours=hours, T_C=T_C, f_open=f_open,
                    delta_um=delta_um, dx_um=dx_um, max_overburden_um=max_overburden_um,
                    cfl=cfl, dt_max_s=dt_max, reinit_every=reinit_every,
                    D_m2_s=D, D_cu_m2_s=si["D_cu"],
                    D_choice="E_mig*D_cu (migration enhanced, as fields.py:303)",
                    c_bulk_mol_m3=c_bulk, i0_A_m2=i0, alpha=alpha, Mfac_m3_C=Mfac,
                    omega_m3_mol=omega, F_C_mol=F, f_RT_per_V=f_RT,
                    grid=dict(nr=g.nr, nz=g.nz, dx_um=g.dx_um, R_um=g.R_um, a_um=g.a_um, H_um=g.H_um,
                              z_lo_um=g.z_lo * 1e4, z_top_um=g.z_top * 1e4),
                    bc_cu2="moving Dirichlet: cells with z >= z_face+delta pinned to c_bulk each step by "
                          "row elimination on the FiPy matrix (option b; matches sim.py:178-181, a fixed "
                          "facesTop would be overtaken by the front at ~3.2 h).  FiPy 4.0.3 does not put "
                          "cell constraints into the matrix, so the plane is imposed explicitly",
                    diffusion_face="Hs = 0.5(1+tanh(phi/1.2dx)) smoothed face D (shared with platesim "
                                   "fields.py:40 StepGeom.Hs) instead of upstream's sharp "
                                   "_LevelSetDiffusionVariable, which isolates the solid-side interface cells",
                    field_series_convention="series[*].field_um is the ABSOLUTE z of the field surface in um "
                                            "(H + overburden); the summary field_um is the overburden above H",
                    seed="sharp-mask signed distance function; platesim additionally rounds the trench-mouth "
                         "and floor corners with corner_cells=1.0 dx of reinit"))


# --------------------------------------------------------------------------------------- platesim reference
def run_platesim(d_um, H_um, j_mA_cm2=15.0, hours=4.5, T_C=22.0, f_open=0.2, delta_um=64.0,
                 dx_um=4.0, max_overburden_um=140.0, dt_max=120.0, verbose=False):
    """The same case through platesim's level-set engine (no parameter may differ except dt_max)."""
    from ..feature.engine.levelset import Grid, column_crossing, initial_psi
    from ..feature.engine.params import Params
    from ..feature.engine.recipes import dc
    from ..feature.engine.run_case import run_case

    p = Params(geometry="trench", H_um=H_um, ohmic=False, suppressor=False, k_mix=0.0, efficiency=1.0,
               delta_um=delta_um, dx_um=dx_um, max_overburden_um=max_overburden_um, dt_max=dt_max,
               verbose=verbose)
    t0 = time.time()
    res, sim = run_case(d_um, T_C, dc(j_mA_cm2, hours), p, f_open, label=f"xcheck_T4_{d_um:g}um", out=None,
                        return_sim=True)
    g = sim.g

    def surface(psi):
        out = np.full(psi.shape[0], np.nan)
        for i in range(psi.shape[0]):
            _, zc = column_crossing(psi[i], g.zc[0], upper=True)
            if zc is not None:
                out[i] = zc
        return out

    surf, surf0 = surface(sim.psi), surface(initial_psi(g, p.corner_cells))
    r, meta = res["result"], res["meta"]
    field_um = r["overburden_um"]                       # face thickness at x = R (last column)
    floor_um = None
    for v in reversed(res["series"]["floor_height_um"]):
        if v is not None:
            floor_um = v
            break
    far = meta["faraday"]
    charge_rel = (far["deposited_volume_cm3"] / far["charge_volume_cm3"] - 1.0) if far["charge_volume_cm3"] else None
    return dict(
        code="platesim-levelset",
        field_um=float(field_um), floor_um=float(floor_um) if floor_um is not None else None,
        field_mean_um=float(np.nanmean(surf - g.H) * 1e4),
        mean_thickness_um=float(np.nanmean(surf - surf0) * 1e4),
        ratio=float(floor_um / field_um) if floor_um is not None else None,
        stop_reason=r["stop_reason"], t_end_h=res["meta"]["steps"] and res["series"]["t_s"][-1] / 3600.0,
        steps=meta["steps"], charge_rel=charge_rel,
        max_balance_error=meta["max_balance_error"], cu_balance_rel=r["cu_balance_rel"],
        wall_s=time.time() - t0, warnings=[] if r["stop_reason"] == "recipe_end" else [r["stop_reason"]],
        inputs=dict(d_um=d_um, H_um=H_um, j_mA_cm2=j_mA_cm2, hours=hours, T_C=T_C, f_open=f_open,
                    delta_um=delta_um, dx_um=dx_um, max_overburden_um=max_overburden_um, dt_max_s=dt_max,
                    params=p.to_dict(), grid=dict(nr=g.nr, nz=g.nz, dx_um=g.dx_um, R_um=g.R_um,
                                                  z_lo_um=g.z_lo * 1e4, z_top_um=g.z_top * 1e4)),
        note="field = overburden at x=R, floor = zero-level height at x=0 (same definitions as the FiPy side)")


def compare(f, p, tol=0.02):
    """Side-by-side comparison of one site; pass = |d(floor/field)| <= tol."""
    d_ratio = abs(f["ratio"] - p["ratio"])
    return dict(
        field_um=dict(fipy=f["field_um"], platesim=p["field_um"], diff_um=f["field_um"] - p["field_um"]),
        floor_um=dict(fipy=f["floor_um"], platesim=p["floor_um"], diff_um=f["floor_um"] - p["floor_um"]),
        floor_over_field=dict(fipy=f["ratio"], platesim=p["ratio"], diff=f["ratio"] - p["ratio"],
                              abs_diff=d_ratio),
        mean_thickness_um=dict(fipy=f["mean_thickness_um"], platesim=p["mean_thickness_um"]),
        tol=tol, passed=bool(d_ratio <= tol))


# --------------------------------------------------------------------------------------- CLI
def main(argv=None):
    global OUT_JSON
    ap = argparse.ArgumentParser(description="FiPy cross-check of the platesim level-set trench model")
    ap.add_argument("--sites", default="narrow", choices=["narrow", "wide", "all"])
    ap.add_argument("--j", type=float, default=15.0, help="mA/cm2 (DC)")
    ap.add_argument("--hours", type=float, default=4.5)
    ap.add_argument("--dx", type=float, default=4.0, help="um")
    ap.add_argument("--delta", type=float, default=64.0, help="um")
    ap.add_argument("--max-overburden", type=float, default=140.0, help="um")
    ap.add_argument("--fipy-only", action="store_true", help="skip the platesim reference run")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    names = list(SITES) if a.sites == "all" else [a.sites]
    OUT_JSON = a.out
    t0 = time.time()
    out = dict(task="plan.md Task 4 - independent FiPy cross-check of the additive-free trench model",
               generated=time.strftime("%Y-%m-%dT%H:%M:%S"), fipy_available=_FIPY_OK,
               common=dict(j_mA_cm2=a.j, hours=a.hours, dx_um=a.dx, delta_um=a.delta, efficiency=1.0,
                           ohmic=False, suppressor=False, k_mix=0.0, f_open=0.2,
                           units="SI inside FiPy (m, s, mol/m3, A/m2); platesim cm-based",
                           I_t="j * R per unit length in y", sites={}, warnings=[]),
               sites={}, summary={})
    for nm in names:
        s = SITES[nm]
        print(f"[{nm}] FiPy ...", flush=True)
        f = run_fipy(s["d_um"], s["H_um"], a.j, a.hours, 22.0, 0.2, a.delta, a.dx, a.max_overburden,
                     verbose=a.verbose)
        print(f"[{nm}] FiPy done: field={f['field_um']:.2f} floor={f['floor_um']:.2f} "
              f"ratio={f['ratio']:.4f} ({f['wall_s']:.0f} s, {f['steps']} steps)", flush=True)
        entry = dict(fipy=f)
        out["common"]["sites"][nm] = s
        out["common"]["warnings"] += [f"{nm}: {w}" for w in f["warnings"]]
        if not a.fipy_only:
            print(f"[{nm}] platesim ...", flush=True)
            pr = run_platesim(s["d_um"], s["H_um"], a.j, a.hours, 22.0, 0.2, a.delta, a.dx,
                              a.max_overburden, verbose=a.verbose)
            print(f"[{nm}] platesim done: field={pr['field_um']:.2f} floor={pr['floor_um']:.2f} "
                  f"ratio={pr['ratio']:.4f} ({pr['wall_s']:.0f} s)", flush=True)
            entry["platesim"] = pr
            entry["compare"] = compare(f, pr)
            out["common"]["warnings"] += [f"{nm} (platesim): {w}" for w in pr["warnings"]]
        out["sites"][nm] = entry
    comps = [e["compare"] for e in out["sites"].values() if "compare" in e]
    out["summary"] = dict(
        n_compared=len(comps),
        max_abs_d_ratio=max((c["floor_over_field"]["abs_diff"] for c in comps), default=None),
        passed=all(c["passed"] for c in comps) if comps else None,
        tol=0.02, runtime_s=round(time.time() - t0, 1),
        passed_by_site={k: e["compare"]["passed"] for k, e in out["sites"].items() if "compare" in e},
        fipy_wall_s={k: round(e["fipy"]["wall_s"], 1) for k, e in out["sites"].items()},
        platesim_wall_s={k: round(e["platesim"]["wall_s"], 1) for k, e in out["sites"].items()
                         if "platesim" in e},
        bc_choice="option (b), a genuinely moving top boundary: cells with z >= z_face+delta are pinned "
                  "to c_bulk every step (row elimination on the FiPy matrix).  The quasi-steady flux "
                  "matches platesim's delta-plane Dirichlet, but NOT exactly: the pin sits on the first "
                  "cell row at or above the plane, so the effective plane is 0-1 dx above platesim's, "
                  "i.e. 0-6.25 % (mean 0.5 dx = 2 um, 3.1 %) of delta = 64 um at dx 4.  Because the flux "
                  "scales as 1/delta, that perturbs the surface concentration by ~(j/j_lim)*(0-6.25 %) "
                  "<~0.7 % of c_bulk (mean ~0.4 %), and floor/field by a few 1e-3.  It is the leading "
                  "known boundary-condition discretisation term of this comparison, not negligible by "
                  "construction",
        fipy_version=__import__("fipy").__version__,
        notes=["FiPy 4.0.3 does not apply CellVariable cell constraints to the linear system "
               "(fipy/variables/variable.py only masks .value), so the moving Dirichlet is imposed by "
               "row elimination instead of var.constrain(...)",
               "D_face uses platesim's smeared Hs = 0.5(1+tanh(phi/1.2dx)) (fields.py StepGeom.Hs); the "
               "upstream sharp _LevelSetDiffusionVariable isolates solid-side interface cells and the "
               "implicit sink then drains them to c=0"])
    out = _clean(out)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(f"wrote {OUT_JSON}")
    for nm, e in out["sites"].items():
        c = e.get("compare")
        if c:
            print(f"{nm:7s} field {c['field_um']['fipy']:7.2f}/{c['field_um']['platesim']:7.2f}  "
                  f"floor {c['floor_um']['fipy']:7.2f}/{c['floor_um']['platesim']:7.2f}  "
                  f"ratio {c['floor_over_field']['fipy']:.4f}/{c['floor_over_field']['platesim']:.4f}  "
                  f"|d|={c['floor_over_field']['abs_diff']:.4f}  {'PASS' if c['passed'] else 'FAIL'}")
    return out


if __name__ == "__main__":
    main()

