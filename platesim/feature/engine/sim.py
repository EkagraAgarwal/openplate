"""Time loop of the level-set feature-fill model (via or trench).

One step:  geometry (Hs, active cells, voids)  ->  fixed-point solve of (theta, s, phi/V, c)  ->  interface speed
extended along normals  ->  CFL time step  ->  Godunov advance of psi  ->  theta extension / re-initialisation  ->
bookkeeping (series, frames, voids, stop tests).  All r-weighting enters through Grid (geometry switch)."""
from __future__ import annotations
import math
import time as _time
import numpy as np
import scipy.ndimage as ndi

from .params import F, Params, derive
from .levelset import (Grid, initial_psi, reinit, extend, advect, band_average, electrolyte_fraction, column_crossing,
                       contour_lines, curvature_div_n, label_regions, UM)
from .fields import StepGeom, StepSolver, theta_be
from .recipes import Recipe, pr_forward


def recipe_delta_max(recipe: Recipe, p: Params):
    """Largest boundary-layer thickness used by any step (sets the domain height)."""
    dm = p.delta_um
    for s in recipe.steps:
        if s.stir:
            d = s.stir.get('delta_um', p.delta_um * s.stir.get('delta_factor', 1.0))
            dm = max(dm, d)
    return dm


class ViaSim:
    def __init__(self, d_um, T_C, params: Params, f_open, recipe: Recipe):
        self.p, self.ph = params, derive(params, T_C)
        p = params
        self.d_um, self.T_C, self.f_open = d_um, T_C, f_open
        self.recipe = recipe
        self.g = Grid(d_um, p.H_um, f_open, recipe_delta_max(recipe, p), p.dx_um, p.max_overburden_um,
                      geometry=p.geometry)
        g = self.g
        self.psi = initial_psi(g, p.corner_cells)
        shp = (g.nr, g.nz)
        self.st = dict(theta=np.full(shp, p.theta0 if p.suppressor else 0.0), theta_a=np.zeros(shp),
                       s=np.ones(shp), ctil=np.ones(shp), phi=np.zeros(shp), phi_r=np.zeros(shp),
                       V=0.0, V_r=0.0)
        if p.transport == "tertiary":
            # explicit species state (absolute mol/cm3); Cu2+ stays normalised in st['ctil'] and the Cu
            # snapshot/series keep reading st['ctil'].  StepSolver would create these lazily, so seeding
            # them here makes the initial state explicit and keeps it in the step-backup/reject cycle.
            self.st['c_h'] = np.full(shp, self.ph.c_h_bulk)
            self.st['c_so4'] = np.full(shp, self.ph.c_so4_bulk)
        ef0 = electrolyte_fraction(self.psi, g)
        self.via_mask = g.via_w > 0.5                                           # boolean: used by the sealing test
        self.cav0 = float((g.vol * ef0 * g.via_w).sum())
        self.ef0 = ef0
        self.solid0 = float((g.vol * (1 - ef0)).sum())
        self.q_total = 0.0                       # integrated net charge (C), for the Faraday check
        self.cu_influx = 0.0                     # integrated Cu2+ moles entering through the top Dirichlet plane
        self.cu_adv = 0.0                        # ... plus the net Cu2+ the advection operator adds (trench_full)
        self.cu_balance_rel = None               # (Cu2+ in) / (Cu deposited) - 1, filled by run()
        self.cu_balance_adv_rel = None           # (diffusive influx + net advective source) / deposit - 1
        self.void_records: list[dict] = []

    # ------------------------------------------------------------------ geometry diagnostics
    def face_height(self, psi=None):
        if self.g.f_open >= 0.999:                 # 1-D slab test geometry: flat interface
            if not self.g.periodic_x:
                return self.g.H                    # via / trench slab test: the wall is the top of the domain
            # trench_full flat slab: initial_psi puts the wall at z = 0, so the interface height must be read
            # from psi instead of being pinned at 0.  A pinned delta plane stays behind the rising front, the
            # rows below it lose their electrolyte, and the active set of StepGeom collapses (empty bincounts
            # in K_data) as soon as the front has grown past the plane.
            psi = self.psi if psi is None else psi
            g = self.g
            zs = [z for z in (column_crossing(psi[i], g.zc[0], upper=True)[1] for i in range(g.nr))
                  if z is not None]
            return float(np.mean(zs)) if zs else 0.0
        psi = self.psi if psi is None else psi
        g = self.g
        if g.periodic_x:
            # two field columns (x ~ 0 and x ~ pitch); average the two crossings when both exist
            zs = []
            for i in (0, g.nr - 1):
                _, z = column_crossing(psi[i], g.zc[0], upper=True)
                if z is not None:
                    zs.append(z)
            return float(np.mean(zs)) if zs else g.H
        _, z = column_crossing(psi[-1], self.g.zc[0], upper=True)
        return z if z is not None else self.g.H

    def fill_fraction(self, ef=None):
        ef = electrolyte_fraction(self.psi, self.g) if ef is None else ef
        return float((self.g.vol * (self.ef0 - ef) * self.g.via_w).sum() / self.cav0)

    def eddy_map(self, z_face, k_mix):
        p, ph, g = self.p, self.ph, self.g
        if g.periodic_x:
            return np.zeros((g.nr, g.nz))          # resolved cavity flow replaces the eddy-diffusivity surrogate
        if k_mix <= 0 or p.D_t0 <= ph.D_cu:
            return np.zeros((g.nr, g.nz))
        ell = k_mix * (2 * g.a) / math.log(p.D_t0 / ph.D_cu)
        depth = np.maximum(z_face - g.zc, 0.0)
        return p.D_t0 * np.exp(-depth / ell) * (g.rc < g.a) * (g.zc <= z_face)

    def sample_column(self, i, fields):
        jf, z = column_crossing(self.psi[i], self.g.zc[0], upper=True)
        if jf is None:
            return None, [np.nan] * len(fields)
        j0 = int(math.floor(jf))
        j0 = min(j0, self.g.nz - 2)
        t = jf - j0
        return z, [float(f[i, j0] * (1 - t) + f[i, j0 + 1] * t) for f in fields]

    def field_floor_metrics(self, z_plane):
        """Front statistics of the periodic full-pitch cell (trench_full only), per output step.

        field_*_um : deposit thickness = upper psi = 0 crossing minus g.H, over the columns that are FAR
                     from both walls: |x - x_c| > a + 150 um; when the pitch leaves no such column (small
                     pitch) the outer quarter of each field side is used instead,
                     |x - x_c| > a + 0.25 (pitch/2 - a).  std is the population std (ddof=0).
        floor_*_um : z of the LOWEST psi = 0 crossing (absolute, um) over the central 40 % of the trench
                     width (|x - x_c| < 0.4 a).
        plane_gap_min_um : min over ALL columns of (z_plane - upper crossing), um: the thinnest gap left
                     between the front and the Dirichlet plane of the diffusion boundary layer.
        Keys that have no valid column are NaN."""
        g = self.g
        off = np.abs(g.rc[:, 0] / UM - g.x_c / UM)                  # |x - x_c| in um
        sel = off > g.a_um + 150.0
        if not sel.any():
            sel = off > g.a_um + 0.25 * (0.5 * g.R_um - g.a_um)
        floor_sel = off < 0.4 * g.a_um
        out = dict(field_mean_um=np.nan, field_min_um=np.nan, field_max_um=np.nan, field_std_um=np.nan,
                   floor_mean_um=np.nan, floor_min_um=np.nan, floor_max_um=np.nan, plane_gap_min_um=np.nan)
        zc = g.zc[0]
        t_far, z_up_all, z_floor = [], [], []
        for i in range(g.nr):
            _, z_up = column_crossing(self.psi[i], zc, upper=True)
            if z_up is not None:
                z_up_all.append(z_up)
                if sel[i]:
                    t_far.append(z_up)
            if floor_sel[i]:
                _, z_lo = column_crossing(self.psi[i], zc, upper=False)
                if z_lo is not None:
                    z_floor.append(z_lo)
        if t_far:
            t = (np.asarray(t_far) - g.H) / UM
            out.update(field_mean_um=float(t.mean()), field_min_um=float(t.min()),
                       field_max_um=float(t.max()), field_std_um=float(t.std()))
        if z_floor:
            zf = np.asarray(z_floor) / UM
            out.update(floor_mean_um=float(zf.mean()), floor_min_um=float(zf.min()),
                       floor_max_um=float(zf.max()))
        if z_up_all:
            out['plane_gap_min_um'] = float((z_plane - max(z_up_all)) / UM)
        return out

    def solid_volume(self):
        return float((self.g.vol * (1 - electrolyte_fraction(self.psi, self.g))).sum())

    # ------------------------------------------------------------------ resolved cavity flow (trench_full)
    def cavity_flow(self, shear_rate_s):
        """LBM shear flow over the current level set.  Returns (ux, uz) arrays shaped (nr, nz) in cm/s.

        The trenchflow CavityGrid convention is [iz, ix] with z = 0 at the wafer (field) plane and z < 0
        in the trench, so z_um = (g.zc - g.H)/UM and the field is transposed back to this engine's (nr, nz).
        CavityGrid lengths are in um while the engine works in cm (UM = 1e-4 cm), hence the /UM on x/z/height."""
        from ...trenchflow.geometry import CavityGrid
        from ...trenchflow.cavity_lbm import solve_flow
        g = self.g
        grid = CavityGrid(dx_um=g.dx_um, x_um=g.rc[:, 0] / UM, z_um=(g.zc[0] - g.H) / UM,
                          solid=(self.psi <= 0).T.copy(), width_um=g.d_um, depth_um=g.H_um,
                          pitch_um=g.R_um, height_um=(g.z_top - g.H) / UM, deposit_um=0.0)
        ff = solve_flow(grid, float(shear_rate_s))
        self.flow_info = ff.info
        return ff.ux.T.copy(), ff.uz.T.copy()

    # ------------------------------------------------------------------ voids
    def update_void_records(self, void_m, t):
        """Track trapped electrolyte pockets (components of void_m); keep the largest state of each pocket."""
        g = self.g
        if not void_m.any():
            return
        lab, n = ndi.label(void_m)
        ef = electrolyte_fraction(self.psi, g)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < 4:
                continue
            ii, jj = np.nonzero(m)
            z0, z1 = g.zc[0, jj.min()] - 0.5 * g.dx, g.zc[0, jj.max()] + 0.5 * g.dx
            rmax = float(np.max(np.abs(g.rc[ii, 0] - g.x_c))) if g.periodic_x else (ii.max() + 1) * g.dx
            vol = float((g.vol * ef * m).sum())
            vol_cells = float((g.vol * m).sum())
            zc = float((g.zc[0, jj] * g.vol[ii, 0]).sum() / g.vol[ii, 0].sum())
            for rec in self.void_records:                         # same pocket seen before?
                if z0 <= rec['z1'] and z1 >= rec['z0'] and rmax >= 0.5 * rec['rmax'] and rec['rmax'] >= 0.5 * rmax \
                        and abs(zc - rec['zc']) < 0.5 * (rec['z1'] - rec['z0'] + z1 - z0):
                    if vol > rec['vol']:
                        rec.update(z0=z0, z1=z1, rmax=rmax, vol=vol, zc=zc)
                    break
            else:
                self.void_records.append(dict(t_s=t, z0=z0, z1=z1, rmax=rmax, vol=vol, zc=zc))

    def classify(self, rec):
        g = self.g
        h = rec['z1'] - rec['z0']
        rmax = rec['rmax']
        if rmax <= 0.4 * g.a and h >= 3.0 * max(2 * rmax, g.dx):
            return 'seam'
        if rec['zc'] < g.H / 3.0:
            return 'bottom'
        if rec['z1'] > 0.6 * g.H:
            return 'keyhole'
        return 'void'

    # ------------------------------------------------------------------ frames
    def snapshot(self, t, z_plane):
        g, p = self.g, self.p
        lines = contour_lines(self.psi, g)
        vlines = []
        jt = min(int((z_plane - 0.5 * g.dx - g.z_lo) / g.dx - 0.5), g.nz - 1)
        _, void_m, _ = label_regions(self.psi, g, jt, -0.5 * g.dx)
        if void_m.any():                                   # trapped pockets: contour of psi restricted to them
            m1 = ndi.binary_dilation(void_m, iterations=1)
            vlines = contour_lines(np.where(m1, self.psi, -1.0), g)
        # coarse Cu2+ field: 0 = solid, 1..255 = c/c_bulk in [0, 1]
        nrc, nzc = 48, 64
        z_view = g.H + p.delta_um * UM + min(g.max_overburden_um, 0.25 * g.H_um) * UM
        z_view = min(z_view, g.z_top - 2 * g.dx)
        rs = (np.arange(nrc) + 0.5) * g.R / nrc
        zs = (np.arange(nzc) + 0.5) * z_view / nzc
        ii = np.minimum((rs / g.dx).astype(int), g.nr - 1)
        jj = np.clip(((zs - g.z_lo) / g.dx).astype(int), 0, g.nz - 1)
        ctil = np.clip(self.st['ctil'][np.ix_(ii, jj)], 0, 1)
        psi_s = self.psi[np.ix_(ii, jj)]
        val = (1 + np.round(ctil * 254)).astype(np.uint8)
        val = np.where(zs[None, :] >= z_plane, 255, val)
        val = np.where(psi_s <= 0, 0, val).astype(np.uint8)
        return dict(t_s=float(t), contour=lines, voids=vlines, cfield=val.T.copy(),     # rows = z (bottom first)
                    extent=[0.0, g.R_um, 0.0, float(z_view / UM)])

    # ------------------------------------------------------------------ main loop
    def run(self, label='', progress=None):
        """progress: optional callback progress(frac, msg) with frac = t / recipe duration (0..1)."""
        p, ph, g, st, recipe = self.p, self.ph, self.g, self.st, self.recipe
        t0_wall = _time.time()
        T_total = recipe.duration_s
        if progress is not None:
            progress(0.0, f'{label or "run"}: start')
        bounds = np.concatenate([[0.0], np.cumsum([s.duration_s for s in recipe.steps])])
        t, k = 0.0, 0
        dt_prev, vmax_prev = None, None
        growth = 1.6                        # dt growth factor chosen by the iteration-count controller
        lu_reuse = None                     # LU factors carried from step to step while the active set is unchanged
        flow_cache = None                   # (ux, uz, z_face, G, step) of the last LBM solve (trench_full)
        series = {key: [] for key in ('t_s', 'V', 'eta_floor', 'eta_face', 'I_total', 'fill_fraction',
                                      'face_thickness_um', 'floor_height_um', 'void_volume_frac', 'theta_floor',
                                      'theta_face', 'c_floor', 'bal', 'it', 'dt_s', 'attempts')}
        if g.periodic_x:                       # trench_full only: field/floor statistics of the whole pitch
            series.update({key: [] for key in ('field_mean_um', 'field_min_um', 'field_max_um', 'field_std_um',
                                               'floor_mean_um', 'floor_min_um', 'floor_max_um',
                                               'plane_gap_min_um')})
        frames, next_frame = [], 0.0
        frame_dt = T_total / 240.0
        stop_reason, fill_time = None, None
        max_bal, tot_it = 0.0, 0
        fill_prev, t_prev = 0.0, 0.0
        n_reject = 0
        while t < T_total - 1e-9:
            si = int(np.searchsorted(bounds, t, side='right') - 1)
            si = min(si, len(recipe.steps) - 1)
            step = recipe.steps[si]
            t_end_step = bounds[si + 1]
            stir = step.stir or {}
            delta_um = stir.get('delta_um', p.delta_um * stir.get('delta_factor', 1.0))
            k_mix = stir.get('k_mix', p.k_mix * stir.get('k_mix_factor', 1.0))
            z_face = self.face_height()
            z_plane = z_face + delta_um * UM
            if z_plane > g.z_top - 3 * g.dx:
                stop_reason = 'overburden_exceeded'
                break
            geo = StepGeom(g, self.psi, z_plane, p)
            self.update_void_records(geo.void_m, t)
            # ---- stop test: open path sealed (a pocket exists and no open cavity remains below z = H)
            if self.void_records and not (geo.open_m & self.via_mask).any():
                stop_reason = 'sealed'
                break
            if g.periodic_x:
                from ...trenchflow.model import shear_from_delta
                G_flow = p.flow_shear_s if p.flow_shear_s is not None else shear_from_delta(delta_um, ph.D_c)
                need = (flow_cache is None or k % max(int(p.flow_every), 1) == 0
                        or abs(z_face - flow_cache[2]) > p.flow_move_cells * g.dx
                        or abs(G_flow - flow_cache[3]) > 1e-9 * max(1.0, abs(G_flow)))
                if need:
                    ux_f, uz_f = self.cavity_flow(G_flow)
                    flow_cache = (ux_f, uz_f, z_face, G_flow, k)
                flow = flow_cache[:2]
                Dt = np.zeros((g.nr, g.nz))        # resolved flow replaces the eddy-diffusivity surrogate
            else:
                flow = None
                Dt = self.eddy_map(z_face, k_mix)
            solver = StepSolver(geo, p, ph, Dt, reuse=lu_reuse, flow=flow)
            # ---- control
            if step.j_mA_cm2 is not None:
                pr = step.pr
                if pr:
                    j_f, duty_f, duty_r = pr_forward(step.j_mA_cm2, pr)
                    rr = pr['rr']
                else:
                    rr, j_f, duty_f, duty_r = 0.0, step.j_mA_cm2, 1.0, 0.0
                ctrl = dict(mode='I', I_t=j_f * 1e-3 * g.area_cell, duty_f=duty_f, duty_r=duty_r, rr=rr,
                            pr=bool(pr))
            else:
                ctrl = dict(mode='V', V_set=step.V_mV * 1e-3, I_t=1.0, duty_f=1.0, duty_r=0.0, rr=0.0, pr=False)
            # ---- predicted time step
            dt_pred = min(p.dt_max, t_end_step - t)
            if vmax_prev:
                dt_pred = min(dt_pred, p.cfl * g.dx / vmax_prev)
            dt_pred = min(dt_pred, 5.0 if dt_prev is None else dt_prev * growth)
            dt_pred = max(dt_pred, min(p.dt_min, t_end_step - t))
            backup = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in st.items()}
            attempt = 0
            for attempt in range(6):
                out = solver.solve(st, ctrl, dt_pred)
                info = out['info']
                ok = info['converged'] or (abs(info['bal']) < 2 * p.fix_tol_bal and info['d_th'] < 4 * p.fix_tol_theta
                                           and info['d_c'] < 4 * p.fix_tol_c)
                if ok or dt_pred <= 4 * p.dt_min:
                    break
                n_reject += 1                                  # stiff transition (e.g. suppression breakdown): retry smaller
                for kk, vv in backup.items():
                    st[kk] = vv.copy() if isinstance(vv, np.ndarray) else vv
                dt_pred = max(0.4 * dt_pred, p.dt_min)
            act = geo.act
            if p.transport == "tertiary" and out.get('c_h') is not None:
                # mirror the species state the solver accepted (out carries active cells only; the solver
                # already wrote them to st, this makes the hand-off explicit and keeps it after any retry)
                st['c_h'][act] = out['c_h']
                st['c_so4'][act] = out['c_so4']
            lu_reuse = solver.lu_state()
            # ---- step-size controller (iteration count): slow or rejected solves keep dt small, easy ones let it grow
            nit = info['it']
            growth = 0.8 if attempt > 0 else (1.6 if nit <= 5 else (1.0 if nit <= 9 else 0.7))
            # ---- interface speed (cm/s): time-averaged over a pulse-reverse cycle
            i_net = ctrl['duty_f'] * out['i_f'] - ctrl['duty_r'] * out['i_a']      # time-averaged net current (A/cm2)
            icell = np.zeros((g.nr, g.nz))
            icell[act] = i_net
            i_s = band_average(self.psi, geo.w_full, icell, g)                       # interface value (see levelset)
            vfield = ph.Mfac * i_s
            vfield[np.abs(self.psi) > 3 * g.dx] = 0.0
            _, vext = extend(self.psi, vfield, g.dx)
            vmax = float(np.abs(vext).max())
            vmax_prev = max(vmax, 1e-12)
            dt = min(dt_pred, p.cfl * g.dx / max(vmax, 1e-30), t_end_step - t)
            dt = max(dt, min(p.dt_min, t_end_step - t))
            # ---- Cu2+ mass balance, independent of the level set: moles entering through the top Dirichlet
            # plane of the Cu2+ field of this step.  K_data builds bc for u_bulk = 1, so the top-row Dirichlet
            # coupling is bc_c and the influx per unit driving concentration is bc_c (1 - ct_top).  Computed
            # once per ACCEPTED step, after dt is final (rejected retries are not counted) and committed below
            # together with q_total, so influx and charge cover exactly the same steps (a 'filled' break
            # before the front advance must not leave an uncounted influx behind).
            cu_step = float(np.dot(solver.bc_c[geo.ptop], 1.0 - out['ct'][geo.ptop])) * p.c_bulk * dt
            # Net Cu2+ the advection operator adds to the active unknowns in this step: with an OPEN top the
            # upwind 'gradient' operator is not discretely conservative (its rows sum to zero, so its column
            # sum is the flux imbalance), and the Cu2+ that the flow carries through the Dirichlet plane is
            # NOT part of cu_step (which only counts the diffusive Dirichlet coupling).  Reported as a
            # separate number instead of being forced into cu_balance_rel: -sum_p (A_adv c)_p c_bulk dt.
            cu_adv_step = (-float(np.sum(solver.adv @ out['ct'])) * p.c_bulk * dt
                           if solver.adv is not None else 0.0)
            if p.suppressor and dt < 0.9 * dt_pred:
                # shorter step than predicted: redo the (cell-local) coverage update with the true dt
                acc = (1.0 + p.b_acc * st['theta_a'][act]) if p.use_accel else 1.0
                b_ex = p.k_rd * ctrl['duty_r'] * out['i_a'] / 1e-3 if ctrl['pr'] else 0.0
                th = theta_be(out['th0'], dt, p.k_plus * p.s_bulk * out['s_t'], ph.Mfac * out['ct'] * out['gu'],
                              ph.Mfac * out['ct'] * out['gs'], p.k_minus * ctrl['duty_f'], b_ex)
                st['theta'][act] = th
            # ---- diagnostics before the interface moves
            ef = electrolyte_fraction(self.psi, g)
            fill = self.fill_fraction(ef)
            I_net = float(np.dot(geo.w, ctrl['duty_f'] * out['i_f'] - ctrl['duty_r'] * out['i_a']))
            z_fl, (eta_fl, th_fl, c_fl) = self.sample_column(
                g.nr // 2 if g.periodic_x else 0,
                [st['V'] - st['phi'] if p.ohmic else np.full((g.nr, g.nz), st['V']), st['theta'], st['ctil']])
            z_fc, (eta_fc, th_fc, c_fc) = self.sample_column(
                0 if g.periodic_x else g.nr - 1,
                [st['V'] - st['phi'] if p.ohmic else np.full((g.nr, g.nz), st['V']), st['theta'], st['ctil']])
            void_frac = float((g.vol * ef * geo.void_m).sum() / self.cav0)
            series['t_s'].append(t)
            series['V'].append(st['V'])
            series['eta_floor'].append(eta_fl)
            series['eta_face'].append(eta_fc)
            series['I_total'].append(I_net)
            series['fill_fraction'].append(fill)
            series['face_thickness_um'].append((z_face - g.H) / UM)
            series['floor_height_um'].append(np.nan if z_fl is None else z_fl / UM)
            series['void_volume_frac'].append(void_frac)
            series['theta_floor'].append(th_fl)
            series['theta_face'].append(th_fc)
            series['c_floor'].append(c_fl)
            series['bal'].append(info['bal'])
            series['it'].append(info['it'])
            series['dt_s'].append(dt)
            series['attempts'].append(attempt + 1)
            if g.periodic_x:                   # trench_full: whole-pitch front statistics (see field_floor_metrics)
                for key, val in self.field_floor_metrics(z_plane).items():
                    series[key].append(val)
            max_bal = max(max_bal, abs(info['bal']))
            tot_it += info['it']
            if progress is not None:
                progress(min(t / T_total, 1.0), f'{t / 60:.1f} min, fill {fill:.3f}')
            if t >= next_frame:
                frames.append(self.snapshot(t, z_plane))
                next_frame += frame_dt
            if p.verbose and k % 10 == 0:
                print(f't={t / 60:7.1f} min dt={dt:6.1f} fill={fill:.3f} V={st["V"] * 1e3:7.1f} mV it={info["it"]:2d} '
                      f'bal={info["bal"]:+.1e} nph={info["nph"]} vmax={vmax * 1e4 * 1e3:.2f} nm/s '
                      f'th_fl={th_fl:.2f} th_fc={th_fc:.2f} c_fl={c_fl:.2f}', flush=True)
            # ---- fill-complete stop
            if fill >= p.fill_stop:
                fill_time = t_prev + (p.fill_stop - fill_prev) / max(fill - fill_prev, 1e-12) * (t - t_prev) \
                    if fill_prev < p.fill_stop else t
                stop_reason = 'filled'
                break
            # ---- advance the interface
            psi_new = advect(self.psi, vext, dt, g)
            if p.use_accel:
                kap = np.clip(-curvature_div_n(self.psi, g), -1 / (2 * g.dx), 1 / (2 * g.dx))
                ta = st['theta_a']
                ta = np.clip(ta + dt * (p.k_acc * p.c_acc * (1 - ta) + ta * kap * vext), 0.0, 1.0)
                st['theta_a'] = ta
            d_new, th_ext = (extend(psi_new, st['theta'], g.dx, periodic=g.periodic_x)
                             if p.suppressor else (None, None))
            if p.suppressor:
                st['theta'] = th_ext
            if p.use_accel:
                _, st['theta_a'] = extend(psi_new, st['theta_a'], g.dx, periodic=g.periodic_x)
            if k % p.reinit_every == 0:
                self.psi = d_new if d_new is not None else reinit(psi_new, g.dx, periodic=g.periodic_x)
            else:
                self.psi = psi_new
            self.q_total += I_net * dt                          # charge of the step just taken (front advanced)
            self.cu_influx += cu_step                           # ... and its Cu2+ influx (same accepted steps)
            self.cu_adv += cu_adv_step                          # ... and its net advective Cu2+ source
            fill_prev, t_prev = fill, t
            t += dt
            dt_prev = dt
            k += 1
        # ---- final state
        z_plane = self.face_height() + p.delta_um * UM
        fill = self.fill_fraction()
        if stop_reason is None:
            stop_reason = 'recipe_end'
        if stop_reason in ('recipe_end', 'filled', 'overburden_exceeded'):
            # final void check on the final geometry
            jt = min(int((z_plane - 0.5 * g.dx - g.z_lo) / g.dx - 0.5), g.nz - 1)
            open_m, void_m, _ = label_regions(self.psi, g, jt, -0.5 * g.dx)
            self.update_void_records(void_m, t)
        frames.append(self.snapshot(t, z_plane))
        series['t_s'].append(t)
        series['V'].append(st['V'])
        for key in series:
            if key not in ('t_s', 'V') and len(series[key]) < len(series['t_s']):
                series[key].append(series[key][-1] if series[key] else np.nan)
        cu_dep = self.q_total * p.efficiency / (2 * F)      # moles of Cu in the deposit (net charge x efficiency)
        self.cu_balance_rel = float(self.cu_influx / cu_dep - 1.0) if cu_dep > 0 else None
        # honest version for the OPEN-top trench_full cell: the advective Cu2+ crossing of the Dirichlet plane
        # is not in cu_influx, and the 'gradient' operator is not discretely conservative, so the net advective
        # source of the operator is added on top of the diffusive influx (equal to cu_balance_rel when the
        # flow is off or the fluxes are discretely divergence free, e.g. the legacy 'divergence' form).
        self.cu_balance_adv_rel = (float((self.cu_influx + self.cu_adv) / cu_dep - 1.0)
                                   if cu_dep > 0 else None)
        raw = dict(series=series, frames=frames, t_end=t, stop_reason=stop_reason, fill_time=fill_time,
                    fill_final=fill, steps=k, n_reject=n_reject, max_bal=max_bal, mean_it=tot_it / max(k, 1),
                    wall_s=_time.time() - t0_wall, face_thickness_um=(self.face_height() - g.H) / UM,
                    voids=[dict(rec, cls=self.classify(rec)) for rec in self.void_records],
                    solid_gain=self.solid_volume() - self.solid0, q_total=self.q_total,
                    cu_influx=self.cu_influx, cu_balance_rel=self.cu_balance_rel,
                    cu_adv=self.cu_adv, cu_balance_adv_rel=self.cu_balance_adv_rel)
        if progress is not None:
            progress(1.0, f'{stop_reason} (fill {fill:.3f})')
        return raw
