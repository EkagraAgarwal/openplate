"""D3a: time-resolved 1-D pulse-reverse (PR) cycle model, and what duty averaging costs.

The level-set feature engine treats a PR waveform by DUTY AVERAGING: `engine/recipes.pr_forward` returns
the instantaneous forward current j_f and the duties, and `engine/fields.StepSolver.solve` weights the
Cu2+ interface sink of the forward pulse by duty_f and the anodic source of the reverse pulse by duty_r
(one quasi-steady field solve per step; the double layer does not exist in that model).  That picture
cannot see (a) the pulsating diffusion layer of a 9 ms pulse, thickness sqrt(D t_f) ~ 1.5 um, or (b) the
charge the double layer takes at every switch.  This module resolves ONE cycle in 1-D and measures the
error of the duty-averaged answer.

Model (x normal to an electrode patch, 0 <= x <= delta, all engine units cm/s/mol/A/V)
---------------------------------------------------------------------------------------
    dc/dt = D d2c/dx2                                    0 < x < delta
    c(delta, t) = c_far = c_far_frac * c_bulk
    D dc/dx|0 = +(j_c - j_a)/(2F)                        Cu2+ consumed when the net current is cathodic
    j_c = i0eff (c_s/c_bulk) exp(-alpha_u f eta)         i0eff = (1-theta) i0_u + theta i0_s
    j_a = i0_a exp(alpha_a f eta)                        (same kinetics as engine/fields.Kinetics)
    j(t) = j_c - j_a - C_dl deta/dt                      applied (cathodic positive), piecewise constant
    v = Mfac * <j_c - j_a>                               net deposition velocity (um/h in the output)

    f = F/(RT), i0_u/i0_a/alpha_* from Params (Arrhenius at T_C), D = derive(Params(), T_C).D_c (the
    migration-enhanced value, i.e. the engine's `enhanced` transport path), Mfac = M_CU/(2 F RHO_CU).

Discretisation: finite volumes on a graded mesh refined at x = 0 (first cell 0.02 um [J], geometric growth
1.07 [J], 60-90 cells for delta 20-128 um), implicit backward Euler in time with a graded step schedule
(geometric growth from the switch [J]), Newton on the coupled (c_s, eta) boundary at every step (bordered
tridiagonal, one Schur complement for eta, two banded solves per iteration).

Periodic steady state: marched cycle by cycle from the steady DC profile at the net current.  The
march is accelerated [J] because the layer's slowest diffusion mode decays only by
exp(-D pi^2 P/(4 delta^2)) per cycle (~3000 cycles at delta 64 um): every cycle the measured cycle-mean
error of the profile is removed, and the eta drift gets one scalar Aitken step.  This is legitimate
because of an exact identity that the model then confirms to 1e-9: the period-average of the linear
diffusion equation with the period-average faradaic flux (= j_net exactly, charge conservation) gives
d2<c>/dx2 = 0, <D dc/dx|0> = j_net/(2F), <c>(delta) = c_far, so the cycle-MEAN profile of the periodic
solution is EXACTLY the linear steady profile at the net current - the duty-averaged one.  Only the
fluctuating part is left, and it lives in sqrt(D P) ~ 2 um.  Consequence for D3: duty averaging cannot
be wrong about the net rate or the mean profile; it is a statement about instantaneous values only.

CLI:  python -m platesim.feature.pulse_cycle [--c-dl 3e-5] [--delta 20,64,128] [--cfar 1.0,0.7,0.4]
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from ..core.constants import F
from .engine.params import Params, derive
from .engine.recipes import pr_forward

# --------------------------------------------------------------------------- [J] model constants
TEAM_WAVEFORM = dict(j_net_mA_cm2=15.0, t_f_ms=9.0, t_r_ms=0.5, t_off_ms=0.5, rr=2.0)
C_DL_DEFAULT = 30e-6        # F/cm2, mid of the 20-40 uF/cm2 range quoted for copper plating [J]
DX0_UM = 0.02               # first cell width, << sqrt(D t_r) = 0.45 um [J]
GROWTH = 1.07               # geometric mesh growth, 60-90 cells over 20-128 um [J]
DT0_S = 2e-6                # first step after a switch (resolves the C_dl/(dj/deta) ~ 0.1 ms transient) [J]
DT_PER_STEP = 2e-5          # mid-phase step ceiling -> ~450 steps in the 9 ms forward pulse [J]
N_MIN_PHASE = 70            # minimum steps per phase [J]
MAX_CYCLES = 40


# --------------------------------------------------------------------------- mesh / time schedule
def _mesh(delta_cm, dx0_cm=DX0_UM * 1e-4, growth=GROWTH, n_max=800):
    """Graded finite-volume mesh: cell edges from 0 to delta, first cell dx0_cm, ratio `growth`."""
    edges, x, dx = [0.0], 0.0, dx0_cm
    while x < delta_cm * (1.0 - 1e-12) and len(edges) < n_max:
        rem = delta_cm - x
        x = delta_cm if rem <= 1.05 * dx else x + dx        # last cell is a full cell, never a sliver
        edges.append(x)
        dx *= growth
    edges = np.asarray(edges)
    xc = 0.5 * (edges[:-1] + edges[1:])
    vol = edges[1:] - edges[:-1]
    return xc, vol


def _phase_dts(T, dt_per_step=DT_PER_STEP, n_min=N_MIN_PHASE, growth=1.25, dt0=DT0_S):
    """Step sizes for one phase: dt0 at the switch, geometric growth, then the phase ceiling."""
    n_tgt = int(min(max(round(T / dt_per_step), n_min), 4000))
    dcap = T / n_tgt
    out, t, dt = [], 0.0, min(dt0, T / 40.0, dcap)
    while t < T * (1.0 - 1e-12):
        d = min(dt, dcap, T - t)
        if T - t - d < 0.3 * d:
            d = T - t                                       # keep the switch exactly on a step edge
        out.append(d)
        t += d
        dt *= growth
    return out


def _waveform_phases(waveform):
    """(j_f, duty_f, duty_r) from the engine's own pr_forward, then [(duration_s, j_A_cm2)] phases."""
    wf = dict(TEAM_WAVEFORM)
    if waveform:
        wf.update(waveform)
    pr = {'t_f_ms': float(wf['t_f_ms']), 't_r_ms': float(wf['t_r_ms']),
          't_off_ms': float(wf.get('t_off_ms', 0.0)), 'rr': float(wf['rr'])}
    j_net = float(wf['j_net_mA_cm2'])
    j_f, duty_f, duty_r = pr_forward(j_net, pr)
    phases = []
    if pr['t_f_ms'] > 0:
        phases.append((pr['t_f_ms'] * 1e-3, j_f * 1e-3))
    if pr['t_r_ms'] > 0:
        phases.append((pr['t_r_ms'] * 1e-3, -pr['rr'] * j_f * 1e-3))
    if pr['t_off_ms'] > 0:
        phases.append((pr['t_off_ms'] * 1e-3, 0.0))
    return wf, pr, j_net, j_f, duty_f, duty_r, phases


# --------------------------------------------------------------------------- one implicit step
def _step(u_old, eta_old, dt, j_app, g, tol=1e-11, maxit=40):
    """One backward-Euler step of the coupled (c, eta) problem.  u is c/c_bulk on the mesh, x in units of
    delta (X = x/delta, tau = D t/delta^2), so the diffusion operator is the unit-coefficient Laplacian.

    Unknowns: (u_0 .. u_{n-1}, eta).  Rows 0..n-1 are the finite-volume balances (row 0 carries the
    kinetics Robin flux), row n is the applied-current balance j(t) = j_c - j_a - C_dl deta/dt (eta is
    negative for a cathodic electrode, as in engine/fields.Kinetics, so a cathodic step lowers eta).  The
    Jacobian is tridiagonal plus a rank-1 border, solved with a Schur complement for eta (two banded
    solves, no iteration on the border).  Returns (u, eta, j_faradaic, n_newton, u_min)."""
    n, vol, xc, delta = g['n'], g['vol'], g['xc'], g['delta']
    i0e, au, aa, i0a, f, q, Cdl, ufar = g['i0e'], g['au'], g['aa'], g['i0a'], g['f_RT'], g['q'], g['C_dl'], g['u_far']
    dtau = g['Dod'] * dt                                    # scaled time step, tau = D t/delta^2
    h01 = xc[1] - xc[0]
    hl_in = xc[1:-1] - xc[:-2]
    hr_in = xc[2:] - xc[1:-1]
    hl_last = xc[-1] - xc[-2]
    hE = delta - xc[-1]
    ab = np.zeros((3, n))
    e0 = np.zeros(n)
    e0[0] = 1.0
    u = u_old.copy()
    eta = float(eta_old)
    it = 0
    for it in range(1, maxit + 1):
        e1 = math.exp(min(max(-au * f * eta, -60.0), 60.0))
        e2 = math.exp(min(max(aa * f * eta, -60.0), 60.0))
        A, B = i0e * e1, i0a * e2
        jf = A * u[0] - B                                  # faradaic current density, A/cm2, cathodic +
        G = q * jf                                         # scaled surface flux  du/dX|0 = +G
        dG_du0 = q * A
        dG_deta = q * (-au * f * A * u[0] - aa * f * B)
        R = np.empty(n)
        R[0] = vol[0] / dtau * (u[0] - u_old[0]) - (u[1] - u[0]) / h01 + G
        ab[1, 0] = vol[0] / dtau + 1.0 / h01 + dG_du0
        ab[0, 0] = 0.0
        ab[0, 1] = -1.0 / h01                              # solve_banded: ab[0, j] = A[j-1, j]
        R[1:-1] = (vol[1:-1] / dtau * (u[1:-1] - u_old[1:-1])
                   - (u[2:] - u[1:-1]) / hr_in + (u[1:-1] - u[:-2]) / hl_in)
        ab[1, 1:-1] = vol[1:-1] / dtau + 1.0 / hr_in + 1.0 / hl_in
        ab[0, 2:] = -1.0 / hr_in
        ab[2, :-2] = -1.0 / hl_in                          # ab[2, j] = A[j+1, j]
        R[-1] = (vol[-1] / dtau * (u[-1] - u_old[-1])
                 - (ufar - u[-1]) / hE + (u[-1] - u[-2]) / hl_last)
        ab[1, -1] = vol[-1] / dtau + 1.0 / hE + 1.0 / hl_last
        ab[2, -2] = -1.0 / hl_last
        R_eta = j_app - jf + Cdl * (eta - eta_old) / dt    # C_dl deta/dt = j_far - j_app (eta negative = cathodic)
        v0 = -A                                            # dR_eta/du_0
        v_eta = au * f * A * u[0] + aa * f * B + Cdl / dt  # dR_eta/deta
        u0vec = dG_deta                                    # dR_0/deta
        a = solve_banded((1, 1), ab, -R)
        bb = solve_banded((1, 1), ab, e0)
        den = v_eta - v0 * u0vec * bb[0]
        d_eta = (-R_eta - v0 * a[0]) / den if den != 0.0 else 0.0
        du = a - d_eta * u0vec * bb
        s = 1.0
        if abs(d_eta) > 0.25:
            s = min(s, 0.25 / abs(d_eta))
        dm = float(np.max(np.abs(du)))
        if dm > 0.25:
            s = min(s, 0.25 / dm)
        u += s * du
        eta += s * d_eta
        if s * dm < tol and abs(s * d_eta) < tol:
            break
    e1 = math.exp(min(max(-au * f * eta, -60.0), 60.0))
    e2 = math.exp(min(max(aa * f * eta, -60.0), 60.0))
    jf = i0e * e1 * u[0] - i0a * e2
    return u, eta, jf, it, float(u.min())


# --------------------------------------------------------------------------- steady helpers
def _geometry(p, ph, delta_cm, c_far_frac, C_dl, theta, xc, vol):
    """Everything the stepper needs.  The mesh is rescaled to X = x/delta (cell widths ~1e-4..0.02) so the
    diffusion operator is the unit Laplacian and the matrix stays well conditioned; time enters as
    tau = D t/delta^2 (`Dod` converts), while the double-layer term keeps the physical dt."""
    return dict(n=len(xc), xc=xc / delta_cm, vol=vol / delta_cm, delta=1.0, u_far=float(c_far_frac),
                i0e=(1.0 - theta) * ph.i0_u + theta * ph.i0_s,
                au=p.alpha_u, aa=p.alpha_a, i0a=ph.i0_a, f_RT=ph.f_RT,
                q=delta_cm / (2.0 * F * ph.D_c * p.c_bulk), C_dl=C_dl, Mfac=ph.Mfac,
                Dod=ph.D_c / delta_cm ** 2)


def _steady_u_s(g, j_cath):
    """Surface concentration c_s/c_bulk of the steady linear 1-D profile that carries j_cath (A/cm2).

    Steady state of D d2c/dx2 = 0 with c(delta) = c_far and D dc/dx|0 = +j/(2F) (the -x flux into the
    electrode is j/(2F), cathodic positive) gives c_s = c_far - G, G = delta j/(2F D c_bulk) = q j --
    i.e. exactly the duty-averaged profile of the engine."""
    return g['u_far'] - g['q'] * j_cath


def _eta_for(g, j_target, u_s):
    """eta (V) such that i0e u_s exp(-au f eta) - i0a exp(aa f eta) = j_target (cathodic positive).

    Residual is strictly decreasing in eta, so a bracketed solve is safe (same root as the engine's
    Newton on the forward / reverse potential, without the ohmic drop)."""
    au, aa, f, i0e, i0a = g['au'], g['aa'], g['f_RT'], g['i0e'], g['i0a']

    def res(eta):
        e1 = math.exp(min(max(-au * f * eta, -60.0), 60.0))
        e2 = math.exp(min(max(aa * f * eta, -60.0), 60.0))
        return i0e * u_s * e1 - i0a * e2 - j_target

    return brentq(res, -2.0, 2.0, xtol=1e-15, rtol=8.9e-16, maxiter=300)


# --------------------------------------------------------------------------- periodic march
def _march(g, phases, dts, u0, eta0, u_mean, max_cycles=MAX_CYCLES, tol=1e-8, verbose=False):
    """March whole cycles from (u0, eta0) to the periodic state; remove the cycle-mean error each cycle
    (see the comment below) [J].  Returns the diagnostics of the last cycle, the profile and eta."""
    u, eta = u0.copy(), float(eta0)
    out: dict = {}
    n_cycles, converged, change = 0, False, float('inf')
    de_prev = None
    # [J] acceleration: the cycle-MEAN faradaic flux is j_net exactly (charge conservation: the applied
    # charge per cycle is j_net P and the double layer returns what it takes), so the cycle-mean profile of
    # the periodic state is the linear steady profile u_mean to machine precision.  While marching, any
    # deviation of the marched cycle mean from u_mean is the slow (DC) diffusion mode of the transient,
    # which decays only by exp(-D pi^2 P/(4 delta^2)) per cycle (0.25 % at delta 64 um, P 10 ms: ~3000
    # cycles).  Each cycle we move the start profile by the measured cycle-mean error, which removes that
    # mode exactly; the residual (pulsating) modes decay by >= 9x per cycle, so a handful suffice.
    P = sum(T for (T, _j) in phases)
    xc = g['xc']
    w0 = xc[0] / (xc[1] - xc[0])                            # linear extrapolation of c_s back to x = 0
    for k in range(1, max_cycles + 1):
        us, es = u.copy(), eta
        rec = dict(t=[], cs=[], eta=[], jf=[], dt=[], japp=[])
        usum = np.zeros_like(u)
        t = 0.0
        for iph, (T, japp) in enumerate(phases):
            for dt in dts[iph]:
                u, eta, jf, nit, umin = _step(u, eta, dt, japp, g)
                t += dt
                usum += u * dt
                rec['t'].append(t)
                rec['cs'].append(u[0] - (u[1] - u[0]) * w0)
                rec['eta'].append(eta)
                rec['jf'].append(jf)
                rec['dt'].append(dt)
                rec['japp'].append(japp)
        out = _cycle_stats(g, rec)
        out['u'], out['eta_end'] = u, eta
        d = u - us
        de = eta - es
        change = float(np.max(np.abs(d)))
        out['cycle_change'] = change
        out['mean_error'] = float(np.max(np.abs(usum / t - u_mean)))
        n_cycles = k
        if verbose:
            print(f'  cycle {k:2d}: change {change:.3e}  v {out["v_um_h"]:.4f} um/h  '
                  f'cs {out["c_s_mean"]:.6f}  eta {eta:+.4f} V')
        if change < tol:
            converged = True
            break
        u = u + np.clip(u_mean - usum / t, -0.5, 0.5)      # remove the (slow) cycle-mean error
        np.clip(u, 0.0, 3.0 * g['u_far'], out=u)
        # eta: one scalar geometric (Aitken) step.  Its cycle-to-cycle drift relaxes as
        # 1 - (dj/deta) P / C_dl per cycle, i.e. instantly for the ms waveforms but only ~0.9 per cycle
        # when the period is short compared with C_dl/(dj/deta) [J].
        if de_prev not in (None, 0.0):
            r = de / de_prev
            if 0.0 < r < 0.99:
                eta += float(np.clip(de * r / (1.0 - r), -0.05, 0.05))
        de_prev = de
    out['n_cycles'] = n_cycles
    out['converged'] = converged
    out['u_final'] = u
    out['eta_final'] = eta
    return out


def _cycle_stats(g, rec):
    """Cycle averages and extremes from the recorded step ends (right-endpoint weights = the BE scheme)."""
    dt = np.asarray(rec['dt'])
    cs = np.asarray(rec['cs'])
    eta = np.asarray(rec['eta'])
    jf = np.asarray(rec['jf'])
    japp = np.asarray(rec['japp'])
    P = float(dt.sum())
    q_far = float(np.sum(jf * dt))                          # C/cm2, cathodic positive
    q_abs = float(np.sum(np.abs(japp) * dt))
    d_eta = np.abs(np.diff(np.concatenate(([eta[0]], eta))))
    q_dl = float(g['C_dl'] * np.sum(d_eta))                 # gross double-layer charge per cycle
    rev = japp < 0.0                                        # the reverse (anodic) phase, if any
    q_rev_app = float(np.sum(japp[rev] * dt[rev]))          # negative (anodic)
    q_rev_far = float(np.sum(jf[rev] * dt[rev]))
    return dict(
        period_s=P, q_faradaic_c_cm2=q_far, q_applied_abs_c_cm2=q_abs,
        j_far_avg_a_cm2=q_far / P, v_um_h=q_far / P * g['Mfac'] * 1e4 * 3600.0,
        dl_charge_frac=q_dl / q_abs if q_abs > 0 else 0.0,
        rev_charge_c_cm2=-q_rev_app, rev_faradaic_c_cm2=-q_rev_far,
        rev_dl_frac=(1.0 - (-q_rev_far) / (-q_rev_app)) if q_rev_app < 0 else 0.0,
        c_s_mean=float(np.sum(cs * dt) / P), c_s_min=float(cs.min()), c_s_max=float(cs.max()),
        eta_mean_V=float(np.sum(eta * dt) / P), eta_min_V=float(eta.min()), eta_max_V=float(eta.max()))


# --------------------------------------------------------------------------- the duty-averaged answer
def _duty_average(g, j_net, j_f, rr, duty_f, duty_r):
    """Replicate the engine's duty-averaged answer in 1-D.

    The engine does not average two steady profiles: it solves ONE field with the forward sink weighted
    by duty_f and the reverse source by duty_r (`fields.StepSolver.solve`, `extra = duty_f w ci /(2F c_b)`
    and `rev_src = duty_r w Ia /(2F c_b)`), and its galvanostat fixes the forward current j_f while the
    reverse current is Ia = rr j_f.  The two weighted interface terms therefore add up to
    (duty_f j_f - duty_r rr j_f)/(2F) = j_net/(2F) (that identity IS `recipes.pr_forward`), so the
    duty-averaged concentration field is the single steady profile carrying the NET current, and the
    surface concentration is ONE number, c_s = c_far - delta j_net/(2F D).  The two phase overpotentials
    are then evaluated with that same c_s (forward at j_f, reverse at -rr j_f).

    `c_s_split` is the naive alternative (duty-weighted average of the two phase steady states), kept for
    comparison only - it is NOT what the engine does."""
    c_s = _steady_u_s(g, j_net)
    return dict(c_s=c_s, eta_f=_eta_for(g, j_f, c_s), eta_r=_eta_for(g, -rr * j_f, c_s),
                c_s_split=duty_f * _steady_u_s(g, j_f) + duty_r * _steady_u_s(g, -rr * j_f),
                v_um_h=j_net * g['Mfac'] * 1e4 * 3600.0)


def cycle_vs_duty_average(delta_um=64.0, c_far_frac=1.0, C_dl=C_DL_DEFAULT, theta=0.0, T_C=22.0,
                          waveform=None, n_x_factor=1.0, dt_scale=1.0, max_cycles=MAX_CYCLES,
                          tol=1e-8, verbose=False) -> dict:
    """Resolve one PR cycle in 1-D and compare it with the engine's duty-averaged prediction.

    delta_um    Nernst layer thickness (um); c_far_frac = c(delta)/c_bulk (a depleted trench floor < 1);
    C_dl        double-layer capacitance (F/cm2, 0 switches the capacitor off); theta constant suppressor
    coverage blending i0_u and i0_s (no coverage dynamics); waveform dict overriding TEAM_WAVEFORM
    (j_net_mA_cm2, t_f_ms, t_r_ms, t_off_ms, rr); n_x_factor / dt_scale / max_cycles / tol are the
    numerical knobs used by the convergence test.

    Returns a dict: time-resolved cycle averages (`v_um_h`, `c_s_mean`, `c_s_min`, `c_s_max`, `eta_*`,
    `dl_charge_frac`, `rev_dl_frac`, `q_faradaic_c_cm2`, `n_cycles`, `converged`), the duty-averaged
    prediction (`da_*`) and the relative differences (`rel_diff_v`, `rel_diff_c_s_mean`)."""
    p = Params()
    ph = derive(p, T_C)
    wf, pr, j_net, j_f, duty_f, duty_r, phases = _waveform_phases(waveform)
    delta_cm = float(delta_um) * 1e-4
    xc, vol = _mesh(delta_cm, dx0_cm=DX0_UM * 1e-4 / n_x_factor,
                    growth=1.0 + (GROWTH - 1.0) / n_x_factor)
    g = _geometry(p, ph, delta_cm, c_far_frac, C_dl, theta, xc, vol)
    dts = [_phase_dts(T, dt_per_step=DT_PER_STEP * dt_scale, dt0=DT0_S * dt_scale)
           for (T, _j) in phases]
    j_net_A = j_net * 1e-3
    u_s0 = _steady_u_s(g, j_net_A)
    u0 = c_far_frac - g['q'] * j_net_A * (1.0 - xc / delta_cm)     # linear steady profile at j_net
    eta0 = _eta_for(g, j_net_A, u_s0)
    m = _march(g, phases, dts, u0, eta0, u0, max_cycles=max_cycles, tol=tol, verbose=verbose)
    da = _duty_average(g, j_net_A, j_f * 1e-3, pr['rr'], duty_f, duty_r)
    out = dict(
        delta_um=float(delta_um), c_far_frac=float(c_far_frac), C_dl=float(C_dl), theta=float(theta),
        T_C=float(T_C), waveform=wf, j_net_mA_cm2=j_net, j_f_mA_cm2=j_f,
        j_r_mA_cm2=-pr['rr'] * j_f, duty_f=duty_f, duty_r=duty_r, period_s=m['period_s'],
        D_c_cm2_s=ph.D_c, c_bulk_mol_cm3=p.c_bulk, Mfac_cm_s=ph.Mfac, n_cells=len(xc),
        n_steps=sum(len(d) for d in dts), n_cycles=m['n_cycles'], mean_error=m['mean_error'],
        converged=m['converged'], cycle_change=m['cycle_change'],
        v_um_h=m['v_um_h'], c_s_mean=m['c_s_mean'], c_s_min=m['c_s_min'], c_s_max=m['c_s_max'],
        eta_mean_V=m['eta_mean_V'], eta_min_V=m['eta_min_V'], eta_max_V=m['eta_max_V'],
        q_faradaic_c_cm2=m['q_faradaic_c_cm2'], q_net_applied_c_cm2=j_net_A * m['period_s'],
        q_applied_abs_c_cm2=m['q_applied_abs_c_cm2'], dl_charge_frac=m['dl_charge_frac'],
        rev_charge_c_cm2=m['rev_charge_c_cm2'], rev_faradaic_c_cm2=m['rev_faradaic_c_cm2'],
        rev_dl_frac=m['rev_dl_frac'],
        da_v_um_h=da['v_um_h'], da_c_s=da['c_s'], da_eta_f_V=da['eta_f'], da_eta_r_V=da['eta_r'],
        da_c_s_split=da['c_s_split'], da_c_s_min=da['c_s'], da_c_s_max=da['c_s'],
        x_um=xc * float(delta_um), c_profile=m['u_final'], eta_final=m['eta_final'],
    )
    out['rel_diff_v'] = out['v_um_h'] / da['v_um_h'] - 1.0
    out['rel_diff_c_s_mean'] = m['c_s_mean'] / da['c_s'] - 1.0
    out['rel_diff_c_s_split'] = m['c_s_mean'] / da['c_s_split'] - 1.0
    out['c_s_swing'] = m['c_s_max'] - m['c_s_min']
    out['c_s_swing_frac'] = out['c_s_swing'] / m['c_s_mean']
    out['eta_swing_V'] = m['eta_max_V'] - m['eta_min_V']
    return out


# --------------------------------------------------------------------------- CLI
def _fmt_row(r):
    return (f'{r["delta_um"]:5.0f} {r["c_far_frac"]:5.2f} {r["v_um_h"]:8.3f} {r["da_v_um_h"]:8.3f} '
            f'{100 * r["rel_diff_v"]:+6.2f} {r["c_s_mean"]:7.4f} {r["da_c_s"]:7.4f} '
            f'{100 * r["rel_diff_c_s_mean"]:+6.2f} {r["c_s_min"]:7.4f} {r["c_s_max"]:7.4f} '
            f'{100 * r["c_s_swing_frac"]:6.2f} {r["eta_min_V"]:+7.4f} {r["eta_max_V"]:+7.4f} '
            f'{100 * r["dl_charge_frac"]:6.2f} {100 * r["rev_dl_frac"]:6.1f} {r["n_cycles"]:3d}')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Time-resolved 1-D PR cycle vs the duty-averaged engine answer')
    ap.add_argument('--delta', default='20,64,128', help='Nernst layer thicknesses, um (comma list)')
    ap.add_argument('--cfar', default='1.0,0.7,0.4', help='c(delta)/c_bulk, i.e. a depleted trench floor')
    ap.add_argument('--c-dl', type=float, default=C_DL_DEFAULT, help='double-layer capacitance, F/cm2')
    ap.add_argument('--theta', type=float, default=0.0, help='constant suppressor coverage')
    ap.add_argument('--T', type=float, default=22.0, help='temperature, C')
    ap.add_argument('--max-cycles', type=int, default=MAX_CYCLES)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args(argv)
    deltas = [float(x) for x in a.delta.split(',') if x.strip()]
    cfars = [float(x) for x in a.cfar.split(',') if x.strip()]
    print(f'team waveform {TEAM_WAVEFORM["t_f_ms"]:g}/{TEAM_WAVEFORM["t_r_ms"]:g}/'
          f'{TEAM_WAVEFORM["t_off_ms"]:g} ms, rr {TEAM_WAVEFORM["rr"]:g}, '
          f'{TEAM_WAVEFORM["j_net_mA_cm2"]:g} mA/cm2 net, C_dl {a.c_dl * 1e6:g} uF/cm2, theta {a.theta:g}, '
          f'T {a.T:g} C')
    print(' delta cfar    v_res   v_duty  d_v%  cs_mean  cs_duty  d_cs%  cs_min  cs_max swing%   eta_lo'
          '   eta_hi   dl%  revDL% cyc')
    for d in deltas:
        for cf in cfars:
            r = cycle_vs_duty_average(delta_um=d, c_far_frac=cf, C_dl=a.c_dl, theta=a.theta, T_C=a.T,
                                      max_cycles=a.max_cycles, verbose=a.verbose)
            print(_fmt_row(r), '' if r['converged'] else ' (not converged)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
