"""Quasi-steady field solve of one 3-D tile, physics level L1 (no flow, no additives).

Same model as the 2-D engine's additive-free 'enhanced' path (platesim/feature/engine/fields.py,
StepGeom 'diffuse' + newton_potential), written per unit volume on a uniform grid:

  Cu2+ (ct = c / c_bulk):   -div(D_c Hs grad ct) + eff * w * ci(eta) / (2 F c_bulk) * ct = 0
  potential:                -div(kappa Hs grad phi) + w * ct * ci(eta) = 0,   eta = V - phi
  galvanostat:              sum(w * ct * ci) * dx^3 = j_net * Lx * Ly

  Hs = max(0.5 (1 + tanh(psi / eps)), 1e-6),  eps = eps_fac * dx      (smoothed electrolyte indicator)
  w  = |grad Hs|  [1/cm]                                              (interface area per unit volume)
  ci = i0_u exp(-alpha_u f eta)  [A/cm2]                              (cathodic, unsuppressed)

Boundary conditions: ct = 1 and phi = 0 on the bulk plane z = z_plane (at a distance d_top in
[0.5, 1.5) dx above the last solved cell layer, as in 2-D); no flux through the bottom; lateral sides
periodic or no flux.  Units: cm, s, mol, A, V.  Pulse-reverse enters only through the net current.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu

from ..core.constants import F
from ..feature.engine.params import Params, derive
from . import operators
from .backend import ls_numpy

DIRECT_MAX = 40000      # unknowns up to which a sparse LU is used; above it AMG-preconditioned CG


class _Linear:
    """One SPD matrix, several right-hand sides."""

    def __init__(self, A, tol=1e-10):
        self.A, self.tol, self.iterations = A, tol, 0
        if A.shape[0] <= DIRECT_MAX:
            self._lu, self._M = splu(A.tocsc()), None
        else:
            import pyamg
            self._lu = None
            self._M = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=500).aspreconditioner()

    def solve(self, b, x0=None):
        if self._lu is not None:
            return self._lu.solve(b)
        count = [0]
        x, info = cg(self.A, b, x0=x0, rtol=self.tol, atol=0.0, M=self._M, maxiter=400,
                     callback=lambda _: count.__setitem__(0, count[0] + 1))
        self.iterations += count[0]
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return x


def _faces(H, periodic, top_scale=None):
    """Arithmetic face means of the cell field H; outer faces 0 unless periodic.  top_scale (nx, ny)
    replaces the outer z_hi face (used for the Dirichlet plane at distance d_top)."""
    out = []
    for axis in range(3):
        v = np.moveaxis(H, axis, 0)
        f = np.zeros((v.shape[0] + 1, *v.shape[1:]))
        f[1:-1] = 0.5 * (v[:-1] + v[1:])
        if periodic[axis]:
            f[0] = f[-1] = 0.5 * (v[0] + v[-1])
        out.append(np.moveaxis(f, 0, axis).copy())
    if top_scale is not None:
        out[2][:, :, -1] = top_scale
    return out


class FieldSolver:
    """L1 field solve.  Construct once per run, call solve() once per front position."""

    def __init__(self, params: Params | None = None, T_C: float = 22.0):
        self.p = Params(suppressor=False) if params is None else params
        self.ph = derive(self.p, T_C)
        self.T_C = T_C

    def solve(self, psi, dx, z0, z_plane, j_net_A_cm2, periodic=(True, True, False), state=None,
              tol=2e-4, maxit=30):
        """psi: (nx, ny, nz) level set in cm (positive in electrolyte); z0: z of the low domain face;
        z_plane: height of the bulk plane (cm); j_net_A_cm2: net current per projected area.
        state: dict returned by a previous call (warm start) or None.
        Returns dict(ct, phi, i, speed, V, w, active, kt, info) on the full grid; i is the interface
        current density (A/cm2) of each cell, speed = Mfac * i (cm/s)."""
        p, ph = self.p, self.ph
        psi = np.asarray(psi, dtype=np.float64)
        nx, ny, nz = psi.shape
        zc = z0 + (np.arange(nz) + 0.5) * dx
        kt = int(np.floor((z_plane - 0.5 * dx - z0) / dx - 0.5))
        if kt < 2:
            raise ValueError("bulk plane is below the grid")
        if kt > nz - 1:
            raise ValueError("bulk plane is above the grid: enlarge the domain in z "
                             f"(needs z up to {z_plane * 1e4:.1f} um)")
        d_top = z_plane - zc[kt]
        sl = np.s_[:, :, :kt + 1]
        eps = p.eps_fac * dx
        x = np.clip(psi / eps, -30, 30)
        Hs_full = np.maximum(0.5 * (1 + np.tanh(x)), 1e-6)
        gm = np.sqrt(sum(((ls_numpy.shifted(psi, 1, ax, periodic)
                           - ls_numpy.shifted(psi, -1, ax, periodic)) / (2 * dx)) ** 2 for ax in range(3)))
        gHs = 0.5 / eps / np.cosh(x) ** 2 * gm
        Hs = Hs_full[sl]
        active = psi[sl] > -p.n_active_eps * eps
        w = np.where(active, gHs[sl], 0.0)
        if not w.any():
            raise ValueError("no plating front below the bulk plane")
        shape = Hs.shape
        top = Hs[:, :, -1] * dx / (2.0 * d_top)          # so that 2 k / dx^2 = k Hs / (dx d_top)
        hx, hy, hz = _faces(Hs, periodic, top_scale=top)
        per = tuple(bool(v) for v in periodic)

        def system(coef, value):
            return operators.assemble(coef * hx, coef * hy, coef * hz, dx, active=active,
                                      dirichlet={"z_hi": value}, periodic=per)

        A_c, b_c = system(ph.D_c, 1.0)
        A_p, _ = system(ph.kappa, 0.0)
        wf = w.ravel()
        act = active.ravel()
        It = j_net_A_cm2 * nx * ny / dx                 # target of sum(w * i) (volume-normalised)
        af = p.alpha_u * ph.f_RT
        sink = p.efficiency / (2.0 * F * p.c_bulk)

        def ci_of(eta):
            return ph.i0_u * np.exp(np.clip(-af * eta, -60, 60))

        def V_fixed_phi(ct, phi):
            s = np.dot(wf * ct, ph.i0_u * np.exp(np.clip(af * phi, -60, 60)))
            return -np.log(It / (s + 1e-300)) / af

        n = wf.size
        if state is not None and state.get("shape") == shape:
            ct, phi = state["ct"].copy(), state["phi"].copy()
        else:
            ct, phi = np.where(act, 1.0, 0.0), np.zeros(n)
        ct = np.where(act, np.clip(ct, 1e-6, None), 0.0)
        V = V_fixed_phi(ct, phi)
        info = dict(outer=0, newton=0, cg=0)
        relax = 1.0
        last_change = np.inf
        for outer in range(maxit):
            # ---- potential and electrode potential V for fixed ct (bordered Newton, as in 2-D)
            for _ in range(14):
                eta = V - phi
                ci = ci_of(eta)
                I = ct * ci
                q = wf * af * I                          # d(w I)/d(phi) > 0
                R = A_p @ phi + wf * I
                g = np.dot(wf, I) - It
                lin = _Linear(A_p + sparse.diags(q))
                x1, x2 = lin.solve(-R), lin.solve(q)
                info["cg"] += lin.iterations
                dV = (g + np.dot(q, x1)) / (q.sum() - np.dot(q, x2) + 1e-300)
                dphi = x1 + x2 * dV
                m = float(np.max(np.abs(dV - dphi)))
                lam = min(1.0, 0.15 / (m + 1e-300))
                phi, V = phi + lam * dphi, V + lam * dV
                info["newton"] += 1
                if lam == 1.0 and m < 2e-7:
                    break
            # ---- Cu2+ for fixed phi.  For a given electrode potential the problem is linear in ct (implicit
            # sink), and the total current is a monotone function of u = -af V, so V is found by a scalar
            # root-find with one linear solve per evaluation.  This stays stable up to the limiting current,
            # where alternating "ct at fixed eta" / "V at fixed ct" oscillates.
            shape_k = wf * ph.i0_u * np.exp(np.clip(af * phi, -60, 60))     # w * ci / exp(u)
            guess = [ct]

            def total(u):
                lin_c = _Linear(A_c + sparse.diags(sink * np.exp(u) * shape_k))
                c = np.where(act, np.clip(lin_c.solve(b_c, x0=guess[0]), 1e-12, None), 0.0)
                info["cg"] += lin_c.iterations
                guess[0] = c
                return np.log(np.dot(shape_k, c) + 1e-300) + u - np.log(It), c

            u1 = -af * V
            f1, new = total(u1)
            u0, f0 = u1 - 0.5 * np.sign(f1 if f1 != 0 else 1.0), None
            for _ in range(40):
                if abs(f1) < 1e-10:
                    break
                if f0 is None:
                    f0, _ = total(u0)
                slope = (f1 - f0) / (u1 - u0) if u1 != u0 else 1.0
                slope = slope if slope > 1e-6 else 1e-6        # f is increasing in u; flat near the limit
                step = float(np.clip(-f1 / slope, -4.0, 4.0))
                u0, f0 = u1, f1
                u1 = u1 + step
                f1, new = total(u1)
            V = -u1 / af
            info["limited"] = bool(abs(f1) > 1e-6)       # set current above the transport limit
            change = float(np.max(np.abs(new - ct)))
            last_change = change
            ct = new
            info["outer"] = outer + 1
            if change < tol:
                break
        info["converged"] = bool(change < tol)
        info["ct_change"] = change
        I = ct * ci_of(V - phi)
        info["current_error"] = float(np.dot(wf, I) / It - 1.0)
        # Cu2+ entering through the bulk plane vs the deposition sink (both mol/s per tile)
        ct3 = ct.reshape(shape)
        influx = float(np.sum(ph.D_c * Hs[:, :, -1] / (dx * d_top) * (1.0 - ct3[:, :, -1])
                              * active[:, :, -1]) * p.c_bulk * dx ** 3)
        need = j_net_A_cm2 * nx * ny * dx * dx * p.efficiency / (2.0 * F)
        info["cu_balance_rel"] = influx / need - 1.0
        info["ct_min_front"] = float(np.min(ct[wf > 0.25 * wf.max()]))

        def full(a, fill):
            out = np.full(psi.shape, fill, dtype=np.float64)
            out[sl] = a.reshape(shape)
            return out

        i_full = full(np.where(act, I, 0.0), 0.0)
        return dict(ct=full(np.where(act, ct, 0.0), 1.0), phi=full(phi, 0.0), i=i_full,
                    speed=ph.Mfac * i_full, V=float(V), w=full(wf, 0.0), active=full(act, False).astype(bool),
                    kt=kt, info=info, state=dict(shape=shape, ct=ct, phi=phi))
