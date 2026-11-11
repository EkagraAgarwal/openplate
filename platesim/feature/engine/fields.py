"""Smoothed-boundary finite-volume field solvers for one time step (frozen geometry).

Smoothed-boundary form (Hs = 0.5(1+tanh(psi/eps)), psi > 0 electrolyte):
    div(Hs K grad u) = |grad Hs| * (K du/dn at the interface)
 so with a consumption flux N at the interface   div(Hs D grad c) = |grad Hs| N.
Discretisation: cell-centred finite volume (axisymmetric via: face area ~ r; planar trench per unit length:
weight 1 - both cases enter only through Grid.vol / Grid.AE / Grid.AZ), symmetric M-matrix K with
  K u = -(sink), solved on cells with psi > -n_active_eps*eps only, Dirichlet (bulk) value on the delta plane.
Interface weights  w_p = V_p |grad Hs|_p  [cm2]  carry all interface fluxes:  total interface current = sum w_p i_p.

Fields per step (fixed-point loop): theta (implicit backward Euler per cell), suppressor s, potential phi with
the metal potential V (Newton, galvanostatic constraint bordered), Cu2+ concentration c (implicit sink)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.ndimage as ndi
from scipy.optimize import brentq

from .levelset import Grid, grad_mag, label_regions
from .params import F


# ======================================================================================= Scharfetter-Gummel
def bern(x):
    """Bernoulli function B(x) = x / (exp(x) - 1), vectorised and overflow-safe.

    Branch strategy (research/tertiary_transport_notes.md sect. 4): Taylor series for |x| < 1e-3,
    x/expm1(x) in the middle, asymptotic x exp(-x) for x > 40; the argument is clipped to +-50, where
    the middle branch is still exact to 1e-15 (expm1(-50) = -1).  B(-x) = B(x) + x is applied by the
    caller, so only one exponential per face is needed."""
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, -50.0, 50.0)
    out = np.empty_like(xc)
    small = np.abs(xc) < 1e-3
    big = xc > 40.0
    mid = ~(small | big)
    xs = xc[small]
    out[small] = 1.0 - 0.5 * xs + xs * xs / 12.0
    xb = xc[big]
    out[big] = xb * np.exp(-xb)
    xm = xc[mid]
    out[mid] = xm / np.expm1(xm)
    return out


# ======================================================================================= geometry for a step
class StepGeom:
    """Everything that depends only on psi at the start of a step: smoothed indicator, active cells, interface
    weights and the sparse stencil (pattern) shared by all fields."""

    def __init__(self, g: Grid, psi, z_plane, p):
        self.g, dx = g, g.dx
        self.psi = psi
        eps = p.eps_fac * dx
        self.eps = eps
        self.z_plane = z_plane
        jt = int(np.floor((z_plane - 0.5 * dx - g.z_lo) / dx - 0.5))
        jt = min(jt, g.nz - 1)
        self.jt = jt
        self.d_top = z_plane - g.zc[0, jt]                     # in [0.5 dx, 1.5 dx)
        gm = grad_mag(psi, dx)
        x = np.clip(psi / eps, -30, 30)
        self.Hs = np.maximum(0.5 * (1 + np.tanh(x)), 1e-6)
        self.gHs = 0.5 / eps / np.cosh(x) ** 2 * gm
        # ---- electrolyte connectivity -> open cavity / trapped voids
        conn = -0.5 * dx
        open_m, void_m, _ = label_regions(psi, g, jt, conn)
        band = (psi <= conn) & (psi > -p.n_active_eps * eps)
        band[:, jt + 1:] = False
        if void_m.any():
            E = open_m | void_m
            _, (ii, jj) = ndi.distance_transform_edt(~E, return_indices=True)
            band_void = band & void_m[ii, jj]
            act = open_m | (band & ~band_void)
            self.void_cells = void_m | band_void
        else:
            act = open_m | band
            self.void_cells = void_m
        # keep only the connected piece that touches the Dirichlet row (guards against singular islands)
        lab, n = ndi.label(act)
        if n > 1:
            top = np.unique(lab[:, jt])
            act = np.isin(lab, top[top > 0])
        self.open_m, self.void_m, self.act = open_m, void_m, act
        n = int(act.sum())
        self.n = n
        idx = -np.ones(act.shape, np.int64)
        idx[act] = np.arange(n)
        self.idx = idx
        # ---- interface weights (zero for inert surfaces in test mode)
        w = g.vol * self.gHs
        w = np.where(act, w, 0.0)
        self.w_full = w
        self.w = w[act]
        # ---- stencil
        ar = act[:-1, :] & act[1:, :]
        self.ir, self.jr = np.nonzero(ar)
        self.pr, self.qr = idx[self.ir, self.jr], idx[self.ir + 1, self.jr]
        self.geo_r = g.AE[self.ir, 0] / dx
        self.Hf_r = 0.5 * (self.Hs[self.ir, self.jr] + self.Hs[self.ir + 1, self.jr])
        az = act[:, :-1] & act[:, 1:]
        self.iz, self.jz = np.nonzero(az)
        self.pz, self.qz = idx[self.iz, self.jz], idx[self.iz, self.jz + 1]
        self.geo_z = g.AZ[self.iz, 0] / dx
        self.Hf_z = 0.5 * (self.Hs[self.iz, self.jz] + self.Hs[self.iz, self.jz + 1])
        # periodic x seam: cell 0 (x ~ 0+) and cell nr-1 (x ~ pitch-) are neighbours.  p = column 0,
        # q = column nr-1, so the p -> q face normal points in -x (used for the advective flux sign).
        if g.periodic_x:
            aw = act[-1, :] & act[0, :]
            self.jw = np.nonzero(aw)[0]
            self.pw = idx[0, self.jw]
            self.qw = idx[g.nr - 1, self.jw]
            self.geo_w = g.AE[0, 0] / dx
            self.Hf_w = 0.5 * (self.Hs[0, self.jw] + self.Hs[g.nr - 1, self.jw])
        else:
            self.jw = np.empty(0, np.int64)
            self.pw = self.qw = np.empty(0, np.int64)
            self.geo_w = 1.0
            self.Hf_w = np.empty(0)
        self.itop = np.nonzero(act[:, jt])[0]
        self.ptop = idx[self.itop, jt]
        self.geo_top = g.AZ[self.itop, 0] / self.d_top
        self.Hs_top = self.Hs[self.itop, jt]
        if g.periodic_x:
            rows = np.concatenate([self.pr, self.qr, self.pz, self.qz, self.pw, self.qw, np.arange(n)])
            cols = np.concatenate([self.qr, self.pr, self.qz, self.pz, self.qw, self.pw, np.arange(n)])
        else:
            rows = np.concatenate([self.pr, self.qr, self.pz, self.qz, np.arange(n)])
            cols = np.concatenate([self.qr, self.pr, self.qz, self.pz, np.arange(n)])
        nnz = rows.size
        P = sp.csr_matrix((np.arange(nnz) + 1.0, (rows, cols)), shape=(n, n))
        self.perm = (P.data - 1).astype(np.int64)
        self.indices, self.indptr = P.indices, P.indptr
        inv = np.empty(nnz, np.int64)
        inv[self.perm] = np.arange(nnz)
        self.diag_pos = inv[nnz - n + np.arange(n)]
        self.vh = (g.vol * self.Hs)[act]                      # V_p Hs_p (transient storage term)

    # ------------------------------------------------------------------ matrix assembly
    def K_data(self, D):
        """Data vector (CSR order) of the symmetric operator  -div(Hs D grad .)  incl. Dirichlet at the top plane,
        and the Dirichlet coupling vector bc (n,) so that the bulk value u0 enters the rhs as bc*u0."""
        n = self.n
        g = self.g
        Gr = self.Hf_r * 0.5 * (D[self.ir, self.jr] + D[self.ir + 1, self.jr]) * self.geo_r
        Gz = self.Hf_z * 0.5 * (D[self.iz, self.jz] + D[self.iz, self.jz + 1]) * self.geo_z
        # float accumulator: np.bincount returns int64 when its index array is empty, and adding bc (float)
        # to that would be an illegal cast.  The summation order is unchanged, so the sum is bit-identical.
        diag = np.zeros(n)
        diag += np.bincount(self.pr, Gr, n)
        diag += np.bincount(self.qr, Gr, n)
        diag += np.bincount(self.pz, Gz, n)
        diag += np.bincount(self.qz, Gz, n)
        Gw = np.zeros(0)
        if g.periodic_x:
            jw = self.jw
            Gw = self.Hf_w * 0.5 * (D[0, jw] + D[g.nr - 1, jw]) * self.geo_w
            diag += np.bincount(self.pw, Gw, n) + np.bincount(self.qw, Gw, n)
        bc = np.zeros(n)
        bc[self.ptop] = self.Hs_top * D[self.itop, self.jt] * self.geo_top
        diag += bc
        if g.periodic_x:
            data = np.concatenate([-Gr, -Gr, -Gz, -Gz, -Gw, -Gw, diag])
        else:
            data = np.concatenate([-Gr, -Gr, -Gz, -Gz, diag])
        return data[self.perm], bc

    # ------------------------------------------------------------------ Scharfetter-Gummel (tertiary transport)
    def sg_operator(self, phi, z, D, f_RT, chi_max=30.0, adv=None):
        """Scharfetter-Gummel species operator A with (A c)_p = sum of the outward SG face fluxes.

        Face p -> q:  N = g [ B(chi) c_p - B(-chi) c_q ],  chi = z (phi_q - phi_p) F/(RT),
        g = D_face * geo * Hf (the same face measure as K_data), D_face = harmonic mean.
        B(-chi) = B(chi) + chi is used, so the row entries are g*B(chi) (diagonal side) and
        -g*(B(chi)+chi).  The Dirichlet coupling at the delta plane is folded into the diagonal and
        returned as bc (bulk value enters the rhs as bc*c_b).  Non-symmetric for z != 0; B > 0 keeps
        the operator an M-matrix for any chi, so c stays positive."""
        g = self.g
        D = np.asarray(D, dtype=float)
        phi = np.asarray(phi, dtype=float)
        rows, cols, vals = [], [], []

        def add(p, q, geo, hf, dp, dq, phip, phiq):
            if p.size == 0:
                return
            Df = 2.0 * dp * dq / np.maximum(dp + dq, 1e-300)          # harmonic face diffusivity
            gs = Df * geo * hf
            chi = np.clip(z * f_RT * (phiq - phip), -chi_max, chi_max)
            bb = bern(chi)
            bm = bb + chi                                             # B(-chi)
            rows.extend([p, p, q, q])
            cols.extend([p, q, p, q])
            vals.extend([gs * bb, -gs * bm, -gs * bb, gs * bm])

        add(self.pr, self.qr, self.geo_r, self.Hf_r,
            D[self.ir, self.jr], D[self.ir + 1, self.jr], phi[self.ir, self.jr], phi[self.ir + 1, self.jr])
        add(self.pz, self.qz, self.geo_z, self.Hf_z,
            D[self.iz, self.jz], D[self.iz, self.jz + 1], phi[self.iz, self.jz], phi[self.iz, self.jz + 1])
        if g.periodic_x and self.jw.size:
            jw = self.jw
            add(self.pw, self.qw, self.geo_w, self.Hf_w,
                D[0, jw], D[g.nr - 1, jw], phi[0, jw], phi[g.nr - 1, jw])
        n = self.n
        if not vals:
            return sp.csr_matrix((n, n)), np.zeros(n)
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
        bc = np.zeros(n)
        bc[self.ptop] = self.Hs_top * D[self.itop, self.jt] * self.geo_top
        A = A + sp.diags(bc)
        if adv is not None:
            A = A + adv
        return A.tocsr(), bc

    # ------------------------------------------------------------------ advection (resolved cavity flow)
    def face_fluxes(self, ux, uz):
        """Signed volume fluxes F [cm2/s] of every stencil face, oriented p -> q.

        Face-normal velocity = arithmetic mean of the two cell-centred velocities (zero when either cell is
        solid, psi <= 0); the volume flux is Hs_face * geo * dx * u_n, the same face measure that carries the
        diffusive conductances in K_data (planar: geo = 1, dx = cell width, so F = u * dx as in
        trenchflow.transport).  The wrap face p -> q points in -x (p is column 0, q column nr-1)."""
        g = self.g
        dx = g.dx
        elyt = self.psi > 0
        u_r = 0.5 * (ux[self.ir, self.jr] + ux[self.ir + 1, self.jr])
        ok_r = elyt[self.ir, self.jr] & elyt[self.ir + 1, self.jr]
        F_r = np.where(ok_r, self.Hf_r * self.geo_r * dx * u_r, 0.0)
        u_z = 0.5 * (uz[self.iz, self.jz] + uz[self.iz, self.jz + 1])
        ok_z = elyt[self.iz, self.jz] & elyt[self.iz, self.jz + 1]
        F_z = np.where(ok_z, self.Hf_z * self.geo_z * dx * u_z, 0.0)
        if g.periodic_x and self.jw.size:
            jw = self.jw
            u_w = -0.5 * (ux[0, jw] + ux[g.nr - 1, jw])          # p -> q is -x across the seam
            ok_w = elyt[0, jw] & elyt[g.nr - 1, jw]
            F_w = np.where(ok_w, self.Hf_w * self.geo_w * dx * u_w, 0.0)
        else:
            F_w = np.zeros(0)
        return F_r, F_z, F_w

    def mac_face_fluxes(self, ufx, ufz):
        """Signed volume fluxes F [cm2/s] of every stencil face, oriented p -> q, taken DIRECTLY from the
        staggered (MAC) face velocities of a flow solver such as platesim.trenchflow.stokes.solve_stokes.

        No interpolation and no Hs weighting: the face velocity is already the face-normal MAC unknown, so
        the discrete divergence of the resulting flux set is the discrete divergence of the velocity field
        (which the Stokes solver makes ~1e-14 in every open cell).  ufx has the engine orientation
        (nr, nz): ufx[i, j] is the x-face between cells (i-1, j) and (i, j), i.e. between (ir, jr) and
        (ir+1, jr) it is ufx[ir+1, jr]; ufz (nr, nz+1): ufz[i, m] is the z-face between rows m-1 and m,
        i.e. between (iz, jz) and (iz, jz+1) it is ufz[iz, jz+1].  Both index conventions are the
        transposed CavityGrid layout documented in trenchflow.stokes._layout, and the faces of cells that
        touch a solid cell (psi <= 0) are exactly 0 in the MAC field.

        The face measure is the same geometric factor face_fluxes uses (geo * dx for planar trench_full,
        where geo = 1) but WITHOUT the Hs face weight (the Stokes field is sharp).  The wrap face p -> q
        points in -x, so F_w = -ufx[0, jw] * dx * geo.  The returned F_top (one value per active top cell
        ptop, positive = leaving the species domain upward) is the flux through the z-face between rows
        jt and jt+1, which the species stencil truncates at the Dirichlet plane."""
        g = self.g
        dx = g.dx
        F_r = ufx[self.ir + 1, self.jr] * dx * self.geo_r
        F_z = ufz[self.iz, self.jz + 1] * dx * self.geo_z
        if g.periodic_x and self.jw.size:
            F_w = -ufx[0, self.jw] * dx * self.geo_w
        else:
            F_w = np.zeros(0)
        geo_top_z = g.AZ[self.itop, 0] / dx
        F_top = ufz[self.itop, self.jt + 1] * dx * geo_top_z
        return F_r, F_z, F_w, F_top

    def mac_advection(self, ufx, ufz):
        """Conservative advection operator from the MAC face velocities, with the open top boundary.

        Returns (adv, adv_bc, F_top):
          * adv = adv_matrix(F_r, F_z, F_w, form='divergence') plus the outflow diagonal: every top cell
            with F_top > 0 loses  F_top * c_P  through the Dirichlet plane, so +F_top is added to its
            diagonal (row sum = full discrete flux divergence, including the plane).
          * adv_bc = rhs source of the inflow top cells: F_top < 0 brings bulk fluid in, |F_top| times the
            species bulk value (the caller scales it: 1 for the normalised Cu2+/suppressor unknowns,
            c_bulk / c_h_bulk for the absolute tertiary species).
          * F_top as in mac_face_fluxes (per active top cell).
        With these terms the operator preserves a uniform field exactly: (adv @ 1 - adv_bc) == 0 in
        every row, because the interior flux divergence of a top cell is -F_top."""
        F_r, F_z, F_w, F_top = self.mac_face_fluxes(ufx, ufz)
        adv = self.adv_matrix(F_r, F_z, F_w, form="divergence")
        diag = np.zeros(self.n)
        rhs = np.zeros(self.n)
        if F_top.size:
            out = F_top > 0
            diag[self.ptop[out]] = F_top[out]
            rhs[self.ptop[~out]] = -F_top[~out]
        if np.any(diag):
            adv = adv + sp.diags(diag)
        return adv, rhs, F_top

    def adv_matrix(self, F_r, F_z, F_w, form="gradient"):
        """First-order upwind advection operator (CSR), non-symmetric.

        form='gradient' (default): rows sum to zero, i.e. the operator is  u.grad(c)  discretised as
            sum_faces F (c_f - c_P) with upwind face values.  For a face p -> q with signed flux F:
            F > 0 -> A[q,q] += F, A[q,p] -= F;  F < 0 -> A[p,p] += -F, A[p,q] -= -F.
            A uniform field is therefore preserved EXACTLY, whatever the face fluxes do (the LBM fluxes
            are not discretely divergence free: cell-mean interpolation plus the truncation at the
            Dirichlet top plane leave a residual, and the conservative form below turns that residual
            into a spurious source -c div(Hs u) per cell -> the trench_full shoulder ripple).
        form='divergence' (legacy): the conservative form  +div(Hs u c); column sums are zero.
            F > 0 -> A[p,p] += F, A[q,p] -= F;  F < 0 -> A[p,q] += F, A[q,q] -= F.
        """
        if form not in ("gradient", "divergence"):
            raise ValueError(f"adv_matrix form must be 'gradient' or 'divergence', got {form!r}")
        rows, cols, vals = [], [], []
        for p, q, F in ((self.pr, self.qr, F_r), (self.pz, self.qz, F_z), (self.pw, self.qw, F_w)):
            if F.size == 0:
                continue
            pos, neg = F > 0, F < 0
            if form == "gradient":
                rows += [q[pos], q[pos], p[neg], p[neg]]
                cols += [q[pos], p[pos], p[neg], q[neg]]
                vals += [F[pos], -F[pos], -F[neg], F[neg]]
            else:
                rows += [p[pos], q[pos], p[neg], q[neg]]
                cols += [p[pos], p[pos], q[neg], q[neg]]
                vals += [F[pos], -F[pos], F[neg], -F[neg]]
        if not rows:
            return sp.csr_matrix((self.n, self.n))
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.n, self.n)).tocsr()

    def flux_divergence(self, F_r, F_z, F_w):
        """Per-unknown discrete divergence of the face fluxes: (div F)_p = sum of the OUTGOING signed
        fluxes of cell p (length n).  Zero everywhere iff the flux field is discretely divergence free;
        for a uniform c, adv_matrix(form='divergence') @ ones == flux_divergence(...)."""
        div = np.zeros(self.n)
        for p, q, F in ((self.pr, self.qr, F_r), (self.pz, self.qz, F_z), (self.pw, self.qw, F_w)):
            if F.size == 0:
                continue
            np.add.at(div, p, F)
            np.add.at(div, q, -F)
        return div

    def factor(self, data, extra=None):
        d = data.copy() if extra is not None else data
        if extra is not None:
            d[self.diag_pos] += extra
        A = sp.csc_matrix((d, self.indices, self.indptr), shape=(self.n, self.n))   # symmetric: CSR == CSC
        return spla.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                         options=dict(SymmetricMode=True))

    def matvec(self, data, x):
        A = sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))
        return A @ x


# ======================================================================================= LU reuse
class CachedSym:
    """Symmetric sparse operator K + diag(extra) with a cached LU.  The diagonal changes between fixed-point sweeps
    (kinetic sinks), so later solves reuse the first factorisation as a preconditioner (iterative refinement) and only
    refactor when the refinement stalls."""

    def __init__(self, geo, data, tol=1e-7, max_ref=14, lu=None):
        self.geo, self.data, self.lu, self.tol, self.max_ref = geo, data, lu, tol, max_ref
        self.n_fac = 0
        self.n_sol = 0

    def _A(self, extra):
        d = self.data.copy()
        d[self.geo.diag_pos] += extra
        return sp.csr_matrix((d, self.geo.indices, self.geo.indptr), shape=(self.geo.n, self.geo.n))

    def _fac(self, extra):
        self.lu = self.geo.factor(self.data, extra)
        self.n_fac += 1

    def solve(self, extra, rhs):
        rhs2 = rhs if rhs.ndim == 2 else rhs[:, None]
        self.n_sol += 1
        if self.lu is None:
            self._fac(extra)
            x = self.lu.solve(rhs2)
        else:
            A = self._A(extra)
            x = self.lu.solve(rhs2)
            bn = np.linalg.norm(rhs2, axis=0) + 1e-300
            rn_old = np.inf
            ok = False
            for k in range(self.max_ref + 1):
                r = rhs2 - A @ x
                rn = np.max(np.linalg.norm(r, axis=0) / bn)
                if rn < self.tol:
                    ok = True
                    break
                if k == self.max_ref or rn > 0.9 * rn_old:
                    break
                rn_old = rn
                x = x + self.lu.solve(r)
            if not ok:
                self._fac(extra)
                x = self.lu.solve(rhs2)
            elif k >= 10:
                self._fac(extra)                     # converging slowly: refresh for the following solves
        return x if rhs.ndim == 2 else x[:, 0]


class CachedGen:
    """General (non-symmetric) sparse operator A0 + diag(extra) with a cached COLAMD LU.

    Used for the advection-diffusion equations of 'trench_full'.  Same refinement strategy as CachedSym:
    the diagonal changes between fixed-point sweeps, so the first factorisation is reused as a
    preconditioner and refreshed only when the refinement stalls."""

    def __init__(self, geo, base, tol=1e-7, max_ref=14):
        self.geo, self.base, self.lu, self.tol, self.max_ref = geo, base, None, tol, max_ref
        self.n_fac = 0
        self.n_sol = 0

    def _A(self, extra):
        return (self.base + sp.diags(extra, format='csr')).tocsr()

    def _fac(self, extra):
        self.lu = spla.splu(self._A(extra).tocsc(), permc_spec='COLAMD')
        self.n_fac += 1

    def solve(self, extra, rhs):
        rhs2 = rhs if rhs.ndim == 2 else rhs[:, None]
        self.n_sol += 1
        if self.lu is None:
            self._fac(extra)
            x = self.lu.solve(rhs2)
        else:
            A = self._A(extra)
            x = self.lu.solve(rhs2)
            bn = np.linalg.norm(rhs2, axis=0) + 1e-300
            rn_old = np.inf
            ok = False
            for k in range(self.max_ref + 1):
                r = rhs2 - A @ x
                rn = np.max(np.linalg.norm(r, axis=0) / bn)
                if rn < self.tol:
                    ok = True
                    break
                if k == self.max_ref or rn > 0.9 * rn_old:
                    break
                rn_old = rn
                x = x + self.lu.solve(r)
            if not ok:
                self._fac(extra)
                x = self.lu.solve(rhs2)
            elif k >= 10:
                self._fac(extra)
        return x if rhs.ndim == 2 else x[:, 0]


# ======================================================================================= kinetics
class Kinetics:
    def __init__(self, ph, p):
        self.ph, self.p = ph, p
        self.f = ph.f_RT

    def cath(self, eta, accel):
        """Unsuppressed / suppressed cathodic current density at c = c_bulk (A/cm2) for overpotential eta (V)."""
        ph, p = self.ph, self.p
        gu = ph.i0_u * accel * np.exp(np.clip(-p.alpha_u * self.f * eta, -60, 60))
        gs = ph.i0_s * accel * np.exp(np.clip(-p.alpha_s * self.f * eta, -60, 60))
        return gu, gs

    def anod(self, eta):
        a = self.ph.i0_a * np.exp(np.clip(self.p.alpha_a * self.f * eta, -60, 60))
        return a, self.p.alpha_a * self.f * a        # (i_a, d i_a / d eta)


def theta_be(th_old, dt, a_rate, vu, vs, km_duty, b_extra):
    """Backward-Euler solution of  dth/dt = a (1-th) - [km (vu (1-th) + vs th) + b_extra] th  (per cell, exact quadratic).
    a_rate = k_plus*s_surf [1/s]; vu, vs: growth velocities of fully free / fully suppressed surface [cm/s];
    km_duty = k_minus*duty_f [1/cm]. Unique root in [0,1]; stable for any dt (quasi-static at large dt)."""
    A2 = dt * km_duty * (vs - vu)
    B = 1.0 + dt * a_rate + dt * km_duty * vu + dt * b_extra
    C = -(th_old + dt * a_rate)
    disc = np.maximum(B * B - 4 * A2 * C, 0.0)
    return np.clip(-2 * C / (B + np.sqrt(disc)), 0.0, 1.0)


# ======================================================================================= potential problems
def scalar_V_cathodic(kin, w, ct, th, acc, phi, I_t, V0=None, lo=-4.0, hi=2.0):
    """V such that sum w ct ci(V - phi) = I_t with phi fixed.  I(V) is monotone decreasing and ln I is nearly linear
    in V, so Newton on ln I - ln I_t converges in 3-5 iterations; brentq is the fallback."""
    p = kin.p
    f_ = kin.f
    lnT = np.log(I_t)
    V = -0.2 if V0 is None else float(np.clip(V0, lo, hi))
    for _ in range(40):
        gu, gs = kin.cath(V - phi, acc)
        wu, ws = w * ct * (1 - th) * gu, w * ct * th * gs
        I = wu.sum() + ws.sum()
        if I <= 0:
            break
        dlnI = -f_ * (p.alpha_u * wu.sum() + p.alpha_s * ws.sum()) / I
        r = np.log(I) - lnT
        if abs(r) < 1e-11:
            return V
        step = r / dlnI
        V = float(np.clip(V - np.clip(step, -0.5, 0.5), lo, hi))

    def g(V):
        gu, gs = kin.cath(V - phi, acc)
        return np.dot(w * ct, (1 - th) * gu + th * gs) - I_t
    glo, ghi = g(lo), g(hi)
    if glo < 0:
        return lo
    if ghi > 0:
        return hi
    return brentq(g, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200)


def newton_potential(geo, cs, kin, w, ct, th, acc, V0, phi0, I_t, galvano, anodic=False, maxit=14, tol_eta=2e-7,
                     src=None):
    """Solve  K phi + sigma w I(V-phi) = 0  [sigma = +1 cathodic sink, -1 anodic source] and, if galvano,
    sum w I = I_t for the unknowns (phi, V) with Newton (bordered system reduced by a Schur complement).
    src (tertiary path only): the divergence of the diffusion current F sum z_i D_i grad c_i, so the
    residual is  K phi + sigma w I - src = 0  (charge balance  div i_l = 0  with i_l = -kappa grad phi
    - F sum z_i D_i grad c_i).  src=None reproduces the enhanced path exactly.
    Returns V, phi, I_p (A/cm2 per cell incl. ct factor for cathodic), iterations."""
    sigma = -1.0 if anodic else 1.0
    phi = phi0.copy()
    V = V0
    n = geo.n

    def current(eta):
        if anodic:
            i, di = kin.anod(eta)                      # di = dI/deta > 0
            return i, di
        gu, gs = kin.cath(eta, acc)
        i = ct * ((1 - th) * gu + th * gs)
        di = kin.f * ct * (kin.p.alpha_u * (1 - th) * gu + kin.p.alpha_s * th * gs)   # -dI/deta > 0
        return i, di

    if galvano:
        if anodic:
            # closed form for fixed phi: sum w i0a exp(af (V-phi)) = I_t
            ph = kin.ph
            s = np.dot(w, np.exp(np.clip(-kin.p.alpha_a * kin.f * phi, -60, 60)))
            V = np.log(I_t / (ph.i0_a * s + 1e-300)) / (kin.p.alpha_a * kin.f)
        else:
            V = scalar_V_cathodic(kin, w, ct, th, acc, phi, I_t, V0=V0)
    for it in range(maxit):
        eta = V - phi
        I, q0 = current(eta)
        q = w * q0
        R = geo.matvec(cs.data, phi) + sigma * w * I
        if src is not None:
            R = R - src
        g = np.dot(w, I) - I_t
        sol = cs.solve(q, np.column_stack([-R, q]))
        x1, x2 = sol[:, 0], sol[:, 1]
        if galvano:
            dV = (sigma * g + np.dot(q, x1)) / (q.sum() - np.dot(q, x2) + 1e-300)
        else:
            dV = 0.0
        dphi = x1 + x2 * dV
        deta = dV - dphi
        m = np.max(np.abs(deta)) if n else 0.0
        lam = min(1.0, 0.15 / (m + 1e-300))
        phi = phi + lam * dphi
        V = V + lam * dV
        if lam == 1.0 and m < tol_eta:
            break
    eta = V - phi
    I, _ = current(eta)
    return V, phi, I, it + 1


# ======================================================================================= main step solver
class StepSolver:
    PHI_SC = 0.05        # V   weights for the Anderson least-squares norm
    V_SC = 0.05

    def __init__(self, geo: StepGeom, p, ph, Dt, reuse=None, flow=None, flow_faces=None):
        """reuse: LU factors (c, s, phi) of the previous step; valid when the active cell set is unchanged, they
        then serve as preconditioners (refinement) for the slightly different matrices of this step.
        flow: (ux, uz) cell-centred velocities (nr, nz) in cm/s for the resolved cavity flow
        ('trench_full'); the advective operator is added to the Cu2+ and suppressor equations.
        flow_faces: (ufx, ufz) MAC face velocities in the engine orientation (nr, nz) / (nr, nz+1);
        required by flow_solver='stokes', which then uses the conservative divergence-form operator of
        StepGeom.mac_advection (plus its open-top rhs adv_bc) instead of the interpolated face fluxes."""
        self.geo, self.p, self.ph = geo, p, ph
        self.kin = Kinetics(ph, p)
        self.Dt = Dt
        self.flow = flow
        self.flow_faces = flow_faces
        self.adv_bc = None                  # rhs source of the open-top inflow (stokes MAC path only)
        self.adv_top = None                 # top-plane z-face flux per active top cell (stokes path only)
        self.tertiary = p.transport == "tertiary"
        g = geo.g
        lu_c = lu_s = lu_p = None
        if reuse is not None and reuse[0].shape == geo.act.shape and np.array_equal(reuse[0], geo.act):
            lu_c, lu_s, lu_p = reuse[1:]
        if self.tertiary:
            self._init_tertiary(Dt, reuse, lu_s)
            return
        Kc, self.bc_c = geo.K_data(ph.D_c + Dt)
        Ks, self.bc_s = geo.K_data(ph.D_s + Dt)
        self.adv = None
        if g.periodic_x:
            # non-symmetric advection-diffusion: general LU (no symmetric mode), no cross-step LU reuse
            ux, uz = (np.zeros((g.nr, g.nz)), np.zeros((g.nr, g.nz))) if flow is None else flow
            if p.flow_solver == "stokes" and flow_faces is not None:
                adv, self.adv_bc, self.adv_top = geo.mac_advection(*flow_faces)
            else:
                adv = geo.adv_matrix(*geo.face_fluxes(ux, uz), form=p.adv_form)
            self.adv = adv                                # exposed for the sim.py Cu2+ balance
            self.cs_c = CachedGen(geo, self._base(Kc, adv))
            self.cs_s = CachedGen(geo, self._base(Ks, adv))
        else:
            self.cs_c, self.cs_s = CachedSym(geo, Kc, lu=lu_c), CachedSym(geo, Ks, lu=lu_s)
        self.cs_p = None
        if p.ohmic:
            Kp, _ = geo.K_data(ph.kappa * np.ones((g.nr, g.nz)))
            self.cs_p = CachedSym(geo, Kp, lu=lu_p)

    def _base(self, K_data, adv):
        """Constant part (diffusion + advection) of a trench_full operator, CSR."""
        geo = self.geo
        return sp.csr_matrix((K_data, geo.indices, geo.indptr), shape=(geo.n, geo.n)) + adv

    # ------------------------------------------------------------------ tertiary transport
    def _init_tertiary(self, Dt, reuse=None, lu_s=None):
        """Operators for transport == 'tertiary' (explicit Cu2+ / H+ / SO4^2-).

        Cu2+ stays in the Anderson vector as ctil = c_Cu/c_bulk, so every existing diagnostic and the
        Cu2+ mass balance of sim.py keep their meaning; H+ and SO4^2- live in st['c_h']/st['c_so4'] in
        mol/cm3 (absolute).  c_SO4 is closed by electroneutrality sum z_i c_i = 0 (COMSOL/Newman:
        eliminate the majority inert anion; an eliminated species cannot carry a flux BC), so only Cu2+
        and H+ are solved implicitly and the potential comes from the charge balance
        div(-kappa grad phi - F sum z_i D_i grad c_i) = 0 with kappa evaluated locally."""
        geo, p, ph, g = self.geo, self.p, self.ph, self.geo.g
        self.D_cu_a = ph.D_cu + Dt
        self.D_h_a = ph.D_H + Dt
        self.D_s_a = ph.D_so4 + Dt
        self.cb = (p.c_bulk, ph.c_h_bulk, ph.c_so4_bulk)
        self.zv = (ph.z_cu, ph.z_h, ph.z_so4)
        adv = None
        if g.periodic_x:
            ux, uz = (np.zeros((g.nr, g.nz)), np.zeros((g.nr, g.nz))) if self.flow is None else self.flow
            if p.flow_solver == "stokes" and self.flow_faces is not None:
                adv, self.adv_bc, self.adv_top = geo.mac_advection(*self.flow_faces)
            else:
                adv = geo.adv_matrix(*geo.face_fluxes(ux, uz), form=p.adv_form)
        self.adv = adv
        zero = np.zeros((g.nr, g.nz))
        A0, self.bc_c = geo.sg_operator(zero, ph.z_cu, self.D_cu_a, ph.f_RT,
                                        chi_max=p.tertiary_chi_max, adv=adv)
        Ah0, self.bc_h = geo.sg_operator(zero, ph.z_h, self.D_h_a, ph.f_RT,
                                         chi_max=p.tertiary_chi_max, adv=adv)
        self.sg_c, self.sg_h = CachedGen(geo, A0), CachedGen(geo, Ah0)
        self.cs_c = self.sg_c                       # lu_state()/n_fac bookkeeping
        Ks, self.bc_s = geo.K_data(ph.D_s + Dt)
        self.cs_s = CachedGen(geo, self._base(Ks, adv)) if adv is not None else CachedSym(geo, Ks, lu=lu_s)
        # diffusion-current source: constant-D operators per species (K_data data vectors, CSR order)
        self.kd = [geo.K_data(self.D_cu_a), geo.K_data(self.D_h_a), geo.K_data(self.D_s_a)]
        self.D_n = (self.D_cu_a[geo.act], self.D_h_a[geo.act], self.D_s_a[geo.act])
        self.kappa_floor = max(1e-3 * ph.kappa_ne, 1e-9)
        self.c_floor = 1e-6 * p.c_bulk
        self.c_h_scale = max(ph.c_h_bulk, 1e-3 * p.c_bulk)
        self.cs_p = None
        lu_p = None
        if reuse is not None and reuse[0].shape == geo.act.shape and np.array_equal(reuse[0], geo.act):
            lu_p = reuse[3]
        if p.ohmic:
            kp, _ = geo.K_data(np.full((g.nr, g.nz), ph.kappa_ne))
            self.cs_p = CachedSym(geo, kp, lu=lu_p)

    def _kappa(self, c_cu, c_h, c_s):
        """Local Nernst-Einstein conductivity (S/cm) on the active cells, floored, full grid."""
        geo, ph = self.geo, self.ph
        kap = (ph.z_cu ** 2 * self.D_n[0] * c_cu + ph.z_h ** 2 * self.D_n[1] * c_h
               + ph.z_so4 ** 2 * self.D_n[2] * c_s)
        kap = np.maximum(F * ph.f_RT * kap, self.kappa_floor)
        out = np.full(geo.act.shape, ph.kappa_ne)
        out[geo.act] = kap
        return out

    def _diff_src(self, c_cu, c_h, c_s):
        """Charge source of the diffusion current in the potential equation:  -F sum_i z_i * (outward
        diffusion flux)_i, with the outward flux  K_i c_i - bc_i c_b,i  on the K_data face measure.

        The potential residual is  K phi + w I - src = 0, i.e.  sum of outward currents = 0  for
        i_l = -kappa grad phi - F sum z_i D_i grad c_i  (ohmic term K phi, interface term w I).  For a
        uniform c the source vanishes in the interior and at the top plane (c = c_b).  Omitting it
        would leave only the ohmic field and under-predict the limiting current by the diffusion
        potential (research/tertiary_transport_notes.md sect. 3)."""
        geo, ph = self.geo, self.ph
        src = np.zeros(geo.n)
        for z, c, (kd, bc), cb in ((ph.z_cu, c_cu, self.kd[0], self.cb[0]),
                                   (ph.z_h, c_h, self.kd[1], self.cb[1]),
                                   (ph.z_so4, c_s, self.kd[2], self.cb[2])):
            src -= z * F * (geo.matvec(kd, c) - bc * cb)
        return src

    def lu_state(self):
        return (self.geo.act.copy(), self.cs_c.lu, self.cs_s.lu, self.cs_p.lu if self.cs_p else None)

    # ------------------------------------------------------------------
    def solve(self, st, ctrl, dt):
        """Fixed-point solution of the coupled (theta, s, phi/V, c) problem on the frozen geometry, Anderson
        accelerated.  st: dict of full-grid arrays + scalars (updated in place).
        ctrl: dict(mode 'I'|'V', I_t (A, instantaneous forward total), V_set, duty_f, duty_r, rr, pr)."""
        geo, p, ph, kin = self.geo, self.p, self.ph, self.kin
        act, w, n = geo.act, geo.w, geo.n
        galvano = ctrl['mode'] == 'I'
        duty_f, duty_r = ctrl['duty_f'], ctrl['duty_r']
        acc = (1.0 + p.b_acc * st['theta_a'][act]) if p.use_accel else 1.0
        th0 = st['theta'][act].copy()
        s_old = st['s'][act].copy()
        Mf = ph.Mfac
        # ---- reverse (anodic) problem: independent of c; solved once per step, warm started
        Ia = np.zeros(n)
        phi_r, V_r = None, st['V_r']
        if ctrl['pr']:
            I_ta = ctrl['rr'] * ctrl['I_t']
            if p.ohmic:
                V_r, phi_r, Ia, _ = newton_potential(geo, self.cs_p, kin, w, None, None, 1.0, st['V_r'],
                                                     st['phi_r'][act], I_ta, True, anodic=True)
            else:
                V_r, phi_r, _, _ = self._rev_nonohmic(w, I_ta)
                Ia = kin.anod(V_r - phi_r)[0]
        # [J] the Cu2+ sink/source is scaled by the current efficiency, like the front velocity Mfac: only
        # the fraction of the charge that deposits Cu consumes (or returns) Cu2+, so the Cu mass balance
        # closes against the deposit at efficiency < 1.  x * 1.0 is exact -> efficiency = 1 is bit-identical.
        rev_src = p.efficiency * duty_r * w * Ia / (2 * F * p.c_bulk) if ctrl['pr'] else 0.0
        wm = w > 0.02 * w.max() if n else w > 0
        diag = dict(nph=0)
        # ---- tertiary transport state (absolute mol/cm3; Cu2+ stays normalised in the Anderson vector)
        rev_src_abs = p.efficiency * duty_r * w * Ia / (2 * F) if ctrl['pr'] else 0.0
        if self.tertiary:
            if 'c_h' not in st:
                st['c_h'] = np.full(st['ctil'].shape, ph.c_h_bulk)
            if 'c_so4' not in st:
                st['c_so4'] = np.full(st['ctil'].shape, ph.c_so4_bulk)
        ter = dict(c_h=st['c_h'][act].copy() if self.tertiary else None,
                   c_so4=st['c_so4'][act].copy() if self.tertiary else None, d_h=0.0)

        def sweep_enh(x):
            """One Gauss-Seidel pass  theta -> s -> (V, phi) -> c  (the fixed-point map)."""
            th, s_t, ct, phi, V = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n] * self.PHI_SC, x[4 * n] * self.V_SC
            if not p.ohmic:
                phi = np.zeros(n)
            if not galvano:
                V = ctrl['V_set']
            eta = V - phi
            gu, gs = kin.cath(eta, acc)
            if p.suppressor:
                b_ex = p.k_rd * duty_r * Ia / 1e-3 if ctrl['pr'] else 0.0
                th_new = theta_be(th0, dt, p.k_plus * p.s_bulk * s_t, Mf * ct * gu, Mf * ct * gs,
                                  p.k_minus * duty_f, b_ex)
                extra = w * p.Gamma * p.k_plus * (1 - th_new)          # sink = Gamma k_plus s (1-theta)
                rhs = self.bc_s.copy()
                if self.adv_bc is not None:                   # bulk normalised: s_bulk -> 1
                    rhs = rhs + self.adv_bc
                if p.s_transient:
                    extra = extra + geo.vh / dt
                    rhs += geo.vh / dt * s_old
                s_new = np.clip(self.cs_s.solve(extra, rhs), 0.0, 1.5)
            else:
                th_new, s_new = th0 * 0.0, s_t
            if p.ohmic:
                V_new, phi_new, _, nph = newton_potential(geo, self.cs_p, kin, w, ct, th_new, acc, V, phi,
                                                          ctrl['I_t'], galvano)
            else:
                V_new, phi_new, _, nph = self._nonohmic(w, ct, th_new, acc, V, ctrl, galvano)
            diag['nph'] += nph
            gu, gs = kin.cath(V_new - phi_new, acc)
            ci = (1 - th_new) * gu + th_new * gs
            extra = p.efficiency * duty_f * w * ci / (2 * F * p.c_bulk)  # Cu2+ sink, time-averaged over PR cycle
            rhs_c = self.bc_c + rev_src
            if self.adv_bc is not None:                       # open-top inflow, bulk normalised to 1
                rhs_c = rhs_c + self.adv_bc
            ct_new = np.clip(self.cs_c.solve(extra, rhs_c), 0.0, 3.0)
            diag['bal'] = (np.dot(w * ci, ct_new) / ctrl['I_t'] - 1.0) if galvano else 0.0
            return np.concatenate([th_new, s_new, ct_new, phi_new / self.PHI_SC, [V_new / self.V_SC]])

        def sweep_ter(x):
            """One outer Picard pass for transport == 'tertiary':
            theta -> s -> [Cu2+/H+ SG species -> electroneutral sulfate -> local kappa -> (V, phi) with the
            diffusion current -> kinetics] x tertiary_inner."""
            th, s_t, ct, phi, V = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n] * self.PHI_SC, x[4 * n] * self.V_SC
            if not p.ohmic:
                phi = np.zeros(n)
            if not galvano:
                V = ctrl['V_set']
            eta = V - phi
            gu, gs = kin.cath(eta, acc)
            if p.suppressor:
                b_ex = p.k_rd * duty_r * Ia / 1e-3 if ctrl['pr'] else 0.0
                th_new = theta_be(th0, dt, p.k_plus * p.s_bulk * s_t, Mf * ct * gu, Mf * ct * gs,
                                  p.k_minus * duty_f, b_ex)
                extra = w * p.Gamma * p.k_plus * (1 - th_new)
                rhs = self.bc_s.copy()
                if self.adv_bc is not None:                   # bulk normalised: s_bulk -> 1
                    rhs = rhs + self.adv_bc
                if p.s_transient:
                    extra = extra + geo.vh / dt
                    rhs += geo.vh / dt * s_old
                s_new = np.clip(self.cs_s.solve(extra, rhs), 0.0, 1.5)
            else:
                th_new, s_new = th0 * 0.0, s_t
            c_h_old = ter['c_h']
            c_cu, c_h, c_s = ct * p.c_bulk, c_h_old, ter['c_so4']
            phi_w, V_w = phi, V
            for _ in range(max(int(p.tertiary_inner), 1)):
                phi_g = np.zeros(geo.act.shape)
                phi_g[act] = phi_w
                # ---- kinetics at the current eta fixes the Cu2+ interface sink.  The sink is linear in c
                # (i_loc = ci * c_Cu/c_bulk), so the diagonal coefficient is w*ci/(2F c_bulk) and it
                # multiplies the absolute concentration c_Cu -- the same number as the enhanced path's
                # extra, which multiplies ctil.
                gu_w, gs_w = kin.cath(V_w - phi_w, acc)
                ci = (1 - th_new) * gu_w + th_new * gs_w
                sink = p.efficiency * duty_f * w * ci / (2 * F * p.c_bulk)
                A, bc = geo.sg_operator(phi_g, ph.z_cu, self.D_cu_a, ph.f_RT,
                                        chi_max=p.tertiary_chi_max, adv=self.adv)
                self.sg_c.base = A
                rhs_cu = bc * p.c_bulk + rev_src_abs
                if self.adv_bc is not None:                   # absolute concentration: bulk = c_bulk
                    rhs_cu = rhs_cu + self.adv_bc * p.c_bulk
                c_cu = np.maximum(self.sg_c.solve(sink, rhs_cu), self.c_floor)
                Ah, bch = geo.sg_operator(phi_g, ph.z_h, self.D_h_a, ph.f_RT,
                                          chi_max=p.tertiary_chi_max, adv=self.adv)
                self.sg_h.base = Ah
                rhs_h = bch * self.cb[1]
                if self.adv_bc is not None:                   # absolute concentration: bulk = c_h_bulk
                    rhs_h = rhs_h + self.adv_bc * self.cb[1]
                c_h = np.maximum(self.sg_h.solve(np.zeros(n), rhs_h), self.c_floor)
                # ---- electroneutrality closes the sulfate pool (majority inert anion, no flux BC)
                c_s = np.maximum(-(ph.z_cu * c_cu + ph.z_h * c_h) / ph.z_so4, self.c_floor)
                ct_w = np.clip(c_cu / p.c_bulk, 0.0, 3.0)
                if p.ohmic:
                    kp, _ = geo.K_data(self._kappa(c_cu, c_h, c_s))
                    self.cs_p = CachedSym(geo, kp, lu=self.cs_p.lu if self.cs_p is not None else None)
                    src = self._diff_src(c_cu, c_h, c_s)
                    V_w, phi_w, _, nph = newton_potential(geo, self.cs_p, kin, w, ct_w, th_new, acc, V_w, phi_w,
                                                          ctrl['I_t'], galvano, src=src)
                else:
                    V_w, phi_w, _, nph = self._nonohmic(w, ct_w, th_new, acc, V_w, ctrl, galvano)
                diag['nph'] += nph
            ter['c_h'], ter['c_so4'] = c_h, c_s
            ter['d_h'] = float(np.max(np.abs(c_h - c_h_old)) / self.c_h_scale) if n else 0.0
            ct_new = np.clip(c_cu / p.c_bulk, 0.0, 3.0)
            gu_w, gs_w = kin.cath(V_w - phi_w, acc)
            ci = (1 - th_new) * gu_w + th_new * gs_w
            # galvanostatic balance: newton_potential solved sum(w I) = I_t at this ct, so bal is ~0 by
            # construction here; convergence of the tertiary path rests on d_c / d_th (and ter['d_h']).
            diag['bal'] = (np.dot(w * ci, ct_new) / ctrl['I_t'] - 1.0) if galvano else 0.0
            return np.concatenate([th_new, s_new, ct_new, phi_w / self.PHI_SC, [V_w / self.V_SC]])

        sweep = sweep_ter if self.tertiary else sweep_enh

        x = np.concatenate([th0, s_old, st['ctil'][act], (st['phi'][act] if p.ohmic else np.zeros(n)) / self.PHI_SC,
                            [(st['V'] if galvano else ctrl['V_set']) / self.V_SC]])
        hist_f, hist_g = [], []
        f_prev = g_prev = None
        info = dict(it=0, converged=False)
        best = None                                  # (residual norm, gx) of the best sweep so far
        n_restart = 0
        gx = x
        sel = np.concatenate([wm, np.zeros(n, bool), wm, np.zeros(n, bool), [True]])
        for it in range(p.fix_maxit):
            diag['nph'] = 0
            gx = sweep(x)
            f = gx - x
            d_th = np.max(np.abs(f[:n])[wm]) if wm.any() else 0.0
            d_c = np.max(np.abs(f[2 * n:3 * n])[wm]) if wm.any() else 0.0
            if self.tertiary:
                d_c = max(d_c, ter['d_h'])         # H+ is not in the Anderson vector; track its change too
            bal = diag['bal']
            fnorm = np.linalg.norm(f[sel]) / np.sqrt(max(sel.sum(), 1))
            if p.debug_iter:
                print(f'   it{it} bal={bal:+.2e} d_c={d_c:.2e} d_th={d_th:.2e} dV={f[4 * n] * self.V_SC * 1e3:+.3f}mV '
                      f'nph={diag["nph"]} fnorm={fnorm:.2e}')
            info.update(it=it + 1, bal=bal, d_c=d_c, d_th=d_th, nph=diag['nph'])
            if abs(bal) < p.fix_tol_bal and d_c < p.fix_tol_c and d_th < p.fix_tol_theta and it >= 1:
                x = gx
                info['converged'] = True
                break
            # ---- safeguard: a sweep whose residual blew up relative to the best one restarts the Anderson history
            if best is None or fnorm < best[0]:
                best = (fnorm, gx)
            elif fnorm > 5.0 * best[0] and fnorm > 1e-2:
                n_restart += 1
                if n_restart >= 3:
                    break                            # not converging at this dt: let the caller retry with smaller dt
                hist_f, hist_g, f_prev, g_prev = [], [], None, None
                x = best[1]
                continue
            # ---- Anderson mixing
            if f_prev is not None:
                hist_f.append(f - f_prev)
                hist_g.append(gx - g_prev)
                if len(hist_f) > p.aa_depth:
                    hist_f.pop(0)
                    hist_g.pop(0)
            f_prev, g_prev = f, gx
            if p.aa_depth > 0 and hist_f:
                dF = np.stack(hist_f, axis=1)
                gam, *_ = np.linalg.lstsq(dF, f, rcond=1e-8)
                x = gx - np.stack(hist_g, axis=1) @ gam
                x[:n] = np.clip(x[:n], 0.0, 1.0)
                x[n:3 * n] = np.clip(x[n:3 * n], 0.0, None)
            else:
                x = gx
        if not info['converged'] and best is not None and it + 1 >= p.fix_maxit - 0:
            pass
        th, s_t, ct = x[:n], x[n:2 * n], x[2 * n:3 * n]
        phi = x[3 * n:4 * n] * self.PHI_SC
        V = x[4 * n] * self.V_SC
        if not galvano:
            V = ctrl['V_set']
        # ---- final polish of V at fixed phi, c so the currents used for the front motion balance exactly
        if galvano:
            V = scalar_V_cathodic(kin, w, ct, th, acc, phi, ctrl['I_t'], V0=V)
        eta = V - phi
        gu, gs = kin.cath(eta, acc)
        ci = (1 - th) * gu + th * gs
        i_f = ct * ci                                      # forward cathodic current density per cell (A/cm2)
        info['I_f'] = float(np.dot(w, i_f))
        st['V'] = V
        if ctrl['pr']:
            st['V_r'] = V_r
            st['phi_r'][act] = phi_r
        st['theta'][act] = th
        st['s'][act] = s_t
        st['ctil'][act] = ct
        if self.tertiary:
            st['c_h'][act] = ter['c_h']
            st['c_so4'][act] = ter['c_so4']
        if p.ohmic:
            st['phi'][act] = phi
        info['n_fac'] = (self.cs_c.n_fac + self.cs_s.n_fac + (self.cs_p.n_fac if self.cs_p else 0)
                         + (self.sg_h.n_fac if self.tertiary else 0))
        return dict(act=act, i_f=i_f, i_a=Ia, th=th, th0=th0, ct=ct, eta=eta, phi=phi, s_t=s_t, gu=gu, gs=gs,
                    c_h=ter['c_h'] if self.tertiary else None,
                    c_so4=ter['c_so4'] if self.tertiary else None, info=info)

    # ---------------------------------------------------------------- phi = 0 helpers
    def _nonohmic(self, w, ct, th, acc, V, ctrl, galvano):
        n = self.geo.n
        phi = np.zeros(n)
        if galvano:
            V = scalar_V_cathodic(self.kin, w, ct, th, acc, phi, ctrl['I_t'], V0=V)
        else:
            V = ctrl['V_set']
        gu, gs = self.kin.cath(V - phi, acc)
        return V, phi, ct * ((1 - th) * gu + th * gs), 0

    def _rev_nonohmic(self, w, I_ta):
        ph, p = self.ph, self.p
        V_r = np.log(I_ta / (ph.i0_a * w.sum() + 1e-300)) / (p.alpha_a * self.kin.f)
        return V_r, np.zeros(self.geo.n), None, 0
