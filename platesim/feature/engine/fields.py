"""Finite-volume field solvers for one time step (frozen geometry).

Two interface representations, selected by Params.interface (plan v2 C4):

'diffuse' (default, bit-identical to the pre-C4 engine) -- smoothed-boundary form
(Hs = 0.5(1+tanh(psi/eps)), psi > 0 electrolyte):
    div(Hs K grad u) = |grad Hs| * (K du/dn at the interface)
 so with a consumption flux N at the interface   div(Hs D grad c) = |grad Hs| N.
Discretisation: cell-centred finite volume (axisymmetric via: face area ~ r; planar trench per unit length:
weight 1 - both cases enter only through Grid.vol / Grid.AE / Grid.AZ), symmetric M-matrix K with
  K u = -(sink), solved on cells with psi > -n_active_eps*eps only, Dirichlet (bulk) value on the delta plane.
Interface weights  w_p = V_p |grad Hs|_p  [cm2]  carry all interface fluxes:  total interface current = sum w_p i_p.

'sharp' -- cut-cell interface built from the same level set (plan v2 C4; see StepGeom):
cell electrolyte fractions ef in the storage, cut-cell face apertures in place of Hs on the faces, the
interface area of a cut cell (the chord of the linear level-set reconstruction) as w_p on the sink node,
and the interface rate rejected to the ghost-fluid distance |psi| of that node (k_eff = k/(1+k|psi|/D)).
In 1-D that is exact for the steady planar limiting current at any front position inside a cell, where the
tanh band is first-order biased (the band leaks the sink into the cells above the front).  Small cut cells
(ef < EF_FLOOR) lose their sink to the majority neighbour and keep a floored storage.

Fields per step (fixed-point loop): theta (implicit backward Euler per cell), suppressor s, potential phi with
the metal potential V (Newton, galvanostatic constraint bordered), Cu2+ concentration c (implicit sink)."""
from __future__ import annotations
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.ndimage as ndi
from scipy.optimize import brentq

from .levelset import Grid, grad_mag, label_regions, curvature_div_n, electrolyte_fraction
from .params import F, OMEGA_CU_M3_MOL
from .additives import (AdditiveSet, displace_suppressor, suppressor_cl_rate)


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


def sg_face_coeffs(g, F):
    """Exponentially fitted face coefficients (g B(-P), g B(P)) with P = F/g, for the face flux

        N_pq = gp * c_p - gq * c_q,      gp = g B(-P),  gq = g B(P).

    g is the same diffusive face conductance K_data uses (Hf * D_face * geo), F the signed volume flux
    oriented p -> q.  B(x) = x/(exp x - 1) (B(0) = 1):  P -> 0 gives the plain diffusive flux g(c_p - c_q),
    P >> 1 gives the upwind flux F c_p, and the row/face sums  gp - gq = g P = F  hold exactly, so a
    uniform field has zero flux divergence for a discretely divergence-free F.

    B(-P) is evaluated through the identity B(-P) = B(P) + P (one exponential per face, as in
    sg_operator).  bern() clips its argument to +-50, so B(P) for P -> -inf is replaced by its asymptote
    -P (the branch where B(-P) -> 0 and the face is fully upwind from q); without that the clipped
    branch would drive gp = g B(-P) negative for P < -50 and destroy the M-matrix.

    Faces with g == 0 but F != 0 cannot occur on the Stokes path (a MAC face that touches solid carries
    F = 0); they fall back to pure upwind here instead of dividing by zero."""
    g = np.asarray(g, dtype=float)
    F = np.asarray(F, dtype=float)
    safe = np.where(g > 0.0, g, 1.0)
    P = F / safe
    bp = bern(P)                                        # B(P)
    neg = P < -40.0
    if np.any(neg):
        # B(P) = -P / (1 - exp(P))  ->  -P (1 + exp(P)) for P << 0; bern() would clip this to B(-50).
        bp = np.where(neg, -P * (1.0 + np.exp(np.maximum(P, -700.0))), bp)
    gp = g * bp + F                                     # g B(-P) = g (B(P) + P)
    gq = g * bp                                         # g B(P)
    dead = ~(g > 0.0)
    if np.any(dead):
        gp = np.where(dead, np.maximum(F, 0.0), gp)     # upwind fallback
        gq = np.where(dead, np.maximum(-F, 0.0), gq)
    return gp, gq


# ======================================================================================= TVD (MUSCL / van Leer)
def vanleer(r):
    """van Leer flux limiter  psi(r) = (r + |r|) / (1 + |r|)  (the harmonic-mean form).

    psi(r) = 0 for r <= 0 (no anti-diffusive extrapolation across an extremum) and psi(r) = 2r/(1+r)
    for r > 0, rising monotonically to 2 as r -> inf.  Used with the MOVERS/MUSCL reconstruction
    c_face = c_u + 0.5 psi(r) (c_d - c_u), which is TVD for any such Sweby-region limiter."""
    r = np.asarray(r, dtype=float)
    ar = np.abs(r)
    return (r + ar) / (1.0 + ar)


def muscl_face(cu, cd, cuu, ok=None):
    """Limited second-order (MUSCL) face value from the upwind side, vectorised over faces.

    For a face whose signed flux runs upwind cell u -> downwind cell d, with uu the next cell further
    upstream (all uniform spacing):
        r      = (c_u - c_uu) / (c_d - c_u)
        c_face = c_u + 0.5 * vanleer(r) * (c_d - c_u)
    r ~ 1 on a smooth monotone field, so psi(r) ~ 1 and the face value is centred (second order);
    psi -> 0 at extrema/steps, which collapses it to the first-order upwind value c_u (TVD).

    ok (bool array, optional) marks the faces that have a valid far-upwind cell; the others fall back
    to c_u (solid neighbour, domain end, the top Dirichlet plane).  A locally uniform field (c_d ==
    c_u) returns c_u exactly, so a constant field never generates a correction."""
    cu = np.asarray(cu, dtype=float)
    cd = np.asarray(cd, dtype=float)
    cuu = np.asarray(cuu, dtype=float)
    dc = cd - cu
    live = dc != 0.0
    den = np.where(live, dc, 1.0)
    r = (cu - cuu) / den
    psi = vanleer(np.where(live, r, 0.0))
    ho = cu + 0.5 * psi * dc
    if ok is not None:
        ho = np.where(np.asarray(ok, dtype=bool), ho, cu)
    return np.where(live, ho, cu)


# ======================================================================================= sharp interface (C4)
# Below ~5 % electrolyte a cut cell is no longer a useful unknown: its storage would be a needle and its
# sink belongs to the neighbour that holds the bulk of the control volume (see StepGeom._sharp_weights).
EF_FLOOR = 0.05


def _seg_pos_frac(a, b):
    """Fraction of the unit segment [a, b] on which the linear interpolant from a to b is > 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mix = (a > 0.0) != (b > 0.0)
    t = np.where(mix, a / np.where(mix, a - b, 1.0), 1.0)
    return np.where((a <= 0.0) & (b <= 0.0), 0.0,
                    np.where(mix, np.where(a > 0.0, t, 1.0 - t), 1.0))


def _grad_comp(psi, dx, periodic):
    """Central-difference gradient components of psi: the (gr, gz) that levelset.grad_mag hypot's."""
    p = np.pad(psi, 1, mode='edge')
    if periodic:
        p[0, 1:-1] = psi[-1, :]
        p[-1, 1:-1] = psi[0, :]
    return ((p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx),
            (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx))


def _face_apertures(psi, g: Grid):
    """Open (electrolyte) fraction of every grid face -- cut-cell apertures, plan v2 C4.

    A face between two cells is a segment of length dx.  psi at its two ends is the bilinear mean of the
    four cell values around that corner, psi is interpolated linearly along the segment and the aperture
    is the length fraction with psi > 0 (psi > 0 = electrolyte).  The estimate is exact for a straight
    interface -- in particular 1 for every face that lies wholly on the electrolyte side and 0 for one
    wholly in the metal, which is what the smoothed band gets wrong for a face that is adjacent to a
    mostly-solid cut cell -- and monotone in psi, so a_k in [0, 1].

    Returns (ax, az):
      ax[i, j]  x-face between (i, j) and (i+1, j)  (i = nr-1 is the periodic seam when g.periodic_x)
      az[i, j]  z-face between (i, j) and (i, j+1)
    The stencil picks ax[ir, jr], az[iz, jz] and ax[nr-1, jw]; degenerate edge entries (i = nr-1 or
    j = nz-1 on a non-periodic grid) are never used."""
    per = g.periodic_x

    def up(a):
        r = np.empty_like(a)
        r[:, :-1] = a[:, 1:]
        r[:, -1] = a[:, -1]
        return r

    def dn(a):
        r = np.empty_like(a)
        r[:, 1:] = a[:, :-1]
        r[:, 0] = a[:, 0]
        return r

    def nxt(a):
        if per:
            return np.roll(a, -1, axis=0)
        r = np.empty_like(a)
        r[:-1] = a[1:]
        r[-1] = a[-1]
        return r

    def prv(a):
        if per:
            return np.roll(a, 1, axis=0)
        r = np.empty_like(a)
        r[1:] = a[:-1]
        r[0] = a[0]
        return r

    pu, pn = up(psi), nxt(psi)
    lo = 0.25 * (psi + pn + dn(psi) + dn(pn))          # corner (i+1/2, j-1/2)
    hi = 0.25 * (psi + pn + pu + up(pn))               # corner (i+1/2, j+1/2)
    ax = _seg_pos_frac(lo, hi)
    le = 0.25 * (psi + pu + prv(psi) + up(prv(psi)))   # corner (i-1/2, j+1/2)
    ri = 0.25 * (psi + pu + pn + up(pn))               # corner (i+1/2, j+1/2)
    az = _seg_pos_frac(le, ri)
    return ax, az


def _iface_cells(psi, g: Grid, gm=None):
    """Interface area and sink node of every cut cell -- cut-cell geometry, plan v2 C4.

    Interface area.  Inside a cell psi is replaced by its linear reconstruction
        psi(x) = psi_p + (x - x_p) . grad psi,
    i.e. the psi = 0 line sits at signed distance delta = psi_p/|grad psi| from the centre with unit
    normal n = grad psi/|grad psi|.  The interface area of the cell is the length of that line's chord
    through the square [x_p +- dx/2] x [z_p +- dx/2] (the closing segment of the cut polygon).  The
    chord is exact for a straight interface (so the total interface area of a planar front is exactly
    the front length, and of a 45 deg line the sqrt(2) dx of the diagonal), 2nd order for a curved one,
    and 0 when the line misses the cell.  It is scaled to the cell's projected measure (1 per unit
    length for the planar geometries, 2 pi r_c for the axisymmetric 'via'), matching the units of the
    legacy  w = V |grad Hs|  so that sum(w) is the interface area in every geometry.

    Sink node.  The sink of a cut cell lives at the node of the cell whose centre is in the
    ELECTROLYTE: the cell itself when ef >= 0.5 (see levelset.electrolyte_fraction), otherwise the
    neighbour along +grad psi (the outward normal, i.e. the 'merge the small sliver into the larger
    neighbour along the normal' rule).  That keeps every sink |psi_node| <= ~dx from the interface
    instead of up to dx/2 inside the metal, and it is what makes the 1-D chain exact (see
    StepSolver._ghost).  Returns (chord, ti, tj):
      chord [cm]      interface area per cell (>= 0, 0 = the cell is not cut)
      ti, tj (int)    sink-node indices per cell (-1 where the cell is not cut); the caller validates
                      them against the active set and falls back to the cell itself."""
    nr, nz = psi.shape
    dx = g.dx
    gr, gz = _grad_comp(psi, dx, g.periodic_x)
    gm = np.hypot(gr, gz) if gm is None else np.asarray(gm, dtype=float)
    live = gm > 1e-8                                    # 1/cm; |grad psi| ~ 1 for a distance function
    gs = np.where(live, gm, 1.0)
    nx = np.where(live, gr / gs, 0.0)
    nz_ = np.where(live, gz / gs, 0.0)
    delta = np.where(live, psi / gs, 0.0)
    h = 0.5 * dx

    def bounds(c, s):
        """t-interval of |c + s t| <= h (empty -> lo > hi; unbounded -> +-inf)."""
        tiny = np.abs(s) < 1e-12
        ss = np.where(tiny, 1.0, s)
        a, b = (-h - c) / ss, (h - c) / ss
        inside = np.abs(c) <= h + 1e-9 * dx      # tolerance: a front exactly ON the face is a valid case
        lo = np.where(tiny, np.where(inside, -np.inf, np.inf), np.minimum(a, b))
        hi = np.where(tiny, np.where(inside, np.inf, -np.inf), np.maximum(a, b))
        return lo, hi

    # the chord runs along t_hat = (n_z, -n_x): x = delta n_x + t n_z, z = delta n_z - t n_x
    with np.errstate(invalid='ignore'):
        lo1, hi1 = bounds(delta * nx, nz_)
        lo2, hi2 = bounds(delta * nz_, -nx)
        chord = np.maximum(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0.0)
    chord = np.where(live, np.clip(chord, 0.0, np.sqrt(2.0) * dx), 0.0)
    # A line that lies (numerically) ON a cell boundary is reconstructed by both cells that share the
    # face, so a grid-aligned front would be counted twice.  Keep it in the electrolyte-side cell only
    # (psi >= 0); equality also covers a line through a corner, where the chord is ~0 anyway.
    on_face = np.abs(delta) >= h * (np.abs(nx) + np.abs(nz_)) - 1e-9 * dx
    chord = np.where(on_face & (psi < 0.0), 0.0, chord)
    chord = np.where(chord < 1e-9 * dx, 0.0, chord)
    chord = chord * (g.vol[:, 0] / dx ** 2)[:, None]
    ef = electrolyte_fraction(psi, g)
    ti = np.full((nr, nz), -1, np.int64)
    tj = np.full((nr, nz), -1, np.int64)
    ii, jj = np.nonzero(chord > 0.0)
    if ii.size:
        own = ef[ii, jj] >= 0.5
        cx = np.abs(nx[ii, jj]) >= np.abs(nz_[ii, jj])   # step along the dominant normal component
        si = np.where(cx, np.sign(nx[ii, jj]), 0.0).astype(np.int64)
        sj = np.where(cx, 0.0, np.sign(nz_[ii, jj])).astype(np.int64)
        ti[ii, jj] = np.where(own, ii, ii + si)
        tj[ii, jj] = np.where(own, jj, jj + sj)
    return chord, ti, tj


# ======================================================================================= geometry for a step
class StepGeom:
    """Everything that depends only on psi at the start of a step: smoothed indicator (or the sharp
    cut-cell fractions), active cells, interface weights and the sparse stencil (pattern) shared by all
    fields.

    Params.interface selects the interface representation:
      'diffuse' (default, bit-identical to the pre-C4 engine): Hs = 0.5(1+tanh(psi/eps)) on cells and
          faces, interface weights w_p = V_p |grad Hs|_p.
      'sharp' (plan v2 C4): cell electrolyte fractions ef (levelset.electrolyte_fraction) replace Hs in
          the storage, cut-cell face apertures replace Hs on the faces (Hf_*), and w_p is the interface
          area of the cut cell, placed on the sink node returned by _iface_cells.  Small cut cells
          (ef < EF_FLOOR = 0.05) keep their storage floored at EF_FLOOR * V and their sink in the
          majority neighbour, so no row of the operator loses its diagonal dominance; the sink area
          itself is conserved (sum w is the interface area in both representations).  The interface
          fluxes (interface current, the Cu2+ sink of the Cu balance, the level-set speed) all use w,
          and StepSolver rejects the sink to the ghost-fluid distance |psi_node| (k_eff = k/(1+k d/D)),
          which is exact for the 1-D planar steady limiting current.  Not applied: the suppressor /
          leveller adsorption rates keep the plain interface area (documented in StepSolver)."""

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
        # ---- plan v2 C4: 'sharp' cut-cell interface (None / False on the legacy 'diffuse' path)
        self.sharp = str(getattr(p, "interface", "diffuse")) == "sharp"
        self.ef = None                     # cell electrolyte volume fraction (sharp only)
        self.d_if = None                   # |psi| of every sink node (ghost-fluid distance, sharp only)
        if self.sharp:
            self.ef = electrolyte_fraction(psi, g)
            self.ape_x, self.ape_z = _face_apertures(psi, g)
            self._chord, self._ti, self._tj = _iface_cells(psi, g, gm)
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
        if self.sharp:
            # a cell with no electrolyte by the level set has no capacity and no conductance: drop it.
            # (For the flat band this is exactly the psi <= -dx/2 tail the tanh band still carries.)
            act = act & (self.ef > 0.0)
        # keep only the connected piece that touches the Dirichlet row (guards against singular islands)
        lab, n = ndi.label(act)
        if n > 1:
            top = np.unique(lab[:, jt])
            act = np.isin(lab, top[top > 0])
        if self.sharp:
            # A cut cell whose every face is closed (or opens only onto a cell that has just been dropped)
            # has no conductance to anywhere: its row of the operator would be empty and the LU singular.
            # Such a sliver is a sub-grid electrolyte pocket that the level set keeps at ef ~ 0.1 inside the
            # metal; it is dropped (its interface area then vanishes -- the galvanostatic constraint
            # renormalises over the remaining w, so the Cu balance is unaffected).  Every cell on the
            # Dirichlet row keeps its plane coupling and stays.
            ax_, az_ = self.ape_x, self.ape_z
            keep0 = np.zeros_like(act)
            keep0[:, jt] = True
            for _ in range(g.nr + g.nz):
                nb = np.zeros_like(act)
                nb[:-1] |= act[1:] & (ax_[:-1] > 0.0)
                nb[1:] |= act[:-1] & (ax_[:-1] > 0.0)
                nb[:, :-1] |= act[:, 1:] & (az_[:, :-1] > 0.0)
                nb[:, 1:] |= act[:, :-1] & (az_[:, :-1] > 0.0)
                if g.periodic_x:
                    nb[0] |= act[-1] & (ax_[-1] > 0.0)
                    nb[-1] |= act[0] & (ax_[-1] > 0.0)
                keep = act & (nb | keep0)
                if keep.sum() == act.sum():
                    break
                act = keep
        self.open_m, self.void_m, self.act = open_m, void_m, act
        n = int(act.sum())
        self.n = n
        idx = -np.ones(act.shape, np.int64)
        idx[act] = np.arange(n)
        self.idx = idx
        # ---- interface weights (zero for inert surfaces in test mode)
        if self.sharp:
            w, d_if = self._sharp_weights(act)
            self.d_if = d_if
        else:
            w = g.vol * self.gHs
        w = np.where(act, w, 0.0)
        self.w_full = w
        self.w = w[act]
        # ---- stencil
        ar = act[:-1, :] & act[1:, :]
        self.ir, self.jr = np.nonzero(ar)
        self.pr, self.qr = idx[self.ir, self.jr], idx[self.ir + 1, self.jr]
        self.geo_r = g.AE[self.ir, 0] / dx
        self.Hf_r = (self.ape_x[self.ir, self.jr] if self.sharp
                     else 0.5 * (self.Hs[self.ir, self.jr] + self.Hs[self.ir + 1, self.jr]))
        az = act[:, :-1] & act[:, 1:]
        self.iz, self.jz = np.nonzero(az)
        self.pz, self.qz = idx[self.iz, self.jz], idx[self.iz, self.jz + 1]
        self.geo_z = g.AZ[self.iz, 0] / dx
        self.Hf_z = (self.ape_z[self.iz, self.jz] if self.sharp
                     else 0.5 * (self.Hs[self.iz, self.jz] + self.Hs[self.iz, self.jz + 1]))
        # periodic x seam: cell 0 (x ~ 0+) and cell nr-1 (x ~ pitch-) are neighbours.  p = column 0,
        # q = column nr-1, so the p -> q face normal points in -x (used for the advective flux sign).
        if g.periodic_x:
            aw = act[-1, :] & act[0, :]
            self.jw = np.nonzero(aw)[0]
            self.pw = idx[0, self.jw]
            self.qw = idx[g.nr - 1, self.jw]
            self.geo_w = g.AE[0, 0] / dx
            self.Hf_w = (self.ape_x[g.nr - 1, self.jw] if self.sharp
                         else 0.5 * (self.Hs[0, self.jw] + self.Hs[g.nr - 1, self.jw]))
        else:
            self.jw = np.empty(0, np.int64)
            self.pw = self.qw = np.empty(0, np.int64)
            self.geo_w = 1.0
            self.Hf_w = np.empty(0)
        self.itop = np.nonzero(act[:, jt])[0]
        self.ptop = idx[self.itop, jt]
        self.geo_top = g.AZ[self.itop, 0] / self.d_top
        if self.sharp:
            # Dirichlet-plane coupling: the aperture of the z-face above the top cell (1 when that face
            # does not exist, i.e. jt is the last row)
            self.Hs_top = (self.ape_z[self.itop, jt] if jt < g.nz - 1
                           else np.ones(self.itop.size, float))
        else:
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
        if self.sharp:
            # electrolyte storage: V_p ef_p, with the tiny-cell floor (see EF_FLOOR / _sharp_weights)
            self.vh = (g.vol * np.where(self.ef > 0.0, np.maximum(self.ef, EF_FLOOR), 0.0))[act]

    # ------------------------------------------------------------------ sharp (cut-cell) interface weights
    def _sharp_weights(self, act):
        """Interface area per cell and the ghost-fluid distance of every sink node (plan v2 C4).

        w_p is the chord length of the level-set reconstruction inside cell p (_iface_cells), moved onto
        the sink node: the cell itself when the centre is in the electrolyte (ef >= 0.5), otherwise the
        neighbour along +grad psi.  Small cells (ef < EF_FLOOR) therefore carry neither the sink nor
        (beyond the floor) the storage, which keeps the operator an M-matrix with positive diagonal:
        every face conductance is aperture * geo * D >= 0 and every w_p >= 0.
        The total interface area is conserved by the merge (sum w == sum of the cut-cell chords), so the
        current sum(w i) and the Cu balance are unaffected by the small-cell rule.
        d_if is |psi| of the sink node when that node is on the ELECTROLYTE side (psi >= 0), i.e. the
        distance from the node to the interface, and 0 otherwise: a node on the metal side (only possible
        when the majority neighbour is itself inactive, a degenerate sliver) already sits beyond the
        interface, so its ghost correction would double-count the sub-cell distance.  0 disables the
        correction there and recovers the plain interface rate."""
        chord, ti, tj = self._chord, self._ti, self._tj
        nr, nz = chord.shape
        ii, jj = np.nonzero(chord > 0.0)
        w = np.zeros((nr, nz))
        d_if = np.zeros((nr, nz))
        if ii.size == 0:
            return w, d_if
        ti_v, tj_v = ti[ii, jj], tj[ii, jj]
        inb = (ti_v >= 0) & (ti_v < nr) & (tj_v >= 0) & (tj_v < nz)
        tgt_ok = np.zeros(ii.size, bool)
        tgt_ok[inb] = act[ti_v[inb], tj_v[inb]]
        own_ok = act[ii, jj]
        keep = tgt_ok | own_ok
        ti2 = np.where(tgt_ok, ti_v, ii)[keep]
        tj2 = np.where(tgt_ok, tj_v, jj)[keep]
        np.add.at(w, (ti2, tj2), chord[ii[keep], jj[keep]])
        d_if[ti2, tj2] = np.maximum(self.psi[ti2, tj2], 0.0)
        return w, d_if

    # ------------------------------------------------------------------ matrix assembly
    def K_data(self, D):
        """Data vector (CSR order) of the symmetric operator  -div(a K grad .)  incl. Dirichlet at the top plane,
        and the Dirichlet coupling vector bc (n,) so that the bulk value u0 enters the rhs as bc*u0.

        a is Hs (the smoothed band, 'diffuse') or the cut-cell face aperture ('sharp'), so the same assembly
        serves both interface representations."""
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
        where geo = 1) but WITHOUT the Hs face weight (the Stokes field is sharp).  With interface='sharp'
        the aperture is deliberately NOT applied either: the MAC flux is already the face-normal velocity
        of the sharp flow field (which is zero on faces that touch solid), and scaling it by an aperture
        < 1 would destroy the discrete divergence-freeness that the conservative advective operator
        (adv_matrix form='divergence', mac_advection) relies on to keep a uniform field uniform and the
        Cu2+ balance closed.  The wrap face p -> q
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
        diag, rhs = self.top_advection(F_top)
        if np.any(diag):
            adv = adv + sp.diags(diag)
        return adv, rhs, F_top

    def top_advection(self, F_top):
        """Open-top advective boundary terms of the Dirichlet plane: (diag, rhs).

        Only the plane is treated this way -- it is a modelling boundary (the delta layer), not a resolved
        face of the species stencil, so it keeps the upwind face value: an outflow cell (F_top > 0) loses
        F_top c_P (diagonal), an inflow cell (F_top < 0) gains |F_top| c_bulk (rhs, caller scales the bulk
        value).  Shared by mac_advection (upwind scheme) and the exponentially fitted operator, which
        replaces the INTERIOR faces only."""
        diag = np.zeros(self.n)
        rhs = np.zeros(self.n)
        if F_top.size:
            out = F_top > 0
            diag[self.ptop[out]] = F_top[out]
            rhs[self.ptop[~out]] = -F_top[~out]
        return diag, rhs

    def sg_advdiff(self, D, F_r, F_z, F_w):
        """Exponentially fitted (Scharfetter-Gummel) diffusion+advection operator and its Dirichlet vector.

        Every interior stencil face carries the combined flux  N_pq = g B(-P) c_p - g B(P) c_q  with
        P = F/g and g the K_data diffusive conductance (arithmetic face diffusivity, same face measure),
        instead of the sum of a separate diffusive conductance g and an upwind advective flux F.  The top
        Dirichlet plane is unchanged (diffusive coupling bc folded into the diagonal and returned as the
        rhs vector bc*c_bulk, exactly as K_data); the caller still adds StepGeom.top_advection(F_top).

        Row sums: with c = 1 on every face the interior part contributes exactly F per face (gp - gq =
        g P = F), i.e. the net flux divergence, which is 0 for a discretely divergence-free MAC field, so
        a uniform field is preserved exactly (row sum at a top cell = -F_top, cancelled by top_advection).
        Returns (A, bc) with A CSR including diag(bc)."""
        D = np.asarray(D, dtype=float)
        if D.ndim == 0:
            D = np.full(self.act.shape, float(D))
        g = self.g
        n = self.n
        Gr = self.Hf_r * 0.5 * (D[self.ir, self.jr] + D[self.ir + 1, self.jr]) * self.geo_r
        Gz = self.Hf_z * 0.5 * (D[self.iz, self.jz] + D[self.iz, self.jz + 1]) * self.geo_z
        rows, cols, vals = [], [], []
        for p, q, gf, F in ((self.pr, self.qr, Gr, F_r), (self.pz, self.qz, Gz, F_z)):
            if p.size == 0:
                continue
            Np, Nq = sg_face_coeffs(gf, F)
            rows.extend([p, p, q, q])
            cols.extend([p, q, p, q])
            vals.extend([Np, -Nq, -Np, Nq])
        if g.periodic_x and self.jw.size:
            jw = self.jw
            Gw = self.Hf_w * 0.5 * (D[0, jw] + D[g.nr - 1, jw]) * self.geo_w
            Np, Nq = sg_face_coeffs(Gw, F_w)
            rows.extend([self.pw, self.pw, self.qw, self.qw])
            cols.extend([self.pw, self.qw, self.pw, self.qw])
            vals.extend([Np, -Nq, -Np, Nq])
        if vals:
            A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, n)).tocsr()
        else:
            A = sp.csr_matrix((n, n))
        bc = np.zeros(n)
        bc[self.ptop] = self.Hs_top * D[self.itop, self.jt] * self.geo_top
        return (A + sp.diags(bc)).tocsr(), bc

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

    # ------------------------------------------------------------------ TVD (MUSCL) deferred correction
    def tvd_topology(self):
        """Cached per-face (upwind, downwind, far-upwind) active-cell ids for the MUSCL reconstruction.

        One tuple per stencil face group, in the (F_r, F_z, F_w) order of mac_face_fluxes:
            (p, q, up_pos, dn_pos, uu_pos, up_neg, dn_neg, uu_neg)
        where the *_pos entries are the upwind / downwind / upwind-upwind cell ids used on faces with
        F > 0 and *_neg those for F < 0.  The far-upwind id uu is -1 when that cell is inactive (solid),
        off the grid, or past the top Dirichlet plane, in which case muscl_face falls back to first
        order.  The far-upwind cell across the periodic x seam wraps (columns nr-1 / 0 / 1 / nr-2);
        z is not periodic, so the bottom/top ends fall back.  Built once per StepGeom."""
        topo = getattr(self, "_tvd_topo", None)
        if topo is not None:
            return topo
        idx, g = self.idx, self.g
        nr, nz = g.nr, g.nz
        wx = g.periodic_x

        def cell(i, j):
            if j < 0 or j >= nz:
                return -1
            if i < 0 or i >= nr:
                if not wx:
                    return -1
                i %= nr
            return int(idx[i, j])

        def ids(i_arr, j_arr):
            return np.array([cell(i, j) for i, j in zip(i_arr, j_arr)], dtype=np.int64)

        pr, qr = np.asarray(self.pr), np.asarray(self.qr)
        pz, qz = np.asarray(self.pz), np.asarray(self.qz)
        pw, qw = np.asarray(self.pw), np.asarray(self.qw)
        # r-faces p -> q point +x: F>0 upwind p, far-upwind (ir-1); F<0 upwind q, far-upwind (ir+2)
        uu_r_pos = ids(self.ir - 1, self.jr)
        uu_r_neg = ids(self.ir + 2, self.jr)
        # z-faces p -> q point +z: F>0 far-upwind (iz, jz-1); F<0 far-upwind (iz, jz+2)
        uu_z_pos = ids(self.iz, self.jz - 1)
        uu_z_neg = ids(self.iz, self.jz + 2)
        # wrap face p=(0,j) -> q=(nr-1,j) points -x: F>0 far-upwind column 1; F<0 column nr-2
        jw = np.asarray(self.jw)
        uu_w_pos = ids(np.ones_like(jw), jw) if nr >= 2 else np.full(jw.size, -1, np.int64)
        uu_w_neg = ids(np.full(jw.size, nr - 2, np.int64), jw) if nr >= 3 else np.full(jw.size, -1, np.int64)
        topo = (
            (pr, qr, pr, qr, uu_r_pos, qr, pr, uu_r_neg),
            (pz, qz, pz, qz, uu_z_pos, qz, pz, uu_z_neg),
            (pw, qw, pw, qw, uu_w_pos, qw, pw, uu_w_neg),
        )
        self._tvd_topo = topo
        return topo

    def tvd_correction(self, c, F_r, F_z, F_w):
        """Deferred-correction vector of the limited (MUSCL / van Leer) advective face fluxes.

        Returns  D(c) = sum over faces of  F (c_face_limited - c_face_upwind)  accumulated with + on p
        and - on q, i.e. the flux divergence of the extra anti-diffusive flux, so that the second-order
        scheme is  A_upwind c + D(c) = rhs  and a Picard sweep freed with  rhs - D(c^k)  converges to it.
        The linear operator (and therefore the CachedGen reuse and the classic upwind matrix) is
        unchanged.  Each face enters p and q with opposite signs, so sum(D) == 0 exactly (up to the
        summation round-off) and a uniform field gives D == 0.  The top Dirichlet-plane terms of
        mac_advection are untouched: only the interior stencil faces reconstructed here get the
        correction."""
        c = np.asarray(c, dtype=float)
        corr = np.zeros(self.n)
        for (p, q, up_p, dn_p, uu_p, up_n, dn_n, uu_n), F in zip(self.tvd_topology(),
                                                                 (F_r, F_z, F_w)):
            if F.size == 0:
                continue
            assert p.size == q.size == up_p.size == up_n.size == F.size == uu_p.size == uu_n.size
            pos = F > 0
            u = np.where(pos, up_p, up_n)
            d = np.where(pos, dn_p, dn_n)
            uu = np.where(pos, uu_p, uu_n)
            ho = muscl_face(c[u], c[d], c[np.maximum(uu, 0)], uu >= 0)
            extra = F * (ho - c[u])
            corr += np.bincount(p, extra, self.n) - np.bincount(q, extra, self.n)
        return corr

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


# ======================================================================================= bulk speciation
def _speciation_mod():
    """platesim.core.speciation, imported lazily.

    That module imports feature.engine.params at module scope; importing it from fields at import time
    would close the cycle  core.speciation -> feature.engine.params -> platesim.feature.__init__ ->
    engine.run_case -> sim -> fields -> core.speciation.  Deferring keeps the legacy import graph intact.
    """
    from ...core import speciation as mod
    return mod


@lru_cache(maxsize=16)
def _bulk_speciation(c_cu_mol_L, c_acid_mol_L, T_C, Ka2, logK_ip, activity):
    """Bulk HSO4-/CuSO4^0 speciation of the bath (plan v2 D1b), cached across StepSolver rebuilds.

    Trend-level only: the Davies activity model is outside its validity range at I ~ 0.8 mol/L and
    log10 K_ip = 2.3 is not verified (see platesim/core/speciation.py).  The StepSolver is rebuilt
    every time step, so caching keeps this out of the time loop."""
    return _speciation_mod().speciate(c_cu_mol_L, c_acid_mol_L, T_C=T_C, Ka2=Ka2,
                                      K_ip=10.0 ** float(logK_ip), activity=activity)


def bulk_speciation(p, ph):
    """Bulk speciation + speciated conductivity of the bath for Params p at Phys temperature ph.T_C.

    Returns dict(spec, kappa [S/cm], kappa_ratio, r_cu, r_so4, r_react).  ``kappa_ratio`` is
    kappa_speciated / kappa_fully_dissociated at the bulk point, computed with the ENGINE diffusivities
    (p.D_cu25 / D_H25 / D_SO425 + the Stokes-Einstein T factor), so the ratio is directly the factor by
    which the tertiary path's local Nernst-Einstein kappa must be rescaled.

    Model error [J]: Davies at I ~ 0.8 mol/L is an extrapolation and log10 K_ip = 2.3 is not verified,
    so kappa and the free fraction are TREND-LEVEL.  Use activity='ideal'/'bdot' to bracket them.

    Approximation used by the engine (plan v2 D1b): the transported Cu2+ field stays TOTAL copper and
    the H+/sulfate pools stay lumped, fully dissociated; the bulk free-ion ratios are applied everywhere
    (frozen speciation).  Local re-equilibration per Picard sweep is NOT implemented.

    TODO(D1b): recompute the speciation from the local (c_Cu_total, c_H_total, c_SO4_total) inside every
    sweep_ter inner iteration and feed back the local free fraction/gamma to the sink, the diffusion
    current and _kappa.  That is the stretch goal; until then the bulk ratio is used everywhere.
    """
    c_cu_L = p.c_bulk * 1e3                        # mol/L
    c_acid_L = 0.5 * ph.c_h_bulk * 1e3             # C_H = 2 C_acid
    D25 = {"cu": p.D_cu25, "h": p.D_H25, "so4": p.D_SO425}   # 25 C; conductivity applies the SE T factor
    spec = _bulk_speciation(c_cu_L, c_acid_L, float(ph.T_C),
                            None if p.Ka2 is None else float(p.Ka2),
                            float(p.logK_ip), str(p.activity))
    if not spec["converged"]:
        raise RuntimeError(f"bath speciation did not converge: residual={spec['residual']:.3e} "
                           f"(T={ph.T_C} C, activity={p.activity}, log10 K_ip={p.logK_ip})")
    mod = _speciation_mod()
    k_spec = mod.conductivity(spec, D=D25, T_C=ph.T_C)
    k_ref = mod.conductivity(mod.dissociated_spec(c_cu_L, c_acid_L, T_C=ph.T_C), D=D25, T_C=ph.T_C)
    if not (k_spec > 0.0 and k_ref > 0.0):
        raise RuntimeError(f"non-positive speciated conductivity: k_spec={k_spec}, k_ref={k_ref}")
    r_cu = spec["c_cu"] / c_cu_L
    return dict(spec=spec, kappa=k_spec, kappa_ratio=k_spec / k_ref, r_cu=r_cu,
                r_so4=spec["c_so4"] / spec["c_so4_total"],
                r_react=float(spec["gamma"]["cu"]) * r_cu)


# ======================================================================================= kinetics
def additive_set(p) -> AdditiveSet:
    """``AdditiveSet`` with the engine's own Params in the kinetics/suppressor slots (plan v2 D2 wiring).

    ``c_s``/``Gamma_s``/``k_plus``/``k_minus`` and the ``i0_u``/``i0_s`` ratio are the engine's literature
    set, so the Cl-gated suppressor of ``additives='cl_ndr'`` reduces EXACTLY to the legacy update when
    ``theta_cl -> 1`` (``suppressor_i0_factor`` reproduces the ``(1-theta) i0_u + theta i0_s``
    interpolation the Kinetics/sweeps use, see tests/test_feature_additives.py).  The accelerator slots are
    the engine's ``c_acc``/``k_acc``/``b_acc``; the leveller ``k_l``/``Gamma_l`` come from Params so the
    transported leveller uses the engine's own transport constants.  Everything else is the module's
    literature-order default, flagged [J] there.
    """
    return AdditiveSet(c_s=p.s_bulk, Gamma_s=p.Gamma, k_plus=p.k_plus, k_minus=p.k_minus,
                       k_des_s=0.0, n_cl=1.0, i0_u=p.i0_u, i0_s=p.i0_s,
                       c_a=p.c_acc, k_a=p.k_acc, b_acc=p.b_acc,
                       k_l=p.k_lev, Gamma_l=p.Gamma_lev)


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


# ======================================================================================= capillarity
def gibbs_thomson_eta(psi, g: Grid, gamma_J_m2, omega_m3_mol=None):
    """Per-cell Gibbs-Thomson shift of the CATHODIC overpotential, eta_GT [V] (plan v2 D6).

    Sign convention.  psi > 0 in the electrolyte and psi < 0 in the metal, so n = grad psi/|grad psi| points
    OUT of the metal and  kappa = levelset.curvature_div_n(psi) = div n  is POSITIVE on a convex metal
    protrusion (deposit growing into the electrolyte; on a metal circle of radius R in the planar 2-D
    geometries kappa = +1/R everywhere).  A curved interface carries the excess metal chemical potential
    mu = mu_flat + gamma Omega kappa  (Kelvin), i.e. a raised local equilibrium potential
    E_eq = E_eq,flat + gamma Omega kappa / (2F) for z = 2, so plating needs a LARGER driving force:

        eta_GT = -gamma Omega kappa / (2 F)                     [V]

    is returned, and the kinetics use  eta_eff = eta - eta_GT = (V - phi) + gamma Omega kappa / (2F).
    kappa > 0 (protrusion) therefore shifts eta_eff TOWARD ZERO, i.e. reduces the cathodic driving force and
    makes the protrusion plate more slowly; a groove (kappa < 0) plates faster.  That is the physical sign.

    kappa is clipped to |kappa| <= 1/(2 dx) (the guard the CEAC use_accel code applies to the same operator)
    because the finite-difference curvature of a re-initialised field is only usable up to the grid scale.
    omega_m3_mol defaults to params.OMEGA_CU_M3_MOL = M_CU/rho_CU (molar volume of solid Cu); curvature_div_n
    returns 1/cm, hence the 100 for metres in gamma [J/m2] * Omega [m3/mol] * kappa [1/m]."""
    omega = OMEGA_CU_M3_MOL if omega_m3_mol is None else float(omega_m3_mol)
    kap = np.clip(curvature_div_n(psi, g), -1.0 / (2.0 * g.dx), 1.0 / (2.0 * g.dx))     # 1/cm
    return -float(gamma_J_m2) * omega * (kap * 100.0) / (2.0 * F)                        # V


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
def scalar_V_cathodic(kin, w, ct, th, acc, phi, I_t, V0=None, lo=-4.0, hi=2.0, eta_gt=None, i_ghost=None):
    """V such that sum w ct ci(V - phi) = I_t with phi fixed.  I(V) is monotone decreasing and ln I is nearly linear
    in V, so Newton on ln I - ln I_t converges in 3-5 iterations; brentq is the fallback.
    eta_gt (plan v2 D6): per-cell Gibbs-Thomson shift [V], the kinetics see eta = V - phi - eta_gt;
    None (default) reproduces the legacy expression exactly.
    i_ghost (plan v2 C4, sharp interface): per-cell a [cm2/A] of the ghost-fluid correction
    ci -> ci/(1 + a ci) of the interface rate (see StepSolver._ghost); None (default) leaves the
    legacy expression untouched."""
    p = kin.p
    f_ = kin.f
    lnT = np.log(I_t)
    V = -0.2 if V0 is None else float(np.clip(V0, lo, hi))
    a = i_ghost

    def cur(eta):
        """(i_eff, |di_eff/deta|) per cell: i_eff = ct ci, times 1/(1 + a ci) on the sharp path."""
        gu, gs = kin.cath(eta, acc)
        ci = (1 - th) * gu + th * gs
        di = f_ * ct * (p.alpha_u * (1 - th) * gu + p.alpha_s * th * gs)
        if a is not None:
            fac = 1.0 / (1.0 + a * ci)
            ci, di = ci * fac, di * fac * fac
        return ct * ci, di

    def tot(eta):
        """(sum w i_eff, sum w |di_eff/deta|); identical summation to the legacy expression when a is None."""
        if a is None:
            gu, gs = kin.cath(eta, acc)
            wu, ws = w * ct * (1 - th) * gu, w * ct * th * gs
            return wu.sum() + ws.sum(), f_ * (p.alpha_u * wu.sum() + p.alpha_s * ws.sum())
        i, di = cur(eta)
        return float(np.dot(w, i)), float(np.dot(w, di))

    for _ in range(40):
        eta = V - phi if eta_gt is None else V - phi - eta_gt
        I, dI = tot(eta)
        if I <= 0:
            break
        dlnI = -dI / I
        r = np.log(I) - lnT
        if abs(r) < 1e-11:
            return V
        step = r / dlnI
        V = float(np.clip(V - np.clip(step, -0.5, 0.5), lo, hi))

    def g(V):
        eta = V - phi if eta_gt is None else V - phi - eta_gt
        if a is None:
            gu, gs = kin.cath(eta, acc)
            return np.dot(w * ct, (1 - th) * gu + th * gs) - I_t
        return float(np.dot(w, cur(eta)[0])) - I_t
    glo, ghi = g(lo), g(hi)
    if glo < 0:
        return lo
    if ghi > 0:
        return hi
    return brentq(g, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200)


def newton_potential(geo, cs, kin, w, ct, th, acc, V0, phi0, I_t, galvano, anodic=False, maxit=14, tol_eta=2e-7,
                     src=None, eta_gt=None, i_ghost=None):
    """Solve  K phi + sigma w I(V-phi) = 0  [sigma = +1 cathodic sink, -1 anodic source] and, if galvano,
    sum w I = I_t for the unknowns (phi, V) with Newton (bordered system reduced by a Schur complement).
    src (tertiary path only): the divergence of the diffusion current F sum z_i D_i grad c_i, so the
    residual is  K phi + sigma w I - src = 0  (charge balance  div i_l = 0  with i_l = -kappa grad phi
    - F sum z_i D_i grad c_i).  src=None reproduces the enhanced path exactly.
    eta_gt (plan v2 D6): per-cell Gibbs-Thomson shift [V] of the cathodic overpotential, so the kinetics see
    eta = V - phi - eta_gt (see gibbs_thomson_eta).  It is constant in V and phi, so the Newton Jacobian is
    unchanged.  None (default) is bit-identical to the pre-D6 solver; the anodic branch ignores it.
    i_ghost (plan v2 C4): per-cell a [cm2/A] of the sharp-interface ghost-fluid correction
    ci -> ci/(1 + a ci) of the cathodic interface rate ci = (1-theta) gu + theta gs
    (StepSolver._ghost).  None (default) is the legacy diffuse-band expression, bit-identical; the
    anodic branch ignores it.
    Returns V, phi, I_p (A/cm2 per cell incl. ct factor for cathodic), iterations."""
    sigma = -1.0 if anodic else 1.0
    phi = phi0.copy()
    V = V0
    n = geo.n
    shift = eta_gt if (eta_gt is not None and not anodic) else None
    a = None if anodic else i_ghost

    def current(eta):
        if anodic:
            i, di = kin.anod(eta)                      # di = dI/deta > 0
            return i, di
        gu, gs = kin.cath(eta, acc)
        ci = (1 - th) * gu + th * gs
        di = kin.f * ct * (kin.p.alpha_u * (1 - th) * gu + kin.p.alpha_s * th * gs)   # -dI/deta > 0
        if a is not None:
            # ghost-fluid: the transported interface rate is ci/(1 + a ci) with d/deta -> di/(1+a ci)^2
            fac = 1.0 / (1.0 + a * ci)
            ci, di = ci * fac, di * fac * fac
        return ct * ci, di

    if galvano:
        if anodic:
            # closed form for fixed phi: sum w i0a exp(af (V-phi)) = I_t
            ph = kin.ph
            s = np.dot(w, np.exp(np.clip(-kin.p.alpha_a * kin.f * phi, -60, 60)))
            V = np.log(I_t / (ph.i0_a * s + 1e-300)) / (kin.p.alpha_a * kin.f)
        else:
            V = scalar_V_cathodic(kin, w, ct, th, acc, phi, I_t, V0=V0, i_ghost=i_ghost)
    for it in range(maxit):
        eta = V - phi if shift is None else V - phi - shift
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
    eta = V - phi if shift is None else V - phi - shift
    I, _ = current(eta)
    return V, phi, I, it + 1


# ======================================================================================= main step solver
class StepSolver:
    PHI_SC = 0.05        # V   weights for the Anderson least-squares norm
    V_SC = 0.05
    TVD_OMEGA = 1.0      # under-relaxation of the MUSCL deferred correction (1 = full limited face value).
                         # The correction is an explicit function of the current iterate, so it only adds a
                         # local O(dx^2) term to the Picard map; the Anderson loop handled omega = 1 on the
                         # narrow-trench gate case (same iteration count as upwind), hence no damping here.
                         # Lower it (e.g. 0.5) if a stiff case stalls on the correction.

    def __init__(self, geo: StepGeom, p, ph, Dt, reuse=None, flow=None, flow_faces=None, shade=None):
        """reuse: LU factors (c, s, phi) of the previous step; valid when the active cell set is unchanged, they
        then serve as preconditioners (refinement) for the slightly different matrices of this step.
        flow: (ux, uz) cell-centred velocities (nr, nz) in cm/s for the resolved cavity flow
        ('trench_full'); the advective operator is added to the Cu2+ and suppressor equations.
        flow_faces: (ufx, ufz) MAC face velocities in the engine orientation (nr, nz) / (nr, nz+1);
        required by flow_solver='stokes', which then uses the conservative divergence-form operator of
        StepGeom.mac_advection (plus its open-top rhs adv_bc) instead of the interpolated face fluxes.
        shade: optional per-active-cell factor in [0, 1] that multiplies the interface weights w (plan v2
        E2, bubble shading).  None (default) -> w = geo.w, the legacy operator, bit-identical."""
        self.geo, self.p, self.ph = geo, p, ph
        self.kin = Kinetics(ph, p)
        self.Dt = Dt
        self.flow = flow
        self.flow_faces = flow_faces
        # ---- plan v2 C4: ghost-fluid distance of the sharp interface (None -> legacy diffuse band)
        self.ghost_d = None
        self.ghost_a = None
        # ---- plan v2 E2: bubble shading of the interface weights (None = legacy)
        self.shade = shade
        self.adv_bc = None                  # rhs source of the open-top inflow (stokes MAC path only)
        self.adv_top = None                 # top-plane z-face flux per active top cell (stokes path only)
        self.tvd_faces = None               # (F_r, F_z, F_w) of the MUSCL deferred correction (tvd only)
        self.tertiary = p.transport == "tertiary"
        # Plan v2 D6 (Gibbs-Thomson): per-active-cell shift of the local cathodic overpotential, computed once
        # per step from the frozen level set (see gibbs_thomson_eta for the sign convention).  None - the
        # default - makes every kinetic expression below exactly the legacy one.
        self.eta_gt = None
        if p.gibbs_thomson:
            self.eta_gt = gibbs_thomson_eta(geo.psi, geo.g, p.gamma_cu_J_m2)[geo.act]
        # Plan v2 D1b: bulk speciation state (None = legacy fully dissociated bath).  The transported Cu2+
        # field stays TOTAL Cu; r_react/r_cu/r_so4 apply the bulk free-ion ratios (frozen-speciation
        # approximation) to the kinetics, the diffusion current and the conductivity.
        self.spec = None
        self.kappa_ratio = 1.0              # kappa(speciated) / kappa(fully dissociated) at the bulk point
        self.r_react = 1.0                  # reactive Cu: gamma_Cu * c_Cu_free / c_Cu_total
        self.r_cu = 1.0                     # free Cu2+ / total Cu (diffusion current)
        self.r_so4 = 1.0                    # free SO4^2- / total sulfate (diffusion current)
        self.kappa_bulk = 0.0               # speciated bulk conductivity (S/cm), 0 when speciation off
        # Plan v2 D2 (full additive system): mode flags and the additive constants.  'legacy' keeps every
        # legacy expression and operation order below untouched (bit-identical defaults).
        self.add_mode = p.additives
        self.cl_on = p.additives in ("cl_ndr", "full") and bool(p.suppressor)
        self.full = p.additives == "full"
        self.lev_on = bool(self.full and p.c_lev_bulk > 0.0)
        self.aset = additive_set(p) if self.add_mode != "legacy" else None
        g = geo.g
        lu_c = lu_s = lu_p = lu_l = None
        if reuse is not None and reuse[0].shape == geo.act.shape and np.array_equal(reuse[0], geo.act):
            lu_c, lu_s, lu_p, lu_l = reuse[1:]
        if self.tertiary:
            self._init_tertiary(Dt, reuse, lu_s, lu_l)
            if geo.d_if is not None:
                self._init_ghost(ph.D_cu + Dt)
            return
        Kc, self.bc_c = geo.K_data(ph.D_c + Dt)
        Ks, self.bc_s = geo.K_data(ph.D_s + Dt)
        if geo.d_if is not None:
            self._init_ghost(ph.D_c + Dt)
        self.adv = None
        # adv_scheme only has a meaning on the Stokes (MAC) path; the LBM path keeps upwind + adv_form.
        self.adv_scheme = p.adv_scheme if (p.flow_solver == "stokes" and flow_faces is not None) else "upwind"
        if g.periodic_x:
            # non-symmetric advection-diffusion: general LU (no symmetric mode), no cross-step LU reuse
            ux, uz = (np.zeros((g.nr, g.nz)), np.zeros((g.nr, g.nz))) if flow is None else flow
            if p.flow_solver == "stokes" and flow_faces is not None:
                adv, self.adv_bc, self.adv_top = geo.mac_advection(*flow_faces)
            else:
                adv = geo.adv_matrix(*geo.face_fluxes(ux, uz), form=p.adv_form)
            self.adv = adv                                # exposed for the sim.py Cu2+ balance
            if self.adv_scheme == "sg":
                # Exponentially fitted faces: replace K_data + upwind by the SG operator on every interior
                # face, per species with its own D.  The top plane (Dirichlet diffusion + upwind advection)
                # is unchanged, so the plane influx counted by sim.py is unchanged.
                F_r, F_z, F_w, _ = geo.mac_face_fluxes(*flow_faces)
                top_diag, _ = geo.top_advection(self.adv_top)
                Ac, self.bc_c = geo.sg_advdiff(ph.D_c + Dt, F_r, F_z, F_w)
                As, self.bc_s = geo.sg_advdiff(ph.D_s + Dt, F_r, F_z, F_w)
                self.cs_c = CachedGen(geo, (Ac + sp.diags(top_diag)).tocsr())
                self.cs_s = CachedGen(geo, (As + sp.diags(top_diag)).tocsr())
            else:
                self.cs_c = CachedGen(geo, self._base(Kc, adv))
                self.cs_s = CachedGen(geo, self._base(Ks, adv))
                if self.adv_scheme == "tvd":
                    # Limited (MUSCL / van Leer) face values by DEFERRED CORRECTION: the implicit operator
                    # stays the upwind one above (CachedGen / LU reuse unchanged), and every Picard sweep
                    # adds  -D(c)  to the rhs (D = flux divergence of the limited minus the upwind face
                    # flux, StepGeom.tvd_correction).  D is conservative per face and vanishes for a
                    # uniform field, so the Cu balance still closes and c = const stays exact.
                    self.tvd_faces = geo.mac_face_fluxes(*flow_faces)[:3]
        else:
            self.cs_c, self.cs_s = CachedSym(geo, Kc, lu=lu_c), CachedSym(geo, Ks, lu=lu_s)
        # Plan v2 D2 ('full' only): the transported leveller copies the suppressor machinery (diffusion +
        # Dirichlet bulk + an interface sink linear in the normalised concentration, so only the diagonal
        # changes between sweeps).  adv_scheme='sg' fits the SG faces of Cu2+/suppressor only; the leveller
        # keeps the upwind operator of the same face fluxes there (flagged, new mode).
        self.cs_l, self.bc_l = None, None
        if self.lev_on:
            Kl, self.bc_l = geo.K_data(ph.D_lev + Dt)
            self.cs_l = (CachedGen(geo, self._base(Kl, self.adv)) if self.adv is not None
                         else CachedSym(geo, Kl, lu=lu_l))
        self.cs_p = None
        if p.ohmic:
            Kp, _ = geo.K_data(ph.kappa * np.ones((g.nr, g.nz)))
            self.cs_p = CachedSym(geo, Kp, lu=lu_p)

    def _base(self, K_data, adv):
        """Constant part (diffusion + advection) of a trench_full operator, CSR."""
        geo = self.geo
        return sp.csr_matrix((K_data, geo.indices, geo.indptr), shape=(geo.n, geo.n)) + adv

    # ------------------------------------------------------------------ sharp interface (C4)
    def _init_ghost(self, D_cu):
        """Ghost-fluid distance of the Cu2+ interface sink (plan v2 C4, StepGeom.interface='sharp').

        The cut-cell sink of cell p is the interface flux evaluated at the NODE, but the interface is
        |psi_p| = d_p away from it.  With the interface rate k = ci/(2F c_bulk) [cm/s] (ci = the kinetic
        rate that maps concentration to current density, (1-theta) gu + theta gs) and the local
        diffusivity D, the series conductance of 'node -> interface' and 'interface reaction' is
            k_eff = k / (1 + k d_p / D),  i.e.  ci_eff = ci / (1 + a_p ci),  a_p = d_p / (2F c_bulk D_p),
        and the interface current density is  ct ci_eff  (reported as i_f, so sum(w i_f) is what the sink
        actually consumes).  For k d/D << 1 (activation control) this is 1, i.e. identical to the legacy
        sink; for k d/D >> 1 (limiting current) it gives the diffusive flux D c_bulk/d_p, which makes the
        1-D planar steady limiting current exact for any front position inside the cell:
            c_bulk = sum_faces (dx/D) N + d_p/D N + N/k  ->  N = c_bulk/(L/D + 1/k),  L = distance
        from the front to the Dirichlet plane.  The correction is 2nd order in the front position
        (first-order error removed), and it is applied to the Cu2+ sink only -- the suppressor and
        leveller sinks keep the plain interface area w (their rate constants are partly lumped
        mass-transfer coefficients, so a Damkoehler correction there would not be meaningful)."""
        self.ghost_d = self.geo.d_if[self.geo.act]
        self.D_cu_loc = np.asarray(D_cu, dtype=float)[self.geo.act]
        self.ghost_a = self.ghost_d / (2.0 * F * self.p.c_bulk * np.maximum(self.D_cu_loc, 1e-30))

    def _ghost(self, ci, scale=1.0):
        """Per-cell 1/(1 + scale * a * ci): the ghost-fluid factor of the sharp-interface sink.

        ci is the interface rate at BULK concentration, ci = (1-theta) gu + theta gs [A/cm2] (the
        coefficient that maps concentration -> current density), a = |psi_node|/(2F c_bulk D) [cm2/A]
        and scale = r_react on the tertiary path.  a*ci is the Damkoehler number k d/D of the sink, so
        the factor is concentration independent and the sink stays LINEAR in c (M-matrix).
        Returns 1.0 (a Python float, exactly neutral) for the legacy diffuse band."""
        if self.ghost_a is None:
            return 1.0
        return 1.0 / (1.0 + scale * self.ghost_a * np.asarray(ci, dtype=float))

    def _ghost_scaled(self):
        """The ghost-fluid a of the current transport path: a * r_react (r_react == 1 on the enhanced
        path, so this is exactly ghost_a there).  None when the interface is diffuse."""
        return None if self.ghost_a is None else self.ghost_a * self.r_react

    # ------------------------------------------------------------------ tertiary transport
    def _init_tertiary(self, Dt, reuse=None, lu_s=None, lu_l=None):
        """Operators for transport == 'tertiary' (explicit Cu2+ / H+ / SO4^2-).

        Cu2+ stays in the Anderson vector as ctil = c_Cu/c_bulk, so every existing diagnostic and the
        Cu2+ mass balance of sim.py keep their meaning; H+ and SO4^2- live in st['c_h']/st['c_so4'] in
        mol/cm3 (absolute).  c_SO4 is closed by electroneutrality sum z_i c_i = 0 (COMSOL/Newman:
        eliminate the majority inert anion; an eliminated species cannot carry a flux BC), so only Cu2+
        and H+ are solved implicitly and the potential comes from the charge balance
        div(-kappa grad phi - F sum z_i D_i grad c_i) = 0 with kappa evaluated locally."""
        geo, p, ph, g = self.geo, self.p, self.ph, self.geo.g
        if p.speciation:
            self._init_speciation()
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
        # Plan v2 D2 ('full'): transported leveller, same transport machinery as the suppressor above.
        self.cs_l, self.bc_l = None, None
        if self.lev_on:
            Kl, self.bc_l = geo.K_data(ph.D_lev + Dt)
            self.cs_l = (CachedGen(geo, self._base(Kl, adv)) if adv is not None
                         else CachedSym(geo, Kl, lu=lu_l))
        # diffusion-current source: constant-D operators per species (K_data data vectors, CSR order)
        self.kd = [geo.K_data(self.D_cu_a), geo.K_data(self.D_h_a), geo.K_data(self.D_s_a)]
        self.D_n = (self.D_cu_a[geo.act], self.D_h_a[geo.act], self.D_s_a[geo.act])
        kap0 = self.kappa_bulk if self.spec is not None else ph.kappa_ne
        self.kappa_floor = max(1e-3 * kap0, 1e-9)
        self.c_floor = 1e-6 * p.c_bulk
        self.c_h_scale = max(ph.c_h_bulk, 1e-3 * p.c_bulk)
        self.cs_p = None
        lu_p = None
        if reuse is not None and reuse[0].shape == geo.act.shape and np.array_equal(reuse[0], geo.act):
            lu_p = reuse[3]
        if p.ohmic:
            kp, _ = geo.K_data(np.full((g.nr, g.nz), kap0))
            self.cs_p = CachedSym(geo, kp, lu=lu_p)

    def _init_speciation(self):
        """Bulk speciation state from :func:`bulk_speciation`; see its docstring for the approximation."""
        p, ph = self.p, self.ph
        sp = bulk_speciation(p, ph)
        self.spec = sp["spec"]
        self.kappa_bulk = sp["kappa"]
        self.kappa_ratio = sp["kappa_ratio"]
        self.r_cu = sp["r_cu"]
        self.r_so4 = sp["r_so4"]
        self.r_react = sp["r_react"]

    def _kappa(self, c_cu, c_h, c_s):
        """Local Nernst-Einstein conductivity (S/cm) on the active cells, floored, full grid.

        With speciation the fully-dissociated local sum is scaled by the bulk speciated ratio
        kappa_speciated / kappa_dissociated (frozen speciation: ion pairing and HSO4- formation remove
        carriers everywhere in the same proportion).  Caveat: a scalar rescale keeps the local transport
        numbers (Cu2+ share of the current) at their fully-dissociated values; a free-ion-resolved kappa
        would lower the Cu2+ migration share too (see the TODO in bulk_speciation)."""
        geo, ph = self.geo, self.ph
        kap = (ph.z_cu ** 2 * self.D_n[0] * c_cu + ph.z_h ** 2 * self.D_n[1] * c_h
               + ph.z_so4 ** 2 * self.D_n[2] * c_s)
        kap = np.maximum(F * ph.f_RT * kap * self.kappa_ratio, self.kappa_floor)
        out = np.full(geo.act.shape, self.kappa_bulk if self.spec is not None else ph.kappa_ne)
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
        # Speciation (plan v2 D1b): only the FREE ions carry the diffusion current, so the Cu2+ and
        # SO4^2- contributions are scaled by their bulk free fractions (frozen speciation; HSO4- carries
        # no current in this lumped-source model).  r = 1 when speciation is off -> bit-identical.
        for z, c, (kd, bc), cb, rfree in ((ph.z_cu, c_cu, self.kd[0], self.cb[0], self.r_cu),
                                          (ph.z_h, c_h, self.kd[1], self.cb[1], 1.0),
                                          (ph.z_so4, c_s, self.kd[2], self.cb[2], self.r_so4)):
            src -= z * F * rfree * (geo.matvec(kd, c) - bc * cb)
        return src

    def lu_state(self):
        return (self.geo.act.copy(), self.cs_c.lu, self.cs_s.lu, self.cs_p.lu if self.cs_p else None,
                self.cs_l.lu if self.cs_l is not None else None)

    # ------------------------------------------------------------------
    def solve(self, st, ctrl, dt):
        """Fixed-point solution of the coupled (theta, s, phi/V, c) problem on the frozen geometry, Anderson
        accelerated.  st: dict of full-grid arrays + scalars (updated in place).
        ctrl: dict(mode 'I'|'V', I_t (A, instantaneous forward total), V_set, duty_f, duty_r, rr, pr)."""
        geo, p, ph, kin = self.geo, self.p, self.ph, self.kin
        act, n = geo.act, geo.n
        # plan v2 E2: bubble shading multiplies the interface weights, so the remaining area carries the
        # galvanostatic current.  shade=None (default) leaves w = geo.w, i.e. the legacy operator.
        w = geo.w if self.shade is None else geo.w * self.shade
        galvano = ctrl['mode'] == 'I'
        duty_f, duty_r = ctrl['duty_f'], ctrl['duty_r']
        cl_on, lev_on, full = self.cl_on, self.lev_on, self.full
        aset = self.aset
        acc_a = (1.0 + p.b_acc * st['theta_a'][act]) if (p.use_accel or full) else 1.0
        th0 = st['theta'][act].copy()
        s_old = st['s'][act].copy()
        # ---- plan v2 D2 coverage state of the step (theta_cl is owned by sim.run, which re-steps it with the
        # true dt after the CFL clip; the sweeps below use the start-of-step value as the Cl gate).
        th_clr = st['theta_cl'][act].copy() if cl_on else None
        ta = st['theta_a'][act] if full else None
        th_l0 = st['theta_l'][act].copy() if lev_on else None
        lev_old = st['lev'][act].copy() if lev_on else None
        lev_t = lev_old
        lev_new = lev_old
        th_l_new = th_l0

        def acc_now(th_l_now):
            """i0 multiplier of the accelerator (plan v2 D2 'full': x the leveller blocking factor)."""
            if not lev_on:
                return acc_a
            return acc_a * aset.leveller_factor(th_l_now)
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
            nonlocal lev_t, lev_new, th_l_new
            th, s_t, ct, phi, V = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n] * self.PHI_SC, x[4 * n] * self.V_SC
            if not p.ohmic:
                phi = np.zeros(n)
            if not galvano:
                V = ctrl['V_set']
            eta = V - phi if self.eta_gt is None else V - phi - self.eta_gt   # D6: local GT-corrected eta
            acc_s = acc_now(lev_t) if lev_on else acc_a
            gu, gs = kin.cath(eta, acc_s)
            if p.suppressor:
                # 'full' moves the reverse-pulse coverage loss out of theta_be into the additive kernel
                # (additives.reverse_pulse_desorption, applied by sim.run with the accepted dt).
                b_ex = 0.0 if full else (p.k_rd * duty_r * Ia * (1.0 - p.pr_dl_fraction) / 1e-3 if ctrl['pr'] else 0.0)
                if cl_on:
                    # D2: the Cl- adlayer gates the polyether adsorption (Braun 2019/2020 co-adsorption).
                    # suppressor_cl_rate = k_plus c_s theta_cl**n_cl, so theta_cl -> 1 reproduces the legacy
                    # rate (and sink coefficient) exactly and theta_cl -> 0 adsorbs no suppressor.
                    gate = np.clip(th_clr, 0.0, 1.0) ** aset.n_cl
                    a_rate = suppressor_cl_rate(p.s_bulk * s_t, th_clr, p.k_plus, n_cl=aset.n_cl)
                    th_new = theta_be(th0, dt, a_rate, Mf * ct * gu, Mf * ct * gs,
                                      p.k_minus * duty_f, b_ex)
                    extra = w * p.Gamma * p.k_plus * gate * (1 - th_new)
                else:
                    th_new = theta_be(th0, dt, p.k_plus * p.s_bulk * s_t, Mf * ct * gu, Mf * ct * gs,
                                      p.k_minus * duty_f, b_ex)
                    extra = w * p.Gamma * p.k_plus * (1 - th_new)      # sink = Gamma k_plus s (1-theta)
                rhs = self.bc_s.copy()
                if self.adv_bc is not None:                   # bulk normalised: s_bulk -> 1
                    rhs = rhs + self.adv_bc
                if self.tvd_faces is not None:                # limited (MUSCL) face values, deferred
                    rhs = rhs - self.TVD_OMEGA * geo.tvd_correction(s_t, *self.tvd_faces)
                if p.s_transient:
                    extra = extra + geo.vh / dt
                    rhs += geo.vh / dt * s_old
                s_new = np.clip(self.cs_s.solve(extra, rhs), 0.0, 1.5)
            else:
                th_new, s_new = th0 * 0.0, s_t
            if lev_on:
                # D2 'full': transport-limited leveller coverage from the local front supply c_l = c_lev_bulk
                # * lev, then the transported leveller equation (same machinery as the suppressor, always
                # transient here because the species is new).
                v_cell = Mf * ct * ((1.0 - th_new) * gu + th_new * gs)
                th_l_new = aset.leveller_step(th_l0, dt, c_l=p.c_lev_bulk * np.clip(lev_t, 0.0, None),
                                              v=v_cell)
                # Sink coefficient of the NORMALISED leveller field: the molar sink is k_l c_l (1-theta), so
                # after dividing by c_lev_bulk the diagonal carries k_l (1-theta) [cm/s], not Gamma_l*k_l.
                extra_l = w * p.k_lev * (1.0 - th_l_new) + geo.vh / dt
                rhs_l = self.bc_l.copy() + geo.vh / dt * lev_old
                if self.adv_bc is not None:                   # normalised: bulk leveller -> 1
                    rhs_l = rhs_l + self.adv_bc
                if self.tvd_faces is not None:
                    rhs_l = rhs_l - self.TVD_OMEGA * geo.tvd_correction(lev_t, *self.tvd_faces)
                lev_new = np.clip(self.cs_l.solve(extra_l, rhs_l), 0.0, 1.5)
            thk = displace_suppressor(th_new, ta) if full else th_new
            acc_k = acc_now(th_l_new) if lev_on else acc_a
            if p.ohmic:
                V_new, phi_new, _, nph = newton_potential(geo, self.cs_p, kin, w, ct, thk, acc_k, V, phi,
                                                          ctrl['I_t'], galvano, eta_gt=self.eta_gt,
                                                          i_ghost=self._ghost_scaled())
            else:
                V_new, phi_new, _, nph = self._nonohmic(w, ct, thk, acc_k, V, ctrl, galvano)
            diag['nph'] += nph
            gu, gs = kin.cath(V_new - phi_new if self.eta_gt is None else V_new - phi_new - self.eta_gt, acc_k)
            ci = (1 - thk) * gu + thk * gs
            extra = p.efficiency * duty_f * w * ci / (2 * F * p.c_bulk)  # Cu2+ sink, time-averaged over PR cycle
            if self.ghost_a is not None:
                # C4 ghost-fluid: sink of the cut cell at distance |psi| from its node (see _init_ghost)
                extra = extra * self._ghost(ci)
            rhs_c = self.bc_c + rev_src
            if self.adv_bc is not None:                       # open-top inflow, bulk normalised to 1
                rhs_c = rhs_c + self.adv_bc
            if self.tvd_faces is not None:                    # limited (MUSCL) face values, deferred
                rhs_c = rhs_c - self.TVD_OMEGA * geo.tvd_correction(ct, *self.tvd_faces)
            ct_new = np.clip(self.cs_c.solve(extra, rhs_c), 0.0, 3.0)
            if self.ghost_a is not None:
                diag['bal'] = (np.dot(w * ci * self._ghost(ci), ct_new) / ctrl['I_t'] - 1.0) \
                    if galvano else 0.0
            else:
                diag['bal'] = (np.dot(w * ci, ct_new) / ctrl['I_t'] - 1.0) if galvano else 0.0
            lev_t = lev_new
            return np.concatenate([th_new, s_new, ct_new, phi_new / self.PHI_SC, [V_new / self.V_SC]])

        def sweep_ter(x):
            """One outer Picard pass for transport == 'tertiary':
            theta -> s -> [Cu2+/H+ SG species -> electroneutral sulfate -> local kappa -> (V, phi) with the
            diffusion current -> kinetics] x tertiary_inner."""
            nonlocal lev_t, lev_new, th_l_new
            th, s_t, ct, phi, V = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:4 * n] * self.PHI_SC, x[4 * n] * self.V_SC
            if not p.ohmic:
                phi = np.zeros(n)
            if not galvano:
                V = ctrl['V_set']
            eta = V - phi if self.eta_gt is None else V - phi - self.eta_gt   # D6: local GT-corrected eta
            acc_s = acc_now(lev_t) if lev_on else acc_a
            gu, gs = kin.cath(eta, acc_s)
            r_react = self.r_react                   # gamma_Cu * free-Cu fraction (1.0 = legacy)
            if p.suppressor:
                b_ex = 0.0 if full else (p.k_rd * duty_r * Ia * (1.0 - p.pr_dl_fraction) / 1e-3 if ctrl['pr'] else 0.0)
                if cl_on:
                    gate = np.clip(th_clr, 0.0, 1.0) ** aset.n_cl
                    a_rate = suppressor_cl_rate(p.s_bulk * s_t, th_clr, p.k_plus, n_cl=aset.n_cl)
                    th_new = theta_be(th0, dt, a_rate, Mf * ct * r_react * gu,
                                      Mf * ct * r_react * gs, p.k_minus * duty_f, b_ex)
                    extra = w * p.Gamma * p.k_plus * gate * (1 - th_new)
                else:
                    th_new = theta_be(th0, dt, p.k_plus * p.s_bulk * s_t, Mf * ct * r_react * gu,
                                      Mf * ct * r_react * gs, p.k_minus * duty_f, b_ex)
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
            if lev_on:
                v_cell = Mf * ct * r_react * ((1.0 - th_new) * gu + th_new * gs)
                th_l_new = aset.leveller_step(th_l0, dt, c_l=p.c_lev_bulk * np.clip(lev_t, 0.0, None),
                                              v=v_cell)
                extra_l = w * p.k_lev * (1.0 - th_l_new) + geo.vh / dt
                rhs_l = self.bc_l.copy() + geo.vh / dt * lev_old
                if self.adv_bc is not None:                   # normalised: bulk leveller -> 1
                    rhs_l = rhs_l + self.adv_bc
                lev_new = np.clip(self.cs_l.solve(extra_l, rhs_l), 0.0, 1.5)
            thk = displace_suppressor(th_new, ta) if full else th_new
            acc_k = acc_now(th_l_new) if lev_on else acc_a
            c_h_old = ter['c_h']
            c_cu, c_h, c_s = ct * p.c_bulk, c_h_old, ter['c_so4']
            phi_w, V_w = phi, V
            for _ in range(max(int(p.tertiary_inner), 1)):
                # TODO(D1b, stretch): re-speciate locally from (c_Cu, c_H, c_SO4) here and use the local
                # free fraction / gamma instead of the bulk r_react, r_cu, r_so4 (frozen speciation now).
                phi_g = np.zeros(geo.act.shape)
                phi_g[act] = phi_w
                # ---- kinetics at the current eta fixes the Cu2+ interface sink.  The sink is linear in c
                # (i_loc = ci * c_Cu/c_bulk), so the diagonal coefficient is w*ci/(2F c_bulk) and it
                # multiplies the absolute concentration c_Cu -- the same number as the enhanced path's
                # extra, which multiplies ctil.
                gu_w, gs_w = kin.cath(V_w - phi_w if self.eta_gt is None else V_w - phi_w - self.eta_gt, acc_k)
                ci = (1 - thk) * gu_w + thk * gs_w
                # i_loc = ci * (reactive Cu)/(c_bulk) with reactive Cu = r_react * total Cu (bulk-ratio
                # approximation).  The dissolved species stays TOTAL Cu, so the sink coefficient is
                # scaled by r_react and multiplies c_Cu_total -- the Cu balance still closes because
                # i_f = r_react * ct * ci is the current that the deposit and the sink both use.
                sink = p.efficiency * duty_f * w * ci * r_react / (2 * F * p.c_bulk)
                if self.ghost_a is not None:
                    # C4 ghost-fluid (see _init_ghost): a = d/(2F c_bulk D_Cu), i = r_react ct ci
                    sink = sink * self._ghost(ci, scale=r_react)
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
                ct_w = np.clip(c_cu / p.c_bulk, 0.0, 3.0) * r_react   # reactive Cu (BV), bulk-ratio approx.
                if p.ohmic:
                    kp, _ = geo.K_data(self._kappa(c_cu, c_h, c_s))
                    self.cs_p = CachedSym(geo, kp, lu=self.cs_p.lu if self.cs_p is not None else None)
                    src = self._diff_src(c_cu, c_h, c_s)
                    V_w, phi_w, _, nph = newton_potential(geo, self.cs_p, kin, w, ct_w, thk, acc_k, V_w, phi_w,
                                                          ctrl['I_t'], galvano, src=src, eta_gt=self.eta_gt,
                                                          i_ghost=self._ghost_scaled())
                else:
                    V_w, phi_w, _, nph = self._nonohmic(w, ct_w, thk, acc_k, V_w, ctrl, galvano)
                diag['nph'] += nph
            ter['c_h'], ter['c_so4'] = c_h, c_s
            ter['d_h'] = float(np.max(np.abs(c_h - c_h_old)) / self.c_h_scale) if n else 0.0
            ct_new = np.clip(c_cu / p.c_bulk, 0.0, 3.0)
            gu_w, gs_w = kin.cath(V_w - phi_w if self.eta_gt is None else V_w - phi_w - self.eta_gt, acc_k)
            ci = (1 - thk) * gu_w + thk * gs_w
            # galvanostatic balance: newton_potential solved sum(w I) = I_t at this reactive ct, so bal is ~0
            # by construction here; convergence of the tertiary path rests on d_c / d_th (and ter['d_h']).
            if self.ghost_a is not None:
                # C4 ghost-fluid: the transported interface flux of the node current r_react ct ci
                diag['bal'] = (np.dot(w * ci * self._ghost(ci, scale=r_react), ct_new * r_react)
                               / ctrl['I_t'] - 1.0) if galvano else 0.0
            else:
                diag['bal'] = (np.dot(w * ci, ct_new * r_react) / ctrl['I_t'] - 1.0) if galvano else 0.0
            lev_t = lev_new
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
        ct_r = ct * self.r_react                 # reactive Cu (bulk-ratio speciation); = ct when off
        # D2 'full': the kinetics see the displaced suppressor coverage and the accelerator x leveller factor;
        # identical values to `th`/`acc` when the mode is 'legacy'.
        thk_f = displace_suppressor(th, ta) if full else th
        acc_f = acc_now(th_l_new) if lev_on else acc_a
        if galvano:
            V = scalar_V_cathodic(kin, w, ct_r, thk_f, acc_f, phi, ctrl['I_t'], V0=V, eta_gt=self.eta_gt,
                                  i_ghost=self._ghost_scaled())
        eta = V - phi                               # physical (reported) overpotential
        eta_k = eta if self.eta_gt is None else eta - self.eta_gt   # D6: what the kinetics see
        gu, gs = kin.cath(eta_k, acc_f)
        ci = (1 - thk_f) * gu + thk_f * gs
        i_f = ct_r * ci                            # forward cathodic current density per cell (A/cm2)
        if self.ghost_a is not None:
            # C4: i_f is reported at the INTERFACE (the node value rejected by the ghost-fluid factor),
            # so sum(w i_f) is the transported interface current that the sink actually consumes (and
            # that the Cu2+ balance and the level-set speed use), not the node value ci*ct.
            i_f = i_f * self._ghost(ci, scale=self.r_react)
        info['I_f'] = float(np.dot(w, i_f))
        st['V'] = V
        if ctrl['pr']:
            st['V_r'] = V_r
            st['phi_r'][act] = phi_r
        st['theta'][act] = th
        st['s'][act] = s_t
        st['ctil'][act] = ct
        if lev_on:
            st['theta_l'][act] = th_l_new
            st['lev'][act] = lev_new
        if self.tertiary:
            st['c_h'][act] = ter['c_h']
            st['c_so4'][act] = ter['c_so4']
        if p.ohmic:
            st['phi'][act] = phi
        info['n_fac'] = (self.cs_c.n_fac + self.cs_s.n_fac + (self.cs_p.n_fac if self.cs_p else 0)
                         + (self.sg_h.n_fac if self.tertiary else 0)
                         + (self.cs_l.n_fac if self.cs_l is not None else 0))
        return dict(act=act, i_f=i_f, i_a=Ia, th=th, th0=th0, ct=ct, ct_react=ct_r, eta=eta, phi=phi,
                    s_t=s_t, gu=gu, gs=gs,
                    theta_cl=th_clr, theta_l=th_l_new, lev=lev_new, th_k=thk_f,
                    c_h=ter['c_h'] if self.tertiary else None,
                    c_so4=ter['c_so4'] if self.tertiary else None, info=info)

    # ---------------------------------------------------------------- phi = 0 helpers
    def _nonohmic(self, w, ct, th, acc, V, ctrl, galvano):
        n = self.geo.n
        phi = np.zeros(n)
        if galvano:
            V = scalar_V_cathodic(self.kin, w, ct, th, acc, phi, ctrl['I_t'], V0=V, eta_gt=self.eta_gt,
                                  i_ghost=self._ghost_scaled())
        else:
            V = ctrl['V_set']
        eta = V - phi if self.eta_gt is None else V - phi - self.eta_gt
        gu, gs = self.kin.cath(eta, acc)
        return V, phi, ct * ((1 - th) * gu + th * gs), 0

    def _rev_nonohmic(self, w, I_ta):
        ph, p = self.ph, self.p
        V_r = np.log(I_ta / (ph.i0_a * w.sum() + 1e-300)) / (p.alpha_a * self.kin.f)
        return V_r, np.zeros(self.geo.n), None, 0
