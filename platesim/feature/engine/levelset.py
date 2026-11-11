"""Grid and level-set utilities for the via (r, z) / trench (x, z) feature-fill model.

Conventions
  * cell-centred uniform grid, array shape (nr, nz), index [i, j] <-> (r_i, z_j) or (x_i, z_j);
    r_i / x_i = (i+0.5) dx, z_j = z_lo + (j+0.5) dx
  * psi = signed distance, > 0 in the electrolyte, < 0 in metal (Si + Cu); the zero set is the plating front
  * symmetry (zero flux) at the first and last column (r = 0 / R for the via, x = 0 / R for the trench);
    the half plane is mirrored for every FMM call.  Geometry 'trench_full' instead uses the FULL pitch
    x in [0, pitch] with periodic (wrap) x boundaries and no mirror anywhere
  * geometry = 'via'     : 2-D axisymmetric (r, z), face areas ~ r, cell volumes ~ r dr dz, projected
                           area pi R^2 (all quantities per full unit cell / full circle)
    geometry = 'trench'  : planar 2-D (x, z) per unit LENGTH in y, a long trench of width w = d = 2a,
                           depth H, half-domain x in [0, R] with R = a / f_open (open fraction of a
                           trench array = a / R); every r-weighting replaced by weight 1 (volumes and
                           face areas become dx^2 and dx, set current j * R per unit length)
    geometry = 'trench_full' : planar 2-D (x, z) per unit LENGTH in y, the FULL pitch x in [0, pitch]
                           with pitch = d / f_open and two walls (no mirror symmetry); periodic in x.
                           nr = 2 * nr_half so dx equals the half-domain ('trench') value; every
                           r-weighting replaced by weight 1, projected width = pitch
  * all lengths in cm inside the solver; *_um names are micrometres"""
from __future__ import annotations
import math
import numpy as np
import scipy.ndimage as ndi
import skfmm
import contourpy

UM = 1e-4

GEOMETRIES = ("via", "trench", "trench_full")


class Grid:
    """Geometry of the unit cell: feature of width d = 2a, depth H, outer radius R
    (via: R = a / sqrt(f_open); trench: R = a / f_open half-domain; trench_full: R = pitch = d / f_open)."""

    def __init__(self, d_um, H_um, f_open, delta_um, dx_um=None, max_overburden_um=None, nbot=4,
                 margin_cells=6, geometry="via"):
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        self.geometry = geometry
        trench = geometry == "trench"
        full = geometry == "trench_full"
        self.planar = trench or full                # Cartesian: no 1/r terms anywhere
        self.periodic_x = full                      # x wraps at 0 / pitch (two walls, no mirror)
        self.d_um, self.H_um, self.f_open = float(d_um), float(H_um), float(f_open)
        a_um = d_um / 2.0
        dx_t = dx_um if dx_um is not None else min(max(a_um / 15.0, 0.75), 8.0)
        if full:
            # full pitch d / f_open; the half-domain cell count is doubled so dx is the half-domain dx
            R_half_um = a_um / f_open
            self.nr_half = max(int(round(R_half_um / dx_t)), 6)
            dx_um = R_half_um / self.nr_half
            self.nr = 2 * self.nr_half
            R_um = 2.0 * R_half_um                  # pitch, an integer number of cells
        else:
            self.nr_half = None
            R_um = a_um / f_open if trench else a_um / math.sqrt(f_open)
            self.nr = max(int(round(R_um / dx_t)), 6)
            dx_um = R_um / self.nr                  # R is an integer number of cells
        if max_overburden_um is None:
            max_overburden_um = 0.4 * H_um if d_um >= 500 else 80.0
        self.max_overburden_um = max_overburden_um
        self.dx_um, self.a_um, self.R_um = dx_um, a_um, R_um
        self.dx, self.a, self.R, self.H = dx_um * UM, a_um * UM, R_um * UM, H_um * UM
        self.x_c = 0.5 * self.R                     # centre of the full pitch (trench_full only)
        self.z_lo = -nbot * self.dx
        z_top = self.H + (max_overburden_um + delta_um) * UM + margin_cells * self.dx
        self.nz = int(math.ceil((z_top - self.z_lo) / self.dx))
        self.z_top = self.z_lo + self.nz * self.dx
        nr, nz, dx = self.nr, self.nz, self.dx
        self.rc = ((np.arange(nr) + 0.5) * dx)[:, None]                  # (nr, 1) cell centres in x
        self.zc = (self.z_lo + (np.arange(nz) + 0.5) * dx)[None, :]      # (1, nz)
        if self.planar:
            # planar 2-D, per unit LENGTH in y: no r factors anywhere (weight 1)
            self.vol = np.full((nr, 1), dx * dx)                         # cell cross-section (nr,1)
            self.AE = np.full((nr, 1), dx)                               # east face: dx x 1
            self.AZ = np.full((nr, 1), dx)                               # axial face: dx x 1
            self.area_cell = self.R                                      # projected half-cell / pitch width
            self.via_volume = 2.0 * self.a * self.H                      # trench cross-section per unit length
        else:
            self.vol = 2 * np.pi * self.rc * dx * dx                     # cell volume (nr,1)
            self.AE = 2 * np.pi * (np.arange(1, nr + 1) * dx)[:, None] * dx   # east face area
            self.AZ = 2 * np.pi * self.rc * dx                           # axial face area
            self.area_cell = math.pi * self.R ** 2                       # projected area of the unit cell
            self.via_volume = math.pi * self.a ** 2 * self.H
        # partial-cell weights of the original via/trench region r < a (x < a), 0 < z < H
        r_lo, r_hi = np.arange(nr) * dx, (np.arange(nr) + 1) * dx
        if full:
            # both walls: cells that overlap [x_c - a, x_c + a]
            x_lo, x_hi = self.x_c - self.a, self.x_c + self.a
            wr = np.clip((np.minimum(r_hi, x_hi) - np.maximum(r_lo, x_lo)) / dx, 0, 1)[:, None]
        else:
            wr = np.clip((np.minimum(r_hi, self.a) - r_lo) / dx, 0, 1)[:, None]
        z_lo_c, z_hi_c = (self.zc[0] - dx / 2), (self.zc[0] + dx / 2)
        wz = np.clip((np.minimum(z_hi_c, self.H) - np.maximum(z_lo_c, 0.0)) / dx, 0, 1)[None, :]
        # fill-fraction weights: every cell that overlaps the feature region counts (the deposit sits in the
        # electrolyte part of a wall-straddling cell), weighted by its overlap with 0 < z < H only
        self.via_w = (wr > 0) * wz

    def __repr__(self):
        return (f'Grid({self.geometry}, d={self.d_um}um, H={self.H_um}um, R={self.R_um:.1f}um, '
                f'dx={self.dx_um:.3f}um, nr={self.nr}, nz={self.nz})')


# ------------------------------------------------------------------------------------------- FMM helpers
def _mirror(a):
    return np.concatenate([a[::-1], a], axis=0)


def _halo_x(a, k):
    """Periodic halo along x: [last k rows | a | first k rows] (trench_full, no mirror)."""
    return np.concatenate([a[-k:], a, a[:k]], axis=0)


NARROW_CELLS = 8          # FMM narrow band half-width in cells (values outside keep their old/advected state)


def _halo_width(n):
    """Halo width in rows: NARROW_CELLS + 2, clipped to the grid (tiny test grids)."""
    return int(min(NARROW_CELLS + 2, n))


def _fmm_src(psi, periodic):
    """(padded field for skfmm, crop slice) - mirror across the symmetry plane, or a periodic x halo."""
    n = psi.shape[0]
    if periodic:
        k = _halo_width(n)
        return _halo_x(psi, k), slice(k, k + n)
    return _mirror(psi), slice(n, None)


def reinit(psi, dx, narrow=True, periodic=False):
    """Fast-marching re-initialisation to a signed distance function (mirrored across r = 0, or periodic in x
    when periodic=True: skfmm runs on a halo and the centre rows are returned).  With narrow=True only
    |psi| < NARROW_CELLS*dx is recomputed; farther cells keep their (sign-correct) old values."""
    src, sl = _fmm_src(psi, periodic)
    if not narrow:
        return skfmm.distance(src, dx=dx, order=2)[sl]
    d = skfmm.distance(src, dx=dx, order=2, narrow=NARROW_CELLS * dx)[sl]
    return np.where(np.ma.getmaskarray(d), psi, np.ma.getdata(d))


def extend(psi, field, dx, periodic=False):
    """Return (signed distance, field extended constant along the normals from the zero level set), both within
    the narrow band; outside it psi and field are returned unchanged.  periodic=True: x wraps (trench_full)."""
    src, sl = _fmm_src(psi, periodic)
    d, f = skfmm.extension_velocities(src, _fmm_src(field, periodic)[0], dx=dx, order=2,
                                      narrow=NARROW_CELLS * dx)
    d, f = d[sl], f[sl]
    if np.ma.is_masked(d):
        m = np.ma.getmaskarray(d)
        d = np.where(m, psi, d.data)
    else:
        d = np.asarray(d)
    if np.ma.is_masked(f):
        m = np.ma.getmaskarray(f)
        f = np.where(m, field, f.data)
    else:
        f = np.asarray(f)
    return d, f


# ------------------------------------------------------------------------------------------- initial psi
def initial_psi(g: Grid, corner_cells=1.0):
    """Signed distance to the Si surface: Si = {z < 0} U {r > a, z < H} (r = |x - x_c| for trench_full),
    corners rounded by ~corner_cells (periodic FMM for trench_full)."""
    z, a, H = g.zc, g.a, g.H
    if g.periodic_x:
        r = np.abs(g.rc - g.x_c)                   # full pitch: distance to the trench centre
    else:
        r = g.rc
    if g.f_open >= 0.999:                      # 1-D slab test geometry: the unit cell IS the via, no wall, no field
        return np.broadcast_to(z, (g.nr, g.nz)).copy()
    d_floor = np.hypot(np.maximum(r - a, 0.0), z)                              # segment z=0, |r|<=a
    d_wall = np.hypot(r - a, z - np.clip(z, 0.0, H))                           # segment r=a, 0<=z<=H
    d_top = np.hypot(np.maximum(a - r, 0.0), z - H)                            # segment z=H, r>=a
    dist = np.minimum(np.minimum(d_floor, d_wall), d_top)
    inside_si = (z < 0) | ((r > a) & (z < H))
    psi = np.where(inside_si, -dist, dist)
    psi = np.broadcast_to(psi, (g.nr, g.nz)).copy()
    rc = corner_cells * g.dx
    if rc > 0:
        # opening rounds the convex (floor) corners of the electrolyte, closing rounds the concave (mouth) corner
        per = g.periodic_x
        psi = reinit(psi - rc, g.dx, narrow=False, periodic=per) + rc
        psi = reinit(psi + rc, g.dx, narrow=False, periodic=per) - rc
        psi = reinit(psi, g.dx, narrow=False, periodic=per)
    return psi


# ------------------------------------------------------------------------------------------- geometry ops
def _dx_periodic(dx_or_g, periodic=None):
    """Accept either a grid (Grid -> its dx and periodic_x) or a plain dx with an explicit periodic flag."""
    if isinstance(dx_or_g, Grid):
        per = dx_or_g.periodic_x if periodic is None else bool(periodic)
        return dx_or_g.dx, per
    return float(dx_or_g), bool(periodic)


def _pad(psi, dx_or_g, periodic=None):
    """1-cell pad: edge (mirror) in z, periodic x wrap when the grid says so."""
    p = np.pad(psi, 1, mode='edge')
    if _dx_periodic(dx_or_g, periodic)[1]:
        p[0, 1:-1] = psi[-1, :]
        p[-1, 1:-1] = psi[0, :]
    return p


def grad_mag(psi, dx, periodic=None):
    dx, per = _dx_periodic(dx, periodic)
    p = _pad(psi, dx, per)                           # edge pad == mirror at r=0 / R for cell-centred data
    gr = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
    gz = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    return np.hypot(gr, gz)


def advect(psi, v, dt, dx, periodic=None):
    """First-order Godunov upwind step of  psi_t + v |grad psi| = 0  (v > 0: metal grows into the electrolyte,
    psi decreases).  v is an extension velocity field (nr, nz).  x wraps when the grid is periodic in x."""
    dx, per = _dx_periodic(dx, periodic)
    p = _pad(psi, dx, per)
    c = p[1:-1, 1:-1]
    dxm, dxp = (c - p[:-2, 1:-1]) / dx, (p[2:, 1:-1] - c) / dx
    dzm, dzp = (c - p[1:-1, :-2]) / dx, (p[1:-1, 2:] - c) / dx
    gpos = np.sqrt(np.maximum(dxm, 0) ** 2 + np.minimum(dxp, 0) ** 2 + np.maximum(dzm, 0) ** 2 + np.minimum(dzp, 0) ** 2)
    gneg = np.sqrt(np.minimum(dxm, 0) ** 2 + np.maximum(dxp, 0) ** 2 + np.minimum(dzm, 0) ** 2 + np.maximum(dzp, 0) ** 2)
    return psi - dt * (np.maximum(v, 0) * gpos + np.minimum(v, 0) * gneg)


def electrolyte_fraction(psi, g: Grid):
    """Sub-cell electrolyte volume fraction of every cell (sharp interface, linear ramp across one cell width
    measured along the normal)."""
    p = _pad(psi, g)
    gr = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * g.dx)
    gz = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * g.dx)
    n = np.hypot(gr, gz) + 1e-30
    l1 = np.maximum((np.abs(gr) + np.abs(gz)) / n, 1.0)   # |nr| + |nz| >= 1 for a unit normal (guards flat regions)
    return np.clip(0.5 + psi / (g.dx * l1), 0.0, 1.0)


def curvature_div_n(psi, g: Grid):
    """div(grad psi/|grad psi|): axisymmetric  ~ psi_rr + psi_r/r + psi_zz for a signed distance,
    plain Cartesian psi_xx + psi_zz for the trench / trench_full (no 1/r term)."""
    dx = g.dx
    p = _pad(psi, g)
    c = p[1:-1, 1:-1]
    prr = (p[2:, 1:-1] - 2 * c + p[:-2, 1:-1]) / dx ** 2
    pzz = (p[1:-1, 2:] - 2 * c + p[1:-1, :-2]) / dx ** 2
    if g.planar:
        return prr + pzz
    pr = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
    return prr + pzz + pr / g.rc


def column_crossing(psi_col, zc, upper=True):
    """z of the topmost (upper=True) or lowest sign change psi<0 -> psi>0 along z in one column; None if none."""
    s = (psi_col[:-1] <= 0) & (psi_col[1:] > 0)
    idx = np.flatnonzero(s)
    if idx.size == 0:
        return None, None
    j = idx[-1] if upper else idx[0]
    t = psi_col[j] / (psi_col[j] - psi_col[j + 1])
    return j + t, zc[j] + t * (zc[j + 1] - zc[j])


# ------------------------------------------------------------------------------------------- voids
def label_regions(psi, g: Grid, jt, conn_thresh):
    """Electrolyte connectivity.  conn_thresh: cells with psi > conn_thresh count as electrolyte (slightly
    negative -> a gap thinner than the grid is still seen).  Returns (open_mask, void_mask, labels).
    Periodic in x (trench_full): the domain is tripled along x so a cavity straddling the seam is one
    region; the centre copy is returned."""
    E = psi > conn_thresh
    E[:, jt + 1:] = False
    if g.periodic_x:
        nr = E.shape[0]
        lab, n = ndi.label(np.concatenate([E, E, E], axis=0))
        if n == 0:
            z = np.zeros_like(E)
            return z, z, np.zeros_like(E, dtype=lab.dtype)
        top = np.unique(lab[:, jt])              # column jt of every copy: the open label set
        top = top[top > 0]
        open_m = np.isin(lab, top)[nr:2 * nr]
        return open_m, E & ~open_m, lab[nr:2 * nr]
    lab, n = ndi.label(E)
    if n == 0:
        z = np.zeros_like(E)
        return z, z, lab
    top = np.unique(lab[:, jt])
    top = top[top > 0]
    open_m = np.isin(lab, top)
    return open_m, E & ~open_m, lab


def contour_lines(psi, g: Grid, mask_val=None, tol_um=0.2, max_pts=250):
    """psi = 0 polylines (r, z in um, rounded to 0.1) on the r >= 0 half plane, RDP-simplified.
    Periodic in x (trench_full): the x pad wraps, so lines cross the seam continuously; x stays 0..R."""
    nr, nz = psi.shape
    x = np.concatenate([[0.0], g.rc[:, 0], [g.R]]) / UM
    y = g.zc[0] / UM
    zz = np.pad(psi, ((1, 1), (0, 0)), mode='edge').T                 # (nz, nr+2)
    if g.periodic_x:
        zz[:, 0], zz[:, -1] = psi[-1, :], psi[0, :]                   # wrap instead of edge-replicate
    gen = contourpy.contour_generator(x, y, zz, name='serial', line_type=contourpy.LineType.Separate)
    out = []
    for ln in gen.lines(0.0):
        ln = np.asarray(ln)
        if len(ln) < 2:
            continue
        ln = rdp(ln, tol_um)
        if len(ln) > max_pts:
            idx = np.unique(np.linspace(0, len(ln) - 1, max_pts).round().astype(int))
            ln = ln[idx]
        out.append(np.round(ln, 1))
    return out


def rdp(pts, tol):
    """Ramer-Douglas-Peucker polyline simplification (iterative)."""
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        seg = p1 - p0
        L = np.hypot(*seg)
        sub = pts[i0 + 1:i1]
        if L < 1e-12:
            dist = np.hypot(*(sub - p0).T)
        else:
            dist = np.abs(seg[0] * (sub[:, 1] - p0[1]) - seg[1] * (sub[:, 0] - p0[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i0 + 1 + k
            keep[m] = True
            stack += [(i0, m), (m, i1)]
    return pts[keep]


# ------------------------------------------------------------------------------------------- interface values
def band_average(psi, w, q, g: Grid, half_width_cells=5.0):
    """Interface value of a cell quantity q (e.g. current density) from the smoothed-boundary band.

    In the diffuse interface the sink is spread over the band, and in the transport-limited regime (large
    Damkoehler number across the band) the cell-wise k*c is concentrated at its electrolyte edge, so q at psi = 0 is
    NOT the interface flux.  The interface flux is the weight-averaged value along the normal:
        q_s(x_s) = sum_p w_p q_p / sum_p w_p   over cells p that project (closest-point) onto the same interface
    location x_s.  Cells are projected with x_s = x - psi n, binned onto the grid and smoothed over 3x3 bins.
    Returns an (nr, nz) field defined near the interface (0 elsewhere)."""
    dx = g.dx
    p = _pad(psi, g)
    gr = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
    gz = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    gm = np.hypot(gr, gz) + 1e-30
    m = (w > 0) & (np.abs(psi) < half_width_cells * dx)
    ii, jj = np.nonzero(m)
    nr_, nz_ = psi.shape
    r_s = g.rc[ii, 0] - psi[ii, jj] * gr[ii, jj] / gm[ii, jj]
    if g.periodic_x:
        r_s = np.mod(r_s, g.R)                     # full pitch: signed projection folded into [0, pitch)
    z_s = g.zc[0, jj] - psi[ii, jj] * gz[ii, jj] / gm[ii, jj]
    # bilinear splatting of (w q, w) from the projected points onto the four surrounding cell centres
    if g.periodic_x:
        fi = r_s / dx - 0.5
    else:
        fi = np.abs(r_s) / dx - 0.5
    fj = (z_s - g.z_lo) / dx - 0.5
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    ti, tj = fi - i0, fj - j0
    wv = w[ii, jj]
    wq = wv * q[ii, jj]
    num = np.zeros(nr_ * nz_)
    den = np.zeros(nr_ * nz_)
    for di, wi in ((0, 1 - ti), (1, ti)):
        for dj, wj in ((0, 1 - tj), (1, tj)):
            if g.periodic_x:
                ib = (i0 + di) % nr_                       # x wraps
            else:
                ib = np.clip(i0 + di, 0, nr_ - 1)           # cell -1 mirrors onto cell 0 (clip), cell nr clips too
            jb = np.clip(j0 + dj, 0, nz_ - 1)
            flat = ib * nz_ + jb
            num += np.bincount(flat, weights=wq * wi * wj, minlength=nr_ * nz_)
            den += np.bincount(flat, weights=wv * wi * wj, minlength=nr_ * nz_)
    smode = 'wrap' if g.periodic_x else 'reflect'
    num = ndi.uniform_filter(num.reshape(nr_, nz_), size=3, mode=smode)
    den = ndi.uniform_filter(den.reshape(nr_, nz_), size=3, mode=smode)
    # keep bins that hold a reasonably complete normal column (a full column has sum w ~ vol_p, 3x3-averaged;
    # vol = 2 pi r dx^2 for the via, dx^2 per unit length for the trench)
    full = g.vol
    ok = den > 0.12 * full
    out = np.zeros_like(num)
    out[ok] = num[ok] / den[ok]
    return out
