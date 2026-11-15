"""Steady Stokes / Oseen flow over a periodic trench on a staggered MAC grid.

Drop-in alternative to :func:`platesim.trenchflow.cavity_lbm.solve_flow`: the same geometry
(:class:`~platesim.trenchflow.geometry.CavityGrid`), the same [iz, ix] cell arrays in cm/s, but the
steady flow is obtained from ONE sparse saddle-point solve instead of a time-marching lattice.

Discretisation
- Staggered MAC grid on the Cartesian cells of `grid`. u lives on x-faces (nz, nx), one face per row
  because x is periodic; w on z-faces (nz+1, nx); p in the open cells.
- Sharp solid mask (default): a face between a fluid cell and a solid cell is not an unknown and its
  normal velocity is exactly 0. The tangential no-slip at a horizontal wall (spacing dx/2 from the
  face) enters the viscous stencil as the ghost value u_ghost = -u; a face straddling a step corner
  takes the fractional weight 0, 0.5 or 1 (the average over its two cells), which is what makes the
  stencil exactly reflection-symmetric. At a vertical wall the wall plane coincides with the
  neighbouring x-face, so it is a plain Dirichlet node. The momentum stencil stays compact.
- Pressure lives in every open cell; its null space is fixed by replacing the continuity row of ONE open
  cell PER CONNECTED FLUID REGION with p = 0 (a pure gauge pin, no continuity equation is lost), which
  makes the matrix non-singular for SuperLU. Regions are 4-connected with a periodic x seam, because a
  sealed void pocket (a disconnected fluid bubble the feature engine can leave inside the deposit) has
  its own constant-pressure null space; without its own pin that block is singular. A region with no
  contact with the lid has no driving term and comes out at u = w = 0 to round-off. One region -> the
  pinned cell is the same as the single-gauge code, so those results stay bit-identical.
- Non-dimensionalisation: x by dx, u by U, and p by nu*U/dx, with U the lid speed |u_top| (or
  max(|u_top|, max|f| dx^2/nu) when a body force is given, so a buoyancy-only solve has a scale too).
  The viscosity then drops out of the Stokes momentum equation entirely (as it must for Stokes
  similarity), and the assembled matrix has O(1) entries: A = -Laplacian, G = gradient, D = divergence.
- Oseen: `oseen_passes` Picard iterations add Re_dx * (U_prev . grad) u, Re_dx = U * dx / nu.
- Optional Brinkman penalisation -(nu/K)(1-H)u with a smoothed solid fraction H; faces are then kept
  as unknowns unless both adjacent cells are solid (diffuse mask, no ghost values).
- Optional body force `body_force=(fx, fz)` in cm/s2 (force per unit mass, cell centres): averaged onto
  the u and w faces over the fluid adjacent cells and added to the momentum right-hand side. The
  velocity scale is then U = max(|u_top|, max|f| dx^2 / nu), so a buoyancy-only solve (shear 0) keeps
  the assembled matrix O(1); the dimensionless face force is f dx^2 / (nu U). The system is linear in
  the right-hand side, so shear and body force superpose exactly, and a force that is the gradient of a
  pressure (f parallel to gravity, varying only along it, in a closed box) is absorbed by p: no flow.

The top boundary (z = height_um) is always the moving lid with u_x = shear_rate_s *
(height_um - deposit_um) * 1e-4 cm/s and u_z = 0, matching cavity_lbm._scaling / the LBM lid row.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.linalg import spsolve

from .geometry import CavityGrid, FlowField

_TINY = 1e-300

# 4-connectivity: N, S, E, W of a cell
_STRUCT4 = np.array([[False, True, False], [True, True, True], [False, True, False]])


def _fluid_regions(fluid):
    """Label the 4-connected fluid regions, PERIODIC in x. Returns ``(labels, n_regions)``.

    ``scipy.ndimage.label`` only sees the in-grid neighbours, so labels that meet across the x seam
    (a cell in column nx-1 is a left/right neighbour of the same row in column 0) are merged with a
    union-find. Labels are 1..n_regions, 0 on solid cells.
    """
    lab, n = ndimage.label(fluid, structure=_STRUCT4)
    n = int(n)
    if n < 2:
        return lab, n
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    merged = False
    for a, b in zip(lab[:, 0].tolist(), lab[:, -1].tolist()):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                merged = True
    if not merged:
        return lab, n
    remap = np.zeros(n + 1, np.int64)
    nxt = 0
    for i in range(1, n + 1):
        r = find(i)
        if remap[r] == 0:
            nxt += 1
            remap[r] = nxt
        remap[i] = remap[r]
    return remap[lab], int(nxt)


def _region_pins(id_p, labels, n_regions):
    """Pressure unknown of the gauge cell of every connected fluid region (exactly one pin per region).

    The region that owns the legacy gauge cell is pinned at that same cell. The single-gauge code used
    ``id_p == flat index of the first open cell`` as the pin, which is the (f0+1)-th open cell because
    ``id_p`` is only contiguous from 0 -- keeping it means a one-region grid assembles the identical
    matrix and stays bit-identical. Every other region is pinned at its FIRST cell in row-major order
    (bottom row up, then +x), which is where the legacy cell sits when the fluid is one region.
    """
    if n_regions <= 0:
        return np.zeros(0, np.int64)
    flat_lab = labels.reshape(-1)
    order = np.flatnonzero(flat_lab)                     # ascending flat index of the open cells
    ids = id_p.reshape(-1)[order]                        # 0 .. n_open-1 in that same order
    first = np.unique(flat_lab[order], return_index=True)[1]
    pins = {int(flat_lab[order[i]]): int(ids[i]) for i in first}
    f0 = int(order[0])                                   # legacy pin: the cell with id_p == f0
    if f0 < ids.size:
        pins[int(flat_lab[order[f0]])] = f0
    return np.array([pins[lab] for lab in sorted(pins)], np.int64)


def _solid_fraction(solid, passes: int = 2):
    """Binomial-smoothed solid indicator H in [0, 1] (1 in solid, ~0 in the bulk fluid)."""
    h = solid.astype(np.float64)
    for _ in range(int(passes)):
        h = 0.5 * h + 0.25 * (np.roll(h, 1, axis=1) + np.roll(h, -1, axis=1))
        h = 0.5 * h + 0.25 * (np.vstack([h[:1], h[:-1]]) + np.vstack([h[1:], h[-1:]]))
    return np.clip(h, 0.0, 1.0)


def _layout(grid: CavityGrid, diffuse: bool = False) -> dict:
    """Index maps for the MAC faces and the open (pressure) cells.

    u face (k, j) sits at x = j*dx, z = z_um[k], between cells (k, j-1) and (k, j); it is an unknown
    when the two cells are not both solid (sharp: when both are fluid).
    w face (m, j) sits at x = x_um[j], z = z_um[0] + (m - 0.5)*dx, between cells (m-1, j) and (m, j);
    only m = 1 .. nz-1 can be an unknown (the top and bottom z-faces are impermeable walls).
    """
    nz, nx = grid.shape
    solid = np.ascontiguousarray(grid.solid, dtype=bool)
    fluid = ~solid
    if diffuse:
        open_u = ~(solid & np.roll(solid, 1, axis=1))
        open_w = np.zeros((nz + 1, nx), bool)
        open_w[1:nz] = ~(solid[:-1] & solid[1:])
    else:
        open_u = fluid & np.roll(fluid, 1, axis=1)
        open_w = np.zeros((nz + 1, nx), bool)
        open_w[1:nz] = fluid[:-1] & fluid[1:]

    id_u = np.full((nz, nx), -1, np.int64)
    id_u[open_u] = np.arange(np.count_nonzero(open_u), dtype=np.int64)
    id_w = np.full((nz + 1, nx), -1, np.int64)
    id_w[open_w] = np.arange(np.count_nonzero(open_w), dtype=np.int64)

    # pressure unknown in every open cell; the gauge is fixed later by replacing the continuity row
    # of ONE cell per connected fluid region with p = 0 (see _assemble), so no continuity equation is
    # lost and every region's constant-pressure null space is removed.
    id_p = np.full((nz, nx), -1, np.int64)
    open_p = fluid.copy()
    id_p[open_p] = np.arange(np.count_nonzero(open_p), dtype=np.int64)
    labels, n_regions = _fluid_regions(open_p)
    pins = _region_pins(id_p, labels, n_regions)

    nu = int(np.count_nonzero(open_u))
    nw = int(np.count_nonzero(open_w))
    np_ = int(np.count_nonzero(open_p))
    idx_m = (np.arange(nx) - 1) % nx
    idx_p = (np.arange(nx) + 1) % nx
    return dict(nz=nz, nx=nx, dx=float(grid.dx_um), solid=solid, fluid=fluid,
                open_u=open_u, open_w=open_w, open_p=open_p,
                id_u=id_u, id_w=id_w, id_p=id_p, nu=nu, nw=nw, np=np_,
                off_w=nu, off_p=nu + nw, size=nu + nw + np_,
                labels=labels, n_regions=int(n_regions),
                pin_rows=np.asarray(pins, np.int64) + (nu + nw),
                idx_m=idx_m, idx_p=idx_p)


def _assemble(lay: dict, u_lid: float, brinkman_K, rec: float | None, uprev, body_force=None):
    """Assemble the sparse saddle-point system [[A, G], [D, 0]] (row-major unknown order u, w, p).

    All coefficients are dimensionless (lengths in dx, velocities in U, pressure in nu*U/dx), so A is the
    plain discrete -Laplacian and G / D are the plain gradient / divergence. The lid enters through the
    ghost value u_ghost = 2*u_lid - u (wall at dx/2 above the top row); a stationary horizontal wall
    gives u_ghost = -u, weighted by the fraction of the face that actually faces solid (0, 0.5 or 1).
    ``body_force=(fx_face, fz_face)`` are already dimensionless face forces (see _face_body_force) and
    are added to the momentum right-hand side.
    """
    nz, nx = lay["nz"], lay["nx"]
    solid = lay["solid"]
    id_u, id_w, id_p = lay["id_u"], lay["id_w"], lay["id_p"]
    open_u, open_w = lay["open_u"], lay["open_w"]
    off_w, off_p = lay["off_w"], lay["off_p"]
    idx_m, idx_p = lay["idx_m"], lay["idx_p"]
    diffuse = brinkman_K is not None

    rows, cols, vals = [], [], []

    def emit(r, c, v):
        r, c, v = np.broadcast_arrays(np.asarray(r), np.asarray(c), np.asarray(v, np.float64))
        m = (r >= 0) & (c >= 0)
        rows.append(np.asarray(r[m], np.int64))
        cols.append(np.asarray(c[m], np.int64))
        vals.append(np.asarray(v[m], np.float64))

    def gcol(local, off):
        return np.where(local >= 0, local + off, -1)

    # ---------------- x-momentum rows (one per open u face) ----------------
    iu_l, iu_r = id_u[:, idx_m], id_u[:, idx_p]
    iu_u = np.full((nz, nx), -1, np.int64)
    iu_u[:-1] = id_u[1:]                       # u face at z + dx (same x position)
    iu_d = np.full((nz, nx), -1, np.int64)
    iu_d[1:] = id_u[:-1]
    top = np.zeros((nz, nx), bool)
    top[-1] = True
    bot = np.zeros((nz, nx), bool)
    bot[0] = True
    su_l = np.zeros((nz, nx), bool)            # solid just above the face, left column
    su_l[:-1] = solid[1:][:, idx_m]
    su_r = np.zeros((nz, nx), bool)            # solid just above the face, right column
    su_r[:-1] = solid[1:]
    sd_l = np.zeros((nz, nx), bool)
    sd_l[1:] = solid[:-1][:, idx_m]
    sd_r = np.zeros((nz, nx), bool)
    sd_r[1:] = solid[:-1]
    # sub-cell wall position: the face is shared by two columns, so the ghost weight is the average.
    # cu = 1 puts the wall at dx/2 (full ghost u_ghost = -u), cu = 0.5 a half-covered face, cu = 0 no
    # wall.  Keeping both columns makes the stencil exactly reflection-symmetric at step corners.
    cu_up = 0.5 * (su_l.astype(np.float64) + su_r)
    cu_dn = 0.5 * (sd_l.astype(np.float64) + sd_r)
    if diffuse:
        cu_up = np.zeros((nz, nx))
        cu_dn = np.zeros((nz, nx))
    cu_up[top] = 1.0                           # moving lid at z = height_um
    cu_dn[bot] = 1.0                           # no-slip floor outside the grid

    emit(id_u, id_u, np.where(open_u, 4.0 + cu_up + cu_dn, 0.0))
    emit(id_u, iu_l, -1.0 * open_u)
    emit(id_u, iu_r, -1.0 * open_u)
    emit(id_u, iu_u, -1.0 * open_u)
    emit(id_u, iu_d, -1.0 * open_u)
    emit(id_u, gcol(id_p, off_p), open_u)                     # + dp/dx (cell j)
    emit(id_u, gcol(id_p[:, idx_m], off_p), -1.0 * open_u)    # - dp/dx (cell j-1)

    # ---------------- z-momentum rows (one per open w face) ----------------
    iw_l, iw_r = id_w[:, idx_m], id_w[:, idx_p]
    iw_u = np.full((nz + 1, nx), -1, np.int64)
    iw_u[:-1] = id_w[1:]
    iw_d = np.full((nz + 1, nx), -1, np.int64)
    iw_d[1:] = id_w[:-1]
    sw_l = np.zeros((nz + 1, nx), bool)
    sw_l[1:nz] = solid[1:, :][:, idx_m]        # cell (m, j-1)
    sw_l2 = np.zeros((nz + 1, nx), bool)
    sw_l2[1:nz] = solid[:-1, :][:, idx_m]      # cell (m-1, j-1)
    sw_r = np.zeros((nz + 1, nx), bool)
    sw_r[1:nz] = solid[1:, :][:, idx_p]
    sw_r2 = np.zeros((nz + 1, nx), bool)
    sw_r2[1:nz] = solid[:-1, :][:, idx_p]
    # w face in x: walls sit at the cell boundary, dx/2 away; average over the face's two cells
    cw_l = 0.5 * (sw_l.astype(np.float64) + sw_l2)
    cw_r = 0.5 * (sw_r.astype(np.float64) + sw_r2)
    if diffuse:
        cw_l = np.zeros((nz + 1, nx))
        cw_r = np.zeros((nz + 1, nx))
    rw = gcol(id_w, off_w)
    emit(rw, rw, np.where(open_w, 4.0 + cw_l + cw_r, 0.0))
    emit(rw, gcol(iw_l, off_w), -1.0 * open_w)
    emit(rw, gcol(iw_r, off_w), -1.0 * open_w)
    emit(rw, gcol(iw_u, off_w), -1.0 * open_w)
    emit(rw, gcol(iw_d, off_w), -1.0 * open_w)
    pw_up = np.full((nz + 1, nx), -1, np.int64)
    pw_up[:nz] = gcol(id_p, off_p)
    pw_dn = np.full((nz + 1, nx), -1, np.int64)
    pw_dn[1:] = gcol(id_p, off_p)
    emit(rw, pw_up, open_w)                                   # + dp/dz (cell m)
    emit(rw, pw_dn, -1.0 * open_w)                            # - dp/dz (cell m-1)

    # ---------------- continuity rows (one per open cell) ----------------
    rp = gcol(id_p, off_p)
    emit(rp, gcol(id_u, 0), -1.0 * lay["fluid"])              # - u_left
    emit(rp, gcol(id_u[:, idx_p], 0), 1.0 * lay["fluid"])     # + u_right
    emit(rp, gcol(id_w[:nz], off_w), -1.0 * lay["fluid"])     # - w_bottom
    emit(rp, gcol(id_w[1:], off_w), 1.0 * lay["fluid"])       # + w_top

    if diffuse:
        kd = float(brinkman_K) / lay["dx"] ** 2
        hc = _solid_fraction(solid)
        hu = 0.5 * (hc + hc[:, idx_m])
        hw = np.zeros((nz + 1, nx))
        hw[1:nz] = 0.5 * (hc[:-1] + hc[1:])
        emit(id_u, id_u, np.where(open_u, (1.0 - hu) / max(kd, 1e-12), 0.0))
        emit(rw, rw, np.where(open_w, (1.0 - hw) / max(kd, 1e-12), 0.0))

    if rec is not None and uprev is not None:
        _add_oseen(emit, lay, uprev, rec)

    b = np.zeros(lay["size"])
    b[id_u[top & open_u]] = 2.0 * u_lid
    if body_force is not None:
        fx_f, fz_f = (np.asarray(a, np.float64) for a in body_force)
        b[id_u[open_u]] += fx_f[open_u]
        b[id_w[open_w] + off_w] += fz_f[open_w]

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    # pressure gauge: replace the continuity row of the pinned cell of EVERY fluid region with p = 0
    pin_rows = lay["pin_rows"]
    if pin_rows.size:
        drop = np.zeros(lay["size"], bool)
        drop[pin_rows] = True
        keep = ~drop[r]
        r = np.concatenate([r[keep], pin_rows])
        c = np.concatenate([c[keep], pin_rows])
        v = np.concatenate([v[keep], np.ones(pin_rows.size)])
    A = sp.coo_matrix((v, (r, c)), shape=(lay["size"], lay["size"])).tocsr()
    return A, b


def _add_oseen(emit, lay: dict, uprev, rec: float):
    """Add the Picard linearisation rec * (U_prev . grad) u of the convective term.

    Face values of the previous field are interpolated to the staggered location of each unknown
    (4-point average); the advecting gradient uses central differences, with blocked faces acting as
    zero (the sharp no-slip value).
    """
    nz, nx = lay["nz"], lay["nx"]
    id_u, id_w = lay["id_u"], lay["id_w"]
    open_u, open_w = lay["open_u"], lay["open_w"]
    off_w = lay["off_w"]
    idx_m, idx_p = lay["idx_m"], lay["idx_p"]
    up, wp = uprev

    def gcol(local, off):
        return np.where(local >= 0, local + off, -1)

    # --- u rows ---
    iu_l, iu_r = id_u[:, idx_m], id_u[:, idx_p]
    iu_u = np.full((nz, nx), -1, np.int64)
    iu_u[:-1] = id_u[1:]
    iu_d = np.full((nz, nx), -1, np.int64)
    iu_d[1:] = id_u[:-1]
    wu = 0.25 * (wp[:-1] + wp[1:] + wp[:-1][:, idx_m] + wp[1:][:, idx_m])
    a = 0.5 * rec * up
    bz = 0.5 * rec * wu
    emit(id_u, gcol(iu_r, 0), a * open_u)
    emit(id_u, gcol(iu_l, 0), -a * open_u)
    emit(id_u, gcol(iu_u, 0), bz * open_u)
    emit(id_u, gcol(iu_d, 0), -bz * open_u)

    # --- w rows ---
    iw_l, iw_r = id_w[:, idx_m], id_w[:, idx_p]
    iw_u = np.full((nz + 1, nx), -1, np.int64)
    iw_u[:-1] = id_w[1:]
    iw_d = np.full((nz + 1, nx), -1, np.int64)
    iw_d[1:] = id_w[:-1]
    U = np.zeros((nz + 1, nx))
    U[1:] = up                                 # U[m] = u of cell m-1
    Ud = U
    Uu = np.vstack([U[1:], U[-1:]])
    Udr = U[:, idx_p]
    Uur = np.vstack([U[1:][:, idx_p], U[-1:][:, idx_p]])
    uw = 0.25 * (Ud + Uu + Udr + Uur)
    a2 = 0.5 * rec * uw
    b2 = 0.5 * rec * wp
    rw = gcol(id_w, off_w)
    emit(rw, gcol(iw_r, off_w), a2 * open_w)
    emit(rw, gcol(iw_l, off_w), -a2 * open_w)
    emit(rw, gcol(iw_u, off_w), b2 * open_w)
    emit(rw, gcol(iw_d, off_w), -b2 * open_w)


def _face_body_force(fx, fz, lay: dict):
    """Average a cell-centred body force (force per unit mass) onto the MAC faces, fluid cells only.

    u face (k, j) is shared by cells (k, j-1) and (k, j); w face (m, j) by cells (m-1, j) and (m, j).
    A face with a solid neighbour takes the value of the fluid one; a face with no fluid neighbour
    (never an unknown) is 0. Returns ``(fx_face (nz, nx), fz_face (nz+1, nx))``.
    """
    nz, nx = lay["nz"], lay["nx"]
    fluid = lay["fluid"].astype(np.float64)
    idx_m = lay["idx_m"]
    fx = np.where(lay["fluid"], np.asarray(fx, np.float64), 0.0)
    fz = np.where(lay["fluid"], np.asarray(fz, np.float64), 0.0)

    wu_l, wu_r = fluid[:, idx_m], fluid
    fx_f = (fx[:, idx_m] * wu_l + fx * wu_r) / np.maximum(wu_l + wu_r, 1.0)

    num = np.zeros((nz + 1, nx))
    den = np.zeros((nz + 1, nx))
    num[1:] += fz * fluid                 # cell m-1 feeds face m
    den[1:] += fluid
    num[:-1] += fz * fluid                # cell m feeds face m
    den[:-1] += fluid
    fz_f = num / np.maximum(den, 1.0)
    return fx_f, fz_f


def solve_stokes(grid: CavityGrid, shear_rate_s: float, nu_cm2_s: float = 0.0114,
                 oseen_passes: int = 0, brinkman_K: float | None = None,
                 body_force=None) -> FlowField:
    """Steady Stokes (or Picard-Oseen) flow over `grid`, driven by the far-field shear.

    Same interface and conventions as :func:`~platesim.trenchflow.cavity_lbm.solve_flow`: ux, uz are
    cell-centred cm/s arrays of shape (nz, nx), exactly 0 in solid cells; z increases upward, x is
    periodic; the lid at z = height_um moves with shear_rate_s * (height_um - deposit_um) * 1e-4 cm/s
    (negative shear -> lid in -x -> the mirrored flow).

    ``body_force=(fx, fz)``: optional arrays (nz, nx) in cm/s2, force per unit mass at the cell centres,
    averaged onto the fluid MAC faces and added to the momentum right-hand side (Boussinesq buoyancy).
    With it the velocity scale becomes U = max(|u_top|, max|f| dx^2/nu) -- a buoyancy-only solve has no
    lid speed -- and the dimensionless face force is f dx^2/(nu U), so the matrix stays O(1) and both
    drives superpose exactly (the system is linear in its right-hand side).

    info["u_face_x"] (nz, nx) and info["u_face_z"] (nz+1, nx) are the staggered face velocities in
    cm/s; they are discretely divergence-free in the open cells. info["method"] == "stokes_mac".
    info["n_fluid_regions"] is the number of 4-connected fluid regions (periodic in x) and every one of
    them gets its own pressure gauge, so a sealed void pocket does not make the matrix singular.
    nu_cm2_s only enters the Oseen term, the body-force scaling and the reported Reynolds number: the
    Stokes velocity field is independent of viscosity (it only sets the pressure scale).
    """
    t0 = time.time()
    nz, nx = grid.shape
    solid = np.ascontiguousarray(grid.solid, dtype=bool)
    fluid = ~solid
    shear = float(shear_rate_s)
    h_cm = max(float(grid.height_um) - float(grid.deposit_um), 0.0) * 1e-4
    u_top = shear * h_cm                                    # cm/s at the lid (signed)
    dx_cm = float(grid.dx_um) * 1e-4

    body = None
    f_body_max = 0.0
    if body_force is not None:
        fx_in, fz_in = body_force
        fx_in = np.asarray(fx_in, np.float64)
        fz_in = np.asarray(fz_in, np.float64)
        if fx_in.shape != (nz, nx) or fz_in.shape != (nz, nx):
            raise ValueError(f"body_force arrays must have shape {grid.shape}, got "
                             f"{fx_in.shape} and {fz_in.shape}")
        if not (np.isfinite(fx_in).all() and np.isfinite(fz_in).all()):
            raise ValueError("body_force contains non-finite values")
        if nu_cm2_s <= 0:
            raise ValueError("body_force requires nu_cm2_s > 0 (it sets the buoyant velocity scale)")
        body = (np.where(fluid, fx_in, 0.0), np.where(fluid, fz_in, 0.0))
        if fluid.any():
            f_body_max = float(np.hypot(body[0], body[1])[fluid].max())

    u_body = f_body_max * dx_cm ** 2 / nu_cm2_s if nu_cm2_s > 0 else 0.0
    u_scale = max(abs(u_top), u_body)
    re = u_scale * h_cm / nu_cm2_s if nu_cm2_s > 0 else float("inf")
    n_rows = int(np.count_nonzero(grid.z_um >= grid.deposit_um))
    common = dict(tau=None, u_lid_lb=None, re=float(re), re_sim=None, max_mach=None, nu_lb=None,
                  re_clamped=None, n_fluid_rows=n_rows, method="stokes_mac",
                  scheme="MAC staggered, sharp no-slip, direct sparse SuperLU",
                  dx_um=float(grid.dx_um), u_scale_cm_s=float(u_scale),
                  body_force_max_cm_s2=float(f_body_max), buoyant_velocity_scale_cm_s=float(u_body))

    if not np.isfinite(shear) or u_scale == 0.0 or not fluid.any():
        z = np.zeros((nz, nx))
        return FlowField(grid=grid, ux=z.copy(), uz=z.copy(), shear_rate_s=shear, converged=True,
                         info=dict(common, steps=0, wall_s=time.time() - t0, residual=0.0,
                                   u_top_cm_s=float(u_top), u_face_x=z.copy(),
                                   u_face_z=np.zeros((nz + 1, nx)), div_max_rel=0.0,
                                   n_fluid_regions=int(_fluid_regions(fluid)[1]),
                                   converged=True))

    diffuse = brinkman_K is not None
    lay = _layout(grid, diffuse=diffuse)
    rec = u_scale * lay["dx"] * 1e-4 / nu_cm2_s if nu_cm2_s > 0 else 0.0
    u_lid = u_top / u_scale                                  # +-1 (or smaller with buoyancy only)
    npass = max(0, int(oseen_passes))
    if body is not None:
        fx_f, fz_f = _face_body_force(body[0], body[1], lay)
        scale_f = dx_cm ** 2 / (nu_cm2_s * u_scale)
        face_force = (fx_f * scale_f, fz_f * scale_f)
    else:
        face_force = None

    uprev = None
    converged, residual, steps = True, 0.0, 0
    uf = np.zeros((nz, nx))
    wf = np.zeros((nz + 1, nx))
    for it in range(npass + 1):
        A, b = _assemble(lay, u_lid, brinkman_K, rec if it else None, uprev, body_force=face_force)
        x = spsolve(A, b)
        un = np.zeros((nz, nx))
        un[lay["open_u"]] = x[:lay["nu"]]
        wn = np.zeros((nz + 1, nx))
        wn[lay["open_w"]] = x[lay["off_w"]:lay["off_w"] + lay["nw"]]
        steps = it + 1
        if uprev is not None:
            du = max(float(np.abs(un - uprev[0]).max()), float(np.abs(wn - uprev[1]).max()))
            ref = max(float(np.abs(un).max()), float(np.abs(wn).max()), _TINY)
            residual = du / ref
            converged = bool(residual < 1e-4)
        uprev = (un, wn)
        uf, wf = un, wn
        if it and converged:
            break

    ux = 0.5 * (uf + np.roll(uf, -1, axis=1)) * u_scale
    uz = 0.5 * (wf[:-1] + wf[1:]) * u_scale
    ux = np.where(solid, 0.0, ux)
    uz = np.where(solid, 0.0, uz)

    div = (np.roll(uf, -1, axis=1) - uf) + (wf[1:] - wf[:-1])
    fmax = max(float(np.abs(uf).max()), float(np.abs(wf).max()), _TINY)
    div_max_rel = float(np.abs(div[fluid]).max() / fmax) if fluid.any() else 0.0

    info = dict(common, steps=steps, wall_s=time.time() - t0, residual=float(residual),
                u_top_cm_s=float(u_top), u_face_x=uf * u_scale, u_face_z=wf * u_scale,
                div_max_rel=div_max_rel, oseen_passes=npass, brinkman_K=brinkman_K,
                mesh=(nz, nx), n_fluid_regions=int(lay["n_regions"]),
                pin_cells=[int(r - lay["off_p"]) for r in lay["pin_rows"]],
                converged=bool(converged))
    return FlowField(grid=grid, ux=ux, uz=uz, shear_rate_s=shear, converged=bool(converged),
                     info=info)
