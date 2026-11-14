"""Sharp-mask steady Stokes on a uniform staggered grid.

The transverse half-wall weights intentionally match trenchflow.stokes, including
half-covered step corners. Length, velocity and pressure scales are dx, U and
nu*U/dx. The only dependencies of the direct path are NumPy and SciPy.
"""
from time import perf_counter
import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, gmres, spsolve


def buoyancy_force(ct, c_bulk_mol_L, beta_L_mol, gravity_dir, g_cm_s2=981.0):
    """Boussinesq acceleration, cm/s²; depleted electrolyte rises against gravity.

    ``ct`` is concentration divided by bulk concentration. ``gravity_dir`` is
    used verbatim (not normalised), allowing zero gravity or projected gravity.
    """
    ct = np.asarray(ct, dtype=np.float64)
    direction = np.asarray(gravity_dir, dtype=np.float64)
    params = np.asarray([c_bulk_mol_L, beta_L_mol, g_cm_s2], dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('gravity_dir must contain three finite components')
    if not np.isfinite(ct).all() or not np.isfinite(params).all():
        raise ValueError('buoyancy inputs must be finite')
    if c_bulk_mol_L < 0 or g_cm_s2 < 0:
        raise ValueError('bulk concentration and gravity magnitude must be nonnegative')
    f = float(g_cm_s2) * float(beta_L_mol) * float(c_bulk_mol_L) * (ct - 1.)
    return tuple(f * d for d in direction)


def _shift(a, axis, step, periodic, fill):
    """Neighbour at index + step; no wrap on physical boundaries."""
    out = np.roll(a, -step, axis=axis)
    if not periodic[axis]:
        edge = [slice(None)] * 3
        edge[axis] = slice(-1, None) if step == 1 else slice(0, 1)
        out[tuple(edge)] = fill
    return out


def _face_cells(a, axis, periodic, fill):
    """Left/right cell values on unique faces (periodic high face omitted)."""
    n = a.shape[axis]
    inds = np.arange(n if periodic[axis] else n + 1)
    left = np.take(a, (inds - 1) % n, axis=axis)
    right = np.take(a, inds % n, axis=axis)
    if not periodic[axis]:
        edge = [slice(None)] * 3
        edge[axis] = slice(0, 1)
        left[tuple(edge)] = fill
        edge[axis] = slice(-1, None)
        right[tuple(edge)] = fill
    return left, right


def _regions(fluid, periodic):
    """Six-connected labels with union-find merges across both periodic seams."""
    labels, n = ndimage.label(fluid)
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for axis in range(3):
        if periodic[axis]:
            lo = np.take(labels, 0, axis=axis).ravel()
            hi = np.take(labels, -1, axis=axis).ravel()
            pairs = np.unique(np.stack((lo, hi), axis=1), axis=0)
            for a, b in pairs:
                if a and b:
                    parent[find(b)] = find(a)
    roots = np.array([find(i) for i in range(n + 1)])
    _, inverse = np.unique(roots[1:], return_inverse=True)
    remap = np.r_[0, inverse + 1]
    return remap[labels], len(np.unique(roots[1:]))


def _field_height(psi, dx, z0):
    """Highest upward solid/electrolyte zero crossing, or low face if none.

    This identifies the field (not the trench floor) for an etched planar wafer.
    It is linear in the signed distance, hence resolves fractional deposits.
    A rough/nonplanar wafer needs an explicit field-height policy in the contract.
    """
    crossing = (psi[:, :, :-1] <= 0) & (psi[:, :, 1:] > 0)
    i, j, k = np.where(crossing)
    if not k.size:
        return float(z0)
    low, high = psi[i, j, k], psi[i, j, k + 1]
    return float(np.max(z0 + (k + .5 - low / (high - low)) * dx))


def _assemble(fluid, periodic, lid, force, top_distance):
    """Return A, G, momentum RHS, face maps, pressure map and gauge metadata.

    G is the gradient and -G.T the divergence. Remove one pressure column and
    its redundant continuity equation per region, preserving symmetry. Unlike
    row-only gauge replacement, this also admits symmetric block methods.
    """
    labels, nregions = _regions(fluid, periodic)
    flat_labels = labels.ravel()
    active = np.flatnonzero(flat_labels)
    _, first = np.unique(flat_labels[active], return_index=True)
    pins = active[first]
    pressure_active = fluid.copy()
    pressure_active.ravel()[pins] = False
    pid = np.full(fluid.shape, -1, dtype=np.int64)
    pid[pressure_active] = np.arange(pressure_active.sum())

    maps, adjacent, count = [], [], 0
    for c in range(3):
        left, right = _face_cells(fluid, c, periodic, False)
        opened = left & right
        ids = np.full(opened.shape, -1, dtype=np.int64)
        ids[opened] = np.arange(count, count + opened.sum())
        count += int(opened.sum())
        maps.append(ids)
        adjacent.append(_face_cells(pid, c, periodic, -1))

    rows, cols, vals = [], [], []
    gr, gc, gv = [], [], []
    b = np.zeros(count)

    def emit(r, col, value, rr, cc, vv):
        r, col, value = np.broadcast_arrays(r, col, value)
        keep = (r >= 0) & (col >= 0)
        rr.append(r[keep])
        cc.append(col[keep])
        vv.append(value[keep])

    for c, ids in enumerate(maps):
        opened = ids >= 0
        diag = np.zeros(ids.shape)
        left, right = _face_cells((~fluid).astype(float), c, periodic, 1.)
        solid_fraction = .5 * (left + right)
        for axis in range(3):
            for step in (-1, 1):
                neighbour = _shift(ids, axis, step, periodic, -1)
                weight = np.ones(ids.shape)
                if axis != c:
                    weight += _shift(solid_fraction, axis, step, periodic, 0.)
                    if not periodic[axis]:
                        edge = [slice(None)] * 3
                        edge[axis] = slice(0, 1) if step == -1 else slice(-1, None)
                        edge = tuple(edge)
                        # Lateral sides are impermeable/free-slip; z walls no-slip.
                        weight[edge] = (2. if step == -1 else 1. / top_distance) if axis == 2 else 0.
                        if axis == 2 and step == 1:
                            r = ids[edge]
                            good = r >= 0
                            b[r[good]] += lid[c] / top_distance
                diag += weight
                emit(ids, neighbour, -1., rows, cols, vals)
        emit(ids, ids, diag, rows, cols, vals)
        pl, pr = adjacent[c]
        emit(ids, pr, 1., gr, gc, gv)
        emit(ids, pl, -1., gr, gc, gv)
        fl, fr = _face_cells(force[c], c, periodic, 0.)
        b[ids[opened]] += (.5 * (fl + fr))[opened]

    def matrix(rr, cc, vv, shape):
        return sparse.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=shape).tocsr()

    a = matrix(rows, cols, vals, (count, count))
    g = matrix(gr, gc, gv, (count, int(pressure_active.sum())))
    return a, g, b, maps, pid, labels, pins, nregions


def _solve(a, g, b, solver, pressure_labels):
    """GMRES with an upper triangular AMG/mass-Schur block preconditioner.

    Viscous Stokes has an order-zero pressure Schur complement: the pressure
    mass matrix is its mesh-independent approximation. In pinned coordinates
    its inverse includes one rank-one correction per region (I + 11.T), avoiding
    a spurious near-null mode from a single-cell pressure gauge. AMG approximates
    the three SPD velocity Poisson blocks without 3-D sparse-LU fill-in.
    """
    nv, np_ = g.shape
    system = sparse.bmat([[a, g], [g.T, None]], format='csr')
    rhs = np.r_[b, np.zeros(np_)]
    if solver == 'direct':
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', MatrixRankWarning)
                x = spsolve(system, rhs)
        except (MatrixRankWarning, RuntimeError) as exc:
            raise RuntimeError('Stokes direct factorisation failed') from exc
        residual = float(np.linalg.norm(system @ x - rhs) / max(np.linalg.norm(rhs), 1e-30))
        if not np.isfinite(x).all() or not np.isfinite(residual) or residual > 1e-10:
            raise RuntimeError(f'Stokes direct solve failed: residual={residual:.3e}')
        return x, 1, residual
    try:
        import pyamg
    except ImportError as exc:
        raise ImportError('iterative Stokes requires pyamg; install the optional ref dependencies or use solver="direct"') from exc
    amg = pyamg.smoothed_aggregation_solver(a, symmetry='symmetric', max_coarse=100)
    ainv = amg.aspreconditioner(cycle='V')
    iterations = 0

    def precondition(r):
        rp = r[nv:]
        sums = np.bincount(pressure_labels, weights=rp)
        p = -(rp + sums[pressure_labels])
        return np.r_[ainv @ (r[:nv] - g @ p), p]

    def callback(_):
        nonlocal iterations
        iterations += 1

    pre = LinearOperator(system.shape, matvec=precondition, dtype=float)
    x, status = gmres(system, rhs, M=pre, rtol=2e-12, atol=0., restart=60,
                      maxiter=20, callback=callback, callback_type='pr_norm')
    residual = float(np.linalg.norm(system @ x - rhs) / max(np.linalg.norm(rhs), 1e-30))
    if status or not np.isfinite(x).all() or residual > 1e-10:
        raise RuntimeError(f'Stokes GMRES failed: status={status}, iterations={iterations}, residual={residual:.3e}')
    return x, iterations, residual


def solve_stokes3d(psi, dx, z0, z_lid, periodic=(True, True, False), shear_rate_s=0.0,
                   shear_dir=(1.0, 0.0), body_force=None, nu_cm2_s=0.0114, solver='auto') -> dict:
    """Steady incompressible MAC Stokes, section 10e of CONTRACTS_3D.md.

    Arrays are [x,y,z]; psi > 0 is fluid. The bottom and solid walls are
    stationary/no-slip; nonperiodic lateral sides are impermeable/free-slip.
    z is never periodic. The lid can lie inside the supplied grid: cells whose
    centres are at/above it return zero. The last active centre must be at least
    1e-5 dx below it; tangential wall extrapolation uses its exact height.
    Normal flux closes at the last retained cell face (not a cut-cell lid).

    Lid speed is shear_rate_s * (z_lid - z_field), with z_field the highest
    upward zero crossing of psi, or z0 for an entirely open domain. shear_dir
    is normalised. For arbitrary rough fronts this field-height policy is an
    explicit convention, pending an additional contract argument.

    Returns cell velocities, full faces (nx+1,ny,nz), (nx,ny+1,nz),
    (nx,ny,nz+1), pressure (cm²/s²), and solver diagnostics. Periodic seam
    faces are duplicated exactly. Pressure is zero at one cell per region.
    'auto' uses SuperLU up to 12000 unknowns, block-AMG GMRES above that.
    """
    start = perf_counter()
    psi = np.asarray(psi, dtype=float)
    if psi.ndim != 3 or min(psi.shape) < 1 or not np.isfinite(psi).all():
        raise ValueError('psi must be a nonempty finite 3-D array')
    if not np.isfinite([dx, z0, z_lid, shear_rate_s, nu_cm2_s]).all() or dx <= 0 or nu_cm2_s <= 0:
        raise ValueError('dx and nu_cm2_s must be positive; coordinates and shear must be finite')
    if len(periodic) != 3 or periodic[2]:
        raise ValueError('periodic must be (px, py, False): the bottom and lid are walls')
    periodic = tuple(bool(p) for p in periodic)
    direction = np.asarray(shear_dir, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('shear_dir must be a finite nonzero in-plane vector')
    direction = direction / np.linalg.norm(direction)
    if solver not in ('auto', 'direct', 'iterative'):
        raise ValueError('solver must be auto, direct, or iterative')
    extent = (z_lid - z0) / dx
    if extent <= .5 or extent > psi.shape[2] + 1e-10:
        raise ValueError('z_lid must lie above the lowest cell centre and within the grid')
    nz = min(psi.shape[2], int(np.ceil(extent - .5 - 1e-12)))
    top_distance = extent - (nz - .5)
    if top_distance < 1e-5:
        raise ValueError('lid is too close to a cell centre (<1e-5 dx); move z_lid to a grid face')
    cut = psi[:, :, :nz]
    fluid = cut > 0
    field_height = _field_height(cut, dx, z0)
    lid_speed = float(shear_rate_s) * max(z_lid - field_height, 0.)
    lid = np.r_[direction * lid_speed, 0.]
    if body_force is None:
        body = tuple(np.zeros(cut.shape) for _ in range(3))
    else:
        if len(body_force) != 3:
            raise ValueError('body_force must contain three cell-centred arrays')
        body = tuple(np.asarray(f, dtype=float) for f in body_force)
        if any(f.shape != psi.shape or not np.isfinite(f).all() for f in body):
            raise ValueError('body_force must contain finite arrays with the same shape as psi')
        body = tuple(np.where(fluid, f[:, :, :nz], 0.) for f in body)
    fmax = float(np.max(np.sqrt(sum(f * f for f in body))))
    scale = max(abs(lid_speed), fmax * dx * dx / nu_cm2_s, 1e-30)
    force = tuple(f * (dx * dx / (nu_cm2_s * scale)) for f in body)
    a, g, b, maps, pid, labels, pins, nregions = _assemble(fluid, periodic, lid / scale, force, top_distance)
    method = ('direct' if a.shape[0] + g.shape[1] <= 12000 else 'iterative') if solver == 'auto' else solver
    pressure = np.zeros(psi.shape)
    if a.shape[0] and np.any(b):
        x, iterations, residual = _solve(a, g, b, method, labels[pid >= 0])
        pressure[:, :, :nz][pid >= 0] = x[a.shape[0]:] * (nu_cm2_s * scale / dx)
    else:
        x = np.zeros(a.shape[0] + g.shape[1])
        iterations, residual = 0, 0.
    faces, cells = [], []
    for c, ids in enumerate(maps):
        face = np.zeros(ids.shape)
        face[ids >= 0] = x[ids[ids >= 0]] * scale
        if periodic[c]:
            face = np.concatenate((face, np.take(face, [0], axis=c)), axis=c)
        shape = list(psi.shape)
        shape[c] += 1
        full = np.zeros(shape)
        sl = [slice(None)] * 3
        sl[2] = slice(0, face.shape[2])
        full[tuple(sl)] = face
        faces.append(full)
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[c], hi[c] = slice(None, -1), slice(1, None)
        cells.append(np.where(psi > 0, .5 * (full[tuple(lo)] + full[tuple(hi)]), 0.))
    divergence = sum(np.diff(f, axis=c) for c, f in enumerate(faces)) / dx
    divmax = float(np.max(np.abs(divergence)))
    vmax = max(float(np.max(np.abs(f))) for f in faces)
    info = dict(iterations=iterations, divergence_max=divmax, seconds=perf_counter() - start,
                solver=method, converged=True, rel_residual=residual, n_fluid_regions=nregions,
                pin_cells=[tuple(int(i) for i in np.unravel_index(p, cut.shape)) for p in pins],
                div_max_rel=divmax * dx / max(vmax, 1e-30), z_field_cm=field_height,
                u_lid_cm_s=lid_speed, u_scale_cm_s=scale, unknowns=a.shape[0] + g.shape[1])
    return dict(zip(('ux', 'uy', 'uz'), cells), faces=tuple(faces), p=pressure, info=info)
