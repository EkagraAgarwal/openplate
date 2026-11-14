"""Cell-centred, conservative seven-point operators (cm-based units).

Non-periodic outer faces have zero transport coefficients. Since that erases
the conductivity needed by a later Dirichlet condition, ``face_coefficients``
returns ndarray subclasses retaining the adjacent cell conductivity. Keep these
arrays intact (``copy`` preserves metadata). For plain ndarray inputs, set the
requested Dirichlet outer face to its physical conductivity explicitly.
Periodic seams have equal coefficients on both copies and are counted once.
Inactive interfaces are insulating; caller-supplied source terms must be zero
in inactive cells. No volume factor is included: rhs is f, not f*dx**3.
"""
import numpy as np
from scipy import sparse


class _FaceArray(np.ndarray):
    """An ordinary face array with the otherwise lost outer-cell data."""

    def __array_finalize__(self, parent):
        self.boundary_k = getattr(parent, "boundary_k", None)
        self.periodic = getattr(parent, "periodic", None)


def _harmonic(a, b):
    # Stable even for large k; exactly zero if either cell is impermeable.
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ratio = np.divide(lo, hi, out=np.zeros_like(lo), where=hi > 0)
    return lo * (2 / (1 + ratio))


def face_coefficients(k_cell, periodic):
    """Harmonic interior/seam k; zero outer faces on non-periodic axes.

    Outputs are float64 arrays shaped (nx+1,ny,nz), etc. Coefficients must
    be finite and nonnegative. ``periodic`` has one boolean per axis.
    """
    k = np.asarray(k_cell, dtype=np.float64)
    if k.ndim != 3 or min(k.shape) < 1 or not np.isfinite(k).all() or np.any(k < 0):
        raise ValueError("k_cell must be a nonempty finite nonnegative 3-D array")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three booleans")
    result = []
    for axis, wraps in enumerate(periodic):
        v = np.moveaxis(k, axis, 0)
        f = np.zeros((v.shape[0]+1, *v.shape[1:]))
        f[1:-1] = _harmonic(v[:-1], v[1:])
        if wraps:
            f[0] = f[-1] = _harmonic(v[-1], v[0])
        face = np.moveaxis(f, 0, axis).copy().view(_FaceArray)
        face.boundary_k = (v[0].copy(), v[-1].copy())
        face.periodic = bool(wraps)
        result.append(face)
    return tuple(result)


def _prepare(kx, ky, kz, dx, reaction, active, dirichlet, periodic_axes=None):
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive (cm)")
    faces = (kx, ky, kz)
    arrays = tuple(np.asarray(f, dtype=np.float64) for f in faces)
    if any(f.ndim != 3 for f in arrays):
        raise ValueError("face arrays must be 3-D")
    shape = (arrays[0].shape[0]-1, arrays[0].shape[1], arrays[0].shape[2])
    if min(shape) < 1:
        raise ValueError("cell shape must be nonempty")
    bc = {} if dirichlet is None else dirichlet
    valid = {f"{name}_{side}" for name in "xyz" for side in ("lo", "hi")}
    if set(bc)-valid:
        raise ValueError(f"unknown Dirichlet face(s): {set(bc)-valid}")
    mask = np.ones(shape, bool) if active is None else np.asarray(active, dtype=bool)
    if mask.shape != shape:
        raise ValueError("active must have cell shape")
    r = np.broadcast_to(0.0 if reaction is None else reaction, shape).astype(np.float64)
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("reaction must be finite and nonnegative")
    diagonal = np.where(mask, r, 1.0)
    rhs = np.zeros(shape)
    wraps = []
    for axis, (face, arr) in enumerate(zip(faces, arrays)):
        expected = list(shape)
        expected[axis] += 1
        if arr.shape != tuple(expected) or not np.isfinite(arr).all() or np.any(arr < 0):
            raise ValueError("face arrays must have compatible shapes and finite nonnegative values")
        v = np.moveaxis(arr, axis, 0)
        keys = (f"{'xyz'[axis]}_lo", f"{'xyz'[axis]}_hi")
        explicit = any(key in bc for key in keys)
        # An explicit periodic=(px, py, pz) from the caller wins over array metadata and guessing.
        periodic = (bool(periodic_axes[axis]) if periodic_axes is not None
                    else getattr(face, "periodic", None))
        if periodic is None:
            periodic = not explicit and (np.any(v[0]) or np.any(v[-1]))
        if periodic and (explicit or not np.array_equal(v[0], v[-1])):
            raise ValueError("periodic faces must match and cannot also be Dirichlet")
        wraps.append(periodic)
        d, b, m = (np.moveaxis(t, axis, 0) for t in (diagonal, rhs, mask))
        outer_k = getattr(face, "boundary_k", None)
        for side, index in enumerate((0, -1)):
            if keys[side] not in bc:
                continue
            value = np.asarray(bc[keys[side]], dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError("Dirichlet values must be finite")
            # Explicit nonzero face coefficients override the saved cell values.
            conductivity = v[index]
            if outer_k is not None:
                conductivity = np.where(conductivity != 0, conductivity, outer_k[side])
            weight = 2*conductivity/dx**2 * m[index]
            d[index] += weight
            b[index] += np.where(m[index], weight*value, 0)
    return shape, arrays, mask, diagonal, rhs, wraps


def _edges(shape, faces, mask, wraps, dx):
    """Yield vectorized undirected edge batches, never loop over cells."""
    ids = np.arange(np.prod(shape)).reshape(shape)
    for axis, face in enumerate(faces):
        i, m, f = (np.moveaxis(a, axis, 0) for a in (ids, mask, face))
        ok = m[:-1] & m[1:] & (f[1:-1] > 0)
        yield i[:-1][ok], i[1:][ok], f[1:-1][ok]/dx**2
        if wraps[axis] and shape[axis] > 1:
            ok = m[-1] & m[0] & (f[0] > 0)
            yield i[-1][ok], i[0][ok], f[0][ok]/dx**2


def assemble(kx, ky, kz, dx, reaction=None, active=None, dirichlet=None, periodic=None):
    """Return CSR A and flat b_bc for -div(k grad u) + r*u.

    Dirichlet values (scalar or face-shaped) contribute 2*k/dx**2 to
    the diagonal and 2*k*value/dx**2 to b_bc. All other non-periodic
    outer boundaries are insulating. Inactive rows are identity and
    have no coupling to active rows; their source must be zero.
    """
    shape, faces, mask, diagonal, rhs, wraps = _prepare(
        kx, ky, kz, dx, reaction, active, dirichlet, periodic)
    n = int(np.prod(shape))
    diag = diagonal.ravel()
    rows, cols, data = [], [], []
    for i, j, w in _edges(shape, faces, mask, wraps, dx):
        np.add.at(diag, i, w)
        np.add.at(diag, j, w)
        rows.extend((i, j))
        cols.extend((j, i))
        data.extend((-w, -w))
    ids = np.arange(n)
    rows.append(ids)
    cols.append(ids)
    data.append(diag)
    A = sparse.coo_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    A.eliminate_zeros()
    return A, rhs.ravel()


def apply_matrix_free(u, kx, ky, kz, dx, reaction=None, active=None, dirichlet=None,
                      periodic=None):
    """Apply only A to a cell-shaped u; return cell shape, without b_bc."""
    shape, faces, mask, diagonal, _, wraps = _prepare(
        kx, ky, kz, dx, reaction, active, dirichlet, periodic)
    u = np.asarray(u, dtype=np.float64)
    if u.shape != shape:
        raise ValueError("u must have cell shape")
    flat = u.ravel()
    out = (diagonal*u).ravel()
    for i, j, w in _edges(shape, faces, mask, wraps, dx):
        flux = w*(flat[i]-flat[j])
        np.add.at(out, i, flux)
        np.add.at(out, j, -flux)
    return out.reshape(shape)
