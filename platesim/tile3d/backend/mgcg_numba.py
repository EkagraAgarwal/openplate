"""Matrix-free geometric V-cycle preconditioned CG for the tile FV operator.

Faces, diagonals, V-cycle and Krylov workspaces use ``dtype``; all scalar
reductions accumulate in float64. Native float32 can have a true-residual floor
above the requested tolerance; nonconvergence is reported, not hidden by a
recursive residual. No fine-grid sparse matrix is built. Only the final, small
grid is factored. Dirichlet data affect A, not b:
the caller must include the boundary RHS returned by operators.assemble.

Cell-centred linear prolongation and its scaled transpose form a symmetric
transfer pair. Coarse faces are transverse arithmetic averages at coincident
fine faces, not averages of cell conductivities. Axes that cannot be halved
exactly while retaining four cells stop coarsening independently.
"""
from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import numpy as np
from numba import njit, prange

from ..operators import _prepare


@njit(parallel=True, cache=True)
def _dot(a, b):
    total = 0.0
    av, bv = a.ravel(), b.ravel()
    for i in prange(av.size):
        total += np.float64(av[i]) * np.float64(bv[i])
    return total


@njit(parallel=True, cache=True)
def _kernel(u, b, fx, fy, fz, base, diag, periodic, out, color, omega):
    """Apply, or one simultaneous Jacobi / independent RBGS colour update."""
    nx, ny, nz = u.shape
    for ij in prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nz):
            if color >= 0 and (i + j + k) % 2 != color:
                continue
            v = np.float64(u[i, j, k])
            a = np.float64(base[i, j, k]) * v
            if i > 0:
                a += fx[i, j, k] * (v - u[i-1, j, k])
            elif periodic[0] and nx > 1:
                a += fx[0, j, k] * (v - u[nx-1, j, k])
            if i + 1 < nx:
                a += fx[i+1, j, k] * (v - u[i+1, j, k])
            elif periodic[0] and nx > 1:
                a += fx[0, j, k] * (v - u[0, j, k])
            if j > 0:
                a += fy[i, j, k] * (v - u[i, j-1, k])
            elif periodic[1] and ny > 1:
                a += fy[i, 0, k] * (v - u[i, ny-1, k])
            if j + 1 < ny:
                a += fy[i, j+1, k] * (v - u[i, j+1, k])
            elif periodic[1] and ny > 1:
                a += fy[i, 0, k] * (v - u[i, 0, k])
            if k > 0:
                a += fz[i, j, k] * (v - u[i, j, k-1])
            elif periodic[2] and nz > 1:
                a += fz[i, j, 0] * (v - u[i, j, nz-1])
            if k + 1 < nz:
                a += fz[i, j, k+1] * (v - u[i, j, k+1])
            elif periodic[2] and nz > 1:
                a += fz[i, j, 0] * (v - u[i, j, 0])
            if omega == 0:
                out[i, j, k] = a
            else:
                out[i, j, k] = v + omega * (b[i, j, k] - a) / diag[i, j, k]


@njit(cache=True)
def _index(i, n, periodic):
    return i % n if periodic else min(max(i, 0), n-1)


@njit(parallel=True, cache=True)
def _prolong(c, out, mask, factors, periodic):
    nx, ny, nz = out.shape
    for ij in prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nz):
            value = 0.0
            for a in range(factors[0]):
                ci = i if factors[0] == 1 else (i-1)//2 + a
                wi = 1.0 if factors[0] == 1 else (.75 if a == (1-i % 2) else .25)
                ci = _index(ci, c.shape[0], periodic[0])
                for b in range(factors[1]):
                    cj = j if factors[1] == 1 else (j-1)//2 + b
                    wj = 1.0 if factors[1] == 1 else (.75 if b == (1-j % 2) else .25)
                    cj = _index(cj, c.shape[1], periodic[1])
                    for d in range(factors[2]):
                        ck = k if factors[2] == 1 else (k-1)//2 + d
                        wk = 1.0 if factors[2] == 1 else (.75 if d == (1-k % 2) else .25)
                        ck = _index(ck, c.shape[2], periodic[2])
                        value += wi * wj * wk * c[ci, cj, ck]
            out[i, j, k] = value if mask[i, j, k] else 0


@njit(cache=True)
def _restriction_entry(c, offset, n, factor, periodic):
    if factor == 1:
        return c, 1.0
    # Transpose of clamped (Neumann extension) cell-centred interpolation.
    raw = 2*c - 1 + offset
    w = .25 if offset == 0 or offset == 3 else .75
    if periodic:
        return raw % n, w
    if raw < 0 or raw >= n:
        return 0, 0.0
    if (c == 0 and offset == 1) or (c == n//2-1 and offset == 2):
        w = 1.0
    return raw, w


@njit(parallel=True, cache=True)
def _restrict(f, out, fine_mask, coarse_mask, factors, periodic):
    nx, ny, nz = out.shape
    scale = 1.0 / (factors[0] * factors[1] * factors[2])
    for ij in prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nz):
            value = 0.0
            for a in range(1 if factors[0] == 1 else 4):
                fi, wi = _restriction_entry(i, a, f.shape[0], factors[0], periodic[0])
                for b in range(1 if factors[1] == 1 else 4):
                    fj, wj = _restriction_entry(j, b, f.shape[1], factors[1], periodic[1])
                    for d in range(1 if factors[2] == 1 else 4):
                        fk, wk = _restriction_entry(k, d, f.shape[2], factors[2], periodic[2])
                        if fine_mask[fi, fj, fk]:
                            value += wi * wj * wk * f[fi, fj, fk]
            out[i, j, k] = scale * value if coarse_mask[i, j, k] else 0


def _cell_average(a, factors):
    shape = tuple(s // f for s, f in zip(a.shape, factors))
    return a.reshape(shape[0], factors[0], shape[1], factors[1], shape[2], factors[2]).mean(axis=(1, 3, 5))


def _coarse_faces(faces, factors):
    result = []
    for axis, face in enumerate(faces):
        v = face
        for dim, factor in enumerate(factors):
            if dim == axis:
                ix = [slice(None)] * 3
                ix[dim] = slice(None, None, factor)
                v = v[tuple(ix)]
            elif factor == 2:
                sh = list(v.shape)
                sh[dim:dim+1] = [sh[dim] // 2, 2]
                v = v.reshape(sh).mean(axis=dim+1)
        result.append(np.ascontiguousarray(v))
    return result


class MGCG:
    """Reusable geometric hierarchy; all connected components must be anchored.

    ``rbgs`` uses reverse colour order after coarse correction. Odd periodic
    grids are not bipartite and automatically use damped Jacobi on that level.
    Equal positive pre/post counts are required for a symmetric preconditioner.
    ``levels`` caps depth; a large terminal grid uses 200 symmetric smoothing
    sweeps instead of a direct factorization. solve returns the shape of b.
    """

    def __init__(self, kx, ky, kz, dx, reaction=None, active=None, dirichlet=None,
                 periodic=(True, True, False), dtype=np.float64, levels=None,
                 smoother="rbgs", pre=2, post=2):
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        if smoother not in ("rbgs", "jacobi"):
            raise ValueError("smoother must be rbgs or jacobi")
        if pre != post or pre < 1 or int(pre) != pre:
            raise ValueError("pre and post must be equal positive integers for symmetric CG")
        if len(periodic) != 3:
            raise ValueError("periodic must contain three booleans")
        if levels is not None and (levels < 1 or int(levels) != levels):
            raise ValueError("levels must be a positive integer")
        shape, arrays, mask, _, _, wraps = _prepare(
            kx, ky, kz, dx, reaction, active, dirichlet, periodic)
        self.shape, self.periodic = shape, tuple(wraps)
        self.pre, self.post, self.smoother = int(pre), int(post), smoother
        bc = {} if dirichlet is None else dirichlet
        faces = [a.copy() for a in arrays]
        # Resolve metadata before converting to a plain storage array.
        for axis, (original, face) in enumerate(zip((kx, ky, kz), faces)):
            v = np.moveaxis(face, axis, 0)
            saved = getattr(original, "boundary_k", None)
            for side, index in enumerate((0, -1)):
                if f"{'xyz'[axis]}_{('lo', 'hi')[side]}" in bc:
                    if saved is not None:
                        v[index] = np.where(v[index] != 0, v[index], saved[side])
                elif not wraps[axis]:
                    v[index] = 0
        self._bc = set(bc)
        self.hierarchy = []
        spacing = np.full(3, dx)
        while True:
            factors = tuple(2 if n >= 8 and n % 2 == 0 else 1 for n in mask.shape)
            level = self._level(faces, mask, spacing, factors)
            self.hierarchy.append(level)
            if all(f == 1 for f in factors) or (levels is not None and len(self.hierarchy) >= levels):
                break
            faces = _coarse_faces(faces, factors)
            mask = _cell_average(mask.astype(float), factors) > 0
            spacing = spacing * factors
        self.set_reaction(reaction)

    def _level(self, physical, mask, spacing, factors):
        faces = [np.array(f / h**2, dtype=self.dtype, order="C", copy=True)
                 for f, h in zip(physical, spacing)]
        boundary = np.zeros(mask.shape, dtype=self.dtype)
        for axis, face in enumerate(faces):
            v, m, d = (np.moveaxis(a, axis, 0) for a in (face, mask, boundary))
            v[1:-1] *= m[:-1] & m[1:]
            if self.periodic[axis]:
                v[0] *= m[0] & m[-1]
                v[-1] = v[0]
            else:
                for side, index in enumerate((0, -1)):
                    if f"{'xyz'[axis]}_{('lo', 'hi')[side]}" in self._bc:
                        d[index] += 2 * v[index] * m[index]
                    v[index] = 0
        edge_diagonal = np.zeros(mask.shape, dtype=self.dtype)
        for axis, face in enumerate(faces):
            v, d = (np.moveaxis(a, axis, 0) for a in (face, edge_diagonal))
            d[:-1] += v[1:-1]
            d[1:] += v[1:-1]
            if self.periodic[axis] and mask.shape[axis] > 1:
                d[0] += v[0]
                d[-1] += v[0]
        rbgs = self.smoother == "rbgs" and not any(
            p and n % 2 for p, n in zip(self.periodic, mask.shape))
        return SimpleNamespace(faces=tuple(faces), mask=mask.copy(), boundary=boundary,
                               edge_diagonal=edge_diagonal, factors=factors, rbgs=rbgs,
                               x=np.zeros(mask.shape, dtype=self.dtype),
                               rhs=np.zeros(mask.shape, dtype=self.dtype),
                               work=np.zeros(mask.shape, dtype=self.dtype),
                               correction=np.zeros(mask.shape, dtype=self.dtype))

    def set_reaction(self, reaction):
        """Replace only diagonal data and the terminal factor; retain all faces/transfers."""
        r = np.broadcast_to(0.0 if reaction is None else reaction, self.shape).astype(np.float64)
        if np.any(r < 0) or not np.isfinite(r).all():
            raise ValueError("reaction must be finite and nonnegative")
        for level in self.hierarchy:
            level.base = np.where(level.mask, level.boundary + r, 1).astype(self.dtype)
            level.diag = level.base + level.edge_diagonal
            if np.any(level.diag <= 0) or not np.isfinite(level.diag).all():
                raise ValueError("operator has an unanchored isolated cell or storage overflow")
            r = _cell_average(np.where(level.mask, r, 0), level.factors)
        last = self.hierarchy[-1]
        last.cholesky = None
        if last.x.size <= 512:
            # At most 512 unknowns, constructed from the seven-point coarse stencil.
            from scipy import sparse
            from scipy.linalg import cho_factor
            ids = np.arange(last.x.size).reshape(last.x.shape)
            rows, cols, values = [ids.ravel()], [ids.ravel()], [last.diag.ravel().astype(float)]
            for axis, face in enumerate(last.faces):
                v, ix = (np.moveaxis(a, axis, 0) for a in (face, ids))
                pairs = [(ix[:-1].ravel(), ix[1:].ravel(), v[1:-1].ravel())]
                if self.periodic[axis] and last.x.shape[axis] > 1:
                    pairs.append((ix[-1].ravel(), ix[0].ravel(), v[0].ravel()))
                for i, j, w in pairs:
                    rows.extend((i, j)); cols.extend((j, i)); values.extend((-w, -w))
            matrix = sparse.coo_matrix((np.concatenate(values),
                                       (np.concatenate(rows), np.concatenate(cols))),
                                      shape=(last.x.size, last.x.size)).toarray()
            try:
                last.cholesky = cho_factor(matrix, check_finite=False)
            except np.linalg.LinAlgError as exc:
                raise ValueError("operator must be SPD: anchor every active component") from exc

    def _apply(self, level, u, out):
        _kernel(u, u, *level.faces, level.base, level.diag, self.periodic, out, -1, 0.0)

    def apply(self, u):
        """Return A*u with the input shape; never add Dirichlet boundary RHS."""
        u = np.asarray(u)
        if u.size != np.prod(self.shape) or u.shape not in (self.shape, (int(np.prod(self.shape)),)):
            raise ValueError("u must be cell-shaped or C-order flat")
        v = np.ascontiguousarray(u.reshape(self.shape), dtype=np.result_type(u.dtype, self.dtype))
        out = np.empty_like(v)
        self._apply(self.hierarchy[0], v, out)
        return out.reshape(u.shape)

    def _smooth(self, level, sweeps, reverse=False):
        for _ in range(sweeps):
            if level.rbgs:
                for color in ((1, 0) if reverse else (0, 1)):
                    _kernel(level.x, level.rhs, *level.faces, level.base, level.diag,
                            self.periodic, level.x, color, 1.0)
            else:
                _kernel(level.x, level.rhs, *level.faces, level.base, level.diag,
                        self.periodic, level.work, -1, 2/3)
                level.x[:] = level.work

    def _cycle(self, index=0):
        level = self.hierarchy[index]
        level.x.fill(0)
        if index == len(self.hierarchy)-1:
            if level.cholesky is not None:
                from scipy.linalg import cho_solve
                level.x[:] = cho_solve(level.cholesky, level.rhs.ravel(),
                                       check_finite=False).reshape(level.x.shape)
            else:
                self._smooth(level, 100)
                self._smooth(level, 100, reverse=True)
            return
        self._smooth(level, self.pre)
        self._apply(level, level.x, level.work)
        level.work[:] = level.rhs - level.work
        coarse = self.hierarchy[index+1]
        _restrict(level.work, coarse.rhs, level.mask, coarse.mask, level.factors, self.periodic)
        self._cycle(index+1)
        _prolong(coarse.x, level.correction, level.mask, level.factors, self.periodic)
        level.x += level.correction
        self._smooth(level, self.post, reverse=True)

    def solve(self, b, x0=None, tol=1e-8, maxiter=200):
        """PCG, reporting a freshly recomputed residual, including failed solves.

        All dot products/norms accumulate in float64. Vectors and the returned
        solution use the requested storage dtype. For float32 storage, the
        residual is for the stored (rounded) coefficients, evaluated in float64.
        Setup/JIT time is excluded unless first triggered within this call.
        Instances own workspaces and are not reentrant.
        """
        start = perf_counter()
        b = np.asarray(b, dtype=np.float64)
        if b.size != np.prod(self.shape) or not np.isfinite(b).all():
            raise ValueError("b must be finite and match cell count")
        if not np.isfinite(tol) or tol <= 0 or maxiter < 0 or int(maxiter) != maxiter:
            raise ValueError("tol must be positive and maxiter a nonnegative integer")
        output_shape = b.shape
        b = b.reshape(self.shape)
        x = (np.zeros(self.shape, dtype=self.dtype) if x0 is None else
             np.array(x0, dtype=self.dtype).reshape(self.shape).copy())
        if not np.isfinite(x).all():
            raise ValueError("x0 must be finite")
        norm = float(np.sqrt(_dot(b, b))) or 1.0
        # Float64 operator evaluation avoids cancellation / output rounding when
        # checking the true residual, without changing solution storage.
        r = (b - self.apply(x.astype(np.float64))).astype(self.dtype)
        relative = float(np.sqrt(_dot(r, r)) / norm)
        p = np.zeros(self.shape, dtype=self.dtype)
        old = 0.0
        restart = True
        iterations = 0
        best_checked = relative
        plateau = 0
        stagnated = False
        for step in range(int(maxiter)):
            if relative <= tol:
                break
            self.hierarchy[0].rhs[:] = r
            self._cycle()
            z = self.hierarchy[0].x.copy()
            # Inactive identity rows are not passed through masked transfers.
            mask = self.hierarchy[0].mask
            z[~mask] = r[~mask]
            rho = float(_dot(r, z))
            if not np.isfinite(rho) or rho <= 0:
                break
            p[:] = z + (rho / old * p if not restart else 0)
            restart = False
            ap = self.apply(p)
            denominator = float(_dot(p, ap))
            if not np.isfinite(denominator) or denominator <= 0:
                break
            alpha = rho / denominator
            x += alpha * p
            r -= alpha * ap
            iterations = step + 1
            relative = float(np.sqrt(_dot(r, r)) / norm)
            old = rho
            if relative <= tol or (self.dtype == np.dtype("float32") and iterations % 8 == 0):
                r = (b - self.apply(x.astype(np.float64))).astype(self.dtype)
                relative = float(np.sqrt(_dot(r, r)) / norm)
                if relative > tol:
                    # Restart the recurrence after reliable residual replacement.
                    restart = True
                    if self.dtype == np.dtype("float32"):
                        plateau = plateau + 1 if relative >= .95 * best_checked else 0
                        best_checked = min(best_checked, relative)
                        if plateau >= 3:
                            stagnated = True
                            break
        residual = b - self.apply(x.astype(np.float64))
        relative = float(np.sqrt(_dot(residual, residual)) / norm)
        return x.reshape(output_shape), dict(iterations=iterations, rel_residual=relative,
                                             converged=bool(relative <= tol),
                                             seconds=perf_counter()-start, stagnated=stagnated)
