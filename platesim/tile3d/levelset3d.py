"""Array-only 3-D level sets: positive electrolyte, negative solid (cm, s).

Advection is explicit: callers must keep dt*max(abs(speed))/dx <= 0.3
for a conservative 3-D CFL choice. Stored input precision is preserved;
all kernels, root solves, and volume sums use float64 internally.
"""
import importlib

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import brentq

from .backend import ls_numpy

DEFAULT_PERIODIC = (True, True, False)
RELAX_ITERS = 20           # pseudo-time sweeps (4 cells) of the post-FMM gradient relaxation
                           # (fewer sweeps grow the near-front gradient error: 12 -> 0.054)


def _array(psi, dx=None):
    a = np.asarray(psi)
    if a.ndim != 3 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError("psi must be a nonempty finite 3-D array")
    if dx is not None and (not np.isfinite(dx) or dx <= 0):
        raise ValueError("dx must be finite and positive (cm)")
    return np.ascontiguousarray(a, dtype=np.float64)


def _periodic(periodic):
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    return tuple(bool(p) for p in periodic)


def _restore(a, original):
    dtype = np.asarray(original).dtype
    return np.asarray(a, dtype=dtype if dtype.kind == "f" else np.float64)


def _kernels(backend):
    if backend not in ("numpy", "numba", "auto"):
        raise ValueError("backend must be 'numpy', 'numba' or 'auto'")
    if backend != "numpy":
        try:
            return importlib.import_module(".backend.ls_numba", __package__)
        except ImportError:
            pass
    return ls_numpy


def advect(psi, speed, dt, dx, periodic=DEFAULT_PERIODIC, scheme="weno5",
           band_cells=None, backend="auto"):
    """Advance psi_t + speed*|grad psi| = 0; positive speed grows solid.

    The original strict band alone is written. RK3 intermediate stages use
    the full grid so no stale neighbours contaminate band cells. This is a
    correctness-first band implementation, not a sparse-work optimisation.
    Missing Numba falls back to NumPy even when explicitly requested.
    """
    a = _array(psi, dx)
    per = _periodic(periodic)
    if scheme not in ("weno5", "godunov1"):
        raise ValueError("scheme must be 'weno5' or 'godunov1'")
    if backend not in ("numpy", "numba", "auto"):
        raise ValueError("backend must be 'numpy', 'numba' or 'auto'")
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative (s)")
    if band_cells is not None and (not np.isfinite(band_cells) or band_cells <= 0):
        raise ValueError("band_cells must be finite and positive")
    v = np.ascontiguousarray(np.broadcast_to(speed, a.shape), dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("speed must be finite (cm/s)")
    if dt == 0 or not np.any(v):
        return _restore(a.copy(), psi)
    kernels = _kernels(backend)
    fn = getattr(kernels, "advect_" + scheme)
    return _restore(fn(a, v, dt, dx, per, band_cells), psi)


def reinit(psi, dx, periodic=DEFAULT_PERIODIC, narrow_cells=None, backend="auto"):
    """Second-order scikit-fmm distance; masked cells retain input values.

    The relaxation RHS uses the same optional backend as advection; NumPy
    remains available as a reference. A field with no zero contour is
    returned unchanged. A narrow FMM band
    is measured by computed distance, not by the distorted input field.
    """
    import skfmm
    a = _array(psi, dx)
    per = _periodic(periodic)
    if backend not in ("numpy", "numba", "auto"):
        raise ValueError("backend must be 'numpy', 'numba' or 'auto'")
    if narrow_cells is not None and (not np.isfinite(narrow_cells) or narrow_cells <= 0):
        raise ValueError("narrow_cells must be finite and positive")
    if not (np.min(a) <= 0 <= np.max(a)) or np.all(a == 0):
        return _restore(a.copy(), psi)
    kernels = _kernels(backend)
    # Six extra cells isolate the returned narrow band from relaxation's
    # artificial outer edge (four cells pseudo-time plus stencil margin).
    distance = skfmm.distance(a, dx=dx, order=2, periodic=per,
                            narrow=0 if narrow_cells is None else (narrow_cells+6)*dx)
    out = np.where(np.ma.getmaskarray(distance), a, np.ma.getdata(distance))
    # FMM's edge-interpolated interface seeds can leave O(0.2) local
    # gradient spikes in the first cell layer. Relax these with the same
    # high-order Hamiltonian and a frozen, scale-invariant smoothed sign.
    # The finite pseudo-time is four cells; it is not physical motion.
    grad0 = np.sqrt(sum(g*g for g in _gradient(a, dx, per)))
    sign = a/np.maximum(np.sqrt(a*a + (dx*grad0)**2), 1e-300)
    active = ~np.ma.getmaskarray(distance)
    step = 0.2*dx

    def residual(u):
        return kernels.rhs(u, sign, dx, per, True)-sign

    for _ in range(RELAX_ITERS):
        u1 = out-step*residual(out)
        u2 = 0.75*out+0.25*(u1-step*residual(u1))
        relaxed = out/3+(2/3)*(u2-step*residual(u2))
        out = np.where(active, relaxed, a)
    if narrow_cells is not None:
        returned = active & (np.abs(np.ma.getdata(distance)) < narrow_cells*dx)
        out = np.where(returned, out, a)
    return _restore(out, psi)


def extend(psi, field, dx, periodic=DEFAULT_PERIODIC):
    """scikit-fmm extension velocity, seeded by edge-interpolated front data.

    Values away from interface-adjacent cells do not seed the extension.
    All cells are extended, unlike the 2-D engine's fixed narrow band.
    """
    import skfmm
    a = _array(psi, dx)
    f = np.ascontiguousarray(np.broadcast_to(field, a.shape), dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("field must be finite; provide data at front-adjacent cells")
    _, out = skfmm.extension_velocities(a, f, dx=dx, order=2,
                                       periodic=_periodic(periodic))
    return _restore(np.where(np.ma.getmaskarray(out), f, np.ma.getdata(out)), field)


def _gradient(a, dx, periodic):
    return [(ls_numpy.shifted(a, 1, ax, periodic)
             - ls_numpy.shifted(a, -1, ax, periodic))/(2*dx) for ax in range(3)]


def grad_mag(psi, dx, periodic=DEFAULT_PERIODIC):
    """Centred gradient magnitude with wrap/edge boundaries."""
    a = _array(psi, dx)
    grad = _gradient(a, dx, _periodic(periodic))
    return _restore(np.sqrt(sum(g*g for g in grad)), psi)


def curvature(psi, dx, periodic=DEFAULT_PERIODIC):
    """div(grad psi/|grad psi|), positive 2/R on a solid sphere, in 1/cm."""
    a = _array(psi, dx)
    per = _periodic(periodic)
    grad = _gradient(a, dx, per)
    norm = np.sqrt(sum(g*g for g in grad))
    normals = [g/np.maximum(norm, 1e-30) for g in grad]
    out = sum((ls_numpy.shifted(n, 1, ax, per)
               - ls_numpy.shifted(n, -1, ax, per))/(2*dx)
              for ax, n in enumerate(normals))
    return _restore(out, psi)


def front_curvature(psi, dx, periodic=DEFAULT_PERIODIC, smooth=False, band_cells=1.5,
                    clip=True):
    """Front curvature ``kappa = div(grad psi / |grad psi|)`` [1/cm], carried off the front.

    Sign convention: psi > 0 is electrolyte, psi < 0 is solid, so the normal
    ``n = grad psi / |grad psi|`` points out of the solid and ``kappa`` is
    positive on a convex solid sphere (``+2/R``) and negative on an electrolyte
    cavity inside solid (``-2/R``).  This is the kappa of the CEAC balance
    ``d theta_a/dt = ... - theta_a kappa v``: a growing solid sphere dilutes the
    carried coverage, a shrinking cavity enriches it.

    With ``smooth=True`` psi is Gaussian-filtered with sigma = 1 cell (``mode``
    'wrap' on periodic axes, 'nearest' otherwise, passed as a per-axis tuple)
    before differencing, which damps cell-scale noise of the discrete operator.
    The curvature is evaluated only on the ``|psi| < band_cells*dx`` band and
    that band is extended to every cell along the normals of the ORIGINAL psi
    (constant off the front).  With ``clip=True`` the magnitude is capped at
    ``1/(2 dx)``, the 2-D engine's resolution guard.

    Resolution note: the finite-difference curvature is meaningful only for a
    local radius of curvature >= 3-4 dx; below that it is dominated by the
    stencil and the clip, not by geometry.
    """
    a = _array(psi, dx)
    per = _periodic(periodic)
    if not np.isfinite(band_cells) or band_cells <= 0:
        raise ValueError("band_cells must be finite and positive")
    src = a
    if smooth:
        mode = tuple("wrap" if p else "nearest" for p in per)
        src = ndi.gaussian_filter(a, sigma=1.0, mode=mode)
    k = np.asarray(curvature(src, dx, per), dtype=np.float64)
    band = np.where(np.abs(a) < band_cells*dx, k, 0.0)
    out = extend(a, band, dx, per)
    if clip:
        out = np.clip(out, -0.5/dx, 0.5/dx)
    return _restore(out, psi)


def electrolyte_fraction(psi, dx):
    """Regularised Heaviside, cosine-smoothed over +/- one cell.

    This fixed-width, monotone smooth fraction is a volume quadrature, not
    an exact planar cut-cell reconstruction. It also makes band-shift
    volume correction a continuous monotone scalar root problem.
    """
    a = _array(psi, dx)
    s = np.clip(a/dx, -1, 1)
    return np.clip(0.5*(1+s+np.sin(np.pi*s)/np.pi), 0, 1)


def solid_volume(psi, dx):
    """Smooth solid volume (cm^3), accumulated in float64."""
    return float(np.sum(1-electrolyte_fraction(psi, dx), dtype=np.float64)*dx**3)


def volume_correct(psi, target_solid_volume, dx, band_cells=8.0):
    """Uniform shift inside the original band found with Brent's root solve.

    Unreachable targets raise ValueError rather than silently missing the
    requested balance. Outside-band cells are never modified. The output
    is float64 so even float32 inputs can meet tight volume tolerances.
    """
    a = _array(psi, dx)
    if not np.isfinite(band_cells) or band_cells <= 0:
        raise ValueError("band_cells must be finite and positive")
    target = float(target_solid_volume)
    if not np.isfinite(target) or not 0 <= target <= a.size*dx**3:
        raise ValueError("target_solid_volume must lie within the domain volume")
    band = band_cells*dx
    mask = np.abs(a) < band

    def error(shift_cells):
        return (solid_volume(np.where(mask, a+shift_cells*dx, a), dx)-target)/dx**3

    if error(0) == 0:
        return a.copy()
    lo, hi = -band_cells-1, band_cells+1
    if not mask.any() or error(lo) < 0 or error(hi) > 0:
        raise ValueError("target_solid_volume is unreachable within the correction band")
    shift = brentq(error, lo, hi, xtol=1e-12, rtol=1e-14)
    return np.where(mask, a+shift*dx, a)


def label_voids(psi, bulk_mask, periodic=(False, False, False)):
    """Label sealed electrolyte components using six-face connectivity.

    Labels 1..count identify sealed pockets; solid and bulk-connected
    electrolyte are 0. Opposite faces connect along each periodic axis.
    """
    a = _array(psi)
    per = _periodic(periodic)
    bulk = np.asarray(bulk_mask, dtype=bool)
    if bulk.shape != a.shape:
        raise ValueError("bulk_mask must have the same shape as psi")
    labels, count = ndi.label(a > 0, structure=ndi.generate_binary_structure(3, 1))
    parent = np.arange(count+1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for axis, wraps in enumerate(per):
        if not wraps:
            continue
        lo, hi = np.take(labels, 0, axis), np.take(labels, -1, axis)
        pairs = np.unique(np.column_stack((lo.ravel(), hi.ravel())), axis=0)
        for left, right in pairs:
            if left and right:
                parent[root(right)] = root(left)
    lookup = np.array([root(i) for i in range(count+1)])
    labels = lookup[labels]
    connected = np.unique(labels[bulk])
    sealed = np.setdiff1d(np.unique(labels), np.r_[0, connected])
    remap = np.zeros(count+1, dtype=np.int32)
    n = len(sealed)
    remap[sealed] = np.arange(1, n+1)
    return remap[labels], n
