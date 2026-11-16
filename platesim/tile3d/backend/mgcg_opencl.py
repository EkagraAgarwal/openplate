"""OpenCL 1.2 GPU-resident geometric multigrid PCG (optional PyOpenCL).

The CPU hierarchy builder is the single source of conservative singleton
coarsening, masks, boundary coefficients and transpose transfer tables. Setup
and set_reaction run on the host; solve transfers b/x0 once, x once on return,
and only reduction scalars in between. Even the <=4096-cell coarse inverse is
applied on-device. Larger terminal levels use the reference's fixed sweeps.

``mixed`` stores/applies the *same fixed* V-cycle in float32, while the fine
operator, Krylov vectors, updates and reductions remain float64. This is PCG,
not flexible CG: there are no residual-dependent inner tolerances or adaptive
cycles. As with any floating-point implementation, rounding makes the computed
preconditioner only approximately linear/SPD; the f64 true residual is checked.
Instances own mutable workspaces and are not reentrant.
"""
from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import numpy as np


def available_devices():
    """Return all OpenCL devices, or raise a clear optional-runtime error."""
    try:
        import pyopencl as cl
    except ImportError as exc:
        raise ImportError("MGCG_CL requires optional dependency pyopencl") from exc
    try:
        devices = [d for p in cl.get_platforms() for d in p.get_devices()]
    except cl.Error as exc:
        raise RuntimeError("No usable OpenCL platform/device") from exc
    if not devices:
        raise RuntimeError("No OpenCL device available")
    return devices


def _source(shape, periodic, scalar):
    # C-order linear work-item IDs make neighbouring lanes span k (z), the
    # fastest axis. In mixed mode every stencil operand/accumulator is float;
    # there is no accidental double promotion in the bandwidth-heavy smoother.
    _nx, ny, nz = shape
    strides = (ny*nz, nz, 1)
    coords = ("i", "j", "k")
    expr = "real a = base[t]*v;\n"
    for axis, (n, stride, coord, wrap) in enumerate(zip(shape, strides, coords, periodic)):
        fs = list(shape)
        fs[axis] += 1
        fstride = (fs[1]*fs[2], fs[2], 1)[axis]
        index = f"(i*{fs[1]}+j)*{fs[2]}+k"
        # The periodic edge always uses face zero on the selected axis.
        edge = ["i", "j", "k"]
        edge[axis] = "0"
        edgeidx = f"({edge[0]}*{fs[1]}+{edge[1]})*{fs[2]}+{edge[2]}"
        f = f"f{axis}"
        expr += f"if ({coord}>0) a += {f}[{index}]*(v-u[t-{stride}]);\n"
        if wrap and n > 1:
            expr += f"else a += {f}[{edgeidx}]*(v-u[t+{(n-1)*stride}]);\n"
        expr += f"if ({coord}+1<{n}) a += {f}[({index})+{fstride}]*(v-u[t+{stride}]);\n"
        if wrap and n > 1:
            expr += f"else a += {f}[{edgeidx}]*(v-u[t-{(n-1)*stride}]);\n"
    return f"""
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #pragma OPENCL FP_CONTRACT OFF
    typedef {scalar} real;
    __kernel void stencil(__global const real *u, __global const real *b,
        __global const real *f0, __global const real *f1, __global const real *f2,
        __global const real *base, __global const real *diag,
        __global real *out, int color, real omega) {{
        int t=get_global_id(0), i=t/{ny*nz}, j=(t/{nz})%{ny}, k=t%{nz};
        if(color>=0 && (i+j+k)%2!=color) return;
        real v=u[t]; {expr}
        out[t] = omega==0 ? a : v+omega*(b[t]-a)/diag[t];
    }}
    __kernel void dense(__global const real *inv, __global const real *b,
                        __global real *x, int n) {{
        int t=get_global_id(0); real v=0;
        for(int j=0;j<n;j++) v+=inv[t*n+j]*b[j];
        x[t]=v;
    }}
    """


def _transfer_source(outshape, inshape, sizes):
    # One output per linear work-item: coalesced z stores and neighbouring
    # interpolation/restriction gathers, including singleton transfer tables.
    _nx, ny, nz = outshape
    sy, sz = inshape[1:]
    a, b, c = sizes
    return f"""
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #pragma OPENCL FP_CONTRACT OFF
    __kernel void transfer(__global const real *u, __global real *out,
        __global const uchar *imask, __global const uchar *omask,
        __global const int *ix, __global const int *iy, __global const int *iz,
        __global const real *wx, __global const real *wy, __global const real *wz,
        real scale) {{
        int t=get_global_id(0), i=t/{ny*nz}, j=(t/{nz})%{ny}, k=t%{nz};
        real v=0;
        for(int a=0;a<{a};a++) for(int b=0;b<{b};b++) for(int c=0;c<{c};c++) {{
            int s=(ix[i*{a}+a]*{sy}+iy[j*{b}+b])*{sz}+iz[k*{c}+c];
            if(imask[s]) v+=wx[i*{a}+a]*wy[j*{b}+b]*wz[k*{c}+c]*u[s];
        }}
        out[t]=omask[t] ? scale*v : 0;
    }}
    """


class MGCG_CL:
    """MGCG API plus ``device`` (name substring or enumeration index).

    dtype is float64 (default) or "mixed"; output is always float64.
    Device enumeration is platform order, then device order. No vendor features
    beyond cl_khr_fp64 are used. Host hierarchy construction requires Numba/SciPy
    just like MGCG, but invokes no Numba kernels during solve.
    """

    def __init__(self, kx, ky, kz, dx, reaction=None, active=None, dirichlet=None,
                 periodic=(True, True, False), dtype=np.float64, levels=None,
                 smoother="rbgs", pre=2, post=2, device: str | int = "gfx1102"):
        import pyopencl as cl
        import pyopencl.array as ca

        from .mgcg_numba import MGCG

        self.mixed = isinstance(dtype, str) and dtype == "mixed"
        if not self.mixed and np.dtype(dtype) != np.dtype("float64"):
            raise ValueError('dtype must be float64 or "mixed"')
        self.dtype = np.dtype("float64")
        self._vdtype = np.dtype("float32" if self.mixed else "float64")
        devices = available_devices()
        if isinstance(device, (int, np.integer)):
            index = int(device)
            if not 0 <= index < len(devices):
                raise RuntimeError(f"OpenCL device index {device} unavailable")
            selected = devices[index]
        else:
            matches = [d for d in devices if str(device).lower() in d.name.lower()]
            if not matches:
                raise RuntimeError(f"OpenCL device matching {device!r} unavailable")
            selected = matches[0]
        if "cl_khr_fp64" not in selected.extensions:
            raise RuntimeError(f"OpenCL device {selected.name} requires cl_khr_fp64")
        self.device = selected
        self.context = cl.Context([selected])
        self.queue = cl.CommandQueue(self.context)
        self._ca, self._cl = ca, cl
        self._host = MGCG(kx, ky, kz, dx, reaction, active, dirichlet,
                          periodic, np.float64, levels, smoother, pre, post)
        self.shape, self.periodic = self._host.shape, self._host.periodic
        self.hierarchy = []
        for h in self._host.hierarchy:
            l = SimpleNamespace(shape=h.x.shape, size=h.x.size, rbgs=h.rbgs,
                                factors=h.factors, transfers=h.transfers)
            l.faces = tuple(self._upload(f, self._vdtype) for f in h.faces)
            l.mask = self._upload(h.mask, np.uint8)
            for name in ("x", "rhs", "work", "correction"):
                setattr(l, name, ca.zeros(self.queue, (l.size,), self._vdtype))
            p = cl.Program(self.context, _source(l.shape, periodic,
                           "float" if self.mixed else "double")).build(options=["-cl-std=CL1.2"])
            l.stencil, l.dense = p.stencil, p.dense
            self.hierarchy.append(l)
        self._fine = SimpleNamespace(shape=self.shape, size=int(np.prod(self.shape)))
        self._fine.faces = tuple(self._upload(f) for f in self._host.hierarchy[0].faces)
        self._fine.stencil = cl.Program(self.context, _source(self.shape, periodic, "double")).build(
            options=["-cl-std=CL1.2"]).stencil
        # Hand-written reduction avoids PyOpenCL.reduction's optional Mako
        # dependency. All partial sums and products are double on the device.
        self._reduce = cl.Program(self.context, """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #pragma OPENCL FP_CONTRACT OFF
            __kernel void reduce(__global const double *a, __global const double *b,
                                 __global double *out, int n, int dot) {
                int lid=get_local_id(0); double value=0;
                for(int t=get_global_id(0);t<n;t+=get_global_size(0))
                    value += dot ? a[t]*b[t] : a[t];
                __local double sums[128]; sums[lid]=value;
                barrier(CLK_LOCAL_MEM_FENCE);
                for(int s=64;s>0;s/=2) {
                    if(lid<s) sums[lid]+=sums[lid+s];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if(lid==0) out[get_group_id(0)]=sums[0];
            }
        """).build(options=["-cl-std=CL1.2"]).reduce
        for fine, coarse in zip(self.hierarchy, self.hierarchy[1:]):
            fine.prolong = self._transfer(fine, coarse, fine.transfers[0:2])
            fine.restrict = self._transfer(coarse, fine, fine.transfers[2:4])
        self._update_diagonal()
        self.queue.finish()

    def _upload(self, a, dtype=np.float64):
        values = np.ascontiguousarray(a, dtype=dtype).ravel()
        if not np.isfinite(values).all():
            raise ValueError("device storage overflow or nonfinite coefficients")
        return self._ca.to_device(self.queue, values)

    def _transfer(self, out, inp, tables):
        ids, weights = tables
        source = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        source += "typedef " + ("float" if self.mixed else "double") + " real;\n"
        source += _transfer_source(out.shape, inp.shape, [a.shape[1] for a in ids])
        kernel = self._cl.Program(self.context, source).build(options=["-cl-std=CL1.2"]).transfer
        return kernel, tuple(self._upload(a, np.int32) for a in ids), tuple(
            self._upload(a, self._vdtype) for a in weights)

    def _update_diagonal(self):
        for l, h in zip(self.hierarchy, self._host.hierarchy):
            for name in ("base", "diag"):
                a = np.asarray(getattr(h, name), dtype=self._vdtype).ravel()
                if not np.isfinite(a).all():
                    raise ValueError("device storage overflow in diagonal")
                if hasattr(l, name):
                    getattr(l, name).set(a)
                else:
                    setattr(l, name, self._upload(a, self._vdtype))
        l = self.hierarchy[-1]
        inverse = self._host._coarse_solve(np.eye(l.size)) if l.size <= 4096 else None
        l.inverse = None if inverse is None else self._upload(inverse, self._vdtype)
        for name in ("base", "diag"):
            setattr(self._fine, name, self._upload(getattr(self._host.hierarchy[0], name)))

    def set_reaction(self, reaction):
        """Refresh diagonal buffers and coarse inverse; retain faces/transfers."""
        self._host.set_reaction(reaction)
        self._update_diagonal()
        self.queue.finish()

    def _stencil(self, l, u, out, rhs=None, color=-1, omega: float = 0):
        cast = u.dtype.type
        l.stencil(self.queue, (l.size,), None, u.data, (u if rhs is None else rhs).data,
                  *(f.data for f in l.faces), l.base.data, l.diag.data, out.data,
                  np.int32(color), cast(omega))

    def _smooth(self, l, sweeps, reverse=False):
        if any(p and n % 2 for p, n in zip(self.periodic, l.shape)):
            sweeps *= 2
        for _ in range(sweeps):
            if l.rbgs:
                for color in ((1, 0) if reverse else (0, 1)):
                    self._stencil(l, l.x, l.x, l.rhs, color, 1)
            else:
                self._stencil(l, l.x, l.work, l.rhs, -1, 2/3)
                l.x, l.work = l.work, l.x

    def _move(self, table, inp, out, imask, omask, scale):
        kernel, ids, weights = table
        kernel(self.queue, (out.size,), None, inp.data, out.data, imask.data, omask.data,
               *(a.data for a in (*ids, *weights)), self._vdtype.type(scale))

    def _cycle(self, index=0):
        l = self.hierarchy[index]
        l.x.fill(0)
        if index == len(self.hierarchy)-1:
            if l.inverse is not None:
                l.dense(self.queue, (l.size,), None, l.inverse.data, l.rhs.data,
                        l.x.data, np.int32(l.size))
            else:
                self._smooth(l, 100)
                self._smooth(l, 100, True)
            return
        self._smooth(l, self._host.pre)
        self._stencil(l, l.x, l.work)
        l.work[:] = l.rhs - l.work
        c = self.hierarchy[index+1]
        self._move(l.restrict, l.work, c.rhs, l.mask, c.mask, 1/np.prod(l.factors))
        self._cycle(index+1)
        self._move(l.prolong, c.x, l.correction, c.mask, l.mask, 1)
        l.x += l.correction
        self._smooth(l, self._host.post, True)

    def _precondition(self, r):
        l = self.hierarchy[0]
        l.rhs[:] = r.astype(self._vdtype)
        self._cycle()
        return self._ca.if_positive(l.mask, l.x.astype(np.float64), r)

    def _input(self, u, strict=False):
        u = np.asarray(u, dtype=np.float64)
        if u.size != np.prod(self.shape) or not np.isfinite(u).all():
            raise ValueError("input must be finite and match cell count")
        if strict and u.shape not in (self.shape, (int(np.prod(self.shape)),)):
            raise ValueError("u must be cell-shaped or C-order flat")
        return u

    def apply(self, u):
        """Return the f64 fine operator applied to a cell-shaped/flat input."""
        u = self._input(u, True)
        d = self._upload(u)
        out = self._ca.empty_like(d)
        self._stencil(self._fine, d, out)
        return out.get().reshape(u.shape)

    def vcycle(self, r):
        """Apply one fixed V-cycle, including inactive identity rows."""
        r = self._input(r)
        return self._precondition(self._upload(r)).get().reshape(r.shape)

    def _dot(self, a, b):
        n, dot = a.size, 1
        while True:
            groups = min(1024, (n+127)//128)
            out = self._ca.empty(self.queue, (groups,), np.float64)
            self._reduce(self.queue, (groups*128,), (128,), a.data, b.data,
                         out.data, np.int32(n), np.int32(dot))
            if groups == 1:
                return float(out.get()[0])
            a, b, n, dot = out, out, groups, 0

    def solve(self, b, x0=None, tol=1e-8, maxiter=200):
        """Device PCG; scalar reductions only cross the bus inside iterations."""
        start = perf_counter()
        b = self._input(b)
        if not np.isfinite(tol) or tol <= 0 or maxiter < 0 or int(maxiter) != maxiter:
            raise ValueError("tol must be positive and maxiter a nonnegative integer")
        bd = self._upload(b)
        x = self._ca.zeros_like(bd) if x0 is None else self._upload(self._input(x0))
        ap = self._ca.empty_like(x)
        self._stencil(self._fine, x, ap)
        r = bd - ap
        norm = np.sqrt(self._dot(bd, bd)) or 1.0
        relative = np.sqrt(self._dot(r, r)) / norm
        p = self._ca.zeros_like(x)
        old, restart, iterations = 0., True, 0
        for step in range(int(maxiter)):
            if relative <= tol:
                break
            z = self._precondition(r)
            rho = self._dot(r, z)
            if not np.isfinite(rho) or rho <= 0:
                break
            p[:] = z if restart else z + (rho/old)*p
            restart = False
            self._stencil(self._fine, p, ap)
            denominator = self._dot(p, ap)
            if not np.isfinite(denominator) or denominator <= 0:
                break
            alpha = rho/denominator
            x += alpha*p
            r -= alpha*ap
            iterations = step+1
            relative = np.sqrt(self._dot(r, r))/norm
            old = rho
            if relative <= tol:
                self._stencil(self._fine, x, ap)
                r[:] = bd-ap
                relative = np.sqrt(self._dot(r, r))/norm
                restart = relative > tol
        self._stencil(self._fine, x, ap)
        r[:] = bd-ap
        relative = np.sqrt(self._dot(r, r))/norm
        result = x.get().reshape(b.shape)
        return result, {"iterations": iterations, "rel_residual": float(relative),
                        "converged": bool(relative <= tol), "seconds": perf_counter()-start,
                        "stagnated": False}
