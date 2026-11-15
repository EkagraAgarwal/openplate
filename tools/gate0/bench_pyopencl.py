"""OpenCL C RBGS, with common host-orchestrated geometric MGPCG."""
import numpy as np
from common import benchmark, parser, select, failure

SOURCE = r'''
#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#else
typedef float real;
#endif
__kernel void stencil(__global real *u, __global const real *b,
 __global const real *k, __global real *out, int n, int color, int apply) {
 int t=get_global_id(0); if(t>=n*n*n) return;
 int z=t%n, j=(t/n)%n, i=t/(n*n);
 if(!apply && ((i+j+z)&1)!=color) return;
 real diagonal=0, neighbors=0, center=k[t];
 for(int axis=0;axis<3;axis++) for(int d=-1;d<=1;d+=2) {
   int coord=axis==0?i:(axis==1?j:z);
   int stride=axis==0?n*n:(axis==1?n:1);
   if(coord+d<0 || coord+d>=n) diagonal+=2*center*n*n;
   else { int q=t+d*stride; real w=2*center*k[q]/(center+k[q])*n*n;
          diagonal+=w; neighbors+=w*u[q]; }
 }
 if(apply) out[t]=diagonal*u[t]-neighbors;
 else u[t]=(b[t]+neighbors)/diagonal;
}
'''


def devices():
    import pyopencl as cl
    return [d for p in cl.get_platforms() for d in p.get_devices()
            if d.type & cl.device_type.GPU]


class OpenCL:
    def __init__(self, device):
        import pyopencl as cl
        self.cl = cl
        self.name = device.name.strip()
        self.memory = device.global_mem_size
        self.fp64 = bool(device.double_fp_config)
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self.programs = {}

    def execute(self, u, b, k, sweeps=0, reverse=False):
        cl, ctx, queue = self.cl, self.context, self.queue
        dtype = u.dtype.name
        if dtype not in self.programs:
            options = ['-DUSE_DOUBLE'] if dtype == 'float64' else []
            program = cl.Program(ctx, SOURCE).build(options=options)
            self.programs[dtype] = cl.Kernel(program, 'stencil')
        kernel = self.programs[dtype]
        mf = cl.mem_flags
        bufs = [cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=np.ascontiguousarray(a)) for a in (u, b, k)]
        out = cl.Buffer(ctx, mf.READ_WRITE, u.nbytes)
        n = np.int32(len(u))
        if sweeps:
            for _ in range(sweeps):
                for color in ((1, 0) if reverse else (0, 1)):
                    kernel(queue, (u.size,), None, *bufs, out, n,
                           np.int32(color), np.int32(0))
            result = np.empty_like(u)
            cl.enqueue_copy(queue, result, bufs[0]).wait()
        else:
            kernel(queue, (u.size,), None, *bufs, out, n, np.int32(0), np.int32(1))
            result = np.empty_like(u)
            cl.enqueue_copy(queue, result, out).wait()
        return result

    def apply(self, u, k):
        return self.execute(u, u, k)

    def smooth(self, u, b, k, sweeps, reverse):
        return self.execute(u, b, k, sweeps, reverse)


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        ds = devices()
        if args.list:
            for i, d in enumerate(ds):
                print(f'{i}: {d.name.strip()}; fp64={bool(d.double_fp_config)}; bytes={d.global_mem_size}')
        else:
            backend = OpenCL(select(ds, args.device))
            raise SystemExit(benchmark('pyopencl', backend, args, backend.fp64))
    except Exception as exc:
        failure('pyopencl', args.device, exc)
        raise SystemExit(1)
