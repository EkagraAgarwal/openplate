"""Same kernels for Taichi and its Quadrants fork (imported by entrypoints)."""
import os
import numpy as np
from common import benchmark, select
from vulkan_devices import devices


def run(name, args):
    ds = devices()
    if args.list:
        import importlib
        importlib.import_module(name)
        for d in ds:
            print(f'{d.index}: {d.name}; hardware fp64={d.fp64}; bytes={d.memory}')
        return 0
    device = select(ds, args.device)
    os.environ['TI_VISIBLE_DEVICE' if name == 'taichi' else 'QD_VISIBLE_DEVICE'] = str(device.index)
    if name == 'taichi':
        import taichi as ti
    else:
        import quadrants as ti
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False,
            log_level=ti.WARN)
    fp64 = device.fp64 and ti._lib.core.is_extension_supported(ti.vulkan, ti.extension.data64)

    @ti.kernel
    def stencil(u: ti.types.ndarray(), b: ti.types.ndarray(), k: ti.types.ndarray(),
                out: ti.types.ndarray(), n: ti.i32, color: ti.i32, apply: ti.i32,
                scalar: ti.template()):
        for i, j, z in ti.ndrange(n, n, n):
            if apply != 0 or (i+j+z) % 2 == color:
                diagonal = ti.cast(0, scalar)
                neighbors = ti.cast(0, scalar)
                for axis in ti.static(range(3)):
                    for direction in ti.static((-1, 1)):
                        a, c, d = i, j, z
                        if ti.static(axis == 0):
                            a += direction
                        elif ti.static(axis == 1):
                            c += direction
                        else:
                            d += direction
                        if a < 0 or c < 0 or d < 0 or a >= n or c >= n or d >= n:
                            diagonal += 2*k[i,j,z]*n*n
                        else:
                            w = 2*k[i,j,z]*k[a,c,d]/(k[i,j,z]+k[a,c,d])*n*n
                            diagonal += w
                            neighbors += w*u[a,c,d]
                if apply != 0:
                    out[i,j,z] = diagonal*u[i,j,z]-neighbors
                else:
                    u[i,j,z] = (b[i,j,z]+neighbors)/diagonal

    class Backend:
        name = device.name
        memory = device.memory

        def execute(self, u, b, k, sweeps=0, reverse=False):
            dt = ti.f64 if u.dtype == np.float64 else ti.f32
            arrays = [ti.ndarray(dt, shape=u.shape) for _ in range(4)]
            for array, host in zip(arrays, (u, b, k)):
                array.from_numpy(np.ascontiguousarray(host))
            if sweeps:
                for _ in range(sweeps):
                    for color in ((1, 0) if reverse else (0, 1)):
                        stencil(*arrays, len(u), color, 0, dt)
                ti.sync()
                return arrays[0].to_numpy()
            stencil(*arrays, len(u), 0, 1, dt)
            ti.sync()
            return arrays[3].to_numpy()

        def apply(self, u, k):
            return self.execute(u, u, k)

        def smooth(self, u, b, k, sweeps, reverse):
            return self.execute(u, b, k, sweeps, reverse)

    print(f'Vulkan loader device index {device.index} selected via visible-device environment variable.')
    return benchmark(name, Backend(), args, fp64)
