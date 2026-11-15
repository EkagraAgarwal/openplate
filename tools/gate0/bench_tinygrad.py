"""Float32 tensor RBGS on tinygrad's cross-vendor CL backend (first platform)."""
import ctypes
import numpy as np
from common import benchmark, coefficients, parser, select, failure


def devices():
    from tinygrad import Device
    from tinygrad.runtime.ops_cl import CLDevice, cl, check
    Device['CL']  # initializes first platform's device list
    names = []
    for i, handle in enumerate(CLDevice.device_ids):
        buffer = ctypes.create_string_buffer(1024)
        check(cl.clGetDeviceInfo(handle, cl.CL_DEVICE_NAME, len(buffer), buffer, None))
        names.append((i, buffer.value.decode().strip()))
    return names


class Tiny:
    def __init__(self, index, name):
        from tinygrad import Device, Tensor
        self.Tensor = Tensor
        self.device = f'CL:{index}'
        self.name = Device[self.device].device_name.strip()
        if self.name != name:
            raise RuntimeError('tinygrad device name does not match enumerated device')

    def execute(self, u, b, k, sweeps=0, reverse=False):
        Tensor = self.Tensor
        device = self.device
        tensor = lambda a: Tensor(np.ascontiguousarray(a), device=device).realize()
        diag, faces = coefficients(k)
        diagonal, rhs, value = tensor(diag), tensor(b), tensor(u)
        weights = [(axis, direction, tensor(w)) for axis, direction, w in faces]
        n = len(u)
        def neighbors(v):
            padded = v.pad(((1, 1), (1, 1), (1, 1)))
            result = v*0
            for axis, direction, w in weights:
                slices = [(1, n+1)]*3
                slices[axis] = (1+direction, n+1+direction)
                result = result + w*padded.shrink(tuple(slices))
            return result
        if not sweeps:
            return (diagonal*value-neighbors(value)).numpy()
        parity = np.indices(u.shape).sum(axis=0) % 2
        masks = [tensor(parity == color) for color in (0, 1)]
        for _ in range(sweeps):
            for color in ((1, 0) if reverse else (0, 1)):
                value = masks[color].where((rhs+neighbors(value))/diagonal, value).realize()
        return value.numpy()

    def apply(self, u, k):
        return self.execute(u, u, k)

    def smooth(self, u, b, k, sweeps, reverse):
        return self.execute(u, b, k, sweeps, reverse)


if __name__ == '__main__':
    args = parser(False).parse_args()
    try:
        ds = devices()
        if args.list:
            for i, name in ds:
                print(f'{i}: {name}; backend=CL (first OpenCL platform)')
        else:
            if args.device.isdecimal():
                index, name = ds[int(args.device)]
            else:
                name = select([name for _, name in ds], args.device)
                index = next(i for i, label in ds if label == name)
            raise SystemExit(benchmark('tinygrad', Tiny(index, name), args, False, True))
    except Exception as exc:
        failure('tinygrad', args.device, exc)
        raise SystemExit(1)
