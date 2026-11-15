"""Isolated CPU baseline; see run_all.md for uv command."""
import numpy as np
import os
import platform
import subprocess
from numba import njit, prange, get_num_threads, set_num_threads
from common import benchmark, parser, failure


@njit(parallel=True, cache=False)
def kernel(u, b, k, out, color, apply):
    n = len(u)
    for ii in prange(n):
        i = np.int64(ii)
        for j in range(n):
            for z in range(n):
                if not apply and (i+j+z) % 2 != color:
                    continue
                diagonal = 0.0
                neighbors = 0.0
                for axis in range(3):
                    for direction in (-1, 1):
                        a, c, d = i, j, z
                        if axis == 0:
                            a += direction
                        elif axis == 1:
                            c += direction
                        else:
                            d += direction
                        if a < 0 or c < 0 or d < 0 or a >= n or c >= n or d >= n:
                            diagonal += 2*k[i,j,z]*n*n
                        else:
                            w = 2*k[i,j,z]*k[a,c,d]/(k[i,j,z]+k[a,c,d])*n*n
                            diagonal += w
                            neighbors += w*u[a,c,d]
                if apply:
                    out[i,j,z] = diagonal*u[i,j,z]-neighbors
                else:
                    u[i,j,z] = (b[i,j,z]+neighbors)/diagonal


class CPU:
    def __init__(self):
        self.name = platform.processor() or 'CPU'
        if os.name == 'nt':
            try:
                name = subprocess.check_output(
                    ['powershell', '-NoProfile', '-Command',
                     '(Get-CimInstance Win32_Processor).Name'],
                    text=True, timeout=10).strip()
                if name:
                    self.name = name
            except (OSError, subprocess.SubprocessError):
                pass

    def apply(self, u, k):
        out = np.empty_like(u)
        kernel(u, u, k, out, 0, True)
        return out

    def smooth(self, u, b, k, sweeps, reverse):
        for _ in range(sweeps):
            for color in ((1, 0) if reverse else (0, 1)):
                kernel(u, b, k, u, color, False)
        return u


if __name__ == '__main__':
    args = parser().parse_args()
    if args.list:
        print('0: CPU (numba parallel); threads:', get_num_threads())
    else:
        # Eight physical cores; avoid oversubscribing this 16-GB laptop.
        set_num_threads(min(8, get_num_threads()))
        try:
            raise SystemExit(benchmark('numba', CPU(), args))
        except Exception as exc:
            failure('numba', 'CPU', exc)
            raise SystemExit(1)
