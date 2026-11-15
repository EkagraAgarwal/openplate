| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pyopencl | 2026.1.5 | gfx1102 | sweep | float32 | 160 | 604.609 | 0.0270985 | 4 | 1.41217 | 0.998555 | 0.731945 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1102 | sweep | float32 | 256 | 975.966 | 0.0687615 | 4 | 1.41341 | 0.999435 | 0.731945 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1102 | sweep | float64 | 160 | 268.528 | 0.0610141 | 4 | 1.41217 | 0.998555 | 0.663414 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1102 | sweep | float64 | 256 | 333.432 | 0.201267 | 4 | 1.41342 | 0.999435 | 0.663414 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1102 | mgpcg | float64 | 160 |  | 3.73463 | 7 | 1.09705e-09 | 4.48395e-11 | 0.663414 |  | passed |  |
