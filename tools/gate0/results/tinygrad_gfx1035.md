| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tinygrad | 0.14.0 | gfx1035 | sweep | float32 | 160 | 31.3333 | 0.522894 | 4 | 1.41217 | 0.998555 | 1.28133 |  | passed |  |
| tinygrad | 0.14.0 | gfx1035 | sweep | float32 | 256 | 37.9832 | 1.7668 | 4 | 1.41341 | 0.999435 | 1.28133 |  | passed |  |
| tinygrad | 0.14.0 | gfx1035 | capability | float64 |  |  |  |  |  |  |  |  | unsupported | float64 unavailable in this backend/device; float32 used |
