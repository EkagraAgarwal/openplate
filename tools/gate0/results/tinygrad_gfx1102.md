| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tinygrad | 0.14.0 | gfx1102 | sweep | float32 | 160 | 39.4941 | 0.414847 | 4 | 1.41217 | 0.998555 | 2.21576 |  | passed |  |
| tinygrad | 0.14.0 | gfx1102 | sweep | float32 | 256 | 48.9084 | 1.37213 | 4 | 1.41341 | 0.999435 | 2.21576 |  | passed |  |
| tinygrad | 0.14.0 | gfx1102 | capability | float64 |  |  |  |  |  |  |  |  | unsupported | float64 unavailable in this backend/device; float32 used |
