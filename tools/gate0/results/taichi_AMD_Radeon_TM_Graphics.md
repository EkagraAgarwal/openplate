| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taichi | 1.7.4 | AMD Radeon(TM) Graphics | sweep | float32 | 160 | 260.355 | 0.0629295 | 4 | 1.41217 | 0.998555 | 0.633937 |  | passed |  |
| taichi | 1.7.4 | AMD Radeon(TM) Graphics | sweep | float32 | 256 | 160.424 | 0.418321 | 4 | 1.41341 | 0.999435 | 0.633937 |  | passed |  |
| taichi | 1.7.4 | AMD Radeon(TM) Graphics | capability | float64 |  |  |  |  |  |  |  |  | unsupported | float64 unavailable in this backend/device; float32 used |
| taichi | 1.7.4 | AMD Radeon(TM) Graphics | mgpcg | float32 | 160 |  |  |  |  |  |  |  | failed | ArithmeticError: CG failed: residual=2.59202e+27, error=4.82507e+25 |
