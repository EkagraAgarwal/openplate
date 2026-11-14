| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quadrants | 1.4.0 | AMD Radeon(TM) Graphics | sweep | float32 | 160 | 239.994 | 0.0682685 | 4 | 1.41217 | 0.998555 | 0.233402 |  | passed |  |
| quadrants | 1.4.0 | AMD Radeon(TM) Graphics | sweep | float32 | 256 | 244.713 | 0.274235 | 4 | 1.41341 | 0.999435 | 0.233402 |  | passed |  |
| quadrants | 1.4.0 | AMD Radeon(TM) Graphics | capability | float64 |  |  |  |  |  |  |  |  | unsupported | float64 unavailable in this backend/device; float32 used |
| quadrants | 1.4.0 | AMD Radeon(TM) Graphics | mgpcg | float32 | 160 |  |  |  |  |  |  |  | failed | ArithmeticError: CG failed: residual=2.59202e+27, error=4.82507e+25 |
