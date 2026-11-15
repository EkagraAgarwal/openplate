| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taichi | 1.7.4 | AMD Radeon RX 7700S | sweep | float32 | 160 | 306.403 | 0.053472 | 4 | 1.41217 | 0.998555 | 0.975942 |  | passed |  |
| taichi | 1.7.4 | AMD Radeon RX 7700S | sweep | float32 | 256 | 407.124 | 0.164837 | 4 | 1.41341 | 0.999435 | 0.975942 |  | passed |  |
| taichi | 1.7.4 | AMD Radeon RX 7700S | capability | float64 |  |  |  |  |  |  |  |  | unsupported | float64 unavailable in this backend/device; float32 used |
| taichi | 1.7.4 | AMD Radeon RX 7700S | mgpcg | float32 | 160 |  |  |  |  |  |  |  | failed | ArithmeticError: CG failed: residual=3.45432e+27, error=4.57993e+25 |
