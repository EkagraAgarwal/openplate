| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quadrants | 1.4.0 | AMD Radeon RX 7700S | sweep | float32 | 160 | 394.523 | 0.0415286 | 4 | 1.41217 | 0.998555 | 0.572077 |  | passed |  |
| quadrants | 1.4.0 | AMD Radeon RX 7700S | sweep | float32 | 256 | 338.703 | 0.198135 | 4 | 1.41341 | 0.999435 | 0.572077 |  | passed |  |
| quadrants | 1.4.0 | AMD Radeon RX 7700S | capability | float64 |  |  |  |  |  |  |  |  | unsupported | float64 unavailable in this backend/device; float32 used |
| quadrants | 1.4.0 | AMD Radeon RX 7700S | mgpcg | float32 | 160 |  |  |  |  |  |  |  | failed | ArithmeticError: CG failed: residual=3.45432e+27, error=4.57993e+25 |
