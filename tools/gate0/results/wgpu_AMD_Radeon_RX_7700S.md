| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wgpu | 0.33.0 | AMD Radeon RX 7700S | sweep | float32 | 160 | 616.239 | 0.0265871 | 4 | 1.41217 | 0.998555 | 0.393362 |  | passed |  |
| wgpu | 0.33.0 | AMD Radeon RX 7700S | sweep | float32 | 256 | 478.202 | 0.140336 | 4 | 1.41341 | 0.999435 | 0.393362 |  | passed |  |
| wgpu | 0.33.0 | AMD Radeon RX 7700S | capability | float64 |  |  |  |  |  |  |  |  | unsupported | float64 unavailable in this backend/device; float32 used |
