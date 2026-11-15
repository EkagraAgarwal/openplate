| framework | version | device | test | dtype | N | mcups | seconds | iterations | relative_residual | relative_error | compile_seconds | peak_bytes | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pyopencl | 2026.1.5 | gfx1035 | sweep | float32 | 160 | 466.718 | 0.0351047 | 4 | 1.41217 | 0.998555 | 0.0849879 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1035 | sweep | float32 | 256 | 427.409 | 0.157013 | 4 | 1.41341 | 0.999435 | 0.0849879 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1035 | sweep | float64 | 160 | 195.963 | 0.0836076 | 4 | 1.41217 | 0.998555 | 0.0168421 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1035 | sweep | float64 | 256 | 170.613 | 0.393339 | 4 | 1.41342 | 0.999435 | 0.0168421 |  | passed |  |
| pyopencl | 2026.1.5 | gfx1035 | mgpcg | float64 | 160 |  | 4.20228 | 7 | 1.09705e-09 | 4.48395e-11 | 0.0168421 |  | passed |  |
