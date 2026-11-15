# Gate 0: isolated, cross-vendor backend experiment

Run from the repository root in PowerShell. These commands use Python 3.12
and uv's isolated cache, **not** the project environment or `uv.lock`.
All outputs are under `tools/gate0/results/`. No project dependency changes.

## Environment and enumeration

```powershell
uv run --no-project --python 3.12 --with numpy --with numba --with taichi==1.7.4 --with quadrants --with pyopencl --with tinygrad --with wgpu --with vulkan --with pyvista --with gdstk --with ezdxf --with mcp --with onnxruntime python tools/gate0/check_env.py --list
uv run --no-project --python 3.12 --with numpy --with numba python tools/gate0/bench_numba.py --list
uv run --no-project --python 3.12 --with numpy --with taichi==1.7.4 --with vulkan python tools/gate0/bench_taichi.py --list
uv run --no-project --python 3.12 --with numpy --with quadrants --with vulkan python tools/gate0/bench_quadrants.py --list
uv run --no-project --python 3.12 --with numpy --with pyopencl python tools/gate0/bench_pyopencl.py --list
uv run --no-project --python 3.12 --with numpy --with tinygrad python tools/gate0/bench_tinygrad.py --list
uv run --no-project --python 3.12 --with numpy --with wgpu python tools/gate0/bench_wgpu.py --list
```

`check_env` checks installed packages, not packages in other isolated uv runs.
The long command installs all requested optional packages into one disposable
environment. The sphere PNG is `results/pyvista_sphere.png`.

## CPU and both GPU commands

On this checkout only the integrated adapter is currently exposed: Vulkan
index 0 `AMD Radeon(TM) Graphics` (device ID 0x1681); OpenCL index 0 `gfx1035`.
The RX 7700S is **absent** from both APIs. The commands below intentionally
fail rather than silently benchmarking the integrated GPU as the RX 7700S.
When the discrete adapter is enabled in firmware/driver/OS, rerun `--list`.
Device indices are framework-specific; do not assume index 1 is discrete.

```powershell
uv run --no-project --python 3.12 --with numpy --with numba python tools/gate0/bench_numba.py

# Taichi: integrated / discrete
uv run --no-project --python 3.12 --with numpy --with taichi==1.7.4 --with vulkan python tools/gate0/bench_taichi.py --device 'Radeon(TM)'
uv run --no-project --python 3.12 --with numpy --with taichi==1.7.4 --with vulkan python tools/gate0/bench_taichi.py --device 'RX 7700S'
# Quadrants: integrated / discrete
uv run --no-project --python 3.12 --with numpy --with quadrants --with vulkan python tools/gate0/bench_quadrants.py --device 'Radeon(TM)'
uv run --no-project --python 3.12 --with numpy --with quadrants --with vulkan python tools/gate0/bench_quadrants.py --device 'RX 7700S'
# OpenCL: integrated / discrete
uv run --no-project --python 3.12 --with numpy --with pyopencl python tools/gate0/bench_pyopencl.py --device gfx1035
uv run --no-project --python 3.12 --with numpy --with pyopencl python tools/gate0/bench_pyopencl.py --device 'RX 7700S'
# tinygrad CL: integrated / discrete (first OpenCL platform only)
uv run --no-project --python 3.12 --with numpy --with tinygrad python tools/gate0/bench_tinygrad.py --device gfx1035
uv run --no-project --python 3.12 --with numpy --with tinygrad python tools/gate0/bench_tinygrad.py --device 'RX 7700S'
# wgpu Vulkan WGSL: integrated / discrete
uv run --no-project --python 3.12 --with numpy --with wgpu python tools/gate0/bench_wgpu.py --device 'Radeon(TM)'
uv run --no-project --python 3.12 --with numpy --with wgpu python tools/gate0/bench_wgpu.py --device 'RX 7700S'

uv run --no-project --python 3.12 --with numpy --with numba python tools/gate0/test_common.py
uv run --no-project --python 3.12 --with numpy python tools/gate0/summarize.py
```

Executed here: all full default benchmarks with `--device 0` (both sweep sizes),
CPU defaults, every framework's `--list`, full environment check, and discrete
requests with `--device 'RX 7700S' --sizes 160`. Initial N=160-only runs were
also made. To reproduce those, use `--device 0 --sizes 160`.
Defaults benchmark both 160 and 256; solve defaults to 160. `--no-solve` limits
to sweeps; `--solve-n 16 --sizes 16` is a quick correctness smoke test.
`--maxiter 200` bounds solves. GPU scripts select devices explicitly, never CPU
fallback. Invalid/ambiguous selectors fail and produce JSON with exact errors.

## Algorithm, validation, and timing contract

- Unit cube in cm; C-order `[x,y,z]`, cell centres `(i+.5)/N`, zero Dirichlet
  values at outer faces. Positive `-div(k grad u)` uses harmonic face means,
  doubled boundary-face diagonal (half-cell distance), and no wrapped neighbors.
- Sweep: k=1, four complete red-then-black GS sweeps, reset u=0 each trial.
  One cell update means updating one colour once (N^3 updates per full sweep).
  Residual need not decrease monotonically after a colour split; the sine-mode
  example has residual near sqrt(2), while solution error decreases. This is
  not a converged solve. Random N=8 operator and sweep reference comparisons
  must pass before timings are accepted; residual and solution error are saved.
- Variable k = 1 + .2x + .1y + .1z. The exact solve solution is
  sin(4*pi*x) sin(4*pi*y) sin(4*pi*z); using four half-waves keeps the float32
  cancellation-limited true-residual floor below 1e-5 at N=160. The sweep uses
  sin(pi*x) sin(pi*y) sin(pi*z). The manufactured exact solution and discrete
  RHS are shared across all frameworks; no discretization error is hidden in
  the stopping tolerance. The reference operator independently checks kernels.
- MG: halve even grid sizes down to <=5 (160,80,40,20,10,5). Rediscretize k
  per level. Two RB pre-sweeps, two BR post-sweeps. Coarsest: 12 RB then
  12 BR sweeps (24 total). Cell-centred trilinear interpolation with odd
  reflection at faces; full weighting R=P^T/8. This symmetric cycle preconditions
  CG. Float64 reductions and true recomputed residual each iteration. Tests
  check transfer adjointness, operator symmetry/positivity, and manufactured solve.
- All backends use identical host CG/transfer operations. Only stencil apply
  and smoothing kernels are framework-specific. Host/device allocation,
  upload/download and synchronization are included. This is **not** a fully
  GPU-resident solver: the decision applies to these reference implementations,
  not the maximum performance of each framework. tinygrad constructs coefficient
  arrays on the host too; this cost is included. wgpu/tinygrad are sweep-only.
- One warm-up is excluded, then **best of three** complete calls. Solve timing
  includes problem/hierarchy setup. `compile_seconds` is first apply+smoother
  validation call wall time (upper bound including launch/transfers, not a pure
  compiler profiler); `warmup_seconds` is separately recorded. Backend/device
  initialization before validation is not timed. Per-process caches are disabled
  for Numba/Taichi; driver caches can persist. FP64 support is reported and
  unsupported float64 rows are explicit; float32 solve target is 1e-5 vs 1e-8
  float64. No unconverged or wrong answer is a passing solve.
- `peak_bytes=null` means not obtainable, **not** zero. N=256 is skipped if the
  conservative 24-field estimate exceeds 70% of reported device memory. There
  is no reliable portable host-available-memory estimate; allocation failures
  are recorded. Shared system memory reported by drivers is not dedicated VRAM.

## Decision

A framework qualifies only if it runs sweeps AND the MGPCG solve on **both**
GPUs, and RX 7700S N=160 MGPCG time is <= CPU N=160 MGPCG time / 5. Compare
`test=mgpcg`, not MCUPS, only `status=passed`. Report dtype/tolerance differences
when comparing a float32-only GPU to float64 CPU. Sweep-only candidates cannot
qualify yet. An absent adapter or failed solve is missing evidence, never a win.
`summarize.py` merges every JSON (including failures) into `results/summary.md`.

Quadrants initialization/device-selection sources:
https://github.com/Genesis-Embodied-AI/quadrants (qd.init(arch=qd.vulkan));
`quadrants/rhi/vulkan/vulkan_loader.cpp` (`QD_VISIBLE_DEVICE`);
Taichi v1.7.4 `taichi/rhi/vulkan/vulkan_loader.cpp` (`TI_VISIBLE_DEVICE`).
Indices come from the Vulkan loader enumeration, not wgpu adapter ordering.

## Observed limitations (this machine)

All installed-package checks and sphere rendering passed. CPU N=160 float64
MGPCG passed in 3.10892 s; OpenCL float64 passed in 4.20228 s (both 7 iterations,
relative residual 1.09705e-9). Taichi/Quadrants Vulkan expose no float64 in these
runtimes despite hardware FP64 support. Their float32 N=160 solves failed:
`ArithmeticError: CG failed: residual=2.59202e+27, error=4.82507e+25`.
This is a failure of this float32 solver/backend combination, not proof that
the frameworks cannot implement a robust solver. No solve speedup is claimed
for them. Initial compile and solve failures are retained as historical JSON.
No RX 7700S measurement was possible: every explicit selector failed with
`ValueError: device selector 'RX 7700S': 0 matches`.
