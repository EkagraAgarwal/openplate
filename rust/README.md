# Lane E: Rust hot-loop spike (measurement only)

No engine integration, dispatch changes, or production dependency changes.
The extension implements float64, C-contiguous versions of the current
`mgcg_numba.MGCG._smooth` and `ls_numpy.rhs(..., high_order=True)` contracts.
It deliberately does not implement the solver, SuperLU, RK3, narrow-band
selection, float32 storage, arbitrary strides, or broadcasting at the FFI boundary.
Python callers must normalize fields as the existing numba backend does.

## Reproduce on Windows / build friction

From the repository root, with Rust/MSVC and Python available:

```powershell
$env:PYTHONPATH='.'
$env:RAYON_NUM_THREADS='4'
uv run --project . --extra fast --extra ref --with maturin --with pytest maturin build --release --manifest-path rust/platesim_rs/Cargo.toml --out rust/wheels
uv pip install --python .venv/Scripts/python.exe --reinstall --no-deps (Get-ChildItem rust/wheels/*.whl | Select-Object -First 1 -ExpandProperty FullName)
uv run --project . --extra fast --extra ref --with maturin --with pytest python tools/bench_rust.py --threads 4 --repeats 7 --inner 3
uv run --project . --extra fast --extra ref --with maturin --with pytest python -m pytest rust/tests -q
cargo fmt --manifest-path rust/platesim_rs/Cargo.toml -- --check
$env:PYO3_PYTHON=(Resolve-Path .venv/Scripts/python.exe).Path
cargo clippy --manifest-path rust/platesim_rs/Cargo.toml --release -- -D warnings
```

Tested starting with no worktree `.venv`, no crate build artifacts and no
installed extension. `uv run --with maturin ... maturin develop --release
--manifest-path rust/platesim_rs/Cargo.toml` **returned success but was not
persistent**: maturin selected the ephemeral uv overlay's Python, and the next
identical `uv run ... python` failed with `ModuleNotFoundError: platesim_rs`.
Thus this exact develop command is not a reliable clean-clone workflow here.
The wheel build plus explicit install into `.venv` above worked across subsequent
uv overlay invocations. Do not use `uv sync --exact` afterward without reinstalling
this intentionally undeclared spike dependency. For develop, install maturin into
the actual project venv and call `.venv/Scripts/maturin.exe develop ...` with that
venv activated instead of putting maturin in an ephemeral `--with` overlay.
That alternate develop route is advice, not a tested clean-venv result.

Initial develop build downloaded/compiled 34 Rust packages and took 54.6 s of
Cargo time, plus about 12 s of Python environment setup. The subsequent wheel
build took 40.2 s (PyO3 rebuilt against another overlay interpreter). UV warned
that cross-volume hardlinks were unavailable and copied packages instead; this
was harmless. Rust 1.96 stable MSVC linked without additional installation on
this host. A clean CI runner would still need the MSVC build tools and Windows
SDK, a Rust toolchain, Python, network/cache access and maturin. Cargo.lock is
included for reproducible crate versions; production CI should build `--locked`.
Direct Cargo verification initially failed with `no Python 3.x interpreter
found`: Python exists through uv, not as a global `python` here. Setting
`PYO3_PYTHON` to the project venv interpreter, as above, fixes that discovery.
For repeat builds, pass `-i .venv/Scripts/python.exe` to maturin to avoid selecting
a different transient overlay interpreter.

The produced wheel is `cp312-cp312-win_amd64`, **not** a universal Python wheel.
Distribution needs a tested OS/architecture/Python-version matrix (e.g. maturin
in cibuildwheel or maturin-action), NumPy compatibility tests, wheel installation
tests, and Windows runtime dependency checks; Linux needs a manylinux policy and
macOS deployment targets. No `abi3` configuration or cross-platform wheel was
tested. The default release target is portable, not `target-cpu=native`; no fast
math or explicit FMA is enabled. Users of supplied wheels would not need Rust.

## Measurement method and numerical coverage

`tools/bench_rust.py` contains independent copies of both numba loop bodies,
compiled with and without `parallel=True`. The original engine numba functions
are also measured with one worker (`numba-engine-1`); the existing engine kernels
are **already parallel**, not an unoptimized Python baseline. No engine file is
edited. `numba-serial` uses `prange` as a serial range. In particular, WENO's
seven-element scratch array allocation can be hoisted by numba's parallel
transformation, so the serial copy and engine-with-one-thread need not perform
identically. Compare against the better numba option, not just the serial copy.

The three prescribed shapes use periodic x/y and clamped z, two forward RBGS
sweeps per smoother call, positive spatially varying conductances, reaction,
and 8% masked identity cells. WENO uses signed speeds and a noisy tilted-distance
field, dx = 8e-4 cm (8 um); results include boundary stencils. These are synthetic
representative inputs, not a replay of captured engine arrays or a full solve.

JIT and pool startup are excluded. For every variant the initial smoother array
is reset outside each timed batch. Each batch then makes three consecutive calls
with the same trajectory; the reported time is the **minimum of seven batch
means**. WENO allocation and the PyO3 boundary are timed; no array-copy wrapper
is hidden around Rust. Rust single and rayon use identical arithmetic. Rayon
uses four workers fixed at process start; parallel numba uses four workers except
the explicit one-worker engine measurement. BLAS threads default to one. Runs
are serial, never numba and Rust concurrently. No exclusive machine reservation,
CPU pinning or governor control was possible; retain the raw seven samples in
`results.json` and treat small differences as provisional on this shared host.

All benchmark outputs are compared to the actual engine smoother or numpy WENO
using `rtol=atol=1e-12`; maximum absolute errors are retained per row.
Additional checks cover forward and reverse colour order, zero/one/three sweeps,
forced Jacobi, automatic doubled Jacobi for odd periodic axes, nonperiodic and
fully periodic boundaries, tiny/singleton axes, constant fields, nonsmooth
random fields, mixed speed signs, and invalid shape/layout/spacing/aliasing.

Rust RBGS updates in place and has a barrier after each colour, without a
snapshot or changed ordering. A small documented raw-pointer section permits
disjoint parallel writes: every cell reads itself and opposite-colour neighbours
only. Odd periodic axes disable RBGS. Jacobi writes a distinct workspace then
copies back each pass, matching `_smooth`. Arrays are exclusively borrowed via
rust-numpy and the Python GIL remains held during computation, including rayon
work. Concurrent mutation through other GIL-free native code is unsupported;
the caller must ensure no external writer uses these buffers during a call.
This choice also means the API does not permit concurrent Python work while a
call runs. These constraints and the unsafe colour proof are maintenance costs,
not evidence of a production-ready extension.

## Results and recommendation

Host: AMD Ryzen 7 7735HS, 8 cores / 16 logical processors, Windows x86-64,
Rust 1.96.0 MSVC, CPython 3.12.13, NumPy 2.5.4, numba 0.68.1.
Four workers were intentionally used on this shared host,
not all logical CPUs. Raw timings and exact Python/library versions are in
`results.json`.

Times in ms per call, minimum of seven warmed three-call batches. `NB seq` is
the serial copy; `NB pool1` is the actual engine's parallel kernel with one
worker; `NB4` is the parallel copy with four workers. `R1` and `R4` are Rust
single-worker and four-worker rayon. The last column is `NB4 / R4`, so **greater
than 1 favours Rust**.

| Kernel | Shape | NB seq | NB pool1 | NB4 | R1 | R4 | Rust speedup vs NB4 |
|---|---|---:|---:|---:|---:|---:|---:|
| smoother, 2 sweeps | 590x4x48 | 4.657 | 4.299 | 1.470 | 6.888 | 2.188 | 0.67x |
| smoother, 2 sweeps | 142x4x47 | 1.005 | 1.317 | 0.351 | 1.306 | 0.671 | 0.52x |
| smoother, 2 sweeps | 128x128x64 | 57.222 | 42.641 | 14.425 | 61.558 | 19.407 | 0.74x |
| WENO5 rhs | 590x4x48 | 92.533 | 40.907 | 13.056 | 39.275 | 9.140 | 1.43x |
| WENO5 rhs | 142x4x47 | 20.888 | 8.629 | 3.258 | 8.067 | 2.031 | 1.60x |
| WENO5 rhs | 128x128x64 | 857.609 | 406.007 | 106.257 | 337.974 | 94.062 | 1.13x |

The numpy WENO baselines were **117.804, 17.539, 1996.987 ms**, respectively.
On the profile's 590x4x48 tile, numba4 therefore gives **9.02x** over numpy and
Rust4 **12.89x**; comparing Rust only against the serial numba copy would greatly
overstate the incremental benefit. All 33 timed outputs had **zero maximum
absolute error** against their engine/numpy reference. The worst seven-sample
max/min ratio was **3.37**, showing material shared-host interference despite
minimum filtering. In particular the 13% large-grid WENO advantage is not a
dedicated-host confidence estimate.

**Do not replace the smoother with Rust.** The existing numba parallel kernel
already closes the gap. Its 590x4x48 time is close to the supplied ~1.5 ms
profile observation, while Rust rayon is slower. Rust's single-worker smoother
is also slower than numba, so this is not a case where adding threads conceals
a single-thread language advantage. Even the large 3-D case shows no Rust win
in the final run: Rust rayon takes 19.407 ms versus numba's 14.425 ms.

**Prefer enabling/retaining numba WENO before adding Rust.** Rust rayon wins the
WENO microbenchmark, but most of the benefit over the numpy profile is already
available through the existing optional numba backend. There is no evidence
here that a second compiled language, a wheel pipeline and an unsafe parallel
stencil are justified for this workload. WENO is the only plausible candidate
for a future Rust experiment if larger real workloads, stable dedicated-host
measurements or Python-free deployment requirements change that tradeoff.
Rust avoids numba JIT warmup, which was excluded here. Startup-sensitive or
numba-free deployments could have a different tradeoff; those benefits were not
measured, and long runs with ~58,000 smoother calls amortize startup costs.

## How to interpret speedups

For a kernel share `p` and measured speedup `s`, the ideal whole-run speedup is
`1 / ((1-p) + p/s)`. For the supplied no-flow profile with both kernels replaced,
use `1 / (0.28 + 0.62/s_smooth + 0.10/s_weno)`. These are upper-bound estimates
holding iteration counts, precision, time steps, all other work and workload mix
fixed; this spike never timed the full engine. The 10% WENO share belongs to the
**numpy** profile, not a hypothetical run already using numba WENO.

Using the 590x4x48 measurements and the supplied 62% / 10% profile shares:

| Change from current parallel smoother + numpy WENO | Ideal no-flow end-to-end speedup |
|---|---:|
| retain smoother; use numba4 WENO | 1.098x |
| retain smoother; use Rust4 WENO | 1.102x |
| Rust4 smoother only | 0.768x (regression) |
| Rust4 smoother and Rust4 WENO | 0.826x (regression) |

Thus Rust WENO adds only **1.0037x**, about **0.37%**, whole-run speedup beyond
using the already-available numba WENO, under this profile. That is much smaller
than the measurement uncertainty and does not warrant a new production backend.
If a *different* profile already spent 10% in numba WENO, its 1.43x Rust kernel
gain would instead imply about 1.031x whole-run speedup; that is not the supplied
numpy-based profile.

An infinitely fast smoother alone would cap the no-flow gain at 2.63x; infinitely
fast WENO alone caps it at 1.11x. In the sheared profile the 63% SuperLU
factorization is untouched: even eliminating every other cost could reach only
1.59x. The no-flow shares cannot be reused to predict a sheared-run kernel gain.

Maintenance would include two numerical implementations, synchronization and
boundary invariants, Python/NumPy/Rust ABI compatibility, wheel build/release
infrastructure, dependency/toolchain updates, platform testing and allocation /
GIL policies. The unsafe colour implementation deserves specialist review before
any production use. A worthwhile follow-up before adopting another language is
to optimize numba WENO scratch allocation and stencil index arithmetic, tune
worker counts on dedicated hardware, and benchmark actual solver arrays and
full runs. Neither the Rust implementation nor its wheel should be promoted
from this measurement-only branch based on synthetic microbenchmarks alone.

## Verification and scope

The final release wheel passed all **8 pytest tests**, including an AST check
that the copied numba loop bodies have not drifted from the engine. Cargo
formatting and release-mode Clippy with warnings denied passed. This is a
Python/Rust repository with no `bun typecheck` script; that command reports
`Script not found "typecheck"` and is not a relevant verifier for this spike.
One malformed-input test initially exposed a rust-numpy automatic argument
conversion panic on overlapping arrays. The smoother now uses explicit fallible
borrows and raises a normal Python ValueError instead; the final wheel was
rebuilt, reinstalled and retested before the final timing run.

Only new files under `rust/` and `tools/bench_rust.py` are deliverables. Build
outputs, wheels, numba caches and test caches are ignored under `rust/`. The
root project dependency declaration and every existing engine file are unchanged.
No commit or branch switch was made.
