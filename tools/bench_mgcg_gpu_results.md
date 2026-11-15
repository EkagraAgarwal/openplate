# GPU-resident MGCG benchmark

Tolerance: 1e-8 true relative residual; zero initial guess for every solve.
CPU: Numba, 4 threads. Devices: gfx1035, gfx1102, gfx1035, gfx1102.
Setup includes hierarchy, coarse factor/inverse, and OpenCL context/program construction.
Solve timings exclude one warm-up solve; repeated timing includes set_reaction and all transfers.
The reaction cycles through 1.00–1.04 times the initial field; no warm-starting.
Both AMD devices appear on two platforms; these measurements use the first matching device
(gfx1102 is enumeration index 1). Setup times include compilation/cache loading and are
not uniform cold-cache measurements: cache state differs across shapes and precisions.

| Field | Shape | Backend | Setup s | Solve s | Iterations | 20 updates + solves s | Repeat iterations |
|---|---|---|---:|---:|---:|---:|---|
| trench | 48x56x48 | Numba f64 | 0.0382 | 0.0874 | 11 | 2.3939 | 11–11 |
| trench | 48x56x48 | gfx1102 float64 | 1.0146 | 0.1261 | 11 | 2.7526 | 11–11 |
| trench | 48x56x48 | gfx1102 mixed | 0.3273 | 0.1335 | 11 | 2.8830 | 11–11 |
| trench | 48x56x48 | gfx1035 float64 | 1.0088 | 0.1590 | 11 | 3.4281 | 11–11 |
| trench | 48x56x48 | gfx1035 mixed | 0.3333 | 0.1526 | 11 | 3.2936 | 11–11 |
| trench | 128x128x96 | Numba f64 | 0.4131 | 1.7702 | 12 | 37.7224 | 12–12 |
| trench | 128x128x96 | gfx1102 float64 | 9.8721 | 0.2657 | 12 | 8.3403 | 12–12 |
| trench | 128x128x96 | gfx1102 mixed | 13.0253 | 0.2783 | 13 | 8.0674 | 13–13 |
| trench | 192x192x128 | Numba f64 | 1.4672 | 5.3849 | 13 | 112.2362 | 13–13 |
| trench | 192x192x128 | gfx1102 float64 | 14.2684 | 0.5145 | 13 | 19.6593 | 13–13 |
| trench | 192x192x128 | gfx1102 mixed | 14.5016 | 0.3881 | 13 | 17.8146 | 13–13 |
| smooth | 256x256x256 | Numba f64 | 6.4459 | 16.4951 | 10 | 386.2946 | 10–10 |
| smooth | 256x256x256 | gfx1102 float64 | 28.6135 | 1.1299 | 10 | 60.5125 | 10–10 |
| smooth | 256x256x256 | gfx1102 mixed | 25.9706 | 0.7493 | 10 | 51.8434 | 10–10 |

## Contention check

After the full run, the small case was repeated with the GPU confirmed free. Results below
are a separate check, not replacements or cherry-picked minima for the table above.
GPU timings are consistent with the full run; the short single CPU solve is noisier.

| Backend | Solve s | 20 updates + solves s | Iterations |
|---|---:|---:|---:|
| Numba f64 | 0.1146 | 2.5201 | 11 |
| gfx1102 float64 | 0.1272 | 2.8792 | 11 |
| gfx1102 mixed | 0.1259 | 2.8591 | 11 |
| gfx1035 float64 | 0.1613 | 3.5571 | 11 |
| gfx1035 mixed | 0.1499 | 3.3930 | 11 |

## Recommendation

- **48x56x48: keep Numba CPU.** Neither GPU pays off; launches and scalar synchronization
  dominate, and mixed precision provides no meaningful speedup. The integrated GPU is slower.
- **128x128x96: GPU is worthwhile for a reused hierarchy.** The 20-update pattern is
  4.5x faster in f64 and 4.7x in mixed, excluding initial setup. F64 and mixed are effectively
  tied here; mixed takes 13 rather than 12 iterations. Including measured setup reduces the
  end-to-end gain to roughly 2.1x (f64) / 1.8x (mixed), so avoid rebuilding each solve.
- **192x192x128 and 256^3: prefer gfx1102 mixed.** Single solves are 13.9x and 22.0x faster
  than Numba. The realistic 20-update pattern is 6.3x and 7.5x faster (3.5x and 5.0x including
  measured setup). Mixed beats device f64 by 1.33x / 1.51x for solves and 1.10x / 1.17x for
  the repeated pattern, with identical iteration counts at these two sizes.
- Host diagonal coarsening, coarse-inverse preparation and coefficient uploads in
  `set_reaction` remain substantial overhead. Repeated-pattern speedup is deliberately not
  inferred from either sweep throughput or solve-only timing. These results compare this
  implementation against 4-thread Numba, not an optimized CPU bandwidth roofline.

## Implementation and validation

Mixed means f32 hierarchy/smoother/transfers/coarse inverse, with the original fine operator,
all outer CG vectors/updates and reductions in f64. The V-cycle uses fixed sweep counts, not
an adaptive inner tolerance or flexible CG. Work-items are C-order linear indices, with
consecutive lanes along z, the fastest axis. Coarse inversion is prepared on the host only
during setup/diagonal updates; its application is on-device. No fine-grid sparse matrix is built.

The solve-residency test instruments every PyOpenCL array download: all downloads are scalars
except the final returned solution. Tests cover f64 operator parity to 1e-12, CPU V-cycle
parity/linearity, trench solutions to 1e-10 true relative residual (stronger than the requested
1e-8), iteration parity, mixed iteration overhead, 43x55x23 singleton coarsening, odd periodic
Jacobi fallback, reaction reuse, zero RHS, warm starts, and both gfx1102 and gfx1035.

Test command (replace `<external-temp>` with a directory outside the repository):

Final result: **14 passed in 90.30 s**, exercising both physical GPUs. The isolated process
includes repeated context/program setup; it exceeds the general 60 s test-file budget.

```text
PYTHONPATH=. uv run --no-project --python 3.12 --with numpy --with scipy --with numba --with pyamg --with pyopencl --with pytest python -m pytest tests/test_tile3d_mgcg_opencl.py -q -p no:cacheprovider --basetemp=<external-temp>
```

Benchmark command:

```text
PYTHONPATH=. uv run --no-project --python 3.12 --with numpy --with scipy --with numba --with pyopencl python tools/bench_mgcg_gpu.py
```

No Mako dependency is needed: the f64 reduction is a hand-written OpenCL 1.2 kernel.
Python syntax and targeted Ruff checks pass. `bun typecheck` is unavailable (no such script
in this Python project). The repository graph indexes the sibling main worktree rather
than these new files, so numerical tests and direct file inspection are the validation source.
