# Uncoupled 3-D flow

`from platesim.flow3d import solve_stokes3d, buoyancy_force`

Section 10e of `CONTRACTS_3D.md`; no tile-engine coupling is installed here.
Coordinates, distance and velocity use cm, cm and cm/s. Arrays use `[x,y,z]`.
Pressure in the additional `p` output is kinematic pressure, cm²/s².
`info.divergence_max` is the maximum absolute MAC divergence in 1/s;
`info.div_max_rel` is that value times dx divided by the largest face speed.
Face shapes include both domain faces, with exactly duplicated periodic seams.

## Discretisation and solver

Only faces shared by two open cells are unknowns. Transverse solid-wall ghosts
use the mean solid occupancy of the two neighbouring cells, matching the 2-D
reference at step corners. Bottom/lid ghosts enforce no-slip; lateral
nonperiodic boundaries have zero normal flux and free-slip tangential velocity.
Forces are averaged from cells to faces. The scales are dx, U and nu U/dx, where
U is the greater of lid speed and max |f| dx²/nu. One pressure unknown and one
redundant continuity equation are removed per six-connected, periodic-aware
fluid region. This symmetric gauge elimination also handles sealed pockets and
isolated cells without modifying the mask.

`solver="auto"` selects direct sparse SuperLU at up to 12,000 unknowns. Larger
systems use GMRES with an upper triangular block preconditioner: a PyAMG
smoothed-aggregation V-cycle for the velocity Laplacian and a pressure mass
approximation for the order-zero Stokes Schur complement. The inverse pressure
mass includes a rank-one gauge correction per region. This avoids 3-D LU fill-in
and the ill-conditioned point-gauge mode of a plain identity pressure block.
GMRES tolerates the approximate triangular inverse; MINRES would require a
symmetric positive-definite preconditioner. No fallback silently switches a
failed iterative solve to a potentially memory-exhausting direct solve.
`solver="iterative"` requires the existing optional `pyamg` dependency. Failure
to converge raises an exception. Reported iterations count inner Krylov steps;
seconds include assembly, preconditioner setup and solution.

## Contract interpretation / gaps

The interface does not supply or define `z_field`. Here it is the highest
upward zero crossing of the signed distance, linearly interpolated between
centres, or z0 when no such crossing exists. This reproduces planar field and
uniform-deposit reference cases; a nonplanar wafer needs an agreed field-height
policy (or an explicit parameter). A binary mask cannot encode a fractional
deposit height; supply the actual signed distance for that information.

The lid may be inside the grid. Centres at/above it return zero; the last
tangential momentum row uses its actual distance to the lid. A lid less than
1e-5 dx above the last retained centre is rejected to prevent ill-scaled rows;
move it to a grid face. Normal flux closes at the last retained cell face,
so an off-face lid is not a geometrically exact cut-cell boundary. Use a
grid-face-aligned lid for general nonuniform flow. A sharp mask is still used
for solid geometry: subcell no-slip geometry is not a cut-cell method. Only x/y
may be periodic. Neither Oseen inertia nor Brinkman smoothing is included.

## Verification

From the repository root, without modifying the project environment:

```powershell
$env:PYTHONPATH='.'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PLATESIM_RUN_SLOW='1' # omit to skip the 48x48x40 benchmark
uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-fmm --with scikit-image --with numba --with pyamg --with contourpy --with pytest python -m pytest tests/test_flow3d.py -q -s -p no:cacheprovider --basetemp="$env:TEMP\banyancode\flow3d-tests"
```

The slow benchmark is opt-in as well as marked `slow`. On the development CPU,
the reviewed 26-case run took 6.22 s total. The 48x48x40 square via had 199,871
unknowns, 38 iterations and 3.78 s solver time, with relative face divergence
1.36e-11. Timings depend on the machine and BLAS threading.

Maximum differences normalised by the reference maximum x speed:

| Extruded reference | max absolute ux difference / U | max absolute uz difference / U |
| --- | ---: | ---: |
| Cross shear, deposit 0, ny=1 | 1.40e-15 | 6.19e-16 |
| Cross shear, deposit 0, ny=2 | 2.20e-15 | 5.24e-16 |
| Cross shear, deposit 0, ny=4 | 1.70e-15 | 1.20e-15 |
| Cross shear, deposit 4 um, ny=1 | 8.09e-16 | 5.16e-16 |
| Cross shear, deposit 4 um, ny=2 | 8.09e-16 | 6.98e-16 |
| Cross shear, deposit 4 um, ny=4 | 1.39e-15 | 7.06e-16 |
| In-plane gravity, ny=2 | 1.59e-15 | 1.57e-15 |

Tests also compare along-trench flow with an independently assembled scalar
Poisson solve, check hydrostatic cancellation and trapped/seam-crossing pockets,
verify hydrostatic pressure sign/units/gauges and fractional-height Couette lids,
and compare direct and iterative solutions on a genuinely 3-D via. Both direct
and iterative paths validate finite solutions and the actual saddle residual.
