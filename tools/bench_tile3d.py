"""Public, data-free 3-D trench solver benchmark (wall time = min of 3).

Run with PYTHONPATH=. and the fast/ref extras. Geometry construction and a
zero-duration JIT warmup are excluded. Set NUMBA_NUM_THREADS explicitly when
sharing a machine; the default here is four. JSON contains every timing, not
only the minimum. --profile adds a separate, untimed cProfile run per case.

Example (PowerShell, from the repository root)::

    $env:PYTHONPATH='.'; $env:NUMBA_NUM_THREADS='4'
    uv run --project . --extra fast --extra ref --with pytest --with httpx python tools/bench_tile3d.py --output before.json --profile profiles_before

Repeat after editing with --output after.json --reference before.json.
The defaults are a narrow sheared trench (0.5 h) and a wide stagnant trench
(1 h), with synthetic geometry and parameters only. --micro times individual
hotspots; --check-kernels checks signed RHS/reinit equivalence and LU reuse.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from platesim.feature.engine.params import Params
from platesim.tile3d import levelset3d as ls
from platesim.tile3d.grid import Grid3D
from platesim.tile3d.sim import run_tile, top_height


def setup(width_um, depth_um, dx_um=8.0):
    """Rectangular trench, 20% open area, four periodic axial cells."""
    nx = int(np.ceil(5 * width_um / dx_um))
    bottom = -np.ceil((depth_um + 32) / dx_um) * dx_um
    top = 256.0
    grid = Grid3D((nx, 4, int(round((top-bottom)/dx_um))), dx_um*1e-4,
                  (-nx*dx_um*0.5e-4, 0., bottom*1e-4))
    x, _, z = grid.meshgrid()
    psi = np.maximum(z, np.minimum(width_um*0.5e-4-np.abs(x), z+depth_um*1e-4))
    psi = ls.reinit(psi, grid.dx, periodic=grid.periodic)
    xc = grid.centers()[0]*1e4
    field = np.broadcast_to((np.abs(xc) > width_um/2+32)[:, None], grid.shape[:2])
    floor = np.broadcast_to((np.abs(xc) < width_um/4)[:, None], grid.shape[:2])
    return psi, grid, field, floor


def benchmark(case, repeats=3, profile=None, duration=None):
    wide, shear = case.startswith("wide"), case.endswith("shear")
    width, depth, seconds = (940., 97., 3600.) if wide else (230., 90., 1800.)
    psi, grid, field, floor = setup(width, depth)
    seconds = seconds if duration is None else duration
    kw = dict(delta_um=64., params=Params(suppressor=False, k_mix=0., flow_shear_s=0.),
              n_frames=2, dt_max=120., cfl=.3, reinit_every=4, scheme="weno5",
              backend="auto", linear="auto", field_mask=field,
              flow=dict(shear_rate_s=59., lid_um=256.) if shear else None)
    def run(t):
        return run_tile(psi, grid, 15., t, **kw)
    run(0.)
    h0 = top_height(psi, grid)
    samples, metrics = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        out = run(seconds)
        samples.append(time.perf_counter()-start)
        growth = (top_height(out["psi"], grid)-h0)*1e4
        thickness = float(growth[field].mean())
        metrics.append(dict(field_um=thickness, floor_over_field=float(growth[floor].mean()/thickness),
                            steps=out["summary"]["steps"],
                            max_outer=max(out["series"]["outer_iterations"]),
                            faraday_error_rel=out["summary"]["faraday_error_rel"]))
    hotspots = {}
    if profile:
        profiler = cProfile.Profile()
        profiler.runcall(run, seconds)
        profiler.dump_stats(str(profile))
        for (filename, _, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
            if name in ("_smooth", "_restrict", "_prolong", "_apply", "rhs", "weno_left", "gmres") or "gstrf" in name or "cFastMarcher" in name:
                key = Path(filename).name+":"+name
                hotspots[key] = dict(calls=calls, own_s=own, cumulative_s=cumulative)
    return dict(case=case, shape=grid.shape, width_um=width, depth_um=depth, duration_s=seconds,
                wall_s=min(samples), timings_s=samples, metrics=metrics,
                numba_threads=os.environ["NUMBA_NUM_THREADS"], hotspots=hotspots)


def check_kernels():
    """Exercise the signed WENO RHS, including seams and degenerate axes."""
    from platesim.tile3d.backend import ls_numpy, ls_numba
    rng = np.random.default_rng(2718)
    error = 0.0
    for shape in ((19, 4, 17), (1, 4, 9), (7, 1, 1)):
        psi = rng.normal(size=shape)*8e-4
        speed = rng.uniform(-1, 1, shape)
        for periodic in ((True, True, False), (False, False, False), (True, True, True)):
            for high_order in (False, True):
                a = ls_numpy.rhs(psi, speed, 8e-4, periodic, high_order)
                b = ls_numba.rhs(psi, speed, 8e-4, periodic, high_order)
                error = max(error, float(np.max(np.abs(a-b))))
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    print(f"RHS numpy/numba max absolute error: {error:.3g}", flush=True)
    psi, grid, _, _ = setup(230., 90.)
    a = ls.reinit(psi, grid.dx, grid.periodic, backend="numpy")
    b = ls.reinit(psi, grid.dx, grid.periodic, backend="numba")
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    from unittest.mock import patch
    with patch.object(ls.importlib, "import_module", side_effect=ImportError("optional numba absent")):
        fallback = ls.reinit(psi, grid.dx, grid.periodic)
    np.testing.assert_array_equal(a, fallback)
    print(f"Reinit max absolute error: {np.max(np.abs(a-b)):.3g} cm", flush=True)
    from platesim.tile3d.fields import _AdvLinear, _faces
    from platesim.tile3d import operators
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    shape = (14, 4, 11)
    faces = _faces(np.ones(shape), grid.periodic, top_scale=np.ones(shape[:2]))
    velocity = tuple(np.full_like(f, 20. if axis == 0 else 0.)
                     for axis, f in enumerate(faces))
    base, rhs = operators.assemble_advdiff(*faces, 1., velocity,
                                         dirichlet={"z_hi": 1.}, periodic=grid.periodic)
    cache, x = {}, None
    q_base = rng.uniform(.1, 1, rhs.shape)
    for scale in (1., 1.001, 1.1, 2., 1000.):
        q = scale*q_base
        A = base+sparse.diags(q)
        solver = _AdvLinear(A, q, True, ilu_cache=cache, tol=1e-10)
        x = solver.solve(rhs, x0=x)
        np.testing.assert_allclose(x, spsolve(A, rhs), rtol=1e-8, atol=1e-10)
        assert np.linalg.norm(A@x-rhs)/np.linalg.norm(rhs) <= 1e-10
        if scale == 1.001:
            assert cache["lu_builds"] == 1, "nearby system refactored instead of reusing LU"
    print(f"Cached-LU true-residual checks passed; {cache['lu_builds']} builds/5 systems", flush=True)
    from platesim.tile3d.backend.mgcg_numba import MGCG
    shape = (588, 4, 8)
    faces = operators.face_coefficients(np.ones(shape), grid.periodic)
    kw = dict(reaction=.1, dirichlet={"z_hi": 0.}, periodic=grid.periodic)
    mg = MGCG(*faces, 1., **kw)
    assert mg.hierarchy[-1].x.shape == (147, 2, 2)
    assert mg.hierarchy[-1].lu is not None
    a, b = rng.normal(size=shape), rng.normal(size=shape)
    ma, mb = mg.vcycle(a), mg.vcycle(b)
    assert np.vdot(a, ma) > 0
    np.testing.assert_allclose(np.vdot(a, mb), np.vdot(ma, b), rtol=1e-11, atol=1e-12)
    matrix = mg.hierarchy[-1].matrix
    mg.set_reaction(.2)
    assert mg.hierarchy[-1].matrix is matrix
    kw["reaction"] = .2
    fresh = MGCG(*faces, 1., **kw)
    x, info = mg.solve(a)
    y, other = fresh.solve(a)
    assert info["converged"] and other["converged"]
    np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)
    A, _ = operators.assemble(*faces, 1., **kw)
    assert np.linalg.norm(A@x.ravel()-a.ravel())/np.linalg.norm(a) < 1e-8
    print(f"Sparse coarse symmetry/reaction reuse/true residual checks passed: {info}", flush=True)


def microbench(baseline_mg=None):
    """Public hotspot experiments; compilation/setup excluded, min of three."""
    from platesim.tile3d.backend import ls_numpy, ls_numba
    from platesim.tile3d.backend.mgcg_numba import MGCG
    from platesim.tile3d.fields import _faces
    from scipy import sparse
    from scipy.sparse.linalg import splu, spilu

    def timed(fn):
        fn()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = fn()
            times.append(time.perf_counter()-start)
        return min(times), result

    psi, grid, _, _ = setup(940., 97.)
    speed = np.sin(psi/grid.dx)
    for module in (ls_numpy, ls_numba):
        seconds, _ = timed(lambda: module.rhs(psi, speed, grid.dx, grid.periodic, True))
        print(f"RHS {module.__name__}: {seconds:.6f} s", flush=True)
    eps = 1.2*grid.dx
    H = np.maximum(.5*(1+np.tanh(psi/eps)), 1e-6)
    active = psi > -3.5*eps
    faces = _faces(.065*H, grid.periodic, top_scale=.065*H[:, :, -1])
    reaction = np.where(active, .3/eps/np.cosh(np.clip(psi/eps, -30, 30))**2, 0)
    rng = np.random.default_rng(19)
    b = np.where(active, 1+rng.random(psi.shape), 0)
    if baseline_mg:
        import importlib.util
        import sys
        name = "platesim.tile3d.backend._bench_baseline"
        spec = importlib.util.spec_from_file_location(name, baseline_mg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        solver = module.MGCG(*faces, grid.dx, reaction=reaction, active=active,
                            dirichlet={"z_hi": 0.})
        seconds, (_, info) = timed(lambda: solver.solve(b))
        print(f"MG baseline sweeps=2: {seconds:.6f} s {info}", flush=True)
    for sweeps in (1, 2, 3):
        solver = MGCG(*faces, grid.dx, reaction=reaction, active=active,
                      dirichlet={"z_hi": 0.}, pre=sweeps, post=sweeps)
        seconds, (_, info) = timed(lambda: solver.solve(b))
        print(f"MG sweeps={sweeps}: {seconds:.6f} s {info}", flush=True)
    from numba import njit
    from platesim.tile3d.backend import mgcg_numba as mg_module
    parallel = mg_module._kernel
    serial = njit(parallel.py_func)
    try:
        for cutoff in (4096, 16384):
            def hybrid(*args):
                return (serial if args[0].size <= cutoff else parallel)(*args)
            mg_module._kernel = hybrid
            solver = MGCG(*faces, grid.dx, reaction=reaction, active=active,
                          dirichlet={"z_hi": 0.})
            seconds, (_, info) = timed(lambda: solver.solve(b))
            print(f"MG serial cutoff={cutoff}: {seconds:.6f} s {info}", flush=True)
        for fast_math in (False, True):
            mg_module._kernel = njit(parallel=True, fastmath=fast_math)(parallel.py_func)
            solver = MGCG(*faces, grid.dx, reaction=reaction, active=active,
                          dirichlet={"z_hi": 0.})
            seconds, (_, info) = timed(lambda: solver.solve(b))
            print(f"MG fastmath={fast_math}: {seconds:.6f} s {info}", flush=True)
    finally:
        mg_module._kernel = parallel
    # Same seven-point sparsity, with strong directional transport and changing sink.
    from platesim.tile3d import operators
    psi, grid, _, _ = setup(230., 90.)
    faces = _faces(np.ones(grid.shape)*5e-6, grid.periodic,
                   top_scale=np.ones(grid.shape[:2])*5e-6)
    velocity = [np.zeros_like(f) for f in faces]
    velocity[0][:] = .5
    A, _ = operators.assemble_advdiff(*faces, grid.dx, velocity,
                                     dirichlet={"z_hi": 1.}, periodic=grid.periodic)
    A = (A+sparse.eye(A.shape[0])*.1).tocsc()
    for ordering in ("COLAMD", "MMD_AT_PLUS_A", "MMD_ATA", "NATURAL"):
        seconds, lu = timed(lambda: splu(A, permc_spec=ordering))
        print(f"LU {ordering}: {seconds:.6f} s nnz={lu.L.nnz+lu.U.nnz}", flush=True)
    for fill in (5, 10, 15):
        seconds, lu = timed(lambda: spilu(A, drop_tol=1e-5, fill_factor=fill))
        print(f"ILU fill={fill}: {seconds:.6f} s nnz={lu.L.nnz+lu.U.nnz}", flush=True)


def field_microbench():
    """Tune smoothing and previous-Newton guesses on a complete field solve."""
    from platesim.tile3d import fields
    from platesim.tile3d.backend import mgcg_numba as mg_module
    from platesim.tile3d.backend.mgcg_numba import MGCG
    from numba import njit
    psi, grid, _, _ = setup(940., 97.)
    original_init = MGCG.__init__
    original_system = fields._MGFamily.system
    original_solve = fields._MGFamily.solve
    original_kernel = mg_module._kernel
    strict_kernel = njit(parallel=True)(original_kernel.py_func)
    reference = None
    try:
        configs = ((2, False, False), (3, False, False), (3, True, False), (3, False, True))
        samples = {c: [] for c in configs}
        results = {}
        def configure(config):
            sweeps, warm, fast_math = config
            mg_module._kernel = original_kernel if fast_math else strict_kernel
            def init(self, *args, **kw):
                kw.update(pre=sweeps, post=sweeps)
                original_init(self, *args, **kw)
            def system(self, q):
                self._bench_slot = 0
                return original_system(self, q)
            def solve(self, b, x0=None):
                slot = self._bench_slot
                self._bench_slot += 1
                if not hasattr(self, "_bench_guesses"):
                    self._bench_guesses = {}
                if x0 is None:
                    x0 = self._bench_guesses.get(slot)
                x = original_solve(self, b, x0)
                self._bench_guesses[slot] = x.copy()
                return x
            MGCG.__init__ = init
            fields._MGFamily.system = system if warm else original_system
            fields._MGFamily.solve = solve if warm else original_solve
        def run():
            return fields.FieldSolver().solve(psi, grid.dx, grid.origin[2], 64e-4, .015,
                                               periodic=grid.periodic)
        # Compile/warm every variant, then interleave samples to reduce load drift.
        for config in configs:
            configure(config)
            run()
        for round_index in range(3):
            order = configs[round_index:]+configs[:round_index]
            for config in order:
                configure(config)
                start = time.perf_counter()
                out = run()
                samples[config].append(time.perf_counter()-start)
                results[config] = out
                if reference is None:
                    reference = out
                np.testing.assert_allclose(out["ct"], reference["ct"], rtol=0, atol=1e-6)
                assert abs(out["V"]-reference["V"]) < 1e-6
        for config in configs:
            sweeps, warm, fast_math = config
            print(f"Field sweeps={sweeps} warm={warm} fastmath={fast_math}: {min(samples[config]):.6f} s "
                  f"samples={samples[config]} info={results[config]['info']}", flush=True)
    finally:
        MGCG.__init__ = original_init
        fields._MGFamily.system = original_system
        fields._MGFamily.solve = original_solve
        mg_module._kernel = original_kernel


def flow_microbench():
    """Compare a full extruded Stokes system with its exact y-invariant mode."""
    from platesim.flow3d import solve_stokes3d
    from platesim.flow3d import stokes
    from platesim.flow3d.stokes import _field_height
    from contextlib import nullcontext
    from unittest.mock import patch
    psi, grid, _, _ = setup(230., 90.)
    kw = dict(periodic=grid.periodic, shear_rate_s=59.)
    samples, outputs = {}, {}
    y = max(range(grid.shape[1]), key=lambda j: _field_height(psi[:, j:j+1, :], grid.dx, grid.origin[2]))
    for key, geometry in (("full", psi), ("reduced", psi[:, y:y+1, :])):
        def run():
            return solve_stokes3d(geometry, grid.dx, grid.origin[2], 256e-4, **kw)
        context = patch.object(stokes, "_extruded_stokes", return_value=None) if key == "full" else nullcontext()
        with context:
            run()
            samples[key] = []
            for _ in range(3):
                start = time.perf_counter()
                outputs[key] = run()
                samples[key].append(time.perf_counter()-start)
        print(f"Stokes {key}: {min(samples[key]):.6f} s samples={samples[key]} "
              f"info={outputs[key]['info']}", flush=True)
    full, reduced = outputs["full"], outputs["reduced"]
    error = 0.
    for axis in range(3):
        restored = np.repeat(reduced["faces"][axis][:, :1, :], grid.shape[1]+(axis == 1), axis=1)
        error = max(error, float(np.max(np.abs(restored-full["faces"][axis]))))
        np.testing.assert_allclose(restored, full["faces"][axis], rtol=1e-8, atol=1e-8)
    print(f"Full/reduced maximum face velocity error: {error:.3g} cm/s", flush=True)
    # psi need not be y-invariant: only the exact sharp mask must be. This
    # also checks that the reduction preserves the global maximum crossing.
    z = grid.centers()[2][None, None, :]
    varied = psi*(1+.1*np.sin(z/grid.dx+np.arange(grid.shape[1])[None, :, None]))
    with patch.object(stokes, "_extruded_stokes", return_value=None):
        full = solve_stokes3d(varied, grid.dx, grid.origin[2], 256e-4, **kw)
    reduced = solve_stokes3d(varied, grid.dx, grid.origin[2], 256e-4, **kw)
    assert full["info"]["z_field_cm"] == reduced["info"]["z_field_cm"]
    for a, b in zip(full["faces"], reduced["faces"]):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-8)
    print("Nonuniform signed-distance/global field-height reduction check passed", flush=True)


def ilu_microbench():
    """Compare amortized ILU fill costs at two Peclet numbers, against exact LU."""
    from platesim.core.constants import F
    from platesim.tile3d.fields import FieldSolver
    from platesim.flow3d import solve_stokes3d
    from scipy.sparse import linalg as spla
    psi, grid, _, _ = setup(230., 90.)
    original = spla.spilu
    model = FieldSolver()
    current = .5*2*F*model.ph.D_c*model.p.c_bulk/64e-4
    try:
        for shear in (59., 3000.):
            flow = solve_stokes3d(psi, grid.dx, grid.origin[2], 256e-4,
                                  periodic=grid.periodic, shear_rate_s=shear)
            def run(linear="mg"):
                return FieldSolver(linear=linear).solve(psi, grid.dx, grid.origin[2], 64e-4, current,
                                                        periodic=grid.periodic, flow_faces=flow["faces"])
            reference = run("direct")
            for fill in (15, 10, 5):
                builds = [0]
                def factor(*args, **kw):
                    builds[0] += 1
                    kw["fill_factor"] = fill
                    return original(*args, **kw)
                spla.spilu = factor
                run()
                samples = []
                for _ in range(3):
                    builds[0] = 0
                    start = time.perf_counter()
                    out = run()
                    samples.append(time.perf_counter()-start)
                    assert out["info"]["converged"]
                # At extreme Pe, the existing 1e-8 linear tolerance does not
                # imply 1e-6 forward accuracy. Report it rather than silently
                # relaxing production tolerances or hiding a rejected trial.
                error = float(np.max(np.abs(out["ct"]-reference["ct"])))
                v_error = abs(out["V"]-reference["V"])
                print(f"ILU shear={shear:g} fill={fill}: {min(samples):.6f} s "
                      f"samples={samples} builds={builds[0]} ct_error_vs_lu={error:.3g} "
                      f"V_error_vs_lu={v_error:.3g} matches_lu_1e-6={error < 1e-6 and v_error < 1e-6} "
                      f"info={out['info']}", flush=True)
    finally:
        spla.spilu = original


def direct_microbench():
    """Interleaved exact-vs-cached SPD LU families, with/without advection."""
    from platesim.tile3d import fields
    from platesim.flow3d import solve_stokes3d
    psi, grid, _, _ = setup(230., 90.)
    flow = solve_stokes3d(psi, grid.dx, grid.origin[2], 256e-4,
                          periodic=grid.periodic, shear_rate_s=59.)["faces"]
    original_init, original_factor = fields._Linear.__init__, fields.splu
    builds = [0]
    def factor(*args, **kw):
        builds[0] += 1
        return original_factor(*args, **kw)
    try:
        fields.splu = factor
        for faces in (None, flow):
            samples = {False: [], True: []}
            counts, results = {}, {}
            def configure(enabled):
                def init(self, A, tol=1e-8, precond=None, mode="auto"):
                    original_init(self, A, tol, precond if enabled else None, mode)
                fields._Linear.__init__ = init
            def run():
                return fields.FieldSolver().solve(psi, grid.dx, grid.origin[2], 64e-4, .015,
                                                   periodic=grid.periodic, flow_faces=faces)
            for enabled in (False, True):
                configure(enabled)
                results[enabled] = run()
            reference = results[False]
            for round_index in range(3):
                for enabled in ((False, True) if round_index % 2 == 0 else (True, False)):
                    configure(enabled)
                    builds[0] = 0
                    start = time.perf_counter()
                    out = run()
                    samples[enabled].append(time.perf_counter()-start)
                    counts[enabled], results[enabled] = builds[0], out
                    assert out["info"]["converged"]
                    np.testing.assert_allclose(out["ct"], reference["ct"], rtol=0, atol=1e-6)
                    assert abs(out["V"]-reference["V"]) < 1e-6
            for enabled in (False, True):
                print(f"SPD LU reuse={enabled} flow={faces is not None}: {min(samples[enabled]):.6f} s "
                      f"samples={samples[enabled]} factors={counts[enabled]} "
                      f"info={results[enabled]['info']}", flush=True)
    finally:
        fields._Linear.__init__, fields.splu = original_init, original_factor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=("narrow-none", "narrow-shear", "wide-none", "wide-shear"),
                        default=["narrow-shear", "wide-none"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, help="override physical seconds (smoke tests only)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, help="baseline JSON; enforce physics tolerances")
    parser.add_argument("--profile", type=Path, help="directory for separate untimed profiles")
    parser.add_argument("--check-kernels", action="store_true", help="check WENO RHS to 1e-12 and exit")
    parser.add_argument("--micro", action="store_true", help="time synthetic hotspots and exit")
    parser.add_argument("--baseline-mg", type=Path, help="original mgcg_numba.py to compare in --micro")
    parser.add_argument("--field-micro", action="store_true", help="tune complete synthetic field solves and exit")
    parser.add_argument("--flow-micro", action="store_true", help="time exact extruded Stokes reduction and exit")
    parser.add_argument("--ilu-micro", action="store_true", help="compare ILU fill on complete field solves and exit")
    parser.add_argument("--direct-micro", action="store_true", help="time SPD LU reuse on complete field solves and exit")
    args = parser.parse_args()
    if args.check_kernels:
        check_kernels()
        return
    if args.micro:
        microbench(args.baseline_mg)
        return
    if args.field_micro:
        field_microbench()
        return
    if args.flow_micro:
        flow_microbench()
        return
    if args.ilu_micro:
        ilu_microbench()
        return
    if args.direct_micro:
        direct_microbench()
        return
    if args.repeats < 3:
        parser.error("at least three repeats are required")
    if args.profile:
        args.profile.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = {r["case"]: r for r in json.loads(args.reference.read_text())} if args.reference else {}
    for case in args.cases:
        row = benchmark(case, args.repeats, args.profile / (case+".prof") if args.profile else None,
                        args.duration)
        rows.append(row)
        if args.reference:
            before = reference[case]
            assert row["duration_s"] == before["duration_s"]
            assert list(row["shape"]) == list(before["shape"])
            ref = before["metrics"][0]
            for metric in row["metrics"]:
                assert abs(metric["floor_over_field"]-ref["floor_over_field"]) < .002
                assert abs(metric["field_um"]-ref["field_um"]) < .3
        m = row["metrics"][0]
        print(f"{case:14s} {str(row['shape']):15s} floor/field={m['floor_over_field']:.9f} "
              f"field={m['field_um']:.9f} um wall={row['wall_s']:.3f} s "
              f"samples={row['timings_s']}", flush=True)
        if args.reference:
            print(f"  speedup={before['wall_s']/row['wall_s']:.3f}x "
                  f"delta floor/field={m['floor_over_field']-ref['floor_over_field']:+.3g} "
                  f"delta field={m['field_um']-ref['field_um']:+.3g} um", flush=True)
        if args.output:
            args.output.write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
