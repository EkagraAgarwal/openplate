"""Gate 0: cell-centred, homogeneous face Dirichlet, C-order arrays.

Discrete manufactured RHS eliminates discretization error as a confounder.
All solves use rediscretized geometric MG: 2 forward RB pre-sweeps,
2 reverse RB post-sweeps, 12 symmetric coarse sweeps; R = P.T / 8.
Transfers, CG vectors and float64 reductions live on the host for EVERY
backend. GPU stencil timings include transfers (deliberately conservative).
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore', category=SyntaxWarning)

RESULTS = Path(__file__).resolve().parent / 'results'


def clean(text):
    text = str(text)
    for key in ('USERPROFILE', 'HOME', 'USERNAME'):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, '<redacted>')
    return text


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'unavailable'


# Third-party warning formatters otherwise print uv cache paths under a home
# directory. Keep category and exact message, never a user's source path.
warnings.formatwarning = lambda message, category, *args, **kwargs: (
    f'{category.__name__}: {clean(message)}\n')


def parser(solve=True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--device', default='0')
    p.add_argument('--sizes', nargs='+', type=int, default=[160, 256])
    p.add_argument('--solve-n', type=int, default=160)
    p.add_argument('--no-solve', action='store_true', default=not solve)
    p.add_argument('--sweeps', type=int, default=4)
    p.add_argument('--maxiter', type=int, default=200)
    return p


def select(devices, requested):
    if requested.isdecimal():
        return devices[int(requested)]
    hits = [d for d in devices if requested.lower() in str(d).lower()]
    if len(hits) != 1:
        raise ValueError(f'device selector {requested!r}: {len(hits)} matches')
    return hits[0]


def table(rows):
    keys = ['framework', 'version', 'device', 'test', 'dtype', 'N',
            'mcups', 'seconds', 'iterations', 'relative_residual',
            'relative_error', 'compile_seconds', 'peak_bytes', 'status', 'error']
    def fmt(x):
        return f'{x:.6g}' if isinstance(x, float) else str(x or '')
    return ('| ' + ' | '.join(keys) + ' |\n| ' + ' | '.join(['---'] * len(keys))
            + ' |\n' + '\n'.join('| ' + ' | '.join(
                fmt(r.get(k, '')).replace('|', '/').replace('\n', ' ')
                for k in keys) + ' |' for r in rows))


def write_chunks(path, text):
    """Keep generated writes within the task's 120-line chunk budget too."""
    lines = text.splitlines(keepends=True)
    with path.open('w', encoding='utf8') as stream:
        for offset in range(0, len(lines), 120):
            stream.write(''.join(lines[offset:offset+120]))


def save(framework, device, rows, announce=True):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', clean(f'{framework}_{device}'))
    path = RESULTS / (stem + '.json')
    write_chunks(path, clean(json.dumps(rows, indent=2, allow_nan=False)))
    write_chunks(path.with_suffix('.md'), table(rows) + '\n')
    if announce:
        print(table(rows), flush=True)


def failure(framework, device, exc):
    row = dict(framework=framework, version=version(framework), device=clean(device),
               status='failed', error=clean(f'{type(exc).__name__}: {exc}'))
    save(framework, device, [row])


def setup(n, dtype, variable=True):
    x = (np.arange(n, dtype=np.float64) + .5) / n
    # Four half-waves for the solve keep the float32 true-residual floor below
    # 1e-5 at N=160; sweep uses the slower-decaying fundamental mode.
    wave = (4 if variable else 1)*np.pi*x
    exact = (np.sin(wave)[:, None, None] * np.sin(wave)[None, :, None]
             * np.sin(wave)[None, None, :]).astype(dtype)
    k = np.broadcast_to(1 + .2*x[:, None, None] + .1*x[None, :, None]
                        + .1*x[None, None, :], (n, n, n)).astype(dtype).copy()
    if not variable:
        k.fill(1)
    return exact, k


def coefficients(k):
    n = len(k)
    faces = []
    diag = np.zeros_like(k)
    for axis in range(3):
        for direction in (-1, 1):
            neighbor = np.roll(k, -direction, axis=axis)
            w = 2*k*neighbor/(k+neighbor) * n*n
            edge: list = [slice(None)]*3
            edge[axis] = 0 if direction == -1 else -1
            w[tuple(edge)] = 0
            diag += w
            diag[tuple(edge)] += 2*k[tuple(edge)] * n*n
            faces.append((axis, direction, w))
    return diag, faces


def reference_apply(u, k):
    diag, faces = coefficients(k)
    out = diag*u
    for axis, direction, w in faces:
        out -= w*np.roll(u, -direction, axis=axis)
    return out


def prolong(coarse):
    """Cell-centred trilinear interpolation, odd reflection at zero faces."""
    a = coarse
    for axis in range(3):
        left, right = np.roll(a, 1, axis), np.roll(a, -1, axis)
        lo: list = [slice(None)]*3
        hi: list = [slice(None)]*3
        lo[axis], hi[axis] = 0, -1
        left[tuple(lo)] = -a[tuple(lo)]
        right[tuple(hi)] = -a[tuple(hi)]
        shape = list(a.shape)
        shape[axis] *= 2
        b = np.empty(shape, dtype=a.dtype)
        even, odd = [slice(None)]*3, [slice(None)]*3
        even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
        b[tuple(even)] = .75*a + .25*left
        b[tuple(odd)] = .75*a + .25*right
        a = b
    return a


def restrict(fine):
    """Full weighting: exact transpose of prolongation divided by eight."""
    a = fine
    for axis in range(3):
        even, odd = [slice(None)]*3, [slice(None)]*3
        even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
        e, o = a[tuple(even)], a[tuple(odd)]
        right, left = np.roll(e, -1, axis), np.roll(o, 1, axis)
        lo: list = [slice(None)]*3
        hi: list = [slice(None)]*3
        lo[axis], hi[axis] = 0, -1
        left[tuple(lo)], right[tuple(hi)] = -e[tuple(lo)], -o[tuple(hi)]
        a = (.75*(e+o) + .25*(left+right))*.5
    return a


def dot(a, b):
    # Accumulate AND multiply in float64; return a Python scalar so NumPy 2
    # does not accidentally promote float32 CG vectors to float64 (NEP 50).
    return float(np.einsum('ijk,ijk->', a, b, dtype=np.float64))


def solve(backend, n, dtype, maxiter):
    if n < 4 or maxiter < 1:
        raise ValueError('solve requires N >= 4 and maxiter >= 1')
    exact, k = setup(n, dtype)
    rhs = backend.apply(exact, k)
    levels = [k]
    while len(levels[-1]) % 2 == 0 and len(levels[-1]) > 5:
        levels.append(setup(len(levels[-1])//2, dtype)[1])
    def cycle(b, level=0):
        kk = levels[level]
        u = np.zeros_like(b)
        if level == len(levels)-1:
            u = backend.smooth(u, b, kk, 12, False)
            return backend.smooth(u, b, kk, 12, True)
        u = backend.smooth(u, b, kk, 2, False)
        residual = b-backend.apply(u, kk)
        u += prolong(cycle(restrict(residual), level+1))
        return backend.smooth(u, b, kk, 2, True)
    u = np.zeros_like(rhs)
    r = rhs.copy()
    normb = np.sqrt(dot(rhs, rhs))
    tolerance = 1e-8 if dtype == np.float64 else 1e-5
    z = cycle(r)
    p, rz = z.copy(), dot(r, z)
    relative, iteration = 1.0, 0
    for iteration in range(1, maxiter+1):
        if p.dtype != np.dtype(dtype):
            raise TypeError(f'CG vector dtype drift: {p.dtype}, expected {np.dtype(dtype)}')
        ap = backend.apply(p, k)
        pap = dot(p, ap)
        if pap <= 0 or not np.isfinite(pap):
            raise ArithmeticError('CG lost positive definiteness')
        alpha = rz/pap
        u += alpha*p
        r = rhs-backend.apply(u, k)  # true residual, no recurrence drift
        relative = np.sqrt(dot(r, r))/normb
        if relative <= tolerance:
            break
        z = cycle(r)
        newrz = dot(r, z)
        p = z + (newrz/rz)*p
        rz = newrz
    error = np.sqrt(dot(u-exact, u-exact)/dot(exact, exact))
    if relative > tolerance or error > tolerance*10:
        raise ArithmeticError(f'CG failed: residual={relative:g}, error={error:g}')
    return iteration, float(relative), float(error)


def timed(call):
    start = time.perf_counter()
    call()
    first = time.perf_counter()-start
    trials = []
    for _ in range(3):
        start = time.perf_counter()
        value = call()
        trials.append((time.perf_counter()-start, value))
    elapsed, result = min(trials, key=lambda v: v[0])
    return elapsed, result, first


def validate(backend, dtype):
    n = 8
    rng = np.random.default_rng(17)
    u = rng.random((n, n, n)).astype(dtype)
    _, k = setup(n, dtype)
    b = rng.random(u.shape).astype(dtype)
    diag, faces = coefficients(k)
    expected = u.copy()
    parity = np.indices(u.shape).sum(axis=0) % 2
    for color in (0, 1):
        residual = b-reference_apply(expected, k)
        mask = parity == color
        expected[mask] += residual[mask]/diag[mask]
    tolerance = 2e-5 if dtype == np.float32 else 2e-12
    start = time.perf_counter()
    actual_apply = backend.apply(u, k)
    actual_smooth = backend.smooth(u.copy(), b, k, 1, False)
    first_calls = time.perf_counter()-start
    np.testing.assert_allclose(actual_apply, reference_apply(u, k),
                               rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_smooth,
                               expected, rtol=tolerance, atol=tolerance)
    return first_calls


def benchmark(framework, backend, args, fp64=True, sweep_only=False):
    print(clean(f'Actual device: {backend.name}; float64 support: {fp64}'), flush=True)
    rows = []
    compile_time = None
    for dtype in ([np.float32, np.float64] if fp64 else [np.float32]):
        compile_time = validate(backend, dtype)
        for n in args.sizes:
            base = dict(framework=framework, version=version(framework),
                        device=backend.name, dtype=np.dtype(dtype).name, N=n,
                        test='sweep', peak_bytes=None, status='passed')
            # Conservative available-memory guard includes host validation arrays.
            need = n**3*np.dtype(dtype).itemsize*24
            available = getattr(backend, 'memory', 2**63)
            if n >= 256 and need > available*.7:
                rows.append(dict(base, status='skipped', error='memory guard: 24 fields'))
                continue
            try:
                exact, k = setup(n, dtype, False)
                rhs = backend.apply(exact, k)
                def call():
                    return backend.smooth(np.zeros_like(rhs), rhs, k, args.sweeps, False)
                elapsed, u, first = timed(call)
                residual = rhs-backend.apply(u, k)
                rel = float(np.sqrt(dot(residual, residual)/dot(rhs, rhs)))
                err = float(np.sqrt(dot(u-exact, u-exact)/dot(exact, exact)))
                if not np.isfinite(rel) or rel >= 1.5 or err >= 1:
                    raise ArithmeticError(f'sweep failed residual={rel}, error={err}')
                rows.append(dict(base, mcups=n**3*args.sweeps/elapsed/1e6,
                                 seconds=elapsed, iterations=args.sweeps,
                                 relative_residual=rel, relative_error=err,
                                 compile_seconds=compile_time, warmup_seconds=first,
                                 timing_scope='host orchestration + transfers + device'))
                del exact, k, rhs, u, residual
            except Exception as exc:
                rows.append(dict(base, status='failed', error=clean(f'{type(exc).__name__}: {exc}')))
        save(framework, backend.name, rows, announce=False)
    if not fp64:
        rows.append(dict(framework=framework, version=version(framework), device=backend.name,
                         test='capability', dtype='float64', status='unsupported',
                         error='float64 unavailable in this backend/device; float32 used'))
    if not args.no_solve and not sweep_only:
        dtype = np.float64 if fp64 else np.float32
        base = dict(framework=framework, version=version(framework), device=backend.name,
                    dtype=np.dtype(dtype).name, N=args.solve_n, test='mgpcg', peak_bytes=None)
        try:
            elapsed, result, first = timed(lambda: solve(backend, args.solve_n, dtype, args.maxiter))
            iterations, residual, error = result
            rows.append(dict(base, status='passed', seconds=elapsed, iterations=iterations,
                             relative_residual=residual, relative_error=error,
                             compile_seconds=compile_time, warmup_seconds=first,
                             timing_scope='end-to-end including setup'))
        except Exception as exc:
            rows.append(dict(base, status='failed', error=clean(f'{type(exc).__name__}: {exc}')))
    save(framework, backend.name, rows)
    return int(any(r['status'] == 'failed' for r in rows))
