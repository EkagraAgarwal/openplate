"""Synthetic cm/s level-set checks; no Grid3D or measured inputs."""
import importlib.util
import math

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from platesim.tile3d import levelset3d as ls

PER = (False, False, False)


@pytest.fixture(scope="module", autouse=True)
def bounded_numba_threads():
    """Keep tiny synthetic parallel kernels below the per-file time budget."""
    try:
        import numba
    except ImportError:
        yield
        return
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    try:
        yield
    finally:
        numba.set_num_threads(previous)


def mesh(n=48):
    dx = 1.0 / n
    a = (np.arange(n) + 0.5) * dx - 0.5
    return np.meshgrid(a, a, a, indexing="ij"), dx


def sphere(n=48):
    (x, y, z), dx = mesh(n)
    return np.sqrt(x*x + y*y + z*z) - 0.23, dx


def evolve(p, dx, distance, scheme="weno5", periodic=PER):
    steps = math.ceil(abs(distance) / (0.25 * dx))
    for _ in range(steps):
        p = ls.advect(p, np.sign(distance), abs(distance)/steps, dx,
                      periodic, scheme, backend="numpy")
    return p


def sample(p, points, dx):
    return map_coordinates(p, (np.asarray(points).T + 0.5)/dx - 0.5,
                           order=1, mode="nearest")


def test_rectangular_trench_offset():
    (x, y, z), dx = mesh(40)
    a, depth, growth = 0.18, 0.22, 0.06
    r = np.abs(x)
    floor = np.hypot(np.maximum(r-a, 0), z+depth)
    wall = np.hypot(r-a, z-np.clip(z, -depth, 0))
    top = np.hypot(np.maximum(a-r, 0), z)
    p = np.minimum(np.minimum(floor, wall), top)
    p = np.where((z < -depth) | ((r > a) & (z < 0)), -p, p)
    out = evolve(p, dx, growth)
    # Exact Minkowski offset: mouth convex arcs have radius growth;
    # the intersections of offset floor/walls remain sharp inside corners.
    exact = p - growth
    near = np.abs(exact) < dx
    error = np.max(np.abs(out[near]-exact[near])) / dx
    theta = np.linspace(0, np.pi/2, 40)
    arc = np.column_stack((a-growth*np.cos(theta), np.zeros(40),
                           growth*np.sin(theta)))
    arc_error = np.max(np.abs(sample(out, arc, dx))) / dx
    inner = sample(out, [[a-growth, 0, -depth+growth]], dx)[0]/dx
    print(f"\nT1 trench offset max={error:.6g} cells; arc={arc_error:.6g}; inner={inner:.6g}")
    assert max(error, arc_error, abs(inner)) < 1.5


def test_sphere_growth():
    p, dx = sphere()
    out = evolve(p, dx, 0.06)
    radius = 0.29
    rng = np.random.default_rng(12)
    directions = rng.normal(size=(600, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radius_error = np.max(np.abs(sample(out, radius*directions, dx))) / radius
    volume_error = abs(ls.solid_volume(out, dx)/(4*np.pi*radius**3/3)-1)
    print(f"\nT2 sphere radius_error={radius_error:.8g}; volume_error={volume_error:.8g}")
    assert radius_error < 0.01
    assert volume_error < 0.02


@pytest.mark.parametrize("slotted", [False, True])
def test_grow_shrink(slotted):
    p, dx = sphere(36)
    if slotted:
        (x, y, z), _ = mesh(36)
        # Wide open slot: growth does not close it (closure is irreversible).
        p = np.maximum(p, np.minimum(0.105-np.abs(x), z+0.06))
        p = ls.reinit(p, dx, PER)
    v0 = ls.solid_volume(p, dx)
    errors = {}
    for scheme in ("godunov1", "weno5"):
        out = evolve(evolve(p, dx, 0.035, scheme), dx, -0.035, scheme)
        errors[scheme] = abs(ls.solid_volume(out, dx)/v0-1)
    print(f"\nT3 slotted={slotted}: {errors}")
    assert errors["weno5"] < errors["godunov1"]


def test_reinit_distance_and_front():
    p, dx = sphere()
    (x, y, z), _ = mesh()
    distorted = p * (1.5 + 0.25*np.sin(4*x)*np.cos(3*z))
    out = ls.reinit(distorted, dx, PER)
    near = np.abs(p) < 2*dx
    grad_error = np.max(np.abs(ls.grad_mag(out, dx, PER)[near]-1))
    assert grad_error < 0.05
    max_motion = 0.0
    # Compare edge-interpolated zero crossings, not just signs at cells.
    for axis in range(3):
        a = np.moveaxis(distorted, axis, 0)
        b = np.moveaxis(out, axis, 0)
        cross = a[:-1]*a[1:] < 0
        t0 = a[:-1][cross]/(a[:-1][cross]-a[1:][cross])
        t1 = b[:-1][cross]/(b[:-1][cross]-b[1:][cross])
        max_motion = max(max_motion, np.max(np.abs(t1-t0)))
        assert np.max(np.abs(t1-t0)) < 0.2
    print(f"\nT4 gradient_error={grad_error:.8g}; zero_motion={max_motion:.8g} cells")
    narrow = ls.reinit(distorted, dx, PER, narrow_cells=4)
    far = np.abs(out) > 5*dx
    np.testing.assert_array_equal(narrow[far], distorted[far])
    assert np.max(np.abs(ls.grad_mag(narrow, dx, PER)[near]-1)) < 0.05


def test_extend_normal_constancy():
    p, dx = sphere(40)
    (x, y, z), _ = mesh(40)
    r = p + 0.23
    field = 2 + x/np.maximum(r, dx)
    out = ls.extend(p, field, dx, PER)
    # Deliberately corrupt values away from the front: these are not seeds.
    corrupted = np.where(np.abs(p) < 2*dx, field, 17.0)
    again = ls.extend(p, corrupted, dx, PER)
    shell = (r > 0.14) & (r < 0.34)
    assert np.max(np.abs(out[shell]-field[shell])) < 0.08
    np.testing.assert_allclose(again[shell], out[shell], atol=0.03)


def test_volume_correction_and_fraction():
    p, dx = sphere(32)
    target = ls.solid_volume(p, dx)*1.08
    out = ls.volume_correct(p, target, dx)
    assert abs(ls.solid_volume(out, dx)/target-1) < 1e-4
    band = np.abs(p) < 8*dx
    np.testing.assert_array_equal(out[~band], p[~band])
    assert np.ptp((out-p)[band]) < 1e-14
    frac = ls.electrolyte_fraction(p, dx)
    assert np.all((frac >= 0) & (frac <= 1))
    assert isinstance(ls.solid_volume(p.astype(np.float32), dx), float)
    with pytest.raises(ValueError):
        ls.volume_correct(p, p.size*dx**3*2, dx)


def test_void_labels():
    p = -np.ones((20, 20, 20))
    p[:, :, 16:] = 1
    p[8:12, 8:12, 4:8] = 1
    bulk = np.zeros(p.shape, bool)
    bulk[:, :, -1] = True
    labels, count = ls.label_voids(p, bulk)
    assert count == 1
    assert np.count_nonzero(labels) == 64
    p[8:12, 8:12, 4:] = 1
    labels, count = ls.label_voids(p, bulk)
    assert count == 0 and not labels.any()


@pytest.mark.parametrize("scheme", ["godunov1", "weno5"])
def test_backends_periodic_and_band(scheme):
    p, dx = sphere(18)
    speed = 0.1*np.cos(20*p)
    per = (True, True, True)
    out = ls.advect(p, speed, dx, dx, per, scheme, backend="numpy")
    shift = (8, -5, 3)
    axes = (0, 1, 2)
    shifted = ls.advect(np.roll(p, shift, axes), np.roll(speed, shift, axes),
                        dx, dx, per, scheme, backend="numpy")
    np.testing.assert_allclose(shifted, np.roll(out, shift, axes), atol=1e-14)
    band = ls.advect(p, speed, dx, dx, per, scheme, 3, "numpy")
    mask = np.abs(p) < 3*dx
    np.testing.assert_allclose(band[mask], out[mask], atol=1e-14)
    np.testing.assert_array_equal(band[~mask], p[~mask])
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba missing")
    for bc in (per, (True, False, False)):
        for width in (None, 3):
            a = ls.advect(p, speed, dx, dx, bc, scheme, width, "numpy")
            b = ls.advect(p, speed, dx, dx, bc, scheme, width, "numba")
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def test_curvature_and_validation():
    p, dx = sphere(32)
    shell = np.abs(p) < dx
    expected = 2/(p[shell]+0.23)
    np.testing.assert_allclose(ls.curvature(p, dx, PER)[shell], expected, rtol=0.04)
    for kw in ({"backend": "cuda"}, {"scheme": "eno2"}):
        with pytest.raises(ValueError):
            ls.advect(p, 1, dx/8, dx, **kw)


def test_missing_numba_fallback_and_precision(monkeypatch):
    p, dx = sphere(12)
    p = p.astype(np.float32)
    original = p.copy()

    def missing(*args, **kwargs):
        raise ImportError("synthetic absent optional numba")

    monkeypatch.setattr(ls.importlib, "import_module", missing)
    ref = ls.advect(p, 1, dx/8, dx, backend="numpy")
    for backend in ("auto", "numba"):
        out = ls.advect(p, 1, dx/8, dx, backend=backend)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(p, original)
