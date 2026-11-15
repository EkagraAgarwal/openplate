"""Device MGCG parity; absent OpenCL runtimes/devices are optional."""
import numpy as np
import pytest
from numba import get_num_threads, set_num_threads
from test_tile3d_mgcg import trench

from platesim.tile3d import operators
from platesim.tile3d.backend.mgcg_numba import MGCG

pytestmark = pytest.mark.gpu


@pytest.fixture(params=["gfx1102", "gfx1035"])
def device(request):
    cl = pytest.importorskip("pyopencl")
    try:
        devices = [d for p in cl.get_platforms() for d in p.get_devices()]
    except cl.Error as exc:
        pytest.skip(str(exc))
    if not any(request.param in d.name and "cl_khr_fp64" in d.extensions for d in devices):
        pytest.skip(f"{request.param} with fp64 unavailable")
    return request.param


@pytest.fixture(autouse=True)
def threads():
    old = get_num_threads()
    set_num_threads(min(old, 4))
    yield
    set_num_threads(old)


def test_apply(device):
    from platesim.tile3d.backend.mgcg_opencl import MGCG_CL
    rng = np.random.default_rng(31)
    shape = (12, 8, 10)
    faces = operators.face_coefficients(np.exp(rng.normal(size=shape)), (True, True, False))
    args = {"active": rng.random(shape) > .3, "reaction": rng.random(shape),
            "dirichlet": {"z_hi": 2.0}}
    cpu = MGCG(*faces, .07, **args)
    gpu = MGCG_CL(*faces, .07, device=device, **args)
    u = rng.normal(size=shape)
    np.testing.assert_allclose(gpu.apply(u), cpu.apply(u), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gpu.apply(u.ravel()), cpu.apply(u).ravel(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape,periodic", [
    ((48, 8, 48), (True, True, False)),
    ((43, 55, 23), (False, False, False)),
    ((9, 8, 11), (True, True, False)),
])
def test_solve_and_mixed(device, shape, periodic):
    from platesim.tile3d.backend.mgcg_opencl import MGCG_CL
    faces, dx, args, A, b = trench(shape, periodic)
    cpu = MGCG(*faces, dx, **args)
    expected, ref = cpu.solve(b, tol=1e-10)
    counts = []
    for dtype in (np.float64, "mixed"):
        gpu = MGCG_CL(*faces, dx, device=device, dtype=dtype, **args)
        x, info = gpu.solve(b, tol=1e-10)
        assert info["converged"], info
        assert np.linalg.norm(b - A @ x) / np.linalg.norm(b) <= 1e-10
        assert np.linalg.norm(x - expected) / np.linalg.norm(expected) < 1e-8
        counts.append(info["iterations"])
        if dtype != "mixed":
            assert abs(info["iterations"] - ref["iterations"]) <= 2
    assert counts[1] <= 1.5 * counts[0]


@pytest.mark.parametrize("smoother", ["rbgs", "jacobi"])
def test_cycle_reuse_and_edges(device, smoother):
    from platesim.tile3d.backend.mgcg_opencl import MGCG_CL
    faces, dx, args, _, b = trench((17, 9, 15), (False, False, False))
    gpu = MGCG_CL(*faces, dx, device=device, smoother=smoother, **args)
    cpu = MGCG(*faces, dx, smoother=smoother, **args)
    r = np.random.default_rng(4).normal(size=gpu.shape)
    np.testing.assert_allclose(gpu.vcycle(r), cpu.vcycle(r), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(gpu.vcycle(2*r), 2*gpu.vcycle(r), rtol=1e-12, atol=1e-12)
    faces_before = [tuple(id(f) for f in l.faces) for l in gpu.hierarchy]
    args["reaction"] = args["reaction"] * 2 + .01
    gpu.set_reaction(args["reaction"])
    assert faces_before == [tuple(id(f) for f in l.faces) for l in gpu.hierarchy]
    fresh = MGCG_CL(*faces, dx, device=device, smoother=smoother, **args)
    x, info = gpu.solve(b)
    y, other = fresh.solve(b)
    assert info["converged"] and other["converged"]
    np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-12)
    _, warm = gpu.solve(b, x0=x)
    assert warm["iterations"] == 0 and warm["converged"]
    zero, info = gpu.solve(np.zeros(gpu.shape))
    assert info["converged"] and not zero.any()
    assert not gpu.solve(b, maxiter=0)[1]["converged"]
    with pytest.raises(ValueError):
        gpu.set_reaction(-1)


def test_residency_and_selection(device, monkeypatch):
    import pyopencl.array as ca

    from platesim.tile3d.backend.mgcg_opencl import MGCG_CL, available_devices
    faces, dx, args, _, b = trench((12, 8, 12))
    index = next(i for i, d in enumerate(available_devices()) if device in d.name)
    solver = MGCG_CL(*faces, dx, device=index, dtype="mixed", **args)
    transfers = []
    original = ca.Array.get

    def get(array, *a, **kw):
        transfers.append(array.size)
        return original(array, *a, **kw)

    monkeypatch.setattr(ca.Array, "get", get)
    _, info = solver.solve(b)
    assert info["converged"], info
    # Exactly one non-scalar download: the returned solution, not per-cycle data.
    assert [n for n in transfers if n > 1] == [b.size]
    with pytest.raises(RuntimeError, match="unavailable"):
        MGCG_CL(*faces, dx, device="no-such-opencl-device", **args)
    with pytest.raises(RuntimeError, match="index"):
        MGCG_CL(*faces, dx, device=-1, **args)
