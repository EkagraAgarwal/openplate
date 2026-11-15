"""Synthetic finite-volume verification; all lengths are in cm."""
import numpy as np
import pytest

from platesim.tile3d.operators import assemble, apply_matrix_free, face_coefficients
from platesim.tile3d.solvers_ref import solve_amg, solve_direct


def manufactured(n):
    h = 1.0 / n
    x, y, z = np.meshgrid(*([(np.arange(n) + 0.5) * h] * 3), indexing="ij")
    q = 2 * np.pi
    a, b, c = np.cos(q*x), np.cos(q*y), np.cos(np.pi*z/2)
    u = 1 + 0.2*a*b*c
    k = 2 + 0.3*np.cos(q*x)*np.cos(q*y)*np.cos(np.pi*z)
    ux = -0.2*q*np.sin(q*x)*b*c
    uy = -0.2*q*a*np.sin(q*y)*c
    uz = -0.1*np.pi*a*b*np.sin(np.pi*z/2)
    kx = -0.3*q*np.sin(q*x)*np.cos(q*y)*np.cos(np.pi*z)
    ky = -0.3*q*np.cos(q*x)*np.sin(q*y)*np.cos(np.pi*z)
    kz = -0.3*np.pi*np.cos(q*x)*np.cos(q*y)*np.sin(np.pi*z)
    r = 1 + 0.1*z
    f = k*(2*q*q + (np.pi/2)**2)*(u-1) - (kx*ux+ky*uy+kz*uz) + r*u
    A, bc = assemble(*face_coefficients(k, (True, True, False)), h,
                     reaction=r, dirichlet={"z_hi": 1.0})
    return A, f.ravel()+bc, u


def test_manufactured_convergence():
    errors = []
    print("\n n     RMS error       order      AMG iterations")
    for n in (12, 24, 48):
        A, b, exact = manufactured(n)
        u, info = solve_amg(A, b, tol=1e-11)
        assert info["converged"]
        error = np.sqrt(np.mean((u-exact.ravel())**2))
        order = np.log2(errors[-1]/error) if errors else np.nan
        print(f"{n:2d}  {error:.9e}  {order:8.5f}  {info['iterations']:5d}")
        if errors:
            assert order > 1.8
        errors.append(error)


def test_column_series_resistance():
    n, k, sink = 48, 2.0, 1e5
    h = 1/n
    reaction = np.zeros((3, 2, n))
    reaction[:, :, 0] = sink
    A, b = assemble(*face_coefficients(np.full(reaction.shape, k), (True, True, False)),
                    h, reaction=reaction, dirichlet={"z_hi": 1.0})
    u = solve_direct(A, b).reshape(reaction.shape)
    flux = 2*k*(1-u[:, :, -1])/h
    analytic = 1 / ((1-h/2)/k + 1/(sink*h))
    np.testing.assert_allclose(flux, analytic, rtol=0.01)
    np.testing.assert_allclose(k*np.diff(u, axis=2)/h, analytic, rtol=0.01)
    np.testing.assert_allclose(sink*h*u[:, :, 0], flux, rtol=1e-11)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("periodic", [(True, True, False), (False, False, False), (True, True, True)])
def test_matrix_free(masked, periodic):
    rng = np.random.default_rng(194)
    shape = (5, 4, 3)
    faces = face_coefficients(np.exp(rng.normal(size=shape)), periodic)
    active = rng.random(shape) > 0.3 if masked else None
    bc = None if periodic[2] else {"z_hi": 1.7}
    r, u = rng.random(shape), rng.normal(size=shape)
    A, _ = assemble(*faces, 0.17, reaction=r, active=active, dirichlet=bc)
    got = apply_matrix_free(u, *faces, 0.17, reaction=r, active=active, dirichlet=bc)
    assert got.shape == shape
    np.testing.assert_allclose(got.ravel(), A@u.ravel(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reaction,bc", [(0.2, None), (None, {"z_hi": 0.0})])
def test_symmetric_positive_definite(reaction, bc):
    k = np.random.default_rng(8).uniform(0.1, 2, (4, 3, 3))
    A, _ = assemble(*face_coefficients(k, (True, True, False)), 0.2,
                    reaction=reaction, dirichlet=bc)
    np.testing.assert_allclose(A.toarray(), A.T.toarray(), atol=1e-13)
    assert np.linalg.eigvalsh(A.toarray()).min() > 0


def test_inactive_zero_conductivity_equivalence():
    k = np.ones((7, 6, 5))
    active = np.ones(k.shape, dtype=bool)
    active[2:5, 2:4, 1:3] = False
    kwargs = dict(reaction=0.7, dirichlet={"z_hi": 1.0})
    A, b = assemble(*face_coefficients(k, (True, True, False)), 0.2, active=active, **kwargs)
    u = solve_direct(A, b).reshape(k.shape)
    k[~active] = 0
    B, c = assemble(*face_coefficients(k, (True, True, False)), 0.2, **kwargs)
    v = solve_direct(B, c).reshape(k.shape)
    assert np.all(u[~active] == 0)
    np.testing.assert_allclose(u, v, atol=1e-13)
    np.testing.assert_allclose(A.toarray(), A.T.toarray(), atol=1e-13)
def test_amg_accuracy_and_scaling():
    counts = []
    for n in (16, 32):
        A, b, _ = manufactured(n)
        u, info = solve_amg(A, b, tol=1e-11)
        assert info["converged"]
        assert info["relative_residual"] < 1e-10
        counts.append(info["iterations"])
        if n == 16:
            np.testing.assert_allclose(u, solve_direct(A, b), rtol=1e-8, atol=1e-10)
        else:
            # A manufactured algebraic rhs avoids a costly 32^3 direct factorization.
            exact = np.random.default_rng(2).normal(size=A.shape[0])
            got, check = solve_amg(A, A@exact, tol=1e-12, x0=np.zeros_like(exact))
            assert check["converged"]
            assert np.linalg.norm(got-exact)/np.linalg.norm(exact) < 1e-8
    print(f"\nAMG scaling 16^3 -> 32^3: {counts}")
    assert counts[1] < 2*counts[0]


def test_strong_contrast():
    n = 24
    x, y, z = np.meshgrid(*([(np.arange(n)+0.5)/n]*3), indexing="ij")
    # Smooth log-conductivity band, synthetic (not measured) contrast 1e6.
    band = np.clip((z-0.25)/0.5, 0, 1)
    band = band*band*(3-2*band)
    k = 10**(6*band)
    A, b = assemble(*face_coefficients(k, (True, True, False)), 1/n,
                    reaction=1.0, dirichlet={"z_hi": 1.0})
    u, info = solve_amg(A, b, tol=1e-10)
    print(f"\nContrast 1e6: {info}")
    assert info["converged"]
    assert info["relative_residual"] < 1e-10
    assert np.isfinite(u).all()


def test_faces_and_boundary_flux():
    k = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
    k[0, 0, 0] = 0
    faces = face_coefficients(k, (True, False, False))
    assert [f.shape for f in faces] == [(3, 3, 4), (2, 4, 4), (2, 3, 5)]
    np.testing.assert_array_equal(faces[1][:, 0], 0)
    np.testing.assert_array_equal(faces[2][:, :, -1], 0)
    np.testing.assert_array_equal(faces[0][0], faces[0][-1])
    expected = np.divide(2*k[0]*k[1], k[0]+k[1])
    np.testing.assert_allclose(faces[0][1], expected)
    # Every outer face: half-cell resistance, with preserved cell conductivity.
    for axis, name in enumerate("xyz"):
        for side, index in (("lo", 0), ("hi", -1)):
            faces = face_coefficients(k, (False, False, False))
            key = f"{name}_{side}"
            A, b = assemble(*faces, 0.5, dirichlet={key: 2.0})
            diagonal = np.zeros(k.shape)
            sl = [slice(None)]*3
            sl[axis] = index
            sl = tuple(sl)
            diagonal[sl] = 2*k[sl]/0.5**2
            np.testing.assert_allclose(A@np.ones(k.size), diagonal.ravel(), atol=1e-12)
            np.testing.assert_allclose(b, 2*diagonal.ravel())


def test_degenerate_axes_and_zero_rhs():
    for shape in ((1, 1, 1), (2, 1, 2)):
        faces = face_coefficients(np.ones(shape), (True, True, True))
        A, b = assemble(*faces, 1.0, reaction=1)
        np.testing.assert_allclose(A@np.ones(np.prod(shape)), 1)
        u = np.arange(np.prod(shape), dtype=float).reshape(shape)
        np.testing.assert_allclose(apply_matrix_free(u, *faces, 1, reaction=1).ravel(), A@u.ravel())
        got, info = solve_amg(A, b)
        np.testing.assert_array_equal(got, 0)
        assert info["iterations"] == 0
        assert info["relative_residual"] == 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        face_coefficients(-np.ones((2, 2, 2)), (False,)*3)
    faces = face_coefficients(np.ones((2, 2, 2)), (True,)*3)
    with pytest.raises(ValueError):
        assemble(*faces, 1, dirichlet={"z_hi": 1})
    with pytest.raises(ValueError):
        assemble(*faces, 0)
