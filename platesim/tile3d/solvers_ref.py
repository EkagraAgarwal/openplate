"""CPU reference solvers for symmetric positive definite tile operators."""
import numpy as np
from scipy.sparse.linalg import cg, spsolve


def solve_direct(A, b):
    """Solve A*x=b using SciPy SuperLU; return C-ordered flat unknowns."""
    return np.asarray(spsolve(A, np.asarray(b, dtype=np.float64).ravel()))


def solve_amg(A, b, tol=1e-10, x0=None):
    """Smoothed-aggregation V-cycle preconditioned conjugate gradients.

    A must be SPD (anchor every connected active component with reaction
    or Dirichlet data). ``relative_residual`` is the independently recomputed
    Euclidean norm ||b-A*x||/||b||; for zero b it is the absolute residual.
    Nonconvergence is returned in info, not silently treated as success.
    PyAMG is optional and imported only when this function is called.
    """
    import pyamg

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    A = A.astype(np.float64).tocsr()
    b = np.asarray(b, dtype=np.float64).ravel()
    if A.shape != (b.size, b.size) or not np.isfinite(b).all():
        raise ValueError("A must be square and match the finite rhs")
    initial = None if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0:
        return np.zeros_like(b), dict(iterations=0, relative_residual=0.0,
                                     converged=True, cg_info=0)
    hierarchy = pyamg.smoothed_aggregation_solver(
        A, symmetry="symmetric", strength="symmetric",
        presmoother=("gauss_seidel", {"sweep": "symmetric"}),
        postsmoother=("gauss_seidel", {"sweep": "symmetric"}))
    iterations = 0

    def count(_):
        nonlocal iterations
        iterations += 1

    x, status = cg(A, b, x0=initial, rtol=tol, atol=0.0, maxiter=1000,
                   M=hierarchy.aspreconditioner(cycle="V"), callback=count)
    relative = float(np.linalg.norm(b-A@x)/norm_b)
    return x, dict(iterations=iterations, relative_residual=relative,
                   converged=bool(status == 0 and relative <= tol), cg_info=int(status))
