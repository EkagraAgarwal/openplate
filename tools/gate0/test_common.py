"""Fast correctness checks; isolated uv command documented in run_all.md."""
import unittest
import numpy as np
from bench_numba import CPU
from common import dot, prolong, reference_apply, restrict, setup, solve, validate


class Numerics(unittest.TestCase):
    def test_transpose_transfers(self):
        rng = np.random.default_rng(7)
        c, f = rng.normal(size=(4,)*3), rng.normal(size=(8,)*3)
        self.assertAlmostEqual(dot(prolong(c), f), 8*dot(c, restrict(f)), places=12)

    def test_operator_spd(self):
        rng = np.random.default_rng(8)
        u, v = rng.normal(size=(8,)*3), rng.normal(size=(8,)*3)
        _, k = setup(8, np.float64)
        au, av = reference_apply(u, k), reference_apply(v, k)
        self.assertAlmostEqual(dot(u, av), dot(au, v), places=9)
        self.assertGreater(dot(u, au), 0)

    def test_numba_reference(self):
        for dtype in (np.float32, np.float64):
            validate(CPU(), dtype)

    def test_solve(self):
        for dtype, tol in ((np.float64, 1e-8), (np.float32, 1e-5)):
            iterations, residual, error = solve(CPU(), 16, dtype, 100)
            self.assertLessEqual(residual, tol)
            self.assertLess(error, tol*10)
            self.assertLess(iterations, 30)


if __name__ == '__main__':
    from numba import set_num_threads
    set_num_threads(4)
    unittest.main()
