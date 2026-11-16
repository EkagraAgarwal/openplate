"""Run with PYTHONPATH=. and the installed measurement-only extension."""
import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import bench_rust as bench


def test_numba_copies_have_not_drifted():
    for original, copied in [(bench.mg._kernel.py_func, bench.smooth_kernel),
                              (bench.ls_numba.rhs.py_func, bench.weno_kernel)]:
        bodies = []
        for fn in (original, copied):
            node = ast.parse(inspect.getsource(fn)).body[0]
            assert isinstance(node, ast.FunctionDef)
            body = node.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]  # optional engine docstring
            bodies.append(ast.dump(ast.Module(body=body, type_ignores=[])))
        assert bodies[0] == bodies[1]


def test_reference_equivalence():
    bench.check_contracts()


@pytest.mark.parametrize("dx", [0.0, -1.0, float("nan"), float("inf")])
def test_rhs_rejects_bad_spacing(dx):
    a = np.ones((4, 4, 4))
    with pytest.raises(ValueError):
        bench.rs.rhs(a, a, dx, (False,)*3)


def test_smoother_rejects_bad_shapes_and_layouts():
    l = bench.level((4, 4, 4), (False,)*3)
    with pytest.raises(ValueError):
        bench.rs.smooth(l.x, l.rhs, l.faces[0][:-1], *l.faces[1:], l.base,
                        l.diag, l.work, (False,)*3)
    with pytest.raises(ValueError):
        bench.rs.smooth(l.x.transpose(1, 0, 2), l.rhs, *l.faces, l.base,
                        l.diag, l.work, (False,)*3)


def test_smoother_rejects_aliasing():
    l = bench.level((4, 4, 4), (False,)*3)
    with pytest.raises((ValueError, RuntimeError)):
        bench.rs.smooth(l.x, l.x, *l.faces, l.base, l.diag, l.work, (False,)*3)
