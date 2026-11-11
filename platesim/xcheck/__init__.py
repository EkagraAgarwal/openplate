"""Independent cross-checks of the platesim models against external solvers.

``fipy_trench`` re-implements the additive-free trench deposition of
:mod:`platesim.feature.engine` in FiPy (Task 4 of ``plan.md``) so the level-set
engine can be validated against a second code.  It needs the optional extras
``fipy`` and ``scikit-fmm``::

    uv run --no-project --with numpy --with scipy --with scikit-fmm --with fipy \\
        python -m platesim.xcheck.fipy_trench --sites narrow

The FiPy imports inside the module are guarded, so importing this package does
not require FiPy.
"""
