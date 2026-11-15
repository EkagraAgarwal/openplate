# Verification notes

Changes are confined to `tools/gate0/`; no commit, project dependency edits,
or changes to `uv.lock`, `.venv`, or `platesim` were made by this task.

The repository-intelligence tools are bound to a different open project,
`openplate`, not this `openplate_3d` worktree. Queries for `tools/gate0`
returned no matching symbols. Explicit graph build for this worktree failed:

```text
codegraph build: root '<repo>' is outside the open project ('<repo>'); the code graph only indexes the open project
```

The required generic JavaScript verification command `bun typecheck` is not
applicable to this Python repository and failed with:

```text
error: Script not found "typecheck"
```

Python numerical unit tests, per-backend reference validation, manufactured
solution checks and isolated Python syntax compilation are the relevant checks.
Exact execution outcomes are in the result JSON files and `summary.md`.

Completed verification:

- `test_common.py`: 4 tests passed (transfer adjointness, SPD operator,
  Numba/reference agreement, float32/float64 manufactured solves on N=16).
- Syntax parsing: all 12 Python files passed.
- Full `check_env.py --list` environment: all 12 package checks passed;
  off-screen sphere PNG was nonempty.
- All six benchmark scripts ran their full default sweep sizes (160, 256).
  Every accepted sweep passed its independent reference validation.
- CPU and OpenCL N=160 float64 solves passed. Taichi/Quadrants float32 solves
  failed validation; no solve timing or success was accepted for them.
- Each GPU entrypoint explicitly requested RX 7700S at N=160 and rejected the
  absent adapter instead of silently choosing integrated graphics.
- `summarize.py` completed and merged current and historical JSON records.
