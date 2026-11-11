# platesim: module contracts

`platesim` is the simulation framework for Cu electroplating (TSV project). Python 3.11 (global interpreter `python`),
run everything from `D:\Workspace\Semiconductors\platesim` (so `import platesim` works). Desktop app in `app/` talks
to the FastAPI server in `platesim/server/`.

## Rules for every worker
- Only touch the files/directories your task names. Never edit `platesim/core/`, `platesim/data/planar_doe.json`,
  this file, or anything under `D:\Workspace\Semiconductors\research\` (read-only reference) unless told.
- Units: cm, s, mol, A, V, K inside code; `_um`, `_mA_cm2`, `_min` suffixes for other units.
- Use `platesim.core.constants`, `platesim.core.bath.Bath`, `platesim.core.cell.CellGeometry`; do not duplicate constants.
- Tag every literature number in a comment: [FT] full text, [AB] abstract, [SH] secondhand, [C] calc, [J] judgement, [M] measured.
  Give a citation (author year journal DOI) for every [FT]/[AB]/[SH] number. Never invent DOIs or data.
- Tests: `tests/test_<module>.py`, pytest, must run in < 60 s total per module (`python -m pytest tests/test_<module>.py -q`).
- No commits. No network calls at import time. Never put personal data (emails) in URLs/headers.
- Install new pip deps only if needed and list them in your final report (the lead adds them to pyproject).
- Finish with a short report: files created, how to run, test output, assumptions, known limitations.

## Model plug-in contract (what the server discovers)
Each simulation module exposes in its `__init__.py`:

```python
MODEL_SPEC = {
  "id": "cell_current",            # unique, snake_case
  "title": "Coupon current distribution",
  "description": "...",
  "params": [                       # rendered as a form by the app
     {"name": "I_total_mA", "type": "float", "default": 48.0, "min": 0.1, "max": 500, "unit": "mA", "help": "..."},
     {"name": "stir_rpm", "type": "float", ...},
     {"name": "recipe", "type": "enum", "choices": ["dc", "plan_ramp"], "default": "dc"},
     {"name": "taped", "type": "bool", "default": True},
  ],
}
def run(params: dict, progress=lambda frac, msg="": None) -> dict: ...
```
`run` returns JSON-serialisable output:
```python
{"summary": {"key": value, ...},                       # scalar results shown as a table
 "plots": [ {"kind": "line", "title": ..., "x": [...], "series": [{"name":..., "y": [...]}], "xlabel":..., "ylabel":...},
            {"kind": "heatmap", "title": ..., "x": [...], "y": [...], "z": [[...]], "zlabel": ...},
            {"kind": "contours", "title": ..., "frames": [{"t": s, "lines": [[[x,y],...], ...]}]} ],
 "warnings": [...], "assumptions": [...]}
```
Registered model packages (server imports these names): `platesim.cell`, `platesim.flow`, `platesim.efficiency`,
`platesim.feature`, `platesim.imaging`, `platesim.coupon`, `platesim.atomistic`.

## Shared data
- `platesim/data/planar_doe.json`: team planar efficiency DOE (W9 deck) + vendor rate table.
- `platesim/data/literature/*.json`: digitised literature datasets, each with `source` (full citation + DOI), `tag`, `data`.
- `platesim/data/sem_measurements.json`: thickness measurements extracted from the SEM TIFFs (imaging module output).
