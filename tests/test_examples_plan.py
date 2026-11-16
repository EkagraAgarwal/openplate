"""Public examples must plan as shipped, without running a solver."""
import json
import math
from pathlib import Path

import pytest

from platesim.sample3d.pipeline import plan_tile, run_sample
from platesim.sample3d.planner import CELL_LIMIT

EXAMPLES = Path(__file__).resolve().parents[1] / 'platesim/spec3d/examples'


@pytest.mark.parametrize('path', sorted(EXAMPLES.glob('*.json')), ids=lambda p: p.stem)
def test_example_plans_with_defaults(path):
    plans = plan_tile(json.loads(path.read_text(encoding='utf-8')))
    assert plans
    for plan in plans:
        assert 0 < plan['n_cells'] == math.prod(plan['shape']) <= CELL_LIMIT


@pytest.mark.slow
def test_quick_trench_public_golden(tmp_path):
    """Unmodified preview example, full solve/render/export; measured 95.38 s on Windows."""
    spec = json.loads((EXAMPLES / 'quick_trench.json').read_text(encoding='utf-8'))
    assert spec['numerics']['quality'] == 'preview'
    result = run_sample(spec, tmp_path / 'quick_trench')
    quantities = {q['name']: q['value'] for q in result['quantities']}
    assert abs(quantities['floor_over_field'] - 1.2318707123122348) < 0.003
    assert abs(quantities['field_um'] - 26.05796267877618) < 0.5
