"""Synthetic input-contract tests; no measured data."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from platesim.spec3d import Manifest, SampleSpec, export_schema, spec_hash, validate_spec
from platesim.spec3d.__main__ import app

EXAMPLES = Path(__file__).parents[1] / "platesim/spec3d/examples"


def sample():
    return json.loads((EXAMPLES / "single_trench.json").read_text())


@pytest.mark.parametrize("name", ["coupon_example", "single_trench", "tsv_array"])
def test_examples(name):
    spec, errors, warnings = validate_spec(EXAMPLES / f"{name}.json")
    assert spec is not None and errors == [] and warnings == []


def test_waveform():
    spec, _, _ = validate_spec(EXAMPLES / "coupon_example.json")
    step = spec.recipe.steps[0]
    assert step.j_net_mA_cm2 == pytest.approx((20 * 10 - 20 * 1) / 11)
    assert step.duty == pytest.approx(10 / 11)
    data = sample()
    data["recipe"]["steps"][0].update(t_reverse_ms=0, t_off_ms=0)
    spec, errors, _ = validate_spec(data)
    assert not errors
    assert spec.recipe.steps[0].duty == 1


@pytest.mark.parametrize("path,value", [
    ("layout.die_um.0", -1),
    ("layout.features.0.w_um", 0),
    ("layout.features.0.x_um", -1000),
    ("layout.features.0.r_um", 1000),
    ("etch.default.depth_um", 0),
    ("etch.default.sidewall_deg", 59),
    ("etch.default.bottom_radius_um", -1),
    ("bath.temperature_C", 61),
    ("bath.cu_g_L", -1),
    ("setup.area_cm2", 0),
    ("setup.delta_um", 0),
    ("numerics.dx_um", -1),
    ("numerics.device", -1),
    ("recipe.steps.0.t_forward_ms", 0),
    ("recipe.steps.0.j_reverse_mA_cm2", 10000),
    ("recipe.steps.0.duration_h", -1),
    ("recipe.steps.0.charge_C_cm2", 1),
    ("layout.features.0.surprise", True),
    ("surprise", True),
])
def test_invalid_paths(path, value):
    data = sample()
    keys = path.split(".")
    node = data
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    node[int(keys[-1]) if isinstance(node, list) else keys[-1]] = value
    spec, errors, _ = validate_spec(data)
    assert spec is None
    assert any(e["path"] == path and e["fix"] and e["allowed"] for e in errors)


def test_hash():
    data = sample()
    spec = SampleSpec.model_validate(data)
    filled = spec.model_dump(mode="json")
    assert spec_hash(data) == spec_hash(dict(reversed(list(filled.items()))))
    filled["etch"]["default"]["depth_um"] += 1
    assert spec_hash(data) != spec_hash(filled)
    filled = spec.model_dump(mode="json")
    filled["layout"]["die_um"] = [int(v) for v in filled["layout"]["die_um"]]
    assert spec_hash(filled) == spec_hash(spec)


def test_schema(tmp_path):
    path = tmp_path / "schema.json"
    export_schema(path)
    schema = json.loads(path.read_text())
    assert set(schema["$defs"]["SampleSpec"]["properties"]) == {
        "layout", "etch", "recipe", "bath", "setup", "numerics", "additives"}
    assert "Additives" in schema["$defs"]
    assert "Manifest" in schema["$defs"]


def test_cli(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["validate", str(EXAMPLES / "single_trench.json")]).exit_code == 0
    bad = tmp_path / "bad.json"
    bad.write_text('{"unknown": 1}')
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["errors"]
    assert runner.invoke(app, ["hash", str(EXAMPLES / "single_trench.json")]).exit_code == 0


@pytest.mark.parametrize("bad", [None, 42, [], "{broken", {"layout": None}, Path("missing-spec.json")])
def test_bad_input_never_raises(bad):
    spec, errors, warnings = validate_spec(bad)
    assert spec is None and errors and not warnings
    assert all(set(e) == {"path", "value", "allowed", "fix"} and e["fix"] for e in errors)
    json.dumps(errors, allow_nan=False)


def test_charge_and_high_current():
    data = sample()
    step = data["recipe"]["steps"][0]
    del step["duration_h"]
    spec, errors, _ = validate_spec(data)
    assert spec is None and errors[0]["path"] == "recipe.steps.0.charge_C_cm2"
    step.update(charge_C_cm2=10, j_forward_mA_cm2=100)
    spec, errors, warnings = validate_spec(data)
    assert not errors and spec is not None
    assert warnings[0]["path"] == "recipe.steps.0.j_net_mA_cm2"
    assert spec.recipe.steps[0].charge_C_cm2 == 10


def test_array_bounds_and_rotation():
    data = json.loads((EXAMPLES / "tsv_array.json").read_text())
    data["layout"]["features"][0]["array"]["nx"] = 100
    assert validate_spec(data)[1][0]["path"] == "layout.features.0.x_um"
    data = sample()
    data["layout"]["features"][0].update(x_um=200, rotation_deg=90)
    assert validate_spec(data)[1][0]["path"] == "layout.features.0.x_um"


def test_finite_and_depth_table():
    data = sample()
    data["bath"]["temperature_C"] = float("nan")
    assert validate_spec(data)[1][0]["path"] == "bath.temperature_C"
    json.dumps(validate_spec(data)[1], allow_nan=False)
    data = sample()
    data["etch"]["default"]["depth_vs_width"] = [[20, 30], [10, 20]]
    assert validate_spec(data)[1][0]["path"] == "etch.default.depth_vs_width"


def test_manifest():
    data = dict(spec_hash="a" * 64, code_version="synthetic", backend="numpy", device="cpu",
                dx_um=4, physics="L1", created_utc="2026-01-01T00:00:00Z",
                waveform_sensitivity="duty-averaged", summary_text="Synthetic example",
                quantities=[dict(name="depth", value=88, unit="um", flag="preview")],
                checks=[dict(name="balance", value=0, limit=0.01, passed=True)],
                artifacts=[dict(kind="json", path="series.json", title="Series")])
    manifest = Manifest.model_validate(data)
    assert Manifest.model_validate_json(manifest.model_dump_json()) == manifest
    for key, value in [("spec_hash", "bad"), ("dx_um", 0),
                       ("created_utc", "2026-01-01T00:00:00"), ("unknown", 1)]:
        with pytest.raises(ValidationError):
            Manifest.model_validate({**data, key: value})


def test_json_input_and_schema_cli(tmp_path):
    assert validate_spec(json.dumps(sample()))[1] == []
    schema = tmp_path / "schema.json"
    assert CliRunner().invoke(app, ["schema", str(schema)]).exit_code == 0
    assert json.loads(schema.read_text())["$defs"]["Manifest"]["additionalProperties"] is False
