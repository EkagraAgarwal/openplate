"""Contract 10i: CEAC carried accelerator coverage in the sample spec."""
import copy
import json
from pathlib import Path

import pytest

from platesim.spec3d import Additives, SampleSpec, spec_hash, validate_spec

EXAMPLES = Path(__file__).parents[1] / "platesim/spec3d/examples"


def _data(name="quick_trench"):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


def _l3_data(additives=None):
    data = _data()
    data["numerics"].update(physics="L3", quality="preview", dx_um=8.0, device="cpu")
    data["recipe"]["steps"][0]["duration_h"] = 0.05
    if additives is not None:
        data["additives"] = additives
    return data


def _validate(data):
    return validate_spec(data)[:2]


# --------------------------------------------------------------- validation
def test_ceac_requires_mean_pulse_coverage():
    data = _l3_data({"mode": "cl_ndr", "ceac": True, "pulse_coverage": "resolved"})
    data["numerics"]["pulse_model"] = "resolved"
    spec, errors = _validate(data)
    assert spec is None
    assert any(e["path"] == "additives.ceac" and 'pulse_coverage="mean"' in e["allowed"]
               for e in errors)


def test_ceac_smooth_requires_ceac():
    spec, errors = _validate(_l3_data({"mode": "cl_ndr", "ceac_smooth": True}))
    assert spec is None
    assert any(e["path"] == "additives.ceac_smooth" and "ceac" in e["allowed"]
               for e in errors)


def test_valid_cl_ndr_ceac_spec():
    spec, errors = _validate(_l3_data({"mode": "cl_ndr", "ceac": True, "ceac_smooth": True}))
    assert errors == []
    assert spec is not None
    assert isinstance(spec.additives, Additives)
    assert spec.additives.ceac is True
    assert spec.additives.ceac_smooth is True
    assert spec.additives.pulse_coverage == "mean"


# --------------------------------------------------------------------- hash
@pytest.mark.parametrize("name,prefix", [
    ("coupon_example", "c3128a4943c69c5b"),
    ("quick_trench", "3a33c86673777779"),
    ("single_trench", "7f1d0dd26430b349"),
    ("tsv_array", "add7c7658ea90070"),
])
def test_example_hashes_unchanged(name, prefix):
    assert spec_hash(_data(name)).startswith(prefix)


def test_ceac_false_explicit_hash_unchanged():
    data = _l3_data({"mode": "cl_ndr"})
    base = spec_hash(data)
    explicit = copy.deepcopy(data)
    explicit["additives"].update(ceac=False, ceac_smooth=False)
    assert spec_hash(explicit) == base
    assert spec_hash(SampleSpec.model_validate(explicit)) == base
    on = copy.deepcopy(data)
    on["additives"].update(ceac=True, ceac_smooth=True)
    assert spec_hash(on) != base


# -------------------------------------------------------------- Params map
def test_spec_params_ceac_mapping():
    from platesim.sample3d.pipeline import _spec_params

    spec = SampleSpec.model_validate(
        _l3_data({"mode": "cl_ndr", "ceac": True, "ceac_smooth": True}))
    params = _spec_params(spec)
    assert params.ceac is True
    assert params.ceac_smooth is True
    assert params.additives == "cl_ndr"
    assert params.pulse_coverage == "mean"


def test_spec_params_without_ceac_is_false():
    from platesim.sample3d.pipeline import _spec_params

    spec = SampleSpec.model_validate(_l3_data({"mode": "cl_ndr"}))
    params = _spec_params(spec)
    assert params.ceac is False
    assert params.ceac_smooth is False


# ------------------------------------------------------------- end to end
def test_end_to_end_ceac_manifest(tmp_path):
    from platesim.sample3d.pipeline import run_sample

    data = _l3_data({"mode": "cl_ndr", "ceac": True})
    manifest = run_sample(data, tmp_path / "ceac")
    assert manifest["physics"] == "L3"
    assert manifest["spec_hash"] == spec_hash(data)
    assert "CEAC" in manifest["summary_text"]
    assert "radii >= 3-4 dx" in manifest["summary_text"]
    theta = [q for q in manifest["quantities"]
             if "theta_a_floor" in q["name"] or "theta_a_field" in q["name"]]
    assert theta
    assert all(q["flag"] in ("uncalibrated", "preview") for q in theta)
