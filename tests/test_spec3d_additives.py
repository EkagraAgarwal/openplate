"""Contract 10g: L3 additives in the sample spec (synthetic examples only)."""
import copy
import json
from dataclasses import asdict
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


def test_additives_requires_l3():
    data = _data()
    data["additives"] = {"mode": "cl_ndr"}
    spec, errors = _validate(data)
    assert spec is None
    assert any(e["path"] == "additives" and "numerics.physics" in e["allowed"] for e in errors)


def test_l3_requires_additives():
    spec, errors = _validate(_l3_data())
    assert spec is None
    assert any(e["path"] == "numerics.physics" and "additives" in e["allowed"] for e in errors)


@pytest.mark.parametrize("name,value", [
    ("c_acc_mol_cm3", 1e-8), ("Gamma_acc_mol_cm2", 2e-10),
    ("k_acc_cm3_mol_s", 4e5), ("b_acc", -0.5),
    ("k_inc_a_1_cm", 2e4), ("D_acc_cm2_s", 3e-6),
])
def test_accelerator_fields_need_acc_mode(name, value):
    spec, errors = _validate(_l3_data({"mode": "cl_ndr", name: value}))
    assert spec is None
    assert any(e["path"] == f"additives.{name}" and 'mode="acc"' in e["allowed"] for e in errors)


def test_pulse_coverage_resolved_needs_resolved_pulse_model():
    data = _l3_data({"mode": "cl_ndr", "pulse_coverage": "resolved"})
    data["numerics"]["pulse_model"] = "net"
    spec, errors = _validate(data)
    assert spec is None
    assert any(e["path"] == "additives.pulse_coverage" and "pulse_model" in e["allowed"] for e in errors)


@pytest.mark.parametrize("name", ["k_strip_s", "k_strip_a"])
def test_strip_rates_need_resolved_coverage(name):
    spec, errors = _validate(_l3_data({"mode": "cl_ndr", name: 1.0}))
    assert spec is None
    assert any(e["path"] == f"additives.{name}" and "resolved" in e["allowed"] for e in errors)


@pytest.mark.parametrize("name,value", [("b_acc", -1.5), ("strip_return_a", 1.5)])
def test_numeric_bounds(name, value):
    spec, errors = _validate(_l3_data({"mode": "acc", name: value}))
    assert spec is None
    assert any(e["path"] == f"additives.{name}" and e["allowed"] and e["fix"] for e in errors)


def test_valid_cl_ndr_spec():
    spec, errors = _validate(_l3_data({"mode": "cl_ndr"}))
    assert spec is not None and errors == []
    assert isinstance(spec.additives, Additives)
    assert spec.additives.mode == "cl_ndr"
    assert spec.additives.suppressor is True
    assert spec.additives.pulse_coverage == "mean"


def test_valid_acc_spec():
    data = _l3_data({"mode": "acc", "pulse_coverage": "resolved", "k_strip_a": 300.0,
                     "c_acc_mol_cm3": 1e-8})
    data["numerics"]["pulse_model"] = "resolved"
    spec, errors = _validate(data)
    assert errors == []
    assert spec is not None
    assert spec.additives is not None
    assert spec.additives.mode == "acc"
    assert spec.additives.k_strip_a == 300.0
    assert spec.numerics.pulse_model == "resolved"


def test_hash_invariance_without_additives():
    data = _data("single_trench")
    base = spec_hash(data)
    with_none = copy.deepcopy(data)
    with_none["additives"] = None
    no_key = {k: v for k, v in data.items() if k != "additives"}
    assert "additives" not in no_key
    assert spec_hash(with_none) == base
    assert spec_hash(no_key) == base
    assert spec_hash(SampleSpec.model_validate(with_none)) == base
    changed = copy.deepcopy(data)
    changed["numerics"]["physics"] = "L3"
    changed["additives"] = {"mode": "cl_ndr"}
    assert spec_hash(changed) != base


def test_spec_params_additive_mapping():
    from platesim.core.constants import M_CU
    from platesim.feature.engine.params import Params, migration_enhancement
    from platesim.sample3d.pipeline import _spec_params

    override_values = dict(
        mode="acc", suppressor=False, s_bulk_mol_cm3=1.2e-7, c_acc_mol_cm3=3e-9,
        Gamma_acc_mol_cm2=2e-10, k_acc_cm3_mol_s=4e5, b_acc=-0.5,
        k_inc_a_1_cm=2e4, D_acc_cm2_s=3e-6, pulse_coverage="resolved",
        k_strip_s=2.0, k_strip_a=300.0, strip_return_a=0.25)
    data = _l3_data(override_values)
    data["numerics"]["pulse_model"] = "resolved"
    spec = SampleSpec.model_validate(data)
    params = _spec_params(spec)
    assert params.additives == "acc"
    assert params.suppressor is False
    assert params.c_cl_bulk == pytest.approx(50 / 35.453 * 1e-6)
    for field, key in (
            ("s_bulk", "s_bulk_mol_cm3"), ("c_acc", "c_acc_mol_cm3"),
            ("Gamma_acc", "Gamma_acc_mol_cm2"), ("k_acc", "k_acc_cm3_mol_s"),
            ("b_acc", "b_acc"), ("k_inc_a", "k_inc_a_1_cm"),
            ("D_acc25", "D_acc_cm2_s"), ("pulse_coverage", "pulse_coverage"),
            ("k_strip_s", "k_strip_s"), ("k_strip_a", "k_strip_a"),
            ("strip_return_a", "strip_return_a")):
        assert getattr(params, field) == override_values[key]

    # L1 regression: no additive path, constructed exactly as on origin/main.
    legacy_data = _data()
    legacy_data["numerics"].update(quality="preview", dx_um=8.0, device="cpu")
    legacy_spec = SampleSpec.model_validate(legacy_data)
    cu, acid = 40 / M_CU, 100 / 98.079
    legacy = Params(suppressor=False, c_bulk=cu * 1e-3,
                    E_mig=migration_enhancement(cu, [(1, 2 * acid), (-2, cu + acid)]),
                    c_h_bulk=2 * acid * 1e-3, c_so4_bulk=(cu + acid) * 1e-3)
    result = _spec_params(legacy_spec)
    assert result.additives == "legacy"
    assert result.c_cl_bulk == Params().c_cl_bulk
    assert result == legacy
    assert asdict(result) == asdict(legacy)


def test_end_to_end_l3_manifest(tmp_path):
    from platesim.sample3d.pipeline import run_sample

    data = _l3_data({"mode": "cl_ndr"})
    manifest = run_sample(data, tmp_path / "l3")
    assert manifest["physics"] == "L3"
    assert manifest["spec_hash"] == spec_hash(data)
    additive_names = [q["name"] for q in manifest["quantities"]
                      if q["name"].startswith("step_1.additives.")]
    assert additive_names
    assert any(name.endswith(".theta_s_mean") for name in additive_names)
    assert "cl_ndr" in manifest["summary_text"]
    assert "uncalibrated" in manifest["summary_text"].lower()
    assert all(q["flag"] in ("uncalibrated", "preview") for q in manifest["quantities"])
