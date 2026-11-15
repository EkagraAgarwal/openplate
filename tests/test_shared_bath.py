"""Tests for platesim.core.shared_bath (plan v2 A4 shared conductivity, F2 one bath object).

Run: uv run --with pytest python -m pytest tests/test_shared_bath.py -q -p no:cacheprovider

Fast: no PDE, no meshing.  The only non-trivial cost is the speciated conductivity
(a 4-species Newton solve), a few ms.

The two hard requirements:

* ``to_core_bath()`` with ``constants='core_legacy'`` equals the default ``Bath()`` field by
  field, exactly (bit for bit, not ``approx``).
* ``Params(**to_feature_kwargs())`` with ``constants='feature_legacy'`` equals the default
  ``Params()`` field by field, exactly.
"""
from __future__ import annotations

import dataclasses

import pytest

from platesim.core import speciation
from platesim.core.bath import Bath
from platesim.core.constants import F as CORE_F
from platesim.core.constants import R_GAS as CORE_R
from platesim.core.shared_bath import (
    CONSTANT_SETS,
    KAPPA_SOURCES,
    SharedBath,
    diff_report,
    format_diff_report,
)
from platesim.feature.engine.params import R_GAS as FEATURE_R
from platesim.feature.engine.params import Params, default_enhancement, derive

M_CU = 63.546
M_H2SO4 = 98.079
C_CU = 65.0 / M_CU
C_ACID = 8.0 / M_H2SO4


def _bath_fields(b: Bath) -> dict:
    return {f.name: getattr(b, f.name) for f in dataclasses.fields(Bath)}


# ---------------------------------------------------------------- legacy round trips
def test_core_roundtrip_is_exact_field_by_field():
    """SharedBath(constants='core_legacy').to_core_bath() == Bath(), bit for bit."""
    sb = SharedBath(constants="core_legacy")
    got = sb.to_core_bath()
    ref = Bath()
    assert isinstance(got, Bath)
    for name, value in _bath_fields(ref).items():
        assert getattr(got, name) == value, f"Bath.{name}: {getattr(got, name)!r} != {value!r}"
    assert _bath_fields(got) == _bath_fields(ref)
    # the same for every derived quantity the cell/coupon/efficiency code reads
    for name in ("c_cu", "D_cu", "D_cu_eff", "kappa", "i0_cu", "i0_cu_supp", "T_K", "f_RT"):
        assert getattr(got, name) == getattr(ref, name), name


def test_feature_roundtrip_is_exact_field_by_field():
    """Params(**SharedBath(constants='feature_legacy').to_feature_kwargs()) == Params()."""
    kw = SharedBath(constants="feature_legacy").to_feature_kwargs()
    got = dataclasses.asdict(Params(**kw))
    ref = dataclasses.asdict(Params())
    for name in ref:
        assert got[name] == ref[name], f"Params.{name}: {got[name]!r} != {ref[name]!r}"
    assert got == ref


def test_constructor_roundtrips():
    """from_core_bath / from_feature_params survive their own adapters exactly."""
    b0 = Bath()
    sb_core = SharedBath.from_core_bath(b0)
    assert sb_core.constants == "core_legacy"
    assert sb_core.E_mig == b0.E_mig == 1.67
    assert _bath_fields(sb_core.to_core_bath()) == _bath_fields(b0)

    p0 = Params()
    sb_feat = SharedBath.from_feature_params(p0)
    assert sb_feat.constants == "feature_legacy"
    assert sb_feat.E_mig == p0.E_mig == default_enhancement()
    assert dataclasses.asdict(Params(**sb_feat.to_feature_kwargs())) == dataclasses.asdict(p0)

    # a modified Bath still round-trips (E_mig is carried as a measured override)
    b1 = Bath(T_C=40.0, kappa_22=0.08, E_mig=1.9)
    assert _bath_fields(SharedBath.from_core_bath(b1).to_core_bath()) == _bath_fields(b1)


# ---------------------------------------------------------------- conductivity options (A4)
def test_kappa_placeholder():
    sb = SharedBath()
    value, label = sb.kappa()
    assert value == 0.065
    assert label == "placeholder_0.065"
    assert sb.kappa_source in KAPPA_SOURCES
    # the linear T rule is the one both legacy paths use
    for T in (22.0, 40.0, 60.0):
        assert sb.at(T).kappa()[0] == Bath().at(T).kappa
        assert sb.at(T).kappa()[0] == sb.kappa_placeholder(T)


def test_kappa_nernst_einstein_matches_phys_and_speciation():
    """NE with the bath-effective D set == Phys.kappa_ne (exact); == speciation with D override."""
    sb = SharedBath()
    for T in (22.0, 25.0, 40.0):
        k = sb.kappa_nernst_einstein(T)
        assert k == derive(Params(), T).kappa_ne, T
        ref = speciation.conductivity(
            speciation.dissociated_spec(C_CU, C_ACID, T_C=T),
            D={"cu": 2.65e-6, "h": 9.31e-5, "so4": 1.07e-5, "hso4": 1.33e-5},
            T_C=T,
        )
        assert k == pytest.approx(ref, rel=1e-6)
    assert 0.20 < sb.kappa_nernst_einstein(22.0) < 0.32


def test_kappa_speciated_matches_speciation_module():
    sb = SharedBath(kappa_source="speciated")
    value, label = sb.kappa()
    assert label == "speciated"
    ref = speciation.conductivity(speciation.speciate(C_CU, C_ACID, T_C=22.0), T_C=22.0)
    assert value == ref
    assert value == pytest.approx(0.0610, abs=5e-4)
    # pairing + HSO4- remove carriers: the speciated value is far below free-ion NE
    assert value < 0.5 * sb.kappa_nernst_einstein(22.0)
    # explicit chemistry settings are honoured
    ideal = sb.with_(activity="ideal")
    assert ideal.kappa()[0] != value


def test_kappa_measured_and_labels():
    sb = SharedBath(kappa_source="measured", kappa_measured_S_cm=0.21)
    assert sb.kappa() == (0.21, "measured")
    assert sb.kappa(40.0)[0] == 0.21 * (1 + 0.018 * (40.0 - 22.0))
    assert sb.to_feature_kwargs()["kappa_22"] == 0.21
    with pytest.raises(ValueError):
        SharedBath(kappa_source="measured")
    with pytest.raises(ValueError):
        SharedBath(kappa_source="bogus")
    with pytest.raises(ValueError):
        SharedBath(constants="bogus")


def test_conductivity_choice_changes_exactly_one_adapter_field():
    """A4 wiring: selecting a kappa source must move only kappa_22 in Params / Bath."""
    ref_bath = _bath_fields(Bath())
    ref_params = dataclasses.asdict(Params())

    b = SharedBath(constants="core_legacy", kappa_source="speciated").to_core_bath()
    changed = {k for k, v in _bath_fields(b).items() if v != ref_bath[k]}
    assert changed == {"kappa_22"}

    p = Params(**SharedBath(kappa_source="nernst_einstein").to_feature_kwargs())
    changed = {k for k, v in dataclasses.asdict(p).items() if v != ref_params[k]}
    assert changed == {"kappa_22"}
    assert p.kappa_22 == derive(Params(), 22.0).kappa_ne
    assert derive(p, 22.0).kappa == p.kappa_22


# ---------------------------------------------------------------- constant sets
def test_legacy_constant_sets_are_distinct():
    feat = SharedBath(constants="feature_legacy")
    core = SharedBath(constants="core_legacy")
    assert feat.r_gas == FEATURE_R == 8.314
    assert core.r_gas == CORE_R == 8.314462618
    assert feat.faraday == core.faraday == CORE_F == 96485.332
    assert core.E_mig == 1.67
    assert feat.E_mig == default_enhancement() != core.E_mig
    # E_mig is a thermodynamics-free geometric quantity: the computed rule is set-independent
    assert SharedBath(constants="codata").E_mig == feat.E_mig
    # kappa_per_K and the placeholder do not depend on the constant set
    assert feat.kappa() == core.kappa() == (0.065, "placeholder_0.065")


def test_codata_set_differs_from_legacy_as_expected():
    feat = SharedBath(constants="feature_legacy")
    core = SharedBath(constants="core_legacy")
    cod = SharedBath(constants="codata")
    assert cod.faraday == 96485.33212 > feat.faraday
    assert cod.r_gas == CORE_R != feat.r_gas
    # CODATA R equals the core legacy R; only F and the E_mig rule separate them
    assert cod.r_gas == core.r_gas and cod.faraday != core.faraday
    assert cod.E_mig == feat.E_mig != core.E_mig
    k_cod = cod.kappa_nernst_einstein(22.0)
    k_feat = feat.kappa_nernst_einstein(22.0)
    assert k_cod != k_feat
    assert k_cod == pytest.approx(k_feat, rel=1e-3)
    # the placeholder/measured conductivity is unaffected by the constant set
    assert cod.kappa() == feat.kappa()


# ---------------------------------------------------------------- F2 audit
def test_diff_report_between_the_two_legacy_paths():
    a = SharedBath.from_core_bath(Bath())
    b = SharedBath.from_feature_params(Params())
    labels = ("core/bath.py", "feature/params.py")
    rows = diff_report(a, b, print_table=True, labels=labels)
    table = format_diff_report(rows, labels=labels)

    by_name = {r["quantity"]: r for r in rows}
    differing = {r["quantity"] for r in rows if r["differ"]}

    # the quantities the brief asks to audit are all present
    for name in ("kappa", "E_mig", "R", "F", "D_cu25", "i0_u", "efficiency", "T_ref_arrhenius_K"):
        assert name in by_name, name

    # and these are the ones that actually disagree between the legacy paths
    assert "E_mig" in differing
    assert "R" in differing
    # the kappa rows that differ are the physics alternatives: the placeholder VALUE agrees
    # (both paths use 0.065) but the NE conductivity does not, because R differs, and the A4
    # gap row (NE / placeholder = 3.95x) is the real finding
    assert "kappa_nernst_einstein" in differing
    assert "kappa_ne_over_placeholder" in differing
    assert by_name["kappa"]["differ"] is False
    assert by_name["E_mig"]["path_a"] == 1.67
    assert by_name["E_mig"]["path_b"] == default_enhancement()
    assert by_name["R"]["path_a"] == 8.314462618
    assert by_name["R"]["path_b"] == 8.314
    # both legacy paths use F = 96485.332, so F agrees (codata is the set that changes it)
    assert "F" not in differing and by_name["F"]["differ"] is False
    # i0 is identical at the 22 C reference (exp(0) = 1 whatever R is) and differs at 40 C;
    # D_cu_eff differs through E_mig
    assert by_name["i0_u"]["differ"] is False
    assert {"i0_u_hot", "D_cu_eff"} <= differing
    # the A4 gap: the placeholder conductivity is 3-4x below free-ion NE
    gap = by_name["kappa_ne_over_placeholder"]["path_b"]
    assert 3.0 < gap < 5.0
    # core-only / feature-only parameters are reported, not silently defaulted
    assert {"cl_ppm", "D_o2_25", "s_bulk", "D_H25"} <= differing
    # both paths would call the same core.speciation, so the speciated value itself agrees
    assert by_name["kappa_speciated"]["differ"] is False

    # the table is printable and mentions the quantities
    assert "quantity" in table and "E_mig" in table and "core/bath.py" in table
    assert "YES" in table


def test_diff_report_identical_objects_agree():
    """The audit is not rigged: one object against itself shows no differences."""
    a = SharedBath.from_core_bath(Bath())
    rows = diff_report(a, a)
    assert not any(r["differ"] for r in rows)
    assert all(r["path_a"] == r["path_b"] for r in rows)


# ---------------------------------------------------------------- adapter coverage
def test_feature_kwargs_cover_the_additive_and_kinetic_sets():
    kw = SharedBath().to_feature_kwargs()
    for name in ("s_bulk", "D_s25", "Gamma", "k_plus", "k_minus", "theta0", "s_transient",
                 "k_rd", "c_acc", "k_acc", "b_acc", "use_accel", "efficiency", "alpha_s",
                 "i0_a", "D_H25", "D_SO425", "c_h_bulk", "c_so4_bulk", "T_ref_C", "logK_ip",
                 "activity"):
        assert name in kw, name
    # a non-default additive value survives the trip into Params
    sb = SharedBath(k_plus=1.0e9, efficiency=0.97, use_accel=True, c_acc=7e-9)
    p = Params(**sb.to_feature_kwargs())
    assert (p.k_plus, p.efficiency, p.use_accel, p.c_acc) == (1.0e9, 0.97, True, 7e-9)


def test_core_bath_adapter_uses_the_shared_kinetics():
    sb = SharedBath(constants="core_legacy", kappa_measured_S_cm=None)
    b = sb.at(40.0).to_core_bath()
    assert b.T_C == 40.0
    assert b.kappa == sb.at(40.0).kappa()[0]
    assert b.D_cu_eff == sb.at(40.0).D_cu_eff
    assert b.i0_cu == sb.at(40.0).i0_cu
    assert b.j_lim_cu(0.01) == sb.at(40.0).j_lim_cu(0.01)


def test_constant_sets_registry_is_complete():
    assert set(CONSTANT_SETS) == {"feature_legacy", "core_legacy", "codata"}
    for name, cs in CONSTANT_SETS.items():
        assert cs.r_gas > 8.0 and cs.faraday > 9e4, name
        assert cs.e_mig_rule in ("computed", "fixed"), name
