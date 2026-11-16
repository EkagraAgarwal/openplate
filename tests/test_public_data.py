"""The optional-data guard must not disable tests when data is available."""
import pytest

import conftest


def test_require_data_skips_only_missing_files(tmp_path, monkeypatch, require_data):
    monkeypatch.setattr(conftest, "DATA_ROOT", tmp_path)
    (tmp_path / "present.json").write_text("{}", encoding="utf-8")
    require_data("present.json")
    with pytest.raises(pytest.skip.Exception, match="optional team/measured data.*missing.json"):
        require_data("present.json", "missing.json")
