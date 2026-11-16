"""Shared guards for optional, non-public measured datasets.

Only explicitly requested files are optional: missing public fixtures and other
FileNotFoundError exceptions must still fail tests.
"""
from pathlib import Path

import pytest


DATA_ROOT = Path(__file__).resolve().parents[1] / "platesim" / "data"


@pytest.fixture(scope="session")
def require_data():
    """Skip a measured-data test before doing work when its inputs are absent."""
    def require(*names):
        missing = [name for name in names if not (DATA_ROOT / name).is_file()]
        if missing:
            pytest.skip("optional team/measured data not shipped in public repo: "
                        + ", ".join(f"platesim/data/{name}" for name in missing))
    return require
