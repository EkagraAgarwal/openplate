"""Tests for platesim.imaging (must run in < 60 s)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from platesim.imaging.measure import (  # noqa: E402
    detect_thickness_axis,
    measure_band_auto,
    measure_manual,
)
from platesim.imaging.metadata import parse_sem_metadata  # noqa: E402

REAL_SEM = (
    ROOT.parents[0]
    / "Sample Results"
    / "Planar 2 (SiO2-Ti-Cu)"
    / "planar2_9mA_top.tif"
)


def _synthetic_cross_section(
    n_rows: int = 400,
    n_cols: int = 300,
    band_rows: int = 120,
    air: int = 30,
    film: int = 170,
    substrate: int = 70,
    noise: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """Horizontal layer stack: air / bright Cu band / darker substrate."""
    rng = np.random.default_rng(seed)
    img = np.empty((n_rows, n_cols), dtype=np.float64)
    top = (n_rows - band_rows) // 2
    bot = top + band_rows
    img[:top] = air
    img[top:bot] = film
    img[bot:] = substrate
    img += rng.normal(0.0, noise, img.shape)
    return img


def test_synthetic_band_thickness_within_2pct():
    band_rows = 120
    ps_nm = 10.0
    img = _synthetic_cross_section(band_rows=band_rows)
    res = measure_band_auto(img, ps_nm)
    expected_um = band_rows * ps_nm / 1000.0  # 1.2 um
    assert res["axis"] == "y"
    assert abs(res["mean_um"] - expected_um) / expected_um < 0.02
    assert abs(res["median_um"] - expected_um) / expected_um < 0.02


def test_synthetic_rotated_band_thickness():
    """Rotated cross-section: layers vertical, thickness along x."""
    band_cols = 90
    ps_nm = 20.0
    img = _synthetic_cross_section(n_rows=300, n_cols=400, band_rows=band_cols).T
    assert detect_thickness_axis(img) == "x"
    res = measure_band_auto(img, ps_nm, axis="x")
    expected_um = band_cols * ps_nm / 1000.0  # 1.8 um
    assert abs(res["mean_um"] - expected_um) / expected_um < 0.02


def test_manual_lines_measurement():
    img = _synthetic_cross_section()
    lines = [{"x1": 50, "y1": 140, "x2": 50, "y2": 260}]
    res = measure_manual(img, pixel_size_nm=10.0, lines=lines)
    assert res["mean_um"] == pytest.approx(1.2, rel=1e-6)


def test_tilt_correction_factor():
    img = _synthetic_cross_section(band_rows=100)
    res0 = measure_band_auto(img, 10.0, tilt_deg=0.0)
    res52 = measure_band_auto(img, 10.0, tilt_deg=52.0)
    assert res0["tilt_factor"] == 1.0
    assert res52["tilt_factor"] == pytest.approx(1.0 / np.sin(np.deg2rad(52.0)))
    assert res52["mean_um"] == pytest.approx(
        res0["mean_um"] * res52["tilt_factor"], rel=0.05
    )


@pytest.mark.skipif(not REAL_SEM.exists(), reason="real SEM TIFF not present")
def test_metadata_parser_real_file_pixel_size():
    meta = parse_sem_metadata(REAL_SEM)
    assert meta["pixel_size_nm"] > 0
    # FEI Apreo export: 6.74479 nm/px at HFW 10.36 um [M from tag 34682]
    assert 5.0 < meta["pixel_size_nm"] < 8.0
    assert meta.get("vendor") == "fei"
    assert meta.get("tilt_deg") is not None
