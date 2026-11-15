"""Synthetic polygon fitting and editable-spec round trips (no private layouts)."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("shapely")
from shapely.geometry import Polygon, box

from platesim.layout import load_layout, import_gds, import_image
from platesim.layout.model import make_layout
from platesim.layout.to_features import fit_features
from platesim.spec3d import SampleSpec


def round_trip(layout):
    fitted = fit_features(layout)
    assert not fitted["unfit"]
    data = {k: fitted[k] for k in ("die_um", "features")}
    spec = json.loads((Path(__file__).parents[1] / "platesim/spec3d/examples/quick_trench.json").read_text())
    spec["layout"] = data
    SampleSpec.model_validate(spec)
    restored = load_layout(data)
    assert restored.geometry("etch").area == pytest.approx(layout.geometry("etch").area, rel=.03)
    return fitted


@pytest.mark.parametrize("kind,rotation,radius", [
    ("rect", 0, 0), ("rect", 30, 0), ("rounded_rect", 0, 3), ("circle", 0, 5),
])
def test_shapes(kind, rotation, radius):
    layout = load_layout(dict(die_um=[100, 80], features=[dict(
        kind=kind, x_um=30, y_um=30, w_um=20, h_um=12, r_um=radius, rotation_deg=rotation)]))
    fitted = round_trip(layout)
    feature = fitted["features"][0]
    assert feature["kind"] == kind
    assert feature["rotation_deg"] == pytest.approx(rotation, abs=.01)
    if kind == "rounded_rect":
        assert feature["r_um"] == pytest.approx(radius, rel=.001)
    assert len(fitted["preview"]) == 1


def test_circle_array():
    layout = load_layout(dict(die_um=[100, 100], features=[dict(
        kind="circle", x_um=15, y_um=20, r_um=5,
        array=dict(nx=3, ny=2, pitch_x_um=25, pitch_y_um=30))]))
    fitted = round_trip(layout)
    assert len(fitted["features"]) == 1
    assert fitted["features"][0]["array"] == pytest.approx(dict(nx=3, ny=2, pitch_x_um=25, pitch_y_um=30))


def test_unfit_not_dropped():
    polygon = Polygon([(0, 0), (20, 0), (20, 5), (5, 5), (5, 20), (0, 20)])
    layout = make_layout([("etch", polygon)], skipped=[{"reason": "unsupported"}])
    fitted = fit_features(layout)
    assert fitted["features"] == []
    assert fitted["unfit"][0]["index"] == 0
    assert fitted["unfit"][0]["area_um2"] == polygon.area
    assert fitted["unfit"][0]["error_rel"] > .03
    assert fitted["skipped"] == layout.skipped_entities
    assert len(fitted["preview"]) == 1


def test_missing_array_member_and_hole():
    layout = make_layout([("etch", box(x, y, x+4, y+4))
                          for x, y in [(0, 0), (10, 0), (0, 10)]], box(0, 0, 20, 20))
    assert all("array" not in f for f in fit_features(layout)["features"])
    hole = box(0, 0, 20, 20).difference(box(5, 5, 15, 15))
    assert fit_features(make_layout([("etch", hole)]))["unfit"]


def test_import_gds_round_trip(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    library = gdstk.Library(unit=1e-6)
    cell = library.new_cell("SYNTHETIC")
    cell.add(gdstk.rectangle((10, 20), (30, 32), layer=4))
    path = tmp_path / "synthetic.gds"
    library.write_gds(path)
    round_trip(import_gds(path, {"etch": 4}))


def test_import_png_round_trip(tmp_path):
    imageio = pytest.importorskip("imageio.v3")
    pytest.importorskip("skimage")
    pixels = np.zeros((80, 100), dtype=np.uint8)
    pixels[20:60, 20:80] = 255
    path = tmp_path / "synthetic.png"
    imageio.imwrite(path, pixels)
    round_trip(import_image(path, pixel_um=.5))
