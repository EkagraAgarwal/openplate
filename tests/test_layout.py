"""Synthetic layout and etch checks; all lengths and volumes here are um-based."""
import math

import numpy as np
import pytest

from platesim.layout import (load_layout, import_image, import_gds, import_dxf,
                             rasterize, signed_distance_2d, etch_to_psi)


def test_parametric():
    features = []
    for i, w in enumerate((6, 8, 10, 12)):
        features.append(dict(kind="rect", x_um=20*i+10, y_um=20,
                             w_um=w, h_um=24, layer="trench"))
        features.append(dict(kind="circle", x_um=20*i+10, y_um=60,
                             r_um=w/2, layer="via"))
    spec = dict(die_um=[100, 100], features=features)
    layout = load_layout(spec)
    again = load_layout(layout.to_spec())
    assert len(layout.features) == len(again.features) == 8
    assert len({f.id for f in layout.features}) == 8
    for layer, area in (("trench", 24*36), ("via", math.pi*86)):
        assert layout.geometry(layer).area == pytest.approx(area, rel=.005)
        assert again.geometry(layer).area == pytest.approx(area, rel=.005)
    array = load_layout(dict(die_um=[100, 100], features=[dict(
        kind="rounded_rect", x_um=15, y_um=15, w_um=10, h_um=8,
        r_um=2, rotation_deg=30, array=dict(nx=2, ny=3, pitch_x_um=25,
                                          pitch_y_um=20))]))
    assert len(array.features) == 6
    assert array.features[0].polygon.area == pytest.approx(80-16+4*math.pi, rel=.005)
    assert len(array.features[0].bounds_um) == 4


def test_image(tmp_path):
    from skimage.io import imsave
    a = np.zeros((100, 120), dtype=np.uint8)
    a[10:90, 10:110] = 255
    a[30:70, 40:80] = 0
    path = tmp_path / "mask.png"
    imsave(path, a)
    layout = import_image(path, .5, threshold=128)
    assert layout.geometry("etch").area == pytest.approx((8000-1600)*.25, rel=.02)
    assert len(layout.geometry("etch").interiors) == 1


def test_gds(tmp_path):
    import gdstk
    lib = gdstk.Library(unit=1e-6)
    cell = lib.new_cell("feature")
    cell.add(gdstk.rectangle((0, 0), (8, 10), layer=4))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(cell, columns=2, rows=2, spacing=(20, 20)))
    path = tmp_path / "layout.gds"
    lib.write_gds(path)
    layout = import_gds(path, {"etch": 4})
    assert len(layout.features) == 4
    assert layout.geometry("etch").area == pytest.approx(320, rel=.001)


def test_dxf(tmp_path):
    import ezdxf
    doc = ezdxf.new()
    space = doc.modelspace()
    space.add_lwpolyline([(0, 0), (10, 0), (10, 8), (0, 8)], close=True)
    space.add_circle((20, 20), 5)
    space.add_lwpolyline([(30, 30), (40, 40)])
    path = tmp_path / "layout.dxf"
    doc.saveas(path)
    layout = import_dxf(path, {"etch": "0"})
    assert layout.geometry("etch").area == pytest.approx(80+25*math.pi, rel=.001)
    assert len(layout.skipped_entities) == 1
    assert "open" in layout.skipped_entities[0]["reason"]


@pytest.mark.parametrize("kind", ["rect", "circle"])
def test_raster(kind):
    layout = load_layout(dict(die_um=[48, 48], features=[dict(
        kind=kind, x_um=24, y_um=24, w_um=18, h_um=30, r_um=15)]))
    mask = rasterize(layout, "etch", (0, 0, 48, 48), 1)
    assert mask.shape == (48, 48) and mask.dtype == bool
    assert mask.sum() == pytest.approx(layout.geometry("etch").area, rel=.02)
    distance = signed_distance_2d(mask, 1)
    assert np.all(distance[mask] > 0) and np.all(distance[~mask] < 0)


def _etched(kind="rect", angle=90, radius=0):
    layout = load_layout(dict(die_um=[48, 48], features=[dict(
        kind=kind, x_um=24, y_um=24, w_um=20, h_um=32, r_um=15)]))
    mask = rasterize(layout, "etch", (0, 0, 48, 48), 1)
    psi, grid = etch_to_psi(mask, 1, dict(depth_um=16, sidewall_deg=angle,
                                        bottom_radius_um=radius), (-24, 8))
    return psi, grid


@pytest.mark.parametrize("name,kind,angle,radius", [
    ("vertical trench", "rect", 90, 0), ("80-degree trench", "rect", 80, 0),
    ("via", "circle", 90, 0), ("rounded via", "circle", 90, 4)])
def test_volume(name, kind, angle, radius):
    psi, grid = _etched(kind, angle, radius)
    z = (np.arange(psi.shape[2])+.5) + grid["origin"][2]*1e4
    numeric = np.count_nonzero(psi[:, :, z < 0] > 0)
    if kind == "rect":
        a = 1/math.tan(math.radians(angle))
        analytic = 20*32*16 - a*(20+32)*16**2 + 4*a*a*16**3/3
    else:
        analytic = math.pi*15**2*16
        analytic -= 2*math.pi*((15-radius)*radius**2*(1-math.pi/4)+radius**3/6)
    print(f"\n{name}: analytic={analytic:.3f}, numeric={numeric:.3f} um^3")
    assert numeric == pytest.approx(analytic, rel=.02)
    assert grid["shape"] == psi.shape
    assert grid["dx"] == pytest.approx(1e-4)
    assert psi.dtype == np.float32
    assert np.all(psi[:, :, z > 0] > 0)


def test_signed_distance_band():
    psi, _ = _etched()
    # Check planar surface patches in a band, excluding corners and medial axes
    # by geometric distance, not by selecting on the computed gradient.
    grad = np.sqrt(sum(g*g for g in np.gradient(psi.astype(float), 1e-4)))
    x, y, z = np.meshgrid(np.arange(48)+.5, np.arange(48)+.5,
                           np.arange(32)+.5-24, indexing="ij")
    floor = (abs(x-24) < 5) & (abs(y-24) < 10) & (z < -12)
    wall = (abs(y-24) < 10) & (z > -10) & (z < -6)
    field = ((abs(x-24) > 16) | (abs(y-24) > 22)) & (z > -4)
    band = (abs(psi) > 1.5e-4) & (abs(psi) < 4e-4)
    band &= floor | wall | field
    band[:3] = band[-3:] = False
    band[:, :3] = band[:, -3:] = False
    band[:, :, :3] = band[:, :, -3:] = False
    assert band.sum() > 100
    assert np.max(abs(grad[band]-1)) < .05


def test_width_dependent_depth():
    mask = np.zeros((48, 48), bool)
    mask[4:10, 8:40] = True
    mask[22:38, 8:40] = True
    psi, grid = etch_to_psi(mask, 1, dict(depth_um=20,
        depth_vs_width=[[6, 6], [16, 16]]), (-24, 8))
    z = (np.arange(32)+.5)-24
    assert z[psi[6, 24] > 0].min() == pytest.approx(-5.5, abs=1)
    assert z[psi[30, 24] > 0].min() == pytest.approx(-15.5, abs=1)


def test_empty_and_full_masks():
    for value in (False, True):
        mask = np.full((12, 12), value)
        psi, _ = etch_to_psi(mask, 1, {"depth_um": 4}, (-8, 4))
        assert np.isfinite(psi).all()
        expected = np.arange(12)+.5-8+(4 if value else 0)
        np.testing.assert_allclose(psi[6, 6]*1e4, expected, atol=.05)


@pytest.mark.parametrize("kwargs", [dict(dx_um=0), dict(z_extent_um=(1, 0)),
                                     dict(etch_layer={"depth_um": -1})])
def test_invalid_etch(kwargs):
    args = dict(mask=np.zeros((8, 8), bool), dx_um=1,
                etch_layer={"depth_um": 2}, z_extent_um=(-4, 4))
    args.update(kwargs)
    with pytest.raises(ValueError):
        etch_to_psi(**args)


def test_gds_units_and_datatype(tmp_path):
    import gdstk
    lib = gdstk.Library(unit=1e-9, precision=1e-12)
    cell = lib.new_cell("nano")
    cell.add(gdstk.rectangle((0, 0), (1000, 2000), layer=2, datatype=3))
    cell.add(gdstk.rectangle((3000, 0), (4000, 2000), layer=2, datatype=4))
    path = tmp_path / "nano.gds"
    lib.write_gds(path)
    assert import_gds(path, {"etch": [2, 3]}).geometry("etch").area == pytest.approx(2)
    assert import_gds(path, {"etch": [2, 3]}, unit_um=1).geometry("etch").area == 2e6


def test_rotation_and_xy_order():
    layout = load_layout(dict(die_um=[24, 32], features=[dict(
        kind="rect", x_um=8, y_um=16, w_um=12, h_um=6, rotation_deg=90)]))
    mask = rasterize(layout, "etch", (0, 0, 24, 32), 1)
    assert mask.shape == (24, 32)
    assert mask[5:11, 10:22].all()
    assert mask.sum() == 72


def test_image_islands_and_native_rgb_threshold(tmp_path):
    from skimage.io import imsave
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    a[5:95, 5:95] = 255
    a[20:80, 20:80] = 0
    a[35:65, 35:65] = 255
    path = tmp_path / "islands.png"
    imsave(path, a)
    layout = import_image(path, 1, threshold=128)
    assert len(layout.features) == 2
    assert layout.geometry("etch").area == pytest.approx(8100-3600+900, rel=.02)


def test_taper_and_radius_nest():
    vertical, _ = _etched("circle")
    taper, _ = _etched("circle", angle=80)
    rounded, _ = _etched("circle", radius=4)
    assert np.count_nonzero(taper > 0) < np.count_nonzero(vertical > 0)
    assert np.count_nonzero(rounded > 0) < np.count_nonzero(vertical > 0)


def test_gds_hole(tmp_path):
    import gdstk
    lib = gdstk.Library()
    cell = lib.new_cell("hole")
    polygons = gdstk.boolean(gdstk.rectangle((0, 0), (30, 30)),
                            gdstk.rectangle((10, 10), (20, 20)), "not", layer=1)
    cell.add(*polygons)
    path = tmp_path / "hole.gds"
    lib.write_gds(path)
    assert import_gds(path, {"etch": 1}).geometry("etch").area == pytest.approx(800)


@pytest.mark.parametrize("kind", ["circle", "rect"])
def test_six_cell_features_and_clipping(kind):
    layout = load_layout(dict(die_um=[24, 24], features=[dict(
        kind=kind, x_um=12.3, y_um=12.2, r_um=3, w_um=6.2, h_um=6.4)]))
    mask = rasterize(layout, "etch", (0, 0, 24, 24), 1)
    assert mask.sum() == pytest.approx(layout.geometry("etch").area, rel=.02)
    from shapely.geometry import box
    clipped = rasterize(layout, "etch", (12, 0, 24, 24), 1)
    area = layout.geometry("etch").intersection(box(12, 0, 24, 24)).area
    assert abs(clipped.sum()-area) <= .5
