"""Contract 10f against real spawned workers and on-disk results."""
import json
import base64
import csv
import io
import math
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from platesim.server.app import create_app

SPEC = json.loads((Path(__file__).parents[1] / "platesim/spec3d/examples/quick_trench.json").read_text())


def wait(client, run_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/3d/runs/{run_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(.05)
    pytest.fail(f"run {run_id} did not finish within {timeout}s")


def test_import_is_light():
    script = ("import sys; import platesim.server.routes3d; "
              "assert not {'numpy','numba','pyvista','platesim.sample3d','platesim.analysis3d',"
              "'platesim.tile3d'} & sys.modules.keys()")
    subprocess.run([sys.executable, "-c", script], check=True)


def test_strict_json():
    from platesim.mcp3d.service import _safe
    assert _safe({"values": np.array([np.nan, np.inf, -np.inf, 1.])}) == {
        "values": [None, None, None, 1.]}


def test_fast_endpoints(tmp_path):
    with TestClient(create_app(tmp_path, max_workers=1)) as client:
        assert "properties" in client.get("/api/3d/schema").json()["spec_schema"]
        examples = client.get("/api/3d/examples").json()["examples"]
        assert any(e["name"] == "quick_trench" for e in examples)
        assert client.post("/api/3d/validate", json={"spec": SPEC}).json() == {"ok": True, "errors": []}
        invalid = dict(SPEC, numerics={"quality": "wrong"})
        errors = client.post("/api/3d/validate", json={"spec": invalid}).json()
        assert not errors["ok"]
        assert {"field", "message", "fix"} <= errors["errors"][0].keys()
        rejected = client.post("/api/3d/runs", json={"spec": invalid})
        assert rejected.status_code == 422
        assert {"field", "error", "fix"} <= rejected.json()["detail"].keys()
        planned = client.post("/api/3d/plan", json={"spec": SPEC}).json()
        assert planned["ok"] and planned["tiles"] and planned["n_cells"] > 0
        assert all({"tile_id", "kind", "feature_ids", "shape", "dx_um", "n_cells", "est_time_s",
                    "weight", "assumptions", "flags"} == t.keys() for t in planned["tiles"])
        assert not client.post("/api/3d/plan", json={"spec": invalid}).json()["ok"]
        assert client.get("/api/3d/runs/deadbeef").status_code == 404
        assert client.get("/api/3d/runs/deadbeef/manifest").status_code == 404
        assert client.post("/api/3d/runs/deadbeef/cancel").status_code == 404


def test_real_run(tmp_path):
    with TestClient(create_app(tmp_path, max_workers=1)) as client:
        response = client.post("/api/3d/runs", json={"spec": SPEC, "label": "Synthetic trench"})
        assert response.status_code == 202, response.text
        rid = response.json()["run_id"]
        base = f"/api/3d/runs/{rid}"
        for path in ("manifest", "tiles", "height", "series", "frames", "export?kind=height_csv"):
            assert client.get(f"{base}/{path}").status_code == 409
        assert client.post(f"{base}/section", json={}).status_code == 409
        assert client.get(f"{base}/artifact?path=anything").status_code == 409
        assert wait(client, rid)["status"] == "done", client.get(base).json()
        manifest = client.get(f"{base}/manifest").json()
        assert manifest["quantities"] and manifest["artifacts"]
        tiles = client.get(f"{base}/tiles").json()["tiles"]
        assert tiles
        for tile in tiles:
            tid = tile["tile_id"]
            for frame in (0, -1):
                h = client.get(f"{base}/height", params={"tile_id": tid, "frame": frame, "max_n": 5}).json()
                assert len(h["x_um"]) <= 5 and len(h["y_um"]) <= 5
                assert np.asarray(h["height_um"]).shape == (len(h["x_um"]), len(h["y_um"]))
                assert np.asarray(h["bare_height_um"]).shape == np.asarray(h["height_um"]).shape
                assert all(v is None or math.isfinite(v) for row in h["height_um"] for v in row)
                if frame == 0:
                    assert h["height_um"] == h["bare_height_um"]
            h = client.get(f"{base}/height", params={"tile_id": tid, "max_n": 400}).json()
            assert np.asarray(h["height_um"]).shape == tuple(tile["shape"][:2])
            line = [[h["x_um"][0], h["y_um"][len(h["y_um"])//2]],
                    [h["x_um"][-1], h["y_um"][len(h["y_um"])//2]]]
            cut_response = client.post(f"{base}/section", json={"polyline_um": line, "tile_id": tid})
            assert cut_response.status_code == 200, cut_response.text
            cut = cut_response.json()
            assert cut["cu_outline"] and cut["si_outline"] and cut["cu_outline_initial"]
            assert cut["cu_outline"] != cut["cu_outline_initial"]
            assert 2 <= len(cut["s_um"]) <= 600
            for kind in ("height_csv", "section_csv", "stl"):
                exported = client.get(f"{base}/export", params=dict(
                    kind=kind, tile_id=tid, polyline=";".join(",".join(map(str, p)) for p in line)))
                assert exported.status_code == 200, exported.text[:200]
                assert "attachment; filename=" in exported.headers["content-disposition"]
                if kind == "stl":
                    assert exported.headers["content-type"] == "model/stl"
                    assert exported.content.startswith(b"solid") or len(exported.content) >= 84
                else:
                    rows = list(csv.DictReader(io.StringIO(exported.text)))
                    assert rows
                    if kind == "height_csv":
                        assert set(rows[0]) == {"x_um", "y_um", "height_um"}
                        assert len(rows) == tile["shape"][0]*tile["shape"][1]
                        assert float(rows[0]["x_um"]) == h["x_um"][0]
                    else:
                        assert set(rows[0]) == {"contour", "kind", "s_um", "z_um"}
                        assert {r["kind"] for r in rows} == {"cu", "cu_initial", "si"}
                        assert all(math.isfinite(float(r["z_um"])) for r in rows)
            series = client.get(f"{base}/series", params={"tile_id": tid}).json()
            frames = client.get(f"{base}/frames", params={"tile_id": tid}).json()["t_s"]
            assert frames == series["t_s"] and len(frames) >= 2
            assert all(len(v) == len(frames) for v in series.values())
        artifact = next(a for a in manifest["artifacts"] if a["kind"] == "png")
        response = client.get(f"{base}/artifact", params={"path": artifact["path"]})
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")
        for path in ("../x", "unlisted.png", "manifest.json"):
            assert 400 <= client.get(f"{base}/artifact", params={"path": path}).status_code < 500
        assert client.get(f"{base}/height?max_n=401").status_code == 422
        assert client.get(f"{base}/height?frame=999").status_code == 422
        assert client.get(f"{base}/height?tile_id=unknown").status_code == 422
        assert client.get(f"{base}/export?kind=invalid").status_code == 422
        assert client.get(f"{base}/export?kind=section_csv&polyline=bad").status_code == 422
        assert client.get(f"{base}/export?kind=stl&frame=999").status_code == 422
        assert client.post(f"{base}/section", json={"polyline_um": [[0, 0], [0, 0]]}).status_code == 422
        second = client.post("/api/3d/runs", json={"spec": SPEC}).json()["run_id"]
        assert client.post(f"/api/3d/runs/{second}/cancel").status_code == 200
        assert wait(client, second)["status"] == "cancelled"
        listing = client.get("/api/3d/runs").json()["runs"]
        assert {r["run_id"] for r in listing} == {rid, second}
        assert listing[0]["run_id"] == second
    # Both index labels/mappings and JobManager terminal history survive restart.
    with TestClient(create_app(tmp_path, max_workers=1)) as client:
        restored = client.get(base).json()
        assert restored["status"] == "done" and restored["label"] == "Synthetic trench"
        assert client.get(f"{base}/manifest").json() == manifest


@pytest.mark.parametrize("kind", ["gds", "png"])
def test_layout_import(tmp_path, kind):
    path = tmp_path / ("synthetic." + kind)
    if kind == "gds":
        gdstk = pytest.importorskip("gdstk")
        library = gdstk.Library(unit=1e-6)
        cell = library.new_cell("SYNTHETIC")
        cell.add(gdstk.rectangle((10, 10), (30, 20), layer=4))
        library.write_gds(path)
    else:
        imageio = pytest.importorskip("imageio.v3")
        pixels = np.zeros((80, 100), dtype=np.uint8)
        pixels[20:60, 20:80] = 255
        imageio.imwrite(path, pixels)
    with TestClient(create_app(tmp_path / "server", max_workers=1)) as client:
        response = client.post("/api/3d/layout/import", json=dict(
            filename="../../" + path.name, content_b64=base64.b64encode(path.read_bytes()).decode(),
            layer_map={"etch": 4}, pixel_um=.5))
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["source"] == dict(filename=path.name, kind=kind if kind == "gds" else "image", n_polygons=1)
        # Image contours bevel pixel corners, so rounded_rect may fit better.
        assert answer["features"][0]["kind"] in ("rect", "rounded_rect")
        assert not answer["unfit"] and answer["preview"]
        spec = dict(SPEC, layout={k: answer[k] for k in ("die_um", "features")})
        assert client.post("/api/3d/validate", json={"spec": spec}).json()["ok"]


def test_layout_import_errors(tmp_path):
    with TestClient(create_app(tmp_path, max_workers=1)) as client:
        for body in (dict(filename="bad.txt", content_b64=""),
                     dict(filename="bad.gds", content_b64="!"),
                     dict(filename="big.png", content_b64=base64.b64encode(b"x"*(20*1024*1024+1)).decode()),
                     dict(filename="bad.png", content_b64="eA==")):
            response = client.post("/api/3d/layout/import", json=body)
            assert response.status_code == 422, response.text
            assert {"error", "field", "fix"} <= response.json()["detail"].keys()
