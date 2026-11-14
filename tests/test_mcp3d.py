"""No solver required except the optional tiny integration test."""
import asyncio
import json
from pathlib import Path
from threading import Event
import time

import pytest

from platesim.mcp3d import RunService
from platesim.spec3d import Manifest, spec_hash
from tests.fixtures3d.make_fixtures import box_step


@pytest.fixture
def spec():
    return json.loads((Path(__file__).parents[1] / "platesim/spec3d/examples/single_trench.json").read_text())


def fake_plan(spec):
    return {"dx_um": 4., "window_um": [-96., -96., 96., 96.], "n_cells": 92160, "est_seconds": 0.1}


def fake_run_sample(spec, out_dir, progress=None):
    if progress:
        progress(.25, "Writing fixture")
    box_step(out_dir)
    manifest = Manifest(spec_hash=spec_hash(spec), code_version="test", backend="analytic",
                        device="cpu", dx_um=4., physics="L1", created_utc="2026-01-01T00:00:00Z",
                        waveform_sensitivity="net-current only", summary_text="Synthetic fixture")
    data = manifest.model_dump(mode="json")
    (Path(out_dir) / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    if progress:
        progress(.9, "Manifest ready")
    return data


def wait(service, run_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = service.get_status(run_id)
        if status.get("state") in ("done", "failed"):
            return status
        time.sleep(.01)
    pytest.fail("Background run did not finish within 20 s")


@pytest.fixture
def completed(tmp_path, spec):
    service = RunService(tmp_path, runner=fake_run_sample, planner=fake_plan)
    run_id = service.run_sample(spec)["run_id"]
    assert wait(service, run_id)["state"] == "done"
    return service, run_id


def test_validation_and_planning(tmp_path, spec):
    service = RunService(tmp_path, runner=fake_run_sample, planner=fake_plan)
    assert service.validate_spec(spec) == {"ok": True, "errors": []}
    spec["etch"]["default"]["depth_um"] = -1
    error = service.validate_spec(spec)["errors"][0]
    assert set(error) == {"field", "value", "message", "fix"}
    assert error["field"] == "etch.default.depth_um"
    assert error["value"] == -1
    assert "error" in service.plan_run(spec)
    assert "error" in service.run_sample(spec)
    assert service.validate_spec("not a dict")["ok"] is False


def test_plan_does_not_run(tmp_path, spec):
    def forbidden(*args, **kwargs):
        pytest.fail("Planner ran solver")
    service = RunService(tmp_path, runner=forbidden, planner=fake_plan)
    assert service.plan_run(spec)["est_time_s"] == .1
    assert not list(tmp_path.iterdir())


def test_background_progress_and_counter(tmp_path, spec):
    entered, release = Event(), Event()
    def runner(model, out_dir, progress):
        progress(.4, "Blocked deliberately")
        progress(-1, "Still blocked")
        entered.set()
        assert release.wait(5)
        return fake_run_sample(model, out_dir, progress)
    service = RunService(tmp_path, runner=runner)
    first = service.run_sample(spec)["run_id"]
    try:
        assert entered.wait(5)
        assert first == f"{spec_hash(spec)[:12]}-1"
        state = service.get_status(first)
        assert state["state"] == "running" and state["progress"] == .4
        assert "error" in service.get_manifest(first)
    finally:
        release.set()
    assert wait(service, first) == dict(state="done", progress=1., message="Done", error=None)
    second = service.run_sample(spec)["run_id"]
    assert second.endswith("-2")
    assert wait(service, second)["state"] == "done"
    assert len(service.list_runs()["runs"]) == 2


def test_failure_and_missing_manifest(tmp_path, spec):
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")
    service = RunService(tmp_path, runner=broken)
    run_id = service.run_sample(spec)["run_id"]
    status = wait(service, run_id)
    assert status["state"] == "failed" and "synthetic solver failure" in status["error"]
    service = RunService(tmp_path, runner=lambda *a, **k: {})
    assert wait(service, service.run_sample(spec)["run_id"])["state"] == "failed"


def test_manifest_restart_and_interrupted(completed):
    service, run_id = completed
    manifest = service.get_manifest(run_id)
    Manifest.model_validate(manifest)
    restarted = RunService(service.root)
    assert restarted.get_status(run_id)["state"] == "done"
    assert restarted.get_manifest(run_id) == manifest
    (service.root / "0123456789ab-1").mkdir()
    assert restarted.get_status("0123456789ab-1")["state"] == "failed"


@pytest.mark.parametrize("run_id", ["../escape", "C:\\outside", "", "a"*12+"-1", "0123456789ab-9"])
def test_unknown_or_unsafe_id(tmp_path, run_id):
    service = RunService(tmp_path)
    for result in (service.get_status(run_id), service.get_manifest(run_id), service.height_map(run_id)):
        assert set(result) == {"error", "field", "fix"}


def test_analysis_and_bounded_json(completed):
    service, run_id = completed
    section = service.cross_section(run_id, [[-90, 0], [90, 0]], render=False)
    assert "error" not in section, section
    assert section["length_um"] == pytest.approx(180)
    assert section["sample_count"] <= 64
    assert not {"psi", "psi_si", "png_path"} & section.keys()
    section = service.cross_section(run_id, [[-90, 0], [90, 0]])
    assert Path(section["png_path"]).is_file()
    height = service.height_map(run_id)
    assert "height_um" not in height
    assert height["stats"]["tir_um"] == pytest.approx(80)
    assert Path(height["png_path"]).is_file()
    height = service.height_map(run_id, include_samples=True)
    assert len(height["height_um"]) <= 64 and len(height["height_um"][0]) <= 64
    json.dumps(height, allow_nan=False)
    assert "error" in service.cross_section(run_id, [[0, 0], [0, 0]])
    assert "error" in service.cross_section(run_id, [[float("nan"), 0], [1, 1]])


def test_render_tools(completed, monkeypatch):
    service, run_id = completed
    import platesim.render3d as rendering
    def surface(result, out_png, camera):
        assert camera == "front"
        Path(out_png).write_bytes(b"fake PNG")
    monkeypatch.setattr(rendering, "render_surface", surface)
    for kind in ("surface", "section", "height_map"):
        path = service.render(run_id, kind, "front")
        assert "error" not in path, path
        assert Path(path["path"]).is_file()
    assert "error" in service.render(run_id, "unknown")
    assert "error" in service.render(run_id, "surface", "unknown")


def test_fastmcp_tool_list(tmp_path):
    from platesim.mcp3d.server import create_server
    server = create_server(tmp_path)
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "validate_spec", "plan_run", "run_sample", "get_status", "get_manifest",
        "cross_section", "height_map", "render", "list_runs"}
    section = next(t for t in tools if t.name == "cross_section")
    schema = getattr(section, "inputSchema", None) or section.input_schema
    assert schema["properties"]["render"]["default"] is True
    # Exercise SDK invocation as well as schema registration.
    result = asyncio.run(server.call_tool("validate_spec", {"spec": {}}))
    assert result is not None


@pytest.mark.slow
def test_real_tiny_preview(tmp_path, spec):
    sample = pytest.importorskip("platesim.sample3d")
    if not callable(getattr(sample, "run_sample", None)):
        pytest.skip("sample3d.run_sample is not available yet")
    pytest.importorskip("shapely", reason="Real sample3d geometry needs the optional shapely package")
    spec["layout"] = {"die_um": [32, 32], "features": [
        {"kind": "rect", "x_um": 16, "y_um": 16, "w_um": 16, "h_um": 16}]}
    spec["etch"] = {"default": {"depth_um": 8}}
    spec["recipe"]["steps"] = [{"t_forward_ms": 1, "j_forward_mA_cm2": 1, "duration_h": 1e-5}]
    spec["numerics"] = {"quality": "preview", "dx_um": 8, "device": "cpu"}
    spec["setup"]["delta_um"] = 16
    service = RunService(tmp_path)
    response = service.run_sample(spec)
    assert "run_id" in response, response
    status = wait(service, response["run_id"])
    assert status["state"] == "done", status
    Manifest.model_validate(service.get_manifest(response["run_id"]))
