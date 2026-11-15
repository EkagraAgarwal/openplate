"""Service tests plus real tiny-preview and MCP stdio integration tests."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
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


def test_multi_tile_plan(tmp_path, spec):
    service = RunService(tmp_path, planner=lambda model: [fake_plan(model), fake_plan(model)])
    plan = service.plan_run(spec)
    assert plan["n_cells"] == 2 * fake_plan(spec)["n_cells"]
    assert plan["est_time_s"] == pytest.approx(.2)
    assert len(plan["tiles"]) == 2
    assert all(tile["est_time_s"] == .1 and "est_seconds" not in tile for tile in plan["tiles"])
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


def tiny_preview_spec(spec):
    spec = deepcopy(spec)
    spec["layout"] = {"die_um": [32, 32], "features": [
        {"kind": "rect", "x_um": 16, "y_um": 16, "w_um": 16, "h_um": 16}]}
    spec["etch"] = {"default": {"depth_um": 8}}
    spec["recipe"]["steps"] = [{"t_forward_ms": 1, "j_forward_mA_cm2": 1, "duration_h": 1e-5}]
    spec["numerics"] = {"quality": "preview", "dx_um": 8, "device": "cpu"}
    spec["setup"]["delta_um"] = 16
    return spec


@pytest.mark.slow
def test_real_tiny_preview(tmp_path, spec):
    sample = pytest.importorskip("platesim.sample3d")
    if not callable(getattr(sample, "run_sample", None)):
        pytest.skip("sample3d.run_sample is not available yet")
    pytest.importorskip("shapely", reason="Real sample3d geometry needs the optional shapely package")
    spec = tiny_preview_spec(spec)
    service = RunService(tmp_path)
    response = service.run_sample(spec)
    assert "run_id" in response, response
    status = wait(service, response["run_id"])
    assert status["state"] == "done", status
    Manifest.model_validate(service.get_manifest(response["run_id"]))


def test_stdio_round_trip(tmp_path, spec):
    """Exercise the actual module entry point, SDK transport and real solver."""
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "runs with spaces"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "platesim.mcp3d", "--root", str(root)],
        cwd=str(tmp_path),  # Import must work via PYTHONPATH, not the cwd.
        env={**os.environ, "PYTHONPATH": str(repo), "MPLBACKEND": "Agg"},
    )

    async def round_trip():
        # anyio cancellation lets the SDK's shielded stdio teardown close stdin,
        # wait for exit and kill/reap on timeout (including on Windows).
        with anyio.fail_after(180):
            with (tmp_path / "server-stderr.log").open("w+", encoding="utf-8") as stderr:
                async with stdio_client(parameters, errlog=stderr) as (read, write):
                    async with ClientSession(read, write) as session:
                        initialized = await session.initialize()
                        assert initialized.model_dump(by_alias=True)["serverInfo"]["name"] == "platesim-3d"
                        tools = (await session.list_tools()).tools
                        required = {
                            "validate_spec": {"spec"}, "plan_run": {"spec"},
                            "run_sample": {"spec"}, "get_status": {"run_id"},
                            "get_manifest": {"run_id"}, "cross_section": {"run_id", "polyline_um"},
                            "height_map": {"run_id"}, "render": {"run_id"}, "list_runs": set(),
                        }
                        assert {tool.name for tool in tools} == set(required)
                        for tool in tools:
                            schema = tool.model_dump(by_alias=True)["inputSchema"]
                            assert schema["type"] == "object"
                            assert set(schema.get("required", [])) == required[tool.name]
                            assert required[tool.name] <= schema.get("properties", {}).keys()

                        async def call(name, **arguments):
                            result = (await session.call_tool(name, arguments)).model_dump(by_alias=True)
                            assert not result.get("isError", False), result
                            data = result.get("structuredContent")
                            if data is None:
                                data = json.loads(result["content"][0]["text"])
                            assert isinstance(data, dict), data
                            assert not data.get("error"), data
                            json.dumps(data, allow_nan=False)
                            return data

                        assert await call("validate_spec", spec=spec) == {"ok": True, "errors": []}
                        invalid = deepcopy(spec)
                        invalid["etch"]["default"]["depth_um"] = -1
                        validation = await call("validate_spec", spec=invalid)
                        assert validation["ok"] is False
                        error = next(e for e in validation["errors"]
                                     if e["field"] == "etch.default.depth_um")
                        assert error["value"] == -1 and error["message"] and error["fix"]
                        # The shipped standard-quality trench exceeds the CPU
                        # cell cap. Preserve its actionable error, then explicitly
                        # opt into preview coarsening rather than silently changing it.
                        rejected = (await session.call_tool("plan_run", {"spec": spec})).model_dump(by_alias=True)
                        rejected = rejected.get("structuredContent") or json.loads(rejected["content"][0]["text"])
                        assert rejected["field"] == "spec" and rejected["fix"] and rejected["error"]
                        preview = deepcopy(spec)
                        preview["numerics"]["quality"] = "preview"
                        plan = await call("plan_run", spec=preview)
                        assert plan["n_cells"] > 0 and plan["est_time_s"] >= 0
                        assert len(plan["tiles"]) == 2
                        assert plan["n_cells"] == sum(tile["n_cells"] for tile in plan["tiles"])

                        run = await call("run_sample", spec=tiny_preview_spec(spec))
                        run_id = run["run_id"]
                        with anyio.fail_after(120):
                            while True:
                                status = await call("get_status", run_id=run_id)
                                assert 0 <= status["progress"] <= 1
                                assert status["state"] != "failed", status
                                if status["state"] == "done":
                                    break
                                await anyio.sleep(.1)
                        assert status["progress"] == 1 and status["error"] is None
                        manifest = await call("get_manifest", run_id=run_id)
                        Manifest.model_validate(manifest)
                        assert manifest["spec_hash"] == spec_hash(tiny_preview_spec(spec))

                        def check_png(value):
                            assert isinstance(value, str)
                            path = root / value  # Also accepts absolute plain-string paths.
                            assert path.resolve().is_relative_to(root.resolve())
                            assert path.is_file() and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

                        height = await call("height_map", run_id=run_id, include_samples=True)
                        assert height["stats"]
                        assert 0 < len(height["height_um"]) <= 64
                        assert 0 < len(height["height_um"][0]) <= 64
                        check_png(height["png_path"])
                        surface = await call("render", run_id=run_id, kind="surface")
                        check_png(surface["path"])

    asyncio.run(round_trip())
